"""Contiguous per-type stream segments at reasoning/prose transitions.

Arrival-order consumers (ACP clients, journal printers) render chunks in
arrival order, so the loop must close a thinking segment the moment prose
starts — and close a prose segment when reasoning resumes.
"""

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from kolega_code.agent.baseagent import BaseAgent
from kolega_code.llm.models import Message, MessageChunk, TextBlock

from .compaction_helpers import FakeLLM, FakeStream, build_agent

THINK_ONE = "reason one"
ANSWER_ONE = "answer one"
THINK_TWO = "reason two"
ANSWER_TWO = "answer two"


class _ScriptedStream(FakeStream):
    def __init__(self, events: AsyncIterator[MessageChunk], final_message: Message) -> None:
        super().__init__(final_message)
        self._events = events

    async def __anext__(self) -> MessageChunk:
        return await anext(self._events)


def _think(text: str) -> MessageChunk:
    return MessageChunk(type="thinking", thinking=text)


def _text(text: str) -> MessageChunk:
    return MessageChunk(type="text", text=text)


def _configure(agent: BaseAgent, events: list[MessageChunk] | AsyncIterator[MessageChunk]) -> None:
    agent.system_prompt = Message(role="system", content=[TextBlock(text="sys")])
    agent.tool_collection = MagicMock()
    agent.tool_collection.get_tool_list = MagicMock(return_value=[])
    agent.send_chat_message = AsyncMock()
    agent.log_info = AsyncMock()
    agent.log_error = AsyncMock()
    agent.llm = cast(Any, FakeLLM(token_script=[100]))
    final = Message(role="assistant", content=[TextBlock(text="done")], stop_reason="end_turn")

    async def scripted_events() -> AsyncIterator[MessageChunk]:
        if isinstance(events, list):
            for event in events:
                yield event
        else:
            async for event in events:
                yield event

    agent.llm.stream = AsyncMock(side_effect=lambda *args, **kwargs: _ScriptedStream(scripted_events(), final))


async def _chunks(agent: BaseAgent) -> list[dict]:
    return [chunk async for chunk in agent.process_message_stream("go")]


def _by_uuid(chunks: list[dict], chunk_type: str) -> dict[str, str]:
    ordered: list[tuple[str, str]] = []
    index: dict[str, int] = {}
    for chunk in chunks:
        if chunk["type"] != chunk_type:
            continue
        uuid_value = str(chunk["uuid"])
        if uuid_value in index:
            ordered[index[uuid_value]] = (uuid_value, ordered[index[uuid_value]][1] + chunk["content"])
        else:
            index[uuid_value] = len(ordered)
            ordered.append((uuid_value, chunk["content"]))
    return dict(ordered)


@pytest.mark.asyncio
async def test_interleaved_stream_yields_contiguous_segments(tmp_path: Path) -> None:
    agent, _cm = build_agent(tmp_path)
    _configure(agent, [_think(THINK_ONE), _text(ANSWER_ONE), _think(THINK_TWO), _text(ANSWER_TWO)])
    chunks = await _chunks(agent)

    assert list(_by_uuid(chunks, "thinking").values()) == [THINK_ONE, THINK_TWO]
    assert list(_by_uuid(chunks, "response").values()) == [ANSWER_ONE, ANSWER_TWO]

    first_response = next(i for i, c in enumerate(chunks) if c["type"] == "response")
    thinking_one_closed = next(i for i, c in enumerate(chunks) if c["type"] == "thinking" and c["complete"])
    assert thinking_one_closed < first_response

    first_thinking_uuid = str(chunks[0]["uuid"])
    thinking_two = next(i for i, c in enumerate(chunks) if c["type"] == "thinking" and c["uuid"] != first_thinking_uuid)
    response_one_closed = next(
        i for i, c in enumerate(chunks) if c["type"] == "response" and c["complete"] and c["uuid"] != chunks[-1]["uuid"]
    )
    assert response_one_closed < thinking_two


@pytest.mark.asyncio
async def test_reasoning_first_turn_closes_thinking_before_prose(tmp_path: Path) -> None:
    agent, _cm = build_agent(tmp_path)
    _configure(agent, [_think(THINK_ONE), _text(ANSWER_ONE)])
    chunks = await _chunks(agent)

    assert list(_by_uuid(chunks, "thinking").values()) == [THINK_ONE]
    assert list(_by_uuid(chunks, "response").values()) == [ANSWER_ONE]
    first_response = next(i for i, c in enumerate(chunks) if c["type"] == "response")
    assert first_response > 0
    assert all(c["type"] == "thinking" for c in chunks[:first_response])
    assert any(c["type"] == "thinking" and c["complete"] for c in chunks[:first_response])


@pytest.mark.asyncio
@pytest.mark.parametrize(("delta", "chunk_type"), [(_text, "response"), (_think, "thinking")])
async def test_first_nonempty_delta_arrives_before_second(
    tmp_path: Path, delta: Callable[[str], MessageChunk], chunk_type: str
) -> None:
    agent, _cm = build_agent(tmp_path)
    release = asyncio.Event()
    observed: asyncio.Queue[str] = asyncio.Queue()

    async def gated_events() -> AsyncIterator[MessageChunk]:
        yield delta("")
        yield delta("Hello")
        observed.put_nowait("second delta requested")
        await release.wait()
        yield delta(" again")

    async def consume() -> list[dict]:
        async with aclosing(agent.process_message_stream("go")) as stream:
            first = await anext(stream)
            observed.put_nowait("first chunk received")
            return [first, *[chunk async for chunk in stream]]

    _configure(agent, gated_events())
    consumer = asyncio.create_task(consume())
    try:
        async with asyncio.timeout(10):
            assert await observed.get() == "first chunk received"
    finally:
        release.set()
        chunks = await asyncio.wait_for(consumer, timeout=10)

    segment = [chunk for chunk in chunks if chunk["type"] == chunk_type]
    assert [(chunk["content"], chunk["complete"]) for chunk in segment] == [("Hello", False), (" again", True)]
    assert segment[0]["uuid"] == segment[1]["uuid"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("delta", "chunk_type"), [(_text, "response"), (_think, "thinking")])
async def test_flush_threshold_batches_updates_after_first_delta(
    tmp_path: Path, delta: Callable[[str], MessageChunk], chunk_type: str
) -> None:
    agent, _cm = build_agent(tmp_path)
    _configure(agent, [delta("12345") for _ in range(12)])
    chunks = await _chunks(agent)

    updates = [(len(chunk["content"]), chunk["complete"]) for chunk in chunks if chunk["type"] == chunk_type]
    assert updates == [(5, False), (20, False), (20, False), (15, True)]
