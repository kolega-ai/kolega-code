"""Stream-segment rotation around hosted web-search calls.

Hosted web_search is the only tool that renders mid-stream: Responses
providers interleave reasoning → web_search_call → reasoning inside one
streamed response. Transcript consumers key streamed entries by chunk uuid,
so at each hosted call the loop must close the open thinking/response
segment (a ``complete: True`` flush) and rotate both uuids — otherwise
post-search output folds into the entry rendered above the search rows
instead of opening a new one below them. Hermetic: FakeLLM with a scripted
event stream.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from kolega_code.llm.models import Message, MessageChunk, TextBlock, WebSearchCallBlock

from .compaction_helpers import FakeLLM, FakeStream

# Above the 50-char flush threshold, so each event yields a chunk immediately.
THINK_BEFORE = "reasoning before the search " + "a" * 40
THINK_AFTER = "reasoning after the search " + "b" * 40
ANSWER_BEFORE = "answer text before the search " + "c" * 40
ANSWER_AFTER = "answer text after the search " + "d" * 40


class _ScriptedStream(FakeStream):
    """FakeStream that yields scripted MessageChunks before finishing."""

    def __init__(self, events, final_message):
        super().__init__(final_message)
        self._events = list(events)

    async def __anext__(self):
        if self._events:
            return self._events.pop(0)
        raise StopAsyncIteration


def _hosted_event(item_id: str = "ws_1") -> MessageChunk:
    return MessageChunk(
        type="hosted_tool_call",
        tool_call_delta={
            "id": item_id,
            "name": "web_search",
            "status": "completed",
            "action": {"type": "search", "queries": ["kolega"]},
        },
    )


def _configure(agent, events) -> None:
    agent.system_prompt = Message(role="system", content=[TextBlock(text="sys")])
    agent.tool_collection = MagicMock()
    agent.tool_collection.get_tool_list = MagicMock(return_value=[])
    agent.send_chat_message = AsyncMock()
    agent.log_info = AsyncMock()
    agent.log_error = AsyncMock()
    agent.llm = FakeLLM(token_script=[100])
    final = Message(role="assistant", content=[TextBlock(text="done")], stop_reason="end_turn")
    agent.llm.stream = AsyncMock(side_effect=lambda *args, **kwargs: _ScriptedStream(events, final))


async def _chunks(agent) -> list[dict]:
    return [chunk async for chunk in agent.process_message_stream("go")]


def _of_type(chunks: list[dict], kind: str) -> list[dict]:
    return [chunk for chunk in chunks if chunk.get("type") == kind]


def _hosted_pair_calls(agent) -> list[dict]:
    return [
        call.kwargs
        for call in agent.send_chat_message.await_args_list
        if call.kwargs.get("tool_description") == WebSearchCallBlock.TOOL_LABEL
    ]


@pytest.mark.asyncio
async def test_thinking_uuid_rotates_across_hosted_call(base_agent) -> None:
    _configure(
        base_agent,
        [
            MessageChunk(type="thinking", thinking=THINK_BEFORE),
            _hosted_event(),
            MessageChunk(type="thinking", thinking=THINK_AFTER),
        ],
    )

    thinking = _of_type(await _chunks(base_agent), "thinking")

    uuid_before = thinking[0]["uuid"]
    assert thinking[0] == {"type": "thinking", "content": THINK_BEFORE, "complete": False, "uuid": uuid_before}
    # The pre-search segment is closed before the search rows are emitted...
    assert thinking[1] == {"type": "thinking", "content": "", "complete": True, "uuid": uuid_before}
    # ...and the post-search reasoning opens a fresh segment.
    uuid_after = thinking[2]["uuid"]
    assert uuid_after != uuid_before
    assert thinking[2]["content"] == THINK_AFTER
    assert all(chunk["uuid"] == uuid_after for chunk in thinking[2:])
    assert thinking[-1]["complete"] is True


@pytest.mark.asyncio
async def test_response_uuid_rotates_across_hosted_call(base_agent) -> None:
    _configure(
        base_agent,
        [
            MessageChunk(type="text", text=ANSWER_BEFORE),
            _hosted_event(),
            MessageChunk(type="text", text=ANSWER_AFTER),
        ],
    )

    response = _of_type(await _chunks(base_agent), "response")

    uuid_before = response[0]["uuid"]
    assert response[0] == {"type": "response", "content": ANSWER_BEFORE, "complete": False, "uuid": uuid_before}
    assert response[1] == {"type": "response", "content": "", "complete": True, "uuid": uuid_before}
    uuid_after = response[2]["uuid"]
    assert uuid_after != uuid_before
    assert response[2]["content"] == ANSWER_AFTER
    assert all(chunk["uuid"] == uuid_after for chunk in response[2:])
    assert response[-1]["complete"] is True


@pytest.mark.asyncio
async def test_hosted_call_without_prior_thinking_emits_no_thinking_chunk(base_agent) -> None:
    _configure(
        base_agent,
        [
            _hosted_event(),
            MessageChunk(type="text", text=ANSWER_AFTER),
        ],
    )

    chunks = await _chunks(base_agent)

    # No thinking segment was open, so none is flushed — an empty complete
    # thinking chunk would render a bare glyph line in the transcript.
    assert _of_type(chunks, "thinking") == []
    # The response flush is unconditional: the segment before the search is
    # closed (empty no-op chunk) and the answer arrives under a fresh uuid.
    response = _of_type(chunks, "response")
    assert response[0]["complete"] is True
    assert response[0]["content"] == ""
    assert response[1]["uuid"] != response[0]["uuid"]
    assert response[1]["content"] == ANSWER_AFTER


@pytest.mark.asyncio
async def test_each_hosted_call_rotates_again_and_emits_one_row_pair(base_agent) -> None:
    _configure(
        base_agent,
        [
            MessageChunk(type="thinking", thinking=THINK_BEFORE),
            _hosted_event("ws_1"),
            MessageChunk(type="thinking", thinking=THINK_AFTER),
            _hosted_event("ws_2"),
            MessageChunk(type="text", text=ANSWER_AFTER),
        ],
    )

    chunks = await _chunks(base_agent)

    thinking_uuids = [chunk["uuid"] for chunk in _of_type(chunks, "thinking")]
    assert len(set(thinking_uuids)) == 2  # one segment per search boundary
    pairs = _hosted_pair_calls(base_agent)
    assert [(call["message_type"], call["tool_call_id"]) for call in pairs] == [
        ("tool_call", "ws_1"),
        ("tool_result", "ws_1"),
        ("tool_call", "ws_2"),
        ("tool_result", "ws_2"),
    ]
    assert pairs[0]["content"] == "Calling web_search (hosted): search: 'kolega'"
    assert pairs[1]["content"] == "search: 'kolega' — completed (results injected server-side)"
