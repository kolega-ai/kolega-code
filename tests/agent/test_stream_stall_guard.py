"""The stream stall guard: deterministic failure + a visible countdown.

A silent provider stream used to be bounded only by the transport read timeout,
which cannot tell "the endpoint never answered" from "a long gap mid-answer" and
says nothing to the user while it waits. The guard applies a stage-specific
deadline per event and reports heartbeats in between.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kolega_code.agent.baseagent import BaseAgent
from kolega_code.events import AgentEvent
from kolega_code.llm.exceptions import LLMStreamStalled
from kolega_code.llm.models import MessageChunk


class _HangingStream:
    """Yields the given events, then either ends or never produces another one."""

    def __init__(self, events=(), *, hang: bool = True):
        self._events = list(events)
        self._hang = hang
        self.cancelled = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._events:
            return self._events.pop(0)
        if not self._hang:
            raise StopAsyncIteration
        try:
            import asyncio

            await asyncio.sleep(3600)
        except BaseException:
            self.cancelled = True
            raise
        raise StopAsyncIteration


def _tune(agent, *, first: float = 0.05, idle: float = 0.05, heartbeat: float = 0.02) -> None:
    agent._stream_first_chunk_timeout = first
    agent._stream_idle_timeout = idle
    agent._stream_heartbeat_seconds = heartbeat


def _status_events(mock_connection_manager) -> list[AgentEvent]:
    events = []
    for call in mock_connection_manager.broadcast_event.await_args_list:
        event = call.args[0]
        if event.event_type == "llm_status_update":
            events.append(event)
    return events


@pytest.mark.asyncio
async def test_zero_output_stall_is_classified_as_unreachable(base_agent, mock_connection_manager):
    _tune(base_agent, first=0.05, heartbeat=0.02)
    stream = _HangingStream()

    with pytest.raises(LLMStreamStalled) as excinfo:
        async for _ in base_agent._iter_stream_with_stall_guard(stream):
            pass

    error = excinfo.value
    assert error.stage == "first_chunk"
    assert error.zero_output is True
    assert error.bytes_received == 0
    assert error.timeout_s == 0.05
    assert "no data" in str(error).lower()
    # The read was abandoned rather than left dangling.
    assert stream.cancelled is True


@pytest.mark.asyncio
async def test_mid_stream_stall_keeps_what_arrived(base_agent):
    _tune(base_agent, idle=0.05, heartbeat=0.02)
    stream = _HangingStream([SimpleNamespace(type="text", text="hello", thinking=None)])

    received = []
    with pytest.raises(LLMStreamStalled) as excinfo:
        async for event in base_agent._iter_stream_with_stall_guard(stream):
            received.append(event)

    assert [event.text for event in received] == ["hello"]
    error = excinfo.value
    assert error.stage == "stream"
    assert error.bytes_received == 5
    assert error.partial_output is True
    assert error.zero_output is False


@pytest.mark.asyncio
async def test_silence_reports_a_heartbeat_before_the_deadline(base_agent, mock_connection_manager):
    _tune(base_agent, first=0.15, heartbeat=0.02)

    with pytest.raises(LLMStreamStalled):
        async for _ in base_agent._iter_stream_with_stall_guard(_HangingStream()):
            pass

    heartbeats = _status_events(mock_connection_manager)
    assert heartbeats, "no stall heartbeat was emitted"
    content = heartbeats[0].content
    assert content["stage"] == "first_chunk"
    assert content["seconds_since_data"] >= 0.0
    assert content["timeout_s"] == 0.15
    assert content["message"] == "Waiting for response…"
    assert content["status"] == "info"
    assert not content.get("retry")


@pytest.mark.asyncio
async def test_fast_stream_is_untouched(base_agent, mock_connection_manager):
    """A normal stream must not be slowed down or reported on."""
    _tune(base_agent, first=1.0, idle=1.0, heartbeat=0.5)
    stream = _HangingStream(
        [SimpleNamespace(type="text", text="a"), SimpleNamespace(type="text", text="b")], hang=False
    )

    received = [event.text async for event in base_agent._iter_stream_with_stall_guard(stream)]

    assert received == ["a", "b"]
    assert _status_events(mock_connection_manager) == []


def _release_on_wait(connection_manager: AsyncMock, release: asyncio.Event) -> None:
    async def on_event(event: AgentEvent, _workspace_id: str, _thread_id: str) -> None:
        if event.event_type == "llm_status_update" and event.content.get("stall"):
            release.set()

    connection_manager.broadcast_event.side_effect = on_event


@pytest.mark.asyncio
@pytest.mark.parametrize("mid_stream", [False, True])
@pytest.mark.parametrize(
    "chunk",
    [
        MessageChunk(type="text", text="ready"),
        MessageChunk(type="thinking", thinking="working"),
        MessageChunk(type="tool_use_delta", tool_call_delta={"input_delta": '{"path":'}),
        MessageChunk(type="ignore"),
    ],
    ids=["text", "thinking", "tool_arguments", "lifecycle"],
)
async def test_healthy_stream_waits_neutrally_and_reports_recovery(
    base_agent: BaseAgent, mock_connection_manager: AsyncMock, mid_stream: bool, chunk: MessageChunk
) -> None:
    _tune(base_agent, first=1.0, idle=1.0, heartbeat=0.01)
    release = asyncio.Event()
    _release_on_wait(mock_connection_manager, release)
    initial = MessageChunk(type="text", text="hello")

    async def stream() -> AsyncIterator[MessageChunk]:
        if mid_stream:
            yield initial
        await release.wait()
        yield chunk

    received = [event async for event in base_agent._iter_stream_with_stall_guard(stream())]

    assert received == ([initial, chunk] if mid_stream else [chunk])
    statuses = _status_events(mock_connection_manager)
    assert [event.content["message"] for event in statuses] == [
        "Waiting for response…",
        "Reading model response…",
    ]
    assert statuses[0].content["stage"] == ("stream" if mid_stream else "first_chunk")
    assert statuses[1].content["resumed"] is True
    assert all(event.content["status"] == "info" and not event.content.get("retry") for event in statuses)
    assert all(
        call.args[0].event_type != "llm_error" for call in mock_connection_manager.broadcast_event.await_args_list
    )


@pytest.mark.asyncio
async def test_healthy_setup_wait_reports_recovery(base_agent: BaseAgent, mock_connection_manager: AsyncMock) -> None:
    _tune(base_agent, first=1.0, idle=1.0, heartbeat=0.01)
    release = asyncio.Event()
    _release_on_wait(mock_connection_manager, release)
    chunk = MessageChunk(type="text", text="ready")

    @asynccontextmanager
    async def stream_context() -> AsyncGenerator[_HangingStream]:
        yield _HangingStream([chunk], hang=False)

    async def setup() -> AbstractAsyncContextManager[_HangingStream]:
        await release.wait()
        return stream_context()

    async with base_agent._open_stream_with_stall_guard(setup) as (stream, state):
        received = [event async for event in base_agent._iter_stream_with_stall_guard(stream, state=state)]

    assert received == [chunk]
    assert [event.content["message"] for event in _status_events(mock_connection_manager)] == [
        "Waiting for response…",
        "Reading model response…",
    ]


@pytest.mark.asyncio
async def test_failed_read_does_not_report_recovery(base_agent: BaseAgent, mock_connection_manager: AsyncMock) -> None:
    _tune(base_agent, first=1.0, idle=1.0, heartbeat=0.01)
    release = asyncio.Event()
    _release_on_wait(mock_connection_manager, release)

    class FailingStream(_HangingStream):
        async def __anext__(self) -> MessageChunk:
            await release.wait()
            raise RuntimeError("read failed")

    with pytest.raises(RuntimeError, match="read failed"):
        async for _ in base_agent._iter_stream_with_stall_guard(FailingStream()):
            pass

    assert [event.content["message"] for event in _status_events(mock_connection_manager)] == ["Waiting for response…"]
