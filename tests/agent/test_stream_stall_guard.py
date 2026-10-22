"""The stream stall guard: deterministic failure + a visible countdown.

A silent provider stream used to be bounded only by the transport read timeout,
which cannot tell "the endpoint never answered" from "a long gap mid-answer" and
says nothing to the user while it waits. The guard applies a stage-specific
deadline per event and reports heartbeats in between.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kolega_code.llm.exceptions import LLMStreamStalled


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


def _status_events(mock_connection_manager) -> list:
    events = []
    for call in mock_connection_manager.broadcast_event.await_args_list:
        event = call.args[0]
        if event.event_type == "llm_status_update" and event.content.get("stall"):
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
    assert content["message"] == "Connection issue — waiting for a response…"


@pytest.mark.asyncio
async def test_fast_stream_is_untouched(base_agent):
    """A normal stream must not be slowed down or reported on."""
    _tune(base_agent, first=1.0, idle=1.0, heartbeat=0.5)
    stream = _HangingStream(
        [SimpleNamespace(type="text", text="a"), SimpleNamespace(type="text", text="b")], hang=False
    )

    received = [event.text async for event in base_agent._iter_stream_with_stall_guard(stream)]

    assert received == ["a", "b"]
