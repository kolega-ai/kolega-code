"""A retrying connection must be visible in the transcript, not just the status line.

The status strip is overwritten by the next stream delta, so a flapping connection
used to read as one long freeze. Retry notices become transcript entries; stall
heartbeats stay in the strip so they cannot spam the conversation.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from kolega_code.agent.baseagent import BaseAgent
from kolega_code.events import AgentEvent, AgentEventEmitter, KnownEventType
from kolega_code.llm.models import MessageChunk

from ._app_test_utils import _build_sub_agent_test_app


def _status(content: dict) -> AgentEvent:
    return AgentEvent(sender="coder", event_type=KnownEventType.LLM_STATUS_UPDATE, content=content)


def _transcript_notices(app):
    return [entry for entry in app.conversation_entries if entry.kind == "system"]


@pytest.mark.asyncio
async def test_retry_status_becomes_a_transcript_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        app._begin_turn_progress("Reading model response…")
        app._render_event(
            _status(
                {
                    "status": "info",
                    "message": "Response timed out — retrying (1/3) in 8s…",
                    "retry": True,
                    "attempt": 1,
                    "cap": 3,
                    "stalled": True,
                }
            )
        )
        await pilot.pause()

        notices = _transcript_notices(app)
        assert notices, "retry was not surfaced in the transcript"
        assert "retrying (1/3)" in notices[-1].content
        assert notices[-1].tone == "warning"


@pytest.mark.asyncio
async def test_stall_heartbeat_stays_out_of_the_transcript(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        app._begin_turn_progress("Reading model response…")
        app._render_event(
            _status(
                {
                    "status": "info",
                    "message": "Waiting for response…",
                    "stall": True,
                    "seconds_since_data": 75.0,
                    "timeout_s": 90.0,
                }
            )
        )
        await pilot.pause()

        assert _transcript_notices(app) == []
        assert app._turn_status_text == "Waiting for response…"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chunk", "expected"),
    [
        (MessageChunk(type="ignore"), "Reading model response…"),
        (MessageChunk(type="thinking", thinking="working"), "Thinking…"),
    ],
    ids=["lifecycle", "thinking"],
)
async def test_queued_stream_recovery_clears_only_the_waiting_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, chunk: MessageChunk, expected: str
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    release = asyncio.Event()
    waiting_labels: list[str] = []
    broadcast = app.connection_manager.broadcast_event

    async def defer_events() -> None:
        await asyncio.Event().wait()

    async def broadcast_and_release(event: AgentEvent, workspace_id: str, thread_id: str) -> None:
        await broadcast(event, workspace_id, thread_id)
        if event.content.get("stall"):
            await app._drain_pending_events()
            waiting_labels.append(app._turn_status_text)
            release.set()

    monkeypatch.setattr(app, "_consume_events", defer_events)
    monkeypatch.setattr(app.connection_manager, "broadcast_event", broadcast_and_release)
    guard = object.__new__(BaseAgent)
    guard._stream_first_chunk_timeout = 1.0
    guard._stream_idle_timeout = 1.0
    guard._stream_heartbeat_seconds = 0.01
    guard._llm_retry_budget_started = None
    guard.emitter = AgentEventEmitter(app.recording_connection_manager, "workspace", "thread", "coder")

    async def chunks() -> AsyncIterator[MessageChunk]:
        await release.wait()
        yield chunk

    async def stream() -> AsyncIterator[dict[str, str]]:
        async for message in guard._iter_stream_with_stall_guard(chunks()):
            if message.type == "thinking":
                yield {"type": "thinking", "content": message.thinking or ""}

    async with app.run_test() as pilot:
        app._begin_turn_progress("Reading model response…")
        await app._consume_agent_stream(stream())
        assert waiting_labels == ["Waiting for response…"]
        await app._drain_pending_events()
        await pilot.pause()

        assert app._turn_status_text == expected
        assert _transcript_notices(app) == []
