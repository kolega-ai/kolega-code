"""A retrying connection must be visible in the transcript, not just the status line.

The status strip is overwritten by the next stream delta, so a flapping connection
used to read as one long freeze. Retry notices become transcript entries; stall
heartbeats stay in the strip so they cannot spam the conversation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kolega_code.events import AgentEvent, KnownEventType

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
                    "message": "Connection issue — stream stalled. Retrying (1/3) in 8s…",
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
        assert "Retrying (1/3)" in notices[-1].content
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
                    "message": "Waiting for the first response — no data for 1m 15s (gives up at 1m 30s)",
                    "stall": True,
                    "seconds_since_data": 75.0,
                    "timeout_s": 90.0,
                }
            )
        )
        await pilot.pause()

        assert _transcript_notices(app) == []
        assert "no data" in app._turn_status_text.lower()
