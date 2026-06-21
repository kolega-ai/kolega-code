"""Tests that compaction emits status events and the TUI toggles its indicator."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from .compaction_helpers import FakeLLM, build_agent, compaction_status_events, long_history


@pytest.mark.asyncio
async def test_compress_history_emits_started_then_finished(tmp_path):
    agent, cm = build_agent(tmp_path, llm=FakeLLM(token_script=[100]))
    agent.history = long_history(6)

    await agent.compress_history()

    events = compaction_status_events(cm)
    phases = [e.content["phase"] for e in events]
    assert phases[0] == "started"
    assert phases[-1] == "finished"
    # The finished event carries the summary text so the UI can show it.
    assert events[-1].content["summary"].strip()


@pytest.mark.asyncio
async def test_compress_history_emits_error_phase_on_failure(tmp_path):
    llm = FakeLLM(token_script=[100])
    llm.stream = AsyncMock(side_effect=RuntimeError("boom"))
    agent, cm = build_agent(tmp_path, llm=llm)
    agent.history = long_history(6)

    await agent.compress_history()

    phases = [e.content["phase"] for e in compaction_status_events(cm)]
    assert "started" in phases
    assert "error" in phases


def test_app_apply_compaction_status_toggles_indicator():
    # Exercise the dashboard toggle without constructing the full Textual app.
    from kolega_code.cli.app import KolegaCodeApp, StatusDashboardState

    refreshed = []
    fake = SimpleNamespace(
        _status_state=StatusDashboardState(),
        _refresh_status_dashboard=lambda: refreshed.append(True),
    )

    KolegaCodeApp._apply_compaction_status(fake, {"phase": "started", "message": "Compacting…"})
    assert fake._status_state.is_compacting is True
    assert fake._status_state.compaction_message == "Compacting…"

    KolegaCodeApp._apply_compaction_status(fake, {"phase": "finished", "message": ""})
    assert fake._status_state.is_compacting is False
    assert fake._status_state.compaction_message == ""

    assert refreshed  # the dashboard was refreshed on each phase


def test_app_apply_compaction_status_adds_summary_collapsible():
    # On finish with a summary, a collapsible transcript entry is created.
    from kolega_code.cli.app import ConversationEntry, KolegaCodeApp, StatusDashboardState

    added = []
    fake = SimpleNamespace(
        _status_state=StatusDashboardState(),
        _refresh_status_dashboard=lambda: None,
        _add_conversation_entry=lambda entry: added.append(entry),
    )

    KolegaCodeApp._apply_compaction_status(
        fake, {"phase": "finished", "message": "done", "summary": "## Goal\nDo the thing"}
    )

    assert len(added) == 1
    assert isinstance(added[0], ConversationEntry)
    assert added[0].kind == "compaction_summary"
    assert "Do the thing" in added[0].content
