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

    await agent.compress_history(trigger="auto")

    events = compaction_status_events(cm)
    phases = [e.content["phase"] for e in events]
    assert "started" in phases
    assert "error" in phases
    error_event = [e for e in events if e.content["phase"] == "error"][-1]
    assert error_event.content["trigger"] == "auto"
    assert error_event.content["reason"] == "llm_error"
    assert "boom" in error_event.content["message"]


def test_app_apply_compaction_status_toggles_indicator():
    # Exercise the dashboard toggle without constructing the full Textual app.
    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import StatusDashboardState

    class _FakeApp(KolegaCodeApp):
        def __init__(self):
            pass

    refreshed = []
    fake = _FakeApp()
    object.__setattr__(fake, "_status_state", StatusDashboardState())
    object.__setattr__(fake, "_refresh_status_dashboard", lambda: refreshed.append(True))

    KolegaCodeApp._apply_compaction_status(fake, {"phase": "started", "message": "Compacting…"})
    assert fake._status_state.is_compacting is True
    assert fake._status_state.compaction_message == "Compacting…"

    KolegaCodeApp._apply_compaction_status(fake, {"phase": "finished", "message": ""})
    assert fake._status_state.is_compacting is False
    assert fake._status_state.compaction_message == ""

    assert refreshed  # the dashboard was refreshed on each phase


def test_app_apply_compaction_status_adds_summary_collapsible():
    # On finish with a summary, a collapsible transcript entry is created.
    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import ConversationEntry, StatusDashboardState

    class _FakeApp(KolegaCodeApp):
        def __init__(self):
            pass

    added = []
    fake = _FakeApp()
    object.__setattr__(fake, "_status_state", StatusDashboardState())
    object.__setattr__(fake, "_refresh_status_dashboard", lambda: None)
    object.__setattr__(fake, "_add_conversation_entry", lambda entry: added.append(entry))

    KolegaCodeApp._apply_compaction_status(
        fake, {"phase": "finished", "message": "done", "summary": "## Goal\nDo the thing"}
    )

    assert len(added) == 1
    assert isinstance(added[0], ConversationEntry)
    assert added[0].kind == "compaction_summary"
    assert "Do the thing" in added[0].content


def test_app_resume_compaction_entry():
    # The resume helper builds a collapsible summary entry from session.compaction.
    from kolega_code.cli.app import KolegaCodeApp

    class _FakeApp(KolegaCodeApp):
        def __init__(self):
            pass

    def app_with(compaction):
        app = _FakeApp()
        object.__setattr__(app, "session", SimpleNamespace(compaction=compaction))
        return app

    assert KolegaCodeApp._resume_compaction_entry(app_with({})) is None
    assert KolegaCodeApp._resume_compaction_entry(app_with({"summary": "", "compacted_through": 3})) is None
    assert KolegaCodeApp._resume_compaction_entry(app_with({"summary": "S", "compacted_through": 0})) is None

    entry = KolegaCodeApp._resume_compaction_entry(app_with({"summary": "RESTORED SUMMARY", "compacted_through": 5}))
    assert entry is not None
    assert entry.kind == "compaction_summary"
    assert "RESTORED SUMMARY" in entry.content


def test_app_apply_compaction_status_surfaces_auto_error_once_until_recovery():
    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import ConversationEntry, StatusDashboardState

    class _FakeApp(KolegaCodeApp):
        def __init__(self):
            pass

    added: list[ConversationEntry] = []
    notified: list[tuple[str, str]] = []
    drift_checks: list[bool] = []
    fake = _FakeApp()
    object.__setattr__(fake, "_status_state", StatusDashboardState())
    object.__setattr__(fake, "_last_compaction_error", None)
    object.__setattr__(fake, "_refresh_status_dashboard", lambda: None)
    object.__setattr__(fake, "_add_conversation_entry", lambda entry: added.append(entry))
    object.__setattr__(fake, "_notify_user", lambda msg, severity="information": notified.append((msg, severity)))
    object.__setattr__(fake, "_check_install_drift", lambda: drift_checks.append(True))

    payload = {
        "phase": "error",
        "trigger": "auto",
        "reason": "llm_error",
        "message": "Failed to compress message history: TemplateNotFound",
    }
    KolegaCodeApp._apply_compaction_status(fake, payload)
    # Consecutive identical errors are deduplicated so multi-turn loops don't spam.
    KolegaCodeApp._apply_compaction_status(fake, payload)

    assert len(added) == 1
    assert added[0].kind == "system"
    assert added[0].tone == "warning"
    assert "Automatic compression failed:" in added[0].content
    assert "TemplateNotFound" in added[0].content
    assert len(notified) == 1
    assert notified[0][1] == "warning"
    assert len(drift_checks) == 2

    # A successful compaction resets dedup state so a later regression surfaces again.
    KolegaCodeApp._apply_compaction_status(fake, {"phase": "finished", "message": "done", "summary": ""})
    KolegaCodeApp._apply_compaction_status(fake, payload)
    assert len(added) == 2


def test_app_apply_compaction_status_skips_manual_and_noop_errors():
    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import ConversationEntry, StatusDashboardState

    class _FakeApp(KolegaCodeApp):
        def __init__(self):
            pass

    added: list[ConversationEntry] = []
    fake = _FakeApp()
    object.__setattr__(fake, "_status_state", StatusDashboardState())
    object.__setattr__(fake, "_last_compaction_error", None)
    object.__setattr__(fake, "_refresh_status_dashboard", lambda: None)
    object.__setattr__(fake, "_add_conversation_entry", lambda entry: added.append(entry))

    # Manual /compress already prints its failure via the command response.
    KolegaCodeApp._apply_compaction_status(
        fake,
        {"phase": "error", "trigger": "manual", "reason": "llm_error", "message": "boom"},
    )
    # Benign auto no-ops (e.g. before zero-tail fallback) shouldn't alarm the user.
    KolegaCodeApp._apply_compaction_status(
        fake,
        {"phase": "error", "trigger": "auto", "reason": "nothing_to_summarize", "message": "Already compact"},
    )
    KolegaCodeApp._apply_compaction_status(
        fake,
        {"phase": "error", "trigger": "auto", "reason": "too_few", "message": "Too few"},
    )

    assert added == []
