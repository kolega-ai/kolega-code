# ruff: noqa: F401,F811,E402
from pathlib import Path
import asyncio
import json
import time

import pytest

from kolega_code.cli.tui import agent_runtime as agent_runtime_module

from kolega_code.config import ModelProvider
from kolega_code.llm.exceptions import (
    LLMBillingError,
    LLMAuthenticationError,
    LLMContextWindowExceededError,
    LLMError,
    LLMInternalServerError,
)
from kolega_code.llm.models import Message, TextBlock, ToolCall, ToolResult
from kolega_code.events import AgentEvent
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.cli.config import build_agent_config, config_summary
from kolega_code.cli.provider_registry import (
    DEEPSEEK_DEFAULT_MODEL,
    MOONSHOT_K26_MODEL,
    UI_DEFAULT_MODEL,
    UI_DEFAULT_PROVIDER,
)
from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.settings import CliSettings, SettingsStore

from ._app_test_utils import (
    _build_mention_test_app,
    _build_sub_agent_test_app,
    _sub_agent_context_event,
    _sub_agent_entries,
    _sub_agent_event,
    _workflow_event,
    build_test_config,
    extension_by_name,
    first_text_styles,
    question_payload,
    renderable_text,
)

def test_turn_state_styles_do_not_depend_on_content_text() -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import theme
    from kolega_code.cli.tui.state import TurnState, turn_state_color

    # Resolves role names against the active theme's live Color attributes.
    assert turn_state_color(TurnState.ERROR) == theme.Color.ERROR
    assert turn_state_color(TurnState.STOPPED) == theme.Color.WARNING
    assert turn_state_color(TurnState.STOPPING) == theme.Color.WARNING
    assert turn_state_color(TurnState.IDLE) == theme.Color.SUCCESS
    assert turn_state_color(TurnState.GENERATING) == theme.Color.ACCENT  # falls back to accent

@pytest.mark.asyncio
async def test_progress_entry_tone_drives_styling_not_prose(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import theme
    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import ConversationEntry, TurnState

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        # Prose mentioning "error" with a warning tone must not render as an error
        warning_entry = ConversationEntry(
            kind="progress", content="Stopped before the error handler ran", complete=True, tone="warning"
        )
        rendered = app._format_conversation_entry(warning_entry)
        assert theme.Color.WARNING in first_text_styles(rendered)
        assert theme.Color.ERROR not in first_text_styles(rendered)

        error_entry = ConversationEntry(kind="progress", content="All good otherwise", complete=True, tone="error")
        rendered = app._format_conversation_entry(error_entry)
        assert theme.Color.ERROR in first_text_styles(rendered)

        # Explicit state drives the dashboard, not content keywords
        app._turn_active = True
        app._begin_turn_progress()
        app._finish_turn_progress("Wrapped up without issue", TurnState.STOPPED)
        assert app._status_state.turn_state is TurnState.STOPPED

@pytest.mark.asyncio
async def test_textual_app_shows_working_progress_during_active_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Static

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer
    from kolega_code.cli.messages import COMPOSER_PLACEHOLDER

    started = asyncio.Event()
    release = asyncio.Event()

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

        async def process_message_stream(self, message):
            started.set()
            await release.wait()
            yield {"type": "response", "content": "done", "complete": True, "uuid": "response-1"}

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)
    now = 100.0
    monkeypatch.setattr(app, "_now", lambda: now)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        turn_status = app.query_one("#turn_status", Static)
        assert turn_status.display is False

        task = asyncio.create_task(app._process_message("hi"))
        await started.wait()

        progress_entries = [entry for entry in app.conversation_entries if entry.kind == "progress"]
        assert progress_entries == []
        assert composer.placeholder == COMPOSER_PLACEHOLDER
        assert composer.disabled is True
        assert "Working…" in str(turn_status.render())
        assert "0s" in str(turn_status.render())

        now = 103.0
        app._render_event(
            AgentEvent(event_type="status_update", sender="coder", content={"text": "Indexing workspace"})
        )
        assert composer.placeholder == COMPOSER_PLACEHOLDER
        app._refresh_turn_status_strip()
        assert "Indexing workspace" in str(turn_status.render())
        assert "3s" in str(turn_status.render())

        now = 423.0
        release.set()
        await task

        assert [entry for entry in app.conversation_entries if entry.kind == "progress"] == []
        assert composer.placeholder == COMPOSER_PLACEHOLDER
        assert composer.disabled is False
        assert "Done in 5m 23s" in str(turn_status.render())

@pytest.mark.asyncio
async def test_confirmations_surface_as_logs_without_toasts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        logged: list[tuple[str, str]] = []

        def fake_notify(message, *, severity="information", title=None, **kwargs):
            raise AssertionError("TUI notices should not show transient popups")

        original_log_status = app._log_status

        def spy_log_status(text, level="info"):
            logged.append((text, level))
            original_log_status(text, level)

        monkeypatch.setattr(app, "notify", fake_notify)
        monkeypatch.setattr(app, "_log_status", spy_log_status)

        await app._set_interaction_mode("plan")

        assert ("Switched to plan mode.", "ok") in logged  # diagnostic record kept

        # Blockers are logged as warnings without transient popups.
        app._turn_active = True
        await app.action_toggle_interaction_mode()
        assert ("Stop the current turn before switching modes.", "warn") in logged

@pytest.mark.asyncio
async def test_turn_status_strip_shows_spinner_and_outcome_glyph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli import theme
    from kolega_code.cli.tui.state import TurnState

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    now = 0.0
    monkeypatch.setattr(app, "_now", lambda: now)

    async with app.run_test():
        app._begin_turn_progress()
        content = app._turn_status_content()
        assert any(frame in content for frame in theme.spinner_frames())
        assert "Working…" in content

        now = 12.0
        app._finish_turn_progress("Finished.", TurnState.IDLE)
        content = app._turn_status_content()
        assert theme.g(theme.Glyph.CHECK) in content
        assert "Done in 12s" in content

@pytest.mark.asyncio
async def test_status_dashboard_context_note_uses_alert_level(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from kolega_code.agent import AgentEvent
    from kolega_code.cli import theme

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():

        def context_event(alert_level):
            return AgentEvent(
                event_type="llm_context_update",
                sender="coder",
                content={
                    "input_tokens": 1000,
                    "max_tokens": 2000,
                    "usage_percentage": 50.0,
                    "compression_threshold": 80.0,
                    "alert_level": alert_level,
                    "message": "Context is getting large.",
                },
            )

        app._render_event(context_event("info"))
        dashboard = app._format_status_dashboard()
        warn = theme.Color.WARNING
        assert f"[{warn}]Context is getting large.[/{warn}]" in dashboard

        app._render_event(context_event("critical"))
        dashboard = app._format_status_dashboard()
        err = theme.Color.ERROR
        assert f"[{err}]Context is getting large.[/{err}]" in dashboard

@pytest.mark.asyncio
async def test_planning_sidebar_marks_empty_states(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Markdown

    from kolega_code.cli.messages import PLAN_EMPTY_MESSAGE

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        plan_md = app.query_one("#planning_plan_markdown", Markdown)
        assert plan_md.source == PLAN_EMPTY_MESSAGE
        assert plan_md.has_class("empty-state")

        app._latest_plan = "# Plan\n\n- do the thing"
        app._refresh_planning_sidebar()

        assert plan_md.source == "# Plan\n\n- do the thing"
        assert not plan_md.has_class("empty-state")

@pytest.mark.asyncio
async def test_tab_activity_label_changes_only_on_state_transitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import TabbedContent

    from kolega_code.cli import theme
    from kolega_code.cli.tui.constants import TAB_BASE_LABELS
    from kolega_code.cli.theme import Glyph

    app = _build_sub_agent_test_app(tmp_path, monkeypatch, show_logs=True)

    async with app.run_test():
        tabs = app.query_one("#events", TabbedContent)
        logs_tab = tabs.get_tab("logs_pane")
        expected_marked = f"{TAB_BASE_LABELS['logs_pane']} {theme.g(Glyph.STATUS)}"

        app._mark_tab_activity("logs_pane")
        marked_label = logs_tab.label
        assert str(marked_label) == expected_marked

        app._mark_tab_activity("logs_pane")
        assert logs_tab.label is marked_label

        app._clear_tab_activity("logs_pane")
        cleared_label = logs_tab.label
        assert str(cleared_label) == TAB_BASE_LABELS["logs_pane"]

        app._clear_tab_activity("logs_pane")
        assert logs_tab.label is cleared_label

@pytest.mark.asyncio
async def test_repeated_progress_updates_refresh_status_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.state import TurnState

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        refresh_count = 0

        def refresh_status_dashboard() -> None:
            nonlocal refresh_count
            refresh_count += 1

        monkeypatch.setattr(app, "_refresh_status_dashboard", refresh_status_dashboard)
        app._turn_status_text = ""
        app._status_state.turn_state = TurnState.IDLE
        app._status_state.activity = "Ready"

        app._update_progress("Reading response", complete=False, state=TurnState.GENERATING)
        app._update_progress("Reading response", complete=False, state=TurnState.GENERATING)
        app._update_progress("Reading response", complete=False, state=TurnState.GENERATING)

        assert refresh_count == 1

        app._update_progress("Reading response", complete=False, state=TurnState.THINKING)

        assert refresh_count == 2

