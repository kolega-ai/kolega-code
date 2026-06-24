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
async def test_textual_app_keeps_command_c_for_screen_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

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
        cancel_binding = next(binding for binding in app.BINDINGS if binding.action == "cancel_generation")
        assert cancel_binding.key == "ctrl+c"
        assert all("super+c" not in binding.key for binding in app.BINDINGS)


@pytest.mark.asyncio
async def test_textual_app_merges_streamed_response_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer
    from kolega_code.cli.messages import COMPOSER_PLACEHOLDER

    chunks = [
        {"type": "response", "content": "hello ", "complete": False, "uuid": "response-1"},
        {"type": "response", "content": "world", "complete": False, "uuid": "response-1"},
        {"type": "response", "content": "", "complete": True, "uuid": "response-1"},
    ]

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
            for chunk in chunks:
                yield chunk

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        await app._process_message("hi")

        assistant_entries = [entry for entry in app.conversation_entries if entry.kind == "assistant"]
        assert len(assistant_entries) == 1
        assert assistant_entries[0].content == "hello world"
        assert assistant_entries[0].complete is True
        assert [entry for entry in app.conversation_entries if entry.kind == "progress"] == []
        assert app.query_one("#composer", ChatComposer).placeholder == COMPOSER_PLACEHOLDER


@pytest.mark.asyncio
async def test_textual_app_merges_streamed_thinking_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

    chunks = [
        {"type": "thinking", "content": "checking ", "complete": False, "uuid": "thinking-1"},
        {"type": "thinking", "content": "context", "complete": False, "uuid": "thinking-1"},
        {"type": "thinking", "content": "", "complete": True, "uuid": "thinking-1"},
        {"type": "response", "content": "done", "complete": True, "uuid": "response-1"},
    ]

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
            for chunk in chunks:
                yield chunk

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        await app._process_message("hi")

        thinking_entries = [entry for entry in app.conversation_entries if entry.kind == "thinking"]
        assert len(thinking_entries) == 1
        assert thinking_entries[0].content == "checking context"
        assert thinking_entries[0].complete is True


@pytest.mark.asyncio
async def test_textual_app_formats_thinking_as_italic_chat_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import ConversationEntry

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
        formatted = app._format_conversation_entry(
            ConversationEntry(kind="thinking", content="inspect [red]markup[/red]", complete=False)
        )
        rendered = renderable_text(formatted)

        assert "Thinking" in rendered
        assert "[red]markup[/red]" in rendered
        assert "…" in rendered  # streaming indicator in the header
        assert any("italic" in style and "dim" in style for style in first_text_styles(formatted))


@pytest.mark.asyncio
async def test_textual_app_renders_one_widget_per_chat_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import ConversationEntryWidget

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

    async with app.run_test() as pilot:
        app.conversation_entries = [
            ConversationEntry(kind="user", content="first"),
            ConversationEntry(kind="assistant", content="second", complete=False),
            ConversationEntry(kind="user", content="third"),
        ]
        app._render_conversation()
        await pilot.pause()

        widgets = list(app.query(ConversationEntryWidget))
        assert len(widgets) == 3
        assert [widget.entry.content for widget in widgets] == ["first", "second", "third"]
        assert widgets[0].has_class("entry-user")
        assert widgets[1].has_class("entry-assistant")

        # Streaming into an entry updates its widget in place without remounting
        app.conversation_entries[1].content = "second updated"
        app._invalidate_conversation(app.conversation_entries[1])
        app._flush_conversation_render()
        await pilot.pause()

        same_widgets = list(app.query(ConversationEntryWidget))
        assert len(same_widgets) == 3
        assert same_widgets[1] is widgets[1]
        assert "second updated" in renderable_text(same_widgets[1]._formatted)


@pytest.mark.asyncio
async def test_conversation_render_skips_detached_view_during_teardown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A coalesced render timer can fire while the app is tearing down. The view is
    detached from the DOM (is_attached is False) but query_one still resolves it, so
    mounting into it used to raise MountError and crash the CLI on exit."""
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import ConversationEntryWidget, ConversationView

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

    async with app.run_test() as pilot:
        app.conversation_entries = [ConversationEntry(kind="user", content="first")]
        app._render_conversation()
        await pilot.pause()
        assert len(app.query(ConversationEntryWidget)) == 1

        # Queue a new entry so a flush would reach view.mount(...), then simulate the
        # exit-time race by detaching the view (is_attached False) without removing it
        # from the DOM, so query_one still resolves it.
        app.conversation_entries.append(ConversationEntry(kind="assistant", content="late", complete=False))
        app._render_pending = True
        ConversationView.is_attached = property(lambda self: False)
        try:
            app._flush_conversation_render()  # must not raise (pre-fix: MountError)
            app._render_conversation()  # must not raise
        finally:
            del ConversationView.is_attached

        # The detached render was skipped, so nothing new was mounted.
        assert len(app.query(ConversationEntryWidget)) == 1


@pytest.mark.asyncio
async def test_conversation_flush_uses_dirty_entry_fast_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import ConversationEntryWidget

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        app.conversation_entries = [ConversationEntry(kind="user", content=f"message {index}") for index in range(50)]
        app._render_conversation()
        await pilot.pause()

        widgets = list(app.query(ConversationEntryWidget))
        assert len(widgets) == 50

        def fail_full_rebuild() -> None:
            raise AssertionError("dirty-entry refresh should not rebuild the transcript")

        monkeypatch.setattr(app, "_render_conversation", fail_full_rebuild)
        entry = app.conversation_entries[25]
        entry.content = "message 25 updated"
        app._dirty_entry_ids.add(entry.entry_id)
        app._render_pending = True

        app._flush_conversation_render()
        await pilot.pause()

        refreshed = list(app.query(ConversationEntryWidget))
        assert refreshed == widgets
        assert "message 25 updated" in renderable_text(refreshed[25]._formatted)


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
async def test_conversation_entry_widget_skips_unchanged_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import ConversationEntryWidget

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        app.conversation_entries = [ConversationEntry(kind="assistant", content="stable")]
        app._render_conversation()
        await pilot.pause()

        widget = app.query(ConversationEntryWidget).last()
        updates = 0

        def update(renderable) -> None:
            nonlocal updates
            updates += 1

        monkeypatch.setattr(widget, "update", update)
        widget.refresh_content()
        assert updates == 0

        widget.entry.content = "changed"
        widget.refresh_content()
        assert updates == 1


@pytest.mark.asyncio
async def test_conversation_entry_widget_extracts_plain_selected_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.selection import Selection

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import ConversationEntryWidget

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

    async with app.run_test() as pilot:
        app.conversation_entries = [ConversationEntry(kind="assistant", content="copy this")]
        app._render_conversation()
        await pilot.pause()

        widget = app.query(ConversationEntryWidget).last()
        selected = widget.get_selection(Selection(None, None))

        assert selected is not None
        text, ending = selected
        assert ending == "\n"
        assert "Agent" in text
        assert "copy this" in text
        assert "\x1b" not in text
        assert "[bold]" not in text


@pytest.mark.asyncio
async def test_conversation_entry_supports_mouse_drag_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual import events

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import ConversationEntryWidget

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

    async with app.run_test() as pilot:
        app.conversation_entries = [ConversationEntry(kind="assistant", content="select this text")]
        app._render_conversation()
        await pilot.pause()

        widget = app.query(ConversationEntryWidget).last()

        await pilot.mouse_down(widget, offset=(0, 1))
        await pilot._post_mouse_events([events.MouseMove], widget, offset=(19, 1), button=1)
        await pilot.mouse_up(widget, offset=(19, 1))

        selected_text = app.screen.get_selected_text()
        assert selected_text is not None
        assert selected_text.strip() == "select this text"


@pytest.mark.asyncio
async def test_conversation_entry_selection_styles_rendered_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.geometry import Offset
    from textual.selection import Selection

    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import ConversationEntryWidget

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test(size=(100, 40)) as pilot:
        app.conversation_entries = [
            ConversationEntry(kind="user", content="select this user text"),
            ConversationEntry(kind="assistant", content="select this agent text"),
            ConversationEntry(kind="assistant", content="select this streaming text", complete=False),
        ]
        app._render_conversation()
        await pilot.pause()

        widgets = list(app.query(ConversationEntryWidget))[-3:]
        for widget in widgets:
            app.screen.selections = {widget: Selection(Offset(0, 1), Offset(12, 1))}
            strip = widget.render_line(1)
            selection_bg = widget.selection_style.bgcolor

            assert any(
                segment.style is not None
                and segment.style.bgcolor == selection_bg
                and segment.style.meta.get("offset") is not None
                for segment in strip
            )


@pytest.mark.asyncio
async def test_conversation_entry_selection_preserves_text_foreground(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.geometry import Offset
    from textual.selection import Selection

    from kolega_code.cli import theme
    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import ConversationEntryWidget

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test(size=(100, 40)) as pilot:
        app.conversation_entries = [
            ConversationEntry(kind="user", content="select this user text"),
        ]
        app._render_conversation()
        await pilot.pause()

        # Exercise every theme: the fix must keep selected text readable regardless
        # of the active palette. Theme state is process-global, so restore it after.
        try:
            for theme_name in theme.available_themes():
                app._apply_theme(theme_name)
                await pilot.pause()
                widget = app.query(ConversationEntryWidget).last()

                # Foreground colors present on line 1 with no selection.
                app.screen.selections = {}
                plain_strip = widget.render_line(1)
                plain_colors = {
                    segment.style.color for segment in plain_strip if segment.text and segment.style is not None
                }
                assert plain_colors, f"expected colored content on line 1 for {theme_name}"

                # Select a span on line 1 and re-render.
                app.screen.selections = {widget: Selection(Offset(0, 1), Offset(20, 1))}
                selected_strip = widget.render_line(1)
                selection_style = widget.selection_style
                selection_bg = selection_style.bgcolor

                highlighted = [
                    segment
                    for segment in selected_strip
                    if segment.text and segment.style is not None and segment.style.bgcolor == selection_bg
                ]
                assert highlighted, f"expected highlighted segments for {theme_name}"

                # The selection must not blank out the text: every highlighted segment
                # keeps its original foreground (one seen unselected) and never the
                # transparent selection foreground.
                for segment in highlighted:
                    assert segment.style.color in plain_colors, f"selection wiped the text foreground for {theme_name}"
                    if selection_style.color is not None:
                        assert segment.style.color != selection_style.color, (
                            f"selection foreground overrode text color for {theme_name}"
                        )
        finally:
            theme.apply_theme(theme.DEFAULT_THEME_NAME)


@pytest.mark.asyncio
async def test_conversation_selection_can_start_in_blank_separator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual import events

    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import ConversationEntryWidget

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        app.conversation_entries = [
            ConversationEntry(kind="user", content="first"),
            ConversationEntry(kind="assistant", content="second", complete=False),
        ]
        app._render_conversation()
        await pilot.pause()

        first, second = list(app.query(ConversationEntryWidget))[-2:]
        separator_y = first.region.height - 1

        await pilot.mouse_down(first, offset=(0, separator_y))
        await pilot._post_mouse_events([events.MouseMove], second, offset=(20, 1), button=1)
        await pilot.mouse_up(second, offset=(20, 1))

        selected_text = app.screen.get_selected_text()
        assert selected_text is not None
        assert "second" in selected_text


@pytest.mark.asyncio
async def test_conversation_selection_can_start_after_line_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from textual import events

    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import ConversationEntryWidget

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        app.conversation_entries = [
            ConversationEntry(kind="user", content="first"),
            ConversationEntry(kind="assistant", content="second", complete=False),
        ]
        app._render_conversation()
        await pilot.pause()

        first, second = list(app.query(ConversationEntryWidget))[-2:]

        await pilot.mouse_down(first, offset=(30, 1))
        await pilot._post_mouse_events([events.MouseMove], second, offset=(20, 1), button=1)
        await pilot.mouse_up(second, offset=(20, 1))

        selected_text = app.screen.get_selected_text()
        assert selected_text is not None
        assert "second" in selected_text


@pytest.mark.asyncio
async def test_conversation_selection_spans_multiple_messages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from textual import events

    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import ConversationEntryWidget

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        app.conversation_entries = [
            ConversationEntry(kind="user", content="first message"),
            ConversationEntry(kind="assistant", content="second message", complete=False),
        ]
        app._render_conversation()
        await pilot.pause()

        first, second = list(app.query(ConversationEntryWidget))[-2:]

        await pilot.mouse_down(first, offset=(0, 1))
        await pilot._post_mouse_events([events.MouseMove], second, offset=(20, 1), button=1)
        await pilot.mouse_up(second, offset=(20, 1))

        selected_text = app.screen.get_selected_text()
        assert selected_text is not None
        assert "first message" in selected_text
        assert "second message" in selected_text
        assert selected_text.index("first message") < selected_text.index("second message")


@pytest.mark.asyncio
async def test_collapsed_tool_title_supports_drag_selection_and_toggle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual import events
    from textual.widgets import Collapsible
    from textual.widgets._collapsible import CollapsibleTitle

    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import ToolEntryWidget

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        app.conversation_entries = [
            ConversationEntry(
                kind="tool_result",
                content="preview text",
                full_content="full text",
                tool_name="read_file",
            )
        ]
        app._render_conversation()
        await pilot.pause()

        widget = app.query(ToolEntryWidget).last()
        collapsible = widget.query_one(Collapsible)
        title = widget.query_one(CollapsibleTitle)

        await pilot.mouse_down(title, offset=(1, 0))
        await pilot._post_mouse_events([events.MouseMove], title, offset=(20, 0), button=1)
        await pilot.mouse_up(title, offset=(20, 0))

        selected_text = app.screen.get_selected_text()
        assert selected_text is not None
        assert "read_file" in selected_text

        await pilot.click(title, offset=(1, 0))
        await pilot.pause()
        assert collapsible.collapsed is False

        title.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert collapsible.collapsed is True


@pytest.mark.asyncio
async def test_expanded_tool_body_line_start_selection_copies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from textual import events
    from textual.widgets import Collapsible, Static
    from textual.widgets._collapsible import CollapsibleTitle

    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import ToolEntryWidget

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    copied: list[str] = []
    monkeypatch.setattr(app, "copy_to_clipboard", copied.append)

    async with app.run_test(size=(100, 40)) as pilot:
        app.conversation_entries = [
            ConversationEntry(
                kind="tool_result",
                content="preview text",
                full_content="alpha line\nbeta line\ngamma line",
                tool_name="read_file",
            )
        ]
        app._render_conversation()
        await pilot.pause()

        widget = app.query(ToolEntryWidget).last()
        title = widget.query_one(CollapsibleTitle)
        await pilot.click(title, offset=(1, 0))
        await pilot.pause()

        body = widget.query_one(".tool-body", Static)
        assert widget.query_one(Collapsible).collapsed is False
        assert body.region.x == widget.region.x + 3

        body_y = body.region.y - widget.region.y
        await pilot.mouse_down(widget, offset=(0, body_y))
        await pilot._post_mouse_events([events.MouseMove], widget, offset=(11, body_y + 1), button=1)
        await pilot.mouse_up(widget, offset=(11, body_y + 1))

        selected_text = app.screen.get_selected_text()
        assert selected_text is not None
        assert selected_text == "alpha line\nbeta line"

        await pilot.press("super+c")
        assert copied == ["alpha line\nbeta line"]


@pytest.mark.asyncio
async def test_command_c_copies_selected_chat_text_to_macos_clipboard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.selection import Selection

    from kolega_code.cli import app as app_module
    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import ConversationEntryWidget

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

    pbcopy_calls: list[dict] = []

    def fake_run(args, *, input, text, check):
        pbcopy_calls.append({"args": args, "input": input, "text": text, "check": check})

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)
    monkeypatch.setattr(app_module.sys, "platform", "darwin")
    monkeypatch.setattr(app_module.subprocess, "run", fake_run)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        app.conversation_entries = [ConversationEntry(kind="assistant", content="copy this")]
        app._render_conversation()
        await pilot.pause()

        widget = app.query(ConversationEntryWidget).last()
        app.screen.selections = {widget: Selection(None, None)}

        await pilot.press("super+c")

        assert "copy this" in app.clipboard
        assert "\x1b" not in app.clipboard
        assert len(pbcopy_calls) == 1
        assert pbcopy_calls[0]["args"] == ["pbcopy"]
        assert pbcopy_calls[0]["input"] == app.clipboard


@pytest.mark.asyncio
async def test_textual_app_formats_agent_and_tool_chat_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import ConversationEntry

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
        assistant = app._format_conversation_entry(ConversationEntry(kind="assistant", content="hello", complete=False))
        tool_call = app._format_conversation_entry(
            ConversationEntry(
                kind="tool_call",
                content="inspect [red]markup[/red]\nthen continue",
                tool_name="read_file",
                complete=False,
            )
        )
        tool_result = app._format_conversation_entry(
            ConversationEntry(kind="tool_result", content="completed\nok", tool_name="read_file")
        )
        tool_error = app._format_conversation_entry(
            ConversationEntry(kind="tool_error", content="Permission denied", tool_name="write_file")
        )
        assistant_text = renderable_text(assistant)
        tool_call_text = renderable_text(tool_call)
        tool_result_text = renderable_text(tool_result)
        tool_error_text = renderable_text(tool_error)

        assert "● Agent" in assistant_text
        assert "Kolega" not in assistant_text
        assert "⏺ read_file" in tool_call_text
        assert "· running" in tool_call_text
        assert "inspect [red]markup[/red]" in tool_call_text
        assert "then continue" in tool_call_text
        assert "⏺ read_file" in tool_result_text
        assert "· done" in tool_result_text
        assert "⏺ write_file" in tool_error_text
        assert "· failed" in tool_error_text


@pytest.mark.asyncio
async def test_textual_app_ignores_empty_final_response_without_existing_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer
    from kolega_code.cli.messages import COMPOSER_PLACEHOLDER

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
            yield {"type": "response", "content": "", "complete": True, "uuid": "response-empty"}

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        await app._process_message("hi")

        assert [entry for entry in app.conversation_entries if entry.kind == "assistant"] == []
        assert [entry for entry in app.conversation_entries if entry.kind == "progress"] == []
        assert app.query_one("#composer", ChatComposer).placeholder == COMPOSER_PLACEHOLDER


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
async def test_textual_app_renders_tool_events_in_chat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.theme import TOOL_RESULT_PREVIEW_CHARS

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
        app._render_event(
            AgentEvent(
                event_type="chat_message",
                sender="coder",
                content={
                    "message_type": "tool_call",
                    "text": "Calling read_file",
                    "tool_description": "read_file",
                    "tool_call_id": "tool-1",
                },
            )
        )
        app._render_event(
            AgentEvent(
                event_type="chat_message",
                sender="coder",
                content={
                    "message_type": "tool_result",
                    "text": "short result",
                    "tool_description": "read_file",
                    "tool_call_id": "tool-1",
                },
            )
        )
        app._render_event(
            AgentEvent(
                event_type="chat_message",
                sender="coder",
                content={
                    "message_type": "tool_error",
                    "text": "x" * (TOOL_RESULT_PREVIEW_CHARS + 10),
                    "tool_description": "read_file",
                    "tool_call_id": "tool-2",
                },
            )
        )

        tool_entries = [entry for entry in app.conversation_entries if entry.kind.startswith("tool")]
        assert [entry.kind for entry in tool_entries] == ["tool_result", "tool_error"]
        assert tool_entries[0].content == "short result"
        assert tool_entries[0].tool_call_id == "tool-1"
        assert tool_entries[1].content.endswith("…")
        assert tool_entries[1].tool_call_id == "tool-2"
        assert len(tool_entries[1].content) == TOOL_RESULT_PREVIEW_CHARS + 1


@pytest.mark.asyncio
async def test_textual_app_appends_append_mode_tool_streaming_events_in_chat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

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
        app._render_event(
            AgentEvent(
                event_type="chat_message",
                sender="coder",
                content={
                    "message_type": "tool_call",
                    "text": "Calling think_hard",
                    "tool_description": "think_hard",
                    "tool_call_id": "tool-1",
                },
            )
        )
        app._render_event(
            AgentEvent(
                event_type="tool_streaming_update",
                sender="coder",
                content={
                    "text": "partial analysis",
                    "tool_name": "think_hard",
                    "tool_call_id": "tool-1",
                    "is_complete": False,
                    "stream_mode": "append",
                },
            )
        )
        app._render_event(
            AgentEvent(
                event_type="tool_streaming_update",
                sender="coder",
                content={
                    "text": "\ncontinued analysis",
                    "tool_name": "think_hard",
                    "tool_call_id": "tool-1",
                    "is_complete": False,
                    "stream_mode": "append",
                },
            )
        )

        tool_entries = [entry for entry in app.conversation_entries if entry.kind.startswith("tool")]
        assert len(tool_entries) == 1
        assert tool_entries[0].kind == "tool_call"
        assert tool_entries[0].content == "partial analysis\ncontinued analysis"
        assert tool_entries[0].complete is False

        app._render_event(
            AgentEvent(
                event_type="tool_streaming_update",
                sender="coder",
                content={
                    "text": "final analysis",
                    "tool_name": "think_hard",
                    "tool_call_id": "tool-1",
                    "is_complete": True,
                },
            )
        )

        tool_entries = [entry for entry in app.conversation_entries if entry.kind.startswith("tool")]
        assert len(tool_entries) == 1
        assert tool_entries[0].kind == "tool_result"
        assert tool_entries[0].content == "final analysis"
        assert tool_entries[0].complete is True
        assert app._tool_stream_buffers == {}


@pytest.mark.asyncio
async def test_textual_app_replaces_default_tool_streaming_events_in_chat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

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
        app._render_event(
            AgentEvent(
                event_type="tool_streaming_update",
                sender="coder",
                content={
                    "text": "Fetching content...",
                    "tool_name": "web_fetch",
                    "tool_call_id": "tool-1",
                    "is_complete": False,
                },
            )
        )
        app._render_event(
            AgentEvent(
                event_type="tool_streaming_update",
                sender="coder",
                content={
                    "text": "Processing content...",
                    "tool_name": "web_fetch",
                    "tool_call_id": "tool-1",
                    "is_complete": False,
                },
            )
        )

        tool_entries = [entry for entry in app.conversation_entries if entry.kind.startswith("tool")]
        assert len(tool_entries) == 1
        assert tool_entries[0].kind == "tool_call"
        assert tool_entries[0].content == "Processing content..."


@pytest.mark.asyncio
async def test_textual_app_caps_long_append_mode_tool_streaming_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.theme import TOOL_STREAM_PREVIEW_CHARS

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
        app._render_event(
            AgentEvent(
                event_type="tool_streaming_update",
                sender="coder",
                content={
                    "text": "a" * (TOOL_STREAM_PREVIEW_CHARS + 10),
                    "tool_name": "think_hard",
                    "tool_call_id": "tool-1",
                    "is_complete": False,
                    "stream_mode": "append",
                },
            )
        )

        tool_entries = [entry for entry in app.conversation_entries if entry.kind.startswith("tool")]
        assert len(tool_entries) == 1
        assert tool_entries[0].content.startswith(
            f"[stream truncated to the last {TOOL_STREAM_PREVIEW_CHARS} characters]"
        )
        assert tool_entries[0].content.endswith("a" * TOOL_STREAM_PREVIEW_CHARS)


@pytest.mark.asyncio
async def test_textual_app_renders_queued_tool_events_during_active_turn(
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
            self.connection_manager = kwargs["connection_manager"]
            self.workspace_id = kwargs["workspace_id"]
            self.thread_id = kwargs["thread_id"]

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
            await self.connection_manager.broadcast_event(
                AgentEvent(
                    event_type="chat_message",
                    sender="coder",
                    content={
                        "message_type": "tool_call",
                        "text": "Calling read_file",
                        "tool_description": "read_file",
                        "tool_call_id": "tool-1",
                    },
                ),
                self.workspace_id,
                self.thread_id,
            )
            started.set()
            await release.wait()
            await self.connection_manager.broadcast_event(
                AgentEvent(
                    event_type="chat_message",
                    sender="coder",
                    content={
                        "message_type": "tool_result",
                        "text": "README contents",
                        "tool_description": "read_file",
                        "tool_call_id": "tool-1",
                    },
                ),
                self.workspace_id,
                self.thread_id,
            )
            yield {"type": "response", "content": "done", "complete": True, "uuid": "response-1"}

    async def wait_for_tool_entries(app: KolegaCodeApp, count: int) -> list:
        while True:
            entries = [entry for entry in app.conversation_entries if entry.kind.startswith("tool")]
            if len(entries) >= count:
                return entries
            await asyncio.sleep(0.01)

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)
    now = 10.0
    monkeypatch.setattr(app, "_now", lambda: now)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        turn_status = app.query_one("#turn_status", Static)
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, "hi"))
        worker = app.agent_worker
        assert worker is not None
        assert worker.group == "turns"

        await started.wait()
        event_worker = next(worker for worker in app.workers if worker.name == "kolega-events")
        assert event_worker.group == "events"
        assert not event_worker.is_cancelled

        tool_entries = await asyncio.wait_for(wait_for_tool_entries(app, 1), timeout=1)
        assert tool_entries[0].kind == "tool_call"
        assert tool_entries[0].content == "Calling read_file"
        assert composer.placeholder == COMPOSER_PLACEHOLDER
        assert "Running read_file…" in str(turn_status.render())

        now = 25.0
        release.set()
        await worker.wait()

        tool_entries = [entry for entry in app.conversation_entries if entry.kind.startswith("tool")]
        assert len(tool_entries) == 1
        assert tool_entries[0].kind == "tool_result"
        assert tool_entries[0].content == "README contents"
        assert composer.placeholder == COMPOSER_PLACEHOLDER
        assert "Done in 15s" in str(turn_status.render())


@pytest.mark.asyncio
async def test_textual_app_late_tool_result_updates_existing_tool_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

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
        app._render_event(
            AgentEvent(
                event_type="chat_message",
                sender="coder",
                content={
                    "message_type": "tool_call",
                    "text": "Calling read_file",
                    "tool_description": "read_file",
                    "tool_call_id": "tool-1",
                },
            )
        )
        app._active_progress_entry = None
        app._turn_active = False

        app._render_event(
            AgentEvent(
                event_type="chat_message",
                sender="coder",
                content={
                    "message_type": "tool_result",
                    "text": "late result",
                    "tool_description": "read_file",
                    "tool_call_id": "tool-1",
                },
            )
        )

        tool_entries = [entry for entry in app.conversation_entries if entry.kind.startswith("tool")]
        assert len(tool_entries) == 1
        assert tool_entries[0].kind == "tool_result"
        assert tool_entries[0].content == "late result"


@pytest.mark.asyncio
async def test_textual_app_cancellation_is_visible_in_chat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Static

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer
    from kolega_code.cli.messages import COMPOSER_PLACEHOLDER

    started = asyncio.Event()

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
            while True:
                await asyncio.sleep(1)
                yield {"type": "thinking", "content": "still working", "complete": False, "uuid": "thinking-1"}

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)
    now = 10.0
    monkeypatch.setattr(app, "_now", lambda: now)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        turn_status = app.query_one("#turn_status", Static)
        task = asyncio.create_task(app._process_message("hi"))
        app.agent_worker = task
        await started.wait()

        now = 52.0
        app.action_cancel_generation()
        progress_entries = [entry for entry in app.conversation_entries if entry.kind == "progress"]
        assert progress_entries == []
        assert composer.placeholder == COMPOSER_PLACEHOLDER
        assert "Stopping…" in str(turn_status.render())
        assert "42s" in str(turn_status.render())

        await task

        progress_entries = [entry for entry in app.conversation_entries if entry.kind == "progress"]
        assert len(progress_entries) == 1
        assert progress_entries[0].content == "Stopped by user."
        assert progress_entries[0].complete is True
        assert composer.placeholder == COMPOSER_PLACEHOLDER
        assert "Stopped after 42s" in str(turn_status.render())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,provider,model,expected_message",
    [
        pytest.param(
            LLMBillingError(
                "DeepSeek APIError: Insufficient Balance",
                provider=ModelProvider.DEEPSEEK.value,
            ),
            ModelProvider.DEEPSEEK,
            DEEPSEEK_DEFAULT_MODEL,
            "DeepSeek/deepseek-v4-pro could not run this request",
            id="billing",
        ),
        pytest.param(
            LLMContextWindowExceededError("context too large", provider=ModelProvider.ANTHROPIC.value),
            ModelProvider.ANTHROPIC,
            "claude-haiku-4-5-20251001",
            "The conversation context became too large for the model",
            id="context-window",
        ),
        pytest.param(
            LLMInternalServerError(
                "provider overloaded",
                provider=ModelProvider.ANTHROPIC.value,
            ),
            ModelProvider.ANTHROPIC,
            "claude-haiku-4-5-20251001",
            "There is high traffic on our LLM provider",
            id="internal-server",
        ),
        pytest.param(
            LLMAuthenticationError(
                "invalid key",
                provider=ModelProvider.ANTHROPIC.value,
            ),
            ModelProvider.ANTHROPIC,
            "claude-haiku-4-5-20251001",
            "Anthropic/claude-haiku-4-5-20251001 could not authenticate",
            id="authentication",
        ),
        pytest.param(
            LLMError(
                "unexpected provider error",
                provider=ModelProvider.ANTHROPIC.value,
            ),
            ModelProvider.ANTHROPIC,
            "claude-haiku-4-5-20251001",
            "Anthropic/claude-haiku-4-5-20251001 returned an error",
            id="generic-llm",
        ),
    ],
)
async def test_textual_app_handles_llm_error_without_worker_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error,
    provider,
    model,
    expected_message,
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Static

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import TurnState
    from kolega_code.cli.tui.widgets import ChatComposer
    from kolega_code.cli.messages import COMPOSER_PLACEHOLDER

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
            raise error
            yield {"type": "response", "content": "unreachable"}

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    config.long_context_config.provider = provider
    config.long_context_config.model = model
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)
    monkeypatch.setattr(app, "_now", lambda: 10.0)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        turn_status = app.query_one("#turn_status", Static)

        await app._process_message("hi")

        progress_entries = [entry for entry in app.conversation_entries if entry.kind == "progress"]
        assert len(progress_entries) == 1
        assert expected_message in progress_entries[0].content
        assert progress_entries[0].tone == "error"
        assert composer.placeholder == COMPOSER_PLACEHOLDER
        assert composer.disabled is False
        assert app.agent_worker is None
        assert app._status_state.turn_state is TurnState.ERROR
        assert "Errored after" in str(turn_status.render())


@pytest.mark.asyncio
async def test_textual_app_reraises_non_llm_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import TurnState
    from kolega_code.cli.tui.widgets import ChatComposer

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
            raise RuntimeError("tool host exploded")
            yield {"type": "response", "content": "unreachable"}

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)

        with pytest.raises(RuntimeError, match="tool host exploded"):
            await app._process_message("hi")

        progress_entries = [entry for entry in app.conversation_entries if entry.kind == "progress"]
        assert len(progress_entries) == 1
        assert progress_entries[0].content == "Stopped due to an error: tool host exploded"
        assert progress_entries[0].tone == "error"
        assert composer.disabled is False
        assert app._status_state.turn_state is TurnState.ERROR


@pytest.mark.asyncio
async def test_rapid_stream_chunks_coalesce_renders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        render_calls = 0
        original_render = app._render_conversation

        def counting_render() -> None:
            nonlocal render_calls
            render_calls += 1
            original_render()

        monkeypatch.setattr(app, "_render_conversation", counting_render)

        for index in range(50):
            app._apply_stream_chunk(
                {"uuid": "chunk-1", "content": f"word{index} ", "complete": False}, kind="assistant"
            )
        app._apply_stream_chunk({"uuid": "chunk-1", "content": "done", "complete": True}, kind="assistant")

        await pilot.pause(0.1)

        assert render_calls < 10
        entry = app._stream_entries["chunk-1"]
        assert entry.complete is True
        assert "word0" in entry.content
        assert "word49" in entry.content
        assert entry.content.endswith("done")


@pytest.mark.asyncio
async def test_conversation_body_renders_rich_markup_tokens_literally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kolega_code.cli.tui.state import ConversationEntry

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    literal = "I investigated [/dim]\n[bold]not bold[/bold]\npath\\"

    async with app.run_test():
        for entry in [
            ConversationEntry(kind="user", content=literal),
            ConversationEntry(kind="assistant", content=literal, complete=False),
            ConversationEntry(kind="thinking", content=literal, complete=False),
            ConversationEntry(kind="progress", content=literal, complete=True),
            ConversationEntry(kind="question", content=literal),
            ConversationEntry(kind="skill", content=literal),
            ConversationEntry(kind="system", content=literal),
            ConversationEntry(kind="message", content=literal),
        ]:
            rendered = app._format_conversation_entry(entry)
            text = renderable_text(rendered)
            assert "[/dim]" in text
            assert "[bold]not bold[/bold]" in text
            assert "path\\" in text

        app._render_event(
            _sub_agent_event(
                agent_name="agent[/dim]",
                task="inspect [red]task[/red]",
                uuid="u1",
                text="tail [/dim] [bold]literal[/bold]\\",
            )
        )
        sub_agent_entry = _sub_agent_entries(app)[0]
        sub_agent_text = renderable_text(app._format_conversation_entry(sub_agent_entry))
        assert "agent[/dim]" in sub_agent_text
        assert "[red]task[/red]" in sub_agent_text
        assert "[bold]literal[/bold]\\" in sub_agent_text


@pytest.mark.asyncio
async def test_streaming_assistant_refresh_accepts_literal_markup_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import ConversationEntryWidget

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        entry = ConversationEntry(kind="assistant", content="start [/dim]", complete=False)
        app.conversation_entries = [entry]
        app._render_conversation()
        await pilot.pause()

        widget = app.query(ConversationEntryWidget).last()
        entry.content += "\n[bold]literal[/bold]\\"
        app._invalidate_conversation(entry)
        app._flush_conversation_render()
        await pilot.pause()

        assert app.query(ConversationEntryWidget).last() is widget
        rendered = renderable_text(widget._formatted)
        assert "[/dim]" in rendered
        assert "[bold]literal[/bold]\\" in rendered


@pytest.mark.asyncio
async def test_conversation_scroll_position_survives_streaming(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import JumpToBottomBar

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        view = app._conversation
        for index in range(40):
            app._add_conversation_entry(ConversationEntry(kind="user", content=f"message {index}"))
        app._flush_conversation_render()
        await pilot.pause()
        await pilot.pause()

        assert view.max_scroll_y > 0
        # Anchored: streaming keeps the view pinned to the bottom
        assert view.scroll_y == view.max_scroll_y

        # User scrolls up; new entries must not yank the view back down
        view.scroll_to(y=0, animate=False)
        await pilot.pause()
        for index in range(5):
            app._add_conversation_entry(ConversationEntry(kind="user", content=f"late message {index}"))
        app._flush_conversation_render()
        await pilot.pause()
        await pilot.pause()

        assert view.scroll_y == 0
        assert app.query_one("#jump_to_bottom", JumpToBottomBar).display is True

        # Jump-to-bottom restores the anchor and hides the bar
        app.on_jump_to_bottom_bar_pressed(JumpToBottomBar.Pressed(app.query_one("#jump_to_bottom", JumpToBottomBar)))
        await pilot.pause()
        assert view.scroll_y == view.max_scroll_y
        assert app.query_one("#jump_to_bottom", JumpToBottomBar).display is False


@pytest.mark.asyncio
async def test_streaming_growth_stays_pinned_when_following_bottom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import JumpToBottomBar

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        view = app._conversation
        for index in range(35):
            app._add_conversation_entry(ConversationEntry(kind="user", content=f"message {index}"))
        entry = ConversationEntry(kind="assistant", content="stream start", complete=False)
        app._add_conversation_entry(entry)
        app._flush_conversation_render()
        await pilot.pause()
        await pilot.pause()

        assert view.is_at_bottom()
        assert view.auto_follow_bottom is True

        widget = app._entry_widgets[entry.entry_id]
        for index in range(8):
            entry.content += f"\nstreamed line {index} " + ("x" * 100)
            app._invalidate_conversation(entry)
            app._flush_conversation_render()
            await pilot.pause()
            await pilot.pause()
            assert app._entry_widgets[entry.entry_id] is widget
            assert view.is_at_bottom()
            assert app.query_one("#jump_to_bottom", JumpToBottomBar).display is False


@pytest.mark.asyncio
async def test_jump_to_bottom_keeps_following_continued_stream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import JumpToBottomBar

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        view = app._conversation
        for index in range(35):
            app._add_conversation_entry(ConversationEntry(kind="user", content=f"message {index}"))
        entry = ConversationEntry(kind="assistant", content="stream start", complete=False)
        app._add_conversation_entry(entry)
        app._flush_conversation_render()
        await pilot.pause()
        await pilot.pause()

        view.scroll_to(y=0, animate=False)
        await pilot.pause()
        assert view.auto_follow_bottom is False

        for index in range(6):
            entry.content += f"\nwhile away {index} " + ("x" * 100)
            app._invalidate_conversation(entry)
            app._flush_conversation_render()
            await pilot.pause()

        bar = app.query_one("#jump_to_bottom", JumpToBottomBar)
        assert view.scroll_y == 0
        assert view.is_at_bottom() is False
        assert bar.display is True

        app.on_jump_to_bottom_bar_pressed(JumpToBottomBar.Pressed(bar))
        await pilot.pause()
        await pilot.pause()
        assert view.auto_follow_bottom is True
        assert view.is_at_bottom()
        assert bar.display is False

        for index in range(6):
            entry.content += f"\nafter jump {index} " + ("y" * 100)
            app._invalidate_conversation(entry)
            app._flush_conversation_render()
            await pilot.pause()
            await pilot.pause()
            assert view.is_at_bottom()
            assert bar.display is False


@pytest.mark.asyncio
async def test_markdown_completion_reflow_after_jump_stays_at_bottom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import JumpToBottomBar

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        view = app._conversation
        for index in range(25):
            app._add_conversation_entry(ConversationEntry(kind="user", content=f"message {index}"))
        markdown_lines = [f"- streamed markdown item {index} with `code`" for index in range(45)]
        entry = ConversationEntry(kind="assistant", content="\n".join(markdown_lines), complete=False)
        app._add_conversation_entry(entry)
        app._flush_conversation_render()
        await pilot.pause()
        await pilot.pause()

        view.scroll_to(y=0, animate=False)
        await pilot.pause()
        bar = app.query_one("#jump_to_bottom", JumpToBottomBar)
        assert bar.display is True

        app.on_jump_to_bottom_bar_pressed(JumpToBottomBar.Pressed(bar))
        await pilot.pause()
        await pilot.pause()
        assert view.is_at_bottom()

        entry.complete = True
        app._invalidate_conversation(entry)
        app._flush_conversation_render()
        await pilot.pause()
        await pilot.pause()
        await pilot.pause()

        assert view.auto_follow_bottom is True
        assert view.is_at_bottom()
        assert bar.display is False


@pytest.mark.asyncio
async def test_full_rebuild_preserves_manual_scroll_away(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import JumpToBottomBar

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        view = app._conversation
        for index in range(45):
            app._add_conversation_entry(ConversationEntry(kind="user", content=f"message {index}"))
        app._flush_conversation_render()
        await pilot.pause()
        await pilot.pause()
        assert view.is_at_bottom()

        view.scroll_to(y=0, animate=False)
        await pilot.pause()
        assert view.auto_follow_bottom is False

        app._render_conversation()
        await pilot.pause()
        await pilot.pause()

        assert view.scroll_y == 0
        assert view.is_at_bottom() is False
        assert view.auto_follow_bottom is False
        assert app.query_one("#jump_to_bottom", JumpToBottomBar).display is True


@pytest.mark.asyncio
async def test_full_rebuild_keeps_following_when_already_at_bottom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import JumpToBottomBar

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        view = app._conversation
        for index in range(45):
            app._add_conversation_entry(ConversationEntry(kind="user", content=f"message {index}"))
        app._flush_conversation_render()
        await pilot.pause()
        await pilot.pause()
        assert view.is_at_bottom()
        assert view.auto_follow_bottom is True

        app._render_conversation()
        await pilot.pause()
        await pilot.pause()

        assert view.is_at_bottom()
        assert view.auto_follow_bottom is True
        assert app.query_one("#jump_to_bottom", JumpToBottomBar).display is False


@pytest.mark.asyncio
async def test_assistant_entries_render_markdown_when_complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from rich.console import Group
    from rich.markdown import Markdown as RichMarkdown

    from kolega_code.cli.tui.state import ConversationEntry

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        streaming = app._format_conversation_entry(
            ConversationEntry(kind="assistant", content="# Title\n\nsome `code`", complete=False)
        )
        assert not isinstance(streaming, str)
        assert "…" in renderable_text(streaming)  # header carries the streaming indicator

        complete = app._format_conversation_entry(
            ConversationEntry(kind="assistant", content="# Title\n\nsome `code`", complete=True)
        )
        assert isinstance(complete, Group)
        renderables = list(complete.renderables)
        assert any(isinstance(getattr(item, "renderable", item), RichMarkdown) for item in renderables)

        plan = app._format_conversation_entry(
            ConversationEntry(kind="plan", content="- step one\n- step two", complete=True)
        )
        assert isinstance(plan, Group)


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
async def test_tool_entries_render_as_collapsibles_with_full_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Collapsible

    from kolega_code.cli.tui.widgets import ToolEntryWidget
    from kolega_code.cli.theme import TOOL_RESULT_PREVIEW_CHARS

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        app._turn_active = True
        long_output = "x" * (TOOL_RESULT_PREVIEW_CHARS + 200)
        app._add_tool_message(
            "tool_call", {"tool_name": "read_file", "tool_call_id": "tc-1", "text": "Calling read_file"}
        )
        app._flush_conversation_render()
        await pilot.pause()

        widget = app.query(ToolEntryWidget).last()
        collapsible = widget.query_one(Collapsible)
        assert collapsible.collapsed is True
        assert "running" in str(collapsible.title)

        app._add_tool_message("tool_result", {"tool_name": "read_file", "tool_call_id": "tc-1", "text": long_output})
        app._flush_conversation_render()
        await pilot.pause()

        # The same widget is updated in place: title flips to done, body holds full output
        same_widget = app.query(ToolEntryWidget).last()
        assert same_widget is widget
        assert "done" in str(widget.query_one(Collapsible).title)
        entry = widget.entry
        assert len(entry.content) == TOOL_RESULT_PREVIEW_CHARS + 1  # preview stays truncated
        assert entry.full_content == long_output  # expand-on-demand shows everything


@pytest.mark.asyncio
async def test_log_lines_carry_timestamp_and_level_glyph(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    import re

    from kolega_code.agent import AgentEvent

    app = _build_sub_agent_test_app(tmp_path, monkeypatch, show_logs=True)

    async with app.run_test():
        line = app._format_log_line("boom", "error")
        assert re.fullmatch(r"\d{2}:\d{2}:\d{2} \S+ boom", line.plain)

        written: list[object] = []
        monkeypatch.setattr(app._logs, "write_log", written.append)
        app._render_event(
            AgentEvent(event_type="log_message", sender="coder", content={"level": "error", "message": "it [broke]"})
        )
        assert len(written) == 1
        assert "[error]" not in written[0].plain  # no raw level prefix
        assert "it [broke]" in written[0].plain  # brackets survive without markup errors


@pytest.mark.asyncio
async def test_terminal_commands_render_as_styled_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from kolega_code.agent import AgentEvent
    from kolega_code.cli import theme

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        formatted = app._format_terminal_command("ls -la")
        assert formatted.plain == f"{theme.g(theme.Glyph.USER)} ls -la"

        written: list[object] = []
        monkeypatch.setattr(app._terminal, "write_terminal", written.append)
        app._render_event(AgentEvent(event_type="terminal_command", sender="coder", content={"command": "echo one"}))
        app._render_event(AgentEvent(event_type="terminal_output", sender="coder", content={"output": "one"}))
        app._render_event(AgentEvent(event_type="terminal_command", sender="coder", content={"command": "echo two"}))

        plains = [item.plain if hasattr(item, "plain") else item for item in written]
        # Pending output is flushed before the next command, whose block is preceded
        # by a blank separator line.
        assert plains == [f"{theme.g(theme.Glyph.USER)} echo one", "one", "", f"{theme.g(theme.Glyph.USER)} echo two"]


@pytest.mark.asyncio
async def test_terminal_output_is_batched_until_flush(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from kolega_code.agent import AgentEvent

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        written: list[object] = []
        monkeypatch.setattr(app._terminal, "write_terminal", written.append)

        for index in range(5):
            app._render_event(
                AgentEvent(event_type="terminal_output", sender="coder", content={"output": f"chunk-{index}\n"})
            )

        assert written == []
        app._flush_terminal_output()

        assert written == ["chunk-0\nchunk-1\nchunk-2\nchunk-3\nchunk-4\n"]
        assert app._terminal_output_buffer == []
        assert app._terminal_output_buffer_chars == 0


@pytest.mark.asyncio
async def test_terminal_output_preserves_scrollback_when_user_scrolls_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import TabbedContent

    from kolega_code.agent import AgentEvent

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test(size=(100, 30)) as pilot:
        app.query_one("#events", TabbedContent).active = "terminal_pane"
        await pilot.pause()

        terminal = app._terminal
        terminal.write_terminal("".join(f"line {index}\n" for index in range(120)))
        await pilot.pause()
        terminal.scroll_end(animate=False, immediate=True)
        await pilot.pause()
        assert terminal.max_scroll_y > 0

        terminal.scroll_to(y=0, animate=False, immediate=True)
        await pilot.pause()
        scroll_y = terminal.scroll_y
        assert terminal.auto_follow_bottom is False

        app._render_event(AgentEvent(event_type="terminal_output", sender="coder", content={"output": "new line\n"}))
        app._flush_terminal_output()
        await pilot.pause()

        assert terminal.scroll_y == scroll_y

        terminal.scroll_end(animate=False, immediate=True)
        await pilot.pause()
        app._render_event(AgentEvent(event_type="terminal_output", sender="coder", content={"output": "tail line\n"}))
        app._flush_terminal_output()
        await pilot.pause()

        assert terminal.scroll_y >= terminal.max_scroll_y - terminal.bottom_tolerance


@pytest.mark.asyncio
async def test_terminal_rendered_history_is_capped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from textual.widgets import TabbedContent

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test(size=(100, 30)) as pilot:
        app.query_one("#events", TabbedContent).active = "terminal_pane"
        await pilot.pause()

        terminal = app._terminal
        terminal.max_lines = 5
        terminal.write_terminal("".join(f"line {index}\n" for index in range(12)))
        await pilot.pause()

        rendered = "\n".join(strip.text for strip in terminal.lines)
        assert len(terminal.lines) <= 5
        assert "line 11" in rendered


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
async def test_logs_tab_hidden_by_default_and_write_log_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import TabbedContent

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        tabs = app.query_one("#events", TabbedContent)
        assert tabs.active == "status_pane"
        assert list(app.query("#logs")) == []

        def fail_format(*args, **kwargs):
            raise AssertionError("hidden logs should not format log lines")

        def fail_activity(*args, **kwargs):
            raise AssertionError("hidden logs should not mark tab activity")

        monkeypatch.setattr(app, "_format_log_line", fail_format)
        monkeypatch.setattr(app, "_mark_tab_activity", fail_activity)

        app._write_log("background activity")


@pytest.mark.asyncio
async def test_logs_tab_can_be_enabled_with_sticky_widget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from textual.widgets import TabbedContent

    from kolega_code.cli.tui.widgets import LogOutputLog

    app = _build_sub_agent_test_app(tmp_path, monkeypatch, show_logs=True)

    async with app.run_test():
        tabs = app.query_one("#events", TabbedContent)

        assert tabs.get_tab("logs_pane") is not None
        assert isinstance(app.query_one("#logs"), LogOutputLog)


@pytest.mark.asyncio
async def test_logs_output_preserves_scrollback_when_user_scrolls_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import TabbedContent

    app = _build_sub_agent_test_app(tmp_path, monkeypatch, show_logs=True)

    async with app.run_test(size=(100, 30)) as pilot:
        app.query_one("#events", TabbedContent).active = "logs_pane"
        await pilot.pause()

        logs = app._logs
        logs.write_log("".join(f"line {index}\n" for index in range(120)))
        await pilot.pause()
        logs.scroll_end(animate=False, immediate=True)
        await pilot.pause()
        assert logs.max_scroll_y > 0

        logs.scroll_to(y=0, animate=False, immediate=True)
        await pilot.pause()
        scroll_y = logs.scroll_y
        assert logs.auto_follow_bottom is False

        app._write_log("new line")
        await pilot.pause()

        assert logs.scroll_y == scroll_y

        logs.scroll_end(animate=False, immediate=True)
        await pilot.pause()
        app._write_log("tail line")
        await pilot.pause()

        assert logs.scroll_y >= logs.max_scroll_y - logs.bottom_tolerance


@pytest.mark.asyncio
async def test_logs_rendered_history_is_capped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from textual.widgets import TabbedContent

    app = _build_sub_agent_test_app(tmp_path, monkeypatch, show_logs=True)

    async with app.run_test(size=(100, 30)) as pilot:
        app.query_one("#events", TabbedContent).active = "logs_pane"
        await pilot.pause()

        logs = app._logs
        logs.max_lines = 5
        logs.write_log("".join(f"line {index}\n" for index in range(12)))
        await pilot.pause()

        rendered = "\n".join(strip.text for strip in logs.lines)
        assert len(logs.lines) <= 5
        assert "line 11" in rendered


@pytest.mark.asyncio
async def test_logs_tab_shows_activity_dot_until_visited(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from textual.widgets import TabbedContent

    from kolega_code.cli import theme

    app = _build_sub_agent_test_app(tmp_path, monkeypatch, show_logs=True)

    async with app.run_test() as pilot:
        tabs = app.query_one("#events", TabbedContent)
        assert tabs.active == "status_pane"

        app._write_log("background activity")
        dot = theme.g(theme.Glyph.STATUS)
        assert str(tabs.get_tab("logs_pane").label) == f"Logs {dot}"

        tabs.active = "logs_pane"
        await pilot.pause()
        assert str(tabs.get_tab("logs_pane").label) == "Logs"

        # Writing while the tab is active does not re-add the dot
        app._write_log("foreground activity")
        assert str(tabs.get_tab("logs_pane").label) == "Logs"
