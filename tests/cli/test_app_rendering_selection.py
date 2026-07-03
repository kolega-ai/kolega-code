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

    from textual.binding import Binding

    async with app.run_test():
        cancel_binding = next(
            binding
            for binding in app.BINDINGS
            if isinstance(binding, Binding) and binding.action == "cancel_generation"
        )
        assert cancel_binding.key == "ctrl+c"
        assert all("super+c" not in binding.key for binding in app.BINDINGS if isinstance(binding, Binding))


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
                    assert segment.style is not None
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
