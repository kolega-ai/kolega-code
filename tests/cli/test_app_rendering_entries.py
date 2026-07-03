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
        monkeypatch.setattr(ConversationView, "is_attached", property(lambda self: False))
        app._flush_conversation_render()  # must not raise (pre-fix: MountError)
        app._render_conversation()  # must not raise

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
