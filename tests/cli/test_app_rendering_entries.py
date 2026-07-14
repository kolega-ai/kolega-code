# ruff: noqa: F401,F811,E402
from pathlib import Path
import asyncio
import json
import time

import pytest

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
    FakeCoderAgent,
    _build_mention_test_app,
    _build_sub_agent_test_app,
    _sub_agent_context_event,
    _sub_agent_entries,
    _sub_agent_event,
    _workflow_event,
    build_test_config,
    extension_by_name,
    first_text_styles,
    install_fake_agents,
    question_payload,
    renderable_text,
)


@pytest.mark.asyncio
async def test_textual_app_renders_one_widget_per_chat_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import ConversationEntryWidget

    install_fake_agents(monkeypatch)

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

    install_fake_agents(monkeypatch)

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

    install_fake_agents(monkeypatch)

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

        assert assistant_text.splitlines()[0] == "● hello"
        assert "Agent" not in assistant_text
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
async def test_transcript_bodies_share_two_cell_indent_without_guides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kolega_code.cli.tui.state import ConversationEntry

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        # Glyph-first kinds: the role glyph opens line 1, continuations align at 2.
        user_lines = renderable_text(
            app._format_conversation_entry(ConversationEntry(kind="user", content="body text\nsecond line"))
        ).splitlines()
        assert user_lines[0] == "❯ body text"
        assert user_lines[1] == "  second line"

        streaming_lines = renderable_text(
            app._format_conversation_entry(
                ConversationEntry(kind="assistant", content="body text\nsecond line", complete=False)
            )
        ).splitlines()
        assert streaming_lines[0] == "● body text"
        assert streaming_lines[1] == "  second line"

        complete_lines = renderable_text(
            app._format_conversation_entry(ConversationEntry(kind="assistant", content="body text", complete=True))
        ).splitlines()
        assert complete_lines[0].startswith("● body text")

        # Thinking is bare dim-italic text at the shared indent — no glyph, no label.
        thinking_lines = renderable_text(
            app._format_conversation_entry(
                ConversationEntry(kind="thinking", content="body text\nsecond line", complete=True)
            )
        ).splitlines()
        assert thinking_lines[0] == "  body text"
        assert thinking_lines[1] == "  second line"

        # Header kinds keep the label line with the body indented beneath.
        for entry in [
            ConversationEntry(kind="question", content="body text"),
            ConversationEntry(kind="plan", content="body text"),
            ConversationEntry(kind="lsp", content="body text"),
        ]:
            lines = renderable_text(app._format_conversation_entry(entry)).splitlines()
            assert lines[1].startswith("  body text"), entry.kind
            assert "│" not in lines[1], entry.kind


@pytest.mark.asyncio
async def test_transcript_activity_hierarchy_uses_two_cell_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from textual.widgets import Collapsible, Static
    from textual.widgets._collapsible import CollapsibleTitle

    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import ConversationEntryWidget, ToolEntryWidget

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    top_level = [
        ConversationEntry(kind="user", content="user"),
        ConversationEntry(kind="assistant", content="agent"),
        # Thinking aligns with the message flow (its 2-cell indent lives in the text).
        ConversationEntry(kind="thinking", content="thinking"),
        ConversationEntry(kind="plan", content="plan"),
        ConversationEntry(kind="question", content="question"),
        ConversationEntry(kind="lsp", content="lsp"),
        ConversationEntry(kind="system", content="system"),
    ]
    activities = [
        ConversationEntry(kind="progress", content="status"),
        ConversationEntry(kind="skill", content="skill"),
        ConversationEntry(kind="sub_agent", content="sub-agent"),
        ConversationEntry(kind="workflow", content="workflow"),
    ]
    tool = ConversationEntry(
        kind="tool_result",
        content="preview",
        full_content="full output",
        tool_name="read_file",
    )
    compaction = ConversationEntry(kind="compaction_summary", content="summary")

    async with app.run_test(size=(72, 70)) as pilot:
        app.conversation_entries = [*top_level, *activities, tool, compaction]
        app._render_conversation()
        await pilot.pause()

        for entry in top_level:
            widget = app._entry_widgets[entry.entry_id]
            assert isinstance(widget, ConversationEntryWidget)
            assert widget.content_region.x == widget.region.x, entry.kind
            # The blank line between entries comes from padding-bottom; a more
            # specific padding rule must never zero it (Textual keeps only the
            # most-specific rule per style key, and padding is one key).
            assert widget.styles.padding.bottom == 1, entry.kind

        for entry in activities:
            widget = app._entry_widgets[entry.entry_id]
            assert isinstance(widget, ConversationEntryWidget)
            assert widget.content_region.x == widget.region.x + 2, entry.kind
            assert widget.styles.padding.bottom == 1, entry.kind

        tool_widget = app._entry_widgets[tool.entry_id]
        assert isinstance(tool_widget, ToolEntryWidget)
        assert tool_widget.has_class("agent-activity")
        tool_collapsible = tool_widget.query_one(Collapsible)
        tool_title = tool_widget.query_one(CollapsibleTitle)
        assert tool_collapsible.region.x == tool_widget.region.x
        assert tool_title.content_region.x == tool_widget.region.x + 2

        compaction_widget = app._entry_widgets[compaction.entry_id]
        assert isinstance(compaction_widget, ToolEntryWidget)
        assert not compaction_widget.has_class("agent-activity")
        compaction_title = compaction_widget.query_one(CollapsibleTitle)
        assert compaction_title.content_region.x == compaction_widget.region.x

        tool_collapsible.collapsed = False
        await pilot.pause()
        body = tool_widget.query_one(".tool-body", Static)
        assert body.region.x == tool_widget.region.x + 4
        assert body.region.right <= app._conversation.content_region.right


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
    from rich.table import Table

    from kolega_code.cli.tui.state import ConversationEntry

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        streaming = app._format_conversation_entry(
            ConversationEntry(kind="assistant", content="# Title\n\nsome `code`", complete=False)
        )
        assert renderable_text(streaming).splitlines()[0] == "● # Title"

        complete = app._format_conversation_entry(
            ConversationEntry(kind="assistant", content="# Title\n\nsome `code`", complete=True)
        )
        assert isinstance(complete, Table)
        cells = [cell for column in complete.columns for cell in column._cells]
        assert any(isinstance(cell, RichMarkdown) for cell in cells)
        assert "●" in renderable_text(complete)

        plan = app._format_conversation_entry(
            ConversationEntry(kind="plan", content="- step one\n- step two", complete=True)
        )
        assert isinstance(plan, Group)


@pytest.mark.asyncio
async def test_user_and_assistant_entries_render_inline_glyphs_without_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from rich.text import Text

    from kolega_code.cli import theme
    from kolega_code.cli.tui.state import ConversationEntry

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        user = app._format_conversation_entry(ConversationEntry(kind="user", content="hi\nthere"))
        assert isinstance(user, Text)
        assert user.plain == "❯ hi\n  there"
        span = user._spans[0]
        assert (span.start, span.end, str(span.style)) == (0, 2, theme.Color.USER)

        streaming = app._format_conversation_entry(ConversationEntry(kind="assistant", content="reply", complete=False))
        assert isinstance(streaming, Text)
        span = streaming._spans[0]
        assert (span.start, span.end, str(span.style)) == (0, 2, theme.Color.AGENT)

        for rendered in (user, streaming):
            text = renderable_text(rendered)
            assert "You" not in text
            assert "Agent" not in text
