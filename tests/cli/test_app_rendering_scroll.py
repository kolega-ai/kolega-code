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

