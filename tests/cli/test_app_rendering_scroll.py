# ruff: noqa: F401,F811,E402
from pathlib import Path
import asyncio
import json
import time
from unittest.mock import Mock

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
    wait_for_layout,
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
@pytest.mark.parametrize("stream", ["reply", "tool"])
async def test_manual_scroll_wins_over_pending_live_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stream: str
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import JumpToBottomBar

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    def stream_chunk(text: str) -> None:
        if stream == "reply":
            app._apply_stream_chunk({"uuid": "reply-1", "content": text, "complete": False}, kind="assistant")
        else:
            app._apply_tool_streaming_update(
                {"tool_call_id": "tool-1", "tool_name": "shell", "text": text, "stream_mode": "append"}
            )
        app._flush_conversation_render()

    async with app.run_test(size=(92, 28)) as pilot:
        await wait_for_layout(pilot, app._startup_complete.is_set)
        view = app._conversation
        bar = app.query_one("#jump_to_bottom", JumpToBottomBar)
        history = [ConversationEntry(kind="user", content=f"message {index}") for index in range(40)]
        for entry in history:
            app._add_conversation_entry(entry)
        stream_chunk("stream start")
        await wait_for_layout(pilot, lambda: view.max_scroll_y > 0 and not app._conversation_anchor_pending)
        assert view.auto_follow_bottom and view.is_at_bottom()
        anchor = app._entry_widgets[history[-6].entry_id]

        def anchor_row() -> int | None:
            geometry = view.screen._compositor.visible_widgets.get(anchor)
            return None if geometry is None else geometry[0].y - view.content_region.y

        def reading_frame_visible() -> bool:
            row = anchor_row()
            return bar.display and row is not None and anchor.virtual_region.y - row == view.scroll_offset.y

        old_count = len(app._entry_widgets)
        held = []
        call_after_refresh = app.call_after_refresh

        def capture_anchor(callback, *args, **kwargs):
            if callback.__name__ == "anchor_after_refresh":
                held.append(callback)
                return True
            return call_after_refresh(callback, *args, **kwargs)

        with monkeypatch.context() as capture:
            capture.setattr(app, "call_after_refresh", capture_anchor)
            stream_chunk("\n" + "new live output " * 12)
            assert len(held) == 1 and app._conversation_anchor_pending
            for _ in range(4):
                assert view._scroll_up_for_pointer(animate=False)
            await wait_for_layout(pilot, reading_frame_visible)
            assert not view.auto_follow_bottom
            reading_position = view.scroll_y
            reading_row = anchor_row()

        held.pop()()
        await pilot.pause()
        assert not app._conversation_anchor_pending
        assert view.scroll_y == reading_position
        assert anchor_row() == reading_row
        assert not view.auto_follow_bottom and bar.display
        assert len(app._entry_widgets) == old_count

        app.on_jump_to_bottom_bar_pressed(JumpToBottomBar.Pressed(bar))
        await wait_for_layout(pilot, lambda: view.is_at_bottom() and not bar.display)
        assert view.auto_follow_bottom
        stream_chunk("\n" + "after explicit jump " * 12)
        await wait_for_layout(pilot, lambda: not app._conversation_anchor_pending and view.is_at_bottom())
        assert view.auto_follow_bottom and not bar.display


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
@pytest.mark.parametrize("navigation", ["jump", "wheel", "page", "resize"])
async def test_mounted_reply_defers_updates_below_viewport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, navigation: str
) -> None:
    pytest.importorskip("textual")

    from textual import events
    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import ConversationEntryWidget, JumpToBottomBar

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    def chunk(text: str, *, complete: bool = False) -> None:
        app._apply_stream_chunk({"uuid": "reply", "content": text, "complete": complete}, kind="assistant")
        app._flush_conversation_render()

    async with app.run_test(size=(100, 28)) as pilot:
        await wait_for_layout(pilot, app._startup_complete.is_set)
        for index in range(55):
            app._add_conversation_entry(ConversationEntry(kind="user", content=f"message {index}"))
        start = "**stream start**" + "\n\nopening paragraph" * 4
        chunk(start)
        entry = app.conversation_entries[-1]
        view = app._conversation
        reply = app._entry_widgets[entry.entry_id]
        assert isinstance(reply, ConversationEntryWidget)
        await wait_for_layout(
            pilot, lambda: reply in view.screen._compositor.visible_widgets and not app._conversation_anchor_pending
        )
        view.scroll_to(y=view.max_scroll_y - 18, animate=False, immediate=True)
        await wait_for_layout(
            pilot,
            lambda: (
                not view.auto_follow_bottom
                and reply not in view.screen._compositor.visible_widgets
                and reply.region.y >= view.content_region.bottom
            ),
        )
        scroll_y = view.scroll_y
        compositor = view.screen._compositor
        reader = next(
            row
            for row in app._entry_widgets.values()
            if row in compositor.visible_widgets and row.region.y > view.content_region.y
        )
        reader_offset = reader.region.y - view.content_region.y
        refresh = Mock(wraps=reply.refresh_content)
        reflow = Mock(wraps=compositor.reflow)
        monkeypatch.setattr(reply, "refresh_content", refresh)
        monkeypatch.setattr(compositor, "reflow", reflow)
        deltas = [f"\n\n- batch {index}\n- more\n- end" for index in range(10)]
        for delta in deltas:
            chunk(delta)
            await pilot.pause()
        completed = navigation != "wheel"
        if completed:
            chunk("", complete=True)
            await pilot.pause()
        assert entry.complete is completed
        assert entry.content + "".join(entry.stream_parts) == start + "".join(deltas)
        assert refresh.call_count == reflow.call_count == 0
        assert view.scroll_y == scroll_y
        assert reader.region.y - view.content_region.y == reader_offset

        window = app._get_transcript_window()
        mounted_end = window.mounted_end
        if navigation == "page":
            for index in range(20):
                app._add_conversation_entry(ConversationEntry(kind="user", content=f"newer {index}"))
            app._flush_conversation_render()
            await wait_for_layout(pilot, lambda: view.has_newer_entries)

        frames: list[tuple[int, str]] = []
        original_display = app._display

        def display(screen, renderable) -> None:
            if renderable is not None and not app._batch_count and reply in compositor.visible_widgets:
                frames.append((reply.size.height, reply.render_line(0).text))
            original_display(screen, renderable)

        monkeypatch.setattr(app, "_display", display)
        bar = app.query_one("#jump_to_bottom", JumpToBottomBar)
        if navigation == "jump":
            app.on_jump_to_bottom_bar_pressed(JumpToBottomBar.Pressed(bar))
        elif navigation == "resize":
            await pilot.resize_terminal(100, 52)
        else:
            for _ in range(6):
                await view._on_message(events.MouseScrollDown(view, 0, 0, 0, 0, 0, False, False, False))
        await wait_for_layout(
            pilot,
            lambda: bool(frames) and not app._conversation_anchor_pending and view._scrollback_anchor is None,
        )
        if navigation == "jump":
            await wait_for_layout(pilot, lambda: view.is_at_bottom() and not bar.display)
        elif navigation == "page":
            assert window.mounted_end > mounted_end
        elif navigation == "resize":
            assert view.scroll_y == scroll_y
        assert refresh.call_count == 1
        text = "\n".join(reply.render_line(y).text for y in range(reply.size.height))
        assert frames[0][0] == reply.size.height > 30
        assert "stream start" in frames[0][1]
        assert ("**stream start**" in frames[0][1]) is not completed
        assert all(f"batch {index}" in text for index in range(10))
        if navigation == "wheel":
            assert not view.auto_follow_bottom
            assert view.content_region.y <= reply.region.y < view.content_region.bottom < reply.region.bottom
            chunk("\n\nvisible update")
            assert refresh.call_count == 2


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
