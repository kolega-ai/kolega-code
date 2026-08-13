# ruff: noqa: F401,F811,E402
from collections.abc import Callable
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
        assert written == []
        app._flush_log_output()
        assert len(written) == 1
        assert "[error]" not in getattr(written[0], "plain")  # no raw level prefix
        assert "it [broke]" in getattr(written[0], "plain")  # brackets survive without markup errors


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

        plains = [getattr(item, "plain", item) for item in written]
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

        terminal = app._terminal
        await _wait_for_layout(pilot, lambda: terminal._size_known and terminal.region.height > 0)
        terminal.write_terminal("".join(f"line {index}\n" for index in range(120)))
        await _wait_for_layout(
            pilot,
            lambda: (
                terminal.max_scroll_y > 0
                and terminal.is_at_bottom()
                and not terminal._follow_bottom_pending
                and not terminal._follow_bottom_callback_scheduled
            ),
        )

        terminal.scroll_to(y=0, animate=False, immediate=True)
        await _wait_for_layout(pilot, lambda: terminal.scroll_y == 0 and not terminal.auto_follow_bottom)
        scroll_y = terminal.scroll_y

        app._render_event(AgentEvent(event_type="terminal_output", sender="coder", content={"output": "new line\n"}))
        app._flush_terminal_output()
        await _wait_for_layout(
            pilot,
            lambda: "new line" in "\n".join(strip.text for strip in terminal.lines),
        )

        assert terminal.scroll_y == scroll_y
        assert terminal.auto_follow_bottom is False

        terminal.scroll_end(animate=False, immediate=True)
        await _wait_for_layout(pilot, terminal.is_at_bottom)
        app._render_event(AgentEvent(event_type="terminal_output", sender="coder", content={"output": "tail line\n"}))
        app._flush_terminal_output()
        await _wait_for_layout(
            pilot,
            lambda: (
                "tail line" in "\n".join(strip.text for strip in terminal.lines)
                and terminal.is_at_bottom()
                and not terminal._follow_bottom_pending
                and not terminal._follow_bottom_callback_scheduled
            ),
        )

        assert terminal.is_at_bottom()


@pytest.mark.asyncio
async def test_terminal_output_written_while_hidden_follows_when_shown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import TabbedContent

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test(size=(100, 30)) as pilot:
        tabs = app.query_one("#events", TabbedContent)
        assert tabs.active == "status_pane"
        terminal = app._terminal

        terminal.write_terminal("".join(f"hidden line {index}\n" for index in range(120)))
        assert terminal._follow_bottom_pending is True

        tabs.active = "terminal_pane"
        await _wait_for_layout(
            pilot,
            lambda: (
                "hidden line 119" in "\n".join(strip.text for strip in terminal.lines)
                and terminal.max_scroll_y > 0
                and terminal.is_at_bottom()
                and not terminal._follow_bottom_pending
                and not terminal._follow_bottom_callback_scheduled
            ),
        )


@pytest.mark.asyncio
async def test_terminal_clear_cancels_pending_follow_and_restores_following(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import TabbedContent

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test(size=(100, 30)) as pilot:
        app.query_one("#events", TabbedContent).active = "terminal_pane"
        terminal = app._terminal
        await _wait_for_layout(pilot, lambda: terminal._size_known and terminal.region.height > 0)

        scheduled_follow_count = 0
        call_after_refresh = terminal.call_after_refresh

        def track_call_after_refresh(callback: Callable[..., object], *args: object) -> None:
            nonlocal scheduled_follow_count
            if callback == terminal._apply_pending_follow_bottom:
                scheduled_follow_count += 1
            call_after_refresh(callback, *args)

        monkeypatch.setattr(terminal, "call_after_refresh", track_call_after_refresh)
        terminal.write_terminal("".join(f"stale line {index}\n" for index in range(120)))
        terminal.write_terminal("another stale line\n")
        assert terminal._follow_bottom_pending is True
        assert scheduled_follow_count == 1
        terminal.clear_output()

        assert terminal._follow_bottom_pending is False
        assert terminal.auto_follow_bottom is True
        await _wait_for_layout(pilot, lambda: not terminal._follow_bottom_callback_scheduled)
        assert terminal.lines == []
        assert terminal.scroll_y == 0

        terminal.write_terminal("".join(f"fresh line {index}\n" for index in range(120)))
        await _wait_for_layout(
            pilot,
            lambda: (
                "fresh line 119" in "\n".join(strip.text for strip in terminal.lines)
                and terminal.max_scroll_y > 0
                and terminal.is_at_bottom()
                and not terminal._follow_bottom_pending
                and not terminal._follow_bottom_callback_scheduled
            ),
        )
        assert scheduled_follow_count == 2


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
@pytest.mark.parametrize("command", ["/clear", "/reset"])
async def test_reset_command_clears_terminal_logs_and_pending_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import TabbedContent

    from kolega_code.cli import theme
    from kolega_code.cli.tui.widgets import ChatComposer

    app = _build_sub_agent_test_app(tmp_path, monkeypatch, show_logs=True)

    async with app.run_test(size=(100, 30)) as pilot:
        tabs = app.query_one("#events", TabbedContent)
        assert tabs.active == "status_pane"

        terminal = app._terminal
        logs = app._logs
        clear_calls = {"terminal": 0, "logs": 0}
        terminal_clear_output = terminal.clear_output
        logs_clear_output = logs.clear_output

        def clear_terminal_output() -> None:
            clear_calls["terminal"] += 1
            terminal_clear_output()

        def clear_logs_output() -> None:
            clear_calls["logs"] += 1
            logs_clear_output()

        monkeypatch.setattr(terminal, "clear_output", clear_terminal_output)
        monkeypatch.setattr(logs, "clear_output", clear_logs_output)

        tabs.active = "terminal_pane"
        await pilot.pause()
        terminal.write_terminal("old terminal output\n")
        await pilot.pause()
        assert "old terminal output" in "\n".join(strip.text for strip in terminal.lines)

        tabs.active = "logs_pane"
        await pilot.pause()
        logs.write_log("old log entry")
        await pilot.pause()
        tabs.active = "status_pane"
        await pilot.pause()
        app._queue_terminal_output("stale buffered output\n")
        app._write_log("background log entry")
        terminal.auto_follow_bottom = False
        logs.auto_follow_bottom = False

        assert app._terminal_flush_timer is not None
        assert app._terminal_output_buffer == ["stale buffered output\n"]
        assert app._terminal_output_buffer_chars == len("stale buffered output\n")
        assert app._terminal_has_content is True
        dot = theme.g(theme.Glyph.STATUS)
        assert str(tabs.get_tab("terminal_pane").label) == f"Terminal {dot}"
        assert str(tabs.get_tab("logs_pane").label) == f"Logs {dot}"

        composer = app.query_one("#composer", ChatComposer)
        composer.load_text(command)
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        await pilot.pause(0.1)

        assert clear_calls == {"terminal": 1, "logs": 1}
        assert terminal.lines == []
        assert logs.lines == []
        assert app._terminal_output_buffer == []
        assert app._terminal_output_buffer_chars == 0
        assert app._terminal_flush_timer is None
        assert app._terminal_has_content is False
        assert terminal.auto_follow_bottom is True
        assert logs.auto_follow_bottom is True
        assert str(tabs.get_tab("terminal_pane").label) == "Terminal"
        assert str(tabs.get_tab("logs_pane").label) == "Logs"
        assert composer.text == ""

        await pilot.pause(0.1)
        assert terminal.lines == []
        assert logs.lines == []


@pytest.mark.asyncio
async def test_blocked_reset_command_preserves_terminal_and_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import TabbedContent

    from kolega_code.cli.tui.widgets import ChatComposer

    app = _build_sub_agent_test_app(tmp_path, monkeypatch, show_logs=True)

    async with app.run_test(size=(100, 30)) as pilot:
        tabs = app.query_one("#events", TabbedContent)
        terminal = app._terminal
        logs = app._logs

        def fail_clear_output() -> None:
            raise AssertionError("blocked reset must not clear runtime output")

        monkeypatch.setattr(terminal, "clear_output", fail_clear_output)
        monkeypatch.setattr(logs, "clear_output", fail_clear_output)

        tabs.active = "terminal_pane"
        await pilot.pause()
        terminal.write_terminal("old terminal output\n")
        await pilot.pause()

        tabs.active = "logs_pane"
        await pilot.pause()
        logs.write_log("old log entry")
        await pilot.pause()

        tabs.active = "status_pane"
        await pilot.pause()
        app._queue_terminal_output("pending output\n")

        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("/clear")
        app._turn_active = True

        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        assert "old terminal output" in "\n".join(strip.text for strip in terminal.lines)
        assert app._terminal_output_buffer == ["pending output\n"]
        assert app._terminal_output_buffer_chars == len("pending output\n")
        assert app._terminal_flush_timer is not None
        assert app._terminal_has_content is True
        assert composer.text == "/clear"


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


def test_default_scrollback_caps_are_bounded() -> None:
    from kolega_code.cli.app import LOG_MAX_LINES, TERMINAL_MAX_LINES

    assert LOG_MAX_LINES == 2_000
    assert TERMINAL_MAX_LINES == 2_000


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


@pytest.mark.asyncio
async def test_terminal_output_is_sanitized_before_rendering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from kolega_code.agent import AgentEvent

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        written: list[object] = []
        monkeypatch.setattr(app._terminal, "write_terminal", written.append)

        app._render_event(
            AgentEvent(
                event_type="terminal_output",
                sender="coder",
                content={
                    "output": "raw \x1b[31mred\x1b[0m\x1b]8;;https://example.com\x1b\\link\x1b]8;;\x1b\\\rnext\b!\x07\n"
                },
            )
        )
        app._flush_terminal_output()

        rendered = "".join(str(item) for item in written)
        assert "\x1b" not in rendered
        assert "https://example.com" not in rendered
        assert rendered == "raw redlink\nnex!\n"


@pytest.mark.asyncio
async def test_terminal_output_uses_display_output_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.agent import AgentEvent

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        written: list[object] = []
        monkeypatch.setattr(app._terminal, "write_terminal", written.append)

        app._render_event(
            AgentEvent(
                event_type="terminal_output",
                sender="coder",
                content={"output": "\ufffd", "display_output": "€"},
            )
        )
        app._flush_terminal_output()

        assert written == ["€"]


@pytest.mark.asyncio
async def test_terminal_and_logs_hide_horizontal_scrollbars_when_wrapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    app = _build_sub_agent_test_app(tmp_path, monkeypatch, show_logs=True)

    async with app.run_test():
        assert app._terminal.styles.overflow_x == "hidden"
        assert app._logs.styles.overflow_x == "hidden"


@pytest.mark.asyncio
async def test_log_output_is_batched_until_flush(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    app = _build_sub_agent_test_app(tmp_path, monkeypatch, show_logs=True)

    async with app.run_test():
        written: list[object] = []
        monkeypatch.setattr(app._logs, "write_log", written.append)

        for index in range(5):
            app._write_log(f"line {index}")

        assert written == []
        app._flush_log_output()

        assert len(written) == 1
        rendered = renderable_text(written[0])
        assert "line 0" in rendered
        assert "line 4" in rendered


async def _wait_for_layout(pilot, predicate, *, timeout: float = 6.0) -> None:
    """Poll until ``predicate()`` is truthy or the layout deadline expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await pilot.pause(0.02)
        if predicate():
            return
    raise AssertionError(f"layout did not settle within {timeout}s")


@pytest.mark.asyncio
async def test_terminal_output_extracts_selected_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from textual.geometry import Offset
    from textual.selection import Selection
    from textual.widgets import TabbedContent

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test(size=(100, 30)) as pilot:
        # RichLog defers writes until its size is known; the pane must be visible first.
        app.query_one("#events", TabbedContent).active = "terminal_pane"
        await pilot.pause()

        terminal = app._terminal
        terminal.write_terminal("alpha\nbeta\ngamma\n")
        await pilot.pause()

        selected = terminal.get_selection(Selection(None, None))
        assert selected is not None
        text, ending = selected
        assert ending == "\n"
        assert "alpha" in text
        assert "beta" in text
        assert "gamma" in text
        assert "\x1b" not in text
        assert not any(line != line.rstrip() for line in text.split("\n"))

        # Coordinates are content (line, column) pairs across visual lines.
        partial = terminal.get_selection(Selection(Offset(1, 0), Offset(3, 2)))
        assert partial is not None
        assert partial[0] == "lpha\nbeta\ngam"


@pytest.mark.asyncio
async def test_terminal_render_line_marks_selection_offsets_and_highlight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.geometry import Offset
    from textual.selection import Selection
    from textual.widgets import TabbedContent

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test(size=(100, 30)) as pilot:
        app.query_one("#events", TabbedContent).active = "terminal_pane"
        terminal = app._terminal
        terminal.write_terminal("highlight this line\n")
        await pilot.pause()

        app.screen.selections = {terminal: Selection(Offset(0, 0), Offset(9, 0))}
        strip = terminal.render_line(0)
        selection_bg = terminal.selection_style.bgcolor

        assert any(
            segment.style is not None
            and segment.style.bgcolor == selection_bg
            and segment.style.meta.get("offset") is not None
            for segment in strip
        )

        # Unscrolled log: selection coordinates start at the content origin.
        tagged_offsets = [
            segment.style.meta["offset"]
            for segment in strip
            if segment.style is not None and segment.style.meta.get("offset") is not None
        ]
        assert tagged_offsets
        assert min(tagged_offsets) == (0, 0)

        app.screen.selections = {}


@pytest.mark.asyncio
async def test_terminal_output_supports_mouse_drag_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from textual import events
    from textual.widgets import TabbedContent

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test(size=(100, 30)) as pilot:
        app.query_one("#events", TabbedContent).active = "terminal_pane"
        terminal = app._terminal
        terminal.write_terminal("select this terminal text\nsecond line\n")
        await pilot.pause()

        def selection_targets_are_ready() -> bool:
            for x in (0, 24):
                hit_widget, select_offset = app.screen.get_widget_and_offset_at(
                    terminal.region.x + x, terminal.region.y
                )
                if hit_widget is not terminal or select_offset is None:
                    return False
            return True

        # The widget can be queryable before the compositor's hit map reflects its
        # region. Mouse selection requires both to agree.
        await _wait_for_layout(pilot, selection_targets_are_ready)

        await pilot.mouse_down(terminal, offset=(0, 0))
        await pilot._post_mouse_events([events.MouseMove], terminal, offset=(24, 0), button=1)
        await pilot.mouse_up(terminal, offset=(24, 0))

        def selection_copied_text() -> bool:
            return (app.screen.get_selected_text() or "").strip() == "select this terminal text"

        await _wait_for_layout(pilot, selection_copied_text)
        selected_text = app.screen.get_selected_text()
        assert selected_text is not None
        assert selected_text.strip() == "select this terminal text"


@pytest.mark.asyncio
async def test_terminal_selection_offsets_follow_vertical_scroll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.geometry import Offset
    from textual.selection import Selection
    from textual.widgets import TabbedContent

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test(size=(100, 30)) as pilot:
        app.query_one("#events", TabbedContent).active = "terminal_pane"
        terminal = app._terminal
        await _wait_for_layout(pilot, lambda: terminal._size_known and terminal.region.height > 0)
        terminal.write_terminal("".join(f"scroll line {index}\n" for index in range(120)))
        assert terminal._follow_bottom_pending is True

        terminal.scroll_to(y=10, animate=False, immediate=True)
        await _wait_for_layout(
            pilot,
            lambda: (
                terminal.max_scroll_y > 10
                and terminal.scroll_offset.y == 10
                and not terminal._follow_bottom_pending
                and not terminal._follow_bottom_callback_scheduled
            ),
        )
        assert terminal.auto_follow_bottom is False

        # Viewport line 0 renders content line 10; stamped offsets must follow the scroll.
        strip = terminal.render_line(0)
        offset_meta = [
            segment.style.meta["offset"]
            for segment in strip
            if segment.style is not None and segment.style.meta.get("offset") is not None
        ]
        assert offset_meta
        assert all(y == 10 for _x, y in offset_meta)

        # Extraction uses the same content coordinates as the stamped offsets.
        selected = terminal.get_selection(Selection(Offset(0, 10), Offset(len("scroll line 10"), 10)))
        assert selected is not None
        assert selected[0] == "scroll line 10"


@pytest.mark.asyncio
async def test_logs_output_supports_mouse_drag_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from textual import events
    from textual.widgets import TabbedContent

    app = _build_sub_agent_test_app(tmp_path, monkeypatch, show_logs=True)

    async with app.run_test(size=(100, 30)) as pilot:
        app.query_one("#events", TabbedContent).active = "logs_pane"
        logs = app._logs
        logs.write_log("select this log text\n")
        await pilot.pause()

        def selection_targets_are_ready() -> bool:
            for x in (0, 19):
                hit_widget, select_offset = app.screen.get_widget_and_offset_at(logs.region.x + x, logs.region.y)
                if hit_widget is not logs or select_offset is None:
                    return False
            return True

        await _wait_for_layout(pilot, selection_targets_are_ready)

        await pilot.mouse_down(logs, offset=(0, 0))
        await pilot._post_mouse_events([events.MouseMove], logs, offset=(19, 0), button=1)
        await pilot.mouse_up(logs, offset=(19, 0))

        def selection_copied_text() -> bool:
            return (app.screen.get_selected_text() or "").strip() == "select this log text"

        await _wait_for_layout(pilot, selection_copied_text)
        selected_text = app.screen.get_selected_text()
        assert selected_text is not None
        assert selected_text.strip() == "select this log text"


@pytest.mark.asyncio
async def test_flush_pacing_stretches_terminal_flush_under_loop_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui import pacing

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        await pilot.pause()
        app._flush_pacer.attach(lambda: 60.0)  # saturated loop: pacer sits at the ceiling

        app._queue_terminal_output("paced output\n")
        assert app._terminal_flush_timer is not None
        assert app._terminal_flush_timer._interval == pacing.PACING_CEILING
        # Queueing never flushes synchronously: the only drain path in an idle app
        # is the armed ceiling timer (0.5s), not the healthy base cadence.
        assert app._terminal_output_buffer == ["paced output\n"]

        # Stop the timer so a real-time pause can never race its firing on a loaded
        # runner (a 0.1s sleep stretched past the 0.5s ceiling would otherwise
        # drain the buffer); still verify nothing else flushes during an idle wait.
        app._terminal_flush_timer.stop()
        await pilot.pause(0.1)
        assert app._terminal_output_buffer == ["paced output\n"]

        app._flush_terminal_output()  # forced drains bypass pacing
        assert app._terminal_output_buffer == []


@pytest.mark.asyncio
async def test_render_coalesce_interval_composes_paced_and_size_backoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui import pacing
    from kolega_code.cli.tui.state import ConversationEntry

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    async with app.run_test():
        big_entry = ConversationEntry(kind="assistant", content="x" * (pacing.RENDER_COALESCE_LARGE_CHARS + 1))

        app._flush_pacer.attach(lambda: 0.0)  # healthy loop: the size backoff wins
        assert app._render_coalesce_interval(big_entry) == pacing.RENDER_COALESCE_INTERVAL_LARGE
        assert app._render_coalesce_interval(None) == pacing.FLUSH_BASE_INTERVAL

        app._flush_pacer.attach(lambda: 60.0)  # saturated loop: the paced floor wins
        assert app._render_coalesce_interval(big_entry) == pacing.PACING_CEILING
        assert app._render_coalesce_interval(None) == pacing.PACING_CEILING
