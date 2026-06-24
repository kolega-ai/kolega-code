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
