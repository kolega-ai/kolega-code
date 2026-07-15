from copy import deepcopy
from pathlib import Path

import pytest
from textual.widgets import Static

from kolega_code.cli.tui.state import ConversationEntry
from kolega_code.cli.tui.terminal_display import TerminalControlFilter
from kolega_code.llm.models import Message, TextBlock, ToolResult

from ._app_test_utils import _build_sub_agent_test_app


RESTORED_RAW = "RESTORED_BEFORE\x1b]52;c;OSC_PAYLOAD\x07\x1b[?1049h\x90DCS_PAYLOAD\x9cRESTORED_AFTER"
TOOL_RAW = "TOOL_BEFORE\x1b^PM_PAYLOAD\x1b\\\x9b?1000hTOOL_AFTER"
DYNAMIC_RAW = (
    "DYNAMIC_BEFORE"
    "\x00\x07"
    "\x1b]2;TITLE_PAYLOAD\x1b\\"
    "\x1bPPRIVATE_PAYLOAD\x1b\\"
    "\x9d8;;C1_OSC_PAYLOAD\x9c"
    "\x1bc"
    "DYNAMIC_AFTER"
)


def _terminal_repaint(app) -> str:
    update = app.screen._compositor.render_full_update()
    return update.render_segments(app.console)


def _assert_untrusted_controls_absent(output: str) -> None:
    for payload in (
        "OSC_PAYLOAD",
        "DCS_PAYLOAD",
        "PM_PAYLOAD",
        "TITLE_PAYLOAD",
        "PRIVATE_PAYLOAD",
        "C1_OSC_PAYLOAD",
    ):
        assert payload not in output
    for attack in ("\x1b[?1049h", "\x1b[?1000h", "\x1bc"):
        assert attack not in output
    assert "\x00" not in output
    assert "\x07" not in output
    assert not any("\x80" <= character <= "\x9f" for character in output)


def test_app_registers_one_stable_terminal_control_filter_last(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    first = app.get_line_filters()
    second = app.get_line_filters()

    assert first[-1] is app._terminal_control_filter
    assert second[-1] is app._terminal_control_filter
    assert sum(isinstance(line_filter, TerminalControlFilter) for line_filter in first) == 1


@pytest.mark.asyncio
async def test_compositor_filters_untrusted_widget_text_but_keeps_trusted_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test(size=(200, 50)) as pilot:
        panel = app.query_one("#queued_messages", Static)
        panel.update(DYNAMIC_RAW)
        panel.display = True
        turn_status = app.query_one("#turn_status", Static)
        turn_status.update('[link="https://example.invalid/\x1b]2;STYLE_LINK_PAYLOAD\x07"]MARKUP_LINK[/link]')
        turn_status.display = True
        await pilot.pause()

        output = _terminal_repaint(app)

    _assert_untrusted_controls_absent(output)
    assert "DYNAMIC_BEFORE" in output
    assert "DYNAMIC_AFTER" in output
    assert "MARKUP_LINK" in output
    assert "STYLE_LINK_PAYLOAD" not in output
    # Cursor movement and Rich styling are generated after line filtering and
    # must remain available to Textual's compositor/driver.
    assert "\x1b[" in output


@pytest.mark.asyncio
async def test_restored_model_tool_state_session_and_clipboard_data_stay_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    history = [
        Message(role="assistant", content=[TextBlock(RESTORED_RAW)]).to_dict(),
        Message(
            role="user",
            content=[
                ToolResult(
                    tool_use_id="tool-1",
                    content=TOOL_RAW,
                    name="untrusted_tool",
                    is_error=False,
                )
            ],
        ).to_dict(),
    ]
    original_history = deepcopy(history)
    copied: list[str] = []

    async with app.run_test(size=(200, 50)) as pilot:
        monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
        app._restore_conversation_history(history)
        await pilot.pause()

        assistant_entry = next(entry for entry in app.conversation_entries if entry.kind == "assistant")
        tool_entry = next(entry for entry in app.conversation_entries if entry.kind == "tool_result")
        assert assistant_entry.content == RESTORED_RAW
        assert tool_entry.full_content == TOOL_RAW
        assert history == original_history

        await app._command_copy("")
        assert copied == [RESTORED_RAW]

        assert app.agent is not None
        app.agent.history = [Message.from_dict(item) for item in history]
        await app._save_session_history_async()
        assert app.session.history == original_history

        live_entry = ConversationEntry(kind="assistant", content=DYNAMIC_RAW)
        app._add_conversation_entry(live_entry)
        await pilot.pause()
        assert live_entry.content == DYNAMIC_RAW

        output = _terminal_repaint(app)

    _assert_untrusted_controls_absent(output)
    assert "RESTORED_BEFORE" in output
    assert "RESTORED_AFTER" in output
