"""Compact tool displays retain identity, copyable details, and raw results."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from rich.cells import cell_len
from rich.console import Console
from rich.text import Text
from textual import events
from textual.selection import Selection
from textual.widgets import Collapsible, Static
from textual.widgets._collapsible import CollapsibleTitle

from kolega_code.events import AgentEvent
from kolega_code.llm.models import Message, ToolCall, ToolResult
from kolega_code.cli.tui.state import ConversationEntry
from kolega_code.cli.tui.tool_presentation import command_preview, edit_previews, path_label, tool_title
from kolega_code.cli.tui.widgets import ToolEntryWidget
from kolega_code.tool_subjects import build_tool_display, build_tool_subject

from ._app_test_utils import _build_sub_agent_test_app, _sub_agent_event
from .test_tool_subject_titles import _wait_for_layout


def _render(renderable: Any, width: int = 100) -> str:
    output = StringIO()
    Console(file=output, width=width, color_system=None).print(renderable)
    return output.getvalue()


def _event(kind: str, name: str, call_id: str, arguments: dict[str, str], text: str = "") -> AgentEvent:
    return AgentEvent(
        event_type="chat_message",
        sender="coder",
        content={
            "message_type": kind,
            "tool_description": name,
            "tool_call_id": call_id,
            "tool_subject": build_tool_subject(name, arguments),
            "tool_display": build_tool_display(name, arguments),
            "text": text,
        },
    )


def _preview(call_id: str, path: str, *, added: str = "+new") -> dict:
    return {
        "tool_call_id": call_id,
        "tool_name": "apply_patch",
        "kind": "diff",
        "path": path,
        "lines": [["del", "-old"], ["add", added]],
        "adds": 1,
        "dels": 1,
        "more": 0,
    }


@pytest.mark.parametrize("width", [0, 1, 3, 8, 16, 30, 48, 80, 120])
@pytest.mark.parametrize(
    "path",
    [
        "src/" + "deeply/nested/" * 30 + "settings.py",
        "src/" + "文件" * 100 + ".py",
        "C:\\Users\\developer\\" + "deeply\\nested\\" * 10 + "settings.py",
        "/var/folders/" + "temporary/" * 30 + "settings.py",
        "src/[bold]/[literal].py",
        "a/" + "long" * 40 + ".tar.gz",
    ],
)
def test_path_labels_are_cell_bounded_and_keep_extensions(path: str, width: int) -> None:
    label = path_label(path, width)
    assert label.cell_len <= width
    assert "\n" not in label.plain
    if width >= 16:
        assert label.plain.endswith(".gz" if path.endswith(".gz") else ".py")
    if width >= cell_len(path):
        assert label.plain == path
    elif width:
        assert "…" in label.plain


def test_path_labels_keep_filename_parent_and_style_literal_text() -> None:
    path = "kolega_code/" + "deeply/nested/" * 12 + "tui/transcript.py"
    label = path_label(path, 45)
    assert label.plain.startswith("kolega_code/…/")
    assert label.plain.endswith("/tui/transcript.py")
    assert label.spans[-1].style == "bold"
    assert label.get_style_at_offset(Console(), len(label.plain) - 1).dim is not True
    assert path_label("src/[bold].py").plain == "src/[bold].py"


@pytest.mark.parametrize("filename", ["settings.py", "文件名.py", "archive.tar.gz"])
@pytest.mark.parametrize("spare_cells", [0, 1])
def test_path_label_drops_directory_decoration_before_shortening_a_filename_that_fits(
    filename: str, spare_cells: int
) -> None:
    width = cell_len(filename) + spare_cells
    label = path_label("src/deep/" + filename, width)
    assert label.plain == filename
    assert label.cell_len <= width
    assert label.spans[-1].style == "bold"


@pytest.mark.parametrize("width", [0, 1, 2, 5, 20, 40, 80])
@pytest.mark.parametrize("command", ["git status --short", "echo one\necho two\necho three", "echo 文件 " * 300])
def test_command_preview_has_at_most_two_cell_bounded_lines(command: str, width: int) -> None:
    preview = command_preview(command, width)
    lines = preview.plain.splitlines()
    assert len(lines) <= 2
    assert all(cell_len(line) <= width for line in lines)
    if width >= 5:
        assert lines[0].startswith("$ ")
    if width >= 5 and command != "git status --short":
        assert preview.plain.endswith("…")


def test_short_commands_stay_inline_long_and_multiline_commands_do_not() -> None:
    for command, separate in [
        ("git status --short", False),
        ("echo " + "long-argument " * 30, True),
        ("echo one\necho two", True),
    ]:
        entry = ConversationEntry(
            kind="tool_result", content="OUTPUT", tool_name="exec_command", tool_display={"command": command}
        )
        title, needs_preview = tool_title(entry, 80)
        assert needs_preview is separate
        assert ("git status --short" in title.plain) is (not separate)
        assert title.plain.endswith("done")
        assert title.cell_len <= 80


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["path", "command"])
async def test_details_remain_complete_and_selectable_across_resize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    value = (
        "src/" + "directory/" * 40 + "target.py"
        if kind == "path"
        else "uv run pytest " + "tests/cli/test_rendering.py " * 16 + "\necho finished"
    )
    name = "read" if kind == "path" else "exec_command"
    args = {"file_path" if kind == "path" else "command": value}
    async with app.run_test(size=(100, 40)) as pilot:
        app._set_sidebar_visible(False)
        app._render_event(_event("tool_call", name, "details", args))
        await _wait_for_layout(
            pilot, lambda: any(widget.entry.tool_call_id == "details" for widget in app.query(ToolEntryWidget))
        )
        widget = app.query(ToolEntryWidget).last()
        title = widget.query_one(CollapsibleTitle)
        collapsible = widget.query_one(Collapsible)
        preview = widget.query_one(".tool-command-preview", Static)
        details = widget.query_one(".tool-details", Static)
        await _wait_for_layout(pilot, lambda: title.size.height == 1 and (kind == "path" or preview.size.height == 2))
        if kind == "path":
            assert "target.py" in title.render_line(0).text
            assert not preview.display
        else:
            assert preview.display
        title.focus()
        await pilot.press("enter")
        await _wait_for_layout(pilot, lambda: not collapsible.collapsed and details.size.height > 0)
        assert not preview.display
        assert str(details.render()) == value
        assert details.get_selection(Selection(None, None)) == (value, "\n")
        app._render_event(_event("tool_result", name, "details", args, "UNMODIFIED_OUTPUT"))
        for width in (40, 120, 80):
            await pilot.resize_terminal(width, 40)
            await _wait_for_layout(pilot, lambda: widget.size.width <= width and title.size.height == 1)
            assert app.query(ToolEntryWidget).last() is widget
            assert not collapsible.collapsed
            assert str(details.render()) == value
            assert widget.entry.full_content == "UNMODIFIED_OUTPUT"
        collapsible.collapsed = True
        await _wait_for_layout(pilot, lambda: kind == "path" or preview.display and preview.size.height == 2)
        assert not details.is_on_screen


@pytest.mark.asyncio
async def test_long_command_moves_back_into_title_when_terminal_widens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    command = "uv run pytest tests/cli/test_tool_subject_titles.py -q"
    async with app.run_test(size=(60, 30)) as pilot:
        app._set_sidebar_visible(False)
        app._render_event(_event("tool_result", "exec_command", "resize", {"command": command}, "passed"))
        await _wait_for_layout(
            pilot, lambda: any(widget.entry.tool_call_id == "resize" for widget in app.query(ToolEntryWidget))
        )
        widget = app.query(ToolEntryWidget).last()
        preview = widget.query_one(".tool-command-preview", Static)
        await _wait_for_layout(pilot, lambda: preview.display and preview.size.height > 0)
        await pilot.resize_terminal(140, 30)
        await _wait_for_layout(pilot, lambda: command in Text.from_markup(widget._title).plain and not preview.display)
        assert widget.query_one(Collapsible).collapsed


@pytest.mark.asyncio
@pytest.mark.parametrize("before_call", [False, True])
async def test_multifile_previews_keep_each_file_once_and_single_edits_have_no_duplicate_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, before_call: bool
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    async with app.run_test(size=(100, 40)) as pilot:
        call = _event("tool_call", "apply_patch", "patch", {})
        call.content["tool_display"] = {"paths": ["src/first.py", "src/second.py"]}
        if not before_call:
            app._render_event(call)
            pending = app._tool_entries["patch"]
            preview = _render(app._tool_preview_renderable(pending))
            assert preview.count("src/first.py") == 1
            assert preview.count("src/second.py") == 1
        for path in ("src/first.py", "src/second.py"):
            app._apply_edit_preview(_preview("patch", path))
        if before_call:
            app._render_event(call)
        entry = app._tool_entries["patch"]
        assert [item["path"] for item in edit_previews(entry.edit_preview)] == ["src/first.py", "src/second.py"]
        app._apply_edit_preview(_preview("patch", "src/first.py", added="+updated"))
        assert len(edit_previews(entry.edit_preview)) == 2
        combined = Text.from_markup(app._tool_entry_title(entry, 100)).plain + _render(
            app._tool_preview_renderable(entry)
        )
        assert combined.count("src/first.py") == 1
        assert combined.count("src/second.py") == 1
        assert "+updated" in combined
        assert combined.count("+new") == 1
        app._render_event(_event("tool_result", "edit", "single", {"path": "src/[literal].py"}, "raw"))
        app._apply_edit_preview({**_preview("single", "src/[literal].py"), "tool_name": "edit"})
        single = app._tool_entries["single"]
        combined = Text.from_markup(app._tool_entry_title(single, 100)).plain + _render(
            app._tool_preview_renderable(single)
        )
        assert combined.count("src/[literal].py") == 1
        assert "+1 -1" in combined
        await _wait_for_layout(
            pilot,
            lambda: {widget.entry.tool_call_id for widget in app.query(ToolEntryWidget)} >= {"patch", "single"},
        )
        single_widget = next(widget for widget in app.query(ToolEntryWidget) if widget.entry.tool_call_id == "single")
        single_widget.query_one(Collapsible).collapsed = False
        await _wait_for_layout(pilot, lambda: single_widget.query_one(".tool-body", Static).is_on_screen)
        assert not single_widget.query_one(".tool-details", Static).display


@pytest.mark.asyncio
async def test_restore_retains_safe_long_display_details_without_mutating_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    path = "src/" + "directory/" * 40 + "target.py"
    command = "echo " + "ordinary-argument " * 30 + "\necho finished"
    calls = [
        ToolCall(id="path", name="read", input={"file_path": str(app.project_path / path)}),
        ToolCall(id="command", name="exec_command", input={"command": command}),
    ]
    history = [
        Message(role="assistant", content=list(calls)).to_dict(),
        Message(
            role="user",
            content=[
                ToolResult(tool_use_id=call.id, name=call.name, content="EXACT_OUTPUT", is_error=False)
                for call in calls
            ],
        ).to_dict(),
    ]
    original = repr(history)
    async with app.run_test():
        entries = app._conversation_entries_from_history_items(history)
        assert entries[0].tool_display == {"paths": [path]}
        assert entries[1].tool_display == {"command": command}
        assert all(entry.full_content == "EXACT_OUTPUT" for entry in entries)
        assert repr(history) == original


@pytest.mark.asyncio
async def test_subagent_receives_command_details_and_accumulates_patch_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    command = "echo one\necho two"
    async with app.run_test(size=(120, 40)) as pilot:
        event = _sub_agent_event(message_type="tool_call", text="Calling exec_command", tool_description="exec_command")
        event.content.update(tool_call_id="child", tool_display={"command": command})
        app._render_event(event)
        activity = next(iter(app._sub_agent_activities.values()))
        assert activity.tool_steps["child"].tool_display == {"command": command}
        for path in ("src/one.py", "src/two.py"):
            app._apply_sub_agent_edit_preview(
                AgentEvent(
                    event_type="file_edit_preview",
                    sender=event.sender,
                    sub_agent_info=event.sub_agent_info,
                    content=_preview("patch", path),
                )
            )
        patch = _sub_agent_event(message_type="tool_call", text="Calling apply_patch", tool_description="apply_patch")
        patch.content.update(tool_call_id="patch", tool_display={"paths": ["src/one.py", "src/two.py"]})
        app._render_event(patch)
        assert len(edit_previews(activity.tool_steps["patch"].edit_preview)) == 2
        app.action_open_sub_agent()
        screen = app._sub_agent_inspector
        assert screen is not None
        await _wait_for_layout(pilot, lambda: len(screen.query(ToolEntryWidget)) == 2)
        widget = next(widget for widget in screen.query(ToolEntryWidget) if widget.entry.tool_call_id == "child")
        assert widget.query_one(".tool-command-preview", Static).display
        await pilot.press("escape")


@pytest.mark.asyncio
async def test_drag_selects_expanded_literal_command_and_preserves_toggle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    command = "echo '[bold]literal[/bold]'\necho next"
    async with app.run_test(size=(100, 40)) as pilot:
        app._set_sidebar_visible(False)
        app._render_event(_event("tool_result", "exec_command", "copy", {"command": command}, "OUTPUT"))
        await _wait_for_layout(
            pilot, lambda: any(widget.entry.tool_call_id == "copy" for widget in app.query(ToolEntryWidget))
        )
        widget = app.query(ToolEntryWidget).last()
        collapsible = widget.query_one(Collapsible)
        collapsible.collapsed = False
        details = widget.query_one(".tool-details", Static)
        await _wait_for_layout(pilot, lambda: details.size.height == 2 and details.is_on_screen)
        await pilot.mouse_down(details, offset=(0, 0))
        await pilot._post_mouse_events([events.MouseMove], details, offset=(9, 1), button=1)
        await pilot.mouse_up(details, offset=(9, 1))
        assert app.screen.get_selected_text() == command
        assert not collapsible.collapsed
