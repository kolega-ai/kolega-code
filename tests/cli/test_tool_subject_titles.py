"""Tool subjects through real transcript lifecycles, restoration, and layout."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from rich.style import Style
from rich.text import Text
from textual import events
from textual.widgets import Collapsible
from textual.widgets._collapsible import CollapsibleTitle

from kolega_code.cli.tui.state import ConversationEntry
from kolega_code.cli.tui.transcript import TranscriptRenderingMixin
from kolega_code.cli.tui.widgets import ToolEntryWidget
from kolega_code.events import AgentEvent
from kolega_code.llm.models import ContentBlock, Message, ToolCall, ToolResult, WebSearchCallBlock

from ._app_test_utils import _build_sub_agent_test_app, _sub_agent_event


async def _wait_for_layout(pilot: Any, predicate: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline:
        await pilot.pause()
        if predicate():
            return
    assert predicate(), "tool subject layout did not settle"


def _tool_event(kind: str, call_id: str, *, subject: object = None, text: str = "") -> AgentEvent:
    content: dict[str, object] = {
        "message_type": kind,
        "tool_description": "read",
        "tool_call_id": call_id,
        "text": text,
    }
    if subject is not None:
        content["tool_subject"] = subject
    return AgentEvent(event_type="chat_message", sender="coder", content=content)


@pytest.mark.parametrize("kind,state", [("tool_call", "running"), ("tool_result", "done"), ("tool_error", "failed")])
@pytest.mark.parametrize("width", [0, 1, 7, 15, 30, 40, 80, 120])
def test_title_reserves_status_and_uses_cell_width(kind: str, state: str, width: int) -> None:
    entry = ConversationEntry(
        kind=kind,
        content="900 matches, 10000 lines in 5 seconds",
        tool_name="read",
        tool_subject="src/" + "文件" * 70 + ".py",
    )
    title = Text.from_markup(TranscriptRenderingMixin._tool_entry_title(entry, width))
    assert title.cell_len <= width
    assert "\n" not in title.plain
    if width >= len(state):
        assert state in title.plain
    if width >= 30:
        assert ".py" in title.plain
        assert "…" in title.plain
    assert "matches" not in title.plain
    assert "10000" not in title.plain


@pytest.mark.parametrize("subject", ["[bold]src/file.py[/bold]", "[link=https://example.test]src/file.py[/link]"])
def test_subject_is_literal_and_existing_lsp_badge_is_preserved(subject: str) -> None:
    entry = ConversationEntry(
        kind="tool_result",
        content="LSP diagnostics (2 warnings):\nfile.py:1:1 warning: unused",
        tool_name="read",
        tool_subject=subject,
    )
    title = Text.from_markup(TranscriptRenderingMixin._tool_entry_title(entry, 120))
    assert subject in title.plain
    assert "2 LSP warnings" in title.plain
    assert "done" in title.plain
    assert all(
        not (Style.parse(span.style) if isinstance(span.style, str) else span.style).link for span in title.spans
    )


@pytest.mark.asyncio
async def test_live_parallel_subjects_survive_streaming_results_and_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    async with app.run_test():
        app._render_event(_tool_event("tool_call", "first", subject="src/first.py"))
        app._render_event(_tool_event("tool_call", "second", subject="src/second.py"))
        app._render_event(
            AgentEvent(
                event_type="tool_streaming_update",
                sender="coder",
                content={
                    "tool_name": "read",
                    "tool_call_id": "first",
                    "stream_mode": "append",
                    "text": "partial",
                    "is_complete": False,
                },
            )
        )
        app._render_event(_tool_event("tool_error", "second", subject={}, text="failed to read"))
        app._render_event(_tool_event("tool_result", "first", text="full result"))
        first = app._tool_entries["first"]
        second = app._tool_entries["second"]
        assert first.tool_subject == "src/first.py"
        assert second.tool_subject == "src/second.py"
        assert first.kind == "tool_result" and first.full_content == "full result"
        assert second.kind == "tool_error" and second.full_content == "failed to read"

        app._render_event(_tool_event("tool_call", "late"))
        app._render_event(_tool_event("tool_result", "late", subject="src/late.py", text="result"))
        assert app._tool_entries["late"].tool_subject == "src/late.py"
        app._render_event(_tool_event("tool_error", "orphan", text="result without a call"))
        assert app._tool_entries["orphan"].tool_subject == ""


@pytest.mark.asyncio
async def test_event_subject_is_sanitized_not_inferred_from_raw_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    async with app.run_test():
        event = _tool_event("tool_call", "unsafe", subject="curl --password FAKE_PRIVATE_VALUE")
        event.content["arguments"] = {"contents": "DO_NOT_DISPLAY_RAW_INPUT"}
        app._render_event(event)
        title = Text.from_markup(app._tool_entry_title(app._tool_entries["unsafe"]))
        assert "FAKE_PRIVATE_VALUE" not in title.plain
        assert "DO_NOT_DISPLAY_RAW_INPUT" not in title.plain
        assert "curl" in title.plain
        app._render_event(_tool_event("tool_call", "url", subject="https://fake:pass@example.test/a?token=FAKE_QUERY"))
        assert app._tool_entries["url"].tool_subject == "https://example.test/a"


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["Kolega Dark", "Solarized"])
async def test_tool_title_resizes_without_replacing_expanded_widget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, theme: str
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    subject = "src/" + "文件" * 70 + ".py"
    async with app.run_test(size=(120, 40)) as pilot:
        app._apply_theme(theme)
        app._set_sidebar_visible(False)
        app._render_event(_tool_event("tool_call", "resize", subject=subject))
        await _wait_for_layout(
            pilot, lambda: any(widget.entry.tool_call_id == "resize" for widget in app.query(ToolEntryWidget))
        )
        widget = next(widget for widget in app.query(ToolEntryWidget) if widget.entry.tool_call_id == "resize")
        title = widget.query_one(CollapsibleTitle)
        collapsible = widget.query_one(Collapsible)
        await _wait_for_layout(pilot, lambda: title.size.height == 1 and "src/" in widget._title)
        wide = Text.from_markup(widget._title).plain
        title.focus()
        await pilot.press("enter")
        await _wait_for_layout(pilot, lambda: not collapsible.collapsed)
        app._render_event(_tool_event("tool_result", "resize", text="full output is still expandable"))

        for columns in (80, 40, 120):
            previous_width = widget.size.width
            previous_title = Text.from_markup(widget._title).plain
            await pilot.resize_terminal(columns, 40)
            await _wait_for_layout(
                pilot,
                lambda: (
                    title.size.height == 1
                    and widget.size.width != previous_width
                    and widget.size.width <= columns
                    and "done" in Text.from_markup(widget._title).plain
                    and Text.from_markup(widget._title).plain != previous_title
                ),
            )
            assert app.query(ToolEntryWidget).last() is widget
            assert not collapsible.collapsed
            assert widget.entry.tool_subject == subject
            assert widget.entry.full_content == "full output is still expandable"
            rendered = title.render_line(0).text
            assert "done" in rendered
            assert "src/" in rendered
        assert len(Text.from_markup(widget._title).plain) >= len(wide) - len("running") + len("done")


@pytest.mark.asyncio
async def test_drag_selection_copies_subject_and_click_still_toggles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    async with app.run_test(size=(100, 35)) as pilot:
        app._set_sidebar_visible(False)
        app._render_event(_tool_event("tool_result", "copy", subject="src/copy.py", text="body"))
        await _wait_for_layout(
            pilot, lambda: any(widget.entry.tool_call_id == "copy" for widget in app.query(ToolEntryWidget))
        )
        widget = next(widget for widget in app.query(ToolEntryWidget) if widget.entry.tool_call_id == "copy")
        title = widget.query_one(CollapsibleTitle)
        await _wait_for_layout(pilot, lambda: title.size.height == 1 and "src/copy.py" in title.render_line(0).text)
        await pilot.mouse_down(title, offset=(2, 0))
        await pilot._post_mouse_events([events.MouseMove], title, offset=(title.size.width - 2, 0), button=1)
        await pilot.mouse_up(title, offset=(title.size.width - 2, 0))
        assert "src/copy.py" in (app.screen.get_selected_text() or "")
        await pilot.click(title, offset=(1, 0))
        await _wait_for_layout(pilot, lambda: not widget.query_one(Collapsible).collapsed)


@pytest.mark.asyncio
async def test_sub_agent_subjects_use_shared_inspector_titles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    async with app.run_test(size=(120, 40)) as pilot:
        for call_id, subject in (("child-a", "src/child-a.py"), ("child-b", "src/child-b.py")):
            event = _sub_agent_event(message_type="tool_call", text="Calling read", tool_description="read")
            event.content.update(tool_call_id=call_id, tool_subject=subject)
            app._render_event(event)
        event = _sub_agent_event(message_type="tool_error", text="failure", tool_description="read")
        event.content["tool_call_id"] = "child-a"
        app._render_event(event)
        activity = next(iter(app._sub_agent_activities.values()))
        assert [step.tool_subject for step in activity.tool_steps.values()] == ["src/child-a.py", "src/child-b.py"]
        assert [step.kind for step in activity.tool_steps.values()] == ["tool_error", "tool_call"]
        app.action_open_sub_agent()
        screen = app._sub_agent_inspector
        assert screen is not None
        await _wait_for_layout(pilot, lambda: len(screen.query(ToolEntryWidget)) == 2)
        widgets = list(screen.query(ToolEntryWidget))
        await _wait_for_layout(
            pilot, lambda: all(widget.query_one(CollapsibleTitle).size.height == 1 for widget in widgets)
        )
        assert "src/child-a.py" in Text.from_markup(widgets[0]._title).plain
        assert "failed" in Text.from_markup(widgets[0]._title).plain
        assert "src/child-b.py" in Text.from_markup(widgets[1]._title).plain
        assert "running" in Text.from_markup(widgets[1]._title).plain
        copied = screen._trajectory_text(activity)
        assert "[tool_error] read · src/child-a.py" in copied
        assert "[tool_call] read · src/child-b.py" in copied
        await pilot.press("escape")


@pytest.mark.asyncio
async def test_restore_derives_safe_subjects_from_calls_and_hosted_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    assert app.config is not None
    app.config.openai_api_key = "FAKE_CONFIGURED_SUBJECT_KEY"
    calls: list[ContentBlock] = [
        ToolCall(
            id="provider-a", execution_id="exec-a", name="read", input={"file_path": str(app.project_path / "src/a.py")}
        ),
        ToolCall(
            id="provider-b",
            execution_id="exec-b",
            name="apply_patch",
            input="*** Begin Patch\n*** Update File: src/b.py\n@@\n-old\n+RAW_PATCH_BODY\n*** End Patch\n",
            input_kind="freeform",
        ),
        ToolCall(
            id="provider-c",
            execution_id="exec-c",
            name="exec_command",
            input={"command": "echo FAKE_CONFIGURED_SUBJECT_KEY"},
        ),
    ]
    history = [
        Message(role="assistant", content=calls).to_dict(),
        Message(
            role="user",
            content=[
                ToolResult(
                    tool_use_id="provider-b",
                    execution_id="exec-b",
                    name="apply_patch",
                    content="rejected",
                    is_error=True,
                ),
                ToolResult(
                    tool_use_id="provider-a", execution_id="exec-a", name="read", content="FILE_BODY", is_error=False
                ),
                ToolResult(
                    tool_use_id="provider-c",
                    execution_id="exec-c",
                    name="exec_command",
                    content="COMMAND_OUTPUT",
                    is_error=False,
                ),
            ],
        ).to_dict(),
        Message(
            role="assistant",
            content=[
                WebSearchCallBlock(
                    item_id="hosted-query", action={"type": "search", "queries": ["Textual layouts", "tool titles"]}
                ),
                WebSearchCallBlock(
                    item_id="hosted-url",
                    item_type="fetch_url_results",
                    action={"urls": ["https://fake:pass@example.test/docs?token=FAKE_QUERY"]},
                ),
            ],
        ).to_dict(),
    ]
    async with app.run_test():
        entries = app._conversation_entries_from_history_items(history)
        assert len(entries) == 5
        assert [entry.tool_subject for entry in entries[:2]] == ["src/a.py", "src/b.py"]
        assert [entry.kind for entry in entries[:2]] == ["tool_result", "tool_error"]
        assert "FAKE_CONFIGURED_SUBJECT_KEY" not in entries[2].tool_subject
        assert entries[3].tool_subject == "Textual layouts, tool titles"
        assert entries[4].tool_subject == "https://example.test/docs"
        assert entries[4].tool_name == "fetch_url (hosted)"
        assert "RAW_PATCH_BODY" not in str([entry.tool_subject for entry in entries])
        assert entries[0].full_content == "FILE_BODY"
