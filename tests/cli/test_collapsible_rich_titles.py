"""Rich-generated tool labels must not be reparsed as Textual markup."""

from __future__ import annotations

import pytest
from rich.text import Text
from textual.app import App, ComposeResult
from textual.content import Content
from textual.style import Style
from textual.widgets import Collapsible, Static
from textual.widgets._collapsible import CollapsibleTitle

from kolega_code.cli.tui.state import ConversationEntry
from kolega_code.cli.tui.transcript import TranscriptRenderingMixin
from kolega_code.cli.tui.widgets import SelectableCollapsible, ToolEntryWidget

from .test_tool_subject_titles import _wait_for_layout


LITERAL_SUBJECTS = [
    "echo '=== geometric result ===' \\",
    "echo [",
    "echo [A]",
    r"echo \\[dim]",
    "echo :smile: :warning:",
]


@pytest.mark.parametrize("kind", ["tool_call", "tool_result", "tool_error"])
@pytest.mark.parametrize(
    "tool_name,subject",
    [
        ("exec_command", "echo '=== geometric result ==='"),
        ("exec_command", "pwd"),
        ("read", "src/file.py"),
        ("read", "file.py"),
        ("exec_command", "printf '[bold]literal[/bold] [/dim]'"),
        *(("exec_command", subject) for subject in LITERAL_SUBJECTS),
        ("read", "C:\\src\\file.py"),
        ("read", "/tmp/:smile:/result.py"),
    ],
)
def test_generated_tool_titles_construct_without_markup_errors(kind: str, tool_name: str, subject: str) -> None:
    entry = ConversationEntry(kind=kind, content="", tool_name=tool_name, tool_subject=subject)
    title = TranscriptRenderingMixin._tool_entry_title(entry)
    collapsible = SelectableCollapsible(title=title)
    label = collapsible._title.label
    assert isinstance(label, Content)
    assert tool_name in label.plain
    assert subject in label.plain
    assert {"tool_call": "running", "tool_result": "done", "tool_error": "failed"}[kind] in label.plain


@pytest.mark.asyncio
@pytest.mark.parametrize("subject", LITERAL_SUBJECTS)
async def test_tool_title_refresh_and_resize_keep_literal_subject(subject: str) -> None:
    entry = ConversationEntry(kind="tool_call", content="", tool_name="exec_command", tool_subject="pwd")
    widget = ToolEntryWidget(
        entry,
        TranscriptRenderingMixin._tool_entry_title,
        title_for_width=TranscriptRenderingMixin._tool_entry_title,
    )

    class TitleApp(App[None]):
        def compose(self) -> ComposeResult:
            yield widget

    app = TitleApp()
    async with app.run_test(size=(140, 30)) as pilot:
        title = widget.query_one(CollapsibleTitle)
        collapsible = widget.query_one(Collapsible)
        await _wait_for_layout(pilot, lambda: "pwd" in str(title.label))

        entry.tool_subject = subject
        widget.refresh_content()
        await _wait_for_layout(pilot, lambda: subject in str(title.label))
        assert isinstance(title.label, Content)
        assert "running" in title.label.plain
        assert subject in title.render_line(0).text

        title.focus()
        await pilot.press("enter")
        await _wait_for_layout(pilot, lambda: not collapsible.collapsed)
        entry.kind = "tool_result"
        entry.content = "unchanged tool output"
        widget.refresh_content()
        await _wait_for_layout(pilot, lambda: "done" in str(title.label))

        await pilot.resize_terminal(25, 30)
        details = widget.query_one(".tool-details", Static)
        await _wait_for_layout(pilot, lambda: widget.size.width == 25 and title.size.height == 1 and details.display)
        assert "done" in title.render_line(0).text
        rendered_details = details.render()
        assert isinstance(rendered_details, Content)
        assert rendered_details.plain == subject
        assert entry.tool_subject == subject
        assert not collapsible.collapsed
        assert entry.content == "unchanged tool output"
        await pilot.resize_terminal(140, 30)
        await _wait_for_layout(pilot, lambda: widget.size.width == 140 and subject in title.render_line(0).text)
        assert "done" in str(title.label)
        assert not collapsible.collapsed


def test_rich_title_styles_survive_conversion() -> None:
    title = Text("tool", style="bold")
    title.append(" · echo \\", style="dim")
    collapsible = SelectableCollapsible(title=title.markup)
    label = collapsible._title.label
    assert isinstance(label, Content)
    assert label.plain == "tool · echo \\"
    assert any(
        span.start == 0 and span.end == len(label.plain) and isinstance(span.style, Style) and span.style.bold
        for span in label.spans
    )
    assert any(
        span.start == 4 and span.end == len(label.plain) and isinstance(span.style, Style) and span.style.dim
        for span in label.spans
    )
