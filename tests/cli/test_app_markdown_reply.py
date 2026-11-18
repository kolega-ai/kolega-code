from pathlib import Path

import pytest
from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import Markdown
from textual.selection import Selection

from kolega_code.cli.tui.state import ConversationEntry

from ._app_test_utils import _build_sub_agent_test_app, wait_for_layout


@pytest.mark.asyncio
async def test_cold_markdown_reply_paint_reuses_layout_at_final_width(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    markdown = "A **styled** reply with `inline code`.\n\n```python\nreturn 123\n```\n\n" + "\n".join(
        f"- Item {index}: a deliberately long line to verify soft wrapping near the transcript edge."
        for index in range(400)
    )
    reply = ConversationEntry(kind="assistant", content=markdown, complete=True)
    passes: list[int] = []
    original = Markdown.__rich_console__

    def record(self: Markdown, console: Console, options: ConsoleOptions) -> RenderResult:
        if self.markup == markdown:
            passes.append(options.max_width)
        return original(self, console, options)

    monkeypatch.setattr(Markdown, "__rich_console__", record)
    async with app.run_test(size=(100, 24)) as pilot:
        await wait_for_layout(pilot, app._startup_complete.is_set)
        for index in range(32):
            app._add_conversation_entry(ConversationEntry(kind="user", content=f"before {index}"))
        app._add_conversation_entry(reply)
        for index in range(32):
            app._add_conversation_entry(ConversationEntry(kind="user", content=f"after {index}"))
        app._render_conversation()
        view = app._conversation
        widget = app._entry_widgets[reply.entry_id]
        await wait_for_layout(
            pilot, lambda: view.is_at_bottom() and not app._conversation_anchor_pending and widget.size.height > 400
        )
        assert passes  # measured during layout
        assert widget not in view.screen._compositor.visible_widgets
        assert not widget._render_cache.lines  # no warm paint to satisfy the assertion
        measured = passes.copy()

        view.scroll_to(y=widget.virtual_region.bottom - widget.styles.gutter.bottom - 1, animate=False, immediate=True)
        await wait_for_layout(pilot, lambda: bool(widget._render_cache.lines))
        assert widget.content_region.intersection(view.scrollable_content_region).height == 1
        assert passes == measured

        top = widget.render_line(0)
        assert "A styled reply with inline code." in top.text
        assert any("styled" in segment.text and segment.style is not None and segment.style.bold for segment in top)
        code = next(line for line in widget._render_cache.lines if "return 123" in line.text)
        assert any(segment.text == "return" and segment.style is not None and segment.style.color for segment in code)
        selected = widget.get_selection(Selection(None, None))
        assert selected is not None
        assert "Item 0:" in selected[0] and "Item 399:" in selected[0]
        assert "return 123" in selected[0]
        assert "**" not in selected[0] and "`" not in selected[0]

        old_width = widget.size.width
        old_height = widget.size.height
        await pilot.resize_terminal(72, 24)
        await wait_for_layout(pilot, lambda: widget.size.width < old_width and widget.size.height > old_height)
        assert passes[-1] < measured[-1]
        assert "A styled reply" in widget.render_line(0).text

        count = len(passes)
        widget.styles.text_style = "underline"
        top = widget.render_line(0)
        assert any(
            "styled" in segment.text and segment.style is not None and segment.style.underline for segment in top
        )
        assert len(passes) == count + 1
