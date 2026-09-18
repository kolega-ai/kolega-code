"""Read-only checkbox presentation without changing the saved Markdown."""

from __future__ import annotations

import re
from typing import ClassVar

from markdown_it.token import Token
from rich.cells import cell_len
from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import ListItem, Markdown, MarkdownContext, MarkdownElement, Paragraph
from rich.segment import Segment
from rich.style import Style


_CHECKBOX = re.compile(r"^\[([ xX])\](?:[ \t]+|$)")


class _TaskItem(ListItem):
    checked: bool | None = None

    @classmethod
    def create(cls, markdown: Markdown, token: Token) -> _TaskItem:
        item = cls()
        item.checked = token.meta.get("task_checked")
        return item

    def on_child_close(self, context: MarkdownContext, child: MarkdownElement) -> bool:
        if self.checked is not None and isinstance(child, Paragraph):
            # Style only this item's paragraphs, not a nested list whose tasks
            # may have different states. Keep inline Markdown spans intact.
            child.text.stylize(Style(meta={"task_checked": self.checked}))
        return super().on_child_close(context, child)

    def _render_task(self, console: Console, options: ConsoleOptions, *, number: int | None = None) -> RenderResult:
        marker = "[x] " if self.checked else "[ ] "
        if number is not None:
            marker = f"{number}. {marker}"
        # Leave at least one cell for content when a deeply nested item is narrow.
        marker = marker[: max(0, options.max_width - 1)]
        marker_width = cell_len(marker)
        lines = console.render_lines(
            self.elements, options.update(width=max(1, options.max_width - marker_width)), style=self.style
        )
        marker_style = Style(meta={"task_checked": self.checked, "task_marker": True})
        for index, line in enumerate(lines):
            yield Segment(marker if index == 0 else " " * marker_width, marker_style)
            yield from line
            yield Segment.line()

    def render_bullet(self, console: Console, options: ConsoleOptions) -> RenderResult:
        if self.checked is None:
            yield from super().render_bullet(console, options)
        else:
            yield from self._render_task(console, options)

    def render_number(self, console: Console, options: ConsoleOptions, number: int, last_number: int) -> RenderResult:
        if self.checked is None:
            yield from super().render_number(console, options, number, last_number)
        else:
            yield from self._render_task(console, options, number=number)


class TaskListMarkdown(Markdown):
    """Rich Markdown with literal, tagged checkboxes on actual list items only."""

    elements: ClassVar[dict[str, type[MarkdownElement]]] = {**Markdown.elements, "list_item_open": _TaskItem}

    def __init__(self, source: str, *, code_theme: str) -> None:
        super().__init__(source, code_theme=code_theme, hyperlinks=False)
        for index, token in enumerate(self.parsed[:-2]):
            if token.type != "list_item_open" or self.parsed[index + 1].type != "paragraph_open":
                continue
            inline = self.parsed[index + 2]
            if inline.type != "inline" or not inline.children:
                continue
            # Match the unparsed text too: escaped checkboxes, code spans, and
            # Markdown links with a label of "x" must remain ordinary text.
            match = _CHECKBOX.match(inline.content)
            first = inline.children[0]
            rendered_match = _CHECKBOX.match(first.content) if first.type == "text" else None
            if match is None or rendered_match is None:
                continue
            token.meta["task_checked"] = match[1].lower() == "x"
            first.content = first.content[rendered_match.end() :]

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        # Rich separates the opening list from a preceding block even when it
        # is the first block. A compact task card needs no empty leading row.
        started = False
        for renderable in super().__rich_console__(console, options):
            if not started and isinstance(renderable, Segment) and renderable.text == "\n":
                continue
            started = True
            yield renderable
