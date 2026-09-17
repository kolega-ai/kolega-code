"""Compact, selectable startup card with independently expandable configuration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar, Sequence

from rich.cells import cell_len, split_graphemes
from rich.console import RenderableType
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.message import Message as TextualMessage
from textual.widgets import Collapsible, Static
from textual.widgets._collapsible import CollapsibleTitle

from .state import ConversationEntry
from .widgets import ConversationEntryWidget, SelectableCollapsible, ToolEntryWidget


def display_project_path(path: Path) -> str:
    """Abbreviate home without changing the path's meaning."""
    try:
        relative = path.relative_to(Path.home())
    except ValueError:
        return str(path)
    return "~" if relative == Path(".") else f"~/{relative}"


@dataclass(frozen=True)
class RecentSessionItem:
    session_id: str
    title: str
    updated_at: str
    locked: bool = False


class StartupText(ConversationEntryWidget):
    """Startup fields can change without changing length, and reflow on resize."""

    def _entry_snapshot(self) -> tuple[object, ...]:
        return self.entry.content, self.content_size.width

    def on_resize(self, event: events.Resize) -> None:
        self.refresh_content()


class StartupDisclosure(SelectableCollapsible):
    """Only explicit disclosure clicks may scroll the transcript to this card."""

    def _watch_collapsed(self, collapsed: bool) -> None:
        # Textual's default watcher schedules scroll_visible even on a programmatic
        # fold, undoing the transcript's bottom anchor after the first submission.
        self._update_collapsed(collapsed)
        self.post_message(self.Collapsed(self) if collapsed else self.Expanded(self))

    def _on_collapsible_title_toggle(self, event: CollapsibleTitle.Toggle) -> None:
        event.prevent_default()
        super()._on_collapsible_title_toggle(event)
        self.call_after_refresh(self.scroll_visible)


class StartupEntryWidget(ToolEntryWidget):
    """Keep disclosure state on the entry so scrollback remounts preserve it."""

    @dataclass
    class ResumeRequested(TextualMessage):
        session_id: str

    DEFAULT_CSS = """
    StartupEntryWidget {
        height: auto;
        padding: 0;
    }
    StartupEntryWidget > Collapsible {
        border: round $surface-lighten-2;
        padding: 0 1;
        background: transparent;
    }
    StartupEntryWidget CollapsibleTitle {
        padding: 0;
        color: $text;
    }
    StartupEntryWidget Collapsible Contents {
        padding: 0;
    }
    StartupEntryWidget StartupText {
        height: auto;
        padding: 0;
    }
    StartupEntryWidget .startup-summary {
        margin: 0;
    }
    StartupEntryWidget RecentSessionsWidget {
        display: none;
        height: auto;
        margin: 0;
        padding: 0;
    }
    StartupEntryWidget .startup-recent-header {
        height: 1;
        margin: 0;
        padding: 0;
        color: $text-muted;
    }
    StartupEntryWidget RecentSessionRow {
        height: 1;
        margin: 0;
        padding: 0;
        color: $text;
        background: transparent;
    }
    StartupEntryWidget RecentSessionRow:hover {
        background: $surface-lighten-1;
    }
    StartupEntryWidget RecentSessionRow:focus {
        background: $accent 20%;
    }
    StartupEntryWidget RecentSessionRow:disabled {
        color: $text-muted;
    }
    StartupEntryWidget .startup-configuration {
        border: none;
        padding: 0;
        background: transparent;
    }
    StartupEntryWidget .startup-details {
        margin: 0;
    }
    """

    def __init__(
        self,
        entry: ConversationEntry,
        title_factory: Callable[[ConversationEntry, int | None], str],
        summary_factory: Callable[[ConversationEntry, int], RenderableType],
        details_factory: Callable[[ConversationEntry], RenderableType],
        recent_sessions_factory: Callable[[], Sequence[RecentSessionItem]] | None = None,
    ) -> None:
        super().__init__(entry, lambda item: title_factory(item, None), title_for_width=title_factory)
        self._summary_factory = summary_factory
        self._details_factory = details_factory
        self._recent_sessions_factory = recent_sessions_factory
        self._summary: StartupText | None = None
        self._details: StartupText | None = None
        self._recent_sessions: RecentSessionsWidget | None = None
        self._configuration: Collapsible | None = None

    def compose(self) -> ComposeResult:
        self._summary = StartupText(self.entry, lambda entry: self._summary_factory(entry, self.content_size.width))
        self._summary.add_class("startup-summary")
        self._recent_sessions = RecentSessionsWidget(self._recent_sessions_factory)
        self._details = StartupText(self.entry, self._details_factory)
        self._details.add_class("startup-details")
        self._configuration = StartupDisclosure(
            self._details,
            title="Session & configuration",
            collapsed=not self.entry.startup_details_expanded,
            classes="startup-configuration",
        )
        self._collapsible = StartupDisclosure(
            self._summary,
            self._recent_sessions,
            self._configuration,
            title=self._title_factory(self.entry),
            collapsed=self.entry.startup_collapsed,
        )
        yield self._collapsible

    def refresh_content(self) -> None:
        if self._collapsible is None:
            return
        self._collapsible.collapsed = self.entry.startup_collapsed
        self._refresh_title()
        if self._summary is not None:
            self._summary.refresh_content()
        if self._recent_sessions is not None:
            self._recent_sessions.refresh_sessions()
        if self._details is not None:
            self._details.refresh_content()

    def on_collapsible_expanded(self, event: Collapsible.Expanded) -> None:
        self._remember_disclosure(event.collapsible)

    def on_collapsible_collapsed(self, event: Collapsible.Collapsed) -> None:
        self._remember_disclosure(event.collapsible)

    def _remember_disclosure(self, collapsible: Collapsible) -> None:
        if collapsible is self._collapsible:
            self.entry.startup_collapsed = collapsible.collapsed
            self._refresh_title()
        elif collapsible is self._configuration:
            self.entry.startup_details_expanded = not collapsible.collapsed


class RecentSessionsWidget(Vertical):
    """Compact, bounded recent-session list shown inside the startup card."""

    MAX_ROWS = 3

    def __init__(self, factory: Callable[[], Sequence[RecentSessionItem]] | None) -> None:
        super().__init__(classes="startup-recent-sessions")
        self._factory = factory
        self._header = Static("Recent sessions", markup=False, classes="startup-recent-header")
        self._rows = [RecentSessionRow() for _ in range(self.MAX_ROWS)]

    def compose(self) -> ComposeResult:
        self.refresh_sessions()
        yield self._header
        yield from self._rows

    def refresh_sessions(self) -> None:
        items = self._load_items()
        self.display = bool(items)
        for index, row in enumerate(self._rows):
            row.update_item(items[index] if index < len(items) else None)

    def _load_items(self) -> list[RecentSessionItem]:
        if self._factory is None:
            return []
        return list(self._factory())[: self.MAX_ROWS]


class RecentSessionRow(Static):
    """Focusable one-line row for a resumable session."""

    can_focus = True
    BINDINGS: ClassVar[list[Binding]] = [Binding("enter", "activate", "Resume session", show=False)]

    def __init__(self) -> None:
        super().__init__("", markup=False, classes="startup-recent-row")
        self._item: RecentSessionItem | None = None
        self.can_focus = False
        self.disabled = True
        self.display = False

    def update_item(self, item: RecentSessionItem | None) -> None:
        self._item = item
        self.display = item is not None
        self.disabled = item is None or item.locked
        self.can_focus = item is not None and not item.locked
        self._refresh_label()

    def on_mount(self) -> None:
        self._refresh_label()

    def on_resize(self, event: events.Resize) -> None:
        self._refresh_label()

    def on_click(self) -> None:
        self._activate()

    def action_activate(self) -> None:
        self._activate()

    def _activate(self) -> None:
        if self._item is None or self._item.locked:
            return
        self.post_message(StartupEntryWidget.ResumeRequested(self._item.session_id))

    def _refresh_label(self) -> None:
        item = self._item
        if item is None:
            self.update(Text("", no_wrap=True, overflow="crop"))
            return
        width = max(0, self.content_size.width or self.size.width)
        self.update(_recent_session_label(item, width))


def _recent_session_label(item: RecentSessionItem, width: int) -> Text:
    status = "in use" if item.locked else _relative_age(item.updated_at)
    title = _sanitize_recent_title(item.title)
    if width <= 0:
        return Text("", no_wrap=True, overflow="crop")
    status_width = cell_len(status)
    separator_width = 2 if status else 0
    title_width = max(0, width - status_width - separator_width)
    if title_width == 0 and status_width > width:
        status = _cell_prefix(status, width)
        status_width = cell_len(status)
        separator_width = 0
    rendered_title = _ellipsize_filename_like(title, title_width)
    gap = max(0, width - cell_len(rendered_title) - status_width)
    text = Text(no_wrap=True, overflow="crop")
    text.append(rendered_title)
    if gap:
        text.append(" " * gap)
    if status:
        text.append(status, style="dim")
    return text


def _sanitize_recent_title(title: str) -> str:
    cleaned: list[str] = []
    for character in title:
        if character.isprintable() and character not in "\r\n\t":
            cleaned.append(character)
        else:
            cleaned.append(" ")
    result = " ".join("".join(cleaned).split())
    return result or "Untitled session"


def _relative_age(value: str) -> str:
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return "unknown"
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    delta_seconds = max(0, int((datetime.now(timezone.utc) - timestamp.astimezone(timezone.utc)).total_seconds()))
    if delta_seconds < 60:
        return "just now"
    minutes = delta_seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    if days < 7:
        return f"{days}d ago"
    return timestamp.date().isoformat()


def _ellipsize_filename_like(value: str, width: int) -> str:
    if width <= 0:
        return ""
    if cell_len(value) <= width:
        return value
    if width == 1:
        return "…"
    suffix_width = max(1, min(width // 2, 24))
    suffix = _cell_suffix(value, suffix_width)
    prefix = _cell_prefix(value, width - cell_len(suffix) - 1)
    return f"{prefix}…{suffix}"


def _cell_prefix(value: str, width: int) -> str:
    if width <= 0:
        return ""
    result: list[str] = []
    used = 0
    spans, _ = split_graphemes(value)
    for start, end, grapheme_width in spans:
        if used + grapheme_width > width:
            break
        result.append(value[start:end])
        used += grapheme_width
    return "".join(result)


def _cell_suffix(value: str, width: int) -> str:
    if width <= 0:
        return ""
    result: list[str] = []
    used = 0
    spans, _ = split_graphemes(value)
    for start, end, grapheme_width in reversed(spans):
        if used + grapheme_width > width:
            break
        result.append(value[start:end])
        used += grapheme_width
    result.reverse()
    return "".join(result)
