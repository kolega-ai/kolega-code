"""Compact, selectable startup card with independently expandable configuration."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from rich.console import RenderableType
from textual import events
from textual.app import ComposeResult
from textual.widgets import Collapsible
from textual.widgets._collapsible import CollapsibleTitle

from .discovery import DiscoveryTip
from .state import ConversationEntry
from .widgets import ConversationEntryWidget, SelectableCollapsible, ToolEntryWidget


def display_project_path(path: Path) -> str:
    """Abbreviate home without changing the path's meaning."""
    try:
        relative = path.relative_to(Path.home())
    except ValueError:
        return str(path)
    return "~" if relative == Path(".") else f"~/{relative}"


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

    DEFAULT_CSS = """
    StartupEntryWidget {
        height: auto;
        padding-bottom: 1;
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
        margin: 1 0 0 0;
    }
    StartupEntryWidget .startup-configuration {
        border: none;
        padding: 0;
        background: transparent;
    }
    StartupEntryWidget .startup-details {
        margin-top: 1;
    }
    """

    def __init__(
        self,
        entry: ConversationEntry,
        title_factory: Callable[[ConversationEntry, int | None], str],
        summary_factory: Callable[[ConversationEntry, int], RenderableType],
        details_factory: Callable[[ConversationEntry], RenderableType],
    ) -> None:
        super().__init__(entry, lambda item: title_factory(item, None), title_for_width=title_factory)
        self._summary_factory = summary_factory
        self._details_factory = details_factory
        self._summary: StartupText | None = None
        self._details: StartupText | None = None
        self._configuration: Collapsible | None = None
        self._tip: DiscoveryTip | None = None

    def compose(self) -> ComposeResult:
        self._summary = StartupText(self.entry, lambda entry: self._summary_factory(entry, self.content_size.width))
        self._summary.add_class("startup-summary")
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
            self._configuration,
            title=self._title_factory(self.entry),
            collapsed=self.entry.startup_collapsed,
        )
        yield self._collapsible
        if self.entry.startup_tip_visible:
            self._tip = DiscoveryTip(index=self.entry.startup_tip_index)
            yield self._tip

    def refresh_content(self) -> None:
        if self._collapsible is None:
            return
        self._collapsible.collapsed = self.entry.startup_collapsed
        self._refresh_title()
        if self._summary is not None:
            self._summary.refresh_content()
        if self._details is not None:
            self._details.refresh_content()
        if self._tip is None and self.entry.startup_tip_visible:
            self._tip = DiscoveryTip(index=self.entry.startup_tip_index)
            self.mount(self._tip)
        if self._tip is not None:
            self._tip.display = self.entry.startup_tip_visible

    def on_discovery_tip_changed(self, event: DiscoveryTip.Changed) -> None:
        event.stop()
        self.entry.startup_tip_index = event.index

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
