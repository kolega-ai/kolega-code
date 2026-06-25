"""Session file changes inspector screen for the CLI TUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from rich.console import Group
from rich.markup import escape
from rich.padding import Padding
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message as TextualMessage
from textual.screen import ModalScreen
from textual.widgets import Static

from .. import messages, theme
from ..theme import Color, Glyph
from .state import SessionFileChange

if TYPE_CHECKING:
    from ..app import KolegaCodeApp


class ChangeFileRosterRow(Static):
    """One selectable changed-file row in the changes inspector."""

    @dataclass
    class Selected(TextualMessage):
        path: str

    def __init__(self, path: str) -> None:
        super().__init__("", markup=False)
        self.path = path

    def on_click(self) -> None:
        self.post_message(self.Selected(self.path))


class ChangesInspectorScreen(ModalScreen):
    """Full-screen view of file edit previews captured in this TUI session.

    Left: a roster of changed files. Right: chronological edit previews for the
    selected file, with source/tool attribution. The screen updates live while
    the agent runs and new ``file_edit_preview`` events arrive.
    """

    BINDINGS = [
        Binding("escape", "close", "Close", show=True, priority=True),
        Binding("q", "close", "Close", show=False, priority=True),
        Binding("down", "next_file", "Next file", show=True, priority=True),
        Binding("up", "prev_file", "Prev file", show=False, priority=True),
        Binding("o", "toggle_follow", "Follow", show=True, priority=True),
        Binding("y", "copy_changes", "Copy", show=True, priority=True),
    ]

    def __init__(self, owner: "KolegaCodeApp", selected_path: str) -> None:
        super().__init__()
        self._owner = owner
        self._selected_path = selected_path
        self._follow = True
        self._rows: dict[str, ChangeFileRosterRow] = {}
        self._preview_widgets: dict[str, Static] = {}
        self._rendered_path: Optional[str] = None
        self._empty_shown = False
        self._flush_pending = False

    def compose(self) -> ComposeResult:
        with Horizontal(id="changes_body"):
            yield VerticalScroll(id="changes_roster")
            with Vertical(id="changes_main"):
                yield Static("", id="changes_header", markup=True)
                yield VerticalScroll(id="changes_previews")
        yield Static("", id="changes_footer", markup=False)

    def on_mount(self) -> None:
        self.border_title = f"{theme.g(Glyph.TOOL)} Changes"
        self._sync_roster()
        self._refresh_header()
        self._sync_previews()
        self._refresh_footer()

    def on_unmount(self) -> None:
        if self._owner._changes_inspector is self:
            self._owner._changes_inspector = None

    # ---- live updates ---------------------------------------------------------

    def note_change_updated(self, change: Optional[SessionFileChange] = None) -> None:
        """Called by the owner when a new file edit preview is captured."""
        if self._follow and change is not None:
            self._selected_path = change.path
        self._schedule_flush()

    def _schedule_flush(self) -> None:
        if self._flush_pending:
            return
        self._flush_pending = True
        try:
            self.set_timer(theme.RENDER_COALESCE_INTERVAL, self._flush)
        except Exception:
            self._flush()

    def _flush(self) -> None:
        self._flush_pending = False
        self._sync_roster()
        self._refresh_header()
        self._sync_previews()

    # ---- data helpers ---------------------------------------------------------

    def _ordered_paths(self) -> list[str]:
        paths: list[str] = []
        seen: set[str] = set()
        for change in self._owner._session_file_changes:
            if change.path not in seen:
                seen.add(change.path)
                paths.append(change.path)
        return paths

    def _changes_for_path(self, path: str) -> list[SessionFileChange]:
        return [change for change in self._owner._session_file_changes if change.path == path]

    def _selected_changes(self) -> list[SessionFileChange]:
        return self._changes_for_path(self._selected_path)

    # ---- selection ------------------------------------------------------------

    def action_next_file(self) -> None:
        self._move_selection(1)

    def action_prev_file(self) -> None:
        self._move_selection(-1)

    def _move_selection(self, delta: int) -> None:
        paths = self._ordered_paths()
        if not paths:
            return
        if self._selected_path in paths:
            index = (paths.index(self._selected_path) + delta) % len(paths)
        else:
            index = 0
        self._selected_path = paths[index]
        self._select_changed()

    def on_change_file_roster_row_selected(self, message: ChangeFileRosterRow.Selected) -> None:
        if message.path != self._selected_path:
            self._selected_path = message.path
            self._select_changed()

    def _select_changed(self) -> None:
        self._sync_roster()
        self._refresh_header()
        self._sync_previews()
        row = self._rows.get(self._selected_path)
        if row is not None:
            try:
                row.scroll_visible()
            except Exception:
                pass

    # ---- rendering ------------------------------------------------------------

    def _sync_roster(self) -> None:
        try:
            roster = self.query_one("#changes_roster", VerticalScroll)
        except Exception:
            return
        if not roster.is_attached:
            return
        paths = self._ordered_paths()
        if paths != list(self._rows):
            roster.remove_children()
            self._rows = {}
            rows = []
            for path in paths:
                row = ChangeFileRosterRow(path)
                self._rows[path] = row
                rows.append(row)
            if rows:
                roster.mount(*rows)
        for path in paths:
            row = self._rows.get(path)
            if row is not None:
                row.update(self._roster_row(path, selected=path == self._selected_path))

    def _roster_row(self, path: str, *, selected: bool) -> Text:
        changes = self._changes_for_path(path)
        adds = sum(int(change.preview.get("adds") or 0) for change in changes)
        dels = sum(int(change.preview.get("dels") or 0) for change in changes)
        row_style = "bold" if selected else ""
        line = Text()
        line.append("> " if selected else "  ", style=row_style)
        line.append(f"{theme.g(Glyph.TOOL)} ", style=Color.ACCENT)
        line.append(path, style=row_style)
        meta = f"\n    {len(changes)} edit{'' if len(changes) == 1 else 's'}"
        if adds or dels:
            meta += f"  +{adds} -{dels}"
        line.append(meta, style="dim")
        return line

    def _refresh_header(self) -> None:
        try:
            header = self.query_one("#changes_header", Static)
        except Exception:
            return
        changes = self._selected_changes()
        if not changes:
            header.update(messages.CHANGES_INSPECTOR_NO_SELECTION)
            return
        adds = sum(int(change.preview.get("adds") or 0) for change in changes)
        dels = sum(int(change.preview.get("dels") or 0) for change in changes)
        sep = theme.g(Glyph.BULLET_SEP)
        line = theme.role_header(Glyph.TOOL, escape(self._selected_path), Color.ACCENT, state="session changes")
        line += theme.styled(
            f" {sep} {len(changes)} edit{'' if len(changes) == 1 else 's'} {sep} +{adds} -{dels}",
            "dim",
        )
        header.update(Text.from_markup(line))

    def _sync_previews(self) -> None:
        try:
            view = self.query_one("#changes_previews", VerticalScroll)
        except Exception:
            return
        if not view.is_attached:
            return
        changes = self._selected_changes()
        if self._rendered_path != self._selected_path:
            view.remove_children()
            self._preview_widgets = {}
            self._empty_shown = False
            self._rendered_path = self._selected_path
        if not changes:
            if not self._empty_shown:
                view.remove_children()
                self._preview_widgets = {}
                view.mount(Static(messages.CHANGES_INSPECTOR_NO_SELECTION, classes="changes-empty"))
                self._empty_shown = True
            return
        if self._empty_shown:
            view.remove_children()
            self._preview_widgets = {}
            self._empty_shown = False

        rendered_ids = list(self._preview_widgets)
        current_ids = [change.change_id for change in changes]
        if current_ids[: len(rendered_ids)] != rendered_ids:
            view.remove_children()
            self._preview_widgets = {}
            rendered_ids = []

        for change_id, widget in self._preview_widgets.items():
            change = next((item for item in changes if item.change_id == change_id), None)
            if change is not None:
                widget.update(self._preview_renderable(change))

        new_changes = changes[len(rendered_ids) :]
        if new_changes:
            widgets = []
            for change in new_changes:
                widget = Static(self._preview_renderable(change), markup=False, classes="change-preview")
                self._preview_widgets[change.change_id] = widget
                widgets.append(widget)
            view.mount(*widgets)
        if self._follow:
            self._scroll_previews_end()

    def _preview_renderable(self, change: SessionFileChange) -> Group:
        sep = theme.g(Glyph.BULLET_SEP)
        title = Text()
        title.append(f"#{change.index} ", style="bold")
        title.append(change.source_label or "Agent", style=Color.ACCENT)
        if change.tool_name:
            title.append(f" {sep} {change.tool_name}", style="dim")
        if change.tool_call_id:
            title.append(f" {sep} {change.tool_call_id}", style="dim")
        try:
            preview = self._owner._build_edit_preview(change.preview)
        except Exception:
            preview = Text("Preview unavailable", style="dim")
        return Group(title, Padding(preview, (0, 0, 1, theme.INSET_WIDTH)))

    def _scroll_previews_end(self) -> None:
        def _do() -> None:
            try:
                self.query_one("#changes_previews", VerticalScroll).scroll_end(animate=False)
            except Exception:
                pass

        try:
            self.call_after_refresh(_do)
        except Exception:
            _do()

    def _refresh_footer(self) -> None:
        try:
            footer = self.query_one("#changes_footer", Static)
        except Exception:
            return
        sep = theme.g(Glyph.BULLET_SEP)
        follow = "on" if self._follow else "off"
        footer.update(f"Esc close  {sep}  Up/Down switch file  {sep}  o follow:{follow}  {sep}  y copy")

    # ---- actions --------------------------------------------------------------

    def action_close(self) -> None:
        self._owner._changes_inspector = None
        self.dismiss()
        self._owner._schedule_primary_focus_restore()

    def action_toggle_follow(self) -> None:
        self._follow = not self._follow
        self._refresh_footer()
        if self._follow:
            self._sync_previews()

    def action_copy_changes(self) -> None:
        changes = self._selected_changes()
        if not changes:
            return
        self._owner.copy_to_clipboard(self._changes_text(self._selected_path, changes))
        try:
            self._owner._notify_user(messages.CHANGES_COPIED, severity="information")
        except Exception:
            pass

    def _changes_text(self, path: str, changes: list[SessionFileChange]) -> str:
        lines = [f"Changes for {path}", ""]
        for change in changes:
            header = f"#{change.index} {change.source_label or 'Agent'}"
            if change.tool_name:
                header += f" · {change.tool_name}"
            if change.tool_call_id:
                header += f" · {change.tool_call_id}"
            lines.append(header)
            preview = change.preview
            kind = str(preview.get("kind") or "")
            if kind == "diff":
                lines.append(f"+{int(preview.get('adds') or 0)} -{int(preview.get('dels') or 0)}")
            for row in preview.get("lines") or []:
                if isinstance(row, (list, tuple)) and len(row) >= 2:
                    lines.append(str(row[1]))
                else:
                    lines.append(str(row))
            more = int(preview.get("more") or 0)
            if more > 0:
                lines.append(f"… +{more} more lines")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"
