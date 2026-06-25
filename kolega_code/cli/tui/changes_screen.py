"""Session net-diff inspector screen for the CLI TUI."""

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
from .session_diff import SessionDiffFile

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
    """Full-screen view of net git changes since the TUI session began.

    Left: a roster of files whose current disk state differs from the session
    baseline. Right: the selected file's net diff, followed by any captured edit
    events for attribution/history.
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
        self._rendered_key = ""
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

    def note_change_updated(self, change: Optional[object] = None) -> None:
        """Called by the owner when net diff state or edit-event history changes."""
        if self._follow:
            self._selected_path = self._owner._default_changes_path() or self._selected_path
        elif change is not None and self._diff_for_path(self._selected_path) is None:
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

    def _ordered_diffs(self) -> list[SessionDiffFile]:
        return list(self._owner._session_diff_files)

    def _ordered_paths(self) -> list[str]:
        return [change.path for change in self._ordered_diffs()]

    def _diff_for_path(self, path: str) -> Optional[SessionDiffFile]:
        for change in self._owner._session_diff_files:
            if change.path == path:
                return change
        return None

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
        self._sync_previews(force=True)
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
        for change in self._ordered_diffs():
            row = self._rows.get(change.path)
            if row is not None:
                row.update(self._roster_row(change, selected=change.path == self._selected_path))

    def _roster_row(self, change: SessionDiffFile, *, selected: bool) -> Text:
        row_style = "bold" if selected else ""
        line = Text()
        line.append("> " if selected else "  ", style=row_style)
        line.append(f"{theme.g(Glyph.TOOL)} ", style=Color.ACCENT)
        line.append(change.path, style=row_style)
        meta = f"\n    {change.status}"
        if change.adds or change.dels:
            meta += f"  +{change.adds} -{change.dels}"
        if change.message:
            meta += "  diff unavailable"
        line.append(meta, style="dim")
        return line

    def _refresh_header(self) -> None:
        try:
            header = self.query_one("#changes_header", Static)
        except Exception:
            return
        change = self._diff_for_path(self._selected_path)
        if change is None:
            header.update(messages.CHANGES_INSPECTOR_EMPTY)
            return
        sep = theme.g(Glyph.BULLET_SEP)
        line = Text.from_markup(theme.role_header(Glyph.TOOL, escape(change.path), Color.ACCENT, state=change.status))
        line.append(f" {sep} ", style="dim")
        line.append(f"+{change.adds}", style=Color.SUCCESS)
        line.append(" ", style="dim")
        line.append(f"-{change.dels}", style=Color.ERROR)
        header.update(line)

    def _sync_previews(self, *, force: bool = False) -> None:
        try:
            view = self.query_one("#changes_previews", VerticalScroll)
        except Exception:
            return
        if not view.is_attached:
            return
        change = self._diff_for_path(self._selected_path)
        key = self._render_key(change)
        if force or key != self._rendered_key:
            view.remove_children()
            self._preview_widgets = {}
            self._empty_shown = False
            self._rendered_key = key
        if change is None:
            if not self._empty_shown:
                view.remove_children()
                self._preview_widgets = {}
                view.mount(Static(messages.CHANGES_INSPECTOR_EMPTY, classes="changes-empty"))
                self._empty_shown = True
            return

        widgets = []
        net_widget = self._preview_widgets.get("net")
        if net_widget is None:
            net_widget = Static(self._net_diff_renderable(change), markup=False, classes="change-preview")
            self._preview_widgets["net"] = net_widget
            widgets.append(net_widget)
        else:
            net_widget.update(self._net_diff_renderable(change))

        if widgets:
            view.mount(*widgets)
        if self._follow:
            self._scroll_previews_end()

    def _render_key(self, change: Optional[SessionDiffFile]) -> str:
        if change is None:
            return ""
        return f"{change.path}:{change.status}:{change.adds}:{change.dels}:{repr(change.preview)}"

    def _net_diff_renderable(self, change: SessionDiffFile) -> Group:
        if change.message:
            body = Text(change.message, style="dim")
        elif change.preview:
            try:
                body = self._preview_body_without_meta(change.preview)
            except Exception:
                body = Text("Preview unavailable", style="dim")
        else:
            body = Text("Preview unavailable", style="dim")
        return Group(Padding(body, (0, 0, 1, 0)))

    def _preview_body_without_meta(self, preview: dict):
        """Render only the diff body; the selected file and +/- counts are in the header."""
        kind = str(preview.get("kind") or "")
        if kind == "diff":
            body = self._owner._edit_preview_diff(preview.get("lines") or [])
            more = int(preview.get("more") or 0)
            if more > 0:
                footer = Text(f"{theme.g(Glyph.ELLIPSIS)} +{more} more lines", style="dim")
                return Group(body, footer)
            return body
        return self._owner._build_edit_preview(preview)

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
            self._sync_previews(force=True)

    def action_copy_changes(self) -> None:
        change = self._diff_for_path(self._selected_path)
        if change is None:
            return
        self._owner.copy_to_clipboard(self._changes_text(change))
        try:
            self._owner._notify_user(messages.CHANGES_COPIED, severity="information")
        except Exception:
            pass

    def _changes_text(self, change: SessionDiffFile) -> str:
        lines = [f"Changes for {change.path}", f"Status: {change.status}", ""]
        if change.message:
            lines.append(change.message)
            lines.append("")
        elif change.preview:
            if str(change.preview.get("kind") or "") == "diff":
                lines.append(f"+{int(change.preview.get('adds') or 0)} -{int(change.preview.get('dels') or 0)}")
            for row in change.preview.get("lines") or []:
                if isinstance(row, (list, tuple)) and len(row) >= 2:
                    lines.append(str(row[1]))
                else:
                    lines.append(str(row))
            more = int(change.preview.get("more") or 0)
            if more > 0:
                lines.append(f"… +{more} more lines")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"
