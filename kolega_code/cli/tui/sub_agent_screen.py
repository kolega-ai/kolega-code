"""Sub-agent inspector screen for the CLI TUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from rich.markup import escape
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message as TextualMessage
from textual.screen import ModalScreen
from textual.widgets import Static

from .. import messages, theme
from ..theme import Color, Glyph
from .state import ConversationEntry, SubAgentActivity
from .widgets import ConversationEntryWidget, ToolEntryWidget

if TYPE_CHECKING:
    from ..app import KolegaCodeApp

class SubAgentEntryWidget(ConversationEntryWidget):
    """A sub-agent summary card that opens the inspector when clicked."""

    @dataclass
    class Pressed(TextualMessage):
        entry: ConversationEntry

    def on_click(self) -> None:
        self.post_message(self.Pressed(self.entry))


class SubAgentRosterRow(Static):
    """One selectable row in the inspector's fleet roster."""

    @dataclass
    class Selected(TextualMessage):
        key: str

    def __init__(self, key: str) -> None:
        super().__init__("", markup=False)
        self.key = key

    def on_click(self) -> None:
        self.post_message(self.Selected(self.key))


class SubAgentInspectorScreen(ModalScreen):
    """Full-screen mission-control view of dispatched sub-agents.

    Left: a roster of every sub-agent in the turn (running + finished). Right: the
    selected agent's full trajectory, rendered with the same entry widgets as the main
    transcript (collapsible tool calls, thinking, responses). Switch agents with the
    arrow keys or by clicking a roster row; the view updates live while agents run.
    """

    BINDINGS = [
        Binding("escape", "close", "Close", show=True, priority=True),
        Binding("q", "close", "Close", show=False, priority=True),
        Binding("down", "next_agent", "Next agent", show=True, priority=True),
        Binding("up", "prev_agent", "Prev agent", show=False, priority=True),
        Binding("o", "toggle_follow", "Follow", show=True, priority=True),
        Binding("y", "copy_trajectory", "Copy", show=True, priority=True),
    ]

    def __init__(self, owner: "KolegaCodeApp", selected_key: str) -> None:
        super().__init__()
        self._owner = owner
        self._selected_key = selected_key
        self._follow = True
        self._rows: dict[str, SubAgentRosterRow] = {}
        self._step_widgets: dict[str, ConversationEntryWidget | ToolEntryWidget] = {}
        self._rendered_key: Optional[str] = None
        self._empty_shown = False
        self._spinner_frame = 0
        self._flush_pending = False

    def compose(self) -> ComposeResult:
        with Horizontal(id="inspector_body"):
            yield VerticalScroll(id="inspector_roster")
            with Vertical(id="inspector_main"):
                yield Static("", id="inspector_header", markup=True)
                yield VerticalScroll(id="inspector_trajectory")
        yield Static("", id="inspector_footer", markup=False)

    def on_mount(self) -> None:
        self.border_title = f"{theme.g(Glyph.SUB_AGENT)} Sub-agents"
        self._sync_roster()
        self._refresh_header()
        self._sync_trajectory()
        self._refresh_footer()
        self.set_interval(theme.SPINNER_INTERVAL, self._on_tick)

    def on_unmount(self) -> None:
        # Defensive: the owner clears this on close, but guarantee no dangling reference.
        if self._owner._sub_agent_inspector is self:
            self._owner._sub_agent_inspector = None

    # ---- live updates ---------------------------------------------------------

    def note_activity_updated(self, activity: SubAgentActivity) -> None:
        """Called by the owner when any sub-agent emits an event (coalesced)."""
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
        self._sync_trajectory()

    def _on_tick(self) -> None:
        self._spinner_frame += 1
        self._sync_roster()
        self._refresh_header()

    # ---- selection ------------------------------------------------------------

    def _ordered_activities(self) -> list[SubAgentActivity]:
        return sorted(self._owner._sub_agent_activities.values(), key=lambda a: a.index)

    def action_next_agent(self) -> None:
        self._move_selection(1)

    def action_prev_agent(self) -> None:
        self._move_selection(-1)

    def _move_selection(self, delta: int) -> None:
        keys = [a.agent_id for a in self._ordered_activities()]
        if not keys:
            return
        if self._selected_key in keys:
            index = (keys.index(self._selected_key) + delta) % len(keys)
        else:
            index = 0
        self._selected_key = keys[index]
        self._select_changed()

    def on_sub_agent_roster_row_selected(self, message: SubAgentRosterRow.Selected) -> None:
        if message.key != self._selected_key:
            self._selected_key = message.key
            self._select_changed()

    def _select_changed(self) -> None:
        self._sync_roster()
        self._refresh_header()
        self._sync_trajectory()
        row = self._rows.get(self._selected_key)
        if row is not None:
            try:
                row.scroll_visible()
            except Exception:
                pass

    # ---- rendering ------------------------------------------------------------

    def _sync_roster(self) -> None:
        try:
            roster = self.query_one("#inspector_roster", VerticalScroll)
        except Exception:
            return
        if not roster.is_attached:
            return
        activities = self._ordered_activities()
        keys = [a.agent_id for a in activities]
        if keys != list(self._rows):
            roster.remove_children()
            self._rows = {}
            rows = []
            for activity in activities:
                row = SubAgentRosterRow(activity.agent_id)
                self._rows[activity.agent_id] = row
                rows.append(row)
            if rows:
                roster.mount(*rows)
        for activity in activities:
            row = self._rows.get(activity.agent_id)
            if row is not None:
                row.update(self._roster_row(activity, selected=activity.agent_id == self._selected_key))

    def _status_glyph(self, activity: SubAgentActivity) -> tuple[str, str]:
        if activity.status == "running":
            frames = theme.spinner_frames()
            return frames[self._spinner_frame % len(frames)], Color.ACCENT
        if activity.status == "completed":
            return theme.g(Glyph.CHECK), Color.SUCCESS
        if activity.status == "failed":
            return theme.g(Glyph.CROSS), Color.ERROR
        return theme.g(Glyph.BULLET_SEP), Color.WARNING

    def _elapsed(self, activity: SubAgentActivity) -> str:
        if activity.finished_at is not None:
            seconds = max(0.0, activity.finished_at - activity.started_at)
        else:
            seconds = max(0.0, self._owner._now() - activity.started_at)
        return self._owner._format_turn_duration(seconds)

    def _roster_row(self, activity: SubAgentActivity, *, selected: bool) -> Text:
        sep = theme.g(Glyph.BULLET_SEP)
        glyph, color = self._status_glyph(activity)
        indent = "  " * max(0, activity.depth - 1)
        row_style = "bold" if selected else ""
        line = Text()
        line.append("> " if selected else "  ", style=row_style)
        line.append(f"{indent}{glyph} ", style=color)
        parts = [f"#{activity.index}", activity.agent_name, self._elapsed(activity), f"{activity.tool_calls}t"]
        if activity.tokens:
            parts.append(f"{self._owner._format_token_count(activity.tokens)}tok")
        if activity.context_percentage is not None:
            parts.append(f"ctx {activity.context_percentage:.0f}%")
        line.append(f"  {sep}  ".join(parts), style=row_style)
        tail = activity.current_action if activity.status == "running" else activity.last_activity
        if tail:
            line.append(f"\n    {theme.g(Glyph.INSET_ELBOW)} ", style="dim")
            line.append(tail[:48], style="dim")
        return line

    def _refresh_header(self) -> None:
        try:
            header = self.query_one("#inspector_header", Static)
        except Exception:
            return
        activity = self._owner._sub_agent_activities.get(self._selected_key)
        if activity is None:
            header.update(messages.SUB_AGENT_INSPECTOR_NO_SELECTION)
            return
        header.update(Text.from_markup(self._header_markup(activity)))

    def _header_markup(self, activity: SubAgentActivity) -> str:
        sep = theme.g(Glyph.BULLET_SEP)
        owner = self._owner
        base = owner._sub_agent_header(activity)
        extras = [f"{activity.tool_calls} tool{'' if activity.tool_calls == 1 else 's'}"]
        if activity.tokens:
            extras.append(f"{owner._format_token_count(activity.tokens)} tok")
        if activity.context_percentage is not None:
            extras.append(f"ctx {activity.context_percentage:.0f}%")
        line = base + theme.styled(f" {sep} " + f" {sep} ".join(extras), "dim")
        if activity.task:
            line += "\n" + theme.styled(escape(f"Task: {activity.task}"), "dim")
        return line

    def _sync_trajectory(self) -> None:
        try:
            view = self.query_one("#inspector_trajectory", VerticalScroll)
        except Exception:
            return
        if not view.is_attached:
            return
        activity = self._owner._sub_agent_activities.get(self._selected_key)
        if activity is None:
            view.remove_children()
            self._step_widgets = {}
            self._rendered_key = None
            self._empty_shown = False
            return
        if self._rendered_key != self._selected_key:
            view.remove_children()
            self._step_widgets = {}
            self._empty_shown = False
            self._rendered_key = self._selected_key
        if not activity.steps:
            if not self._empty_shown:
                view.remove_children()
                self._step_widgets = {}
                view.mount(Static(messages.SUB_AGENT_INSPECTOR_NO_STEPS, classes="inspector-empty"))
                self._empty_shown = True
            return
        if self._empty_shown:
            view.remove_children()
            self._step_widgets = {}
            self._empty_shown = False
        rendered_ids = list(self._step_widgets)
        current_ids = [step.entry_id for step in activity.steps]
        if current_ids[: len(rendered_ids)] != rendered_ids:
            view.remove_children()
            self._step_widgets = {}
            rendered_ids = []
        for widget in self._step_widgets.values():
            widget.refresh_content()
        new_steps = activity.steps[len(rendered_ids) :]
        if new_steps:
            widgets = []
            for step in new_steps:
                widget = self._owner._make_entry_widget(step)
                self._step_widgets[step.entry_id] = widget
                widgets.append(widget)
            view.mount(*widgets)
        if self._follow:
            self._scroll_trajectory_end()

    def _scroll_trajectory_end(self) -> None:
        """Scroll to the newest step after layout settles (heights are auto)."""

        def _do() -> None:
            try:
                self.query_one("#inspector_trajectory", VerticalScroll).scroll_end(animate=False)
            except Exception:
                pass

        try:
            self.call_after_refresh(_do)
        except Exception:
            _do()

    def _refresh_footer(self) -> None:
        try:
            footer = self.query_one("#inspector_footer", Static)
        except Exception:
            return
        sep = theme.g(Glyph.BULLET_SEP)
        follow = "on" if self._follow else "off"
        footer.update(
            f"Esc close  {sep}  Up/Down switch agent  {sep}  Tab+Enter expand tool"
            f"  {sep}  o follow:{follow}  {sep}  y copy"
        )

    # ---- actions --------------------------------------------------------------

    def action_close(self) -> None:
        self._owner._sub_agent_inspector = None
        self.dismiss()

    def action_toggle_follow(self) -> None:
        self._follow = not self._follow
        self._refresh_footer()
        if self._follow:
            self._sync_trajectory()

    def action_copy_trajectory(self) -> None:
        activity = self._owner._sub_agent_activities.get(self._selected_key)
        if activity is None:
            return
        self._owner.copy_to_clipboard(self._trajectory_text(activity))
        try:
            self._owner._notify_user(messages.SUB_AGENT_TRAJECTORY_COPIED, severity="information")
        except Exception:
            pass

    def _trajectory_text(self, activity: SubAgentActivity) -> str:
        lines = [f"{activity.agent_name} #{activity.index} ({activity.status})"]
        full_task = activity.task_full or activity.task
        if full_task:
            lines.append(f"Task: {full_task}")
        lines.append("")
        for step in activity.steps:
            # The full task is already printed above as the header Task line.
            if step.kind == "sub_agent_task":
                continue
            label = step.tool_name or step.kind
            lines.append(f"[{step.kind}] {label}")
            body = step.full_content or step.content
            if body:
                lines.append(body)
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"
