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
from .widgets import ConversationEntryWidget, ScrollbackWindow, ToolEntryWidget, TrajectoryScrollView

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
        self._row_rendered: dict[str, Text] = {}
        self._header_rendered: Optional[tuple[str, str]] = None
        self._trajectory_window: Optional[ScrollbackWindow] = None
        self._rendered_key: Optional[str] = None
        self._empty_shown = False
        self._spinner_frame = 0
        self._flush_pending = False

    @property
    def _step_widgets(self) -> dict[str, ConversationEntryWidget | ToolEntryWidget]:
        """Mounted trajectory step widgets (the trailing window of the selected trajectory)."""
        window = self._trajectory_window
        return window.widgets if window is not None else {}

    def compose(self) -> ComposeResult:
        with Horizontal(id="inspector_body"):
            yield VerticalScroll(id="inspector_roster")
            with Vertical(id="inspector_main"):
                yield Static("", id="inspector_header", markup=True)
                yield TrajectoryScrollView(id="inspector_trajectory", on_near_top=self._maybe_expand_trajectory)
        yield Static("", id="inspector_footer", markup=False)

    def on_mount(self) -> None:
        self.border_title = f"{theme.g(Glyph.SUB_AGENT)} Sub-agents"
        self._sync_roster()
        self._refresh_header()
        self._refresh_footer()
        # Defer the trajectory mount until after the first paint: mounting the
        # step window costs real time on long trajectories, and the chrome
        # (roster, header, footer) is enough to show immediately.
        self.call_after_refresh(self._sync_trajectory)
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
            self._row_rendered = {}
            rows = []
            for activity in activities:
                row = SubAgentRosterRow(activity.agent_id)
                self._rows[activity.agent_id] = row
                rows.append(row)
            if rows:
                roster.mount(*rows)
        for activity in activities:
            row = self._rows.get(activity.agent_id)
            if row is None:
                continue
            rendered = self._roster_row(activity, selected=activity.agent_id == self._selected_key)
            # Skip the widget update when nothing changed: finished agents render
            # identically every tick, and each update forces a roster re-layout.
            if self._row_rendered.get(activity.agent_id) != rendered:
                row.update(rendered)
                self._row_rendered[activity.agent_id] = rendered

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
            if self._header_rendered is not None:
                self._header_rendered = None
                header.update(messages.SUB_AGENT_INSPECTOR_NO_SELECTION)
            return
        markup = self._header_markup(activity)
        rendered = (self._selected_key, markup)
        if self._header_rendered == rendered:
            return
        self._header_rendered = rendered
        header.update(Text.from_markup(markup))

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

    def _get_trajectory_window(self, view: TrajectoryScrollView) -> ScrollbackWindow:
        window = self._trajectory_window
        if window is None:
            window = ScrollbackWindow(
                view,
                self._owner._make_entry_widget,
                max_mounted=theme.INSPECTOR_WINDOW_MAX,
                trim_chunk=theme.INSPECTOR_WINDOW_EXPAND_CHUNK,
                expand_chunk=theme.INSPECTOR_WINDOW_EXPAND_CHUNK,
            )
            self._trajectory_window = window
        return window

    def _maybe_expand_trajectory(self) -> None:
        """Mount older trajectory steps when the user scrolls near the top."""
        window = self._trajectory_window
        if window is None:
            return
        activity = self._owner._sub_agent_activities.get(self._selected_key)
        if activity is None:
            return
        window.expand_up(activity.steps)

    def _sync_trajectory(self) -> None:
        try:
            view = self.query_one("#inspector_trajectory", TrajectoryScrollView)
        except Exception:
            return
        if not view.is_attached:
            return
        window = self._get_trajectory_window(view)
        activity = self._owner._sub_agent_activities.get(self._selected_key)
        if activity is None:
            window.rebuild([])
            self._rendered_key = None
            self._empty_shown = False
            return
        if self._rendered_key != self._selected_key:
            # Selection change: mount only the trailing window of the new trajectory.
            window.rebuild(activity.steps)
            self._rendered_key = self._selected_key
            self._empty_shown = False
            if not activity.steps:
                view.mount(Static(messages.SUB_AGENT_INSPECTOR_NO_STEPS, classes="inspector-empty"))
                self._empty_shown = True
            if self._follow:
                # Sticky anchor: the compositor pins the view to the newest step on
                # every arrange until the user scrolls away — no layout-settle race.
                view.anchor()
            return
        if not activity.steps:
            if not self._empty_shown:
                window.rebuild([])
                view.mount(Static(messages.SUB_AGENT_INSPECTOR_NO_STEPS, classes="inspector-empty"))
                self._empty_shown = True
            return
        if self._empty_shown:
            self._empty_shown = False
            window.rebuild(activity.steps)  # also clears the placeholder static
            if self._follow:
                view.anchor()
            return
        # Same agent, steps present: refresh changed mounted steps (bounded by the
        # window), mount newly appended steps, and trim the oldest — but only while
        # pinned to the newest step, never under a user reading older history.
        pinned_to_end = view.max_scroll_y <= 0 or view.scroll_y >= view.max_scroll_y - 1
        window.refresh_all()
        window.sync(activity.steps, set(), follow_bottom=self._follow and pinned_to_end)

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
        self._owner._schedule_primary_focus_restore()

    def action_toggle_follow(self) -> None:
        self._follow = not self._follow
        self._refresh_footer()
        try:
            view = self.query_one("#inspector_trajectory", TrajectoryScrollView)
        except Exception:
            return
        if self._follow:
            view.anchor()
            self._sync_trajectory()
        else:
            view.anchor(False)

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
        routing = activity.routing
        if routing is not None:
            sep = theme.g(Glyph.BULLET_SEP)
            origin = "parent override" if routing.overridden else "inherited default"
            effort = routing.effort or "none"
            lines.append(f"Model: {routing.provider}/{routing.model} {sep} effort: {effort} {sep} {origin}")
        lines.append("")
        for step in activity.steps:
            # The full task is already printed above as the header Task line.
            if step.kind == "sub_agent_task":
                continue
            label = step.tool_name or step.kind
            lines.append(f"[{step.kind}] {label}")
            # Fold any deferred stream deltas so a copy mid-stream isn't missing the tail.
            step.materialize()
            body = step.full_content or step.content
            if body:
                lines.append(body)
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"
