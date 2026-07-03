"""Status dashboard and turn timer behavior for the CLI TUI."""

from __future__ import annotations

import time
from typing import Optional

from rich.markup import escape
from textual.widgets import Static

from kolega_code.agent import AgentEvent

from .. import messages, theme
from ..theme import Color, Glyph
from . import app_base as tui_app_base
from . import state as tui_state


class StatusDashboardMixin(tui_app_base.KolegaAppBase):
    @property
    def _status_dashboard(self) -> Static:
        return self.query_one("#status_dashboard", Static)

    @property
    def _turn_status(self) -> Static:
        return self.query_one("#turn_status", Static)

    def _refresh_status_dashboard(self) -> None:
        provider, model = self._startup_model()
        self._status_state.provider = provider
        self._status_state.model = model
        self._status_state.thinking_effort = self._startup_thinking_effort()
        self._status_state.mode = self.interaction_mode
        self._status_state.permission_mode = self.permission_mode.value
        self._status_state.gigacode_enabled = self._gigacode_enabled
        self._status_state.goal = self._goal_summary()
        try:
            self._status_dashboard.update(self._format_status_dashboard())
        except Exception:
            return

    def _format_status_dashboard(self) -> str:
        state = self._status_state
        disconnected = self.config is None
        provider_model = (
            messages.DISCONNECTED_MODEL
            if disconnected
            else (f"{state.provider}/{state.model}" if state.model else state.provider)
        )
        effort = state.thinking_effort or "not supported"
        mode = state.mode.title()
        permission_mode = state.permission_mode.title()
        gigacode = "On" if state.gigacode_enabled else "Off"
        turn_style = tui_state.turn_state_color(state.turn_state)
        context_style = self._context_style(state.usage_percentage, state.compression_threshold)

        def label(text: str) -> str:
            return theme.styled(text, Color.MUTED)

        goal_line = ""
        if state.goal:
            goal_line = f"{label('Goal')} [bold]{escape(state.goal)}[/bold]\n"
        if state.usage_percentage is None:
            context_lines = theme.styled("Waiting for first context count", Color.MUTED)
        else:
            percentage = f"{state.usage_percentage:.1f}%"
            token_line = self._context_token_line(state.input_tokens, state.max_tokens)
            threshold = self._compression_threshold_line(state.compression_threshold)
            context_lines = (
                f"[{context_style}]{self._context_bar(state.usage_percentage)}[/] "
                f"[bold {context_style}]{percentage}[/]\n"
                f"{token_line}\n"
                f"{theme.styled(threshold, Color.MUTED)}"
            )
            if state.context_note:
                note_style = self._context_note_style(state.alert_level)
                context_lines += f"\n[{note_style}]{escape(state.context_note)}[/{note_style}]"

        if state.is_compacting:
            indicator = escape(state.compaction_message or messages.COMPACTING)
            context_lines += f"\n[{Color.ACCENT}]{theme.g(Glyph.RUNNING)} {indicator}[/{Color.ACCENT}]"

        turn_line = (
            f"{label('Turn')} [{turn_style}]{theme.g(Glyph.STATUS)}[/{turn_style}] "
            f"[bold]{escape(state.turn_state.value)}[/bold]"
        )
        return (
            f"{label('Model')}\n[bold]{escape(provider_model)}[/bold]\n\n"
            f"{label('Thinking effort')} [bold]{escape(effort)}[/bold]\n"
            f"{label('Mode')} [bold]{mode}[/bold]\n"
            f"{label('Permissions')} [bold]{permission_mode}[/bold]\n"
            f"{label('Gigacode')} [bold]{gigacode}[/bold]\n"
            f"{goal_line}"
            f"{turn_line}\n\n"
            f"{label('Context')}\n"
            f"{context_lines}\n\n"
            f"{label('Activity')}\n"
            f"{escape(messages.DISCONNECTED_ACTIVITY if disconnected else state.activity)}"
        )

    def _context_bar(self, usage_percentage: float) -> str:
        return theme.context_bar(usage_percentage)

    def _context_token_line(self, input_tokens: Optional[int], max_tokens: Optional[int]) -> str:
        if input_tokens is None or max_tokens is None:
            return theme.styled(messages.STATUS_TOKENS_UNKNOWN, Color.MUTED)
        return f"Tokens: {input_tokens:,} / {max_tokens:,}"

    def _compression_threshold_line(self, compression_threshold: Optional[float]) -> str:
        if compression_threshold is None:
            return "Compression threshold unknown"
        return f"Compresses at {compression_threshold:.0f}%"

    def _context_style(self, usage_percentage: Optional[float], compression_threshold: Optional[float]) -> str:
        if usage_percentage is None:
            return Color.SUCCESS
        if compression_threshold is not None and usage_percentage >= compression_threshold:
            return Color.ERROR
        if usage_percentage >= 60:
            return Color.WARNING
        return Color.SUCCESS

    def _context_note_style(self, alert_level: str) -> str:
        if alert_level.lower() in {"error", "critical"}:
            return Color.ERROR
        return Color.WARNING

    def _goal_summary(self) -> Optional[str]:
        """One-line goal summary for the status dashboard, or None when inactive."""
        goal = getattr(self, "_goal", None)
        if goal is None or not goal.condition:
            return None
        from ..goal import goal_status_label

        condition = goal.condition
        if len(condition) > 48:
            condition = condition[:47].rstrip() + "…"
        return f"{condition} ({goal_status_label(goal)})"

    def _set_status_activity(self, content: str, *, turn_state: Optional[tui_state.TurnState] = None) -> None:
        changed = False
        if content and self._status_state.activity != content:
            self._status_state.activity = content
            changed = True
        if turn_state is not None and self._status_state.turn_state != turn_state:
            self._status_state.turn_state = turn_state
            changed = True
        if changed:
            self._refresh_status_dashboard()

    def _apply_compaction_status(self, content: dict) -> None:
        """Toggle the 'compaction in progress' indicator and, on finish, drop the
        summary into the transcript as a collapsible the user can expand."""
        phase = str(content.get("phase") or "")
        if phase == "started":
            self._status_state.is_compacting = True
            message = content.get("message")
            self._status_state.compaction_message = (
                message if isinstance(message, str) and message else messages.COMPACTING
            )
        else:  # "finished" | "error"
            self._status_state.is_compacting = False
            self._status_state.compaction_message = ""
            if phase == "finished":
                summary = content.get("summary")
                if isinstance(summary, str) and summary.strip():
                    self._add_conversation_entry(
                        tui_state.ConversationEntry(kind="compaction_summary", content=summary.strip())
                    )
        self._refresh_status_dashboard()

    def _apply_context_status_update(self, content: dict) -> None:
        self._status_state.input_tokens = self._as_optional_int(content.get("input_tokens"))
        self._status_state.max_tokens = self._as_optional_int(content.get("max_tokens"))
        self._status_state.usage_percentage = self._as_optional_float(content.get("usage_percentage"))
        self._status_state.compression_threshold = self._as_optional_float(content.get("compression_threshold"))
        self._status_state.alert_level = str(content.get("alert_level") or "normal")
        message = content.get("message")
        self._status_state.context_note = message if isinstance(message, str) else ""
        self._refresh_status_dashboard()

    def _display_text_from_event(self, event: AgentEvent) -> str:
        for key in ("text", "message"):
            value = event.content.get(key)
            if isinstance(value, str):
                return value
        return ""

    def _as_optional_int(self, value: object) -> Optional[int]:
        try:
            return int(value) if value is not None else None  # pyright: ignore[reportArgumentType]
        except (TypeError, ValueError):
            return None

    def _as_optional_float(self, value: object) -> Optional[float]:
        try:
            return float(value) if value is not None else None  # pyright: ignore[reportArgumentType]
        except (TypeError, ValueError):
            return None

    def _now(self) -> float:
        return time.monotonic()

    def _start_turn_timer(self, status_text: str) -> None:
        if self._turn_timer is not None:
            self._turn_timer.stop()
        self._turn_started_at = self._now()
        self._turn_finished_duration = None
        self._turn_status_text = status_text
        self._turn_final_text = ""
        self._turn_final_state = tui_state.TurnState.IDLE
        self._spinner_frame = 0
        self._turn_timer = self.set_interval(
            theme.SPINNER_INTERVAL, self._refresh_turn_status_strip, name="turn-status"
        )
        self._refresh_turn_status_strip()

    def _complete_turn_timer(self, content: str, state: tui_state.TurnState = tui_state.TurnState.IDLE) -> None:
        if self._turn_timer is not None:
            self._turn_timer.stop()
            self._turn_timer = None
        if self._turn_started_at is None:
            return

        self._turn_finished_duration = max(0.0, self._now() - self._turn_started_at)
        duration = self._format_turn_duration(self._turn_finished_duration)
        self._turn_final_state = state
        if state is tui_state.TurnState.ERROR:
            self._turn_final_text = messages.ERRORED_AFTER.format(duration=duration)
        elif state in {tui_state.TurnState.STOPPED, tui_state.TurnState.STOPPING}:
            self._turn_final_text = messages.STOPPED_AFTER.format(duration=duration)
        else:
            self._turn_final_text = messages.DONE_IN.format(duration=duration)
        self._turn_started_at = None
        self._refresh_turn_status_strip()

    def _clear_turn_status_strip(self) -> None:
        if self._turn_timer is not None:
            self._turn_timer.stop()
            self._turn_timer = None
        self._turn_started_at = None
        self._turn_finished_duration = None
        self._turn_status_text = ""
        self._turn_final_text = ""
        self._turn_final_state = tui_state.TurnState.IDLE
        self._refresh_turn_status_strip()

    def _refresh_turn_status_strip(self) -> None:
        try:
            strip = self._turn_status
        except Exception:
            return

        self._spinner_frame += 1
        content = self._turn_status_content()
        strip.display = bool(content)
        strip.update(content)
        # Tick elapsed time on running sub-agents at most once per second so the
        # faster spinner cadence only touches this cheap status strip.
        now = self._now()
        if now - self._last_sub_agent_tick >= 1.0:
            self._last_sub_agent_tick = now
            self._tick_running_sub_agents()
            self._tick_running_workflows()

    def _turn_status_content(self) -> str:
        if self._turn_started_at is not None:
            elapsed = max(0.0, self._now() - self._turn_started_at)
            status = self._turn_status_text or messages.WORKING
            frames = theme.spinner_frames()
            frame = frames[self._spinner_frame % len(frames)]
            return (
                f"[{Color.ACCENT}]{frame}[/{Color.ACCENT}] {escape(status)} "
                f"[dim]{theme.g(Glyph.BULLET_SEP)} {self._format_turn_duration(elapsed)}[/dim]"
            )
        if self._turn_final_text:
            if self._turn_final_state is tui_state.TurnState.ERROR:
                glyph, color = Glyph.CROSS, Color.ERROR
            elif self._turn_final_state in {tui_state.TurnState.STOPPED, tui_state.TurnState.STOPPING}:
                glyph, color = Glyph.CROSS, Color.WARNING
            else:
                glyph, color = Glyph.CHECK, Color.SUCCESS
            return f"[{color}]{theme.g(glyph)}[/{color}] {escape(self._turn_final_text)}"
        return ""

    def _format_turn_duration(self, seconds: float) -> str:
        total_seconds = max(0, int(seconds))
        minutes, remaining_seconds = divmod(total_seconds, 60)
        if minutes:
            return f"{minutes}m {remaining_seconds:02d}s"
        return f"{remaining_seconds}s"
