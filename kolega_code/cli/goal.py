"""Goal state, prompt builders, and status formatting for the ``/goal`` command.

This is the shared CLI layer: importable by both the Textual TUI mixins and the
non-interactive ``ask`` path in ``main.py``. It depends only on the agent-layer
:class:`~kolega_code.agent.goal.GoalVerdict` and the standard library, so it stays
cheap to import and easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from kolega_code.agent.goal import GoalVerdict

# Subcommands/aliases that clear an active goal (matches Claude Code's /goal clear).
GOAL_CLEAR_ALIASES: frozenset[str] = frozenset({"clear", "stop", "off", "reset", "none", "cancel"})

# Default safety backstop for the autonomous loop. Generous (not a tight limit);
# the primary controls are /goal clear, Esc/cancel, and the status display.
DEFAULT_GOAL_MAX_TURNS = 50


def now_iso() -> str:
    """Current UTC timestamp in the same ISO-8601 format used by the session store."""
    return datetime.now(timezone.utc).isoformat()


def _format_duration(seconds: float) -> str:
    """Format an elapsed duration like the TUI turn timer (e.g. ``3m 02s``)."""
    total = max(0, int(seconds))
    minutes, remaining = divmod(total, 60)
    if minutes:
        return f"{minutes}m {remaining:02d}s"
    return f"{remaining}s"


@dataclass
class GoalState:
    """Persistent state of one autonomous goal.

    Serialized to/from a dict for session persistence (see ``to_dict`` /
    ``from_dict``); unknown/missing keys are tolerated so older sessions load.
    """

    condition: str
    started_at: str
    turns_evaluated: int = 0
    tokens_spent: int = 0
    last_reason: str = ""
    last_evaluated_at: Optional[str] = None
    max_turns: int = DEFAULT_GOAL_MAX_TURNS
    run_to_completion: bool = False
    paused: bool = False
    met: bool = False
    # Free-form note set when the loop paused/aborted (e.g. "Stopped by user").
    status_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "started_at": self.started_at,
            "turns_evaluated": self.turns_evaluated,
            "tokens_spent": self.tokens_spent,
            "last_reason": self.last_reason,
            "last_evaluated_at": self.last_evaluated_at,
            "max_turns": self.max_turns,
            "run_to_completion": self.run_to_completion,
            "paused": self.paused,
            "met": self.met,
            "status_note": self.status_note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GoalState":
        return cls(
            condition=str(data.get("condition") or ""),
            started_at=str(data.get("started_at") or now_iso()),
            turns_evaluated=int(data.get("turns_evaluated") or 0),
            tokens_spent=int(data.get("tokens_spent") or 0),
            last_reason=str(data.get("last_reason") or ""),
            last_evaluated_at=data.get("last_evaluated_at"),
            max_turns=int(data.get("max_turns") or DEFAULT_GOAL_MAX_TURNS),
            run_to_completion=bool(data.get("run_to_completion", False)),
            paused=bool(data.get("paused", False)),
            met=bool(data.get("met", False)),
            status_note=str(data.get("status_note") or ""),
        )

    @classmethod
    def create(
        cls, condition: str, *, max_turns: int = DEFAULT_GOAL_MAX_TURNS, run_to_completion: bool = False
    ) -> "GoalState":
        return cls(
            condition=condition.strip(),
            started_at=now_iso(),
            max_turns=max_turns,
            run_to_completion=run_to_completion,
        )

    @property
    def is_active(self) -> bool:
        """A goal that should drive the loop: set, not met, not paused."""
        return bool(self.condition) and not self.met and not self.paused


def build_goal_task_prompt(condition: str) -> str:
    """The first work-turn message: kick off autonomous work toward the goal."""
    return (
        "Work autonomously toward the following goal until it is verifiably met. "
        "Use the available tools to make progress, then verify your own work.\n\n"
        f"Goal:\n{condition.strip()}"
    )


def build_goal_nudge(condition: str, verdict: GoalVerdict, turns_remaining: int) -> str:
    """The continuation message injected when the goal is not yet met.

    Includes the condition, the verifier's stated remaining gap, and the number of
    turns left before the safety cap so the agent prioritizes accordingly.
    """
    gap = verdict.reason.strip() or "the verifier did not confirm the goal is met"
    remaining_clause = (
        f" About {turns_remaining} evaluation turn(s) remain before the safety cap." if turns_remaining > 0 else ""
    )
    return (
        "The goal is not yet met. Continue working toward it.\n\n"
        f"Goal:\n{condition.strip()}\n\n"
        f"Verifier's assessment of the remaining gap: {gap}.{remaining_clause}\n\n"
        "Make concrete progress now and verify your work."
    )


def build_goal_prompt_extension_markdown(condition: str) -> str:
    """Body for the ``cli-active-goal`` system-prompt extension."""
    return (
        "## Active goal\n\n"
        "You are working autonomously toward the following goal. Keep making "
        "progress across turns until it is verifiably met, and verify your own "
        "work (run tests/commands where relevant). After each turn an independent "
        "read-only verifier checks whether the goal is met.\n\n"
        f"Goal:\n{condition.strip()}"
    )


def goal_status_label(state: GoalState) -> str:
    """Short label for the status dashboard (e.g. ``active``/``paused``/``met``)."""
    if state.met:
        return "met"
    if state.paused:
        return "paused"
    return "active"


def format_goal_status(state: GoalState, *, now: Optional[str] = None) -> str:
    """Render the ``/goal`` (no-args) status block as plain text."""
    now_dt = _parse_iso(now) if now else datetime.now(timezone.utc)
    started = _parse_iso(state.started_at)
    runtime = _format_duration((now_dt - started).total_seconds()) if started else "unknown"
    label = goal_status_label(state)
    lines = [
        f"Goal ({label}): {state.condition}",
        f"Runtime: {runtime}  |  Turns evaluated: {state.turns_evaluated}  |  Tokens spent: {state.tokens_spent:,}",
    ]
    if state.last_reason:
        lines.append(f"Verifier's latest reason: {state.last_reason}")
    if state.status_note:
        lines.append(f"Status: {state.status_note}")
    if state.met:
        lines.append("The goal has been met.")
    elif state.paused:
        lines.append("Paused. Send a message to resume, or /goal clear to remove it.")
    elif state.turns_evaluated >= state.max_turns:
        lines.append(f"Reached the turn cap ({state.max_turns}). Use /goal <condition> to retry.")
    return "\n".join(lines)


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
