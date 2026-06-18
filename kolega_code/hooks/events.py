"""Lifecycle events and the re-entrancy guard for the hooks system.

This module is deliberately free of any dependency on the rest of kolega_code so
the hooks package stays a self-contained, importable unit. The event model here
(``LifecycleEvent``) is distinct from ``kolega_code.events.AgentEvent`` — the
latter is the WebSocket/queue broadcast bus for UI messages, this is the
agent-control hook system.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class HookEvent(str, Enum):
    """The lifecycle events a hook can be attached to (Claude-Code wire names)."""

    SESSION_START = "SessionStart"
    SESSION_END = "SessionEnd"
    USER_PROMPT_SUBMIT = "UserPromptSubmit"
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    STOP = "Stop"
    SUBAGENT_STOP = "SubagentStop"
    PRE_COMPACT = "PreCompact"
    NOTIFICATION = "Notification"


# Tool-scoped events: a tool-spawning ``agent`` hook is forbidden here because the
# sub-agent it spawns would itself trigger these events (recursion + latency).
TOOL_EVENTS = frozenset({HookEvent.PRE_TOOL_USE, HookEvent.POST_TOOL_USE})

# Events whose hooks may alter control flow (deny/end-turn/keep-working) rather
# than being purely advisory.
BLOCKING_EVENTS = frozenset(
    {
        HookEvent.USER_PROMPT_SUBMIT,
        HookEvent.PRE_TOOL_USE,
        HookEvent.POST_TOOL_USE,
        HookEvent.STOP,
        HookEvent.SUBAGENT_STOP,
    }
)


# Re-entrancy guard. While a hook runs (an ``agent`` hook can drive the agent and
# spawn a sub-agent in the same asyncio task), nested fire points must NOT dispatch
# again, or a tool-using hook would recurse forever. ContextVars propagate to
# awaited sub-agents and to child tasks, so this suppresses the whole subtree.
_in_hook: contextvars.ContextVar[bool] = contextvars.ContextVar("kolega_in_hook", default=False)


def in_hook() -> bool:
    """True when the current execution is already inside a hook dispatch."""
    return _in_hook.get()


def enter_hook() -> contextvars.Token:
    """Mark the current context as inside a hook; returns a token for ``exit_hook``."""
    return _in_hook.set(True)


def exit_hook(token: contextvars.Token) -> None:
    """Restore the re-entrancy flag set by ``enter_hook``."""
    _in_hook.reset(token)


@dataclass(frozen=True)
class LifecycleEvent:
    """A fired lifecycle event with its payload and ambient context."""

    name: HookEvent
    payload: dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    cwd: Optional[str] = None
    permission_mode: Optional[str] = None

    def to_hook_input(self) -> dict[str, Any]:
        """The JSON document handed to a hook (stdin for command hooks, $EVENT for LLM hooks)."""
        document: dict[str, Any] = {
            "hook_event_name": self.name.value,
            "session_id": self.session_id,
            "cwd": self.cwd,
            "permission_mode": self.permission_mode,
        }
        document.update(self.payload)
        return document
