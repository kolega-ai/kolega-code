"""Shared value types for the gigacode workflow runtime.

These are intentionally decoupled from the agent stack so the runtime can be
unit-tested with a stub ``dispatch`` callable. The production dispatch adapter
(``WorkflowTool``) translates an :class:`AgentRunSpec` into a real sub-agent run
and returns an :class:`AgentRunResult`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional


@dataclass
class AgentRunSpec:
    """One normalized ``agent()`` invocation, ready to dispatch."""

    prompt: str
    label: Optional[str] = None
    phase: Optional[str] = None
    schema: Optional[dict] = None
    model: Optional[str] = None
    effort: Optional[str] = None
    agent_type: Optional[str] = None
    # Position of this call in the run, assigned by the runtime in invocation
    # order. Combined with ``cache_key`` it drives resume.
    call_index: int = 0

    def cache_key(self) -> str:
        """Stable hash of the semantically-significant inputs of this call.

        Label and phase are cosmetic and deliberately excluded so renaming a
        step does not bust its resume cache entry.
        """
        payload = {
            "prompt": self.prompt,
            "schema": self.schema,
            "model": self.model,
            "effort": self.effort,
            "agent_type": self.agent_type,
        }
        encoded = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha1(encoded.encode("utf-8")).hexdigest()


@dataclass
class AgentRunResult:
    """Outcome of dispatching one ``agent()`` call."""

    text: Optional[str] = None
    structured: Optional[Any] = None
    tokens: int = 0
    agent_id: Optional[str] = None
    status: str = "completed"  # completed | failed | skipped
    error: Optional[str] = None

    @property
    def value(self) -> Any:
        """What ``agent()`` returns to the script.

        The structured dict when a schema was requested, otherwise the recap
        text. ``None`` for any non-completed outcome so scripts can
        ``.filter(...)`` dead agents out.
        """
        if self.status != "completed":
            return None
        return self.structured if self.structured is not None else self.text


# An async callable that actually runs a sub-agent for one spec.
DispatchFn = Callable[[AgentRunSpec], Awaitable[AgentRunResult]]

# An async callable that publishes a workflow progress event: (event_type, content).
EmitFn = Callable[[str, dict], Awaitable[None]]

# Resolves a saved/named workflow (or {"script_path": ...} ref) to its source.
WorkflowResolver = Callable[[Any], str]
