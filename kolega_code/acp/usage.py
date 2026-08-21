"""Context usage reporting via ACP ``usage_update`` notifications.

``used``/``size`` mirror the TUI's context gauge: input tokens from the
agent's per-request ``llm_context_update`` events, window from the agent's
effective model limits.
"""

from __future__ import annotations

from typing import Any

from acp.schema import UsageUpdate


def context_window_for(agent: Any) -> int | None:
    for attribute in ("model_max_input_tokens", "model_context_length"):
        value = getattr(agent, attribute, None)
        if value:
            return int(value)
    primary = getattr(agent, "primary_model_config", None)
    value = getattr(primary, "context_window_tokens", None) if primary is not None else None
    return int(value) if value else None


def build_usage_update(agent: Any, used: int | None = None) -> UsageUpdate | None:
    size = context_window_for(agent)
    if not size:
        return None
    resolved_used = min(int(used or 0), size)
    return UsageUpdate(session_update="usage_update", used=resolved_used, size=size)
