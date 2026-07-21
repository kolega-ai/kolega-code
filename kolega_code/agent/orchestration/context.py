"""Strict validation for private workflow delegation context."""

from __future__ import annotations

from typing import Optional

from .executor import MAX_AGENT_DEPTH


def validated_workflow_depth(context: object) -> Optional[tuple[int, int]]:
    """Return a strict workflow ``(depth, maximum)`` pair, or ``None``."""
    if not isinstance(context, dict) or not isinstance(context.get("workflow_run_id"), str):
        return None
    depth = context.get("depth")
    maximum = context.get("max_agent_depth")
    if type(depth) is not int or type(maximum) is not int:
        return None
    if not 1 <= depth <= maximum <= MAX_AGENT_DEPTH:
        return None
    return depth, maximum


def has_workflow_context_marker(context: object) -> bool:
    """Whether a context claims to belong to a workflow, valid or malformed."""
    return isinstance(context, dict) and "workflow_run_id" in context
