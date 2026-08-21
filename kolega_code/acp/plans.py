"""Plan-mode support: markdown plans mapped to ACP plan updates.

The planning agent's response in plan mode is a markdown plan; the editor's
plan view wants ``PlanEntry`` items. Top-level markdown headings become
entries; a plan with no headings becomes a single entry.
"""

from __future__ import annotations

from acp.helpers import plan_entry
from acp.schema import PlanEntry


def plan_entries_from_markdown(text: str) -> list[PlanEntry]:
    entries: list[PlanEntry] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            content = stripped.lstrip("#").strip()
            if content:
                entries.append(plan_entry(content[:500]))
    if not entries and text.strip():
        entries.append(plan_entry(text.strip()[:500]))
    return entries
