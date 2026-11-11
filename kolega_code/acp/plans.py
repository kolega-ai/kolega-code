"""Task-list support: the agent's shared task list mapped to ACP plan updates.

The build agent's shared task list (the same ``SessionRecord.task_list_markdown``
the TUI renders in its planning sidebar) is the only source of the editor's
plan view. ``update_task_list`` writes are broadcast as ``task_list_updated``
events; each list item becomes a ``PlanEntry``, with checkbox markdown carrying
the item's status. Plans never populate this view — the plan and the task list
are distinct in the TUI, and the editor's plan view mirrors the task list only.
"""

from __future__ import annotations

from acp.helpers import plan_entry
from acp.schema import PlanEntry

from kolega_code.session.task_list import task_list_entries


def task_entries_from_markdown(text: str) -> list[PlanEntry]:
    """The shared task list's markdown as plan entries.

    ``- [x]`` items map to completed entries; everything else pending, so the
    editor's plan view renders the TUI's checklist with live status.
    """
    return [
        plan_entry(entry["text"][:500], status="completed" if entry["status"] == "completed" else "pending")
        for entry in task_list_entries(text)
    ]
