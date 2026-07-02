"""State models and presentation helpers for the Textual UI."""

from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from kolega_code.permissions import PermissionDecision, PermissionMode, PermissionRequest, PermissionRuleOption

from ..provider_registry import UI_DEFAULT_MODEL, UI_DEFAULT_PROVIDER
from ..theme import Color
from .constants import BUILD_INTERACTION_MODE


class TurnState(str, Enum):
    """Explicit lifecycle state of the active turn, shown on the status dashboard."""

    IDLE = "Idle"
    GENERATING = "Generating"
    THINKING = "Thinking"
    RUNNING_TOOL = "Running tool"
    RUNNING_SUB_AGENTS = "Running sub-agents"
    WAITING_FOR_USER = "Waiting for input"
    STOPPING = "Stopping"
    STOPPED = "Stopped"
    ERROR = "Error"


# Role colors are stored as Color attribute NAMES (not values) so they resolve
# against the active theme at render time instead of snapshotting it at import;
# see theme.apply_theme(), which reassigns the Color attributes on theme switch.
TURN_STATE_STYLES = {
    TurnState.IDLE: "SUCCESS",
    TurnState.STOPPING: "WARNING",
    TurnState.STOPPED: "WARNING",
    TurnState.ERROR: "ERROR",
}

TOOL_STATE_PRESENTATION = {
    "tool_call": ("running", "ACCENT"),
    "tool_result": ("done", "SUCCESS"),
    "tool_error": ("failed", "ERROR"),
}


def turn_state_color(state: TurnState) -> str:
    """Live role color for a turn state (resolves against the active theme)."""
    return getattr(Color, TURN_STATE_STYLES.get(state, "ACCENT"))


def tool_state_presentation(kind: str) -> tuple[str, str]:
    """(state label, live role color) for a tool-entry kind."""
    label, attr = TOOL_STATE_PRESENTATION.get(kind, ("running", "ACCENT"))
    return label, getattr(Color, attr)


_ENTRY_ID_COUNTER = itertools.count(1)


def _next_entry_id() -> str:
    return f"entry-{next(_ENTRY_ID_COUNTER)}"


@dataclass
class ConversationEntry:
    kind: str
    content: str
    complete: bool = True
    uuid: Optional[str] = None
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tone: Optional[str] = None  # "warning" | "error" styling hint for progress entries
    full_content: str = ""  # untruncated tool output for expand-on-demand (capped)
    edit_preview: Optional[dict] = None  # UI-only structured diff/head preview for edit tools (not persisted)
    entry_id: str = field(default_factory=_next_entry_id)  # UI-only widget key, not persisted
    # Streaming buffers (UI-only, excluded from equality/repr). Deltas accumulate in
    # stream_parts and are joined into content once per render flush, not per chunk, to
    # avoid O(n^2) attribute concatenation. render_cache holds the incrementally-built
    # inset renderable so each flush appends only the new text rather than re-splitting
    # the whole buffer. See transcript._apply_stream_chunk / _streaming_inset_renderable.
    stream_parts: list[str] = field(default_factory=list, compare=False, repr=False)
    render_cache: object = field(default=None, compare=False, repr=False)

    def materialize(self) -> str:
        """Fold any deferred stream deltas into ``content`` and return it.

        Streaming appends land in ``stream_parts`` (O(1)); they are joined into
        ``content`` once here — on each render flush and when the segment completes —
        rather than on every chunk, so growth stays O(n) instead of O(n^2)."""
        if self.stream_parts:
            self.content += "".join(self.stream_parts)
            self.stream_parts.clear()
        return self.content


@dataclass
class QueuedMessage:
    """One UI-local follow-up queued while the current turn is running.

    ``entry`` is the future transcript entry and stays out of
    ``conversation_entries`` until the queued message begins processing.
    """

    queue_id: str
    text: str
    attachments: list[dict] | None
    entry: ConversationEntry


@dataclass
class SessionFileChange:
    """One UI-only file edit preview captured during the live TUI session."""

    change_id: str
    index: int
    path: str
    preview: dict
    tool_name: str = ""
    tool_call_id: str = ""
    agent_id: str = ""
    agent_name: str = ""
    source_label: str = "Agent"
    created_at: float = 0.0


@dataclass
class SubAgentActivity:
    """Live display state for one dispatched sub-agent."""

    agent_id: str
    agent_name: str
    task: str
    index: int  # display ordinal within the turn: #1, #2, ...
    entry: ConversationEntry  # kind="sub_agent", updated in place
    task_full: str = ""  # untruncated task, shown as the first step in the inspector
    status: str = "running"  # running | completed | failed | stopped
    tool_calls: int = 0
    last_activity: str = ""
    started_at: float = 0.0
    finished_at: Optional[float] = None
    stream_buffers: dict[str, str] = field(default_factory=dict)  # chunk uuid -> accumulated text
    active_stream_uuid: Optional[str] = None
    # Full captured trajectory for the inspector. Steps are ConversationEntry objects so
    # the existing entry widgets/renderers display them verbatim. They live on the activity
    # (not in the top-level conversation) so the main transcript stays an append-only summary.
    steps: list[ConversationEntry] = field(default_factory=list)
    tool_steps: dict[str, ConversationEntry] = field(default_factory=dict)  # tool_call_id/name -> step
    stream_steps: dict[str, ConversationEntry] = field(default_factory=dict)  # chunk uuid -> step
    pending_edit_previews: dict[str, dict] = field(default_factory=dict)  # tool_call_id -> preview payload
    tokens: Optional[int] = None  # cumulative tokens consumed, from lifecycle events
    # Context-window occupancy (how full this sub-agent's own context is right now),
    # from llm_context_update events. Distinct from cumulative `tokens` above.
    context_percentage: Optional[float] = None
    context_input_tokens: Optional[int] = None
    context_max_tokens: Optional[int] = None
    depth: int = 1
    current_action: str = ""  # live "now:" action while running
    workflow_run_id: str = ""  # set when dispatched by a workflow, for card rollup
    workflow_phase: str = ""  # the workflow phase this agent runs under


@dataclass
class PhaseState:
    """One stage of a workflow, shown as a checklist row on the workflow card."""

    title: str
    detail: str = ""
    state: str = "pending"  # pending | active | done | failed
    agents_total: int = 0
    agents_done: int = 0


@dataclass
class WorkflowActivity:
    """Live display state for one running workflow ("gigacode"), shown inline as a card."""

    run_id: str
    name: str
    description: str
    entry: ConversationEntry  # kind="workflow", updated in place
    phases: list[PhaseState] = field(default_factory=list)
    status: str = "running"  # running | completed | failed | stopped
    current_phase: str = ""
    latest_log: str = ""
    agent_count: int = 0
    tokens: int = 0
    started_at: float = 0.0
    finished_at: Optional[float] = None

    def phase_by_title(self, title: str) -> Optional[PhaseState]:
        for phase in self.phases:
            if phase.title == title:
                return phase
        return None


@dataclass
class PendingQuestion:
    question: str
    options: list[str]  # selectable option labels; the selected label is the answer
    future: asyncio.Future[str]
    descriptions: Optional[list[str]] = None  # per-option descriptions, parallel to options


@dataclass
class PendingApproval:
    request: PermissionRequest
    future: asyncio.Future[PermissionDecision]
    rule_options: list[PermissionRuleOption]


@dataclass
class PendingModelSelection:
    provider: str
    options: list[tuple[str, str]]


@dataclass
class PendingEffortSelection:
    provider: str
    model: str
    options: list[tuple[str, str]]


@dataclass
class PendingThemeSelection:
    # (display name, value); value == display name for themes.
    options: list[tuple[str, str]]


@dataclass
class StatusDashboardState:
    provider: str = UI_DEFAULT_PROVIDER
    model: str = UI_DEFAULT_MODEL
    thinking_effort: Optional[str] = None
    mode: str = BUILD_INTERACTION_MODE
    permission_mode: str = PermissionMode.ASK.value
    gigacode_enabled: bool = False
    turn_state: TurnState = TurnState.IDLE
    activity: str = "Ready"
    input_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    usage_percentage: Optional[float] = None
    compression_threshold: Optional[float] = None
    alert_level: str = "normal"
    context_note: str = ""
    is_compacting: bool = False
    compaction_message: str = ""
