"""Presentation projection: turn an event stream into renderable state.

This is the shared brain of every frontend. The TUI, the web client, and the
replay player must agree on what a session looks like, and the only way to
guarantee that is for all three to derive their view from one function rather
than each reimplementing the dispatch.

Deliberately free of UI dependencies: no Textual, no theme, no colours. The
projection decides *what* happened; a frontend decides how it looks.

**Purity and cost.** ``fold`` mutates the state it is given and returns it,
rather than returning a copy. A replay seeking to event N folds N events, so
copying would make seeking quadratic in session length and a long session
unseekable. The function is deterministic — the same events in the same order
always produce the same state — but callers must not share one state object
between two independent folds. Use :func:`replay` to build state from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from kolega_code.events import AgentEvent, ArtifactRef, KnownEventType

#: Cap on retained terminal text. Beyond this the oldest output is dropped and
#: ``terminal_truncated`` is set, so a viewer is told rather than misled.
TERMINAL_BUFFER_CHARS = 2_000_000
#: Cap on retained log lines; logs are diagnostic, not the record of the session.
LOG_BUFFER_LINES = 2_000


class ProjectionError(RuntimeError):
    """Raised when events arrive in an order the projection cannot trust."""


@dataclass
class ConversationItem:
    """One block in the transcript."""

    #: "user" | "assistant" | "thinking" | "tool" | "system" | "status"
    kind: str
    text: str = ""
    complete: bool = True
    #: Groups the deltas of one streamed segment.
    stream_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    #: "running" | "done" | "failed" for tool items.
    status: Optional[str] = None
    tone: Optional[str] = None
    #: Sub-agent this item belongs to, or None for the main agent.
    sub_agent: Optional[str] = None
    seq: Optional[int] = None
    elapsed_ms: int = 0
    artifacts: list[ArtifactRef] = field(default_factory=list)
    #: Structured edit preview attached to a tool item, when the tool edited a file.
    edit_preview: Optional[dict] = None


@dataclass
class SubAgentActivity:
    """Live state of one delegated agent."""

    key: str
    name: str = ""
    task: str = ""
    status: str = "running"
    steps: list[ConversationItem] = field(default_factory=list)
    last_text: str = ""
    context: Optional[dict] = None
    #: Workflow-supplied name for this agent. Every agent in a fan-out shares one
    #: ``name`` ("general-agent"), so without the label they cannot be told apart.
    label: str = ""
    #: Workflow phase this agent ran in, when it was dispatched by one.
    phase: str = ""
    #: Groups the agents belonging to one ``run_workflow`` call.
    workflow_run_id: str = ""


@dataclass
class TurnMarker:
    """A turn boundary, so a player can seek by turn rather than by time."""

    turn_id: str
    status: str = "open"
    user_text: str = ""
    started_seq: Optional[int] = None
    started_ms: int = 0
    ended_ms: Optional[int] = None


@dataclass
class PendingPrompt:
    """A decision point the agent is waiting on, or one already answered."""

    request_id: str
    #: "permission", "question", or another prompt kind.
    kind: str
    payload: dict
    #: None while outstanding; the answer once resolved.
    response: Optional[dict] = None
    #: How it settled: "answered", "timeout", "no_controller", "controller_left".
    reason: Optional[str] = None
    seq: Optional[int] = None
    elapsed_ms: int = 0

    @property
    def resolved(self) -> bool:
        return self.reason is not None


@dataclass
class ContextStatus:
    input_tokens: int = 0
    max_tokens: int = 0
    usage_percentage: float = 0.0
    alert_level: str = ""
    message: Optional[str] = None
    will_compress_at: int = 0


@dataclass
class PresentationState:
    """Everything a frontend needs to draw a session at one point in time."""

    conversation: list[ConversationItem] = field(default_factory=list)
    sub_agents: dict[str, SubAgentActivity] = field(default_factory=dict)
    turns: list[TurnMarker] = field(default_factory=list)
    terminal: str = ""
    terminal_truncated: bool = False
    logs: list[tuple[str, str]] = field(default_factory=list)
    edit_previews: list[dict] = field(default_factory=list)
    context: Optional[ContextStatus] = None
    compaction: Optional[dict] = None
    #: Decision points in the order they arose, resolved or not. A replay shows
    #: where the agent stopped to ask and what the answer was.
    prompts: list[PendingPrompt] = field(default_factory=list)
    #: Latest status line, e.g. a provider overload notice.
    status: str = ""
    #: "idle" | "generating" | "thinking" | "running_tool"
    activity: str = "idle"
    open_browsers: set[str] = field(default_factory=set)
    #: Highest applied seq, used to reject out-of-order folds.
    last_seq: int = 0
    elapsed_ms: int = 0
    #: True once the recording hit a retention ceiling.
    recording_truncated: bool = False
    #: Event types seen but not understood; surfaced so a stale client can say so.
    unknown_event_types: set[str] = field(default_factory=set)

    # -- Lookups kept alongside the lists so folding stays O(1) per event ----
    _streams: dict[str, int] = field(default_factory=dict, repr=False)
    _tools: dict[str, int] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict:
        """JSON-safe view for transport to a web client or player.

        Internal lookup indices are omitted: they are a folding optimization, not
        part of the state a frontend renders, and exposing them would invite a
        client to depend on list positions.
        """
        return {
            "conversation": [_item_dict(item) for item in self.conversation],
            "sub_agents": {
                key: {
                    "key": activity.key,
                    "name": activity.name,
                    "task": activity.task,
                    "status": activity.status,
                    "last_text": activity.last_text,
                    "context": activity.context,
                    "label": activity.label,
                    "phase": activity.phase,
                    "workflow_run_id": activity.workflow_run_id,
                    "steps": [_item_dict(step) for step in activity.steps],
                }
                for key, activity in self.sub_agents.items()
            },
            "turns": [
                {
                    "turn_id": marker.turn_id,
                    "status": marker.status,
                    "user_text": marker.user_text,
                    "started_seq": marker.started_seq,
                    "started_ms": marker.started_ms,
                    "ended_ms": marker.ended_ms,
                }
                for marker in self.turns
            ],
            "terminal": self.terminal,
            "terminal_truncated": self.terminal_truncated,
            "logs": [{"level": level, "text": text} for level, text in self.logs],
            "edit_previews": self.edit_previews,
            "context": None
            if self.context is None
            else {
                "input_tokens": self.context.input_tokens,
                "max_tokens": self.context.max_tokens,
                "usage_percentage": self.context.usage_percentage,
                "alert_level": self.context.alert_level,
                "message": self.context.message,
                "will_compress_at": self.context.will_compress_at,
            },
            "compaction": self.compaction,
            "prompts": [
                {
                    "request_id": prompt.request_id,
                    "kind": prompt.kind,
                    "payload": prompt.payload,
                    "response": prompt.response,
                    "reason": prompt.reason,
                    "resolved": prompt.resolved,
                    "seq": prompt.seq,
                    "elapsed_ms": prompt.elapsed_ms,
                }
                for prompt in self.prompts
            ],
            "status": self.status,
            "activity": self.activity,
            "open_browsers": sorted(self.open_browsers),
            "last_seq": self.last_seq,
            "elapsed_ms": self.elapsed_ms,
            "recording_truncated": self.recording_truncated,
            "unknown_event_types": sorted(self.unknown_event_types),
        }


def _item_dict(item: ConversationItem) -> dict:
    """JSON-safe view of one transcript item, dropping empty optional fields."""
    payload: dict = {
        "kind": item.kind,
        "text": item.text,
        "complete": item.complete,
        "seq": item.seq,
        "elapsed_ms": item.elapsed_ms,
    }
    for key, value in (
        ("stream_id", item.stream_id),
        ("tool_name", item.tool_name),
        ("tool_call_id", item.tool_call_id),
        ("status", item.status),
        ("tone", item.tone),
        ("sub_agent", item.sub_agent),
        ("edit_preview", item.edit_preview),
    ):
        if value is not None:
            payload[key] = value
    if item.artifacts:
        payload["artifacts"] = [ref.model_dump(exclude_none=True) for ref in item.artifacts]
    return payload


def replay(events: Iterable[AgentEvent]) -> PresentationState:
    """Fold ``events`` into fresh state."""
    state = PresentationState()
    for event in events:
        fold(state, event)
    return state


def fold(state: PresentationState, event: AgentEvent) -> PresentationState:
    """Apply one event to ``state`` and return it.

    Unknown event types are recorded and otherwise ignored: a frontend running
    against a newer agent must degrade, not crash.
    """
    if event.seq is not None:
        if event.seq <= state.last_seq:
            raise ProjectionError(
                f"Event seq {event.seq} is not after the last applied seq {state.last_seq}; "
                "the projection requires ascending order",
            )
        state.last_seq = event.seq
    state.elapsed_ms = max(state.elapsed_ms, event.elapsed_ms)

    handler = _HANDLERS.get(event.event_type)
    if handler is None:
        state.unknown_event_types.add(event.event_type)
        return state
    handler(state, event)
    return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sub_agent_key(event: AgentEvent) -> Optional[str]:
    """Identity of the delegate an event belongs to.

    ``agent_id`` matters as much as ``dispatch_id``: a workflow leaves
    ``dispatch_id`` unset, so keying on the name alone folded every agent of a
    fan-out into one activity — a dozen parallel agents appearing as a single
    "general-agent" whose task line was whichever one happened to arrive first.
    """
    info = event.sub_agent_info
    if not info:
        return None
    return str(info.get("dispatch_id") or info.get("agent_id") or info.get("agent_name") or "sub-agent")


def _sub_agent(state: PresentationState, key: str, event: AgentEvent) -> SubAgentActivity:
    activity = state.sub_agents.get(key)
    if activity is None:
        info = event.sub_agent_info or {}
        activity = SubAgentActivity(
            key=key,
            name=str(info.get("agent_name") or key),
            task=str(info.get("task") or ""),
            label=str(info.get("label") or ""),
            phase=str(info.get("phase") or ""),
            workflow_run_id=str(info.get("workflow_run_id") or ""),
        )
        state.sub_agents[key] = activity
    return activity


def _append(state: PresentationState, item: ConversationItem, event: AgentEvent) -> ConversationItem:
    """Add an item to the main transcript or to its sub-agent's trajectory."""
    item.seq = event.seq
    item.elapsed_ms = event.elapsed_ms
    key = _sub_agent_key(event)
    if key is not None:
        item.sub_agent = key
        activity = _sub_agent(state, key, event)
        activity.steps.append(item)
        if item.text:
            activity.last_text = item.text
        return item
    state.conversation.append(item)
    return item


def _text_of(event: AgentEvent, *keys: str) -> str:
    for key in keys:
        value = event.content.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _on_assistant_delta(state: PresentationState, event: AgentEvent, *, kind: str = "assistant") -> None:
    """Accumulate a streamed assistant or reasoning segment.

    Segments are keyed by the event uuid so interleaved reasoning and prose stay
    in separate blocks, which is how the TUI presents them.
    """
    text = _text_of(event, "text")
    complete = bool(event.content.get("complete"))
    stream_key = f"{kind}:{_sub_agent_key(event) or ''}:{event.uuid}"
    index = state._streams.get(stream_key)
    target: Optional[ConversationItem] = None
    if index is not None:
        target = _resolve(state, event, index)
    if target is None:
        # A segment that never carried text produces nothing. The agent closes
        # every iteration with a final empty prose delta even when it only called
        # a tool, and rendering that as an entry puts a bare glyph with no content
        # into the transcript.
        if not text:
            return
        _append(state, ConversationItem(kind=kind, text=text, complete=complete, stream_id=event.uuid), event)
        state._streams[stream_key] = _index_of(state, event)
        if complete:
            state._streams.pop(stream_key, None)
        return
    target.text += text
    target.complete = complete
    if complete:
        state._streams.pop(stream_key, None)


def _on_thinking_delta(state: PresentationState, event: AgentEvent) -> None:
    _on_assistant_delta(state, event, kind="thinking")


def _index_of(state: PresentationState, event: AgentEvent) -> int:
    """Index of the most recently appended item in the sequence the event targets."""
    key = _sub_agent_key(event)
    if key is not None:
        return len(state.sub_agents[key].steps) - 1
    return len(state.conversation) - 1


def _resolve(state: PresentationState, event: AgentEvent, index: int) -> Optional[ConversationItem]:
    key = _sub_agent_key(event)
    sequence = state.sub_agents[key].steps if key is not None and key in state.sub_agents else state.conversation
    if 0 <= index < len(sequence):
        return sequence[index]
    return None


#: AgentTool's dispatch lifecycle statuses, which carry no transcript text.
_SUB_AGENT_LIFECYCLE = {"STOPPED": "completed", "ERROR": "failed"}


#: Workflow lifecycle messages, and the tone each carries into the transcript.
_WORKFLOW_TONES = {"workflow_start": "start", "workflow_phase": "phase", "workflow_end": "end"}


def _on_workflow_message(state: PresentationState, event: AgentEvent, message_type: str) -> None:
    """Render a gigacode run's own lifecycle.

    ``workflow_start`` and ``workflow_end`` carry their detail in named fields
    and leave ``text`` empty, so the generic path dropped them outright: a
    workflow appeared as a few unlabelled lines with no name, no phases, and no
    outcome, while its agents' work was merged into the transcript around them.
    """
    content = event.content
    tone = _WORKFLOW_TONES[message_type]
    status: Optional[str] = None
    if message_type == "workflow_start":
        name = str(content.get("name") or "workflow")
        description = str(content.get("description") or "")
        text = f"{name} — {description}" if description else name
        status = "running"
    elif message_type == "workflow_end":
        status = str(content.get("status") or "completed")
        error = str(content.get("error") or "")
        text = f"{status}: {error}" if error else status
    else:
        text = _text_of(event, "text")
        if not text:
            return
    _append(
        state,
        ConversationItem(
            kind="workflow",
            text=text,
            tone=tone,
            status=status,
            tool_call_id=str(content.get("workflow_run_id") or "") or None,
        ),
        event,
    )


def _on_chat_message(state: PresentationState, event: AgentEvent) -> None:
    message_type = str(event.content.get("message_type") or "message")
    if message_type in ("tool_call", "tool_result", "tool_error"):
        _on_tool_message(state, event, message_type)
        return
    if message_type in _WORKFLOW_TONES:
        _on_workflow_message(state, event, message_type)
        return
    lifecycle = event.content.get("status")
    key = _sub_agent_key(event)
    if lifecycle is not None and key is not None:
        # Dispatch lifecycle, not conversation: without it a finished sub-agent
        # keeps reporting "running" for the rest of the replay.
        _sub_agent(state, key, event).status = _SUB_AGENT_LIFECYCLE.get(str(lifecycle), "running")
        return
    text = _text_of(event, "text")
    if not text:
        return
    kind = "assistant" if message_type == "message" else "system"
    _append(state, ConversationItem(kind=kind, text=text, artifacts=list(event.artifacts)), event)


_TOOL_STATUS = {"tool_call": "running", "tool_result": "done", "tool_error": "failed"}


def _on_tool_message(state: PresentationState, event: AgentEvent, message_type: str) -> None:
    tool_call_id = str(event.content.get("tool_call_id") or "")
    tool_name = str(event.content.get("tool_description") or "")
    text = _text_of(event, "text")
    status = _TOOL_STATUS[message_type]
    key = f"{_sub_agent_key(event) or ''}:{tool_call_id}"
    index = state._tools.get(key) if tool_call_id else None
    existing = _resolve(state, event, index) if index is not None else None
    if existing is not None and existing.kind == "tool":
        existing.status = status
        existing.tool_name = existing.tool_name or tool_name
        if text:
            existing.text = text
        existing.artifacts = list(event.artifacts) or existing.artifacts
        if status != "running":
            state._tools.pop(key, None)
        return
    _append(
        state,
        ConversationItem(
            kind="tool",
            text=text,
            tool_name=tool_name,
            tool_call_id=tool_call_id or None,
            status=status,
            artifacts=list(event.artifacts),
        ),
        event,
    )
    if tool_call_id and status == "running":
        state._tools[key] = _index_of(state, event)
    state.activity = "running_tool"


def _on_tool_streaming_update(state: PresentationState, event: AgentEvent) -> None:
    tool_call_id = str(event.content.get("tool_call_id") or "")
    key = f"{_sub_agent_key(event) or ''}:{tool_call_id}"
    index = state._tools.get(key)
    target = _resolve(state, event, index) if index is not None else None
    text = _text_of(event, "text")
    if target is None:
        target = _append(
            state,
            ConversationItem(
                kind="tool",
                text=text,
                tool_name=str(event.content.get("tool_name") or ""),
                tool_call_id=tool_call_id or None,
                status="running",
            ),
            event,
        )
        if tool_call_id:
            state._tools[key] = _index_of(state, event)
        return
    if str(event.content.get("stream_mode") or "") == "replace":
        target.text = text
    else:
        target.text += text
    if event.content.get("is_complete"):
        target.status = "done"


def _on_terminal_command(state: PresentationState, event: AgentEvent) -> None:
    command = _text_of(event, "command")
    if command:
        _append_terminal(state, f"$ {command}\n")
    state.activity = "running_tool"


def _on_terminal_output(state: PresentationState, event: AgentEvent) -> None:
    _append_terminal(state, _text_of(event, "display_output", "output"))


def _append_terminal(state: PresentationState, text: str) -> None:
    if not text:
        return
    combined = state.terminal + text
    if len(combined) > TERMINAL_BUFFER_CHARS:
        combined = combined[-TERMINAL_BUFFER_CHARS:]
        state.terminal_truncated = True
    state.terminal = combined


def _on_log_message(state: PresentationState, event: AgentEvent) -> None:
    text = _text_of(event, "text", "message")
    if not text:
        return
    level = str(event.content.get("level") or "info")
    state.logs.append((level, text))
    if len(state.logs) > LOG_BUFFER_LINES:
        del state.logs[: len(state.logs) - LOG_BUFFER_LINES]


def _on_file_edit_preview(state: PresentationState, event: AgentEvent) -> None:
    preview = dict(event.content)
    preview["seq"] = event.seq
    preview["elapsed_ms"] = event.elapsed_ms
    key = _sub_agent_key(event)
    if key is not None:
        preview["sub_agent"] = key
    state.edit_previews.append(preview)
    tool_call_id = str(event.content.get("tool_call_id") or "")
    if tool_call_id:
        index = state._tools.get(f"{key or ''}:{tool_call_id}")
        target = _resolve(state, event, index) if index is not None else None
        if target is not None:
            target.edit_preview = preview


def _on_context_update(state: PresentationState, event: AgentEvent) -> None:
    content = event.content
    status = ContextStatus(
        input_tokens=int(content.get("input_tokens") or 0),
        max_tokens=int(content.get("max_tokens") or 0),
        usage_percentage=float(content.get("usage_percentage") or 0.0),
        alert_level=str(content.get("alert_level") or ""),
        message=content.get("message"),
        will_compress_at=int(content.get("will_compress_at") or 0),
    )
    key = _sub_agent_key(event)
    if key is not None:
        _sub_agent(state, key, event).context = dict(content)
        return
    state.context = status


def _on_compaction_status(state: PresentationState, event: AgentEvent) -> None:
    if _sub_agent_key(event) is not None:
        # A delegate's compaction must not overwrite the main indicator.
        return
    state.compaction = dict(event.content)


def _on_status_update(state: PresentationState, event: AgentEvent) -> None:
    text = _text_of(event, "message", "status", "text")
    key = _sub_agent_key(event)
    if key is not None:
        _sub_agent(state, key, event).status = text or "running"
        return
    if text:
        state.status = text


def _on_turn_started(state: PresentationState, event: AgentEvent) -> None:
    turn_id = str(event.content.get("turn_id") or "")
    user_text = str(event.content.get("user_text") or "")
    # A dispatched agent runs its own turns. They are not session turns: indexing
    # them would put work nobody typed into the turn rail, and letting them drive
    # the activity flag would report the session idle while the main agent is
    # still going. The task text still opens the sub-agent's own trajectory.
    if _sub_agent_key(event) is None:
        state.turns.append(
            TurnMarker(turn_id=turn_id, user_text=user_text, started_seq=event.seq, started_ms=event.elapsed_ms)
        )
        state.activity = "generating"
    if user_text:
        _append(state, ConversationItem(kind="user", text=user_text), event)


def _on_turn_ended(state: PresentationState, event: AgentEvent) -> None:
    if _sub_agent_key(event) is not None:
        return
    turn_id = str(event.content.get("turn_id") or "")
    status = str(event.content.get("status") or "completed")
    for marker in reversed(state.turns):
        if marker.turn_id == turn_id:
            marker.status = status
            marker.ended_ms = event.elapsed_ms
            break
    state.activity = "idle"


def _on_browser_launched(state: PresentationState, event: AgentEvent) -> None:
    browser_id = str(event.content.get("browser_id") or "")
    if browser_id:
        state.open_browsers.add(browser_id)


def _on_browser_closed(state: PresentationState, event: AgentEvent) -> None:
    state.open_browsers.discard(str(event.content.get("browser_id") or ""))


def _on_stream_truncated(state: PresentationState, event: AgentEvent) -> None:
    state.recording_truncated = True
    _append(
        state,
        ConversationItem(
            kind="system",
            text="Recording stopped here because this session reached its retention limit.",
            tone="warning",
        ),
        event,
    )


def _on_control_requested(state: PresentationState, event: AgentEvent) -> None:
    request_id = str(event.content.get("request_id") or "")
    if not request_id or any(prompt.request_id == request_id for prompt in state.prompts):
        return
    state.prompts.append(
        PendingPrompt(
            request_id=request_id,
            kind=str(event.content.get("kind") or "prompt"),
            payload=dict(event.content.get("payload") or {}),
            seq=event.seq,
            elapsed_ms=event.elapsed_ms,
        )
    )
    state.activity = "waiting_for_user"


def _on_control_resolved(state: PresentationState, event: AgentEvent) -> None:
    request_id = str(event.content.get("request_id") or "")
    for prompt in reversed(state.prompts):
        if prompt.request_id == request_id:
            response = event.content.get("response")
            prompt.response = dict(response) if isinstance(response, dict) else None
            prompt.reason = str(event.content.get("reason") or "answered")
            break
    if not any(not prompt.resolved for prompt in state.prompts):
        state.activity = "generating"


def _on_system_message(state: PresentationState, event: AgentEvent) -> None:
    text = _text_of(event, "text", "message")
    if text:
        _append(state, ConversationItem(kind="system", text=text), event)


def _ignore(state: PresentationState, event: AgentEvent) -> None:
    """Diagnostics-only events: recorded for debugging, never rendered."""
    return None


_HANDLERS = {
    KnownEventType.ASSISTANT_DELTA: _on_assistant_delta,
    KnownEventType.THINKING_DELTA: _on_thinking_delta,
    KnownEventType.CHAT_MESSAGE: _on_chat_message,
    KnownEventType.TOOL_STREAMING_UPDATE: _on_tool_streaming_update,
    KnownEventType.TERMINAL_COMMAND: _on_terminal_command,
    KnownEventType.TERMINAL_OUTPUT: _on_terminal_output,
    KnownEventType.TERMINAL_LAUNCHED: _ignore,
    KnownEventType.TERMINAL_CLOSED: _ignore,
    KnownEventType.LOG_MESSAGE: _on_log_message,
    KnownEventType.FILE_EDIT_PREVIEW: _on_file_edit_preview,
    KnownEventType.LLM_CONTEXT_UPDATE: _on_context_update,
    KnownEventType.COMPACTION_STATUS: _on_compaction_status,
    KnownEventType.LLM_STATUS_UPDATE: _on_status_update,
    KnownEventType.STATUS_UPDATE: _on_status_update,
    KnownEventType.TURN_STARTED: _on_turn_started,
    KnownEventType.TURN_ENDED: _on_turn_ended,
    KnownEventType.BROWSER_LAUNCHED: _on_browser_launched,
    KnownEventType.BROWSER_CLOSED: _on_browser_closed,
    KnownEventType.STREAM_TRUNCATED: _on_stream_truncated,
    KnownEventType.CONTROL_REQUESTED: _on_control_requested,
    KnownEventType.CONTROL_RESOLVED: _on_control_resolved,
    KnownEventType.SYSTEM_MESSAGE: _on_system_message,
    KnownEventType.MEMORY_SUGGESTIONS: _ignore,
    KnownEventType.CREDIT_ALERT: _on_status_update,
    # Diagnostics timeline only; the TUI does not render these either.
    KnownEventType.LLM_ERROR: _ignore,
    KnownEventType.LLM_REQUEST: _ignore,
}
