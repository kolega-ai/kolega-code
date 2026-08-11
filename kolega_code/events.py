"""Agent events: the event model, connection-manager contract, and emitter.

AgentEvent is the wire format broadcast to hosts; AgentConnectionManager is
the abstract transport hosts implement; AgentEventEmitter is the agent-side
helper that constructs and broadcasts events.

Events form a durable, ordered *session event stream*: every frontend (the
Textual TUI, the web client, the replay player) renders a session by folding
this stream, so anything a UI must show has to travel as an event. Durability
and ordering live behind the protocols in :mod:`kolega_code.session.store`;
``seq`` is assigned there, never by an emitter.
"""

import abc
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

#: Version of the AgentEvent wire format. Bumped when the envelope changes
#: shape; consumers should treat unknown-but-newer envelopes as forward
#: compatible so long as the fields they read are present.
AGENT_EVENT_SCHEMA_VERSION = 2


def utc_now_iso() -> str:
    """Timezone-aware UTC timestamp. Naive local time cannot be ordered or replayed."""
    return datetime.now(timezone.utc).isoformat()


class KnownEventType:
    """Event types emitted by this package.

    ``AgentEvent.event_type`` is an open string rather than a closed enum: hosts
    and future versions must be able to introduce types without a release here,
    and every consumer (fold, replay, UI) must treat unrecognized types as inert
    rather than an error.
    """

    SYSTEM_MESSAGE = "system_message"
    CHAT_MESSAGE = "chat_message"
    LOG_MESSAGE = "log_message"
    TERMINAL_COMMAND = "terminal_command"
    TERMINAL_OUTPUT = "terminal_output"
    TERMINAL_LAUNCHED = "terminal_launched"
    TERMINAL_CLOSED = "terminal_closed"
    BROWSER_LAUNCHED = "browser_launched"
    BROWSER_CLOSED = "browser_closed"
    STATUS_UPDATE = "status_update"
    LLM_STATUS_UPDATE = "llm_status_update"
    LLM_ERROR = "llm_error"
    LLM_REQUEST = "llm_request"
    CREDIT_ALERT = "credit_alert"
    LLM_CONTEXT_UPDATE = "llm_context_update"
    COMPACTION_STATUS = "compaction_status"
    TOOL_STREAMING_UPDATE = "tool_streaming_update"
    FILE_EDIT_PREVIEW = "file_edit_preview"
    MEMORY_SUGGESTIONS = "memory_suggestions"
    # Assistant response and reasoning text. Historically these reached the TUI
    # only through process_message_stream's generator, which left them absent
    # from the event stream and therefore unrenderable by any other frontend.
    ASSISTANT_DELTA = "assistant_delta"
    THINKING_DELTA = "thinking_delta"
    # Turn boundaries, so a frontend can seek by turn without reading the
    # provider-facing journal.
    TURN_STARTED = "turn_started"
    TURN_ENDED = "turn_ended"
    # Emitted in place of payload that exceeded a configured retention bound.
    STREAM_TRUNCATED = "stream_truncated"
    # Interactions that need an answer from whichever client holds control:
    # permission prompts and planning questions. Previously an in-process
    # callback, which meant only a co-located UI could ever answer, and which
    # left decision points absent from the recording entirely.
    CONTROL_REQUESTED = "control_requested"
    CONTROL_RESOLVED = "control_resolved"


class ArtifactPurpose:
    """Why a payload was externalized, and therefore how it may be used.

    Purpose is the export gate. ``TOOL_RESULT`` and ``IMAGE`` are content a
    viewer needs in order to render a session. The remaining four are opaque
    provider state — reasoning signatures and encrypted reasoning — which exist
    only so a conversation can be replayed back to the model that produced it.
    They carry no display value and must never be shared, so export is
    allowlist-based rather than denylist-based.
    """

    TOOL_RESULT = "tool_result"
    IMAGE = "image"
    PROVIDER_SIGNATURE = "provider_signature"
    REDACTED_REASONING = "redacted_reasoning"
    ENCRYPTED_REASONING = "encrypted_reasoning"
    THOUGHT_SIGNATURE = "thought_signature"

    #: The only purposes permitted to leave this machine.
    SHAREABLE: frozenset[str] = frozenset({TOOL_RESULT, IMAGE})


class ArtifactRef(BaseModel):
    """Pointer to content held outside the event, addressed by content hash.

    ``purpose`` drives export policy: only an explicit allowlist is ever shared,
    because provider-opaque payloads (reasoning signatures, encrypted reasoning)
    are secrets that must never leave the machine that produced them.
    """

    sha256: str
    bytes: int
    media_type: str
    purpose: str
    encoding: str
    #: Filesystem hint for local diagnostics. Absent for non-file stores, and
    #: stripped on export so bundles never leak local paths.
    path: Optional[str] = None
    #: Character count for text payloads, used to describe what was elided.
    chars: Optional[int] = None


class AgentEvent(BaseModel):
    schema_version: int = AGENT_EVENT_SCHEMA_VERSION
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=utc_now_iso)
    event_type: str
    sender: str
    recipient: Optional[str] = None
    content: dict = Field(default_factory=dict)
    is_streaming: bool = False
    sub_agent_info: Optional[dict] = None

    # --- Addressing -------------------------------------------------------
    # Carried on the event, not only as broadcast arguments, so a persisted or
    # forwarded event stays self-describing once separated from its call site.
    session_id: str = ""
    workspace_id: str = ""
    thread_id: str = ""

    # --- Ordering and replay ---------------------------------------------
    #: Assigned by SessionEventStore.append(). ``None`` means "not durable":
    #: live-only events (for example coalesced streaming deltas) are broadcast
    #: but not replayable.
    seq: Optional[int] = None
    #: Monotonic milliseconds since the session's first recorded event. This,
    #: not ``timestamp``, is the replay timing key: wall-clock time can jump
    #: backwards and is not comparable across machines.
    elapsed_ms: int = 0
    artifacts: list[ArtifactRef] = Field(default_factory=list)


class AgentStatus(Enum):
    """
    Enum representing the current status of an agent.

    Values:
        STOPPED: The agent is not currently generating content.
        GENERATING: The agent is actively generating content.
        INTERRUPT_REQUESTED: The agent has received a request to stop generation but hasn't fully stopped yet.
    """

    STOPPED = "stopped"
    GENERATING = "generating"
    INTERRUPT_REQUESTED = "interrupt_requested"


class AgentConnectionManager(abc.ABC):
    """Abstract base class for agent connection managers."""

    @abc.abstractmethod
    async def connect(
        self, websocket: Any, workspace_id: str, thread_id: str, connection_type: str, user_info=None
    ) -> None:
        """
        Connect a client to a specific workspace, thread and connection type.

        Args:
            websocket: The WebSocket connection
            workspace_id: ID of the workspace to connect to
            thread_id: ID of the thread to connect to
            connection_type: Type of connection ('chat', 'terminal', or 'logs')
            user_info: Optional user information dictionary
        """

    @abc.abstractmethod
    def disconnect(self, websocket: Any, workspace_id: str, thread_id: str, connection_type: str) -> None:
        """
        Disconnect a client from a specific workspace, thread and connection type.

        Args:
            websocket: The WebSocket connection
            workspace_id: ID of the workspace to disconnect from
            thread_id: ID of the thread to disconnect from
            connection_type: Type of connection ('chat', 'terminal', or 'logs')
        """

    @abc.abstractmethod
    async def broadcast_event(self, event: AgentEvent, workspace_id: str, thread_id: str) -> None:
        """
        Broadcast a chat message to all connected clients for a thread.

        Args:
            event: The event to broadcast
            workspace_id: ID of the workspace
            thread_id: ID of the thread
        """

    @abc.abstractmethod
    def get_connection_count(self, workspace_id: str, thread_id: str) -> dict:
        """
        Get the number of connections for each type for a thread.

        Args:
            workspace_id: ID of the workspace
            thread_id: ID of the thread

        Returns:
            Dictionary with connection counts
        """


class AgentEventEmitter:
    """Constructs and broadcasts AgentEvents for one agent instance."""

    def __init__(
        self,
        connection_manager: AgentConnectionManager,
        workspace_id: str,
        thread_id: str,
        sender: str,
        sub_agent_info_provider: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
    ) -> None:
        self.connection_manager = connection_manager
        self.workspace_id = workspace_id
        self.thread_id = thread_id
        self.sender = sender
        # Callable rather than a value: sub-agent dispatch metadata is set on the
        # agent after construction and changes per tool call.
        self._sub_agent_info_provider = sub_agent_info_provider

    async def emit(self, event: AgentEvent) -> None:
        await self.connection_manager.broadcast_event(event, self.workspace_id, self.thread_id)

    async def chat(
        self,
        message_type: str,
        content: str,
        *,
        is_streaming: bool = False,
        tool_description: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        images: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> None:
        """Send a chat_message event (responses, tool calls/results/errors).

        ``images`` carries ``(media_type, base64_data)`` pairs for a tool that
        produced pictures. They travel as content here and are turned into image
        artifacts by the recording layer, which then drops the payload — without
        this the bytes only ever reach the model, on a history record no
        presentation client reads, and a replay could never show a screenshot.
        """
        sub_agent_info = self._sub_agent_info_provider() if self._sub_agent_info_provider else None
        payload: Dict[str, Any] = {
            "message_type": message_type,
            "text": content,
            "tool_description": tool_description,
            "tool_call_id": tool_call_id,
        }
        if images:
            payload["images"] = [{"media_type": media_type, "data": data} for media_type, data in images]

        await self.emit(
            AgentEvent(
                sender=self.sender,
                event_type="chat_message",
                content=payload,
                is_streaming=is_streaming,
                sub_agent_info=sub_agent_info,
            )
        )

    async def assistant_delta(
        self,
        text: str,
        *,
        complete: bool,
        stream_uuid: str,
        thinking: bool = False,
    ) -> None:
        """Send assistant response or reasoning text as a stream event.

        ``process_message_stream`` also yields this text to its caller, which is
        how the TUI has always rendered it. Emitting it here as well is what lets
        any other frontend render a conversation at all: without these events the
        stream contains tool activity and status but no assistant prose.

        ``stream_uuid`` groups the deltas of one contiguous segment so consumers
        can append to the segment they already started; ``complete`` marks the
        final delta of that segment.
        """
        sub_agent_info = self._sub_agent_info_provider() if self._sub_agent_info_provider else None

        await self.emit(
            AgentEvent(
                sender=self.sender,
                event_type=KnownEventType.THINKING_DELTA if thinking else KnownEventType.ASSISTANT_DELTA,
                uuid=stream_uuid,
                content={"text": text, "complete": complete},
                is_streaming=not complete,
                sub_agent_info=sub_agent_info,
            )
        )

    async def turn_boundary(
        self,
        phase: str,
        *,
        turn_id: str,
        status: Optional[str] = None,
        user_text: Optional[str] = None,
    ) -> None:
        """Mark a turn boundary so frontends can seek by turn.

        ``phase`` is "started" or "ended"; ``status`` carries the terminal
        outcome on "ended".

        A dispatched agent runs turns of its own, so these carry ``sub_agent_info``
        like every other event. Without it a consumer cannot tell a delegated turn
        from a session turn, and folds the sub-agent's task into the main
        transcript as a message the user never sent.
        """
        sub_agent_info = self._sub_agent_info_provider() if self._sub_agent_info_provider else None

        await self.emit(
            AgentEvent(
                sender=self.sender,
                event_type=(KnownEventType.TURN_STARTED if phase == "started" else KnownEventType.TURN_ENDED),
                content={
                    "turn_id": turn_id,
                    "status": status,
                    "user_text": user_text,
                },
                sub_agent_info=sub_agent_info,
            )
        )

    async def context_update(
        self,
        *,
        input_tokens: int,
        model_context_length: int,
        compression_threshold: float,
        alert_level: str,
        message: Optional[str],
    ) -> None:
        """Send an llm_context_update event describing context usage.

        ``model_context_length`` is the usable input budget (window minus the
        output allowance), so percentages and ``will_compress_at`` line up
        with the compaction trigger. The payload key stays ``max_tokens``.
        """
        usage_percentage = (input_tokens / model_context_length) * 100
        sub_agent_info = self._sub_agent_info_provider() if self._sub_agent_info_provider else None

        await self.emit(
            AgentEvent(
                event_type="llm_context_update",
                sender=self.sender,
                content={
                    "input_tokens": input_tokens,
                    "max_tokens": model_context_length,
                    "usage_percentage": round(usage_percentage, 1),
                    "alert_level": alert_level,
                    "message": message,
                    "compression_threshold": compression_threshold * 100,  # Convert to percentage
                    "will_compress_at": int(model_context_length * compression_threshold),
                },
                sub_agent_info=sub_agent_info,
            )
        )

    async def compaction_status(self, phase: str, message: str = "", summary: str = "") -> None:
        """Send a compaction_status event so the UI can show compaction progress.

        ``phase`` is one of "started" | "finished" | "error". On "finished",
        ``summary`` carries the summary text so the UI can show it in the transcript.
        """
        sub_agent_info = self._sub_agent_info_provider() if self._sub_agent_info_provider else None

        await self.emit(
            AgentEvent(
                sender=self.sender,
                event_type=KnownEventType.COMPACTION_STATUS,
                content={"phase": phase, "message": message, "summary": summary},
                is_streaming=False,
                sub_agent_info=sub_agent_info,
            )
        )

    async def llm_status(self, status: str, message: str) -> None:
        """Send an llm_status_update event (e.g. provider overload notices)."""
        sub_agent_info = self._sub_agent_info_provider() if self._sub_agent_info_provider else None

        await self.emit(
            AgentEvent(
                sender=self.sender,
                event_type=KnownEventType.LLM_STATUS_UPDATE,
                content={"status": status, "message": message},
                is_streaming=False,
                sub_agent_info=sub_agent_info,
            )
        )

    async def llm_error(self, **fields: Any) -> None:
        """Structured LLM error for the diagnostics timeline (provider/model/endpoint/status/…).

        Diagnostic-only: the UI surfaces user-facing errors via llm_status; this carries the
        machine-readable detail the CLI persists so a failed turn is debuggable."""
        sub_agent_info = self._sub_agent_info_provider() if self._sub_agent_info_provider else None
        await self.emit(
            AgentEvent(
                sender=self.sender,
                event_type=KnownEventType.LLM_ERROR,
                content=dict(fields),
                is_streaming=False,
                sub_agent_info=sub_agent_info,
            )
        )

    async def llm_request(self, phase: str, **fields: Any) -> None:
        """Mark an LLM request boundary (phase="start"|"end") for the diagnostics timeline,
        so a long-open request (network stall) is visible while the loop watchdog stays quiet."""
        sub_agent_info = self._sub_agent_info_provider() if self._sub_agent_info_provider else None
        await self.emit(
            AgentEvent(
                sender=self.sender,
                event_type=KnownEventType.LLM_REQUEST,
                content={"phase": phase, **fields},
                is_streaming=False,
                sub_agent_info=sub_agent_info,
            )
        )
