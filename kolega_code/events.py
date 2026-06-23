"""Agent events: the event model, connection-manager contract, and emitter.

AgentEvent is the wire format broadcast to hosts; AgentConnectionManager is
the abstract transport hosts implement; AgentEventEmitter is the agent-side
helper that constructs and broadcasts events.
"""

import abc
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Literal, Optional

from pydantic import BaseModel, Field


class AgentEvent(BaseModel):
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    event_type: Literal[
        "system_message",
        "chat_message",
        "log_message",
        "terminal_command",
        "terminal_output",
        "terminal_launched",
        "terminal_closed",
        "browser_launched",
        "browser_closed",
        "status_update",
        "llm_status_update",
        "credit_alert",
        "llm_context_update",
        "compaction_status",
        "tool_streaming_update",
        "file_edit_preview",
        "memory_suggestions",
    ]
    sender: str
    recipient: Optional[str] = None
    content: dict = Field(default_factory=dict)
    is_streaming: bool = False
    sub_agent_info: Optional[dict] = None


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
    ) -> None:
        """Send a chat_message event (responses, tool calls/results/errors)."""
        sub_agent_info = self._sub_agent_info_provider() if self._sub_agent_info_provider else None

        await self.emit(
            AgentEvent(
                sender=self.sender,
                event_type="chat_message",
                content={
                    "message_type": message_type,
                    "text": content,
                    "tool_description": tool_description,
                    "tool_call_id": tool_call_id,
                },
                timestamp=datetime.now().isoformat(),
                is_streaming=is_streaming,
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
        """Send an llm_context_update event describing context-window usage."""
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
                event_type="compaction_status",
                content={"phase": phase, "message": message, "summary": summary},
                timestamp=datetime.now().isoformat(),
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
                event_type="llm_status_update",
                content={"status": status, "message": message},
                timestamp=datetime.now().isoformat(),
                is_streaming=False,
                sub_agent_info=sub_agent_info,
            )
        )
