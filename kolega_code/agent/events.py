"""AgentEventEmitter: one place to construct and broadcast AgentEvents.

Binds the connection manager, workspace/thread routing, and sender identity
so the rest of the agent (and its tools) can emit events without rebuilding
AgentEvent payloads inline.
"""

from datetime import datetime
from typing import Any, Callable, Dict, Optional

from .connection_manager import AgentConnectionManager
from .models.public import AgentEvent


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
            )
        )

    async def llm_status(self, status: str, message: str) -> None:
        """Send an llm_status_update event (e.g. provider overload notices)."""
        await self.emit(
            AgentEvent(
                sender=self.sender,
                event_type="llm_status_update",
                content={"status": status, "message": message},
                timestamp=datetime.now().isoformat(),
                is_streaming=False,
            )
        )
