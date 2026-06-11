"""Base class for tools that support streaming responses."""

from .base_tool import BaseTool
from kolega_code.events import AgentEvent


class StreamingTool(BaseTool):
    """Base class for tools that support streaming responses."""

    async def send_streaming_update(
        self,
        content: str,
        tool_call_id: str,
        tool_name: str,
        is_complete: bool = False,
        stream_mode: str = "replace",
    ):
        """Send a streaming update for this tool's execution.

        Args:
            content: The partial or complete content to stream
            tool_call_id: The ID of the tool call this update belongs to
            tool_name: The name of the tool being executed
            is_complete: Whether this is the final update
            stream_mode: Whether incomplete updates should replace or append to the visible stream
        """
        # Attach dispatch metadata when the calling agent is a sub-agent so the
        # UI can route this stream to the right sub-agent display.
        sub_agent_info = None
        sub_agent_context = getattr(self.caller, "sub_agent_context", None)
        if getattr(self.caller, "sub_agent", False) is True and isinstance(sub_agent_context, dict):
            sub_agent_info = dict(sub_agent_context)

        event = AgentEvent(
            sender=self.caller.agent_name,
            event_type="tool_streaming_update",
            content={
                "text": content,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "is_complete": is_complete,
                "stream_mode": stream_mode,
            },
            is_streaming=not is_complete,
            sub_agent_info=sub_agent_info,
        )
        await self.connection_manager.broadcast_event(event, self.workspace_id, self.thread_id)
