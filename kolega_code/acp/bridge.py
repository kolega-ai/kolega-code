"""Map kolega-code turn output onto ACP ``session/update`` notifications.

Two sources feed one ACP session's rendering:

- the ``process_message_stream`` generator, which yields response/thinking
  text chunks (the primary prose);
- the agent event stream (``AgentConnectionManager``), which is the only
  place tool calls, tool results, and workflow progress surface.

Consuming text from the generator and tools from the event stream avoids the
known double-render: generator chunks are mirrored as ``assistant_delta``
events for non-TUI frontends, so mapping both would duplicate every word.
"""

from __future__ import annotations

import logging
from typing import Any

from acp.helpers import (
    SessionUpdate,
    start_tool_call,
    text_block,
    tool_content,
    update_agent_message,
    update_agent_thought,
    update_tool_call,
)
from acp.interfaces import Client
from acp.schema import ToolCallStatus, ToolKind

from kolega_code.events import AgentEvent

logger = logging.getLogger(__name__)

# ACP v1 tool kinds: coarse buckets that drive client-side icons/UX.
TOOL_KINDS: dict[str, ToolKind] = {
    "read": "read",
    "read_file": "read",
    "read_image": "read",
    "edit": "edit",
    "write": "edit",
    "multi_edit": "edit",
    "apply_patch": "edit",
    "lsp_edit": "edit",
    "claude_edit": "edit",
    "claude_write": "edit",
    "hashline_edit": "edit",
    "hashline_write": "edit",
    "exec_command": "execute",
    "write_stdin": "execute",
    "kill_command": "execute",
    "list_sessions": "execute",
    "eval": "execute",
    "web_search": "search",
    "web_fetch": "fetch",
    "lsp": "search",
    "run_workflow": "other",
    "dispatch_agent": "other",
}


class AcpBridge:
    """Renders one ACP session's turn output to the client."""

    def __init__(self, conn: Client) -> None:
        self._conn = conn

    async def send(self, session_id: str, update: SessionUpdate) -> None:
        await self._conn.session_update(session_id=session_id, update=update, source="kolega_code")

    async def emit_chunk(self, session_id: str, chunk: dict[str, Any]) -> None:
        """Map one generator chunk to an agent message or thought update."""
        content = str(chunk.get("content") or "")
        if not content:
            return
        if chunk.get("type") == "thinking":
            await self.send(session_id, update_agent_thought(text_block(content)))
        else:
            await self.send(session_id, update_agent_message(text_block(content)))

    async def handle_event(self, session_id: str, event: AgentEvent) -> None:
        """Map one agent event to tool-call lifecycle updates.

        Only ``chat_message`` events carry tool activity. ``workflow_*``
        progress and terminal events have no stable ``tool_call_id`` in the
        Phase 1 event stream and surface through the tool_call/tool_result
        pair instead.
        """
        if event.event_type != "chat_message" or not event.content:
            return
        content = event.content
        message_type = str(content.get("message_type") or "")
        tool_call_id = str(content.get("tool_call_id") or "")

        if message_type == "tool_call":
            name = str(content.get("tool_description") or "tool")
            title = str(content.get("text") or f"Calling {name}")
            await self.send(
                session_id,
                start_tool_call(tool_call_id, title, kind=TOOL_KINDS.get(name, "other"), status="pending"),
            )
        elif message_type in ("tool_result", "tool_error"):
            status: ToolCallStatus = "failed" if message_type == "tool_error" else "completed"
            text = str(content.get("text") or "")
            await self.send(
                session_id,
                update_tool_call(
                    tool_call_id,
                    status=status,
                    content=[tool_content(text_block(text))] if text else None,
                ),
            )

    async def emit_stop_reason_note(self, session_id: str, reason: str) -> None:
        """Surface a non-end_turn stop reason as a short agent message."""
        if reason == "end_turn":
            return
        await self.send(session_id, update_agent_message(text_block(f"[stopped: {reason}]")))
