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
    start_tool_call,
    text_block,
    tool_content,
    update_agent_message,
    update_agent_thought,
    update_tool_call,
)
from acp.interfaces import Client
from acp.schema import ToolCallStatus, ToolKind

from kolega_code.acp.diffs import AcpDiffProvider
from kolega_code.acp.usage import build_usage_update
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


EXECUTE_TOOLS = frozenset(name for name, kind in TOOL_KINDS.items() if kind == "execute")


class AcpBridge:
    """Renders one ACP session's turn output to the client."""

    TERMINAL_BUFFER_MAX_CHARS = 20_000

    def __init__(self, conn: Client, diff_provider: AcpDiffProvider | None = None, agent: Any = None) -> None:
        self._conn = conn
        self._diffs = diff_provider
        self._agent = agent
        self._terminal_text: dict[str, str] = {}
        self._active_execute_tool: str | None = None
        self._latest_context_tokens: int | None = None

    async def send(self, session_id: str, update: Any) -> None:
        await self._conn.session_update(session_id=session_id, update=update, source="kolega_code")

    async def emit_usage(self, session_id: str) -> None:
        update = build_usage_update(self._agent, self._latest_context_tokens)
        if update is not None:
            await self.send(session_id, update)

    async def emit_chunk(self, session_id: str, chunk: dict[str, Any]) -> None:
        """Map one generator chunk to an agent message or thought update.

        Each contiguous stream segment (one ``uuid``) becomes one ACP message:
        the messageId tells the client which chunks belong to the same block
        and when a new block starts.
        """
        content = str(chunk.get("content") or "")
        if not content:
            return
        stream_uuid = str(chunk.get("uuid") or "")
        if chunk.get("type") == "thinking":
            update = update_agent_thought(text_block(content))
        else:
            update = update_agent_message(text_block(content))
        if stream_uuid:
            update.message_id = stream_uuid
        await self.send(session_id, update)

    async def handle_event(self, session_id: str, event: AgentEvent) -> None:
        """Map one agent event to tool-call lifecycle updates.

        ``chat_message`` events carry tool-call lifecycle. Terminal events are
        emitted by the terminal manager itself, correlated to the running tool
        call only positionally: exec tools are mutating and never run in
        parallel, so the single active execute tool is unambiguous.
        """
        if event.event_type in ("terminal_command", "terminal_output"):
            await self._handle_terminal_event(session_id, event)
            return
        if event.event_type == "llm_context_update" and event.content:
            tokens = event.content.get("input_tokens")
            if isinstance(tokens, int) and tokens >= 0:
                self._latest_context_tokens = tokens
            return
        if event.event_type != "chat_message" or not event.content:
            return
        content = event.content
        message_type = str(content.get("message_type") or "")
        tool_call_id = str(content.get("tool_call_id") or "")

        if message_type == "tool_call":
            name = str(content.get("tool_description") or "tool")
            title = str(content.get("text") or f"Calling {name}")
            if name in EXECUTE_TOOLS:
                self._active_execute_tool = tool_call_id
            await self.send(
                session_id,
                start_tool_call(tool_call_id, title, kind=TOOL_KINDS.get(name, "other"), status="pending"),
            )
        elif message_type in ("tool_result", "tool_error"):
            if self._active_execute_tool == tool_call_id:
                self._active_execute_tool = None
            status: ToolCallStatus = "failed" if message_type == "tool_error" else "completed"
            terminal_text = self._terminal_text.pop(tool_call_id, None)
            text = str(content.get("text") or "")
            blocks: list[Any] = []
            if status == "completed" and self._diffs is not None:
                blocks.extend(self._diffs.build_for_tool_result(tool_call_id))
            if terminal_text:
                blocks.append(tool_content(text_block(terminal_text)))
            elif text:
                blocks.append(tool_content(text_block(text)))
            await self.send(
                session_id,
                update_tool_call(tool_call_id, status=status, content=blocks or None),
            )

    async def _handle_terminal_event(self, session_id: str, event: AgentEvent) -> None:
        assert event.content is not None
        tool_call_id = str(event.content.get("tool_call_id") or self._active_execute_tool or "")
        if not tool_call_id:
            return
        if event.event_type == "terminal_command":
            command = str(event.content.get("command") or "")
            if not command:
                return
            self._terminal_text[tool_call_id] = f"$ {command}\n"
        else:
            output = str(event.content.get("display_output") or event.content.get("output") or "")
            if not output:
                return
            current = self._terminal_text.get(tool_call_id, "")
            current = (current + output)[-self.TERMINAL_BUFFER_MAX_CHARS :]
            self._terminal_text[tool_call_id] = current
        await self.send(
            session_id,
            update_tool_call(
                tool_call_id,
                status="in_progress",
                content=[tool_content(text_block(self._terminal_text[tool_call_id]))],
            ),
        )

    async def emit_stop_reason_note(self, session_id: str, reason: str) -> None:
        """Surface a non-end_turn stop reason as a short agent message."""
        if reason == "end_turn":
            return
        await self.send(session_id, update_agent_message(text_block(f"[stopped: {reason}]")))
