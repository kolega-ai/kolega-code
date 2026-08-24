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
    update_agent_thought_text,
    update_plan,
    update_tool_call,
    update_user_message,
)
from acp.interfaces import Client
from acp.schema import ToolCallStatus, ToolKind

from kolega_code.acp.diffs import AcpDiffProvider
from kolega_code.acp.plans import task_entries_from_markdown
from kolega_code.acp.usage import build_usage_update
from kolega_code.events import AgentEvent
from kolega_code.session.projection import ConversationItem

logger = logging.getLogger(__name__)

#: Prefix for synthetic sub-agent session ids Zed loads via session/load to
#: render a delegated turn as a nested transcript card under the dispatch.
SUB_SESSION_PREFIX = "sub-"

#: Tools whose calls dispatch delegated agents; their tool cards carry the
#: Zed subagent_session_info meta so the delegate's transcript nests under them.
SUB_AGENT_DISPATCH_TOOLS = {"dispatch_agent", "run_workflow"}


def sub_session_id(parent_session_id: str, tool_call_id: str) -> str:
    """Deterministic child session id for one dispatch (parseable on load)."""
    return f"{SUB_SESSION_PREFIX}{parent_session_id}-{tool_call_id}"


def _subagent_meta(parent_session_id: str, tool_call_id: str, tool_name: str) -> dict[str, Any]:
    """Zed's ``_meta.subagent_session_info`` convention for a dispatch card."""
    return {
        "subagent_session_info": {
            "session_id": sub_session_id(parent_session_id, tool_call_id),
            "message_start_index": 0,
        },
        "tool_name": tool_name,
    }


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
        self._turn_response_parts: list[str] = []
        #: Dispatch tool_call_id -> tool name, so delegate status lines can
        #: label the dispatching card's title.
        self._dispatch_labels: dict[str, str] = {}

    async def send(self, session_id: str, update: Any) -> None:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "acp update kind=%s tc=%s st=%s%s",
                getattr(update, "session_update", None) or type(update).__name__,
                getattr(update, "tool_call_id", ""),
                getattr(update, "status", ""),
                self._content_preview(update),
            )
        await self._conn.session_update(session_id=session_id, update=update, source="kolega_code")

    @staticmethod
    def _content_preview(update: Any) -> str:
        try:
            blocks = getattr(update, "content", None)
            if isinstance(blocks, list):
                if not blocks:
                    return ""
                inner = getattr(blocks[0], "content", blocks[0])
                text = str(getattr(inner, "text", "") or "")[:60]
                return f" ({len(blocks)} block(s), '{text}')"
            if blocks is not None:
                inner = getattr(blocks, "content", blocks)
                text = str(getattr(inner, "text", "") or "")[:60]
                return f" ('{text}')"
        except Exception:  # noqa: BLE001 — diagnostics must never break the wire
            pass
        return ""

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
            self._turn_response_parts.append(content)
            update = update_agent_message(text_block(content))
        if stream_uuid:
            update.message_id = stream_uuid
        await self.send(session_id, update)

    def response_text(self) -> str:
        return "".join(self._turn_response_parts)

    async def replay_conversation(self, session_id: str, items: list[ConversationItem]) -> None:
        """Replay the persisted transcript as session updates (session/load).

        ACP requires the agent to replay the entire conversation before
        responding to ``session/load``. The items come from the same projection
        the TUI restores, so the editor sees what the TUI would show. Live-only
        notice items (system/status) are not conversation history and are
        deliberately not replayed.
        """
        for item in items:
            if item.kind == "user":
                update = update_user_message(text_block(item.text))
                if item.seq is not None:
                    update.message_id = f"replay-{item.seq}-user"
                await self.send(session_id, update)
            elif item.kind == "assistant":
                update = update_agent_message(text_block(item.text))
                if item.seq is not None:
                    update.message_id = f"replay-{item.seq}-assistant"
                await self.send(session_id, update)
            elif item.kind == "thinking":
                await self.send(session_id, update_agent_thought(text_block(item.text)))
            elif item.kind == "tool":
                tool_call_id = item.tool_call_id or f"replay-tool-{item.seq}"
                name = item.tool_name or "tool"
                start = start_tool_call(tool_call_id, name, kind=TOOL_KINDS.get(name, "other"), status="pending")
                if name in SUB_AGENT_DISPATCH_TOOLS:
                    start.field_meta = _subagent_meta(session_id, tool_call_id, name)
                await self.send(session_id, start)
                if item.status == "failed":
                    status: ToolCallStatus = "failed"
                elif item.status == "running":
                    status = "in_progress"
                else:
                    status = "completed"
                content = [tool_content(text_block(item.text))] if item.text else None
                await self.send(session_id, update_tool_call(tool_call_id, status=status, content=content))

    async def handle_event(self, session_id: str, event: AgentEvent) -> None:
        """Map one agent event to tool-call lifecycle updates.

        ``chat_message`` events carry tool-call lifecycle. Terminal events are
        emitted by the terminal manager itself, correlated to the running tool
        call only positionally: exec tools are mutating and never run in
        parallel, so the single active execute tool is unambiguous.
        """
        if event.sub_agent_info:
            await self._handle_sub_agent_event(session_id, event)
            return
        if event.event_type in ("terminal_command", "terminal_output"):
            await self._handle_terminal_event(session_id, event)
            return
        if event.event_type == "compaction_status":
            await self._handle_compaction_event(session_id, event)
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
            if name in SUB_AGENT_DISPATCH_TOOLS:
                self._dispatch_labels[tool_call_id] = name
            start = start_tool_call(tool_call_id, title, kind=TOOL_KINDS.get(name, "other"), status="pending")
            if name in SUB_AGENT_DISPATCH_TOOLS:
                start.field_meta = _subagent_meta(session_id, tool_call_id, name)
            await self.send(session_id, start)
        elif message_type == "task_list_update":
            entries = task_entries_from_markdown(str(content.get("text") or ""))
            await self.send(session_id, update_plan(entries))
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

    async def _handle_compaction_event(self, session_id: str, event: AgentEvent) -> None:
        """Compaction progress as a collapsible thought block (the editor's
        "special view" for the conversation being compressed). A delegate's
        compaction belongs to its own transcript and is not shown here."""
        if event.sub_agent_info:
            return
        assert event.content is not None
        phase = str(event.content.get("phase") or "")
        text = str(event.content.get("summary") or event.content.get("message") or "").strip()
        if phase == "started":
            text = text or "Compacting conversation…"
        elif phase == "finished":
            text = f"Conversation compacted:\n{text}" if text else "Conversation compacted."
        elif phase == "error":
            text = f"Compaction failed: {text}" if text else "Compaction failed."
        else:
            return
        await self.send(session_id, update_agent_thought_text(text))

    async def _handle_sub_agent_event(self, session_id: str, event: AgentEvent) -> None:
        """Route a delegate's activity to the dispatching card's title.

        The nested transcript card only exists after the thread view is
        rebuilt from replayed history; live, the dispatch card's title is the
        one channel that updates as the delegate works (Zed renders tool-card
        titles live). The delegate's own tool calls, terminal commands, and
        lifecycle statuses all land there instead of polluting the parent
        trajectory. Everything else is left to the recorded nested transcript.
        """
        if event.event_type == "compaction_status" or not event.content:
            return
        parent_tool_call_id = str((event.sub_agent_info or {}).get("parent_tool_call_id") or "")
        if not parent_tool_call_id:
            return
        label = self._dispatch_labels.get(parent_tool_call_id, "sub-agent")
        content = event.content
        progress: str | None = None
        status_message: str | None = None
        if event.event_type == "terminal_command":
            progress = str(content.get("command") or "").strip() or None
        elif event.event_type == "chat_message":
            message_type = str(content.get("message_type") or "")
            if message_type == "" and content.get("status") and content.get("message"):
                status_message = str(content.get("message"))
            elif message_type == "tool_call":
                progress = str(content.get("text") or content.get("tool_description") or "").strip() or None
        if progress is None and status_message is None:
            return
        if status_message is not None:
            await self.send(
                session_id,
                update_tool_call(
                    parent_tool_call_id,
                    title=self._progress_label(label, status_message),
                    status="in_progress",
                    content=[tool_content(text_block(status_message))],
                ),
            )
            return
        await self.send(
            session_id,
            update_tool_call(
                parent_tool_call_id,
                title=self._progress_label(label, progress or ""),
                status="in_progress",
            ),
        )

    @staticmethod
    def _progress_label(label: str, text: str) -> str:
        """One-line, length-capped card title (Zed truncates multi-line titles)."""
        line = (text or "").strip().splitlines()[0].strip()
        if len(line) > 80:
            line = line[:79] + "…"
        return f"{label} · {line}" if line else label

    async def _handle_terminal_event(self, session_id: str, event: AgentEvent) -> None:
        assert event.content is not None
        tool_call_id = str(event.content.get("tool_call_id") or self._active_execute_tool or "")
        logger.debug(
            "acp terminal event kind=%s terminal_id=%s attached_tc=%s",
            event.event_type,
            event.content.get("terminal_id"),
            tool_call_id,
        )
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
