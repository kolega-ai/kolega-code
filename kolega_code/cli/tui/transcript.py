"""Transcript, tool, sub-agent, and workflow rendering for the CLI TUI."""

from __future__ import annotations

from typing import Callable, Optional

from rich.console import Group
from rich.markdown import Markdown as RichMarkdown
from rich.markup import escape
from rich.padding import Padding
from rich.text import Text

from kolega_code.agent import AgentEvent
from kolega_code.llm.models import Message, TextBlock, ToolCall, ToolResult
from kolega_code.services.lsp import extract_lsp_label

from .. import messages, theme
from ..skills import skill_names_in_text
from ..theme import Color, Glyph
from .constants import QUESTION_TOOL_NAME, STARTUP_WORDMARK
from .state import (
    ConversationEntry,
    PhaseState,
    SessionFileChange,
    SubAgentActivity,
    TurnState,
    WorkflowActivity,
    TOOL_STATE_PRESENTATION,
    tool_state_presentation,
)
from . import app_base as tui_app_base
from .sub_agent_screen import SubAgentEntryWidget
from .widgets import ConversationEntryWidget, JumpToBottomBar, ToolEntryWidget


class _InsetRenderState:
    """Incremental render cache for a streaming inset body (assistant/thinking).

    Holds the Rich ``Text`` built so far plus enough state to append only newly
    arrived characters on each flush, instead of re-splitting the whole growing
    buffer. The output matches ``_format_inset_text`` for ``\\n``-delimited content
    (the common case); completion does a canonical full reformat as a safety net.
    """

    __slots__ = ("text", "length", "started", "open_has_content", "deferred_newline", "style")

    def __init__(self, style):
        self.text = Text()
        self.length = 0  # chars of entry.content already folded into `text`
        self.started = False  # has the first line's inset bar been emitted?
        self.open_has_content = False  # has the current (last) line emitted its " " + text?
        # One pending line break: a "\n" defers its new line until either more content or
        # another "\n" arrives, so a *trailing* terminator leaves no empty inset line
        # (matching str.splitlines, which drops only the final terminator).
        self.deferred_newline = False
        self.style = style


class TranscriptRenderingMixin(tui_app_base.KolegaAppBase):
    def _restore_conversation_history(self, history: list[dict]) -> None:
        self.conversation_entries = []
        self._stream_entries = {}
        self._tool_entries = {}
        self._tool_stream_buffers = {}
        self._sub_agent_activities = {}
        self._sub_agent_by_tool_call = {}
        self._sub_agent_seq = 0
        self._workflow_activities = {}
        self._active_progress_entry = None
        self._plan_decision_active = False
        self._restore_plan_action_visibility()
        self._cancel_pending_question()
        self._cancel_pending_model_selection()
        self._cancel_pending_effort_selection()
        self._cancel_pending_theme_selection()
        self._refresh_planning_sidebar()
        self._ensure_startup_entry(render=False)
        # If the restored agent is in a compacted state, place the collapsible
        # summary where it sat in the live transcript: after the retained tail
        # messages, before any newer turns. The history length captured when
        # compaction ran tells us that position; old sessions without it fall
        # back to the compaction boundary so the summary still renders.
        summary_entry = self._resume_compaction_entry()
        if summary_entry is not None:
            compaction = self.session.compaction or {}
            through = int(compaction.get("compacted_through") or 0)
            boundary = min(through, len(history))
            summary_index = int(compaction.get("compacted_history_length") or boundary)
            summary_index = min(summary_index, len(history))
            self.conversation_entries.extend(self._conversation_entries_from_history_items(history[:summary_index]))
            self.conversation_entries.append(summary_entry)
            self.conversation_entries.extend(self._conversation_entries_from_history_items(history[summary_index:]))
        else:
            self.conversation_entries.extend(self._conversation_entries_from_history_items(history))
        self._render_conversation()

    def _resume_compaction_entry(self) -> Optional[ConversationEntry]:
        """A collapsible summary entry for the restored compaction boundary, or None.

        Built from the session's persisted compaction metadata (the same data the
        agent was restored from), so it does not depend on agent internals.
        """
        data = self.session.compaction or {}
        summary_text = (data.get("summary") or "").strip()
        if not summary_text or int(data.get("compacted_through") or 0) <= 0:
            return None
        return ConversationEntry(kind="compaction_summary", content=summary_text)

    def _conversation_entries_from_history_items(self, history: list[dict]) -> list[ConversationEntry]:
        """Build restored transcript entries, coalescing completed tool executions.

        Live tool events render one row per execution by mutating a running
        ``tool_call`` entry into ``tool_result``/``tool_error``. Saved history stores
        the assistant ToolCall and the user ToolResult as separate provider messages,
        so restore needs to rejoin them for the transcript to match the live UI.
        """
        entries: list[ConversationEntry] = []
        pending_tool_entries: dict[str, ConversationEntry] = {}

        def remember_tool_entry(block: ToolCall, entry: ConversationEntry) -> None:
            for value in (getattr(block, "id", None), getattr(block, "execution_id", None)):
                if value:
                    pending_tool_entries[str(value)] = entry

        def forget_tool_entry(entry: ConversationEntry) -> None:
            for key, candidate in list(pending_tool_entries.items()):
                if candidate is entry:
                    pending_tool_entries.pop(key, None)

        for item in history:
            try:
                message = Message.from_dict(item)
            except Exception:
                continue
            entries.extend(
                self._conversation_entries_from_message(
                    message,
                    pending_tool_entries=pending_tool_entries,
                    remember_tool_entry=remember_tool_entry,
                    forget_tool_entry=forget_tool_entry,
                )
            )

        return entries

    def _conversation_entries_from_message(
        self,
        message: Message,
        *,
        pending_tool_entries: Optional[dict[str, ConversationEntry]] = None,
        remember_tool_entry: Optional[Callable[[ToolCall, ConversationEntry], None]] = None,
        forget_tool_entry: Optional[Callable[[ConversationEntry], None]] = None,
    ) -> list[ConversationEntry]:
        entries: list[ConversationEntry] = []

        if isinstance(message.content, str):
            content = message.content.strip()
            if content:
                entries.append(self._conversation_entry_for_text(message.role, content))
            return entries

        pending_text: list[str] = []

        def flush_text() -> None:
            text = "\n".join(part for part in pending_text if part).strip()
            pending_text.clear()
            if text:
                entries.append(self._conversation_entry_for_text(message.role, text))

        for block in message.content:
            if isinstance(block, TextBlock):
                pending_text.append(block.text)
            elif isinstance(block, ToolCall):
                flush_text()
                entry = ConversationEntry(
                    kind="tool_call",
                    content=f"Calling {block.name}",
                    complete=False,
                    tool_name=block.name,
                    tool_call_id=getattr(block, "execution_id", None),
                )
                entries.append(entry)
                if remember_tool_entry is not None:
                    remember_tool_entry(block, entry)
            elif isinstance(block, ToolResult):
                flush_text()
                text = self._tool_content_to_text(block.content)
                entry = self._restored_tool_entry_for_result(block, text, pending_tool_entries)
                if entry is None:
                    entries.append(
                        ConversationEntry(
                            kind="tool_error" if block.is_error else "tool_result",
                            content=self._truncate_tool_text(text)
                            if block.is_error
                            else self._tool_result_preview(text),
                            tool_name=block.name,
                            tool_call_id=getattr(block, "execution_id", None),
                            full_content=self._capped_tool_text(text),
                        )
                    )
                else:
                    entry.kind = "tool_error" if block.is_error else "tool_result"
                    entry.content = (
                        self._truncate_tool_text(text) if block.is_error else self._tool_result_preview(text)
                    )
                    entry.complete = True
                    entry.tool_name = block.name or entry.tool_name
                    entry.tool_call_id = getattr(block, "execution_id", None) or entry.tool_call_id
                    entry.full_content = self._capped_tool_text(text)
                    if forget_tool_entry is not None:
                        forget_tool_entry(entry)

        flush_text()
        return entries

    def _restored_tool_entry_for_result(
        self,
        block: ToolResult,
        text: str,
        pending_tool_entries: Optional[dict[str, ConversationEntry]],
    ) -> Optional[ConversationEntry]:
        if not pending_tool_entries:
            return None
        for value in (getattr(block, "tool_use_id", None), getattr(block, "execution_id", None)):
            if value:
                entry = pending_tool_entries.get(str(value))
                if entry is not None:
                    return entry
        return None

    def _conversation_entry_for_text(self, role: str, text: str) -> ConversationEntry:
        names = skill_names_in_text(text)
        if names:
            skill_list = ", ".join(f"`/{name}`" for name in names)
            return ConversationEntry(kind="skill", content=f"Activated skill {skill_list}.")
        return ConversationEntry(kind=self._entry_kind_for_role(role), content=text)

    def _entry_kind_for_role(self, role: str) -> str:
        if role == "assistant":
            return "assistant"
        if role == "user":
            return "user"
        return "system"

    def _tool_content_to_text(self, content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n\n".join(item.to_markdown() if hasattr(item, "to_markdown") else str(item) for item in content)
        return str(content)

    def _apply_stream_chunk(self, chunk: dict, *, kind: str) -> None:
        chunk_uuid = str(chunk.get("uuid") or "")
        content = str(chunk.get("content") or "")
        complete = bool(chunk.get("complete"))
        cache_key = chunk_uuid or f"__nouuid__:{kind}"

        entry = self._stream_entries.get(cache_key)
        if entry is None or (not chunk_uuid and entry.complete):
            if not content:
                return
            entry = ConversationEntry(kind=kind, content="", complete=complete, uuid=chunk_uuid or None)
            self.conversation_entries.append(entry)
            self._stream_entries[cache_key] = entry

        # Defer concatenation to the next render flush (see ConversationEntryWidget.
        # refresh_content): per-chunk `content += content` is O(n^2) over the stream.
        # On completion, materialize now so entry.content is current the moment the
        # segment ends (one O(n) join), without waiting for the flush.
        entry.stream_parts.append(content)
        entry.complete = complete
        if complete:
            entry.materialize()
        self._invalidate_conversation(entry)

    def _begin_turn_progress(self) -> None:
        self._close_sub_agent_inspector()
        self._tool_entries = {}
        self._tool_stream_buffers = {}
        self._sub_agent_activities = {}
        self._sub_agent_by_tool_call = {}
        self._sub_agent_seq = 0
        self._workflow_activities = {}
        self._active_progress_entry = None
        self._turn_active = True
        self._set_chat_enabled(True)
        self._set_composer_status(messages.QUEUE_PLACEHOLDER)
        self._start_turn_timer(messages.WORKING)
        self._set_status_activity(messages.WORKING, turn_state=TurnState.GENERATING)
        self._update_progress(messages.WORKING, complete=False, state=TurnState.GENERATING)

    def _update_progress(self, content: str, complete: bool, state: Optional[TurnState] = None) -> None:
        if complete:
            final_state = state or TurnState.IDLE
            self._complete_turn_timer(content, final_state)
            self._set_status_activity(content, turn_state=final_state)
            if final_state is not TurnState.IDLE:
                tone = "error" if final_state is TurnState.ERROR else "warning"
                self._add_conversation_entry(
                    ConversationEntry(kind="progress", content=content, complete=True, tone=tone)
                )
            self._restore_composer_placeholder()
            return

        text_changed = self._turn_status_text != content
        state_changed = state is not None and self._status_state.turn_state != state
        if not text_changed and not state_changed:
            return
        if text_changed:
            self._turn_status_text = content
            self._refresh_turn_status_strip()
        self._set_status_activity(content, turn_state=state)

    def _update_activity_progress(self, content: str, state: Optional[TurnState] = None) -> None:
        if self._turn_active:
            self._update_progress(content, complete=False, state=state)

    def _finish_turn_progress(self, content: str, state: TurnState = TurnState.IDLE) -> None:
        self._update_progress(content, complete=True, state=state)

    def _persist_agent_into_session(self) -> None:
        """Capture the agent's message history and compaction boundary into the session."""
        if self.agent is None:
            return
        self.session.history = self.agent.dump_message_history()
        self.session.compaction = self.agent.dump_compaction_state()

    def _add_tool_message(self, message_type: str, content: dict) -> None:
        tool_name = str(content.get("tool_description") or content.get("tool_name") or "tool")
        tool_call_id = str(content.get("tool_call_id") or "")
        text = str(content.get("text") or "")
        if tool_name == QUESTION_TOOL_NAME and message_type in {"tool_call", "tool_result"}:
            return
        entry = self._find_tool_entry(tool_call_id, tool_name)

        if message_type == "tool_call":
            self._clear_tool_stream_buffer(tool_call_id, tool_name)
            entry_content = text or f"Calling {tool_name}"
            full_content = ""
            complete = False
            self._update_activity_progress(messages.RUNNING_TOOL.format(tool=tool_name), state=TurnState.RUNNING_TOOL)
        elif message_type == "tool_error":
            entry_content = self._truncate_tool_text(text)
            full_content = self._capped_tool_text(text)
            complete = True
            self._clear_tool_stream_buffer(tool_call_id, tool_name)
            self._update_activity_progress(messages.TOOL_FAILED.format(tool=tool_name))
        else:
            entry_content = self._tool_result_preview(text)
            full_content = self._capped_tool_text(text)
            complete = True
            self._clear_tool_stream_buffer(tool_call_id, tool_name)
            self._update_activity_progress(messages.TOOL_DONE.format(tool=tool_name))

        if entry is None:
            new_entry = ConversationEntry(
                kind=message_type,
                content=entry_content,
                complete=complete,
                tool_name=tool_name,
                tool_call_id=tool_call_id or None,
                full_content=full_content,
            )
            # A preview event can land before this entry exists; apply any stash now.
            pending = getattr(self, "_pending_edit_previews", None)
            if pending and tool_call_id and tool_call_id in pending:
                new_entry.edit_preview = pending.pop(tool_call_id)
            self._add_conversation_entry(new_entry)
            return

        entry.kind = message_type
        entry.content = entry_content
        entry.complete = complete
        entry.tool_name = tool_name
        entry.full_content = full_content
        entry.tool_call_id = tool_call_id or entry.tool_call_id
        if entry.tool_call_id:
            self._tool_entries[entry.tool_call_id] = entry
        self._invalidate_conversation(entry)

    def _apply_tool_streaming_update(self, content: dict) -> None:
        tool_name = str(content.get("tool_name") or content.get("tool_description") or "tool")
        tool_call_id = str(content.get("tool_call_id") or "")
        text = str(content.get("text") or "")
        is_complete = bool(content.get("is_complete"))
        stream_mode = str(content.get("stream_mode") or "replace")
        entry = self._find_tool_entry(tool_call_id, tool_name)
        buffer_key = self._tool_stream_buffer_key(tool_call_id, tool_name)

        if is_complete:
            self._clear_tool_stream_buffer(tool_call_id, tool_name)
            entry_content = self._tool_result_preview(text)
            full_content = self._capped_tool_text(text)
        elif stream_mode == "append":
            # Bound the live buffer to the full-content cap window. The preview is the
            # tail and the live expand view is capped to this many chars anyway, and on
            # completion the entry is recomputed from the complete text. Storing the
            # whole stream via `get()+text` per delta is O(n^2) and stalled the UI on
            # long streamed tool output.
            cap = theme.TOOL_FULL_CONTENT_CAP_CHARS
            buffer_text = (self._tool_stream_buffers.get(buffer_key, "") + text)[-cap:]
            self._tool_stream_buffers[buffer_key] = buffer_text
            entry_content = self._tool_stream_preview(buffer_text)
            full_content = self._capped_tool_text(buffer_text)
        else:
            self._tool_stream_buffers[buffer_key] = text
            entry_content = self._truncate_tool_text(text)
            full_content = self._capped_tool_text(text)

        if is_complete:
            self._update_activity_progress(messages.TOOL_DONE.format(tool=tool_name))
        else:
            self._update_activity_progress(messages.RUNNING_TOOL.format(tool=tool_name), state=TurnState.RUNNING_TOOL)

        if entry is None:
            self._add_conversation_entry(
                ConversationEntry(
                    kind="tool_result" if is_complete else "tool_call",
                    content=entry_content or f"Running {tool_name}",
                    complete=is_complete,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id or None,
                    full_content=full_content,
                )
            )
            return

        entry.kind = "tool_result" if is_complete else "tool_call"
        entry.content = entry_content or entry.content
        entry.complete = is_complete
        entry.tool_name = tool_name
        entry.full_content = full_content or entry.full_content
        entry.tool_call_id = tool_call_id or entry.tool_call_id
        if entry.tool_call_id:
            self._tool_entries[entry.tool_call_id] = entry
        self._invalidate_conversation(entry)

    def _tool_stream_buffer_key(self, tool_call_id: str, tool_name: str) -> str:
        return tool_call_id or f"name:{tool_name}"

    def _clear_tool_stream_buffer(self, tool_call_id: str, tool_name: str) -> None:
        if tool_call_id:
            self._tool_stream_buffers.pop(tool_call_id, None)
        self._tool_stream_buffers.pop(f"name:{tool_name}", None)

    def _apply_edit_preview(self, content: dict) -> None:
        """Attach a UI-only diff/head preview (from a file_edit_preview event) to its tool entry."""
        tool_call_id = str(content.get("tool_call_id") or "")
        tool_name = str(content.get("tool_name") or "")
        if not tool_call_id:
            return
        entry = self._find_tool_entry(tool_call_id, tool_name)
        if entry is None:
            # Preview can arrive before the tool entry exists; stash and apply on creation.
            previews: dict[str, dict] = getattr(self, "_pending_edit_previews", None) or {}
            previews[tool_call_id] = content
            self._pending_edit_previews = previews
            return
        entry.edit_preview = content
        self._invalidate_conversation(entry)

    def _record_file_change_event(self, event: AgentEvent) -> Optional[SessionFileChange]:
        """Capture a UI-only edit preview in the live session changes list."""
        content = event.content or {}
        path = str(content.get("path") or "").strip()
        kind = str(content.get("kind") or "").strip()
        if not path or kind not in {"diff", "head"}:
            return None

        agent_id = ""
        agent_name = ""
        source_label = "Agent"
        if event.sub_agent_info:
            activity = self._ensure_sub_agent_activity(event)
            agent_id = activity.agent_id
            agent_name = activity.agent_name
            source_label = f"Sub-agent {activity.agent_name} #{activity.index}"
        elif event.sender:
            agent_name = str(event.sender)

        index = len(self._session_file_changes) + 1
        change = SessionFileChange(
            change_id=f"change-{index}",
            index=index,
            path=path,
            preview=dict(content),
            tool_name=str(content.get("tool_name") or ""),
            tool_call_id=str(content.get("tool_call_id") or ""),
            agent_id=agent_id,
            agent_name=agent_name,
            source_label=source_label,
            created_at=self._now(),
        )
        self._session_file_changes.append(change)
        self._invalidate_changes_detail(change)
        return change

    def _apply_sub_agent_edit_preview(self, event: AgentEvent) -> None:
        """Attach a file edit preview to the matching sub-agent tool step."""
        content = event.content or {}
        tool_call_id = str(content.get("tool_call_id") or "").strip()
        if not tool_call_id:
            return
        activity = self._ensure_sub_agent_activity(event)
        step = activity.tool_steps.get(tool_call_id)
        if step is None:
            activity.pending_edit_previews[tool_call_id] = dict(content)
        else:
            step.edit_preview = dict(content)
        self._invalidate_sub_agent_detail(activity)

    def _attach_pending_sub_agent_edit_preview(self, activity: SubAgentActivity, step: ConversationEntry) -> None:
        tool_call_id = step.tool_call_id or ""
        if not tool_call_id:
            return
        preview = activity.pending_edit_previews.pop(tool_call_id, None)
        if preview:
            step.edit_preview = preview

    def _invalidate_changes_detail(self, change: Optional[SessionFileChange] = None) -> None:
        """Refresh the open changes inspector if present (no-op when closed)."""
        screen = getattr(self, "_changes_inspector", None)
        if screen is not None:
            screen.note_change_updated(change)

    def _find_tool_entry(self, tool_call_id: str, tool_name: str) -> Optional[ConversationEntry]:
        if tool_call_id:
            return self._tool_entries.get(tool_call_id)
        for entry in reversed(self.conversation_entries):
            if entry.kind not in {"tool_call", "tool_result", "tool_error"}:
                continue
            if entry.complete:
                continue
            if entry.tool_name == tool_name:
                return entry
        return None

    def _sub_agent_key(self, event: AgentEvent) -> str:
        info = event.sub_agent_info or {}
        return str(info.get("agent_id") or info.get("parent_tool_call_id") or info.get("agent_name") or event.sender)

    def _ensure_sub_agent_activity(self, event: AgentEvent) -> SubAgentActivity:
        key = self._sub_agent_key(event)
        activity = self._sub_agent_activities.get(key)
        if activity is None:
            info = event.sub_agent_info or {}
            self._sub_agent_seq += 1
            entry = ConversationEntry(kind="sub_agent", content="", complete=False)
            task_full = str(info.get("task_full") or info.get("task") or "")
            activity = SubAgentActivity(
                agent_id=key,
                agent_name=str(info.get("agent_name") or event.sender or "sub-agent"),
                task=str(info.get("task") or ""),
                index=self._sub_agent_seq,
                entry=entry,
                task_full=task_full,
                workflow_run_id=str(info.get("workflow_run_id") or ""),
                workflow_phase=str(info.get("phase") or ""),
                started_at=self._now(),
            )
            self._sub_agent_activities[key] = activity
            parent_id = info.get("parent_tool_call_id")
            if parent_id:
                self._sub_agent_by_tool_call[str(parent_id)] = key
            if task_full:
                # The full task becomes the first entry in the inspector trajectory; the
                # inline summary keeps showing only the truncated preview.
                activity.steps.append(
                    ConversationEntry(kind="sub_agent_task", content=task_full, full_content=task_full)
                )
            entry.content = self._format_sub_agent_content(activity)
            self._add_conversation_entry(entry)
            self._refresh_sub_agent_activity_status()
            self._note_workflow_sub_agent(activity)
        return activity

    def _render_sub_agent_event(self, event: AgentEvent) -> None:
        activity = self._ensure_sub_agent_activity(event)
        content = event.content
        info = event.sub_agent_info or {}
        depth = info.get("depth")
        if isinstance(depth, int):
            activity.depth = depth
        status = content.get("status")
        if status:  # lifecycle event from AgentTool
            tokens = content.get("total_tokens", content.get("tokens"))
            if isinstance(tokens, int):
                activity.tokens = tokens
            if status != "GENERATING":
                message = str(content.get("message") or "")
                failed = status == "ERROR" or message.startswith("Error")
                activity.status = "failed" if failed else "completed"
                activity.finished_at = self._now()
                activity.entry.complete = True
                activity.current_action = ""
                activity.last_activity = message if failed else ""
                self._refresh_sub_agent_activity_status()
            self._refresh_sub_agent_entry(activity, force=True)
            self._invalidate_sub_agent_detail(activity)
            self._note_workflow_sub_agent(activity)
            return

        message_type = content.get("message_type", "response")
        text = str(content.get("text") or "")
        if message_type == "tool_call":
            activity.tool_calls += 1
            tool = str(content.get("tool_description") or content.get("tool_name") or "tool")
            activity.last_activity = tool
            activity.current_action = tool
            self._record_sub_agent_tool_step(activity, "tool_call", content)
        elif message_type in {"tool_result", "tool_error"}:
            suffix = "failed" if message_type == "tool_error" else "done"
            tool = str(content.get("tool_description") or content.get("tool_name") or "tool")
            activity.last_activity = f"{tool} {suffix}"
            activity.current_action = f"{tool} {suffix}"
            self._record_sub_agent_tool_step(activity, message_type, content)
        elif message_type == "thinking":
            activity.last_activity = "thinking"
            activity.current_action = "thinking"
            self._accumulate_sub_agent_stream(activity, "thinking", event, text)
        else:  # streamed response text - accumulate by chunk uuid
            activity.current_action = "responding"
            if event.uuid and text:
                # Keep only a bounded tail. The card shows just the last
                # SUB_AGENT_TAIL_CHARS (whitespace-collapsed), so storing the whole
                # response via `get()+text` per delta was pure O(n^2) waste that froze
                # the UI on long reasoning streams. A small multiple of the display
                # window preserves the truncation/ellipsis behavior at O(1) per delta.
                tail_cap = theme.SUB_AGENT_TAIL_CHARS * 4
                buffer = (activity.stream_buffers.get(event.uuid, "") + text)[-tail_cap:]
                activity.stream_buffers[event.uuid] = buffer
                activity.active_stream_uuid = event.uuid
            self._accumulate_sub_agent_stream(activity, "assistant", event, text)
        self._refresh_sub_agent_entry(activity)
        self._invalidate_sub_agent_detail(activity)

    def _record_sub_agent_tool_step(self, activity: SubAgentActivity, message_type: str, content: dict) -> None:
        """Capture a sub-agent tool_call/result/error as a ConversationEntry step,
        pairing call->result by tool_call_id exactly like _add_tool_message.

        Sub-agent tool events always carry a stable tool_call_id (the emitter sets it from
        tool_execution_id). The no-id branch is a defensive fallback: a call always gets its
        own step, and a result/error attaches to the most recent unpaired same-name call so
        distinct executions of the same tool can never collide onto one shared step.
        """
        tool_name = str(content.get("tool_description") or content.get("tool_name") or "tool")
        tool_call_id = str(content.get("tool_call_id") or "").strip()
        text = str(content.get("text") or "")
        if message_type == "tool_call":
            entry_content = text or f"Calling {tool_name}"
            full_content = ""
            complete = False
        elif message_type == "tool_error":
            entry_content = self._truncate_tool_text(text)
            full_content = self._capped_tool_text(text)
            complete = True
        else:  # tool_result
            entry_content = self._tool_result_preview(text)
            full_content = self._capped_tool_text(text)
            complete = True

        step = None
        if tool_call_id:
            step = activity.tool_steps.get(tool_call_id)
        elif message_type != "tool_call":
            step = self._last_unpaired_sub_agent_tool_step(activity, tool_name)
        if step is None:
            step = ConversationEntry(
                kind=message_type,
                content=entry_content,
                complete=complete,
                tool_name=tool_name,
                tool_call_id=tool_call_id or None,
                full_content=full_content,
            )
            activity.steps.append(step)
            if tool_call_id:
                activity.tool_steps[tool_call_id] = step
            self._attach_pending_sub_agent_edit_preview(activity, step)
            return
        step.kind = message_type
        step.content = entry_content
        step.complete = complete
        step.tool_name = tool_name
        step.full_content = full_content or step.full_content
        self._attach_pending_sub_agent_edit_preview(activity, step)

    def _last_unpaired_sub_agent_tool_step(
        self, activity: SubAgentActivity, tool_name: str
    ) -> Optional[ConversationEntry]:
        """Most recent still-running tool step for a tool name (name-based result pairing)."""
        for step in reversed(activity.steps):
            if step.kind == "tool_call" and not step.complete and step.tool_name == tool_name:
                return step
        return None

    def _accumulate_sub_agent_stream(self, activity: SubAgentActivity, kind: str, event: AgentEvent, text: str) -> None:
        """Accumulate streamed thinking/response chunks into one step per chunk uuid,
        mirroring the main transcript's _apply_stream_chunk.

        Events normally carry a uuid; the kind-qualified sentinel for the no-uuid case keeps
        consecutive uuid-less chunks of the same kind merged into one step (rather than
        fragmenting) while never merging thinking into response.
        """
        complete = not event.is_streaming
        chunk_uuid = str(event.uuid or "")
        cache_key = chunk_uuid or f"__nouuid__:{kind}"
        step = activity.stream_steps.get(cache_key)
        if step is None:
            if not text and not complete:
                return
            step = ConversationEntry(kind=kind, content="", complete=complete, uuid=chunk_uuid or None)
            activity.steps.append(step)
            activity.stream_steps[cache_key] = step
        # Defer concatenation: deltas land in stream_parts (O(1)) and are folded into
        # content once on completion (and at render time by ConversationEntryWidget.
        # refresh_content), mirroring _apply_stream_chunk. A per-delta `step.content +=
        # text` is O(n^2) over a long reasoning stream — on DeepSeek-class sub-agents it
        # grew to seconds of event-loop CPU and froze scrolling while sub-agents ran.
        if text:
            step.stream_parts.append(text)
        step.complete = complete
        if complete:
            step.materialize()

    def _note_sub_agent_tool_stream(self, event: AgentEvent) -> None:
        activity = self._ensure_sub_agent_activity(event)
        tool_name = str(event.content.get("tool_name") or event.content.get("tool_description") or "tool")
        is_complete = bool(event.content.get("is_complete"))
        activity.last_activity = f"{tool_name} done" if is_complete else f"{tool_name} streaming"
        self._refresh_sub_agent_entry(activity)

    def _note_sub_agent_context(self, event: AgentEvent) -> None:
        """Record a sub-agent's context-window usage on its own card, so it never
        overwrites the main agent's context indicator on the status dashboard."""
        activity = self._ensure_sub_agent_activity(event)
        content = event.content
        activity.context_percentage = self._as_optional_float(content.get("usage_percentage"))
        activity.context_input_tokens = self._as_optional_int(content.get("input_tokens"))
        activity.context_max_tokens = self._as_optional_int(content.get("max_tokens"))
        self._refresh_sub_agent_entry(activity)
        self._invalidate_sub_agent_detail(activity)

    def _note_sub_agent_status(self, event: AgentEvent) -> None:
        """Surface a sub-agent's status notice (e.g. provider overload) on its card,
        keeping it off the main activity line."""
        activity = self._ensure_sub_agent_activity(event)
        message = str(event.content.get("message") or "").strip()
        if message:
            activity.last_activity = message
            if activity.status == "running":
                activity.current_action = message
        self._refresh_sub_agent_entry(activity)
        self._invalidate_sub_agent_detail(activity)

    def _refresh_sub_agent_entry(self, activity: SubAgentActivity, *, force: bool = False) -> None:
        activity.entry.content = self._format_sub_agent_content(activity)
        self._invalidate_conversation(activity.entry)
        if force:
            self._flush_conversation_render()

    def _format_sub_agent_content(self, activity: SubAgentActivity) -> str:
        header = Text.from_markup(self._sub_agent_header(activity)).plain
        body = "\n".join(self._sub_agent_body_lines(activity))
        return f"{header}\n{body}" if body else header

    def _format_sub_agent_renderable(self, activity: SubAgentActivity) -> Text | Group:
        return self._entry_renderable(
            self._sub_agent_header(activity),
            "\n".join(self._sub_agent_body_lines(activity)),
        )

    def _sub_agent_header(self, activity: SubAgentActivity) -> str:
        if activity.finished_at is not None:
            elapsed = max(0.0, activity.finished_at - activity.started_at)
        else:
            elapsed = max(0.0, self._now() - activity.started_at)
        duration = self._format_turn_duration(elapsed)

        if activity.status == "running":
            color, state = Color.ACCENT, f"running {theme.g(Glyph.BULLET_SEP)} {duration}"
        elif activity.status == "completed":
            color, state = Color.SUCCESS, f"completed in {duration}"
        elif activity.status == "failed":
            color, state = Color.ERROR, f"failed after {duration}"
        else:
            color, state = Color.WARNING, f"stopped after {duration}"

        return theme.role_header(
            Glyph.SUB_AGENT,
            escape(activity.agent_name),
            color,
            state=f"#{activity.index} {theme.g(Glyph.BULLET_SEP)} {state}",
        )

    def _sub_agent_body_lines(self, activity: SubAgentActivity) -> list[str]:
        sep = theme.g(Glyph.BULLET_SEP)
        body_lines: list[str] = []
        if activity.task:
            task = activity.task
            if len(task) > theme.SUB_AGENT_TASK_PREVIEW_CHARS:
                task = f"{task[: theme.SUB_AGENT_TASK_PREVIEW_CHARS]}{theme.g(Glyph.ELLIPSIS)}"
            body_lines.append(f"Task: {task}")
        tools_line = f"{activity.tool_calls} tool{'' if activity.tool_calls == 1 else 's'}"
        if activity.tokens:
            tools_line += f" {sep} {self._format_token_count(activity.tokens)} tok"
        if activity.context_percentage is not None:
            tools_line += f" {sep} ctx {activity.context_percentage:.0f}%"
        if activity.last_activity:
            tools_line += f" {sep} last: {activity.last_activity}"
        body_lines.append(tools_line)
        if activity.status == "running" and activity.current_action:
            body_lines.append(f"{theme.g(Glyph.TOOL)} now: {activity.current_action}")
        if activity.status == "running" and activity.active_stream_uuid:
            tail = activity.stream_buffers.get(activity.active_stream_uuid, "")
            tail = " ".join(tail.split())
            if tail:
                if len(tail) > theme.SUB_AGENT_TAIL_CHARS:
                    tail = f"{theme.g(Glyph.ELLIPSIS)}{tail[-theme.SUB_AGENT_TAIL_CHARS :]}"
                body_lines.append(tail)
        if any(step.kind != "sub_agent_task" for step in activity.steps):
            body_lines.append(messages.SUB_AGENT_INSPECT_HINT)

        return body_lines

    def _format_token_count(self, tokens: int) -> str:
        """Compact token count for cards/roster: 980, 3.1k, 1.2M."""
        if tokens < 1000:
            return str(tokens)
        if tokens < 1_000_000:
            return f"{tokens / 1000:.1f}k".replace(".0k", "k")
        return f"{tokens / 1_000_000:.1f}M".replace(".0M", "M")

    def _invalidate_sub_agent_detail(self, activity: SubAgentActivity) -> None:
        """Refresh the open inspector if it is showing this agent (no-op when closed)."""
        screen = self._sub_agent_inspector
        if screen is not None:
            screen.note_activity_updated(activity)

    def _sub_agent_activity_for_entry(self, entry: ConversationEntry) -> Optional[SubAgentActivity]:
        for activity in self._sub_agent_activities.values():
            if activity.entry is entry:
                return activity
        return None

    def _running_sub_agents(self) -> list[SubAgentActivity]:
        return [a for a in self._sub_agent_activities.values() if a.status == "running"]

    def _refresh_sub_agent_activity_status(self) -> None:
        running = self._running_sub_agents()
        if running:
            if len(running) == 1:
                text = messages.RUNNING_SUB_AGENT.format(name=running[0].agent_name, index=running[0].index)
            else:
                text = messages.RUNNING_SUB_AGENTS.format(count=len(running))
            self._update_activity_progress(text, state=TurnState.RUNNING_SUB_AGENTS)
        elif self._turn_active:
            self._update_activity_progress(messages.WORKING, state=TurnState.GENERATING)

    def _finalize_sub_agent_activities(self, status: str = "stopped") -> None:
        """Mark still-running sub-agents as finished (no lifecycle event arrives on cancel)."""
        changed = False
        for activity in self._sub_agent_activities.values():
            if activity.status == "running":
                activity.status = status
                activity.finished_at = self._now()
                activity.entry.complete = True
                activity.entry.content = self._format_sub_agent_content(activity)
                self._invalidate_conversation(activity.entry)
                changed = True
        if changed:
            self._flush_conversation_render()

    def _tick_running_sub_agents(self) -> None:
        running = self._running_sub_agents()
        if not running:
            return
        for activity in running:
            activity.entry.content = self._format_sub_agent_content(activity)
            self._invalidate_conversation(activity.entry)

    # ---- workflow cards ("gigacode") ------------------------------------------

    def _handle_workflow_start(self, content: dict) -> None:
        run_id = str(content.get("workflow_run_id") or "")
        if not run_id or run_id in self._workflow_activities:
            return
        phases: list[PhaseState] = []
        for raw in content.get("phases") or []:
            # meta.phases entries may be dicts ({title, detail}) or bare strings;
            # extract_meta only guarantees name+description, so guard the shape.
            if isinstance(raw, dict):
                title = str(raw.get("title") or "").strip()
                detail = str(raw.get("detail") or "").strip()
            else:
                title, detail = str(raw).strip(), ""
            if title:
                phases.append(PhaseState(title=title, detail=detail))
        entry = ConversationEntry(kind="workflow", content="", complete=False)
        activity = WorkflowActivity(
            run_id=run_id,
            name=str(content.get("name") or "workflow"),
            description=str(content.get("description") or ""),
            entry=entry,
            phases=phases,
            started_at=self._now(),
        )
        self._workflow_activities[run_id] = activity
        entry.content = self._format_workflow_content(activity)
        self._add_conversation_entry(entry)

    def _handle_workflow_phase(self, content: dict) -> None:
        title = str(content.get("text") or "").strip()
        activity = self._workflow_for_run(content)
        if title:
            self._update_activity_progress(f"workflow: {title}", state=TurnState.RUNNING_SUB_AGENTS)
        if activity is None or not title:
            return
        # phase() calls are sequential, so a new explicit phase retires the prior one.
        if activity.current_phase and activity.current_phase != title:
            prev = activity.phase_by_title(activity.current_phase)
            if prev is not None and prev.state == "active":
                prev.state = "done"
        phase = activity.phase_by_title(title)
        if phase is None:
            phase = PhaseState(title=title)
            activity.phases.append(phase)
        if phase.state == "pending":
            phase.state = "active"
        activity.current_phase = title
        self._refresh_workflow_entry(activity)

    def _handle_workflow_log(self, content: dict) -> None:
        message = str(content.get("text") or "").strip()
        activity = self._workflow_for_run(content)
        if activity is None or not message:
            return
        activity.latest_log = message
        self._refresh_workflow_entry(activity)

    def _handle_workflow_end(self, content: dict) -> None:
        activity = self._workflow_for_run(content)
        if activity is None:
            return
        status = str(content.get("status") or "completed")
        activity.status = status
        activity.finished_at = self._now()
        activity.current_phase = ""
        for phase in activity.phases:
            if status == "failed":
                # Only an in-flight phase failed; phases never reached stay pending.
                if phase.state == "active":
                    phase.state = "failed"
            elif phase.state in {"active", "pending"}:
                phase.state = "done"
        activity.entry.complete = True
        self._refresh_workflow_entry(activity, force=True)

    def _workflow_for_run(self, content: dict) -> Optional[WorkflowActivity]:
        run_id = str(content.get("workflow_run_id") or "")
        return self._workflow_activities.get(run_id) if run_id else None

    def _note_workflow_sub_agent(self, activity: SubAgentActivity) -> None:
        """Roll a workflow sub-agent's phase/tokens into its workflow card.

        sub_agent_info carries workflow_run_id + phase for every workflow-dispatched
        agent; consuming it here drives per-phase agent counts and marks a phase active
        even when the script used the agent(phase=...) kwarg (which emits no phase event).
        """
        card = self._workflow_activities.get(activity.workflow_run_id) if activity.workflow_run_id else None
        if card is None:
            return
        self._recompute_workflow_rollup(card)
        self._refresh_workflow_entry(card)

    def _recompute_workflow_rollup(self, card: WorkflowActivity) -> None:
        """Idempotently derive agent counts + tokens for a card from its sub-agents."""
        members = [a for a in self._sub_agent_activities.values() if a.workflow_run_id == card.run_id]
        card.agent_count = len(members)
        card.tokens = sum(a.tokens for a in members if isinstance(a.tokens, int))
        by_phase: dict[str, list[SubAgentActivity]] = {}
        for member in members:
            by_phase.setdefault(member.workflow_phase, []).append(member)
        for phase in card.phases:
            members_for_phase = by_phase.get(phase.title, [])
            phase.agents_total = len(members_for_phase)
            phase.agents_done = sum(1 for a in members_for_phase if a.status != "running")
            if members_for_phase and phase.state == "pending":
                phase.state = "active"
        # Phases that exist only via the agent(phase=...) kwarg, never declared in meta.
        for title, members_for_phase in by_phase.items():
            if title and card.phase_by_title(title) is None:
                card.phases.append(
                    PhaseState(
                        title=title,
                        state="active",
                        agents_total=len(members_for_phase),
                        agents_done=sum(1 for a in members_for_phase if a.status != "running"),
                    )
                )

    def _refresh_workflow_entry(self, activity: WorkflowActivity, *, force: bool = False) -> None:
        activity.entry.content = self._format_workflow_content(activity)
        self._invalidate_conversation(activity.entry)
        if force:
            self._flush_conversation_render()

    def _workflow_activity_for_entry(self, entry: ConversationEntry) -> Optional[WorkflowActivity]:
        for activity in self._workflow_activities.values():
            if activity.entry is entry:
                return activity
        return None

    def _tick_running_workflows(self) -> None:
        for activity in self._workflow_activities.values():
            if activity.status == "running":
                activity.entry.content = self._format_workflow_content(activity)
                self._invalidate_conversation(activity.entry)

    def _finalize_workflow_activities(self, status: str = "stopped") -> None:
        """Mark still-running workflow cards as finished (workflow_end never arrives on cancel)."""
        changed = False
        for activity in self._workflow_activities.values():
            if activity.status == "running":
                activity.status = status
                activity.finished_at = self._now()
                activity.current_phase = ""
                if status == "completed":
                    for phase in activity.phases:
                        if phase.state in {"active", "pending"}:
                            phase.state = "done"
                # On a stop/cancel, leave phase glyphs as they were — an interrupted
                # phase stays "active" rather than misreporting as done or pending.
                activity.entry.complete = True
                activity.entry.content = self._format_workflow_content(activity)
                self._invalidate_conversation(activity.entry)
                changed = True
        if changed:
            self._flush_conversation_render()

    def _workflow_phase_glyph(self, phase: PhaseState) -> tuple[str, str]:
        if phase.state == "done":
            return Glyph.CHECK, Color.SUCCESS
        if phase.state == "failed":
            return Glyph.CROSS, Color.ERROR
        if phase.state == "active":
            return Glyph.RUNNING, Color.ACCENT
        return Glyph.PENDING, Color.MUTED

    def _workflow_header(self, activity: WorkflowActivity) -> str:
        if activity.finished_at is not None:
            elapsed = max(0.0, activity.finished_at - activity.started_at)
        else:
            elapsed = max(0.0, self._now() - activity.started_at)
        duration = self._format_turn_duration(elapsed)
        sep = theme.g(Glyph.BULLET_SEP)
        if activity.status == "running":
            color, state = Color.ACCENT, f"running {sep} {duration}"
        elif activity.status == "completed":
            color, state = Color.SUCCESS, f"completed in {duration}"
        elif activity.status == "failed":
            color, state = Color.ERROR, f"failed after {duration}"
        else:
            color, state = Color.WARNING, f"stopped after {duration}"
        return theme.role_header(
            Glyph.PLAN,
            escape(activity.name or "workflow"),
            color,
            state=f"workflow {sep} {state}",
        )

    def _workflow_footer_line(self, activity: WorkflowActivity) -> str:
        sep = theme.g(Glyph.BULLET_SEP)
        bits: list[str] = []
        if activity.status == "running" and activity.current_phase:
            bits.append(f"now: {activity.current_phase}")
        if activity.agent_count:
            bits.append(f"{activity.agent_count} agent{'' if activity.agent_count == 1 else 's'}")
        if activity.tokens:
            bits.append(f"{self._format_token_count(activity.tokens)} tok")
        if activity.latest_log:
            log = activity.latest_log
            if len(log) > theme.SUB_AGENT_TASK_PREVIEW_CHARS:
                log = f"{log[: theme.SUB_AGENT_TASK_PREVIEW_CHARS]}{theme.g(Glyph.ELLIPSIS)}"
            bits.append(log)
        return f" {sep} ".join(bits)

    def _workflow_phase_rows(self, activity: WorkflowActivity) -> list[Text]:
        sep = theme.g(Glyph.BULLET_SEP)
        bar = f"  {theme.g(Glyph.INSET_BAR)}"
        rows: list[Text] = []
        for phase in activity.phases:
            glyph, color = self._workflow_phase_glyph(phase)
            line = Text()
            line.append(bar, style="dim")
            line.append(" ")
            line.append(f"{theme.g(glyph)} ", style=color)
            title_style = "bold" if phase.state == "active" else ("dim" if phase.state == "pending" else "")
            line.append(phase.title, style=title_style)
            if phase.detail:
                line.append(f"  {sep} {phase.detail}", style="dim")
            if phase.agents_total:
                line.append(f"  {sep} {phase.agents_done}/{phase.agents_total} agents", style="dim")
            rows.append(line)
        return rows

    def _format_workflow_renderable(self, activity: WorkflowActivity) -> Group:
        parts: list = [Text.from_markup(self._workflow_header(activity))]
        if activity.description:
            parts.append(self._format_inset_text(activity.description, style="dim"))
        parts.extend(self._workflow_phase_rows(activity))
        footer = self._workflow_footer_line(activity)
        if footer:
            parts.append(self._format_inset_text(footer, style="dim"))
        return Group(*parts)

    def _format_workflow_content(self, activity: WorkflowActivity) -> str:
        header = Text.from_markup(self._workflow_header(activity)).plain
        lines = [header]
        if activity.description:
            lines.append(activity.description)
        for phase in activity.phases:
            glyph, _ = self._workflow_phase_glyph(phase)
            row = f"{theme.g(glyph)} {phase.title}"
            if phase.detail:
                row += f" — {phase.detail}"
            if phase.agents_total:
                row += f" ({phase.agents_done}/{phase.agents_total} agents)"
            lines.append(row)
        footer = self._workflow_footer_line(activity)
        if footer:
            lines.append(footer)
        return "\n".join(lines)

    def _tool_result_preview(self, text: str) -> str:
        # The entry header already conveys completion; the body is just the preview.
        return self._truncate_tool_text(text)

    def _truncate_tool_text(self, text: str) -> str:
        if len(text) <= theme.TOOL_RESULT_PREVIEW_CHARS:
            return text
        return f"{text[: theme.TOOL_RESULT_PREVIEW_CHARS]}{theme.g(Glyph.ELLIPSIS)}"

    def _capped_tool_text(self, text: str) -> str:
        if len(text) <= theme.TOOL_FULL_CONTENT_CAP_CHARS:
            return text
        return f"{text[: theme.TOOL_FULL_CONTENT_CAP_CHARS]}{theme.g(Glyph.ELLIPSIS)}"

    def _tool_stream_preview(self, text: str) -> str:
        if len(text) <= theme.TOOL_STREAM_PREVIEW_CHARS:
            return text
        notice = messages.STREAM_TRUNCATED.format(chars=theme.TOOL_STREAM_PREVIEW_CHARS)
        return f"{notice}\n{text[-theme.TOOL_STREAM_PREVIEW_CHARS :]}"

    def _invalidate_conversation(self, entry: Optional[ConversationEntry] = None) -> None:
        """Mark the conversation dirty and coalesce re-renders.

        Hot paths (stream chunks, tool updates, sub-agent ticks) call this
        instead of rendering directly, so rapid event bursts produce at most
        one flush per coalesce interval, and a flush only touches new or
        changed entry widgets.
        """
        if entry is not None:
            self._dirty_entry_ids.add(entry.entry_id)
        if self._render_pending:
            return
        self._render_pending = True
        interval = self._render_coalesce_interval(entry)
        try:
            self.set_timer(
                interval,
                self._flush_conversation_render,
                name="conversation-render",
            )
        except Exception:
            # Timers are unavailable before the app is running; render directly.
            self._flush_conversation_render()

    def _render_coalesce_interval(self, entry: Optional[ConversationEntry]) -> float:
        """Back off the flush cadence for very large live entries.

        Each flush makes Textual re-measure the auto-height streaming widget (O(height)),
        so for big reasoning streams fewer, larger flushes beat ~20/sec full re-measures.
        Length lags one flush (deltas materialize at flush time), which is fine: the entry
        only grows, so the cadence steps up within one interval of crossing a threshold.
        """
        if entry is None:
            return theme.RENDER_COALESCE_INTERVAL
        size = len(entry.content)
        if size >= theme.RENDER_COALESCE_LARGE_CHARS:
            return theme.RENDER_COALESCE_INTERVAL_LARGE
        if size >= theme.RENDER_COALESCE_MEDIUM_CHARS:
            return theme.RENDER_COALESCE_INTERVAL_MEDIUM
        return theme.RENDER_COALESCE_INTERVAL

    def _flush_conversation_render(self) -> None:
        if not self._render_pending:
            return
        self._render_pending = False
        try:
            view = self._conversation
        except Exception:
            # A coalesced flush can fire after the widget is unmounted (e.g. on exit).
            self._dirty_entry_ids.clear()
            return
        if not view.is_attached:
            # During teardown the view detaches before it is flagged closing, so query_one
            # still resolves it but mounting into it raises MountError.
            self._dirty_entry_ids.clear()
            return

        should_follow = bool(getattr(view, "auto_follow_bottom", False)) or view.is_at_bottom()

        # Fast path for the common streaming case: the transcript shape is unchanged
        # and only already-mounted entries need their renderables refreshed.
        if (
            self._dirty_entry_ids
            and len(self._entry_widgets) == len(self.conversation_entries)
            and all(entry_id in self._entry_widgets for entry_id in self._dirty_entry_ids)
        ):
            self._refresh_dirty_entry_widgets()
            if should_follow:
                self._schedule_conversation_bottom_anchor()
            else:
                self._update_jump_button()
            return

        rendered_ids = list(self._entry_widgets)
        current_ids = [entry.entry_id for entry in self.conversation_entries]
        if current_ids[: len(rendered_ids)] != rendered_ids:
            # Entries were removed, replaced, or inserted before the end; rebuild.
            self._render_conversation()
            return

        self._refresh_dirty_entry_widgets()

        new_entries = self.conversation_entries[len(rendered_ids) :]
        if new_entries:
            widgets = []
            for entry in new_entries:
                widget = self._make_entry_widget(entry)
                self._entry_widgets[entry.entry_id] = widget
                widgets.append(widget)
            view.mount(*widgets)
        if should_follow:
            self._schedule_conversation_bottom_anchor()
        else:
            self._update_jump_button()

    def _refresh_dirty_entry_widgets(self) -> None:
        for entry_id in self._dirty_entry_ids:
            widget = self._entry_widgets.get(entry_id)
            if widget is not None:
                widget.refresh_content()
        self._dirty_entry_ids.clear()

    def _render_conversation(self) -> None:
        """Full rebuild of the conversation view (restore, reset, startup changes)."""
        self._render_pending = False
        self._dirty_entry_ids.clear()
        try:
            view = self._conversation
        except Exception:
            return
        if not view.is_attached:
            return
        had_rendered_entries = bool(self._entry_widgets)
        should_follow = (
            not had_rendered_entries or bool(getattr(view, "auto_follow_bottom", False)) or view.is_at_bottom()
        )
        view.remove_children()
        self._entry_widgets = {}
        widgets = []
        for entry in self.conversation_entries:
            widget = self._make_entry_widget(entry)
            self._entry_widgets[entry.entry_id] = widget
            widgets.append(widget)
        if widgets:
            view.mount(*widgets)
        if should_follow:
            self._schedule_conversation_bottom_anchor()
        else:
            view.set_auto_follow(False)
            self._update_jump_button()

    def _make_entry_widget(self, entry: ConversationEntry) -> ConversationEntryWidget | ToolEntryWidget:
        if entry.kind in {"tool_call", "tool_result", "tool_error"}:
            return ToolEntryWidget(entry, self._tool_entry_title, self._tool_preview_renderable)
        if entry.kind == "compaction_summary":
            return ToolEntryWidget(entry, self._compaction_summary_title)
        if entry.kind == "sub_agent":
            return SubAgentEntryWidget(entry, self._format_conversation_entry)
        return ConversationEntryWidget(entry, self._format_conversation_entry)

    def _compaction_summary_title(self, entry: ConversationEntry) -> str:
        return theme.role_header(Glyph.STATUS, messages.COMPACTION_SUMMARY_TITLE, Color.ACCENT)

    def _schedule_conversation_bottom_anchor(self, *, update_button: bool = True) -> None:
        """Pin the transcript to the latest output after layout has settled."""
        try:
            view = self._conversation
        except Exception:
            return
        if not view.is_attached:
            return
        view.set_auto_follow(True)
        if getattr(self, "_conversation_anchor_pending", False):
            return
        self._conversation_anchor_pending = True

        def anchor_after_refresh() -> None:
            self._conversation_anchor_pending = False
            if not view.is_attached:
                return
            view.set_auto_follow(True)
            view.anchor()
            if update_button:
                self.call_after_refresh(self._update_jump_button)

        self.call_after_refresh(anchor_after_refresh)

    def _update_jump_button(self) -> None:
        try:
            view = self._conversation
            bar = self.query_one("#jump_to_bottom", JumpToBottomBar)
        except Exception:
            return
        should_display = (not bool(getattr(view, "auto_follow_bottom", False))) and not view.is_at_bottom()
        if bar.display != should_display:
            bar.display = should_display
            try:
                self.call_after_refresh(self._update_jump_button)
            except Exception:
                pass

    def on_jump_to_bottom_bar_pressed(self, message: JumpToBottomBar.Pressed) -> None:
        self._schedule_conversation_bottom_anchor()

    def _format_conversation_entry(self, entry: ConversationEntry) -> Text | Group:
        """Render an entry using the shared header grammar.

        GRAMMAR: <colored glyph> <bold label> [ · state] — body inset beneath.
        """
        if entry.kind == "startup":
            return self._format_startup_entry(entry)
        streaming = None if entry.complete else theme.g(Glyph.ELLIPSIS)
        if entry.kind == "user":
            header = theme.role_header(Glyph.USER, "You", Color.USER)
            return self._entry_renderable(header, entry.content)
        if entry.kind == "assistant":
            header = theme.role_header(Glyph.AGENT, "Agent", Color.AGENT, state=streaming)
            if entry.complete and entry.content.strip():
                return self._markdown_entry(header, entry.content)
            if not entry.complete:
                return self._streaming_inset_renderable(entry, header)
            return self._entry_renderable(header, entry.content)
        if entry.kind == "thinking":
            header = theme.role_header(
                Glyph.STATUS, "Thinking", Color.THINKING, label_style="dim italic", state=streaming
            )
            if not entry.complete:
                return self._streaming_inset_renderable(entry, header, body_style="italic dim")
            return self._entry_renderable(header, entry.content, body_style="italic dim")
        if entry.kind == "progress":
            color = Color.ERROR if entry.tone == "error" else Color.WARNING
            header = theme.role_header(Glyph.STATUS, "Status", color, state=streaming)
            return self._entry_renderable(header, entry.content)
        if entry.kind == "queued":
            header = theme.role_header(Glyph.STATUS, "Queued", Color.ACCENT)
            return self._entry_renderable(header, entry.content, body_style="dim")
        if entry.kind == "plan":
            header = theme.role_header(Glyph.PLAN, "Plan", Color.SUCCESS)
            if entry.content.strip():
                return self._markdown_entry(header, entry.content)
            return Text.from_markup(header)
        if entry.kind == "question":
            header = theme.role_header(Glyph.QUESTION, "Question", Color.ACCENT)
            return self._entry_renderable(header, entry.content)
        if entry.kind == "skill":
            header = theme.role_header(Glyph.PLAN, "Skill", Color.SUCCESS)
            return self._entry_renderable(header, entry.content)
        if entry.kind == "workflow":
            wf = self._workflow_activity_for_entry(entry)
            if wf is not None:
                return self._format_workflow_renderable(wf)
            return Text(entry.content)
        if entry.kind == "sub_agent_task":
            header = theme.role_header(Glyph.USER, "Task", Color.USER)
            return self._entry_renderable(header, entry.content)
        if entry.kind == "sub_agent":
            activity = self._sub_agent_activity_for_entry(entry)
            if activity is not None:
                return self._format_sub_agent_renderable(activity)
            return Text(entry.content)
        if entry.kind in TOOL_STATE_PRESENTATION:
            state, color = tool_state_presentation(entry.kind)
            return self._format_tool_entry(entry, state=state, color=color)
        if entry.kind == "system":
            return Text(entry.content, style="dim")
        if entry.kind == "lsp":
            return self._markdown_entry(
                theme.role_header(Glyph.STATUS, "LSP", Color.ACCENT),
                entry.content,
            )
        return Text(entry.content)

    def _entry_renderable(
        self,
        header: str,
        body: Optional[str] = None,
        *,
        body_style: Optional[str] = None,
    ) -> Text | Group:
        header_text = Text.from_markup(header)
        if body is None:
            return header_text
        return Group(header_text, self._format_inset_text(body, style=body_style))

    def _streaming_inset_renderable(self, entry, header: str, *, body_style: Optional[str] = None) -> Group:
        """Inset body for a still-streaming entry, built incrementally (O(delta) per flush).

        Caches the rendered Text on the entry so each flush only appends the newly
        arrived characters instead of re-splitting and re-rendering the whole buffer
        (the O(n^2) cliff). The header is small and rebuilt each call. On completion the
        dispatcher routes to the canonical _entry_renderable/_markdown_entry instead.
        """
        state = entry.render_cache
        content = entry.content
        if not isinstance(state, _InsetRenderState) or state.style != body_style or state.length > len(content):
            # First render for this entry, a body-style change, or a defensive shrink:
            # fold the whole current buffer once.
            state = _InsetRenderState(body_style)
            entry.render_cache = state
            self._extend_inset(state, content)
        elif state.length < len(content):
            self._extend_inset(state, content[state.length :])
        return Group(Text.from_markup(header), state.text)

    def _extend_inset(self, state: "_InsetRenderState", delta: str) -> None:
        """Append ``delta`` to the cached inset Text, matching _format_inset_text for "\\n".

        A pending newline is flushed (its new line materialized) as soon as another
        newline *or* content follows it; only a still-pending terminator at the very end
        is left unrendered, so the result equals ``_format_inset_text`` over the buffer.
        """
        if not delta and state.started:
            return
        bar = f"  {theme.g(Glyph.INSET_BAR)}"
        text = state.text

        def flush_deferred() -> None:
            if state.deferred_newline:
                text.append("\n")
                text.append(bar, style="dim")
                state.deferred_newline = False
                state.open_has_content = False

        if not state.started:
            text.append(bar, style="dim")
            state.started = True
            state.open_has_content = False

        for index, segment in enumerate(delta.split("\n")):
            if index > 0:
                # A "\n" boundary confirms any prior deferred line, then defers its own.
                flush_deferred()
                state.deferred_newline = True
            if segment:
                flush_deferred()
                if not state.open_has_content:
                    text.append(" ")
                    state.open_has_content = True
                text.append(segment, style=state.style)
        state.length += len(delta)

    def _markdown_entry(self, header: str, content: str) -> Group:
        return Group(
            Text.from_markup(header),
            Padding(
                RichMarkdown(content, code_theme=theme.markdown_code_theme()),
                (0, 0, 0, theme.INSET_WIDTH),
            ),
        )

    def _format_startup_entry(self, entry: ConversationEntry) -> Text:
        lines = entry.content.splitlines()
        try:
            separator = lines.index("")
        except ValueError:
            separator = len(STARTUP_WORDMARK)
        rendered = Text()
        logo_lines = lines[:separator]
        if logo_lines:
            top, bottom = theme.splash_colors()
            gradient = (
                theme.gradient_hex(top, bottom, len(logo_lines)) if theme.supports_truecolor(self.console) else []
            )
            if gradient:
                # Two-tone vertical gradient: accent at top -> secondary at bottom.
                for index, line in enumerate(logo_lines):
                    if index:
                        rendered.append("\n")
                    rendered.append(line, style=f"bold {gradient[index]}")
            else:
                # 256-color terminal: flat bold primary (matches the primary buttons).
                rendered.append("\n".join(logo_lines), style=f"bold {top}")
        for line in lines[separator + 1 :]:
            rendered.append("\n")
            label, sep, value = line.partition(": ")
            if sep and label and len(label) <= 12:
                # Aligned two-column key/value line: muted label, normal value.
                rendered.append(f"{label + ':':<13}", style="dim")
                rendered.append(value)
            else:
                rendered.append(line, style="dim")
        return rendered

    def _format_tool_entry(self, entry: ConversationEntry, *, state: str, color: str) -> Text | Group:
        tool_name = escape(entry.tool_name or "tool")
        header = theme.role_header(Glyph.TOOL, tool_name, color, state=state)
        if not entry.content:
            return Text.from_markup(header)
        return self._entry_renderable(header, entry.content)

    def _tool_entry_title(self, entry: ConversationEntry) -> str:
        state, color = tool_state_presentation(entry.kind)
        header = theme.role_header(Glyph.TOOL, escape(entry.tool_name or "tool"), color, state=state)
        # Surface an LSP diagnostics summary as a severity-colored badge in the title
        # (e.g. "· 2 LSP warnings") so warnings are visible without expanding. The
        # label is recovered from the result text, which covers live edits, sub-agent
        # steps, and restored sessions through this one shared title factory.
        label = extract_lsp_label(entry.full_content or entry.content)
        if label:
            head, sep, rest = label.partition(" ")
            badge = f"{head} LSP {rest}" if sep else f"LSP {label}"  # "2 warnings" -> "2 LSP warnings"
            badge_color = Color.ERROR if "error" in label else Color.WARNING if "warning" in label else Color.MUTED
            header += " " + theme.styled(f"{theme.g(Glyph.BULLET_SEP)} {badge}", badge_color)
        return header

    def _tool_preview_renderable(self, entry: ConversationEntry) -> Optional[Group]:
        """Inline diff/file-head preview for an edit tool, or None to hide the preview region."""
        preview = entry.edit_preview
        if not preview:
            return None
        try:
            return self._build_edit_preview(preview)
        except Exception:
            return None

    def _build_edit_preview(self, preview: dict) -> Group:
        kind = str(preview.get("kind") or "")
        path = str(preview.get("path") or "file")
        lines = preview.get("lines") or []
        more = int(preview.get("more") or 0)

        meta = Text()
        meta.append(escape(path), style="bold")
        if kind == "diff":
            meta.append("  ")
            meta.append(f"+{int(preview.get('adds') or 0)}", style=Color.SUCCESS)
            meta.append(" ")
            meta.append(f"-{int(preview.get('dels') or 0)}", style=Color.ERROR)

        if kind == "head":
            code = "\n".join(str(row[1]) for row in lines if isinstance(row, (list, tuple)) and len(row) >= 2)
            body = self._edit_preview_code(code, str(preview.get("language") or "text"))
        else:
            body = self._edit_preview_diff(lines)

        parts: list = [meta, Padding(body, (0, 0, 0, theme.INSET_WIDTH))]
        if more > 0:
            footer = Text(f"{theme.g(Glyph.ELLIPSIS)} +{more} more lines", style="dim")
            parts.append(Padding(footer, (0, 0, 0, theme.INSET_WIDTH)))
        return Group(*parts)

    def _edit_preview_diff(self, lines: list) -> Text:
        bar = f"{theme.g(Glyph.INSET_BAR)} "
        styles = {"add": Color.SUCCESS, "del": Color.ERROR, "meta": "dim", "context": "dim"}
        rendered = Text()
        for index, row in enumerate(lines):
            if isinstance(row, (list, tuple)) and len(row) >= 2:
                tag, text = str(row[0]), str(row[1])
            else:
                tag, text = "context", str(row)
            if index:
                rendered.append("\n")
            rendered.append(bar, style="dim")
            rendered.append(text, style=styles.get(tag))
        return rendered

    def _edit_preview_code(self, code: str, language: str):
        try:
            from rich.syntax import Syntax

            return Syntax(
                code,
                language or "text",
                theme=theme.markdown_code_theme(),
                background_color="default",
                word_wrap=False,
            )
        except Exception:
            # Unknown lexer / pathological content: fall back to plain inset text.
            bar = f"{theme.g(Glyph.INSET_BAR)} "
            rendered = Text()
            for index, line in enumerate(code.split("\n")):
                if index:
                    rendered.append("\n")
                rendered.append(bar, style="dim")
                rendered.append(line)
            return rendered

    def _format_inset_text(self, content: str, style: Optional[str] = None) -> Text:
        bar = f"  {theme.g(Glyph.INSET_BAR)}"
        lines = content.splitlines() or [""]
        rendered = Text()
        for index, line in enumerate(lines):
            if index:
                rendered.append("\n")
            rendered.append(bar, style="dim")
            if line:
                rendered.append(" ")
                rendered.append(line, style=style)
        return rendered
