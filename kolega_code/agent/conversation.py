"""Conversation: owns the message history and its invariants.

Centralizes the rules that keep a conversation valid for LLM providers:
tool-call/tool-result pairing, cache checkpoints, the compression boundary,
and oversized-tool-result sanitization. BaseAgent delegates here; host
subclasses should reach this through the BaseAgent methods rather than
holding their own reference.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from kolega_code.llm.models import (
    ImageBlock,
    Message,
    MessageHistory,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResult,
)

logger = logging.getLogger(__name__)


def _image_placeholder(media_type: str, model_name: str) -> str:
    """Text substituted for an image block when the active model can't see images."""
    return f"[An image ({media_type}) was shared earlier in this thread but is not visible to {model_name}.]"


def count_image_blocks(messages: List[Message]) -> int:
    """Count ``ImageBlock`` instances across messages, including ones nested in tool results."""
    count = 0
    for message in messages:
        if not isinstance(message.content, list):
            continue
        for block in message.content:
            if isinstance(block, ImageBlock):
                count += 1
            elif isinstance(block, ToolResult) and isinstance(block.content, list):
                count += sum(1 for inner in block.content if isinstance(inner, ImageBlock))
    return count


def _provider_value(provider: Any) -> str:
    value = getattr(provider, "value", provider)
    return str(value or "")


def _message_with_content(message: Message, content: List[Any]) -> Message:
    return Message(
        role=message.role,
        content=content,
        stop_reason=message.stop_reason,
        tool_calls=message.tool_calls,
        usage_metadata=message.usage_metadata,
    )


def _tool_result_with_content(tool_result: ToolResult, content: List[Any]) -> ToolResult:
    return ToolResult(
        tool_use_id=tool_result.tool_use_id,
        content=content,
        name=tool_result.name,
        is_error=tool_result.is_error,
        cache_checkpoint=tool_result.cache_checkpoint,
        execution_id=tool_result.execution_id,
    )


def replace_image_blocks_with_placeholders(messages: List[Message], model_name: str) -> List[Message]:
    """Return a new message list with every ``ImageBlock`` replaced by a text placeholder.

    Non-mutating: messages without images are returned as-is. When a replacement
    occurs, new ``Message``/``ToolResult`` objects are built so the caller's stored
    history is never altered — switching back to a vision-capable model restores
    the original images. ``ImageBlock``s nested inside a ``ToolResult.content`` list
    (e.g. from the ``read_image`` tool) are handled too.
    """
    result: List[Message] = []
    for message in messages:
        if not isinstance(message.content, list):
            result.append(message)
            continue
        new_blocks: List[Any] = []
        changed = False
        for block in message.content:
            if isinstance(block, ImageBlock):
                new_blocks.append(TextBlock(text=_image_placeholder(block.media_type, model_name)))
                changed = True
            elif isinstance(block, ToolResult) and isinstance(block.content, list):
                inner_new: List[Any] = []
                inner_changed = False
                for inner in block.content:
                    if isinstance(inner, ImageBlock):
                        inner_new.append(TextBlock(text=_image_placeholder(inner.media_type, model_name)))
                        inner_changed = True
                    else:
                        inner_new.append(inner)
                if inner_changed:
                    new_blocks.append(_tool_result_with_content(block, inner_new))
                    changed = True
                else:
                    new_blocks.append(block)
            else:
                new_blocks.append(block)
        if changed:
            result.append(_message_with_content(message, new_blocks))
        else:
            result.append(message)
    return result


def _preserve_reasoning_block(block: Any, *, source_provider: str, target_provider: str) -> bool:
    return bool(source_provider) and source_provider == target_provider


def _reasoning_placeholder(block: Any, source_provider: str) -> TextBlock:
    source = source_provider or "unknown provider"
    if isinstance(block, RedactedThinkingBlock):
        return TextBlock(text=f"[Prior redacted reasoning from {source} omitted for compatibility.]")
    return TextBlock(text=f"[Prior reasoning from {source} omitted for compatibility.]")


def _adapt_content_blocks_for_provider(
    blocks: List[Any], *, source_provider: str, target_provider: str, target_model: str, supports_vision: bool
) -> tuple[List[Any], bool]:
    adapted: List[Any] = []
    changed = False

    for block in blocks:
        if isinstance(block, ImageBlock) and not supports_vision:
            adapted.append(TextBlock(text=_image_placeholder(block.media_type, target_model)))
            changed = True
        elif isinstance(block, (ThinkingBlock, RedactedThinkingBlock)):
            if _preserve_reasoning_block(block, source_provider=source_provider, target_provider=target_provider):
                adapted.append(block)
            else:
                adapted.append(_reasoning_placeholder(block, source_provider))
                changed = True
        elif isinstance(block, ToolResult) and isinstance(block.content, list):
            inner, inner_changed = _adapt_content_blocks_for_provider(
                block.content,
                source_provider=source_provider,
                target_provider=target_provider,
                target_model=target_model,
                supports_vision=supports_vision,
            )
            if inner_changed:
                adapted.append(_tool_result_with_content(block, inner))
                changed = True
            else:
                adapted.append(block)
        else:
            adapted.append(block)

    return adapted, changed


def adapt_history_for_provider(
    messages: List[Message], *, target_provider: str, target_model: str, supports_vision: bool
) -> List[Message]:
    """Return a request-safe history for the target provider without mutating storage.

    Provider-managed reasoning blocks are not portable across providers, but
    reasoning produced by a provider is safe to send back to the same provider.
    Preserve same-provider reasoning; convert foreign or unknown reasoning
    blocks to text placeholders. Image blocks are preserved for vision models
    and replaced with placeholders for non-vision models, matching the previous
    image compatibility behavior.
    """
    target_provider = _provider_value(target_provider)
    result: List[Message] = []
    changed_any = False

    for message in messages:
        if not isinstance(message.content, list):
            result.append(message)
            continue

        source_provider = _provider_value((message.usage_metadata or {}).get("provider"))
        adapted_content, changed = _adapt_content_blocks_for_provider(
            message.content,
            source_provider=source_provider,
            target_provider=target_provider,
            target_model=target_model,
            supports_vision=supports_vision,
        )
        if changed:
            result.append(_message_with_content(message, adapted_content))
            changed_any = True
        else:
            result.append(message)

    return result if changed_any else messages


class Conversation:
    """Message history plus the invariants that keep it valid for providers."""

    skill_content_pattern = re.compile(r'<skill_content name="[^"]+">')

    def __init__(
        self,
        messages: Optional[List[Message]] = None,
        *,
        max_tool_result_chars: int = 100_000,
    ) -> None:
        self._history = MessageHistory(list(messages) if messages else [])
        # Compaction state: ``summary`` stands in for ``history[:compacted_through]``
        # in the effective view; messages from ``compacted_through`` onward are kept
        # verbatim. ``summary is None`` / ``compacted_through == 0`` means uncompacted.
        # ``compacted_history_length`` is the history length captured when compaction
        # ran, so the UI can place the summary after the retained tail on restore.
        self.summary: Optional[Message] = None
        self.compacted_through: int = 0
        self.compacted_history_length: int = 0
        self.max_tool_result_chars = max_tool_result_chars

    # ------------------------------------------------------------------
    # History access (compaction-aware)
    # ------------------------------------------------------------------

    @property
    def history(self) -> MessageHistory:
        return self._history

    @history.setter
    def history(self, value) -> None:
        # Wholesale replacement of the log invalidates any compaction boundary.
        # Internal mutators that must preserve an active summary assign to
        # ``self._history`` directly instead of going through this setter.
        self._history = value if isinstance(value, MessageHistory) else MessageHistory(list(value))
        self.summary = None
        self.compacted_through = 0
        self.compacted_history_length = 0

    @property
    def last_compression_index(self) -> Optional[int]:
        """Back-compat: index of the last message folded into the summary, or None."""
        if self.summary is not None and self.compacted_through > 0:
            return self.compacted_through - 1
        return None

    @last_compression_index.setter
    def last_compression_index(self, value: Optional[int]) -> None:
        # Legacy setter. The only meaningful operation is clearing the boundary;
        # a real summary is recorded through ``apply_compaction``/``record_compression``.
        if value is None:
            self.summary = None
            self.compacted_through = 0
            self.compacted_history_length = 0
        else:
            self.compacted_through = value + 1

    def clear(self) -> None:
        """Drop all history and reset compaction state."""
        self._history = MessageHistory([])
        self.summary = None
        self.compacted_through = 0
        self.compacted_history_length = 0

    def has_image_blocks(self) -> bool:
        """True if the effective (compaction-aware) history still carries any images.

        Images folded into a compaction summary are already gone, so this reflects
        only what would actually be sent to the model.
        """
        return count_image_blocks(list(self.effective_history())) > 0

    # ------------------------------------------------------------------
    # Appending
    # ------------------------------------------------------------------

    def append_user(self, content) -> None:
        """
        Safely append a user message, reconciling incoming tool results with
        any placeholder or duplicate results already in history.

        Args:
            content: Either a string (converted to TextBlock) or list of ContentBlocks
        """
        if isinstance(content, str):
            content_blocks = [TextBlock(text=content)]
        elif isinstance(content, list):
            content_blocks = content
        else:
            content_blocks = [content]

        if isinstance(content_blocks, list):
            new_tool_results = {}
            other_blocks = []

            for block in content_blocks:
                if isinstance(block, ToolResult):
                    new_tool_results[block.tool_use_id] = block
                else:
                    other_blocks.append(block)

            if new_tool_results:
                # Find and update any existing tool results with the same IDs
                for i, msg in enumerate(self.history):
                    if msg.role == "user" and isinstance(msg.content, list):
                        updated_content = []
                        replaced_any = False

                        for block in msg.content:
                            if isinstance(block, ToolResult) and block.tool_use_id in new_tool_results:
                                new_result = new_tool_results[block.tool_use_id]
                                # Replace if: old is dummy error OR new is success and old is error
                                should_replace = (block.is_error and "Operation was interrupted" in block.content) or (
                                    not new_result.is_error and block.is_error
                                )

                                if should_replace:
                                    updated_content.append(new_result)
                                    replaced_any = True
                                    logger.debug("Replaced tool result for tool_use_id: %s", block.tool_use_id)
                                    del new_tool_results[block.tool_use_id]
                                else:
                                    updated_content.append(block)
                                    if block.tool_use_id in new_tool_results:
                                        logger.debug(
                                            "Skipping duplicate tool result for tool_use_id: %s", block.tool_use_id
                                        )
                                        del new_tool_results[block.tool_use_id]
                            else:
                                updated_content.append(block)

                        if replaced_any:
                            self.history[i] = Message(
                                role=msg.role, content=updated_content, stop_reason=msg.stop_reason
                            )

                # Add any remaining new tool results along with other blocks
                content_blocks = list(new_tool_results.values()) + other_blocks

                # If all blocks were handled (replaced or skipped), don't add empty message
                if not content_blocks:
                    return

        if not content_blocks or (isinstance(content_blocks, list) and len(content_blocks) == 0):
            logger.warning("User message has empty content, replacing with placeholder")
            content_blocks = [TextBlock(text="[User provided no message content]")]

        self.history.append(Message(role="user", content=content_blocks))

    def append_assistant(self, message: Message) -> None:
        """Safely append an assistant message, replacing empty content with a placeholder."""
        if not message.content or (isinstance(message.content, list) and len(message.content) == 0):
            logger.warning("Assistant message has empty content, replacing with placeholder")
            message = Message(
                role=message.role,
                content=[TextBlock(text="[Assistant returned no message content]")],
                stop_reason=message.stop_reason,
                tool_calls=message.tool_calls,
            )

        self.history.append(message)

    def extend(self, messages: List[Message]) -> None:
        """Extend history with multiple messages, repairing incomplete tool calls in the result."""
        all_messages = list(self.history) + messages
        # Assign to _history (not the public setter) so an active compaction survives
        # appending live turns.
        self._history = MessageHistory(self.repaired(all_messages))

    # ------------------------------------------------------------------
    # Views and validity
    # ------------------------------------------------------------------

    def effective_history(self) -> MessageHistory:
        """
        Return the subset of history to send to the LLM:
        - If compacted: protected skill content from the folded prefix, then the
          summary, then the most-recent turns kept verbatim.
        - Else: the full history.
        """
        if self.summary is not None and self.compacted_through > 0 and self._history:
            cut = min(self.compacted_through, len(self._history))
            protected = [message for message in self._history[:cut] if self.is_protected(message)]
            tail = list(self._history[cut:])
            return MessageHistory(protected + [self.summary] + tail)

        return MessageHistory(list(self._history))

    def is_protected(self, message: Message) -> bool:
        """True for user messages carrying skill content that must survive compression."""
        if message.role != "user":
            return False
        if isinstance(message.content, str):
            return bool(self.skill_content_pattern.search(message.content))
        if not isinstance(message.content, list):
            return False
        return any(
            isinstance(block, TextBlock) and self.skill_content_pattern.search(block.text)
            for block in message.content
        )

    def is_valid_for_anthropic(self, messages: Optional[List[Message]] = None) -> bool:
        """
        Check that every tool_use block is followed by a matching tool_result block,
        as the Anthropic API requires.
        """
        if messages is None:
            messages = list(self.history)

        for i, message in enumerate(messages):
            if message.role == "assistant" and isinstance(message.content, list):
                tool_calls = [block for block in message.content if isinstance(block, ToolCall)]

                if tool_calls:
                    if i + 1 >= len(messages):
                        return False  # No next message

                    next_message = messages[i + 1]
                    if next_message.role != "user":
                        return False  # Next message should be user role

                    if not isinstance(next_message.content, list):
                        return False  # Should contain list of tool results

                    tool_call_ids = {call.id for call in tool_calls}
                    tool_result_ids = {
                        block.tool_use_id for block in next_message.content if isinstance(block, ToolResult)
                    }

                    if not tool_call_ids.issubset(tool_result_ids):
                        return False  # Missing tool results

        return True

    def needs_tool_call_fix(self) -> bool:
        """True if the last message is an assistant message with pending tool calls."""
        if not self.history:
            return False

        last_message = self.history[-1]

        if last_message.role != "assistant":
            return False

        if not isinstance(last_message.content, list):
            return False

        return any(isinstance(block, ToolCall) for block in last_message.content)

    def repaired(self, messages: Optional[List[Message]] = None) -> List[Message]:
        """
        Repair incomplete tool call sequences by reuniting displaced tool results
        with their tool calls and adding placeholder results for orphaned calls.

        Args:
            messages: Messages to repair; defaults to the current history.

        Returns:
            List[Message]: Repaired messages safe to send to a provider.
        """
        if messages is None:
            messages = list(self.history)
        if not messages:
            return messages

        fixed_messages = []
        i = 0
        processed_indices = set()  # Track which messages we've already processed

        while i < len(messages):
            if i in processed_indices:
                i += 1
                continue

            current_message = messages[i]

            if current_message.role == "assistant" and isinstance(current_message.content, list):
                tool_calls = [block for block in current_message.content if isinstance(block, ToolCall)]

                if tool_calls:
                    fixed_messages.append(current_message)
                    processed_indices.add(i)

                    # Collect all tool results from the entire remaining conversation
                    tool_call_ids = {call.id for call in tool_calls}
                    all_tool_results = {}
                    other_content_blocks = []  # Non-tool-result content from the next user message

                    # First, check the immediately following message (expected position)
                    next_user_message = None
                    if i + 1 < len(messages) and messages[i + 1].role == "user":
                        next_user_message = messages[i + 1]
                        if isinstance(next_user_message.content, list):
                            for block in next_user_message.content:
                                if isinstance(block, ToolResult) and block.tool_use_id in tool_call_ids:
                                    all_tool_results[block.tool_use_id] = block
                                else:
                                    other_content_blocks.append(block)

                            if all_tool_results:
                                processed_indices.add(i + 1)

                    # Search the entire remaining conversation for any missing tool results
                    missing_ids = tool_call_ids - set(all_tool_results.keys())
                    if missing_ids:
                        for j in range(i + 1, len(messages)):
                            if j in processed_indices:
                                continue

                            msg = messages[j]
                            if msg.role == "user" and isinstance(msg.content, list):
                                remaining_content = []
                                found_any = False

                                for block in msg.content:
                                    if isinstance(block, ToolResult) and block.tool_use_id in missing_ids:
                                        logger.warning(
                                            "Found tool result %s at position %s instead of expected position %s",
                                            block.tool_use_id,
                                            j,
                                            i + 1,
                                        )
                                        all_tool_results[block.tool_use_id] = block
                                        missing_ids.remove(block.tool_use_id)
                                        found_any = True
                                    else:
                                        remaining_content.append(block)

                                if found_any:
                                    if remaining_content:
                                        # Message has other content - keep it but remove tool results
                                        updated_msg = Message(
                                            role=msg.role, content=remaining_content, stop_reason=msg.stop_reason
                                        )
                                        messages[j] = updated_msg
                                    else:
                                        # Message only had tool results - mark for skipping
                                        processed_indices.add(j)

                    # Create the complete tool results list in the correct order
                    complete_tool_results = []
                    for tool_call in tool_calls:
                        if tool_call.id in all_tool_results:
                            complete_tool_results.append(all_tool_results[tool_call.id])
                        else:
                            logger.warning("Adding placeholder result for missing tool call: %s", tool_call.id)
                            complete_tool_results.append(
                                ToolResult(
                                    tool_use_id=tool_call.id,
                                    content="Operation was interrupted. Please retry if needed.",
                                    name=tool_call.name,
                                    is_error=True,
                                )
                            )

                    # Create the user message with all tool results, plus any other
                    # content that was in the original next user message
                    all_content = complete_tool_results + other_content_blocks
                    if all_content:
                        complete_user_message = Message(
                            role="user",
                            content=all_content,
                            stop_reason=next_user_message.stop_reason if next_user_message else None,
                        )
                        fixed_messages.append(complete_user_message)

                    i += 1
                else:
                    fixed_messages.append(current_message)
                    processed_indices.add(i)
                    i += 1
            else:
                # Not an assistant message with tool calls
                # Skip if already processed (was a tool result message we moved)
                if i not in processed_indices:
                    fixed_messages.append(current_message)
                i += 1

        return fixed_messages

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def mark_cache_checkpoint(self) -> None:
        """
        Mark only the last content block of the last message for prompt caching,
        clearing the marker everywhere else (Anthropic allows max 4 cache blocks).
        """
        for message in self.history:
            if hasattr(message, "content") and isinstance(message.content, list):
                for content_block in message.content:
                    if hasattr(content_block, "cache_checkpoint"):
                        content_block.cache_checkpoint = False

        if self.history:
            last_message = self.history[-1]
            if hasattr(last_message, "content") and isinstance(last_message.content, list) and last_message.content:
                last_content_block = last_message.content[-1]
                if hasattr(last_content_block, "cache_checkpoint"):
                    last_content_block.cache_checkpoint = True

    def sanitize_oversized_tool_results(self) -> int:
        """Replace tool results above the size cap with an explanatory placeholder."""
        sanitized_count = 0
        for message in self.history:
            if not isinstance(message.content, list):
                continue

            for block in message.content:
                if not isinstance(block, ToolResult) or not isinstance(block.content, str):
                    continue

                content_length = len(block.content)
                if content_length <= self.max_tool_result_chars:
                    continue

                block.content = (
                    f"[Tool result omitted from history because it was {content_length:,} characters, "
                    f"exceeding the {self.max_tool_result_chars:,} character safety cap. "
                    f"Re-run `{block.name}` with narrower inputs if the content is still needed.]"
                )
                sanitized_count += 1

        return sanitized_count

    def record_compression(self, summary: Message) -> None:
        """Back-compat shim: fold the entire current history into ``summary``.

        Newer callers use ``apply_compaction`` with an explicit split point so the
        most recent turns are kept verbatim.
        """
        self.summary = (
            summary if isinstance(summary, Message) else Message(role="user", content=[TextBlock(text=str(summary))])
        )
        self.compacted_through = len(self._history)
        self.compacted_history_length = len(self._history)

    def apply_compaction(self, summary_text: str, split_point: int) -> None:
        """Record ``summary_text`` as standing in for ``history[:split_point]``.

        Non-destructive: the full history is untouched; only the compaction
        boundary moves. Messages from ``split_point`` onward stay verbatim.
        """
        self.summary = Message(role="user", content=[TextBlock(text=summary_text)])
        self.compacted_through = max(0, min(split_point, len(self._history)))
        self.compacted_history_length = len(self._history)

    def compaction_split_point(self, *, keep_recent: int, min_prefix: int) -> Optional[int]:
        """Index where the verbatim recent tail should begin.

        Keeps the last ``keep_recent`` messages verbatim, then snaps the cut
        backward so it never lands between an assistant tool_use and its
        tool_result. Returns None when the prefix to summarize would be smaller
        than ``min_prefix``.
        """
        n = len(self._history)
        if n <= min_prefix:
            return None
        candidate = self._snap_to_safe_boundary(max(0, n - keep_recent))
        if candidate < min_prefix:
            return None
        return candidate

    def _snap_to_safe_boundary(self, idx: int) -> int:
        """Move ``idx`` backward past any assistant tool_use so a tool_use/tool_result
        group is never split across the compaction boundary."""
        while idx > 0:
            prev = self._history[idx - 1]
            if (
                prev.role == "assistant"
                and isinstance(prev.content, list)
                and any(isinstance(block, ToolCall) for block in prev.content)
            ):
                idx -= 1
                continue
            break
        return idx

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def dump(self) -> List[Dict[str, Any]]:
        """Serialize the message history into a list of dictionaries."""
        return [message.to_dict() for message in self.history]

    def restore(self, serialized_history: List[Dict[str, Any]]) -> None:
        """Restore the message history from a list of dictionaries."""
        parsed_messages = [Message.from_dict(item) for item in serialized_history]
        # Keep history authentic - no fixing here
        self.history = MessageHistory(parsed_messages)
        self.sanitize_oversized_tool_results()

    def dump_compaction(self) -> Dict[str, Any]:
        """Serialize the compaction boundary so it survives save/restore."""
        return {
            "summary": self.summary.get_text_content() if self.summary is not None else "",
            "compacted_through": self.compacted_through,
            "compacted_history_length": self.compacted_history_length,
        }

    def restore_compaction(self, data: Optional[Dict[str, Any]]) -> None:
        """Restore a compaction boundary saved by ``dump_compaction``.

        Call AFTER ``restore`` (which resets compaction). Assigns the boundary
        fields directly so it does not trip the history setter's reset. Old
        sessions without ``compacted_history_length`` fall back to 0.
        """
        data = data or {}
        text = (data.get("summary") or "").strip()
        through = int(data.get("compacted_through") or 0)
        if text and through > 0:
            self.summary = Message(role="user", content=[TextBlock(text=text)])
            self.compacted_through = min(through, len(self._history))
            self.compacted_history_length = int(data.get("compacted_history_length") or 0)
        else:
            self.summary = None
            self.compacted_through = 0
            self.compacted_history_length = 0
