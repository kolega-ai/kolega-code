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

from .llm.models import Message, MessageHistory, TextBlock, ToolCall, ToolResult

logger = logging.getLogger(__name__)


class Conversation:
    """Message history plus the invariants that keep it valid for providers."""

    skill_content_pattern = re.compile(r'<skill_content name="[^"]+">')

    def __init__(
        self,
        messages: Optional[List[Message]] = None,
        *,
        max_tool_result_chars: int = 100_000,
    ) -> None:
        self.history = MessageHistory(list(messages) if messages else [])
        # Compression marker: index of the last message before a summary was appended
        self.last_compression_index: Optional[int] = None
        self.max_tool_result_chars = max_tool_result_chars

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
        self.history = MessageHistory(self.repaired(all_messages))

    # ------------------------------------------------------------------
    # Views and validity
    # ------------------------------------------------------------------

    def effective_history(self) -> MessageHistory:
        """
        Return the subset of history to send to the LLM:
        - If compressed: protected skill content + summary + all messages after the summary
        - Else: the full history
        """
        if self.last_compression_index is not None and self.history:
            # Summary is the message immediately after the boundary
            summary_idx = self.last_compression_index + 1
            if summary_idx < len(self.history):
                summary_msg = self.history[summary_idx]
                protected = [message for message in self.history[:summary_idx] if self.is_protected(message)]
                # Tail starts after the summary
                tail = list(self.history[summary_idx + 1 :]) if summary_idx + 1 < len(self.history) else []
                return MessageHistory(protected + [summary_msg] + tail)

        return MessageHistory(list(self.history))

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
        """Append a compression summary and mark the boundary before it."""
        self.history.append(summary)
        self.last_compression_index = len(self.history) - 2

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
