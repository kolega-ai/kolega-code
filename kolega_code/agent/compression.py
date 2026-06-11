"""History compression: summarize a conversation when it outgrows the context window."""

import logging
from typing import Awaitable, Callable, Optional

from .conversation import Conversation
from kolega_code.llm.models import Message, MessageHistory, TextBlock
from .prompts import (
    COMPRESSION_SUMMARY_SYSTEM_PROMPT,
    COMPRESSION_SUMMARY_USER_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)

LogCallback = Callable[[str], Awaitable[None]]


class HistoryCompressor:
    """Summarizes a conversation non-destructively when it crosses the budget threshold."""

    MIN_MESSAGES_TO_COMPRESS = 5

    def __init__(self, threshold: float = 0.8) -> None:
        # Fraction of the model context window above which compression kicks in
        self.threshold = threshold

    def over_budget(self, input_tokens: int, model_context_length: int) -> bool:
        return input_tokens > model_context_length * self.threshold

    async def summarize(
        self,
        conversation: Conversation,
        *,
        llm,
        model: str,
        max_completion_tokens: int,
        temperature: float,
        thinking,
        on_info: Optional[LogCallback] = None,
        on_error: Optional[LogCallback] = None,
    ) -> bool:
        """
        Non-destructively summarize the conversation and mark a compression boundary.

        Returns True if a summary was recorded.
        """
        if not conversation.history or len(conversation.history) < self.MIN_MESSAGES_TO_COMPRESS:
            return False

        if on_info:
            await on_info("Compressing message history...")

        try:
            conversation_markdown = conversation.history.get_markdown_conversation()
            user_prompt_filled = COMPRESSION_SUMMARY_USER_PROMPT_TEMPLATE.replace("{HISTORY}", conversation_markdown)

            messages = MessageHistory([Message(role="user", content=[TextBlock(text=user_prompt_filled)])])
            system_message = Message(role="system", content=[TextBlock(text=COMPRESSION_SUMMARY_SYSTEM_PROMPT)])

            response = await llm.generate(
                messages=messages,
                system=system_message,
                temperature=temperature,
                model=model,
                max_completion_tokens=max_completion_tokens,
                thinking=thinking,
            )

            summary = response.get_text_content()
            conversation.record_compression(Message(role="user", content=[TextBlock(text=summary)]))

            if on_info:
                await on_info("Message history compressed (non-destructive).")
            return True

        except Exception as e:
            if on_error:
                await on_error(f"Failed to compress message history: {str(e)}")
            else:
                logger.error("Failed to compress message history: %s", e)
            return False
