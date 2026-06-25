"""History compression: summarize a conversation when it outgrows the context window."""

import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from .conversation import Conversation
from kolega_code.llm.models import Message, MessageHistory, TextBlock
from .prompts import (
    build_compression_summary_user_prompt,
    COMPRESSION_SUMMARY_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

LogCallback = Callable[[str], Awaitable[None]]


@dataclass(frozen=True)
class CompactionResult:
    """Outcome of a compaction attempt, surfaced to callers and the UI.

    ``reason`` is a machine tag: "ok" | "too_few" | "nothing_to_summarize" | "llm_error".
    ``message`` is a human-readable line for the command output / logs.
    """

    ok: bool
    reason: str
    summarized_messages: int = 0
    message: str = ""


class HistoryCompressor:
    """Summarizes a conversation non-destructively when it crosses the budget threshold."""

    MIN_MESSAGES_TO_COMPRESS = 5
    # How many of the most recent messages to keep verbatim after the summary.
    KEEP_RECENT_MESSAGES = 6
    # Don't bother summarizing a trivially short prefix.
    MIN_PREFIX_TO_SUMMARIZE = 3
    # Cap the summary length: the prompt targets ~600 words (~900 tokens), so a
    # small ceiling keeps it tight and avoids the model's full completion budget.
    SUMMARY_MAX_TOKENS = 2048

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
        temperature: float,
        thinking,
        on_info: Optional[LogCallback] = None,
        on_error: Optional[LogCallback] = None,
        system_prompt_text: Optional[str] = None,
    ) -> CompactionResult:
        """
        Non-destructively summarize the aged-out prefix while keeping the most
        recent turns verbatim, and move the compaction boundary.

        Incremental: only the messages that have aged out since the previous
        boundary are summarized, with the prior summary folded in for continuity.
        Returns a CompactionResult describing what happened — never a silent no-op.
        """
        history = conversation.history
        if not history or len(history) < self.MIN_MESSAGES_TO_COMPRESS:
            return CompactionResult(
                ok=False,
                reason="too_few",
                message=(
                    f"Nothing to compress yet ({len(history)} message(s); "
                    f"need at least {self.MIN_MESSAGES_TO_COMPRESS})."
                ),
            )

        split = conversation.compaction_split_point(
            keep_recent=self.KEEP_RECENT_MESSAGES, min_prefix=self.MIN_PREFIX_TO_SUMMARIZE
        )
        prior_through = conversation.compacted_through if conversation.summary is not None else 0
        if split is None or split <= prior_through:
            return CompactionResult(
                ok=False,
                reason="nothing_to_summarize",
                message="Already compact — no older messages to summarize.",
            )

        if on_info:
            await on_info("Compressing message history...")

        try:
            prefix = MessageHistory(list(history[prior_through:split]))
            prefix_markdown = prefix.get_markdown_conversation()
            previous_summary = conversation.summary.get_text_content() if conversation.summary is not None else None
            user_prompt_filled = build_compression_summary_user_prompt(
                prefix_markdown, previous_summary=previous_summary
            )

            messages = MessageHistory([Message(role="user", content=[TextBlock(text=user_prompt_filled)])])
            system_text = system_prompt_text or COMPRESSION_SUMMARY_SYSTEM_PROMPT
            system_message = Message(role="system", content=[TextBlock(text=system_text)])

            # Stream and drain rather than calling generate(): the Anthropic SDK
            # rejects non-streaming requests whose max_tokens is large enough to risk
            # a >10-minute response, which the model's full completion budget triggers.
            async with await llm.stream(
                messages=messages,
                system=system_message,
                temperature=temperature,
                model=model,
                max_completion_tokens=self.SUMMARY_MAX_TOKENS,
                thinking=thinking,
            ) as stream:
                async for _event in stream:
                    pass
            response = await stream.get_final_message()

            summary_text = response.get_text_content()
            if not summary_text or not summary_text.strip():
                msg = "Compression produced an empty summary; history left unchanged."
                if on_error:
                    await on_error(msg)
                else:
                    logger.error(msg)
                return CompactionResult(ok=False, reason="llm_error", message=msg)

            folded = split - prior_through
            conversation.apply_compaction(summary_text, split)

            done = f"Compressed {folded} older message(s) into a summary; kept the latest turns verbatim."
            if on_info:
                await on_info(done)
            return CompactionResult(ok=True, reason="ok", summarized_messages=folded, message=done)

        except Exception as e:
            # Never swallow into a fake success: log the full traceback and surface
            # the real error to the caller.
            logger.exception("Failed to compress message history")
            msg = f"Failed to compress message history: {type(e).__name__}: {e}"
            if on_error:
                await on_error(msg)
            return CompactionResult(ok=False, reason="llm_error", message=msg)
