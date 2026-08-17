"""History compression: summarize a conversation when it outgrows the context window."""

import json
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, List, Optional

from .conversation import Conversation, replace_image_blocks_with_placeholders
from kolega_code.llm.exceptions import LLMContextWindowExceededError
from kolega_code.llm.ledger import helper_origin, llm_call_origin
from kolega_code.llm.models import Message, MessageHistory, TextBlock, ToolCall, ToolResult
from .prompts import (
    build_compression_summary_user_prompt,
    COMPRESSION_SUMMARY_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

LogCallback = Callable[[str], Awaitable[None]]


@dataclass(frozen=True)
class CompactionResult:
    """Outcome of a compaction attempt, surfaced to callers and the UI.

    ``reason`` is a machine tag: "ok" | "too_few" | "nothing_to_summarize" | "empty_summary" | "llm_error".
    ``message`` is a human-readable line for the command output / logs.
    """

    ok: bool
    reason: str
    summarized_messages: int = 0
    message: str = ""


def _render_transcript(messages: List[Message]) -> str:
    """Role-tagged plain text: JSON only for tool arguments, tool results raw.

    JSON-escaping newline-heavy tool output costs ~30% more tokens than
    leaving it raw.
    """
    out: List[str] = []
    for message in messages:
        if isinstance(message.content, str):
            if message.content:
                out.append(f"[{message.role}]\n{message.content}")
            continue
        for block in message.content:
            if isinstance(block, ToolCall):
                try:
                    args = json.dumps(block.input, ensure_ascii=False)
                except (TypeError, ValueError):
                    args = str(block.input)
                out.append(f"[assistant → {block.name}] {args}")
            elif isinstance(block, ToolResult):
                if isinstance(block.content, str):
                    body = block.content
                else:
                    body = "\n".join(getattr(item, "text", "") or "" for item in block.content)
                tag = "error" if block.is_error else "result"
                out.append(f"[{block.name} {tag}]\n{body}")
            else:
                text = getattr(block, "text", None)
                thinking = getattr(block, "thinking", None)
                if text:
                    out.append(f"[{message.role}]\n{text}")
                elif thinking:
                    out.append(f"[{message.role} thinking]\n{thinking}")
    return "\n\n".join(out)


def _omission_notice(omitted: int) -> str:
    return f"[... {omitted} conversation message(s) omitted here so the compaction request fits the context budget ...]"


def _approx_message_chars(message: Message) -> int:
    """Size proxy for picking the omission gap; the request is re-counted before dispatch."""
    if isinstance(message.content, str):
        return len(message.content)
    chars = 0
    for block in message.content:
        if isinstance(block, ToolCall):
            try:
                chars += len(json.dumps(block.input))
            except (TypeError, ValueError):
                chars += len(str(block.input))
        elif isinstance(block, ToolResult):
            if isinstance(block.content, str):
                chars += len(block.content)
            else:
                chars += sum(len(getattr(item, "text", "") or "") for item in block.content)
        else:
            chars += len(getattr(block, "text", "") or "") + len(getattr(block, "thinking", "") or "")
    return chars


def _middle_gap(weights: List[int], target_chars: float) -> tuple[int, int]:
    """Pick the (head_end, tail_start) gap keeping ~``target_chars``: half from the
    oldest messages (task statement), half from the newest (continuity)."""
    half = target_chars / 2
    head_end, acc = 0, 0
    for index, weight in enumerate(weights):
        if acc + weight > half:
            break
        acc += weight
        head_end = index + 1
    tail_start, acc = len(weights), 0
    for index in range(len(weights) - 1, head_end - 1, -1):
        if acc + weights[index] > half:
            break
        acc += weights[index]
        tail_start = index
    return head_end, tail_start


def _bundles_tool_results(message: Message) -> bool:
    return isinstance(message.content, list) and any(isinstance(block, ToolResult) for block in message.content)


async def _stream_summary(
    llm,
    messages: MessageHistory,
    system: Message,
    temperature: float,
    model: str,
    thinking,
) -> Message:
    """Issue the compaction helper request and return the completed message.

    Stream and consume events rather than calling ``generate()``: the Anthropic
    SDK rejects non-streaming requests whose max_tokens is large enough to risk
    a >10-minute response. The helper origin spans the whole request because
    providers may sample and emit their trace record at context entry or
    final-message time, not at ``stream()``.
    """
    with llm_call_origin(helper_origin("compression")):
        stream_cm = await llm.stream(
            messages=messages,
            system=system,
            temperature=temperature,
            model=model,
            max_completion_tokens=HistoryCompressor.SUMMARY_MAX_TOKENS,
            thinking=thinking,
        )
        async with stream_cm as stream:
            async for _event in stream:
                pass
        return await stream.get_final_message()


class HistoryCompressor:
    """Summarizes a conversation non-destructively when it crosses the budget threshold."""

    MIN_MESSAGES_TO_COMPRESS = 5
    # How many of the most recent messages to keep verbatim after the summary.
    KEEP_RECENT_MESSAGES = 6
    # Don't bother summarizing a trivially short prefix.
    MIN_PREFIX_TO_SUMMARIZE = 3
    # Cap the summary length. The prompt targets ~600 words, but reasoning
    # models (deepseek-v4-flash Responses, etc.) can consume the entire budget
    # on chain-of-thought and leave no room for text output, producing an empty
    # summary. A generous ceiling gives reasoning headroom while still staying
    # well under the model's full completion budget.
    SUMMARY_MAX_TOKENS = 8192
    # Completed-but-empty summaries (reasoning-only, whitespace, no text) are
    # retried on the same fitted request before the pass is reported unusable.
    SUMMARY_EMPTY_ATTEMPTS = 3
    # Attempts at widening the omission gap per budget rung.
    MAX_FIT_ATTEMPTS = 6

    def __init__(self, threshold: float = 0.8) -> None:
        # Fraction of the model context window above which compression kicks in
        self.threshold = threshold

    def over_budget(self, input_tokens: int, input_budget_tokens: int) -> bool:
        """True when input exceeds the threshold fraction of the input budget."""
        return input_tokens > input_budget_tokens * self.threshold

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
        keep_recent: Optional[int] = None,
        max_input_tokens: Optional[int] = None,
    ) -> CompactionResult:
        """
        Non-destructively summarize the aged-out prefix while keeping the most
        recent turns verbatim, and move the compaction boundary.

        Incremental: only the messages that have aged out since the previous
        boundary are summarized, with the prior summary folded in for continuity.
        The aged-out span is presented to the summarizer as a role-tagged
        plain-text transcript (tool arguments as JSON, tool results raw).
        Returns a CompactionResult describing what happened — never a silent no-op.

        ``keep_recent`` overrides KEEP_RECENT_MESSAGES (the verbatim tail). A
        literal ``0`` is the aggressive fallback: everything before the safe
        boundary — potentially including recent user content — becomes eligible
        for lossy summarization. The raw history is never modified, and the
        safe-boundary snap still keeps tool_use/tool_result pairs together.

        ``max_input_tokens`` bounds the summarization request itself: while the
        rendered request exceeds it, whole messages are dropped from the middle
        of the span (never splitting a tool exchange) and an omission notice is
        left in the gap. Dispatch walks a descending budget ladder (100%, 90%,
        80%, ... of the bound): if the enforcing side rejects a request the
        counter accepted, the next rung re-widens the gap and retries until a
        request is accepted. An unbounded request would be rejected before
        dispatch and read as ``llm_error``, permanently disabling
        auto-compaction.
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

        if keep_recent is None:
            keep_recent = self.KEEP_RECENT_MESSAGES
        split = conversation.compaction_split_point(keep_recent=keep_recent, min_prefix=self.MIN_PREFIX_TO_SUMMARIZE)
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
            # A summary needs no pixels, and base64 must stay out of the transcript.
            prefix = replace_image_blocks_with_placeholders(list(history[prior_through:split]), model)
            previous_summary = conversation.summary.get_text_content() if conversation.summary is not None else None

            def build_request(head_end: int, tail_start: int) -> MessageHistory:
                omitted = tail_start - head_end
                if omitted:
                    parts = [
                        _render_transcript(prefix[:head_end]),
                        _omission_notice(omitted),
                        _render_transcript(prefix[tail_start:]),
                    ]
                    transcript = "\n\n".join(part for part in parts if part)
                else:
                    transcript = _render_transcript(prefix)
                prompt = build_compression_summary_user_prompt(transcript, previous_summary=previous_summary)
                return MessageHistory([Message(role="user", content=[TextBlock(text=prompt)])])

            head_end, tail_start = len(prefix), len(prefix)
            messages = build_request(head_end, tail_start)
            system_text = system_prompt_text or COMPRESSION_SUMMARY_SYSTEM_PROMPT
            system_message = Message(role="system", content=[TextBlock(text=system_text)])
            weights = [_approx_message_chars(message) for message in prefix]

            async def widen_gap_until_fits(budget: int) -> None:
                nonlocal head_end, tail_start, messages
                for _ in range(self.MAX_FIT_ATTEMPTS):
                    counted = await llm.count_tokens(
                        messages=messages, system=system_message, model=model, thinking=thinking
                    )
                    if counted.input_tokens <= budget:
                        return
                    kept_chars = sum(weights[:head_end]) + sum(weights[tail_start:])
                    target_chars = kept_chars * budget / counted.input_tokens * 0.9
                    new_head, new_tail = _middle_gap(weights, target_chars)
                    # Keep tool exchanges whole on both edges of the gap.
                    new_head = conversation.snap_split_point(prior_through + new_head) - prior_through
                    while new_head < new_tail < len(prefix) and _bundles_tool_results(prefix[new_tail]):
                        new_tail += 1
                    if (new_head, new_tail) == (head_end, tail_start):
                        # Stalled (fixed prompt parts dominate): let the budget
                        # check or provider give the definitive answer.
                        return
                    head_end, tail_start = new_head, new_tail
                    messages = build_request(head_end, tail_start)

            response = None
            last_error: Optional[LLMContextWindowExceededError] = None
            dispatched_gap: Optional[tuple[int, int]] = None
            rungs = range(10, 0, -1) if max_input_tokens is not None else range(1)
            for tenths in rungs:
                if max_input_tokens is not None:
                    await widen_gap_until_fits(max_input_tokens * tenths // 10)
                    if dispatched_gap == (head_end, tail_start):
                        # This rung could not shrink the request any further;
                        # re-sending it would only be rejected again.
                        continue
                try:
                    response = await _stream_summary(
                        llm,
                        messages=messages,
                        system=system_message,
                        temperature=temperature,
                        model=model,
                        thinking=thinking,
                    )
                    break
                except LLMContextWindowExceededError as exc:
                    if max_input_tokens is None:
                        raise
                    dispatched_gap = (head_end, tail_start)
                    last_error = exc
            if response is None:
                # Every rung was rejected or would have re-sent an identical request.
                assert last_error is not None
                raise last_error

            last_empty_message = "Compression produced an empty summary; history left unchanged."
            for attempt in range(1, self.SUMMARY_EMPTY_ATTEMPTS + 1):
                summary_text = response.get_text_content()
                if summary_text and summary_text.strip():
                    folded = split - prior_through
                    conversation.apply_compaction(summary_text, split)
                    done = f"Compressed {folded} older message(s) into a summary; kept the latest turns verbatim."
                    if on_info:
                        await on_info(done)
                    return CompactionResult(ok=True, reason="ok", summarized_messages=folded, message=done)

                stop = getattr(response, "stop_reason", None)
                stop_note = f" (stop_reason={stop})" if stop else ""
                last_empty_message = f"Compression produced an empty summary{stop_note}; history left unchanged."
                if attempt >= self.SUMMARY_EMPTY_ATTEMPTS:
                    break
                retry_msg = (
                    f"Compression produced an empty summary{stop_note}; "
                    f"retrying ({attempt + 1}/{self.SUMMARY_EMPTY_ATTEMPTS})."
                )
                if on_info:
                    await on_info(retry_msg)
                else:
                    logger.info(retry_msg)
                # Same fitted request: empty completions are not a budget-fit failure.
                response = await _stream_summary(
                    llm,
                    messages=messages,
                    system=system_message,
                    temperature=temperature,
                    model=model,
                    thinking=thinking,
                )

            if on_error:
                await on_error(last_empty_message)
            else:
                logger.error(last_empty_message)
            return CompactionResult(ok=False, reason="empty_summary", message=last_empty_message)

        except Exception as e:
            # Never swallow into a fake success: log the full traceback and surface
            # the real error to the caller.
            logger.exception("Failed to compress message history")
            msg = f"Failed to compress message history: {type(e).__name__}: {e}"
            if on_error:
                await on_error(msg)
            return CompactionResult(ok=False, reason="llm_error", message=msg)
