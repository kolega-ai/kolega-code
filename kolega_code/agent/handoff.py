"""Handoff: summarize a session into a document that seeds a fresh session.

Mirrors the helper-request pattern of ``compression.py``: a one-shot streamed
LLM call that renders the session as a role-tagged transcript and asks for a
structured handoff document. The result is injected as the sole context of a
brand-new session, so the summary must stand alone.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Union

from kolega_code.llm.ledger import helper_origin, llm_call_origin
from kolega_code.llm.models import Message, MessageHistory, TextBlock

from .compression import render_transcript
from .conversation import replace_image_blocks_with_placeholders
from .prompts import HANDOFF_SYSTEM_PROMPT, build_handoff_user_prompt

logger = logging.getLogger(__name__)


class HandoffCancelledError(Exception):
    """Raised when the user cancels handoff document generation."""


@dataclass(frozen=True)
class HandoffResult:
    """Outcome of a handoff document generation attempt.

    ``reason`` is a machine tag: "ok" | "too_few" | "empty_summary" | "llm_error".
    Cancellation is not a result — it raises :class:`HandoffCancelledError`.
    """

    ok: bool
    reason: str
    document: str = ""
    message: str = ""


# Don't summarize a session that has no real exchange to summarize.
HANDOFF_MIN_MESSAGES = 2
# Generous ceiling matching the compaction helper: reasoning models can consume
# most of the budget on chain-of-thought before emitting the document.
HANDOFF_MAX_TOKENS = 8192
# Completed-but-empty documents (reasoning-only, whitespace, no text) are
# retried on the same request before the attempt is reported unusable.
HANDOFF_EMPTY_ATTEMPTS = 3


def build_handoff_context_text(document: str) -> str:
    """Wrap a handoff document for injection as a new session's opening context."""
    return (
        "<handoff-context>\n"
        f"{document}\n"
        "</handoff-context>\n\n"
        "The above is a handoff document from a previous session. "
        "Use this context to continue the work seamlessly."
    )


async def _stream_document(
    llm: Any,
    messages: MessageHistory,
    system: Message,
    temperature: float,
    model: str,
    thinking: Optional[Union[int, str]],
    cancel_event: asyncio.Event,
) -> Message:
    """Issue the handoff helper request and return the completed message.

    Stream and consume events rather than calling ``generate()``: the Anthropic
    SDK rejects non-streaming requests whose max_tokens is large enough to risk
    a >10-minute response. Cancellation is cooperative — the stream is closed
    and :class:`HandoffCancelledError` raised when the event fires.
    """
    if cancel_event.is_set():
        raise HandoffCancelledError()
    with llm_call_origin(helper_origin("handoff")):
        stream_cm = await llm.stream(
            messages=messages,
            system=system,
            temperature=temperature,
            model=model,
            max_completion_tokens=HANDOFF_MAX_TOKENS,
            thinking=thinking,
        )
        async with stream_cm as stream:
            async for _event in stream:
                if cancel_event.is_set():
                    break
        if cancel_event.is_set():
            raise HandoffCancelledError()
        return await stream.get_final_message()


async def generate_handoff_document(
    messages: List[Message],
    *,
    llm: Any,
    model: str,
    temperature: float,
    thinking: Optional[Union[int, str]],
    focus: Optional[str] = None,
    cancel_event: Optional[asyncio.Event] = None,
) -> HandoffResult:
    """Summarize ``messages`` into a handoff document via a one-shot LLM call.

    Images are replaced with placeholders (base64 must stay out of the
    transcript); tool calls render as JSON arguments and tool results raw,
    exactly like compaction's transcript. Raises :class:`HandoffCancelledError`
    when ``cancel_event`` fires before a document is produced; otherwise never
    raises — failures come back as ``HandoffResult(ok=False, ...)``.
    """
    if len(messages) < HANDOFF_MIN_MESSAGES:
        return HandoffResult(
            ok=False,
            reason="too_few",
            message=(f"Nothing to hand off ({len(messages)} message(s); need at least {HANDOFF_MIN_MESSAGES})."),
        )

    if cancel_event is None:
        cancel_event = asyncio.Event()

    try:
        stripped = replace_image_blocks_with_placeholders(list(messages), model)
        prompt = build_handoff_user_prompt(render_transcript(stripped), focus=focus)
        request_messages = MessageHistory([Message(role="user", content=[TextBlock(text=prompt)])])
        system_message = Message(role="system", content=[TextBlock(text=HANDOFF_SYSTEM_PROMPT)])

        response = await _stream_document(
            llm, request_messages, system_message, temperature, model, thinking, cancel_event
        )
        for attempt in range(1, HANDOFF_EMPTY_ATTEMPTS + 1):
            document = response.get_text_content()
            if document and document.strip():
                return HandoffResult(ok=True, reason="ok", document=document.strip())
            stop = getattr(response, "stop_reason", None)
            stop_note = f" (stop_reason={stop})" if stop else ""
            if attempt >= HANDOFF_EMPTY_ATTEMPTS:
                break
            logger.info(
                "Handoff generation produced an empty document%s; retrying (%d/%d).",
                stop_note,
                attempt + 1,
                HANDOFF_EMPTY_ATTEMPTS,
            )
            response = await _stream_document(
                llm, request_messages, system_message, temperature, model, thinking, cancel_event
            )

        message = "Handoff generation produced an empty document; session left unchanged."
        logger.error(message)
        return HandoffResult(ok=False, reason="empty_summary", message=message)
    except HandoffCancelledError:
        raise
    except Exception as exc:
        logger.exception("Failed to generate handoff document")
        return HandoffResult(
            ok=False,
            reason="llm_error",
            message=f"Failed to generate handoff document: {type(exc).__name__}: {exc}",
        )
