"""Generate descriptive session title and description via the active LLM."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from kolega_code.gateway.redaction import scrub
from kolega_code.llm.ledger import helper_origin, llm_call_origin
from kolega_code.llm.models import Message, MessageHistory, TextBlock

from .compression import render_transcript
from .conversation import replace_image_blocks_with_placeholders

logger = logging.getLogger(__name__)

SESSION_METADATA_MIN_MESSAGES = 2
SESSION_METADATA_MAX_TOKENS = 256
SESSION_METADATA_HEAD_CHARS = 2000
SESSION_METADATA_TAIL_CHARS = 6000

TITLE_MAX_CHARS = 60
DESCRIPTION_MAX_CHARS = 200

SESSION_METADATA_SYSTEM_PROMPT = """\
You are an assistant that summarizes developer coding sessions.
Analyze the provided conversation transcript and generate:
1. title: A concise, descriptive title in Natural Title Case (e.g. "Fix Authentication Middleware", "Add PostgreSQL Connection Pool", "Investigate Bug 662"). Maximum 60 characters. No surrounding quotes or Markdown formatting.
2. description: A clear 1 to 2 sentence summary of what the session is working on, goals, and key progress or findings so far. Maximum 200 characters.

Respond ONLY with valid JSON in this exact structure:
{
  "title": "Natural Title Case Title",
  "description": "One or two sentences describing the session."
}
"""


@dataclass(frozen=True)
class SessionMetadata:
    """The generated descriptive metadata for a session."""

    title: str
    description: str


class SessionMetadataCancelledError(Exception):
    """Raised when metadata generation is cancelled."""


def _truncate_transcript(transcript: str) -> str:
    """Bound the transcript sent to the metadata generator."""
    total_budget = SESSION_METADATA_HEAD_CHARS + SESSION_METADATA_TAIL_CHARS
    if len(transcript) <= total_budget:
        return transcript
    head = transcript[:SESSION_METADATA_HEAD_CHARS]
    tail = transcript[-SESSION_METADATA_TAIL_CHARS:]
    return f"{head}\n\n[... middle conversation omitted ...]\n\n{tail}"


def parse_session_metadata_json(raw_text: str) -> Optional[SessionMetadata]:
    """Parse and sanitize the JSON returned by the model."""
    text = raw_text.strip()
    if not text:
        return None

    # Strip code fences if present
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    data: Optional[dict[str, Any]] = None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            data = parsed
    except Exception:
        # Fallback regex extraction for title and description
        title_match = re.search(r'"title"\s*:\s*"([^"]+)"', text)
        desc_match = re.search(r'"description"\s*:\s*"([^"]+)"', text)
        if title_match:
            data = {
                "title": title_match.group(1),
                "description": desc_match.group(1) if desc_match else "",
            }

    if not data or not isinstance(data.get("title"), str):
        return None

    raw_title = data["title"].strip().strip("\"'")
    # Clean control characters and normalize spaces
    cleaned_title = " ".join(raw_title.split())
    if not cleaned_title:
        return None

    if len(cleaned_title) > TITLE_MAX_CHARS:
        cleaned_title = cleaned_title[:TITLE_MAX_CHARS].rstrip()

    raw_description = str(data.get("description") or "").strip()
    cleaned_description = " ".join(raw_description.split())
    if len(cleaned_description) > DESCRIPTION_MAX_CHARS:
        cleaned_description = cleaned_description[:DESCRIPTION_MAX_CHARS].rstrip()

    return SessionMetadata(title=cleaned_title, description=cleaned_description)


async def generate_session_metadata(
    messages: List[Message],
    *,
    llm: Any,
    model: str,
    temperature: float = 0.2,
    cancel_event: Optional[asyncio.Event] = None,
    secret_values: Sequence[str] = (),
) -> Optional[SessionMetadata]:
    """Summarize ``messages`` into a title and description via the active LLM.

    Returns ``None`` if generation fails or is cancelled. Never raises.
    """
    if len(messages) < SESSION_METADATA_MIN_MESSAGES:
        return None

    if cancel_event is None:
        cancel_event = asyncio.Event()

    try:
        stripped = replace_image_blocks_with_placeholders(list(messages), model)
        transcript = render_transcript(stripped)
        bounded_transcript = _truncate_transcript(transcript)
        scrubbed_transcript = scrub(bounded_transcript, secret_values)

        user_content = f"Conversation transcript:\n\n{scrubbed_transcript}"
        request_messages = MessageHistory([Message(role="user", content=[TextBlock(text=user_content)])])
        system_message = Message(role="system", content=[TextBlock(text=SESSION_METADATA_SYSTEM_PROMPT)])

        if cancel_event.is_set():
            return None

        with llm_call_origin(helper_origin("session_metadata")):
            stream_cm = await llm.stream(
                messages=request_messages,
                system=system_message,
                temperature=temperature,
                model=model,
                max_completion_tokens=SESSION_METADATA_MAX_TOKENS,
                thinking=None,
            )
            async with stream_cm as stream:
                async for _event in stream:
                    if cancel_event.is_set():
                        return None

            if cancel_event.is_set():
                return None

            response = await stream.get_final_message()
            raw_output = response.get_text_content()
            return parse_session_metadata_json(raw_output)
    except Exception as exc:
        logger.warning("Failed to generate session metadata: %s", exc)
        return None
