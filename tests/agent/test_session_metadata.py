import asyncio
from typing import Any, AsyncIterator
from unittest.mock import MagicMock

import pytest

from kolega_code.agent.session_metadata import (
    SESSION_METADATA_HEAD_CHARS,
    SESSION_METADATA_TAIL_CHARS,
    TITLE_MAX_CHARS,
    _truncate_transcript,
    generate_session_metadata,
    parse_session_metadata_json,
)
from kolega_code.llm.models import Message, TextBlock


def test_parse_session_metadata_json_valid() -> None:
    raw = '{"title": "Fix Authentication Deadlock", "description": "Resolves DB lock during token refresh."}'
    result = parse_session_metadata_json(raw)
    assert result is not None
    assert result.title == "Fix Authentication Deadlock"
    assert result.description == "Resolves DB lock during token refresh."


def test_parse_session_metadata_json_code_fence() -> None:
    raw = """```json
{
  "title": "Add PostgreSQL Pool",
  "description": "Configures async connection pooling."
}
```"""
    result = parse_session_metadata_json(raw)
    assert result is not None
    assert result.title == "Add PostgreSQL Pool"
    assert result.description == "Configures async connection pooling."


def test_parse_session_metadata_json_regex_fallback() -> None:
    raw = 'Here is the result: {"title": "Refactor User Service", "description": "Cleaned up methods"}, hope it helps!'
    result = parse_session_metadata_json(raw)
    assert result is not None
    assert result.title == "Refactor User Service"
    assert result.description == "Cleaned up methods"


def test_parse_session_metadata_json_clamps_lengths() -> None:
    long_title = "A" * 100
    long_desc = "B" * 300
    raw = f'{{"title": "{long_title}", "description": "{long_desc}"}}'
    result = parse_session_metadata_json(raw)
    assert result is not None
    assert len(result.title) <= TITLE_MAX_CHARS
    assert len(result.description) <= 200


def test_parse_session_metadata_json_invalid_empty() -> None:
    assert parse_session_metadata_json("") is None
    assert parse_session_metadata_json("not json at all") is None
    assert parse_session_metadata_json('{"description": "No title"}') is None
    assert parse_session_metadata_json('{"title": "   "}') is None


def test_truncate_transcript_within_budget() -> None:
    short = "Hello world"
    assert _truncate_transcript(short) == short


def test_truncate_transcript_over_budget() -> None:
    long_text = "H" * (SESSION_METADATA_HEAD_CHARS + 500) + "T" * (SESSION_METADATA_TAIL_CHARS + 500)
    truncated = _truncate_transcript(long_text)
    assert "[... middle conversation omitted ...]" in truncated
    assert truncated.startswith("H" * 100)
    assert truncated.endswith("T" * 100)


@pytest.mark.asyncio
async def test_generate_session_metadata_too_few_messages() -> None:
    messages = [Message(role="user", content=[TextBlock(text="hi")])]
    result = await generate_session_metadata(messages, llm=MagicMock(), model="test-model")
    assert result is None


class FakeStream:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text

    async def __aenter__(self) -> "FakeStream":
        return self

    async def __aexit__(self, *args) -> None:
        pass

    def __aiter__(self) -> AsyncIterator[None]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[None]:
        yield None

    async def get_final_message(self) -> Message:
        return Message(role="assistant", content=[TextBlock(text=self.response_text)])


@pytest.mark.asyncio
async def test_generate_session_metadata_success_and_redaction() -> None:
    fake_stream = FakeStream(
        '{"title": "Fix Auth Token", "description": "Updated token refresh logic without SECRET_KEY."}'
    )
    llm = MagicMock()
    captured_messages: list[Any] = []

    async def mock_stream(**kwargs):
        captured_messages.extend(kwargs.get("messages", []))
        return fake_stream

    llm.stream = mock_stream

    messages = [
        Message(role="user", content=[TextBlock(text="Please fix SECRET_KEY_12345 in the auth service")]),
        Message(role="assistant", content=[TextBlock(text="I have fixed the issue.")]),
    ]

    result = await generate_session_metadata(
        messages,
        llm=llm,
        model="test-model",
        secret_values=["SECRET_KEY_12345"],
    )

    assert result is not None
    assert result.title == "Fix Auth Token"
    assert result.description == "Updated token refresh logic without SECRET_KEY."

    # Verify secret was scrubbed in the outgoing prompt to the LLM
    assert len(captured_messages) > 0
    prompt_text = captured_messages[0].content[0].text
    assert "SECRET_KEY_12345" not in prompt_text
    assert "[redacted]" in prompt_text


@pytest.mark.asyncio
async def test_generate_session_metadata_cancelled() -> None:
    cancel_event = asyncio.Event()
    cancel_event.set()

    messages = [
        Message(role="user", content=[TextBlock(text="first")]),
        Message(role="assistant", content=[TextBlock(text="second")]),
    ]

    result = await generate_session_metadata(
        messages,
        llm=MagicMock(),
        model="test-model",
        cancel_event=cancel_event,
    )
    assert result is None
