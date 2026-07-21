"""Unit tests for Gemini thought_signature round-tripping (no API key required).

Gemini 3.x attaches an encrypted thought_signature to each function-call part and rejects
resent history whose function-call parts omit it. These tests verify the signature is captured
on the way in, emitted on the way out, and survives session persistence.
"""

import json
from collections.abc import AsyncGenerator

import pytest
from google.genai import types as genai_types

from kolega_code.agent.conversation import Conversation
from kolega_code.llm.models import Message, ToolCall
from kolega_code.llm.providers.google import GoogleStreamWrapper


def _fake_google_response(signature: bytes) -> genai_types.GenerateContentResponse:
    """Minimal stand-in for a genai GenerateContentResponse with one function-call part."""
    part = genai_types.Part(
        function_call=genai_types.FunctionCall(id="call-1", name="list_directory", args={"path": "."}),
        thought_signature=signature,
        thought=False,
    )
    candidate = genai_types.Candidate(content=genai_types.Content(parts=[part]))
    return genai_types.GenerateContentResponse(
        candidates=[candidate],
        usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
            prompt_token_count=1,
            candidates_token_count=1,
            total_token_count=2,
        ),
    )


def test_to_google_emits_thought_signature() -> None:
    tc = ToolCall(id="c1", name="list_directory", input={"path": "."}, thought_signature=b"\x01\x02SIG")
    part = tc.to_google()
    assert part.function_call is not None
    assert part.function_call.name == "list_directory"
    assert part.thought_signature == b"\x01\x02SIG"


def test_to_google_without_signature_is_none() -> None:
    tc = ToolCall(id="c1", name="x", input={})
    assert tc.to_google().thought_signature is None


def test_to_dict_from_dict_round_trip_preserves_signature() -> None:
    tc = ToolCall(id="c1", name="list_directory", input={"path": "."}, thought_signature=b"\x01\x02SIG")
    data = tc.to_dict()
    # Must be JSON-serializable (base64 string), not raw bytes.
    assert isinstance(data["thought_signature"], str)
    restored = ToolCall.from_dict(json.loads(json.dumps(data)))
    assert restored.thought_signature == b"\x01\x02SIG"


def test_to_dict_omits_signature_when_absent() -> None:
    assert "thought_signature" not in ToolCall(id="c1", name="x", input={}).to_dict()


def test_from_google_captures_signature() -> None:
    msg = Message.from_google(_fake_google_response(b"GSIG"))
    tool_calls = [b for b in msg.content if isinstance(b, ToolCall)]
    assert len(tool_calls) == 1
    assert tool_calls[0].thought_signature == b"GSIG"
    # Also exposed via the dedicated tool_calls field.
    assert msg.tool_calls[0].thought_signature == b"GSIG"


@pytest.mark.asyncio
async def test_google_stream_preserves_final_usage_metadata() -> None:
    async def stream() -> AsyncGenerator[genai_types.GenerateContentResponse, None]:
        yield _fake_google_response(b"GSIG")

    async with GoogleStreamWrapper(stream()) as wrapper:
        await anext(wrapper)
        message = await wrapper.get_final_message()

    assert message.usage_metadata == {
        "prompt_token_count": 1,
        "candidates_token_count": 1,
        "total_token_count": 2,
        "provider": "google",
    }


def test_persisted_session_preserves_signature() -> None:
    """dump -> JSON -> restore keeps the signature so a resumed Gemini session re-emits it."""
    conversation = Conversation()
    conversation.history.append(Message.from_google(_fake_google_response(b"GSIG")))

    reloaded = json.loads(json.dumps(conversation.dump()))
    restored = Conversation()
    restored.restore(reloaded)

    restored_tool_call = next(b for b in restored.history[0].content if isinstance(b, ToolCall))
    assert restored_tool_call.thought_signature == b"GSIG"
    # And it makes it back into the request payload.
    assert restored_tool_call.to_google().thought_signature == b"GSIG"
