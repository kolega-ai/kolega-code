# ruff: noqa: F401,F811,E402
"""Tests for the ChatGPT-subscription Responses provider and its wiring."""

import types

import httpx
import pytest

from kolega_code.auth import constants as chatgpt_constants
from kolega_code.auth.tokens import ChatGPTTokenManager, OAuthTokens
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider
from kolega_code.llm.client import LLMClient
from kolega_code.llm.models import (
    ImageBlock,
    Message,
    MessageHistory,
    ResponsesReasoningBlock,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)
from kolega_code.llm.providers.chatgpt_oauth import (
    ChatGPTAuth,
    ChatGPTOAuthProvider,
    ResponsesStreamWrapper,
    instructions_from,
    responses_tools,
    to_responses_input,
)
from kolega_code.llm.providers.models import GenerationParams


def _ns(**kwargs):
    return types.SimpleNamespace(**kwargs)


def _tokens():
    return OAuthTokens(access_token="at", refresh_token="rt", expires_at=10**12, account_id="acct_1", plan_type="pro")


class _FakeStream:
    def __init__(self, events):
        self._events = events

    def __aiter__(self):
        async def gen():
            for event in self._events:
                yield event

        return gen()


class _FakeResponses:
    def __init__(self, result):
        self._result = result
        self.last_kwargs = None

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._result


@pytest.mark.asyncio
async def test_responses_stream_wrapper_text_tools_and_usage():
    completed_response = _ns(
        output=[
            _ns(type="message", content=[_ns(type="output_text", text="Done.")]),
            _ns(type="function_call", call_id="call_1", id="fc_1", name="read_file", arguments='{"path": "a.py"}'),
        ],
        usage=_ns(input_tokens=12, output_tokens=4, total_tokens=16, input_tokens_details=None),
        status="completed",
        incomplete_details=None,
    )
    events = [
        _ns(type="response.output_text.delta", delta="Do"),
        _ns(type="response.output_text.delta", delta="ne."),
        _ns(
            type="response.output_item.added",
            item=_ns(type="function_call", call_id="call_1", id="fc_1", name="read_file"),
        ),
        _ns(type="response.function_call_arguments.delta", item_id="fc_1", delta='{"path"'),
        _ns(type="response.function_call_arguments.done", item_id="fc_1", arguments='{"path": "a.py"}'),
        _ns(type="response.completed", response=completed_response),
    ]

    wrapper = ResponsesStreamWrapper(_FakeStream(events))
    text_chunks = []
    tool_starts = []
    async with wrapper as stream:
        async for chunk in stream:
            if chunk.type == "text" and chunk.text:
                text_chunks.append(chunk.text)
            elif chunk.type == "tool_use_start":
                tool_starts.append(chunk.tool_call_delta)

    assert "".join(text_chunks) == "Done."
    assert tool_starts == [{"id": "call_1", "name": "read_file", "input": ""}]

    message = await wrapper.get_final_message()
    assert message.get_text_content() == "Done."
    assert message.stop_reason == "tool_use"
    tool_calls = [b for b in message.content if isinstance(b, ToolCall)]
    assert tool_calls[0].id == "call_1"
    assert tool_calls[0].input == {"path": "a.py"}
    assert message.usage_metadata["prompt_tokens"] == 12
    assert message.usage_metadata["completion_tokens"] == 4
    # Messages must be tagged with the real provider so history adaptation treats
    # them as openai_chatgpt (not the api-key "openai" provider).
    assert message.usage_metadata["provider"] == "openai_chatgpt"


@pytest.mark.asyncio
async def test_responses_stream_wrapper_uses_deltas_when_completed_output_empty():
    # The ChatGPT/Codex backend streams the answer as deltas but sends
    # response.completed with an empty output[]. The final message must come from
    # the deltas, not the empty output (this was the "agent says one thing and
    # stops" bug).
    completed = _ns(
        output=[],
        usage=_ns(input_tokens=5, output_tokens=1, total_tokens=6, input_tokens_details=None),
        status="completed",
        incomplete_details=None,
    )
    events = [
        _ns(type="response.output_text.delta", delta="ready"),
        _ns(type="response.completed", response=completed),
    ]
    wrapper = ResponsesStreamWrapper(_FakeStream(events))
    async with wrapper as stream:
        async for _chunk in stream:
            pass
    message = await wrapper.get_final_message()
    assert message.get_text_content() == "ready"
    assert message.usage_metadata["completion_tokens"] == 1
    assert message.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_responses_stream_wrapper_tool_call_from_output_item_done():
    # Tool calls must survive even when args arrive only via output_item.done
    # (and the completed output is empty, as on the ChatGPT backend).
    completed = _ns(output=[], usage=None, status="completed", incomplete_details=None)
    events = [
        _ns(
            type="response.output_item.done",
            item=_ns(type="function_call", id="fc_1", call_id="call_1", name="read_file", arguments='{"path": "a.py"}'),
        ),
        _ns(type="response.completed", response=completed),
    ]
    wrapper = ResponsesStreamWrapper(_FakeStream(events))
    async with wrapper as stream:
        async for _chunk in stream:
            pass
    message = await wrapper.get_final_message()
    tool_calls = [b for b in message.content if isinstance(b, ToolCall)]
    assert tool_calls and tool_calls[0].id == "call_1"
    assert tool_calls[0].name == "read_file"
    assert tool_calls[0].input == {"path": "a.py"}
    assert message.stop_reason == "tool_use"


@pytest.mark.asyncio
async def test_provider_always_sends_non_empty_instructions():
    # The backend 400s with "Instructions are required" on an empty instructions.
    provider = ChatGPTOAuthProvider(token_manager=ChatGPTTokenManager(_tokens()))
    fake = _FakeResponses(_FakeStream([]))
    provider.async_client = _ns(responses=fake)
    await provider.stream(
        MessageHistory([Message(role="user", content=[TextBlock(text="hi")])]),
        params=GenerationParams(),
        model="gpt-5.5",
    )
    assert fake.last_kwargs["instructions"]  # present and non-empty


@pytest.mark.asyncio
async def test_responses_stream_wrapper_max_tokens_stop_reason():
    completed = _ns(
        output=[_ns(type="message", content=[_ns(type="output_text", text="partial")])],
        usage=_ns(input_tokens=5, output_tokens=5, total_tokens=10, input_tokens_details=None),
        status="incomplete",
        incomplete_details=_ns(reason="max_output_tokens"),
    )
    wrapper = ResponsesStreamWrapper(_FakeStream([_ns(type="response.completed", response=completed)]))
    async with wrapper as stream:
        async for _chunk in stream:
            pass
    message = await wrapper.get_final_message()
    assert message.stop_reason == "max_tokens"


@pytest.mark.asyncio
async def test_responses_stream_wrapper_captures_reasoning_for_continuity():
    # With include=["reasoning.encrypted_content"], the backend emits a completed
    # reasoning item carrying the encrypted blob. We capture it (leading the
    # assistant message) so it round-trips into the next request for continuity.
    completed = _ns(output=[], usage=None, status="completed", incomplete_details=None)
    events = [
        _ns(
            type="response.output_item.done",
            item=_ns(
                type="reasoning",
                id="rs_1",
                encrypted_content="ENC",
                summary=[_ns(type="summary_text", text="planning")],
            ),
        ),
        _ns(
            type="response.output_item.done",
            item=_ns(type="function_call", id="fc_1", call_id="call_1", name="read_file", arguments='{"path": "a.py"}'),
        ),
        _ns(type="response.completed", response=completed),
    ]
    wrapper = ResponsesStreamWrapper(_FakeStream(events))
    async with wrapper as stream:
        async for _chunk in stream:
            pass
    message = await wrapper.get_final_message()

    # Reasoning leads the assistant content so it precedes the tool call on resend.
    assert isinstance(message.content[0], ResponsesReasoningBlock)
    assert message.content[0].encrypted_content == "ENC"
    assert message.content[0].summary == ["planning"]

    items = to_responses_input(MessageHistory([message]))
    assert items[0]["type"] == "reasoning"
    assert items[0]["encrypted_content"] == "ENC"
    assert items[1]["type"] == "function_call"


@pytest.mark.asyncio
async def test_responses_stream_wrapper_reasoning_from_final_output_fallback():
    # Some backends populate reasoning only on the final response output, not via
    # output_item.done events. The fallback path must still capture it.
    completed = _ns(
        output=[
            _ns(type="reasoning", id="rs_1", encrypted_content="ENC2", summary=[]),
            _ns(type="message", content=[_ns(type="output_text", text="hi")]),
        ],
        usage=None,
        status="completed",
        incomplete_details=None,
    )
    wrapper = ResponsesStreamWrapper(_FakeStream([_ns(type="response.completed", response=completed)]))
    async with wrapper as stream:
        async for _chunk in stream:
            pass
    message = await wrapper.get_final_message()
    reasoning = [b for b in message.content if isinstance(b, ResponsesReasoningBlock)]
    assert reasoning and reasoning[0].encrypted_content == "ENC2"
    assert reasoning[0].summary == []


@pytest.mark.asyncio
async def test_responses_stream_wrapper_skips_reasoning_without_encrypted_content():
    # A reasoning item lacking encrypted_content is useless for continuity (and
    # would be rejected on resend), so it must be dropped.
    completed = _ns(output=[], usage=None, status="completed", incomplete_details=None)
    events = [
        _ns(
            type="response.output_item.done", item=_ns(type="reasoning", id="rs_1", encrypted_content=None, summary=[])
        ),
        _ns(type="response.output_text.delta", delta="ok"),
        _ns(type="response.completed", response=completed),
    ]
    wrapper = ResponsesStreamWrapper(_FakeStream(events))
    async with wrapper as stream:
        async for _chunk in stream:
            pass
    message = await wrapper.get_final_message()
    assert not [b for b in message.content if isinstance(b, ResponsesReasoningBlock)]
