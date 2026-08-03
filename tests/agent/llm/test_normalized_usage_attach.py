"""Attach-path tests: every provider wrapper and generate() must return messages
carrying NormalizedUsage, idempotently across repeated finalization."""

from types import SimpleNamespace

import pytest

from kolega_code.llm.client import LLMClient
from kolega_code.llm.models import Message, MessageHistory
from kolega_code.llm.providers.google import GoogleStreamWrapper
from kolega_code.llm.providers.anthropic import AnthropicStreamWrapper
from kolega_code.llm.providers.openai import OpenAIProvider, OpenAIStreamWrapper
from kolega_code.llm.providers.responses_common import ResponsesStreamWrapper, _usage_from_response
from kolega_code.llm.usage import REASON_NOT_REPORTED


class _Chunk:
    def __init__(self, text=None, usage=None):
        delta = SimpleNamespace(content=text, tool_calls=[])
        self.choices = [SimpleNamespace(delta=delta, finish_reason="stop" if usage else None)]
        self.usage = usage


class _AsyncStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)

    async def aclose(self):
        pass


def _openai_usage(**overrides):
    base = dict(
        prompt_tokens=15,
        completion_tokens=28,
        total_tokens=43,
        prompt_tokens_details=None,
        completion_tokens_details=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


async def _drain(wrapper):
    async with wrapper as s:
        async for _ in s:
            pass
        return await s.get_final_message()


@pytest.mark.asyncio
async def test_openai_stream_wrapper_attaches_usage_and_is_idempotent():
    usage = _openai_usage(completion_tokens_details={"reasoning_tokens": 25})
    wrapper = OpenAIStreamWrapper(
        _AsyncStream([_Chunk("hi"), _Chunk(usage=usage)]),
        requested_include_usage=True,
        provider_name="together",
        model="moonshotai/Kimi-K2.7-Code",
    )
    message = await _drain(wrapper)
    assert message.usage is not None
    assert message.usage.reported is True
    assert message.usage.provider == "together"
    assert message.usage.model == "moonshotai/Kimi-K2.7-Code"
    assert message.usage.input_tokens == 15
    assert message.usage.output_tokens == 28
    assert message.usage.total_tokens == 43
    assert message.usage.reasoning_output_tokens == 25

    # Repeated finalization (baseagent calls after __aexit__): equal values.
    again = await wrapper.get_final_message()
    assert again.usage == message.usage
    assert again.usage_metadata == message.usage_metadata


@pytest.mark.asyncio
async def test_openai_stream_wrapper_without_usage_reports_unavailable():
    wrapper = OpenAIStreamWrapper(
        _AsyncStream([_Chunk("hi")]), requested_include_usage=True, provider_name="groq", model="m"
    )
    message = await _drain(wrapper)
    assert message.usage is not None
    assert message.usage.reported is False
    assert message.usage.unavailable_reason == REASON_NOT_REPORTED
    assert message.usage.input_tokens is None


@pytest.mark.asyncio
async def test_xai_stream_wrapper_adds_reasoning_to_output():
    usage = _openai_usage(
        prompt_tokens=213,
        completion_tokens=1,
        total_tokens=238,
        completion_tokens_details={"reasoning_tokens": 24},
        prompt_tokens_details={"cached_tokens": 128},
    )
    wrapper = OpenAIStreamWrapper(
        _AsyncStream([_Chunk(usage=usage)]), requested_include_usage=True, provider_name="xai", model="grok-4.5"
    )
    message = await _drain(wrapper)
    assert message.usage is not None
    assert message.usage.output_tokens == 25
    assert message.usage.total_tokens == 238  # matches xAI's own reported total
    assert message.usage.cache_read_input_tokens == 128


@pytest.mark.asyncio
async def test_openrouter_stream_wrapper_subtracts_then_adds_back_cache_writes():
    usage = _openai_usage(
        prompt_tokens=19026,
        completion_tokens=4,
        total_tokens=19030,
        prompt_tokens_details={"cached_tokens": 0, "cache_write_tokens": 19023},
    )
    wrapper = OpenAIStreamWrapper(
        _AsyncStream([_Chunk(usage=usage)]),
        requested_include_usage=True,
        provider_name="openrouter",
        model="anthropic/claude-haiku-4.5",
    )
    message = await _drain(wrapper)
    # Raw metadata keeps the anthropic-style exclusive prompt count...
    assert message.usage_metadata["prompt_tokens"] == 3
    assert message.usage_metadata["cache_write_input_tokens"] == 19023
    # ...and the normalized view restores the provider-inclusive input.
    assert message.usage is not None
    assert message.usage.input_tokens == 19026
    assert message.usage.total_tokens == 19030


@pytest.mark.asyncio
async def test_anthropic_stream_wrapper_attaches_inclusive_input():
    wrapper = AnthropicStreamWrapper(SimpleNamespace(), provider_name="anthropic", model="claude-haiku-4-5-20251001")

    async def fake_final_message():
        return SimpleNamespace(
            role="assistant",
            content=[],
            stop_reason="end_turn",
            usage=SimpleNamespace(
                input_tokens=13,
                output_tokens=4,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=19013,
            ),
        )

    wrapper.generator = SimpleNamespace(get_final_message=fake_final_message)
    message = await wrapper.get_final_message()
    assert message.usage is not None
    assert message.usage.input_tokens == 19026
    assert message.usage.cache_write_input_tokens == 19013
    assert message.usage.model == "claude-haiku-4-5-20251001"

    again = await wrapper.get_final_message()
    assert again.usage == message.usage


@pytest.mark.asyncio
async def test_google_stream_wrapper_attaches_usage_with_thoughts():
    chunk = SimpleNamespace(
        text="ok",
        usage_metadata=SimpleNamespace(
            prompt_token_count=8,
            candidates_token_count=1,
            total_token_count=111,
            cached_content_token_count=None,
            thoughts_token_count=102,
            tool_use_prompt_token_count=None,
        ),
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[]), finish_reason=SimpleNamespace(value="STOP"))],
    )
    wrapper = GoogleStreamWrapper(_AsyncStream([chunk]), model="gemini-3.5-flash")
    message = await _drain(wrapper)
    assert message.usage_metadata["thoughts_token_count"] == 102
    assert message.usage is not None
    assert message.usage.input_tokens == 8
    assert message.usage.output_tokens == 103
    assert message.usage.total_tokens == 111  # matches Google's own total
    assert message.usage.reasoning_output_tokens == 102

    again = await wrapper.get_final_message()
    assert again.usage == message.usage


@pytest.mark.asyncio
async def test_responses_stream_wrapper_attaches_usage_with_reasoning():
    wrapper = ResponsesStreamWrapper(SimpleNamespace(), provider_name="openai", model="gpt-5.4-mini")
    wrapper._final_response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=93,
            output_tokens=57,
            total_tokens=150,
            input_tokens_details=SimpleNamespace(cached_tokens=0),
            output_tokens_details=SimpleNamespace(reasoning_tokens=48),
        ),
        output=[],
        status="completed",
    )
    message = await wrapper.get_final_message()
    assert message.usage_metadata["reasoning_output_tokens"] == 48
    assert message.usage is not None
    assert message.usage.reported is True
    assert message.usage.provider == "openai"
    assert message.usage.model == "gpt-5.4-mini"
    assert message.usage.input_tokens == 93
    assert message.usage.output_tokens == 57
    assert message.usage.total_tokens == 150

    again = await wrapper.get_final_message()
    assert again.usage == message.usage


@pytest.mark.asyncio
async def test_responses_stream_wrapper_without_completed_event_reports_unavailable():
    wrapper = ResponsesStreamWrapper(SimpleNamespace(), provider_name="openai_chatgpt", model="gpt-5.6-sol")
    message = await wrapper.get_final_message()
    assert message.usage is not None
    assert message.usage.reported is False
    assert message.usage.provider == "openai_chatgpt"
    assert message.usage.unavailable_reason == REASON_NOT_REPORTED


def test_usage_from_response_does_not_capture_cache_writes():
    # Guard for the double-count trap documented in _usage_from_response: the
    # Responses shape must never store cache_write_input_tokens.
    usage = SimpleNamespace(
        input_tokens=100,
        output_tokens=10,
        total_tokens=110,
        input_tokens_details=SimpleNamespace(cached_tokens=50, cache_write_tokens=25),
        output_tokens_details=None,
    )
    metadata = _usage_from_response(SimpleNamespace(usage=usage))
    assert metadata["prompt_tokens"] == 100
    assert metadata["cache_read_input_tokens"] == 50
    assert "cache_write_input_tokens" not in metadata


@pytest.mark.asyncio
async def test_plain_llm_client_generate_attaches_usage_without_langfuse(monkeypatch):
    client = LLMClient(provider="together", api_key="sk-test")

    async def fake_create(*args, **kwargs):
        return SimpleNamespace(
            usage=_openai_usage(),
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None, finish_reason="stop"))],
        )

    provider = client.provider
    assert isinstance(provider, OpenAIProvider)
    monkeypatch.setattr(provider.async_client.chat.completions, "create", fake_create)

    message = await client.generate(MessageHistory([Message(role="user", content="hi")]))
    assert message.usage is not None
    assert message.usage.reported is True
    assert message.usage.provider == "together"
    assert message.usage.input_tokens == 15
    assert message.usage.output_tokens == 28
