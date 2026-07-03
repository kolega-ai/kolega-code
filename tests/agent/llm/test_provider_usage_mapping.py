# ruff: noqa: F401,F811,E402
import asyncio
import os
from collections.abc import Awaitable
from unittest.mock import AsyncMock, patch

import pytest

from kolega_code.llm.client import (
    GenerationParams,
    LLMClient,
    TokenCount,
)
from kolega_code.llm.models import (
    Message,
    MessageChunk,
    MessageHistory,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResult,
)
from kolega_code.llm.providers.anthropic import AnthropicProvider, AnthropicStreamWrapper
from kolega_code.llm.providers.openai import OpenAIProvider

TEST_MESSAGES = MessageHistory([Message("user", [TextBlock("Hello, how are you?")])])
TEST_SYSTEM = Message("system", [TextBlock("You are a helpful assistant.")])


def test_fireworks_uses_openai_compatible_provider():
    client = LLMClient("fireworks", "test-key")

    assert isinstance(client.provider, OpenAIProvider)
    assert not isinstance(client.provider, AnthropicProvider)
    assert client.provider.provider_name == "fireworks"
    assert client.provider.base_url == "https://api.fireworks.ai/inference/v1"


def test_ollama_cloud_uses_openai_compatible_provider():
    client = LLMClient("ollama_cloud", "test-key")

    assert isinstance(client.provider, OpenAIProvider)
    assert not isinstance(client.provider, AnthropicProvider)
    assert client.provider.provider_name == "ollama_cloud"
    assert client.provider.base_url == "https://ollama.com/v1"


@pytest.mark.asyncio
async def test_fireworks_generate_maps_openai_provider_response_usage(capsys):
    client = LLMClient("fireworks", "test-key")
    assert isinstance(client.provider, OpenAIProvider)

    class MessageObj:
        content = "ok"
        reasoning_content = "brief reasoning"
        tool_calls = None

    class Choice:
        message = MessageObj()

    class PromptTokenDetails:
        cached_tokens = 76

    class Usage:
        prompt_tokens = 321
        completion_tokens = 54
        total_tokens = 375
        prompt_tokens_details = PromptTokenDetails()

    class OpenAIResponse:
        choices = [Choice()]
        usage = Usage()

    create = AsyncMock(return_value=OpenAIResponse())
    with patch.object(client.provider.async_client.chat.completions, "create", create):
        response = await client.generate(
            messages=TEST_MESSAGES,
            system=TEST_SYSTEM,
            model="accounts/fireworks/models/glm-5p2",
            temperature=1.0,
            max_completion_tokens=8,
            thinking="high",
        )

    assert create.await_count == 1
    assert create.await_args is not None
    assert create.await_args.kwargs["model"] == "accounts/fireworks/models/glm-5p2"
    assert create.await_args.kwargs["reasoning_effort"] == "high"
    assert response.usage_metadata == {
        "provider": "fireworks",
        "prompt_tokens": 321,
        "completion_tokens": 54,
        "total_tokens": 375,
        "cache_read_input_tokens": 76,
    }
    assert isinstance(response.content, list)
    assert response.content[0].type == "thinking"
    assert isinstance(response.content[1], TextBlock)
    assert response.content[1].text == "ok"
    assert capsys.readouterr().out == ""


@pytest.mark.asyncio
async def test_fireworks_stream_final_message_maps_openai_usage():
    client = LLMClient("fireworks", "test-key")
    assert isinstance(client.provider, OpenAIProvider)

    class Delta:
        def __init__(self, content=None, reasoning_content=None):
            self.content = content
            self.reasoning_content = reasoning_content
            self.tool_calls = []

    class Choice:
        def __init__(self, delta, finish_reason=None):
            self.delta = delta
            self.finish_reason = finish_reason

    class PromptTokenDetails:
        cached_tokens = 33

    class Usage:
        prompt_tokens = 11
        completion_tokens = 22
        total_tokens = 33
        prompt_tokens_details = PromptTokenDetails()

    class Chunk:
        def __init__(self, delta, finish_reason=None, usage=None):
            self.choices = [Choice(delta, finish_reason)]
            self.usage = usage

    class FakeOpenAIStream:
        def __init__(self):
            self._chunks = iter(
                [
                    Chunk(Delta(reasoning_content="think ")),
                    Chunk(Delta(reasoning_content="hard")),
                    Chunk(Delta(content="ok"), finish_reason="stop", usage=Usage()),
                ]
            )

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._chunks)
            except StopIteration:
                raise StopAsyncIteration

        async def aclose(self):
            pass

    create = AsyncMock(return_value=FakeOpenAIStream())
    with patch.object(client.provider.async_client.chat.completions, "create", create):
        # ``client.stream`` is typed as ``AsyncContextManager | Coroutine[...,
        # AsyncContextManager]`` (some providers return a plain context manager).
        # For OpenAI-compatible providers it returns a coroutine; narrow to the
        # awaitable branch before awaiting.
        stream_result = client.stream(
            messages=TEST_MESSAGES,
            system=TEST_SYSTEM,
            model="accounts/fireworks/models/glm-5p2",
            temperature=1.0,
            max_completion_tokens=8,
            thinking="low",
        )
        assert isinstance(stream_result, Awaitable)
        fireworks_stream = await stream_result

        chunks = []
        async with fireworks_stream as stream_ctx:
            async for chunk in stream_ctx:
                chunks.append(chunk)
            final_message = await stream_ctx.get_final_message()

    assert create.await_count == 1
    assert create.await_args is not None
    assert create.await_args.kwargs["model"] == "accounts/fireworks/models/glm-5p2"
    assert create.await_args.kwargs["reasoning_effort"] == "low"
    assert [chunk.type for chunk in chunks] == ["thinking", "thinking", "text"]
    assert final_message.content[0].type == "thinking"
    assert final_message.content[0].thinking == "think hard"
    assert final_message.content[1].text == "ok"
    assert final_message.usage_metadata == {
        "prompt_tokens": 11,
        "completion_tokens": 22,
        "total_tokens": 33,
        "cache_read_input_tokens": 33,
        "provider": "fireworks",
    }


@pytest.mark.asyncio
async def test_moonshot_generate_maps_provider_response_usage(capsys):
    """Kimi billing metadata should come from Moonshot's Anthropic-shaped usage block."""
    client = LLMClient("moonshot", "test-key")
    assert isinstance(client.provider, AnthropicProvider)

    class TextContent:
        type = "text"
        text = "ok"

    class Usage:
        input_tokens = 123
        output_tokens = 45
        cache_read_input_tokens = 67
        cache_creation_input_tokens = 89
        prompt_tokens = 999
        completion_tokens = 888
        total_tokens = 1887

    class AnthropicMessage:
        role = "assistant"
        content = [TextContent()]
        stop_reason = "end_turn"
        usage = Usage()

    with patch.object(client.provider.async_client.messages, "create", AsyncMock(return_value=AnthropicMessage())):
        response = await client.generate(
            messages=TEST_MESSAGES,
            system=TEST_SYSTEM,
            model="kimi-k2.6",
            temperature=1.0,
            max_completion_tokens=8,
        )

    assert response.usage_metadata == {
        "input_tokens": 123,
        "output_tokens": 45,
        "cache_read_input_tokens": 67,
        "cache_write_input_tokens": 89,
        "provider": "moonshot",
    }
    assert capsys.readouterr().out == ""


@pytest.mark.asyncio
async def test_anthropic_opus_47_generate_omits_deprecated_temperature():
    client = LLMClient("anthropic", "test-key")
    assert isinstance(client.provider, AnthropicProvider)

    class TextContent:
        type = "text"
        text = "ok"

    class AnthropicMessage:
        role = "assistant"
        content = [TextContent()]
        stop_reason = "end_turn"
        usage = None

    create = AsyncMock(return_value=AnthropicMessage())
    with patch.object(client.provider.async_client.messages, "create", create):
        await client.generate(
            messages=TEST_MESSAGES,
            system=TEST_SYSTEM,
            model="claude-opus-4-7",
            temperature=0.7,
            max_completion_tokens=8,
        )

    assert create.await_args is not None
    assert "temperature" not in create.await_args.kwargs


@pytest.mark.asyncio
async def test_anthropic_opus_47_stream_omits_deprecated_temperature():
    client = LLMClient("anthropic", "test-key")
    assert isinstance(client.provider, AnthropicProvider)

    with patch.object(client.provider.async_client.messages, "stream", return_value=object()) as stream:
        stream_result = client.stream(
            messages=TEST_MESSAGES,
            system=TEST_SYSTEM,
            model="claude-opus-4-7",
            temperature=0.7,
            max_completion_tokens=8,
        )
        assert isinstance(stream_result, Awaitable)
        await stream_result

    assert stream.call_args is not None
    assert "temperature" not in stream.call_args.kwargs


@pytest.mark.asyncio
async def test_anthropic_non_opus_47_generate_keeps_temperature():
    client = LLMClient("anthropic", "test-key")
    assert isinstance(client.provider, AnthropicProvider)

    class TextContent:
        type = "text"
        text = "ok"

    class AnthropicMessage:
        role = "assistant"
        content = [TextContent()]
        stop_reason = "end_turn"
        usage = None

    create = AsyncMock(return_value=AnthropicMessage())
    with patch.object(client.provider.async_client.messages, "create", create):
        await client.generate(
            messages=TEST_MESSAGES,
            system=TEST_SYSTEM,
            model="claude-sonnet-4-5-20250929",
            temperature=0.7,
            max_completion_tokens=8,
        )

    assert create.await_args is not None
    assert create.await_args.kwargs["temperature"] == 0.7
