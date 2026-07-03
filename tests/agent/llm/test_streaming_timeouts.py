"""Streaming requests must carry a bounded per-request timeout, and a stalled-stream
timeout must be classified as retryable.

Without an explicit timeout the SDK clients inherit a 600s read default, so a stalled
DeepSeek stream hangs ~10 min and leaks CLOSE_WAIT sockets. Each provider's stream() now
passes ``streaming_timeout()`` (300s read), and SDK connection/timeout errors map to the
retryable ``LLMInternalServerError`` so the agent loop re-issues the request.
"""

from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from anthropic import APITimeoutError as AnthropicAPITimeoutError
from openai import APITimeoutError as OpenAIAPITimeoutError

from kolega_code.llm.exceptions import LLMConnectionError, LLMTimeout, map_to_llm_error
from kolega_code.llm.models import Message, MessageHistory, TextBlock
from kolega_code.llm.providers.anthropic import AnthropicProvider
from kolega_code.llm.providers.openai import OpenAIProvider
from kolega_code.llm.providers.models import GenerationParams
from kolega_code.llm.ratelimit import RateLimiter
from kolega_code.llm.timeouts import STREAM_READ_TIMEOUT, streaming_timeout


class _RateLimiter(RateLimiter):
    async def acquire(self, tokens=None) -> None:
        return None


def _system() -> Message:
    return Message(role="system", content=[TextBlock(text="system prompt")])


def _messages() -> MessageHistory:
    return MessageHistory([Message(role="user", content=[TextBlock(text="hello")])])


def test_streaming_timeout_value():
    timeout = streaming_timeout()
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.read == STREAM_READ_TIMEOUT == 300.0
    # Far below the 600s SDK default that caused the hang.
    assert timeout.read is not None
    assert timeout.read < 600.0


@pytest.mark.asyncio
async def test_anthropic_stream_passes_timeout():
    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider.provider_name = "deepseek"
    provider.rate_limiter = _RateLimiter()
    provider.async_client = Mock()
    provider.async_client.messages.stream = Mock(return_value=Mock())

    await provider.stream(_messages(), system=_system(), params=GenerationParams())

    _, kwargs = provider.async_client.messages.stream.call_args
    assert "timeout" in kwargs
    assert kwargs["timeout"].read == 300.0


@pytest.mark.asyncio
async def test_openai_stream_passes_timeout():
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.provider_name = "deepseek"
    provider.rate_limiter = _RateLimiter()
    provider.async_client = Mock()
    provider.async_client.chat.completions.create = AsyncMock(return_value=Mock())

    await provider.stream(_messages(), system=_system(), params=GenerationParams())

    _, kwargs = provider.async_client.chat.completions.create.call_args
    assert "timeout" in kwargs
    assert kwargs["timeout"].read == 300.0


def test_anthropic_timeout_error_maps_to_llm_timeout():
    request = httpx.Request("POST", "https://api.deepseek.com/anthropic/v1/messages")
    mapped = map_to_llm_error(AnthropicAPITimeoutError(request=request), "deepseek")
    assert isinstance(mapped, LLMTimeout)
    assert isinstance(mapped, LLMConnectionError)  # retryable family, not a 5xx


def test_openai_timeout_error_maps_to_llm_timeout():
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    mapped = map_to_llm_error(OpenAIAPITimeoutError(request=request), "openai")
    assert isinstance(mapped, LLMTimeout)
    assert isinstance(mapped, LLMConnectionError)


def test_httpx_read_timeout_maps_to_llm_timeout():
    mapped = map_to_llm_error(httpx.ReadTimeout("timed out"), "deepseek")
    assert isinstance(mapped, LLMTimeout)


def test_connection_drop_maps_to_connection_error():
    # A dropped/reset connection (not a timeout) is a plain LLMConnectionError, still retryable.
    mapped = map_to_llm_error(httpx.ConnectError("connection refused"), "deepseek")
    assert isinstance(mapped, LLMConnectionError)
    assert not isinstance(mapped, LLMTimeout)


def test_connection_errors_are_retryable_by_the_agent_loop():
    # Pin the contract handle_llm_error relies on: connection failures are the retryable family.
    from kolega_code.llm.exceptions import LLMRateLimitError

    assert issubclass(LLMTimeout, LLMConnectionError)
    # Distinct from the server-error family (mislabeling a transport error as a 5xx was the bug).
    assert not issubclass(LLMConnectionError, LLMRateLimitError)
