# ruff: noqa: F401,F811,E402
"""Tests for the ChatGPT-subscription Responses provider and its wiring."""

from typing import Dict, Optional

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


def test_provider_default_model_is_gpt56_sol():
    provider = ChatGPTOAuthProvider(token_manager=ChatGPTTokenManager(_tokens()))

    assert provider._default_model() == "gpt-5.6-sol"


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
        self.last_kwargs: Optional[Dict[str, object]] = None

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._result


@pytest.mark.asyncio
async def test_provider_generate_builds_codex_shaped_request(monkeypatch):
    provider = ChatGPTOAuthProvider(token_manager=ChatGPTTokenManager(_tokens()))
    completed = _ns(
        output=[_ns(type="message", content=[_ns(type="output_text", text="hi")])],
        usage=_ns(input_tokens=3, output_tokens=2, total_tokens=5, input_tokens_details=None),
        status="completed",
        incomplete_details=None,
    )
    fake = _FakeResponses(_FakeStream([_ns(type="response.completed", response=completed)]))
    monkeypatch.setattr(provider, "async_client", _ns(responses=fake))

    params = GenerationParams(max_completion_tokens=256, thinking="high")
    message = await provider.generate(
        MessageHistory([Message(role="user", content=[TextBlock(text="hello")])]),
        system=Message(role="system", content=[TextBlock(text="sys")]),
        params=params,
        model="gpt-5.5",
    )

    assert message.get_text_content() == "hi"
    assert message.usage_metadata["provider"] == "openai_chatgpt"
    kwargs = fake.last_kwargs
    assert kwargs is not None
    assert kwargs["model"] == "gpt-5.5"
    assert kwargs["store"] is False
    assert kwargs["stream"] is True  # backend is SSE-only; generate streams too
    assert kwargs["instructions"] == "sys"
    # summary="auto" so the backend streams a reasoning summary for the thinking
    # display; this is independent of continuity, which rides on `include` below.
    assert kwargs["reasoning"] == {"effort": "high", "summary": "auto"}
    # Reasoning continuity: ask the backend for the encrypted reasoning blob so it
    # can be resent next turn (matches Codex; the cause of the long-thinking gap).
    assert kwargs["include"] == ["reasoning.encrypted_content"]
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["parallel_tool_calls"] is False
    assert "prompt_cache_key" in kwargs
    # Codex never sends max_output_tokens; sending it triggers a 400.
    assert "max_output_tokens" not in kwargs
    assert kwargs["input"] == [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}]


@pytest.mark.asyncio
async def test_provider_generate_omits_reasoning_and_include_without_thinking(monkeypatch):
    provider = ChatGPTOAuthProvider(token_manager=ChatGPTTokenManager(_tokens()))
    completed = _ns(
        output=[_ns(type="message", content=[_ns(type="output_text", text="hi")])],
        usage=None,
        status="completed",
        incomplete_details=None,
    )
    fake = _FakeResponses(_FakeStream([_ns(type="response.completed", response=completed)]))
    monkeypatch.setattr(provider, "async_client", _ns(responses=fake))
    await provider.generate(
        MessageHistory([Message(role="user", content=[TextBlock(text="hello")])]),
        params=GenerationParams(),  # no thinking effort
        model="gpt-5.5",
    )
    assert fake.last_kwargs is not None
    assert "reasoning" not in fake.last_kwargs
    assert "include" not in fake.last_kwargs


@pytest.mark.asyncio
async def test_provider_stream_returns_wrapper(monkeypatch):
    provider = ChatGPTOAuthProvider(token_manager=ChatGPTTokenManager(_tokens()))
    monkeypatch.setattr(provider, "async_client", _ns(responses=_FakeResponses(_FakeStream([]))))
    stream = await provider.stream(
        MessageHistory([Message(role="user", content=[TextBlock(text="hi")])]),
        params=GenerationParams(),
        model="gpt-5.5",
    )
    assert isinstance(stream, ResponsesStreamWrapper)
