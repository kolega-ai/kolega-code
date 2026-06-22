"""Tests for the api-key OpenAI Responses provider and its routing.

gpt-5.x reject ``function tools + reasoning_effort`` on Chat Completions, so the
api-key ``openai`` provider routes to the Responses API. The OpenAI-compatible
providers (fireworks, xai, …) keep using the Chat Completions ``OpenAIProvider``.
"""

import types

import pytest

from kolega_code.llm.client import LLMClient
from kolega_code.llm.models import (
    Message,
    MessageHistory,
    ResponsesReasoningBlock,
    TextBlock,
    ToolDefinition,
    ToolParameter,
)
from kolega_code.llm.providers.models import GenerationParams
from kolega_code.llm.providers.openai import OpenAIProvider
from kolega_code.llm.providers.openai_responses import OpenAIResponsesProvider


def _ns(**kwargs):
    return types.SimpleNamespace(**kwargs)


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


# --- routing --------------------------------------------------------------------


def test_llmclient_routes_openai_to_responses_provider():
    client = LLMClient(provider="openai", api_key="sk-test")
    assert isinstance(client.provider, OpenAIResponsesProvider)
    assert client.provider.provider_name == "openai"


def test_llmclient_routes_compatible_provider_to_chat_completions():
    client = LLMClient(provider="fireworks", api_key="sk-test")
    assert isinstance(client.provider, OpenAIProvider)
    assert not isinstance(client.provider, OpenAIResponsesProvider)


# --- request building (no network) ----------------------------------------------


def test_build_request_is_responses_shaped_with_reasoning():
    provider = OpenAIResponsesProvider(api_key="sk-test")
    tool = ToolDefinition(
        name="read_file",
        description="Read a file",
        parameters=[ToolParameter(name="path", type="string", description="path", required=True)],
    )
    params = GenerationParams(tools=[tool], thinking="high", max_completion_tokens=256, temperature=0.5)
    request = provider._build_request(
        MessageHistory([Message(role="user", content=[TextBlock(text="hello")])]),
        Message(role="system", content=[TextBlock(text="sys")]),
        params,
        {"model": "gpt-5.5"},
    )

    assert request["model"] == "gpt-5.5"
    assert request["stream"] is True
    assert request["store"] is False
    assert request["tool_choice"] == "auto"
    assert request["instructions"] == "sys"
    assert request["reasoning"] == {"effort": "high", "summary": "auto"}
    assert request["include"] == ["reasoning.encrypted_content"]
    assert request["tools"][0]["type"] == "function"
    assert request["tools"][0]["name"] == "read_file"
    assert request["input"] == [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}]
    # The Responses path must NOT send Chat-Completions-only fields.
    assert "temperature" not in request
    assert "max_completion_tokens" not in request
    assert "max_output_tokens" not in request


def test_build_request_default_model_and_no_reasoning_without_thinking():
    provider = OpenAIResponsesProvider(api_key="sk-test")
    request = provider._build_request(
        MessageHistory([Message(role="user", content=[TextBlock(text="hi")])]),
        None,
        GenerationParams(),
        {},
    )
    assert request["model"] == "gpt-5.5"
    assert "reasoning" not in request
    assert "include" not in request


# --- generate / stream (no network) ---------------------------------------------


@pytest.mark.asyncio
async def test_generate_tags_openai_provider_and_sends_include():
    provider = OpenAIResponsesProvider(api_key="sk-test")
    completed = _ns(
        output=[_ns(type="message", content=[_ns(type="output_text", text="hi")])],
        usage=_ns(input_tokens=3, output_tokens=2, total_tokens=5, input_tokens_details=None),
        status="completed",
        incomplete_details=None,
    )
    fake = _FakeResponses(_FakeStream([_ns(type="response.completed", response=completed)]))
    provider.async_client = _ns(responses=fake)

    msg = await provider.generate(
        MessageHistory([Message(role="user", content=[TextBlock(text="hello")])]),
        params=GenerationParams(thinking="medium"),
        model="gpt-5.5",
    )

    assert msg.get_text_content() == "hi"
    assert msg.usage_metadata["provider"] == "openai"
    assert fake.last_kwargs["include"] == ["reasoning.encrypted_content"]
    assert fake.last_kwargs["reasoning"] == {"effort": "medium", "summary": "auto"}


@pytest.mark.asyncio
async def test_stream_captures_reasoning_for_continuity():
    provider = OpenAIResponsesProvider(api_key="sk-test")
    completed = _ns(output=[], usage=None, status="completed", incomplete_details=None)
    events = [
        _ns(type="response.output_item.done", item=_ns(type="reasoning", id="rs_1", encrypted_content="ENC", summary=[])),
        _ns(type="response.output_text.delta", delta="hi"),
        _ns(type="response.completed", response=completed),
    ]
    provider.async_client = _ns(responses=_FakeResponses(_FakeStream(events)))

    wrapper = await provider.stream(
        MessageHistory([Message(role="user", content=[TextBlock(text="x")])]),
        params=GenerationParams(thinking="high"),
        model="gpt-5.5",
    )
    async with wrapper as stream:
        async for _chunk in stream:
            pass
    message = await wrapper.get_final_message()

    assert isinstance(message.content[0], ResponsesReasoningBlock)
    assert message.content[0].encrypted_content == "ENC"
