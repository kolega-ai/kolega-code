"""Credential-gated live smoke for deepseek-v4-flash over the **Responses API**.

`deepseek-v4-flash` is the one DeepSeek model routed to the Responses API (see
DeepSeekResponsesProvider); every other DeepSeek model stays on Chat Completions.
These tests hit the real ``https://api.deepseek.com`` Responses endpoint, so they are
marked ``slow``/``integration`` and skip when ``DEEPSEEK_API_KEY`` is unset. Run with:

    ./run_tests.sh --all tests/agent/llm/test_deepseek_responses_live.py
"""

import os

import pytest

from kolega_code.llm.client import LLMClient
from kolega_code.llm.models import Message, MessageHistory, TextBlock, ToolCall, ToolDefinition, ToolParameter
from kolega_code.llm.providers.deepseek_responses import DeepSeekResponsesProvider

pytestmark = [pytest.mark.slow, pytest.mark.integration]

_MODEL = "deepseek-v4-flash"


@pytest.fixture
def deepseek_flash_client() -> LLMClient:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        pytest.skip("DEEPSEEK_API_KEY not set")
    return LLMClient(provider="deepseek", api_key=api_key, model=_MODEL)


def test_flash_client_routes_to_responses_provider(deepseek_flash_client):
    # The live client must actually be on the Responses transport (bare host, no /v1),
    # otherwise the tests below would silently exercise Chat Completions instead.
    provider = deepseek_flash_client.provider
    assert isinstance(provider, DeepSeekResponsesProvider)
    assert provider.base_url == "https://api.deepseek.com"
    assert provider.provider_name == "deepseek"


@pytest.mark.asyncio
async def test_generate_text_over_responses_api(deepseek_flash_client):
    messages = MessageHistory([Message("user", [TextBlock("Reply with exactly: deepseek-ok")])])
    system = Message("system", [TextBlock("Follow the user's instruction exactly.")])

    response = await deepseek_flash_client.generate(
        messages=messages,
        system=system,
        model=_MODEL,
        thinking="high",
        temperature=1.0,
    )

    assert isinstance(response, Message)
    assert response.role == "assistant"
    # Reasoning is on by default; assert on extracted text rather than a fixed index.
    assert response.get_text_content().strip()

    usage = response.usage_metadata or {}
    assert usage.get("provider") == "deepseek"
    # The Responses path reports prompt/completion/total tokens; reasoning tokens are
    # folded into completion_tokens (DeepSeek bills thinking at the output rate).
    assert usage.get("completion_tokens", 0) > 0
    assert usage.get("prompt_tokens", 0) > 0


@pytest.mark.asyncio
async def test_tool_call_over_responses_api(deepseek_flash_client):
    # The main reason to move flash to Responses is the native tool-calling format —
    # exercise a real function-call round trip.
    weather_tool = ToolDefinition(
        name="get_weather",
        description="Get the current weather for a city.",
        parameters=[ToolParameter(name="city", type="string", description="City name", required=True)],
    )
    messages = MessageHistory(
        [Message("user", [TextBlock("What is the weather in Paris right now? Use the get_weather tool.")])]
    )
    system = Message("system", [TextBlock("You are a helpful assistant. Use the provided tools when relevant.")])

    response = await deepseek_flash_client.generate(
        messages=messages,
        system=system,
        model=_MODEL,
        tools=[weather_tool],
        thinking="high",
        temperature=1.0,
    )

    assert isinstance(response, Message)
    tool_calls = [block for block in response.content if isinstance(block, ToolCall)]
    assert tool_calls, f"expected a get_weather tool call, got: {[type(b).__name__ for b in response.content]}"
    call = tool_calls[0]
    assert call.name == "get_weather"
    # Native Responses function args parse to a dict with the declared parameter.
    assert isinstance(call.input, dict)
    assert "city" in call.input
    assert response.stop_reason == "tool_use"
