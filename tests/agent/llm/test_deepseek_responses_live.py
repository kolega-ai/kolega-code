"""Credential-gated live smoke for DeepSeek over the **Responses API**.

The first-party ``deepseek`` provider (``deepseek-flash`` and ``deepseek-v4-pro``,
the only two models DeepSeek still lists) routes to the Responses API (see
DeepSeekResponsesProvider); DeepSeek models on other providers stay on Chat
Completions. These tests hit the real ``https://api.deepseek.com`` Responses
endpoint, so they are marked ``slow``/``integration`` and skip when
``DEEPSEEK_API_KEY`` is unset. Run with:

    ./run_tests.sh --all tests/agent/llm/test_deepseek_responses_live.py
"""

import os

import pytest

from kolega_code.llm.client import LLMClient
from kolega_code.llm.models import (
    ImageBlock,
    Message,
    MessageHistory,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolParameter,
    WebSearchCallBlock,
)
from kolega_code.llm.providers.deepseek_responses import DeepSeekResponsesProvider

pytestmark = [pytest.mark.slow, pytest.mark.integration]

_MODEL = "deepseek-flash"
# The released V4.1 Flash is multimodal, so the image-path tests use the same id;
# kept as a named constant to make those sections' intent explicit.
_VISION_MODEL = _MODEL
_PRO_MODEL = "deepseek-v4-pro"

# 64x64 solid red PNG (generated with PIL, base64-encoded).
_RED_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAf0lEQVR4nNXOQREA"
    "IAzAsFIN868HWYjgsWsU5NwZyiRO4iRO4iRO4iRO4iRO4iRO4iRO4iRO4iRO4iRO"
    "4iRO4iRO4iRO4iRO4iRO4iRO4iRO4iRO4iRO4iRO4iRO4iRO4iRO4iRO4iRO4iRO"
    "4iRO4iRO4iRO4iRO4iRO4twO/Hp+awFwlXGz8wAAAABJRU5ErkJggg=="
)


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


# --- deepseek-flash (first multimodal DeepSeek model) ----------------


@pytest.fixture
def deepseek_vision_client() -> LLMClient:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        pytest.skip("DEEPSEEK_API_KEY not set")
    return LLMClient(provider="deepseek", api_key=api_key, model=_VISION_MODEL)


def test_vision_client_routes_to_responses_provider(deepseek_vision_client):
    # The live client must actually be on the Responses transport (bare host, no /v1).
    provider = deepseek_vision_client.provider
    assert isinstance(provider, DeepSeekResponsesProvider)
    assert provider.base_url == "https://api.deepseek.com"
    assert provider.provider_name == "deepseek"


@pytest.mark.asyncio
async def test_vision_generate_text_with_thinking(deepseek_vision_client):
    messages = MessageHistory([Message("user", [TextBlock("Reply with exactly: vision-ok")])])
    system = Message("system", [TextBlock("Follow the user's instruction exactly.")])

    response = await deepseek_vision_client.generate(
        messages=messages,
        system=system,
        model=_VISION_MODEL,
        thinking="high",
        temperature=1.0,
    )

    assert isinstance(response, Message)
    assert response.get_text_content().strip() == "vision-ok"
    usage = response.usage_metadata or {}
    assert usage.get("provider") == "deepseek"
    assert usage.get("completion_tokens", 0) > 0


@pytest.mark.asyncio
async def test_vision_image_input_identifies_color(deepseek_vision_client):
    # Real multimodal check: a solid red PNG must come back as "red" — verifies the
    # base64 input_image path (DeepSeek bills images as tokens; one <= 384 tokens).
    image = ImageBlock(image_type="base64", media_type="image/png", data=_RED_PNG_B64)
    messages = MessageHistory(
        [Message("user", [TextBlock("What color is this image? Reply with one color name only."), image])]
    )

    response = await deepseek_vision_client.generate(
        messages=messages,
        system=None,
        model=_VISION_MODEL,
        thinking="none",
        temperature=0.0,
    )

    assert isinstance(response, Message)
    assert "red" in response.get_text_content().strip().lower()


@pytest.fixture
def deepseek_pro_client() -> LLMClient:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        pytest.skip("DEEPSEEK_API_KEY not set")
    return LLMClient(provider="deepseek", api_key=api_key, model=_PRO_MODEL)


@pytest.mark.asyncio
async def test_pro_hosted_web_search(deepseek_pro_client):
    # Hosted search is gone from the flash architecture (V4.1 removed it silently: the
    # docs' tool table moved web_search to "Ignored"), but the V4 Pro model still
    # executes it, so this is the live coverage of the server-side path. NOTE: DeepSeek
    # routes deepseek-v4-pro to V4.1 Flash from 12:00 Beijing Time 2026-09-14 — re-check
    # this test and supports_hosted_web_search for pro after that date.
    # Forcing a page open, rather than asking a question the model could answer from
    # memory, is what makes the assertion test the tool instead of the model's recall.
    messages = MessageHistory(
        [
            Message(
                "user",
                [
                    TextBlock(
                        "Use the web search tool to open https://blog.rust-lang.org/releases/ and reply with "
                        "just the newest release version listed there."
                    )
                ],
            )
        ]
    )

    response = await deepseek_pro_client.generate(
        messages=messages,
        system=None,
        model=_PRO_MODEL,
        thinking="none",
        max_completion_tokens=2048,
        hosted_web_search=True,
    )

    assert isinstance(response, Message)
    assert any(isinstance(block, WebSearchCallBlock) for block in response.content)
    assert response.get_text_content().strip()


@pytest.mark.asyncio
async def test_vision_accepts_full_output_cap(deepseek_vision_client):
    # Flash-family ceiling: the server accepts max_output_tokens=384000 (the
    # value the unclamped catalog entry puts on the wire when no cap is requested).
    messages = MessageHistory([Message("user", [TextBlock("Say ok")])])

    response = await deepseek_vision_client.generate(
        messages=messages,
        system=None,
        model=_VISION_MODEL,
        thinking="none",
        temperature=0.0,
        max_completion_tokens=384000,
    )

    assert isinstance(response, Message)
    assert response.get_text_content().strip()
