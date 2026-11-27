"""Opt-in live API integration tests for Vercel AI Gateway.

Run with:
    KOLEGA_TEST_LIVE_API=1 ./run_tests.sh tests/agent/llm/test_vercel_ai_gateway_live.py -n 0 -ra
"""

from __future__ import annotations

import os
from typing import Any, cast

import pytest

from kolega_code.cli.config import provider_api_key_env_names
from kolega_code.config import ModelProvider
from kolega_code.llm.accounting import reported_cost
from kolega_code.llm.client import LLMClient
from kolega_code.llm.exceptions import LLMBillingError, LLMNotFoundError
from kolega_code.llm.models import (
    ContentBlock,
    ImageBlock,
    Message,
    MessageHistory,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolResult,
)

pytestmark = [pytest.mark.integration, pytest.mark.live_api]

PROVIDER = ModelProvider.VERCEL_AI_GATEWAY.value
DEFAULT_MODEL = "moonshotai/kimi-k2.7-code"
CACHE_MODEL = "alibaba/qwen3.7-flash"
VISION_MODEL = "google/gemini-2.5-flash"

# 1x1 red PNG
_RED_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


def _require_live_key() -> str:
    if os.getenv("CI") or os.getenv("GITLAB_CI"):
        pytest.skip("Skipping live provider call in CI")
    if os.getenv("KOLEGA_TEST_LIVE_API") != "1":
        pytest.skip("Set KOLEGA_TEST_LIVE_API=1 to run live Vercel AI Gateway tests")
    for env_name in provider_api_key_env_names(ModelProvider.VERCEL_AI_GATEWAY):
        key = os.getenv(env_name)
        if key:
            return key
    pytest.skip("VERCEL_API_KEY / AI_GATEWAY_API_KEY not set")


@pytest.mark.asyncio
async def test_live_streaming_and_reported_cost() -> None:
    client = LLMClient(provider=PROVIDER, api_key=_require_live_key(), max_retries=1)
    stream = await cast(
        Any,
        client.stream(
            messages=MessageHistory([Message(role="user", content=[TextBlock(text="Reply with just 4.")])]),
            system=Message(role="system", content=[TextBlock(text="Be terse.")]),
            model=DEFAULT_MODEL,
            max_completion_tokens=128,
            thinking="low",
        ),
    )
    async with stream:
        async for _ in stream:
            pass
    message = await stream.get_final_message()
    assert message.usage is not None
    assert message.usage.reported is True
    assert (message.usage.input_tokens or 0) > 0
    assert (message.usage.output_tokens or 0) > 0
    cost = reported_cost(PROVIDER, message.usage_metadata)
    assert cost is not None
    assert cost["source"] == "provider_reported"


@pytest.mark.asyncio
async def test_live_tool_call_with_reasoning_replay() -> None:
    client = LLMClient(provider=PROVIDER, api_key=_require_live_key(), max_retries=1)
    weather_tool = ToolDefinition(
        name="get_weather",
        description="Get weather for a city",
        parameters=[],
        input_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )
    history = MessageHistory(
        [Message(role="user", content=[TextBlock(text="Call get_weather for Dublin, then tell me the result.")])]
    )
    turn1 = await client.generate(
        messages=history,
        system=Message(role="system", content=[TextBlock(text="Always call get_weather when asked for weather.")]),
        model=DEFAULT_MODEL,
        tools=[weather_tool],
        max_completion_tokens=256,
        thinking="low",
    )
    tool_calls = [block for block in turn1.content if isinstance(block, ToolCall)]
    assert tool_calls, f"Expected a tool call, got: {turn1.content}"

    tc = tool_calls[0]
    history.append(turn1)
    tool_result_blocks: list[ContentBlock] = [
        ToolResult(tool_use_id=tc.id, name=tc.name, content="12C and cloudy", is_error=False)
    ]
    history.append(Message(role="user", content=tool_result_blocks))
    turn2 = await client.generate(
        messages=history,
        system=Message(role="system", content=[TextBlock(text="Be terse.")]),
        model=DEFAULT_MODEL,
        tools=[weather_tool],
        max_completion_tokens=128,
        thinking="low",
    )
    text = " ".join(block.text for block in turn2.content if isinstance(block, TextBlock))
    assert "12" in text or "cloudy" in text.lower()


@pytest.mark.asyncio
async def test_live_vision_and_prompt_caching() -> None:
    key = _require_live_key()
    client = LLMClient(provider=PROVIDER, api_key=key, max_retries=1)

    vision_blocks: list[ContentBlock] = [
        TextBlock(text="Name the color of this pixel in one word."),
        ImageBlock(image_type="base64", media_type="image/png", data=_RED_PNG_B64),
    ]
    vis = await client.generate(
        messages=MessageHistory([Message(role="user", content=vision_blocks)]),
        model=VISION_MODEL,
        max_completion_tokens=32,
        thinking="none",
    )
    assert vis.usage is not None and vis.usage.reported is True

    prefix = "Reference guide for Kolega Code Vercel AI Gateway prompt caching live verification. " * 260
    cache_client = LLMClient(provider=PROVIDER, api_key=key, max_retries=1)
    first = await cache_client.generate(
        messages=MessageHistory([Message(role="user", content=[TextBlock(text="Reply with 1.")])]),
        system=Message(role="system", content=[TextBlock(text=prefix)]),
        model=CACHE_MODEL,
        max_completion_tokens=16,
        thinking="none",
    )
    second = await cache_client.generate(
        messages=MessageHistory([Message(role="user", content=[TextBlock(text="Reply with 2.")])]),
        system=Message(role="system", content=[TextBlock(text=prefix)]),
        model=CACHE_MODEL,
        max_completion_tokens=16,
        thinking="none",
    )
    assert first.usage is not None and second.usage is not None
    assert (first.usage.cache_write_input_tokens or 0) > 0 or (second.usage.cache_read_input_tokens or 0) > 0


@pytest.mark.asyncio
async def test_live_unknown_model_and_optional_free_tier_restriction() -> None:
    client = LLMClient(provider=PROVIDER, api_key=_require_live_key(), max_retries=0)
    with pytest.raises(LLMNotFoundError):
        await client.generate(
            messages=MessageHistory([Message(role="user", content=[TextBlock(text="hi")])]),
            model="nonexistent/not-a-model",
            max_completion_tokens=16,
        )

    # On free-tier accounts, paid frontier models raise LLMBillingError; on paid accounts, verify Anthropic + GPT-6.2 + DeepSeek.
    try:
        anthropic_resp = await client.generate(
            messages=MessageHistory([Message(role="user", content=[TextBlock(text="Reply with OK.")])]),
            model="anthropic/claude-sonnet-4.6",
            max_completion_tokens=16,
            thinking="none",
        )
        assert anthropic_resp.usage is not None and anthropic_resp.usage.reported is True

        gpt62_resp = await client.generate(
            messages=MessageHistory([Message(role="user", content=[TextBlock(text="Reply with OK.")])]),
            model="openai/gpt-6.2-sol",
            max_completion_tokens=64,
            thinking="low",
        )
        assert gpt62_resp.usage is not None and gpt62_resp.usage.reported is True

        ds_off = await client.generate(
            messages=MessageHistory(
                [Message(role="user", content=[TextBlock(text="What is 2+2? Reply with just 4.")])]
            ),
            model="deepseek/deepseek-v4.1-flash",
            max_completion_tokens=16,
            thinking="none",
        )
        assert ds_off.usage is not None and ds_off.usage.reported is True
        assert (ds_off.usage.reasoning_output_tokens or 0) == 0
    except LLMBillingError:
        pass
