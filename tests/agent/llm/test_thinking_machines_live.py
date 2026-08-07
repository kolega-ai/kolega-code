"""Live integration tests for the thinking_machines (Tinker) provider.

These hit the real Thinking Machines API through Kolega's Anthropic-compatible
provider route, so they are marked ``integration`` and skip unless
``TINKER_API_KEY`` is present (loaded from the repo ``.env`` by ``conftest.py``).
The auto-discovered smoke tests in ``test_live_providers.py`` already cover
basic generate + thinking effort; this file exercises what they don't: the
tool-call round trip, streaming reasoning capture, reasoning off, image input,
and ``count_tokens``.

Run with::

    pytest -m integration tests/agent/llm/test_thinking_machines_live.py -v
"""

import os

import pytest

from kolega_code.llm.client import LLMClient
from kolega_code.llm.exceptions import LLMBillingError, LLMRateLimitError
from kolega_code.llm.models import (
    ImageBlock,
    Message,
    MessageHistory,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)

pytestmark = pytest.mark.integration

SKIP_IN_CI = bool(os.getenv("CI")) or bool(os.getenv("GITLAB_CI"))

MODEL = "thinkingmachines/Inkling"
PNG_1PX_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

SYSTEM = Message(role="system", content=[TextBlock(text="You are a concise, helpful assistant.")])


def _require_key() -> str:
    if SKIP_IN_CI:
        pytest.skip("Skipping live provider call in CI")
    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        pytest.skip("TINKER_API_KEY not set")
    return api_key


async def _live_generate(client: LLMClient, **generate_kwargs):
    """Run a live ``generate`` call, skipping on provider-side quota exhaustion."""
    try:
        return await client.generate(**generate_kwargs)
    except (LLMRateLimitError, LLMBillingError) as exc:
        pytest.skip(f"provider quota exhausted for this key: {exc}")


@pytest.mark.asyncio
async def test_live_tool_call_round_trip() -> None:
    """The model requests a tool, and answers correctly after the result is fed back."""
    api_key = _require_key()
    client = LLMClient(provider="thinking_machines", api_key=api_key)

    tools = [
        ToolDefinition(
            name="get_weather",
            description="Get the current weather for a city",
            parameters=[ToolParameter(name="city", type="string", description="City name", required=True)],
        )
    ]
    first = await _live_generate(
        client,
        messages=MessageHistory(
            [Message(role="user", content=[TextBlock(text="What is the weather in Paris? Use the tool.")])]
        ),
        system=SYSTEM,
        model=MODEL,
        max_completion_tokens=4096,
        temperature=1.0,
        thinking="medium",
        tools=tools,
    )
    tool_calls = [b for b in (first.content or []) if isinstance(b, ToolCall)]
    assert tool_calls, f"model did not call the tool: {first.get_text_content()!r}"
    tool_call = tool_calls[0]
    assert tool_call.name == "get_weather"
    assert isinstance(tool_call.input, dict) and "city" in tool_call.input

    follow_up = MessageHistory(
        [
            Message(role="user", content=[TextBlock(text="What is the weather in Paris? Use the tool.")]),
            first,
            Message(
                role="user",
                content=[
                    ToolResult(
                        tool_use_id=tool_call.id,
                        content=[TextBlock(text='{"city": "Paris", "temp": 22, "condition": "sunny"}')],
                        name=tool_call.name,
                        is_error=False,
                    )
                ],
            ),
        ]
    )
    final = await _live_generate(
        client,
        messages=follow_up,
        system=SYSTEM,
        model=MODEL,
        max_completion_tokens=4096,
        temperature=1.0,
        thinking="medium",
        tools=tools,
    )
    text = final.get_text_content()
    assert text, "empty reply after tool result"
    assert "22" in text, f"reply did not use the tool result: {text!r}"


@pytest.mark.asyncio
async def test_live_reasoning_is_captured_and_can_be_disabled() -> None:
    """Thinking blocks stream and are captured; effort 'none' disables them."""
    api_key = _require_key()
    client = LLMClient(provider="thinking_machines", api_key=api_key)

    with_thinking = await _live_generate(
        client,
        messages=MessageHistory(
            [Message(role="user", content=[TextBlock(text="What is 17 * 23? Reply with just the number.")])]
        ),
        system=SYSTEM,
        model=MODEL,
        max_completion_tokens=4096,
        temperature=1.0,
        thinking="max",
    )
    thinking = [b for b in (with_thinking.content or []) if isinstance(b, ThinkingBlock)]
    assert thinking, "expected thinking blocks at effort=max"
    assert any(b.thinking for b in thinking), "thinking blocks are empty"
    assert with_thinking.get_text_content()

    without = await _live_generate(
        client,
        messages=MessageHistory([Message(role="user", content=[TextBlock(text="Say hello in one word.")])]),
        system=SYSTEM,
        model=MODEL,
        max_completion_tokens=2048,
        temperature=1.0,
        thinking="none",
    )
    thinking_off = [b for b in (without.content or []) if isinstance(b, ThinkingBlock)]
    assert not thinking_off, "effort=none must disable thinking"
    assert without.get_text_content()


@pytest.mark.asyncio
async def test_live_image_input() -> None:
    """The multimodal model accepts image inputs through the standard block shape."""
    api_key = _require_key()
    client = LLMClient(provider="thinking_machines", api_key=api_key)

    response = await _live_generate(
        client,
        messages=MessageHistory(
            [
                Message(
                    role="user",
                    content=[
                        ImageBlock(image_type="base64", media_type="image/png", data=PNG_1PX_BASE64),
                        TextBlock(text="Describe this image in one sentence."),
                    ],
                )
            ]
        ),
        system=SYSTEM,
        model=MODEL,
        max_completion_tokens=4096,
        temperature=1.0,
        thinking="low",
    )
    assert response.get_text_content(), "empty reply to image input"


@pytest.mark.asyncio
async def test_live_count_tokens() -> None:
    """Tinker supports the Anthropic /v1/messages/count_tokens endpoint."""
    api_key = _require_key()
    client = LLMClient(provider="thinking_machines", api_key=api_key)

    count = await client.count_tokens(
        messages=MessageHistory([Message(role="user", content=[TextBlock(text="What is 2 + 2?")])]),
        system=SYSTEM,
        model=MODEL,
    )
    assert count.input_tokens > 0
