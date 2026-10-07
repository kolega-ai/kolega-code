"""Small, key-gated native Mistral live tests.

Disabled unless ``KOLEGA_TEST_LIVE_API=1`` and ``MISTRAL_API_KEY`` are loaded
with python-dotenv. Do not source the repo .env in a shell.
"""

from __future__ import annotations

import base64
import io
import os
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, cast

import pytest
import pytest_asyncio
from dotenv import load_dotenv
from PIL import Image

from kolega_code.llm.client import LLMClient
from kolega_code.llm.exceptions import LLMBillingError, LLMPermissionDeniedError, LLMRateLimitError
from kolega_code.llm.models import (
    ImageBlock,
    Message,
    MessageHistory,
    TextBlock,
    ThinkingBlock,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)
from kolega_code.llm.providers.mistral import MistralProvider

pytestmark = [pytest.mark.integration, pytest.mark.live_api]

MODEL_REASONING = "mistral-medium-3-5"
MODELS_REASONING = (MODEL_REASONING, "mistral-small-2603", "mistral-large-4")
MODEL_FAST = "ministral-3b-2512"
MAX_TOKENS_SMALL = 512
MAX_TOKENS_REASONING = 2048


def _load_key() -> str:
    if os.getenv("KOLEGA_TEST_LIVE_API") != "1":
        pytest.skip("set KOLEGA_TEST_LIVE_API=1 to run native Mistral live tests")
    repo_root = Path(__file__).resolve().parents[3]
    load_dotenv(repo_root / ".env", override=False)
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        pytest.skip("MISTRAL_API_KEY not set")
    return api_key


@pytest_asyncio.fixture
async def client_factory() -> AsyncIterator[Callable[[str], LLMClient]]:
    clients: list[LLMClient] = []

    def create(model: str) -> LLMClient:
        client = LLMClient(provider="mistral", api_key=_load_key(), model=model, max_retries=0)
        clients.append(client)
        return client

    try:
        yield create
    finally:
        for client in clients:
            assert isinstance(client.provider, MistralProvider)
            await client.provider.async_client.aclose()


def _history(text: str) -> MessageHistory:
    return MessageHistory([Message("user", [TextBlock(text)])])


async def _live_generate(client: LLMClient, **kwargs: Any) -> Message:
    try:
        message = await client.generate(**kwargs)
        _assert_usage(message)
        return message
    except (LLMRateLimitError, LLMBillingError, LLMPermissionDeniedError) as exc:
        pytest.skip(f"native Mistral live API not verified for this account/model: {type(exc).__name__}: {exc}")


async def _live_stream(client: LLMClient, **kwargs: Any) -> Message:
    try:
        cm = await cast(Any, client.stream(**kwargs))
        async with cm as stream:
            chunks = [chunk async for chunk in stream]
        message = await stream.get_final_message()
        assert "".join(chunk.text or "" for chunk in chunks) == message.get_text_content()
        assert "".join(chunk.thinking or "" for chunk in chunks) == "".join(
            block.thinking for block in _thinking_blocks(message)
        )
        _assert_usage(message)
        return message
    except (LLMRateLimitError, LLMBillingError, LLMPermissionDeniedError) as exc:
        pytest.skip(f"native Mistral live stream not verified for this account/model: {type(exc).__name__}: {exc}")


def _thinking_blocks(message: Message) -> list[ThinkingBlock]:
    return [
        block
        for block in (message.content if isinstance(message.content, list) else [])
        if isinstance(block, ThinkingBlock)
    ]


def _assert_usage(message: Message) -> None:
    assert message.usage is not None and message.usage.reported
    assert message.usage.provider == "mistral"
    assert message.usage.input_tokens is not None and message.usage.input_tokens > 0
    assert message.usage.output_tokens is not None and message.usage.output_tokens > 0
    assert message.usage.total_tokens == message.usage.input_tokens + message.usage.output_tokens


@pytest.mark.asyncio
@pytest.mark.parametrize("thinking", ["none", "high"])
@pytest.mark.parametrize("model", MODELS_REASONING)
async def test_live_native_generate_reasoning_none_and_high(
    thinking: str, model: str, client_factory: Callable[[str], LLMClient]
) -> None:
    client = client_factory(model)
    response = await _live_generate(
        client,
        messages=_history("Answer in one short sentence: what is 2+2?"),
        model=model,
        max_completion_tokens=MAX_TOKENS_REASONING if thinking == "high" else MAX_TOKENS_SMALL,
        thinking=thinking,
        temperature=0,
    )
    assert response.role == "assistant"
    assert response.get_text_content().strip()
    blocks = _thinking_blocks(response)
    if thinking == "high":
        assert blocks, "high reasoning must return an actual ThinkingBlock"
        assert any(block.thinking.strip() for block in blocks)
    else:
        assert not blocks


@pytest.mark.asyncio
@pytest.mark.parametrize("thinking", ["none", "high"])
@pytest.mark.parametrize("model", MODELS_REASONING)
async def test_live_native_stream_reasoning_none_and_high(
    thinking: str, model: str, client_factory: Callable[[str], LLMClient]
) -> None:
    client = client_factory(model)
    response = await _live_stream(
        client,
        messages=_history("Reply with exactly: stream-ok"),
        model=model,
        max_completion_tokens=MAX_TOKENS_REASONING if thinking == "high" else MAX_TOKENS_SMALL,
        thinking=thinking,
        temperature=0,
    )
    assert "stream-ok" in response.get_text_content().lower()
    blocks = _thinking_blocks(response)
    if thinking == "high":
        assert blocks and any(block.thinking.strip() for block in blocks)
    else:
        assert not blocks


def _calc_tool() -> ToolDefinition:
    return ToolDefinition(
        name="add_small_ints",
        description="Add two small integers exactly.",
        parameters=[
            ToolParameter(name="a", type="integer", description="first integer", required=True),
            ToolParameter(name="b", type="integer", description="second integer", required=True),
        ],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [*MODELS_REASONING, "ministral-8b-2512", "codestral-2508"])
async def test_live_streaming_tool_cycle_then_second_round_native_reasoning_replay(
    model: str,
    client_factory: Callable[[str], LLMClient],
) -> None:
    client = client_factory(model)
    thinking = "high" if model in MODELS_REASONING else None
    tools = [_calc_tool()]
    history = MessageHistory(
        [Message("user", [TextBlock("Use the tool to add 17 and 25, then answer with only the sum.")])]
    )

    first = await _live_stream(
        client,
        messages=history,
        model=model,
        tools=tools,
        max_completion_tokens=MAX_TOKENS_REASONING,
        thinking=thinking,
        temperature=0,
        tool_choice="any",
        parallel_tool_calls=False,
    )
    assert first.tool_calls, "expected Mistral to request the local addition tool"
    call = first.tool_calls[0]
    assert call.name == "add_small_ints"
    assert isinstance(call.input, dict)
    result = call.input["a"] + call.input["b"]
    assert result == 42
    history.append(Message.from_dict(first.to_dict()))
    history.append(
        Message(
            "user",
            [
                ToolResult(
                    tool_use_id=call.id,
                    execution_id=call.execution_id,
                    name=call.name,
                    content=str(result),
                    is_error=False,
                )
            ],
        )
    )
    second = await _live_generate(
        client,
        messages=history,
        model=model,
        tools=tools,
        max_completion_tokens=MAX_TOKENS_REASONING,
        thinking=thinking,
        temperature=0,
    )
    assert "42" in second.get_text_content()
    if thinking:
        assert _thinking_blocks(first), "first tool round must expose actual native reasoning"
        assert _thinking_blocks(second), "second tool round should preserve same-provider native reasoning replay"
    # Resume the completed exchange and execute another call: wire IDs may
    # repeat across responses but durable execution identities must not.
    history.append(Message.from_dict(second.to_dict()))
    history.append(Message("user", "Now use the tool to add 42 and 1. Answer only the new sum."))
    third = await _live_stream(
        client,
        messages=history,
        model=model,
        tools=tools,
        max_completion_tokens=MAX_TOKENS_REASONING,
        thinking=thinking,
        temperature=0,
        tool_choice="any",
        parallel_tool_calls=False,
    )
    assert len(third.tool_calls) == 1
    third_call = third.tool_calls[0]
    assert third_call.name == "add_small_ints" and isinstance(third_call.input, dict)
    assert third_call.input["a"] + third_call.input["b"] == 43
    assert third_call.execution_id != call.execution_id
    history.extend(
        [
            Message.from_dict(third.to_dict()),
            Message(
                "user",
                [
                    ToolResult(
                        tool_use_id=third_call.id,
                        execution_id=third_call.execution_id,
                        name=third_call.name,
                        content="43",
                        is_error=False,
                    )
                ],
            ),
        ]
    )
    fourth = await _live_stream(
        client,
        messages=history,
        model=model,
        tools=tools,
        max_completion_tokens=MAX_TOKENS_REASONING,
        thinking=thinking,
        temperature=0,
    )
    assert "43" in fourth.get_text_content()


def _red_blue_png_b64() -> str:
    # Large solid panels stay unambiguous through the provider's image resizing.
    image = Image.new("RGB", (128, 64), (255, 0, 0))
    image.paste((0, 0, 255), (64, 0, 128, 64))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [*MODELS_REASONING, MODEL_FAST, "ministral-8b-2512", "ministral-14b-2512"])
async def test_live_vision_generated_local_png_exact_red_blue_content(
    model: str,
    client_factory: Callable[[str], LLMClient],
) -> None:
    client = client_factory(model)
    image = ImageBlock(image_type="base64", media_type="image/png", data=_red_blue_png_b64())
    response = await _live_generate(
        client,
        messages=MessageHistory(
            [Message("user", [TextBlock("What colors are the two panels, from left to right?"), image])]
        ),
        model=model,
        max_completion_tokens=MAX_TOKENS_SMALL,
        thinking="none" if model in MODELS_REASONING else None,
        temperature=0,
    )
    answer = response.get_text_content().lower()
    assert "red" in answer and "blue" in answer
    assert answer.find("red") < answer.find("blue")


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [MODEL_REASONING, "mistral-large-4", "ministral-8b-2512"])
async def test_live_tool_image_result_native_replay_if_supported(
    model: str, client_factory: Callable[[str], LLMClient]
) -> None:
    client = client_factory(model)
    thinking = "high" if model in MODELS_REASONING else None
    image_tool = ToolDefinition(
        name="make_reference_image",
        description="Return a PNG reference image.",
        parameters=[],
    )
    history = MessageHistory(
        [
            Message(
                "user",
                "Call the reference image tool once. Read its attached image, which contains two solid-color "
                "panels side by side. Identify both panel colors from left to right. Do not generate another image.",
            )
        ]
    )
    first = await _live_stream(
        client,
        messages=history,
        model=model,
        tools=[image_tool],
        max_completion_tokens=MAX_TOKENS_REASONING,
        thinking=thinking,
        temperature=0,
        tool_choice={"type": "function", "function": {"name": "make_reference_image"}},
        parallel_tool_calls=False,
    )
    assert first.tool_calls
    assert all(call.name == "make_reference_image" for call in first.tool_calls)
    assert len({call.execution_id for call in first.tool_calls}) == len(first.tool_calls)
    history.append(Message.from_dict(first.to_dict()))
    history.append(
        Message(
            "user",
            [
                ToolResult(
                    tool_use_id=call.id,
                    execution_id=call.execution_id,
                    name=call.name,
                    content=[TextBlock("reference image"), ImageBlock("base64", "image/png", _red_blue_png_b64())],
                    is_error=False,
                )
                for call in first.tool_calls
            ],
        )
    )
    second = await _live_generate(
        client,
        messages=history,
        model=model,
        tools=[image_tool],
        max_completion_tokens=MAX_TOKENS_REASONING,
        thinking=thinking,
        temperature=0,
        tool_choice="none",
    )
    answer = second.get_text_content().lower()
    assert "red" in answer and "blue" in answer
    assert answer.find("red") < answer.find("blue")


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [MODEL_FAST, "ministral-8b-2512", "ministral-14b-2512", "codestral-2508"])
async def test_live_nonreasoning_models_smoke(model: str, client_factory: Callable[[str], LLMClient]) -> None:
    client = client_factory(model)
    response = await _live_generate(
        client,
        messages=_history("Reply with exactly: ok"),
        model=model,
        max_completion_tokens=64,
        temperature=0,
    )
    assert "ok" in response.get_text_content().lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        "Explain why unit tests should isolate external state. " * 30,
        "中文、日本語、한국어、🙂🚀 — Unicode must be counted. " * 30,
        "def build_request(messages: list[dict[str, str]]) -> str:\n    return json.dumps(messages)\n" * 30,
    ],
    ids=["prose", "unicode", "code"],
)
async def test_live_local_count_has_margin_over_server_usage(
    content: str, client_factory: Callable[[str], LLMClient]
) -> None:
    client = client_factory("ministral-8b-2512")
    history = _history(content + "\nReply only: counted")
    system = Message("system", "You are concise.")
    count = await client.count_tokens(messages=history, system=system, tools=[_calc_tool()], model="ministral-8b-2512")
    response = await _live_generate(
        client,
        messages=history,
        system=system,
        tools=[_calc_tool()],
        model="ministral-8b-2512",
        max_completion_tokens=64,
        temperature=0,
    )
    assert response.usage is not None and response.usage.input_tokens is not None
    assert count.input_tokens >= response.usage.input_tokens
