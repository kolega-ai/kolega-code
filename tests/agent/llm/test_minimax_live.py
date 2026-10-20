"""Key-gated live tests for the first-party MiniMax provider.

These hit ``https://api.minimax.io/anthropic`` for real. Opt in with
``KOLEGA_TEST_LIVE_API=1`` and a ``MINIMAX_API_KEY`` in the repo-root ``.env``
(loaded with python-dotenv — never ``source`` it):

    KOLEGA_TEST_LIVE_API=1 ./run_tests.sh --all tests/agent/llm/test_minimax_live.py

Each assertion mirrors a probe recorded in ``specs/catalog/minimax.py``; when the
API changes, these fail before the catalog misleads an agent.
"""

from __future__ import annotations

import base64
import os
import struct
import zlib
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, cast

import pytest
import pytest_asyncio
from dotenv import load_dotenv

from kolega_code.llm.client import LLMClient
from kolega_code.llm.exceptions import (
    LLMBadRequestError,
    LLMBillingError,
    LLMInvalidRequestError,
    LLMPermissionDeniedError,
    LLMRateLimitError,
    LLMUnsupportedParamsError,
)
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
from kolega_code.llm.providers.anthropic import AnthropicProvider

pytestmark = [pytest.mark.integration, pytest.mark.live_api]

FLASH = "MiniMax-M3.1-Flash"
M3 = "MiniMax-M3"
M2_7 = "MiniMax-M2.7"
MAX_TOKENS = 2048
EFFORTS = ("low", "medium", "high", "xhigh", "max")
# Errors that mean "this account/model cannot be probed right now", not "kolega is wrong".
_UNAVAILABLE = (LLMRateLimitError, LLMBillingError, LLMPermissionDeniedError)
_REJECTED = (LLMBadRequestError, LLMInvalidRequestError, LLMUnsupportedParamsError)


def _red_png_b64(size: int = 48) -> str:
    """A valid solid-red PNG, built rather than transcribed (byte-exact by construction)."""
    raw = b"".join(b"\x00" + bytes((220, 30, 30)) * size for _ in range(size))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode()


def _load_key() -> str:
    if os.getenv("KOLEGA_TEST_LIVE_API") != "1":
        pytest.skip("set KOLEGA_TEST_LIVE_API=1 to run MiniMax live tests")
    load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)
    api_key = os.getenv("MINIMAX_API_KEY")
    if not api_key:
        pytest.skip("MINIMAX_API_KEY not set")
    return api_key


@pytest_asyncio.fixture
async def client_factory() -> AsyncIterator[Callable[[str], LLMClient]]:
    clients: list[LLMClient] = []

    def create(model: str) -> LLMClient:
        client = LLMClient(provider="minimax", api_key=_load_key(), model=model, max_retries=0)
        clients.append(client)
        return client

    try:
        yield create
    finally:
        for client in clients:
            assert isinstance(client.provider, AnthropicProvider)
            await cast(Any, client.provider.async_client).close()


def _history(text: str) -> MessageHistory:
    return MessageHistory([Message("user", [TextBlock(text)])])


def _system() -> Message:
    # The Anthropic provider family requires a system message on every request.
    return Message("system", [TextBlock("You are a helpful assistant.")])


def _thinking(message: Message) -> list[ThinkingBlock]:
    blocks = message.content if isinstance(message.content, list) else []
    return [block for block in blocks if isinstance(block, ThinkingBlock)]


def _usage(message: Message) -> None:
    assert message.usage is not None and message.usage.reported
    assert message.usage.provider == "minimax"
    assert message.usage.input_tokens is not None and message.usage.input_tokens > 0
    assert message.usage.output_tokens is not None
    assert message.usage.total_tokens == message.usage.input_tokens + message.usage.output_tokens


async def _generate(client: LLMClient, **kwargs: Any) -> Message:
    kwargs.setdefault("system", _system())
    try:
        message = await client.generate(**kwargs)
    except _UNAVAILABLE as exc:  # pragma: no cover - account-dependent
        pytest.skip(f"MiniMax live API unavailable: {type(exc).__name__}: {exc}")
    _usage(message)
    return message


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
async def test_live_flash_generate_reports_anthropic_usage(client_factory: Callable[[str], LLMClient]) -> None:
    client = client_factory(FLASH)
    message = await _generate(
        client,
        messages=_history("Reply with exactly: OK"),
        model=FLASH,
        max_completion_tokens=64,
    )
    assert "OK" in message.get_text_content()


@pytest.mark.asyncio
async def test_live_count_tokens_uses_the_provider_endpoint(client_factory: Callable[[str], LLMClient]) -> None:
    client = client_factory(FLASH)
    count = await client.count_tokens(_history("hello world"), system=_system())
    # Non-zero and provider-sourced: local tiktoken counting is disabled for this
    # provider precisely because /v1/messages/count_tokens answers.
    assert count.input_tokens > 0
    assert isinstance(client.provider, AnthropicProvider)
    assert client.provider.use_local_token_counting is False


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", EFFORTS)
async def test_live_flash_accepts_every_documented_effort(
    effort: str, client_factory: Callable[[str], LLMClient]
) -> None:
    client = client_factory(FLASH)
    message = await _generate(
        client,
        messages=_history(
            "A farmer has 17 sheep. All but 9 run away. How many remain? Think step by step, then answer."
        ),
        model=FLASH,
        max_completion_tokens=MAX_TOKENS,
        thinking=effort,
    )
    assert message.get_text_content().strip()


@pytest.mark.asyncio
async def test_live_flash_rejects_disabling_thinking(client_factory: Callable[[str], LLMClient]) -> None:
    # The endpoint answers 400 for both "none" effort and thinking.type=disabled,
    # because Flash requires adaptive thinking. Kolega rejects it before the wire
    # and the client surfaces that as a request error.
    client = client_factory(FLASH)
    with pytest.raises(LLMInvalidRequestError):
        await client.generate(
            messages=_history("Say OK"),
            system=_system(),
            model=FLASH,
            max_completion_tokens=64,
            thinking="none",
        )


@pytest.mark.asyncio
async def test_live_m3_emits_no_thinking_blocks(client_factory: Callable[[str], LLMClient]) -> None:
    client = client_factory(M3)
    message = await _generate(
        client,
        messages=_history(
            "A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. "
            "How much is the ball? Think it through."
        ),
        model=M3,
        max_completion_tokens=MAX_TOKENS,
    )
    assert _thinking(message) == [], "M3 does not surface reasoning as thinking blocks"
    assert message.get_text_content().strip()


@pytest.mark.asyncio
async def test_live_m2_7_streams_thinking_blocks(client_factory: Callable[[str], LLMClient]) -> None:
    client = client_factory(M2_7)
    message = await _generate(
        client,
        messages=_history("Say OK"),
        model=M2_7,
        max_completion_tokens=512,
    )
    assert _thinking(message), "M2.x reasoning is always on and surfaced as thinking blocks"


@pytest.mark.asyncio
async def test_live_tool_cycle_replays_thinking_blocks(client_factory: Callable[[str], LLMClient]) -> None:
    client = client_factory(FLASH)
    tools = [_calc_tool()]
    history = MessageHistory(
        [Message("user", [TextBlock("Use the tool to add 17 and 25, then answer with only the sum.")])]
    )
    first = await _generate(
        client,
        messages=history,
        model=FLASH,
        tools=tools,
        max_completion_tokens=MAX_TOKENS,
        thinking="low",
    )
    assert first.tool_calls, "expected MiniMax to request the local addition tool"
    call = first.tool_calls[0]
    assert call.name == "add_small_ints"
    assert isinstance(call.input, dict)
    total = int(call.input["a"]) + int(call.input["b"])
    assert total == 42

    history.append(Message.from_dict(first.to_dict()))
    history.append(
        Message(
            "user",
            [
                ToolResult(
                    tool_use_id=call.id,
                    execution_id=call.execution_id,
                    name=call.name,
                    content=str(total),
                    is_error=False,
                )
            ],
        )
    )
    second = await _generate(
        client,
        messages=history,
        model=FLASH,
        tools=tools,
        max_completion_tokens=MAX_TOKENS,
        thinking="low",
    )
    assert "42" in second.get_text_content()


@pytest.mark.asyncio
async def test_live_vision_reads_a_red_image(client_factory: Callable[[str], LLMClient]) -> None:
    client = client_factory(FLASH)
    image = ImageBlock(image_type="base64", media_type="image/png", data=_red_png_b64())
    message = await _generate(
        client,
        messages=MessageHistory(
            [Message("user", [TextBlock("What color is this image? Reply with one color name only."), image])]
        ),
        model=FLASH,
        max_completion_tokens=64,
    )
    assert "red" in message.get_text_content().lower()


@pytest.mark.asyncio
async def test_live_streaming_delivers_text(client_factory: Callable[[str], LLMClient]) -> None:
    client = client_factory(FLASH)
    try:
        stream_context = await cast(
            Any,
            client.stream(
                messages=_history("Count from one to three, digits only."),
                system=_system(),
                model=FLASH,
                max_completion_tokens=256,
            ),
        )
        async with stream_context as stream:
            chunks = [chunk async for chunk in stream]
        message = await stream.get_final_message()
    except _UNAVAILABLE as exc:  # pragma: no cover - account-dependent
        pytest.skip(f"MiniMax live API unavailable: {type(exc).__name__}: {exc}")
    assert chunks
    _usage(message)
    assert message.get_text_content().strip()


@pytest.mark.asyncio
async def test_live_output_cap_above_the_documented_ceiling_is_rejected(
    client_factory: Callable[[str], LLMClient],
) -> None:
    # 524288 is the ceiling; one token more must be refused rather than silently clamped.
    client = client_factory(FLASH)
    with pytest.raises(_REJECTED):
        await client.generate(
            messages=_history("Say OK"),
            system=_system(),
            model=FLASH,
            max_completion_tokens=524_289,
        )
