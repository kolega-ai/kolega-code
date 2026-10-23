"""Key-gated live tests for the Cloudflare Workers AI provider.

These hit the account-scoped endpoint for real. Opt in with
``KOLEGA_TEST_LIVE_API=1`` plus ``CLOUDFLARE_API_TOKEN`` and
``CLOUDFLARE_ACCOUNT_ID`` (optionally ``CLOUDFLARE_AI_GATEWAY_ID``) in the
repo-root ``.env`` (loaded with python-dotenv — never ``source`` it):

    KOLEGA_TEST_LIVE_API=1 ./run_tests.sh --all tests/agent/llm/test_cloudflare_workers_ai_live.py

Each assertion mirrors a catalog or request-shaping decision; when Workers AI
changes, these fail before the catalog misleads an agent.
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

from kolega_code.llm import cloudflare_workers_ai
from kolega_code.llm.client import LLMClient
from kolega_code.llm.exceptions import LLMBillingError, LLMPermissionDeniedError, LLMRateLimitError
from kolega_code.llm.models import (
    ContentBlock,
    ImageBlock,
    Message,
    MessageHistory,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)
from kolega_code.llm.providers.openai import OpenAIProvider
from kolega_code.llm.specs import MODEL_SPECS
from kolega_code.llm.specs import cloudflare_workers_ai_catalog

pytestmark = [pytest.mark.integration, pytest.mark.live_api]

PROVIDER = cloudflare_workers_ai.PROVIDER
GLM = "@cf/zai-org/glm-5.3"
DEEPSEEK_FLASH = "@cf/deepseek-ai/deepseek-v4.1-flash-1017"
KIMI_CODE = "@cf/moonshotai/kimi-k2.7-code"
GPT_OSS_20B = "@cf/openai/gpt-oss-20b"
_UNAVAILABLE = (LLMRateLimitError, LLMBillingError, LLMPermissionDeniedError)


def _red_png_b64(size: int = 48) -> str:
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


def _credentials() -> tuple[str, str]:
    if os.getenv("KOLEGA_TEST_LIVE_API") != "1":
        pytest.skip("set KOLEGA_TEST_LIVE_API=1 to run Cloudflare Workers AI live tests")
    load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)
    token = os.getenv(cloudflare_workers_ai.API_TOKEN_ENV)
    account = os.getenv(cloudflare_workers_ai.ACCOUNT_ID_ENV)
    if not token or not account:
        pytest.skip("CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID must both be set")
    return token, account


@pytest_asyncio.fixture
async def client_factory() -> AsyncIterator[Callable[[str], LLMClient]]:
    clients: list[LLMClient] = []

    def create(model: str) -> LLMClient:
        token, _account = _credentials()
        # No base_url: exercises the CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_AI_GATEWAY_ID fallback.
        client = LLMClient(provider=PROVIDER, api_key=token, model=model, max_retries=0)
        clients.append(client)
        return client

    try:
        yield create
    finally:
        for client in clients:
            assert isinstance(client.provider, OpenAIProvider)
            await cast(Any, client.provider.async_client).close()


def _history(text: str) -> MessageHistory:
    return MessageHistory([Message("user", [TextBlock(text)])])


def _usage(message: Message) -> None:
    assert message.usage is not None and message.usage.reported
    assert message.usage.provider == PROVIDER
    assert message.usage.input_tokens is not None and message.usage.input_tokens > 0
    assert message.usage.output_tokens is not None and message.usage.output_tokens > 0


async def _generate(client: LLMClient, **kwargs: Any) -> Message:
    try:
        message = await client.generate(**kwargs)
    except _UNAVAILABLE as exc:  # pragma: no cover - account-dependent
        pytest.skip(f"Workers AI unavailable for this account: {type(exc).__name__}: {exc}")
    _usage(message)
    return message


async def _stream(client: LLMClient, **kwargs: Any) -> Message:
    try:
        stream_cm = await cast(Any, client.stream(**kwargs))
        async with stream_cm as stream:
            async for _ in stream:
                pass
            message = await stream.get_final_message()
    except _UNAVAILABLE as exc:  # pragma: no cover - account-dependent
        pytest.skip(f"Workers AI unavailable for this account: {type(exc).__name__}: {exc}")
    _usage(message)
    return message


def _reasoning_tokens(message: Message) -> int:
    value = message.usage_metadata.get("reasoning_output_tokens")
    return value if isinstance(value, int) else 0


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
async def test_live_default_model_generates_with_openai_usage(client_factory: Callable[[str], LLMClient]) -> None:
    message = await _generate(
        client_factory(GLM),
        messages=_history("Reply with exactly: OK"),
        model=GLM,
        max_completion_tokens=2048,
        thinking="low",
    )
    assert "OK" in message.get_text_content()


@pytest.mark.asyncio
async def test_live_stream_reports_usage_and_a_clean_stop(client_factory: Callable[[str], LLMClient]) -> None:
    message = await _stream(
        client_factory(GPT_OSS_20B),
        messages=_history("Reply with exactly: OK"),
        model=GPT_OSS_20B,
        max_completion_tokens=1024,
        thinking="low",
    )
    assert message.stop_reason == "end_turn"
    assert "OK" in message.get_text_content()


@pytest.mark.asyncio
async def test_live_effort_none_disables_reasoning(client_factory: Callable[[str], LLMClient]) -> None:
    client = client_factory(DEEPSEEK_FLASH)
    prompt = "A farmer has 17 sheep. All but 9 run away. How many remain? Think step by step, then answer."

    off = await _generate(
        client, messages=_history(prompt), model=DEEPSEEK_FLASH, max_completion_tokens=4096, thinking="none"
    )
    on = await _generate(
        client, messages=_history(prompt), model=DEEPSEEK_FLASH, max_completion_tokens=4096, thinking="high"
    )

    assert _reasoning_tokens(off) == 0
    assert _reasoning_tokens(on) > 0


@pytest.mark.asyncio
async def test_live_explicit_cap_truncates_honestly(client_factory: Callable[[str], LLMClient]) -> None:
    message = await _generate(
        client_factory(GPT_OSS_20B),
        messages=_history("Count from 1 to 500, one number per line."),
        model=GPT_OSS_20B,
        max_completion_tokens=64,
        thinking="low",
    )
    assert message.stop_reason == "max_tokens"


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [GLM, KIMI_CODE])
async def test_live_tool_round_trip_replays_reasoning(model: str, client_factory: Callable[[str], LLMClient]) -> None:
    client = client_factory(model)
    tools = [_calc_tool()]
    history = _history("Use the add_small_ints tool to add 19 and 23, then state the result.")
    effort = "low" if MODEL_SPECS[(PROVIDER, model)].get("thinking_effort") else None

    first = await _stream(
        client, messages=history, model=model, tools=tools, max_completion_tokens=4096, thinking=effort
    )
    calls = [block for block in first.content if isinstance(block, ToolCall)]
    assert first.stop_reason == "tool_use"
    assert calls and calls[0].name == "add_small_ints"

    result: list[ContentBlock] = [ToolResult(tool_use_id=calls[0].id, content="42", name=calls[0].name, is_error=False)]
    history = MessageHistory([*history, first, Message("user", result)])
    second = await _stream(
        client, messages=history, model=model, tools=tools, max_completion_tokens=4096, thinking=effort
    )

    assert "42" in second.get_text_content()


@pytest.mark.asyncio
async def test_live_vision_reads_an_image(client_factory: Callable[[str], LLMClient]) -> None:
    content: list[ContentBlock] = [
        ImageBlock(image_type="base64", media_type="image/png", data=_red_png_b64()),
        TextBlock("What single colour fills this image? Answer with one word."),
    ]
    message = await _generate(
        client_factory(DEEPSEEK_FLASH),
        messages=MessageHistory([Message("user", content)]),
        model=DEEPSEEK_FLASH,
        max_completion_tokens=1024,
        thinking="none",
    )
    assert "red" in message.get_text_content().lower()


@pytest.mark.asyncio
async def test_live_cached_token_field_parses(client_factory: Callable[[str], LLMClient]) -> None:
    client = client_factory(GPT_OSS_20B)
    prefix = "Reference text. " * 600
    kwargs: dict[str, Any] = {"model": GPT_OSS_20B, "max_completion_tokens": 256, "thinking": "low"}
    await _generate(client, messages=_history(prefix + "Reply OK."), **kwargs)
    second = await _generate(client, messages=_history(prefix + "Reply OK again."), **kwargs)

    cached = second.usage_metadata.get("cache_read_input_tokens")
    assert cached is None or (isinstance(cached, int) and cached >= 0)


def test_live_refresh_lists_every_curated_model() -> None:
    token, account = _credentials()
    payload = cloudflare_workers_ai_catalog.fetch_models(api_key=token, account_id=account)
    discovered = dict(cloudflare_workers_ai_catalog.catalog_entries(payload))
    curated = [model for provider, model in MODEL_SPECS if provider == PROVIDER]

    assert [model for model in curated if model not in discovered] == []
