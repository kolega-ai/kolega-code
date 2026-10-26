"""Cloudflare Workers AI request shaping, connection plumbing, and error mapping (no network)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai import APIStatusError
from openai.types.chat import ChatCompletionChunk

from kolega_code.config import ModelProvider
from kolega_code.llm import cloudflare_workers_ai
from kolega_code.llm.client import LLMClient
from kolega_code.llm.exceptions import (
    LLMAuthenticationError,
    LLMBillingError,
    LLMContextWindowExceededError,
    LLMError,
    LLMInvalidRequestError,
    LLMNotFoundError,
    LLMRateLimitError,
    billing_error_message,
    llm_error_message,
    map_to_llm_error,
)
from kolega_code.llm.models import ContentBlock, Message, MessageHistory, TextBlock, ThinkingBlock, ToolCall
from kolega_code.llm.providers.openai import CLOUDFLARE_FALLBACK_MAX_TOKENS, OpenAIProvider, OpenAIStreamWrapper

PROVIDER = cloudflare_workers_ai.PROVIDER
ACCOUNT_ID = "0123456789abcdef0123456789abcdef"
BASE_URL = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/v1"
GLM = "@cf/zai-org/glm-5.3"
DEEPSEEK = "@cf/deepseek-ai/deepseek-v4.1-flash-1017"
MESSAGES = MessageHistory([Message(role="user", content=[TextBlock(text="hi")])])


@pytest.fixture(autouse=True)
def _no_ambient_cloudflare_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (cloudflare_workers_ai.ACCOUNT_ID_ENV, cloudflare_workers_ai.GATEWAY_ID_ENV):
        monkeypatch.delenv(name, raising=False)


def _client(**kwargs: Any) -> LLMClient:
    return LLMClient(provider=PROVIDER, api_key="cf-test-token", base_url=kwargs.pop("base_url", BASE_URL), **kwargs)


def _openai(client: LLMClient) -> OpenAIProvider:
    assert isinstance(client.provider, OpenAIProvider)
    return client.provider


def _sdk_headers(client: LLMClient) -> dict[str, str]:
    return dict(_openai(client).async_client._custom_headers)


def _base_url(client: LLMClient) -> str:
    return str(_openai(client).async_client.base_url).rstrip("/")


def _response() -> SimpleNamespace:
    message = SimpleNamespace(content="ok", tool_calls=None, reasoning_content=None, reasoning=None)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=usage)


async def _create_kwargs(client: LLMClient, **generate_kwargs: Any) -> dict[str, Any]:
    create = AsyncMock(return_value=_response())
    with patch.object(_openai(client).async_client.chat.completions, "create", create):
        await client.generate(messages=MESSAGES, **generate_kwargs)
    assert create.await_args is not None
    return dict(create.await_args.kwargs)


# --- connection ------------------------------------------------------------


def test_account_scoped_base_url_and_session_affinity_header() -> None:
    client = _client()

    assert _base_url(client) == BASE_URL
    affinity = _sdk_headers(client)[cloudflare_workers_ai.SESSION_AFFINITY_HEADER]
    assert affinity and affinity == _sdk_headers(client)[cloudflare_workers_ai.SESSION_AFFINITY_HEADER]
    assert cloudflare_workers_ai.GATEWAY_HEADER not in _sdk_headers(client)


def test_each_provider_instance_gets_its_own_affinity_id() -> None:
    first = _sdk_headers(_client())[cloudflare_workers_ai.SESSION_AFFINITY_HEADER]
    second = _sdk_headers(_client())[cloudflare_workers_ai.SESSION_AFFINITY_HEADER]
    assert first != second


def test_gateway_header_rides_alongside_affinity() -> None:
    client = _client(default_headers={cloudflare_workers_ai.GATEWAY_HEADER: "my-gateway"})

    headers = _sdk_headers(client)
    assert headers[cloudflare_workers_ai.GATEWAY_HEADER] == "my-gateway"
    assert cloudflare_workers_ai.SESSION_AFFINITY_HEADER in headers


def test_env_fallback_applies_only_without_an_explicit_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(cloudflare_workers_ai.ACCOUNT_ID_ENV, ACCOUNT_ID.upper())
    monkeypatch.setenv(cloudflare_workers_ai.GATEWAY_ID_ENV, "env-gateway")

    bare = LLMClient(provider=PROVIDER, api_key="cf-test-token")
    assert _base_url(bare) == BASE_URL
    assert _sdk_headers(bare)[cloudflare_workers_ai.GATEWAY_HEADER] == "env-gateway"

    other = "fedcba9876543210fedcba9876543210"
    configured = _client(base_url=cloudflare_workers_ai.base_url(other))
    assert other in _base_url(configured)
    assert cloudflare_workers_ai.GATEWAY_HEADER not in _sdk_headers(configured)


def test_missing_account_id_is_a_clear_error() -> None:
    with pytest.raises(LLMError, match=cloudflare_workers_ai.ACCOUNT_ID_ENV):
        LLMClient(provider=PROVIDER, api_key="cf-test-token")


def test_malformed_account_id_from_env_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(cloudflare_workers_ai.ACCOUNT_ID_ENV, "../../evil")
    with pytest.raises(LLMError, match="32-character hex"):
        LLMClient(provider=PROVIDER, api_key="cf-test-token")


def test_default_headers_are_rejected_for_non_chat_completions_providers() -> None:
    with pytest.raises(LLMInvalidRequestError, match="default_headers"):
        LLMClient(provider="anthropic", api_key="sk-ant-test", default_headers={"x-test": "1"})


def test_default_headers_reach_custom_openai_chat_endpoints() -> None:
    client = LLMClient(
        provider="custom:local",
        api_key="",
        base_url="http://localhost:1234/v1",
        api_style="openai_chat",
        default_headers={"x-test": "1"},
    )
    assert _sdk_headers(client)["x-test"] == "1"


# --- request shaping -------------------------------------------------------


@pytest.mark.asyncio
async def test_cap_is_sent_as_max_completion_tokens() -> None:
    kwargs = await _create_kwargs(_client(), model=GLM, max_completion_tokens=1234)

    assert kwargs["max_completion_tokens"] == 1234
    assert "max_tokens" not in kwargs


@pytest.mark.asyncio
async def test_missing_cap_is_filled_from_the_catalog() -> None:
    kwargs = await _create_kwargs(_client(), model=GLM)

    assert kwargs["max_completion_tokens"] == 131072


@pytest.mark.asyncio
async def test_uncatalogued_model_still_gets_an_explicit_cap() -> None:
    kwargs = await _create_kwargs(_client(), model="@cf/example/new-model")

    assert kwargs["max_completion_tokens"] == CLOUDFLARE_FALLBACK_MAX_TOKENS


@pytest.mark.asyncio
async def test_deepseek_clamp_applies_before_the_rename() -> None:
    kwargs = await _create_kwargs(_client(), model=DEEPSEEK, max_completion_tokens=65536)

    assert kwargs["max_completion_tokens"] == 64000
    assert "max_tokens" not in kwargs


@pytest.mark.asyncio
async def test_reasoning_effort_is_a_flat_field() -> None:
    kwargs = await _create_kwargs(_client(), model=DEEPSEEK, thinking="high")

    assert kwargs["reasoning_effort"] == "high"
    assert "extra_body" not in kwargs


@pytest.mark.asyncio
async def test_effort_none_disables_thinking_through_the_chat_template() -> None:
    kwargs = await _create_kwargs(_client(), model=DEEPSEEK, thinking="none")

    assert kwargs["reasoning_effort"] == "none"
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


@pytest.mark.asyncio
async def test_streaming_requests_usage_and_renames_the_cap() -> None:
    client = _client()

    async def chunks() -> AsyncIterator[ChatCompletionChunk]:
        if False:
            yield  # pragma: no cover

    create = AsyncMock(return_value=chunks())
    with patch.object(_openai(client).async_client.chat.completions, "create", create):
        await _openai(client).stream(MESSAGES, None, None, model=GLM)
    assert create.await_args is not None
    kwargs = create.await_args.kwargs
    assert kwargs["stream_options"] == {"include_usage": True}
    assert kwargs["max_completion_tokens"] == 131072


# --- reasoning replay ------------------------------------------------------


def _assistant_turn(source_provider: str, reasoning_field: str | None) -> Message:
    content: list[ContentBlock] = [
        ThinkingBlock(thinking="look at the file first"),
        ToolCall(id="call_1", name="read", input={"path": "a.py"}),
    ]
    message = Message(role="assistant", content=content)
    message.usage_metadata["provider"] = source_provider
    if reasoning_field:
        message.usage_metadata["reasoning_field"] = reasoning_field
    return message


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
def test_reasoning_replays_in_the_field_the_server_emitted(field: str) -> None:
    payload = _assistant_turn(PROVIDER, field).to_openai(provider=PROVIDER, model=GLM)

    assert payload[field] == "look at the file first"
    assert "Thinking" not in str(payload["content"])


def test_reasoning_without_a_recorded_field_defaults_to_reasoning() -> None:
    payload = _assistant_turn(PROVIDER, None).to_openai(provider=PROVIDER, model=GLM)

    assert payload["reasoning"] == "look at the file first"


def test_foreign_reasoning_is_never_replayed_natively() -> None:
    payload = _assistant_turn("fireworks", "reasoning_content").to_openai(provider=PROVIDER, model=GLM)

    assert "reasoning" not in payload
    assert "reasoning_content" not in payload


# --- streaming response ----------------------------------------------------


def _chunk(delta: dict[str, Any], finish_reason: str | None = None, usage: dict | None = None) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": GLM,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            "usage": usage,
        }
    )


@pytest.mark.asyncio
async def test_stream_captures_finish_reason_and_usage_on_the_final_chunk() -> None:
    async def chunks() -> AsyncIterator[ChatCompletionChunk]:
        yield _chunk({"role": "assistant", "reasoning": "think"})
        yield _chunk({"tool_calls": [{"index": 0, "id": "call_9", "function": {"name": "read", "arguments": "{}"}}]})
        yield _chunk(
            {},
            finish_reason="tool_calls",
            usage={
                "prompt_tokens": 50,
                "completion_tokens": 7,
                "total_tokens": 57,
                "prompt_tokens_details": {"cached_tokens": 32},
                "completion_tokens_details": {"reasoning_tokens": 3},
            },
        )

    wrapper = OpenAIStreamWrapper(chunks(), requested_include_usage=True, provider_name=PROVIDER, model=GLM)
    async with wrapper:
        async for _ in wrapper:
            pass
    message = await wrapper.get_final_message()

    assert message.stop_reason == "tool_use"
    assert message.usage_metadata["reasoning_field"] == "reasoning"
    assert message.usage_metadata["prompt_tokens"] == 50
    assert message.usage_metadata["cache_read_input_tokens"] == 32
    assert message.usage_metadata["reasoning_output_tokens"] == 3
    assert any(isinstance(block, ToolCall) for block in message.content)


@pytest.mark.asyncio
async def test_stream_matches_the_probed_workers_ai_chunk_sequence() -> None:
    """Every chunk carries a per-chunk usage delta; the last chunk carries the totals.

    ``finish_reason`` arrives on the final two chunks (probed against gpt-oss-20b
    and qwen3.8-27b, 2026-10-26).
    """

    def delta_usage(prompt: int, completion: int) -> dict[str, Any]:
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "prompt_tokens_details": {"cached_tokens": 0},
            "neurons": 0.1,
        }

    async def chunks() -> AsyncIterator[ChatCompletionChunk]:
        yield _chunk({"role": "assistant", "content": ""}, usage=delta_usage(87, 0))
        yield _chunk({"reasoning_content": "Count"}, usage=delta_usage(0, 3))
        yield _chunk({"content": "1\n2\n"}, usage=delta_usage(0, 4))
        yield _chunk({}, finish_reason="length", usage=delta_usage(0, 0))
        yield _chunk({}, finish_reason="length", usage=delta_usage(87, 96))

    wrapper = OpenAIStreamWrapper(chunks(), requested_include_usage=True, provider_name=PROVIDER, model=GLM)
    async with wrapper:
        async for _ in wrapper:
            pass
    message = await wrapper.get_final_message()

    assert message.stop_reason == "max_tokens"
    assert message.usage_metadata["prompt_tokens"] == 87
    assert message.usage_metadata["completion_tokens"] == 96
    assert message.usage_metadata["reasoning_field"] == "reasoning_content"
    assert message.usage is not None and message.usage.input_tokens == 87 and message.usage.output_tokens == 96


# --- error mapping ---------------------------------------------------------


def _status_error(status: int, body: dict[str, Any]) -> APIStatusError:
    response = httpx.Response(status_code=status, request=httpx.Request("POST", f"{BASE_URL}/chat/completions"))
    return APIStatusError(f"Error code: {status} - {body}", response=response, body=body)


def _cf_body(code: int, message: str) -> dict[str, Any]:
    return {"errors": [{"code": code, "message": message}], "success": False, "result": None, "messages": []}


def test_daily_free_allocation_is_a_billing_error_not_a_retryable_429() -> None:
    error = _status_error(
        429,
        _cf_body(
            3036,
            "You have used up your daily free allocation of 10,000 neurons. Please upgrade to Cloudflare's "
            "Workers Paid plan if you would like to continue usage.",
        ),
    )

    mapped = map_to_llm_error(error, provider=PROVIDER)

    assert isinstance(mapped, LLMBillingError)
    assert "Workers Paid" in billing_error_message(mapped, model=GLM)


def test_free_plan_model_403_5035_is_a_billing_error() -> None:
    error = _status_error(
        403,
        _cf_body(
            5035,
            "AiError: Model @cf/zai-org/glm-5.3 is not available on the Workers Free plan: Model "
            "@cf/zai-org/glm-5.3 is not available on the Workers Free plan. Upgrade to access this model: "
            "https://dash.cloudflare.com/?to=/:account/workers/plans (00000000-0000-0000-0000-000000000000)",
        ),
    )

    mapped = map_to_llm_error(error, provider=PROVIDER)

    assert isinstance(mapped, LLMBillingError)
    assert "Workers Paid" in llm_error_message(mapped, model=GLM)


def test_capacity_exhaustion_stays_a_retryable_rate_limit() -> None:
    error = _status_error(429, _cf_body(3040, "Capacity temporarily exceeded, please try again."))

    assert isinstance(map_to_llm_error(error, provider=PROVIDER), LLMRateLimitError)


def test_input_plus_output_over_the_window_413_5021_is_context_window_exceeded() -> None:
    error = _status_error(
        413,
        _cf_body(
            5021,
            "AiError: Ai: The estimated number of input and maximum output tokens (175033) exceeded this model "
            "context window limit (128000). (00000000-0000-0000-0000-000000000000)",
        ),
    )

    assert isinstance(map_to_llm_error(error, provider=PROVIDER), LLMContextWindowExceededError)


def test_vllm_style_400_8007_maximum_context_length_is_context_window_exceeded() -> None:
    error = _status_error(
        400,
        _cf_body(
            8007,
            'AiError: AiError: {"error":{"message":"This model\'s maximum context length is 256000 tokens. '
            "However, you requested 256000 output tokens and your prompt contains 92 characters (more than 0 "
            "characters, which is the upper bound for 0 input tokens). Please reduce the length of the input "
            'prompt or the number of requested output tokens. (parameter=input_text, value=92)",'
            '"type":"BadRequestError","param":"input_text","code":400}} (00000000-0000-0000-0000-000000000000)',
        ),
    )

    assert isinstance(map_to_llm_error(error, provider=PROVIDER), LLMContextWindowExceededError)


def test_unknown_model_400_5007_is_not_found() -> None:
    error = _status_error(
        400,
        _cf_body(
            5007, "AiError: No such model: No such model @cf/nope/nope or task (00000000-0000-0000-0000-000000000000)"
        ),
    )

    assert isinstance(map_to_llm_error(error, provider=PROVIDER), LLMNotFoundError)


def test_bad_token_401_10000_is_an_authentication_error() -> None:
    error = _status_error(401, _cf_body(10000, "Authentication error"))

    assert isinstance(map_to_llm_error(error, provider=PROVIDER), LLMAuthenticationError)


def test_unconfigured_gateway_400_2001_surfaces_the_detail() -> None:
    error = _status_error(400, _cf_body(2001, "Please configure AI Gateway in the Cloudflare dashboard"))

    mapped = map_to_llm_error(error, provider=PROVIDER)

    assert isinstance(mapped, LLMInvalidRequestError)
    assert "configure AI Gateway" in llm_error_message(mapped, model=GLM)


def test_cloudflare_rules_do_not_leak_to_other_openai_compatible_providers() -> None:
    error = _status_error(429, _cf_body(3036, "You have used up your daily free allocation of 10,000 neurons."))

    assert isinstance(map_to_llm_error(error, provider=ModelProvider.GROQ.value), LLMRateLimitError)
