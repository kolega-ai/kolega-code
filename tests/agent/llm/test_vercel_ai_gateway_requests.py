"""Request shaping, reasoning replay, usage/cost accounting, and error mapping for Vercel AI Gateway."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest

from kolega_code.llm.accounting import reported_cost
from kolega_code.llm.client import LLMClient
from kolega_code.llm.exceptions import (
    LLMAuthenticationError,
    LLMBillingError,
    LLMContextWindowExceededError,
    LLMNotFoundError,
    LLMPermissionDeniedError,
    billing_error_message,
    map_openai_errors,
)
from kolega_code.llm.models import MessageHistory
from kolega_code.llm.providers.models import GenerationParams
from kolega_code.llm.providers.openai import OpenAIProvider, OpenAIStreamWrapper, _usage_metadata
from kolega_code.llm.specs import MODEL_SPECS, ThinkingEffortSpec
from kolega_code.llm.specs.thinking import reasoning_replay_field
from kolega_code.llm.usage import normalize_usage

PROVIDER = "vercel_ai_gateway"


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.chat = self
        self.completions = self

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)

        class _Stream:
            async def __anext__(self) -> Any:
                raise StopAsyncIteration

            async def aclose(self) -> None:
                return None

        return _Stream()

    @property
    def last(self) -> dict[str, Any]:
        assert self.calls
        return self.calls[-1]


def _stream_request(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model: str,
    thinking: str | None = None,
    max_completion_tokens: int | None = 4096,
    temperature: float = 0.7,
) -> dict[str, Any]:
    provider = OpenAIProvider(api_key="vck_test_key_00000000000000000000", provider_name=PROVIDER)
    recorder = _Recorder()
    monkeypatch.setattr(provider, "async_client", recorder)

    async def run() -> None:
        await provider.stream(
            MessageHistory([]),
            params=GenerationParams(
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
                thinking=thinking,
            ),
            model=model,
        )

    asyncio.run(run())
    return recorder.last


def test_client_targets_vercel_ai_gateway_chat_completions() -> None:
    client = LLMClient(PROVIDER, "vck_test_key_00000000000000000000")
    assert isinstance(client.provider, OpenAIProvider)
    assert client.provider.provider_name == PROVIDER
    assert client.provider.base_url == "https://ai-gateway.vercel.sh/v1"


def test_thinking_modes_and_auto_caching_merge_cleanly(isolated_cli_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    effort_model = "vendor/effort-model"
    toggle_model = "vendor/toggle-model"
    monkeypatch.setitem(
        MODEL_SPECS,
        (PROVIDER, effort_model),
        {
            "context_length": 128000,
            "max_completion_tokens": 16384,
            "default_temperature": 1.0,
            "supports_vision": False,
            "thinking_effort": ThinkingEffortSpec(
                options=("none", "low", "medium", "high"),
                default="medium",
                mode="vercel_ai_gateway_reasoning",
            ),
        },
    )
    monkeypatch.setitem(
        MODEL_SPECS,
        (PROVIDER, toggle_model),
        {
            "context_length": 128000,
            "max_completion_tokens": 16384,
            "default_temperature": 1.0,
            "supports_vision": False,
            "thinking_effort": ThinkingEffortSpec(
                options=("none", "enabled"),
                default="enabled",
                mode="vercel_ai_gateway_reasoning",
            ),
        },
    )

    req_effort = _stream_request(monkeypatch, model=effort_model, thinking="high")
    assert req_effort["extra_body"]["reasoning"] == {"effort": "high"}
    assert req_effort["extra_body"]["providerOptions"] == {"gateway": {"caching": "auto"}}
    assert req_effort["max_tokens"] == 4096

    req_off = _stream_request(monkeypatch, model=effort_model, thinking="none")
    assert req_off["extra_body"]["reasoning"] == {"enabled": False}
    assert req_off["extra_body"]["providerOptions"] == {"gateway": {"caching": "auto"}}

    req_toggle = _stream_request(monkeypatch, model=toggle_model, thinking="enabled")
    assert req_toggle["extra_body"]["reasoning"] == {"enabled": True}


def test_optional_gateway_routing_env_vars(isolated_cli_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    model = "vendor/plain-model"
    monkeypatch.setitem(
        MODEL_SPECS,
        (PROVIDER, model),
        {
            "context_length": 128000,
            "max_completion_tokens": 16384,
            "default_temperature": 1.0,
            "supports_vision": False,
        },
    )
    monkeypatch.setenv("KOLEGA_CODE_VERCEL_AI_GATEWAY_PROVIDER_ORDER", "anthropic, vertex")
    monkeypatch.setenv("KOLEGA_CODE_VERCEL_AI_GATEWAY_SORT", "TTFT")

    req = _stream_request(monkeypatch, model=model)
    assert req["extra_body"]["providerOptions"] == {
        "gateway": {"caching": "auto", "order": ["anthropic", "vertex"], "sort": "ttft"}
    }


def test_no_temperature_and_deepseek_cap_clamping(isolated_cli_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    no_temp = "vendor/no-temp"
    monkeypatch.setitem(
        MODEL_SPECS,
        (PROVIDER, no_temp),
        {
            "context_length": 128000,
            "max_completion_tokens": 16384,
            "default_temperature": 1.0,
            "supports_temperature": False,
            "supports_vision": False,
        },
    )
    req_no_temp = _stream_request(monkeypatch, model=no_temp, temperature=0.5)
    assert "temperature" not in req_no_temp

    ds_model = "deepseek/deepseek-v4-test"
    monkeypatch.setitem(
        MODEL_SPECS,
        (PROVIDER, ds_model),
        {
            "context_length": 1048576,
            "max_completion_tokens": 262144,
            "default_temperature": 1.0,
            "supports_vision": False,
        },
    )
    req_ds = _stream_request(monkeypatch, model=ds_model, max_completion_tokens=200000)
    assert req_ds["max_tokens"] == 64000
    req_ds_small = _stream_request(monkeypatch, model=ds_model, max_completion_tokens=2048)
    assert req_ds_small["max_tokens"] == 2048


def test_reasoning_replay_field_drops_anthropic_and_echoes_others(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = ThinkingEffortSpec(options=("none", "low", "high"), default="low", mode="vercel_ai_gateway_reasoning")
    monkeypatch.setitem(
        MODEL_SPECS,
        (PROVIDER, "moonshotai/replay-test"),
        {
            "context_length": 128000,
            "max_completion_tokens": 16384,
            "default_temperature": 1.0,
            "supports_vision": False,
            "thinking_effort": spec,
        },
    )
    monkeypatch.setitem(
        MODEL_SPECS,
        (PROVIDER, "anthropic/replay-test"),
        {
            "context_length": 128000,
            "max_completion_tokens": 16384,
            "default_temperature": 1.0,
            "supports_vision": False,
            "drop_prior_reasoning": True,
            "thinking_effort": spec,
        },
    )
    assert reasoning_replay_field(PROVIDER, "moonshotai/replay-test") == "reasoning"
    assert reasoning_replay_field(PROVIDER, "anthropic/replay-test") is None


@pytest.mark.asyncio
async def test_usage_normalization_cache_creation_and_reported_cost() -> None:
    raw_usage = {
        "prompt_tokens": 3405,
        "completion_tokens": 18,
        "total_tokens": 3423,
        "prompt_tokens_details": {"cached_tokens": 3384, "audio_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 15},
        "cache_creation_input_tokens": 13,
        "cost": 0.000129456,
        "market_cost": 0.000129456,
        "gateway_cost": 0.000129456,
        "is_byok": False,
        "cost_details": {"upstream_inference_cost": None},
    }
    metadata = _usage_metadata(raw_usage)
    assert metadata["prompt_tokens"] == 3392
    assert metadata["cache_write_input_tokens"] == 13
    assert metadata["cache_read_input_tokens"] == 3384
    assert metadata["reasoning_output_tokens"] == 15

    normalized = normalize_usage(metadata, PROVIDER, "alibaba/qwen3.7-flash")
    assert normalized.reported is True
    assert normalized.input_tokens == 3405
    assert normalized.output_tokens == 18
    assert normalized.total_tokens == 3423
    assert normalized.cache_read_input_tokens == 3384
    assert normalized.cache_write_input_tokens == 13
    assert normalized.reasoning_output_tokens == 15

    cost = reported_cost(PROVIDER, metadata)
    assert cost is not None
    assert cost["source"] == "provider_reported"
    assert cost["amount"] == "0.000129456"
    assert cost["scope"] == "gateway_request_charge"
    assert cost["provenance"]["field"] == "usage.cost"
    assert cost["provenance"]["gateway_cost"] == 0.000129456
    assert cost["provenance"]["is_byok"] is False

    # Streaming wrapper captures delta.reasoning and top-level cache_creation_input_tokens.
    class _FakeStream:
        def __init__(self) -> None:
            self._items = [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(delta=SimpleNamespace(reasoning="step 1", content=None), finish_reason=None)
                    ],
                    usage=None,
                    model="moonshotai/kimi-k2.7-code",
                ),
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="4"), finish_reason="stop")],
                    usage=None,
                    model="moonshotai/kimi-k2.7-code",
                ),
                SimpleNamespace(
                    choices=[],
                    usage=raw_usage,
                    model="moonshotai/kimi-k2.7-code",
                ),
            ]
            self._index = 0

        async def __anext__(self) -> Any:
            if self._index >= len(self._items):
                raise StopAsyncIteration
            item = self._items[self._index]
            self._index += 1
            return item

        async def aclose(self) -> None:
            return None

    wrapper = OpenAIStreamWrapper(
        _FakeStream(),
        requested_include_usage=True,
        provider_name=PROVIDER,
        model="moonshotai/kimi-k2.7-code",
    )
    chunks = [chunk async for chunk in wrapper]
    assert any(chunk.type == "thinking" and chunk.thinking == "step 1" for chunk in chunks)
    assert any(chunk.type == "text" and chunk.text == "4" for chunk in chunks)
    final_msg = await wrapper.get_final_message()
    assert final_msg.usage is not None
    assert final_msg.usage.input_tokens == 3405
    assert final_msg.usage.cache_write_input_tokens == 13
    assert final_msg.usage_metadata["reasoning_field"] == "reasoning"


def _status_error(cls: type[openai.APIStatusError], status: int, body: dict[str, Any]) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://ai-gateway.vercel.sh/v1/chat/completions")
    response = httpx.Response(status, request=request, json=body)
    return cls(message=str(body), response=response, body=body)


def test_vercel_error_mapping() -> None:
    err_402 = _status_error(
        openai.APIStatusError,
        402,
        {"error": {"message": "Insufficient balance", "type": "payment_required"}},
    )
    mapped_402 = map_openai_errors(err_402, PROVIDER)
    assert isinstance(mapped_402, LLMBillingError)
    assert "Vercel AI Gateway" in billing_error_message(mapped_402, model="anthropic/claude-opus-5")
    assert "AI Gateway credits" in billing_error_message(mapped_402, model="anthropic/claude-opus-5")

    err_403_free = _status_error(
        openai.PermissionDeniedError,
        403,
        {
            "error": {
                "message": "Free tier users do not have access to this model. Add AI Gateway credits...",
                "type": "no_providers_available",
                "param": {"name": "RestrictedModelsError", "statusCode": 403},
            }
        },
    )
    assert isinstance(map_openai_errors(err_403_free, PROVIDER), LLMBillingError)

    err_403_perm = _status_error(
        openai.PermissionDeniedError,
        403,
        {"error": {"message": "Access forbidden by policy", "type": "permission_error"}},
    )
    assert isinstance(map_openai_errors(err_403_perm, PROVIDER), LLMPermissionDeniedError)

    err_400_ctx = _status_error(
        openai.BadRequestError,
        400,
        {
            "error": {
                "message": "This model's maximum context length is 40960 tokens. However, you requested 16 output tokens...",
                "type": "AI_APICallError",
            }
        },
    )
    assert isinstance(map_openai_errors(err_400_ctx, PROVIDER), LLMContextWindowExceededError)

    err_401 = _status_error(
        openai.AuthenticationError,
        401,
        {"error": {"message": "Authentication failed.", "type": "authentication_error"}},
    )
    assert isinstance(map_openai_errors(err_401, PROVIDER), LLMAuthenticationError)

    err_404 = _status_error(
        openai.NotFoundError,
        404,
        {"error": {"message": "Model 'nonexistent/model' not found", "type": "model_not_found"}},
    )
    assert isinstance(map_openai_errors(err_404, PROVIDER), LLMNotFoundError)
