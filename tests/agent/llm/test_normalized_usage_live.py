"""Live checks that every keyed provider's completed Message carries a reported
NormalizedUsage on both the generate and streaming paths.

Same conventions as test_live_providers.py: marked ``integration``, skipped in
CI and for providers whose API key is absent (keys load from repo ``.env``).
The streaming test also finalizes twice, mirroring BaseAgent's
call-after-__aexit__ pattern, to lock idempotency against live payloads.
"""

import os
from typing import Any, cast

import anthropic
import openai
import pytest

from kolega_code.cli.config import API_KEY_ENV, OAUTH_PROVIDERS
from kolega_code.cli.provider_registry import default_model_for_provider
from kolega_code.config import ModelProvider
from kolega_code.llm.client import LLMClient
from kolega_code.llm.exceptions import LLMBillingError, LLMPermissionDeniedError, LLMRateLimitError
from kolega_code.llm.models import Message, MessageHistory, TextBlock
from kolega_code.llm.specs import MODEL_SPECS, default_thinking_effort, get_model_specs
from kolega_code.llm.usage import NormalizedUsage

pytestmark = pytest.mark.integration

SKIP_IN_CI = bool(os.getenv("CI")) or bool(os.getenv("GITLAB_CI"))

# generate() maps SDK errors to the repo's LLM exceptions, but the streaming
# path surfaces raw SDK errors — skip on both shapes of quota/capacity failure.
# Permission errors mean the key lacks access to the API surface (e.g.
# Perplexity's preview-gated Router), which is likewise not the integration.
_QUOTA_ERRORS = (
    LLMRateLimitError,
    LLMBillingError,
    LLMPermissionDeniedError,
    anthropic.RateLimitError,
    openai.RateLimitError,
    openai.PermissionDeniedError,
)

SYSTEM = Message(
    role="system",
    content=[TextBlock(text="You are a concise math assistant. Answer with as few tokens as possible.")],
)

OPENROUTER_SMOKE_MODEL = "openai/gpt-5.4-mini"


def _messages() -> MessageHistory:
    return MessageHistory(
        [Message(role="user", content=[TextBlock(text="What is 2 + 2? Reply with just the number.")])]
    )


def _provider_models() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    added: set[str] = set()
    for provider_value, _model in MODEL_SPECS:
        if provider_value in added or ModelProvider(provider_value) in OAUTH_PROVIDERS:
            continue
        added.add(provider_value)
        if provider_value == ModelProvider.OPENROUTER.value:
            model = OPENROUTER_SMOKE_MODEL
        else:
            model = default_model_for_provider(ModelProvider(provider_value))
        pairs.append((provider_value, model))
    # The one model-level dispatch split: deepseek-v4-flash speaks the Responses
    # API while the rest of the deepseek catalog stays on Chat Completions.
    pairs.append((ModelProvider.DEEPSEEK.value, "deepseek-v4-flash"))
    return pairs


PROVIDER_MODELS = _provider_models()
_IDS = [f"{p}-{m.rsplit('/', 1)[-1]}" for p, m in PROVIDER_MODELS]


def _require_key(provider_value: str) -> str:
    if SKIP_IN_CI:
        pytest.skip("Skipping live provider call in CI")
    env_name = API_KEY_ENV[ModelProvider(provider_value)]
    api_key = os.getenv(env_name)
    if not api_key:
        pytest.skip(f"{env_name} not set")
    return api_key


def _generation_kwargs(provider_value: str, model: str) -> dict:
    return dict(
        system=SYSTEM,
        model=model,
        max_completion_tokens=4096,
        temperature=get_model_specs(provider_value, model)["default_temperature"],
        thinking=default_thinking_effort(provider_value, model),
    )


def _assert_reported_usage(message: Message, provider_value: str, model: str) -> NormalizedUsage:
    usage = message.usage
    assert usage is not None, f"{provider_value}/{model}: message has no normalized usage"
    assert usage.reported is True, f"{provider_value}/{model}: usage not reported ({usage.unavailable_reason})"
    assert usage.provider == provider_value
    assert usage.model == model
    assert usage.input_tokens is not None and usage.input_tokens > 0
    assert usage.output_tokens is not None and usage.output_tokens > 0
    assert usage.total_tokens == usage.input_tokens + usage.output_tokens
    if usage.cache_read_input_tokens is not None:
        assert usage.cache_read_input_tokens <= usage.input_tokens
    if usage.cache_write_input_tokens is not None:
        assert usage.cache_write_input_tokens <= usage.input_tokens
    if usage.reasoning_output_tokens is not None:
        assert usage.reasoning_output_tokens <= usage.output_tokens
    assert message.usage_metadata, f"{provider_value}/{model}: raw usage_metadata went missing"
    return usage


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_value,model", PROVIDER_MODELS, ids=_IDS)
async def test_live_generate_attaches_reported_usage(provider_value: str, model: str) -> None:
    api_key = _require_key(provider_value)
    client = LLMClient(provider=provider_value, api_key=api_key, model=model)
    try:
        message = await client.generate(messages=_messages(), **_generation_kwargs(provider_value, model))
    except _QUOTA_ERRORS as exc:
        pytest.skip(f"provider quota exhausted for this key: {exc}")
    _assert_reported_usage(message, provider_value, model)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_value,model", PROVIDER_MODELS, ids=_IDS)
async def test_live_stream_attaches_reported_usage_idempotently(provider_value: str, model: str) -> None:
    api_key = _require_key(provider_value)
    client = LLMClient(provider=provider_value, api_key=api_key, model=model)
    try:
        stream_cm = await cast(Any, client.stream(messages=_messages(), **_generation_kwargs(provider_value, model)))
        async with stream_cm as stream:
            async for _ in stream:
                pass
        # Finalize AFTER __aexit__, exactly like BaseAgent.process_message_stream.
        message = await stream.get_final_message()
    except _QUOTA_ERRORS as exc:
        pytest.skip(f"provider quota exhausted for this key: {exc}")
    usage = _assert_reported_usage(message, provider_value, model)
    again = await stream.get_final_message()
    assert again.usage == usage
