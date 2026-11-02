"""Live provider integration tests for provider smoke coverage.

These hit the real provider APIs, so they are marked ``integration`` and are
skipped for any provider whose API key is not present in the environment (loaded
from the repo ``.env`` by ``conftest.py``). They intentionally smoke-test one
model per key-based provider instead of exhaustively exercising large provider
catalogs.

Run them explicitly with the relevant keys set, e.g.::

    pytest -m integration tests/agent/llm/test_live_providers.py -v
"""

import os

import pytest

from kolega_code.cli.config import API_KEY_ENV
from kolega_code.config import ModelProvider
from kolega_code.llm.client import LLMClient
from kolega_code.llm.exceptions import LLMBillingError, LLMPermissionDeniedError, LLMRateLimitError
from kolega_code.llm.models import Message, MessageHistory, TextBlock
from kolega_code.llm.specs import (
    get_model_specs,
    thinking_effort_options,
)

from ._live_models import provider_smoke_models, smoke_generation_settings

pytestmark = [pytest.mark.integration, pytest.mark.live_api]

# Skip the live calls inside CI, where keys are typically absent and we don't want
# to spend tokens on every pipeline run.
SKIP_IN_CI = bool(os.getenv("CI")) or bool(os.getenv("GITLAB_CI"))

SYSTEM = Message(
    role="system",
    content=[TextBlock(text="You are a concise math assistant. Answer with as few tokens as possible.")],
)


def _messages() -> MessageHistory:
    return MessageHistory(
        [Message(role="user", content=[TextBlock(text="What is 2 + 2? Reply with just the number.")])]
    )


PROVIDER_SMOKE_MODELS = provider_smoke_models()

# Subset whose smoke model exposes a thinking/reasoning-effort control. Ollama
# Cloud is intentionally excluded to keep live Ollama coverage to one smoke call.
PROVIDER_THINKING_MODELS = [
    (provider, model)
    for provider, model in PROVIDER_SMOKE_MODELS
    if provider != ModelProvider.OLLAMA_CLOUD.value
    and smoke_generation_settings(provider, model)["thinking"] is not None
]


def _require_key(provider_value: str) -> str:
    if SKIP_IN_CI:
        pytest.skip("Skipping live provider call in CI")
    env_name = API_KEY_ENV[ModelProvider(provider_value)]
    api_key = os.getenv(env_name)
    if not api_key:
        pytest.skip(f"{env_name} not set")
    # Workers AI URLs are account-scoped; LLMClient reads the ID from the environment.
    if provider_value == ModelProvider.CLOUDFLARE_WORKERS_AI.value and not os.getenv("CLOUDFLARE_ACCOUNT_ID"):
        pytest.skip("CLOUDFLARE_ACCOUNT_ID not set")
    return api_key


async def _live_generate(client: "LLMClient", **generate_kwargs):
    """Run a live ``generate`` call, skipping on provider-side quota/access limits.

    Rate-limit and billing errors reflect the API key's quota/billing state, and
    permission errors reflect missing access to an API surface (e.g. Perplexity's
    preview-gated Router) — none are the integration under test, so they skip
    instead of failing the suite. Real integration regressions (auth, request
    formatting, parsing) still surface.
    """
    try:
        return await client.generate(**generate_kwargs)
    except (LLMRateLimitError, LLMBillingError, LLMPermissionDeniedError) as exc:
        pytest.skip(f"provider quota or access exhausted for this key: {exc}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_value,model",
    PROVIDER_SMOKE_MODELS,
    ids=[p for p, _ in PROVIDER_SMOKE_MODELS],
)
async def test_live_provider_generate(provider_value: str, model: str) -> None:
    """The provider's smoke model resolves and returns a non-empty assistant reply.

    Uses the model's default thinking effort, mirroring how the agent calls each
    model — some models (e.g. Moonshot kimi-k2.7-code) force thinking on and reject
    a request that omits it.
    """
    api_key = _require_key(provider_value)
    client = LLMClient(provider=provider_value, api_key=api_key)

    response = await _live_generate(
        client,
        messages=_messages(),
        system=SYSTEM,
        model=model,
        max_completion_tokens=4096,
        **smoke_generation_settings(provider_value, model),
    )

    assert response is not None
    assert response.role == "assistant"
    assert response.get_text_content(), f"empty response from {provider_value}/{model}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_value,model",
    PROVIDER_THINKING_MODELS,
    ids=[p for p, _ in PROVIDER_THINKING_MODELS],
)
async def test_live_provider_thinking_effort(provider_value: str, model: str) -> None:
    """A non-default thinking-effort serializes correctly for each provider's effort mode."""
    api_key = _require_key(provider_value)
    client = LLMClient(provider=provider_value, api_key=api_key)
    # Exercise a different effort than the smoke test (the last/highest option).
    effort = thinking_effort_options(provider_value, model)[-1]

    response = await _live_generate(
        client,
        messages=_messages(),
        system=SYSTEM,
        model=model,
        max_completion_tokens=8192,
        temperature=get_model_specs(provider_value, model)["default_temperature"],
        thinking=effort,
    )

    assert response is not None
    assert response.role == "assistant"
    assert response.get_text_content(), f"empty response from {provider_value}/{model} (effort={effort})"
