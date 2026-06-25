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

from kolega_code.cli.config import API_KEY_ENV, OAUTH_PROVIDERS
from kolega_code.cli.provider_registry import default_model_for_provider
from kolega_code.config import ModelProvider
from kolega_code.llm.client import LLMClient
from kolega_code.llm.models import Message, MessageHistory, TextBlock
from kolega_code.llm.specs import (
    MODEL_SPECS,
    default_thinking_effort,
    get_model_specs,
    get_thinking_effort_spec,
    thinking_effort_options,
)

pytestmark = pytest.mark.integration

# Skip the live calls inside CI, where keys are typically absent and we don't want
# to spend tokens on every pipeline run.
SKIP_IN_CI = bool(os.getenv("CI")) or bool(os.getenv("GITLAB_CI"))

SYSTEM = Message(
    role="system",
    content=[TextBlock(text="You are a concise math assistant. Answer with as few tokens as possible.")],
)

# Ollama Cloud has a large, mixed-access catalog. Keep live coverage to one
# non-subscription smoke model so all-test runs avoid expensive/flaky/gated calls.
OLLAMA_CLOUD_SMOKE_MODEL = "gpt-oss:20b"


def _messages() -> MessageHistory:
    return MessageHistory(
        [Message(role="user", content=[TextBlock(text="What is 2 + 2? Reply with just the number.")])]
    )


def _provider_smoke_models() -> list[tuple[str, str]]:
    """One (provider, smoke-model) pair per key-based provider in the catalog."""
    seen: list[tuple[str, str]] = []
    added: set[str] = set()
    for provider_value, _model in MODEL_SPECS:
        if provider_value in added:
            continue
        # OAuth providers authenticate via interactive sign-in, not an env key, so
        # they can't be driven from this key-based matrix (see test_chatgpt_live.py).
        if ModelProvider(provider_value) in OAUTH_PROVIDERS:
            continue
        added.add(provider_value)
        if provider_value == ModelProvider.OLLAMA_CLOUD.value:
            model = OLLAMA_CLOUD_SMOKE_MODEL
        else:
            model = default_model_for_provider(ModelProvider(provider_value))
        seen.append((provider_value, model))
    return seen


PROVIDER_SMOKE_MODELS = _provider_smoke_models()

# Subset whose smoke model exposes a thinking/reasoning-effort control. Ollama
# Cloud is intentionally excluded to keep live Ollama coverage to one smoke call.
PROVIDER_THINKING_MODELS = [
    (provider, model)
    for provider, model in PROVIDER_SMOKE_MODELS
    if provider != ModelProvider.OLLAMA_CLOUD.value and get_thinking_effort_spec(provider, model) is not None
]


def _require_key(provider_value: str) -> str:
    if SKIP_IN_CI:
        pytest.skip("Skipping live provider call in CI")
    env_name = API_KEY_ENV[ModelProvider(provider_value)]
    api_key = os.getenv(env_name)
    if not api_key:
        pytest.skip(f"{env_name} not set")
    return api_key


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

    response = await client.generate(
        messages=_messages(),
        system=SYSTEM,
        model=model,
        max_completion_tokens=4096,
        temperature=get_model_specs(provider_value, model)["default_temperature"],
        thinking=default_thinking_effort(provider_value, model),
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

    response = await client.generate(
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
