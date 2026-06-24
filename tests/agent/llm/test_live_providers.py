"""Live provider integration tests for the whole model catalog.

These hit the real provider APIs, so they are marked ``integration`` and are
skipped for any provider whose API key is not present in the environment (loaded
from the repo ``.env`` by ``conftest.py``). They are the authoritative check that
every model ID in ``MODEL_SPECS`` actually resolves at the provider — a renamed or
retired ID surfaces here as a 4xx instead of silently shipping.

Run them explicitly with the relevant keys set, e.g.::

    pytest -m integration tests/agent/llm/test_live_providers.py -v

The provider/model matrix is derived from the catalog itself, so new models are
covered automatically.
"""

import os

import httpx
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


def _messages() -> MessageHistory:
    return MessageHistory(
        [Message(role="user", content=[TextBlock(text="What is 2 + 2? Reply with just the number.")])]
    )


def _provider_default_models() -> list[tuple[str, str]]:
    """One (provider, recommended-default-model) pair per provider in the catalog."""
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
        model = default_model_for_provider(ModelProvider(provider_value))
        seen.append((provider_value, model))
    return seen


PROVIDER_DEFAULT_MODELS = _provider_default_models()
OLLAMA_CLOUD_MODELS = [model for provider, model in MODEL_SPECS if provider == ModelProvider.OLLAMA_CLOUD.value]

# Subset whose default model exposes a thinking/reasoning-effort control.
PROVIDER_THINKING_MODELS = [
    (provider, model)
    for provider, model in PROVIDER_DEFAULT_MODELS
    if get_thinking_effort_spec(provider, model) is not None
]


def _require_key(provider_value: str) -> str:
    if SKIP_IN_CI:
        pytest.skip("Skipping live provider call in CI")
    env_name = API_KEY_ENV[ModelProvider(provider_value)]
    api_key = os.getenv(env_name)
    if not api_key:
        pytest.skip(f"{env_name} not set")
    return api_key


def test_live_ollama_cloud_model_catalog_contains_specs() -> None:
    """Ollama Cloud advertises every model ID in our supported catalog.

    The provider may advertise additional IDs that are not usable through chat
    completions yet; those should not force us to expose a broken model.
    """
    api_key = _require_key(ModelProvider.OLLAMA_CLOUD.value)
    response = httpx.get(
        "https://ollama.com/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    response.raise_for_status()

    api_models = {model["id"] for model in response.json()["data"]}

    assert set(OLLAMA_CLOUD_MODELS) <= api_models


@pytest.mark.parametrize("model", OLLAMA_CLOUD_MODELS)
def test_live_ollama_cloud_model_metadata_matches_specs(model: str) -> None:
    """Ollama Cloud metadata agrees with our context/capability flags."""
    api_key = _require_key(ModelProvider.OLLAMA_CLOUD.value)
    response = httpx.post(
        "https://ollama.com/api/show",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model},
        timeout=30,
    )
    response.raise_for_status()

    metadata = response.json()
    capabilities = set(metadata.get("capabilities") or [])
    model_info = metadata.get("model_info") or {}
    context_lengths = [value for key, value in model_info.items() if key.endswith(".context_length")]
    specs = get_model_specs(ModelProvider.OLLAMA_CLOUD.value, model)

    assert context_lengths == [specs["context_length"]]
    assert ("thinking" in capabilities) == (
        get_thinking_effort_spec(ModelProvider.OLLAMA_CLOUD.value, model) is not None
    )
    assert ("vision" in capabilities) == bool(specs.get("supports_vision"))


@pytest.mark.asyncio
@pytest.mark.parametrize("model", OLLAMA_CLOUD_MODELS)
async def test_live_ollama_cloud_model_generate(model: str) -> None:
    """Every supported Ollama Cloud model accepts our configured generation params."""
    api_key = _require_key(ModelProvider.OLLAMA_CLOUD.value)
    client = LLMClient(provider=ModelProvider.OLLAMA_CLOUD.value, api_key=api_key)
    specs = get_model_specs(ModelProvider.OLLAMA_CLOUD.value, model)

    response = await client.generate(
        messages=MessageHistory(
            [Message(role="user", content=[TextBlock(text="Reply with exactly OK and no other text.")])]
        ),
        model=model,
        max_completion_tokens=specs["max_completion_tokens"],
        temperature=specs["default_temperature"],
        thinking=default_thinking_effort(ModelProvider.OLLAMA_CLOUD.value, model),
    )

    assert response is not None
    assert response.role == "assistant"
    assert response.get_text_content(), f"empty response from ollama_cloud/{model}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_value,model",
    PROVIDER_DEFAULT_MODELS,
    ids=[p for p, _ in PROVIDER_DEFAULT_MODELS],
)
async def test_live_provider_generate(provider_value: str, model: str) -> None:
    """The provider's default model resolves and returns a non-empty assistant reply.

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
