"""Sub-agent model discovery and overrides for the OpenRouter gateway.

The discovery catalog is rendered into dispatch prompts, so it must stay bounded
even when a provider catalogs hundreds of models — while an explicitly named
model outside that listing still has to be dispatchable.
"""

import pytest

from kolega_code.agent.model_routing import (
    render_subagent_model_catalog,
    resolve_subagent_model,
    subagent_model_catalog,
)
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider
from kolega_code.llm.specs import MODEL_SPECS, is_featured_model
from kolega_code.services.lsp.config import LspConfig

OPENROUTER_MODELS = [model for provider, model in MODEL_SPECS if provider == ModelProvider.OPENROUTER.value]
FEATURED = [model for model in OPENROUTER_MODELS if is_featured_model(ModelProvider.OPENROUTER.value, model)]
UNLISTED = next(model for model in OPENROUTER_MODELS if model not in set(FEATURED))


def _config() -> AgentConfig:
    model = ModelConfig(provider=ModelProvider.OPENROUTER, model="moonshotai/kimi-k3", thinking_effort="max")
    return AgentConfig(
        openrouter_api_key="sk-or-test",
        long_context_config=model,
        fast_config=model,
        thinking_config=model,
        lsp=LspConfig(enabled=False),
    )


def test_discovery_lists_only_featured_gateway_models() -> None:
    catalog = subagent_model_catalog(_config(), ModelProvider.OPENROUTER.value)

    entry = next(item for item in catalog["providers"] if item["provider"] == ModelProvider.OPENROUTER.value)
    listed = [model["model"] for model in entry["models"]]

    assert listed == FEATURED
    assert UNLISTED not in listed


def test_rendered_catalog_stays_small_and_says_other_ids_are_accepted() -> None:
    catalog = subagent_model_catalog(_config(), ModelProvider.OPENROUTER.value)
    rendered = render_subagent_model_catalog(catalog)

    assert "list only their most-used models" in rendered
    # A full catalog would put hundreds of rows into every dispatch prompt.
    listing = rendered.split("## Configured models", 1)[1]
    assert listing.count("| `openrouter/") == len(FEATURED)


def test_unlisted_gateway_model_is_still_dispatchable() -> None:
    config = _config()
    efforts = MODEL_SPECS[(ModelProvider.OPENROUTER.value, UNLISTED)].get("thinking_effort")
    override = {
        "provider": ModelProvider.OPENROUTER.value,
        "model": UNLISTED,
        "thinking_effort": efforts.default if efforts else None,
    }

    resolution = resolve_subagent_model(config, "general-agent", override, effort_key="thinking_effort")

    assert resolution.model_config.provider == ModelProvider.OPENROUTER
    assert resolution.model_config.model == UNLISTED


def test_unknown_gateway_model_is_rejected_with_a_usable_message() -> None:
    override = {
        "provider": ModelProvider.OPENROUTER.value,
        "model": "vendor/not-a-real-model",
        "thinking_effort": None,
    }

    with pytest.raises(ValueError, match="not supported"):
        resolve_subagent_model(_config(), "general-agent", override, effort_key="thinking_effort")
