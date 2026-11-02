from kolega_code.cli.config import OAUTH_PROVIDERS
from kolega_code.cli.provider_registry import default_model_for_provider
from kolega_code.config import ModelProvider
from kolega_code.llm.specs import MODEL_SPECS


_SMOKE_OVERRIDES = {
    ModelProvider.OPENROUTER: "openai/gpt-5.4-mini",
    ModelProvider.FIREWORKS: "accounts/fireworks/models/ember-1",
    ModelProvider.DASHSCOPE: "qwen3-coder-flash",
}


def provider_smoke_models() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    added: set[str] = set()
    for provider_value, _ in MODEL_SPECS:
        provider = ModelProvider(provider_value)
        if provider_value in added or provider in OAUTH_PROVIDERS:
            continue
        added.add(provider_value)
        pairs.append((provider_value, _SMOKE_OVERRIDES.get(provider) or default_model_for_provider(provider)))
    return pairs
