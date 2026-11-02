from typing import Any

from kolega_code.cli.config import OAUTH_PROVIDERS
from kolega_code.cli.provider_registry import default_model_for_provider
from kolega_code.config import ModelProvider
from kolega_code.llm.specs import MODEL_SPECS, default_thinking_effort, get_model_specs


_SMOKE_OVERRIDES = {
    ModelProvider.OPENROUTER: "openai/gpt-5.4-mini",
    ModelProvider.FIREWORKS: "accounts/fireworks/models/ember-1",
    ModelProvider.DASHSCOPE: "qwen3-coder-flash",
    ModelProvider.TOGETHER: "openai/gpt-oss-120b",
}


def smoke_generation_settings(provider_value: str, model: str) -> dict[str, Any]:
    if (provider_value, model) == ("together", _SMOKE_OVERRIDES[ModelProvider.TOGETHER]):
        # Test-only serverless fixture: https://docs.together.ai/docs/gpt-oss
        return {"temperature": 1.0, "thinking": None}
    return {
        "temperature": get_model_specs(provider_value, model)["default_temperature"],
        "thinking": default_thinking_effort(provider_value, model),
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
