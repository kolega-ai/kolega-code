"""Provider and model registry for the CLI settings UI."""

from __future__ import annotations

from dataclasses import dataclass

from kolega_code.config import ModelProvider
from kolega_code.llm.specs import get_model_specs

UI_DEFAULT_PROVIDER = ModelProvider.MOONSHOT.value
UI_DEFAULT_MODEL = "kimi-k2.6"
DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-pro"


@dataclass(frozen=True)
class ModelOption:
    provider: str
    provider_label: str
    model: str
    model_label: str
    api_key_env: str
    context_length: int
    max_completion_tokens: int


def _model_option(
    provider: ModelProvider,
    provider_label: str,
    model: str,
    model_label: str,
    api_key_env: str,
) -> ModelOption:
    specs = get_model_specs(provider, model)
    return ModelOption(
        provider=provider.value,
        provider_label=provider_label,
        model=model,
        model_label=model_label,
        api_key_env=api_key_env,
        context_length=int(specs["context_length"]),
        max_completion_tokens=int(specs["max_completion_tokens"]),
    )


UI_MODEL_OPTIONS = [
    _model_option(
        ModelProvider.MOONSHOT,
        "Moonshot AI",
        UI_DEFAULT_MODEL,
        "Kimi K2.6",
        "MOONSHOT_API_KEY",
    ),
    _model_option(
        ModelProvider.DEEPSEEK,
        "DeepSeek AI",
        DEEPSEEK_DEFAULT_MODEL,
        "DeepSeek V4 Pro",
        "DEEPSEEK_API_KEY",
    ),
]


def ui_provider_options() -> list[tuple[str, str]]:
    """Return Textual Select options for supported UI providers."""
    seen: set[str] = set()
    options: list[tuple[str, str]] = []
    for option in UI_MODEL_OPTIONS:
        if option.provider in seen:
            continue
        seen.add(option.provider)
        options.append((option.provider_label, option.provider))
    return options


def ui_model_options(provider: str) -> list[tuple[str, str]]:
    """Return Textual Select options for supported UI models."""
    return [(option.model_label, option.model) for option in UI_MODEL_OPTIONS if option.provider == provider]


def get_ui_model(provider: str, model: str) -> ModelOption | None:
    """Return a supported UI model option."""
    for option in UI_MODEL_OPTIONS:
        if option.provider == provider and option.model == model:
            return option
    return None


def default_model_for_provider(provider: ModelProvider) -> str:
    """Return a usable default model for a provider when only the provider is selected."""
    if provider == ModelProvider.MOONSHOT:
        return UI_DEFAULT_MODEL
    if provider == ModelProvider.DEEPSEEK:
        return DEEPSEEK_DEFAULT_MODEL
    if provider == ModelProvider.ANTHROPIC:
        return "claude-opus-4-7"
    raise ValueError(f"No default CLI model is registered for provider '{provider.value}'.")
