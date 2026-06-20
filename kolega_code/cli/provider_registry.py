"""Provider and model registry for the CLI settings UI.

The list of models the UI exposes is derived directly from ``MODEL_SPECS`` (the
single source of truth in ``kolega_code/llm/specs.py``). Adding or removing a
model there automatically updates the Settings UI and the ``/model`` picker — no
separate whitelist to maintain.
"""

from __future__ import annotations

from dataclasses import dataclass

from kolega_code.config import AgentRole, ModelProvider
from kolega_code.llm.specs import (
    MODEL_SPECS,
    default_thinking_effort,
    get_model_specs,
    thinking_effort_options,
)

# Human-readable provider labels. Also defines the display order of providers in
# the UI. Providers with no MODEL_SPECS entries (e.g. llama, groq) simply don't
# appear.
PROVIDER_LABELS: dict[ModelProvider, str] = {
    ModelProvider.MOONSHOT: "Moonshot AI",
    ModelProvider.DEEPSEEK: "DeepSeek AI",
    ModelProvider.ZAI: "Z.AI (GLM Coding Plan)",
    ModelProvider.KIMI_CODING: "Kimi Coding Plan",
    ModelProvider.ANTHROPIC: "Anthropic",
    ModelProvider.OPENAI: "OpenAI",
    ModelProvider.OPENAI_CHATGPT: "OpenAI (ChatGPT subscription)",
    ModelProvider.GOOGLE: "Google",
    ModelProvider.XAI: "xAI",
    ModelProvider.FIREWORKS: "Fireworks",
    ModelProvider.TOGETHER: "Together AI",
    ModelProvider.DASHSCOPE: "DashScope / Qwen",
}

# Friendly display names for models. Anything not listed falls back to its raw
# model ID, so newly added models stay visible with zero extra maintenance.
MODEL_LABELS: dict[str, str] = {
    # Moonshot
    "kimi-k2.7-code": "Kimi K2.7 Code",
    "kimi-k2.7-code-highspeed": "Kimi K2.7 Code (High-Speed)",
    "kimi-k2.6": "Kimi K2.6",
    # Kimi Coding Plan
    "kimi-for-coding": "Kimi for Coding",
    # DeepSeek
    "deepseek-v4-pro": "DeepSeek V4 Pro",
    "deepseek-v4-flash": "DeepSeek V4 Flash",
    # Z.AI (GLM Coding Plan)
    "glm-5.2": "GLM-5.2",
    "glm-5.1": "GLM-5.1",
    # Anthropic
    "claude-opus-4-8": "Claude Opus 4.8",
    "claude-opus-4-7": "Claude Opus 4.7",
    "claude-opus-4-6": "Claude Opus 4.6",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
    "claude-opus-4-5-20251101": "Claude Opus 4.5",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    # OpenAI
    "gpt-5.5": "GPT-5.5",
    "gpt-5.4-mini": "GPT-5.4 Mini",
    # OpenAI via ChatGPT subscription
    "gpt-5-codex": "GPT-5 Codex",
    "gpt-5": "GPT-5",
    # Google
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "gemini-3.5-flash": "Gemini 3.5 Flash",
    # xAI
    "grok-4.3": "Grok 4.3",
    "grok-build-0.1": "Grok Build 0.1",
    # Fireworks
    "accounts/fireworks/models/glm-5p1": "GLM-5.1",
    "accounts/fireworks/models/kimi-k2p7-code": "Kimi K2.7 Code",
    # Together
    "moonshotai/Kimi-K2.7-Code": "Kimi K2.7 Code",
    "zai-org/GLM-5.1": "GLM-5.1",
    # DashScope / Qwen
    "qwen3-coder-plus": "Qwen3 Coder Plus",
    "qwen3-coder-flash": "Qwen3 Coder Flash",
}

# Per-provider default model used when only a provider is selected. Covers the
# "available set is everything, default pick is curated" split.
PROVIDER_DEFAULT_MODEL: dict[ModelProvider, str] = {
    ModelProvider.MOONSHOT: "kimi-k2.7-code",
    ModelProvider.DEEPSEEK: "deepseek-v4-pro",
    ModelProvider.ZAI: "glm-5.2",
    ModelProvider.KIMI_CODING: "kimi-for-coding",
    ModelProvider.ANTHROPIC: "claude-opus-4-8",
    ModelProvider.OPENAI: "gpt-5.5",
    ModelProvider.OPENAI_CHATGPT: "gpt-5-codex",
    ModelProvider.GOOGLE: "gemini-3.1-pro-preview",
    ModelProvider.XAI: "grok-4.3",
    ModelProvider.FIREWORKS: "accounts/fireworks/models/glm-5p1",
    ModelProvider.TOGETHER: "moonshotai/Kimi-K2.7-Code",
    ModelProvider.DASHSCOPE: "qwen3-coder-plus",
}

UI_DEFAULT_PROVIDER = ModelProvider.MOONSHOT.value
UI_DEFAULT_MODEL = "kimi-k2.7-code"
MOONSHOT_K26_MODEL = "kimi-k2.6"
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
    thinking_efforts: tuple[str, ...]
    default_thinking_effort: str | None


def _api_key_env(provider: ModelProvider) -> str:
    """Env var name holding the provider's API key (matches cli/config.API_KEY_ENV).

    OAuth providers (ChatGPT subscription) authenticate via sign-in, not an env
    key, so they have no API-key env var.
    """
    if provider == ModelProvider.OPENAI_CHATGPT:
        return ""
    return f"{provider.value.upper()}_API_KEY"


def _model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def _model_option(provider: ModelProvider, model: str) -> ModelOption:
    specs = get_model_specs(provider, model)
    return ModelOption(
        provider=provider.value,
        provider_label=PROVIDER_LABELS[provider],
        model=model,
        model_label=_model_label(model),
        api_key_env=_api_key_env(provider),
        context_length=int(specs["context_length"]),
        max_completion_tokens=int(specs["max_completion_tokens"]),
        thinking_efforts=thinking_effort_options(provider, model),
        default_thinking_effort=default_thinking_effort(provider, model),
    )


def _build_ui_model_options() -> list[ModelOption]:
    """Generate the UI model list from MODEL_SPECS, grouped by PROVIDER_LABELS order."""
    # Models per provider, preserving MODEL_SPECS insertion order.
    models_by_provider: dict[str, list[str]] = {}
    for provider_value, model in MODEL_SPECS:
        models_by_provider.setdefault(provider_value, []).append(model)

    options: list[ModelOption] = []
    for provider in PROVIDER_LABELS:
        for model in models_by_provider.get(provider.value, []):
            options.append(_model_option(provider, model))
    return options


UI_MODEL_OPTIONS = _build_ui_model_options()


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


def ui_thinking_effort_options(provider: str, model: str) -> list[tuple[str, str]]:
    """Return Textual Select options for supported model thinking efforts."""
    option = get_ui_model(provider, model)
    if option is None:
        return []
    return [(_thinking_effort_label(effort), effort) for effort in option.thinking_efforts]


def default_ui_thinking_effort(provider: str, model: str) -> str | None:
    """Return the default thinking effort for a supported UI model."""
    option = get_ui_model(provider, model)
    return option.default_thinking_effort if option is not None else None


def get_ui_model(provider: str, model: str) -> ModelOption | None:
    """Return a supported UI model option."""
    for option in UI_MODEL_OPTIONS:
        if option.provider == provider and option.model == model:
            return option
    return None


def _thinking_effort_label(effort: str) -> str:
    return {
        "auto": "Auto",
        "none": "None",
        "minimal": "Minimal",
        "low": "Low",
        "medium": "Medium",
        "high": "High",
        "xhigh": "Extra high",
        "max": "Max",
    }.get(effort, effort)


# Sentinel value used by the Settings "Agent Models" provider selects to mean
# "no override — inherit the active model". Kept distinct from any real provider id.
INHERIT_SENTINEL = "__inherit__"

# Display labels and render order for the configurable agent roles in the UI.
AGENT_ROLE_LABELS: dict[AgentRole, str] = {
    AgentRole.PLANNING: "Planning",
    AgentRole.BUILDING: "Building (Coder)",
    AgentRole.INVESTIGATION: "Investigation",
    AgentRole.GENERAL: "General",
    AgentRole.BROWSER: "Browser",
}


def agent_role_options() -> list[tuple[str, str]]:
    """Return (label, role-value) pairs for the configurable agent roles, in order."""
    return [(label, role.value) for role, label in AGENT_ROLE_LABELS.items()]


def agent_role_provider_options() -> list[tuple[str, str]]:
    """Provider Select options for a per-agent row, with an inherit option first."""
    return [("Default (inherit)", INHERIT_SENTINEL), *ui_provider_options()]


def default_model_for_provider(provider: ModelProvider) -> str:
    """Return a usable default model for a provider when only the provider is selected."""
    default = PROVIDER_DEFAULT_MODEL.get(provider)
    if default is not None:
        return default
    # Fall back to the first model the catalog exposes for this provider.
    for option in UI_MODEL_OPTIONS:
        if option.provider == provider.value:
            return option.model
    raise ValueError(f"No default CLI model is registered for provider '{provider.value}'.")
