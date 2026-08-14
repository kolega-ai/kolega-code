"""Provider and model registry for the CLI settings UI.

The list of models the UI exposes is derived directly from ``MODEL_SPECS`` (the
single source of truth in the ``kolega_code/llm/specs/`` catalog). Adding or removing a
model there automatically updates the Settings UI and the ``/model`` picker — no
separate whitelist to maintain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from kolega_code.config import AgentRole, ModelProvider
from kolega_code.llm.specs import (
    MODEL_SPECS,
    CUSTOM_PROVIDER_PREFIX,
    default_thinking_effort,
    get_model_specs,
    is_custom_provider,
    is_featured_model,
    model_is_known,
    provider_has_featured_models,
    provider_has_wildcard_models,
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
    ModelProvider.OLLAMA_CLOUD: "Ollama Cloud",
    ModelProvider.OPENROUTER: "OpenRouter",
    ModelProvider.THINKING_MACHINES: "Thinking Machines",
    ModelProvider.TINKER: "Tinker",
    ModelProvider.PERPLEXITY_AGENT: "Perplexity Agent API",
}

# Friendly display names for models. Anything not listed falls back to its raw
# model ID, so newly added models stay visible with zero extra maintenance.
MODEL_LABELS: dict[str, str] = {
    # Moonshot
    "kimi-k3": "Kimi K3",
    "kimi-k2.7-code": "Kimi K2.7 Code",
    "kimi-k2.7-code-highspeed": "Kimi K2.7 Code (High-Speed)",
    "kimi-k2.6": "Kimi K2.6",
    # Kimi Coding Plan
    "k3": "Kimi K3 (256K)",
    "k3-256k": "Kimi K3-256K",
    "kimi-for-coding": "Kimi for Coding",
    "kimi-for-coding-highspeed": "Kimi for Coding (HighSpeed)",
    # DeepSeek
    "deepseek-v4-pro": "DeepSeek V4 Pro",
    "deepseek-v4-flash": "DeepSeek V4 Flash",
    # Z.AI (GLM Coding Plan)
    "glm-5.3": "GLM-5.3",
    "glm-5.2": "GLM-5.2",
    "glm-5.1": "GLM-5.1",
    # Anthropic
    "claude-fable-5": "Claude Fable 5",
    "claude-opus-5": "Claude Opus 5",
    "claude-opus-4-8": "Claude Opus 4.8",
    "claude-opus-4-7": "Claude Opus 4.7",
    "claude-opus-4-6": "Claude Opus 4.6",
    "claude-sonnet-5": "Claude Sonnet 5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
    "claude-opus-4-5-20251101": "Claude Opus 4.5",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    # OpenAI (shared labels across the API and ChatGPT-subscription providers)
    "gpt-5.6-sol": "GPT-5.6 Sol",
    "gpt-5.6-terra": "GPT-5.6 Terra",
    "gpt-5.6-luna": "GPT-5.6 Luna",
    "gpt-5.5": "GPT-5.5",
    "gpt-5.4": "GPT-5.4",
    "gpt-5.4-mini": "GPT-5.4 Mini",
    "gpt-5.3-codex-spark": "GPT-5.3 Codex Spark",
    # Google
    "gemini-3.6-flash": "Gemini 3.6 Flash",
    "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "gemini-3.5-flash": "Gemini 3.5 Flash",
    # xAI
    "grok-4.6": "Grok 4.6",
    "grok-4.5": "Grok 4.5",
    "grok-4.3": "Grok 4.3",
    "grok-build-0.1": "Grok Build 0.1",
    # Fireworks
    "accounts/fireworks/models/glm-5p2": "GLM-5.2",
    "accounts/fireworks/models/glm-5p1": "GLM-5.1",
    "accounts/fireworks/models/kimi-k3": "Kimi K3",
    "accounts/fireworks/models/kimi-k2p7-code": "Kimi K2.7 Code",
    "accounts/fireworks/models/deepseek-v4-pro": "DeepSeek V4 Pro",
    "accounts/fireworks/models/deepseek-v4-flash": "DeepSeek V4 Flash",
    "accounts/fireworks/models/minimax-m3": "MiniMax M3",
    "accounts/fireworks/models/qwen3p7-plus": "Qwen 3.7 Plus",
    # Together
    "moonshotai/Kimi-K2.7-Code": "Kimi K2.7 Code",
    "zai-org/GLM-5.1": "GLM-5.1",
    # DashScope / Qwen
    "qwen3-coder-plus": "Qwen3 Coder Plus",
    "qwen3-coder-flash": "Qwen3 Coder Flash",
    # Ollama Cloud
    "deepseek-v3.1:671b": "DeepSeek V3.1 671B",
    "deepseek-v3.2": "DeepSeek V3.2",
    "devstral-2:123b": "Devstral 2 123B",
    "devstral-small-2:24b": "Devstral Small 2 24B",
    "gemini-3-flash-preview": "Gemini 3 Flash Preview",
    "gemma3:4b": "Gemma 3 4B",
    "gemma3:12b": "Gemma 3 12B",
    "gemma3:27b": "Gemma 3 27B",
    "gemma4:31b": "Gemma 4 31B",
    "glm-4.7": "GLM-4.7",
    "glm-5": "GLM-5",
    "gpt-oss:20b": "GPT-OSS 20B",
    "gpt-oss:120b": "GPT-OSS 120B",
    "kimi-k2.5": "Kimi K2.5",
    "minimax-m2.1": "MiniMax M2.1",
    "minimax-m2.5": "MiniMax M2.5",
    "minimax-m2.7": "MiniMax M2.7",
    "minimax-m3": "MiniMax M3",
    "ministral-3:3b": "Ministral 3 3B",
    "ministral-3:8b": "Ministral 3 8B",
    "ministral-3:14b": "Ministral 3 14B",
    "mistral-large-3:675b": "Mistral Large 3 675B",
    "nemotron-3-nano:30b": "Nemotron 3 Nano 30B",
    "nemotron-3-super": "Nemotron 3 Super",
    "nemotron-3-ultra": "Nemotron 3 Ultra",
    "qwen3-coder-next": "Qwen3 Coder Next",
    "qwen3-coder:480b": "Qwen3 Coder 480B",
    "qwen3.5:397b": "Qwen3.5 397B",
    # Thinking Machines
    "thinkingmachines/Inkling": "Inkling",
    "thinkingmachines/Inkling-Small": "Inkling Small",
}

# Per-provider default model used when only a provider is selected. Covers the
# "available set is everything, default pick is curated" split.
PROVIDER_DEFAULT_MODEL: dict[ModelProvider, str] = {
    ModelProvider.MOONSHOT: "kimi-k3",
    ModelProvider.DEEPSEEK: "deepseek-v4-pro",
    ModelProvider.ZAI: "glm-5.3",
    ModelProvider.KIMI_CODING: "kimi-for-coding",
    ModelProvider.ANTHROPIC: "claude-opus-5",
    ModelProvider.OPENAI: "gpt-5.6-sol",
    ModelProvider.OPENAI_CHATGPT: "gpt-5.6-sol",
    ModelProvider.GOOGLE: "gemini-3.1-pro-preview",
    ModelProvider.XAI: "grok-4.5",
    ModelProvider.FIREWORKS: "accounts/fireworks/models/glm-5p2",
    ModelProvider.TOGETHER: "moonshotai/Kimi-K2.7-Code",
    ModelProvider.DASHSCOPE: "qwen3-coder-plus",
    ModelProvider.OLLAMA_CLOUD: "gpt-oss:20b",
    # Hardcoded on purpose: regenerating the OpenRouter catalog reorders it by
    # live usage, and that must never silently change what new users get.
    ModelProvider.OPENROUTER: "moonshotai/kimi-k3",
    ModelProvider.THINKING_MACHINES: "thinkingmachines/Inkling",
    # Cheap and renderable without the optional tinker-inkling extra, so the
    # provider works out of the box for anyone who installs the base extras.
    ModelProvider.TINKER: "Qwen/Qwen3-8B",
    ModelProvider.PERPLEXITY_AGENT: "openai/gpt-5.6-sol",
}

UI_DEFAULT_PROVIDER = ModelProvider.MOONSHOT.value
UI_DEFAULT_MODEL = "kimi-k3"
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
    supports_vision: bool
    thinking_efforts: tuple[str, ...]
    default_thinking_effort: str | None
    # Gateway providers catalog hundreds of models. Only featured ones are
    # listed in pickers; every catalogued model stays resolvable by id.
    featured: bool = False


def _api_key_env(provider: ModelProvider) -> str:
    """Env var name holding the provider's API key (matches cli/config.API_KEY_ENV).

    OAuth providers (ChatGPT subscription) authenticate via sign-in, not an env
    key, so they have no API-key env var. Custom endpoints carry their optional
    key on the endpoint definition.
    """
    if is_custom_provider(provider):
        return ""
    if provider == ModelProvider.OPENAI_CHATGPT:
        return ""
    if provider == ModelProvider.OLLAMA_CLOUD:
        return "OLLAMA_API_KEY"
    if provider == ModelProvider.THINKING_MACHINES:
        # Thinking Machines documents TINKER_API_KEY, not THINKING_MACHINES_API_KEY.
        return "TINKER_API_KEY"
    if provider == ModelProvider.TINKER:
        return "TINKER_API_KEY"
    if provider == ModelProvider.PERPLEXITY_AGENT:
        return "PERPLEXITY_API_KEY"
    return f"{provider.value.upper()}_API_KEY"


# Display labels for custom endpoint providers, keyed by endpoint id. Refreshed
# by build_agent_config from the resolved endpoint definitions.
CUSTOM_PROVIDER_LABELS: dict[str, str] = {}


def set_custom_provider_labels(labels: Mapping[str, str]) -> None:
    CUSTOM_PROVIDER_LABELS.clear()
    CUSTOM_PROVIDER_LABELS.update(labels)


def _provider_label(provider: ModelProvider) -> str:
    if is_custom_provider(provider):
        endpoint_id = provider.value[len(CUSTOM_PROVIDER_PREFIX) :]
        return CUSTOM_PROVIDER_LABELS.get(endpoint_id, endpoint_id)
    return PROVIDER_LABELS[provider]


def _model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def _model_option(provider: ModelProvider, model: str) -> ModelOption:
    specs = get_model_specs(provider, model)
    return ModelOption(
        provider=provider.value,
        provider_label=_provider_label(provider),
        model=model,
        model_label=_model_label(model),
        api_key_env=_api_key_env(provider),
        context_length=int(specs["context_length"]),
        max_completion_tokens=int(specs["max_completion_tokens"]),
        supports_vision=bool(specs.get("supports_vision", False)),
        thinking_efforts=thinking_effort_options(provider, model),
        default_thinking_effort=default_thinking_effort(provider, model),
        featured=is_featured_model(provider, model),
    )


def _build_ui_model_options() -> list[ModelOption]:
    """Generate the UI model list from MODEL_SPECS, grouped by PROVIDER_LABELS order."""
    # Models per provider, preserving MODEL_SPECS insertion order. For gateway
    # catalogs that order is the provider's own popularity ranking.
    models_by_provider: dict[str, list[str]] = {}
    for provider_value, model in MODEL_SPECS:
        models_by_provider.setdefault(provider_value, []).append(model)

    options: list[ModelOption] = []
    for provider in PROVIDER_LABELS:
        for model in models_by_provider.get(provider.value, []):
            options.append(_model_option(provider, model))
    for provider_value in sorted(value for value in models_by_provider if value.startswith(CUSTOM_PROVIDER_PREFIX)):
        for model in models_by_provider[provider_value]:
            options.append(_model_option(ModelProvider(provider_value), model))
    return options


UI_MODEL_OPTIONS = _build_ui_model_options()


def rebuild_ui_model_options() -> list[ModelOption]:
    """Rebuild the cached UI model list after MODEL_SPECS gains entries.

    ``UI_MODEL_OPTIONS`` is materialized at import time, so a runtime catalog
    overlay (``kolega-code models refresh``) has to ask for a rebuild.
    """
    global UI_MODEL_OPTIONS
    UI_MODEL_OPTIONS = _build_ui_model_options()
    return UI_MODEL_OPTIONS


def ui_provider_options() -> list[tuple[str, str]]:
    """Return Textual Select options for supported UI providers."""
    seen: set[str] = set()
    options: list[tuple[str, str]] = []
    for option in UI_MODEL_OPTIONS:
        if option.provider in seen:
            continue
        seen.add(option.provider)
        options.append((option.provider_label, option.provider))
    # Custom endpoints without exact model entries (wildcard-only) still belong
    # in the provider picker; their models are typed via the "Other…" entry.
    for endpoint_id in sorted(CUSTOM_PROVIDER_LABELS):
        provider_value = f"{CUSTOM_PROVIDER_PREFIX}{endpoint_id}"
        if provider_value not in seen:
            options.append((CUSTOM_PROVIDER_LABELS[endpoint_id], provider_value))
    return options


def ui_model_options(provider: str, *, vision_only: bool = False) -> list[tuple[str, str]]:
    """Return Textual Select options for supported UI models.

    A provider that marks some models ``featured`` (currently only the
    OpenRouter gateway, whose catalog runs to hundreds of entries) is listed by
    its featured models alone. Everything else is listed in full, unchanged.
    Non-featured models stay reachable by id via ``--model``, ``/model``,
    ``KOLEGA_CODE_MODEL`` and ``settings.json``.
    """
    featured_only = provider_has_featured_models(provider)
    return [
        (option.model_label, option.model)
        for option in UI_MODEL_OPTIONS
        if option.provider == provider
        and (option.supports_vision or not vision_only)
        and (option.featured or not featured_only)
    ]


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
    """Return a supported UI model option.

    Also resolves wildcard-addressed models (Tinker sampler checkpoint paths):
    they are not listed in pickers, but typed ids must resolve so ``/model``,
    the Settings "Other…" entry, and settings.json all accept them.
    """
    for option in UI_MODEL_OPTIONS:
        if option.provider == provider and option.model == model:
            return option
    if provider_has_wildcard_models(provider) and model_is_known(provider, model):
        return _model_option(ModelProvider(provider), model)
    return None


def _thinking_effort_label(effort: str) -> str:
    return {
        "auto": "Auto",
        "none": "None",
        "enabled": "Enabled",
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


# Display labels and render order for the operational model slots the UI exposes.
# The long-context slot is the active model itself, so it is not listed here.
MODEL_SLOT_LABELS: dict[str, str] = {
    "fast": "Fast",
}


def agent_role_options() -> list[tuple[str, str]]:
    """Return (label, role-value) pairs for the configurable agent roles, in order."""
    return [(label, role.value) for role, label in AGENT_ROLE_LABELS.items()]


def model_slot_options() -> list[tuple[str, str]]:
    """Return (label, slot-value) pairs for the configurable model slots, in order."""
    return [(label, slot) for slot, label in MODEL_SLOT_LABELS.items()]


def agent_role_provider_options() -> list[tuple[str, str]]:
    """Provider Select options for an override row, with an inherit option first.

    Shared by the per-agent rows and the model-slot rows: both mean "no override —
    inherit the active model" when the sentinel is selected.
    """
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
