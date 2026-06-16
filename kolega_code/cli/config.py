"""Configuration helpers for the Kolega Code CLI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from dotenv import dotenv_values

from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.llm.specs import get_model_specs, normalize_thinking_effort

from .provider_registry import default_model_for_provider
from .settings import CliSettings

DEFAULT_LONG_PROVIDER = ModelProvider.ANTHROPIC
DEFAULT_LONG_MODEL = "claude-opus-4-8"
DEFAULT_FAST_PROVIDER = ModelProvider.ANTHROPIC
DEFAULT_FAST_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_THINKING_PROVIDER = ModelProvider.ANTHROPIC
DEFAULT_THINKING_MODEL = "claude-opus-4-8"
DEPRECATED_THINKING_TOKENS_MESSAGE = (
    "Thinking token budgets have been replaced by model-specific named effort. "
    "Use --thinking-effort or KOLEGA_CODE_THINKING_EFFORT."
)
MISSING_MODEL_SELECTION_MESSAGE = (
    "No provider/model configured. Choose a provider and model in Settings, "
    "or set --provider/--model or KOLEGA_CODE_PROVIDER/KOLEGA_CODE_MODEL."
)

API_KEY_ENV = {
    ModelProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
    ModelProvider.OPENAI: "OPENAI_API_KEY",
    ModelProvider.GOOGLE: "GOOGLE_API_KEY",
    ModelProvider.GROQ: "GROQ_API_KEY",
    ModelProvider.TOGETHER: "TOGETHER_API_KEY",
    ModelProvider.FIREWORKS: "FIREWORKS_API_KEY",
    ModelProvider.XAI: "XAI_API_KEY",
    ModelProvider.DASHSCOPE: "DASHSCOPE_API_KEY",
    ModelProvider.MOONSHOT: "MOONSHOT_API_KEY",
    ModelProvider.DEEPSEEK: "DEEPSEEK_API_KEY",
    ModelProvider.ZAI: "ZAI_API_KEY",
}


class CliConfigError(ValueError):
    """Raised when CLI configuration is incomplete or invalid."""


@dataclass(frozen=True)
class CliConfigOverrides:
    """Model and provider overrides supplied by CLI flags."""

    provider: Optional[str] = None
    model: Optional[str] = None
    fast_provider: Optional[str] = None
    fast_model: Optional[str] = None
    thinking_provider: Optional[str] = None
    thinking_model: Optional[str] = None
    thinking_effort: Optional[str] = None
    environment: Optional[str] = None


def load_cli_env(project_path: Path, env: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """Load process environment over a project-local .env file."""
    base_env = dict(env if env is not None else os.environ)
    dotenv_path = project_path / ".env"
    if not dotenv_path.exists():
        return base_env

    file_env = {key: value for key, value in dotenv_values(dotenv_path).items() if value is not None}
    return {**file_env, **base_env}


def _provider(value: str) -> ModelProvider:
    try:
        return ModelProvider(value.lower())
    except ValueError as exc:
        valid = ", ".join(provider.value for provider in ModelProvider)
        raise CliConfigError(f"Unsupported provider '{value}'. Valid providers: {valid}") from exc


def _api_key_for_provider(
    provider: ModelProvider,
    env: Mapping[str, str],
    settings: Optional[CliSettings],
) -> Optional[str]:
    env_name = API_KEY_ENV.get(provider)
    if env_name and env.get(env_name):
        return env[env_name]
    if settings:
        return settings.get_api_key(provider.value)
    return None


def _active_provider_model(
    env: Mapping[str, str],
    overrides: CliConfigOverrides,
    settings: Optional[CliSettings],
) -> tuple[Optional[ModelProvider], Optional[str]]:
    provider_value = overrides.provider or env.get("KOLEGA_CODE_PROVIDER")
    model_value = overrides.model or env.get("KOLEGA_CODE_MODEL")

    if provider_value or model_value:
        provider = _provider(provider_value or DEFAULT_LONG_PROVIDER.value)
        return provider, model_value or default_model_for_provider(provider)

    if settings and settings.active_provider and settings.active_model:
        provider = _provider(settings.active_provider)
        return provider, settings.active_model

    return None, None


def _slot_provider_model(
    env: Mapping[str, str],
    provider_env_key: str,
    model_env_key: str,
    provider_override: Optional[str],
    model_override: Optional[str],
    default_provider: ModelProvider,
    default_model: str,
    active_provider: Optional[ModelProvider],
    active_model: Optional[str],
) -> tuple[ModelProvider, str]:
    provider_value = provider_override or env.get(provider_env_key)
    model_value = model_override or env.get(model_env_key)

    if provider_value or model_value:
        provider = _provider(provider_value or (active_provider.value if active_provider else default_provider.value))
        return provider, model_value or (
            active_model if active_provider == provider and active_model else default_model_for_provider(provider)
        )

    if active_provider and active_model:
        return active_provider, active_model

    return default_provider, default_model


def _model_config(provider: ModelProvider, model: str, thinking_effort: Optional[str] = None) -> ModelConfig:
    try:
        get_model_specs(provider, model)
        resolved_thinking_effort = normalize_thinking_effort(provider, model, thinking_effort)
    except ValueError as exc:
        raise CliConfigError(str(exc)) from exc

    return ModelConfig(
        provider=provider,
        model=model,
        rate_limits=RateLimitConfig(),
        thinking_effort=resolved_thinking_effort,
    )


def _resolve_active_thinking_effort(
    provider: ModelProvider,
    model: str,
    env: Mapping[str, str],
    overrides: CliConfigOverrides,
    settings: Optional[CliSettings],
) -> Optional[str]:
    explicit_effort = overrides.thinking_effort or env.get("KOLEGA_CODE_THINKING_EFFORT")
    if explicit_effort is None and settings:
        settings_model_matches = settings.active_provider == provider.value and settings.active_model == model
        if settings_model_matches:
            explicit_effort = settings.active_thinking_effort
    try:
        return normalize_thinking_effort(provider, model, explicit_effort)
    except ValueError as exc:
        raise CliConfigError(str(exc)) from exc


def build_agent_config(
    project_path: Path,
    overrides: Optional[CliConfigOverrides] = None,
    env: Optional[Mapping[str, str]] = None,
    settings: Optional[CliSettings] = None,
) -> AgentConfig:
    """Build an AgentConfig for CLI-hosted agents."""
    overrides = overrides or CliConfigOverrides()
    loaded_env = load_cli_env(project_path, env)
    if "KOLEGA_CODE_THINKING_TOKENS" in loaded_env:
        raise CliConfigError(DEPRECATED_THINKING_TOKENS_MESSAGE)

    active_provider, active_model = _active_provider_model(loaded_env, overrides, settings)
    if active_provider is None or active_model is None:
        raise CliConfigError(MISSING_MODEL_SELECTION_MESSAGE)

    long_provider, long_model = _slot_provider_model(
        loaded_env,
        "KOLEGA_CODE_PROVIDER",
        "KOLEGA_CODE_MODEL",
        overrides.provider,
        overrides.model,
        DEFAULT_LONG_PROVIDER,
        DEFAULT_LONG_MODEL,
        active_provider,
        active_model,
    )
    fast_provider, fast_model = _slot_provider_model(
        loaded_env,
        "KOLEGA_CODE_FAST_PROVIDER",
        "KOLEGA_CODE_FAST_MODEL",
        overrides.fast_provider,
        overrides.fast_model,
        DEFAULT_FAST_PROVIDER,
        DEFAULT_FAST_MODEL,
        active_provider,
        active_model,
    )
    thinking_provider, thinking_model = _slot_provider_model(
        loaded_env,
        "KOLEGA_CODE_THINKING_PROVIDER",
        "KOLEGA_CODE_THINKING_MODEL",
        overrides.thinking_provider,
        overrides.thinking_model,
        DEFAULT_THINKING_PROVIDER,
        DEFAULT_THINKING_MODEL,
        active_provider,
        active_model,
    )
    active_thinking_effort = _resolve_active_thinking_effort(
        long_provider,
        long_model,
        loaded_env,
        overrides,
        settings,
    )
    think_hard_effort = (
        active_thinking_effort
        if thinking_provider == long_provider and thinking_model == long_model
        else None
    )

    required_providers = {long_provider, fast_provider, thinking_provider}
    missing_keys = [
        API_KEY_ENV[provider]
        for provider in sorted(required_providers, key=lambda item: item.value)
        if provider != ModelProvider.LLAMA and not _api_key_for_provider(provider, loaded_env, settings)
    ]
    if missing_keys:
        raise CliConfigError(f"Missing required API key environment variable(s): {', '.join(missing_keys)}")

    try:
        return AgentConfig(
            anthropic_api_key=_api_key_for_provider(ModelProvider.ANTHROPIC, loaded_env, settings),
            openai_api_key=_api_key_for_provider(ModelProvider.OPENAI, loaded_env, settings),
            google_api_key=_api_key_for_provider(ModelProvider.GOOGLE, loaded_env, settings),
            groq_api_key=_api_key_for_provider(ModelProvider.GROQ, loaded_env, settings),
            together_api_key=_api_key_for_provider(ModelProvider.TOGETHER, loaded_env, settings),
            fireworks_api_key=_api_key_for_provider(ModelProvider.FIREWORKS, loaded_env, settings),
            xai_api_key=_api_key_for_provider(ModelProvider.XAI, loaded_env, settings),
            dashscope_api_key=_api_key_for_provider(ModelProvider.DASHSCOPE, loaded_env, settings),
            moonshot_api_key=_api_key_for_provider(ModelProvider.MOONSHOT, loaded_env, settings),
            deepseek_api_key=_api_key_for_provider(ModelProvider.DEEPSEEK, loaded_env, settings),
            zai_api_key=_api_key_for_provider(ModelProvider.ZAI, loaded_env, settings),
            environment=overrides.environment or loaded_env.get("KOLEGA_CODE_ENVIRONMENT", "development"),
            long_context_config=_model_config(long_provider, long_model, thinking_effort=active_thinking_effort),
            fast_config=_model_config(fast_provider, fast_model),
            thinking_config=_model_config(thinking_provider, thinking_model, thinking_effort=think_hard_effort),
        )
    except ValueError as exc:
        raise CliConfigError(str(exc)) from exc


def key_status(provider: str, project_path: Path, settings: Optional[CliSettings] = None) -> str:
    """Return the API-key source for display without exposing the key."""
    provider_value = _provider(provider)
    env = load_cli_env(project_path)
    env_name = API_KEY_ENV.get(provider_value)
    if env_name and env.get(env_name):
        return f"present via {env_name}"
    if settings and settings.get_api_key(provider_value.value):
        return "present in local settings"
    return "missing"


def config_summary(config: AgentConfig) -> dict[str, str | int | None]:
    """Return a session-safe summary of model configuration."""
    return {
        "environment": config.environment,
        "long_provider": config.long_context_config.provider.value,
        "long_model": config.long_context_config.model,
        "fast_provider": config.fast_config.provider.value,
        "fast_model": config.fast_config.model,
        "thinking_provider": config.thinking_config.provider.value,
        "thinking_model": config.thinking_config.model,
        "thinking_effort": config.long_context_config.thinking_effort,
    }
