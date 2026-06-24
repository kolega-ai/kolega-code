"""Configuration helpers for the Kolega Code CLI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from dotenv import dotenv_values

from pydantic import ValidationError

from kolega_code.auth.tokens import ChatGPTTokenManager, OAuthTokens
from kolega_code.config import AgentConfig, AgentRole, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.llm.specs import MODEL_SPECS, get_model_specs, normalize_thinking_effort

from .provider_registry import default_model_for_provider
from .settings import CliSettings, SettingsStore

# Providers authenticated by an OAuth sign-in instead of a static API key.
OAUTH_PROVIDERS = frozenset({ModelProvider.OPENAI_CHATGPT})

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
    ModelProvider.KIMI_CODING: "KIMI_CODING_API_KEY",
    ModelProvider.OLLAMA_CLOUD: "OLLAMA_API_KEY",
}

DEFAULT_WEB_SEARCH_BACKEND = "duckduckgo"
WEB_SEARCH_BACKEND_ENV = "KOLEGA_CODE_WEB_SEARCH_BACKEND"
SEARXNG_BASE_URL_ENV = "SEARXNG_BASE_URL"
# Env vars that supply a cloud web-search backend's key, keyed by backend name.
SEARCH_BACKEND_KEY_ENV = {
    "firecrawl": "FIRECRAWL_API_KEY",
    "tavily": "TAVILY_API_KEY",
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


def _resolve_chatgpt_tokens(settings: Optional[CliSettings]) -> Optional[OAuthTokens]:
    """Load stored ChatGPT OAuth tokens from settings, if present and well-formed."""
    if not settings:
        return None
    raw = settings.get_oauth_token(ModelProvider.OPENAI_CHATGPT.value)
    if not raw:
        return None
    try:
        return OAuthTokens.model_validate(raw)
    except ValidationError:
        return None


def _build_chatgpt_token_manager(
    tokens: OAuthTokens,
    settings: CliSettings,
    settings_store: SettingsStore,
) -> ChatGPTTokenManager:
    """A token manager whose refreshes are persisted back to settings.json."""

    def _persist(new_tokens: OAuthTokens) -> None:
        settings.set_oauth_token(ModelProvider.OPENAI_CHATGPT.value, new_tokens.model_dump(mode="json"))
        settings_store.save(settings)

    return ChatGPTTokenManager(tokens, persist=_persist)


def _search_config(
    env: Mapping[str, str],
    settings: Optional[CliSettings],
) -> tuple[str, Optional[str], Optional[str]]:
    """Resolve (backend, api_key, base_url) for the web_search tool.

    Backend keys follow the same env-over-settings precedence as model-provider keys
    (see ``_api_key_for_provider``). A missing key is never an error here — the default
    backend is keyless, and a missing cloud key surfaces at tool-call time instead.
    """
    backend = (
        env.get(WEB_SEARCH_BACKEND_ENV)
        or (settings.web_search_backend if settings else None)
        or DEFAULT_WEB_SEARCH_BACKEND
    )
    env_name = SEARCH_BACKEND_KEY_ENV.get(backend)
    api_key = (env.get(env_name) if env_name else None) or (settings.get_api_key(backend) if settings else None)
    base_url = env.get(SEARXNG_BASE_URL_ENV) or (settings.web_search_base_url if settings else None)
    return backend, api_key, base_url


def _coerce_known_model(provider: ModelProvider, model: Optional[str]) -> str:
    """Return ``model`` if it's a known spec for ``provider``, else the default.

    Guards against a settings.json that points at a model which has since been
    renamed or removed (e.g. an old ChatGPT slug): without this, config building
    raises and the TUI disables the composer, locking the user out.
    """
    if model and (provider.value, model) in MODEL_SPECS:
        return model
    return default_model_for_provider(provider)


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
        try:
            provider = _provider(settings.active_provider)
        except CliConfigError:
            return None, None
        return provider, _coerce_known_model(provider, settings.active_model)

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


def _agent_role_env_keys(role: AgentRole) -> tuple[str, str, str]:
    """Env var names that override a role's provider/model/effort, e.g.
    KOLEGA_CODE_INVESTIGATION_PROVIDER / _MODEL / _EFFORT."""
    token = role.value.upper()
    return (
        f"KOLEGA_CODE_{token}_PROVIDER",
        f"KOLEGA_CODE_{token}_MODEL",
        f"KOLEGA_CODE_{token}_EFFORT",
    )


def _agent_model_overrides(
    env: Mapping[str, str],
    settings: Optional[CliSettings],
    active_provider: ModelProvider,
    active_model: str,
) -> dict[str, ModelConfig]:
    """Resolve per-agent-role model overrides from env vars over saved settings.

    A role with neither an env nor a settings provider/model is omitted, so it
    inherits the active model. Field-level precedence is env > settings, mirroring
    the per-slot resolution used for long/fast/thinking.
    """
    saved = settings.agent_models if settings else {}
    overrides: dict[str, ModelConfig] = {}
    for role in AgentRole:
        provider_key, model_key, effort_key = _agent_role_env_keys(role)
        entry = saved.get(role.value) or {}
        provider_value = env.get(provider_key) or entry.get("provider")
        model_value = env.get(model_key) or entry.get("model")
        effort_value = env.get(effort_key) or entry.get("thinking_effort")

        if not provider_value and not model_value:
            continue

        provider = _provider(provider_value or active_provider.value)
        if model_value:
            model = model_value
        elif active_provider == provider and active_model:
            model = active_model
        else:
            model = default_model_for_provider(provider)
        overrides[role.value] = _model_config(provider, model, thinking_effort=effort_value)
    return overrides


def build_agent_config(
    project_path: Path,
    overrides: Optional[CliConfigOverrides] = None,
    env: Optional[Mapping[str, str]] = None,
    settings: Optional[CliSettings] = None,
    settings_store: Optional[SettingsStore] = None,
) -> AgentConfig:
    """Build an AgentConfig for CLI-hosted agents.

    When ``settings_store`` is provided and a ChatGPT-subscription provider is in
    use, a persisting token manager is attached so refreshed tokens survive
    restarts.
    """
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
        active_thinking_effort if thinking_provider == long_provider and thinking_model == long_model else None
    )

    agent_model_overrides = _agent_model_overrides(loaded_env, settings, active_provider, active_model)
    web_search_backend, web_search_api_key, web_search_base_url = _search_config(loaded_env, settings)

    required_providers = {long_provider, fast_provider, thinking_provider}
    required_providers.update(override.provider for override in agent_model_overrides.values())
    # API-key providers: env/settings key required. OAuth and local providers are
    # exempt (OAuth is checked via stored tokens below; LLAMA is keyless).
    missing_keys = [
        API_KEY_ENV[provider]
        for provider in sorted(required_providers, key=lambda item: item.value)
        if provider != ModelProvider.LLAMA
        and provider not in OAUTH_PROVIDERS
        and not _api_key_for_provider(provider, loaded_env, settings)
    ]
    if missing_keys:
        raise CliConfigError(f"Missing required API key environment variable(s): {', '.join(missing_keys)}")

    chatgpt_tokens = _resolve_chatgpt_tokens(settings)
    if ModelProvider.OPENAI_CHATGPT in required_providers and chatgpt_tokens is None:
        raise CliConfigError("Not signed in to ChatGPT. Run /login chatgpt to sign in with your ChatGPT subscription.")

    try:
        config = AgentConfig(
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
            kimi_coding_api_key=_api_key_for_provider(ModelProvider.KIMI_CODING, loaded_env, settings),
            ollama_cloud_api_key=_api_key_for_provider(ModelProvider.OLLAMA_CLOUD, loaded_env, settings),
            environment=overrides.environment or loaded_env.get("KOLEGA_CODE_ENVIRONMENT", "development"),
            long_context_config=_model_config(long_provider, long_model, thinking_effort=active_thinking_effort),
            fast_config=_model_config(fast_provider, fast_model),
            thinking_config=_model_config(thinking_provider, thinking_model, thinking_effort=think_hard_effort),
            agent_models=agent_model_overrides,
            web_search_backend=web_search_backend,
            web_search_api_key=web_search_api_key,
            web_search_base_url=web_search_base_url,
            openai_chatgpt_tokens=chatgpt_tokens,
        )
    except ValueError as exc:
        raise CliConfigError(str(exc)) from exc

    # Attach a persisting token manager so mid-session refreshes are written back
    # to settings.json (only possible when a store is supplied by the caller).
    if chatgpt_tokens is not None and settings is not None and settings_store is not None:
        config.attach_chatgpt_token_manager(_build_chatgpt_token_manager(chatgpt_tokens, settings, settings_store))
    return config


def key_status(provider: str, project_path: Path, settings: Optional[CliSettings] = None) -> str:
    """Return the API-key (or sign-in) status for display without exposing secrets."""
    provider_value = _provider(provider)
    if provider_value in OAUTH_PROVIDERS:
        if settings and settings.has_oauth_token(provider_value.value):
            token = settings.get_oauth_token(provider_value.value) or {}
            email = token.get("email") or "ChatGPT account"
            plan = token.get("plan_type") or "subscription"
            return f"signed in as {email} ({plan})"
        return "not signed in"
    env = load_cli_env(project_path)
    env_name = API_KEY_ENV.get(provider_value)
    if env_name and env.get(env_name):
        return f"present via {env_name}"
    if settings and settings.get_api_key(provider_value.value):
        return "present in local settings"
    return "missing"


def config_summary(config: AgentConfig) -> dict[str, object]:
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
        "agent_models": {
            role: f"{model_config.provider.value}/{model_config.model}"
            for role, model_config in config.agent_models.items()
        },
    }
