"""Configuration helpers for the Kolega Code CLI."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, NoReturn, Optional

from pydantic import ValidationError

from kolega_code.auth.tokens import ChatGPTTokenManager, OAuthTokens
from kolega_code.config import (
    AgentConfig,
    AgentRole,
    CustomEndpointConfig,
    EditProtocol,
    ModelConfig,
    ModelProvider,
    RateLimitConfig,
)
from kolega_code.llm.specs import MODEL_SPECS, get_model_specs, model_is_known, normalize_thinking_effort
from kolega_code.llm.specs.custom_endpoints import (
    API_STYLES,
    CUSTOM_PROVIDER_PREFIX,
    CUSTOM_THINKING_PRESETS,
    REASONING_REPLAY_VALUES,
    custom_endpoint_id,
    is_custom_provider,
    sync_custom_endpoint_specs,
)
from kolega_code.mcp.config import load_mcp_config
from kolega_code.services.lsp.config import LspConfig

from .provider_registry import default_model_for_provider, rebuild_ui_model_options, set_custom_provider_labels
from .settings import (
    COMPRESSION_THRESHOLD_MAX_PERCENT,
    COMPRESSION_THRESHOLD_MIN_PERCENT,
    CliSettings,
    SettingsStore,
)

# Providers authenticated by an OAuth sign-in instead of a static API key.
OAUTH_PROVIDERS = frozenset({ModelProvider.OPENAI_CHATGPT})

# No built-in provider or model defaults by design. A model must be named together with
# its provider (see _require_provider_for_model), and an unset operational slot inherits
# the active model rather than pinning itself to any particular model.
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
    ModelProvider.OPENROUTER: "OPENROUTER_API_KEY",
    ModelProvider.THINKING_MACHINES: "TINKER_API_KEY",
    # Native Tinker sampling authenticates with the same key.
    ModelProvider.TINKER: "TINKER_API_KEY",
}

DEFAULT_WEB_SEARCH_BACKEND = "duckduckgo"
WEB_SEARCH_BACKEND_ENV = "KOLEGA_CODE_WEB_SEARCH_BACKEND"
# Web tool mode: auto (hosted server-side search when the model supports it,
# else the client tools), hosted, client, or off (no web tools at all).
WEB_SEARCH_MODES = ("auto", "hosted", "client", "off")
DEFAULT_WEB_SEARCH_MODE = "auto"
WEB_SEARCH_MODE_ENV = "KOLEGA_CODE_WEB_SEARCH_MODE"
SEARXNG_BASE_URL_ENV = "SEARXNG_BASE_URL"
# LSP master switch: on or off. Absent means defer to settings.lsp_enabled.
LSP_MODES = ("on", "off")
LSP_MODE_ENV = "KOLEGA_CODE_LSP"
# Sub-agent dispatch (dispatch_agent) master switch: on or off. Absent means
# defer to settings.subagents_enabled, then the default (enabled).
SUBAGENT_MODES = ("on", "off")
SUBAGENTS_MODE_ENV = "KOLEGA_CODE_SUBAGENTS"
# Agent Skills master switch: on or off. Absent means defer to
# settings.skills_enabled, then the default (enabled).
SKILL_MODES = ("on", "off")
SKILLS_MODE_ENV = "KOLEGA_CODE_SKILLS"
# Compression threshold override, in percent (10-100). Absent means defer to
# settings.compression_threshold, then the agent's built-in default (95%).
COMPRESSION_THRESHOLD_ENV = "KOLEGA_CODE_COMPRESSION_THRESHOLD"
# Env vars that supply a cloud web-search backend's key, keyed by backend name.
SEARCH_BACKEND_KEY_ENV = {
    "firecrawl": "FIRECRAWL_API_KEY",
    "tavily": "TAVILY_API_KEY",
}

# Custom-endpoint definition layers: full JSON object, or flat fields describing
# one ephemeral endpoint under the id below. Never persisted to settings.json.
CUSTOM_ENDPOINTS_ENV = "KOLEGA_CODE_CUSTOM_ENDPOINTS"
ENDPOINT_URL_ENV = "KOLEGA_CODE_ENDPOINT_URL"
ENDPOINT_STYLE_ENV = "KOLEGA_CODE_ENDPOINT_STYLE"
ENDPOINT_API_KEY_ENV = "KOLEGA_CODE_ENDPOINT_API_KEY"
ENDPOINT_CONTEXT_ENV = "KOLEGA_CODE_ENDPOINT_CONTEXT"
ENDPOINT_MAX_OUTPUT_ENV = "KOLEGA_CODE_ENDPOINT_MAX_OUTPUT"
ENDPOINT_VISION_ENV = "KOLEGA_CODE_ENDPOINT_VISION"
ENDPOINT_THINKING_ENV = "KOLEGA_CODE_ENDPOINT_THINKING"
ENDPOINT_REASONING_ENV = "KOLEGA_CODE_ENDPOINT_REASONING"
ENDPOINT_TEMPERATURE_ENV = "KOLEGA_CODE_ENDPOINT_TEMPERATURE"
CLI_ENDPOINT_ID = "cli"
CLI_ENDPOINT_PROVIDER = f"{CUSTOM_PROVIDER_PREFIX}{CLI_ENDPOINT_ID}"


class CliConfigError(ValueError):
    """Raised when CLI configuration is incomplete or invalid."""


class StrictContextBudgetError(CliConfigError):
    """Raised when the paired strict context-budget flags are invalid."""


@dataclass(frozen=True)
class CliConfigOverrides:
    """Model and provider overrides supplied by CLI flags."""

    provider: Optional[str] = None
    model: Optional[str] = None
    fast_provider: Optional[str] = None
    fast_model: Optional[str] = None
    thinking_effort: Optional[str] = None
    environment: Optional[str] = None
    edit_protocol: Optional[str] = None
    web_search_mode: Optional[str] = None
    lsp_mode: Optional[str] = None
    # Sub-agent dispatch override from the --subagents flag ("on"/"off"); None
    # defers to KOLEGA_CODE_SUBAGENTS, then settings.subagents_enabled.
    subagents_mode: Optional[str] = None
    # Agent Skills override from the --skills flag ("on"/"off"); None defers
    # to KOLEGA_CODE_SKILLS, then settings.skills_enabled.
    skills_mode: Optional[str] = None
    # Compression threshold in percent, as typed on the command line; parsed and
    # validated in _compression_threshold.
    compression_threshold: Optional[str] = None
    # Strict per-run context budget, as typed on the command line; parsed and
    # validated as a pair in _strict_context_budget. Never persisted.
    context_window_tokens: Optional[str] = None
    max_output_tokens: Optional[str] = None
    # Custom endpoint definition layers. --custom-endpoints carries the full JSON
    # object; the --endpoint-* flags define one ephemeral endpoint "custom:cli".
    # Neither is ever persisted.
    custom_endpoints_json: Optional[str] = None
    endpoint_url: Optional[str] = None
    endpoint_style: Optional[str] = None
    endpoint_api_key: Optional[str] = None
    endpoint_context: Optional[str] = None
    endpoint_max_output: Optional[str] = None
    endpoint_vision: bool = False
    endpoint_thinking: Optional[str] = None
    endpoint_reasoning: Optional[str] = None
    endpoint_temperature: Optional[str] = None


def load_cli_env(project_path: Path, env: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """Load Kolega Code's explicit process environment.

    ``project_path`` is accepted for backwards-compatible call sites, but project-local
    ``.env`` files are intentionally ignored: they belong to the project being edited,
    not to Kolega Code's own provider/model configuration.
    """
    _ = project_path
    return dict(env if env is not None else os.environ)


def _parse_endpoints_json(raw: str, source: str) -> dict[str, dict]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CliConfigError(f"{source} must be valid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict) or not all(
        isinstance(endpoint_id, str) and isinstance(entry, dict) for endpoint_id, entry in parsed.items()
    ):
        raise CliConfigError(f"{source} must be a JSON object of endpoint definitions.")
    return dict(parsed)


def _flat_endpoint_fields(env: Mapping[str, str], overrides: CliConfigOverrides) -> dict[str, str]:
    fields: dict[str, str] = {}
    for name, env_key, value in (
        ("api_style", ENDPOINT_STYLE_ENV, overrides.endpoint_style),
        ("api_key", ENDPOINT_API_KEY_ENV, overrides.endpoint_api_key),
        ("context_length", ENDPOINT_CONTEXT_ENV, overrides.endpoint_context),
        ("max_output_tokens", ENDPOINT_MAX_OUTPUT_ENV, overrides.endpoint_max_output),
        ("thinking", ENDPOINT_THINKING_ENV, overrides.endpoint_thinking),
        ("reasoning_replay", ENDPOINT_REASONING_ENV, overrides.endpoint_reasoning),
        ("temperature", ENDPOINT_TEMPERATURE_ENV, overrides.endpoint_temperature),
    ):
        resolved = value or env.get(env_key)
        if resolved:
            fields[name] = resolved
    return fields


def _cli_endpoint(url: str, env: Mapping[str, str], overrides: CliConfigOverrides) -> dict[str, Any]:
    """Build the ephemeral `custom:cli` endpoint from --endpoint-url and friends."""
    if not url.strip():
        raise CliConfigError("--endpoint-url/KOLEGA_CODE_ENDPOINT_URL must not be empty.")
    if not url.startswith(("http://", "https://")):
        raise CliConfigError(f"--endpoint-url must be an http(s) URL, got '{url}'.")
    flat = _flat_endpoint_fields(env, overrides)
    api_style = flat.get("api_style", "openai_chat")
    if api_style not in API_STYLES:
        raise CliConfigError(f"Unsupported endpoint style '{api_style}'. Valid styles: {', '.join(API_STYLES)}")
    entry: dict[str, Any] = {"api_style": api_style, "base_url": url.rstrip("/")}
    if flat.get("api_key"):
        entry["api_key"] = flat["api_key"]
    for key, flag in (("context_length", "context"), ("max_output_tokens", "max-output")):
        raw = flat.get(key)
        if raw is None:
            continue
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value <= 0:
            raise CliConfigError(f"--endpoint-{flag} must be a positive integer, got '{raw}'.")
        entry[key] = value
    raw_temperature = flat.get("temperature")
    if raw_temperature is not None:
        try:
            temperature = float(raw_temperature)
        except ValueError:
            temperature = 0.0
        if not (0 < temperature <= 2):
            raise CliConfigError(f"--endpoint-temperature must be between 0 and 2, got '{raw_temperature}'.")
        entry["temperature"] = temperature
    thinking = flat.get("thinking")
    if thinking:
        if thinking not in CUSTOM_THINKING_PRESETS:
            valid = ", ".join(sorted(CUSTOM_THINKING_PRESETS))
            raise CliConfigError(f"Unsupported --endpoint-thinking '{thinking}'. Valid modes: {valid}")
        preset = CUSTOM_THINKING_PRESETS[thinking]
        if preset["budgets_required"]:
            raise CliConfigError(
                f"--endpoint-thinking {thinking} needs per-effort budgets; define it with --custom-endpoints JSON."
            )
        entry["thinking"] = {"mode": thinking, "options": list(preset["options"]), "default": preset["default"]}
    reasoning = flat.get("reasoning_replay")
    if reasoning:
        if reasoning not in REASONING_REPLAY_VALUES:
            valid = ", ".join(REASONING_REPLAY_VALUES)
            raise CliConfigError(f"Unsupported --endpoint-reasoning '{reasoning}'. Valid values: {valid}")
        entry["reasoning_replay"] = reasoning
    if overrides.endpoint_vision or env.get(ENDPOINT_VISION_ENV):
        entry["supports_vision"] = True
    return entry


def _merged_custom_endpoints(
    env: Mapping[str, str],
    overrides: CliConfigOverrides,
    settings: Optional[CliSettings],
) -> dict[str, dict]:
    """Custom endpoints from settings.json overlaid with env JSON then flag JSON (per id)."""
    merged: dict[str, dict] = dict(settings.custom_endpoints) if settings else {}
    env_json = env.get(CUSTOM_ENDPOINTS_ENV)
    if env_json:
        merged.update(_parse_endpoints_json(env_json, CUSTOM_ENDPOINTS_ENV))
    if overrides.custom_endpoints_json:
        merged.update(_parse_endpoints_json(overrides.custom_endpoints_json, "--custom-endpoints"))
    endpoint_url = overrides.endpoint_url or env.get(ENDPOINT_URL_ENV)
    if endpoint_url:
        merged[CLI_ENDPOINT_ID] = _cli_endpoint(endpoint_url, env, overrides)
    return merged


def _provider(value: str) -> ModelProvider:
    try:
        return ModelProvider(value.lower())
    except ValueError as exc:
        valid = ", ".join(provider.value for provider in ModelProvider)
        raise CliConfigError(f"Unsupported provider '{value}'. Valid providers: {valid}") from exc


def _explicit_value(
    flag_value: Optional[str],
    env: Mapping[str, str],
    env_key: str,
    flag_name: str,
) -> tuple[Optional[str], Optional[str]]:
    """Return an explicit CLI/process-env value plus a human-readable source."""
    if flag_value:
        return flag_value, flag_name
    env_value = env.get(env_key)
    if env_value:
        return env_value, env_key
    return None, None


def _model_sources(model: str) -> list[str]:
    return sorted(provider for provider, candidate in MODEL_SPECS if candidate == model)


def _require_provider_for_model(model: str, model_source: Optional[str], provider_names: str) -> NoReturn:
    """Reject a model named without a provider.

    Nothing is guessed here. The same model id can be served by several providers on
    different credentials (e.g. an OpenAI id reachable both by API key and by ChatGPT
    subscription), so picking one for the user would silently choose an account.
    """
    label = f"{model_source}={model}" if model_source else f"Model '{model}'"
    sources = _model_sources(model)
    hint = (
        f" It is offered by: {', '.join(sources)}."
        if sources
        else f" Valid providers: {', '.join(provider.value for provider in ModelProvider)}."
    )
    raise CliConfigError(f"{label} also needs a provider; set {provider_names}.{hint}")


def _require_model_for_provider(provider: str, provider_source: Optional[str], model_names: str) -> NoReturn:
    """Reject a provider named without a model.

    Provider and model are always specified together: there is no house default model
    to fall back on, and picking one from the catalog would pin a choice the user did
    not make.
    """
    label = f"{provider_source}={provider}" if provider_source else f"Provider '{provider}'"
    raise CliConfigError(f"{label} also needs a model; set {model_names}.")


def _ensure_explicit_model_supported(
    provider: ModelProvider,
    model: str,
    *,
    model_source: Optional[str],
    provider_source: Optional[str],
) -> None:
    """Raise a targeted error when an explicit model is paired with the wrong provider."""
    if is_custom_provider(provider) and not model_is_known(provider, model):
        # A custom provider resolves any model only through its endpoint's
        # wildcard; its absence here means the endpoint is not defined.
        raise CliConfigError(
            f"Custom endpoint '{custom_endpoint_id(provider)}' is not defined. Add it under custom_endpoints in "
            "settings.json, or via --custom-endpoints/--endpoint-url for this session."
        )
    if model_is_known(provider, model):
        return

    model_label = f"{model_source}={model}" if model_source else f"Model '{model}'"
    provider_label = f"{provider_source}={provider.value}" if provider_source else f"provider {provider.value}"
    if provider == ModelProvider.TINKER:
        # Checkpoint paths (tinker://...) always resolve via the wildcard; an
        # unknown base model id needs a catalog refresh.
        raise CliConfigError(
            f"{model_label} is not a known Tinker base model. "
            "Run `kolega-code models refresh --provider tinker` to pick up newer bases, "
            "or use a saved checkpoint path (tinker://<run-id>/sampler_weights/<step>)."
        )
    sources = _model_sources(model)
    if sources:
        preferred = sources[0]
        if provider_source:
            suggestion = f"set {provider_source}={preferred} or remove {model_source or 'the model override'}"
        else:
            suggestion = f"set the provider to {preferred} or remove {model_source or 'the model override'}"
    else:
        suggestion = f"choose a supported model for {provider.value} or remove {model_source or 'the model override'}"
    raise CliConfigError(f"{model_label} is not available for {provider_label}; {suggestion}.")


def _api_key_for_provider(
    provider: ModelProvider,
    env: Mapping[str, str],
    settings: Optional[CliSettings],
    custom_endpoints: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    endpoint_id = custom_endpoint_id(provider)
    if endpoint_id is not None:
        merged = custom_endpoints or _merged_custom_endpoints(env, CliConfigOverrides(), settings)
        entry = merged.get(endpoint_id) or {}
        return entry.get("api_key") or None
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


def _web_search_mode(
    env: Mapping[str, str],
    settings: Optional[CliSettings],
    override: Optional[str],
) -> str:
    """Resolve the web tool mode with flag-over-env-over-settings precedence."""
    mode = (
        override
        or env.get(WEB_SEARCH_MODE_ENV)
        or (settings.web_search_mode if settings else None)
        or DEFAULT_WEB_SEARCH_MODE
    ).lower()
    if mode not in WEB_SEARCH_MODES:
        valid = ", ".join(WEB_SEARCH_MODES)
        raise CliConfigError(f"Unsupported web search mode '{mode}'. Valid modes: {valid}")
    return mode


def _lsp_enabled(
    env: Mapping[str, str],
    settings: Optional[CliSettings],
    override: Optional[str],
) -> bool:
    """Resolve the LSP master switch with flag-over-env-over-settings precedence."""
    mode = (override or env.get(LSP_MODE_ENV) or "").strip().lower()
    if mode:
        if mode not in LSP_MODES:
            valid = ", ".join(LSP_MODES)
            raise CliConfigError(f"Unsupported LSP mode '{mode}'. Valid modes: {valid}")
        return mode == "on"
    if settings is not None and settings.lsp_enabled is not None:
        return bool(settings.lsp_enabled)
    return True


def _subagents_enabled(
    env: Mapping[str, str],
    settings: Optional[CliSettings],
    override: Optional[str],
) -> bool:
    """Resolve the sub-agent dispatch master switch with flag-over-env-over-settings precedence."""
    mode = (override or env.get(SUBAGENTS_MODE_ENV) or "").strip().lower()
    if mode:
        if mode not in SUBAGENT_MODES:
            valid = ", ".join(SUBAGENT_MODES)
            raise CliConfigError(f"Unsupported subagents mode '{mode}'. Valid modes: {valid}")
        return mode == "on"
    if settings is not None and settings.subagents_enabled is not None:
        return bool(settings.subagents_enabled)
    return True


def _skills_enabled(
    env: Mapping[str, str],
    settings: Optional[CliSettings],
    override: Optional[str],
) -> bool:
    """Resolve the Agent Skills master switch with flag-over-env-over-settings precedence."""
    mode = (override or env.get(SKILLS_MODE_ENV) or "").strip().lower()
    if mode:
        if mode not in SKILL_MODES:
            valid = ", ".join(SKILL_MODES)
            raise CliConfigError(f"Unsupported skills mode '{mode}'. Valid modes: {valid}")
        return mode == "on"
    if settings is not None and settings.skills_enabled is not None:
        return bool(settings.skills_enabled)
    return True


def _compression_threshold(
    env: Mapping[str, str],
    settings: Optional[CliSettings],
    override: Optional[str],
) -> Optional[float]:
    """Resolve the compression threshold with flag-over-env-over-settings precedence.

    User-facing surfaces carry a percent (10-100); the returned value is the
    fraction the agent compares against, or None for the built-in default (95%).
    Explicit flag/env values are validated strictly (a typo must fail loudly);
    settings values were already range-coerced at load."""
    # Resolve presence per layer instead of via _explicit_value, whose truthiness
    # check treats an explicit empty value as absent; here it is invalid input to
    # reject, not a missing layer to fall through.
    explicit: Optional[str] = None
    source: Optional[str] = None
    if override is not None:
        explicit, source = override, "--compression-threshold"
    elif COMPRESSION_THRESHOLD_ENV in env:
        explicit, source = env[COMPRESSION_THRESHOLD_ENV], COMPRESSION_THRESHOLD_ENV
    percent: Optional[float] = None
    if explicit is not None:
        try:
            percent = float(explicit)
        except ValueError:
            percent = None
        if (
            percent is None
            or percent < COMPRESSION_THRESHOLD_MIN_PERCENT
            or percent > COMPRESSION_THRESHOLD_MAX_PERCENT
        ):
            raise CliConfigError(
                f"Unsupported compression threshold '{explicit}' from {source}. "
                f"Valid range: {COMPRESSION_THRESHOLD_MIN_PERCENT}-{COMPRESSION_THRESHOLD_MAX_PERCENT} (percent)."
            )
    elif settings is not None and settings.compression_threshold is not None:
        percent = settings.compression_threshold
    if percent is None:
        return None
    return percent / 100.0


def _strict_context_budget(
    overrides: CliConfigOverrides,
    provider: ModelProvider,
    model: str,
) -> tuple[Optional[int], Optional[int]]:
    """Parse and validate the paired strict context-budget flags.

    Process-local by design: no env or settings layer. Validated against the
    resolved active model here for an early error; build_agent_config then
    re-validates against the resolved coder route after applying saved role
    overrides. BaseAgent keeps the same check as defense in depth.
    """
    raw_window, raw_output = overrides.context_window_tokens, overrides.max_output_tokens
    if raw_window is None and raw_output is None:
        return None, None
    if raw_window is None or raw_output is None:
        raise StrictContextBudgetError("--context-window-tokens and --max-output-tokens must be supplied together.")

    def _positive_int(raw: str, flag: str) -> int:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value <= 0 or str(value) != raw.strip():
            raise StrictContextBudgetError(f"{flag} must be a positive integer, got '{raw}'.")
        return value

    window = _positive_int(raw_window, "--context-window-tokens")
    output = _positive_int(raw_output, "--max-output-tokens")
    if output >= window:
        raise StrictContextBudgetError(
            f"--max-output-tokens ({output}) must be strictly smaller than --context-window-tokens ({window})."
        )
    _validate_strict_context_budget_limits(window, output, provider, model)
    return window, output


def _validate_strict_context_budget_limits(
    window: int,
    output: int,
    provider: ModelProvider,
    model: str,
) -> None:
    """Validate a strict context-budget pair against one resolved model."""
    specs = get_model_specs(provider.value, model)
    if window > specs["context_length"]:
        raise StrictContextBudgetError(
            f"--context-window-tokens ({window}) exceeds the catalogued context length "
            f"({specs['context_length']}) for {provider.value}/{model}."
        )
    if output > specs["max_completion_tokens"]:
        raise StrictContextBudgetError(
            f"--max-output-tokens ({output}) exceeds the catalogued completion maximum "
            f"({specs['max_completion_tokens']}) for {provider.value}/{model}."
        )


def _coerce_known_model(provider: ModelProvider, model: Optional[str]) -> Optional[str]:
    """Return ``model`` if it's a known spec for ``provider``, else the default.

    Guards against a settings.json that points at a model which has since been
    renamed or removed (e.g. an old ChatGPT slug): without this, config building
    raises and the TUI disables the composer, locking the user out. Wildcard-
    addressed models (Tinker checkpoint paths, custom endpoints) count as known
    and are kept. A custom provider whose endpoint was deleted has no default:
    return None so the caller degrades to unconfigured.
    """
    if model and model_is_known(provider, model):
        return model
    if is_custom_provider(provider):
        return None
    return default_model_for_provider(provider)


def _active_provider_model(
    env: Mapping[str, str],
    overrides: CliConfigOverrides,
    settings: Optional[CliSettings],
    default_custom_provider: Optional[str] = None,
) -> tuple[Optional[ModelProvider], Optional[str]]:
    provider_value, provider_source = _explicit_value(overrides.provider, env, "KOLEGA_CODE_PROVIDER", "--provider")
    model_value, model_source = _explicit_value(overrides.model, env, "KOLEGA_CODE_MODEL", "--model")

    if provider_value or model_value:
        if not provider_value:
            assert model_value is not None
            if default_custom_provider:
                provider_value = default_custom_provider
                provider_source = None
            else:
                _require_provider_for_model(model_value, model_source, "--provider or KOLEGA_CODE_PROVIDER")
        if not model_value:
            _require_model_for_provider(provider_value, provider_source, "--model or KOLEGA_CODE_MODEL")
        provider = _provider(provider_value)
        _ensure_explicit_model_supported(
            provider,
            model_value,
            model_source=model_source,
            provider_source=provider_source,
        )
        return provider, model_value

    if settings and settings.active_provider and settings.active_model:
        try:
            provider = _provider(settings.active_provider)
        except CliConfigError:
            return None, None
        model = _coerce_known_model(provider, settings.active_model)
        if model is None:
            return None, None
        return provider, model

    return None, None


def _slot_provider_model(
    env: Mapping[str, str],
    provider_env_key: str,
    model_env_key: str,
    provider_override: Optional[str],
    model_override: Optional[str],
    active_provider: ModelProvider,
    active_model: str,
    saved: Optional[Mapping[str, str]] = None,
    default_custom_provider: Optional[str] = None,
) -> tuple[ModelProvider, str]:
    """Resolve one operational slot: CLI flag > env var > saved slot > active model.

    ``saved`` is a persisted ``{provider, model}`` override from Settings, always
    carrying both fields (``CliSettings`` drops half-written entries). Provider and
    model must be given together here too, so a flag/env that names only one of them
    is an error rather than a silent pick.

    Only explicit values are validated against the catalog: a *saved* model that has
    since left the catalog degrades to the provider's default rather than locking the
    user out of Settings.

    There is no built-in per-slot default: an unset slot inherits the active model, and
    ``build_agent_config`` raises ``MISSING_MODEL_SELECTION_MESSAGE`` before calling this
    when no active model is configured, so ``active_provider``/``active_model`` are always
    present here.
    """
    provider_flag = f"--{provider_env_key.removeprefix('KOLEGA_CODE_').lower().replace('_', '-')}"
    model_flag = f"--{model_env_key.removeprefix('KOLEGA_CODE_').lower().replace('_', '-')}"
    provider_value, provider_source = _explicit_value(provider_override, env, provider_env_key, provider_flag)
    model_value, model_source = _explicit_value(model_override, env, model_env_key, model_flag)

    if provider_value and not model_value:
        _require_model_for_provider(provider_value, provider_source, f"{model_flag} or {model_env_key}")
    if model_value and not provider_value:
        if default_custom_provider:
            provider_value = default_custom_provider
            provider_source = None
        else:
            _require_provider_for_model(model_value, model_source, f"{provider_flag} or {provider_env_key}")

    if provider_value:
        assert model_value is not None
        provider = _provider(provider_value)
        _ensure_explicit_model_supported(
            provider,
            model_value,
            model_source=model_source,
            provider_source=provider_source,
        )
        return provider, model_value

    saved = saved or {}
    saved_provider, saved_model = saved.get("provider"), saved.get("model")
    if saved_provider and saved_model:
        provider = _provider(saved_provider)
        if not model_is_known(provider, saved_model):
            if is_custom_provider(provider):
                # Endpoint deleted from settings: drop the override, inherit the active.
                return active_provider, active_model
            saved_model = default_model_for_provider(provider)
        return provider, saved_model

    return active_provider, active_model


def _model_config(
    provider: ModelProvider,
    model: str,
    thinking_effort: Optional[str] = None,
    custom_endpoints: Optional[Mapping[str, Any]] = None,
) -> ModelConfig:
    endpoint_id = custom_endpoint_id(provider)
    if endpoint_id is not None and endpoint_id not in (custom_endpoints or {}):
        raise CliConfigError(
            f"Custom endpoint '{endpoint_id}' is not defined. Add it under custom_endpoints in settings.json, "
            "or via --custom-endpoints/--endpoint-url for this session."
        )
    try:
        specs = get_model_specs(provider, model)
        resolved_thinking_effort = normalize_thinking_effort(provider, model, thinking_effort)
    except ValueError as exc:
        raise CliConfigError(str(exc)) from exc
    if endpoint_id is not None:
        thinking_spec = specs.get("thinking_effort")
        if thinking_spec is not None and thinking_spec.mode == "anthropic_budget":
            cap = specs["max_completion_tokens"]
            over = [f"{option}={budget}" for option, budget in thinking_spec.budgets.items() if budget >= cap]
            if over:
                raise CliConfigError(
                    f"Custom endpoint '{endpoint_id}': anthropic_budget budgets must stay below max_output_tokens "
                    f"({cap}); over-limit: {', '.join(over)}."
                )

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
    custom_endpoints: Optional[Mapping[str, Any]] = None,
) -> dict[str, ModelConfig]:
    """Resolve per-agent-role model overrides from env vars over saved settings.

    A role with neither an env nor a settings provider/model is omitted, so it inherits
    the active model. Provider and model are specified together, exactly as for the
    operational slots: an env var naming only one of them is an error rather than a
    silent pick. Saved entries always carry both (``CliSettings`` drops half-written
    ones), and a saved model that has left the catalog degrades to the provider default
    instead of locking the user out of Settings.
    """
    saved = settings.agent_models if settings else {}
    overrides: dict[str, ModelConfig] = {}
    for role in AgentRole:
        provider_key, model_key, effort_key = _agent_role_env_keys(role)
        entry = saved.get(role.value) or {}
        provider_env_value = env.get(provider_key)
        model_env_value = env.get(model_key)
        effort_value = env.get(effort_key) or entry.get("thinking_effort")

        if provider_env_value and not model_env_value and not entry.get("model"):
            _require_model_for_provider(provider_env_value, provider_key, model_key)
        if model_env_value and not provider_env_value and not entry.get("provider"):
            _require_provider_for_model(model_env_value, model_key, provider_key)

        provider_value = provider_env_value or entry.get("provider")
        model_value = model_env_value or entry.get("model")
        if not provider_value or not model_value:
            continue

        provider = _provider(provider_value)
        if model_env_value:
            _ensure_explicit_model_supported(
                provider,
                model_value,
                model_source=model_key,
                provider_source=provider_key if provider_env_value else None,
            )
        elif not model_is_known(provider, model_value):
            if is_custom_provider(provider):
                # Endpoint deleted from settings: drop the override, role inherits.
                continue
            model_value = default_model_for_provider(provider)
        overrides[role.value] = _model_config(
            provider, model_value, thinking_effort=effort_value, custom_endpoints=custom_endpoints
        )
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

    if not (overrides.endpoint_url or loaded_env.get(ENDPOINT_URL_ENV)) and (
        _flat_endpoint_fields(loaded_env, overrides) or overrides.endpoint_vision
    ):
        raise CliConfigError("--endpoint-* flags require --endpoint-url or KOLEGA_CODE_ENDPOINT_URL.")

    merged_endpoints = _merged_custom_endpoints(loaded_env, overrides, settings)
    sync_custom_endpoint_specs(merged_endpoints)
    default_custom_provider = CLI_ENDPOINT_PROVIDER if CLI_ENDPOINT_ID in merged_endpoints else None

    active_provider, active_model = _active_provider_model(
        loaded_env, overrides, settings, default_custom_provider=default_custom_provider
    )
    if active_provider is None or active_model is None:
        raise CliConfigError(MISSING_MODEL_SELECTION_MESSAGE)

    long_provider, long_model = _slot_provider_model(
        loaded_env,
        "KOLEGA_CODE_PROVIDER",
        "KOLEGA_CODE_MODEL",
        overrides.provider,
        overrides.model,
        active_provider,
        active_model,
        default_custom_provider=default_custom_provider,
    )
    fast_provider, fast_model = _slot_provider_model(
        loaded_env,
        "KOLEGA_CODE_FAST_PROVIDER",
        "KOLEGA_CODE_FAST_MODEL",
        overrides.fast_provider,
        overrides.fast_model,
        active_provider,
        active_model,
        settings.model_slots.get("fast") if settings else None,
    )
    active_thinking_effort = _resolve_active_thinking_effort(
        long_provider,
        long_model,
        loaded_env,
        overrides,
        settings,
    )

    agent_model_overrides = _agent_model_overrides(loaded_env, settings, custom_endpoints=merged_endpoints)
    web_search_backend, web_search_api_key, web_search_base_url = _search_config(loaded_env, settings)
    web_search_mode = _web_search_mode(loaded_env, settings, overrides.web_search_mode)
    compression_threshold = _compression_threshold(loaded_env, settings, overrides.compression_threshold)
    context_window_tokens, max_output_tokens = _strict_context_budget(overrides, long_provider, long_model)
    state_dir = settings_store.root if settings_store is not None else SettingsStore().root
    mcp_config = load_mcp_config(
        project_path,
        state_dir,
        project_trusted=bool(settings and settings.is_mcp_project_trusted(project_path)),
    )
    lsp_project_trusted = bool(settings and settings.is_lsp_project_trusted(project_path))

    required_providers = {long_provider, fast_provider}
    required_providers.update(override.provider for override in agent_model_overrides.values())
    # API-key providers: env/settings key required. OAuth, local, and custom
    # endpoint providers are exempt (endpoint keys are optional).
    missing_keys = [
        API_KEY_ENV[provider]
        for provider in sorted(required_providers, key=lambda item: item.value)
        if provider != ModelProvider.LLAMA
        and not is_custom_provider(provider)
        and provider not in OAUTH_PROVIDERS
        and not _api_key_for_provider(provider, loaded_env, settings)
    ]
    if missing_keys:
        raise CliConfigError(f"Missing required API key environment variable(s): {', '.join(missing_keys)}")

    chatgpt_tokens = _resolve_chatgpt_tokens(settings)
    if ModelProvider.OPENAI_CHATGPT in required_providers and chatgpt_tokens is None:
        raise CliConfigError("Not signed in to ChatGPT. Run /login chatgpt to sign in with your ChatGPT subscription.")

    try:
        edit_protocol_value = overrides.edit_protocol or loaded_env.get("KOLEGA_CODE_EDIT_PROTOCOL")
        try:
            edit_protocol = EditProtocol(edit_protocol_value) if edit_protocol_value else None
        except ValueError as exc:
            valid = ", ".join(protocol.value for protocol in EditProtocol)
            raise CliConfigError(
                f"Unsupported edit protocol '{edit_protocol_value}'. Valid protocols: {valid}"
            ) from exc
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
            openrouter_api_key=_api_key_for_provider(ModelProvider.OPENROUTER, loaded_env, settings),
            thinking_machines_api_key=_api_key_for_provider(ModelProvider.THINKING_MACHINES, loaded_env, settings),
            environment=overrides.environment or loaded_env.get("KOLEGA_CODE_ENVIRONMENT", "development"),
            edit_protocol=edit_protocol,
            long_context_config=_model_config(
                long_provider, long_model, thinking_effort=active_thinking_effort, custom_endpoints=merged_endpoints
            ),
            fast_config=_model_config(fast_provider, fast_model, custom_endpoints=merged_endpoints),
            agent_models=agent_model_overrides,
            custom_endpoints={
                endpoint_id: CustomEndpointConfig.model_validate(entry)
                for endpoint_id, entry in merged_endpoints.items()
            },
            web_search_mode=web_search_mode,
            web_search_backend=web_search_backend,
            web_search_api_key=web_search_api_key,
            web_search_base_url=web_search_base_url,
            openai_chatgpt_tokens=chatgpt_tokens,
            mcp_config=mcp_config,
            lsp=LspConfig(enabled=_lsp_enabled(loaded_env, settings, overrides.lsp_mode)),
            lsp_project_trusted=lsp_project_trusted,
            eval_enabled=(True if settings is None or settings.eval_enabled is None else bool(settings.eval_enabled)),
            subagents_enabled=_subagents_enabled(loaded_env, settings, overrides.subagents_mode),
            skills_enabled=_skills_enabled(loaded_env, settings, overrides.skills_mode),
            history_compression_threshold=compression_threshold,
            context_window_tokens=context_window_tokens,
            max_output_tokens=max_output_tokens,
        )
    except ValueError as exc:
        raise CliConfigError(str(exc)) from exc

    # The active model was checked while parsing the flags above, but the coder
    # route may be replaced by a saved "building" role override. Validate that
    # resolved route before any session run marker or agent construction.
    if config.context_window_tokens is not None:
        assert config.max_output_tokens is not None
        coder = config.model_config_for_agent("coder")
        _validate_strict_context_budget_limits(
            config.context_window_tokens,
            config.max_output_tokens,
            coder.provider,
            coder.model,
        )

    # Attach a persisting token manager so mid-session refreshes are written back
    # to settings.json (only possible when a store is supplied by the caller).
    if chatgpt_tokens is not None and settings is not None and settings_store is not None:
        config.attach_chatgpt_token_manager(_build_chatgpt_token_manager(chatgpt_tokens, settings, settings_store))

    # Keep the UI picker registry in sync with the endpoints resolved for this
    # process (includes env/flag-defined ones for the session).
    set_custom_provider_labels(
        {endpoint_id: endpoint.label or endpoint_id for endpoint_id, endpoint in config.custom_endpoints.items()}
    )
    rebuild_ui_model_options()
    return config


def key_status(provider: str, project_path: Path, settings: Optional[CliSettings] = None) -> str:
    """Return the API-key (or sign-in) status for display without exposing secrets."""
    provider_value = _provider(provider)
    if provider_value == ModelProvider.LLAMA:
        return "not required for the local provider"
    endpoint_id = custom_endpoint_id(provider_value)
    if endpoint_id is not None:
        env = load_cli_env(project_path)
        merged = _merged_custom_endpoints(env, CliConfigOverrides(), settings)
        entry = merged.get(endpoint_id) or {}
        if endpoint_id not in merged:
            return "endpoint not defined"
        if entry.get("api_key"):
            saved_entry = (settings.custom_endpoints if settings else {}).get(endpoint_id) or {}
            if saved_entry.get("api_key") == entry["api_key"]:
                return "present in local settings"
            return "present via environment override"
        return "key not set (optional)"
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


def resolved_api_key(provider: str, project_path: Path, settings: Optional[CliSettings] = None) -> Optional[str]:
    """The API key a provider would actually use: env var first, then stored settings.

    The value counterpart to ``key_status``, which reports the same precedence in words.
    """
    return _api_key_for_provider(_provider(provider), load_cli_env(project_path), settings)


def probe_token_manager(settings: Optional[CliSettings]) -> Optional[ChatGPTTokenManager]:
    """A ChatGPT token manager for credential probes, deliberately non-persisting.

    A probe runs against an unapplied Settings draft, so a token refresh triggered by it
    must not write back to settings.json the way the session's own manager does.
    """
    tokens = _resolve_chatgpt_tokens(settings)
    return ChatGPTTokenManager(tokens) if tokens is not None else None


def active_model_override_message(
    config: AgentConfig,
    project_path: Path,
    overrides: Optional[CliConfigOverrides] = None,
    settings: Optional[CliSettings] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Describe an explicit CLI/process-env active-model override, if one is in effect."""
    overrides = overrides or CliConfigOverrides()
    loaded_env = load_cli_env(project_path, env)
    endpoint_override = any(
        (
            overrides.custom_endpoints_json,
            overrides.endpoint_url,
            loaded_env.get(CUSTOM_ENDPOINTS_ENV),
            loaded_env.get(ENDPOINT_URL_ENV),
        )
    )
    explicit_active_override = any(
        (
            overrides.provider,
            overrides.model,
            loaded_env.get("KOLEGA_CODE_PROVIDER"),
            loaded_env.get("KOLEGA_CODE_MODEL"),
        )
    )
    if not explicit_active_override and not endpoint_override:
        return None
    if not settings or not (settings.active_provider and settings.active_model):
        if endpoint_override and is_custom_provider(config.long_context_config.provider):
            return "CLI/environment-defined custom endpoints apply to this session."
        return None

    effective_provider = config.long_context_config.provider.value
    effective_model = config.long_context_config.model
    if settings.active_provider == effective_provider and settings.active_model == effective_model:
        if endpoint_override and is_custom_provider(config.long_context_config.provider):
            return "CLI/environment-defined custom endpoints apply to this session."
        return None

    return (
        "Environment/CLI override active: "
        f"using {effective_provider}/{effective_model} instead of saved "
        f"{settings.active_provider}/{settings.active_model}."
    )


def config_summary(config: AgentConfig) -> dict[str, object]:
    """Return a session-safe summary of model configuration."""
    mcp_config = getattr(config, "mcp_config", None)
    mcp_servers = getattr(mcp_config, "servers", {}) or {}
    mcp_enabled = [server for server in mcp_servers.values() if getattr(server, "enabled", False)]
    edit_protocol, edit_protocol_source = config.resolve_edit_protocol_with_source(config.long_context_config)
    return {
        "environment": config.environment,
        "edit_protocol": edit_protocol.value,
        "edit_protocol_source": edit_protocol_source,
        "long_provider": config.long_context_config.provider.value,
        "long_model": config.long_context_config.model,
        "fast_provider": config.fast_config.provider.value,
        "fast_model": config.fast_config.model,
        "thinking_effort": config.long_context_config.thinking_effort,
        "agent_models": {
            role: f"{model_config.provider.value}/{model_config.model}"
            for role, model_config in config.agent_models.items()
        },
        "mcp_servers": len(mcp_servers),
        "mcp_enabled_servers": len(mcp_enabled),
        "mcp_project_trusted": bool(getattr(mcp_config, "project_trusted", False)),
        **({"context_budget": strict_context_budget_marker(config)} if config.context_window_tokens else {}),
    }


def strict_context_budget_marker(config: AgentConfig) -> Optional[dict[str, Any]]:
    """The strict-budget declaration for run evidence, or None when inactive.

    Built from the RESOLVED coder route (an agent-role override can supersede
    the global model selection), so the marker cannot disagree with the route
    the agent actually uses. The experiment side audits every episode's billed
    usage against these registered values.
    """
    window, output = config.context_window_tokens, config.max_output_tokens
    if window is None or output is None:
        return None
    coder = config.model_config_for_agent("coder")
    specs = get_model_specs(coder.provider.value, coder.model)
    threshold = config.history_compression_threshold
    if threshold is None:
        from kolega_code.agent.baseagent import BaseAgent

        threshold = BaseAgent.history_compression_threshold
    max_input = window - output
    return {
        "provider": coder.provider.value,
        "model": coder.model,
        "compression_threshold": threshold,
        "catalog_context_window_tokens": specs["context_length"],
        "catalog_max_output_tokens": specs["max_completion_tokens"],
        "context_window_tokens": window,
        "max_output_tokens": output,
        "max_input_tokens": max_input,
        "first_compaction_input_tokens": int(max_input * threshold) + 1,
        "source": "cli",
    }
