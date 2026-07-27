"""Model discovery and atomic per-dispatch routing for sub-agents."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from kolega_code.config import AGENT_ROLE_BY_NAME, AgentConfig, AgentRole, ModelConfig, ModelProvider
from kolega_code.llm.specs import (
    MODEL_SPECS,
    default_thinking_effort,
    get_model_specs,
    supports_vision,
    thinking_effort_options,
    validate_thinking_effort,
)


@dataclass(frozen=True)
class AtomicModelOverride:
    """A complete provider/model/effort selection for one dispatched worker."""

    provider: str
    model: str
    effort: Optional[str]

    def as_dict(self, *, effort_key: str = "effort") -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model, effort_key: self.effort}


@dataclass(frozen=True)
class ModelRoutingResolution:
    """Resolved child config and sanitized identity for one dispatch."""

    config: AgentConfig
    model_config: ModelConfig
    requested: Optional[AtomicModelOverride]

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "provider": self.model_config.provider.value,
            "model": self.model_config.model,
            "thinking_effort": self.model_config.thinking_effort,
        }


def provider_is_configured(config: AgentConfig, provider: ModelProvider) -> bool:
    """Whether this config has the credential mechanism needed by ``provider``."""

    if provider == ModelProvider.LLAMA:
        return True
    if provider == ModelProvider.OPENAI_CHATGPT:
        return config.openai_chatgpt_tokens is not None
    return bool(config.get_api_key(provider))


# The two OpenAI backends serve the same models, so a caller that names one when only
# the other is configured is one word away from a valid route. Preference between them
# is advice, never enforcement: a subscription can run out of quota, and the user may
# have their own reason to direct a worker at a particular backend.
_SIBLING_PROVIDERS: dict[ModelProvider, ModelProvider] = {
    ModelProvider.OPENAI: ModelProvider.OPENAI_CHATGPT,
    ModelProvider.OPENAI_CHATGPT: ModelProvider.OPENAI,
}
_PREFERRED_SIBLING = ModelProvider.OPENAI_CHATGPT

SIBLING_PREFERENCE_ADVICE = (
    f"`{ModelProvider.OPENAI.value}` and `{ModelProvider.OPENAI_CHATGPT.value}` serve the same models. "
    f"Prefer `{_PREFERRED_SIBLING.value}` when configured."
)


def configured_provider_names(config: AgentConfig) -> list[str]:
    """Configured providers that actually have dispatchable models, in catalog order.

    Mirrors what ``subagent_model_catalog`` reports so an error message and the
    discovery tool can never disagree about what is available.
    """
    names: list[str] = []
    for provider_value, _model_name in MODEL_SPECS:
        if provider_value in names:
            continue
        try:
            provider = ModelProvider(provider_value)
        except ValueError:
            continue
        if provider_is_configured(config, provider):
            names.append(provider_value)
    return names


def _configured_providers_sentence(config: AgentConfig) -> str:
    names = configured_provider_names(config)
    if not names:
        return "No providers are configured for sub-agent model overrides."
    return f"Configured providers: {', '.join(names)}."


def _sibling_sentence(config: AgentConfig, provider: Optional[ModelProvider]) -> str:
    """Advice naming the configured sibling backend, when one is relevant."""
    sibling = _SIBLING_PROVIDERS.get(provider) if provider is not None else None
    if sibling is None or not provider_is_configured(config, sibling):
        return ""
    return (
        f"'{ModelProvider.OPENAI.value}' and '{ModelProvider.OPENAI_CHATGPT.value}' serve the same models; "
        f"prefer '{_PREFERRED_SIBLING.value}' when it is configured."
    )


def _omit_override_sentence(inherited: ModelConfig) -> str:
    effort = inherited.thinking_effort if inherited.thinking_effort is not None else "null"
    return (
        "Omit model_override entirely to run with the default: "
        f"{inherited.provider.value}/{inherited.model} ({effort})."
    )


def _override_error(*parts: str) -> ValueError:
    return ValueError(" ".join(part for part in parts if part))


def parse_atomic_model_override(
    value: Any,
    *,
    effort_key: str,
) -> Optional[AtomicModelOverride]:
    """Validate the shape of an optional all-or-nothing model override."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("model_override must be an object or null.")

    expected = {"provider", "model", effort_key}
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise ValueError(f"model_override is missing required field(s): {', '.join(missing)}.")
    if extra:
        raise ValueError(f"model_override contains unsupported field(s): {', '.join(extra)}.")

    provider_value = value["provider"]
    model = value["model"]
    if not isinstance(provider_value, str) or not provider_value.strip():
        raise ValueError("model_override.provider must be a non-empty string.")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model_override.model must be a non-empty string.")

    effort = value[effort_key]
    if effort is not None and (not isinstance(effort, str) or not effort.strip()):
        raise ValueError(f"model_override.{effort_key} must be a non-empty string or null.")

    return AtomicModelOverride(
        provider=provider_value.strip().lower(),
        model=model.strip(),
        effort=effort.strip().lower() if isinstance(effort, str) else None,
    )


def _validated_override_model(
    config: AgentConfig, override: AtomicModelOverride, inherited: ModelConfig
) -> ModelConfig:
    # Every failure below names what *is* available and how to proceed without an
    # override. Listing the supported enum instead of the configured providers sent
    # callers straight into another unusable guess.
    try:
        provider = ModelProvider(override.provider)
    except ValueError as exc:
        raise _override_error(
            f"Unsupported model_override provider '{override.provider}'.",
            _configured_providers_sentence(config),
            _omit_override_sentence(inherited),
        ) from exc

    try:
        get_model_specs(provider, override.model)
    except ValueError as exc:
        raise _override_error(
            str(exc),
            _sibling_sentence(config, provider),
            _omit_override_sentence(inherited),
        ) from exc

    if not provider_is_configured(config, provider):
        raise _override_error(
            f"Provider '{provider.value}' is not configured.",
            _configured_providers_sentence(config),
            _sibling_sentence(config, provider),
            _omit_override_sentence(inherited),
        )

    efforts = thinking_effort_options(provider, override.model)
    if efforts:
        if override.effort is None:
            raise ValueError(
                f"model_override effort must be a string for {provider.value}/{override.model}. "
                f"Valid values: {', '.join(efforts)}."
            )
        effort = validate_thinking_effort(provider, override.model, override.effort)
    else:
        if override.effort is not None:
            raise ValueError(
                f"Model {override.model} from provider {provider.value} does not support thinking effort; "
                "set the override effort to null."
            )
        effort = None

    return ModelConfig(
        provider=provider,
        model=override.model,
        rate_limits=inherited.rate_limits.model_copy(deep=True),
        thinking_effort=effort,
    )


def _config_for_agent_model(config: AgentConfig, agent_name: str, model_config: ModelConfig) -> AgentConfig:
    role = AGENT_ROLE_BY_NAME.get(agent_name)
    if role is None:
        return config
    agent_models = dict(config.agent_models)
    agent_models[role.value] = model_config
    return config.model_copy(update={"agent_models": agent_models})


def resolve_subagent_model(
    config: AgentConfig,
    agent_name: str,
    model_override: Any,
    *,
    effort_key: str,
    inherited_model: Optional[ModelConfig] = None,
) -> ModelRoutingResolution:
    """Resolve an optional atomic override against one worker's inherited route."""

    inherited = inherited_model or config.model_config_for_agent(agent_name)
    parsed = parse_atomic_model_override(model_override, effort_key=effort_key)
    if parsed is None:
        return ModelRoutingResolution(config=config, model_config=inherited, requested=None)

    selected = _validated_override_model(config, parsed, inherited)
    return ModelRoutingResolution(
        config=_config_for_agent_model(config, agent_name, selected),
        model_config=selected,
        requested=parsed,
    )


def model_routing_fingerprint(config: AgentConfig) -> str:
    """Return a secret-free fingerprint of inherited workflow agent routes."""

    routes: dict[str, dict[str, Any]] = {}
    for role in AgentRole:
        agent_name = next(
            (name for name, candidate in AGENT_ROLE_BY_NAME.items() if candidate == role),
            role.value,
        )
        model = config.model_config_for_agent(agent_name)
        routes[role.value] = {
            "provider": model.provider.value,
            "model": model.model,
            "effort": model.thinking_effort,
        }
    encoded = json.dumps(routes, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def subagent_model_catalog(config: AgentConfig, provider: Optional[str] = None) -> dict[str, Any]:
    """Build a bounded, credential-free model discovery snapshot."""

    provider_filter: Optional[ModelProvider] = None
    if provider is not None:
        if not isinstance(provider, str):
            raise ValueError("provider must be a string when supplied.")
        # Some model providers serialize an omitted optional string as ``""``.
        # Treat blank input exactly like omission so an unfiltered discovery
        # call remains portable across tool-calling APIs.
        if not provider.strip():
            provider = None
    if provider is not None:
        try:
            provider_filter = ModelProvider(provider.strip().lower())
        except ValueError as exc:
            raise ValueError(f"Unsupported provider '{provider}'.") from exc
        if not provider_is_configured(config, provider_filter):
            raise ValueError(f"Provider '{provider_filter.value}' is not configured.")

    defaults: dict[str, dict[str, Any]] = {}
    for agent_name, role in AGENT_ROLE_BY_NAME.items():
        model = config.model_config_for_agent(agent_name)
        defaults[role.value] = {
            "provider": model.provider.value,
            "model": model.model,
            "thinking_effort": model.thinking_effort,
        }

    models_by_provider: dict[str, list[dict[str, Any]]] = {}
    for provider_value, model_name in MODEL_SPECS:
        try:
            model_provider = ModelProvider(provider_value)
        except ValueError:
            continue
        if provider_filter is not None and model_provider != provider_filter:
            continue
        if not provider_is_configured(config, model_provider):
            continue
        efforts = thinking_effort_options(model_provider, model_name)
        models_by_provider.setdefault(provider_value, []).append(
            {
                "model": model_name,
                "thinking_efforts": list(efforts),
                "default_thinking_effort": default_thinking_effort(model_provider, model_name),
                "override_effort": "string" if efforts else "null",
                "supports_vision": supports_vision(model_provider, model_name),
            }
        )

    if provider_filter is not None and not models_by_provider:
        raise ValueError(f"Provider '{provider_filter.value}' has no supported sub-agent models.")

    return {
        "override_contract": {
            "ordinary_dispatch": {
                "required": ["provider", "model", "thinking_effort"],
                "effort_field": "thinking_effort",
            },
            "gigacode": {
                "required": ["provider", "model", "effort"],
                "effort_field": "effort",
            },
            "nullable_effort": "Use null only when thinking_efforts is empty.",
        },
        "agent_defaults": defaults,
        "providers": [
            {"provider": provider_value, "models": models} for provider_value, models in models_by_provider.items()
        ],
    }


def render_subagent_model_catalog(catalog: dict[str, Any]) -> str:
    """Render discovery data compactly for an LLM and the visible transcript."""

    configured = {entry["provider"] for entry in catalog["providers"]}
    lines = [
        "# Available sub-agent models",
        "",
        "Use a complete atomic override:",
        "- Ordinary dispatch: `provider`, `model`, `thinking_effort`",
        "- Gigacode: `provider`, `model`, `effort`",
        "- Effort must be an exact listed value; use `null` only where efforts are `null`.",
    ]
    if {ModelProvider.OPENAI.value, ModelProvider.OPENAI_CHATGPT.value} <= configured:
        lines.append(f"- {SIBLING_PREFERENCE_ADVICE}")
    lines.extend(
        [
            "",
            "## Effective role defaults",
            "",
            "| Role | Provider/model | Effort |",
            "| --- | --- | --- |",
        ]
    )
    for role, route in catalog["agent_defaults"].items():
        effort = route["thinking_effort"] if route["thinking_effort"] is not None else "null"
        lines.append(f"| {role} | `{route['provider']}/{route['model']}` | `{effort}` |")

    lines.extend(
        [
            "",
            "## Configured models",
            "",
            "| Provider/model | Efforts | Default | Vision |",
            "| --- | --- | --- | --- |",
        ]
    )
    for provider_entry in catalog["providers"]:
        provider = provider_entry["provider"]
        for model in provider_entry["models"]:
            efforts = ", ".join(model["thinking_efforts"]) or "null"
            default = model["default_thinking_effort"] or "null"
            vision = "yes" if model["supports_vision"] else "no"
            lines.append(f"| `{provider}/{model['model']}` | `{efforts}` | `{default}` | {vision} |")

    return "\n".join(lines)
