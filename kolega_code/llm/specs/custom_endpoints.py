"""Custom endpoint provider identity and runtime model-spec registration."""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

from .catalog import MODEL_SPECS, WILDCARD_MODEL_SPECS
from .thinking import REASONING_REPLAY_VALUES
from .types import ThinkingEffortSpec
from .validation import validate_model_spec

CUSTOM_PROVIDER_PREFIX = "custom:"
DEFAULT_CONTEXT_LENGTH = 32768
DEFAULT_MAX_OUTPUT_TOKENS = 8192
DEFAULT_TEMPERATURE = 1.0
API_STYLES = ("openai_chat", "openai_responses", "anthropic")

_CUSTOM_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]*\Z")

# Wire-safe thinking modes custom endpoints may declare.
CUSTOM_THINKING_PRESETS: dict[str, dict[str, Any]] = {
    "openai_reasoning_effort": {
        "options": ("none", "low", "medium", "high", "xhigh", "max"),
        "default": "high",
        "budgets_required": False,
    },
    "openai_responses_reasoning": {
        "options": ("none", "low", "medium", "high", "xhigh"),
        "default": "medium",
        "budgets_required": False,
    },
    "thinking_toggle": {
        "options": ("none", "enabled"),
        "default": "enabled",
        "budgets_required": False,
    },
    "anthropic_budget": {
        "options": ("none", "low", "medium", "high", "xhigh", "max"),
        "default": "medium",
        "budgets_required": True,
    },
}

CUSTOM_THINKING_MODES = frozenset(CUSTOM_THINKING_PRESETS)


def is_custom_provider(provider: Any) -> bool:
    value = provider.value if hasattr(provider, "value") else provider
    return isinstance(value, str) and value.startswith(CUSTOM_PROVIDER_PREFIX)


def custom_endpoint_id(provider: Any) -> Optional[str]:
    value = provider.value if hasattr(provider, "value") else provider
    if not isinstance(value, str) or not value.startswith(CUSTOM_PROVIDER_PREFIX):
        return None
    return value[len(CUSTOM_PROVIDER_PREFIX) :]


def valid_custom_endpoint_id(endpoint_id: str) -> bool:
    return bool(_CUSTOM_ID_RE.match(endpoint_id))


def _positive_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return default
    return value


def _thinking_spec(raw: Any) -> Optional[ThinkingEffortSpec]:
    if not isinstance(raw, dict):
        return None
    mode = raw.get("mode")
    if mode not in CUSTOM_THINKING_MODES:
        return None
    preset = CUSTOM_THINKING_PRESETS[mode]
    if isinstance(raw.get("options"), (list, tuple)):
        options = tuple(str(option) for option in raw["options"])
    else:
        options = preset["options"]
    if not options:
        return None
    default = str(raw.get("default") or preset["default"])
    if default not in options:
        default = options[0]
    budgets: dict[str, int] = {}
    if preset["budgets_required"]:
        raw_budgets = raw.get("budgets")
        if not isinstance(raw_budgets, dict):
            return None
        for option in options:
            if option == "none":
                continue
            budget = _positive_int(raw_budgets.get(option))
            if budget is None:
                return None
            budgets[option] = budget
    return ThinkingEffortSpec(options=options, default=default, mode=mode, budgets=budgets)


def _valid_temperature(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and 0 < float(value) <= 2


def _endpoint_spec(entry: Mapping[str, Any], model_override: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    merged: Mapping[str, Any] = {**entry, **model_override} if model_override else entry
    temperature = merged.get("temperature")
    if not _valid_temperature(temperature):
        temperature = DEFAULT_TEMPERATURE
    assert isinstance(temperature, (int, float))
    spec: dict[str, Any] = {
        "context_length": _positive_int(merged.get("context_length"), DEFAULT_CONTEXT_LENGTH),
        "max_completion_tokens": _positive_int(merged.get("max_output_tokens"), DEFAULT_MAX_OUTPUT_TOKENS),
        "input_budget": "window_minus_output",
        "supports_vision": bool(merged.get("supports_vision", False)),
        "supports_temperature": True,
        "default_temperature": float(temperature),
    }
    thinking = _thinking_spec(merged.get("thinking"))
    if thinking is not None:
        spec["thinking_effort"] = thinking
    if merged.get("api_style") == "openai_chat":
        replay = merged.get("reasoning_replay", "auto")
        spec["reasoning_replay"] = str(replay) if replay in REASONING_REPLAY_VALUES else "auto"
    return spec


def sync_custom_endpoint_specs(endpoints: Mapping[str, Any]) -> None:
    """Rebuild MODEL_SPECS/WILDCARD_MODEL_SPECS entries for custom endpoints. Idempotent."""
    for registry in (MODEL_SPECS, WILDCARD_MODEL_SPECS):
        for key in [key for key in registry if key[0].startswith(CUSTOM_PROVIDER_PREFIX)]:
            del registry[key]
    for endpoint_id, entry in endpoints.items():
        if not isinstance(endpoint_id, str) or not valid_custom_endpoint_id(endpoint_id):
            continue
        if not isinstance(entry, Mapping):
            continue
        provider = f"{CUSTOM_PROVIDER_PREFIX}{endpoint_id}"
        base = _endpoint_spec(entry)
        try:
            validate_model_spec(base)
        except ValueError:
            continue
        WILDCARD_MODEL_SPECS[(provider, "*")] = base
        models = entry.get("models")
        if not isinstance(models, dict):
            continue
        for model_name, override in models.items():
            if not isinstance(model_name, str) or not model_name or not isinstance(override, dict):
                continue
            spec = _endpoint_spec(entry, override)
            try:
                validate_model_spec(spec)
            except ValueError:
                continue
            MODEL_SPECS[(provider, model_name)] = spec
