"""Vercel AI Gateway catalog conversion: API payloads to ``MODEL_SPECS`` entries.

Shared by ``scripts/refresh_vercel_ai_gateway_catalog.py`` (dev-time bundled
catalog generator) and ``kolega-code models refresh --provider vercel_ai_gateway``
(runtime catalog overlay).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from .pricing_feeds import feed_pricing, sanitize_spec_pricing
from .types import ThinkingEffortSpec
from .validation import validate_model_spec

PROVIDER = "vercel_ai_gateway"
BASE_URL = "https://ai-gateway.vercel.sh/v1"
MODELS_URL = f"{BASE_URL}/models"

# Gateway /chat/completions reasoning.effort enum (probed live).
CANONICAL_EFFORT_ORDER: tuple[str, ...] = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

DEFAULT_EDIT_PROTOCOL = "claude_code"
_CODEX_APPLY_PATCH_PREFIXES = ("openai/",)

# Anthropic thinking requires signed blocks that plain Chat Completions
# ``reasoning`` strings cannot round-trip.
_DROP_PRIOR_REASONING_PREFIXES = ("anthropic/",)

# Curated picker head: /v1/models is alphabetical with no public usage ranking.
# Leading entry matches PROVIDER_DEFAULT_MODEL[ModelProvider.VERCEL_AI_GATEWAY]
# so Settings provider switching lands on the same default.
FEATURED_MODEL_IDS: tuple[str, ...] = (
    "moonshotai/kimi-k2.7-code",
    "minimax/minimax-m3.1-flash",
    "openai/gpt-4.1",
    "google/gemini-2.5-flash",
    "alibaba/qwen3.7-flash",
    "anthropic/claude-opus-5",
    "anthropic/claude-sonnet-5",
    "anthropic/claude-fable-5.2",
    "anthropic/claude-haiku-4.5",
    "openai/gpt-6.2-sol",
    "openai/gpt-6.2-astra",
    "openai/gpt-5.6-sol",
    "openai/gpt-5.4-mini",
    "google/gemini-4-neon",
    "google/gemini-4-hydrogen",
    "deepseek/deepseek-v4.1-flash",
    "deepseek/deepseek-v4-pro",
    "moonshotai/kimi-k3.1-pro",
    "zai/glm-5.3",
    "spacexai/grok-4.7",
    "mistral/mistral-large-4",
)

CACHE_SCHEMA_VERSION = 1
CACHE_FILENAME = "vercel_ai_gateway_models.json"


class VercelAIGatewayCatalogError(RuntimeError):
    """Raised when a Vercel AI Gateway payload cannot be used to build a catalog."""


def _as_positive_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def model_id(entry: Mapping[str, Any]) -> str:
    return str(entry.get("id") or "")


def is_catalogable(entry: Mapping[str, Any]) -> bool:
    """Whether a ``/v1/models`` entry belongs in the Kolega Code catalog."""
    identifier = model_id(entry)
    if not identifier or identifier.endswith("-batch"):
        return False

    if entry.get("type") != "language":
        return False

    supported = entry.get("supported_parameters") or []
    if not isinstance(supported, Sequence) or isinstance(supported, (str, bytes)) or "tools" not in supported:
        return False

    modalities = entry.get("modalities") or {}
    input_modalities = modalities.get("input") if isinstance(modalities, Mapping) else None
    if not isinstance(input_modalities, Sequence) or isinstance(input_modalities, (str, bytes)):
        return False
    if "text" not in input_modalities:
        return False

    if _as_positive_int(entry.get("context_window")) is None:
        return False
    if _as_positive_int(entry.get("max_tokens")) is None:
        return False

    return True


def _thinking_effort_spec(entry: Mapping[str, Any]) -> Optional[ThinkingEffortSpec]:
    raw_options = entry.get("reasoning_options")
    if not isinstance(raw_options, Sequence) or isinstance(raw_options, (str, bytes)):
        return None

    has_toggle = False
    effort_values: set[str] = set()
    for item in raw_options:
        if not isinstance(item, Mapping):
            continue
        opt_type = str(item.get("type") or "").strip().lower()
        if opt_type == "toggle":
            has_toggle = True
        elif opt_type == "effort":
            values = item.get("values")
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                effort_values.update(str(v).strip().lower() for v in values)

    options = [effort for effort in CANONICAL_EFFORT_ORDER if effort in effort_values]
    if options:
        if has_toggle and "none" not in options:
            options.insert(0, "none")
        if "medium" in options:
            default = "medium"
        elif "low" in options:
            default = "low"
        else:
            graded = [effort for effort in options if effort != "none"]
            default = graded[-1] if graded else options[-1]
        return ThinkingEffortSpec(options=tuple(options), default=default, mode="vercel_ai_gateway_reasoning")

    if has_toggle:
        return ThinkingEffortSpec(options=("none", "enabled"), default="enabled", mode="vercel_ai_gateway_reasoning")

    return None


def _preferred_edit_protocol(identifier: str) -> str:
    if identifier.startswith(_CODEX_APPLY_PATCH_PREFIXES):
        return "codex_apply_patch"
    return DEFAULT_EDIT_PROTOCOL


def model_spec(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one ``/v1/models`` entry into a ``MODEL_SPECS`` value."""
    identifier = model_id(entry)
    context_length = _as_positive_int(entry.get("context_window"))
    max_completion_tokens = _as_positive_int(entry.get("max_tokens"))
    if context_length is None or max_completion_tokens is None:
        raise VercelAIGatewayCatalogError(f"Model {identifier!r} has no usable context/output limits.")

    modalities = entry.get("modalities") or {}
    input_modalities = modalities.get("input") if isinstance(modalities, Mapping) else ()
    supported = entry.get("supported_parameters") or ()

    spec: dict[str, Any] = {
        "context_length": context_length,
        "max_completion_tokens": max_completion_tokens,
        "input_budget": "output_shares_window" if max_completion_tokens >= context_length else "window_minus_output",
        "default_temperature": 1.0,
    }
    if entry.get("temperature") is False or "temperature" not in supported:
        spec["supports_temperature"] = False
    spec["supports_vision"] = isinstance(input_modalities, Sequence) and "image" in input_modalities
    spec["preferred_edit_protocol"] = _preferred_edit_protocol(identifier)

    thinking = _thinking_effort_spec(entry)
    if thinking is not None:
        spec["thinking_effort"] = thinking

    if identifier.startswith(_DROP_PRIOR_REASONING_PREFIXES):
        spec["drop_prior_reasoning"] = True
    spec["pricing"] = feed_pricing(PROVIDER, identifier, entry)
    return spec


def catalog_entries(
    models_payload: Any,
    *,
    featured_order: Sequence[str] = FEATURED_MODEL_IDS,
) -> list[tuple[str, dict[str, Any]]]:
    """Return ``(model_id, spec)`` pairs: curated featured models first, then alphabetical."""
    rows = models_payload.get("data") if isinstance(models_payload, Mapping) else models_payload
    if not isinstance(rows, Sequence):
        raise VercelAIGatewayCatalogError("Models payload has no 'data' list.")

    selected = [entry for entry in rows if isinstance(entry, Mapping) and is_catalogable(entry)]
    if not selected:
        raise VercelAIGatewayCatalogError("Models payload contained no tool-capable language models.")

    featured_rank = {identifier: index for index, identifier in enumerate(featured_order)}

    def sort_key(entry: Mapping[str, Any]) -> tuple[int, int, str]:
        identifier = model_id(entry)
        rank = featured_rank.get(identifier)
        if rank is not None:
            return (0, rank, identifier)
        return (1, 0, identifier)

    return [(model_id(entry), model_spec(entry)) for entry in sorted(selected, key=sort_key)]


def featured_ids(
    entries: Sequence[tuple[str, dict[str, Any]]],
    *,
    featured_order: Sequence[str] = FEATURED_MODEL_IDS,
    always_include: Iterable[str] = (),
) -> list[str]:
    """Return the catalogued ids to mark ``featured``."""
    known = {identifier for identifier, _ in entries}
    featured: list[str] = []
    seen: set[str] = set()
    for identifier in (*featured_order, *always_include):
        if identifier in known and identifier not in seen:
            featured.append(identifier)
            seen.add(identifier)
    return featured


def spec_to_jsonable(spec: Mapping[str, Any]) -> dict[str, Any]:
    payload = sanitize_spec_pricing(spec)
    thinking = payload.get("thinking_effort")
    if isinstance(thinking, ThinkingEffortSpec):
        payload["thinking_effort"] = {
            "options": list(thinking.options),
            "default": thinking.default,
            "mode": thinking.mode,
        }
    return payload


def spec_from_jsonable(payload: Mapping[str, Any]) -> dict[str, Any]:
    spec = sanitize_spec_pricing(payload)
    thinking = spec.get("thinking_effort")
    if isinstance(thinking, Mapping):
        options = thinking.get("options") or []
        spec["thinking_effort"] = ThinkingEffortSpec(
            options=tuple(str(option) for option in options),
            default=str(thinking.get("default") or ""),
            mode=str(thinking.get("mode") or "vercel_ai_gateway_reasoning"),
        )
    validate_model_spec(spec)
    return spec


_SPEC_KEY_ORDER: tuple[str, ...] = (
    "context_length",
    "max_completion_tokens",
    "input_budget",
    "default_temperature",
    "supports_temperature",
    "supports_vision",
    "preferred_edit_protocol",
    "drop_prior_reasoning",
    "featured",
    "thinking_effort",
)


def _render_value(value: Any) -> str:
    if isinstance(value, ThinkingEffortSpec):
        options = ", ".join(repr(option) for option in value.options)
        trailing = "," if len(value.options) == 1 else ""
        return (
            "ThinkingEffortSpec(\n"
            f"            options=({options}{trailing}),\n"
            f"            default={value.default!r},\n"
            f"            mode={value.mode!r},\n"
            "        )"
        )
    return repr(value)


def render_catalog_module(
    entries: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    header_lines: Sequence[str] = (),
) -> str:
    lines: list[str] = []
    for line in header_lines:
        lines.append(f"# {line}".rstrip())
    lines.append("")
    lines.append("from kolega_code.llm.specs.types import ThinkingEffortSpec")
    lines.append("")
    lines.append("VERCEL_AI_GATEWAY_SPECS = {")
    for identifier, spec in entries:
        spec = sanitize_spec_pricing(spec)
        lines.append(f"    ({PROVIDER!r}, {identifier!r}): {{")
        ordered_keys = [key for key in _SPEC_KEY_ORDER if key in spec]
        ordered_keys += sorted(key for key in spec if key not in _SPEC_KEY_ORDER)
        for key in ordered_keys:
            lines.append(f"        {key!r}: {_render_value(spec[key])},")
        lines.append("    },")
    lines.append("}")
    return "\n".join(lines) + "\n"


def fetch_models(*, timeout: float = 30.0) -> Any:
    """Fetch the keyless ``/v1/models`` payload."""
    import httpx

    response = httpx.get(MODELS_URL, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    return response.json()


def save_cache(path: Path, entries: Sequence[tuple[str, Mapping[str, Any]]], *, fetched_at: str) -> None:
    from kolega_code.local_state import write_private_text

    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "fetched_at": fetched_at,
        "provider": PROVIDER,
        "models": [{"id": identifier, "spec": spec_to_jsonable(spec)} for identifier, spec in entries],
    }
    write_private_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_cache(path: Path) -> list[tuple[str, dict[str, Any]]]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, Mapping) or payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        return []
    models = payload.get("models")
    if not isinstance(models, Sequence):
        return []

    entries: list[tuple[str, dict[str, Any]]] = []
    for item in models:
        if not isinstance(item, Mapping):
            continue
        identifier = item.get("id")
        spec = item.get("spec")
        if not identifier or not isinstance(spec, Mapping):
            continue
        try:
            entries.append((str(identifier), spec_from_jsonable(spec)))
        except (TypeError, ValueError):
            continue
    return entries
