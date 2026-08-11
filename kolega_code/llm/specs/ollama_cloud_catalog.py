"""Ollama Cloud catalog conversion and runtime-cache helpers.

Ollama exposes the cloud membership list through its OpenAI-compatible
``/v1/models`` endpoint, but that response does not contain the capabilities or
context window Kolega Code needs. Each listed ID is therefore resolved through
``/api/show`` and the two payloads are transformed together.

This module is shared by:

* ``scripts/refresh_ollama_cloud_catalog.py`` — the release-time generator for
  ``kolega_code/llm/specs/catalog/ollama_cloud.py``.
* ``kolega-code models refresh --provider ollama_cloud`` — the opt-in additive
  runtime overlay for models published after the bundled release snapshot.

The transform is deliberately strict. A transient or malformed detail response
must never produce a partial catalog that looks authoritative.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import httpx

from .types import ThinkingEffortSpec

PROVIDER = "ollama_cloud"
MODELS_URL = "https://ollama.com/v1/models"
SHOW_URL = "https://ollama.com/api/show"

CACHE_SCHEMA_VERSION = 1
CACHE_FILENAME = "ollama_cloud_models.json"

DEFAULT_TEMPERATURE = 1.0
FALLBACK_MAX_COMPLETION_TOKENS = 32768

THINKING_EFFORT_SPEC = ThinkingEffortSpec(
    options=("low", "medium", "high"),
    default="medium",
    mode="openai_reasoning_effort",
)

# Ollama's model metadata does not publish output ceilings. Preserve only the
# live, already-reviewed exceptions from the former hand-maintained catalog;
# every new or otherwise unverified model gets the conservative fallback above.
MAX_COMPLETION_TOKEN_OVERRIDES = {
    "deepseek-v4-pro": 65536,
    "glm-5.2": 65536,
    "minimax-m3": 65536,
}

_SPEC_KEY_ORDER = (
    "context_length",
    "max_completion_tokens",
    "default_temperature",
    "supports_vision",
    "thinking_effort",
)


class OllamaCloudCatalogError(RuntimeError):
    """Raised when Ollama payloads cannot produce a complete catalog."""


@dataclass(frozen=True)
class CatalogTransformResult:
    """Complete transform output plus intentionally filtered model IDs."""

    entries: list[tuple[str, dict[str, Any]]]
    filtered_ids: tuple[str, ...]


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _model_ids(models_payload: Any) -> list[str]:
    if not isinstance(models_payload, Mapping):
        raise OllamaCloudCatalogError("Expected /v1/models to return a JSON object.")
    rows = models_payload.get("data")
    if not _is_sequence(rows) or not rows:
        raise OllamaCloudCatalogError("Expected /v1/models.data to be a non-empty list.")

    identifiers: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(rows):
        if not isinstance(item, Mapping):
            raise OllamaCloudCatalogError(f"Expected /v1/models.data[{index}] to be an object.")
        raw_identifier = item.get("id")
        if not isinstance(raw_identifier, str) or not raw_identifier.strip():
            raise OllamaCloudCatalogError(f"Expected /v1/models.data[{index}].id to be a non-empty string.")
        identifier = raw_identifier.strip()
        if identifier in seen:
            raise OllamaCloudCatalogError(f"Duplicate Ollama Cloud model id {identifier!r}.")
        seen.add(identifier)
        identifiers.append(identifier)
    return identifiers


def _combined_payload_parts(payload: Any) -> tuple[Any, Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        raise OllamaCloudCatalogError("Expected a combined Ollama Cloud catalog object.")
    models_payload = payload.get("models")
    details = payload.get("details")
    if not isinstance(details, Mapping):
        raise OllamaCloudCatalogError("Expected combined payload.details to be an object keyed by model id.")
    return models_payload, details


def _capabilities(identifier: str, detail: Any) -> set[str]:
    if not isinstance(detail, Mapping):
        raise OllamaCloudCatalogError(f"Expected /api/show details for {identifier!r} to be an object.")
    raw_capabilities = detail.get("capabilities")
    if not _is_sequence(raw_capabilities):
        raise OllamaCloudCatalogError(f"Expected /api/show capabilities for {identifier!r} to be a list.")
    assert isinstance(raw_capabilities, Sequence)

    capabilities: set[str] = set()
    for capability in raw_capabilities:
        if not isinstance(capability, str) or not capability.strip():
            raise OllamaCloudCatalogError(f"Invalid /api/show capability for {identifier!r}: {capability!r}.")
        capabilities.add(capability.strip())
    return capabilities


def _context_length(identifier: str, detail: Mapping[str, Any]) -> int:
    model_info = detail.get("model_info")
    if not isinstance(model_info, Mapping):
        raise OllamaCloudCatalogError(f"Expected /api/show model_info for {identifier!r} to be an object.")

    candidates = [(key, value) for key, value in model_info.items() if str(key).endswith(".context_length")]
    if len(candidates) != 1:
        raise OllamaCloudCatalogError(
            f"Expected exactly one *.context_length field for {identifier!r}; found {len(candidates)}."
        )
    key, value = candidates[0]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OllamaCloudCatalogError(f"Expected {key!r} for {identifier!r} to be a positive integer.")
    return value


def _model_spec(identifier: str, detail: Mapping[str, Any], capabilities: set[str]) -> dict[str, Any]:
    context_length = _context_length(identifier, detail)
    spec: dict[str, Any] = {
        "context_length": context_length,
        "max_completion_tokens": min(
            context_length,
            MAX_COMPLETION_TOKEN_OVERRIDES.get(identifier, FALLBACK_MAX_COMPLETION_TOKENS),
        ),
        "default_temperature": DEFAULT_TEMPERATURE,
        "supports_vision": "vision" in capabilities,
    }
    # Upstream declares no budget convention; classify from the pair itself.
    spec["input_budget"] = (
        "output_shares_window" if spec["max_completion_tokens"] >= context_length else "window_minus_output"
    )
    if "thinking" in capabilities:
        spec["thinking_effort"] = THINKING_EFFORT_SPEC
    return spec


def transform_catalog(payload: Any) -> CatalogTransformResult:
    """Transform a combined list/detail payload into deterministic model specs."""
    models_payload, details = _combined_payload_parts(payload)
    identifiers = _model_ids(models_payload)

    entries: list[tuple[str, dict[str, Any]]] = []
    filtered: list[str] = []
    for identifier in identifiers:
        if identifier not in details:
            raise OllamaCloudCatalogError(f"Missing /api/show details for {identifier!r}.")
        detail = details[identifier]
        capabilities = _capabilities(identifier, detail)
        if not {"completion", "tools"} <= capabilities:
            filtered.append(identifier)
            continue
        assert isinstance(detail, Mapping)
        entries.append((identifier, _model_spec(identifier, detail, capabilities)))

    entries.sort(key=lambda entry: entry[0])
    filtered.sort()
    if not entries:
        raise OllamaCloudCatalogError("Ollama returned no completion-and-tools-capable cloud models.")
    return CatalogTransformResult(entries=entries, filtered_ids=tuple(filtered))


def catalog_entries(payload: Any) -> list[tuple[str, dict[str, Any]]]:
    """Return transformed ``(model id, spec)`` entries for overlay callers."""
    return transform_catalog(payload).entries


def fetch_models(*, timeout: float = 30.0, api_key: Optional[str] = None) -> dict[str, Any]:
    """Fetch `/v1/models` plus every matching `/api/show` detail response."""
    token = api_key if api_key is not None else os.environ.get("OLLAMA_API_KEY")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
            response = client.get(MODELS_URL)
            response.raise_for_status()
            models_payload = response.json()
            identifiers = _model_ids(models_payload)

            details: dict[str, Any] = {}
            for identifier in identifiers:
                try:
                    detail_response = client.post(SHOW_URL, json={"model": identifier})
                    detail_response.raise_for_status()
                    details[identifier] = detail_response.json()
                except Exception as exc:
                    raise OllamaCloudCatalogError(
                        f"Could not fetch /api/show details for Ollama Cloud model {identifier!r}: {exc}"
                    ) from exc
    except OllamaCloudCatalogError:
        raise
    except Exception as exc:
        raise OllamaCloudCatalogError(f"Could not fetch the Ollama Cloud model list: {exc}") from exc

    return {"models": models_payload, "details": details}


def spec_to_jsonable(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-serializable copy of a model spec."""
    payload = dict(spec)
    thinking = payload.get("thinking_effort")
    if isinstance(thinking, ThinkingEffortSpec):
        payload["thinking_effort"] = {
            "options": list(thinking.options),
            "default": thinking.default,
            "mode": thinking.mode,
        }
    return payload


def spec_from_jsonable(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild and validate a runtime model spec from cached JSON."""
    spec = dict(payload)
    thinking = spec.get("thinking_effort")
    if thinking is not None:
        if not isinstance(thinking, Mapping):
            raise ValueError("thinking_effort must be an object")
        options = thinking.get("options")
        default = thinking.get("default")
        mode = thinking.get("mode")
        if not _is_sequence(options) or not options or not all(isinstance(option, str) for option in options):
            raise ValueError("thinking_effort.options must be a non-empty string list")
        normalized_options = tuple(options)
        if not isinstance(default, str) or default not in normalized_options:
            raise ValueError("thinking_effort.default must be one of its options")
        if mode != "openai_reasoning_effort":
            raise ValueError("unexpected thinking_effort.mode")
        spec["thinking_effort"] = ThinkingEffortSpec(
            options=normalized_options,
            default=default,
            mode="openai_reasoning_effort",
        )

    for key in ("context_length", "max_completion_tokens"):
        value = spec.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    temperature = spec.get("default_temperature")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("default_temperature must be numeric")
    if not isinstance(spec.get("supports_vision"), bool):
        raise ValueError("supports_vision must be a boolean")
    return spec


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
    """Render the generated ``catalog/ollama_cloud.py`` source."""
    lines: list[str] = []
    for line in header_lines:
        lines.append(f"# {line}".rstrip())
    lines.append("")
    lines.append("from kolega_code.llm.specs.types import ThinkingEffortSpec")
    lines.append("")
    lines.append("OLLAMA_CLOUD_SPECS = {")
    for identifier, spec in entries:
        lines.append(f"    ({PROVIDER!r}, {identifier!r}): {{")
        ordered_keys = [key for key in _SPEC_KEY_ORDER if key in spec]
        ordered_keys += sorted(key for key in spec if key not in _SPEC_KEY_ORDER)
        for key in ordered_keys:
            lines.append(f"        {key!r}: {_render_value(spec[key])},")
        lines.append("    },")
    lines.append("}")
    return "\n".join(lines) + "\n"


def save_cache(path: Path, entries: Sequence[tuple[str, Mapping[str, Any]]], *, fetched_at: str) -> None:
    """Write a refreshed catalog overlay to ``path`` with private permissions."""
    from kolega_code.local_state import write_private_text

    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "fetched_at": fetched_at,
        "provider": PROVIDER,
        "models": [{"id": identifier, "spec": spec_to_jsonable(spec)} for identifier, spec in entries],
    }
    write_private_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_cache(path: Path) -> list[tuple[str, dict[str, Any]]]:
    """Read a cache, returning ``[]`` for any unusable or foreign file."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != CACHE_SCHEMA_VERSION
        or payload.get("provider") != PROVIDER
    ):
        return []
    models = payload.get("models")
    if not _is_sequence(models):
        return []
    assert isinstance(models, Sequence)

    entries: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for item in models:
        if not isinstance(item, Mapping):
            return []
        identifier = item.get("id")
        spec = item.get("spec")
        if not isinstance(identifier, str) or not identifier or identifier in seen or not isinstance(spec, Mapping):
            return []
        seen.add(identifier)
        try:
            entries.append((identifier, spec_from_jsonable(spec)))
        except (TypeError, ValueError):
            return []
    return entries
