"""First-party Mistral catalog conversion for ``models refresh``.

Mistral publishes its account-visible models at ``GET /v1/models``. Unlike the
public overlay endpoints (OpenRouter, Ollama Cloud, Tinker), this one is
authenticated, so ``fetch_models`` takes an API key instead of reading ambient
configuration; the CLI resolves the credential and passes it in.

The payload is not one row per model. A single model appears several times
(``mistral-medium-3-5``, ``mistral-medium-latest``, ``mistral-vibe-cli-*``, …)
and every row repeats the same ``aliases`` list. This module groups rows by
billing identity and emits exactly one entry per underlying model under a
canonical id. Alias coverage stays a curated concern of the bundled snapshot: a
refresh discovers models, it does not re-derive the reviewed alias list.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import httpx

from .catalog.mistral import MISTRAL_REASONING
from .types import ThinkingEffortSpec
from .validation import validate_model_spec

PROVIDER = "mistral"
MODELS_URL = "https://api.mistral.ai/v1/models"

# Marks this overlay as requiring a credential. ``models refresh`` resolves it
# from ``MISTRAL_API_KEY`` (or a key saved in Settings) before calling
# ``fetch_models``; overlays without this attribute are fetched anonymously.
API_KEY_ENV = "MISTRAL_API_KEY"

CACHE_FILENAME = "mistral_models.json"
CACHE_SCHEMA_VERSION = 1

# Kolega's reserved output budget, not a native ceiling. Mistral's model list
# reports ``max_context_length`` but no maximum-output field; the native
# Chat Completion validation is the only place a real ceiling appears (Large 4
# reports 262,144). Keeping a 32,768 reservation preserves input headroom under
# the shared window.
MAX_COMPLETION_TOKENS = 32768
DEFAULT_TEMPERATURE = 1.0

# Capabilities that make a chat-capable endpoint unusable as an agent model:
# audio/OCR/moderation/embedding endpoints are published in the same list.
_NON_CHAT_CAPABILITIES = (
    "audio",
    "audio_transcription",
    "audio_transcription_realtime",
    "audio_speech",
    "ocr",
    "moderation",
    "classification",
)

# Organization-gated research models. The account lists them but inference
# returns HTTP 403 until an admin enables Labs access, so they are not offered.
_EXCLUDED_ID_PREFIXES = ("labs-",)

# Only first-party families get Kolega's thinking control. Mistral-hosted
# third-party models report ``reasoning: true`` but do not share the effort
# contract: ``zai-glm-5-3`` rejects ``reasoning_effort: "none"`` (only
# low/high/max), which would break the default request path. They keep the
# server default instead.
_REASONING_ID_PREFIXES = ("mistral-", "ministral-", "magistral-")


class MistralCatalogError(RuntimeError):
    """Raised when a Mistral payload cannot be used to build a catalog."""


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _temperature(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return DEFAULT_TEMPERATURE
    temperature = float(value)
    return temperature if 0.0 <= temperature <= 2.0 else DEFAULT_TEMPERATURE


def _group_key(item: Mapping[str, Any]) -> str:
    """The billing identity shared by every row of the same underlying model."""
    return _text(item.get("billing_model_name")) or _text(item.get("name")) or _text(item.get("id"))


def _canonical_id(identifier: str, group: Sequence[Mapping[str, Any]]) -> str:
    """Pick the reviewed id for a model group: versioned, never a mutable alias."""
    ids = [value for value in (_text(item.get("id")) for item in group) if value]
    if not ids:
        return identifier
    billing = _text(group[0].get("billing_model_name"))
    if billing in ids:
        return billing
    name = _text(group[0].get("name"))
    if name in ids:
        return name
    return min(ids)


def _capabilities(item: Mapping[str, Any]) -> Mapping[str, Any]:
    capabilities = item.get("capabilities")
    return capabilities if isinstance(capabilities, Mapping) else {}


def _is_usable(item: Mapping[str, Any]) -> bool:
    if item.get("deprecation"):
        return False
    capabilities = _capabilities(item)
    if not capabilities.get("completion_chat"):
        return False
    return not any(capabilities.get(name) for name in _NON_CHAT_CAPABILITIES)


def _spec_for(identifier: str, item: Mapping[str, Any]) -> dict[str, Any] | None:
    """Build a Kolega spec, or ``None`` when the entry is missing its window."""
    context_length = _positive_int(item.get("max_context_length"))
    if context_length is None:
        return None
    capabilities = _capabilities(item)
    spec: dict[str, Any] = {
        "context_length": context_length,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "input_budget": "window_minus_output",
        "default_temperature": _temperature(item.get("default_model_temperature")),
        "supports_vision": bool(capabilities.get("vision")),
        "supports_function_calling": bool(capabilities.get("function_calling")),
        "supports_hosted_web_search": False,
    }
    if capabilities.get("reasoning") and identifier.startswith(_REASONING_ID_PREFIXES):
        spec["thinking_effort"] = MISTRAL_REASONING
    return spec


def catalog_entries(payload: Any) -> list[tuple[str, dict[str, Any]]]:
    """Transform a Mistral ``/v1/models`` payload into ``(id, spec)`` entries.

    One entry per underlying model, in the payload's own order (which is the
    order the cache and the picker will show). Rows that share a billing
    identity collapse onto their canonical id; aliases are not emitted.
    """
    rows = payload.get("data") if isinstance(payload, Mapping) else payload
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise MistralCatalogError("Expected a JSON object with a 'data' list.")

    groups: dict[str, list[Mapping[str, Any]]] = {}
    for item in rows:
        if not isinstance(item, Mapping):
            continue
        if _text(item.get("id")) == "":
            continue
        key = _group_key(item)
        if not key:
            continue
        groups.setdefault(key, []).append(item)

    entries: list[tuple[str, dict[str, Any]]] = []
    for key, group in groups.items():
        identifier = _canonical_id(key, group)
        if identifier.startswith(_EXCLUDED_ID_PREFIXES):
            continue
        representative = next(
            (item for item in group if _text(item.get("id")) == identifier),
            group[0],
        )
        if not _is_usable(representative):
            continue
        spec = _spec_for(identifier, representative)
        if spec is None:
            continue
        entries.append((identifier, spec))
    return entries


def fetch_models(*, api_key: str, timeout: float = 30.0) -> Any:
    """Fetch the account-visible Mistral model list."""
    response = httpx.get(
        MODELS_URL,
        timeout=timeout,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )
    response.raise_for_status()
    return response.json()


def spec_to_jsonable(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-serializable copy of a spec (``ThinkingEffortSpec`` expanded)."""
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
    """Rebuild a runtime spec from ``spec_to_jsonable`` output."""
    spec = dict(payload)
    thinking = spec.get("thinking_effort")
    if isinstance(thinking, Mapping) and thinking.get("mode") == "mistral_effort":
        spec["thinking_effort"] = ThinkingEffortSpec(
            options=tuple(thinking.get("options") or ()),
            default=str(thinking.get("default") or ""),
            mode="mistral_effort",
        )
    validate_model_spec(spec)
    return spec


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
    """Read a catalog overlay, returning ``[]`` for any unusable file.

    A malformed or future-versioned cache must never be fatal: the bundled
    snapshot alone is always a working catalog.
    """
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
    if not isinstance(models, Sequence) or isinstance(models, (str, bytes)):
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
