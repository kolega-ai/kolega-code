"""Cloudflare Workers AI catalog conversion for ``models refresh``.

The model search API is account-scoped and authenticated, so ``fetch_models``
takes both the token and the account ID rather than reading ambient config.
A refresh discovers models; the reviewed specs stay in the bundled catalog.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import httpx

from kolega_code.llm import cloudflare_workers_ai

from .types import ThinkingEffortSpec
from .validation import validate_model_spec

PROVIDER = cloudflare_workers_ai.PROVIDER
MODELS_URL = cloudflare_workers_ai.models_search_url("{account_id}")

# Marks this overlay as credential-gated; the account ID additionally scopes the URL.
API_KEY_ENV = cloudflare_workers_ai.API_TOKEN_ENV
ACCOUNT_ID_ENV = cloudflare_workers_ai.ACCOUNT_ID_ENV
ACCOUNT_ID_SETTING = "cloudflare_account_id"

CACHE_FILENAME = "cloudflare_workers_ai_models.json"
CACHE_SCHEMA_VERSION = 1

TEXT_GENERATION_TASK = "Text Generation"
MIN_CONTEXT_LENGTH = 65536
MAX_COMPLETION_TOKENS = 32768
DEFAULT_TEMPERATURE = 1.0
PAGE_SIZE = 100
MAX_PAGES = 20

EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
_THINKING_MODE = "cloudflare_reasoning_effort"
_CODEX_PATCH_PREFIX = "@cf/openai/"


class CloudflareWorkersAICatalogError(RuntimeError):
    """Raised when a model-search payload cannot be used to build a catalog."""


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return isinstance(value, str) and value.strip().lower() == "true"


def _properties(item: Mapping[str, Any]) -> dict[str, Any]:
    raw = item.get("properties")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return {}
    properties: dict[str, Any] = {}
    for entry in raw:
        if isinstance(entry, Mapping) and isinstance(entry.get("property_id"), str):
            properties[entry["property_id"]] = entry.get("value")
    return properties


def _context_window(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        window = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return window if window > 0 else None


def _thinking_spec(value: Any) -> ThinkingEffortSpec | None:
    if not isinstance(value, Mapping):
        return None
    supported = value.get("supported_efforts")
    if not isinstance(supported, Sequence) or isinstance(supported, (str, bytes)):
        return None
    labels = {str(effort).strip().lower() for effort in supported}
    options = tuple(effort for effort in EFFORT_ORDER if effort in labels)
    if not options:
        return None
    default = str(value.get("default_effort") or "").strip().lower()
    return ThinkingEffortSpec(
        options=options,
        default=default if default in options else options[-1],
        mode=_THINKING_MODE,
    )


def _spec_for(identifier: str, item: Mapping[str, Any]) -> dict[str, Any] | None:
    task = item.get("task")
    if not isinstance(task, Mapping) or task.get("name") != TEXT_GENERATION_TASK:
        return None
    if _truthy(item.get("deprecated")) or identifier.endswith("-lora"):
        return None
    properties = _properties(item)
    if not _truthy(properties.get("function_calling")):
        return None
    window = _context_window(properties.get("context_window"))
    if window is None or window < MIN_CONTEXT_LENGTH:
        return None
    spec: dict[str, Any] = {
        "context_length": window,
        "max_completion_tokens": min(MAX_COMPLETION_TOKENS, window // 4),
        "input_budget": "window_minus_output",
        "default_temperature": DEFAULT_TEMPERATURE,
        "supports_vision": _truthy(properties.get("vision")),
    }
    thinking = _thinking_spec(properties.get("reasoning_effort"))
    if thinking is not None:
        spec["thinking_effort"] = thinking
    if identifier.startswith(_CODEX_PATCH_PREFIX):
        spec["preferred_edit_protocol"] = "codex_apply_patch"
    return spec


def _rows(payload: Any) -> Sequence[Any]:
    if isinstance(payload, Mapping):
        rows = payload.get("result")
    else:
        rows = payload
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise CloudflareWorkersAICatalogError("Expected a JSON object with a 'result' list.")
    return rows


def catalog_entries(payload: Any) -> list[tuple[str, dict[str, Any]]]:
    """Transform a model-search payload into one ``(id, spec)`` per agent-capable model."""
    entries: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for item in _rows(payload):
        if not isinstance(item, Mapping):
            continue
        identifier = item.get("name")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in seen:
            continue
        spec = _spec_for(identifier, item)
        if spec is None:
            continue
        seen.add(identifier)
        entries.append((identifier, spec))
    return entries


def fetch_models(
    *,
    api_key: str,
    account_id: str,
    timeout: float = 30.0,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Fetch every Text Generation model visible to the account, following pagination."""
    url = cloudflare_workers_ai.models_search_url(account_id)
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    owned = client is None
    http = client or httpx.Client(timeout=timeout)
    results: list[Any] = []
    try:
        for page in range(1, MAX_PAGES + 1):
            response = http.get(
                url,
                headers=headers,
                params={"task": TEXT_GENERATION_TASK, "per_page": PAGE_SIZE, "page": page},
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, Mapping) or payload.get("success") is False:
                errors = payload.get("errors") if isinstance(payload, Mapping) else None
                raise CloudflareWorkersAICatalogError(f"Model search failed: {errors or payload!r}")
            rows = list(_rows(payload))
            results.extend(rows)
            if len(rows) < PAGE_SIZE:
                break
    finally:
        if owned:
            http.close()
    return {"success": True, "result": results}


def spec_to_jsonable(spec: Mapping[str, Any]) -> dict[str, Any]:
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
    spec = dict(payload)
    thinking = spec.get("thinking_effort")
    if isinstance(thinking, Mapping):
        options = tuple(str(option) for option in thinking.get("options") or ())
        default = str(thinking.get("default") or "")
        if thinking.get("mode") != _THINKING_MODE or not options or default not in options:
            raise ValueError("invalid thinking_effort in cached spec")
        spec["thinking_effort"] = ThinkingEffortSpec(options=options, default=default, mode=_THINKING_MODE)
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
    """Read a catalog overlay, returning ``[]`` for any unusable file."""
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
