"""Tinker catalog conversion: ``models.json`` payloads to ``MODEL_SPECS`` entries.

Tinker publishes its model catalog as machine-readable JSON — the documented
stable interface for scripts, not the presentational HTML tables:

    https://tinker-docs.thinkingmachines.ai/tinker/models.json

Each entry carries ``name``, ``tinker_id`` (the exact model string), ``context``
("32K"/"64K"/"128K"/"256K"), ``type`` ("Hybrid + Vision", "Reasoning", "Base",
...), ``arch``, and prices. This module owns the whole transform and is shared
by two callers so they can never drift:

* ``scripts/refresh_tinker_catalog.py`` — the dev-time generator that rewrites
  ``kolega_code/llm/specs/catalog/tinker.py`` before a release.
* ``kolega-code models refresh`` — the opt-in runtime overlay that lets a user
  reach Tinker base models published after the current Kolega Code release.

The transform layers the derived defaults Kolega needs on top of the JSON
(output-token cap, temperature, per-family thinking-effort spec) and excludes
``Base`` type models: they are raw pretrained checkpoints with no chat template,
so no chat renderer can produce a meaningful conversation for them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import httpx

from .types import ThinkingEffortSpec

PROVIDER = "tinker"
MODELS_URL = "https://tinker-docs.thinkingmachines.ai/tinker/models.json"

CACHE_SCHEMA_VERSION = 1
CACHE_FILENAME = "tinker_models.json"

# Tinker's docs sit behind Cloudflare and return HTTP 403 (error 1010) for
# plain script user agents; a browser-like UA is required to fetch models.json.
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

# Client-side output cap and temperature for every Tinker base model. The
# catalog never sends values the service does not document; 32768 is the cap
# used across the direct-provider catalogs.
MAX_COMPLETION_TOKENS = 32768
DEFAULT_TEMPERATURE = 1.0

# Named-effort spec for the Inkling family (tml-renderers effort floats).
INKLING_EFFORT_SPEC = ThinkingEffortSpec(
    options=("none", "low", "medium", "high", "xhigh", "max"),
    default="high",
    mode="tinker_native_effort",
)

# Thinking on/off spec for cookbook-rendered families with documented
# thinking and disable-thinking renderer variants.
_TOGGLE_EFFORT_SPEC = ThinkingEffortSpec(
    options=("none", "auto"),
    default="auto",
    mode="tinker_native_effort",
)

# Per-family derived specs, keyed by base-model id prefix (longest first).
# Families absent here get no thinking spec: no effort parameter is sent and
# the renderer's default behavior applies (GPT-OSS always reasons; Nemotron
# hybrids reason by default).
_FAMILY_SPECS: tuple[tuple[str, Mapping[str, Any]], ...] = (
    ("thinkingmachines/", {"thinking_effort": INKLING_EFFORT_SPEC}),
    ("Qwen/Qwen3.6-", {"thinking_effort": _TOGGLE_EFFORT_SPEC}),
    ("Qwen/Qwen3.5-", {"thinking_effort": _TOGGLE_EFFORT_SPEC}),
    ("deepseek-ai/", {"thinking_effort": _TOGGLE_EFFORT_SPEC}),
    ("moonshotai/Kimi-K2.6", {"thinking_effort": _TOGGLE_EFFORT_SPEC}),
)

_CONTEXT_TOKENS = {
    "32K": 32768,
    "64K": 65536,
    "128K": 131072,
    "256K": 262144,
}


class TinkerCatalogError(RuntimeError):
    """Raised when a Tinker payload cannot be used to build a catalog."""


def _context_tokens(context: Any) -> int:
    value = str(context or "").strip()
    if value not in _CONTEXT_TOKENS:
        raise TinkerCatalogError(f"Unhandled Tinker context window {context!r}.")
    return _CONTEXT_TOKENS[value]


def _family_spec(tinker_id: str) -> dict[str, Any]:
    for prefix, spec in _FAMILY_SPECS:
        if tinker_id.startswith(prefix):
            return dict(spec)
    return {}


def catalog_entries(payload: Any) -> list[tuple[str, dict[str, Any]]]:
    """Transform a Tinker ``models.json`` payload into ``(id, spec)`` entries.

    ``Base`` type models are excluded (no chat template). Models are kept in
    the payload's own order.
    """
    if not isinstance(payload, Sequence):
        raise TinkerCatalogError("Expected a JSON list of models.")
    entries: list[tuple[str, dict[str, Any]]] = []
    for item in payload:
        if not isinstance(item, Mapping):
            continue
        identifier = item.get("tinker_id")
        if not identifier:
            continue
        model_type = str(item.get("type") or "")
        if model_type == "Base":
            continue
        spec: dict[str, Any] = {
            "context_length": _context_tokens(item.get("context")),
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "default_temperature": DEFAULT_TEMPERATURE,
            "supports_vision": "Vision" in model_type,
        }
        spec.update(_family_spec(str(identifier)))
        entries.append((str(identifier), spec))
    return entries


def fetch_models(*, timeout: float = 30.0) -> Any:
    """Fetch Tinker's published ``models.json``."""
    response = httpx.get(MODELS_URL, timeout=timeout, headers={"User-Agent": _USER_AGENT}, follow_redirects=True)
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
    if isinstance(thinking, Mapping) and thinking.get("mode") == "tinker_native_effort":
        spec["thinking_effort"] = ThinkingEffortSpec(
            options=tuple(thinking.get("options") or ()),
            default=str(thinking.get("default") or ""),
            mode="tinker_native_effort",
        )
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
        entries.append((str(identifier), spec_from_jsonable(spec)))
    return entries


# Fallback spec for a user's own fine-tuned checkpoints, addressed by sampler
# weights path (tinker://<run-id>:train:<n>/sampler_weights/<step>). Any
# "tinker://" model id resolves to this template so the agent never hard-fails
# on an uncatalogued checkpoint; the provider resolves the checkpoint's real
# base model at runtime (SamplingClient.get_base_model) and uses the base's
# catalog spec when available. Values here are deliberately conservative:
# 64K context (the smallest Tinker offering — a larger-base checkpoint only
# triggers earlier compression, never overflow), vision off per repo
# convention (uncertain => clear guard over a mid-flight API error), and no
# thinking spec (effort support is base-dependent and unknown from the path).
TINKER_WILDCARD_SPECS = {
    (PROVIDER, "tinker://*"): {
        "context_length": 65536,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "default_temperature": DEFAULT_TEMPERATURE,
        "supports_vision": False,
    },
}
