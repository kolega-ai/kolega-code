"""Perplexity catalog conversion: ``/models`` payloads to ``MODEL_SPECS`` entries.

The Agent API's OpenAI-format ``/models`` endpoint carries only ``id``/
``owned_by``/``pricing`` — no context windows, output caps, or capability
flags. Specs are therefore built from conservative uniform values plus what
has been **probed against Perplexity itself** (the effort vocabulary); nothing
is derived or copied from other providers' catalogs — the same underlying
model behind a different API has no guaranteed shared behavior.

Perplexity's other surface, the inference Gateway (``/router/v1``), is
preview-gated (403 without enrollment, probed 2026-08-14) and is deliberately
not shipped as a provider.

This module is shared by two callers so they can never drift:

* ``scripts/refresh_perplexity_catalog.py`` — the dev-time generator that
  rewrites ``kolega_code/llm/specs/catalog/perplexity_agent.py`` before a
  release.
* ``kolega-code models refresh --provider perplexity_agent`` — the opt-in
  runtime overlay for models published after the bundled snapshot.

The endpoint requires a valid ``PERPLEXITY_API_KEY`` (the docs' "no
authentication required" claim does not hold — probed 2026-08-14, 401 with and
without the header).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import httpx

from .types import ThinkingEffortSpec
from .validation import validate_model_spec

AGENT_PROVIDER = "perplexity_agent"

AGENT_MODELS_URL = "https://api.perplexity.ai/v1/models"

# Server-side tools: a property of the service, stamped on every catalog entry
# like supports_hosted_web_search. Not user configurable.
AGENT_SERVER_TOOLS: tuple[str, ...] = ("web_search", "fetch_url", "finance_search", "people_search")

# v2 matches the input_budget convention required by validate_model_spec.
CACHE_SCHEMA_VERSION = 2
AGENT_CACHE_FILENAME = "perplexity_agent_models.json"

API_KEY_ENV = "PERPLEXITY_API_KEY"

# Conservative uniform values: the endpoint publishes no context/output caps,
# and nothing is copied from native-provider catalogs.
FALLBACK_CONTEXT_LENGTH = 131072
FALLBACK_MAX_COMPLETION_TOKENS = 32768
FALLBACK_EDIT_PROTOCOL = "claude_code"

# Agent API effort vocabulary, live-probed 2026-08-14: an invalid value makes
# the endpoint enumerate the accepted set, uniformly for every model —
# "validation failed: effort must be one of: minimal low medium high xhigh max"
# (notably no "none").
AGENT_EFFORT_OPTIONS: tuple[str, ...] = ("minimal", "low", "medium", "high", "xhigh", "max")
AGENT_EFFORT_DEFAULT = "medium"


class PerplexityCatalogError(RuntimeError):
    """Raised when a Perplexity payload cannot be used to build a catalog."""


def model_spec() -> dict[str, Any]:
    """The uniform spec for one Agent API model."""
    return {
        "context_length": FALLBACK_CONTEXT_LENGTH,
        "max_completion_tokens": FALLBACK_MAX_COMPLETION_TOKENS,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": False,
        "preferred_edit_protocol": FALLBACK_EDIT_PROTOCOL,
        "thinking_effort": ThinkingEffortSpec(
            options=AGENT_EFFORT_OPTIONS,
            default=AGENT_EFFORT_DEFAULT,
            mode="openai_responses_reasoning",
        ),
    }


def _model_ids(payload: Any) -> list[str]:
    if not isinstance(payload, Mapping):
        raise PerplexityCatalogError("Expected /models to return a JSON object.")
    rows = payload.get("data")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise PerplexityCatalogError("Expected /models.data to be a non-empty list.")
    identifiers: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(rows):
        if not isinstance(item, Mapping):
            raise PerplexityCatalogError(f"Expected /models.data[{index}] to be an object.")
        raw_identifier = item.get("id")
        if not isinstance(raw_identifier, str) or not raw_identifier.strip():
            raise PerplexityCatalogError(f"Expected /models.data[{index}].id to be a non-empty string.")
        identifier = raw_identifier.strip()
        if identifier in seen:
            raise PerplexityCatalogError(f"Duplicate Perplexity model id {identifier!r}.")
        seen.add(identifier)
        identifiers.append(identifier)
    return identifiers


def catalog_entries(payload: Any) -> list[tuple[str, dict[str, Any]]]:
    """Transform a Perplexity ``/models`` payload into ``(id, spec)`` entries.

    Every entry gets the same uniform spec. Model order is the payload's own.
    """
    entries: list[tuple[str, dict[str, Any]]] = []
    for identifier in _model_ids(payload):
        spec = model_spec()
        validate_model_spec(spec)
        entries.append((identifier, spec))
    if not entries:
        raise PerplexityCatalogError("Perplexity /models returned no models.")
    return entries


def fetch_models(
    models_url: str,
    *,
    timeout: float = 30.0,
    api_key: Optional[str] = None,
) -> dict[str, Any]:
    """Fetch one Perplexity ``/models`` endpoint (Bearer ``PERPLEXITY_API_KEY``)."""
    token = api_key if api_key is not None else os.environ.get(API_KEY_ENV)
    if not token:
        raise PerplexityCatalogError(
            f"{API_KEY_ENV} is not set; a valid Perplexity API key is required to fetch {models_url}"
        )
    try:
        response = httpx.get(
            models_url,
            timeout=timeout,
            headers={"Authorization": f"Bearer {token}"},
            follow_redirects=True,
        )
        response.raise_for_status()
        return response.json()
    except PerplexityCatalogError:
        raise
    except Exception as exc:
        raise PerplexityCatalogError(
            f"Could not fetch {models_url}: {exc}. If this is an auth error (401), the "
            f"{API_KEY_ENV} key is invalid or inactive."
        ) from exc


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
        if (
            not isinstance(options, Sequence)
            or isinstance(options, (str, bytes))
            or not options
            or not all(isinstance(option, str) for option in options)
        ):
            raise ValueError("thinking_effort.options must be a non-empty string list")
        if not isinstance(default, str) or default not in options:
            raise ValueError("thinking_effort.default must be one of its options")
        if not isinstance(mode, str) or not mode:
            raise ValueError("thinking_effort.mode must be a non-empty string")
        spec["thinking_effort"] = ThinkingEffortSpec(options=tuple(options), default=default, mode=mode)
    for key in ("context_length", "max_completion_tokens"):
        value = spec.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    validate_model_spec(spec)
    return spec


@dataclass(frozen=True)
class PerplexityCatalogSource:
    """Duck-typed overlay source for the Perplexity Agent API (see model_catalog)."""

    PROVIDER: str
    MODELS_URL: str
    CACHE_FILENAME: str
    server_tools: tuple[str, ...] = ()

    def fetch_models(self, *, timeout: float = 30.0, api_key: Optional[str] = None) -> dict[str, Any]:
        return fetch_models(self.MODELS_URL, timeout=timeout, api_key=api_key)

    def catalog_entries(self, payload: Any) -> list[tuple[str, dict[str, Any]]]:
        entries = catalog_entries(payload)
        if self.server_tools:
            stamped = [(identifier, {**spec, "server_tools": list(self.server_tools)}) for identifier, spec in entries]
            return stamped
        return entries

    def save_cache(self, path: Path, entries: Sequence[tuple[str, Mapping[str, Any]]], *, fetched_at: str) -> None:
        """Write a refreshed catalog overlay to ``path`` with private permissions."""
        from kolega_code.local_state import write_private_text

        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "fetched_at": fetched_at,
            "provider": self.PROVIDER,
            "models": [{"id": identifier, "spec": spec_to_jsonable(spec)} for identifier, spec in entries],
        }
        write_private_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def load_cache(self, path: Path) -> list[tuple[str, dict[str, Any]]]:
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
        if payload.get("provider") != self.PROVIDER:
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
            if not isinstance(identifier, str) or not isinstance(spec, Mapping):
                continue
            try:
                entries.append((identifier, spec_from_jsonable(spec)))
            except ValueError:
                continue
        return entries


AGENT_CATALOG = PerplexityCatalogSource(
    PROVIDER=AGENT_PROVIDER,
    MODELS_URL=AGENT_MODELS_URL,
    CACHE_FILENAME=AGENT_CACHE_FILENAME,
    server_tools=AGENT_SERVER_TOOLS,
)
