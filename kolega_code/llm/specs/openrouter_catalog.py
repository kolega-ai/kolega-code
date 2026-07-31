"""OpenRouter catalog conversion: API payloads to ``MODEL_SPECS`` entries.

OpenRouter is a gateway in front of hundreds of models, so its catalog is
generated rather than hand-maintained. This module owns the whole transform and
is shared by two callers so they can never drift:

* ``scripts/refresh_openrouter_catalog.py`` — the dev-time generator that
  rewrites ``kolega_code/llm/specs/catalog/openrouter.py`` before a release.
* ``kolega-code models refresh`` — the opt-in runtime overlay that lets a user
  reach models published after the current Kolega Code release.

Two upstream endpoints are involved:

``GET /api/v1/models``
    Documented, keyless, and authoritative for every spec field. Required.

``GET /api/frontend/v1/rankings/models?view=week``
    Undocumented; it backs OpenRouter's public LLM Leaderboard. Summing
    ``total_prompt_tokens + total_completion_tokens`` per
    ``(model_permaslug, variant)`` over the ``week`` view reproduces the site's
    "This Week / All models" ranking exactly. It is used **only** by the
    generator, only for ordering, and every caller must tolerate its absence —
    see ``catalog_entries`` and the fallbacks in the generator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from .types import ThinkingEffortSpec

PROVIDER = "openrouter"
BASE_URL = "https://openrouter.ai/api/v1"
MODELS_URL = f"{BASE_URL}/models"
RANKINGS_URL = "https://openrouter.ai/api/frontend/v1/rankings/models"
DEFAULT_RANKINGS_VIEW = "week"

# Number of leading (most-used) catalog entries marked ``featured``. Featured
# models are the ones every list-style surface shows; the rest of the catalog
# stays resolvable by id. See ``kolega_code/cli/provider_registry.py``.
FEATURED_COUNT = 20

# Effort names ordered from least to most reasoning. OpenRouter reports its
# supported efforts in the opposite order and omits some values, so options are
# re-sorted into this canonical order for a stable UI.
CANONICAL_EFFORT_ORDER: tuple[str, ...] = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

# Meta-routers: they report ``pricing.prompt == "-1"`` and no usable output cap,
# so no honest spec can be derived for them.
META_ROUTER_MODEL_IDS = frozenset({"openrouter/auto", "openrouter/auto-beta", "openrouter/free"})

# Output cap used when OpenRouter reports no ``top_provider.max_completion_tokens``.
# Only Kolega Code's own context budgeting reads this value: the OpenRouter
# request path deliberately omits ``max_tokens`` (see the provider).
FALLBACK_MAX_COMPLETION_TOKENS = 32768

# Edit protocol offered to OpenRouter models. The gateway fronts a long tail of
# models that were never individually benchmarked here, so the default is the
# Claude Code-style JSON edit tool, which the broadest set of tool-calling models
# handles well. OpenAI models are the exception: they are trained on the Codex
# apply_patch freeform format and do better with it, matching what the direct
# `openai` / `openai_chatgpt` catalogs already select.
DEFAULT_EDIT_PROTOCOL = "claude_code"
_CODEX_APPLY_PATCH_PREFIXES = ("openai/",)

# Anthropic reasoning is carried by signed thinking blocks. Replaying it through
# OpenRouter's plain ``reasoning`` string produces an unsigned block, so prior
# reasoning is dropped for these ids instead (see ``prior_reasoning_is_replayable``).
_DROP_PRIOR_REASONING_PREFIXES = ("anthropic/",)

CACHE_SCHEMA_VERSION = 1
CACHE_FILENAME = "openrouter_models.json"


class OpenRouterCatalogError(RuntimeError):
    """Raised when an OpenRouter payload cannot be used to build a catalog."""


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_positive_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def model_id(entry: Mapping[str, Any]) -> str:
    """Return the wire model id (``vendor/model`` with an optional ``:variant``)."""
    return str(entry.get("id") or "")


def base_model_id(identifier: str) -> str:
    """Return a model id without its ``:variant`` suffix."""
    return identifier.split(":", 1)[0]


def variant_of(identifier: str) -> str:
    """Return the OpenRouter routing variant encoded in a model id.

    OpenRouter's rankings report the variant separately from the permaslug, and
    an id without a suffix is the ``standard`` variant.
    """
    _, separator, suffix = identifier.partition(":")
    return suffix if separator and suffix else "standard"


def is_catalogable(entry: Mapping[str, Any]) -> bool:
    """Whether an ``/api/v1/models`` entry belongs in the Kolega Code catalog.

    The agent cannot run without tool calling, so a model that cannot call tools
    is deliberately absent rather than present-and-broken.
    """
    identifier = model_id(entry)
    if not identifier or identifier in META_ROUTER_MODEL_IDS:
        return False
    # Floating "latest" aliases churn between releases and resolve elsewhere.
    if identifier.startswith("~"):
        return False
    # Batch endpoints are asynchronous and cannot back an interactive agent.
    if identifier.endswith(":batch"):
        return False

    supported = entry.get("supported_parameters") or []
    if not isinstance(supported, Sequence) or "tools" not in supported:
        return False

    architecture = entry.get("architecture") or {}
    input_modalities = architecture.get("input_modalities") or []
    if not isinstance(input_modalities, Sequence) or "text" not in input_modalities:
        return False

    if _as_positive_int(entry.get("context_length")) is None:
        return False

    pricing = entry.get("pricing") or {}
    prompt_price = _as_float(pricing.get("prompt"))
    # A negative price marks a meta-router whose real model is chosen upstream.
    if prompt_price is None or prompt_price < 0:
        return False

    return True


def _thinking_effort_spec(entry: Mapping[str, Any]) -> Optional[ThinkingEffortSpec]:
    reasoning = entry.get("reasoning")
    if not isinstance(reasoning, Mapping):
        return None
    raw_efforts = reasoning.get("supported_efforts")
    if not isinstance(raw_efforts, Sequence) or isinstance(raw_efforts, (str, bytes)):
        return None

    known = {str(effort).strip().lower() for effort in raw_efforts}
    options = [effort for effort in CANONICAL_EFFORT_ORDER if effort in known]
    if not options:
        return None

    # "none" disables reasoning entirely, which a reasoning-mandatory model rejects.
    if not reasoning.get("mandatory") and "none" not in options:
        options.insert(0, "none")

    api_default = str(reasoning.get("default_effort") or "").strip().lower()
    if "medium" in options:
        default = "medium"
    elif api_default in options:
        default = api_default
    else:
        graded = [effort for effort in options if effort != "none"]
        default = graded[-1] if graded else options[-1]

    return ThinkingEffortSpec(options=tuple(options), default=default, mode="openrouter_reasoning")


def _preferred_edit_protocol(identifier: str) -> str:
    """Return the edit protocol this model should be offered."""
    if base_model_id(identifier).startswith(_CODEX_APPLY_PATCH_PREFIXES):
        return "codex_apply_patch"
    return DEFAULT_EDIT_PROTOCOL


def model_spec(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one ``/api/v1/models`` entry into a ``MODEL_SPECS`` value.

    The returned mapping never carries ``featured``: featuring is a property of
    a model's rank within the generated ordering, not of the upstream payload.
    """
    identifier = model_id(entry)
    context_length = _as_positive_int(entry.get("context_length"))
    if context_length is None:
        raise OpenRouterCatalogError(f"Model {identifier!r} has no usable context length.")

    top_provider = entry.get("top_provider") or {}
    max_completion_tokens = _as_positive_int(top_provider.get("max_completion_tokens"))
    if max_completion_tokens is None:
        max_completion_tokens = min(context_length, FALLBACK_MAX_COMPLETION_TOKENS)

    architecture = entry.get("architecture") or {}
    input_modalities = architecture.get("input_modalities") or []
    supported = entry.get("supported_parameters") or []

    spec: dict[str, Any] = {
        "context_length": context_length,
        "max_completion_tokens": max_completion_tokens,
        "default_temperature": 1.0,
    }
    # Reasoning models such as openai/gpt-5.6-* reject an explicit temperature.
    if "temperature" not in supported:
        spec["supports_temperature"] = False
    spec["supports_vision"] = "image" in input_modalities

    spec["preferred_edit_protocol"] = _preferred_edit_protocol(identifier)

    thinking = _thinking_effort_spec(entry)
    if thinking is not None:
        spec["thinking_effort"] = thinking

    if identifier.startswith(_DROP_PRIOR_REASONING_PREFIXES):
        spec["drop_prior_reasoning"] = True

    return spec


def ranking_key(entry: Mapping[str, Any]) -> tuple[str, str]:
    """Return the ``(permaslug, variant)`` key that joins a model to its ranking.

    ``canonical_slug`` is the dated, variant-free slug the rankings feed reports
    as ``model_permaslug``; the ``:variant`` suffix travels in its own column.
    """
    identifier = model_id(entry)
    permaslug = str(entry.get("canonical_slug") or base_model_id(identifier))
    return permaslug, variant_of(identifier)


def ranking_scores(payload: Any) -> dict[tuple[str, str], int]:
    """Aggregate a rankings payload into ``(permaslug, variant) -> total tokens``.

    Summing prompt plus completion tokens over the requested view is what
    reproduces OpenRouter's public leaderboard ordering and totals.
    """
    rows = payload.get("data") if isinstance(payload, Mapping) else payload
    if not isinstance(rows, Sequence):
        raise OpenRouterCatalogError("Rankings payload has no 'data' list.")

    scores: dict[tuple[str, str], int] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        permaslug = row.get("model_permaslug")
        if not permaslug:
            continue
        key = (str(permaslug), str(row.get("variant") or "standard"))
        prompt_tokens = _as_positive_int(row.get("total_prompt_tokens")) or 0
        completion_tokens = _as_positive_int(row.get("total_completion_tokens")) or 0
        scores[key] = scores.get(key, 0) + prompt_tokens + completion_tokens
    if not scores:
        raise OpenRouterCatalogError("Rankings payload contained no usable rows.")
    return scores


def catalog_entries(
    models_payload: Any,
    scores: Optional[Mapping[tuple[str, str], int]] = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Return ``(model_id, spec)`` pairs, most-used first.

    With ``scores`` the order is descending popularity; without them (the
    rankings feed is optional and undocumented) it degrades to alphabetical.
    Ties and unranked models always break alphabetically, so the output is
    deterministic for a given pair of payloads.
    """
    rows = models_payload.get("data") if isinstance(models_payload, Mapping) else models_payload
    if not isinstance(rows, Sequence):
        raise OpenRouterCatalogError("Models payload has no 'data' list.")

    selected = [entry for entry in rows if isinstance(entry, Mapping) and is_catalogable(entry)]
    if not selected:
        raise OpenRouterCatalogError("Models payload contained no tool-capable models.")

    def sort_key(entry: Mapping[str, Any]) -> tuple[int, str]:
        score = int(scores.get(ranking_key(entry), 0)) if scores else 0
        return (-score, model_id(entry))

    return [(model_id(entry), model_spec(entry)) for entry in sorted(selected, key=sort_key)]


def featured_ids(
    entries: Sequence[tuple[str, dict[str, Any]]],
    *,
    always_include: Iterable[str] = (),
    count: int = FEATURED_COUNT,
) -> list[str]:
    """Return the ids to mark ``featured``: the top ``count``, plus pinned ids.

    ``always_include`` exists so the provider's default model is guaranteed to
    appear in the Settings picker even on a week when it drops out of the top
    ``count``; it never reorders anything.
    """
    catalogued = [identifier for identifier, _ in entries]
    featured = catalogued[:count]
    known = set(catalogued)
    seen = set(featured)
    for identifier in always_include:
        if identifier in known and identifier not in seen:
            featured.append(identifier)
            seen.add(identifier)
    return featured


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
    if isinstance(thinking, Mapping):
        options = thinking.get("options") or []
        spec["thinking_effort"] = ThinkingEffortSpec(
            options=tuple(str(option) for option in options),
            default=str(thinking.get("default") or ""),
            mode=str(thinking.get("mode") or "openrouter_reasoning"),
        )
    return spec


# Spec keys are emitted in this order so a regenerated catalog produces a
# reviewable diff rather than a reshuffle. Unknown keys are appended sorted.
_SPEC_KEY_ORDER: tuple[str, ...] = (
    "context_length",
    "max_completion_tokens",
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
    """Render the generated ``catalog/openrouter.py`` source.

    Generator-only helper, kept beside the transform it serializes so both can
    be exercised by the same offline tests.
    """
    lines: list[str] = []
    for line in header_lines:
        lines.append(f"# {line}".rstrip())
    lines.append("")
    lines.append("from kolega_code.llm.specs.types import ThinkingEffortSpec")
    lines.append("")
    lines.append("OPENROUTER_SPECS = {")
    for identifier, spec in entries:
        lines.append(f"    ({PROVIDER!r}, {identifier!r}): {{")
        ordered_keys = [key for key in _SPEC_KEY_ORDER if key in spec]
        ordered_keys += sorted(key for key in spec if key not in _SPEC_KEY_ORDER)
        for key in ordered_keys:
            lines.append(f"        {key!r}: {_render_value(spec[key])},")
        lines.append("    },")
    lines.append("}")
    return "\n".join(lines) + "\n"


def fetch_models(*, timeout: float = 30.0) -> Any:
    """Fetch the documented ``/api/v1/models`` payload."""
    import httpx

    response = httpx.get(MODELS_URL, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    return response.json()


def fetch_rankings(*, view: str = DEFAULT_RANKINGS_VIEW, timeout: float = 30.0) -> Any:
    """Fetch the undocumented leaderboard payload backing catalog ordering."""
    import httpx

    response = httpx.get(RANKINGS_URL, params={"view": view}, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    return response.json()


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
        try:
            entries.append((str(identifier), spec_from_jsonable(spec)))
        except (TypeError, ValueError):
            continue
    return entries
