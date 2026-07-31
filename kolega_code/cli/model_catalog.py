"""Model-catalog inspection and the optional OpenRouter catalog overlay.

Backs ``kolega-code models list`` and ``kolega-code models refresh``.

The bundled OpenRouter catalog is a snapshot taken when a Kolega Code release is
cut, so a model published after that release is unknown to it. ``models refresh``
fetches the documented ``/api/v1/models`` endpoint once, on request, and caches
the result in the state directory; the cache is merged into ``MODEL_SPECS`` at
CLI startup. Startup itself never touches the network, and cached entries are
never marked ``featured`` — Settings pickers lead with the release's curated-by-
usage set (and admit any catalogued id through their "Other…" entry), while
sub-agent prompts keep showing the featured set alone.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from kolega_code.config import ModelProvider
from kolega_code.llm.specs import MODEL_SPECS, is_featured_model
from kolega_code.llm.specs import openrouter_catalog as openrouter

from .provider_registry import PROVIDER_LABELS, get_ui_model, rebuild_ui_model_options
from .session_store import default_state_dir

# Point the overlay at an explicit file (useful for pinning a catalog in CI).
CATALOG_PATH_ENV = "KOLEGA_CODE_OPENROUTER_CATALOG"
# Ignore any cached overlay and use the bundled snapshot alone.
CATALOG_DISABLE_ENV = "KOLEGA_CODE_DISABLE_OPENROUTER_CATALOG"

SORT_CHOICES = ("popularity", "id")


def overlay_path(state_dir: Optional[Path] = None, env: Optional[dict[str, str]] = None) -> Path:
    """Resolve the OpenRouter overlay cache path."""
    environ = env if env is not None else dict(os.environ)
    override = environ.get(CATALOG_PATH_ENV)
    if override:
        return Path(override).expanduser()
    root = state_dir or default_state_dir(environ)
    return Path(root).expanduser() / openrouter.CACHE_FILENAME


def overlay_disabled(env: Optional[dict[str, str]] = None) -> bool:
    environ = env if env is not None else dict(os.environ)
    value = (environ.get(CATALOG_DISABLE_ENV) or "").strip().lower()
    return value not in ("", "0", "false", "no")


def apply_catalog_overlay(
    state_dir: Optional[Path] = None,
    env: Optional[dict[str, str]] = None,
) -> int:
    """Merge a cached OpenRouter catalog into ``MODEL_SPECS``; return entries added.

    Bundled entries always win: a refresh may add models, never redefine the
    reviewed specs (or the featured set) that shipped with the release. Any
    failure is silent by design — the bundled snapshot is a complete catalog on
    its own, and no CLI command should fail because a cache file is unreadable.
    """
    if overlay_disabled(env):
        return 0
    path = overlay_path(state_dir, env)
    entries = openrouter.load_cache(path)
    if not entries:
        return 0

    added = 0
    for identifier, spec in entries:
        key = (openrouter.PROVIDER, identifier)
        if key in MODEL_SPECS:
            continue
        # Overlay models are never featured: the picker keeps showing the
        # release's reviewed most-used set.
        MODEL_SPECS[key] = {key_name: value for key_name, value in spec.items() if key_name != "featured"}
        added += 1

    if added:
        rebuild_ui_model_options()
    return added


def catalog_rows(
    *,
    provider: Optional[str] = None,
    featured_only: bool = False,
    sort: str = "popularity",
) -> list[dict[str, Any]]:
    """Return one row per catalogued model, in catalog order by default.

    ``MODEL_SPECS`` preserves insertion order, and for a gateway provider that
    order is its own usage ranking, so ``rank`` is meaningful per provider.
    """
    if sort not in SORT_CHOICES:
        raise ValueError(f"Unsupported sort {sort!r}. Valid values: {', '.join(SORT_CHOICES)}")

    rows: list[dict[str, Any]] = []
    ranks: dict[str, int] = {}
    for provider_value, model in MODEL_SPECS:
        ranks[provider_value] = ranks.get(provider_value, 0) + 1
        if provider is not None and provider_value != provider:
            continue
        featured = is_featured_model(provider_value, model)
        if featured_only and not featured:
            continue
        option = get_ui_model(provider_value, model)
        specs = MODEL_SPECS[(provider_value, model)]
        rows.append(
            {
                "rank": ranks[provider_value],
                "provider": provider_value,
                "model": model,
                "context_length": int(specs["context_length"]),
                "max_completion_tokens": int(specs["max_completion_tokens"]),
                "supports_vision": bool(specs.get("supports_vision", False)),
                "thinking_efforts": list(option.thinking_efforts) if option else [],
                "default_thinking_effort": option.default_thinking_effort if option else None,
                "featured": featured,
            }
        )

    if sort == "id":
        rows.sort(key=lambda row: (row["provider"], row["model"]))
    return rows


def format_rows(rows: Sequence[dict[str, Any]]) -> str:
    """Render catalog rows as an aligned plain-text table."""
    if not rows:
        return "No models matched."

    header = ("RANK", "PROVIDER", "MODEL", "CONTEXT", "MAX OUT", "VISION", "EFFORTS", "DEFAULT", "")
    body = [
        (
            str(row["rank"]),
            row["provider"],
            row["model"],
            f"{row['context_length']:,}",
            f"{row['max_completion_tokens']:,}",
            "yes" if row["supports_vision"] else "no",
            ",".join(row["thinking_efforts"]) or "-",
            row["default_thinking_effort"] or "-",
            "*" if row["featured"] else "",
        )
        for row in rows
    ]
    widths = [max(len(str(cell)) for cell in column) for column in zip(header, *body)]
    lines = ["  ".join(str(cell).ljust(width) for cell, width in zip(row, widths)).rstrip() for row in (header, *body)]
    if any(row["featured"] for row in rows):
        lines.append("")
        lines.append("* listed in the model picker and offered to sub-agents; other models work by exact id.")
    return "\n".join(lines)


def _resolve_provider(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip().lower()
    try:
        ModelProvider(normalized)
    except ValueError as exc:
        valid = ", ".join(sorted({provider.value for provider in PROVIDER_LABELS}))
        raise ValueError(f"Unsupported provider '{value}'. Valid providers: {valid}") from exc
    return normalized


def run_models_list(args: Any, printer) -> int:
    """Handle ``kolega-code models list``."""
    provider = _resolve_provider(getattr(args, "provider", None))
    rows = catalog_rows(
        provider=provider,
        featured_only=bool(getattr(args, "featured", False)),
        sort=getattr(args, "sort", "popularity"),
    )
    if getattr(args, "json", False):
        printer(json.dumps(rows, indent=2))
    else:
        printer(format_rows(rows))
    return 0


def run_models_refresh(args: Any, printer, error_printer) -> int:
    """Handle ``kolega-code models refresh``."""
    provider = _resolve_provider(getattr(args, "provider", None)) or openrouter.PROVIDER
    if provider != openrouter.PROVIDER:
        error_printer(f"Only '{openrouter.PROVIDER}' has a refreshable catalog.")
        return 2

    printer(f"Fetching {openrouter.MODELS_URL} ...")
    try:
        payload = openrouter.fetch_models()
        entries = openrouter.catalog_entries(payload)
    except Exception as exc:  # noqa: BLE001 - surface the cause, whatever the transport raised
        error_printer(f"Could not refresh the OpenRouter catalog: {exc}")
        return 1

    bundled = {model for (provider_value, model) in MODEL_SPECS if provider_value == openrouter.PROVIDER}
    path = overlay_path(getattr(args, "state_dir", None))
    previously_cached = {identifier for identifier, _ in openrouter.load_cache(path)}
    new_models = [identifier for identifier, _ in entries if identifier not in bundled]

    openrouter.save_cache(path, entries, fetched_at=dt.datetime.now(dt.timezone.utc).isoformat())

    added = [identifier for identifier in new_models if identifier not in previously_cached]
    printer(
        f"Cached {len(entries)} models to {path}.\n"
        f"{len(new_models)} not in the bundled catalog ({len(added)} new since the last refresh)."
    )
    if added:
        printer("Newly available: " + ", ".join(sorted(added)[:20]) + ("..." if len(added) > 20 else ""))
    if overlay_disabled():
        printer(f"NOTE: {CATALOG_DISABLE_ENV} is set, so the cache is ignored at startup.")
    return 0


def known_openrouter_models() -> Iterable[str]:
    """Every OpenRouter model id currently resolvable (bundled plus overlay)."""
    return (model for (provider_value, model) in MODEL_SPECS if provider_value == openrouter.PROVIDER)
