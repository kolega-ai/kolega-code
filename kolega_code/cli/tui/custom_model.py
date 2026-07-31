"""Settings-only "Other…" model picker affordance for gateway providers.

The Settings screen (main dropdown and every per-agent role row) shows only a
gateway provider's featured models so each picker stays short, but OpenRouter
catalogs hundreds of tool-capable models beyond that curated head. This module
owns the UI-side escape hatch: a trailing "Other…" select option backed by a
free-text input for any catalogued model id.

Everything here is deliberately a UI concern — ``kolega_code/cli/provider_registry.py``
(the shared catalog/config layer) is untouched, and the sentinel value is never
persisted: settings.json always stores the real model id.
"""

from __future__ import annotations

from kolega_code.cli.provider_registry import (
    UI_MODEL_OPTIONS,
    get_ui_model,
    provider_has_featured_models,
    ui_model_options,
)

# Sentinel option value for the "Other…" picker entry. Every real catalogued
# model id contains a '/', so this cannot collide with any of them.
CUSTOM_MODEL_LABEL = "Other…"
CUSTOM_MODEL_SENTINEL = "__custom_model__"

# Placeholder shown in every custom-model input on the Settings screen.
CUSTOM_MODEL_PLACEHOLDER = "Custom model id, e.g. anthropic/claude-sonnet-4.5"


def settings_model_options(provider: str, *, vision_only: bool = False) -> list[tuple[str, str]]:
    """Return model Select options for the Settings pickers, with "Other…".

    Delegates to ``provider_registry.ui_model_options`` unchanged, then appends
    the "Other…" entry when the provider is a gateway (it marks some models
    featured) and catalogs at least one non-featured model matching the vision
    filter. For every other provider the output is byte-identical to
    ``ui_model_options``.
    """
    options = ui_model_options(provider, vision_only=vision_only)
    has_non_featured = any(
        option.provider == provider and not option.featured and (option.supports_vision or not vision_only)
        for option in UI_MODEL_OPTIONS
    )
    if provider_has_featured_models(provider) and has_non_featured:
        options.append((CUSTOM_MODEL_LABEL, CUSTOM_MODEL_SENTINEL))
    return options


def resolve_custom_model(provider: str, value: str) -> str | None:
    """Return ``value`` (whitespace-trimmed) iff it names a catalogued model.

    Exact-match semantics shared with the ``/model`` fallback and the
    ``build_agent_config`` gate: anything not in the merged catalog (bundled
    snapshot plus any ``kolega-code models refresh`` overlay) resolves to
    ``None``.
    """
    cleaned = value.strip()
    if not cleaned:
        return None
    return cleaned if get_ui_model(provider, cleaned) is not None else None
