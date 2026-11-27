"""Catalog conversion, bundled invariants, and runtime overlay tests for Vercel AI Gateway."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from kolega_code.cli.model_catalog import apply_catalog_overlay, overlay_path, run_models_refresh
from kolega_code.cli.provider_registry import (
    PROVIDER_DEFAULT_MODEL,
    PROVIDER_LABELS,
    default_model_for_provider,
    get_ui_model,
    rebuild_ui_model_options,
    ui_model_options,
)
from kolega_code.cli.tui.custom_model import CUSTOM_MODEL_SENTINEL, resolve_custom_model, settings_model_options
from kolega_code.config import ModelProvider
from kolega_code.llm.specs import MODEL_SPECS, is_featured_model
from kolega_code.llm.specs import vercel_ai_gateway_catalog as catalog
from kolega_code.llm.specs.types import ThinkingEffortSpec

PROVIDER = catalog.PROVIDER


def _raw_entry(
    model_id: str,
    *,
    model_type: str = "language",
    context_window: int = 128000,
    max_tokens: int = 16384,
    supported_parameters: list[str] | None = None,
    input_modalities: list[str] | None = None,
    temperature: bool | None = True,
    reasoning_options: list[dict[str, Any]] | None = None,
    pricing: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": model_id,
        "type": model_type,
        "context_window": context_window,
        "max_tokens": max_tokens,
        "supported_parameters": (
            supported_parameters if supported_parameters is not None else ["tools", "temperature", "max_tokens"]
        ),
        "modalities": {
            "input": input_modalities if input_modalities is not None else ["text"],
            "output": ["text"],
        },
        "pricing": pricing if pricing is not None else {"input": "0.000001", "output": "0.000002"},
    }
    if temperature is not None:
        entry["temperature"] = temperature
    if reasoning_options is not None:
        entry["reasoning_options"] = reasoning_options
    if tags is not None:
        entry["tags"] = tags
    return entry


def test_is_catalogable_filters_non_language_non_tool_and_batch_models() -> None:
    assert catalog.is_catalogable(_raw_entry("vendor/chat-model")) is True
    assert catalog.is_catalogable(_raw_entry("vendor/chat-model-batch")) is False
    assert catalog.is_catalogable(_raw_entry("vendor/embed", model_type="embedding")) is False
    assert catalog.is_catalogable(_raw_entry("vendor/no-tools", supported_parameters=["temperature"])) is False
    assert catalog.is_catalogable(_raw_entry("vendor/image-only", input_modalities=["image"])) is False
    assert catalog.is_catalogable(_raw_entry("vendor/zero-ctx", context_window=0)) is False
    assert catalog.is_catalogable(_raw_entry("vendor/zero-max", max_tokens=0)) is False


def test_reasoning_options_conversion_handles_efforts_toggles_and_unknown_values() -> None:
    # Effort + toggle without explicit "none" prepends "none" and drops non-canonical "instant".
    spec_mixed = catalog.model_spec(
        _raw_entry(
            "inception/mercury-test",
            reasoning_options=[
                {"type": "toggle"},
                {"type": "effort", "values": ["instant", "low", "medium", "high"]},
            ],
        )
    )
    assert spec_mixed["thinking_effort"] == ThinkingEffortSpec(
        options=("none", "low", "medium", "high"),
        default="medium",
        mode="vercel_ai_gateway_reasoning",
    )

    # Toggle-only model exposes none/enabled.
    spec_toggle = catalog.model_spec(
        _raw_entry(
            "alibaba/qwen-toggle",
            reasoning_options=[{"type": "toggle"}, {"type": "budget_tokens", "min": 1024}],
        )
    )
    assert spec_toggle["thinking_effort"] == ThinkingEffortSpec(
        options=("none", "enabled"),
        default="enabled",
        mode="vercel_ai_gateway_reasoning",
    )

    # Mandatory effort-only model does not fabricate "none".
    spec_effort = catalog.model_spec(
        _raw_entry(
            "vendor/mandatory-reasoning",
            reasoning_options=[{"type": "effort", "values": ["low", "high", "max"]}],
        )
    )
    assert spec_effort["thinking_effort"] == ThinkingEffortSpec(
        options=("low", "high", "max"),
        default="low",
        mode="vercel_ai_gateway_reasoning",
    )


def test_model_spec_maps_temperature_vision_edit_protocol_and_anthropic_drop() -> None:
    openai_spec = catalog.model_spec(
        _raw_entry(
            "openai/gpt-test",
            context_window=400000,
            max_tokens=128000,
            supported_parameters=["tools", "max_tokens"],
            input_modalities=["text", "image"],
            temperature=False,
        )
    )
    assert openai_spec["supports_temperature"] is False
    assert openai_spec["supports_vision"] is True
    assert openai_spec["preferred_edit_protocol"] == "codex_apply_patch"
    assert openai_spec["input_budget"] == "window_minus_output"
    assert "drop_prior_reasoning" not in openai_spec

    anthropic_spec = catalog.model_spec(
        _raw_entry(
            "anthropic/claude-test",
            context_window=1000000,
            max_tokens=1000000,
            input_modalities=["text"],
        )
    )
    assert anthropic_spec["drop_prior_reasoning"] is True
    assert anthropic_spec["preferred_edit_protocol"] == "claude_code"
    assert anthropic_spec["input_budget"] == "output_shares_window"
    assert anthropic_spec["supports_vision"] is False


def test_bundled_catalog_invariants_and_picker_order() -> None:
    keys = [key for key in MODEL_SPECS if key[0] == PROVIDER]
    assert len(keys) >= 100
    assert PROVIDER_LABELS[ModelProvider.VERCEL_AI_GATEWAY] == "Vercel AI Gateway"

    default_model = PROVIDER_DEFAULT_MODEL[ModelProvider.VERCEL_AI_GATEWAY]
    assert default_model_for_provider(ModelProvider.VERCEL_AI_GATEWAY) == default_model
    assert (PROVIDER, default_model) in MODEL_SPECS
    assert is_featured_model(PROVIDER, default_model)

    picker_options = ui_model_options(PROVIDER)
    assert picker_options
    assert picker_options[0][1] == default_model

    settings_options = settings_model_options(PROVIDER)
    assert settings_options[-1][1] == CUSTOM_MODEL_SENTINEL

    non_featured = next(model for _, model in keys if not is_featured_model(PROVIDER, model))
    assert get_ui_model(PROVIDER, non_featured) is not None
    assert resolve_custom_model(PROVIDER, non_featured) == non_featured

    for _, model in keys:
        spec = MODEL_SPECS[(PROVIDER, model)]
        assert not model.endswith("-batch")
        if model.startswith("anthropic/"):
            assert spec.get("drop_prior_reasoning") is True
        if model.startswith("openai/"):
            assert spec.get("preferred_edit_protocol") == "codex_apply_patch"


def test_cache_roundtrip_and_runtime_refresh_overlay(
    isolated_cli_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    before = copy.copy(MODEL_SPECS)
    try:
        bundled = next(model for provider, model in MODEL_SPECS if provider == PROVIDER)
        original = dict(MODEL_SPECS[(PROVIDER, bundled)])

        payload = {
            "data": [
                _raw_entry(
                    "vendor/brand-new-model",
                    context_window=200000,
                    max_tokens=32768,
                    input_modalities=["text", "image"],
                    reasoning_options=[{"type": "effort", "values": ["low", "high"]}],
                ),
                _raw_entry(
                    bundled,
                    context_window=999,
                    max_tokens=123,
                ),
            ]
        }
        monkeypatch.setattr(catalog, "fetch_models", lambda **kwargs: payload)

        messages: list[str] = []
        errors: list[str] = []
        state_dir = tmp_path / "state"
        exit_code = run_models_refresh(
            SimpleNamespace(provider=PROVIDER, state_dir=state_dir),
            messages.append,
            errors.append,
        )
        assert exit_code == 0, errors
        cache_file = overlay_path(state_dir, catalog=PROVIDER)
        assert cache_file.exists()

        added = apply_catalog_overlay(state_dir, catalog=PROVIDER)
        assert added == 1
        assert (PROVIDER, "vendor/brand-new-model") in MODEL_SPECS
        new_spec = MODEL_SPECS[(PROVIDER, "vendor/brand-new-model")]
        assert new_spec["context_length"] == 200000
        assert new_spec["supports_vision"] is True
        assert new_spec.get("featured", False) is False
        assert new_spec["thinking_effort"] == ThinkingEffortSpec(
            options=("low", "high"),
            default="low",
            mode="vercel_ai_gateway_reasoning",
        )
        # Bundled capabilities always win over cached overlay capabilities.
        assert MODEL_SPECS[(PROVIDER, bundled)]["context_length"] == original["context_length"]
    finally:
        MODEL_SPECS.clear()
        MODEL_SPECS.update(before)
        rebuild_ui_model_options()
