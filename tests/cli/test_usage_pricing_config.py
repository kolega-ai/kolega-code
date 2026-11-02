import argparse
import copy
import json
from pathlib import Path

import pytest

from kolega_code.cli import model_catalog
from kolega_code.cli.main import _build_subcommand_parser
from kolega_code.cli.settings import CliSettings
from kolega_code.llm.specs.catalog import MODEL_SPECS
from kolega_code.llm.specs.accessors import get_model_pricing


def pricing(provider: str, model: str, checked_at: str = "2026-11-01") -> dict:
    return {
        "schema_version": 1,
        "pricing_id": "synthetic-v1",
        "provider": provider,
        "model": model,
        "status": "priced",
        "billing_mode": "api",
        "currency": "USD",
        "checked_at": checked_at,
        "rates": {"input": "1", "output": "2"},
        "sources": [{"url": "https://example.org/prices", "model_id": model}],
    }


def test_usage_pricing_optional_settings_roundtrip() -> None:
    settings = CliSettings()
    assert CliSettings.from_dict(settings.to_dict()).usage_pricing == {}
    settings.usage_pricing = {"custom:example/model": False, "openai/example": {"rates": {"input": "bad"}}}
    assert CliSettings.from_dict(settings.to_dict()).usage_pricing == settings.usage_pricing
    settings_data = settings.to_dict()
    settings_data["usage_pricing"] = ["not", "a", "mapping"]
    assert CliSettings.from_dict(settings_data).usage_pricing == {}


def test_model_pricing_flag_parses() -> None:
    args = _build_subcommand_parser().parse_args(["models", "list", "--pricing"])
    assert args.pricing is True


def test_human_pricing_is_opt_in_and_json_additive(monkeypatch: pytest.MonkeyPatch) -> None:
    key = ("openrouter", "test/pricing")
    spec = {
        "context_length": 100,
        "max_completion_tokens": 10,
        "input_budget": "window_minus_output",
        "pricing": pricing(*key),
    }
    monkeypatch.setitem(MODEL_SPECS, key, spec)
    rows = model_catalog.catalog_rows(provider=key[0])
    selected = [row for row in rows if row["model"] == key[1]]
    assert selected[0]["pricing"] == spec["pricing"]
    ordinary = model_catalog.format_rows(selected)
    detailed = model_catalog.format_rows(selected, pricing=True)
    assert "synthetic-v1" not in ordinary
    assert "synthetic-v1" in detailed and "https://example.org/prices" in detailed
    lines: list[str] = []
    model_catalog.run_models_list(
        argparse.Namespace(provider=key[0], featured=False, sort="popularity", json=True), lines.append
    )
    assert any(row["pricing"] for row in json.loads("\n".join(lines)))


@pytest.mark.parametrize("checked_at,expected", [("2026-10-31", "1"), ("2026-11-02", "9")])
def test_existing_overlay_only_replaces_newer_pricing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checked_at: str, expected: str
) -> None:
    key = ("openrouter", "test/pricing")
    original = {
        "context_length": 100,
        "max_completion_tokens": 10,
        "input_budget": "window_minus_output",
        "featured": True,
        "pricing": pricing(*key),
    }
    monkeypatch.setitem(MODEL_SPECS, key, original)
    newer = pricing(*key, checked_at)
    newer["rates"]["input"] = "9"
    module = model_catalog.OVERLAY_SOURCES[key[0]]
    monkeypatch.setattr(module, "load_cache", lambda _: [(key[1], {"context_length": 999, "pricing": newer})])
    assert model_catalog.apply_catalog_overlay(tmp_path, env={}, catalog=key[0]) == 0
    assert MODEL_SPECS[key]["context_length"] == 100
    assert MODEL_SPECS[key]["featured"] is True
    selected = get_model_pricing(*key)
    assert selected is not None
    assert selected["rates"]["input"] == expected
    assert list(MODEL_SPECS).index(key) == len(MODEL_SPECS) - 1


def test_invalid_overlay_price_preserves_new_model_capabilities(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    key = ("openrouter", "test/invalid-price")
    spec = {
        "context_length": 100,
        "max_completion_tokens": 10,
        "input_budget": "window_minus_output",
        "pricing": pricing(*key),
    }
    spec["pricing"]["rates"]["input"] = "NaN"
    module = model_catalog.OVERLAY_SOURCES[key[0]]
    monkeypatch.setattr(module, "load_cache", lambda _: [(key[1], spec)])
    monkeypatch.setattr(model_catalog, "rebuild_ui_model_options", lambda: None)
    before = copy.copy(MODEL_SPECS)
    try:
        assert model_catalog.apply_catalog_overlay(tmp_path, env={}, catalog=key[0]) == 1
        assert MODEL_SPECS[key]["context_length"] == 100
        assert get_model_pricing(*key) is None
    finally:
        MODEL_SPECS.clear()
        MODEL_SPECS.update(before)
