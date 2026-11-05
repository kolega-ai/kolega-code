from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kolega_code.llm.specs import catalog
from kolega_code.llm.specs.catalog import MODEL_SPECS, WILDCARD_MODEL_SPECS
from kolega_code.llm.specs.catalog.pricing_sources import PROVIDER_SOURCES
from scripts.check_pricing_catalog import pricing_coverage


def test_every_bundled_route_has_valid_exact_pricing_or_a_sourced_exception() -> None:
    report = pricing_coverage(now=datetime(2026, 11, 1, tzinfo=timezone.utc))
    assert report["provider_count"] == 23
    assert report["model_count"] == len(MODEL_SPECS)
    assert report["errors"] == []
    assert report["valid"]


def test_pricing_is_defined_on_each_provider_catalogue_entry_not_attached_by_an_overlay() -> None:
    expected = {**MODEL_SPECS, **WILDCARD_MODEL_SPECS}
    found: dict[tuple[str, str], dict[str, Any]] = {}
    directory = Path(catalog.__file__).parent
    for provider in {provider for provider, _ in expected}:
        source = ast.parse((directory / f"{provider}.py").read_text())
        for node in ast.walk(source):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                if not isinstance(key, ast.Tuple) or not isinstance(value, (ast.Dict, ast.Call)):
                    continue
                identity = ast.literal_eval(key)
                if identity not in expected:
                    continue
                fields = (
                    {ast.literal_eval(name): field for name, field in zip(value.keys, value.values) if name}
                    if isinstance(value, ast.Dict)
                    else {keyword.arg: keyword.value for keyword in value.keywords}
                )
                assert "pricing" in fields, identity
                found[identity] = ast.literal_eval(fields["pricing"])
    assert found.keys() == expected.keys()
    assert all(found[key] == spec["pricing"] for key, spec in expected.items())


def test_new_bundled_model_without_a_pricing_policy_fails_audit(monkeypatch: Any) -> None:
    monkeypatch.setitem(MODEL_SPECS, ("openai", "unresearched-model"), {"context_length": 1000})
    report = pricing_coverage()
    assert not report["valid"]
    assert any(row["model"] == "unresearched-model" for row in report["errors"])


def test_another_route_card_cannot_cover_a_same_named_model(monkeypatch: Any) -> None:
    pricing = {
        "schema_version": 1,
        "pricing_id": "operator-example",
        "provider": "together",
        "model": "exact-model",
        "billing_mode": "pay_as_you_go",
        "status": "priced",
        "currency": "USD",
        "rates": {"input": "1", "output": "2"},
        "checked_at": "2026-11-01",
        "sources": [{"url": "https://www.together.ai/pricing", "model_id": "exact-model", "checked_at": "2026-11-01"}],
    }
    monkeypatch.setitem(MODEL_SPECS, ("openai", "exact-model"), {"context_length": 1000, "pricing": pricing})
    report = pricing_coverage()
    assert not report["valid"]
    assert any("identity differs" in row["error"] for row in report["errors"] if row["model"] == "exact-model")


def test_provider_without_selectable_models_still_requires_a_billing_policy(monkeypatch: Any) -> None:
    monkeypatch.delitem(PROVIDER_SOURCES, "llama")
    report = pricing_coverage()
    assert not report["valid"]
    assert any(row["provider"] == "llama" and "provider billing policy" in row["error"] for row in report["errors"])


def test_custom_routes_and_checkpoints_have_explicit_policies(monkeypatch: Any) -> None:
    monkeypatch.delitem(PROVIDER_SOURCES, "custom:*")
    monkeypatch.delitem(PROVIDER_SOURCES, "tinker://*")
    report = pricing_coverage()
    assert not report["valid"]
    assert {"custom:*", "tinker://*"} <= {row["provider"] for row in report["errors"]}


def test_unfinished_research_cannot_be_hidden_as_an_unverified_exception(monkeypatch: Any) -> None:
    model = "unresearched-model"
    monkeypatch.setitem(
        MODEL_SPECS,
        ("openai", model),
        {
            "context_length": 1000,
            "pricing": {
                "schema_version": 1,
                "pricing_id": "pending",
                "provider": "openai",
                "model": model,
                "status": "unverified",
                "billing_mode": "pay_as_you_go",
                "reason": "Research pending for this model.",
                "checked_at": "2026-11-01",
                "sources": [
                    {
                        "url": "https://developers.openai.com/api/docs/pricing",
                        "model_id": model,
                        "checked_at": "2026-11-01",
                    }
                ],
            },
        },
    )
    report = pricing_coverage()
    assert not report["valid"]
    assert any("unfinished" in row["error"] for row in report["errors"] if row["model"] == model)
