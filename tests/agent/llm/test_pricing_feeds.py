from __future__ import annotations

import importlib
import json
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from kolega_code.llm.pricing import PricingResolver, _match
from kolega_code.llm.specs import openrouter_catalog, perplexity_catalog, tinker_catalog
from kolega_code.llm.specs.catalog import MODEL_SPECS, WILDCARD_MODEL_SPECS
from kolega_code.llm.specs.catalog.pricing_sources import PROVIDER_SOURCES
from kolega_code.llm.specs.pricing import pricing_to_json
from kolega_code.llm.specs.pricing_feeds import feed_pricing
from kolega_code.llm.specs.types import ThinkingEffortSpec
from kolega_code.llm.usage import NormalizedUsage

PRICING = {key: spec["pricing"] for key, spec in {**MODEL_SPECS, **WILDCARD_MODEL_SPECS}.items()}
CATALOGS = ["openrouter", "tinker", "perplexity", "ollama_cloud", "cloudflare_workers_ai"]
DEFAULT_TIER_CASES = [
    ("minimax", "MiniMax-M3", "standard", "0.00042"),
    ("perplexity_agent", "openai/gpt-6.1-sol", "default", "0.003"),
    ("perplexity_agent", "openai/gpt-6-astra", "default", "0.015"),
    ("perplexity_agent", "openai/gpt-6-sol", "default", "0.003"),
    ("perplexity_agent", "openai/gpt-6-luna", "default", "0.00015"),
    ("perplexity_agent", "openai/gpt-5.6-sol", "default", "0.006"),
    ("perplexity_agent", "openai/gpt-5.6-terra", "default", "0.0032"),
    ("perplexity_agent", "openai/gpt-5.6-luna", "default", "0.00032"),
    ("perplexity_agent", "openai/gpt-5.5", "default", "0.008"),
]


def _estimate(
    provider: str,
    model: str,
    *,
    read: int | None = 0,
    write: int | None = 0,
    **context: Any,
) -> dict[str, Any]:
    usage = NormalizedUsage(
        provider=provider,
        model=model,
        input_tokens=1000,
        output_tokens=100,
        total_tokens=1100,
        cache_read_input_tokens=read,
        cache_write_input_tokens=write,
        reasoning_output_tokens=0,
        reported=True,
        unavailable_reason=None,
    )
    resolver = PricingResolver()
    pinned = resolver.pin(provider, model, {"requested_at": "2026-11-01T12:00:00Z", "tools": [], **context})
    return resolver.calculate(pinned, usage)


def _spec() -> dict[str, Any]:
    return {
        "context_length": 65536,
        "max_completion_tokens": 8192,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": False,
        "preferred_edit_protocol": "claude_code",
        "pricing": PRICING[("deepseek", "deepseek-flash")],
    }


def test_openrouter_decimal_conversion_preserves_structured_overrides() -> None:
    raw = {
        "prompt": "0.000000123456789",
        "completion": "0.000002",
        "input_cache_write_1h": "0.0000004",
        "overrides": [
            {"min_prompt_tokens": 200000, "prompt": "0.000000246913578", "completion": "0.000004"},
            {"utc_days": ["monday", "friday"], "utc_start": 1500, "utc_end": 100, "prompt": "0.0000001"},
        ],
    }
    card = feed_pricing("openrouter", "test/provider-model", {"pricing": raw})
    assert card["status"] == "priced"
    assert Decimal(card["rates"]["input"]) == Decimal("0.123456789")
    assert Decimal(card["rates"]["cache_write_1h"]) == Decimal("0.4")
    assert card["upstream_pricing"] == raw
    assert card["schedules"][0]["when"] == {"min_input_tokens": 200001}
    assert card["schedules"][1]["when"] == {
        "weekdays": [0, 4],
        "timezone": "UTC",
        "start_time": "15:00",
        "end_time": "01:00",
    }
    assert len(card["source_payload_sha256"]) == 64
    assert pricing_to_json(card) == card


@pytest.mark.parametrize("rate", [True, "-2", "NaN", "Infinity", "-Infinity", {}, [], None, "oops"])
def test_malformed_feed_pricing_does_not_drop_capabilities(rate: Any) -> None:
    entry = {
        "id": "test/new-model",
        "context_length": 65536,
        "supported_parameters": ["tools", "temperature"],
        "architecture": {"input_modalities": ["text"]},
        "pricing": {"prompt": rate, "completion": "0.000001"},
    }
    entries = openrouter_catalog.catalog_entries({"data": [entry]})
    assert len(entries) == 1
    assert entries[0][0] == "test/new-model"
    assert entries[0][1]["context_length"] == 65536
    assert entries[0][1]["pricing"]["status"] == "unverified"


def test_router_variant_zero_is_not_a_paid_sibling_or_metadata_router() -> None:
    free = feed_pricing("openrouter", "test/model:free", {"pricing": {"prompt": "0", "completion": "0"}})
    paid = feed_pricing("openrouter", "test/model", {"pricing": {"prompt": "0.000001", "completion": "0.000002"}})
    assert free["status"] == "priced"
    assert free["routing_variant"] == "free"
    assert set(free["rates"].values()) == {"0"}
    assert paid["status"] == "priced"
    assert free["verified_free"] is True
    assert paid["verified_free"] is False
    assert free["pricing_id"] != paid["pricing_id"]
    for extra in [{"web_search": "0.01"}, {"overrides": [{"min_prompt_tokens": 10, "prompt": "0.000001"}]}]:
        not_all_free = feed_pricing(
            "openrouter",
            "test/paid-components:free",
            {"pricing": {"prompt": "0", "completion": "0", **extra}},
        )
        assert not_all_free["status"] == "priced"
        assert not_all_free["verified_free"] is False


@pytest.mark.parametrize(
    "model",
    [model for (provider, model), card in PRICING.items() if provider == "openrouter" and "upstream_pricing" in card],
)
def test_every_published_openrouter_card_is_enabled_without_an_override(model: str) -> None:
    assert PRICING[("openrouter", model)]["status"] == "priced"


def test_openrouter_published_rates_produce_default_estimate_and_reported_cost_wins() -> None:
    model = "deepseek/deepseek-v4.1-flash-1017"
    cost = _estimate("openrouter", model)
    assert cost["source"] == "estimated"
    assert cost["pricing_source"] == "catalogue"
    assert cost["complete"] is True
    assert cost["amount"] == "0.00021"
    assert any("routed" in assumption for assumption in cost["assumptions"])
    for amount in ("0", "0.0042"):
        reported = PricingResolver().calculate(
            cost["pinned_pricing"], None, SimpleNamespace(cost={"amount": amount, "currency": "USD"})
        )
        assert reported["source"] == "provider_reported"
        assert reported["amount"] == amount
        assert reported["complete"] is True


@pytest.mark.parametrize(("tokens", "amount"), [(100, "0.0001"), (101, "0.000202"), (201, "0.000603")])
def test_openrouter_overrides_are_strict_thresholds_with_later_values_winning(
    monkeypatch: pytest.MonkeyPatch, tokens: int, amount: str
) -> None:
    model = "test/threshold"
    card = feed_pricing(
        "openrouter",
        model,
        {
            "pricing": {
                "prompt": "0.000001",
                "completion": "0.000002",
                "overrides": [
                    {"min_prompt_tokens": 100, "prompt": "0.000002"},
                    {"min_prompt_tokens": 200, "prompt": "0.000003"},
                ],
            }
        },
    )
    monkeypatch.setitem(MODEL_SPECS, ("openrouter", model), {"pricing": card})
    usage = NormalizedUsage(
        provider="openrouter",
        model=model,
        input_tokens=tokens,
        output_tokens=0,
        total_tokens=tokens,
        cache_read_input_tokens=0,
        cache_write_input_tokens=0,
        reasoning_output_tokens=0,
        reported=True,
        unavailable_reason=None,
    )
    resolver = PricingResolver()
    cost = resolver.calculate(resolver.pin("openrouter", model), usage)
    assert cost["complete"] is True
    assert cost["amount"] == amount


def test_openrouter_search_fee_uses_observed_count_and_disabled_tools_are_free(monkeypatch: pytest.MonkeyPatch) -> None:
    model = "test/search"
    card = feed_pricing(
        "openrouter", model, {"pricing": {"prompt": "0.000001", "completion": "0.000002", "web_search": "0.01"}}
    )
    monkeypatch.setitem(MODEL_SPECS, ("openrouter", model), {"pricing": card})
    assert _estimate("openrouter", model)["amount"] == "0.0012"
    assert _estimate("openrouter", model, tools=["web_search"])["complete"] is False
    cost = _estimate("openrouter", model, tools=["web_search"], pricing_quantities={"web_search_calls": 2})
    assert cost["complete"] is True
    assert cost["amount"] == "0.0212"


def test_openrouter_unmetered_extra_pricing_keeps_token_subtotal_without_claiming_a_complete_charge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = "test/multimodal"
    raw = {"prompt": "0.000001", "completion": "0.000002", "image": "0.01"}
    card = feed_pricing("openrouter", model, {"pricing": raw})
    monkeypatch.setitem(MODEL_SPECS, ("openrouter", model), {"pricing": card})
    cost = _estimate("openrouter", model)
    assert card["status"] == "priced"
    assert card["upstream_pricing"] == raw
    assert cost["source"] == "estimated"
    assert cost["complete"] is False
    assert cost["amount"] is None
    assert cost["known_subtotal"] == "0.0012"
    assert any("image" in reason for reason in cost["missing_components"])


@pytest.mark.parametrize("suffix", ["", ":free"])
def test_openrouter_override_only_audio_charge_is_not_silently_free(
    monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    model = f"test/audio{suffix}"
    card = feed_pricing(
        "openrouter",
        model,
        {"pricing": {"prompt": "0", "completion": "0", "overrides": [{"min_prompt_tokens": 100, "audio": "0.01"}]}},
    )
    monkeypatch.setitem(MODEL_SPECS, ("openrouter", model), {"pricing": card})
    cost = _estimate("openrouter", model)
    assert card["status"] == "priced"
    assert card["verified_free"] is False
    assert cost["source"] == "estimated"
    assert not cost["complete"]
    assert cost["amount"] is None
    assert cost["known_subtotal"] == "0"
    assert any("audio" in missing for missing in cost["missing_components"])


@pytest.mark.parametrize(
    ("model", "amount"),
    [("thinkingmachines/Inkling", "0.001405"), ("thinkingmachines/Inkling-Small", "0.00042")],
)
def test_thinking_machines_uses_published_serverless_rates_without_an_override(model: str, amount: str) -> None:
    cost = _estimate("thinking_machines", model)
    assert cost["source"] == "estimated"
    assert cost["pricing_source"] == "catalogue"
    assert cost["complete"] is True
    assert cost["amount"] == amount
    assert any(":peft:262144:sampling-nvfp4" in assumption for assumption in cost["assumptions"])


@pytest.mark.parametrize(
    ("model", "amount"), [("thinkingmachines/Inkling", "0.00099"), ("thinkingmachines/Inkling-Small", "0.0003")]
)
def test_thinking_machines_published_cache_discount_requires_observed_cache_counts(model: str, amount: str) -> None:
    cached = _estimate("thinking_machines", model, read=500)
    assert cached["complete"]
    assert cached["amount"] == amount
    unknown = _estimate("thinking_machines", model, read=None)
    assert unknown["source"] == "estimated"
    assert not unknown["complete"]
    assert unknown["amount"] is None
    assert any("cache_read" in missing for missing in unknown["missing_components"])


@pytest.mark.parametrize("catalog_name", CATALOGS)
def test_cache_roundtrip_preserves_pricing_and_accepts_old_capabilities(catalog_name: str, tmp_path: Path) -> None:
    module = importlib.import_module(f"kolega_code.llm.specs.{catalog_name}_catalog")
    spec = _spec()
    serialized = json.loads(json.dumps(module.spec_to_jsonable(spec)))
    assert module.spec_from_jsonable(serialized) == spec
    path = tmp_path / f"{catalog_name}.json"
    cache = module.AGENT_CATALOG if catalog_name == "perplexity" else module
    cache.save_cache(path, [("test/model", spec)], fetched_at="2026-11-01T00:00:00Z")
    assert cache.load_cache(path) == [("test/model", spec)]
    spec.pop("pricing")
    assert module.spec_from_jsonable(spec) == spec


@pytest.mark.parametrize("catalog_name", CATALOGS)
@pytest.mark.parametrize("bad_pricing", [None, {"schema_version": True}, {"rates": {"input": "NaN"}}])
def test_invalid_cached_pricing_is_removed_without_discarding_capabilities(
    catalog_name: str,
    bad_pricing: Any,
) -> None:
    module = importlib.import_module(f"kolega_code.llm.specs.{catalog_name}_catalog")
    spec = {**_spec(), "pricing": bad_pricing}
    loaded = module.spec_from_jsonable(spec)
    assert "pricing" not in loaded
    assert loaded["context_length"] == 65536
    assert "pricing" not in module.spec_to_jsonable(spec)
    assert spec["pricing"] == bad_pricing


def test_id_only_perplexity_refresh_preserves_reviewed_prices_and_check_date() -> None:
    model = "openai/gpt-6.1-sol"
    spec = dict(perplexity_catalog.catalog_entries({"data": [{"id": model}]}))[model]
    assert spec["pricing"] == PRICING[("perplexity_agent", model)]
    assert spec["pricing"]["status"] == "priced"
    unknown = perplexity_catalog.catalog_entries({"data": [{"id": "test/new-model"}]})[0][1]
    assert unknown["pricing"]["status"] == "unverified"


def test_explicit_feed_units_are_required_and_complete_perplexity_rows_survive() -> None:
    model = "test/new-model"
    rows = [{"id": model, "pricing": {"usd_per_1m_tokens": {"input": "0.123", "output": "1.5"}}}]
    card = perplexity_catalog.catalog_entries({"data": rows})[0][1]["pricing"]
    assert card["rates"] == {"input": "0.123", "output": "1.5"}
    rows[0]["pricing"] = {"input": "0.123", "output": "1.5"}
    assert perplexity_catalog.catalog_entries({"data": rows})[0][1]["pricing"]["status"] == "unverified"


def test_native_tinker_rates_retirements_promotions_and_wildcard_remain_distinct() -> None:
    native = PRICING[("tinker", "thinkingmachines/Inkling")]
    serverless = PRICING[("thinking_machines", "thinkingmachines/Inkling")]
    assert native["rates"]["input"] == "1.87"
    assert native["regular_rates"]["input"] == "3.74"
    assert serverless["rates"]["input"] == "1.00"
    assert serverless["status"] == "priced"
    wildcard = PRICING[("tinker", "tinker://*")]
    assert wildcard["status"] == "requires_configuration" and not wildcard["rates"]
    assert wildcard["checkpoint_policy"] == {"automatic_estimation": False, "inherit": "none"}
    retired = "Qwen/Qwen3.6-27B"
    payload = [
        {
            "tinker_id": retired,
            "type": "Hybrid",
            "context": "256K",
            "prefill": "$1.86",
            "cached_prefill": "$0.372",
            "sample": "$5.595",
        }
    ]
    card = tinker_catalog.catalog_entries(payload)[0][1]["pricing"]
    assert card["status"] == "retired" and card["effective_to"] == "2026-10-23"


@pytest.mark.parametrize(
    ("timestamp", "peak"),
    [
        ("2026-11-02T00:59:59Z", False),
        ("2026-11-02T01:00:00Z", True),
        ("2026-11-02T04:00:00Z", False),
        ("2026-11-02T06:00:00Z", True),
        ("2026-11-02T10:00:00Z", False),
        ("2026-10-01T01:00:00Z", False),
        ("2026-09-20T01:00:00Z", False),
    ],
)
def test_deepseek_beijing_peaks_respect_holidays_and_exclusive_ends(timestamp: str, peak: bool) -> None:
    card = PRICING[("deepseek", "deepseek-flash")]
    matched = [_match(schedule["when"], {"requested_at": timestamp}, {})[0] for schedule in card["schedules"]]
    assert any(matched) is peak


def test_deepseek_calendar_expiration_and_successful_tool_fee_transition() -> None:
    card = PRICING[("deepseek", "deepseek-v4-pro")]
    assert _match(card["conditions"], {"requested_at": "2027-01-04T01:00:00Z"}, {})[0] is None
    for timestamp, expected in [("2026-10-31T15:59:59Z", "0"), ("2026-10-31T16:00:00Z", "0.00075")]:
        selected = [
            component
            for component in card["components"]
            if _match(component["when"], {"requested_at": timestamp}, {})[0]
        ]
        assert len(selected) == 2
        assert {component["rate"] for component in selected} == {expected}
        assert {component["quantity"] for component in selected} == {
            "server_search_successful_requests",
            "server_fetch_successful_requests",
        }


def test_ollama_has_independent_utc_schedule_and_no_invented_cache_read_rate() -> None:
    card = PRICING[("ollama_cloud", "deepseek-v4-pro:0813")]
    when = card["schedules"][0]["when"]
    assert "holidays" not in when
    assert _match(when, {"requested_at": "2026-10-01T12:00:00Z"}, {})[0] is True
    assert _match(when, {"requested_at": "2026-11-02T18:00:00Z"}, {})[0] is False
    no_cache_rate = PRICING[("ollama_cloud", "mistral-large-3:675b")]
    assert "cache_read" not in no_cache_rate["rates"]
    assert "cache_read" not in no_cache_rate["cache_policy"]
    assert not no_cache_rate.get("missing_components")
    assert any("no documented rate" in assumption for assumption in no_cache_rate["assumptions"])


@pytest.mark.parametrize(("provider", "model", "tier", "amount"), DEFAULT_TIER_CASES)
def test_documented_default_tiers_produce_complete_ordinary_request_estimates(
    provider: str, model: str, tier: str, amount: str
) -> None:
    assert PRICING[(provider, model)]["default_context"] == {"service_tier": tier}
    cost = _estimate(provider, model)
    assert cost["source"] == "estimated"
    assert cost["complete"] is True
    assert cost["amount"] == amount
    assert cost["missing_components"] == []
    assert any("Documented default billing context" in assumption for assumption in cost["assumptions"])


@pytest.mark.parametrize(("provider", "model"), [(provider, model) for provider, model, _, _ in DEFAULT_TIER_CASES])
def test_unspecified_tier_without_a_documented_default_remains_a_gap(
    monkeypatch: pytest.MonkeyPatch, provider: str, model: str
) -> None:
    card = deepcopy(PRICING[(provider, model)])
    card.pop("default_context")
    monkeypatch.setitem(MODEL_SPECS, (provider, model), {"pricing": card})
    cost = _estimate(provider, model)
    assert cost["complete"] is False
    assert cost["amount"] is None
    assert any("service_tier" in reason for reason in cost["missing_components"])


@pytest.mark.parametrize(
    ("provider", "model", "tier", "amount"),
    [
        ("minimax", "MiniMax-M3", "standard", "0.00042"),
        ("minimax", "MiniMax-M3", "priority", "0.00063"),
        ("perplexity_agent", "openai/gpt-6.1-sol", "default", "0.003"),
        ("perplexity_agent", "openai/gpt-6.1-sol", "auto", "0.003"),
        ("perplexity_agent", "openai/gpt-6.1-sol", "flex", "0.0015"),
        ("perplexity_agent", "openai/gpt-6.1-sol", "priority", "0.006"),
        ("perplexity_agent", "openai/gpt-6.1-sol", "fast", "0.006"),
    ],
)
def test_explicit_service_tier_overrides_defaults_without_overlapping_bands(
    provider: str, model: str, tier: str, amount: str
) -> None:
    cost = _estimate(provider, model, service_tier=tier)
    assert cost["complete"] is True
    assert cost["amount"] == amount
    assert cost["missing_components"] == []
    assert not any("Documented default billing context" in assumption for assumption in cost["assumptions"])


@pytest.mark.parametrize(
    ("model", "amount"), [("mistral-large-3:675b", "0.00065"), ("nemotron-3-nano:30b", "0.000084")]
)
def test_ollama_unknown_cache_rate_is_not_needed_for_explicit_zero_cache(model: str, amount: str) -> None:
    cost = _estimate("ollama_cloud", model)
    assert cost["source"] == "estimated"
    assert cost["complete"] is True
    assert cost["amount"] == amount
    assert cost["missing_components"] == []


@pytest.mark.parametrize("model", ["mistral-large-3:675b", "nemotron-3-nano:30b"])
@pytest.mark.parametrize("read", [100, None])
def test_ollama_nonzero_or_unknown_unpriced_cache_reads_remain_partial(model: str, read: int | None) -> None:
    cost = _estimate("ollama_cloud", model, read=read)
    assert cost["source"] == "estimated"
    assert cost["complete"] is False
    assert cost["amount"] is None
    assert Decimal(cost["known_subtotal"]) > 0
    assert any("cache_read" in reason for reason in cost["missing_components"])


@pytest.mark.parametrize("provider,model", [("minimax", "MiniMax-M3"), ("perplexity_agent", "openai/gpt-6.1-sol")])
def test_documented_default_tier_does_not_fill_unpriced_cache_writes(provider: str, model: str) -> None:
    cost = _estimate(provider, model, write=100)
    assert cost["complete"] is False
    assert cost["amount"] is None
    assert any("cache_write" in reason for reason in cost["missing_components"])


@pytest.mark.parametrize(
    ("tokens", "tier", "input_rate"),
    [
        (512000, "standard", "0.30"),
        (512001, "standard", "0.60"),
        (512000, "priority", "0.45"),
        (512001, "priority", "0.90"),
    ],
)
def test_minimax_whole_request_context_and_priority_bands(tokens: int, tier: str, input_rate: str) -> None:
    card = PRICING[("minimax", "MiniMax-M3")]
    matches = [row for row in card["schedules"] if _match(row["when"], {"service_tier": tier}, {"input": tokens})[0]]
    assert matches[-1]["rates"]["input"] == input_rate
    assert all(row.get("mode", "whole_request") == "whole_request" for row in matches)


@pytest.mark.parametrize(
    ("tokens", "tier", "input_rate"),
    [(272000, "default", "2.00"), (272001, "default", "4.00"), (272000, "flex", "1.000"), (272001, "priority", "8.00")],
)
def test_perplexity_context_and_service_tiers_do_not_multiply_hosted_tools(
    tokens: int,
    tier: str,
    input_rate: str,
) -> None:
    card = PRICING[("perplexity_agent", "openai/gpt-6.1-sol")]
    matches = [row for row in card["schedules"] if _match(row["when"], {"service_tier": tier}, {"input": tokens})[0]]
    assert Decimal(matches[-1]["rates"]["input"]) == Decimal(input_rate)
    assert next(row["rate"] for row in card["components"] if row["name"] == "web_search") == "0.0025"
    assert not any(row.get("components") for row in matches)


@pytest.mark.parametrize(
    ("timestamp", "peak"),
    [
        ("2026-11-02T05:59:59Z", False),
        ("2026-11-02T06:00:00Z", True),
        ("2026-11-02T10:00:00Z", False),
        ("2026-11-01T06:00:00Z", False),
    ],
)
def test_coding_plan_credit_schedule_is_singapore_weekdays(timestamp: str, peak: bool) -> None:
    card = PRICING[("zai", "glm-5.3")]
    schedule = card["schedules"][0]
    assert _match(schedule["when"], {"requested_at": timestamp}, {})[0] is peak
    assert Decimal(card["rates"]["input"]) == Decimal("345")
    assert Decimal(schedule["rates"]["input"]) == Decimal("690")


def test_subscription_self_hosted_and_coding_credits_do_not_become_api_dollars() -> None:
    assert PROVIDER_SOURCES["llama"]["billing_surface"] == "self_hosted"
    for (provider, _), card in PRICING.items():
        if provider == "kimi_coding":
            assert card["status"] == "subscription" and not card["rates"]
            assert card.get("currency") is None
        elif provider == "zai":
            assert card["currency"] is None
            assert card["denomination"] == "zai_coding_plan_credit"
            assert card["alternative_billing_surface"]["currency"] == "USD"
    assert PRICING[("zai", "glm-5.2")]["documented_alias"]["billing_model"] == "GLM-5.3"


def test_all_four_generators_preserve_json_safe_pricing_and_tinker_singleton_effort() -> None:
    openrouter = importlib.import_module("scripts.refresh_openrouter_catalog")
    ollama = importlib.import_module("scripts.refresh_ollama_cloud_catalog")
    tinker = importlib.import_module("scripts.refresh_tinker_catalog")
    perplexity = importlib.import_module("scripts.refresh_perplexity_catalog")
    spec = _spec()
    entries = [("test/model", spec)]
    sources = [
        openrouter.catalog.render_catalog_module(entries, header_lines=[]),
        ollama.catalog.render_catalog_module(entries, header_lines=[]),
        perplexity._render_catalog(provider="perplexity_agent", variable="SPECS", entries=entries, header_lines=[]),
    ]
    effort = ThinkingEffortSpec(options=("high",), default="high", mode="tinker_native_effort")
    sources.append(
        tinker._SNAPSHOT_TEMPLATE.format(
            generator="test",
            source="https://tinker-docs.thinkingmachines.ai/tinker/models.json",
            date="2026-11-01",
            count=1,
            specs=tinker._format_spec("test/model", {**spec, "thinking_effort": effort}),
            wildcard_pricing=repr(PRICING[("tinker", "tinker://*")]),
        )
    )
    for source in sources:
        namespace: dict[str, Any] = {}
        exec(compile(source, "<offline-generated-pricing>", "exec"), namespace)
        generated = next(
            value
            for value in namespace.values()
            if isinstance(value, dict) and any(isinstance(k, tuple) and k[1] == "test/model" for k in value)
        )
        assert next(iter(generated.values()))["pricing"] == spec["pricing"]
    assert namespace["TINKER_SPECS"][("tinker", "test/model")]["thinking_effort"].options == ("high",)
    assert namespace["TINKER_WILDCARD_SPECS"][("tinker", "tinker://*")]["pricing"]["status"] == "requires_configuration"
