import copy
import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any

import pytest

from kolega_code.llm.pricing import PricingResolver, usage_cost_summary
from kolega_code.llm.specs.catalog import MODEL_SPECS
from kolega_code.llm.specs.pricing import PricingCard, pricing_is_newer, validate_pricing
from kolega_code.llm.usage import NormalizedUsage, normalize_usage
from kolega_code.llm.ledger import UsageLedger


def card(**kwargs: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "pricing_id": "test-v1",
        "provider": "openai",
        "model": "test",
        "status": "priced",
        "billing_mode": "api",
        "currency": "USD",
        "unit": "per_million_tokens",
        "rates": {"input": "2", "output": "8", "cache_read": "0.5", "cache_write": "3"},
        "checked_at": "2026-11-01",
        "sources": [{"url": "https://example.org/pricing", "model_id": "test", "checked_at": "2026-11-01"}],
        **kwargs,
    }


@pytest.fixture
def catalog(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    value = card()
    monkeypatch.setitem(MODEL_SPECS, ("openai", "test"), {"pricing": value})
    return value


def usage(input: int = 100, output: int = 10, read: int = 0, write: int = 0) -> NormalizedUsage:
    return normalize_usage(
        {
            "prompt_tokens": input - write,
            "completion_tokens": output,
            "cache_read_input_tokens": read,
            "cache_write_input_tokens": write,
        },
        "openai",
        "test",
    )


def pin(resolver: PricingResolver | None = None, **context: Any) -> dict[str, Any]:
    return (resolver or PricingResolver()).pin("openai", "test", {"requested_at": "2026-11-01T00:00:00Z", **context})


@pytest.mark.parametrize("rate", [True, -1, 1.5, "-1", "NaN", "Infinity", "oops"])
def test_rejects_invalid_rates(rate: Any) -> None:
    value = card()
    value["rates"]["input"] = rate
    with pytest.raises(ValueError):
        validate_pricing(value)


def test_immutable_record_and_json_snapshot(catalog: dict[str, Any]) -> None:
    immutable = PricingCard.from_dict(catalog)
    with pytest.raises(FrozenInstanceError):
        setattr(immutable, "serialized", "{}")
    serialized = immutable.to_dict()
    serialized["rates"]["input"] = "999"
    assert immutable.to_dict()["rates"]["input"] == "2"
    resolver = PricingResolver()
    pinned = pin(resolver)
    catalog["rates"]["input"] = "999"
    result = resolver.calculate(pinned, usage())
    assert result["amount"] == "0.00028"
    assert json.loads(json.dumps(result)) == result
    pinned["pricing"]["rates"]["input"] = "111"
    assert result["pinned_pricing"]["pricing"]["rates"]["input"] == "2"


def test_inclusive_cache_reasoning_not_double_billed(catalog: dict[str, Any]) -> None:
    result = PricingResolver().calculate(pin(), usage(input=100, output=20, read=30, write=10))
    assert result["amount"] == "0.000325"
    assert result["complete"] is True


def test_missing_cache_details_are_not_zero(catalog: dict[str, Any]) -> None:
    actual = normalize_usage({"prompt_tokens": 100, "completion_tokens": 10}, "openai", "test")
    result = PricingResolver().calculate(pin(), actual)
    assert result["complete"] is False
    assert result["amount"] is None
    assert result["known_subtotal"] == "0.00008"


def test_explicit_unsupported_cache_needs_no_counts(catalog: dict[str, Any]) -> None:
    catalog["cache_policy"] = {"cache_read": "unsupported", "cache_write": "unsupported"}
    catalog["rates"] = {"input": "2", "output": "8"}
    actual = normalize_usage({"prompt_tokens": 100, "completion_tokens": 10}, "openai", "test")
    assert PricingResolver().calculate(pin(), actual)["amount"] == "0.00028"


def test_valid_reported_zero_beats_disable_override(catalog: dict[str, Any]) -> None:
    resolver = PricingResolver({"openai/test": False})
    result = resolver.calculate(pin(resolver), None, SimpleNamespace(cost={"amount": "0", "currency": "USD"}))
    assert result["source"] == "provider_reported"
    assert result["amount"] == "0"
    assert result["complete"] is True


@pytest.mark.parametrize("amount", [True, -1, "-1", "NaN", "Infinity", None, 1.25])
def test_invalid_reported_cost_falls_back(catalog: dict[str, Any], amount: Any) -> None:
    result = PricingResolver().calculate(pin(), usage(), SimpleNamespace(cost={"amount": amount, "currency": "USD"}))
    assert result["source"] == "estimated"
    assert result["amount"] == "0.00028"


def test_exact_override_and_invalid_fallback(catalog: dict[str, Any], caplog: pytest.LogCaptureFixture) -> None:
    resolver = PricingResolver(
        {
            "openai/test": {
                "currency": "USD",
                "rates": {"input": "1", "output": "1"},
                "cache_policy": {"cache_read": "included", "cache_write": "included"},
            }
        }
    )
    result = resolver.calculate(pin(resolver), usage())
    assert result["amount"] == "0.00011"
    assert result["pricing_source"] == "override"
    resolver = PricingResolver({"openai/test": {"currency": "USD", "rates": {"input": "NaN"}}})
    result = resolver.calculate(pin(resolver), usage())
    assert result["pricing_source"] == "catalogue"
    assert result["amount"] == "0.00028"
    assert "Invalid usage_pricing" in caplog.text


def test_disable_exact_only(catalog: dict[str, Any]) -> None:
    resolver = PricingResolver({"openai/test": False})
    assert resolver.calculate(pin(resolver), usage())["source"] == "unavailable"
    assert resolver.pin("custom:local", "test")["reason"].startswith("custom endpoint")


def test_tier_and_region_conditions(catalog: dict[str, Any]) -> None:
    catalog["conditions"] = {"service_tier": "standard", "region": "us"}
    resolver = PricingResolver()
    assert not resolver.calculate(pin(), usage())["complete"]
    assert resolver.calculate(pin(region="eu", service_tier="standard"), usage())["source"] == "unavailable"
    assert resolver.calculate(pin(region="us", service_tier="standard"), usage())["complete"]


@pytest.mark.parametrize("count,expected", [(100, "0.0002"), (101, "0.000404")])
def test_whole_request_length_boundary(catalog: dict[str, Any], count: int, expected: str) -> None:
    catalog["schedules"] = [{"when": {"min_input_tokens": 101}, "rates": {"input": "4", "output": "16"}}]
    assert PricingResolver().calculate(pin(), usage(input=count, output=0))["amount"] == expected


def test_marginal_bands(catalog: dict[str, Any]) -> None:
    catalog["schedules"] = [
        {"mode": "marginal", "min_tokens": 0, "max_tokens": 100, "rates": {"input": "2"}},
        {"mode": "marginal", "min_tokens": 100, "rates": {"input": "4"}},
    ]
    assert PricingResolver().calculate(pin(), usage(input=150, output=0))["amount"] == "0.0004"


def test_supplemental_cache_write(catalog: dict[str, Any]) -> None:
    catalog["cache_policy"] = {"cache_write": "supplemental"}
    assert PricingResolver().calculate(pin(), usage(input=100, output=0, write=10))["amount"] == "0.00023"


def test_cache_ttl_splits(catalog: dict[str, Any]) -> None:
    catalog["rates"].pop("cache_write")
    catalog["rates"].update(cache_write_5m="3", cache_write_1h="4")
    observed = SimpleNamespace(pricing_quantities={"cache_write_5m": 5, "cache_write_1h": 5})
    result = PricingResolver().calculate(pin(), usage(input=100, output=0, write=10), observed)
    assert result["amount"] == "0.000215"
    assert result["complete"] is True


def test_tools_effective_dated_transition(catalog: dict[str, Any]) -> None:
    catalog["schedules"] = [
        {
            "when": {"effective_to": "2026-11-01T00:00:00+08:00"},
            "components": [{"name": "search", "quantity": "search_calls", "rate": "0"}],
        },
        {
            "when": {"effective_from": "2026-11-01T00:00:00+08:00"},
            "components": [{"name": "search", "quantity": "search_calls", "rate": "0.01"}],
        },
    ]
    resolver = PricingResolver()
    observed = SimpleNamespace(pricing_quantities={"search_calls": 2})
    before = resolver.pin("openai", "test", {"requested_at": "2026-10-31T15:59:59Z"})
    after = resolver.pin("openai", "test", {"requested_at": "2026-10-31T16:00:00Z"})
    assert resolver.calculate(before, usage(input=0, output=0), observed)["amount"] == "0"
    assert resolver.calculate(after, usage(input=0, output=0), observed)["amount"] == "0.02"


def test_tool_unknown_retains_token_subtotal(catalog: dict[str, Any]) -> None:
    catalog["components"] = [{"name": "search", "quantity": "search_calls", "rate": "0.01"}]
    result = PricingResolver().calculate(pin(), usage())
    assert result["amount"] is None
    assert result["token_cost_subtotal"] == "0.00028"
    assert result["missing_components"] == ["search: quantity search_calls unavailable"]


@pytest.mark.parametrize(
    "timestamp,expected",
    [
        ("2026-11-02T15:59:59Z", "0.0002"),
        ("2026-11-02T16:00:00Z", "0.0001"),
        ("2026-11-03T15:59:59Z", "0.0001"),
        ("2026-11-03T16:00:00Z", "0.0002"),
    ],
)
def test_timezone_weekday_and_holiday(catalog: dict[str, Any], timestamp: str, expected: str) -> None:
    catalog["schedules"] = [
        {"when": {"timezone": "Asia/Shanghai", "weekdays": [6], "holidays": ["2026-11-03"]}, "rates": {"input": "1"}}
    ]
    pinned = PricingResolver().pin("openai", "test", {"requested_at": timestamp})
    assert PricingResolver().calculate(pinned, usage(input=100, output=0))["amount"] == expected


@pytest.mark.parametrize(
    "time,expected", [("21:59:59", "0.0002"), ("22:00:00", "0.0001"), ("04:59:59", "0.0001"), ("05:00:00", "0.0002")]
)
def test_cross_midnight_schedule(catalog: dict[str, Any], time: str, expected: str) -> None:
    catalog["schedules"] = [{"when": {"start_time": "22:00", "end_time": "05:00"}, "rates": {"input": "1"}}]
    pinned = PricingResolver().pin("openai", "test", {"requested_at": f"2026-11-01T{time}Z"})
    assert PricingResolver().calculate(pinned, usage(input=100, output=0))["amount"] == expected


def test_expired_promotion_and_stale_review(catalog: dict[str, Any]) -> None:
    catalog["checked_at"] = "2026-09-01"
    pinned = pin()
    assert pinned["review_due"]
    assert PricingResolver().calculate(pinned, usage())["amount"] == "0.00028"
    catalog["effective_to"] = "2026-11-01"
    assert PricingResolver().calculate(pin(), usage())["source"] == "unavailable"
    catalog["regular_pricing"] = card(pricing_id="regular")
    assert PricingResolver().calculate(pin(), usage())["amount"] == "0.00028"


def test_credit_denomination_is_not_currency(catalog: dict[str, Any]) -> None:
    catalog.pop("currency")
    catalog["denomination"] = "provider_credits"
    result = PricingResolver().calculate(pin(), usage())
    assert result["currency"] is None
    assert result["denomination"] == "provider_credits"


@pytest.mark.parametrize("context", [{}, {"base_model": "base"}, {"billing_model": "base"}, {"base_model": "unknown"}])
def test_native_checkpoint_has_no_catalogue_estimate(monkeypatch: pytest.MonkeyPatch, context: dict[str, Any]) -> None:
    base = card(provider="tinker", model="base")
    monkeypatch.setitem(MODEL_SPECS, ("tinker", "base"), {"pricing": base})
    resolver = PricingResolver()
    pinned = resolver.pin("tinker", "tinker://private/checkpoint", context)
    assert not pinned["pricing"]
    result = resolver.calculate(pinned, usage())
    assert result["source"] == "unavailable"
    assert result["amount"] is None
    assert result["missing_components"] == ["saved Tinker checkpoints have no automatic catalogue pricing"]
    native = resolver.calculate(resolver.pin("tinker", "base"), usage())
    assert native["amount"] == "0.00028"
    assert native["pricing_source"] == "catalogue"
    assert resolver.pin("thinking_machines", "base")["pricing"] == {}


@pytest.mark.parametrize("context", [{}, {"base_model": "base"}])
def test_explicit_checkpoint_override_does_not_inherit_base_pricing(
    monkeypatch: pytest.MonkeyPatch, context: dict[str, Any]
) -> None:
    model = "tinker://checkpoint/run"
    monkeypatch.setitem(MODEL_SPECS, ("tinker", "base"), {"pricing": card(provider="tinker", model="base")})
    resolver = PricingResolver(
        {
            "tinker": {
                model: {
                    "currency": "USD",
                    "rates": {"input": "2", "output": "8"},
                    "cache_policy": {"cache_read": "included", "cache_write": "included"},
                }
            }
        }
    )
    pinned = resolver.pin("tinker", model, context)
    assert "catalogue_fallback" not in pinned
    result = resolver.calculate(pinned, usage())
    assert result["amount"] == "0.00028"
    assert result["pricing_source"] == "override"


@pytest.mark.parametrize("amount", ["0", "0.25"])
def test_checkpoint_provider_reported_charge_remains_authoritative(amount: str) -> None:
    resolver = PricingResolver()
    pinned = resolver.pin("tinker", "tinker://checkpoint/run")
    result = resolver.calculate(pinned, None, SimpleNamespace(cost={"amount": amount, "currency": "USD"}))
    assert result["source"] == "provider_reported"
    assert result["amount"] == amount
    assert result["complete"] is True


def test_inapplicable_required_override_schedule_uses_catalogue(catalog: dict[str, Any]) -> None:
    resolver = PricingResolver(
        {
            "openai/test": {
                "currency": "USD",
                "requires_schedule": True,
                "schedules": [{"when": {"region": "eu"}, "rates": {"input": "99"}}],
            }
        }
    )
    result = resolver.calculate(pin(resolver, region="us"), usage())
    assert result["amount"] == "0.00028"
    assert result["pricing_source"] == "catalogue"


def test_context_and_provenance_do_not_retain_credentials(catalog: dict[str, Any]) -> None:
    catalog["sources"][0]["url"] = "https://example.org/pricing?token=private#secret"
    pinned = pin(api_key="private", endpoint="https://private.example?token=secret")
    assert "private" not in json.dumps(pinned)
    assert pinned["pricing"]["sources"][0]["url"] == "https://example.org/pricing"


def test_checked_at_newer_selection(catalog: dict[str, Any]) -> None:
    new = copy.deepcopy(catalog)
    new["checked_at"] = "2026-11-02"
    assert pricing_is_newer(new, catalog)
    assert not pricing_is_newer(catalog, new)
    assert not pricing_is_newer(catalog, catalog)
    new["model"] = "other"
    assert not pricing_is_newer(new, catalog)


def test_cost_summary_separates_currency_and_credits() -> None:
    baseline = {"complete": True, "currencies": {"USD": "0.1"}, "denominations": {"credits": "3"}}
    live = {"complete": False, "currencies": {"USD": "0.2", "EUR": "1"}}
    summary = usage_cost_summary(baseline, live)
    assert "USD 0.3" in summary and "EUR 1" in summary
    assert "3 credits (non-monetary)" in summary and "partial subtotal" in summary


def test_observation_cost_uses_shared_currency_marker(catalog: dict[str, Any]) -> None:
    observed = SimpleNamespace(
        cost={"source": "provider_reported", "amount": "0.1", "currency": "USD", "denomination": "currency"}
    )
    result = PricingResolver().calculate(pin(), None, observed)
    assert result["source"] == "provider_reported"
    assert result["currency"] == "USD"
    assert result["amount"] == "0.1"


def test_observed_estimate_does_not_become_reported_charge(catalog: dict[str, Any]) -> None:
    observed = SimpleNamespace(cost={"source": "estimated", "amount": "0.1", "currency": "USD"})
    result = PricingResolver().calculate(pin(), usage(), observed)
    assert result["source"] == "estimated"


@pytest.mark.parametrize("currency", ["U1D", "usd", "EURO", "ÜSD"])
def test_malformed_reported_currency_falls_back(catalog: dict[str, Any], currency: str) -> None:
    observed = SimpleNamespace(cost={"amount": "1", "currency": currency})
    result = PricingResolver().calculate(pin(), usage(), observed)
    assert result["source"] == "estimated"
    assert result["currency"] == "USD"
    assert result["amount"] == "0.00028"


@pytest.mark.parametrize("count,expected,source", [(100, "0.00028", "catalogue"), (101, "0.000181", "override")])
def test_usage_dependent_override_pins_both_exact_cards(
    catalog: dict[str, Any], count: int, expected: str, source: str
) -> None:
    custom = card(
        rates={"input": "1", "output": "8"},
        conditions={"min_input_tokens": 101},
        cache_policy={"cache_read": "included", "cache_write": "included"},
    )
    resolver = PricingResolver({"openai/test": custom})
    pinned = pin(resolver)
    catalog["rates"]["input"] = "999"
    result = resolver.calculate(pinned, usage(input=count))
    assert result["amount"] == expected
    assert result["pricing_source"] == source


def test_unknown_schedule_does_not_claim_component_subtotal(catalog: dict[str, Any]) -> None:
    catalog["schedules"] = [{"when": {"region": "eu"}, "rates": {"input": "5"}}]
    result = PricingResolver().calculate(pin(), usage())
    assert result["complete"] is False
    assert all(part["component"] != "input" for part in result["subtotals"])


def test_ambiguous_schedules_are_incomplete(catalog: dict[str, Any]) -> None:
    schedules: list[dict[str, Any]] = [{"rates": {"input": "1"}}, {"rates": {"input": "3"}}]
    catalog["schedules"] = schedules
    result = PricingResolver().calculate(pin(), usage())
    assert result["amount"] is None
    assert "input: ambiguous overlapping schedules" in result["missing_components"]
    schedules[0]["priority"] = 1
    schedules[1]["priority"] = 2
    assert PricingResolver().calculate(pin(), usage(input=100, output=0))["amount"] == "0.0003"


def test_calendar_expiry_is_not_future_holiday_guess(catalog: dict[str, Any]) -> None:
    catalog["schedules"] = [
        {"when": {"weekdays": [6], "calendar_complete_through": "2026-10-31"}, "rates": {"input": "1"}}
    ]
    result = PricingResolver().calculate(pin(), usage())
    assert result["complete"] is False
    assert "calendar coverage expired" in " ".join(result["missing_components"])


@pytest.mark.parametrize("time,complete", [("21:00:00", True), ("21:59:59", False), ("22:00:00", False)])
def test_billing_time_uncertainty_only_gaps_near_boundary(catalog: dict[str, Any], time: str, complete: bool) -> None:
    catalog["schedules"] = [{"when": {"start_time": "22:00", "end_time": "05:00"}, "rates": {"input": "1"}}]
    pinned = PricingResolver().pin(
        "openai", "test", {"requested_at": f"2026-11-01T{time}Z", "billing_time_uncertainty_seconds": 2}
    )
    result = PricingResolver().calculate(pinned, usage())
    assert result["complete"] is complete


def test_marginal_cache_allocations_are_explicit(catalog: dict[str, Any]) -> None:
    catalog["schedules"] = [
        {
            "mode": "marginal",
            "rates": {"input": "2", "cache_read": "0.5"},
            "quantities": {"input": "uncached_low", "cache_read": "cached_low"},
        },
        {
            "mode": "marginal",
            "rates": {"input": "4", "cache_read": "1"},
            "quantities": {"input": "uncached_high", "cache_read": "cached_high"},
        },
    ]
    observed = SimpleNamespace(
        pricing_quantities={"uncached_low": 90, "cached_low": 10, "uncached_high": 30, "cached_high": 20}
    )
    result = PricingResolver().calculate(pin(), usage(input=150, output=0, read=30), observed)
    assert result["complete"] is True
    assert result["amount"] == "0.000325"


def test_multimodal_components_cover_only_their_partition(catalog: dict[str, Any]) -> None:
    catalog["components"] = [
        {
            "name": "text_input",
            "quantity": "text_input_tokens",
            "rate": "2",
            "unit": "per_million_tokens",
            "covers": "input",
        },
        {
            "name": "audio_input",
            "quantity": "audio_input_tokens",
            "rate": "10",
            "unit": "per_million_tokens",
            "covers": "input",
        },
    ]
    observed = SimpleNamespace(pricing_quantities={"text_input_tokens": 75, "audio_input_tokens": 25})
    result = PricingResolver().calculate(pin(), usage(input=100, output=0), observed)
    assert result["amount"] == "0.0004"
    observed.pricing_quantities["audio_input_tokens"] = 24
    result = PricingResolver().calculate(pin(), usage(input=100, output=0), observed)
    assert result["amount"] is None
    assert "input: component quantities do not cover normalized partition" in result["missing_components"]


@pytest.mark.parametrize(
    "unit,rate", [("per_token", "0.000002"), ("per_thousand_tokens", "0.002"), ("per_million_tokens", "2")]
)
def test_units_are_decimal_not_float(catalog: dict[str, Any], unit: str, rate: str) -> None:
    catalog["unit"] = unit
    catalog["rates"] = {"input": rate, "output": "0"}
    catalog["cache_policy"] = {"cache_read": "included", "cache_write": "included"}
    assert PricingResolver().calculate(pin(), usage())["amount"] == "0.0002"


def test_actual_ledger_pins_each_request_and_summary_is_a_snapshot(catalog: dict[str, Any]) -> None:
    resolver = PricingResolver()
    ledger = UsageLedger(pricing_resolver=resolver)
    first = ledger.begin("openai", "test", {"tools": []})
    second = ledger.begin("openai", "test", {"tools": []})
    catalog["rates"]["input"] = "999"
    ledger.record_response(first, usage(input=100, output=10))
    ledger.record_response(second, usage(input=140, output=20))
    snapshot = ledger.snapshot()
    assert (snapshot.input_tokens, snapshot.output_tokens, snapshot.total_tokens) == (240, 30, 270)
    assert snapshot.costs["currencies"]["USD"]["total"] == "0.00072"
    prior = {"complete": True, "currencies": {"USD": {"total": "1", "reported": "1", "estimated": "0"}}}
    summary = usage_cost_summary(prior, snapshot.costs)
    assert "USD 1.00072" in summary
    assert "includes estimates" in summary
    assert usage_cost_summary(prior, ledger.snapshot().costs) == summary


def test_mixed_cache_ttl_uses_pinned_schedule_rates(catalog: dict[str, Any]) -> None:
    catalog["default_context"] = {"cache_ttl": "5m"}
    catalog["schedules"] = [
        {"when": {"cache_ttl": "5m"}, "rates": {"cache_write": "3"}},
        {"when": {"cache_ttl": "1h"}, "rates": {"cache_write": "4"}},
    ]
    observed = SimpleNamespace(pricing_quantities={"cache_write_5m": 5, "cache_write_1h": 5})
    result = PricingResolver().calculate(pin(), usage(input=100, output=0, write=10), observed)
    assert result["amount"] == "0.000215"
    assert result["complete"] is True


def test_known_disabled_tool_zero_and_fractional_meter(catalog: dict[str, Any]) -> None:
    catalog["components"] = [{"name": "code_execution", "quantity": "code_execution_hours", "rate": "0.05"}]
    resolver = PricingResolver()
    assert resolver.calculate(pin(tools=[]), usage())["amount"] == "0.00028"
    observed = SimpleNamespace(pricing_quantities={"code_execution_hours": "0.5"})
    result = resolver.calculate(pin(tools=["code_execution"]), usage(), observed)
    assert result["amount"] == "0.02528"
    assert json.loads(json.dumps(result)) == result
    assert not resolver.calculate(pin(tools=["code_execution"]), usage())["complete"]


def test_anthropic_reported_search_requests_match_card_quantity(monkeypatch: pytest.MonkeyPatch) -> None:
    from kolega_code.llm.usage import observation_from_metadata

    exact = card(
        provider="anthropic",
        components=[{"name": "web_search", "quantity": "web_search_calls", "rate": "0.01"}],
    )
    monkeypatch.setitem(MODEL_SPECS, ("anthropic", "test"), {"pricing": exact})
    raw = {
        "input_tokens": 100,
        "output_tokens": 10,
        "cache_read_input_tokens": 0,
        "cache_write_input_tokens": 0,
        "server_tool_use": {"web_search_requests": 2},
    }
    tokens = normalize_usage(raw, "anthropic", "test")
    observed = observation_from_metadata(raw, "anthropic", "test", complete=True)
    resolver = PricingResolver()
    pinned = resolver.pin("anthropic", "test", {"tools": ["web_search"]})
    result = resolver.calculate(pinned, tokens, observed)
    assert result["complete"] is True
    assert result["amount"] == "0.02028"


@pytest.mark.parametrize(
    "provider,model",
    [
        ("openai", "gpt-6.1-sol"),
        ("anthropic", "claude-opus-5"),
        ("google", "gemini-4-argon"),
        ("deepseek", "deepseek-flash"),
        ("ollama_cloud", "deepseek-v4.1-flash"),
    ],
)
def test_researched_cards_support_ordinary_default_request(provider: str, model: str) -> None:
    resolver = PricingResolver()
    pinned = resolver.pin(provider, model, {"requested_at": "2026-11-01T00:00:00Z", "tools": []})
    result = resolver.calculate(pinned, usage(input=100, output=10))
    assert result["source"] == "estimated"
    assert result["amount"] is not None
    assert result["complete"] is True
