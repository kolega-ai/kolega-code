from __future__ import annotations

import json
from copy import deepcopy
from decimal import Decimal
from typing import Any

import pytest

from kolega_code.llm.pricing import PricingResolver
from kolega_code.llm.specs.catalog import MODEL_SPECS, WILDCARD_MODEL_SPECS
from kolega_code.llm.specs.mistral_catalog import catalog_entries, spec_from_jsonable, spec_to_jsonable
from kolega_code.llm.specs.pricing import validate_pricing
from kolega_code.llm.usage import NormalizedUsage

PRICING = {key: spec["pricing"] for key, spec in {**MODEL_SPECS, **WILDCARD_MODEL_SPECS}.items()}


def _estimate(
    provider: str,
    model: str,
    *,
    input_tokens: int = 100,
    output_tokens: int = 0,
    read: int | None = 0,
    write: int | None = 0,
    **context: Any,
) -> dict[str, Any]:
    resolver = PricingResolver()
    pinned = resolver.pin(
        provider,
        model,
        {
            "requested_at": "2026-11-01T12:00:00Z",
            "tools": [],
            "pricing_quantities": {"usage_violation_requests": 0},
            **context,
        },
    )
    usage = NormalizedUsage(
        provider=provider,
        model=model,
        reported=True,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cache_read_input_tokens=read,
        cache_write_input_tokens=write,
        reasoning_output_tokens=0,
        unavailable_reason=None,
    )
    return resolver.calculate(pinned, usage)


def test_catalogue_pricing_has_exact_identity_and_json_decimal_rates() -> None:
    for (provider, model), card in PRICING.items():
        validate_pricing(card)
        assert (card["provider"], card["model"]) == (provider, model)
        assert card["checked_at"] == "2026-11-01"
        assert json.loads(json.dumps(card)) == card
        for row in [card, *card["schedules"]]:
            for rate in row.get("rates", {}).values():
                assert isinstance(rate, str)
                assert Decimal(rate).is_finite() and Decimal(rate) >= 0
        if card["status"] != "priced":
            assert card["reason"]


@pytest.mark.parametrize(
    ("provider", "model", "count", "rate"),
    [
        ("openai", "gpt-5.4", 272000, "2.5"),
        ("openai", "gpt-5.4", 272001, "5"),
        ("google", "gemini-3.1-pro-preview", 200000, "2"),
        ("google", "gemini-3.1-pro-preview", 200001, "4"),
        ("xai", "grok-4.7", 199999, "2"),
        ("xai", "grok-4.7", 200000, "4"),
        ("dashscope", "qwen3-coder-flash", 32000, "0.3"),
        ("dashscope", "qwen3-coder-flash", 32001, "0.5"),
        ("dashscope", "qwen3-coder-flash", 128000, "0.5"),
        ("dashscope", "qwen3-coder-flash", 128001, "0.8"),
        ("dashscope", "qwen3-coder-flash", 256000, "0.8"),
        ("dashscope", "qwen3-coder-flash", 256001, "1.6"),
    ],
)
def test_whole_request_boundaries(provider: str, model: str, count: int, rate: str) -> None:
    result = _estimate(provider, model, input_tokens=count)
    assert result["complete"], result
    assert Decimal(result["amount"]) == Decimal(count) * Decimal(rate) / 1_000_000


def test_openai_literal_rounded_batch_cached_rate() -> None:
    result = _estimate("openai", "gpt-5.4", input_tokens=100, read=100, service_tier="batch")
    assert result["amount"] == "0.000013"


@pytest.mark.parametrize(
    ("provider", "model", "context"),
    [
        ("openai", "gpt-5.4-mini", {"service_tier": "ultrafast"}),
        ("mistral", "codestral-2508", {"service_tier": "auto"}),
        ("fireworks", "accounts/fireworks/models/minimax-m3", {"service_tier": "fast"}),
        ("dashscope", "qwen3-coder-flash", {"region": "unknown"}),
    ],
)
def test_unsupported_context_never_becomes_a_complete_base_rate(
    provider: str,
    model: str,
    context: dict[str, Any],
) -> None:
    result = _estimate(provider, model, **context)
    assert not result["complete"]
    assert result["amount"] is None


def test_google_alias_requires_explicit_routing_evidence() -> None:
    card = PRICING["google", "gemini-3.7-flash"]
    assert card["billing_model"] == "gemini-3.8-flash"
    assert any("automatically routes" in source.get("evidence", "") for source in card["sources"])
    assert _estimate("google", "gemini-3.7-flash")["amount"] == _estimate("google", "gemini-3.8-flash")["amount"]


@pytest.mark.parametrize(
    ("stamp", "rate"),
    [("2026-11-30T23:59:59Z", "2"), ("2026-12-01T00:00:00Z", "4")],
)
def test_google_promotional_expiry_has_documented_regular_rate(stamp: str, rate: str) -> None:
    result = _estimate("google", "gemini-4-argon", requested_at=stamp)
    assert Decimal(result["amount"]) == Decimal(100) * Decimal(rate) / 1_000_000


def test_chatgpt_credits_are_not_api_dollars_or_included_subscription_usage() -> None:
    result = _estimate("openai_chatgpt", "gpt-6-astra")
    assert result["denomination"] == "openai_codex_credits"
    assert result.get("currency") is None
    assert result["amount"] == "0.025"
    included = _estimate("openai_chatgpt", "gpt-6-astra", billing_surface="subscription")
    assert included["source"] == "unavailable"
    assert PRICING["openai_chatgpt", "gpt-5.4"]["status"] == "subscription"


def test_anthropic_cache_ttl_and_us_uplift_are_disjoint() -> None:
    result = _estimate(
        "anthropic",
        "claude-opus-5",
        input_tokens=100,
        write=100,
        cache_ttl="1h",
        region="us",
    )
    assert result["amount"] == "0.0011"


def test_groq_batch_does_not_stack_prompt_cache_discount() -> None:
    result = _estimate("groq", "openai/gpt-oss-20b", input_tokens=100, read=100, service_tier="batch")
    assert result["amount"] == "0.00000375"


def test_fireworks_batch_never_inherits_undocumented_standard_cache_rate() -> None:
    uncached = _estimate("fireworks", "accounts/fireworks/models/kimi-k3", service_tier="batch")
    assert uncached["amount"] == "0.00015"
    cached = _estimate("fireworks", "accounts/fireworks/models/kimi-k3", read=50, service_tier="batch")
    assert cached["amount"] is None
    assert not cached["complete"]


@pytest.mark.parametrize(
    ("model", "retired_at"),
    [
        ("glm-5p2", "2026-09-25"),
        ("kimi-k2p7-code", "2026-09-25"),
        ("deepseek-v4-pro", "2026-08-27"),
        ("deepseek-v4-flash", "2026-08-14"),
    ],
)
def test_fireworks_serverless_retirements_never_use_successor_or_gpu_rates(model: str, retired_at: str) -> None:
    identifier = f"accounts/fireworks/models/{model}"
    card = PRICING["fireworks", identifier]
    assert card["status"] == "retired"
    assert card["retired_at"] == retired_at
    assert card["billing_mode"] == "serverless_inference"
    assert any(source["url"] == "https://docs.fireworks.ai/updates/changelog" for source in card["sources"])
    assert not card["rates"] and not card["schedules"]
    assert _estimate("fireworks", identifier)["source"] == "unavailable"


@pytest.mark.parametrize("model", ["glm-5p1", "qwen3p7-plus"])
def test_fireworks_nonserverless_rows_remain_sourced_exceptions(model: str) -> None:
    identifier = f"accounts/fireworks/models/{model}"
    assert PRICING["fireworks", identifier]["status"] == "requires_configuration"
    assert not _estimate("fireworks", identifier)["complete"]


@pytest.mark.parametrize(
    ("model", "tier", "amount"),
    [
        ("kimi-k3", "standard", "0.000396"),
        ("kimi-k3", "priority", "0.000495"),
        ("ember-1", "standard", "0.000396"),
        ("ember-1", "priority", "0.000495"),
        ("minimax-m3", "standard", "0.0000372"),
        ("minimax-m3", "priority", "0.0000558"),
    ],
)
def test_fireworks_exact_published_serverless_input_cache_output_prices(model: str, tier: str, amount: str) -> None:
    cost = _estimate(
        "fireworks",
        f"accounts/fireworks/models/{model}",
        input_tokens=100,
        output_tokens=10,
        read=20,
        service_tier=tier,
    )
    assert cost["complete"]
    assert cost["amount"] == amount


@pytest.mark.parametrize(
    "context", [{"service_tier": "fast"}, {"region": "us"}, {"service_tier": "priority", "region": "us"}]
)
def test_fireworks_fast_and_us_router_prices_are_not_base_model_tiers(context: dict[str, Any]) -> None:
    assert not _estimate("fireworks", "accounts/fireworks/models/kimi-k3", **context)["complete"]


def test_dashscope_implicit_cache_uses_regional_input_discount() -> None:
    result = _estimate("dashscope", "qwen3-coder-flash", input_tokens=100, read=50)
    assert result["amount"] == "0.000018"
    assert not _estimate("dashscope", "qwen3-coder-flash", cache_ttl="5m")["complete"]


def test_kimi_write_cache_partitions_are_not_an_additional_base_input_charge() -> None:
    result = _estimate(
        "moonshot",
        "kimi-k3",
        input_tokens=100,
        read=20,
        write=30,
        output_tokens=10,
        cache_ttl="1h",
    )
    assert result["amount"] == "0.000486"


def test_missing_cache_details_do_not_become_authoritative_zero() -> None:
    result = _estimate("anthropic", "claude-opus-5", read=None)
    assert not result["complete"]
    assert result["amount"] is None


def test_mistral_deprecated_medium_uses_launch_price_not_replacement_price() -> None:
    card = PRICING["mistral", "mistral-medium-3-5"]
    assert card["rates"] == {"input": "1.5", "output": "7.5"}
    assert any("vibe-remote-agents" in source["url"] for source in card["sources"])
    assert _estimate("mistral", "mistral-medium-3-5")["amount"] == "0.00015"
    assert not _estimate("mistral", "mistral-medium-3-5", read=20)["complete"]
    assert PRICING["mistral", "mistral-medium-2604"]["status"] == "unpublished"


def test_mistral_enrichment_preserves_capabilities_and_serializes_independently() -> None:
    payload = {
        "data": [
            {
                "id": "codestral-2508",
                "billing_model_name": "codestral-2508",
                "max_context_length": 256000,
                "default_model_temperature": 0.3,
                "capabilities": {"completion_chat": True, "function_calling": True, "vision": False},
            }
        ]
    }
    identifier, spec = catalog_entries(payload)[0]
    assert identifier == "codestral-2508"
    assert spec["context_length"] == 256000
    assert spec["default_temperature"] == 0.3
    assert spec["supports_vision"] is False
    assert spec["pricing"] == PRICING["mistral", identifier]
    saved = spec_to_jsonable(spec)
    assert spec_from_jsonable(json.loads(json.dumps(saved))) == spec
    saved["pricing"]["rates"]["input"] = "999"
    assert PRICING["mistral", identifier]["rates"]["input"] == "0.3"


def test_mistral_invalid_cached_pricing_does_not_discard_capabilities() -> None:
    spec = deepcopy(MODEL_SPECS["mistral", "codestral-2508"])
    spec["pricing"] = {"schema_version": 99}
    restored = spec_from_jsonable(spec)
    assert restored["context_length"] == spec["context_length"]
    assert "pricing" not in restored


@pytest.mark.parametrize(
    ("provider", "stamp", "expected"),
    [("mistral", "2026-11-02", "0.4"), ("mistral", "2026-10-31", "0.3"), ("openai", "2026-11-02", "0.3")],
)
def test_mistral_refresh_accepts_only_newer_exact_route_pricing(
    provider: str,
    stamp: str,
    expected: str,
) -> None:
    card = deepcopy(PRICING["mistral", "codestral-2508"])
    card.update(provider=provider, checked_at=stamp)
    card["rates"]["input"] = "0.4"
    payload = {
        "data": [
            {
                "id": "codestral-2508",
                "billing_model_name": "codestral-2508",
                "max_context_length": 256000,
                "capabilities": {"completion_chat": True},
                "pricing": card,
            }
        ]
    }
    _, spec = catalog_entries(payload)[0]
    assert spec["pricing"]["rates"]["input"] == expected
    assert spec["context_length"] == 256000


def test_new_mistral_overlay_id_gets_an_explicit_gap_not_a_family_rate() -> None:
    entries = catalog_entries(
        {
            "data": [
                {
                    "id": "mistral-small-undocumented",
                    "max_context_length": 256000,
                    "capabilities": {"completion_chat": True},
                }
            ]
        }
    )
    _, spec = entries[0]
    validate_pricing(spec["pricing"])
    assert spec["pricing"]["status"] == "unverified"
    assert "rates" not in spec["pricing"]
