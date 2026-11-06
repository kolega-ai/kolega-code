from __future__ import annotations

import pytest

from kolega_code.llm.pricing import PricingResolver
from kolega_code.llm.specs import (
    get_model_specs,
    preferred_edit_protocol,
    prior_reasoning_is_replayable,
    resolve_max_input_tokens,
    supports_hosted_web_search,
)
from kolega_code.llm.usage import NormalizedUsage

MODELS = (
    "glm-5.3-abliterated",
    "glm-5.3-flash-abliterated",
    "deepseek-v4.1-flash-abliterated",
    "deepseek-v4.1-flash-fast",
    "minimax-m3.1-flash",
)


@pytest.mark.parametrize(
    ("model", "context", "output"),
    (
        ("glm-5.3-abliterated", 1048576, 131072),
        ("glm-5.3-flash-abliterated", 1048576, 131072),
        ("deepseek-v4.1-flash-abliterated", 1048576, 65536),
        ("deepseek-v4.1-flash-fast", 1048576, 262144),
        ("minimax-m3.1-flash", 1048576, 512000),
    ),
)
def test_refuseless_exact_route_budgets(model: str, context: int, output: int) -> None:
    spec = get_model_specs("refuseless", model)
    assert spec["context_length"] == context
    assert spec["max_completion_tokens"] == output
    assert spec["input_budget"] == "output_shares_window"
    assert resolve_max_input_tokens(spec) == context
    assert spec["default_temperature"] == 1.0
    assert spec["supports_temperature"] is True


@pytest.mark.parametrize(
    ("model", "input_rate", "cache_rate", "output_rate", "amount"),
    (
        ("glm-5.3-abliterated", "1.00", "0.15", "2.00", "0.000196"),
        ("glm-5.3-flash-abliterated", "0.12", "0.03", "0.50", "0.00003546"),
        ("deepseek-v4.1-flash-abliterated", "0.12", "0.025", "0.45", "0.00003311"),
        ("deepseek-v4.1-flash-fast", "0.01", "0.002", "0.08", "0.00000467"),
        ("minimax-m3.1-flash", "0.02", "0.002", "0.08", "0.0000057"),
    ),
)
def test_refuseless_published_pricing_counts_cache_and_reasoning_subsets_once(
    model: str, input_rate: str, cache_rate: str, output_rate: str, amount: str
) -> None:
    card = get_model_specs("refuseless", model)["pricing"]
    assert card["status"] == "priced"
    assert (card["provider"], card["model"]) == ("refuseless", model)
    assert card["billing_mode"] == "prepaid_credits"
    assert card["currency"] == "USD"
    assert card["unit"] == "per_million_tokens"
    assert card["rates"] == {"input": input_rate, "cache_read": cache_rate, "output": output_rate}
    assert card["checked_at"] == "2026-11-05"
    assert any(
        source["url"] == f"https://refuseless.com/models/{model}" and source["model_id"] == model
        for source in card["sources"]
    )

    resolver = PricingResolver()
    pinned = resolver.pin("refuseless", model, {"requested_at": "2026-11-05T12:00:00Z"})
    result = resolver.calculate(
        pinned,
        NormalizedUsage(
            provider="refuseless",
            model=model,
            reported=True,
            input_tokens=123,
            output_tokens=45,
            total_tokens=168,
            cache_read_input_tokens=20,
            cache_write_input_tokens=0,
            reasoning_output_tokens=10,
            unavailable_reason=None,
        ),
    )
    assert result["amount"] == amount
    assert result["currency"] == "USD"
    assert result["complete"] is True
    assert result["missing_components"] == []


@pytest.mark.parametrize("model", MODELS)
def test_refuseless_uses_native_reasoning_and_local_tool_fallbacks(model: str) -> None:
    assert prior_reasoning_is_replayable("refuseless", model)
    assert not supports_hosted_web_search("refuseless", model)
    assert preferred_edit_protocol("refuseless", model) is None
