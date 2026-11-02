from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from acp.schema import UsageUpdate
import pytest

from kolega_code.acp.accounting import AcpUsageAccounting
from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.session_usage import SessionUsageSink
from kolega_code.llm.accounting import UsageObservation
from kolega_code.llm.ledger import UsageLedger
from kolega_code.llm.pricing import PricingResolver, usage_cost_summary
from kolega_code.llm.usage import normalize_usage


def _owner(*, rates: dict[str, str], components: list[dict[str, Any]] | None = None) -> AcpUsageAccounting:
    resolver = PricingResolver(
        {
            "openai/accounting-test": {
                "currency": "USD",
                "rates": rates,
                "cache_policy": {"cache_read": "included", "cache_write": "included"},
                "components": components or [],
            }
        }
    )
    owner = AcpUsageAccounting("cost-coverage", ledger=UsageLedger(pricing_resolver=resolver))
    owner.rebind(
        SimpleNamespace(model_context_length=1000, model_max_input_tokens=800, _acp_generation="cost-generation")
    )
    owner.observe_context(73, "cost-generation")
    return owner


def _respond(owner: AcpUsageAccounting) -> None:
    request = owner.ledger.begin("openai", "accounting-test", {"tools": ["web_search"]})
    owner.ledger.record_response(
        request,
        normalize_usage({"prompt_tokens": 100, "completion_tokens": 10}, "openai", "accounting-test"),
    )


def test_incomplete_request_retains_known_token_cost_without_publishing_native_total() -> None:
    owner = _owner(
        rates={"input": "2", "output": "8"},
        components=[{"name": "web_search", "quantity": "web_search_calls", "rate": "0.01"}],
    )
    _respond(owner)
    costs = owner.extension()["session"]["cost"]
    assert costs["complete"] is False
    assert costs["partial"] == 1
    assert costs["missing"] == 0
    assert Decimal(costs["monetary"][0]["amount"]) == Decimal("0.00028")
    assert Decimal(costs["monetary"][0]["token_cost_subtotal"]) == Decimal("0.00028")
    assert costs["monetary"][0]["source"] == "estimated"
    update = owner.update()
    assert isinstance(update, UsageUpdate)
    assert update.cost is None


@pytest.mark.parametrize("reported", [False, True])
def test_zero_cost_retains_reported_or_estimated_source(reported: bool) -> None:
    owner = _owner(rates={"input": "0", "output": "0"})
    if reported:
        request = owner.ledger.begin("openai", "accounting-test")
        owner.ledger.record_observation(
            request,
            UsageObservation(
                provider="openai",
                cost={"source": "provider_reported", "amount": "0", "currency": "USD", "complete": True},
            ),
        )
        owner.ledger.record_response(request, None)
    else:
        _respond(owner)
    costs = owner.extension()["session"]["cost"]
    assert costs["complete"] is True
    row = costs["monetary"][0]
    assert Decimal(row["amount"]) == 0
    assert row["source"] == ("reported" if reported else "estimated")
    update = owner.update()
    assert isinstance(update, UsageUpdate)
    assert update.cost is not None
    assert Decimal(str(update.cost.amount)) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "partial,estimated_rate,reported_amount,expected_source,expected_total",
    [
        (True, "2", None, "estimated", "0.00028"),
        (False, "0", None, "estimated", "0"),
        (False, "2", "0", "mixed", "0.00028"),
        (False, "0", "1", "mixed", "1"),
    ],
)
async def test_cost_coverage_and_zero_provenance_survive_real_journal_reload(
    tmp_path: Path,
    partial: bool,
    estimated_rate: str,
    reported_amount: str | None,
    expected_source: str,
    expected_total: str,
) -> None:
    components = [{"name": "web_search", "quantity": "web_search_calls", "rate": "0.01"}] if partial else []
    owner = _owner(
        rates={"input": estimated_rate, "output": "8" if estimated_rate == "2" else "0"},
        components=components,
    )
    store = SessionStore(tmp_path / "state")
    session = store.create(tmp_path / "project", "code", {})
    sink = SessionUsageSink(
        store.journal(session.session_id), store.recorder(session.session_id), owner.ledger, mode="acp"
    )
    owner.ledger.observer = sink
    await sink.start()
    try:
        _respond(owner)
        if reported_amount is not None:
            request = owner.ledger.begin("openai", "accounting-test")
            owner.ledger.record_observation(
                request,
                UsageObservation(
                    provider="openai",
                    cost={
                        "source": "provider_reported",
                        "amount": reported_amount,
                        "currency": "USD",
                        "complete": True,
                    },
                ),
            )
            owner.ledger.record_response(request, None)
        assert await sink.flush()
    finally:
        await sink.aclose()

    loaded = store.load(session.session_id).usage
    restored = AcpUsageAccounting("restored", baseline=loaded)
    for metadata in (owner.extension()["session"], restored.extension()["session"]):
        cost = metadata["cost"]
        assert cost["complete"] is not partial
        assert cost["partial"] == int(partial)
        assert cost["missing"] == 0
        row = cost["monetary"][0]
        assert row["source"] == expected_source
        assert Decimal(row["amount"]) == Decimal(expected_total)
        assert row["estimated_requests"] == 1
        assert row["reported_requests"] == int(reported_amount is not None)
        breakdown = metadata["breakdown"][0]["costs"]["monetary"][0]
        assert Decimal(breakdown["amount"]) == Decimal(expected_total)
        if partial:
            assert Decimal(row["token_cost_subtotal"]) == Decimal("0.00028")
    expected_label = "partial subtotal" if partial else "includes estimates"
    assert expected_label in usage_cost_summary({}, owner.ledger.snapshot().costs)
    assert expected_label in usage_cost_summary(loaded["costs"], {})


@pytest.mark.parametrize("reported,estimated,source", [("0.25", "0.25", "mixed"), ("0", "0", "unknown")])
def test_legacy_source_counts_are_unknown_not_inferred_from_amounts(reported: str, estimated: str, source: str) -> None:
    total = Decimal(reported) + Decimal(estimated)
    baseline = {
        "requests": 5,
        "costs": {
            "currencies": {
                "USD": {
                    "reported": reported,
                    "estimated": estimated,
                    "total": str(total),
                    "requests": 5,
                }
            },
            "complete": True,
        },
    }
    owner = AcpUsageAccounting("legacy", baseline=baseline)
    row = owner.extension()["session"]["cost"]["monetary"][0]
    assert row["source"] == source
    assert row["reported_requests"] is None
    assert row["estimated_requests"] is None
    assert Decimal(row["amount"]) == total
    # A new request cannot recover the lost legacy source-count split.
    request = owner.ledger.begin("openrouter", "accounting-test")
    owner.ledger.record_observation(
        request,
        UsageObservation(
            provider="openrouter",
            cost={"source": "provider_reported", "amount": "0", "currency": "USD", "complete": True},
        ),
    )
    owner.ledger.record_response(request, None)
    row = owner.extension()["session"]["cost"]["monetary"][0]
    assert row["source"] == ("mixed" if source == "mixed" else "reported")
    assert row["requests"] == 6
    assert row["reported_requests"] is None
    assert row["estimated_requests"] is None
    assert Decimal(row["amount"]) == total
