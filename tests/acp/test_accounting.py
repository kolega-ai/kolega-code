from __future__ import annotations

import asyncio
from copy import deepcopy
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest
from acp.schema import SessionInfoUpdate, UsageUpdate

from kolega_code.acp.accounting import USAGE_META_KEY, AcpUsageAccounting, native_prompt_usage
from kolega_code.llm.accounting import UsageObservation
from kolega_code.llm.ledger import logical_prompt_scope
from kolega_code.llm.usage import NormalizedUsage


def _usage(input_tokens: int, output_tokens: int, **details: Any) -> NormalizedUsage:
    return NormalizedUsage(
        provider="openrouter",
        model="example/model",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        reported=True,
        unavailable_reason=None,
        **{
            "cache_read_input_tokens": None,
            "cache_write_input_tokens": None,
            "reasoning_output_tokens": None,
            **details,
        },
    )


def _agent(generation: str = "g1", size: int = 1000, budget: int = 800) -> Any:
    return SimpleNamespace(model_context_length=size, model_max_input_tokens=budget, _acp_generation=generation)


def _charge(amount: str, currency: str = "USD") -> dict[str, Any]:
    return {
        "source": "provider_reported",
        "amount": amount,
        "currency": currency,
        "denomination": "currency",
        "complete": True,
    }


def test_occupancy_lifetime_and_prompt_consumption_are_independent() -> None:
    owner = AcpUsageAccounting("s")
    owner.rebind(_agent())
    owner.observe_context(73, "g1")
    with logical_prompt_scope("A"):
        a = owner.ledger.begin("openrouter", "example/model")
        owner.ledger.record_response(a, _usage(100, 10))
    with logical_prompt_scope("B"):
        b = owner.ledger.begin("openrouter", "example/model")
        owner.ledger.record_response(b, _usage(140, 20))
    update = owner.update()
    assert isinstance(update, UsageUpdate)
    assert (update.used, update.size) == (73, 1000)
    extension = owner.extension("B")
    assert extension["session"]["inputTokens"] == 240
    assert extension["session"]["outputTokens"] == 30
    assert extension["session"]["totalTokens"] == 270
    assert extension["prompt"] is not None
    assert extension["prompt"]["totalTokens"] == 160
    assert owner.extension("A")["prompt"]["totalTokens"] == 110  # type: ignore[index]
    assert owner.extension()["session"]["totalTokens"] == 270


@pytest.mark.asyncio
async def test_background_request_retains_launch_prompt_across_task_and_settlement() -> None:
    owner = AcpUsageAccounting("s")
    ready, settle = asyncio.Event(), asyncio.Event()

    async def child() -> None:
        request = owner.ledger.begin("openrouter", "example/model")
        ready.set()
        await settle.wait()
        owner.ledger.record_response(request, _usage(100, 10))

    with logical_prompt_scope("A"):
        task = asyncio.create_task(child())
    await ready.wait()
    with logical_prompt_scope("B"):
        request = owner.ledger.begin("openrouter", "example/model")
        owner.ledger.record_response(request, _usage(140, 20))
        settle.set()
        await task
    assert owner.extension("A")["prompt"]["totalTokens"] == 110  # type: ignore[index]
    assert owner.extension("B")["prompt"]["totalTokens"] == 160  # type: ignore[index]
    assert owner.extension()["session"]["totalTokens"] == 270


def test_unmeasured_context_is_metadata_only_and_model_generation_is_fenced() -> None:
    owner = AcpUsageAccounting("s")
    owner.rebind(_agent())
    assert isinstance(owner.update(), SessionInfoUpdate)
    owner.observe_context(77, "g1")
    assert isinstance(owner.update(), UsageUpdate)
    owner.rebind(_agent("g2", 2000, 1500))
    owner.observe_context(999, "g1")
    assert isinstance(owner.update(), SessionInfoUpdate)
    owner.observe_context(55, "g2")
    update = owner.update()
    assert isinstance(update, UsageUpdate)
    assert (update.used, update.size) == (55, 2000)
    assert owner.extension()["session"]["context"]["usableInputBudget"] == 1500


def test_native_prompt_usage_is_opt_in_disjoint_and_requires_known_partitions() -> None:
    owner = AcpUsageAccounting("s")
    with logical_prompt_scope("A"):
        request = owner.ledger.begin("openrouter", "example/model")
        owner.ledger.record_response(
            request,
            _usage(
                120,
                30,
                cache_read_input_tokens=100,
                cache_write_input_tokens=10,
                reasoning_output_tokens=5,
            ),
        )
    extension = owner.extension("A")
    assert native_prompt_usage(extension, enabled=False) is None
    native = native_prompt_usage(extension, enabled=True)
    assert native is not None
    assert (
        native.input_tokens,
        native.output_tokens,
        native.cached_read_tokens,
        native.cached_write_tokens,
        native.thought_tokens,
        native.total_tokens,
    ) == (10, 25, 100, 10, 5, 150)
    unknown = deepcopy(extension)
    unknown["prompt"]["thoughtTokens"] = None  # type: ignore[index]
    assert native_prompt_usage(unknown, enabled=True) is None


def test_empty_missing_and_partial_are_not_equivalent() -> None:
    owner = AcpUsageAccounting("s")
    empty = owner.extension()["session"]
    assert empty["totalTokens"] == 0 and empty["complete"]
    request = owner.ledger.begin("openrouter", "example/model")
    unknown = owner.extension()["session"]
    assert unknown["totalTokens"] is None and not unknown["complete"]
    owner.ledger.record_observation(request, UsageObservation(provider="openrouter", input_tokens=12))
    partial = owner.extension()["session"]
    assert partial["inputTokens"] == 12
    assert partial["outputTokens"] is None
    assert partial["totalTokens"] is None
    assert not partial["complete"]


def test_missing_cache_splits_are_null_with_separate_known_subtotals() -> None:
    owner = AcpUsageAccounting("s")
    first = owner.ledger.begin("openrouter", "example/model")
    owner.ledger.record_response(first, _usage(100, 10, cache_read_input_tokens=80))
    second = owner.ledger.begin("openrouter", "example/model")
    owner.ledger.record_response(second, _usage(140, 20))
    metadata = owner.extension()["session"]
    assert metadata["cachedReadTokens"] is None
    assert metadata["knownSubtotals"]["cachedReadTokens"] == 80
    assert metadata["totalTokens"] == 270
    assert metadata["complete"]


def test_reported_cost_is_independent_of_tokens_and_unknown_coverage_hides_native_total() -> None:
    owner = AcpUsageAccounting("s")
    owner.rebind(_agent())
    owner.observe_context(22, "g1")
    request = owner.ledger.begin("openrouter", "example/model")
    owner.ledger.record_observation(request, UsageObservation(provider="openrouter", cost=_charge("0.005")))
    owner.ledger.record_failure(request, "cancelled")
    update = owner.update()
    assert isinstance(update, UsageUpdate)
    assert update.cost is not None
    assert Decimal(str(update.cost.amount)) == Decimal("0.005")
    assert not owner.extension()["session"]["complete"]
    missing = owner.ledger.begin("openrouter", "example/model")
    owner.ledger.record_failure(missing, "failed without a charge")
    assert owner.update().cost is None  # type: ignore[union-attr]
    assert owner.extension()["session"]["cost"]["monetary"][0]["amount"] == "0.005"


def test_mixed_currency_and_provider_credits_never_become_one_native_cost() -> None:
    owner = AcpUsageAccounting("s")
    owner.rebind(_agent())
    owner.observe_context(22, "g1")
    for currency in ("USD", "EUR"):
        request = owner.ledger.begin("openrouter", "example/model")
        owner.ledger.record_observation(request, UsageObservation(provider="openrouter", cost=_charge("1", currency)))
        owner.ledger.record_response(request, None)
    assert owner.update().cost is None  # type: ignore[union-attr]
    assert len(owner.extension()["session"]["cost"]["monetary"]) == 2


def test_resume_baseline_is_frozen_and_never_repriced_or_reread() -> None:
    baseline = {
        "requests": 1,
        "responses": 1,
        "reported": 1,
        "input_tokens": 100,
        "output_tokens": 10,
        "total_tokens": 110,
        "known_fields": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 1},
        "coverage": {"full": True},
        "costs": {
            "currencies": [{"currency": "USD", "reported": "0.01", "estimated": "0", "total": "0.01", "requests": 1}],
            "complete": True,
        },
    }
    owner = AcpUsageAccounting("s", baseline=baseline)
    baseline["total_tokens"] = 999
    request = owner.ledger.begin("openrouter", "example/model")
    owner.ledger.record_response(request, _usage(140, 20))
    assert owner.extension()["session"]["totalTokens"] == 270
    assert owner.extension()["session"]["cost"]["monetary"][0]["amount"] == "0.01"


@pytest.mark.asyncio
async def test_publisher_is_session_owned_coalescing_and_transport_failure_is_not_sink_failure() -> None:
    owner = AcpUsageAccounting("s")
    notifications: list[Any] = []
    first, second = asyncio.Event(), asyncio.Event()
    fail = True

    async def send(update: Any) -> None:
        nonlocal fail
        notifications.append(update)
        if fail:
            fail = False
            first.set()
            raise OSError("editor disconnected")
        second.set()

    owner.start_publisher(send)
    await first.wait()
    await asyncio.sleep(0)
    assert len(notifications) == 1
    assert owner.ledger.persist_failures == 0
    owner.changed()
    await second.wait()
    assert len(notifications) == 2
    wire = notifications[-1].model_dump(by_alias=True, exclude_none=True)
    assert USAGE_META_KEY in wire["_meta"]
    await owner.aclose()


def test_two_session_owners_do_not_share_requests_or_run_ids() -> None:
    first, second = AcpUsageAccounting("one"), AcpUsageAccounting("two")
    request = first.ledger.begin("openrouter", "example/model")
    first.ledger.record_response(request, _usage(100, 10))
    assert first.ledger.run_id != second.ledger.run_id
    assert second.extension()["session"]["totalTokens"] == 0


def test_provider_model_breakdowns_keep_baseline_and_live_cost_subtotals() -> None:
    baseline = {
        "requests": 1,
        "breakdown": [
            {
                "provider": "openrouter",
                "model": "example/model",
                "requests": 1,
                "costs": {
                    "currencies": {"USD": {"reported": "0.01", "estimated": "0", "total": "0.01", "requests": 1}},
                    "complete": True,
                },
            }
        ],
    }
    owner = AcpUsageAccounting("s", baseline=baseline)
    request = owner.ledger.begin("openrouter", "example/model")
    owner.ledger.record_observation(request, UsageObservation(provider="openrouter", cost=_charge("0.005")))
    owner.ledger.record_response(request, None)
    row = owner.extension()["session"]["breakdown"][0]
    assert row["requests"] == 2
    assert Decimal(row["costs"]["monetary"][0]["amount"]) == Decimal("0.015")


def test_unclosed_resume_run_is_unhealthy_even_when_no_write_failure_was_recorded() -> None:
    owner = AcpUsageAccounting("s", baseline={"coverage": {"full": False, "unclosed_runs": 1}})
    persistence = owner.extension()["session"]["persistence"]
    assert persistence["failures"] == 0
    assert persistence["unclosed_runs"] == 1
    assert not persistence["healthy"]


@pytest.mark.asyncio
async def test_close_failure_remains_observational_and_always_releases_publisher() -> None:
    owner = AcpUsageAccounting("s")
    sent = asyncio.Event()

    async def broken_close() -> None:
        raise OSError("journal storage unavailable")

    async def send(update: Any) -> None:
        sent.set()

    owner.sink = SimpleNamespace(aclose=broken_close)  # type: ignore[assignment]
    owner.start_publisher(send)
    await sent.wait()
    await owner.aclose()
    assert owner._publisher is None
    assert owner._unsubscribe is None
    assert owner.extension()["session"]["persistence"]["failures"] == 1
    await owner.aclose()
