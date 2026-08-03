"""Unit tests for kolega_code/llm/ledger.py — the UsageLedger state machine,
snapshots, checkpoint deltas, and failure contracts."""

import asyncio
from typing import Any, cast

import pytest

from kolega_code.llm.ledger import UsageLedger, UsageSnapshot
from kolega_code.llm.usage import normalize_usage


def _usage(inp=10, out=5, provider="anthropic", model="claude-opus-5"):
    return normalize_usage({"input_tokens": inp, "output_tokens": out}, provider, model)


def _unreported_usage(provider="openai", model="gpt-5.4-mini"):
    return normalize_usage({}, provider, model)


def test_happy_path_counts_and_sums():
    ledger = UsageLedger()
    r1 = ledger.begin("anthropic", "claude-opus-5")
    ledger.record_response(r1, _usage(10, 5))
    r2 = ledger.begin("anthropic", "claude-opus-5")
    ledger.record_response(r2, _usage(100, 50))

    snap = ledger.snapshot()
    assert snap.requests == 2
    assert snap.responses == 2
    assert snap.reported == 2
    assert snap.unreported == 0
    assert snap.failed == 0
    assert snap.open == 0
    assert snap.input_tokens == 110
    assert snap.output_tokens == 55
    assert snap.total_tokens == 165
    assert snap.total_tokens == snap.input_tokens + snap.output_tokens
    assert snap.complete is True
    assert snap.run_id == ledger.run_id


@pytest.mark.parametrize("first,second", [("response", "failure"), ("failure", "response"), ("response", "response")])
def test_first_settle_wins(first, second):
    ledger = UsageLedger()
    request_id = ledger.begin("anthropic", "m")

    def settle(kind):
        if kind == "response":
            ledger.record_response(request_id, _usage())
        else:
            ledger.record_failure(request_id, "boom")

    settle(first)
    settle(second)
    snap = ledger.snapshot()
    assert snap.requests == 1
    if first == "response":
        assert snap.responses == 1 and snap.failed == 0 and snap.total_tokens == 15
    else:
        assert snap.failed == 1 and snap.responses == 0 and snap.total_tokens == 0


def test_unknown_request_id_is_ignored():
    ledger = UsageLedger()
    ledger.record_response("nope", _usage())
    ledger.record_failure("nope", "boom")
    assert ledger.snapshot().requests == 0


@pytest.mark.parametrize("usage", [None, "junk", _unreported_usage()])
def test_unreported_responses_never_fabricate_zeros(usage):
    ledger = UsageLedger()
    request_id = ledger.begin("openai", "gpt-5.4-mini")
    ledger.record_response(request_id, usage)
    snap = ledger.snapshot()
    assert snap.responses == 1
    assert snap.reported == 0
    assert snap.unreported == 1
    assert snap.total_tokens == 0
    assert snap.complete is False


def test_open_requests_mark_incomplete():
    ledger = UsageLedger()
    ledger.begin("anthropic", "m")
    snap = ledger.snapshot()
    assert snap.open == 1
    assert snap.requests == 1
    assert snap.complete is False


def test_breakdown_keys_and_unknown_model():
    ledger = UsageLedger()
    # Reported: keyed by the usage's own provider/model.
    r1 = ledger.begin("anthropic", "begin-time-model")
    ledger.record_response(r1, _usage(10, 5, model="usage-model"))
    # Failed with no model: keyed by begin-time data with "unknown" fallback.
    r2 = ledger.begin("openai", None)
    ledger.record_failure(r2, "boom")

    snap = ledger.snapshot()
    rows = {(row.provider, row.model): row for row in snap.breakdown}
    assert rows[("anthropic", "usage-model")].reported == 1
    assert rows[("anthropic", "usage-model")].total_tokens == 15
    assert rows[("openai", "unknown")].failed == 1
    assert snap.breakdown == tuple(sorted(snap.breakdown, key=lambda r: (r.provider, r.model)))


def test_since_delta_and_open_semantics():
    ledger = UsageLedger()
    r1 = ledger.begin("anthropic", "m")
    ledger.record_response(r1, _usage(10, 5, model="m"))
    mark = ledger.snapshot()

    r2 = ledger.begin("anthropic", "m")
    ledger.record_response(r2, _usage(100, 50, model="m"))
    ledger.begin("anthropic", "m")  # stays open

    delta = ledger.snapshot().since(mark)
    assert delta.requests == 2
    assert delta.responses == 1
    assert delta.total_tokens == 150
    assert delta.open == 1  # current outstanding, not a difference
    rows = {(row.provider, row.model): row for row in delta.breakdown}
    assert rows[("anthropic", "m")].total_tokens == 150


def test_since_none_or_foreign_mark_yields_full_snapshot():
    ledger = UsageLedger()
    request_id = ledger.begin("anthropic", "m")
    ledger.record_response(request_id, _usage())
    snap = ledger.snapshot()
    assert snap.since(None) is snap
    foreign = UsageLedger().snapshot()
    assert snap.since(foreign) is snap


def test_since_drops_all_zero_breakdown_rows():
    ledger = UsageLedger()
    r1 = ledger.begin("anthropic", "m")
    ledger.record_response(r1, _usage())
    mark = ledger.snapshot()
    r2 = ledger.begin("google", "gemini-3.5-flash")
    ledger.record_response(
        r2, normalize_usage({"prompt_token_count": 3, "candidates_token_count": 2}, "google", "gemini-3.5-flash")
    )

    delta = ledger.snapshot().since(mark)
    keys = {(row.provider, row.model) for row in delta.breakdown}
    assert keys == {("google", "gemini-3.5-flash")}


def test_ledger_methods_never_raise_when_poisoned():
    class _Poison:
        def __setitem__(self, key, value):
            raise RuntimeError("poisoned")

        def get(self, key):
            raise RuntimeError("poisoned")

        def values(self):
            raise RuntimeError("poisoned")

    ledger = UsageLedger()
    cast(Any, ledger)._records = _Poison()

    request_id = ledger.begin("anthropic", "m")
    assert isinstance(request_id, str) and request_id
    ledger.record_response(request_id, _usage())
    ledger.record_failure(request_id, "boom")
    snap = ledger.snapshot()
    assert isinstance(snap, UsageSnapshot)
    assert snap.run_id == ledger.run_id
    assert snap.requests == 0


@pytest.mark.asyncio
async def test_concurrent_settlement_is_exact():
    ledger = UsageLedger()

    async def one_call(i: int):
        request_id = ledger.begin("anthropic", "m")
        await asyncio.sleep(0)
        ledger.record_response(request_id, _usage(10, 5))

    await asyncio.gather(*(one_call(i) for i in range(50)))
    snap = ledger.snapshot()
    assert snap.requests == 50
    assert snap.reported == 50
    assert snap.total_tokens == 50 * 15
    assert snap.complete is True
