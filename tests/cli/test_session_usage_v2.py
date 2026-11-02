import json
from dataclasses import replace
from pathlib import Path

import pytest

from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.session_usage import (
    LLM_MESSAGE_EVENT,
    LLM_REQUEST_SETTLED_EVENT,
    LLM_REQUEST_STARTED_EVENT,
    LLM_RUN_FINISHED_EVENT,
    LLM_RUN_STARTED_EVENT,
    LLM_USAGE_OBSERVED_EVENT,
    derive_session_usage,
)
from kolega_code.llm.usage import normalize_usage


def _store(tmp_path: Path):
    store = SessionStore(tmp_path / "state")
    session = store.create(tmp_path / "project", "code", {})
    return store, session, store.journal(session.session_id)


def _marker(journal, run_id="run", *, version=2):
    return journal.append(
        LLM_RUN_STARTED_EVENT,
        actor="system",
        payload={"run_id": run_id, "mode": "acp", "accounting_version": version},
    )


def _fact(journal, event_type=LLM_REQUEST_SETTLED_EVENT, *, request_id="request", run_id="run", **extra):
    return journal.append(
        event_type,
        actor="system",
        payload={
            "run_id": run_id,
            "request_id": request_id,
            "provider": "anthropic",
            "model": "model",
            "revision": 0,
            "state": "responded",
            **extra,
        },
    )


def _usage(inp=10, out=5):
    return normalize_usage({"input_tokens": inp, "output_tokens": out}, "anthropic", "model").to_dict()


def _cost(amount="0.5", *, source="provider_reported", currency: str | None = "USD", **extra):
    return {"source": source, "amount": amount, "currency": currency, "complete": True, **extra}


def _finish(journal, count=1, run_id="run", **extra):
    return journal.append(
        LLM_RUN_FINISHED_EVENT,
        actor="system",
        payload={"run_id": run_id, "counts": {"requests": count}, "persist_failures": 0, **extra},
    )


def test_compact_facts_win_over_duplicate_facts_and_history_and_helper_messages(tmp_path):
    store, session, journal = _store(tmp_path)
    _marker(journal)
    _fact(journal, LLM_REQUEST_STARTED_EVENT, state="open")
    settled = _fact(journal, usage=_usage(), cost=_cost())
    duplicate = replace(settled, event_id="duplicate", seq=settled.seq + 1)
    for event_type in ("assistant.message", LLM_MESSAGE_EVENT):
        journal.append(
            event_type,
            actor="assistant",
            payload={
                "run_id": "run",
                "llm_call_id": "request",
                "message": {"usage": _usage(1000, 1000)},
            },
        )
    _finish(journal)
    events = journal.read_events()
    result = derive_session_usage([*events, settled, duplicate])
    assert result["requests"] == result["responses"] == result["reported"] == 1
    assert result["total_tokens"] == 15
    assert result["coverage"]["full"] is True
    assert result["costs"]["currencies"]["USD"]["total"] == "0.5"
    assert result["costs"]["complete"] is True
    assert store.load(session.session_id).usage["total_tokens"] == 15


def test_late_evidence_replaces_partial_and_cost_without_replacing_failure(tmp_path):
    _, _, journal = _store(tmp_path)
    _marker(journal)
    _fact(
        journal,
        state="failed",
        revision=1,
        observation={"input_tokens": 9, "complete": False},
        cost=_cost("0.1", source="estimated", complete=False),
    )
    partial = derive_session_usage(journal.read_events(), live_run_id="run")
    assert partial["failed"] == partial["partial"] == 1
    assert partial["input_tokens"] == 9
    assert partial["known_requests"]["output_tokens"] == 0
    _fact(
        journal,
        LLM_USAGE_OBSERVED_EVENT,
        state="responded",
        revision=3,
        observation={**_usage(12, 7), "complete": True},
        cost=_cost("0.4"),
    )
    _fact(journal, LLM_USAGE_OBSERVED_EVENT, revision=2, usage=_usage(100, 100), cost=_cost("99"))
    _finish(journal)
    result = derive_session_usage(journal.read_events())
    assert result["requests"] == result["failed"] == result["reported"] == 1
    assert result["responses"] == result["partial"] == result["missing"] == 0
    assert result["total_tokens"] == 19
    assert result["complete"] is True
    assert result["costs"]["currencies"]["USD"] == {
        "reported": "0.4",
        "estimated": "0",
        "total": "0.4",
        "requests": 1,
        "reported_requests": 1,
        "estimated_requests": 0,
    }


def test_explicit_zero_and_missing_tokens_and_optional_details_are_distinct(tmp_path):
    _, _, journal = _store(tmp_path)
    _marker(journal)
    _fact(journal, request_id="zero", usage=_usage(0, 0), cost=_cost("0"))
    _fact(journal, request_id="missing", usage=None, cost=_cost("1"))
    _finish(journal, count=2)
    result = derive_session_usage(journal.read_events())
    assert result["requests"] == 2
    assert result["reported"] == result["missing"] == 1
    assert result["total_tokens"] == 0
    assert result["known_requests"]["total_tokens"] == 1
    assert result["known_requests"]["cache_read_input_tokens"] == 0
    assert result["detail_complete"]["cache_read_input_tokens"] is False
    assert result["complete"] is False
    assert result["costs"]["complete"] is True


def test_open_live_requests_become_abandoned_on_resume_and_run_is_uncertain(tmp_path):
    store, session, journal = _store(tmp_path)
    _marker(journal)
    _fact(journal, LLM_REQUEST_STARTED_EVENT, state="open")
    live = store.load(session.session_id, live_run_id="run").usage
    assert live["open"] == 1
    assert live["abandoned"] == live["coverage"]["unclosed_runs"] == 0
    resumed = store.load(session.session_id).usage
    assert resumed["open"] == 0
    assert resumed["abandoned"] == resumed["coverage"]["unclosed_runs"] == 1
    assert resumed["coverage"]["full"] is False
    assert resumed["persistence"]["healthy"] is False


def test_unclosed_v2_run_is_uncertain_even_if_all_queued_starts_were_lost(tmp_path):
    _, _, journal = _store(tmp_path)
    _marker(journal)
    result = derive_session_usage(journal.read_events())
    assert result["requests"] == 0
    assert result["coverage"]["unclosed_runs"] == 1
    assert result["complete"] is result["coverage"]["full"] is False
    _finish(journal, count=2)
    result = derive_session_usage(journal.read_events())
    assert result["coverage"]["lost_requests"] == 2
    assert result["persistence"]["healthy"] is False


def test_matching_request_ids_in_different_runs_are_distinct(tmp_path):
    _, _, journal = _store(tmp_path)
    for run_id in ("a", "b"):
        _marker(journal, run_id)
        _fact(journal, run_id=run_id, usage=_usage())
        _finish(journal, run_id=run_id)
    result = derive_session_usage(journal.read_events())
    assert result["requests"] == 2
    assert result["total_tokens"] == 30


def test_legacy_call_ids_deduplicate_before_weak_message_and_event_identity(tmp_path):
    _, _, journal = _store(tmp_path)
    for run_id in ("a", "b"):
        for event_type in ("assistant.message", LLM_MESSAGE_EVENT):
            journal.append(
                event_type,
                actor="assistant",
                payload={
                    "run_id": run_id,
                    "llm_call_id": "legacy",
                    "message": {"usage": _usage()},
                },
            )
    for _ in range(2):
        journal.append(
            "assistant.message",
            actor="assistant",
            payload={"message_id": "same-message", "message": {"usage": _usage(1, 1)}},
        )
    result = derive_session_usage(journal.read_events())
    assert result["requests"] == 3
    assert result["total_tokens"] == 32
    assert result["coverage"]["full"] is False
    assert result["coverage"]["legacy_requests"] == 3


@pytest.mark.parametrize(
    "bad",
    [
        {"request_id": ""},
        {"run_id": []},
        {"revision": -1},
        {"revision": True},
        {"revision": "2"},
    ],
)
def test_malformed_fact_identities_and_revisions_do_not_create_requests(tmp_path, bad):
    _, _, journal = _store(tmp_path)
    _marker(journal)
    _fact(journal, usage=_usage(), **bad)
    _finish(journal, count=0)
    result = derive_session_usage(journal.read_events())
    assert result["requests"] == 0
    assert result["coverage"]["malformed_facts"] == 1
    assert result["coverage"]["full"] is False


@pytest.mark.parametrize("amount", [True, "-1", "NaN", "Infinity", "no"])
def test_invalid_charges_and_tokens_are_not_known_zeros(tmp_path, amount):
    _, _, journal = _store(tmp_path)
    _marker(journal)
    _fact(
        journal,
        observation={"input_tokens": True, "output_tokens": -2, "total_tokens": "0"},
        cost=_cost(amount),
    )
    _finish(journal)
    result = derive_session_usage(journal.read_events())
    assert result["missing"] == 1
    assert result["known_requests"]["total_tokens"] == 0
    assert result["costs"]["missing_requests"] == 1
    assert result["costs"]["currencies"] == {}


def test_currency_and_nonmonetary_denominations_are_never_added_together(tmp_path):
    _, _, journal = _store(tmp_path)
    _marker(journal)
    for request_id, cost in (
        ("usd", _cost("1.25")),
        ("eur", _cost("2.5", currency="EUR")),
        ("credits", _cost("3", currency=None, denomination="provider_credits")),
        ("partial", _cost("0.2", source="estimated", complete=False)),
    ):
        _fact(journal, request_id=request_id, cost=cost, usage=_usage())
    _finish(journal, count=4)
    result = derive_session_usage(journal.read_events())
    assert result["costs"]["currencies"]["USD"]["total"] == "1.45"
    assert result["costs"]["currencies"]["EUR"]["total"] == "2.5"
    assert result["costs"]["non_monetary"]["provider_credits"]["total"] == "3"
    assert result["costs"]["partial_requests"] == 1
    assert result["costs"]["complete"] is False


def test_closed_v2_run_followed_by_uninstrumented_and_legacy_runs_keeps_gaps(tmp_path):
    _, _, journal = _store(tmp_path)
    _marker(journal)
    _fact(journal, usage=_usage())
    _finish(journal)
    journal.append("turn.started", actor="user", turn_id="uninstrumented", payload={})
    journal.append(
        "assistant.message",
        actor="assistant",
        turn_id="uninstrumented",
        payload={"message": {"usage": _usage()}},
    )
    _marker(journal, run_id="legacy", version=1)
    journal.append("turn.started", actor="user", turn_id="legacy", payload={})
    _marker(journal, run_id="new")
    _finish(journal, run_id="new", count=0)
    result = derive_session_usage(journal.read_events())
    assert result["total_tokens"] == 30
    assert result["coverage"]["uninstrumented_turns"] == 1
    assert result["coverage"]["legacy_turns"] == 1
    assert result["coverage"]["full"] is False


def test_rewinds_epochs_exports_and_metadata_do_not_reprice_or_subtract_facts(tmp_path):
    store, session, journal = _store(tmp_path)
    _marker(journal)
    _fact(journal, usage=_usage(), cost=_cost("0.123"), pinned_pricing={"pricing_id": "original"})
    _finish(journal)
    journal.append("history.rewound", actor="system", payload={"target_turn_id": "old"})
    journal.start_epoch("reset")
    _marker(journal, run_id="next")
    _fact(journal, run_id="next", usage=_usage(2, 1), cost=_cost("0.456"))
    _finish(journal, run_id="next")
    baseline = store.load(session.session_id).usage
    assert baseline["total_tokens"] == 18
    assert baseline["costs"]["currencies"]["USD"]["total"] == "0.579"
    record = store.load(session.session_id)
    record.config["pricing_overrides"] = {"unrelated": "new"}
    store.save(record)
    assert store.load(session.session_id).usage == baseline
    assert json.loads(store.export(session.session_id))["usage"] == baseline
    assert json.loads(store.bug_export(session.session_id).session_json)["usage"] == baseline
    assert "usage" not in json.loads(store.path_for(session.session_id).read_text())


def test_frozen_baseline_does_not_change_when_a_new_run_is_attached(tmp_path):
    store, session, journal = _store(tmp_path)
    _marker(journal)
    _fact(journal, usage=_usage())
    _finish(journal)
    baseline = store.load(session.session_id).usage
    _marker(journal, run_id="new")
    _fact(journal, run_id="new", usage=_usage(100, 100))
    assert baseline["total_tokens"] == 15
    assert store.load(session.session_id).usage["total_tokens"] == 215


def test_new_empty_session_is_known_empty_not_a_legacy_gap(tmp_path):
    store, session, _ = _store(tmp_path)
    result = store.load(session.session_id).usage
    assert result["requests"] == result["total_tokens"] == 0
    assert result["complete"] is result["coverage"]["full"] is result["costs"]["complete"] is True


def test_newer_usage_only_fact_supersedes_older_partial_observation(tmp_path):
    _, _, journal = _store(tmp_path)
    _marker(journal)
    _fact(journal, revision=1, observation={"input_tokens": 1, "complete": False})
    _fact(journal, LLM_USAGE_OBSERVED_EVENT, revision=2, usage=_usage(10, 5))
    _finish(journal)
    result = derive_session_usage(journal.read_events())
    assert result["total_tokens"] == 15
    assert result["partial"] == 0


def test_legacy_call_identity_without_run_deduplicates_the_unique_scoped_call(tmp_path):
    _, _, journal = _store(tmp_path)
    for run_id in (None, "legacy-run"):
        journal.append(
            "assistant.message",
            actor="assistant",
            payload={"run_id": run_id, "llm_call_id": "same-call", "message": {"usage": _usage()}},
        )
    result = derive_session_usage(journal.read_events())
    assert result["requests"] == 1
    assert result["total_tokens"] == 15
    assert result["coverage"]["full"] is False


def test_malformed_close_cannot_certify_a_crashed_empty_run(tmp_path):
    _, _, journal = _store(tmp_path)
    _marker(journal)
    _finish(journal, count=True)
    result = derive_session_usage(journal.read_events())
    assert result["coverage"]["malformed_facts"] == 1
    assert result["coverage"]["unclosed_runs"] == 1
    assert result["coverage"]["full"] is False
