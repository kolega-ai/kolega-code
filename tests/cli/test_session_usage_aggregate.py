"""derive_session_usage + SessionRecord.usage through load/export/bug export."""

import json
from pathlib import Path

from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.session_usage import (
    LLM_MESSAGE_EVENT,
    LLM_REQUEST_FAILED_EVENT,
    LLM_RUN_STARTED_EVENT,
    derive_session_usage,
)
from kolega_code.llm.models import Message, TextBlock
from kolega_code.llm.usage import normalize_usage


def _usage_dict(inp=10, out=5, provider="anthropic", model="claude-opus-5"):
    return normalize_usage({"input_tokens": inp, "output_tokens": out}, provider, model).to_dict()


def _assistant_dict(inp=10, out=5):
    message = Message(role="assistant", content=[TextBlock(text="ok")])
    data = message.to_dict()
    data["usage"] = _usage_dict(inp, out)
    return data


def _store(tmp_path: Path):
    store = SessionStore(tmp_path / "state")
    session = store.create(tmp_path / "proj", "code", {})
    return store, session


def _run_turn(recorder, inp=10, out=5):
    recorder.start_turn(Message(role="user", content=[TextBlock(text="hi")]))
    message = Message.from_dict(_assistant_dict(inp, out))
    recorder.record_assistant(message)
    recorder.finish_turn("completed")


def _append_marker(journal, run_id="run1"):
    journal.append(LLM_RUN_STARTED_EVENT, actor="system", payload={"run_id": run_id, "mode": "tui"})


def _append_llm_message(journal, *, usage=None, provider="anthropic", model="m", request_id="r1"):
    payload = {
        "request_id": request_id,
        "run_id": "run1",
        "provider": provider,
        "model": model,
        "origin": {"kind": "helper", "helper": "compression"},
        "message": {"role": "assistant", "content": "x", "usage": usage},
    }
    journal.append(LLM_MESSAGE_EVENT, actor="assistant", payload=payload)


def test_pre_pr3_journal_reports_partial_coverage_not_zero_authority(tmp_path):
    """A session written entirely by the pre-accounting recorder resumes with an
    aggregate that admits its own incompleteness."""
    store, session = _store(tmp_path)
    recorder = store.recorder(session.session_id)
    _run_turn(recorder)
    _run_turn(recorder)

    record = store.load(session.session_id)
    assert record.usage["responses"] == 0
    assert record.usage["total_tokens"] == 0
    assert record.usage["coverage"] == {"accounted_runs": 0, "pre_accounting_turns": 2, "full": False}
    # Resume is unchanged: full history replays.
    assert len(record.history) == 4


def test_post_marker_assistant_messages_counted_pre_marker_excluded(tmp_path):
    store, session = _store(tmp_path)
    recorder = store.recorder(session.session_id)
    journal = store.journal(session.session_id)

    _run_turn(recorder, inp=100, out=1)  # pre-marker: excluded from sums
    _append_marker(journal)
    _run_turn(recorder, inp=10, out=5)  # post-marker: counted

    usage = store.load(session.session_id).usage
    assert usage["responses"] == 1
    assert usage["total_tokens"] == 15
    assert usage["coverage"] == {"accounted_runs": 1, "pre_accounting_turns": 1, "full": False}


def test_rewound_and_prior_epoch_usage_still_counted(tmp_path):
    """Billing is never reversed: rewinds and thread resets don't subtract."""
    store, session = _store(tmp_path)
    recorder = store.recorder(session.session_id)
    journal = store.journal(session.session_id)
    _append_marker(journal)

    _run_turn(recorder, inp=10, out=5)
    turn_to_rewind = recorder.list_rewindable_turns()[-1]
    _append_llm_message(journal, usage=_usage_dict(100, 50))
    recorder.record_rewind(turn_to_rewind.turn_id)
    recorder.start_epoch("thread_reset")
    _run_turn(recorder, inp=1, out=1)

    record = store.load(session.session_id)
    # History only has the new epoch's turn...
    assert len(record.history) == 2
    # ...but usage keeps everything: rewound turn + llm.message + new turn.
    assert record.usage["responses"] == 3
    assert record.usage["total_tokens"] == 15 + 150 + 2


def test_two_markers_and_failures_and_unreported(tmp_path):
    store, session = _store(tmp_path)
    journal = store.journal(session.session_id)
    _append_marker(journal, "run1")
    _append_llm_message(journal, usage=_usage_dict(10, 5), request_id="r1")
    _append_llm_message(journal, usage=None, provider="openai", model="gpt", request_id="r2")
    _append_llm_message(journal, usage={"junk": True}, provider="openai", model="gpt", request_id="r3")
    journal.append(
        LLM_REQUEST_FAILED_EVENT,
        actor="system",
        payload={
            "request_id": "r4",
            "run_id": "run1",
            "provider": "xai",
            "model": None,
            "error": "boom",
            "origin": {"kind": "history"},
        },
    )
    _append_marker(journal, "run2")

    usage = derive_session_usage(journal.read_events())
    assert usage["requests"] == 4
    assert usage["responses"] == 3
    assert usage["reported"] == 1
    assert usage["unreported"] == 2
    assert usage["failed"] == 1
    assert usage["total_tokens"] == 15
    assert usage["coverage"]["accounted_runs"] == 2
    keys = [(row["provider"], row["model"]) for row in usage["breakdown"]]
    assert keys == sorted(keys)
    assert ("xai", "unknown") in keys


def test_detail_counters_track_missing_optional_fields(tmp_path):
    store, session = _store(tmp_path)
    journal = store.journal(session.session_id)
    _append_marker(journal)
    with_cache = normalize_usage(
        {"input_tokens": 5, "output_tokens": 5, "cache_read_input_tokens": 3, "cache_write_input_tokens": 1},
        "anthropic",
        "m",
    ).to_dict()
    _append_llm_message(journal, usage=with_cache, request_id="r1")
    _append_llm_message(journal, usage=_usage_dict(), request_id="r2")

    usage = derive_session_usage(journal.read_events())
    assert usage["detail"] == {
        "reported_missing_cache_read": 1,
        "reported_missing_cache_write": 1,
        "reported_missing_reasoning": 2,
    }
    assert usage["cache_read_input_tokens"] == 3


def test_exports_carry_usage_and_metadata_never_does(tmp_path):
    store, session = _store(tmp_path)
    journal = store.journal(session.session_id)
    _append_marker(journal)
    _append_llm_message(journal, usage=_usage_dict(10, 5))

    exported = json.loads(store.export(session.session_id))
    assert exported["usage"]["total_tokens"] == 15

    bug = store.bug_export(session.session_id)
    assert json.loads(bug.session_json)["usage"]["total_tokens"] == 15

    record = store.load(session.session_id)
    store.save(record)  # metadata-only save cycle
    metadata = json.loads((store.session_dir_for(session.session_id) / "metadata.json").read_text())
    assert "usage" not in metadata


def test_listing_does_not_compute_usage(tmp_path):
    store, session = _store(tmp_path)
    journal = store.journal(session.session_id)
    _append_marker(journal)
    _append_llm_message(journal, usage=_usage_dict())

    listed = [record for record in store.list() if record.session_id == session.session_id]
    assert listed and listed[0].usage == {}


def test_legacy_record_dict_without_usage_key_loads(tmp_path):
    from kolega_code.cli.session_store import SessionRecord

    store, session = _store(tmp_path)
    data = store.load(session.session_id).to_dict()
    data.pop("usage")
    restored = SessionRecord.from_dict(data)
    assert restored.usage == {}
