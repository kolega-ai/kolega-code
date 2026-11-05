"""Final-write outcomes must follow settled disk evidence, not queue completion."""

import asyncio
from pathlib import Path
import subprocess
import sys
import textwrap
import threading
from typing import Any

import pytest

from kolega_code.cli.session_event_protocol import InMemorySessionJournal
from kolega_code.cli.session_journal import SessionEvent, SessionJournal, SessionRecorder
from kolega_code.cli.session_store import SessionRecord, SessionStore
from kolega_code.cli.session_usage import (
    LLM_REQUEST_SETTLED_EVENT,
    LLM_RUN_FINISHED_EVENT,
    LLM_RUN_STARTED_EVENT,
    SessionUsageSink,
    derive_session_usage,
)
from kolega_code.llm.ledger import UsageLedger
from kolega_code.llm.usage import normalize_usage


async def _sink(tmp_path: Path) -> tuple[SessionStore, SessionRecord, SessionJournal, UsageLedger, SessionUsageSink]:
    store = SessionStore(tmp_path / "state")
    session = store.create(tmp_path / "project", "code", {})
    journal = store.journal(session.session_id)
    ledger = UsageLedger()
    sink = SessionUsageSink(journal, store.recorder(session.session_id), ledger, mode="tui")
    ledger.observer = sink
    await sink.start()
    return store, session, journal, ledger, sink


def _response(ledger: UsageLedger) -> None:
    request = ledger.begin("anthropic", "example")
    ledger.record_response(request, normalize_usage({"input_tokens": 10, "output_tokens": 5}, "anthropic", "example"))


@pytest.mark.asyncio
async def test_committed_finish_error_is_reconciled_without_erasing_io_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, session, journal, ledger, sink = await _sink(tmp_path)
    _response(ledger)
    append = journal.append
    captured: list[bytes] = []

    def commit_then_raise(event_type: str, **kwargs: Any) -> SessionEvent:
        result = append(event_type, **kwargs)
        if event_type == LLM_RUN_FINISHED_EVENT:
            captured.append(journal.events_path.read_bytes())
            raise OSError("close failed after full write")
        return result

    monkeypatch.setattr(journal, "append", commit_then_raise)
    result = await sink.aclose()
    assert result.status == "committed"
    assert sink.persistence_failures == ledger.persist_failures == 0
    assert sink.io_errors == 1
    assert journal.events_path.read_bytes() == captured[0]
    replay = store.load(session.session_id).usage
    assert replay["coverage"]["full"] is replay["persistence"]["healthy"] is True
    assert replay["persist_failures"] == sink.persistence_failures
    assert len([event for event in journal.read_events() if event.event_type == LLM_RUN_FINISHED_EVENT]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("late_failure", [False, True])
async def test_timeout_and_cancel_retain_one_actual_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, late_failure: bool
) -> None:
    _, _, journal, ledger, sink = await _sink(tmp_path)
    _response(ledger)
    entered, release = threading.Event(), threading.Event()
    append = journal.append
    attempts: list[dict[str, Any]] = []

    def slow_finish(event_type: str, **kwargs: Any) -> SessionEvent:
        if event_type == LLM_RUN_FINISHED_EVENT:
            attempts.append(kwargs["payload"])
            entered.set()
            assert release.wait(10)
            if late_failure:
                raise OSError("no finish committed")
        return append(event_type, **kwargs)

    monkeypatch.setattr(journal, "append", slow_finish)
    closing = asyncio.create_task(sink.aclose(timeout=None))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        pending = await sink.aclose(timeout=0)
        assert pending.status == "pending"
        assert sink._drainer is not None and not sink._drainer.done()
        second = asyncio.create_task(sink.aclose(timeout=None))
        await asyncio.sleep(0)
        assert not second.done()
    finally:
        release.set()
    result = await second
    assert result.status == ("not_committed" if late_failure else "committed")
    assert len(attempts) == 1
    assert await sink.aclose() == result
    assert sink._drainer.done() and not sink._drainer.cancelled()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["read_error", "torn_tail", "conflict"])
async def test_uncertain_finish_never_repairs_or_certifies_unknown_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    _, _, journal, ledger, sink = await _sink(tmp_path)
    _response(ledger)
    append, read_events = journal.append, journal.read_events
    raw_after_fault: list[bytes] = []

    def uncertain_finish(event_type: str, **kwargs: Any) -> SessionEvent:
        result = append(event_type, **kwargs)
        if event_type == LLM_RUN_FINISHED_EVENT:
            if fault == "torn_tail":
                with journal.events_path.open("ab") as output:
                    output.write(b'{"event_id":"torn')
            elif fault == "conflict":
                append(event_type, **{**kwargs, "payload": {**kwargs["payload"], "persist_failures": 3}})
            raw_after_fault.append(journal.events_path.read_bytes())
            raise OSError("ambiguous finish")
        return result

    def failed_read(*, repair_tail: bool = True) -> list[SessionEvent]:
        assert repair_tail is False
        raise OSError("recovery read failed")

    monkeypatch.setattr(journal, "append", uncertain_finish)
    if fault == "read_error":
        monkeypatch.setattr(journal, "read_events", failed_read)
    result = await sink.aclose()
    assert result.status == "unknown"
    assert sink.persistence_failures > 0
    assert journal.events_path.read_bytes() == raw_after_fault[0]
    monkeypatch.setattr(journal, "read_events", read_events)


@pytest.mark.asyncio
async def test_reconciled_finish_preserves_prior_missing_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, journal, ledger, sink = await _sink(tmp_path)
    append = journal.append

    def faults(event_type: str, **kwargs: Any) -> SessionEvent:
        if event_type == LLM_REQUEST_SETTLED_EVENT:
            raise OSError("lost settlement")
        result = append(event_type, **kwargs)
        if event_type == LLM_RUN_FINISHED_EVENT:
            raise OSError("finish committed")
        return result

    monkeypatch.setattr(journal, "append", faults)
    _response(ledger)
    result = await sink.aclose()
    replay = derive_session_usage(journal.read_events())
    assert result.committed
    assert ledger.persist_failures == replay["persist_failures"] == 1
    assert replay["coverage"]["full"] is replay["persistence"]["healthy"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("lose_settlement", [False, True])
async def test_handoff_finish_uses_sink_local_identity_and_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lose_settlement: bool
) -> None:
    store, session, journal, ledger, first = await _sink(tmp_path)
    append = journal.append

    def fail_settlement(event_type: str, **kwargs: Any) -> SessionEvent:
        if event_type == LLM_REQUEST_SETTLED_EVENT:
            raise OSError("lost settlement")
        return append(event_type, **kwargs)

    with monkeypatch.context() as faults:
        if lose_settlement:
            faults.setattr(journal, "append", fail_settlement)
        _response(ledger)
        await first.aclose()
    second = SessionUsageSink(journal, store.recorder(session.session_id), ledger, mode="tui")
    ledger.observer = second
    await second.start()
    _response(ledger)
    await second.aclose()
    finishes = [event.payload for event in journal.read_events() if event.event_type == LLM_RUN_FINISHED_EVENT]
    assert len({payload["sink_id"] for payload in finishes}) == 2
    assert [payload["counts"]["requests"] for payload in finishes] == [1, 1]
    replay = derive_session_usage(journal.read_events())
    assert replay["requests"] == 2
    assert replay["coverage"]["accounted_runs"] == 2
    assert replay["persist_failures"] == ledger.persist_failures == int(lose_settlement)
    assert replay["coverage"]["full"] is (not lose_settlement)


@pytest.mark.asyncio
@pytest.mark.parametrize("conflict", [False, True])
async def test_replay_identical_finish_is_idempotent_but_conflict_is_not_last_wins(
    tmp_path: Path, conflict: bool
) -> None:
    _, _, journal, ledger, sink = await _sink(tmp_path)
    _response(ledger)
    await sink.aclose()
    finish = next(event.payload for event in journal.read_events() if event.event_type == LLM_RUN_FINISHED_EVENT)
    journal.append(
        LLM_RUN_FINISHED_EVENT,
        actor="system",
        payload={**finish, **({"persist_failures": 2} if conflict else {})},
    )
    replay = derive_session_usage(journal.read_events())
    assert replay["requests"] == 1
    assert replay["coverage"]["full"] is (not conflict)
    assert replay["persistence"]["healthy"] is (not conflict)
    assert replay["persist_failures"] == (2 if conflict else 0)
    if conflict:
        journal.append(LLM_RUN_FINISHED_EVENT, actor="system", payload=finish)
        assert derive_session_usage(journal.read_events())["persist_failures"] == 2


@pytest.mark.asyncio
async def test_later_sink_cannot_hide_prior_missing_finish_or_requests(tmp_path: Path) -> None:
    _, _, journal, ledger, first = await _sink(tmp_path)
    await first.aclose()
    finish = next(event.payload for event in journal.read_events() if event.event_type == LLM_RUN_FINISHED_EVENT)
    journal.append(LLM_RUN_FINISHED_EVENT, actor="system", payload={**finish, "counts": {"requests": 3}})
    second = SessionUsageSink(journal, first._recorder, ledger, mode="tui")
    ledger.observer = second
    await second.start()
    # An older unclosed boundary is not the new live sink just because the run ID matches.
    journal.append(
        LLM_RUN_STARTED_EVENT,
        actor="system",
        payload={"run_id": ledger.run_id, "sink_id": "prior-unclosed", "accounting_version": 2},
    )
    third = SessionUsageSink(journal, first._recorder, ledger, mode="tui")
    await third.start()
    replay = derive_session_usage(journal.read_events(), live_run_id=ledger.run_id)
    assert replay["coverage"]["unclosed_runs"] == 2
    assert replay["coverage"]["lost_requests"] == 3
    assert replay["coverage"]["full"] is False
    await second.aclose()
    await third.aclose()


@pytest.mark.asyncio
async def test_cancelled_start_is_still_owned_by_close(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = SessionStore(tmp_path / "state")
    session = store.create(tmp_path / "project", "code", {})
    journal = store.journal(session.session_id)
    sink = SessionUsageSink(journal, store.recorder(session.session_id), UsageLedger(), mode="tui")
    entered, release, settled = threading.Event(), threading.Event(), threading.Event()
    append = journal.append

    def slow_start(event_type: str, **kwargs: Any) -> SessionEvent:
        if event_type == LLM_RUN_STARTED_EVENT:
            entered.set()
            assert release.wait(10)
        try:
            return append(event_type, **kwargs)
        finally:
            settled.set()

    monkeypatch.setattr(journal, "append", slow_start)
    starting = asyncio.create_task(sink.start())
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        starting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await starting
        closing = asyncio.create_task(sink.wait_closed())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not closing.done()
    finally:
        release.set()
        assert await asyncio.to_thread(settled.wait, 5)
    assert (await closing).committed
    assert [event.event_type for event in journal.read_events()][-2:] == [
        LLM_RUN_STARTED_EVENT,
        LLM_RUN_FINISHED_EVENT,
    ]


@pytest.mark.asyncio
async def test_in_memory_journal_finish_reconciliation_needs_no_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    journal = InMemorySessionJournal("unsaved")
    journal.start_epoch("created")
    recorder = SessionRecorder(journal)
    ledger = UsageLedger()
    sink = SessionUsageSink(journal, recorder, ledger, mode="ask")
    ledger.observer = sink
    await sink.start()
    append = journal.append

    def commit_then_raise(event_type: str, **kwargs: Any) -> SessionEvent:
        result = append(event_type, **kwargs)
        if event_type == LLM_RUN_FINISHED_EVENT:
            raise OSError("committed in memory")
        return result

    monkeypatch.setattr(journal, "append", commit_then_raise)
    assert (await sink.aclose()).committed
    assert sink.persistence_failures == 0
    assert not journal.events_path.exists()


@pytest.mark.asyncio
async def test_finish_preparation_failure_settles_drainer_and_terminal_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, journal, ledger, sink = await _sink(tmp_path)

    def failed_reload(*, repair_tail: bool) -> list[SessionEvent]:
        raise OSError("journal reload failed")

    journal._loaded = False
    monkeypatch.setattr(journal, "_read_events_locked", failed_reload)
    result = await sink.wait_closed()
    assert result.status == "unknown"
    assert sink.close_result == result
    assert ledger.persist_failures == 1
    assert sink._drainer is not None and sink._drainer.done()
    assert await sink.aclose() == result


@pytest.mark.parametrize(
    "close_phase", ["active", "before_entry", "drainer", "idle", "startup", "startup_close", "startup_aclose"]
)
def test_runner_teardown_settles_actual_writer_without_cancelled_task_spin(tmp_path: Path, close_phase: str) -> None:
    code = textwrap.dedent(
        """
        import asyncio
        from pathlib import Path
        import sys
        import threading

        from kolega_code.cli.session_store import SessionStore
        from kolega_code.cli.session_usage import (
            LLM_REQUEST_STARTED_EVENT, LLM_RUN_FINISHED_EVENT, LLM_RUN_STARTED_EVENT, SessionUsageSink
        )
        from kolega_code.llm.ledger import UsageLedger

        root = Path(sys.argv[1])
        phase = sys.argv[2]
        release, entered, returned = threading.Event(), threading.Event(), threading.Event()
        closed = []
        sinks = []

        async def main():
            store = SessionStore(root / "state")
            session = store.create(root / "project", "code", {})
            journal = store.journal(session.session_id)
            ledger = UsageLedger()
            sink = SessionUsageSink(journal, store.recorder(session.session_id), ledger, mode="tui")
            ledger.observer = sink
            sinks.append(sink)
            append = journal.append

            def blocked(event_type, **kwargs):
                blocked_type = LLM_REQUEST_STARTED_EVENT if phase == "drainer" else LLM_RUN_FINISHED_EVENT
                if phase.startswith("startup"):
                    blocked_type = LLM_RUN_STARTED_EVENT
                    if event_type == LLM_RUN_FINISHED_EVENT:
                        threading.Event().wait(0.1)
                if event_type != blocked_type:
                    return append(event_type, **kwargs)
                entered.set()
                try:
                    assert release.wait(10)
                    return append(event_type, **kwargs)
                finally:
                    returned.set()

            journal.append = blocked

            async def close():
                result = await sink.wait_closed()
                assert result.status != "pending"
                assert not entered.is_set() or returned.is_set()
                assert sink._drainer.done()
                closed.append(result.status)

            if phase.startswith("startup"):
                asyncio.create_task(sink.start())
                assert await asyncio.to_thread(entered.wait, 5)
                if phase == "startup_close":
                    asyncio.create_task(close())
                    await asyncio.sleep(0)
                elif phase == "startup_aclose":
                    asyncio.create_task(sink.aclose(timeout=None))
                threading.Timer(0.05, release.set).start()
                return
            await sink.start()
            if phase in ("drainer", "idle"):
                if phase == "drainer":
                    ledger.begin("openrouter", "example/model")
                    assert await asyncio.to_thread(entered.wait, 5)
                threading.Timer(0.05, release.set).start()
                return
            closing = asyncio.create_task(close())
            if phase == "before_entry":
                await asyncio.sleep(0)
                assert sink._close_task is not None
                sink._close_task.cancel()
            else:
                assert await asyncio.to_thread(entered.wait, 5)
            threading.Timer(0.05, release.set).start()

        asyncio.run(main())
        sink = sinks[0]
        assert sink.close_result is not None
        assert sink.close_result.status != "pending"
        assert not entered.is_set() or returned.is_set()
        assert sink._drainer.done()
        assert sink._close_task.done()
        assert sink._start_task.done()
        if phase in ("drainer", "idle", "startup"):
            assert sink.close_result.committed
            assert sum(event.event_type == LLM_RUN_FINISHED_EVENT for event in sink._journal.read_events()) == 1
        elif phase != "startup_aclose":
            assert len(closed) == 1, closed
        print("SETTLED", sink.close_result.status)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), close_phase],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SETTLED" in result.stdout
    assert "Task was destroyed" not in result.stderr
