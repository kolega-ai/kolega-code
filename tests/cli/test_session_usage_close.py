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
@pytest.mark.parametrize("explicit_close", [False, True])
async def test_owned_startup_cancellation_does_not_wait_for_its_live_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit_close: bool
) -> None:
    store = SessionStore(tmp_path / "state")
    session = store.create(tmp_path / "project", "code", {})
    journal = store.journal(session.session_id)
    ledger = UsageLedger()
    entered, release = threading.Event(), threading.Event()
    begin = asyncio.Event()
    append = journal.append

    def held_start(event_type: str, **kwargs: Any) -> SessionEvent:
        if event_type == LLM_RUN_STARTED_EVENT:
            entered.set()
            assert release.wait(15)
        return append(event_type, **kwargs)

    async def produce() -> None:
        await begin.wait()
        try:
            await sink.start()
            assert not sink._closed
            request = ledger.begin("anthropic", "offline")
            ledger.record_failure(request, "settled after startup cancellation")
        finally:
            if explicit_close:
                assert (await sink.wait_closed()).committed

    monkeypatch.setattr(journal, "append", held_start)
    owner = asyncio.create_task(produce())
    sink = SessionUsageSink(journal, store.recorder(session.session_id), ledger, mode="ask", owner_task=owner)
    ledger.observer = sink
    begin.set()
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        assert sink._start_task is not None
        sink._start_task.cancel()
        release.set()
        await asyncio.wait_for(asyncio.shield(owner), timeout=10)
    finally:
        release.set()
        if not owner.done():
            owner.cancel()
        await asyncio.gather(owner, return_exceptions=True)
        result = await sink.wait_closed()
    assert result.committed
    assert sink._start_task is not None
    await asyncio.wait_for(asyncio.shield(sink._start_task), timeout=10)
    assert sink._start_task.done() and sink._drainer is not None and sink._drainer.done()
    events = journal.read_events()
    assert [event.event_type for event in events if event.event_type.startswith("llm.")] == [
        "llm.run_started",
        "llm.request_started",
        "llm.request_settled",
        "llm.request_failed",
        "llm.run_finished",
    ]
    assert events[-1].payload["counts"] == {
        "requests": 1,
        "responses": 0,
        "failed": 1,
        "open": 0,
        "abandoned": 0,
        "partial": 0,
        "missing": 1,
    }


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


@pytest.mark.parametrize("producer_first", [False, True])
@pytest.mark.parametrize("explicit_close", [False, True])
def test_runner_shutdown_keeps_accounting_open_until_request_producer_finishes(
    tmp_path: Path, producer_first: bool, explicit_close: bool, isolated_cli_env: None
) -> None:
    code = textwrap.dedent(
        """
        import asyncio
        import json
        from pathlib import Path
        import sys

        from kolega_code.cli.session_store import SessionStore
        from kolega_code.cli.session_usage import SessionUsageSink, derive_session_usage
        from kolega_code.llm.ledger import UsageLedger
        from kolega_code.llm.usage import normalize_usage

        store = SessionStore(Path(sys.argv[1]) / "state")
        session = store.create(Path(sys.argv[1]) / "project", "code", {})
        journal = store.journal(session.session_id)
        ledger = UsageLedger()
        sink = None
        ready, drainer_cancelled, begin = asyncio.Event(), asyncio.Event(), asyncio.Event()
        producers, errors = [], []

        async def produce():
            await begin.wait()
            request = ledger.begin("anthropic", "offline")
            ready.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                while not drainer_cancelled.is_set():
                    try:
                        await drainer_cancelled.wait()
                    except asyncio.CancelledError:
                        pass
                sink._drainer.cancel()
                await asyncio.sleep(0)
                assert not sink._closed
                ledger.record_failure(request, "producer cancelled")
                assert await sink.flush(timeout=10)
                if sys.argv[3] == "True":
                    assert (await sink.wait_closed()).committed
                raise

        async def main():
            global sink
            asyncio.get_running_loop().set_exception_handler(lambda loop, context: errors.append(str(context)))
            producer = asyncio.create_task(produce())
            producers.append(producer)
            sink = SessionUsageSink(
                journal, store.recorder(session.session_id), ledger, mode="ask", owner_task=producer
            )
            ledger.observer = sink
            await sink.start()
            for inputs, outputs in ((11, 3), (11, 3), (5, 1)):
                request = ledger.begin("anthropic", "offline")
                ledger.record_response(
                    request, normalize_usage({"input_tokens": inputs, "output_tokens": outputs}, "anthropic", "offline")
                )
            assert await sink.flush(timeout=10)
            request_close = sink._request_implicit_close

            def observed_close():
                drainer_cancelled.set()
                return request_close()

            sink._request_implicit_close = observed_close
            begin.set()
            await asyncio.wait_for(ready.wait(), 10)
            assert await sink.flush(timeout=10)
            all_tasks = asyncio.all_tasks
            ordered = (producer, sink._drainer) if sys.argv[2] == "True" else (sink._drainer, producer)

            def teardown_tasks(loop):
                return sorted(all_tasks(loop), key=lambda task: ordered.index(task) if task in ordered else 2)

            asyncio.runners.tasks.all_tasks = teardown_tasks

        asyncio.run(main())
        assert not errors, errors
        assert producers[0].done() and producers[0].cancelled()
        assert sink.close_result.committed
        assert sink._drainer.done()
        events = [json.loads(line) for line in journal.events_path.read_text().splitlines()]
        starts = [event for event in events if event["type"] == "llm.request_started"]
        settlements = [event for event in events if event["type"] == "llm.request_settled"]
        failures = [event for event in events if event["type"] == "llm.request_failed"]
        finishes = [event for event in events if event["type"] == "llm.run_finished"]
        assert len(starts) == len(settlements) == 4, (len(starts), len(settlements))
        assert len(failures) == len(finishes) == 1
        assert finishes[0]["payload"]["counts"] == {
            "requests": 4, "responses": 3, "failed": 1, "open": 0, "abandoned": 0, "partial": 0, "missing": 1
        }
        assert sum((event["payload"].get("usage") or {}).get("total_tokens", 0) for event in settlements) == 34
        assert events[-1]["type"] == "llm.run_finished"
        assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
        replay = derive_session_usage(journal.read_events())
        assert (replay["requests"], replay["responses"], replay["failed"], replay["total_tokens"]) == (4, 3, 1, 34)
        assert replay["open"] == replay["abandoned"] == 0
        store.release_session_locks()
        print("PRODUCER_SETTLED")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), str(producer_first), str(explicit_close)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PRODUCER_SETTLED" in result.stdout
    for diagnostic in ("Task was destroyed", "was never awaited", "unhandled exception"):
        assert diagnostic not in result.stderr


@pytest.mark.parametrize("explicit_close", [False, True])
@pytest.mark.parametrize("owner_first", [False, True])
def test_runner_owned_cancelled_startup_retains_writer_until_producer_settles(
    tmp_path: Path, explicit_close: bool, owner_first: bool, isolated_cli_env: None
) -> None:
    code = textwrap.dedent(
        """
        import asyncio
        import json
        from pathlib import Path
        import sys
        import threading

        from kolega_code.cli.session_store import SessionStore
        from kolega_code.cli.session_usage import SessionUsageSink
        from kolega_code.llm.ledger import UsageLedger

        store = SessionStore(Path(sys.argv[1]) / "state")
        session = store.create(Path(sys.argv[1]) / "project", "code", {})
        journal = store.journal(session.session_id)
        ledger = UsageLedger()
        entered, release = threading.Event(), threading.Event()
        retained, errors = [], []
        sink = None
        append = journal.append

        def held_start(event_type, **kwargs):
            if event_type == "llm.run_started":
                entered.set()
                assert release.wait(10)
            return append(event_type, **kwargs)

        journal.append = held_start

        async def produce():
            try:
                await sink.start()
            except asyncio.CancelledError:
                sink._start_task.cancel()
                release.set()
                while not sink._start_ready.done():
                    try:
                        await asyncio.shield(sink._start_ready)
                    except asyncio.CancelledError:
                        pass
                assert not sink._closed
                request = ledger.begin("anthropic", "offline")
                ledger.record_failure(request, "startup producer cancelled")
                if sys.argv[2] == "True":
                    assert (await sink.wait_closed()).committed
                raise

        async def main():
            global sink
            asyncio.get_running_loop().set_exception_handler(lambda loop, context: errors.append(str(context)))
            producer = asyncio.create_task(produce())
            retained.append(producer)
            sink = SessionUsageSink(
                journal, store.recorder(session.session_id), ledger, mode="ask", owner_task=producer
            )
            ledger.observer = sink
            assert await asyncio.to_thread(entered.wait, 10)
            assert sink._drainer is None
            all_tasks = asyncio.all_tasks
            ordered = (producer, sink._start_task) if sys.argv[3] == "True" else (sink._start_task, producer)

            def teardown_tasks(loop):
                return sorted(all_tasks(loop), key=lambda task: ordered.index(task) if task in ordered else 2)

            asyncio.runners.tasks.all_tasks = teardown_tasks

        try:
            asyncio.run(main())
        finally:
            release.set()
        assert not errors, errors
        assert retained[0].done() and retained[0].cancelled()
        assert sink.close_result.committed
        assert sink._start_task.done() and sink._drainer.done() and sink._close_task.done()
        events = [json.loads(line) for line in journal.events_path.read_text().splitlines()]
        kinds = [event["type"] for event in events if event["type"].startswith("llm.")]
        assert kinds == [
            "llm.run_started", "llm.request_started", "llm.request_settled", "llm.request_failed", "llm.run_finished"
        ], kinds
        assert events[-1]["payload"]["counts"] == {
            "requests": 1, "responses": 0, "failed": 1, "open": 0, "abandoned": 0, "partial": 0, "missing": 1
        }
        store.release_session_locks()
        print("STARTUP_PRODUCER_SETTLED")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), str(explicit_close), str(owner_first)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "STARTUP_PRODUCER_SETTLED" in result.stdout
    for diagnostic in ("Task was destroyed", "was never awaited", "unhandled exception"):
        assert diagnostic not in result.stderr
