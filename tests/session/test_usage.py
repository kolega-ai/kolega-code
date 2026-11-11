from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from copy import deepcopy
from pathlib import Path
import threading
from typing import Any

import pytest

from kolega_code.cli.session_journal import SessionEvent, SessionJournal, SessionRecorder
from kolega_code.cli.session_usage import SessionUsageSink, session_usage_scope
from kolega_code.events import AgentEvent
from kolega_code.llm.ledger import UsageLedger
from kolega_code.llm.usage import normalize_usage
import kolega_code.session.usage as usage_module
from kolega_code.session.usage import UsagePresentation

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("isolated_cli_env")]


class Broadcast:
    def __init__(self, journal: SessionJournal | None = None) -> None:
        self.events: list[AgentEvent] = []
        self.addresses: list[tuple[str, str]] = []
        self.threads: list[int] = []
        self.changed = asyncio.Event()
        self.journal = journal

    async def __call__(self, event: AgentEvent, workspace_id: str, thread_id: str) -> None:
        self.events.append(event)
        self.addresses.append((workspace_id, thread_id))
        self.threads.append(threading.get_ident())
        if self.journal is not None:
            self.journal.append("ui.usage_updated", actor="agent", payload=event.model_dump())
        self.changed.set()


def _journal(tmp_path: Path, name: str = "session") -> SessionJournal:
    journal = SessionJournal(name, tmp_path / name)
    journal.start_epoch("test")
    return journal


def _publisher(
    journal: SessionJournal, broadcast: Broadcast, live_run_id: Callable[[], str | None] | None = None
) -> UsagePresentation:
    return UsagePresentation(journal, broadcast, workspace_id="workspace", thread_id="thread", live_run_id=live_run_id)


def _marker(journal: SessionJournal, run_id: str = "run") -> SessionEvent:
    return journal.append("llm.run_started", actor="system", payload={"run_id": run_id, "accounting_version": 2})


def _fact(
    journal: SessionJournal,
    request_id: str = "request",
    event_type: str = "llm.request_settled",
    **extra: Any,
) -> SessionEvent:
    return journal.append(
        event_type,
        actor="system",
        payload={
            "run_id": "run",
            "request_id": request_id,
            "provider": "test",
            "model": "model",
            "revision": 0,
            "state": "responded",
            **extra,
        },
    )


def _finish(journal: SessionJournal, requests: int = 1) -> SessionEvent:
    return journal.append(
        "llm.run_finished", actor="system", payload={"run_id": "run", "counts": {"requests": requests}}
    )


def _tokens(inp: int, out: int) -> dict[str, int]:
    return {"input_tokens": inp, "output_tokens": out}


async def _eventually(predicate: Callable[[], bool]) -> None:
    async with asyncio.timeout(10):
        while not predicate():
            await asyncio.sleep(0.01)


async def test_empty_initial_snapshot_has_zero_source_and_does_not_create_journal(tmp_path: Path) -> None:
    journal = SessionJournal("empty", tmp_path / "empty")
    broadcast = Broadcast()
    publisher = _publisher(journal, broadcast)
    await publisher.start()
    await publisher.aclose()
    assert len(broadcast.events) == 1
    event = broadcast.events[0]
    assert event.content["source_seq"] == 0
    assert event.content["usage"]["requests"] == event.content["usage"]["total_tokens"] == 0
    assert event.content["usage"]["coverage"]["full"] is True
    assert event.content["usage"]["complete"] is True
    assert not journal.events_path.exists()
    assert journal._listeners == []


async def test_restored_all_epochs_snapshot_is_root_owned_and_ignores_ui_usage(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.append("turn.started", actor="user", turn_id="old")
    journal.append(
        "assistant.message",
        actor="assistant",
        turn_id="old",
        payload={"message": {"usage": _tokens(8, 3)}},
    )
    journal.start_epoch("compact")
    _marker(journal)
    _fact(journal, usage=_tokens(10, 5))
    finish = _finish(journal)
    journal.append(
        "ui.usage_updated",
        actor="agent",
        payload={"usage": {"total_tokens": 999999}, "request_id": "poison", "run_id": "run"},
    )
    journal.append("turn.finished", actor="system")
    original = journal.events_path.read_bytes()
    broadcast = Broadcast()
    publisher = _publisher(journal, broadcast)
    await publisher.start()
    try:
        assert len(broadcast.events) == 1
        event = broadcast.events[0]
        assert (event.event_type, event.sender, event.session_id, event.sub_agent_info) == (
            "usage_updated",
            "agent",
            "session",
            None,
        )
        assert (event.workspace_id, event.thread_id) == ("workspace", "thread")
        assert broadcast.addresses == [("workspace", "thread")]
        assert event.content["scope"] == "session"
        assert event.content["semantics"] == "snapshot"
        assert event.content["source_seq"] == finish.seq
        usage = event.content["usage"]
        assert usage["requests"] == usage["responses"] == 2
        assert (usage["input_tokens"], usage["output_tokens"], usage["total_tokens"]) == (18, 8, 26)
        assert usage["coverage"]["pre_accounting_turns"] == 1
        assert usage["coverage"]["legacy_requests"] == 1
        assert usage["coverage"]["full"] is usage["complete"] is False
    finally:
        await publisher.aclose()
    assert journal.events_path.read_bytes() == original


async def test_unknown_zero_partial_and_separate_cost_denominations_are_preserved(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    _marker(journal)
    _fact(
        journal,
        "zero",
        usage=_tokens(0, 0),
        cost={"source": "provider_reported", "amount": "0", "currency": "USD", "complete": True},
    )
    _fact(journal, "missing", usage=None)
    _fact(
        journal,
        "partial",
        observation={"input_tokens": 7, "complete": False},
        cost={"source": "estimated", "amount": "0.000000000000000123", "currency": "EUR", "complete": False},
    )
    _fact(
        journal,
        "credits",
        usage=_tokens(2, 1),
        cost={"source": "provider_reported", "amount": "1.25", "denomination": "provider_credits", "complete": True},
    )
    _finish(journal, 4)
    broadcast = Broadcast()
    publisher = _publisher(journal, broadcast)
    await publisher.start()
    await publisher.aclose()
    usage = broadcast.events[-1].content["usage"]
    assert usage["requests"] == 4
    assert usage["reported"] == 2
    assert usage["missing"] == usage["partial"] == 1
    assert (usage["input_tokens"], usage["output_tokens"], usage["total_tokens"]) == (9, 1, 3)
    assert usage["known_requests"]["input_tokens"] == 3
    assert usage["known_requests"]["output_tokens"] == usage["known_requests"]["total_tokens"] == 2
    assert usage["known_fields"]["cache_read_input_tokens"] == 0
    assert usage["detail_complete"]["cache_read_input_tokens"] is False
    costs = usage["costs"]
    assert set(costs["currencies"]) == {"USD", "EUR"}
    assert costs["currencies"]["USD"]["total"] == "0"
    assert costs["currencies"]["USD"]["reported_requests"] == 1
    assert costs["currencies"]["EUR"]["total"] == "1.23E-16"
    assert costs["non_monetary"]["provider_credits"]["total"] == "1.25"
    assert costs["missing_requests"] == costs["partial_requests"] == 1
    assert usage["complete"] is costs["complete"] is False


async def test_late_revision_replaces_snapshot_without_double_counting(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    _marker(journal)
    _fact(journal, state="failed", revision=1, observation={"input_tokens": 9, "complete": False})
    broadcast = Broadcast()
    publisher = _publisher(journal, broadcast, lambda: "run")
    await publisher.start()
    try:
        first = deepcopy(broadcast.events[-1].content["usage"])
        assert first["requests"] == first["failed"] == first["partial"] == 1
        assert first["known_requests"]["output_tokens"] == 0
        _fact(journal, event_type="llm.usage_observed", revision=3, usage=_tokens(12, 7), state="responded")
        _fact(journal, event_type="llm.usage_observed", revision=2, usage=_tokens(100, 100))
        finish = _finish(journal)
        await publisher.flush()
        updated = broadcast.events[-1].content["usage"]
        assert updated["requests"] == updated["failed"] == updated["reported"] == 1
        assert updated["responses"] == updated["partial"] == 0
        assert updated["total_tokens"] == 19
        assert updated["complete"] is True
        assert broadcast.events[-1].content["source_seq"] == finish.seq
        count = len(broadcast.events)
        _fact(journal, event_type="llm.usage_observed", revision=1, usage=_tokens(900, 900))
        await publisher.flush()
        assert len(broadcast.events) == count
    finally:
        await publisher.aclose()


async def test_live_run_identity_is_evaluated_on_owner_loop_for_each_refresh(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    _marker(journal)
    _fact(journal, event_type="llm.request_started", state="open")
    current: str | None = "run"
    owner_thread = threading.get_ident()

    def live_run_id() -> str | None:
        assert threading.get_ident() == owner_thread
        return current

    broadcast = Broadcast()
    publisher = _publisher(journal, broadcast, live_run_id)
    await publisher.start()
    try:
        assert broadcast.events[-1].content["usage"]["open"] == 1
        assert broadcast.events[-1].content["usage"]["abandoned"] == 0
        current = None
        await publisher.flush()
        usage = broadcast.events[-1].content["usage"]
        assert usage["open"] == 0
        assert usage["abandoned"] == usage["coverage"]["unclosed_runs"] == 1
        assert len(broadcast.events) == 2
    finally:
        await publisher.aclose()


async def test_final_snapshot_does_not_hide_a_missing_run_finish_with_live_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = _journal(tmp_path)
    ledger = UsageLedger()
    sink = SessionUsageSink(journal, SessionRecorder(journal, recover=False), ledger, mode="tui")
    ledger.observer = sink
    await sink.start()
    request = ledger.begin("anthropic", "model")
    ledger.record_response(request, normalize_usage(_tokens(5, 2), "anthropic", "model"))
    assert await sink.flush()
    broadcast = Broadcast()
    publisher = _publisher(journal, broadcast, lambda: ledger.run_id)
    await publisher.start()
    assert broadcast.events[-1].content["usage"]["complete"] is True
    append = journal.append

    def failed_finish(event_type: str, **kwargs: Any) -> SessionEvent:
        if event_type == "llm.run_finished":
            raise OSError("finish was not committed")
        return append(event_type, **kwargs)

    monkeypatch.setattr(journal, "append", failed_finish)
    try:
        result = await sink.wait_closed()
        assert result.status == "not_committed"
    finally:
        await publisher.aclose()
    final = broadcast.events[-1].content["usage"]
    assert final["requests"] == 1 and final["total_tokens"] == 7
    assert final["coverage"]["unclosed_runs"] == 1
    assert final["complete"] is False
    assert final["persistence"]["healthy"] is False
    assert final["persist_failures"] == 0


async def test_reused_ledger_never_leaks_another_session_and_child_totals_are_root_owned(tmp_path: Path) -> None:
    ledger = UsageLedger()
    old_journal = _journal(tmp_path, "old")
    old_sink = SessionUsageSink(old_journal, SessionRecorder(old_journal, recover=False), ledger, mode="tui")
    ledger.observer = old_sink
    await old_sink.start()
    old_request = ledger.begin("anthropic", "model")
    ledger.record_response(old_request, normalize_usage(_tokens(100, 50), "anthropic", "model"))
    await old_sink.wait_closed()

    journal = _journal(tmp_path, "new")
    recorder = SessionRecorder(journal, recover=False)
    child = recorder.scoped_child(agent_id="child", agent_name="Child", parent_tool_call_id="tool", depth=1)
    sink = SessionUsageSink(journal, recorder, ledger, mode="tui")
    ledger.observer = sink
    await sink.start()
    broadcast = Broadcast()
    publisher = _publisher(journal, broadcast, lambda: ledger.run_id)
    await publisher.start()
    try:
        assert broadcast.events[-1].content["usage"]["requests"] == 0
        with session_usage_scope(child):
            child_request = ledger.begin("anthropic", "model")
            ledger.record_response(child_request, normalize_usage(_tokens(3, 2), "anthropic", "model"))
        root_request = ledger.begin("anthropic", "model")
        ledger.record_response(root_request, normalize_usage(_tokens(4, 1), "anthropic", "model"))
        assert await sink.flush()
        await publisher.flush()
        usage = broadcast.events[-1].content["usage"]
        assert usage["requests"] == usage["responses"] == 2
        assert usage["total_tokens"] == 10
        assert broadcast.events[-1].session_id == "new"
        assert all(event.sub_agent_info is None for event in broadcast.events)
        facts = journal.read_events(repair_tail=False)
        child_fact = next(event for event in facts if event.payload.get("request_id") == child_request)
        assert (child_fact.agent_id, child_fact.depth) == ("child", 1)
        assert ledger.snapshot().total_tokens == 160
    finally:
        await sink.wait_closed()
        await publisher.aclose()
    assert journal._listeners == []


async def test_bursts_coalesce_and_ui_stream_and_terminal_events_do_not_self_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = _journal(tmp_path)
    _marker(journal)
    broadcast = Broadcast(journal)
    publisher = _publisher(journal, broadcast, lambda: "run")
    owner = threading.get_ident()
    reads: list[tuple[int, bool]] = []
    folds: list[int] = []
    read_events = journal.read_events
    derive = usage_module.derive_session_usage

    def read(*, repair_tail: bool = True) -> list[SessionEvent]:
        reads.append((threading.get_ident(), repair_tail))
        return read_events(repair_tail=repair_tail)

    def fold(events: Iterable[SessionEvent], *, live_run_id: str | None = None) -> dict[str, Any]:
        folds.append(threading.get_ident())
        return derive(events, live_run_id=live_run_id)

    monkeypatch.setattr(journal, "read_events", read)
    monkeypatch.setattr(usage_module, "derive_session_usage", fold)
    await publisher.start()
    try:
        assert len(reads) == 1
        broadcast.changed.clear()

        def append_burst() -> None:
            with journal._lock:
                for index in range(60):
                    _fact(journal, str(index), usage=_tokens(2, 1))
                    journal.append("ui.token", actor="agent")
                    journal.append("terminal.output", actor="tool")
                    journal.append("llm.token", actor="assistant")

        await asyncio.to_thread(append_burst)
        await asyncio.wait_for(broadcast.changed.wait(), 10)
        await publisher.flush()
        assert broadcast.events[-1].content["usage"]["requests"] == 60
        assert broadcast.events[-1].content["usage"]["total_tokens"] == 180
        assert len(reads) < 12
        assert all(thread != owner and repair is False for thread, repair in reads)
        assert len(folds) == len(reads)
        assert all(thread != owner for thread in folds)
        assert set(broadcast.threads) == {owner}
        before = len(reads)
        for index in range(100):
            journal.append("ui.usage_updated", actor="agent", payload={"index": index})
            journal.append("llm.token", actor="assistant")
            journal.append("turn.finished", actor="system")
        await asyncio.sleep(0.25)
        assert len(reads) == before
        assert len(broadcast.events) < 12
    finally:
        await publisher.aclose()
    assert journal._listeners == []


@pytest.mark.parametrize("operation", ["flush", "aclose"])
async def test_explicit_refresh_recovers_committed_then_raise_without_listener_notification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    journal = _journal(tmp_path)
    _marker(journal)
    broadcast = Broadcast()
    publisher = _publisher(journal, broadcast, lambda: "run")
    await publisher.start()
    write = journal._write_event_locked

    def committed_then_raise(event: SessionEvent) -> None:
        write(event)
        raise OSError("close failed after durable append")

    monkeypatch.setattr(journal, "_write_event_locked", committed_then_raise)
    with pytest.raises(OSError, match="close failed"):
        _fact(journal, usage=_tokens(11, 6))
    monkeypatch.setattr(journal, "_write_event_locked", write)
    original = journal.events_path.read_bytes()
    try:
        await getattr(publisher, operation)()
        usage = broadcast.events[-1].content["usage"]
        assert usage["requests"] == 1
        assert usage["total_tokens"] == 17
        assert broadcast.events[-1].content["source_seq"] == 3
        assert journal.events_path.read_bytes() == original
    finally:
        await publisher.aclose()


async def test_refresh_never_repairs_or_truncates_partial_tail(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    _marker(journal)
    _fact(journal, usage=_tokens(7, 3))
    with journal.events_path.open("ab") as stream:
        stream.write(b'{"partial":')
    original = journal.events_path.read_bytes()
    broadcast = Broadcast()
    publisher = _publisher(journal, broadcast, lambda: "run")
    await publisher.start()
    await publisher.flush()
    await publisher.aclose()
    assert journal.events_path.read_bytes() == original
    assert broadcast.events[-1].content["usage"]["total_tokens"] == 10
    assert journal.tail_requires_recovery is True


async def test_read_and_broadcast_failures_are_isolated_and_later_updates_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    journal = _journal(tmp_path)
    _marker(journal)
    broadcast = Broadcast()
    failures = 1

    async def flaky_broadcast(event: AgentEvent, workspace: str, thread: str) -> None:
        nonlocal failures
        if failures:
            failures -= 1
            raise RuntimeError("presentation unavailable")
        await broadcast(event, workspace, thread)

    publisher = UsagePresentation(journal, flaky_broadcast, workspace_id="workspace", thread_id="thread")
    await publisher.start()
    try:
        assert broadcast.events == []
        await publisher.flush()
        assert len(broadcast.events) == 1
        real_read = journal.read_events

        def broken_read(*, repair_tail: bool = True) -> list[SessionEvent]:
            raise OSError("read unavailable")

        monkeypatch.setattr(journal, "read_events", broken_read)
        await publisher.flush()
        monkeypatch.setattr(journal, "read_events", real_read)
        _fact(journal, usage=_tokens(5, 2))
        await publisher.flush()
        assert broadcast.events[-1].content["usage"]["total_tokens"] == 7
        assert broadcast.events[-1].content["usage"]["persist_failures"] == 0
        assert "presentation unavailable" in caplog.text
        assert "read unavailable" in caplog.text
    finally:
        await publisher.aclose()


@pytest.mark.parametrize("operation", ["start", "flush", "aclose"])
async def test_repeated_cancellation_drains_owned_executor_and_final_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    journal = _journal(tmp_path)
    _marker(journal)
    broadcast = Broadcast()
    publisher = _publisher(journal, broadcast, lambda: "run")
    if operation != "start":
        await publisher.start()
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    real_read = journal.read_events

    def blocked_read(*, repair_tail: bool = True) -> list[SessionEvent]:
        entered.set()
        assert release.wait(10)
        try:
            return real_read(repair_tail=repair_tail)
        finally:
            finished.set()

    monkeypatch.setattr(journal, "read_events", blocked_read)
    task = asyncio.create_task(getattr(publisher, operation)())
    await _eventually(entered.is_set)
    _fact(journal, usage=_tokens(13, 4))
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    assert not finished.is_set()
    if publisher._worker is not None:
        publisher._worker.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()
    assert publisher._worker is not None and publisher._worker.done()
    assert journal._listeners == []
    assert broadcast.events[-1].content["usage"]["total_tokens"] == 17
    count = len(broadcast.events)
    _fact(journal, "after-close", usage=_tokens(50, 50))
    await publisher.aclose()
    assert len(broadcast.events) == count


async def test_close_waits_for_inflight_broadcast_and_publishes_final_reconciliation(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    _marker(journal)
    broadcast = Broadcast()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def held_broadcast(event: AgentEvent, workspace: str, thread: str) -> None:
        if not broadcast.events:
            entered.set()
            await release.wait()
        await broadcast(event, workspace, thread)

    publisher = UsagePresentation(journal, held_broadcast, workspace_id="workspace", thread_id="thread")
    start = asyncio.create_task(publisher.start())
    await asyncio.wait_for(entered.wait(), 10)
    _fact(journal, usage=_tokens(6, 5))
    close = asyncio.create_task(publisher.aclose())
    await asyncio.sleep(0)
    close.cancel()
    await asyncio.sleep(0)
    close.cancel()
    await asyncio.sleep(0)
    assert not close.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await close
    await start
    assert broadcast.events[-1].content["usage"]["total_tokens"] == 11
    assert journal._listeners == []
    assert publisher._worker is not None and publisher._worker.done()


async def test_worker_cancelled_before_entry_is_recovered_by_close(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    _marker(journal)
    _fact(journal, usage=_tokens(4, 3))
    broadcast = Broadcast()
    publisher = _publisher(journal, broadcast)
    publisher._ensure_worker()
    assert publisher._worker is not None
    publisher._worker.cancel()
    with pytest.raises(asyncio.CancelledError):
        await publisher.aclose()
    assert broadcast.events[-1].content["usage"]["total_tokens"] == 7
    assert journal._listeners == []
    assert publisher._worker is not None and publisher._worker.done()


async def test_asyncio_runner_shutdown_drains_executor_and_unsubscribes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = _journal(tmp_path)
    _marker(journal)
    broadcast = Broadcast()
    publisher = _publisher(journal, broadcast, lambda: "run")
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    blocking = False
    real_read = journal.read_events

    def read(*, repair_tail: bool = True) -> list[SessionEvent]:
        if blocking:
            entered.set()
            assert release.wait(10)
        try:
            return real_read(repair_tail=repair_tail)
        finally:
            if blocking:
                finished.set()

    monkeypatch.setattr(journal, "read_events", read)

    async def main() -> None:
        nonlocal blocking
        await publisher.start()
        blocking = True
        _fact(journal, usage=_tokens(9, 2))
        await _eventually(entered.is_set)

    def run() -> None:
        asyncio.run(main())

    runner = asyncio.create_task(asyncio.to_thread(run))
    try:
        await _eventually(entered.is_set)
        await _eventually(lambda: publisher._worker is not None and bool(publisher._worker.cancelling()))
        assert not runner.done()
        assert not finished.is_set()
    finally:
        release.set()
        await asyncio.wait_for(runner, 10)
    assert finished.is_set()
    assert publisher._worker is not None and publisher._worker.done()
    assert journal._listeners == []
    assert broadcast.events[-1].content["usage"]["total_tokens"] == 11


async def test_unsubscribe_is_idempotent_thread_safe_and_does_not_skip_other_listeners(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    first: list[SessionEvent] = []
    second: list[SessionEvent] = []
    unsubscribe: Callable[[], None]

    def self_removing(event: SessionEvent) -> None:
        first.append(event)
        unsubscribe()

    unsubscribe = journal.add_listener(self_removing)
    remove_second = journal.add_listener(second.append)
    one = journal.append("one", actor="system")
    two = journal.append("two", actor="system")
    assert first == [one]
    assert second == [one, two]
    await asyncio.gather(*(asyncio.to_thread(remove_second) for _ in range(10)))
    unsubscribe()
    journal.append("three", actor="system")
    assert second == [one, two]
    assert journal._listeners == []
