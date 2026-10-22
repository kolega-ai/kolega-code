"""Journal appends are bounded: a wedged write must not hang its caller.

Recording is an enhancement to live delivery, but the await used to have no
deadline. A write blocked inside the journal's process-wide lock parked a whole
turn (see the ``snappy-sloth-4970`` investigation), so the await is bounded and
the stall is reported. The worker thread is deliberately not interrupted: it
finishes the write on its own and the row lands, possibly late.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest

from kolega_code.cli import session_event_store as session_event_store_module
from kolega_code.cli.session_event_store import FileSessionEventStore
from kolega_code.cli.session_journal import SessionJournal, SessionJournalError
from kolega_code.cli.session_store import SessionStore
from kolega_code.events import AgentEvent, KnownEventType


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    return SessionStore(root=tmp_path / "state")


def _journal_with_epoch(store: SessionStore, tmp_path: Path) -> SessionJournal:
    project = tmp_path / "project"
    project.mkdir()
    record = store.create(project, "code", {})
    journal = store.journal(record.session_id)
    journal.start_epoch("test")
    return journal


def _event(journal: SessionJournal) -> AgentEvent:
    return AgentEvent(
        session_id=journal.session_id,
        sender="planning-agent",
        event_type=KnownEventType.TURN_ENDED,
        content={},
    )


@pytest.mark.asyncio
async def test_append_timeout_releases_the_caller_and_the_write_still_lands(
    store: SessionStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = _journal_with_epoch(store, tmp_path)
    gate = threading.Event()
    real_append = journal.append

    def blocked_append(*args, **kwargs):
        gate.wait(10.0)
        return real_append(*args, **kwargs)

    monkeypatch.setattr(journal, "append", blocked_append)
    reports: list[dict] = []
    events = FileSessionEventStore(journal, append_timeout=0.05, on_stall=reports.append)

    with pytest.raises(asyncio.TimeoutError):
        await events.append(_event(journal))

    assert [report["kind"] for report in reports] == ["timeout"]
    assert reports[0]["timeout_s"] == 0.05
    assert reports[0]["event_type"] == KnownEventType.TURN_ENDED

    # The abandoned worker thread is not interrupted: the row lands anyway.
    gate.set()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if any(event.event_type == "ui.turn_ended" for event in journal.read_events()):
            break
        await asyncio.sleep(0.02)
    assert [event.event_type for event in journal.read_events()].count("ui.turn_ended") == 1


@pytest.mark.asyncio
async def test_slow_append_is_reported_without_failing(
    store: SessionStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = _journal_with_epoch(store, tmp_path)
    monkeypatch.setattr(session_event_store_module, "RECORD_APPEND_SLOW_SECONDS", 0.0)
    reports: list[dict] = []
    events = FileSessionEventStore(journal, on_stall=reports.append)

    seq = await events.append(_event(journal))

    assert seq > 0
    assert [report["kind"] for report in reports] == ["slow"]


@pytest.mark.asyncio
async def test_failure_after_the_timeout_is_reported(
    store: SessionStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = _journal_with_epoch(store, tmp_path)
    gate = threading.Event()

    def failing_append(*args, **kwargs):
        gate.wait(10.0)
        raise SessionJournalError("journal went away")

    monkeypatch.setattr(journal, "append", failing_append)
    reports: list[dict] = []
    events = FileSessionEventStore(journal, append_timeout=0.05, on_stall=reports.append)

    with pytest.raises(asyncio.TimeoutError):
        await events.append(_event(journal))
    gate.set()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not any(r["kind"] == "failed" for r in reports):
        await asyncio.sleep(0.02)

    failed = [report for report in reports if report["kind"] == "failed"]
    assert failed and "journal went away" in (failed[0]["error"] or "")
