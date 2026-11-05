from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from filelock import FileLock

from kolega_code.cli.session_journal import SessionEvent, SessionJournal, SessionJournalError

pytestmark = pytest.mark.usefixtures("isolated_cli_env")


@pytest.fixture
def owned_journal(tmp_path: Path) -> Iterator[SessionJournal]:
    lock = FileLock(tmp_path / ".session.lock", timeout=0, thread_local=False)
    with lock:
        journal = SessionJournal(
            "session",
            tmp_path / "session",
            cross_process_lock=lock,
            owns_cross_process_lock=lambda: True,
        )
        journal.start_epoch("test")
        yield journal


def _fail_event_close(monkeypatch: pytest.MonkeyPatch, journal: SessionJournal, error: BaseException) -> None:
    real_close = os.close
    events_stat = journal.events_path.stat()
    failed = False

    def close(fd: int) -> None:
        nonlocal failed
        fd_stat = os.fstat(fd)
        real_close(fd)
        if not failed and (fd_stat.st_dev, fd_stat.st_ino) == (events_stat.st_dev, events_stat.st_ino):
            failed = True
            assert journal._cross_process_lock is not None
            assert journal._cross_process_lock.is_locked
            raise error

    monkeypatch.setattr(os, "close", close)


@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt])
@pytest.mark.parametrize("event_type", ["probe", "context.epoch_started"])
def test_owned_append_reloads_after_close_failure(
    owned_journal: SessionJournal,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[BaseException],
    event_type: str,
) -> None:
    journal = owned_journal
    notified: list[SessionEvent] = []
    journal.add_listener(notified.append)
    lock = journal._cross_process_lock
    assert lock is not None
    real_release = lock.release
    loaded_at_release: list[bool] = []

    def release(force: bool = False) -> None:
        loaded_at_release.append(journal._loaded)
        real_release(force=force)

    monkeypatch.setattr(lock, "release", release)
    error = error_type("injected close failure")
    _fail_event_close(monkeypatch, journal, error)
    original = journal.events_path.read_bytes()
    new_epoch = "new-epoch" if event_type == "context.epoch_started" else None

    with pytest.raises(error_type) as caught:
        journal.append(event_type, actor="system", epoch_id=new_epoch)

    assert caught.value is error
    persisted = journal.events_path.read_bytes()
    assert persisted.startswith(original)
    failed_event = json.loads(persisted.splitlines()[-1])
    assert failed_event["type"] == event_type
    assert failed_event["seq"] == 2
    assert notified == []

    appended = journal.append("after-failure", actor="system")

    assert appended.seq == 3
    assert loaded_at_release[0] is False
    assert journal.events_path.read_bytes().startswith(persisted)
    assert [event.seq for event in journal.read_events(repair_tail=False)] == [1, 2, 3]
    assert appended.epoch_id == (new_epoch or failed_event["epoch_id"])
    assert journal.epoch_id == appended.epoch_id
    assert notified == [appended]
    assert notified[0] is appended


@pytest.mark.parametrize("failure", ["short-write", "write-error", "interrupt"])
@pytest.mark.parametrize("read_before_append", [False, True])
def test_partial_append_with_failed_rollback_is_recovered_only_by_safe_append(
    owned_journal: SessionJournal,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    read_before_append: bool,
) -> None:
    journal = owned_journal
    original = journal.events_path.read_bytes()
    real_write = os.write
    real_ftruncate = os.ftruncate
    error = KeyboardInterrupt("injected interrupt") if failure == "interrupt" else OSError("injected write error")
    partial: list[bytes] = []
    rollback_attempts: list[int] = []
    repairs: list[int] = []

    def write(fd: int, data: bytes) -> int:
        partial.append(data[:27])
        written = real_write(fd, partial[-1])
        if failure != "short-write":
            raise error
        return written

    def fail_rollback(fd: int, length: int) -> None:
        rollback_attempts.append(length)
        raise OSError("injected rollback failure")

    def repair(fd: int, length: int) -> None:
        assert journal._cross_process_lock is not None
        assert journal._cross_process_lock.is_locked
        repairs.append(length)
        real_ftruncate(fd, length)

    with monkeypatch.context() as faults:
        faults.setattr(os, "write", write)
        faults.setattr(os, "ftruncate", fail_rollback)
        expected_error = KeyboardInterrupt if failure == "interrupt" else SessionJournalError
        with pytest.raises(expected_error) as caught:
            journal.append("partial", actor="system")

    if failure == "interrupt":
        assert caught.value is error
        assert rollback_attempts == []
    elif failure == "write-error":
        assert caught.value.__cause__ is error
        assert rollback_attempts == [len(original)]
    else:
        assert "Short event write" in str(caught.value)
        assert rollback_attempts == [len(original)]
    damaged = original + partial[0]
    assert journal.events_path.read_bytes() == damaged

    monkeypatch.setattr(os, "ftruncate", repair)
    if read_before_append:
        assert [event.seq for event in journal.read_events(repair_tail=False)] == [1]
        assert journal.events_path.read_bytes() == damaged
        assert repairs == []

    appended = journal.append("after-failure", actor="system")

    assert appended.seq == 2
    assert repairs == [len(original)]
    assert journal.events_path.read_bytes().startswith(original)
    events = journal.read_events(repair_tail=False)
    assert [event.seq for event in events] == [1, 2]
    assert [event.event_type for event in events] == ["context.epoch_started", "after-failure"]


def test_uncertain_append_does_not_silently_repair_duplicate_sequences(
    owned_journal: SessionJournal, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = owned_journal
    _fail_event_close(monkeypatch, journal, OSError("injected close failure"))
    with pytest.raises(OSError, match="injected close failure"):
        journal.append("probe", actor="system")
    last_line = journal.events_path.read_bytes().splitlines(keepends=True)[-1]
    with journal.events_path.open("ab") as handle:
        handle.write(last_line)
    corrupted = journal.events_path.read_bytes()

    with pytest.raises(SessionJournalError, match="sequence gap"):
        journal.append("after-failure", actor="system")
    with pytest.raises(SessionJournalError, match="sequence gap"):
        journal.read_events(repair_tail=False)
    assert journal.events_path.read_bytes() == corrupted


def test_successful_append_keeps_listener_identity_and_isolates_listener_errors(
    owned_journal: SessionJournal,
) -> None:
    journal = owned_journal
    notified: list[SessionEvent] = []

    def failing_listener(event: SessionEvent) -> None:
        raise RuntimeError("injected listener failure")

    journal.add_listener(failing_listener)
    journal.add_listener(notified.append)
    first = journal.append("first", actor="system")
    second = journal.append("second", actor="system")

    assert first.seq == 2
    assert second.seq == 3
    assert notified == [first, second]
    assert notified[0] is first
    assert notified[1] is second
    assert journal._loaded is True
    assert journal.read_events(repair_tail=False)[1:] == [first, second]
