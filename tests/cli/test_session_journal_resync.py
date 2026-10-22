"""The journal's per-append tail resync is skipped while this process owns the session.

``SessionJournal.append`` normally re-reads the last 64 KB of the log on every
append to reconcile ``seq`` with another writer's tail. Once the store has claimed
the session's cross-process lock for the process lifetime, no other writer can
exist — the resync is two syscalls, a read and a JSON parse on the hot path, all
under the session lock that everything else queues behind.
"""

from __future__ import annotations

from pathlib import Path

from filelock import FileLock

from kolega_code.cli.session_journal import SessionJournal
from kolega_code.cli.session_store import SessionStore


def _counting_journal(tmp_path: Path, *, owns: bool) -> tuple[SessionJournal, list[int]]:
    calls: list[int] = []
    journal = SessionJournal(
        "session-1",
        tmp_path / "session-1",
        cross_process_lock=FileLock(tmp_path / ".session-1.lock", timeout=0, thread_local=False),
        owns_cross_process_lock=lambda: owns,
    )
    journal.start_epoch("test")
    journal._resync_seq_from_tail_locked = lambda: calls.append(1)  # type: ignore[method-assign]
    return journal, calls


def test_append_skips_the_tail_resync_when_the_process_owns_the_lock(tmp_path: Path) -> None:
    journal, calls = _counting_journal(tmp_path, owns=True)

    journal.append("probe", actor="system")

    assert calls == []
    assert [event.event_type for event in journal.read_events()] == ["context.epoch_started", "probe"]


def test_append_keeps_the_tail_resync_when_the_lock_is_shared(tmp_path: Path) -> None:
    journal, calls = _counting_journal(tmp_path, owns=False)

    journal.append("probe", actor="system")

    assert calls == [1]


def test_store_ownership_tracks_claim_and_release(tmp_path: Path) -> None:
    store = SessionStore(root=tmp_path / "state")
    project = tmp_path / "project"
    project.mkdir()
    record = store.create(project, "code", {})

    journal = store.journal(record.session_id)
    journal.start_epoch("test")
    calls: list[int] = []
    journal._resync_seq_from_tail_locked = lambda: calls.append(1)  # type: ignore[method-assign]

    store.claim_for_resume(record.session_id, project)
    journal.append("owned", actor="system")
    assert calls == []

    store.release_session_lock(record.session_id)
    journal.append("shared", actor="system")
    assert calls == [1]
