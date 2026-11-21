"""The non-repairing read reuses its parsed prefix without changing what it returns."""

from __future__ import annotations

import json
import os
from pathlib import Path
import uuid

import pytest

from kolega_code.cli.session_journal import SessionJournal, SessionJournalError

pytestmark = pytest.mark.usefixtures("isolated_cli_env")


def _journal(tmp_path: Path) -> SessionJournal:
    journal = SessionJournal("incremental", tmp_path / "session")
    journal.start_epoch("test")
    return journal


def _line_like(journal: SessionJournal, event_type: str = "turn.started") -> bytes:
    """A complete, valid JSONL row for the next sequence number."""
    last = journal.read_events(repair_tail=False)[-1].to_dict()
    last["id"] = str(uuid.uuid4())
    last["seq"] = last["seq"] + 1
    last["type"] = event_type
    last["payload"] = {"text": "completed later"}
    return (json.dumps(last, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def test_appends_parse_only_the_new_suffix_and_return_the_same_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = _journal(tmp_path)
    for index in range(5):
        journal.append("turn.started", actor="user", payload={"text": str(index)})
    first = journal.read_events(repair_tail=False)
    full_parses = 0
    original_parse = journal._parse_event_bytes_locked

    def parse(raw: bytes, *, strict_seq: bool = True):
        nonlocal full_parses
        full_parses += 1
        return original_parse(raw, strict_seq=strict_seq)

    monkeypatch.setattr(journal, "_parse_event_bytes_locked", parse)
    journal.append("turn.started", actor="user", payload={"text": "six"})
    second = journal.read_events(repair_tail=False)

    assert full_parses == 0
    assert [event.seq for event in second] == [1, 2, 3, 4, 5, 6, 7]
    assert second[:6] == first
    assert second[-1].payload == {"text": "six"}


def test_incremental_reads_match_a_cold_read_of_the_same_file(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    for index in range(20):
        journal.append("turn.started", actor="user", payload={"text": f"step {index}"})
        journal.read_events(repair_tail=False)
    cold = SessionJournal("incremental", tmp_path / "session")
    assert [event.to_dict() for event in journal.read_events(repair_tail=False)] == [
        event.to_dict() for event in cold.read_events(repair_tail=False)
    ]


def test_partial_tail_is_neither_exposed_nor_lost_when_completed(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.append("turn.started", actor="user", payload={"text": "before"})
    complete = journal.events_path.read_bytes()
    pending = _line_like(journal)
    journal.events_path.write_bytes(complete + pending[: len(pending) // 2])

    assert [event.seq for event in journal.read_events(repair_tail=False)] == [1, 2]
    assert journal.tail_requires_recovery is True

    journal.events_path.write_bytes(complete + pending)
    events = journal.read_events(repair_tail=False)
    assert [event.seq for event in events] == [1, 2, 3]
    assert events[-1].payload == {"text": "completed later"}
    assert journal.tail_requires_recovery is False


def test_same_size_rewrite_with_a_new_mtime_is_reread(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.append("turn.started", actor="user", payload={"text": "alpha"})
    journal.read_events(repair_tail=False)
    stat = journal.events_path.stat()
    rewritten = journal.events_path.read_bytes().replace(b"alpha", b"bravo")
    assert len(rewritten) == stat.st_size
    journal.events_path.write_bytes(rewritten)
    os.utime(journal.events_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    assert journal.read_events(repair_tail=False)[-1].payload == {"text": "bravo"}


def test_truncated_journal_forces_a_full_read(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.append("turn.started", actor="user", payload={"text": "one"})
    journal.append("turn.started", actor="user", payload={"text": "two"})
    assert len(journal.read_events(repair_tail=False)) == 3
    journal.events_path.write_bytes(journal.events_path.read_bytes().splitlines()[0] + b"\n")

    assert [event.seq for event in journal.read_events(repair_tail=False)] == [1]


def test_replaced_inode_forces_a_full_read(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.append("turn.started", actor="user", payload={"text": "alpha"})
    journal.read_events(repair_tail=False)
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_bytes(journal.events_path.read_bytes().replace(b"alpha", b"bravo"))
    os.replace(replacement, journal.events_path)

    assert journal.read_events(repair_tail=False)[-1].payload == {"text": "bravo"}


def test_sequence_gap_in_appended_bytes_is_rejected(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.append("turn.started", actor="user", payload={"text": "one"})
    journal.read_events(repair_tail=False)
    row = json.loads(_line_like(journal))
    row["seq"] = 5
    with journal.events_path.open("ab") as stream:
        stream.write((json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8"))

    with pytest.raises(SessionJournalError, match="sequence gap at line 3: expected 3, got 5"):
        journal.read_events(repair_tail=False)
