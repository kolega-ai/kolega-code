"""Filesystem event store: journal integration and isolation guarantees.

Presentation events share the session journal's sequence space. The safety
property that makes that acceptable is that they are invisible to provider
history replay — if a UI event could reach the model's conversation, or could
make a session fail to load, sharing the log would be the wrong design.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kolega_code.cli.session_event_store import UI_EVENT_PREFIX, FileArtifactStore, FileSessionEventStore
from kolega_code.cli.session_store import SessionStore
from kolega_code.events import AgentEvent, ArtifactPurpose, KnownEventType
from kolega_code.llm.models import Message, TextBlock
from kolega_code.session.store import SessionStoreError


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    return SessionStore(root=tmp_path / "state")


def _session(store: SessionStore, project: Path):
    return store.create(project, "code", {"model": "test"})


@pytest.mark.asyncio
async def test_ui_events_do_not_reach_provider_history(store: SessionStore, tmp_path: Path) -> None:
    record = _session(store, tmp_path)
    recorder = store.recorder(record.session_id)
    events = FileSessionEventStore(store.journal(record.session_id))

    recorder.start_turn(Message(role="user", content=[TextBlock(text="build the thing")]))
    await events.append(
        AgentEvent(
            session_id=record.session_id,
            sender="agent",
            event_type=KnownEventType.ASSISTANT_DELTA,
            content={"text": "SHOULD NOT BE IN HISTORY", "complete": True},
        )
    )
    recorder.record_assistant(Message(role="assistant", content=[TextBlock(text="done")]))
    recorder.finish_turn("completed")

    reloaded = store.load(record.session_id)
    texts = [
        block.get("text", "")
        for message in reloaded.history
        for block in message.get("content", [])
        if isinstance(block, dict)
    ]
    assert "build the thing" in texts
    assert "done" in texts
    assert not any("SHOULD NOT BE IN HISTORY" in text for text in texts), (
        "a presentation event leaked into the conversation sent to the model"
    )
    assert len(reloaded.history) == 2, f"history gained unexpected messages: {reloaded.history}"


@pytest.mark.asyncio
async def test_ui_events_interleave_in_one_sequence_space(store: SessionStore, tmp_path: Path) -> None:
    """Shared ordering is the point: a UI event and a message are comparable."""
    record = _session(store, tmp_path)
    recorder = store.recorder(record.session_id)
    journal = store.journal(record.session_id)
    events = FileSessionEventStore(journal)

    recorder.start_turn(Message(role="user", content=[TextBlock(text="go")]))
    ui_seq = await events.append(
        AgentEvent(
            session_id=record.session_id,
            sender="agent",
            event_type=KnownEventType.TERMINAL_COMMAND,
            content={"command": "pytest"},
        )
    )
    recorder.record_assistant(Message(role="assistant", content=[TextBlock(text="ran tests")]))

    raw = journal.read_events(repair_tail=True)
    by_seq = {event.seq: event.event_type for event in raw}
    assert by_seq[ui_seq] == UI_EVENT_PREFIX + KnownEventType.TERMINAL_COMMAND
    assistant_seq = next(seq for seq, kind in by_seq.items() if kind == "assistant.message")
    assert ui_seq < assistant_seq, "the terminal command happened before the assistant message"


@pytest.mark.asyncio
async def test_read_returns_sparse_but_ascending_sequences(store: SessionStore, tmp_path: Path) -> None:
    """Sharing a sequence space leaves gaps; ordering still holds."""
    record = _session(store, tmp_path)
    recorder = store.recorder(record.session_id)
    events = FileSessionEventStore(store.journal(record.session_id))

    recorder.start_turn(Message(role="user", content=[TextBlock(text="x")]))
    for index in range(3):
        await events.append(
            AgentEvent(
                session_id=record.session_id,
                sender="agent",
                event_type=KnownEventType.LOG_MESSAGE,
                content={"text": f"log {index}"},
            )
        )
        recorder.record_assistant(Message(role="assistant", content=[TextBlock(text=f"reply {index}")]))

    stored = await events.read(record.session_id)
    assert len(stored) == 3
    seqs = [event.seq for event in stored if event.seq is not None]
    assert len(seqs) == 3, "every stored event must carry a sequence number"
    assert seqs == sorted(seqs)
    assert seqs != list(range(seqs[0], seqs[0] + 3)), "expected gaps from the interleaved journal records"


@pytest.mark.asyncio
async def test_store_rejects_a_foreign_session_id(store: SessionStore, tmp_path: Path) -> None:
    record = _session(store, tmp_path)
    events = FileSessionEventStore(store.journal(record.session_id))
    with pytest.raises(SessionStoreError):
        await events.read("some-other-session")


@pytest.mark.asyncio
async def test_artifact_round_trip_and_integrity(store: SessionStore, tmp_path: Path) -> None:
    record = _session(store, tmp_path)
    artifacts = FileArtifactStore(store.journal(record.session_id))

    payload = b"large tool output" * 100
    ref = await artifacts.put(
        payload,
        media_type="text/plain; charset=utf-8",
        purpose=ArtifactPurpose.TOOL_RESULT,
        encoding="utf-8",
        chars=len(payload.decode()),
    )
    assert await artifacts.open(ref) == payload

    # Identical bytes must dedupe to the same address.
    again = await artifacts.put(
        payload,
        media_type="text/plain; charset=utf-8",
        purpose=ArtifactPurpose.TOOL_RESULT,
        encoding="utf-8",
    )
    assert again.sha256 == ref.sha256

    corrupted = ref.model_copy(update={"sha256": "0" * 64})
    with pytest.raises(SessionStoreError):
        await artifacts.open(corrupted)


@pytest.mark.asyncio
async def test_tail_survives_a_partially_written_record(store: SessionStore, tmp_path: Path) -> None:
    """A follower reads a log another process is appending to, mid-write."""
    record = _session(store, tmp_path)
    events_path = store.events_path_for(record.session_id)
    events = FileSessionEventStore(store.journal(record.session_id))

    await events.append(
        AgentEvent(
            session_id=record.session_id,
            sender="agent",
            event_type=KnownEventType.LOG_MESSAGE,
            content={"text": "complete record"},
        )
    )
    # Simulate a torn append: a line with no trailing newline yet.
    with open(events_path, "a", encoding="utf-8") as handle:
        handle.write('{"partial": true')

    stored = await events.read(record.session_id)
    assert [event.content["text"] for event in stored] == ["complete record"], (
        "a half-written trailing record must be ignored, not treated as corruption"
    )


@pytest.mark.asyncio
async def test_llm_events_are_invisible_to_the_presentation_stream(tmp_path):
    """share export reads only ui.* events; llm.* journal records never leak."""
    from kolega_code.cli.session_store import SessionStore
    from kolega_code.cli.session_event_store import FileSessionEventStore

    store = SessionStore(tmp_path / "state")
    session = store.create(tmp_path / "proj", "code", {})
    journal = store.journal(session.session_id)
    journal.append(
        "llm.message",
        actor="assistant",
        payload={
            "request_id": "r",
            "run_id": "x",
            "provider": "p",
            "model": "m",
            "origin": {"kind": "helper"},
            "message": None,
        },
    )
    events = await FileSessionEventStore(journal).read(session.session_id)
    assert events == []


def test_append_resyncs_from_tail_after_foreign_writer(tmp_path: Path) -> None:
    """A writer that bypassed this journal object must not cause duplicate seqs."""
    import json

    from filelock import FileLock

    from kolega_code.cli.session_journal import SessionJournal

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    lock = FileLock(tmp_path / "session.lock", timeout=0, thread_local=False)
    journal = SessionJournal("test-session", session_dir, cross_process_lock=lock)
    journal.append("context.epoch_started", actor="system", epoch_id="epoch-1")
    journal.append("test.event", actor="system", epoch_id="epoch-1")

    # A second writer appends directly to the file, advancing the durable tail
    # past this journal's in-memory counter.
    events_path = session_dir / "events.jsonl"
    base = json.loads(events_path.read_text(encoding="utf-8").splitlines()[-1])
    for seq in (3, 4):
        foreign = dict(base)
        foreign["seq"] = seq
        foreign["event_id"] = f"foreign-{seq}"
        with events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(foreign) + "\n")

    event = journal.append("test.event", actor="system", epoch_id="epoch-1")
    assert event.seq == 5
    assert [e.seq for e in journal.read_events()] == [1, 2, 3, 4, 5]


def test_append_repairs_foreign_record_missing_newline_before_resync(tmp_path: Path) -> None:
    import json

    from filelock import FileLock

    from kolega_code.cli.session_journal import SessionJournal

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    lock = FileLock(tmp_path / "session.lock", timeout=0, thread_local=False)
    journal = SessionJournal("test-session", session_dir, cross_process_lock=lock)
    journal.append("context.epoch_started", actor="system", epoch_id="epoch-1")
    journal.append("test.event", actor="system", epoch_id="epoch-1")

    events_path = session_dir / "events.jsonl"
    foreign = json.loads(events_path.read_text(encoding="utf-8").splitlines()[-1])
    foreign["seq"] = 3
    foreign["id"] = "foreign-in-progress"
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(foreign))

    event = journal.append("test.event", actor="system", epoch_id="epoch-1")
    events = journal.read_events()

    assert event.seq == 3
    assert [stored.seq for stored in events] == [1, 2, 3]
    assert events[-1].event_id == event.event_id
    assert all(stored.event_id != "foreign-in-progress" for stored in events)
    assert events_path.read_bytes().endswith(b"\n")


@pytest.mark.parametrize("with_cross_process_lock", [False, True])
def test_append_recovers_tail_ignored_by_prior_read(tmp_path: Path, with_cross_process_lock: bool) -> None:
    import json

    from filelock import FileLock

    from kolega_code.cli.session_journal import SessionJournal

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    lock = FileLock(tmp_path / "session.lock", timeout=0, thread_local=False) if with_cross_process_lock else None
    journal = SessionJournal("test-session", session_dir, cross_process_lock=lock)
    journal.append("context.epoch_started", actor="system", epoch_id="epoch-1")
    journal.append("test.event", actor="system", epoch_id="epoch-1")

    events_path = session_dir / "events.jsonl"
    ignored = json.loads(events_path.read_text(encoding="utf-8").splitlines()[-1])
    ignored["seq"] = 3
    ignored["id"] = "ignored-in-progress"
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(ignored))
    before = events_path.read_bytes()

    assert [event.seq for event in journal.read_events(repair_tail=False)] == [1, 2]
    assert events_path.read_bytes() == before

    appended = journal.append("test.event", actor="system", epoch_id="epoch-1")
    events = journal.read_events()

    assert appended.seq == 3
    assert [event.seq for event in events] == [1, 2, 3]
    assert all(event.event_id != "ignored-in-progress" for event in events)
    assert events_path.read_bytes().endswith(b"\n")


@pytest.mark.parametrize("operation", ["read", "epoch"])
def test_direct_journal_recovery_acquires_supplied_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    from filelock import FileLock

    from kolega_code.cli import session_journal as session_journal_module
    from kolega_code.cli.session_journal import SessionJournal

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    lock = FileLock(tmp_path / "session.lock", timeout=0, thread_local=False)
    seed = SessionJournal("test-session", session_dir, cross_process_lock=lock)
    seed.append("context.epoch_started", actor="system", epoch_id="epoch-1")
    with seed.events_path.open("ab") as handle:
        handle.write(b'{"version":2,"event_id":"in-progress"')

    journal = SessionJournal("test-session", session_dir, cross_process_lock=lock)
    real_ftruncate = session_journal_module.os.ftruncate

    def checked_ftruncate(fd: int, length: int) -> None:
        assert lock.is_locked
        real_ftruncate(fd, length)

    monkeypatch.setattr(session_journal_module.os, "ftruncate", checked_ftruncate)

    if operation == "read":
        assert [event.seq for event in journal.read_events()] == [1]
    else:
        assert journal.epoch_id == "epoch-1"

    assert seed.events_path.read_bytes().endswith(b"\n")


def test_concurrent_appends_keep_sequence_monotonic(tmp_path: Path) -> None:
    """Two processes appending to one journal produce one contiguous sequence."""
    import subprocess
    import sys
    import textwrap

    from kolega_code.cli.session_journal import SessionJournal

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    child_script = tmp_path / "append_loop.py"
    child_script.write_text(
        textwrap.dedent(
            """\
            import sys
            from pathlib import Path

            from filelock import FileLock

            from kolega_code.cli.session_journal import SessionJournal

            session_dir = Path(sys.argv[1])
            who = sys.argv[2]
            lock = FileLock(session_dir / ".lock", timeout=30, thread_local=False)
            journal = SessionJournal("test-session", session_dir, cross_process_lock=lock)
            for index in range(25):
                journal.append("test.event", actor=who, payload={"index": index}, epoch_id="epoch-1")
            """
        ),
        encoding="utf-8",
    )
    procs = [
        subprocess.Popen(
            [sys.executable, str(child_script), str(session_dir), "writer-a"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ),
        subprocess.Popen(
            [sys.executable, str(child_script), str(session_dir), "writer-b"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ),
    ]
    for proc in procs:
        _, stderr = proc.communicate(timeout=60)
        assert proc.returncode == 0, stderr

    journal = SessionJournal("test-session", session_dir)
    events = journal.read_events()
    assert len(events) == 50
    assert [event.seq for event in events] == list(range(1, 51))
    assert len({event.event_id for event in events}) == 50


def _journal_record(seq: int, event: object, *, record_type: str, session_id: str) -> str:
    record = {
        "version": 2,
        "id": f"id-{seq}",
        "session_id": session_id,
        "seq": seq,
        "epoch_id": "epoch",
        "turn_id": "turn",
        "timestamp": "2026-11-24T07:00:00+00:00",
        "actor": "agent",
        "type": record_type,
        "payload": {"event": event},
        "artifacts": [],
    }
    return json.dumps(record, separators=(",", ":"), ensure_ascii=False)


def _expected_head(events: FileSessionEventStore, session_id: str):
    """The summary produced by materializing every event, i.e. the old path."""
    records, _ = events._scan(0)
    if not records:
        return None
    open_turns = 0
    for _, event in records:
        if event.event_type == "turn_started":
            open_turns += 1
        elif event.event_type == "turn_ended":
            open_turns = max(0, open_turns - 1)
    last_seq, last_event = records[-1]
    return {
        "event_count": len(records),
        "last_seq": last_seq,
        "started_at": records[0][1].timestamp,
        "updated_at": last_event.timestamp,
        "duration_ms": last_event.elapsed_ms,
        "status": "open" if open_turns else "idle",
    }


@pytest.mark.asyncio
async def test_head_summary_matches_full_parse_for_unusual_records(tmp_path: Path) -> None:
    """The metadata fast path must agree with validated parsing on every shape."""
    from kolega_code.cli.session_journal import SessionJournal

    session_id = "head-summary"
    session_dir = tmp_path / session_id
    session_dir.mkdir()
    events_path = session_dir / "events.jsonl"

    def event(event_type: str, *, elapsed_ms: int, timestamp: str, **extra) -> dict:
        return AgentEvent(
            sender="agent",
            event_type=event_type,
            content={"text": "chunk"},
            elapsed_ms=elapsed_ms,
            timestamp=timestamp,
            **extra,
        ).model_dump(mode="json")

    coerced = event("assistant_delta", elapsed_ms=0, timestamp="2026-11-24T07:00:04+00:00")
    coerced["elapsed_ms"] = "42"
    extra_key = event("assistant_delta", elapsed_ms=0, timestamp="2026-11-24T07:00:05+00:00")
    extra_key["future_field"] = {"host": "value"}
    with_artifact = event(
        "assistant_delta",
        elapsed_ms=0,
        timestamp="2026-11-24T07:00:06+00:00",
        artifacts=[
            {
                "sha256": "0" * 64,
                "bytes": 7,
                "media_type": "text/plain",
                "purpose": "tool_result",
                "encoding": "utf-8",
            }
        ],
    )
    lines = [
        _journal_record(
            1,
            event("turn_started", elapsed_ms=0, timestamp="2026-11-24T07:00:00+00:00"),
            record_type=UI_EVENT_PREFIX + "turn_started",
            session_id=session_id,
        ),
        # Provider-facing records share the log but never reach the summary.
        _journal_record(2, {"message": None}, record_type="llm.message", session_id=session_id),
        # A malformed record must be ignored exactly like the validated path.
        '{"version":2,"type":"ui.assistant_delta"',
        # A UI record whose payload is not an event object is ignored too.
        _journal_record(3, "not-an-event", record_type=UI_EVENT_PREFIX + "assistant_delta", session_id=session_id),
        _journal_record(4, coerced, record_type=UI_EVENT_PREFIX + "assistant_delta", session_id=session_id),
        _journal_record(5, extra_key, record_type=UI_EVENT_PREFIX + "assistant_delta", session_id=session_id),
        _journal_record(6, with_artifact, record_type=UI_EVENT_PREFIX + "assistant_delta", session_id=session_id),
        _journal_record(
            7,
            event("turn_ended", elapsed_ms=0, timestamp="2026-11-24T07:00:07+00:00"),
            record_type=UI_EVENT_PREFIX + "turn_ended",
            session_id=session_id,
        ),
    ]
    events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    store_events = FileSessionEventStore(SessionJournal(session_id, session_dir))
    meta = await store_events.head(session_id)
    expected = _expected_head(store_events, session_id)
    assert expected is not None
    assert meta.event_count == expected["event_count"]
    assert meta.last_seq == expected["last_seq"]
    assert meta.started_at == expected["started_at"]
    assert meta.updated_at == expected["updated_at"]
    assert meta.duration_ms == expected["duration_ms"]
    assert meta.status == expected["status"]


@pytest.mark.asyncio
async def test_head_summary_refreshes_when_the_journal_appends(tmp_path: Path) -> None:
    """A cached summary must never hide bytes appended since the last call."""
    from kolega_code.cli.session_journal import SessionJournal

    session_id = "head-refresh"
    session_dir = tmp_path / session_id
    session_dir.mkdir()
    journal = SessionJournal(session_id, session_dir)
    journal.start_epoch("test")
    events = FileSessionEventStore(journal)

    await events.append(
        AgentEvent(session_id=session_id, sender="agent", event_type="log_message", content={"text": "a"})
    )
    first = await events.head(session_id)
    assert await events.head(session_id) == first

    await events.append(
        AgentEvent(session_id=session_id, sender="agent", event_type="log_message", content={"text": "b"})
    )
    refreshed = await events.head(session_id)
    assert refreshed.event_count == first.event_count + 1
    assert refreshed.last_seq > first.last_seq
    assert refreshed.updated_at is not None
