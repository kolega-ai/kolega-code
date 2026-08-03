"""Filesystem event store: journal integration and isolation guarantees.

Presentation events share the session journal's sequence space. The safety
property that makes that acceptable is that they are invisible to provider
history replay — if a UI event could reach the model's conversation, or could
make a session fail to load, sharing the log would be the wrong design.
"""

from __future__ import annotations

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
