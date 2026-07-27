"""Conformance checks for :class:`~kolega_code.session.store.SessionEventStore`.

Any implementation — filesystem, in-memory, or a host's database backend — must
pass all of these. They are plain async functions that raise ``AssertionError``,
with no pytest dependency, so a host can run them from its own suite or a script.

The checks deliberately avoid assuming contiguous sequence numbers. A store may
share its sequence space with other records, so the only ordering guarantees are
*strictly increasing* and *exactly once*.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Optional, Protocol

from kolega_code.events import AgentEvent, KnownEventType
from kolega_code.session.store import SessionEventStore

#: Builds a fresh, empty store plus the session id to use with it. Called once
#: per check so no check can observe another's writes.
StoreFactory = Callable[[], Awaitable[tuple[SessionEventStore, str]]]


class ConformanceCheck(Protocol):
    __name__: str

    def __call__(self, factory: StoreFactory) -> Awaitable[None]: ...


def make_event(
    session_id: str,
    *,
    event_type: str = KnownEventType.LOG_MESSAGE,
    text: str = "",
    sender: str = "agent",
    is_streaming: bool = False,
    elapsed_ms: int = 0,
    uuid: Optional[str] = None,
) -> AgentEvent:
    """Build a minimal valid event for store tests."""
    event = AgentEvent(
        session_id=session_id,
        event_type=event_type,
        sender=sender,
        content={"text": text},
        is_streaming=is_streaming,
        elapsed_ms=elapsed_ms,
    )
    if uuid is not None:
        event.uuid = uuid
    return event


async def _drain(
    store: SessionEventStore,
    session_id: str,
    *,
    from_seq: int,
    count: int,
    timeout: float = 5.0,
) -> list[AgentEvent]:
    """Take ``count`` events from a tail, failing rather than hanging forever."""
    collected: list[AgentEvent] = []

    async def _collect() -> None:
        async for event in store.tail(session_id, from_seq=from_seq):
            collected.append(event)
            if len(collected) >= count:
                return

    await asyncio.wait_for(_collect(), timeout=timeout)
    return collected


async def check_append_assigns_increasing_seq(factory: StoreFactory) -> None:
    store, session_id = await factory()
    seqs = [await store.append(make_event(session_id, text=f"m{index}")) for index in range(5)]
    assert all(isinstance(seq, int) for seq in seqs), f"append must return ints, got {seqs}"
    assert seqs == sorted(seqs), f"sequence numbers must ascend, got {seqs}"
    assert len(set(seqs)) == len(seqs), f"sequence numbers must be unique, got {seqs}"


async def check_append_stamps_seq_on_event(factory: StoreFactory) -> None:
    store, session_id = await factory()
    event = make_event(session_id, text="hello")
    assert event.seq is None, "a fresh event must not carry a sequence number"
    seq = await store.append(event)
    assert event.seq == seq, "append must stamp the assigned seq onto the event it stored"


async def check_concurrent_appends_are_unique(factory: StoreFactory) -> None:
    """Two emitters must never collide; this is why seq is the store's job."""
    store, session_id = await factory()
    events = [make_event(session_id, text=f"c{index}") for index in range(24)]
    seqs = await asyncio.gather(*(store.append(event) for event in events))
    assert len(set(seqs)) == len(seqs), f"concurrent appends produced duplicate seqs: {sorted(seqs)}"
    stored = await store.read(session_id)
    assert len(stored) == len(events), f"expected {len(events)} stored events, found {len(stored)}"
    stored_seqs = [event.seq for event in stored]
    assert all(seq is not None for seq in stored_seqs), "every stored event must carry a seq"
    ordered = [seq for seq in stored_seqs if seq is not None]
    assert ordered == sorted(ordered), "read must return events in ascending seq order"


async def check_read_respects_range(factory: StoreFactory) -> None:
    store, session_id = await factory()
    seqs = [await store.append(make_event(session_id, text=f"r{index}")) for index in range(6)]
    middle = await store.read(session_id, from_seq=seqs[2], to_seq=seqs[4])
    assert [event.seq for event in middle] == seqs[2:5], f"inclusive range mismatch: {[e.seq for e in middle]}"
    beyond = await store.read(session_id, from_seq=seqs[-1] + 1_000)
    assert beyond == [], "a range past the end must return empty, not raise"


async def check_read_filters_types(factory: StoreFactory) -> None:
    store, session_id = await factory()
    await store.append(make_event(session_id, event_type=KnownEventType.LOG_MESSAGE))
    await store.append(make_event(session_id, event_type=KnownEventType.CHAT_MESSAGE))
    await store.append(make_event(session_id, event_type=KnownEventType.LOG_MESSAGE))
    only_chat = await store.read(session_id, types={KnownEventType.CHAT_MESSAGE})
    assert len(only_chat) == 1, f"type filter returned {len(only_chat)} events, expected 1"
    assert only_chat[0].event_type == KnownEventType.CHAT_MESSAGE


async def check_read_preserves_payload(factory: StoreFactory) -> None:
    store, session_id = await factory()
    event = make_event(session_id, text="round trip", elapsed_ms=4_321)
    event.content["extra"] = {"nested": [1, 2, 3]}
    await store.append(event)
    (restored,) = await store.read(session_id)
    assert restored.content["text"] == "round trip"
    assert restored.content["extra"] == {"nested": [1, 2, 3]}, "payloads must survive a round trip unchanged"
    assert restored.elapsed_ms == 4_321, "elapsed_ms is the replay timing key and must persist"
    assert restored.session_id == session_id


async def check_tail_replays_backlog(factory: StoreFactory) -> None:
    store, session_id = await factory()
    for index in range(3):
        await store.append(make_event(session_id, text=f"b{index}"))
    collected = await _drain(store, session_id, from_seq=1, count=3)
    assert [event.content["text"] for event in collected] == ["b0", "b1", "b2"]


async def check_tail_follows_live_appends(factory: StoreFactory) -> None:
    store, session_id = await factory()
    await store.append(make_event(session_id, text="before"))

    async def _append_later() -> None:
        await asyncio.sleep(0.05)
        await store.append(make_event(session_id, text="after"))

    writer = asyncio.create_task(_append_later())
    try:
        collected = await _drain(store, session_id, from_seq=1, count=2)
    finally:
        await writer
    assert [event.content["text"] for event in collected] == ["before", "after"], (
        "tail must hand off from backlog to live delivery without a gap"
    )


async def check_tail_has_no_gap_under_concurrent_writes(factory: StoreFactory) -> None:
    """Events appended *while the backlog is being read* must not be skipped."""
    store, session_id = await factory()
    total = 30
    for index in range(10):
        await store.append(make_event(session_id, text=f"x{index}"))

    async def _writer() -> None:
        for index in range(10, total):
            await store.append(make_event(session_id, text=f"x{index}"))
            await asyncio.sleep(0.005)

    writer = asyncio.create_task(_writer())
    try:
        collected = await _drain(store, session_id, from_seq=1, count=total, timeout=10.0)
    finally:
        await writer
    texts = [event.content["text"] for event in collected]
    assert texts == [f"x{index}" for index in range(total)], f"tail lost or reordered events: {texts}"


async def check_tail_resumes_without_duplicates(factory: StoreFactory) -> None:
    """A reconnecting client resumes at last_seen + 1 and sees each event once."""
    store, session_id = await factory()
    for index in range(6):
        await store.append(make_event(session_id, text=f"s{index}"))
    first = await _drain(store, session_id, from_seq=1, count=3)
    last_seen = first[-1].seq
    assert last_seen is not None
    rest = await _drain(store, session_id, from_seq=last_seen + 1, count=3)
    combined = [event.content["text"] for event in first + rest]
    assert combined == [f"s{index}" for index in range(6)], f"resume produced {combined}"


async def check_head_reports_empty(factory: StoreFactory) -> None:
    store, session_id = await factory()
    meta = await store.head(session_id)
    assert meta.event_count == 0 and meta.last_seq == 0, f"empty stream reported {meta}"
    assert meta.status == "empty"
    assert meta.started_at is None


async def check_head_summarizes_stream(factory: StoreFactory) -> None:
    store, session_id = await factory()
    await store.append(make_event(session_id, text="one", elapsed_ms=100))
    last_seq = await store.append(make_event(session_id, text="two", elapsed_ms=2_500))
    meta = await store.head(session_id)
    assert meta.event_count == 2, f"expected 2 events, got {meta.event_count}"
    assert meta.last_seq == last_seq
    assert meta.duration_ms == 2_500, "duration must come from the last event's elapsed_ms"
    assert meta.started_at is not None and meta.updated_at is not None


async def check_head_tracks_open_turn(factory: StoreFactory) -> None:
    store, session_id = await factory()
    await store.append(make_event(session_id, event_type=KnownEventType.TURN_STARTED))
    assert (await store.head(session_id)).status == "open"
    await store.append(make_event(session_id, event_type=KnownEventType.TURN_ENDED))
    assert (await store.head(session_id)).status == "idle"


CONFORMANCE_CHECKS: tuple[ConformanceCheck, ...] = (
    check_append_assigns_increasing_seq,
    check_append_stamps_seq_on_event,
    check_concurrent_appends_are_unique,
    check_read_respects_range,
    check_read_filters_types,
    check_read_preserves_payload,
    check_tail_replays_backlog,
    check_tail_follows_live_appends,
    check_tail_has_no_gap_under_concurrent_writes,
    check_tail_resumes_without_duplicates,
    check_head_reports_empty,
    check_head_summarizes_stream,
    check_head_tracks_open_turn,
)


async def run_conformance(factory: StoreFactory) -> list[str]:
    """Run every check, returning the names that passed. Raises on first failure."""
    passed: list[str] = []
    for check in CONFORMANCE_CHECKS:
        await check(factory)
        passed.append(check.__name__)
    return passed
