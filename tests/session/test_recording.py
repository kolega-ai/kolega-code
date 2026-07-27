"""RecordingConnectionManager: stamping, ordering, and volume control.

The recording wrapper is the only place ordering and durability are added, so
these tests pin the properties the rest of the system assumes: live delivery is
never suppressed, persistence happens before fan-out, and coalescing reduces the
stored volume without losing any streamed text.
"""

from __future__ import annotations

from typing import Any

import pytest

from kolega_code.events import AgentConnectionManager, AgentEvent, KnownEventType
from kolega_code.session.inmemory import InMemoryArtifactStore, InMemorySessionEventStore
from kolega_code.session.recording import RecordingConnectionManager, RetentionPolicy

SESSION_ID = "session-under-test"


class _RecordingInner(AgentConnectionManager):
    """Captures what live clients would see, plus whether it was durable yet."""

    def __init__(self) -> None:
        self.events: list[AgentEvent] = []
        self.seq_at_broadcast: list[int | None] = []
        self.addresses: list[tuple[str, str]] = []

    async def connect(self, websocket: Any, workspace_id: str, thread_id: str, connection_type: str, user_info=None):
        return None

    def disconnect(self, websocket: Any, workspace_id: str, thread_id: str, connection_type: str) -> None:
        return None

    async def broadcast_event(self, event: AgentEvent, workspace_id: str, thread_id: str) -> None:
        self.events.append(event)
        self.seq_at_broadcast.append(event.seq)
        self.addresses.append((workspace_id, thread_id))

    def get_connection_count(self, workspace_id: str, thread_id: str) -> dict:
        return {}


def _manager(
    *,
    policy: RetentionPolicy | None = None,
    artifacts: InMemoryArtifactStore | None = None,
) -> tuple[RecordingConnectionManager, _RecordingInner, InMemorySessionEventStore]:
    inner = _RecordingInner()
    store = InMemorySessionEventStore()
    manager = RecordingConnectionManager(
        inner,
        store,
        session_id=SESSION_ID,
        artifact_store=artifacts,
        policy=policy,
    )
    return manager, inner, store


def _event(event_type: str = KnownEventType.LOG_MESSAGE, **content: Any) -> AgentEvent:
    return AgentEvent(sender="agent", event_type=event_type, content=dict(content))


@pytest.mark.asyncio
async def test_stamps_addressing_and_elapsed_time() -> None:
    manager, inner, store = _manager()

    await manager.broadcast_event(_event(text="hello"), "workspace-1", "thread-1")

    (stored,) = await store.read(SESSION_ID)
    assert stored.session_id == SESSION_ID
    assert stored.workspace_id == "workspace-1"
    assert stored.thread_id == "thread-1"
    assert stored.elapsed_ms >= 0
    # Addressing must be carried on the event, not only in the call arguments,
    # so a persisted event stays self-describing.
    assert inner.events[0].workspace_id == "workspace-1"


@pytest.mark.asyncio
async def test_persists_before_live_fan_out() -> None:
    """A client seeing seq N must be able to find seq N in the store afterwards."""
    manager, inner, _ = _manager()

    await manager.broadcast_event(_event(text="durable first"), "w", "t")

    assert inner.seq_at_broadcast == [1], (
        "the event handed to live clients already carried its stored seq, so persistence completed before fan-out"
    )


@pytest.mark.asyncio
async def test_live_delivery_survives_a_broken_store() -> None:
    class _ExplodingStore(InMemorySessionEventStore):
        async def append(self, event: AgentEvent) -> int:
            raise RuntimeError("disk on fire")

    inner = _RecordingInner()
    manager = RecordingConnectionManager(inner, _ExplodingStore(), session_id=SESSION_ID)

    await manager.broadcast_event(_event(text="still visible"), "w", "t")

    assert len(inner.events) == 1, "recording is an enhancement and must never silence the live UI"


@pytest.mark.asyncio
async def test_append_mode_streaming_loses_no_text() -> None:
    """Coalescing must reduce event count without dropping streamed content."""
    policy = RetentionPolicy(stream_checkpoint_chars=50, stream_checkpoint_ms=10**9)
    manager, inner, store = _manager(policy=policy)

    deltas = [f"chunk-{index:03d} " for index in range(40)]
    for delta in deltas:
        event = AgentEvent(
            sender="agent",
            event_type=KnownEventType.ASSISTANT_DELTA,
            uuid="stream-1",
            content={"text": delta, "complete": False},
            is_streaming=True,
        )
        await manager.broadcast_event(event, "w", "t")
    final = AgentEvent(
        sender="agent",
        event_type=KnownEventType.ASSISTANT_DELTA,
        uuid="stream-1",
        content={"text": "END", "complete": True},
        is_streaming=False,
    )
    await manager.broadcast_event(final, "w", "t")

    stored = await store.read(SESSION_ID)
    assert len(inner.events) == 41, "every delta must still reach live clients"
    assert len(stored) < 41, f"coalescing did not reduce stored volume: {len(stored)} records"
    assert "".join(event.content["text"] for event in stored) == "".join(deltas) + "END", (
        "the concatenated recording must reproduce the streamed text exactly"
    )


@pytest.mark.asyncio
async def test_replace_mode_streaming_keeps_only_the_latest_buffer() -> None:
    policy = RetentionPolicy(stream_checkpoint_chars=1, stream_checkpoint_ms=10**9)
    manager, _, store = _manager(policy=policy)

    for index in range(5):
        event = AgentEvent(
            sender="agent",
            event_type=KnownEventType.TOOL_STREAMING_UPDATE,
            uuid="tool-1",
            content={"text": "x" * (index + 1), "stream_mode": "replace"},
            is_streaming=True,
        )
        await manager.broadcast_event(event, "w", "t")

    stored = await store.read(SESSION_ID)
    # Replace-mode intermediates are genuinely redundant: each carries the whole
    # buffer, so the recording only needs the newest.
    assert stored[-1].content["text"] == "xxxxx"


@pytest.mark.asyncio
async def test_offloads_oversized_payload_to_artifact_store() -> None:
    artifacts = InMemoryArtifactStore()
    policy = RetentionPolicy(inline_payload_chars=200)
    manager, inner, store = _manager(policy=policy, artifacts=artifacts)

    huge = "y" * 5_000
    await manager.broadcast_event(_event(KnownEventType.CHAT_MESSAGE, text=huge), "w", "t")

    (stored,) = await store.read(SESSION_ID)
    assert len(stored.artifacts) == 1, "an oversized payload must be externalized"
    ref = stored.artifacts[0]
    assert ref.chars == 5_000
    assert len(stored.content["text"]) <= 200 + 200, "the stored body must be a bounded preview"
    assert await artifacts.open(ref) == huge.encode(), "the full payload must be retrievable"
    assert inner.events[0].content["text"] == huge, "live clients still get the full text"


@pytest.mark.asyncio
async def test_retention_ceiling_writes_an_explicit_truncation_marker() -> None:
    policy = RetentionPolicy(max_events=3)
    manager, _, store = _manager(policy=policy)

    for index in range(10):
        await manager.broadcast_event(_event(text=f"m{index}"), "w", "t")

    stored = await store.read(SESSION_ID)
    types = [event.event_type for event in stored]
    assert types.count(KnownEventType.STREAM_TRUNCATED) == 1, (
        f"hitting the ceiling must record exactly one truncation marker, got {types}"
    )
    assert types[-1] == KnownEventType.STREAM_TRUNCATED, "the marker must terminate the recording"


@pytest.mark.asyncio
async def test_high_volume_terminal_run_stays_bounded() -> None:
    """A 10k-chunk command must not produce a 10k-record recording."""
    manager, inner, store = _manager()

    chunk = "compiling module and emitting diagnostics\n"
    for _ in range(10_000):
        # Deliberately not is_streaming and with a fresh uuid each time: this is
        # exactly how terminal output really arrives.
        event = AgentEvent(
            sender="agent",
            event_type=KnownEventType.TERMINAL_OUTPUT,
            content={"output": chunk},
        )
        await manager.broadcast_event(event, "w", "t")
    await manager.flush()

    stored = await store.read(SESSION_ID)
    assert len(inner.events) == 10_000
    assert len(stored) < 200, f"expected a coarse recording, stored {len(stored)} records"
    recovered = "".join(event.content["output"] for event in stored)
    assert recovered == chunk * 10_000, "coalesced terminal output must still replay verbatim"


@pytest.mark.asyncio
async def test_elapsed_offset_continues_from_prior_recording() -> None:
    """Resuming a session continues replay time instead of restarting at zero."""
    store = InMemorySessionEventStore()
    first = RecordingConnectionManager(_RecordingInner(), store, session_id=SESSION_ID)
    await first.broadcast_event(_event(text="earlier"), "w", "t")

    resumed = RecordingConnectionManager(_RecordingInner(), store, session_id=SESSION_ID)
    await resumed.prime_elapsed_offset()
    await resumed.broadcast_event(_event(text="later"), "w", "t")

    events = await store.read(SESSION_ID)
    assert events[-1].elapsed_ms >= events[0].elapsed_ms, "a resumed recording must not rewind the replay clock"


def _delta(event_type: str, uuid: str, text: str, *, streaming: bool) -> AgentEvent:
    event = AgentEvent(
        sender="coder",
        event_type=event_type,
        content={"text": text, "complete": not streaming},
        is_streaming=streaming,
    )
    event.uuid = uuid
    return event


@pytest.mark.asyncio
async def test_interleaved_reasoning_and_prose_keep_chronological_order() -> None:
    """Reasoning that began first must be recorded first, however it ends.

    Reasoning and prose stream as two concurrent segments and the prose one
    reliably completes first. Writing buffers as they close therefore recorded
    the reply ahead of the thinking that produced it, and every consumer places
    a segment where its first record lands — so the exported replay showed the
    two swapped relative to the terminal, which renders from the generator.
    """
    manager, _, store = _manager()

    for event_type, uuid, text, streaming in (
        ("thinking_delta", "think", "The user wants me to ", True),
        ("assistant_delta", "prose", "I'll read the file.", True),
        ("assistant_delta", "prose", "", False),
        ("thinking_delta", "think", "read the file.", False),
    ):
        await manager.broadcast_event(_delta(event_type, uuid, text, streaming=streaming), "w", "t")
    await manager.flush()

    stored = await store.read(SESSION_ID)
    assert [event.event_type for event in stored] == [
        "thinking_delta",
        "assistant_delta",
        "thinking_delta",
    ], "the run that began first has to be written first, or a replay reorders the transcript"

    from kolega_code.session.projection import replay

    assert [(item.kind, item.text) for item in replay(stored).conversation] == [
        ("thinking", "The user wants me to read the file."),
        ("assistant", "I'll read the file."),
    ]


@pytest.mark.asyncio
async def test_elapsed_ms_never_runs_backwards_across_stored_events() -> None:
    """The replay timeline is binary-searched, so it must be non-decreasing.

    A coalesced record carries the elapsed_ms of the delta it was built from,
    which can predate a record already written because buffers do not close in
    the order they opened.
    """
    manager, _, store = _manager()

    await manager.broadcast_event(_delta("thinking_delta", "think", "a" * 40, streaming=True), "w", "t")
    await manager.broadcast_event(_delta("assistant_delta", "prose", "b" * 40, streaming=True), "w", "t")
    # Reach into the buffers to force the pathological case deterministically:
    # the younger run now claims to have started long after the older one.
    for state in manager._streams.values():  # pyright: ignore[reportPrivateUsage]
        state.template.elapsed_ms = 9_000 if state.open_index == 2 else 1_000
    await manager.broadcast_event(_delta("assistant_delta", "prose", "", streaming=False), "w", "t")
    await manager.broadcast_event(_delta("thinking_delta", "think", "", streaming=False), "w", "t")
    await manager.flush()

    stored = await store.read(SESSION_ID)
    elapsed = [event.elapsed_ms for event in stored]
    assert elapsed == sorted(elapsed), f"elapsed_ms must never step backwards along the log, got {elapsed}"


@pytest.mark.asyncio
async def test_a_stalled_stream_is_checkpointed_without_another_delta() -> None:
    """A run that goes quiet must still reach the store.

    The size and time checkpoints are only evaluated when the next delta of a
    run arrives, so a model that thinks for twenty seconds left the whole
    segment buffered and anyone watching a shared link saw nothing at all.
    """
    import asyncio

    manager, _, store = _manager(policy=RetentionPolicy(stream_checkpoint_ms=20))

    await manager.broadcast_event(_delta("thinking_delta", "think", "still working", streaming=True), "w", "t")
    assert await store.read(SESSION_ID) == [], "a fresh buffer should not have been written yet"

    for _ in range(50):
        await asyncio.sleep(0.02)
        if await store.read(SESSION_ID):
            break
    await manager.flush()

    stored = await store.read(SESSION_ID)
    assert stored, "an idle streaming buffer must be checkpointed on its own"
    assert stored[0].content["text"] == "still working"


@pytest.mark.asyncio
async def test_tool_images_become_artifacts_and_leave_the_event_body() -> None:
    """Images ride a presentation event, which is what lets a replay show them.

    Image bytes otherwise exist only on the provider-facing history record, and
    no presentation client reads those, so a shared replay could never render a
    screenshot however large it was.
    """
    import base64

    from kolega_code.events import ArtifactPurpose

    artifacts = InMemoryArtifactStore()
    manager, _, store = _manager(artifacts=artifacts)
    raw = b"\x89PNG\r\n\x1a\nnot-really-a-png"

    await manager.broadcast_event(
        _event(
            KnownEventType.CHAT_MESSAGE,
            message_type="tool_result",
            text="# marker.png",
            images=[{"media_type": "image/png", "data": base64.b64encode(raw).decode("ascii")}],
        ),
        "w",
        "t",
    )

    (stored,) = await store.read(SESSION_ID)
    assert "images" not in stored.content, "base64 must not be persisted inline in the event body"
    assert [ref.purpose for ref in stored.artifacts] == [ArtifactPurpose.IMAGE]
    assert await artifacts.open(stored.artifacts[0]) == raw
    assert stored.content["text"] == "# marker.png", "the text payload is unchanged by carrying an image"
