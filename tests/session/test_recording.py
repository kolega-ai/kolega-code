"""RecordingConnectionManager: stamping, ordering, and volume control.

The recording wrapper is the only place ordering and durability are added, so
these tests pin the properties the rest of the system assumes: live delivery is
never suppressed, persistence happens before fan-out, and coalescing reduces the
stored volume without losing any streamed text.
"""

from __future__ import annotations

import base64
import asyncio
from copy import deepcopy
from typing import Any
from unittest.mock import AsyncMock

import pytest

from kolega_code.events import AgentConnectionManager, AgentEvent, ArtifactPurpose, ArtifactRef, KnownEventType
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
    artifacts = InMemoryArtifactStore()
    manager, inner, store = _manager(artifacts=artifacts)
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
    assert inner.events[0].artifacts == stored.artifacts
    assert "images" not in inner.events[0].content


@pytest.mark.asyncio
async def test_user_images_are_offloaded_once_for_live_and_recorded_events() -> None:
    artifacts = InMemoryArtifactStore()
    artifacts.put = AsyncMock(wraps=artifacts.put)
    manager, inner, store = _manager(artifacts=artifacts, policy=RetentionPolicy(inline_payload_chars=200))
    raw_images = [b"first-image", b"second-image"]
    payload = [
        {"media_type": media_type, "data": base64.b64encode(raw).decode("ascii")}
        for media_type, raw in zip(("image/png", "image/jpeg"), raw_images)
    ]
    original_content = {"user_text": "look at both", "text": "x" * 5_000, "images": payload, "image_count": 3}
    event = _event(KnownEventType.TURN_STARTED, **deepcopy(original_content))
    existing = await artifacts.put(
        b"existing", media_type="image/png", purpose=ArtifactPurpose.IMAGE, encoding="base64"
    )
    event.artifacts.append(existing)
    artifacts.put.reset_mock()

    await manager.broadcast_event(event, "w", "t")

    (stored,) = await store.read(SESSION_ID)
    (live,) = inner.events
    assert live.seq == stored.seq == 1
    assert live.content["text"] == original_content["text"]
    assert stored.content["text"] != original_content["text"]
    assert live.content["user_text"] == stored.content["user_text"] == "look at both"
    assert live.content["image_count"] == stored.content["image_count"] == 3
    image_refs = [ref for ref in stored.artifacts if ref.purpose == ArtifactPurpose.IMAGE]
    assert live.artifacts == image_refs
    assert image_refs[0] == existing
    assert [await artifacts.open(ref) for ref in image_refs[1:]] == raw_images
    assert [ref.media_type for ref in image_refs[1:]] == ["image/png", "image/jpeg"]
    assert artifacts.put.await_count == 3, "two image writes and one text write, not a second live-image pass"
    for normalized in (stored, live):
        assert "images" not in normalized.content
        assert all(item["data"] not in normalized.model_dump_json() for item in payload)
    assert event.content == original_content
    assert event.artifacts == [existing]


@pytest.mark.asyncio
@pytest.mark.parametrize("store_mode", ["missing", "failing"])
async def test_unavailable_artifact_store_strips_raw_images_and_keeps_original_count(store_mode: str) -> None:
    artifacts = InMemoryArtifactStore() if store_mode == "failing" else None
    if artifacts is not None:
        artifacts.put = AsyncMock(side_effect=OSError("artifact storage unavailable"))
    manager, inner, store = _manager(artifacts=artifacts)
    event = _event(
        KnownEventType.TURN_STARTED,
        user_text="still visible",
        image_count=4,
        images=[{"media_type": "image/png", "data": base64.b64encode(b"image").decode("ascii")}],
    )
    original = deepcopy(event.content)

    await manager.broadcast_event(event, "w", "t")

    (stored,) = await store.read(SESSION_ID)
    for normalized in (stored, inner.events[0]):
        assert normalized.content == {"user_text": "still visible", "image_count": 4}
        assert normalized.artifacts == []
    assert event.content == original


@pytest.mark.asyncio
async def test_malformed_images_do_not_lose_text_or_other_images() -> None:
    artifacts = InMemoryArtifactStore()
    artifacts.put = AsyncMock(wraps=artifacts.put)
    manager, inner, store = _manager(artifacts=artifacts)
    payload = [
        None,
        "not an image object",
        {"data": b"not base64 text"},
        {"data": "%%%invalid%%%"},
        {"data": ""},
        {"data": "https://example.invalid/image.png", "media_type": "image/png"},
        {"data": base64.b64encode(b"valid").decode("ascii"), "media_type": "image/png"},
    ]
    event = _event(KnownEventType.TURN_STARTED, user_text="retain me", images=payload)

    await manager.broadcast_event(event, "w", "t")

    (stored,) = await store.read(SESSION_ID)
    assert stored.content == {"user_text": "retain me", "image_count": len(payload)}
    assert stored.artifacts == inner.events[0].artifacts
    assert len(stored.artifacts) == 1
    assert await artifacts.open(stored.artifacts[0]) == b"valid"
    assert artifacts.put.await_count == 1
    assert "images" not in inner.events[0].content


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [None, {}, "not a list", []])
async def test_malformed_image_container_never_reaches_live_or_stored_event(payload: Any) -> None:
    manager, inner, store = _manager()

    await manager.broadcast_event(_event(KnownEventType.TURN_STARTED, user_text="retain me", images=payload), "w", "t")

    (stored,) = await store.read(SESSION_ID)
    assert stored.content == inner.events[0].content == {"user_text": "retain me"}


@pytest.mark.asyncio
async def test_failed_image_write_does_not_prevent_remaining_image_writes() -> None:
    artifacts = InMemoryArtifactStore()
    put = artifacts.put
    attempts: list[bytes] = []

    async def partially_failing_put(data: bytes, **kwargs: Any) -> ArtifactRef:
        attempts.append(data)
        if data == b"fail":
            raise OSError("first image failed")
        return await put(data, **kwargs)

    artifacts.put = AsyncMock(side_effect=partially_failing_put)
    manager, inner, store = _manager(artifacts=artifacts)
    event = _event(
        KnownEventType.TURN_STARTED,
        user_text="both images",
        images=[{"media_type": "image/png", "data": base64.b64encode(raw).decode("ascii")} for raw in (b"fail", b"ok")],
    )

    await manager.broadcast_event(event, "w", "t")

    (stored,) = await store.read(SESSION_ID)
    assert attempts == [b"fail", b"ok"]
    assert stored.content == {"user_text": "both images", "image_count": 2}
    assert stored.artifacts == inner.events[0].artifacts
    assert len(stored.artifacts) == 1
    assert await artifacts.open(stored.artifacts[0]) == b"ok"


@pytest.mark.asyncio
async def test_failed_event_append_still_delivers_normalized_live_images() -> None:
    artifacts = InMemoryArtifactStore()
    artifacts.put = AsyncMock(wraps=artifacts.put)
    manager, inner, store = _manager(artifacts=artifacts)
    store.append = AsyncMock(side_effect=OSError("event storage unavailable"))
    event = _event(
        KnownEventType.TURN_STARTED,
        user_text="still live",
        images=[{"data": base64.b64encode(b"image").decode("ascii"), "media_type": "image/png"}],
    )

    await manager.broadcast_event(event, "w", "t")

    (live,) = inner.events
    attempted = store.append.call_args.args[0]
    assert live.content == {"user_text": "still live", "image_count": 1}
    assert live.artifacts == attempted.artifacts
    assert len(live.artifacts) == 1
    assert live.seq is None
    assert artifacts.put.await_count == 1
    assert await artifacts.open(live.artifacts[0]) == b"image"


@pytest.mark.asyncio
async def test_task_list_snapshots_are_durable_complete_and_ordered_between_streams() -> None:
    artifacts = InMemoryArtifactStore()
    manager, inner, store = _manager(
        policy=RetentionPolicy(inline_payload_chars=32, stream_checkpoint_chars=10**9, stream_checkpoint_ms=10**9),
        artifacts=artifacts,
    )
    markdown = "# Keep the full source\n- [ ] " + "完整 task text " * 2_000 + "LAST TASK CHARACTER"
    try:
        await manager.broadcast_event(_delta("assistant_delta", "answer", "Before", streaming=True), "w", "t")
        await manager.broadcast_event(_event("task_list_updated", markdown=markdown), "w", "t")
        await manager.broadcast_event(_delta("assistant_delta", "answer", " after", streaming=False), "w", "t")
        await manager.broadcast_event(_event("task_list_updated", markdown=""), "w", "t")

        recorded = await store.read(SESSION_ID)
        assert [event.event_type for event in recorded] == [
            "assistant_delta",
            "task_list_updated",
            "assistant_delta",
            "task_list_updated",
        ]
        assert [event.seq for event in recorded] == [1, 2, 3, 4]
        assert [event.content for event in recorded] == [
            {"text": "Before", "complete": False},
            {"markdown": markdown},
            {"text": " after", "complete": True},
            {"markdown": ""},
        ]
        assert all(not event.artifacts for event in recorded)
        assert inner.seq_at_broadcast == [None, 2, 3, 4]
        assert inner.events[1].content == {"markdown": markdown}
        assert inner.events[3].content == {"markdown": ""}
    finally:
        await manager.flush()


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_stage", ["append", "delivery"])
async def test_task_list_recording_keeps_ownership_through_repeated_cancellation(blocked_stage: str) -> None:
    manager, inner, store = _manager()
    started = asyncio.Event()
    release = asyncio.Event()
    append = store.append
    deliver = inner.broadcast_event

    async def blocked_append(event: AgentEvent) -> int:
        if blocked_stage == "append":
            started.set()
            await release.wait()
        return await append(event)

    async def blocked_delivery(event: AgentEvent, workspace_id: str, thread_id: str) -> None:
        if blocked_stage == "delivery":
            started.set()
            await release.wait()
        await deliver(event, workspace_id, thread_id)

    store.append = AsyncMock(side_effect=blocked_append)
    inner.broadcast_event = AsyncMock(side_effect=blocked_delivery)
    task = asyncio.create_task(manager.broadcast_event(_event("task_list_updated", markdown=""), "w", "t"))
    try:
        await asyncio.wait_for(started.wait(), timeout=10)
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done(), "a cancelled snapshot must retain its writer and delivery owner"
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=10)
        (stored,) = await store.read(SESSION_ID)
        assert stored.event_type == "task_list_updated"
        assert stored.content == {"markdown": ""}
        assert len(inner.events) == 1
        assert inner.events[0].content == {"markdown": ""}
        assert store.append.await_count == inner.broadcast_event.await_count == 1
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_images_are_normalized_after_recording_retention_limit() -> None:
    artifacts = InMemoryArtifactStore()
    manager, inner, store = _manager(artifacts=artifacts, policy=RetentionPolicy(max_events=0))
    await manager.broadcast_event(_event(text="exhaust recording"), "w", "t")
    event = _event(
        KnownEventType.TURN_STARTED,
        user_text="live only",
        images=[{"data": base64.b64encode(b"image").decode("ascii"), "media_type": "image/png"}],
    )

    await manager.broadcast_event(event, "w", "t")

    (stored,) = await store.read(SESSION_ID)
    assert stored.event_type == KnownEventType.STREAM_TRUNCATED
    live = inner.events[-1]
    assert live.content == {"user_text": "live only", "image_count": 1}
    assert len(live.artifacts) == 1
    assert await artifacts.open(live.artifacts[0]) == b"image"
