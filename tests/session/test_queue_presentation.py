from __future__ import annotations

import asyncio
from copy import deepcopy
from pathlib import Path
import threading
from typing import Any
from unittest.mock import create_autospec

import pytest

from kolega_code.cli.session_event_store import FileSessionEventStore
from kolega_code.cli.session_journal import SessionJournal
from kolega_code.events import AgentConnectionManager, AgentEvent, AgentEventEmitter
from kolega_code.session.queue import QueuePresentation, queue_item, queue_preview
from kolega_code.session.recording import RecordingConnectionManager


def test_preview_projects_display_only_and_preserves_literal_empty_peer_body() -> None:
    assert queue_preview("  hello\n  world\t ") == "hello world"
    assert queue_preview("") == "(empty)"
    assert queue_preview("x" * 120) == "x" * 120
    assert queue_preview("x" * 121) == "x" * 117 + "…"
    assert queue_preview("🐍" * 121) == "🐍" * 117 + "…"
    assert queue_item(
        "peer-1", "", origin={"kind": "peer", "session_id": "s", "title": "<literal>", "private": "never"}
    ) == {
        "queue_id": "peer-1",
        "text": "(empty)",
        "image_count": 0,
        "origin": {"kind": "peer", "session_id": "s", "title": "<literal>"},
    }


@pytest.mark.asyncio
@pytest.mark.skipif(not hasattr(asyncio, "eager_task_factory"), reason="Python 3.12+ eager tasks")
@pytest.mark.parametrize("failure", [None, OSError, asyncio.CancelledError])
async def test_eager_immediate_observer_never_leaves_a_completed_worker_owner(failure: Any) -> None:
    observed: list[int] = []

    async def emit(snapshot: dict[str, Any]) -> None:
        observed.append(snapshot["revision"])
        if failure is not None:
            raise failure()

    owner = QueuePresentation(emit, source="tui")
    loop = asyncio.get_running_loop()
    factory = loop.get_task_factory()
    loop.set_task_factory(getattr(asyncio, "eager_task_factory"))
    try:
        owner.publish([], reason="snapshot")
        assert owner._worker is None or not owner._worker.done()
        await owner.flush()
        owner.publish([queue_item("next", "next input")], reason="enqueued")
        assert owner._worker is None or not owner._worker.done()
        await owner.aclose()
        assert observed == [0, 1]
    finally:
        loop.set_task_factory(factory)


def test_snapshots_captured_before_an_event_loop_are_drained_in_order() -> None:
    observed: list[dict[str, Any]] = []

    async def emit(snapshot: dict[str, Any]) -> None:
        observed.append(snapshot)

    owner = QueuePresentation(emit, source="ask")
    owner.publish([], reason="snapshot")
    owner.publish([queue_item("one", "first input")], reason="enqueued")
    assert owner._worker is None and not observed
    asyncio.run(owner.aclose())
    assert [snapshot["revision"] for snapshot in observed] == [0, 1]
    assert [snapshot["items"] for snapshot in observed] == [
        [],
        [{"queue_id": "one", "text": "first input", "image_count": 0, "origin": None}],
    ]


@pytest.mark.asyncio
async def test_ordered_immutable_snapshots_survive_failed_and_cancelled_publication() -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    observed: list[dict[str, Any]] = []

    async def emit(snapshot: dict[str, Any]) -> None:
        observed.append(deepcopy(snapshot))
        if snapshot["revision"] == 0:
            entered.set()
            await release.wait()
        elif snapshot["revision"] == 1:
            raise OSError("recording failed")
        elif snapshot["revision"] == 2:
            raise asyncio.CancelledError

    owner = QueuePresentation(emit, source="tui")
    owner.publish([], reason="snapshot")
    await entered.wait()
    items = [queue_item("one", "first"), queue_item("two", "second")]
    owner.publish(items, reason="enqueued")
    items[0]["text"] = "mutated later"
    owner.publish(items[1:], reason="dequeued")
    owner.publish([], reason="cleared")
    assert len(observed) == 1  # capture does not wait for a stuck observer
    closing = asyncio.create_task(owner.aclose())
    await asyncio.sleep(0)
    closing.cancel()
    await asyncio.sleep(0)
    closing.cancel()
    await asyncio.sleep(0)
    assert not closing.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert [snapshot["revision"] for snapshot in observed] == [0, 1, 2, 3]
    assert [snapshot["reason"] for snapshot in observed] == ["snapshot", "enqueued", "dequeued", "cleared"]
    assert [[item["text"] for item in snapshot["items"]] for snapshot in observed] == [
        [],
        ["first", "second"],
        ["second"],
        [],
    ]
    assert len({snapshot["queue_id"] for snapshot in observed}) == 1
    assert all(snapshot["source"] == "tui" for snapshot in observed)
    owner.publish([queue_item("late", "not admitted by this closed observer")], reason="enqueued")
    await owner.aclose()
    assert len(observed) == 4
    successor = QueuePresentation(emit, source="ask")
    assert successor._queue_id != owner._queue_id
    await successor.aclose()


@pytest.mark.asyncio
async def test_recording_owns_blocked_thread_and_live_delivery_through_repeated_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = SessionJournal("queue", tmp_path)
    journal.start_epoch("test")
    store = FileSessionEventStore(journal, append_timeout=0.001)
    live = create_autospec(AgentConnectionManager, instance=True)
    recording = RecordingConnectionManager(live, store, session_id="queue")
    emitter = AgentEventEmitter(recording, "workspace", "thread", "cli")
    entered, release = threading.Event(), threading.Event()
    append = store._append_sync

    def blocked(event: AgentEvent) -> int:
        if event.event_type == "queue_updated":
            entered.set()
            assert release.wait(10)
        return append(event)

    monkeypatch.setattr(store, "_append_sync", blocked)
    owner = QueuePresentation(emitter.queue_updated, source="tui")
    owner.publish([queue_item("one", "still captured")], reason="enqueued")
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        worker = owner._worker
        assert worker is not None
        worker.cancel()
        closing = asyncio.create_task(owner.aclose())
        await asyncio.sleep(0.02)  # exceeds append_timeout: the writer remains owned
        closing.cancel()
        await asyncio.sleep(0)
        closing.cancel()
        await asyncio.sleep(0)
        assert not closing.done()
        assert not live.broadcast_event.await_count
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await closing
    finally:
        release.set()
        await owner.aclose()
    events = await store.read("queue")
    assert len(events) == 1
    event = events[0]
    assert event.event_type == "queue_updated" and event.seq is not None
    assert event.content["items"] == [{"queue_id": "one", "text": "still captured", "image_count": 0, "origin": None}]
    assert (event.session_id, event.workspace_id, event.thread_id, event.sender) == (
        "queue",
        "workspace",
        "thread",
        "cli",
    )
    assert event.sub_agent_info is None
    live.broadcast_event.assert_awaited_once()
    assert live.broadcast_event.call_args.args[0].content == event.content


@pytest.mark.asyncio
async def test_child_emitter_cannot_present_a_root_queue() -> None:
    manager = create_autospec(AgentConnectionManager, instance=True)
    emitter = AgentEventEmitter(manager, "workspace", "thread", "child", lambda: {"agent_id": "child"})
    await emitter.queue_updated({"items": []})
    manager.broadcast_event.assert_not_awaited()
