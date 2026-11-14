from __future__ import annotations

import asyncio
from copy import deepcopy
from pathlib import Path
import threading
from typing import Any
from unittest.mock import create_autospec

import pytest

from kolega_code.cli.goal import GoalState
from kolega_code.cli.session_event_store import FileSessionEventStore
from kolega_code.cli.session_journal import SessionJournal
from kolega_code.events import AgentConnectionManager, AgentEvent, AgentEventEmitter
from kolega_code.session.goal import GoalPresentation
from kolega_code.session.recording import RecordingConnectionManager


@pytest.mark.asyncio
async def test_immutable_ordered_lifecycle_survives_observer_failure_and_repeated_close_cancellation() -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    observed: list[dict[str, Any]] = []

    async def emit(snapshot: dict[str, Any]) -> None:
        observed.append(deepcopy(snapshot))
        if snapshot["revision"] == 0:
            entered.set()
            await release.wait()
        elif snapshot["revision"] == 1:
            raise OSError("observer failed")
        elif snapshot["revision"] == 2:
            raise asyncio.CancelledError

    owner = GoalPresentation(emit, source="tui")
    state = GoalState(condition="literal <goal>", started_at="recorded-start").to_dict()
    owner.publish(state, reason="started")
    await entered.wait()
    state["last_reason"] = "not yet"
    owner.publish(state, reason="evaluated", evaluation_met=False)
    state["paused"] = True
    owner.publish(state, reason="paused")
    state["condition"] = "new goal"
    owner.publish(state, reason="replaced")
    owner.publish(None, reason="cleared")
    closing = asyncio.create_task(owner.aclose())
    await asyncio.sleep(0)
    closing.cancel()
    await asyncio.sleep(0)
    closing.cancel()
    await asyncio.sleep(0)
    assert not closing.done() and len(observed) == 1
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert [item["reason"] for item in observed] == ["started", "evaluated", "paused", "replaced", "cleared"]
    assert [item["revision"] for item in observed] == [0, 1, 2, 3, 4]
    assert [item["evaluation_met"] for item in observed] == [None, False, False, None, None]
    assert observed[0]["goal"]["last_reason"] == ""
    assert observed[1]["goal"]["paused"] is False
    assert observed[2]["goal"]["condition"] == "literal <goal>"
    assert observed[-1]["goal"] is None
    assert len({item["generation_id"] for item in observed}) == 1
    owner.publish(state, reason="started")
    await owner.aclose()
    assert len(observed) == 5
    successor = GoalPresentation(emit, source="ask")
    assert successor._generation_id != owner._generation_id
    await successor.aclose()


@pytest.mark.asyncio
@pytest.mark.skipif(not hasattr(asyncio, "eager_task_factory"), reason="Python 3.12+ eager tasks")
@pytest.mark.parametrize("failure", [None, OSError, asyncio.CancelledError])
async def test_eager_immediate_observer_does_not_leave_finished_worker(failure: Any) -> None:
    observed: list[int] = []

    async def emit(snapshot: dict[str, Any]) -> None:
        observed.append(snapshot["revision"])
        if failure is not None:
            raise failure()

    owner = GoalPresentation(emit, source="ask")
    loop = asyncio.get_running_loop()
    factory = loop.get_task_factory()
    loop.set_task_factory(getattr(asyncio, "eager_task_factory"))
    try:
        owner.publish(None, reason="snapshot")
        assert owner._worker is None or not owner._worker.done()
        await owner.flush()
        owner.publish(None, reason="cleared")
        await owner.aclose()
        assert observed == [0, 1]
    finally:
        loop.set_task_factory(factory)


def test_pre_loop_capture_is_drained_without_mutating_original_state() -> None:
    observed: list[dict[str, Any]] = []

    async def emit(snapshot: dict[str, Any]) -> None:
        observed.append(snapshot)

    owner = GoalPresentation(emit, source="ask")
    state = GoalState(condition="saved paused goal", started_at="original", paused=True).to_dict()
    expected = deepcopy(state)
    owner.publish(state, reason="snapshot")
    state["condition"] = "later"
    owner.publish(None, reason="cleared")
    assert owner._worker is None and not observed
    asyncio.run(owner.aclose())
    assert observed[0]["goal"] == expected
    assert observed[0]["evaluation_met"] is None
    assert observed[1]["goal"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("recording_wrapper", [False, True], ids=["headless", "tui"])
async def test_real_journal_and_live_delivery_retain_ownership_through_repeated_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recording_wrapper: bool
) -> None:
    journal = SessionJournal("goal", tmp_path)
    journal.start_epoch("test")
    store = FileSessionEventStore(journal, append_timeout=0.001)
    live = create_autospec(AgentConnectionManager, instance=True)
    recording = RecordingConnectionManager(live, store, session_id="goal")
    emitter = AgentEventEmitter(recording, "workspace", "thread", "cli")
    entered, release = threading.Event(), threading.Event()
    append = store._append_sync

    def blocked(event: AgentEvent) -> int:
        if event.event_type == "goal_updated":
            entered.set()
            assert release.wait(10)
        return append(event)

    monkeypatch.setattr(store, "_append_sync", blocked)

    async def direct(snapshot: dict[str, Any]) -> None:
        await store.append(AgentEvent(sender="ask", event_type="goal_updated", session_id="goal", content=snapshot))

    owner = GoalPresentation(
        emitter.goal_updated if recording_wrapper else direct, source="tui" if recording_wrapper else "ask"
    )
    expected = GoalState(condition="still owned", started_at="original").to_dict()
    owner.publish(expected, reason="started")
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert owner._worker is not None
        owner._worker.cancel()
        closing = asyncio.create_task(owner.aclose())
        await asyncio.sleep(0.02)
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
    events = await store.read("goal")
    assert len(events) == 1
    event = events[0]
    assert event.event_type == "goal_updated" and event.seq is not None
    assert event.content["goal"] == expected
    assert event.sub_agent_info is None
    if recording_wrapper:
        assert (event.session_id, event.workspace_id, event.thread_id, event.sender) == (
            "goal",
            "workspace",
            "thread",
            "cli",
        )
        live.broadcast_event.assert_awaited_once()
        assert live.broadcast_event.call_args.args[0].content == event.content
    else:
        assert event.session_id == "goal" and event.sender == "ask"
        live.broadcast_event.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("recording_wrapper", [False, True], ids=["headless", "tui"])
async def test_runner_wide_task_cancellation_cannot_abandon_goal_writer_or_live_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recording_wrapper: bool
) -> None:
    journal = SessionJournal("goal", tmp_path)
    journal.start_epoch("test")
    store = FileSessionEventStore(journal)
    live = create_autospec(AgentConnectionManager, instance=True)
    recording = RecordingConnectionManager(live, store, session_id="goal")
    emitter = AgentEventEmitter(recording, "workspace", "thread", "cli")
    entered, release = threading.Event(), threading.Event()
    append = store._append_sync

    def blocked(event: AgentEvent) -> int:
        entered.set()
        assert release.wait(20)
        return append(event)

    monkeypatch.setattr(store, "_append_sync", blocked)

    async def direct(snapshot: dict[str, Any]) -> None:
        await store.append(AgentEvent(sender="ask", event_type="goal_updated", session_id="goal", content=snapshot))

    existing = asyncio.all_tasks()
    owner = GoalPresentation(
        emitter.goal_updated if recording_wrapper else direct, source="tui" if recording_wrapper else "ask"
    )
    owner.publish(GoalState(condition="runner shutdown", started_at="original").to_dict(), reason="started")
    closing: asyncio.Task[None] | None = None
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        owned = asyncio.all_tasks() - existing
        assert len(owned) >= 2
        for task in owned:
            task.cancel()
        closing = asyncio.create_task(owner.aclose())
        await asyncio.sleep(0.05)
        assert not closing.done(), "shutdown released the actual writer"
        live.broadcast_event.assert_not_awaited()
        release.set()
        await closing
    finally:
        release.set()
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        await owner.aclose()
    assert len(await store.read("goal")) == 1
    if recording_wrapper:
        live.broadcast_event.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("info", [{}, {"agent_id": "child"}])
async def test_child_emitter_cannot_present_root_goal(info: dict[str, Any]) -> None:
    manager = create_autospec(AgentConnectionManager, instance=True)
    emitter = AgentEventEmitter(manager, "workspace", "thread", "child", lambda: info)
    await emitter.goal_updated({"goal": None})
    manager.broadcast_event.assert_not_awaited()
