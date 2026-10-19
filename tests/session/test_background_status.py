"""Report-free status snapshots and lifecycle notifications for background jobs."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from kolega_code.cli.session_journal import SessionJournalError, SessionRecorder
from kolega_code.session.background import BackgroundAgentManager
from tests.session.test_background import Agent, recorder as recorder, submit

pytestmark = pytest.mark.usefixtures("isolated_cli_env")


@pytest.mark.asyncio
async def test_changed_observes_admission_and_slot_transitions() -> None:
    observed: list[list[tuple[str, str]]] = []
    started, second_started, release, second_release = (asyncio.Event() for _ in range(4))

    def changed() -> None:
        observed.append([(row["job_id"], row["status"]) for row in manager.status_snapshot()])

    manager = BackgroundAgentManager(max_running=1, on_changed=changed)

    async def first() -> str:
        started.set()
        await release.wait()
        return "first report"

    async def second() -> str:
        second_started.set()
        await second_release.wait()
        return "second report"

    try:
        await submit(manager, "first", run=first)
        await asyncio.wait_for(started.wait(), timeout=5)
        await submit(manager, "second", run=second)
        assert [("first", "queued")] in observed
        assert [("first", "running")] in observed
        assert [("first", "running"), ("second", "queued")] in observed
        assert not second_started.is_set()
        release.set()
        await asyncio.wait_for(second_started.wait(), timeout=5)
        assert [("first", "completed"), ("second", "running")] in observed
    finally:
        release.set()
        second_release.set()
        await manager.wait()
        await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["completed", "failed", "cancelled"])
@pytest.mark.parametrize("archive", [False, True])
async def test_terminal_snapshot_waits_for_durable_write_then_notifies_ready_and_delivery(
    recorder: SessionRecorder, monkeypatch: pytest.MonkeyPatch, outcome: str, archive: bool
) -> None:
    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    original = recorder.record_background_job_terminal
    changes: list[list[dict[str, Any]]] = []
    ready: list[list[dict[str, Any]]] = []
    manager = BackgroundAgentManager(
        recorder,
        on_changed=lambda: changes.append(manager.status_snapshot()),
        on_ready=lambda: ready.append(manager.status_snapshot()),
    )

    def blocked(*args: Any, **kwargs: Any) -> None:
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(timeout=10), "terminal write was not released"
        original(*args, **kwargs)

    async def run() -> str:
        if outcome == "failed":
            raise ValueError("private failure report")
        if outcome == "cancelled":
            raise asyncio.CancelledError
        return "private complete report"

    monkeypatch.setattr(recorder, "record_background_job_terminal", blocked)
    try:
        receipt = await submit(manager, run=run)
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert manager.status_snapshot() == [
            {
                "job_id": "job",
                "completion_id": receipt["completion_id"],
                "agent_type": "investigation",
                "status": outcome,
                "delivered": False,
                "label": "research",
                "archived": False,
                "ready": False,
                "settling": True,
                "persistence_error": False,
            }
        ]
        assert manager.active_count == 1
        assert manager.pending_count == 0
        assert ready == []
        assert any(rows and rows[0]["status"] == "queued" for rows in changes)
        assert any(rows and rows[0]["status"] == "running" for rows in changes)
        release.set()
        await manager.wait()
        assert len(ready) == 1
        assert ready[0][0]["status"] == outcome
        assert ready[0][0]["ready"] is True
        assert ready[0][0]["settling"] is False
        assert ready[0][0]["persistence_error"] is False
        assert ready[0] in changes
        agent = Agent(recorder)
        assert await manager.deliver(agent, archive=archive) == 1
        row = manager.status_snapshot()[0]
        assert row["status"] == outcome
        assert row["delivered"] is True
        assert row["archived"] is archive
        assert row["ready"] is False
        assert row["settling"] is False
        assert row["persistence_error"] is False
        assert changes[-1] == [row]
        assert len(agent.history) == (0 if archive else 1)
    finally:
        release.set()
        await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_via", ["deliver", "close"])
async def test_storage_failure_is_visible_until_event_gated_retry_settles(
    recorder: SessionRecorder, monkeypatch: pytest.MonkeyPatch, retry_via: str
) -> None:
    retry_entered = asyncio.Event()
    release_retry = threading.Event()
    loop = asyncio.get_running_loop()
    original = recorder.record_background_job_terminal
    attempts = 0
    changes: list[list[dict[str, Any]]] = []
    manager = BackgroundAgentManager(recorder, on_changed=lambda: changes.append(manager.status_snapshot()))

    def fail_then_block(*args: Any, **kwargs: Any) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise SessionJournalError("storage offline")
        loop.call_soon_threadsafe(retry_entered.set)
        assert release_retry.wait(timeout=10), "retry was not released"
        original(*args, **kwargs)

    monkeypatch.setattr(recorder, "record_background_job_terminal", fail_then_block)
    operation: asyncio.Task[Any] | None = None
    try:
        await submit(manager)
        with pytest.raises(SessionJournalError, match="storage offline"):
            await manager.wait()
        row = manager.status_snapshot()[0]
        assert (row["status"], row["settling"], row["ready"], row["persistence_error"]) == (
            "completed",
            True,
            False,
            True,
        )
        assert row["delivered"] is False
        assert [row] in changes
        operation = asyncio.create_task(manager.deliver(Agent(recorder)) if retry_via == "deliver" else manager.close())
        await asyncio.wait_for(retry_entered.wait(), timeout=5)
        assert manager.status_snapshot() == [row]
        assert not operation.done()
        release_retry.set()
        await operation
        settled = manager.status_snapshot()[0]
        assert settled["status"] == "completed"
        assert settled["settling"] is False
        assert settled["persistence_error"] is False
        assert settled["ready"] is (retry_via == "close")
        assert settled["delivered"] is (retry_via == "deliver")
        if retry_via == "deliver":
            assert changes[-1] == [settled]
        assert attempts == 2
    finally:
        release_retry.set()
        if operation is not None:
            await asyncio.gather(operation, return_exceptions=True)
        await manager.close()


@pytest.mark.asyncio
async def test_status_snapshot_does_not_copy_reports_or_metadata_and_is_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = BackgroundAgentManager()

    async def run() -> str:
        return "report not intended for the indicator"

    try:
        await manager.submit(
            "job",
            run,
            agent_type="investigation",
            task="界" * 81 + "\nprivate task detail",
            metadata={"private": ["secret"]},
        )
        await manager.wait()

        def forbid_copy(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("the status path must not deep-copy retained job data")

        with monkeypatch.context() as patch:
            patch.setattr("kolega_code.session.background.copy.deepcopy", forbid_copy)
            snapshot = manager.status_snapshot()
        assert set(snapshot[0]) == {
            "job_id",
            "completion_id",
            "agent_type",
            "status",
            "delivered",
            "label",
            "archived",
            "ready",
            "settling",
            "persistence_error",
        }
        assert snapshot[0]["label"] == "界" * 80
        original = dict(snapshot[0])
        for key in snapshot[0]:
            snapshot[0][key] = "mutated"
        snapshot.append({"job_id": "injected"})
        assert manager.status_snapshot() == [original]
        retained = manager.snapshot()[0]
        assert retained["result"] == "report not intended for the indicator"
        assert retained["metadata"] == {"private": ["secret"]}
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_callbacks_can_detach_and_reattach_to_a_ready_mailbox() -> None:
    started, release = asyncio.Event(), asyncio.Event()
    notifications: list[str] = []
    manager = BackgroundAgentManager(
        on_changed=lambda: notifications.append("old changed"),
        on_ready=lambda: notifications.append("old ready"),
        on_delivered=lambda row: notifications.append("old delivered"),
    )

    async def run() -> str:
        started.set()
        await release.wait()
        return "report"

    try:
        await submit(manager, run=run)
        await asyncio.wait_for(started.wait(), timeout=5)
        assert notifications
        manager.set_callbacks()
        notifications.clear()
        release.set()
        await manager.wait()
        assert notifications == []
        manager.set_callbacks(
            on_changed=lambda: notifications.append("changed"),
            on_ready=lambda: notifications.append("ready"),
            on_delivered=lambda row: notifications.append("delivered"),
        )
        assert notifications == ["changed", "ready"]
        assert await manager.deliver(Agent()) == 1
        assert notifications == ["changed", "ready", "changed", "delivered"]
    finally:
        release.set()
        await manager.close()


@pytest.mark.asyncio
async def test_callback_exceptions_do_not_break_lifecycle(caplog: pytest.LogCaptureFixture) -> None:
    notifications: set[str] = set()

    def fail(kind: str) -> None:
        notifications.add(kind)
        raise RuntimeError(f"{kind} observer unavailable")

    manager = BackgroundAgentManager(
        on_changed=lambda: fail("changed"),
        on_ready=lambda: fail("ready"),
        on_delivered=lambda row: fail("delivered"),
    )
    try:
        await submit(manager)
        await manager.wait()
        assert await manager.deliver(Agent()) == 1
        assert notifications == {"changed", "ready", "delivered"}
        assert manager.status_snapshot()[0]["delivered"] is True
        assert "Background activity callback failed" in caplog.text
        assert "Background readiness callback failed" in caplog.text
        assert "Background delivery callback failed" in caplog.text
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_closed_manager_suppresses_all_callbacks_including_reattachment() -> None:
    started = asyncio.Event()
    notifications: list[str] = []
    manager = BackgroundAgentManager()

    async def run() -> str:
        started.set()
        await asyncio.Event().wait()
        return "unreachable"

    try:
        await submit(manager, run=run)
        await asyncio.wait_for(started.wait(), timeout=5)
        manager.set_callbacks(
            on_changed=lambda: notifications.append("changed"),
            on_ready=lambda: notifications.append("ready"),
            on_delivered=lambda row: notifications.append("delivered"),
        )
        assert notifications == ["changed"]
        notifications.clear()
        await manager.close()
        row = manager.status_snapshot()[0]
        assert (row["status"], row["ready"], row["settling"]) == ("cancelled", True, False)
        manager.set_callbacks(
            on_changed=lambda: notifications.append("late changed"),
            on_ready=lambda: notifications.append("late ready"),
            on_delivered=lambda row: notifications.append("late delivered"),
        )
        assert await manager.deliver(Agent()) == 1
        assert notifications == []
    finally:
        await manager.close()
