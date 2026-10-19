"""Event-gated tests of background ownership, durable outcomes and mailbox delivery."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest

from kolega_code.cli.session_journal import (
    TOOL_RESULT_PREVIEW_CHARS,
    SessionJournal,
    SessionJournalError,
    SessionRecorder,
)
from kolega_code.cli.session_store import SessionStore
from kolega_code.llm.models import ContentBlock, Message, TextBlock
from kolega_code.session.background import (
    BackgroundAgentManager,
    BackgroundEpochError,
    BackgroundJobError,
    BackgroundJobLimitError,
    BackgroundManagerClosedError,
)


class Agent:
    def __init__(self, recorder: SessionRecorder | None = None) -> None:
        self.session_recorder = recorder
        self.history: list[Message] = []
        self.fail_append = False

    def append_user_message(self, blocks: list[ContentBlock]) -> None:
        if self.fail_append:
            raise RuntimeError("live history unavailable")
        self.history.append(Message(role="user", content=blocks))


@pytest.fixture
def recorder(tmp_path: Path) -> SessionRecorder:
    journal = SessionJournal("background-tests", tmp_path / "session")
    journal.start_epoch("test")
    return SessionRecorder(journal, recover=False)


async def report() -> str:
    return "complete report"


async def submit(manager: BackgroundAgentManager, job_id: str = "job", **kwargs: Any) -> dict[str, Any]:
    return await manager.submit(
        job_id, kwargs.pop("run", report), agent_type="investigation", task="research", **kwargs
    )


def completion_events(recorder: SessionRecorder) -> list[Any]:
    return [
        event
        for event in recorder.journal.read_events()
        if event.event_type == "context.message" and event.payload.get("completion_id")
    ]


@pytest.mark.asyncio
async def test_submission_returns_while_child_blocked_and_retains_full_outcome(recorder: SessionRecorder) -> None:
    started, release = asyncio.Event(), asyncio.Event()
    ready: list[str] = []
    manager = BackgroundAgentManager(recorder, on_ready=lambda: ready.append("ready"))

    async def run() -> str:
        started.set()
        await release.wait()
        return "full report"

    receipt = await submit(manager, run=run, metadata={"workspace": "original"})
    await started.wait()
    assert receipt["job_id"] == "job"
    assert receipt["status"] == "queued"
    assert manager.active_count == 1
    assert manager.pending_count == 0
    assert manager.has_work
    receipt["metadata"]["workspace"] = "mutated"
    release.set()
    await manager.wait()
    assert ready == ["ready"]
    assert manager.active_count == 0
    assert manager.pending_count == 1
    state = manager.snapshot("job")[0]
    assert state["status"] == "completed" and state["result"] == "full report"
    assert state["metadata"]["workspace"] == "original"
    assert recorder.read_background_jobs(epoch_id=recorder.journal.epoch_id)[0]["result"] == "full report"
    await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("open_turn", [False, True])
async def test_delivery_running_or_idle_is_one_root_message(recorder: SessionRecorder, open_turn: bool) -> None:
    if open_turn:
        recorder.start_turn(Message(role="user", content=[TextBlock(text="parent prompt")]))
    delivered: list[dict[str, Any]] = []
    manager = BackgroundAgentManager(recorder, on_delivered=delivered.append)
    agent = Agent(recorder)
    await submit(manager, metadata={"child_id": "child"})
    await manager.wait()
    assert await manager.deliver(agent) == 1
    assert await manager.deliver(agent) == 0
    assert not manager.has_work
    events = completion_events(recorder)
    assert len(events) == 1
    assert events[0].depth == 0
    assert events[0].turn_id == recorder.current_turn_id
    assert events[0].payload["origin"] == {"kind": "subagent_completion"}
    assert events[0].payload["metadata"] == {"child_id": "child"}
    assert events[0].payload["task"] == "research"
    assert len(agent.history) == len(delivered) == 1
    content = agent.history[0].content
    assert isinstance(content, list) and isinstance(content[0], TextBlock)
    text = content[0].text
    assert text.startswith('<system-reminder source="subagent">')
    assert "complete report" in text
    assert delivered[0]["text"] == text
    assert delivered[0]["status"] == "completed"
    replay, _ = SessionStore(recorder.journal.session_dir.parent / "store")._replay(
        recorder.journal.read_events(), recorder.journal
    )
    assert replay[-1]["content"][0]["text"] == text
    await manager.close()


@pytest.mark.asyncio
async def test_failure_and_cancellation_are_durable_outcomes(recorder: SessionRecorder) -> None:
    started, never = asyncio.Event(), asyncio.Event()
    manager = BackgroundAgentManager(recorder)

    async def failed() -> str:
        raise ValueError("child failed with detail")

    async def blocked() -> str:
        started.set()
        await never.wait()
        return "unexpected"

    await submit(manager, "failure", run=failed)
    await submit(manager, "cancel", run=blocked)
    await started.wait()
    cancelled = await manager.cancel("cancel")
    assert cancelled["status"] == "cancelled"
    await manager.wait()
    states = {row["job_id"]: row for row in manager.snapshot()}
    assert states["failure"]["status"] == "failed"
    assert states["failure"]["result"] == "ValueError: child failed with detail"
    assert states["cancel"]["status"] == "cancelled"
    assert manager.pending_count == 2
    assert await manager.deliver(Agent(recorder)) == 2
    assert await manager.cancel("failure") == manager.snapshot("failure")[0]
    with pytest.raises(BackgroundJobError, match="Unknown"):
        await manager.cancel("missing")
    await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("second_action", ["cancel", "close"])
@pytest.mark.parametrize("cancel_waiter", [False, True])
async def test_repeated_cancellation_joins_child_cleanup(
    recorder: SessionRecorder, second_action: str, cancel_waiter: bool
) -> None:
    manager = BackgroundAgentManager(recorder)
    started, cleanup_entered, release_cleanup, cleanup_finished = (asyncio.Event() for _ in range(4))

    async def run() -> str:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_entered.set()
            await release_cleanup.wait()
            cleanup_finished.set()
        return "unexpected"

    await submit(manager, run=run)
    await started.wait()
    child = manager._jobs["job"].task
    assert child is not None
    first = asyncio.create_task(manager.cancel("job"))
    await cleanup_entered.wait()
    assert child.cancelling() == 1
    if cancel_waiter:
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert not child.done()
    second = asyncio.create_task(manager.cancel("job") if second_action == "cancel" else manager.close())
    barrier = asyncio.Event()
    asyncio.get_running_loop().call_soon(barrier.set)
    try:
        await barrier.wait()  # Both public operations have reached their child join.
        assert child.cancelling() == 1
        assert not second.done()
        assert cancel_waiter or not first.done()
        assert manager.active_count == 1 and manager.pending_count == 0
        assert not cleanup_finished.is_set()
    finally:
        release_cleanup.set()
        await asyncio.gather(first, second, return_exceptions=True)
    assert cleanup_finished.is_set()
    if not cancel_waiter:
        assert first.result()["status"] == "cancelled"
    assert second.result() == (None if second_action == "close" else manager.snapshot("job")[0])
    assert manager.active_count == 0 and manager.pending_count == 1
    assert recorder.read_background_jobs(epoch_id=recorder.journal.epoch_id)[0]["status"] == "cancelled"
    assert sum(event.event_type == "background.job.terminal" for event in recorder.journal.read_events()) == 1
    await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["cancel", "close"])
async def test_lifecycle_joins_child_already_cancelling_itself(recorder: SessionRecorder, action: str) -> None:
    manager = BackgroundAgentManager(recorder)
    cleanup_entered, release_cleanup, cleanup_finished = (asyncio.Event() for _ in range(3))

    async def run() -> str:
        child = asyncio.current_task()
        assert child is not None
        child.cancel()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_entered.set()
            await release_cleanup.wait()
            cleanup_finished.set()
        return "unexpected"

    await submit(manager, run=run)
    await cleanup_entered.wait()
    child = manager._jobs["job"].task
    assert child is not None
    operation = asyncio.create_task(manager.cancel("job") if action == "cancel" else manager.close())
    barrier = asyncio.Event()
    asyncio.get_running_loop().call_soon(barrier.set)
    try:
        await barrier.wait()
        assert child.cancelling() == 1
        assert not operation.done() and not cleanup_finished.is_set()
        assert manager.pending_count == 0
    finally:
        release_cleanup.set()
        await asyncio.gather(operation, return_exceptions=True)
    operation.result()
    assert cleanup_finished.is_set()
    assert recorder.read_background_jobs(epoch_id=recorder.journal.epoch_id)[0]["status"] == "cancelled"
    await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["cancel", "close"])
async def test_cancellation_before_execution_starts_persists_without_running_child(
    recorder: SessionRecorder, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    manager = BackgroundAgentManager(recorder)
    execution_entered, release_execution, child_ran = (asyncio.Event() for _ in range(3))
    execute = manager._execute

    async def delayed_execute(*args: Any, **kwargs: Any) -> None:
        execution_entered.set()
        await release_execution.wait()
        await execute(*args, **kwargs)

    async def run() -> str:
        child_ran.set()
        return "unexpected"

    monkeypatch.setattr(manager, "_execute", delayed_execute)
    await submit(manager, run=run)
    await execution_entered.wait()
    job = manager._jobs["job"]
    assert job.task is not None and not job.started
    operation = asyncio.create_task(manager.cancel("job") if action == "cancel" else manager.close())
    barrier = asyncio.Event()
    asyncio.get_running_loop().call_soon(barrier.set)
    try:
        await barrier.wait()
        assert job.cancel_requested
        assert job.task.cancelling() == 0
        assert not operation.done() and not child_ran.is_set()
    finally:
        release_execution.set()
        await asyncio.gather(operation, return_exceptions=True)
    operation.result()
    assert not child_ran.is_set()
    assert manager.active_count == 0 and manager.pending_count == 1
    assert recorder.read_background_jobs(epoch_id=recorder.journal.epoch_id)[0]["status"] == "cancelled"
    await manager.close()


@pytest.mark.asyncio
async def test_admission_and_execution_caps_and_queued_cancellation(recorder: SessionRecorder) -> None:
    started, release = asyncio.Event(), asyncio.Event()
    launches: list[str] = []
    manager = BackgroundAgentManager(recorder, max_running=1, max_jobs=2)

    async def first() -> str:
        launches.append("first")
        started.set()
        await release.wait()
        return "first"

    async def second() -> str:
        launches.append("second")
        return "second"

    await submit(manager, "first", run=first)
    await started.wait()
    await submit(manager, "second", run=second)
    assert manager.active_count == 2
    assert manager.snapshot("second")[0]["status"] == "queued"
    with pytest.raises(BackgroundJobLimitError):
        await submit(manager, "over-cap")
    assert (await manager.cancel("second"))["status"] == "cancelled"
    assert launches == ["first"]
    await submit(manager, "replacement", run=second)
    release.set()
    await manager.wait()
    assert launches == ["first", "second"]
    assert manager.pending_count == 3
    await manager.close()


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_cancel_children(recorder: SessionRecorder) -> None:
    started, release, waiting = asyncio.Event(), asyncio.Event(), asyncio.Event()
    manager = BackgroundAgentManager(recorder)

    async def run() -> str:
        started.set()
        await release.wait()
        return "survived parent cancellation"

    await submit(manager, run=run)
    await started.wait()

    async def wait() -> None:
        waiting.set()
        await manager.wait()

    waiter = asyncio.create_task(wait())
    await waiting.wait()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert manager.active_count == 1
    release.set()
    await manager.wait()
    assert manager.snapshot()[0]["status"] == "completed"
    await manager.close()


@pytest.mark.asyncio
async def test_completion_wait_is_level_triggered_and_does_not_lose_arrival(recorder: SessionRecorder) -> None:
    started, release = asyncio.Event(), asyncio.Event()
    manager = BackgroundAgentManager(recorder)
    await manager.wait_for_completion()  # No work is immediately ready.

    async def run() -> str:
        started.set()
        await release.wait()
        return "arrived"

    await submit(manager, run=run)
    await started.wait()
    waiting = asyncio.Event()

    async def wait() -> None:
        waiting.set()
        await manager.wait_for_completion()

    waiter = asyncio.create_task(wait())
    await waiting.wait()
    assert not waiter.done()
    release.set()
    await waiter
    assert manager.pending_count == 1
    await manager.wait_for_completion()  # Arrival before waiting is not lost.
    await manager.deliver(Agent(recorder))
    await manager.wait_for_completion()
    await manager.close()


@pytest.mark.asyncio
async def test_cancelled_completion_waiter_does_not_discard_pending(recorder: SessionRecorder) -> None:
    started, release, waiting = asyncio.Event(), asyncio.Event(), asyncio.Event()
    manager = BackgroundAgentManager(recorder)

    async def run() -> str:
        started.set()
        await release.wait()
        return "survived"

    async def wait() -> None:
        waiting.set()
        await manager.wait_for_completion()

    await submit(manager, run=run)
    await started.wait()
    waiter = asyncio.create_task(wait())
    await waiting.wait()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    release.set()
    await manager.wait()
    await manager.wait_for_completion()
    assert await manager.deliver(Agent(recorder)) == 1
    await manager.close()


@pytest.mark.asyncio
async def test_duplicate_submissions_and_concurrent_delivery(recorder: SessionRecorder) -> None:
    manager = BackgroundAgentManager(recorder)
    agent = Agent(recorder)
    await submit(manager)
    with pytest.raises(BackgroundJobError, match="already exists"):
        await submit(manager)
    await manager.wait()
    assert sorted(await asyncio.gather(manager.deliver(agent), manager.deliver(agent))) == [0, 1]
    assert len(completion_events(recorder)) == len(agent.history) == 1
    await manager.close()


@pytest.mark.asyncio
async def test_live_history_failure_retries_without_duplicate_durable_insert(recorder: SessionRecorder) -> None:
    manager = BackgroundAgentManager(recorder)
    agent = Agent(recorder)
    agent.fail_append = True
    await submit(manager)
    await manager.wait()
    with pytest.raises(RuntimeError, match="live history"):
        await manager.deliver(agent)
    assert len(completion_events(recorder)) == 1
    assert manager.pending_count == 1
    assert agent.history == []
    agent.fail_append = False
    assert await manager.deliver(agent) == 1
    assert len(completion_events(recorder)) == len(agent.history) == 1
    await manager.close()


@pytest.mark.asyncio
async def test_recovery_preserves_full_artifact_reports_and_skips_replayed_delivery(
    recorder: SessionRecorder,
) -> None:
    full_report = "result-" * (TOOL_RESULT_PREVIEW_CHARS // 7 + 100)
    manager = BackgroundAgentManager(recorder)

    async def large() -> str:
        return full_report

    await submit(manager, "pending", run=large)
    await manager.wait()
    terminal = next(e for e in recorder.journal.read_events() if e.event_type == "background.job.terminal")
    assert terminal.payload["result_artifact"]["chars"] == len(full_report)
    assert "result" not in terminal.payload
    await manager.close()

    # New recorder/journal objects model process restart, not shared in-memory state.
    reopened = SessionRecorder(SessionJournal(recorder.journal.session_id, recorder.journal.session_dir), recover=False)
    recovered = BackgroundAgentManager(reopened)
    await recovered.recover()
    await recovered.recover()
    assert recovered.active_count == 0 and recovered.pending_count == 1
    assert recovered.snapshot()[0]["result"] == full_report
    agent = Agent(reopened)
    assert await recovered.deliver(agent) == 1
    await recovered.close()

    resumed = BackgroundAgentManager(reopened)
    await resumed.recover()
    assert resumed.pending_count == 0
    assert resumed.snapshot()[0]["delivered"]
    assert await resumed.deliver(Agent(reopened)) == 0
    assert len(completion_events(reopened)) == 1
    replay, _ = SessionStore(recorder.journal.session_dir.parent / "store")._replay(
        reopened.journal.read_events(), reopened.journal
    )
    assert full_report in replay[0]["content"][0]["text"]
    await resumed.close()


@pytest.mark.asyncio
async def test_unfinished_prior_process_job_is_interrupted_once(recorder: SessionRecorder) -> None:
    epoch = recorder.journal.epoch_id
    recorder.record_background_job_submitted(
        {
            "job_id": "lost",
            "completion_id": "completion-lost",
            "agent_type": "investigation",
            "task": "interrupted task",
            "metadata": {"workspace": "old"},
            "epoch_id": epoch,
            "status": "queued",
            "result": "",
            "delivered": False,
        },
        epoch_id=epoch,
    )
    manager = BackgroundAgentManager(recorder)
    await asyncio.gather(manager.recover(), manager.recover())
    assert manager.snapshot()[0]["status"] == "interrupted"
    assert manager.pending_count == 1
    assert sum(e.event_type == "background.job.terminal" for e in recorder.journal.read_events()) == 1
    assert await manager.deliver(Agent(recorder)) == 1
    await manager.close()


@pytest.mark.asyncio
async def test_epoch_fencing_retains_late_results_without_injection(recorder: SessionRecorder) -> None:
    started, release = asyncio.Event(), asyncio.Event()
    manager = BackgroundAgentManager(recorder)
    epoch = recorder.journal.epoch_id

    async def run() -> str:
        started.set()
        await release.wait()
        return "old epoch result"

    await submit(manager, run=run)
    await started.wait()
    recorder.start_epoch("reset")
    release.set()
    await manager.wait()
    agent = Agent(recorder)
    with pytest.raises(BackgroundEpochError):
        await manager.deliver(agent)
    with pytest.raises(BackgroundEpochError):
        await submit(manager, "new")
    assert not agent.history and not completion_events(recorder)
    assert recorder.read_background_jobs(epoch_id=epoch)[0]["result"] == "old epoch result"
    fresh = BackgroundAgentManager(recorder)
    await fresh.recover()
    assert fresh.snapshot() == []
    await fresh.close()
    await manager.close()


@pytest.mark.asyncio
async def test_workspace_retirement_archives_only_new_results_and_acknowledges_across_recovery(
    recorder: SessionRecorder,
) -> None:
    recorder.start_turn(Message(role="user", content=[TextBlock("ordinary history")]))
    recorder.record_assistant(Message(role="assistant", content=[TextBlock("ordinary response")]))
    recorder.finish_turn("completed")
    manager = BackgroundAgentManager(recorder)
    old_root = Agent(recorder)
    await submit(manager, "already-consumed")
    await manager.wait()
    await manager.deliver(old_root)
    await submit(manager, "retiring")
    await manager.wait()
    await manager.close()
    assert await manager.deliver(old_root, archive=True) == 1
    assert len(old_root.history) == 1
    events = completion_events(recorder)
    assert [(event.payload["job_id"], event.payload["archived"]) for event in events] == [
        ("already-consumed", False),
        ("retiring", True),
    ]
    reopened = SessionRecorder(SessionJournal(recorder.journal.session_id, recorder.journal.session_dir))
    history, _ = SessionStore(recorder.journal.session_dir.parent / "store")._replay(
        reopened.journal.read_events(), reopened.journal
    )
    assert len(history) == 3
    assert history[0]["content"][0]["text"] == "ordinary history"
    assert history[1]["content"][0]["text"] == "ordinary response"
    assert '"job_id": "already-consumed"' in history[2]["content"][0]["text"]
    resumed = BackgroundAgentManager(reopened)
    await resumed.recover()
    assert {row["job_id"] for row in resumed.snapshot() if row["delivered"]} == {"already-consumed", "retiring"}
    assert resumed.pending_count == 0
    assert await resumed.deliver(Agent(reopened)) == 0
    assert len(completion_events(reopened)) == 2
    await resumed.close()


@pytest.mark.asyncio
async def test_archiving_retries_a_committed_completion_after_live_append_failure(recorder: SessionRecorder) -> None:
    manager = BackgroundAgentManager(recorder)
    root = Agent(recorder)
    root.fail_append = True
    await submit(manager)
    await manager.wait()
    with pytest.raises(RuntimeError, match="live history unavailable"):
        await manager.deliver(root)
    assert len(completion_events(recorder)) == 1
    assert not completion_events(recorder)[0].payload["archived"]
    await manager.close()
    assert await manager.deliver(root, archive=True) == 1
    assert await manager.deliver(root, archive=True) == 0
    assert not root.history
    assert len(completion_events(recorder)) == 1
    assert sum(event.event_type == "context.message_archived" for event in recorder.journal.read_events()) == 1
    history, _ = SessionStore(recorder.journal.session_dir.parent / "store")._replay(
        recorder.journal.read_events(), recorder.journal
    )
    assert history == []
    assert len(recorder.read_archived_background_messages()) == 1
    resumed = BackgroundAgentManager(recorder)
    await resumed.recover()
    assert resumed.pending_count == 0
    assert resumed.snapshot()[0]["archived"] is True
    assert resumed.snapshot()[0]["delivered"] is True
    await resumed.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["queued", "completed", "inserted", "admitted-after-fence"])
async def test_recovery_fences_old_workspace_jobs_at_the_durable_switch(
    recorder: SessionRecorder,
    state: str,
    tmp_path: Path,
) -> None:
    epoch = recorder.journal.epoch_id
    old_path, new_path = str(tmp_path / "old"), str(tmp_path / "new")
    # Reports already acknowledged in the old workspace remain ordinary history.
    previous = BackgroundAgentManager(recorder)
    await submit(previous, "already-consumed", metadata={"project_path": old_path})
    await previous.wait()
    await previous.deliver(Agent(recorder))
    await previous.close()
    lost = {
        "job_id": "lost",
        "completion_id": "completion-lost",
        "agent_type": "investigation",
        "task": "old task",
        "metadata": {"project_path": old_path},
        "epoch_id": epoch,
        "status": "queued",
        "result": "",
        "delivered": False,
    }
    if state != "admitted-after-fence":
        recorder.record_background_job_submitted(lost, epoch_id=epoch)
    recorder.record_workspace_switched(old_root=old_path, new_root=new_path)
    if state == "admitted-after-fence":
        # The tool boundary has committed but the launching turn still runs
        # against the old resources until the host applies the switch.
        recorder.record_background_job_submitted(lost, epoch_id=epoch)
    if state != "queued":
        lost.update(status="completed", result="old-workspace-only report")
        recorder.record_background_job_terminal(lost, epoch_id=epoch)
    if state == "inserted":
        # An older process may have acknowledged after the fence but before
        # applying retirement. Replay must fence this too, before attachment.
        recorder.record_background_completion(
            lost,
            Message(role="user", content=[TextBlock("old-workspace-only report")]),
            epoch_id=epoch,
        )
    reopened = SessionRecorder(SessionJournal(recorder.journal.session_id, recorder.journal.session_dir))
    manager = BackgroundAgentManager(reopened)
    await manager.recover()
    root = Agent(reopened)
    await manager.deliver(root)
    assert root.history == []
    assert not getattr(root, "background_needs_turn", False)
    history, _ = SessionStore(recorder.journal.session_dir.parent / "store")._replay(
        reopened.journal.read_events(), reopened.journal
    )
    assert len(history) == 1
    assert '"job_id": "already-consumed"' in history[0]["content"][0]["text"]
    assert len(reopened.read_archived_background_messages()) == 1
    row = next(row for row in manager.snapshot() if row["job_id"] == "lost")
    assert row["delivered"] is True
    assert row["archived"] is True
    assert row["status"] == ("interrupted" if state == "queued" else "completed")
    await submit(manager, "new-workspace", metadata={"project_path": new_path})
    await manager.wait()
    await manager.deliver(root)
    assert len(root.history) == 1
    assert '"job_id": "new-workspace"' in root.history[0].to_dict()["content"][0]["text"]
    await manager.close()


@pytest.mark.asyncio
async def test_recovery_preserves_irreversible_archives_across_returning_workspace_fences(
    recorder: SessionRecorder,
    tmp_path: Path,
) -> None:
    epoch = recorder.journal.epoch_id
    path_a, path_b, path_c = (str(tmp_path / name) for name in ("a", "b", "c"))
    recorder.start_turn(Message(role="user", content=[TextBlock("ordinary history")]))
    recorder.record_assistant(Message(role="assistant", content=[TextBlock("ordinary response")]))
    recorder.finish_turn("completed")

    def record_job(job_id: str, project_path: str, *, inserted: bool = False) -> None:
        job = {
            "job_id": job_id,
            "completion_id": f"completion-{job_id}",
            "agent_type": "investigation",
            "task": "inspect workspace",
            "metadata": {"project_path": project_path},
            "epoch_id": epoch,
            "status": "queued",
            "result": "",
            "delivered": False,
        }
        recorder.record_background_job_submitted(job, epoch_id=epoch)
        job.update(status="completed", result=f"report for {job_id}")
        recorder.record_background_job_terminal(job, epoch_id=epoch)
        if inserted:
            recorder.record_background_completion(
                job, Message(role="user", content=[TextBlock(job["result"])]), epoch_id=epoch
            )

    record_job("a-consumed", path_a, inserted=True)
    record_job("a-pending", path_a)
    recorder.record_workspace_switched(old_root=path_a, new_root=path_b)
    # A legacy process can acknowledge old-resource work after committing the
    # fence. Returning to A later must never undo that report's disposition.
    record_job("a-late-inserted", path_a, inserted=True)
    record_job("b-consumed", path_b, inserted=True)
    record_job("b-pending", path_b)
    recorder.record_workspace_switched(old_root=path_b, new_root=path_a)
    record_job("b-late", path_b)
    record_job("a-return-consumed", path_a, inserted=True)
    record_job("a-return-pending", path_a)
    recorder.record_workspace_switched(old_root=path_a, new_root=path_c)
    record_job("a-late-again", path_a)
    record_job("c-current", path_c)

    reopened = SessionRecorder(SessionJournal(recorder.journal.session_id, recorder.journal.session_dir))
    manager = BackgroundAgentManager(reopened)
    await manager.recover()
    root = Agent(reopened)
    await manager.deliver(root)
    assert len(root.history) == 1
    assert '"job_id": "c-current"' in root.history[0].to_dict()["content"][0]["text"]

    archived_ids = {
        "a-pending",
        "a-late-inserted",
        "b-pending",
        "b-late",
        "a-return-pending",
        "a-late-again",
    }
    rows = {row["job_id"]: row for row in manager.snapshot()}
    assert {job_id for job_id, row in rows.items() if row["archived"]} == archived_ids
    assert all(row["delivered"] and row["status"] == "completed" for row in rows.values())
    history, _ = SessionStore(recorder.journal.session_dir.parent / "store")._replay(
        reopened.journal.read_events(), reopened.journal
    )
    assert len(history) == 6
    history_text = "\n".join(block["text"] for message in history for block in message["content"])
    for job_id in archived_ids:
        assert f"report for {job_id}" not in history_text
    for job_id in {"a-consumed", "b-consumed", "a-return-consumed", "c-current"}:
        assert f"report for {job_id}" in history_text
    archived = reopened.read_archived_background_messages()
    assert len(archived) == len(archived_ids)
    archive_text = "\n".join(block["text"] for message in archived for block in message["content"])
    for job_id in archived_ids:
        assert f"report for {job_id}" in archive_text

    resumed = BackgroundAgentManager(reopened)
    await resumed.recover()
    assert resumed.pending_count == 0
    assert await resumed.deliver(Agent(reopened)) == 0
    assert len(completion_events(reopened)) == 10
    await resumed.close()
    await manager.close()


@pytest.mark.asyncio
async def test_rebinding_recorder_or_delivering_to_child_is_rejected(recorder: SessionRecorder) -> None:
    manager = BackgroundAgentManager(recorder)
    await submit(manager)
    await manager.wait()
    child = recorder.scoped_child(agent_id="child", agent_name="child", parent_tool_call_id="tool", depth=1)
    with pytest.raises(BackgroundEpochError, match="root"):
        await manager.deliver(Agent(child))
    manager.recorder = None
    with pytest.raises(BackgroundEpochError, match="rebound"):
        await manager.deliver(Agent(recorder))
    await manager.close()


@pytest.mark.asyncio
async def test_close_cancels_running_and_queued_jobs_suppresses_callbacks_and_retains_mailbox(
    recorder: SessionRecorder,
) -> None:
    started, never = asyncio.Event(), asyncio.Event()
    notifications: list[str] = []
    manager = BackgroundAgentManager(recorder, max_running=1, on_ready=lambda: notifications.append("ready"))

    async def run() -> str:
        started.set()
        await never.wait()
        return "unexpected"

    await submit(manager, "running", run=run)
    await started.wait()
    await submit(manager, "queued", run=run)
    await manager.close()
    await manager.close()
    assert manager.active_count == 0 and manager.pending_count == 2
    assert {row["status"] for row in manager.snapshot()} == {"cancelled"}
    assert notifications == []
    assert len(recorder.read_background_jobs(epoch_id=recorder.journal.epoch_id)) == 2
    with pytest.raises(BackgroundManagerClosedError):
        await submit(manager, "too late")
    # Shutdown does not discard outcomes; the host may still explicitly drain.
    assert await manager.deliver(Agent(recorder)) == 2


@pytest.mark.asyncio
async def test_memory_only_manager_callbacks_can_be_reattached() -> None:
    manager = BackgroundAgentManager()
    await submit(manager)
    await manager.wait()
    ready: list[str] = []
    delivered: list[dict[str, Any]] = []
    manager.set_callbacks(on_ready=lambda: ready.append("ready"), on_delivered=delivered.append)
    assert ready == ["ready"]
    assert await manager.deliver(Agent()) == 1
    assert delivered[0]["result"] == "complete report"
    await manager.close()


@pytest.mark.asyncio
async def test_admission_persistence_failure_is_not_accepted_or_started(
    recorder: SessionRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = BackgroundAgentManager(recorder)
    ran = False

    async def run() -> str:
        nonlocal ran
        ran = True
        return "unexpected"

    def fail(*args: Any, **kwargs: Any) -> None:
        raise SessionJournalError("disk unavailable")

    monkeypatch.setattr(recorder, "record_background_job_submitted", fail)
    with pytest.raises(SessionJournalError, match="disk unavailable"):
        await submit(manager, run=run)
    assert not ran and manager.snapshot() == [] and not manager.has_work
    await manager.close()


@pytest.mark.asyncio
async def test_terminal_persistence_failure_is_exposed_and_full_result_retained(
    recorder: SessionRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = BackgroundAgentManager(recorder)

    def fail(*args: Any, **kwargs: Any) -> None:
        raise SessionJournalError("terminal write failed")

    monkeypatch.setattr(recorder, "record_background_job_terminal", fail)
    await submit(manager)
    with pytest.raises(SessionJournalError, match="terminal write failed"):
        await manager.wait()
    assert manager.snapshot()[0]["result"] == "complete report"
    assert manager.pending_count == 0
    with pytest.raises(SessionJournalError, match="terminal write failed"):
        await manager.deliver(Agent(recorder))
    with pytest.raises(SessionJournalError, match="terminal write failed"):
        await manager.close()


@pytest.mark.asyncio
async def test_failed_completion_commit_remains_pending(
    recorder: SessionRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = BackgroundAgentManager(recorder)
    await submit(manager)
    await manager.wait()
    original = recorder.record_background_completion

    def fail(*args: Any, **kwargs: Any) -> bool:
        raise SessionJournalError("completion write failed")

    monkeypatch.setattr(recorder, "record_background_completion", fail)
    agent = Agent(recorder)
    with pytest.raises(SessionJournalError, match="completion write failed"):
        await manager.deliver(agent)
    assert manager.pending_count == 1 and not agent.history and not completion_events(recorder)
    monkeypatch.setattr(recorder, "record_background_completion", original)
    assert await manager.deliver(agent) == 1
    await manager.close()


@pytest.mark.asyncio
async def test_cancelled_threaded_commit_settles_before_delivery_ownership_is_released(
    recorder: SessionRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = BackgroundAgentManager(recorder)
    await submit(manager)
    await manager.wait()
    entered, attempted = asyncio.Event(), asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    original = recorder.record_background_completion

    def blocked(*args: Any, **kwargs: Any) -> bool:
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(timeout=10), "test did not release commit"
        return original(*args, **kwargs)

    monkeypatch.setattr(recorder, "record_background_completion", blocked)
    agent = Agent(recorder)
    delivery = asyncio.create_task(manager.deliver(agent))
    await entered.wait()
    delivery.cancel()
    delivery.cancel()  # Repeated cancellation cannot detach the threaded writer.

    async def second_delivery() -> int:
        attempted.set()
        return await manager.deliver(agent)

    second = asyncio.create_task(second_delivery())
    await attempted.wait()
    assert not delivery.done() and not second.done()
    assert manager.pending_count == 1 and not agent.history
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await delivery
    assert await second == 0
    assert len(completion_events(recorder)) == len(agent.history) == 1
    assert manager.pending_count == 0
    await manager.close()


@pytest.mark.asyncio
async def test_cancellation_of_submission_waits_for_admission_then_retains_owned_job(
    recorder: SessionRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = BackgroundAgentManager(recorder)
    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    original = recorder.record_background_job_submitted

    def blocked(*args: Any, **kwargs: Any) -> None:
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(timeout=10)
        original(*args, **kwargs)

    monkeypatch.setattr(recorder, "record_background_job_submitted", blocked)
    admission = asyncio.create_task(submit(manager))
    await entered.wait()
    admission.cancel()
    assert not admission.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await admission
    await manager.wait()
    assert manager.snapshot()[0]["status"] == "completed"
    assert manager.pending_count == 1
    await manager.close()


def test_recorder_primitives_are_root_only_idempotent_and_epoch_fenced(recorder: SessionRecorder) -> None:
    epoch = recorder.journal.epoch_id
    job = {
        "job_id": "job",
        "completion_id": "completion",
        "agent_type": "type",
        "task": "task",
        "metadata": {},
        "status": "queued",
        "result": "",
    }
    recorder.record_background_job_submitted(job, epoch_id=epoch)
    recorder.record_background_job_submitted(job, epoch_id=epoch)
    terminal = {**job, "status": "completed", "result": "result"}
    recorder.record_background_job_terminal(terminal, epoch_id=epoch)
    recorder.record_background_job_terminal(terminal, epoch_id=epoch)
    message = Message(role="user", content=[TextBlock(text="completion text")])
    assert recorder.record_background_completion(terminal, message, epoch_id=epoch)
    assert not recorder.record_background_completion(terminal, message, epoch_id=epoch)
    assert len(recorder.journal.read_events()) == 4
    assert recorder.read_background_jobs(epoch_id=epoch)[0]["delivered"]
    child = recorder.scoped_child(agent_id="child", agent_name="child", parent_tool_call_id=None, depth=1)
    with pytest.raises(SessionJournalError, match="root"):
        child.read_background_jobs(epoch_id=epoch)
    recorder.start_epoch("reset")
    with pytest.raises(SessionJournalError, match="inactive epoch"):
        recorder.record_background_completion(terminal, message, epoch_id=epoch)
    with pytest.raises(SessionJournalError, match="inactive epoch"):
        recorder.record_background_job_submitted(job, epoch_id=epoch)


@pytest.mark.parametrize("kwargs", [{"max_jobs": 0}, {"max_running": 0}, {"max_jobs": -1}])
def test_invalid_limits(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="positive"):
        BackgroundAgentManager(**kwargs)


@pytest.mark.asyncio
async def test_close_cancellation_still_joins_and_persists_cleanup(recorder: SessionRecorder) -> None:
    started, cleanup, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    never = asyncio.Event()
    manager = BackgroundAgentManager(recorder)

    async def run() -> str:
        started.set()
        try:
            await never.wait()
        finally:
            cleanup.set()
            await release.wait()
        return "unexpected"

    await submit(manager, run=run)
    await started.wait()
    closing = asyncio.create_task(manager.close())
    await cleanup.wait()
    closing.cancel()
    closing.cancel()
    assert not closing.done()
    assert manager.active_count == 1
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert manager.active_count == 0
    assert manager.pending_count == 1
    assert recorder.read_background_jobs(epoch_id=recorder.journal.epoch_id)[0]["status"] == "cancelled"
    await manager.close()


@pytest.mark.asyncio
async def test_partial_batch_failure_only_retries_uncommitted_items(
    recorder: SessionRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = BackgroundAgentManager(recorder)
    await submit(manager, "first")
    await submit(manager, "second")
    await manager.wait()
    original = recorder.record_background_completion
    count = 0

    def fail_second(*args: Any, **kwargs: Any) -> bool:
        nonlocal count
        count += 1
        if count == 2:
            raise SessionJournalError("second completion failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(recorder, "record_background_completion", fail_second)
    agent = Agent(recorder)
    with pytest.raises(SessionJournalError, match="second completion"):
        await manager.deliver(agent)
    assert manager.pending_count == 1
    assert len(completion_events(recorder)) == len(agent.history) == 1
    assert manager.snapshot(completion_events(recorder)[0].payload["job_id"])[0]["delivered"]
    assert await manager.deliver(agent) == 1
    assert len(completion_events(recorder)) == len(agent.history) == 2
    await manager.close()


@pytest.mark.asyncio
async def test_recovery_failure_is_exposed_and_retryable(
    recorder: SessionRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = BackgroundAgentManager(recorder)
    original = recorder.read_background_jobs

    def fail(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        raise SessionJournalError("recovery unavailable")

    monkeypatch.setattr(recorder, "read_background_jobs", fail)
    with pytest.raises(SessionJournalError, match="recovery unavailable"):
        await manager.recover()
    monkeypatch.setattr(recorder, "read_background_jobs", original)
    await manager.recover()
    await submit(manager)
    await manager.wait()
    await manager.close()


@pytest.mark.asyncio
async def test_terminal_commit_settles_during_close_without_losing_success(
    recorder: SessionRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = BackgroundAgentManager(recorder)
    entered, closing_started = asyncio.Event(), asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    original = recorder.record_background_job_terminal

    def blocked(*args: Any, **kwargs: Any) -> None:
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(timeout=10)
        original(*args, **kwargs)

    monkeypatch.setattr(recorder, "record_background_job_terminal", blocked)
    await submit(manager)
    await entered.wait()

    async def close() -> None:
        closing_started.set()
        await manager.close()

    closing = asyncio.create_task(close())
    await closing_started.wait()
    assert not closing.done()
    assert manager.active_count == 1
    release.set()
    await closing
    assert manager.snapshot()[0]["status"] == "completed"
    assert manager.pending_count == 1


@pytest.mark.asyncio
async def test_idempotent_insertion_is_atomic_across_recorder_threads(recorder: SessionRecorder) -> None:
    manager = BackgroundAgentManager(recorder)
    await submit(manager)
    await manager.wait()
    completion = manager.snapshot()[0]
    message = Message(role="user", content=[TextBlock(text="completion")])
    epoch = recorder.journal.epoch_id
    peer = SessionRecorder(recorder.journal, recover=False)
    results = await asyncio.gather(
        *[
            asyncio.to_thread(writer.record_background_completion, completion, message, epoch_id=epoch)
            for writer in [recorder, peer] * 4
        ]
    )
    assert results.count(True) == 1
    assert len(completion_events(recorder)) == 1
    await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_via", ["deliver", "close"])
async def test_terminal_failure_retained_report_can_be_flushed_after_storage_recovers(
    recorder: SessionRecorder, monkeypatch: pytest.MonkeyPatch, retry_via: str
) -> None:
    manager = BackgroundAgentManager(recorder)
    original = recorder.record_background_job_terminal
    calls = 0

    def fail_once(*args: Any, **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise SessionJournalError("transient terminal failure")
        original(*args, **kwargs)

    monkeypatch.setattr(recorder, "record_background_job_terminal", fail_once)
    await submit(manager)
    with pytest.raises(SessionJournalError, match="transient"):
        await manager.wait()
    if retry_via == "close":
        await manager.close()
        assert manager.pending_count == 1
    else:
        assert await manager.deliver(Agent(recorder)) == 1
    assert recorder.read_background_jobs(epoch_id=recorder.journal.epoch_id)[0]["result"] == "complete report"
    await manager.close()
