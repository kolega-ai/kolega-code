from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any
from unittest.mock import create_autospec

import pytest

from kolega_code.cli.session_event_store import FileArtifactStore, FileSessionEventStore
from kolega_code.cli.session_journal import SessionJournal, SessionJournalError, SessionRecorder
from kolega_code.cli.session_store import SessionStore
from kolega_code.events import AgentConnectionManager, AgentEvent, AgentEventEmitter
from kolega_code.llm.models import Message
from kolega_code.session.background import BackgroundAgentManager
from kolega_code.session.recording import RecordingConnectionManager, RetentionPolicy
from tests.session.test_background import Agent, completion_events, submit

pytestmark = pytest.mark.asyncio
DELIVERY = "background_job_delivered"


class _Delivery:
    def __init__(self, directory: Path, *, inline_chars: int = 16_000) -> None:
        self.journal = SessionJournal("delivery", directory)
        self.journal.start_epoch("test")
        self.recorder = SessionRecorder(self.journal, recover=False)
        self.live: Any = create_autospec(AgentConnectionManager, instance=True)
        self.store = FileSessionEventStore(self.journal, append_timeout=None)
        self.recording = RecordingConnectionManager(
            self.live,
            self.store,
            session_id="delivery",
            artifact_store=FileArtifactStore(self.journal),
            policy=RetentionPolicy(inline_payload_chars=inline_chars),
        )
        self.agent = Agent(self.recorder)
        self.agent.emitter = AgentEventEmitter(  # type: ignore[attr-defined]
            self.recording, "workspace", "thread", "root", lambda: {"agent_id": "stale-worker"}
        )
        self.notifications: list[dict[str, Any]] = []
        self.manager = BackgroundAgentManager(self.recorder, on_delivered=self.notifications.append)

    async def events(self) -> list[AgentEvent]:
        return [event for event in await self.store.read("delivery") if event.event_type == DELIVERY]


async def test_worker_completion_is_not_delivery_and_later_turn_owns_the_insertion(tmp_path: Path) -> None:
    delivery = _Delivery(tmp_path)
    recorder, manager = delivery.recorder, delivery.manager
    recorder.start_turn(Message(role="user", content="launch"))
    launch_turn = recorder.current_turn_id
    receipt = await submit(manager, metadata={"parent_tool_call_id": "dispatch", "turn_id": launch_turn})
    await manager.wait()
    recorder.finish_turn("completed")
    assert await delivery.events() == []
    assert not delivery.agent.history and not delivery.notifications
    recorder.start_turn(Message(role="user", content="later"))
    later_turn = recorder.current_turn_id
    assert later_turn != launch_turn
    assert sorted(await asyncio.gather(manager.deliver(delivery.agent), manager.deliver(delivery.agent))) == [0, 1]
    events = await delivery.events()
    assert len(events) == 1
    event = events[0]
    assert event.sub_agent_info is None
    assert (event.sender, event.session_id, event.workspace_id, event.thread_id) == (
        "root",
        "delivery",
        "workspace",
        "thread",
    )
    assert event.content == {
        "job_id": "job",
        "completion_id": receipt["completion_id"],
        "agent_type": "investigation",
        "status": "completed",
        "task": "research",
        "text": "complete report",
        "disposition": "context",
        "epoch_id": delivery.journal.epoch_id,
        "metadata": {"parent_tool_call_id": "dispatch", "turn_id": launch_turn},
    }
    semantic = completion_events(recorder)
    assert len(semantic) == len(delivery.agent.history) == len(delivery.notifications) == 1
    assert semantic[0].turn_id == later_turn
    assert event.seq is not None and semantic[0].seq < event.seq
    assert delivery.notifications[0]["completion_id"] == receipt["completion_id"]
    recorder.finish_turn("completed")
    await manager.close()


@pytest.mark.parametrize("failure", ["persistence", "live_append", "ambiguous_commit"])
async def test_failed_semantic_delivery_has_no_presentation_until_successful_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    delivery = _Delivery(tmp_path)
    manager, agent, recorder = delivery.manager, delivery.agent, delivery.recorder
    await submit(manager)
    await manager.wait()
    commit = recorder.record_background_completion

    def failed(*args: Any, **kwargs: Any) -> bool:
        if failure == "ambiguous_commit":
            commit(*args, **kwargs)
        raise SessionJournalError("commit unavailable")

    if failure == "live_append":
        agent.fail_append = True
    else:
        monkeypatch.setattr(recorder, "record_background_completion", failed)
    with pytest.raises(RuntimeError, match="live history|commit unavailable"):
        await manager.deliver(agent)
    assert await delivery.events() == []
    assert manager.pending_count == 1 and not agent.history and not delivery.notifications
    assert not manager.snapshot()[0]["delivered"]
    monkeypatch.setattr(recorder, "record_background_completion", commit)
    agent.fail_append = False
    assert await manager.deliver(agent) == 1
    assert await manager.deliver(agent) == 0
    assert len(await delivery.events()) == len(completion_events(recorder)) == len(agent.history) == 1
    await manager.close()


@pytest.mark.parametrize("failure", ["recording", "fanout", "cancelled_fanout", "callback"])
async def test_presentation_failure_never_retries_semantic_delivery_or_steals_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    delivery = _Delivery(tmp_path)
    manager = delivery.manager
    await submit(manager)
    await manager.wait()
    delivery.store._append_timeout = 0

    async def fail(*args: Any, **kwargs: Any) -> None:
        if failure == "cancelled_fanout":
            raise asyncio.CancelledError
        raise OSError("presentation unavailable")

    if failure == "recording":
        monkeypatch.setattr(delivery.store, "append", fail)
    elif failure == "callback":

        def failed_callback(data: dict[str, Any]) -> None:
            delivery.notifications.append(data)
            raise RuntimeError("frontend unavailable")

        manager.set_callbacks(on_delivered=failed_callback)
    else:
        delivery.live.broadcast_event.side_effect = fail
    assert await manager.deliver(delivery.agent) == 1
    assert await manager.deliver(delivery.agent) == 0
    assert len(completion_events(delivery.recorder)) == len(delivery.agent.history) == len(delivery.notifications) == 1
    assert manager.pending_count == 0 and manager.snapshot()[0]["delivered"]
    assert len(await delivery.events()) == (0 if failure == "recording" else 1)
    assert delivery.live.broadcast_event.await_count == 1
    await manager.close()


@pytest.mark.parametrize("blocked_at", ["commit", "presentation"])
async def test_repeated_cancellation_drains_delivery_before_releasing_its_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, blocked_at: str
) -> None:
    delivery = _Delivery(tmp_path)
    manager = delivery.manager
    delivery.store._append_timeout = 0
    await submit(manager)
    await manager.wait()
    entered, attempted = asyncio.Event(), asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    commit = delivery.recorder.record_background_completion
    append = delivery.store._append_sync

    def blocked(*args: Any, **kwargs: Any) -> Any:
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(timeout=30), "test did not release delivery"
        return (commit if blocked_at == "commit" else append)(*args, **kwargs)

    monkeypatch.setattr(
        delivery.recorder if blocked_at == "commit" else delivery.store,
        "record_background_completion" if blocked_at == "commit" else "_append_sync",
        blocked,
    )
    operation = asyncio.create_task(manager.deliver(delivery.agent))
    await asyncio.wait_for(entered.wait(), 10)
    operation.cancel()
    operation.cancel()

    async def retry() -> int:
        attempted.set()
        return await manager.deliver(delivery.agent)

    second = asyncio.create_task(retry())
    try:
        await attempted.wait()
        assert not operation.done() and not second.done()
        assert not delivery.notifications
        if blocked_at == "presentation":
            assert manager.snapshot()[0]["delivered"] and len(delivery.agent.history) == 1
        else:
            assert not manager.snapshot()[0]["delivered"] and not delivery.agent.history
    finally:
        release.set()
        await asyncio.gather(operation, second, return_exceptions=True)
    assert operation.cancelled() and second.result() == 0
    assert len(await delivery.events()) == len(completion_events(delivery.recorder)) == len(delivery.notifications) == 1
    await manager.close()


@pytest.mark.parametrize("archive", [False, True])
async def test_retirement_after_close_records_disposition_without_reenabling_callbacks(
    tmp_path: Path, archive: bool
) -> None:
    delivery = _Delivery(tmp_path)
    await submit(delivery.manager)
    await delivery.manager.wait()
    await delivery.manager.close()
    assert await delivery.manager.deliver(delivery.agent, archive=archive) == 1
    assert await delivery.manager.deliver(delivery.agent, archive=archive) == 0
    events = await delivery.events()
    assert len(events) == 1 and events[0].content["disposition"] == ("archived" if archive else "context")
    assert delivery.notifications == []
    assert len(delivery.agent.history) == (0 if archive else 1)
    semantic = completion_events(delivery.recorder)
    assert len(semantic) == 1 and semantic[0].payload["archived"] is archive
    raw, _ = SessionStore(tmp_path / "state")._replay(delivery.journal.read_events(), delivery.journal)
    assert len(raw) == (0 if archive else 1)


async def test_recovery_delivers_pending_reports_but_does_not_invent_past_delivery_events(tmp_path: Path) -> None:
    delivery = _Delivery(tmp_path)
    await submit(delivery.manager, "legacy")
    await delivery.manager.wait()
    # A historical delivery has semantic evidence, but no presentation event.
    legacy = Agent(delivery.recorder)
    assert await delivery.manager.deliver(legacy) == 1
    await submit(delivery.manager, "pending")
    await delivery.manager.wait()
    await delivery.manager.close()
    assert await delivery.events() == []
    reopened = SessionRecorder(SessionJournal("delivery", tmp_path), recover=False)
    delivery.store = FileSessionEventStore(reopened.journal, append_timeout=None)
    delivery.recording.store = delivery.store
    delivery.recording.artifact_store = FileArtifactStore(reopened.journal)
    recovered = BackgroundAgentManager(reopened, on_delivered=delivery.notifications.append)
    delivery.agent.session_recorder = reopened
    await recovered.recover()
    assert recovered.pending_count == 1
    assert await recovered.deliver(delivery.agent) == 1
    events = await delivery.events()
    assert len(events) == 1 and events[0].content["job_id"] == "pending"
    assert events[0].content["metadata"] == {}
    assert len(completion_events(reopened)) == 2
    await recovered.close()
    resumed = BackgroundAgentManager(reopened)
    assert await resumed.deliver(delivery.agent) == 0
    assert len(await delivery.events()) == 1
    await resumed.close()


async def test_report_artifact_survives_recording_and_presentation_does_not_change_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kolega_code.cli.atif_export import export_atif_to_text
    from kolega_code.cli.session_usage import derive_session_usage
    from kolega_code.session.projection import replay

    delivery = _Delivery(tmp_path, inline_chars=160)
    report = "complete report 測試 " * 100 + "REPORT END"

    async def child() -> str:
        return report

    await submit(delivery.manager, run=child)
    await delivery.manager.wait()
    before: dict[str, Any] = {}
    append = delivery.store.append

    async def observe(event: AgentEvent) -> int:
        events = delivery.journal.read_events()
        assert not any(row.event_type.startswith("ui.") for row in events)
        before["history"], _ = SessionStore(tmp_path / "state")._replay(events, delivery.journal)
        before["usage"] = derive_session_usage(events)
        before["atif"] = export_atif_to_text(delivery.journal, kolega_version="test", secret_values=(), state_dirs=())
        assert len(delivery.agent.history) == 1
        return await append(event)

    monkeypatch.setattr(delivery.store, "append", observe)
    await delivery.manager.deliver(delivery.agent)
    assert delivery.live.broadcast_event.call_args.args[0].content["text"] == report
    recorded = await delivery.events()
    assert len(recorded) == 1
    assert recorded[0].content["text"] != report
    assert len(recorded[0].artifacts) == 1
    ref = recorded[0].artifacts[0]
    assert ref.purpose == "tool_result"
    assert ref.chars == len(report)
    assert delivery.journal.read_artifact(ref.model_dump(exclude_none=True)).decode("utf-8") == report
    state = replay(recorded).to_dict()
    assert len(state["conversation"]) == 1
    assert state["conversation"][0]["kind"] == "background_delivery"
    events = delivery.journal.read_events()
    after_history, _ = SessionStore(tmp_path / "state")._replay(events, delivery.journal)
    assert after_history == before["history"]
    assert len(after_history) == 1
    assert report in after_history[0]["content"][0]["text"]
    assert derive_session_usage(events) == before["usage"]
    assert (
        export_atif_to_text(delivery.journal, kolega_version="test", secret_values=(), state_dirs=()) == before["atif"]
    )
    await delivery.manager.close()
