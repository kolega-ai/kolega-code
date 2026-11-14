from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from textual.pilot import Pilot
from textual.worker import Worker, WorkerCancelled

from kolega_code.agent.baseagent import QueuedUserInput
from kolega_code.cli import app as app_module
from kolega_code.cli.app import KolegaCodeApp
from kolega_code.cli.tui.state import TurnState
from kolega_code.cli.tui.widgets import ChatComposer
from kolega_code.session.background import BackgroundAgentManager
from kolega_code.session.runtime import SessionRuntime
from kolega_code.session.turns import StreamFactory, TurnController, TurnOutcome, TurnResult

from ._app_test_utils import FakeCoderAgent, _build_sub_agent_test_app, install_fake_agents
from .test_app_plan_approval_channel import _task_factory
from .test_app_turn_save_timeout import _PlanAgent

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("isolated_cli_env")]


class _GatedAgent(FakeCoderAgent):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.started = asyncio.Event()
        self.pull = asyncio.Event()
        self.pulled = asyncio.Event()
        self.finish = asyncio.Event()
        self.received: list[QueuedUserInput] = []
        self.background_needs_turn = False
        self.continuations = 0

    async def process_message_stream(
        self, message: str, attachments: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        self.started.set()
        await self.pull.wait()
        assert self.queued_input_provider is not None
        self.received = await self.queued_input_provider()
        self.pulled.set()
        await self.finish.wait()
        async for chunk in super().process_message_stream(message, attachments):
            yield chunk

    async def continue_from_history_stream(self) -> AsyncIterator[dict[str, Any]]:
        self.background_needs_turn = False
        self.continuations += 1
        self.started.set()
        await self.finish.wait()
        yield {"type": "response", "content": "background answer", "complete": True, "uuid": "background-answer"}


async def _submit(app: KolegaCodeApp, text: str) -> Worker:
    composer = app.query_one("#composer", ChatComposer)
    composer.load_text(text)
    await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, text))
    worker = app.agent_worker
    assert worker is not None
    return worker


async def test_runtime_receives_controller_at_construction_and_compatibility_fields_are_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    injected: list[TurnController[Any]] = []

    def runtime_factory(**kwargs: Any) -> SessionRuntime:
        injected.append(kwargs["turns"])
        return SessionRuntime(**kwargs)

    monkeypatch.setattr(app_module, "SessionRuntime", runtime_factory)
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    turns = app._turn_controller
    assert injected == [turns]
    assert app.session_runtime.turns is turns
    assert not {
        "agent_worker",
        "_agent_execution_task",
        "_draining_agent_tasks",
        "_queued_messages",
        "_stop_pending",
        "_stop_watchdog",
    }.intersection(vars(app))
    owner = cast(Worker, object())
    app.agent_worker = owner
    assert turns.owner is owner
    turns.owner = None
    assert app.agent_worker is None
    task = asyncio.current_task()
    assert task is not None
    app._agent_execution_task = task
    assert turns.execution_task is task
    turns.execution_task = None
    assert app._agent_execution_task is None
    assert app._queued_messages is turns.inputs.items
    assert app._draining_agent_tasks is turns.draining_tasks
    turns.draining_tasks.add(task)
    assert task in app._draining_agent_tasks
    turns.draining_tasks.remove(task)
    app._stop_pending = True
    assert turns.stop_pending
    turns.stop_pending = False
    assert not app._stop_pending
    turns.arm_watchdog(60, lambda: None)
    try:
        assert app._stop_watchdog is turns.watchdog
        assert app._stop_watchdog is not None
    finally:
        turns.disarm_watchdog()
    assert app._stop_watchdog is None


@pytest.mark.parametrize("has_worker", [False, True], ids=["direct-caller", "worker-already-cancelled"])
async def test_shutdown_timeout_retains_execution_without_another_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, has_worker: bool
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    app._stop_force_after_seconds = 0
    entered, release = asyncio.Event(), asyncio.Event()
    worker = Mock(spec=Worker) if has_worker else None
    if worker is not None:
        worker.wait = AsyncMock(side_effect=release.wait)
    app.agent_worker = worker

    async def execute() -> None:
        app._track_agent_execution()
        entered.set()
        await release.wait()

    task = asyncio.create_task(execute())
    try:
        await asyncio.wait_for(entered.wait(), 10)
        assert not await app._wait_for_agent_shutdown()
        assert app.agent_worker is None
        assert task in app._draining_agent_tasks
        assert app._turn_controller.busy
        assert task.cancelling() == 0
        if worker is not None:
            worker.cancel.assert_called_once_with()
    finally:
        release.set()
        await asyncio.gather(task, *app._shutdown_worker_waiters, return_exceptions=True)
    assert not app._turn_controller.busy
    assert not app._draining_agent_tasks


@pytest.mark.parametrize("eager", [False, True], ids=["lazy", "eager"])
@pytest.mark.parametrize("background", [False, True], ids=["user", "background-continuation"])
async def test_pre_stream_stop_cannot_escalate_the_successors_first_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, eager: bool, background: bool
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    install_fake_agents(monkeypatch, coder_cls=_GatedAgent)
    async with app.run_test():
        async with _task_factory(eager):
            agent = app.agent
            assert isinstance(agent, _GatedAgent)
            entered, release = asyncio.Event(), asyncio.Event()
            jobs = BackgroundAgentManager(app._session_recorder)
            app._background_jobs = jobs

            async def deliver(_agent: Any) -> None:
                entered.set()
                await release.wait()

            monkeypatch.setattr(jobs, "deliver", deliver)
            app._stop_force_after_seconds = 3600
            workers: list[Worker] = []
            try:
                if background:
                    agent.background_needs_turn = True
                    assert app._maybe_start_queued_message()
                    worker = app.agent_worker
                    assert worker is not None
                else:
                    worker = await _submit(app, "cancel before streaming")
                workers.append(worker)
                await asyncio.wait_for(entered.wait(), 10)
                assert app._turn_started_at is None
                app.action_cancel_generation()
                watchdog = app._stop_watchdog
                assert watchdog is not None
                obsolete_callback = watchdog._callback
                assert obsolete_callback is not None
                with pytest.raises(WorkerCancelled):
                    await asyncio.wait_for(worker.wait(), 10)
                obsolete_callback()
                assert app.agent_worker is None
                assert not app._stop_pending
                assert app._stop_watchdog is None
                assert app._status_state.turn_state is TurnState.STOPPED
                assert not app._turn_active
                assert not app._turn_controller.busy
                assert not agent.started.is_set()
                assert agent.messages == []
                assert agent.continuations == 0

                entered.clear()
                successor = await _submit(app, "successor")
                workers.append(successor)
                await asyncio.wait_for(entered.wait(), 10)
                force = Mock(wraps=app._force_settle_stop)
                monkeypatch.setattr(app, "_force_settle_stop", force)
                app.action_cancel_generation()
                force.assert_not_called()
                assert app.agent_worker is successor
                assert app._stop_pending
                assert not app._draining_agent_tasks
                with pytest.raises(WorkerCancelled):
                    await asyncio.wait_for(successor.wait(), 10)
                assert app.agent_worker is None
                assert not app._stop_pending
                assert not app._turn_controller.busy
            finally:
                release.set()
                agent.finish.set()
                agent.pull.set()
                await asyncio.wait_for(
                    asyncio.gather(*(worker.wait() for worker in workers), return_exceptions=True), 10
                )
                app._background_jobs = None
                await jobs.close()


async def _exercise_entry(app: KolegaCodeApp, monkeypatch: pytest.MonkeyPatch, *, background: bool) -> None:
    agent = app.agent
    assert isinstance(agent, _GatedAgent)
    runtime = app.session_runtime
    turns = runtime.turns
    assert turns is app._turn_controller
    calls: list[tuple[object, asyncio.Task[Any] | None]] = []
    results: list[TurnResult] = []
    run = turns.run

    async def observe_run(stream_factory: StreamFactory, **kwargs: Any) -> TurnResult:
        calls.append((turns.owner, asyncio.current_task()))
        result = await run(stream_factory, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(turns, "run", observe_run)
    send = Mock(wraps=runtime.send_message)
    monkeypatch.setattr(runtime, "send_message", send)
    jobs = None
    if background:
        jobs = BackgroundAgentManager(app._session_recorder)
        app._background_jobs = jobs
        agent.background_needs_turn = True
        assert app._maybe_start_queued_message()
        worker = app.agent_worker
        assert worker is not None
    else:
        worker = await _submit(app, "user turn")
    try:
        await asyncio.wait_for(agent.started.wait(), 10)
        assert app.agent_worker is turns.owner is worker
        assert calls == [(worker, turns.execution_task)]
        assert turns.execution_task is not None
        assert not turns.execution_task.done()
        assert turns._active_run is not None
        assert turns.busy
        assert app._turn_active
        assert app._agent_execution_task is turns.execution_task
        if background:
            send.assert_not_called()
            assert agent.continuations == 1
            assert agent.messages == []
        else:
            send.assert_called_once_with("user turn", None)
        agent.pull.set()
        agent.finish.set()
        await asyncio.wait_for(worker.wait(), 10)
        assert [result.outcome for result in results] == [TurnOutcome.COMPLETED]
        assert app.agent_worker is turns.owner is None
        assert turns._active_run is None
        assert turns.execution_task is None
        assert not turns.busy
        assert not app._turn_active
        assert app._turn_final_state is TurnState.IDLE
        expected = "background answer" if background else "done"
        assert [(entry.kind, entry.content) for entry in app.conversation_entries if entry.kind == "assistant"] == [
            ("assistant", expected)
        ]
        assert agent.messages == ([] if background else ["user turn"])
    finally:
        agent.pull.set()
        agent.finish.set()
        await asyncio.wait_for(worker.wait(), 10)
        if jobs is not None:
            app._background_jobs = None
            await jobs.close()


@pytest.mark.parametrize("eager", [False, True], ids=["lazy", "eager"])
@pytest.mark.parametrize("background", [False, True], ids=["user", "background-continuation"])
async def test_tui_entry_runs_shared_controller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, eager: bool, background: bool
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    install_fake_agents(monkeypatch, coder_cls=_GatedAgent)
    async with app.run_test():
        async with _task_factory(eager):
            await _exercise_entry(app, monkeypatch, background=background)


async def test_real_launcher_background_stream_has_shared_controller_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    install_fake_agents(monkeypatch, coder_cls=_GatedAgent)
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()

    async def pilot_turn(_pilot: Pilot) -> None:
        await app._startup_complete.wait()
        factory = getattr(asyncio, "eager_task_factory", None)
        if factory is not None:
            assert loop.get_task_factory() is factory
        try:
            await _exercise_entry(app, monkeypatch, background=True)
        finally:
            app.exit()

    try:
        await asyncio.wait_for(app.run_async(headless=True, auto_pilot=pilot_turn), 20)
    finally:
        loop.set_task_factory(previous)


@pytest.mark.parametrize("eager", [False, True], ids=["lazy", "eager"])
@pytest.mark.parametrize("replace_owner", [False, True], ids=["fifo-owner-only", "post-await-owner-fence"])
async def test_mid_turn_queue_is_owned_and_fenced_after_event_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, eager: bool, replace_owner: bool
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    install_fake_agents(monkeypatch, coder_cls=_GatedAgent)
    async with app.run_test():
        async with _task_factory(eager):
            worker = await _submit(app, "first turn")
            agent = app.agent
            assert isinstance(agent, _GatedAgent)
            replacement = None
            replacement_release = asyncio.Event()
            drain_entered = asyncio.Event()
            drain_release = asyncio.Event()
            try:
                await asyncio.wait_for(agent.started.wait(), 10)
                turns = app.session_runtime.turns
                task = turns.execution_task
                assert task is not None
                attachments = [{"type": "image", "path": "/fake/first.png", "mime_type": "image/png"}]
                origin = {"kind": "peer", "session_id": "peer-session", "title": "Peer"}
                app._queue_user_message("first followup", attachments)
                app._queue_user_message("peer display", model_text="peer model text", origin=origin)
                queued = list(turns.inputs.items)
                assert app._queued_messages is turns.inputs.items
                assert len(queued) == 2
                assert queued[0].attachments == attachments
                assert queued[0].attachments is not attachments
                assert queued[1].origin is origin
                assert await app._provide_queued_user_inputs() == []
                assert turns.inputs.items == queued
                assert not app._maybe_start_queued_message()
                drain = app._drain_pending_events

                async def held_drain() -> None:
                    if asyncio.current_task() is task:
                        drain_entered.set()
                        await drain_release.wait()
                    await drain()

                monkeypatch.setattr(app, "_drain_pending_events", held_drain)
                agent.pull.set()
                await asyncio.wait_for(drain_entered.wait(), 10)
                assert turns.inputs.items == queued
                if replace_owner:
                    replacement = app.run_worker(
                        replacement_release.wait(), name="replacement", group="replacement", exit_on_error=False
                    )
                    turns.owner = replacement
                    app._begin_turn_progress("replacement phase")
                    turns.stop_pending = True
                    turns.arm_watchdog(60, lambda: None)
                    timer = app._turn_timer
                    watchdog = turns.watchdog
                drain_release.set()
                await asyncio.wait_for(agent.pulled.wait(), 10)
                if replace_owner:
                    assert agent.received == []
                    assert turns.inputs.items == queued
                    assert all(item.entry not in app.conversation_entries for item in queued)
                    save = AsyncMock(wraps=app._save_session_history_async)
                    monkeypatch.setattr(app, "_save_session_history_async", save)
                    agent.finish.set()
                    await asyncio.wait_for(worker.wait(), 10)
                    save.assert_not_awaited()
                    assert turns.owner is app.agent_worker is replacement
                    assert turns.stop_pending
                    assert turns.watchdog is watchdog
                    assert app._turn_status_text == "replacement phase"
                    assert app._turn_timer is timer
                    assert not any(entry.kind == "assistant" for entry in app.conversation_entries)
                else:
                    assert agent.received == [
                        QueuedUserInput(text="first followup", attachments=attachments, origin=None),
                        QueuedUserInput(text="peer model text", attachments=None, origin=origin),
                    ]
                    assert turns.inputs.empty()
                    delivered = [
                        entry for entry in app.conversation_entries if entry in [item.entry for item in queued]
                    ]
                    assert delivered == [item.entry for item in queued]
                    assert [(entry.kind, entry.content, entry.origin) for entry in delivered] == [
                        ("user", "first followup", None),
                        ("peer_message", "peer display", origin),
                    ]
            finally:
                drain_release.set()
                agent.pull.set()
                agent.finish.set()
                await asyncio.wait_for(worker.wait(), 10)
                app._clear_queued_messages()
                if replacement is not None:
                    replacement_release.set()
                    await asyncio.wait_for(replacement.wait(), 10)
                    app.agent_worker = None
                    app._turn_controller.stop_pending = False
                    app._turn_controller.disarm_watchdog()
                    app._finish_turn_progress("replacement finished", TurnState.IDLE)


@pytest.mark.parametrize("eager", [False, True], ids=["lazy", "eager"])
async def test_plan_implementation_and_handoff_reuse_controller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, eager: bool
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    install_fake_agents(monkeypatch, planning_cls=_PlanAgent)
    async with app.run_test():
        async with _task_factory(eager):
            turns = app.session_runtime.turns
            runtime = app.session_runtime
            await app.action_toggle_interaction_mode()
            worker = await _submit(app, "plan this")
            await asyncio.wait_for(worker.wait(), 10)
            assert app._plan_decision_active
            planner = app.agent
            await app._implement_pending_plan()
            worker = app.agent_worker
            assert worker is not None
            await asyncio.wait_for(worker.wait(), 10)
            assert app.interaction_mode == "build"
            assert app.agent is not planner
            assert app.session_runtime is runtime
            assert runtime.agent is app.agent
            assert runtime.turns is app._turn_controller is turns
            builder = cast(FakeCoderAgent, app.agent)
            assert len(builder.messages) == 1
            assert "# Plan\n\n- close the Mistral gaps" in builder.messages[0]
            old_session = app.session
            await app._handoff_session("# Handoff\n\nContinue the accepted plan.")
            rebound = app.session_runtime
            assert rebound is not runtime
            assert app.session.parent_session_id == old_session.session_id
            assert rebound.session_id == app.session.session_id
            assert rebound.turns is app._turn_controller is turns
            assert rebound.agent is app.agent
            assert app.agent is not builder
            send = Mock(wraps=rebound.send_message)
            monkeypatch.setattr(rebound, "send_message", send)
            worker = await _submit(app, "continue after handoff")
            await asyncio.wait_for(worker.wait(), 10)
            send.assert_called_once_with("continue after handoff", None)
            assert cast(FakeCoderAgent, app.agent).messages == ["continue after handoff"]
            assert turns.owner is None
            assert turns._active_run is None
