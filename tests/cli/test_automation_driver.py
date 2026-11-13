from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from textual.pilot import Pilot

from kolega_code.agent.goal import GoalVerdict
from kolega_code.cli import main as main_module
from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.tui import state as tui_state
from kolega_code.cli.tui.plan_approval import PlanApprovalOwner
from kolega_code.llm.exceptions import LLMError
from tests.cli._app_test_utils import extension_by_name
from tests.cli.test_app_goal_command import (
    GoalFakeAgent,
    _build_goal_test_app,
    _submit as submit_goal,
    _wait_goal_terminal,
    _wait_turn_idle,
)
from tests.cli.test_app_loop_command import (
    START,
    LoopFakeAgent,
    _build_loop_test_app,
    _ready,
    _submit as submit_loop,
    _wait_for,
    _wait_loop_idle,
    clock as clock,
)
from tests.cli.test_ask_goal import GoalAskFakeAgent, _reset_fake, _setup_project
from tests.cli.test_ask_loop import LoopAskFakeAgent, ask_env as ask_env
from tests.cli.test_app_plan_approval_channel import _task_factory

pytestmark = pytest.mark.usefixtures("isolated_cli_env")


@pytest_asyncio.fixture(params=[False, True], ids=["lazy", "eager"])
async def scheduling(request: pytest.FixtureRequest) -> Any:
    async with _task_factory(request.param):
        yield


@pytest.mark.parametrize("max_turns", [1, 2])
def test_headless_never_sends_unverified_work_after_the_goal_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, max_turns: int
) -> None:
    _reset_fake(monkeypatch)
    GoalAskFakeAgent.default_met = False
    project = _setup_project(tmp_path, monkeypatch)
    code = main_module.main(
        ["ask", "--goal", "verified", "--goal-max-turns", str(max_turns), "--project", str(project)]
    )
    agent = GoalAskFakeAgent.instances[-1]
    assert code == 1
    assert agent.stream_calls == agent.evaluate_calls == max_turns


def test_headless_first_cron_fire_past_expiry_does_no_work(ask_env: Path, capsys: Any) -> None:
    code = main_module.main(
        [
            "ask",
            "poll",
            "--loop-cron",
            "0 9 * * *",
            "--loop-expires",
            "15s",
            "--project",
            str(ask_env),
            "--json",
        ]
    )
    assert code == 0
    assert LoopAskFakeAgent.instances[-1].messages == []
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert not any(record["type"] == "loop.iteration_started" for record in records)
    completed = [record["payload"] for record in records if record["type"] == "loop.completed"]
    assert completed == [{"iterations": 0, "reason": "expired"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["initial", "verification", "continuation"])
@pytest.mark.parametrize("json_mode", [False, True])
async def test_headless_cancellation_pauses_and_persists_goal_after_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any, phase: str, json_mode: bool, scheduling: None
) -> None:
    started = asyncio.Event()
    attempts: list[str] = []

    class CancelAgent(GoalAskFakeAgent):
        async def process_message_stream(self, message: str, attachments: Any = None) -> Any:
            attempt = "initial" if not attempts else "continuation"
            attempts.append(attempt)
            if phase == attempt:
                started.set()
                await asyncio.Event().wait()
            async for chunk in super().process_message_stream(message, attachments):
                yield chunk

        async def evaluate_goal_condition(self, condition: str) -> GoalVerdict:
            if phase == "verification":
                started.set()
                await asyncio.Event().wait()
            return await super().evaluate_goal_condition(condition)

    _reset_fake(monkeypatch)
    monkeypatch.setattr(main_module, "CoderAgent", CancelAgent)
    GoalAskFakeAgent.default_met = False
    project = _setup_project(tmp_path, monkeypatch)
    args = ["ask", "--goal", "verified", "--save", "--project", str(project)]
    if json_mode:
        args.append("--json")
    invocation = asyncio.create_task(main_module._run_ask(main_module.parse_args(args)))
    try:
        await asyncio.wait_for(started.wait(), 10)
        invocation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(invocation, 10)
        agent = GoalAskFakeAgent.instances[-1]
        assert agent.cleanup_calls == 1
        assert attempts == (["initial", "continuation"] if phase == "continuation" else ["initial"])
        store = SessionStore()
        saved = store.list(recover=False)
        assert len(saved) == 1
        goal = store.load(saved[0].session_id).goal
        assert goal["paused"] is True
        assert goal["met"] is False
        assert goal["status_note"] == "Stopped by user."
        records = store.journal(saved[0].session_id).read_events(repair_tail=False)
        terminal = [record for record in records if record.event_type == "goal.completed"]
        assert len(terminal) == 1 and terminal[0].payload["paused"] is True
        assert records[-1].event_type == "run.cancelled"
        if json_mode:
            output = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
            assert [record["seq"] for record in output] == list(range(1, len(output) + 1))
    finally:
        if not invocation.done():
            invocation.cancel()
            await asyncio.gather(invocation, return_exceptions=True)


@pytest.mark.asyncio
async def test_tui_failed_scheduled_turn_stops_instead_of_rearming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: Any, scheduling: None
) -> None:
    class FailedLoopAgent(LoopFakeAgent):
        async def process_message_stream(self, message: str, attachments: Any = None) -> Any:
            raise LLMError("scheduled request failed", provider="anthropic")
            yield

    app = _build_loop_test_app(tmp_path, monkeypatch, agent_cls=FailedLoopAgent)
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        await submit_loop(app, "/loop 5m poll")
        await _wait_loop_idle(app, pilot)
        state = app._scheduled_loop
        assert state is not None
        assert state.stopped and state.iterations == 1
        assert state.last_completed_at is None
        assert state.next_fire_at == START.isoformat()
        assert "scheduled request failed" in state.status_note
        assert app.store.load(app.session.session_id).loop == state.to_dict()
        clock.advance(minutes=30)
        await app._loop_tick()
        assert app.agent_worker is None and state.iterations == 1


@pytest.mark.asyncio
async def test_tui_input_queued_during_verifier_precedes_any_goal_nudge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scheduling: None
) -> None:
    app = _build_goal_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        await asyncio.wait_for(app._startup_complete.wait(), 10)
        agent = GoalFakeAgent.instances[-1]
        started, release = asyncio.Event(), asyncio.Event()
        agent._verifier_started = started
        agent._release_verifier = release
        agent._goal_evaluate_results = [GoalVerdict(met=False, reason="gap"), GoalVerdict(met=True, reason="done")]
        await submit_goal(app, pilot, "/goal verified")
        await asyncio.wait_for(started.wait(), 10)
        await submit_goal(app, pilot, "steer first")
        agent._release_verifier = None
        release.set()
        await _wait_goal_terminal(app, pilot)
        await _wait_turn_idle(app, pilot)
        assert len(agent.messages) == 2
        assert agent.messages[1] == "steer first"
        assert agent._evaluate_calls == ["verified", "verified"]


@pytest.mark.asyncio
async def test_tui_input_arriving_during_fresh_reset_survives_in_new_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: Any, scheduling: None
) -> None:
    app = _build_loop_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        agent = await _ready(app, pilot)
        await submit_loop(app, "/loop --fresh 5m poll")
        await _wait_loop_idle(app, pilot)
        started, release = asyncio.Event(), asyncio.Event()
        retire = app._retire_background_jobs

        async def retire_background(**kwargs: Any) -> None:
            started.set()
            await release.wait()
            await retire(**kwargs)

        monkeypatch.setattr(app, "_retire_background_jobs", retire_background)
        clock.advance(minutes=5)
        await app._loop_tick()
        await asyncio.wait_for(started.wait(), 10)
        await submit_loop(app, "steer in the fresh context")
        assert [item.text for item in app._queued_messages] == ["steer in the fresh context"]
        release.set()
        await _wait_for(app, pilot, lambda: "steer in the fresh context" in agent.messages, timeout=10)
        await _wait_loop_idle(app, pilot)
        assert app._scheduled_loop is not None and app._scheduled_loop.iterations == 1
        assert agent.messages[-1] == "steer in the fresh context"
        assert not app._queued_messages


@pytest.mark.parametrize("phase", ["initial", "verification", "continuation"])
def test_headless_unexpected_failure_pauses_goal_and_preserves_failure_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any, phase: str
) -> None:
    attempts: list[str] = []

    class FailedAgent(GoalAskFakeAgent):
        async def process_message_stream(self, message: str, attachments: Any = None) -> Any:
            attempt = "initial" if not attempts else "continuation"
            attempts.append(attempt)
            if phase == attempt:
                raise ValueError("automation failed")
            async for chunk in super().process_message_stream(message, attachments):
                yield chunk

        async def evaluate_goal_condition(self, condition: str) -> GoalVerdict:
            if phase == "verification":
                raise ValueError("automation failed")
            return await super().evaluate_goal_condition(condition)

    _reset_fake(monkeypatch)
    monkeypatch.setattr(main_module, "CoderAgent", FailedAgent)
    GoalAskFakeAgent.default_met = False
    project = _setup_project(tmp_path, monkeypatch)
    code = main_module.main(["ask", "--goal", "verified", "--save", "--json", "--project", str(project)])
    assert code == 2
    assert GoalAskFakeAgent.instances[-1].cleanup_calls == 1
    assert attempts == (["initial", "continuation"] if phase == "continuation" else ["initial"])
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    terminal = [record for record in records if record["type"] == "goal.completed"]
    assert len(terminal) == 1 and terminal[0]["payload"]["paused"] is True
    assert terminal[0]["payload"]["error"] == "automation failed"
    assert records[-1]["type"] == "run.failed"
    goal = SessionStore().load(records[0]["session_id"]).goal
    assert goal["paused"] and not goal["met"]
    assert goal["status_note"] == "automation failed"


@pytest.mark.asyncio
async def test_tui_initial_goal_cancellation_pauses_and_persists_without_verifying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scheduling: None
) -> None:
    app = _build_goal_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        await asyncio.wait_for(app._startup_complete.wait(), 10)
        agent = GoalFakeAgent.instances[-1]
        agent._cancel_on_calls = {1}
        await submit_goal(app, pilot, "/goal verified")
        await _wait_goal_terminal(app, pilot)
        assert app._goal is not None and app._goal.paused and not app._goal.met
        assert agent._evaluate_calls == []
        assert agent._stream_call_count == 1
        goal = app.store.load(app.session.session_id).goal
        assert goal["paused"] and goal["turns_evaluated"] == 0
        assert goal["status_note"] == "Stopped by user."


@pytest.mark.asyncio
async def test_tui_scheduled_iteration_holds_ownership_through_accounting_and_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: Any, scheduling: None
) -> None:
    app = _build_loop_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        agent = await _ready(app, pilot)
        entered, release = asyncio.Event(), asyncio.Event()
        persist_loop = app._persist_loop_async

        async def persist() -> None:
            if app._loop_iteration_active:
                entered.set()
                await release.wait()
            await persist_loop()

        monkeypatch.setattr(app, "_persist_loop_async", persist)
        await submit_loop(app, "/loop 5m poll")
        await asyncio.wait_for(entered.wait(), 10)
        owner = app.agent_worker
        assert owner is not None and app._turn_controller.busy
        assert not app._turn_active and app._loop_iteration_active
        await submit_loop(app, "steer after persistence")
        await app._loop_tick()
        assert app.agent_worker is owner
        assert len(agent.messages) == 1
        assert [item.text for item in app._queued_messages] == ["steer after persistence"]
        release.set()
        await _wait_for(app, pilot, lambda: "steer after persistence" in agent.messages, timeout=10)
        await _wait_loop_idle(app, pilot)
        assert len(agent.messages) == 2
        assert app._scheduled_loop is not None and app._scheduled_loop.iterations == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("gate", ["draining", "reserved"])
async def test_tui_scheduler_waits_for_retained_plan_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: Any, scheduling: None, gate: str
) -> None:
    app = _build_loop_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        agent = await _ready(app, pilot)
        await asyncio.wait_for(app._startup_complete.wait(), 10)
        await submit_loop(app, "/loop 5m poll")
        await _wait_loop_idle(app, pilot)
        state = app._scheduled_loop
        assert state is not None and state.iterations == 1
        release = asyncio.Event()

        async def retained_plan() -> None:
            await release.wait()

        task = asyncio.create_task(retained_plan())
        app._plan_approval_tasks.add(task)
        if gate == "draining":
            app._draining_plan_approval_tasks.add(task)
        else:
            assert app.agent is not None
            app._plan_approval_owner = PlanApprovalOwner(
                session_id=app.session.session_id,
                runtime=app.session_runtime,
                agent=app.agent,
                revision=1,
                plan="Retained plan",
                task=task,
                action="implement_plan",
            )
        try:
            clock.advance(minutes=5)
            await app._loop_tick()
            assert app.agent_worker is None
            assert state.iterations == 1 and state.deferred
            assert len(agent.messages) == 1
            assert not app._queued_messages
        finally:
            app._plan_approval_owner = None
            release.set()
            await task
            app._plan_approval_tasks.discard(task)
            app._draining_plan_approval_tasks.discard(task)
        await app._loop_tick()
        await _wait_loop_idle(app, pilot)
        assert state.iterations == 2
        assert len(agent.messages) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("escalation", ["escape", "watchdog"])
async def test_tui_detached_loop_releases_its_activity_reservation_after_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: Any, scheduling: None, escalation: str
) -> None:
    entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class HeldLoopAgent(LoopFakeAgent):
        async def process_message_stream(self, message: str, attachments: Any = None) -> Any:
            entered.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    cancelled.set()
            async for chunk in super().process_message_stream(message, attachments):
                yield chunk

    app = _build_loop_test_app(tmp_path, monkeypatch, agent_cls=HeldLoopAgent)
    app._stop_force_after_seconds = 60
    async with app.run_test() as pilot:
        agent = await _ready(app, pilot)
        await asyncio.wait_for(app._startup_complete.wait(), 10)
        await submit_loop(app, "/loop 5m poll")
        state = app._scheduled_loop
        assert state is not None
        execution = None
        try:
            await asyncio.wait_for(entered.wait(), 10)
            execution = app._agent_execution_task
            assert execution is not None
            app.action_cancel_generation()
            await asyncio.wait_for(cancelled.wait(), 10)
            if escalation == "escape":
                app.action_cancel_generation()
            else:
                app._stop_force_after_seconds = 0
                app._arm_stop_watchdog()
            await _wait_for(app, pilot, lambda: app.agent_worker is None, timeout=10)
            assert app._agent_execution_draining()
            assert state.stopped
            assert not app._loop_ready_to_fire()
            await app._loop_tick()
            assert app.agent_worker is None
            await _wait_for(app, pilot, lambda: app.store.load(app.session.session_id).loop["stopped"], timeout=10)
        finally:
            release.set()
            if execution is not None:
                await asyncio.wait_for(asyncio.gather(execution, return_exceptions=True), 10)
        await _wait_for(app, pilot, lambda: not app._agent_execution_draining(), timeout=10)
        assert not app._turn_controller.busy
        assert not app._loop_iteration_active
        await app._loop_tick()
        await _wait_loop_idle(app, pilot)
        assert len(agent.messages) == state.iterations == 1
        await submit_loop(app, "/loop 5m replacement poll")
        await _wait_loop_idle(app, pilot)
        assert app._scheduled_loop is not None and app._scheduled_loop.iterations == 1
        assert len(agent.messages) == 2
        assert "replacement poll" in agent.messages[-1]


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["verification", "continuation"])
@pytest.mark.parametrize("queued_input", [False, True], ids=["autonomous", "queued"])
async def test_tui_replacement_goal_hands_off_with_queued_input_priority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scheduling: None, phase: str, queued_input: bool
) -> None:
    replaced, release = asyncio.Event(), asyncio.Event()
    replacement_messages: list[list[str]] = []

    class ReplacingGoalAgent(GoalFakeAgent):
        async def replace_goal(self) -> None:
            extension = extension_by_name(self.kwargs["tool_extensions"], "cli-goal-control")
            await extension.tools["set_goal"]("replacement condition")
            replaced.set()
            await release.wait()

        async def process_message_stream(self, message: str, attachments: Any = None) -> Any:
            if phase == "continuation" and self._stream_call_count == 1:
                await self.replace_goal()
            async for chunk in super().process_message_stream(message, attachments):
                yield chunk

        async def evaluate_goal_condition(self, condition: str) -> GoalVerdict:
            if phase == "verification" and condition == "original condition":
                await self.replace_goal()
            if condition == "replacement condition":
                replacement_messages.append(list(self.messages))
            return await super().evaluate_goal_condition(condition)

    app = _build_goal_test_app(tmp_path, monkeypatch, agent_cls=ReplacingGoalAgent)
    async with app.run_test() as pilot:
        await asyncio.wait_for(app._startup_complete.wait(), 10)
        agent = GoalFakeAgent.instances[-1]
        agent._goal_evaluate_results = [
            GoalVerdict(phase == "verification", "original verdict"),
            GoalVerdict(True, "replacement verified"),
        ]
        await submit_goal(app, pilot, "/goal original condition")
        try:
            await asyncio.wait_for(replaced.wait(), 10)
            if queued_input:
                await submit_goal(app, pilot, "steer the replacement first")
                assert [item.text for item in app._queued_messages] == ["steer the replacement first"]
        finally:
            release.set()
        if queued_input:
            await _wait_for(app, pilot, lambda: "steer the replacement first" in agent.messages, timeout=10)
        await _wait_turn_idle(app, pilot, timeout=10)
        assert app._goal is not None and app._goal.condition == "replacement condition"
        assert agent._evaluate_calls == ["original condition", "replacement condition"]
        assert app._goal.met
        assert len(agent.messages) == 1 + (phase == "continuation") + queued_input
        assert replacement_messages == [agent.messages]
        assert not app._queued_messages
        assert app.store.load(app.session.session_id).goal["met"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["failed", "cancelled"])
async def test_tui_replacement_goal_preserves_continuation_failure_or_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scheduling: None, terminal: str
) -> None:
    class ReplacingGoalAgent(GoalFakeAgent):
        async def process_message_stream(self, message: str, attachments: Any = None) -> Any:
            if self._stream_call_count == 1:
                extension = extension_by_name(self.kwargs["tool_extensions"], "cli-goal-control")
                await extension.tools["set_goal"]("replacement condition")
            async for chunk in super().process_message_stream(message, attachments):
                yield chunk

    app = _build_goal_test_app(tmp_path, monkeypatch, agent_cls=ReplacingGoalAgent)
    async with app.run_test() as pilot:
        await asyncio.wait_for(app._startup_complete.wait(), 10)
        agent = GoalFakeAgent.instances[-1]
        agent._goal_evaluate_results = [GoalVerdict(False, "gap"), GoalVerdict(True, "must not evaluate")]
        if terminal == "failed":
            agent._fail_on_calls = {2}
        else:
            agent._cancel_on_calls = {2}
        await submit_goal(app, pilot, "/goal original condition")
        await _wait_turn_idle(app, pilot, timeout=10)
        assert app._goal is not None and app._goal.condition == "replacement condition"
        assert agent._evaluate_calls == ["original condition"]
        assert app._goal.paused and not app._goal.met
        expected_reason = "provider rejected the request" if terminal == "failed" else "Stopped by user."
        assert expected_reason in app._goal.status_note
        saved_goal = app.store.load(app.session.session_id).goal
        assert saved_goal["paused"] and not saved_goal["met"]
        assert expected_reason in saved_goal["status_note"]


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["cancelled", "failed"])
async def test_tui_replacement_goal_pauses_on_workspace_switch_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scheduling: None, terminal: str
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    class ReplacingGoalAgent(GoalFakeAgent):
        async def process_message_stream(self, message: str, attachments: Any = None) -> Any:
            if self._stream_call_count == 1:
                extension = extension_by_name(self.kwargs["tool_extensions"], "cli-goal-control")
                await extension.tools["set_goal"]("replacement condition")
                app._pending_workspace_switch = tui_state.PendingWorkspaceSwitch(
                    new_root=tmp_path / "replacement",
                    old_root=app.project_path,
                    branch="replacement",
                    old_branch="main",
                    previous_active_metadata=app.session.active_project_path,
                )
            async for chunk in super().process_message_stream(message, attachments):
                yield chunk

    app = _build_goal_test_app(tmp_path, monkeypatch, agent_cls=ReplacingGoalAgent)

    async def finish_switch(pending: tui_state.PendingWorkspaceSwitch) -> None:
        assert pending is app._pending_workspace_switch
        entered.set()
        await release.wait()
        app._pending_workspace_switch = None
        if terminal == "failed":
            raise ValueError("workspace rebuild failed")

    monkeypatch.setattr(app, "_finish_pending_workspace_switch", finish_switch)
    async with app.run_test() as pilot:
        await asyncio.wait_for(app._startup_complete.wait(), 10)
        agent = GoalFakeAgent.instances[-1]
        agent._goal_evaluate_results = [GoalVerdict(False, "gap"), GoalVerdict(True, "must not evaluate")]
        await submit_goal(app, pilot, "/goal original condition")
        execution = None
        try:
            await asyncio.wait_for(entered.wait(), 10)
            execution = app._agent_execution_task
            assert execution is not None and app.agent_worker is not None
            app.agent_worker.exit_on_error = False
            if terminal == "cancelled":
                app.action_cancel_generation()
            release.set()
            await _wait_turn_idle(app, pilot, timeout=10)
            assert app._goal is not None and app._goal.condition == "replacement condition"
            assert app._goal.paused and not app._goal.met
            assert agent._evaluate_calls == ["original condition"]
            assert agent._stream_call_count == 2
            expected_reason = "Stopped by user." if terminal == "cancelled" else "workspace rebuild failed"
            assert expected_reason in app._goal.status_note
            saved_goal = app.store.load(app.session.session_id).goal
            assert saved_goal["paused"] and not saved_goal["met"]
            assert expected_reason in saved_goal["status_note"]
        finally:
            release.set()
            if execution is not None:
                await asyncio.wait_for(asyncio.gather(execution, return_exceptions=True), 10)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["goal", "loop"])
async def test_real_launcher_releases_automation_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: Any, mode: str
) -> None:
    app = _build_goal_test_app(tmp_path, monkeypatch) if mode == "goal" else _build_loop_test_app(tmp_path, monkeypatch)
    loop = asyncio.get_running_loop()
    factory = loop.get_task_factory()

    async def exercise(pilot: Pilot) -> None:
        await asyncio.wait_for(app._startup_complete.wait(), 10)
        if app._loop_scheduler_timer is not None:
            app._loop_scheduler_timer.stop()
            app._loop_scheduler_timer = None
        if mode == "goal":
            agent = GoalFakeAgent.instances[-1]
            agent._goal_evaluate_results = [GoalVerdict(False, "gap"), GoalVerdict(True, "verified")]
            await submit_goal(app, pilot, "/goal verified")
        else:
            await submit_loop(app, "/loop 5m poll")
        worker = app.agent_worker
        assert worker is not None
        await worker.wait()

        assert getattr(app, "agent_worker") is None
        assert not app._turn_active and not app._loop_iteration_active
        assert not app._turn_controller.busy
        if mode == "goal":
            assert app._goal is not None and app._goal.met
            assert len(GoalFakeAgent.instances[-1].messages) == 2
        else:
            state = app._scheduled_loop
            assert state is not None and state.iterations == 1
            assert state.last_completed_at == START.isoformat()
            clock.advance(minutes=5)
            await app._loop_tick()
            worker = app.agent_worker
            assert worker is not None
            await worker.wait()
            assert app.agent_worker is None
            assert not app._turn_controller.busy
            assert state.iterations == 2
        app.exit()

    try:
        await asyncio.wait_for(app.run_async(headless=True, auto_pilot=exercise), 15)
    finally:
        loop.set_task_factory(factory)
