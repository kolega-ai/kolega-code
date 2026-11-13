from __future__ import annotations

import asyncio
import subprocess
import sys
import textwrap
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio

from kolega_code.agent.goal import GoalVerdict
from kolega_code.cli import loop as loop_module
from kolega_code.cli.goal import GoalState
from kolega_code.cli.loop import LoopState, interval_schedule, parse_schedule_text
from kolega_code.session.automation import (
    AutomationDriver,
    GoalHooks,
    GoalOutcome,
    GoalPolicy,
    LoopDecision,
    LoopHooks,
    LoopOutcome,
    pause_goal,
    resume_goal,
)
from kolega_code.session.turns import TurnController, TurnOutcome, TurnResult

pytestmark = pytest.mark.usefixtures("isolated_cli_env")


@pytest_asyncio.fixture(params=["lazy", "eager"])
async def scheduling(request: pytest.FixtureRequest) -> AsyncIterator[None]:
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    factory = None
    if request.param == "eager":
        factory = getattr(asyncio, "eager_task_factory", None)
        if factory is None:
            pytest.skip("Eager tasks require Python 3.12")
    loop.set_task_factory(factory)
    try:
        yield
    finally:
        loop.set_task_factory(previous)


@pytest_asyncio.fixture
async def turns() -> AsyncIterator[TurnController[str]]:
    controller: TurnController[str] = TurnController()
    try:
        yield controller
    finally:
        pending = set(controller.draining_tasks)
        if controller.execution_task is not None:
            pending.add(controller.execution_task)
        for task in pending:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


@dataclass
class _Clock:
    moment: datetime = datetime(2026, 11, 13, 12)

    def advance(self, seconds: int) -> None:
        self.moment += timedelta(seconds=seconds)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    clock = _Clock()
    monkeypatch.setattr(loop_module, "now_local", lambda: clock.moment)
    return clock


@dataclass
class _Probe:
    gate_at: str | None = None
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    trace: list[str] = field(default_factory=list)
    current: bool = True

    async def step(self, name: str) -> None:
        self.trace.append(name)
        if name == self.gate_at:
            self.entered.set()
            await self.release.wait()

    def drain_tokens(self) -> None:
        self.trace.append("drain")


@dataclass
class _GoalProbe(_Probe):
    verdicts: list[GoalVerdict] = field(default_factory=lambda: [GoalVerdict(True, "verified")])
    result: TurnResult = TurnResult(TurnOutcome.COMPLETED)
    conditions: list[str] = field(default_factory=list)
    continuations: list[tuple[str, GoalVerdict]] = field(default_factory=list)
    evaluations: list[GoalVerdict] = field(default_factory=list)
    finishes: list[tuple[GoalOutcome, str]] = field(default_factory=list)

    async def before_evaluate(self) -> None:
        await self.step("before")

    async def evaluate(self, condition: str) -> GoalVerdict:
        self.conditions.append(condition)
        await self.step("evaluate")
        return self.verdicts.pop(0)

    async def continue_turn(self, prompt: str, verdict: GoalVerdict) -> TurnResult:
        self.continuations.append((prompt, verdict))
        await self.step("continue")
        return self.result

    async def evaluated(self, verdict: GoalVerdict) -> None:
        self.evaluations.append(verdict)
        await self.step("evaluated")

    async def finished(self, outcome: GoalOutcome, error: str) -> None:
        self.finishes.append((outcome, error))
        await self.step("finished")

    def hooks(self) -> GoalHooks:
        return GoalHooks(
            evaluate=self.evaluate,
            continue_turn=self.continue_turn,
            drain_tokens=self.drain_tokens,
            evaluated=self.evaluated,
            finished=self.finished,
            before_evaluate=self.before_evaluate,
            is_current=lambda: self.current,
        )


@dataclass
class _LoopProbe(_Probe):
    prompt: str | None = "Inspect the repository."
    result: TurnResult = TurnResult(TurnOutcome.COMPLETED)
    prompts: list[str] = field(default_factory=list)
    ready: bool = True

    async def settle_background(self) -> None:
        await self.step("settle")

    async def reset_context(self) -> None:
        await self.step("reset")

    async def prepare_prompt(self) -> str | None:
        await self.step("prepare")
        return self.prompt

    async def run_turn(self, prompt: str) -> TurnResult:
        self.prompts.append(prompt)
        await self.step("run")
        return self.result

    def hooks(self) -> LoopHooks:
        return LoopHooks(
            run_turn=self.run_turn,
            prepare_prompt=self.prepare_prompt,
            drain_tokens=self.drain_tokens,
            reset_context=self.reset_context,
            settle_background=self.settle_background,
            is_current=lambda: self.current,
            ready=lambda: self.ready,
        )


async def _wait(event: asyncio.Event) -> None:
    await asyncio.wait_for(event.wait(), 10)


def _loop(*, fresh: bool = False, iterations: int = 0) -> LoopState:
    state = LoopState.create(interval_schedule(60), "Inspect the repository.", fresh=fresh, max_iterations=3)
    state.iterations = iterations
    return state


@pytest.mark.asyncio
@pytest.mark.parametrize(("count_met", "expected_count"), [(False, 1), (True, 2)], ids=["tui", "ask"])
async def test_goal_continues_until_met_with_frontend_counting_policy(
    count_met: bool, expected_count: int, turns: TurnController[str], scheduling: None
) -> None:
    state = GoalState.create("All tests pass.", max_turns=4, run_to_completion=True)
    unmet, met = GoalVerdict(False, "one test fails"), GoalVerdict(True, "all tests pass")
    probe = _GoalProbe(verdicts=[unmet, met])
    driver = AutomationDriver(turns)

    outcome = await turns.start(
        lambda: driver.run_goal(state, probe.hooks(), policy=GoalPolicy(count_met_evaluations=count_met))
    )

    assert outcome is GoalOutcome.MET
    assert state.met and not state.paused
    assert not state.is_active
    assert state.turns_evaluated == expected_count
    assert state.last_reason == met.reason
    assert state.last_evaluated_at
    assert state.status_note == ""
    assert probe.conditions == [state.condition, state.condition]
    assert probe.evaluations == [unmet, met]
    assert len(probe.continuations) == 1
    prompt, verdict = probe.continuations[0]
    assert verdict is unmet
    assert state.condition in prompt
    assert unmet.reason in prompt
    assert "3 evaluation turn(s) remain" in prompt
    assert probe.finishes == [(GoalOutcome.MET, "")]
    assert probe.trace == [
        "before",
        "drain",
        "evaluate",
        "evaluated",
        "continue",
        "before",
        "drain",
        "evaluate",
        "evaluated",
        "drain",
        "finished",
    ]
    assert not turns.busy


@pytest.mark.asyncio
@pytest.mark.parametrize("pause_at_cap", [False, True], ids=["ask", "tui"])
@pytest.mark.parametrize("already_capped", [False, True])
async def test_unmet_goal_cap_never_sends_an_extra_nudge(
    pause_at_cap: bool, already_capped: bool, turns: TurnController[str], scheduling: None
) -> None:
    state = GoalState.create("All tests pass.", max_turns=3, run_to_completion=True)
    state.turns_evaluated = 3 if already_capped else 0
    verdict = GoalVerdict(False, "still failing")
    probe = _GoalProbe(verdicts=[verdict] * 3)
    driver = AutomationDriver(turns)

    outcome = await turns.start(
        lambda: driver.run_goal(state, probe.hooks(), policy=GoalPolicy(pause_at_cap=pause_at_cap))
    )

    assert outcome is GoalOutcome.CAPPED
    assert state.turns_evaluated == 3
    assert not state.met
    assert state.paused is pause_at_cap
    assert state.status_note == ("Reached the turn cap (3)." if pause_at_cap else "")
    assert len(probe.evaluations) == (0 if already_capped else 3)
    assert len(probe.continuations) == (0 if already_capped else 2)
    assert probe.finishes == [(GoalOutcome.CAPPED, "")]
    assert probe.trace[-2:] == ["drain", "finished"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected"), [("  provider failed  ", "provider failed"), (" \n", "The turn failed.")]
)
async def test_failed_goal_continuation_pauses_without_another_verifier(
    error: str, expected: str, turns: TurnController[str], scheduling: None
) -> None:
    state = GoalState.create("All tests pass.")
    probe = _GoalProbe(
        verdicts=[GoalVerdict(False, "remaining gap")],
        result=TurnResult(TurnOutcome.FAILED, error),
    )

    outcome = await turns.start(lambda: AutomationDriver(turns).run_goal(state, probe.hooks()))

    assert outcome is GoalOutcome.FAILED
    assert state.paused and not state.met
    assert state.status_note == expected
    assert state.turns_evaluated == 1
    assert len(probe.conditions) == len(probe.continuations) == 1
    assert probe.finishes == [(GoalOutcome.FAILED, expected)]
    assert probe.trace[-3:] == ["continue", "drain", "finished"]


@pytest.mark.asyncio
async def test_cancelled_goal_turn_result_pauses_without_another_verifier(
    turns: TurnController[str], scheduling: None
) -> None:
    state = GoalState.create("All tests pass.")
    probe = _GoalProbe(
        verdicts=[GoalVerdict(False, "remaining gap")],
        result=TurnResult(TurnOutcome.CANCELLED),
    )

    outcome = await turns.start(lambda: AutomationDriver(turns).run_goal(state, probe.hooks()))

    assert outcome is GoalOutcome.CANCELLED
    assert state.paused and not state.met
    assert state.status_note == "Stopped by user."
    assert len(probe.conditions) == len(probe.continuations) == 1
    assert probe.finishes == [(GoalOutcome.CANCELLED, "")]


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["evaluate", "continue"])
async def test_cancellation_during_goal_verification_or_continuation_pauses_and_propagates(
    phase: str, turns: TurnController[str], scheduling: None
) -> None:
    state = GoalState.create("All tests pass.")
    probe = _GoalProbe(gate_at=phase, verdicts=[GoalVerdict(False, "remaining gap")])
    task = turns.start(lambda: AutomationDriver(turns).run_goal(state, probe.hooks()))
    await _wait(probe.entered)

    assert turns.request_stop()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert state.paused and not state.met
    assert state.status_note == "Stopped by user."
    assert state.turns_evaluated == (1 if phase == "continue" else 0)
    assert probe.finishes == [(GoalOutcome.CANCELLED, "")]
    assert probe.trace[-2:] == ["drain", "finished"]
    assert not turns.busy
    assert not turns.stop_pending


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["before", "evaluate", "evaluated", "continue"])
@pytest.mark.parametrize("invalidate", ["owner", "host", "detach"])
async def test_goal_ownership_is_rechecked_after_each_await(
    phase: str, invalidate: str, turns: TurnController[str], scheduling: None
) -> None:
    state = GoalState.create("All tests pass.")
    probe = _GoalProbe(gate_at=phase, verdicts=[GoalVerdict(False, "remaining gap")])
    task = turns.start(lambda: AutomationDriver(turns).run_goal(state, probe.hooks()))
    await _wait(probe.entered)
    replacement = object()
    if invalidate == "owner":
        turns.owner = replacement
    elif invalidate == "host":
        probe.current = False
    else:
        turns.detach(cancel_execution=False)
    snapshot, trace = state.to_dict(), list(probe.trace)

    probe.release.set()
    assert await task is GoalOutcome.STALE

    assert state.to_dict() == snapshot
    assert probe.trace == trace
    assert probe.finishes == []
    if invalidate == "owner":
        assert turns.owner is replacement


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["evaluate", "continue"])
async def test_cancelling_a_detached_goal_does_not_pause_or_finish_the_old_state(
    phase: str, turns: TurnController[str], scheduling: None
) -> None:
    state = GoalState.create("All tests pass.")
    probe = _GoalProbe(gate_at=phase, verdicts=[GoalVerdict(False, "remaining gap")])
    task = turns.start(lambda: AutomationDriver(turns).run_goal(state, probe.hooks()))
    await _wait(probe.entered)
    snapshot, trace = state.to_dict(), list(probe.trace)

    turns.detach()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert state.to_dict() == snapshot
    assert probe.trace == trace
    assert probe.finishes == []
    assert not turns.busy


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["initial", "evaluated", "continue"])
async def test_goal_yields_to_queued_human_input_without_consuming_it(
    phase: str, turns: TurnController[str], scheduling: None
) -> None:
    state = GoalState.create("All tests pass.")
    probe = _GoalProbe(gate_at=phase, verdicts=[GoalVerdict(False, "remaining gap")])
    if phase == "initial":
        turns.inputs.put_nowait("Human correction")
    task = turns.start(lambda: AutomationDriver(turns).run_goal(state, probe.hooks()))
    if phase != "initial":
        await _wait(probe.entered)
        turns.inputs.put_nowait("Human correction")
        probe.release.set()

    assert await task is GoalOutcome.YIELDED

    assert state.is_active and not state.paused
    assert probe.finishes == []
    assert len(probe.conditions) == (0 if phase == "initial" else 1)
    assert len(probe.continuations) == (1 if phase == "continue" else 0)
    assert turns.take_next() == "Human correction"


@pytest.mark.asyncio
async def test_goal_yields_when_human_input_arrives_during_before_evaluate(
    turns: TurnController[str], scheduling: None
) -> None:
    state = GoalState.create("All tests pass.")
    probe = _GoalProbe(gate_at="before")
    task = turns.start(lambda: AutomationDriver(turns).run_goal(state, probe.hooks()))
    await _wait(probe.entered)
    turns.inputs.put_nowait("Human correction")
    snapshot = state.to_dict()
    probe.release.set()

    assert await task is GoalOutcome.YIELDED
    assert state.to_dict() == snapshot
    assert probe.conditions == []
    assert probe.finishes == []
    assert turns.take_next() == "Human correction"


@pytest.mark.asyncio
async def test_stop_pending_takes_priority_over_a_met_verdict(turns: TurnController[str], scheduling: None) -> None:
    state = GoalState.create("All tests pass.")
    probe = _GoalProbe(gate_at="evaluate")
    task = turns.start(lambda: AutomationDriver(turns).run_goal(state, probe.hooks()))
    await _wait(probe.entered)
    assert turns.request_stop(cancel=lambda: None)
    probe.release.set()

    assert await task is GoalOutcome.CANCELLED
    assert state.paused and not state.met
    assert probe.continuations == []
    assert probe.finishes == [(GoalOutcome.CANCELLED, "")]


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["before", "evaluate", "evaluated"])
async def test_goal_deactivated_during_a_hook_does_not_finish_or_continue(
    phase: str, turns: TurnController[str], scheduling: None
) -> None:
    state = GoalState.create("All tests pass.")
    probe = _GoalProbe(gate_at=phase)
    task = turns.start(lambda: AutomationDriver(turns).run_goal(state, probe.hooks()))
    await _wait(probe.entered)
    pause_goal(state, "Human pause")
    snapshot = state.to_dict()
    probe.release.set()

    assert await task is GoalOutcome.INACTIVE
    assert state.to_dict() == snapshot
    assert probe.continuations == []
    assert probe.finishes == []


def test_goal_pause_resume_preserves_met_goals() -> None:
    state = GoalState.create("All tests pass.")
    assert not resume_goal(state)
    pause_goal(state, "Human pause")
    assert state.paused
    assert resume_goal(state)
    assert state.status_note == ""
    assert state.is_active
    state.met = True
    pause_goal(state, "Do not reopen")
    assert not state.paused
    assert not resume_goal(state)
    assert state.status_note == ""


@pytest.mark.parametrize(
    ("due_in", "ready", "owned", "busy", "queued", "expected"),
    [
        (60, True, False, False, False, LoopDecision.WAIT),
        (60, False, False, True, True, LoopDecision.WAIT),
        (0, True, False, False, False, LoopDecision.FIRE),
        (-3600, True, False, False, False, LoopDecision.FIRE),
        (0, False, False, False, False, LoopDecision.DEFER),
        (0, True, False, True, False, LoopDecision.DEFER),
        (0, True, True, True, False, LoopDecision.FIRE),
        (0, True, False, False, True, LoopDecision.DEFER),
        (0, True, True, True, True, LoopDecision.DEFER),
    ],
)
def test_loop_decision_scheduling_and_human_priority(
    due_in: int, ready: bool, owned: bool, busy: bool, queued: bool, expected: LoopDecision, clock: _Clock
) -> None:
    turns: TurnController[str] = TurnController()
    state = _loop()
    state.deferred = True
    state.next_fire_at = (clock.moment + timedelta(seconds=due_in)).isoformat()
    if busy:
        turns.owner = object()
    if queued:
        turns.inputs.put_nowait("Human correction")
    snapshot = state.to_dict()

    assert AutomationDriver(turns).loop_decision(state, ready=ready, owned=owned, now=clock.moment) is expected

    snapshot["deferred"] = expected is LoopDecision.DEFER
    assert state.to_dict() == snapshot
    assert turns.inputs.qsize() == int(queued)


@pytest.mark.parametrize(
    ("blocked_by", "expected"),
    [
        ("stopped", LoopDecision.INACTIVE),
        ("closed", LoopDecision.INACTIVE),
        ("expiry", LoopDecision.EXPIRED),
        ("cap", LoopDecision.CAPPED),
        ("expiry-and-cap", LoopDecision.EXPIRED),
    ],
)
def test_loop_decision_terminal_states_precede_queued_input(
    blocked_by: str, expected: LoopDecision, clock: _Clock
) -> None:
    turns: TurnController[str] = TurnController()
    state = _loop()
    turns.inputs.put_nowait("Human correction")
    if blocked_by == "stopped":
        state.stopped = True
    elif blocked_by == "closed":
        turns.close_admission()
    if "expiry" in blocked_by:
        state.expires_at = clock.moment.isoformat()
    if "cap" in blocked_by:
        state.iterations = state.max_iterations
    snapshot = state.to_dict()

    assert AutomationDriver(turns).loop_decision(state, now=clock.moment) is expected
    assert state.to_dict() == snapshot
    assert turns.inputs.get_nowait() == "Human correction"


@pytest.mark.asyncio
async def test_loop_decision_is_inactive_while_real_execution_drains(
    turns: TurnController[str], clock: _Clock, scheduling: None
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    async def execute() -> None:
        entered.set()
        await release.wait()

    task = turns.start(execute)
    await _wait(entered)
    turns.detach(cancel_execution=False)
    state = _loop()
    driver = AutomationDriver(turns)

    assert task in turns.draining_tasks
    assert driver.loop_decision(state, owned=True) is LoopDecision.INACTIVE
    release.set()
    await task
    assert driver.loop_decision(state) is LoopDecision.FIRE


@pytest.mark.asyncio
@pytest.mark.parametrize("fresh", [False, True])
@pytest.mark.parametrize("iterations", [0, 1])
async def test_loop_fresh_reset_settles_background_first_and_only_after_first_iteration(
    fresh: bool, iterations: int, turns: TurnController[str], clock: _Clock, scheduling: None
) -> None:
    state = _loop(fresh=fresh, iterations=iterations)
    probe = _LoopProbe()

    outcome = await turns.start(lambda: AutomationDriver(turns).run_loop_iteration(state, probe.hooks()))

    assert outcome is LoopOutcome.COMPLETED
    prefix = ["settle", "reset"] if fresh and iterations else []
    assert probe.trace == prefix + ["prepare", "run", "drain"]
    assert state.iterations == iterations + 1
    assert state.last_fired_at == clock.moment.isoformat()
    assert state.last_completed_at == clock.moment.isoformat()
    assert state.next_fire_at == (clock.moment + timedelta(seconds=60)).isoformat()
    assert len(probe.prompts) == 1
    assert f"[Scheduled loop iteration {iterations + 1} of 3]" in probe.prompts[0]
    assert state.prompt in probe.prompts[0]
    assert ("fresh conversation thread" in probe.prompts[0]) is (fresh and iterations > 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["settle", "reset"])
@pytest.mark.parametrize("blocked_by", ["expiry", "queued", "cap", "closed", "ready", "stopped"])
async def test_loop_rechecks_admission_after_fresh_preparation(
    phase: str, blocked_by: str, turns: TurnController[str], clock: _Clock, scheduling: None
) -> None:
    state = _loop(fresh=True, iterations=1)
    probe = _LoopProbe(gate_at=phase)
    task = turns.start(lambda: AutomationDriver(turns).run_loop_iteration(state, probe.hooks()))
    await _wait(probe.entered)
    if blocked_by == "expiry":
        state.expires_at = clock.moment.isoformat()
    elif blocked_by == "queued":
        turns.inputs.put_nowait("Human correction")
    elif blocked_by == "cap":
        state.iterations = state.max_iterations
    elif blocked_by == "closed":
        turns.close_admission()
    elif blocked_by == "ready":
        probe.ready = False
    else:
        state.stopped = True
    snapshot = state.to_dict()
    probe.release.set()

    assert await task is LoopOutcome.SKIPPED

    if blocked_by in ("queued", "ready"):
        snapshot["deferred"] = True
    assert state.to_dict() == snapshot
    expected_trace = ["settle"] if phase == "settle" else ["settle", "reset"]
    if blocked_by != "closed":
        expected_trace.append("drain")
    assert probe.trace == expected_trace
    assert probe.prompts == []
    if blocked_by == "queued":
        assert turns.take_next() == "Human correction"


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["settle", "reset", "prepare", "run"])
@pytest.mark.parametrize("invalidate", ["owner", "host", "detach"])
async def test_loop_stale_ownership_never_resets_fires_drains_or_advances_late(
    phase: str, invalidate: str, turns: TurnController[str], clock: _Clock, scheduling: None
) -> None:
    state = _loop(fresh=True, iterations=1)
    probe = _LoopProbe(gate_at=phase)
    task = turns.start(lambda: AutomationDriver(turns).run_loop_iteration(state, probe.hooks()))
    await _wait(probe.entered)
    replacement = object()
    if invalidate == "owner":
        turns.owner = replacement
    elif invalidate == "host":
        probe.current = False
    else:
        turns.detach(cancel_execution=False)
    snapshot, trace = state.to_dict(), list(probe.trace)
    probe.release.set()

    assert await task is LoopOutcome.SKIPPED
    assert state.to_dict() == snapshot
    assert probe.trace == trace
    if invalidate == "owner":
        assert turns.owner is replacement


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["settle", "reset", "prepare", "run"])
async def test_loop_cancellation_propagates_without_advancing_the_schedule(
    phase: str, turns: TurnController[str], clock: _Clock, scheduling: None
) -> None:
    state = _loop(fresh=True, iterations=1)
    probe = _LoopProbe(gate_at=phase)
    task = turns.start(lambda: AutomationDriver(turns).run_loop_iteration(state, probe.hooks()))
    await _wait(probe.entered)
    snapshot = state.to_dict()
    assert turns.request_stop()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert state.to_dict() == snapshot
    assert probe.trace.count("drain") == 1
    assert not turns.busy
    assert not turns.stop_pending


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["settle", "reset", "prepare", "run"])
async def test_cancelling_a_detached_loop_does_not_drain_or_advance_late(
    phase: str, turns: TurnController[str], clock: _Clock, scheduling: None
) -> None:
    state = _loop(fresh=True, iterations=1)
    probe = _LoopProbe(gate_at=phase)
    task = turns.start(lambda: AutomationDriver(turns).run_loop_iteration(state, probe.hooks()))
    await _wait(probe.entered)
    snapshot, trace = state.to_dict(), list(probe.trace)

    turns.detach()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert state.to_dict() == snapshot
    assert probe.trace == trace
    assert not turns.busy


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "outcome"),
    [
        (TurnOutcome.FAILED, LoopOutcome.FAILED),
        (TurnOutcome.CANCELLED, LoopOutcome.CANCELLED),
    ],
)
async def test_loop_unsuccessful_turn_result_drains_without_advancing(
    result: TurnOutcome, outcome: LoopOutcome, turns: TurnController[str], clock: _Clock, scheduling: None
) -> None:
    state = _loop()
    next_fire = state.next_fire_at
    probe = _LoopProbe(result=TurnResult(result, "turn failed"))

    assert await turns.start(lambda: AutomationDriver(turns).run_loop_iteration(state, probe.hooks())) is outcome
    assert state.iterations == 1
    assert state.last_fired_at == clock.moment.isoformat()
    assert state.last_completed_at is None
    assert state.next_fire_at == next_fire
    assert probe.trace == ["prepare", "run", "drain"]


@pytest.mark.asyncio
@pytest.mark.parametrize("schedule", ["60s", "*/5 * * * *"])
async def test_loop_schedules_from_completion_without_catching_up_missed_fires(
    schedule: str, turns: TurnController[str], clock: _Clock, scheduling: None
) -> None:
    state = LoopState.create(parse_schedule_text(schedule), "Inspect the repository.")
    clock.moment = datetime.fromisoformat(state.next_fire_at)
    fired_at = clock.moment
    probe = _LoopProbe(gate_at="run")
    driver = AutomationDriver(turns)
    task = turns.start(lambda: driver.run_loop_iteration(state, probe.hooks()))
    await _wait(probe.entered)
    clock.advance(3601)
    probe.release.set()

    assert await task is LoopOutcome.COMPLETED
    assert state.iterations == 1
    assert state.last_fired_at == fired_at.isoformat()
    assert state.last_completed_at == clock.moment.isoformat()
    expected_next = clock.moment + timedelta(seconds=60) if schedule == "60s" else datetime(2026, 11, 13, 13, 10)
    assert state.next_fire_at == expected_next.isoformat()
    assert driver.loop_decision(state) is LoopDecision.WAIT
    trace = list(probe.trace)
    assert await turns.start(lambda: driver.run_loop_iteration(state, probe.hooks())) is LoopOutcome.SKIPPED
    assert probe.trace == trace
    assert state.iterations == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_by", ["expiry", "stopped", "missing-prompt"])
async def test_loop_does_not_run_if_prompt_preparation_invalidates_iteration(
    blocked_by: str, turns: TurnController[str], clock: _Clock, scheduling: None
) -> None:
    state = _loop()
    probe = _LoopProbe(gate_at="prepare")
    task = turns.start(lambda: AutomationDriver(turns).run_loop_iteration(state, probe.hooks()))
    await _wait(probe.entered)
    if blocked_by == "expiry":
        state.expires_at = clock.moment.isoformat()
    elif blocked_by == "stopped":
        state.stopped = True
    else:
        probe.prompt = None
    snapshot = state.to_dict()
    probe.release.set()

    assert await task is LoopOutcome.SKIPPED
    assert state.to_dict() == snapshot
    assert probe.trace == ["prepare", "drain"]
    assert probe.prompts == []


@pytest.mark.asyncio
async def test_loop_yields_if_human_input_arrives_during_prompt_preparation(
    turns: TurnController[str], clock: _Clock, scheduling: None
) -> None:
    state = _loop()
    probe = _LoopProbe(gate_at="prepare")
    task = turns.start(lambda: AutomationDriver(turns).run_loop_iteration(state, probe.hooks()))
    await _wait(probe.entered)
    turns.inputs.put_nowait("Human correction")
    next_fire = state.next_fire_at
    probe.release.set()

    assert await task is LoopOutcome.SKIPPED
    assert state.next_fire_at == next_fire
    assert state.last_completed_at is None
    assert probe.prompts == []
    assert turns.take_next() == "Human correction"


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_by", ["ready", "queued", "expiry", "stopped", "owner", "host", "stop"])
async def test_loop_rechecks_admission_after_start_notification(
    blocked_by: str, turns: TurnController[str], clock: _Clock, scheduling: None
) -> None:
    state = _loop()
    state.max_iterations = 1
    probe = _LoopProbe(gate_at="started")
    hooks = replace(probe.hooks(), started=lambda: probe.step("started"))
    task = turns.start(lambda: AutomationDriver(turns).run_loop_iteration(state, hooks))
    await _wait(probe.entered)
    assert state.iterations == 1
    if blocked_by == "ready":
        probe.ready = False
    elif blocked_by == "queued":
        turns.inputs.put_nowait("Human correction")
    elif blocked_by == "expiry":
        state.expires_at = clock.moment.isoformat()
    elif blocked_by == "stopped":
        state.stopped = True
    elif blocked_by == "owner":
        turns.owner = object()
    elif blocked_by == "host":
        probe.current = False
    else:
        assert turns.request_stop(cancel=lambda: None)
    snapshot = state.to_dict()
    probe.release.set()

    assert await task is (LoopOutcome.CANCELLED if blocked_by == "stop" else LoopOutcome.SKIPPED)
    assert state.to_dict() == snapshot
    assert probe.prompts == []
    assert probe.trace == ["prepare", "started"] + ([] if blocked_by in ("owner", "host") else ["drain"])
    if blocked_by == "queued":
        assert turns.take_next() == "Human correction"


@pytest.mark.asyncio
async def test_loop_cap_prevents_another_iteration(turns: TurnController[str], clock: _Clock, scheduling: None) -> None:
    state = _loop()
    state.max_iterations = 1
    probe = _LoopProbe()
    driver = AutomationDriver(turns)

    assert await turns.start(lambda: driver.run_loop_iteration(state, probe.hooks())) is LoopOutcome.COMPLETED
    clock.advance(60)
    assert driver.loop_decision(state) is LoopDecision.CAPPED
    snapshot, trace = state.to_dict(), list(probe.trace)
    assert await turns.start(lambda: driver.run_loop_iteration(state, probe.hooks())) is LoopOutcome.SKIPPED
    assert state.to_dict() == snapshot
    assert probe.trace == trace


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["before", "evaluate", "evaluated", "before-continue", "continue"])
async def test_goal_hook_failure_pauses_and_propagates_without_retry(
    phase: str, turns: TurnController[str], scheduling: None
) -> None:
    class FailedProbe(_GoalProbe):
        async def step(self, name: str) -> None:
            await super().step(name)
            if name == phase:
                raise ValueError("automation hook failed")

    state = GoalState.create("All tests pass.")
    probe = FailedProbe(verdicts=[GoalVerdict(False, "remaining gap")])
    hooks = replace(probe.hooks(), before_continue=lambda: probe.step("before-continue"))
    with pytest.raises(ValueError, match="automation hook failed"):
        await turns.start(lambda: AutomationDriver(turns).run_goal(state, hooks))

    assert state.paused and not state.met
    assert state.status_note == "automation hook failed"
    assert probe.finishes == [(GoalOutcome.FAILED, "automation hook failed")]
    assert len(probe.conditions) <= 1
    assert len(probe.continuations) <= 1


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["before", "evaluate", "evaluated", "before-continue", "continue"])
async def test_goal_hook_suppressing_cancellation_never_starts_more_work(
    phase: str, turns: TurnController[str], scheduling: None
) -> None:
    class SuppressedProbe(_GoalProbe):
        async def step(self, name: str) -> None:
            try:
                await super().step(name)
            except asyncio.CancelledError:
                self.trace.append("suppressed")

    state = GoalState.create("All tests pass.")
    probe = SuppressedProbe(gate_at=phase, verdicts=[GoalVerdict(False, "remaining gap")])
    hooks = replace(probe.hooks(), before_continue=lambda: probe.step("before-continue"))
    task = turns.start(lambda: AutomationDriver(turns).run_goal(state, hooks))
    await _wait(probe.entered)
    assert turns.request_stop()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert state.paused and not state.met
    assert probe.finishes == [(GoalOutcome.CANCELLED, "")]
    assert len(probe.conditions) == (0 if phase == "before" else 1)
    assert len(probe.continuations) == (1 if phase == "continue" else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["before", "before-continue"])
async def test_goal_pending_stop_after_preparation_prevents_the_next_request(
    phase: str, turns: TurnController[str], scheduling: None
) -> None:
    state = GoalState.create("All tests pass.")
    probe = _GoalProbe(gate_at=phase, verdicts=[GoalVerdict(False, "remaining gap")])
    hooks = replace(probe.hooks(), before_continue=lambda: probe.step("before-continue"))
    task = turns.start(lambda: AutomationDriver(turns).run_goal(state, hooks))
    await _wait(probe.entered)
    assert turns.request_stop(cancel=lambda: None)
    probe.release.set()
    assert await task is GoalOutcome.CANCELLED
    assert state.paused and not state.met
    assert len(probe.conditions) == (0 if phase == "before" else 1)
    assert probe.continuations == []


@pytest.mark.asyncio
async def test_goal_input_during_before_continue_yields_without_consuming_it(
    turns: TurnController[str], scheduling: None
) -> None:
    state = GoalState.create("All tests pass.")
    probe = _GoalProbe(gate_at="before-continue", verdicts=[GoalVerdict(False, "remaining gap")])
    hooks = replace(probe.hooks(), before_continue=lambda: probe.step("before-continue"))
    task = turns.start(lambda: AutomationDriver(turns).run_goal(state, hooks))
    await _wait(probe.entered)
    turns.inputs.put_nowait("Human correction")
    probe.release.set()
    assert await task is GoalOutcome.YIELDED
    assert probe.continuations == []
    assert state.is_active
    assert turns.take_next() == "Human correction"


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["settle", "reset", "prepare", "run"])
async def test_loop_hook_failure_drains_and_propagates_without_advancing(
    phase: str, turns: TurnController[str], clock: _Clock, scheduling: None
) -> None:
    class FailedProbe(_LoopProbe):
        async def step(self, name: str) -> None:
            await super().step(name)
            if name == phase:
                raise ValueError("automation hook failed")

    state = _loop(fresh=True, iterations=1)
    deadline = state.next_fire_at
    probe = FailedProbe()
    with pytest.raises(ValueError, match="automation hook failed"):
        await turns.start(lambda: AutomationDriver(turns).run_loop_iteration(state, probe.hooks()))
    assert state.iterations == (2 if phase == "run" else 1)
    assert state.next_fire_at == deadline
    assert state.last_completed_at is None
    assert probe.trace.count("drain") == 1


def test_shared_automation_imports_and_runs_without_textual_or_rich() -> None:
    program = textwrap.dedent(
        """
        import asyncio
        import importlib.abc
        import sys

        class NoUI(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".", 1)[0] in {"textual", "rich"}:
                    raise AssertionError("UI dependency imported: " + fullname)
                return None

        sys.meta_path.insert(0, NoUI())
        from kolega_code.agent.goal import GoalVerdict
        from kolega_code.cli.goal import GoalState
        from kolega_code.session.automation import AutomationDriver, GoalHooks, GoalOutcome
        from kolega_code.session.turns import TurnController

        async def main():
            turns = TurnController()
            state = GoalState.create("All tests pass.")
            async def evaluate(condition):
                assert condition == "All tests pass."
                return GoalVerdict(True, "verified")
            async def continue_turn(prompt, verdict):
                raise AssertionError("A met goal must not continue")
            async def evaluated(verdict):
                assert verdict.met
            async def finished(outcome, error):
                assert outcome is GoalOutcome.MET and not error
            hooks = GoalHooks(evaluate, continue_turn, lambda: None, evaluated, finished)
            assert await turns.start(lambda: AutomationDriver(turns).run_goal(state, hooks)) is GoalOutcome.MET
            assert state.met and not turns.busy

        asyncio.run(main())
        assert not any(name.split(".", 1)[0] in {"textual", "rich"} for name in sys.modules)
        """
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", program],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
