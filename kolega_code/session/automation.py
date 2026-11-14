"""Goal continuation and scheduled iterations, independent of frontend lifecycles."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from kolega_code.agent.goal import GoalVerdict
from kolega_code.cli.goal import GoalState, build_goal_nudge, now_iso
from kolega_code.cli.loop import LoopState, build_loop_iteration_prompt

from .turns import AsyncStep, TurnController, TurnOutcome, TurnResult


class GoalOutcome(StrEnum):
    MET = "met"
    CAPPED = "capped"
    CANCELLED = "cancelled"
    FAILED = "failed"
    YIELDED = "yielded"
    INACTIVE = "inactive"
    STALE = "stale"


class LoopDecision(StrEnum):
    FIRE = "fire"
    WAIT = "wait"
    DEFER = "defer"
    EXPIRED = "expired"
    CAPPED = "capped"
    INACTIVE = "inactive"


class LoopOutcome(StrEnum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class GoalPolicy:
    # TUI history counts unmet evaluations; ask's semantic events count all verdicts.
    count_met_evaluations: bool = True
    pause_at_cap: bool = True


@dataclass(frozen=True)
class GoalHooks:
    evaluate: Callable[[str], Awaitable[GoalVerdict]]
    continue_turn: Callable[[str, GoalVerdict], Awaitable[TurnResult]]
    drain_tokens: Callable[[], None]
    evaluated: Callable[[GoalVerdict], Awaitable[None]]
    finished: Callable[[GoalOutcome, str], Awaitable[None]]
    before_evaluate: AsyncStep | None = None
    before_continue: AsyncStep | None = None
    is_current: Callable[[], bool] = lambda: True


@dataclass(frozen=True)
class LoopHooks:
    run_turn: Callable[[str], Awaitable[TurnResult]]
    prepare_prompt: Callable[[], Awaitable[str | None]]
    drain_tokens: Callable[[], None]
    reset_context: AsyncStep | None = None
    settle_background: AsyncStep | None = None
    started: AsyncStep | None = None
    is_current: Callable[[], bool] = lambda: True
    ready: Callable[[], bool] = lambda: True


def pause_goal(state: GoalState, reason: str) -> None:
    if not state.met:
        state.paused = True
        state.status_note = reason


def resume_goal(state: GoalState) -> bool:
    if not state.paused or state.met:
        return False
    state.paused = False
    state.status_note = ""
    return True


class AutomationDriver:
    def __init__(self, turns: TurnController[Any]) -> None:
        self.turns = turns

    @staticmethod
    def _check_cancellation() -> None:
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            raise asyncio.CancelledError

    def _fence(self, is_current: Callable[[], bool]) -> Callable[[], bool]:
        owner, execution = self.turns.owner, self.turns.execution_task
        return lambda: (
            self.turns.owner is owner
            and self.turns.execution_task is execution
            and not self.turns.closed
            and not self.turns.draining_tasks
            and is_current()
        )

    async def run_goal(self, state: GoalState, hooks: GoalHooks, *, policy: GoalPolicy = GoalPolicy()) -> GoalOutcome:
        current = self._fence(hooks.is_current)

        async def finish(outcome: GoalOutcome, error: str = "") -> GoalOutcome:
            hooks.drain_tokens()
            if outcome is GoalOutcome.MET:
                state.met = True
                state.status_note = ""
            elif outcome is GoalOutcome.CAPPED:
                if policy.pause_at_cap:
                    pause_goal(state, f"Reached the turn cap ({state.max_turns}).")
            elif outcome in (GoalOutcome.CANCELLED, GoalOutcome.FAILED):
                pause_goal(state, error or "Stopped by user.")
            await hooks.finished(outcome, error)
            return outcome

        try:
            while current() and state.is_active:
                self._check_cancellation()
                if self.turns.stop_pending:
                    return await finish(GoalOutcome.CANCELLED)
                if state.turns_evaluated >= state.max_turns:
                    return await finish(GoalOutcome.CAPPED)
                if not self.turns.inputs.empty():
                    return GoalOutcome.YIELDED
                if hooks.before_evaluate is not None:
                    await hooks.before_evaluate()
                    if not current():
                        return GoalOutcome.STALE
                    if not state.is_active:
                        return GoalOutcome.INACTIVE
                    self._check_cancellation()
                    if self.turns.stop_pending:
                        return await finish(GoalOutcome.CANCELLED)
                    if not self.turns.inputs.empty():
                        return GoalOutcome.YIELDED
                hooks.drain_tokens()
                verdict = await hooks.evaluate(state.condition)
                if not current():
                    return GoalOutcome.STALE
                if not state.is_active:
                    return GoalOutcome.INACTIVE
                self._check_cancellation()
                state.last_reason = verdict.reason
                state.last_evaluated_at = now_iso()
                if not verdict.met or policy.count_met_evaluations:
                    state.turns_evaluated += 1
                await hooks.evaluated(verdict)
                if not current():
                    return GoalOutcome.STALE
                if not state.is_active:
                    return GoalOutcome.INACTIVE
                self._check_cancellation()
                if self.turns.stop_pending:
                    return await finish(GoalOutcome.CANCELLED)
                if verdict.met:
                    return await finish(GoalOutcome.MET)
                if state.turns_evaluated >= state.max_turns:
                    return await finish(GoalOutcome.CAPPED)
                if not self.turns.inputs.empty():
                    return GoalOutcome.YIELDED
                if hooks.before_continue is not None:
                    await hooks.before_continue()
                    if not current():
                        return GoalOutcome.STALE
                    if not state.is_active:
                        return GoalOutcome.INACTIVE
                    self._check_cancellation()
                    if self.turns.stop_pending:
                        return await finish(GoalOutcome.CANCELLED)
                    if not self.turns.inputs.empty():
                        return GoalOutcome.YIELDED
                nudge = build_goal_nudge(state.condition, verdict, state.max_turns - state.turns_evaluated)
                result = await hooks.continue_turn(nudge, verdict)
                if not current():
                    return GoalOutcome.STALE
                if result.outcome is TurnOutcome.CANCELLED:
                    return await finish(GoalOutcome.CANCELLED)
                if result.outcome is TurnOutcome.FAILED:
                    return await finish(GoalOutcome.FAILED, result.error.strip() or "The turn failed.")
            return GoalOutcome.INACTIVE if current() else GoalOutcome.STALE
        except asyncio.CancelledError:
            if current() and state.is_active:
                await finish(GoalOutcome.CANCELLED)
            raise
        except Exception as exc:
            if current() and state.is_active:
                await finish(GoalOutcome.FAILED, str(exc))
            raise

    def loop_decision(
        self, state: LoopState, *, ready: bool = True, owned: bool = False, now: datetime | None = None
    ) -> LoopDecision:
        if state.stopped or self.turns.closed or self.turns.draining_tasks:
            return LoopDecision.INACTIVE
        if state.is_expired(now):
            return LoopDecision.EXPIRED
        if state.reached_cap:
            return LoopDecision.CAPPED
        if not state.is_due(now):
            state.deferred = False
            return LoopDecision.WAIT
        if not ready or not self.turns.inputs.empty() or (not owned and self.turns.busy):
            state.deferred = True
            return LoopDecision.DEFER
        state.deferred = False
        return LoopDecision.FIRE

    async def wait_for_loop(
        self,
        state: LoopState,
        wait: Callable[[float], Awaitable[None]],
        *,
        sleeping: Callable[[float], None],
    ) -> None:
        remaining = state.seconds_until()
        if remaining > 0 and state.is_active:
            sleeping(remaining)
        while state.is_active and not self.turns.closed and not self.turns.draining_tasks:
            self._check_cancellation()
            remaining = state.seconds_until()
            if remaining <= 0:
                return
            await wait(min(1.0, remaining))

    async def run_loop_iteration(self, state: LoopState, hooks: LoopHooks) -> LoopOutcome:
        current = self._fence(hooks.is_current)

        def can_fire() -> bool:
            if current() and self.turns.stop_pending:
                raise asyncio.CancelledError
            return current() and self.loop_decision(state, ready=hooks.ready(), owned=True) is LoopDecision.FIRE

        if not can_fire():
            return LoopOutcome.SKIPPED
        try:
            self._check_cancellation()
            if state.fresh and state.iterations > 0:
                if hooks.settle_background is not None:
                    await hooks.settle_background()
                    if not can_fire():
                        return LoopOutcome.SKIPPED
                    self._check_cancellation()
                if hooks.reset_context is not None:
                    await hooks.reset_context()
                    if not can_fire():
                        return LoopOutcome.SKIPPED
                    self._check_cancellation()
            prompt = await hooks.prepare_prompt()
            if prompt is None or not can_fire():
                return LoopOutcome.SKIPPED
            self._check_cancellation()
            state.mark_fired()
            if hooks.started is not None:
                await hooks.started()
                if not current():
                    return LoopOutcome.SKIPPED
                self._check_cancellation()
                if self.turns.stop_pending:
                    return LoopOutcome.CANCELLED
                if state.stopped or state.is_expired() or not hooks.ready() or not self.turns.inputs.empty():
                    return LoopOutcome.SKIPPED
            result = await hooks.run_turn(
                build_loop_iteration_prompt(
                    prompt, iteration=state.iterations, max_iterations=state.max_iterations, fresh=state.fresh
                )
            )
            if not current():
                return LoopOutcome.SKIPPED
            if result.outcome is TurnOutcome.CANCELLED:
                return LoopOutcome.CANCELLED
            if result.outcome is TurnOutcome.FAILED:
                return LoopOutcome.FAILED
            self._check_cancellation()
            if self.turns.stop_pending:
                return LoopOutcome.CANCELLED
            state.advance_after_completion()
            return LoopOutcome.COMPLETED
        finally:
            if current():
                hooks.drain_tokens()
