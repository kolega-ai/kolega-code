"""Esc/Ctrl-C must always return the TUI to a usable state.

Exercise cancellation-resistant finalization independently of the observed
stale-worker ownership race: ``Worker.cancel()`` alone cannot settle a task
which suppresses ``CancelledError`` while waiting for durable writes.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from kolega_code.cli import messages
from kolega_code.cli.tui import state as tui_state

from ._app_test_utils import FakeCoderAgent, _build_sub_agent_test_app, install_fake_agents


class _UnstoppableStreamAgent(FakeCoderAgent):
    """A turn that cannot unwind on cancel, like a worker parked on journal I/O.

    Swallowing ``CancelledError`` inside the async generator is what makes the
    *consuming task* unstoppable: the worker stays pending with ``cancelling() == 1``
    instead of raising, exactly like ``BaseAgent._finish_recorded_turn``'s
    ``while not commit.done(): await asyncio.shield(commit)`` loop.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.cancellations = 0
        self.fail_after_release = False

    async def process_message_stream(self, message):
        yield {"type": "response", "content": "working", "complete": True, "uuid": "response-1"}
        self.entered.set()
        while not self.release.is_set():
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancellations += 1
        yield {"type": "response", "content": "late response", "complete": True, "uuid": "late-response"}
        yield {"type": "thinking", "content": "late thinking", "complete": True, "uuid": "late-thinking"}
        if self.fail_after_release:
            raise RuntimeError("late detached failure")


async def _wait_for(pilot, predicate, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await pilot.pause(0.01)
    raise AssertionError("Timed out waiting for the app state to settle")


async def _start_stuck_turn(app, pilot):
    agent = app.agent
    assert isinstance(agent, _UnstoppableStreamAgent)
    app.agent_worker = app.run_worker(app._process_message("hello"), name="kolega-turn", group="turns")
    await _wait_for(pilot, lambda: agent.entered.is_set())
    return agent


@pytest.mark.asyncio
async def test_escape_forces_stop_when_the_worker_ignores_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    # Must follow the helper, which installs the default fake agents itself.
    install_fake_agents(monkeypatch, coder_cls=_UnstoppableStreamAgent)
    app._stop_force_after_seconds = 0.05

    async with app.run_test() as pilot:
        agent = await _start_stuck_turn(app, pilot)
        try:
            app.action_cancel_generation()
            # Inspect before the watchdog can fire on a loaded runner.
            assert app._turn_status_text == messages.STOP_REQUESTED
            assert app._turn_started_at is not None

            await _wait_for(pilot, lambda: app.agent_worker is None)
            assert app._turn_started_at is None
            assert app._turn_final_state is tui_state.TurnState.STOPPED
            assert app._turn_active is False
            assert app.query_one("#composer").disabled is False
            assert app._agent_execution_draining()
            await _wait_for(pilot, lambda: agent.cancellations >= 2)
        finally:
            agent.release.set()
            await _wait_for(pilot, lambda: not app._agent_execution_draining())
        assert app._status_state.activity == messages.STOPPED_BY_USER
        assert app._turn_started_at is None
        assert app._turn_final_state is tui_state.TurnState.STOPPED
        assert not any("late " in entry.content for entry in app.conversation_entries)


@pytest.mark.asyncio
async def test_second_escape_settles_a_parked_worker_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fake_agents(monkeypatch, coder_cls=_UnstoppableStreamAgent)
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    install_fake_agents(monkeypatch, coder_cls=_UnstoppableStreamAgent)
    # Long enough that only the escalation path can settle it.
    app._stop_force_after_seconds = 60.0

    async with app.run_test() as pilot:
        agent = await _start_stuck_turn(app, pilot)
        try:
            app.action_cancel_generation()
            await _wait_for(pilot, lambda: agent.cancellations == 1)
            assert app.agent_worker is not None

            app.action_cancel_generation()
            await _wait_for(pilot, lambda: app.agent_worker is None, timeout=1.0)
            assert app._turn_started_at is None
            assert app._turn_final_state is tui_state.TurnState.STOPPED
            assert app.query_one("#composer").disabled is False
            assert app._agent_execution_draining()
        finally:
            agent.release.set()
            await _wait_for(pilot, lambda: not app._agent_execution_draining())


@pytest.mark.asyncio
async def test_escape_clears_a_wedged_stopping_state_with_no_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker that vanished without finalizing must not wedge the dashboard."""
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        app._begin_turn_progress("working")
        app._update_progress(messages.STOP_REQUESTED, complete=False, state=tui_state.TurnState.STOPPING)
        assert app.agent_worker is None
        assert app._turn_started_at is not None

        app.action_cancel_generation()
        await pilot.pause(0.02)

        assert app._turn_started_at is None
        assert app._turn_final_state is tui_state.TurnState.STOPPED
        assert app._turn_active is False
