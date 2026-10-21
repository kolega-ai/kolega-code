"""A stalled journal write must not hide a finished turn's result.

Regression for session ``snappy-sloth-4970``: ``write_plan`` was captured, the agent
turn completed, and then the turn worker parked on its journal/session writes. The
plan card was created *after* those writes, so it never appeared, and the dashboard
showed a finished-looking turn for half an hour.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from kolega_code.cli.tui import state as tui_state

from ._app_test_utils import (
    FakeCoderAgent,
    _build_sub_agent_test_app,
    install_fake_agents,
    wait_for_turn_idle,
)


class _PlanAgent(FakeCoderAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.completed_plan: str | None = "# Plan\n\n- close the Mistral gaps"

    async def process_message_stream(self, message, attachments=None):
        yield {"type": "response", "content": "Writing the plan now.", "complete": True, "uuid": "response-1"}

    def consume_completed_plan(self):
        plan, self.completed_plan = self.completed_plan, None
        return plan


async def _wait_for(pilot, predicate, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await pilot.pause(0.01)
    raise AssertionError("Timed out waiting for the app state")


@pytest.mark.asyncio
async def test_plan_card_is_rendered_before_the_session_save(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    install_fake_agents(monkeypatch, planning_cls=_PlanAgent)
    app._turn_save_timeout_seconds = 0.05

    release = asyncio.Event()

    async def stalled_save() -> None:
        await release.wait()

    async with app.run_test() as pilot:
        await app.action_toggle_interaction_mode()
        monkeypatch.setattr(app, "_save_session_async", stalled_save)

        turn = asyncio.create_task(app._process_message("plan this"))
        await _wait_for(pilot, lambda: app._plan_decision_active)

        # The save is still blocked, yet the plan and its actions are on screen.
        assert app.conversation_entries[-1].kind == "plan"
        assert "close the Mistral gaps" in app.conversation_entries[-1].content
        assert app.query_one("#plan_actions").display is True

        release.set()
        await asyncio.wait_for(turn, 3.0)
        # The stalled write was abandoned, so the turn released its worker and the
        # plan buttons (which require agent_worker is None) can act.
        assert app.agent_worker is None
        assert app._turn_active is False


@pytest.mark.asyncio
async def test_turn_finishes_when_the_post_turn_save_stalls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    install_fake_agents(monkeypatch, planning_cls=_PlanAgent)
    app._turn_save_timeout_seconds = 0.05

    async def never_returns() -> None:
        await asyncio.sleep(3600)

    async with app.run_test() as pilot:
        await app.action_toggle_interaction_mode()
        monkeypatch.setattr(app, "_save_session_history_async", never_returns)

        app.agent_worker = app.run_worker(app._process_message("plan this"), name="kolega-turn", group="turns")
        await _wait_for(pilot, lambda: app._plan_decision_active)

        # The stalled save was abandoned instead of parking the turn forever.
        assert app._turn_started_at is None
        assert app._turn_final_state is tui_state.TurnState.IDLE
        assert app.conversation_entries[-1].kind == "plan"
        await wait_for_turn_idle(app, pilot)
