"""Exercise the real launcher: run_test omits Textual's Python 3.12+ eager factory."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock

import pytest
from textual.pilot import Pilot

from kolega_code.cli.tui import state as tui_state
from kolega_code.cli.tui.widgets import ChatComposer

from ._app_test_utils import FakeCoderAgent, _build_sub_agent_test_app, install_fake_agents
from .test_app_turn_save_timeout import _PlanAgent


@pytest.mark.asyncio
@pytest.mark.parametrize("planning", [False, True], ids=["build", "plan"])
async def test_real_launcher_releases_completed_turn_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, planning: bool
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    install_fake_agents(monkeypatch, planning_cls=_PlanAgent)
    loop = asyncio.get_running_loop()
    original_factory = loop.get_task_factory()

    async def pilot_turn(_pilot: Pilot) -> None:
        await app._startup_complete.wait()
        eager_factory = getattr(asyncio, "eager_task_factory", None)
        if eager_factory is not None:
            assert loop.get_task_factory() is eager_factory
        if planning:
            await app.action_toggle_interaction_mode()
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("plan this" if planning else "hello")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        worker = app.agent_worker
        assert worker is not None
        await worker.wait()

        assert getattr(app, "agent_worker") is None, "a successful worker must not retain the turn slot"
        assert app._turn_active is False
        assert app._turn_started_at is None
        assert app._turn_final_state is tui_state.TurnState.IDLE
        if planning:
            assert app.conversation_entries[-1].kind == "plan"
            assert app._plan_decision_active
            assert "close the Mistral gaps" in app.store.load(app.session.session_id).latest_plan_markdown
        else:
            assert any(entry.kind == "assistant" and entry.content == "done" for entry in app.conversation_entries)
            composer.load_text("another turn")
            await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
            next_worker = app.agent_worker
            assert next_worker is not None and next_worker is not worker
            await next_worker.wait()
            assert getattr(app, "agent_worker") is None
            assert cast(FakeCoderAgent, app.agent).messages == ["hello", "another turn"]
        app.exit()

    try:
        await asyncio.wait_for(app.run_async(headless=True, auto_pilot=pilot_turn), 10)
    finally:
        loop.set_task_factory(original_factory)


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["background-completions", "empty-scheduled-loop"])
async def test_real_launcher_releases_other_turn_entry_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    loop = asyncio.get_running_loop()
    original_factory = loop.get_task_factory()

    async def pilot_turn(_pilot: Pilot) -> None:
        await app._startup_complete.wait()
        if entry == "background-completions":
            jobs = AsyncMock()
            jobs.has_work = False
            jobs.pending_count = 0
            app._background_jobs = jobs
            assert app.agent is not None
            app.agent.background_needs_turn = False
            work = app._process_background_completions(jobs)
        else:
            assert app._scheduled_loop is None
            work = app._run_loop_iteration()
        app.agent_worker = app.run_worker(work, name="kolega-turn", group="turns")
        worker = app.agent_worker
        await worker.wait()
        assert app.agent_worker is None
        assert not app._turn_active
        assert not app._loop_iteration_active
        app._background_jobs = None
        app.exit()

    try:
        await asyncio.wait_for(app.run_async(headless=True, auto_pilot=pilot_turn), 10)
    finally:
        loop.set_task_factory(original_factory)
