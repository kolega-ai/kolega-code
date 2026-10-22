from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest
from textual.widgets import Select

from kolega_code.cli import messages
from kolega_code.cli.loop import LoopState, now_local, parse_schedule_text
from kolega_code.cli.settings import CliSettings, SettingsStore
from kolega_code.cli.tui import state as tui_state
from kolega_code.cli.tui.widgets import ChatComposer
from kolega_code.permissions import PermissionMode

from ._app_test_utils import _build_sub_agent_test_app, install_fake_agents, open_settings_screen
from .test_app_cancel_escalation import _UnstoppableStreamAgent, _start_stuck_turn, _wait_for

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("isolated_cli_env")]

TARGET_MODEL = "claude-sonnet-5"


def _configured_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings_store = SettingsStore(tmp_path / "state")
    settings = CliSettings(active_provider="anthropic", active_model="claude-opus-5")
    settings.set_api_key("anthropic", "test-key")
    settings_store.save(settings)
    app = _build_sub_agent_test_app(tmp_path, monkeypatch, settings_store=settings_store)
    install_fake_agents(monkeypatch, coder_cls=_UnstoppableStreamAgent)
    app._stop_force_after_seconds = 60
    return app


async def _force_stop(app, pilot, agent):
    task = app._agent_execution_task
    assert task is not None
    app.action_cancel_generation()
    await _wait_for(pilot, lambda: agent.cancellations == 1)
    app.action_cancel_generation()
    await _wait_for(pilot, lambda: app.agent_worker is None and app._agent_execution_draining())
    assert not app._turn_active
    return task


@pytest.mark.parametrize("command", ["model", "model-picker", "gigacode"])
async def test_commands_refuse_during_drain_and_work_afterward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    app = _configured_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        if command == "model-picker":
            app._pending_model_selection = tui_state.PendingModelSelection(
                provider="anthropic", options=[("Sonnet", TARGET_MODEL)]
            )
        agent = await _start_stuck_turn(app, pilot)
        task = app._agent_execution_task
        original_settings = deepcopy(app.settings)
        stored_settings = app.settings_store.path.read_bytes()
        notify = Mock()
        monkeypatch.setattr(app, "_notify_user", notify)
        composer = app.query_one("#composer", ChatComposer)

        async def invoke():
            if command == "model-picker":
                await app._answer_model_option(0)
            else:
                text = f"/model {TARGET_MODEL}" if command == "model" else "/gigacode on"
                composer.load_text(text)
                await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        try:
            task = await _force_stop(app, pilot, agent)
            await invoke()
            expected = (
                messages.BLOCK_STOP_BEFORE_MODE_SWITCH
                if command == "gigacode"
                else messages.BLOCK_STOP_BEFORE_MODEL_SWITCH
            )
            notify.assert_called_with(expected, severity="warning")
            assert app.settings == original_settings
            assert app.settings_store.path.read_bytes() == stored_settings
            assert app.agent is agent
            assert agent.cleanup_calls == 0
            assert not app._gigacode_enabled
            assert not app._queued_messages
            assert app.agent_worker is None
            assert not composer.disabled
            composer.load_text("still editable")
            assert composer.text == "still editable"
        finally:
            agent.release.set()
            if task is not None:
                await _wait_for(pilot, task.done)
            await _wait_for(pilot, lambda: not app._agent_execution_draining())

        await invoke()
        if command == "gigacode":
            assert app._gigacode_enabled
            assert agent.gigacode_enabled
            assert app.session.gigacode_enabled
        else:
            assert app.settings_store.load().active_model == TARGET_MODEL
            assert app.agent is not agent
            assert agent.cleanup_calls == 1
        assert not app._queued_messages


@pytest.mark.parametrize("entry_point", ["ui", "candidate"])
async def test_settings_rebuild_refuses_before_persisting_during_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry_point: str
) -> None:
    app = _configured_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        screen = await open_settings_screen(app, pilot)
        await _wait_for(pilot, lambda: bool(screen.query("#model_select")))
        screen.query_one("#model_select", Select).value = TARGET_MODEL
        await pilot.pause()
        candidate = deepcopy(app.settings)
        candidate.active_model = TARGET_MODEL
        original_settings = deepcopy(app.settings)
        stored_settings = app.settings_store.path.read_bytes()
        status = Mock()
        monkeypatch.setattr(app, "_set_settings_status", status)
        agent = await _start_stuck_turn(app, pilot)
        task = app._agent_execution_task

        async def invoke():
            if entry_point == "ui":
                await app._save_settings_from_ui()
                return None
            return await app._apply_settings_candidate(candidate)

        try:
            task = await _force_stop(app, pilot, agent)
            result = await invoke()
            blocked = "Stop the active turn before applying settings."
            if entry_point == "ui":
                status.assert_called_with(blocked, "warning")
            else:
                assert result == (False, blocked)
            assert app.settings == original_settings
            assert app.settings_store.path.read_bytes() == stored_settings
            assert app.agent is agent
            assert agent.cleanup_calls == 0
            assert app.agent_worker is None
            assert not app._queued_messages
            assert screen.query_one("#model_select", Select).value == TARGET_MODEL
        finally:
            agent.release.set()
            if task is not None:
                await _wait_for(pilot, task.done)
            await _wait_for(pilot, lambda: not app._agent_execution_draining())

        result = await invoke()
        if entry_point == "candidate":
            assert result == (True, "")
        assert app.settings_store.load().active_model == TARGET_MODEL
        assert app.agent is not agent
        assert agent.cleanup_calls == 1


@pytest.mark.parametrize("entry_point", ["command", "keyboard", "direct"])
async def test_permission_changes_wait_for_detached_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry_point: str
) -> None:
    app = _configured_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        agent = await _start_stuck_turn(app, pilot)
        task = app._agent_execution_task
        original_mode = app.permission_mode
        target = PermissionMode.AUTO if original_mode == PermissionMode.ASK else PermissionMode.ASK
        original_settings = deepcopy(app.settings)
        stored_settings = app.settings_store.path.read_bytes()
        set_policy = Mock(wraps=app.session_runtime.set_permission_mode)
        set_callback = Mock()
        set_agent_mode = Mock()
        monkeypatch.setattr(app.session_runtime, "set_permission_mode", set_policy)
        monkeypatch.setattr(agent, "set_permission_callback", set_callback, raising=False)
        monkeypatch.setattr(agent, "set_permission_mode", set_agent_mode, raising=False)

        async def invoke():
            if entry_point == "command":
                await app._command_permissions(target.value)
            elif entry_point == "keyboard":
                await app.action_toggle_permission_mode()
            else:
                await app._set_permission_mode(target)

        try:
            task = await _force_stop(app, pilot, agent)
            await invoke()
            assert app.permission_mode == original_mode
            assert app.session.permission_mode == original_mode.value
            assert app.session_runtime.permission_mode == original_mode
            assert app.settings == original_settings
            assert app.settings_store.path.read_bytes() == stored_settings
            set_policy.assert_not_called()
            set_callback.assert_not_called()
            set_agent_mode.assert_not_called()
        finally:
            agent.release.set()
            if task is not None:
                await _wait_for(pilot, task.done)
            await _wait_for(pilot, lambda: not app._agent_execution_draining())

        await invoke()
        assert app.permission_mode == target
        assert app.session_runtime.permission_mode == target
        assert app.settings_store.load().permission_mode == target.value
        set_policy.assert_called_once_with(target)
        set_callback.assert_called_once()
        set_agent_mode.assert_called_once_with(target)


@pytest.mark.parametrize("reason", ["expired", "iteration-cap", "escape"])
async def test_loop_stop_defers_agent_mutations_until_detached_cleanup_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    app = _configured_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        if app._loop_scheduler_timer is not None:
            app._loop_scheduler_timer.stop()
        agent = await _start_stuck_turn(app, pilot)
        task = app._agent_execution_task
        apply_loop = Mock()
        monkeypatch.setattr(agent, "apply_loop", apply_loop)
        state = LoopState.create(parse_schedule_text("5m"), "check progress")
        try:
            task = await _force_stop(app, pilot, agent)
            if reason == "expired":
                state.expires_at = (now_local() - timedelta(seconds=1)).isoformat()
            elif reason == "iteration-cap":
                state.iterations = state.max_iterations
            app._scheduled_loop = state
            if reason == "escape":
                app.action_cancel_generation()
            else:
                await app._loop_tick()
            apply_loop.assert_not_called()
            assert not state.stopped
        finally:
            agent.release.set()
            if task is not None:
                await _wait_for(pilot, task.done)
            await _wait_for(pilot, lambda: not app._agent_execution_draining())

        if reason == "escape":
            app.action_cancel_generation()
        else:
            await app._loop_tick()
        apply_loop.assert_called_once_with(False)
        assert state.stopped
