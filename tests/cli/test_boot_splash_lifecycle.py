"""Startup must paint before waiting for tools, and remain safely interruptible."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from textual.app import App
from textual.pilot import Pilot

from kolega_code.cli.tui.boot_splash import BootSplash
from kolega_code.cli.tui.widgets import ActionList, ChatComposer
from kolega_code.extensions import KolegaExtensionLoadError
from kolega_code.permissions import permission_request_for_tool
from kolega_code.session.runtime import serialize_permission_request

from ._app_test_utils import _FakeToolCollection, _build_sub_agent_test_app, wait_for_onboarding_screen


pytestmark = pytest.mark.usefixtures("isolated_cli_env")


async def _wait_for_layout(pilot: Pilot, predicate: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline:
        await pilot.pause(0.01)
        if predicate():
            return
    raise AssertionError("startup layout did not settle")


@pytest.mark.asyncio
async def test_splash_is_painted_before_agent_initialization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    painted = asyncio.Event()
    entered = asyncio.Event()
    release = asyncio.Event()
    build = app._build_agent

    def after_display() -> None:
        if app._boot_splash.size.width and not app._batch_count:
            painted.set()

    async def slow_build(*args: Any, **kwargs: Any) -> None:
        assert painted.is_set(), "Agent initialization still blocks the first paint"
        entered.set()
        await release.wait()
        await build(*args, **kwargs)

    monkeypatch.setattr(app, "post_display_hook", after_display)
    monkeypatch.setattr(app, "_build_agent", slow_build)
    # Deliberately use Textual's raw lifecycle, not a ready-app test helper.
    async with App.run_test(app, size=(80, 24)) as pilot:
        await asyncio.wait_for(entered.wait(), 6)
        splash = app.query_one(BootSplash)
        assert splash.display
        await _wait_for_layout(pilot, lambda: splash.size == app.screen.size)
        assert splash.size == app.screen.size
        assert app.query_one("#composer", ChatComposer).disabled
        assert app.agent is None
        # Hidden controls must not launch a competing generation.
        await pilot.press("tab", "enter", "shift+tab", "ctrl+p")
        app.action_open_settings()
        assert app._settings_screen is None
        assert len(app.screen_stack) == 1
        assert app._startup_body is not None and app._startup_body.disabled
        release.set()
        await asyncio.wait_for(app._startup_complete.wait(), 6)
        await pilot.pause()
        assert not splash.display
        assert splash._timer is None
        assert app.agent is not None
        assert not app.query_one("#composer", ChatComposer).disabled
        assert not app._startup_body.disabled


@pytest.mark.asyncio
@pytest.mark.parametrize("resuming", [False, True])
async def test_stages_follow_real_tool_and_hook_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resuming: bool
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    if resuming:
        app.session.history = [{"role": "user", "content": "Previous conversation"}]
    stages: list[str] = []
    set_stage = app._set_startup_stage
    tools_entered, tools_release = asyncio.Event(), asyncio.Event()
    hooks_entered, hooks_release = asyncio.Event(), asyncio.Event()

    def record_stage(stage: str) -> None:
        stages.append(stage)
        set_stage(stage)

    async def initialize(_self: _FakeToolCollection) -> list[Any]:
        tools_entered.set()
        await tools_release.wait()
        return []

    async def hook() -> None:
        hooks_entered.set()
        await hooks_release.wait()

    monkeypatch.setattr(app, "_set_startup_stage", record_stage)
    monkeypatch.setattr(_FakeToolCollection, "initialize", initialize)
    monkeypatch.setattr(app, "_fire_session_start_once", hook)
    # A frozen animation clock must not prevent or delay handoff.
    monkeypatch.setattr(app._boot_splash, "_now", lambda: 100.0)
    async with App.run_test(app, size=(80, 24)):
        await asyncio.wait_for(tools_entered.wait(), 6)
        assert stages == ["Preparing workspace…", "Loading skills and agents…", "Initializing tools…"]
        assert not app._startup_complete.is_set()
        assert app._boot_splash.display
        tools_release.set()
        await asyncio.wait_for(hooks_entered.wait(), 6)
        assert stages[3:] == (["Restoring session…"] if resuming else []) + ["Running startup hooks…"]
        assert app._boot_splash.display
        assert not app._maybe_start_queued_message()
        hooks_release.set()
        await asyncio.wait_for(app._startup_complete.wait(), 6)
        assert not app._boot_splash.display
        assert app._boot_splash._timer is None
        assert not app._startup_pending
        assert app._startup_task is not None and app._startup_task.done()


@pytest.mark.asyncio
@pytest.mark.parametrize("queued", [False, True])
async def test_startup_schedules_only_waiting_messages_after_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, queued: bool
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    entered, release = asyncio.Event(), asyncio.Event()
    schedule = Mock(wraps=app._schedule_maybe_start_queued_message)

    async def hook() -> None:
        entered.set()
        await release.wait()

    monkeypatch.setattr(app, "_fire_session_start_once", hook)
    monkeypatch.setattr(app, "_schedule_maybe_start_queued_message", schedule)
    async with App.run_test(app) as pilot:
        await asyncio.wait_for(entered.wait(), 6)
        assert app.agent is not None
        sent: list[str] = getattr(app.agent, "messages")
        if queued:
            app._queue_user_message("Waiting for startup")
            app._schedule_maybe_start_queued_message()
            # An existing generation is not usable until its startup hooks finish.
            assert not app._maybe_start_queued_message()
            assert len(app._queued_messages) == 1
            assert sent == []
        release.set()
        await asyncio.wait_for(app._startup_complete.wait(), 6)
        if queued:
            await _wait_for_layout(pilot, lambda: sent == ["Waiting for startup"] and app.agent_worker is None)
            assert app._queued_messages == []
            assert schedule.call_count >= 2
        else:
            schedule.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["question", "custom-answer", "permission"])
async def test_startup_hook_prompt_replaces_splash_and_accepts_keyboard_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    results: list[Any] = []

    async def hook() -> None:
        if kind == "permission":
            request = permission_request_for_tool("exec_command", {"command": "echo startup"})
            assert request is not None
            results.append(
                await app.control_channel.request(
                    "permission",
                    {"request": serialize_permission_request(request), "rule_options": []},
                    default={"allowed": False},
                )
            )
        else:
            results.append(await app._ask_user_choice("Prepare this workspace?", ["Continue", "Skip"]))

    monkeypatch.setattr(app, "_fire_session_start_once", hook)
    async with App.run_test(app, size=(80, 24)) as pilot:
        actions = app.query_one("#approval_actions" if kind == "permission" else "#question_actions", ActionList)
        await _wait_for_layout(
            pilot, lambda: actions.display and actions.option_count > 0 and app.screen.focused is actions
        )
        assert app._startup_pending
        assert not app._boot_splash.display
        assert app._boot_splash._timer is None
        composer = app.query_one("#composer", ChatComposer)
        assert composer.disabled == (kind == "permission")
        if kind == "custom-answer":
            composer.focus()
            composer.load_text("Use the local configuration")
        await pilot.press("enter")
        await asyncio.wait_for(app._startup_complete.wait(), 6)
        if kind == "permission":
            assert results[0]["allowed"] is True
        else:
            assert results == ["Use the local configuration" if kind == "custom-answer" else "Continue"]
        assert not composer.disabled
        assert not app._boot_splash.display


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["workspace", "tools", "hooks"])
async def test_startup_errors_use_normal_textual_exception_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    error = RuntimeError(f"{phase} startup failed")

    async def fail(*_args: Any) -> None:
        raise error

    if phase == "workspace":
        monkeypatch.setattr(app, "_start_inbox_socket", fail)
    elif phase == "tools":
        monkeypatch.setattr(_FakeToolCollection, "initialize", fail)
    else:
        monkeypatch.setattr(app, "_fire_session_start_once", fail)
    with pytest.raises(RuntimeError, match=f"{phase} startup failed") as caught:
        async with App.run_test(app):
            await asyncio.wait_for(app._startup_complete.wait(), 6)
            assert not app._boot_splash.display
            assert app._boot_splash._timer is None
            assert app.agent is None
    assert caught.value is error


@pytest.mark.asyncio
async def test_unloadable_extension_still_exits_with_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async def fail(*_args: Any) -> None:
        raise KolegaExtensionLoadError("Requested extension is unavailable")

    monkeypatch.setattr(app, "_build_agent", fail)
    async with App.run_test(app):
        await asyncio.wait_for(app._startup_complete.wait(), 6)
        assert app.return_code == 1
        assert not app._boot_splash.display
    assert "Requested extension is unavailable" in capsys.readouterr().err


@pytest.mark.asyncio
@pytest.mark.parametrize("onboarding", [False, True])
async def test_onboarding_and_discovery_are_not_covered_by_splash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, onboarding: bool
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    settings = app.settings_store.load()
    settings.discovery_tips = True
    app.settings_store.save(settings)
    if onboarding:
        app.config = None
    async with App.run_test(app, size=(100, 36)) as pilot:
        await asyncio.wait_for(app._startup_complete.wait(), 6)
        if onboarding:
            await wait_for_onboarding_screen(app, pilot)
            assert app.screen is app._onboarding_screen
            assert app._discovery_tip_screen is None
        else:
            await _wait_for_layout(pilot, lambda: app._discovery_tip_screen is not None)
            assert app.screen is app._discovery_tip_screen
        assert not app._boot_splash.display
        assert app._boot_splash._timer is None


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(35, 11), (24, 5), (12, 2), (1, 1)])
async def test_production_splash_and_handoff_survive_tiny_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, size: tuple[int, int]
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    entered, release = asyncio.Event(), asyncio.Event()

    async def initialize(_self: _FakeToolCollection) -> list[Any]:
        entered.set()
        await release.wait()
        return []

    monkeypatch.setattr(_FakeToolCollection, "initialize", initialize)
    async with App.run_test(app, size=size) as pilot:
        await asyncio.wait_for(entered.wait(), 6)
        await _wait_for_layout(pilot, lambda: app._boot_splash.region == app.screen.region)
        assert app._boot_splash.display
        assert app._boot_splash.size == app.screen.size
        release.set()
        await asyncio.wait_for(app._startup_complete.wait(), 6)
        await pilot.pause()
        assert not app._boot_splash.display
        assert app.agent is not None
        assert app._exception is None


@pytest.mark.asyncio
async def test_cancellation_before_startup_task_launch_settles_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    monkeypatch.setattr(app, "_launch_startup", lambda: None)
    async with App.run_test(app):
        assert app._startup_task is None
        await app._cancel_startup()
        assert app._startup_complete.is_set()
        assert not app._startup_pending
        assert app._startup_task is None
        assert not app._boot_splash.display
        assert app._boot_splash._timer is None
