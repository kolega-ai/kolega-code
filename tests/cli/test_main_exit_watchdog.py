"""The CLI must not hold the terminal open when shutdown cannot finish."""

from __future__ import annotations

import argparse
import asyncio
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from kolega_code.cli import main as main_module
from kolega_code.cli.main import _arm_exit_watchdog
from kolega_code.cli.session_store import SessionLockedError, SessionStore


def test_exit_watchdog_forces_exit_when_shutdown_is_wedged() -> None:
    exits: list[int] = []
    fired = threading.Event()

    def fake_exit(code: int) -> None:
        exits.append(code)
        fired.set()

    _arm_exit_watchdog(0.01, force_exit=fake_exit)

    assert fired.wait(2.0), "watchdog never fired"
    assert exits == [0]


def test_exit_watchdog_stays_quiet_while_shutdown_completes() -> None:
    exits: list[int] = []
    cancel = _arm_exit_watchdog(30.0, force_exit=exits.append)

    try:
        time.sleep(0.05)

        assert exits == []
    finally:
        cancel.set()


def test_exit_watchdog_can_be_disarmed() -> None:
    """A normal return from ``asyncio.run`` must disarm it.

    ``_run_tui`` is called in-process by tests and embedded hosts; a watchdog that
    outlived that call killed xdist workers seconds later ("node down" runs).
    """
    exits: list[int] = []
    cancel = _arm_exit_watchdog(0.01, force_exit=exits.append)

    cancel.set()

    time.sleep(0.05)
    assert exits == []


@pytest.fixture
def watchdogs(monkeypatch: pytest.MonkeyPatch) -> list[threading.Event]:
    events: list[threading.Event] = []

    def arm() -> threading.Event:
        cancel = threading.Event()
        events.append(cancel)
        return cancel

    monkeypatch.setattr(main_module, "_arm_exit_watchdog", arm)
    return events


class _FakeApp:
    def __init__(self, **kwargs: Any) -> None:
        self.session = kwargs["session"]
        self.resume_session = None
        self._quit_cleanly = False
        self._on_shutdown_requested: Callable[[], None] | None = None

    async def run_async(self) -> None:
        pass

    async def _cleanup_agent_generation(self) -> None:
        pass


def _install_tui_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, app_type: type) -> argparse.Namespace:
    import kolega_code.cli.app as app_module

    args = main_module.parse_args(["tui", str(tmp_path), "--state-dir", str(tmp_path / "state")])
    monkeypatch.setattr(app_module, "KolegaCodeApp", app_type)
    monkeypatch.setattr(main_module, "_select_startup_project", lambda path, _args: path)
    monkeypatch.setattr(main_module, "_resolve_extension_selection_from_args", lambda _args: None)
    monkeypatch.setattr(main_module, "build_agent_config", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(main_module, "config_summary", lambda _config: {})
    monkeypatch.setattr(main_module, "known_secret_values", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(main_module, "write_crash_log", lambda *_args, **_kwargs: None)
    return args


def test_startup_selection_error_preserves_exception_and_releases_locks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, watchdogs: list[threading.Event]
) -> None:
    args = _install_tui_app(monkeypatch, tmp_path, _FakeApp)
    store = SessionStore(args.state_dir)
    other_store = SessionStore(args.state_dir)
    session = store.create(tmp_path, main_module.CLI_AGENT_MODE, {})
    failure = ValueError("startup project selection failed")

    def select(*_args: Any) -> Path:
        raise failure

    monkeypatch.setattr(main_module, "_store_from_args", lambda _args: store)
    monkeypatch.setattr(main_module, "_select_startup_project", select)

    try:
        store.claim_for_resume(session.session_id, tmp_path)
        with pytest.raises(SessionLockedError):
            other_store.claim_for_resume(session.session_id, tmp_path)

        with pytest.raises(ValueError) as caught:
            main_module._run_tui(args)

        assert caught.value is failure
        assert watchdogs == []
        assert other_store.claim_for_resume(session.session_id, tmp_path).session_id == session.session_id
    finally:
        store.release_session_locks()
        other_store.release_session_locks()


@pytest.mark.parametrize("stage", ["startup", "session", "generation", "usage"])
def test_watchdog_is_armed_before_awaiting_blocked_teardown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, watchdogs: list[threading.Event], stage: str
) -> None:
    blocked_stages: list[str] = []

    async def block_teardown() -> None:
        assert len(watchdogs) == 1
        assert not watchdogs[0].is_set()
        blocked_stages.append(stage)
        release = asyncio.Event()
        asyncio.get_running_loop().call_soon(release.set)
        await release.wait()
        assert not watchdogs[0].is_set()

    class App(_FakeApp):
        async def run_async(self) -> None:
            assert watchdogs == []
            if stage == "session":
                self._session_shutdown_task = asyncio.create_task(block_teardown())
            if stage == "usage":
                self._usage_sink = SimpleNamespace(aclose=block_teardown)

        async def _cancel_startup(self) -> None:
            if stage == "startup":
                await block_teardown()

        async def _cleanup_agent_generation(self) -> None:
            if stage == "generation":
                await block_teardown()

    args = _install_tui_app(monkeypatch, tmp_path, App)

    assert main_module._run_tui(args) == 0
    assert blocked_stages == [stage]
    assert watchdogs[0].is_set()


def test_shutdown_request_arms_watchdog_before_run_async_returns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, watchdogs: list[threading.Event]
) -> None:
    stages: list[str] = []

    class App(_FakeApp):
        async def run_async(self) -> None:
            assert watchdogs == []
            assert self._on_shutdown_requested is not None
            self._on_shutdown_requested()
            self._on_shutdown_requested()
            assert len(watchdogs) == 1
            assert not watchdogs[0].is_set()
            stages.append("shutdown requested")
            release = asyncio.Event()
            asyncio.get_running_loop().call_soon(release.set)
            await release.wait()
            assert len(watchdogs) == 1
            assert not watchdogs[0].is_set()
            stages.append("run returned")
            self._quit_cleanly = True

        async def _cleanup_agent_generation(self) -> None:
            assert len(watchdogs) == 1
            assert not watchdogs[0].is_set()
            stages.append("cleanup")

    args = _install_tui_app(monkeypatch, tmp_path, App)

    assert main_module._run_tui(args) == 0
    assert stages == ["shutdown requested", "run returned", "cleanup"]
    assert len(watchdogs) == 1
    assert watchdogs[0].is_set()


def test_shutdown_requested_watchdog_covers_default_executor_shutdown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, watchdogs: list[threading.Event]
) -> None:
    release = threading.Event()
    armed_during_executor_shutdown: list[bool] = []

    def worker() -> None:
        if release.wait(2.0):
            armed_during_executor_shutdown.append(len(watchdogs) == 1 and not watchdogs[0].is_set())

    class App(_FakeApp):
        async def run_async(self) -> None:
            assert self._on_shutdown_requested is not None
            self._on_shutdown_requested()
            loop = asyncio.get_running_loop()
            shutdown_executor = loop.shutdown_default_executor

            async def shutdown(*args: Any, **kwargs: Any) -> None:
                try:
                    assert len(watchdogs) == 1
                    assert not watchdogs[0].is_set()
                finally:
                    release.set()
                await shutdown_executor(*args, **kwargs)

            monkeypatch.setattr(loop, "shutdown_default_executor", shutdown)
            loop.run_in_executor(None, worker)

    args = _install_tui_app(monkeypatch, tmp_path, App)

    assert main_module._run_tui(args) == 0
    assert armed_during_executor_shutdown == [True]
    assert len(watchdogs) == 1
    assert watchdogs[0].is_set()


@pytest.mark.parametrize("shutdown_requested", [False, True])
@pytest.mark.parametrize("failure_stage", [None, "run", "teardown"])
def test_watchdog_stays_armed_through_asyncio_run_then_is_disarmed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    watchdogs: list[threading.Event],
    failure_stage: str | None,
    shutdown_requested: bool,
) -> None:
    failure = RuntimeError("app failed")
    real_run = asyncio.run

    class App(_FakeApp):
        async def run_async(self) -> None:
            assert watchdogs == []
            if shutdown_requested:
                assert self._on_shutdown_requested is not None
                self._on_shutdown_requested()
                assert len(watchdogs) == 1
                assert not watchdogs[0].is_set()
            if failure_stage == "run":
                raise failure

        async def _cleanup_agent_generation(self) -> None:
            assert len(watchdogs) == 1
            assert not watchdogs[0].is_set()
            if failure_stage == "teardown":
                raise failure

    def run(coro: Any) -> Any:
        try:
            return real_run(coro)
        finally:
            assert len(watchdogs) == 1
            assert not watchdogs[0].is_set()

    args = _install_tui_app(monkeypatch, tmp_path, App)
    monkeypatch.setattr(main_module.asyncio, "run", run)

    if failure_stage is None:
        assert main_module._run_tui(args) == 0
    else:
        with pytest.raises(RuntimeError) as caught:
            main_module._run_tui(args)
        assert caught.value is failure
    assert watchdogs[0].is_set()


@pytest.mark.parametrize("failure_stage", [None, "run", "teardown"])
def test_session_relaunch_disarms_previous_watchdog_before_running_app(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    watchdogs: list[threading.Event],
    failure_stage: str | None,
) -> None:
    store = SessionStore(tmp_path / "state")
    selected = store.create(tmp_path, main_module.CLI_AGENT_MODE, {})
    runs: list[str] = []
    shutdown_callbacks: list[Callable[[], None]] = []
    failure = RuntimeError("relaunched app failed")

    class App(_FakeApp):
        async def run_async(self) -> None:
            assert all(cancel.is_set() for cancel in watchdogs)
            assert len(watchdogs) == len(runs)
            runs.append(self.session.session_id)
            assert self._on_shutdown_requested is not None
            self._on_shutdown_requested()
            shutdown_callbacks.append(self._on_shutdown_requested)
            for callback in shutdown_callbacks:
                callback()
            assert len(watchdogs) == len(runs)
            assert not watchdogs[-1].is_set()
            if len(runs) == 2 and failure_stage == "run":
                raise failure
            self._quit_cleanly = True
            if len(runs) == 1:
                self.resume_session = selected

        async def _cleanup_agent_generation(self) -> None:
            assert len(watchdogs) == len(runs)
            assert not watchdogs[-1].is_set()
            if len(runs) == 2 and failure_stage == "teardown":
                raise failure

    args = _install_tui_app(monkeypatch, tmp_path, App)
    monkeypatch.setattr(main_module, "_store_from_args", lambda _args: store)

    if failure_stage is None:
        assert main_module._run_tui(args) == 0
    else:
        with pytest.raises(RuntimeError) as caught:
            main_module._run_tui(args)
        assert caught.value is failure
    assert len(runs) == 2
    assert runs[-1] == selected.session_id
    assert len(watchdogs) == 2
    assert all(cancel.is_set() for cancel in watchdogs)
