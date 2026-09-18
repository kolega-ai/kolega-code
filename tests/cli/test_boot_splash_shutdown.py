"""Startup cancellation and shutdown regressions for the boot splash path."""

from __future__ import annotations

import asyncio
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import pytest
from textual.app import App

from kolega_code.extensions import ExtensionSelection, KolegaExtensionBundle, resolve_extension_selection

from ._app_test_utils import FakeCoderAgent, _build_sub_agent_test_app, install_fake_agents


pytestmark = pytest.mark.usefixtures("isolated_cli_env")


class _BlockingToolCollection:
    lsp_manager = None

    def __init__(self, started: asyncio.Event, release: asyncio.Event, events: list[str]) -> None:
        self._started = started
        self._release = release
        self._events = events

    async def initialize(self) -> list[Any]:
        self._events.append("initialize-start")
        self._started.set()
        await self._release.wait()
        self._events.append("initialize-end")
        return []


class _BlockingStartupAgent(FakeCoderAgent):
    initialize_started: ClassVar[asyncio.Event]
    initialize_release: ClassVar[asyncio.Event]
    cleanup_started: ClassVar[asyncio.Event]
    cleanup_release: ClassVar[asyncio.Event]
    cleanup_calls: ClassVar[int]
    instances: ClassVar[list["_BlockingStartupAgent"]]
    events: ClassVar[list[str]]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        type(self).instances.append(self)
        self.tool_collection = _BlockingToolCollection(
            type(self).initialize_started,
            type(self).initialize_release,
            type(self).events,
        )

    async def cleanup(self) -> None:
        type(self).cleanup_calls += 1
        type(self).events.append("agent-cleanup-start")
        type(self).cleanup_started.set()
        await type(self).cleanup_release.wait()
        type(self).events.append("agent-cleanup-end")


class _ExtensionRecorder:
    def __init__(self) -> None:
        self.factory_calls = 0
        self.cleanup_calls = 0
        self.events: list[str] = []

    def cleanup(self) -> None:
        self.cleanup_calls += 1
        self.events.append("bundle-cleanup")


def _reset_blocking_agent(*, cleanup_released: bool = True) -> None:
    _BlockingStartupAgent.initialize_started = asyncio.Event()
    _BlockingStartupAgent.initialize_release = asyncio.Event()
    _BlockingStartupAgent.cleanup_started = asyncio.Event()
    _BlockingStartupAgent.cleanup_release = asyncio.Event()
    if cleanup_released:
        _BlockingStartupAgent.cleanup_release.set()
    _BlockingStartupAgent.cleanup_calls = 0
    _BlockingStartupAgent.instances = []
    _BlockingStartupAgent.events = []


def _install_extension(monkeypatch: pytest.MonkeyPatch, recorder: _ExtensionRecorder) -> ExtensionSelection:
    module_name = f"test_boot_splash_shutdown_ext_{id(recorder)}"

    def create_extension(_host: Any, _config_path: Path | None) -> KolegaExtensionBundle:
        recorder.factory_calls += 1
        return KolegaExtensionBundle(cleanup=recorder.cleanup)

    module = types.ModuleType(module_name)
    module.create_extension = create_extension  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, module)
    return resolve_extension_selection(f"{module_name}:create_extension", None)


def _build_blocked_startup_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    extension_selection: ExtensionSelection | None = None,
) -> Any:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch, extension_selection=extension_selection)
    install_fake_agents(monkeypatch, coder_cls=_BlockingStartupAgent)
    return app


async def _assert_startup_cancelled(app: Any) -> None:
    await asyncio.wait_for(app._startup_complete.wait(), timeout=6)
    assert app._startup_task is not None
    assert app._startup_task.done()
    assert not app._startup_pending
    assert app._startup_complete.is_set()
    assert app.agent is None
    assert app._extension_bundle is None
    assert app._boot_splash._timer is None
    assert not app._boot_splash.display


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["ctrl+q", "ctrl+c", "escape"])
async def test_startup_key_cancellation_cleans_partial_generation_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    _reset_blocking_agent()
    recorder = _ExtensionRecorder()
    app = _build_blocked_startup_app(
        tmp_path,
        monkeypatch,
        extension_selection=_install_extension(monkeypatch, recorder),
    )

    async with App.run_test(app, size=(80, 24)) as pilot:
        await asyncio.wait_for(_BlockingStartupAgent.initialize_started.wait(), timeout=6)

        await pilot.press(key)
        await _assert_startup_cancelled(app)

    assert _BlockingStartupAgent.events == ["initialize-start", "agent-cleanup-start", "agent-cleanup-end"]
    assert _BlockingStartupAgent.cleanup_calls == 1
    assert len(_BlockingStartupAgent.instances) == 1
    assert _BlockingStartupAgent.instances[0].cleanup_calls == 0
    assert recorder.factory_calls == 1
    assert recorder.cleanup_calls == 1
    assert recorder.events == ["bundle-cleanup"]


@pytest.mark.asyncio
async def test_repeated_startup_cancellation_does_not_interrupt_cleanup_in_flight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_blocking_agent(cleanup_released=False)
    recorder = _ExtensionRecorder()
    app = _build_blocked_startup_app(
        tmp_path,
        monkeypatch,
        extension_selection=_install_extension(monkeypatch, recorder),
    )

    async with App.run_test(app, size=(80, 24)):
        await asyncio.wait_for(_BlockingStartupAgent.initialize_started.wait(), timeout=6)

        first_cancel = asyncio.create_task(app._cancel_startup())
        await asyncio.wait_for(_BlockingStartupAgent.cleanup_started.wait(), timeout=6)
        second_cancel = asyncio.create_task(app._cancel_startup())
        third_cancel = asyncio.create_task(app._cancel_startup())

        await asyncio.sleep(0)
        assert _BlockingStartupAgent.events == ["initialize-start", "agent-cleanup-start"]
        assert _BlockingStartupAgent.cleanup_calls == 1
        assert recorder.cleanup_calls == 0
        assert app.agent is None
        assert app._extension_bundle is None

        _BlockingStartupAgent.cleanup_release.set()
        await asyncio.wait_for(asyncio.gather(first_cancel, second_cancel, third_cancel), timeout=6)
        await _assert_startup_cancelled(app)

    assert _BlockingStartupAgent.events == ["initialize-start", "agent-cleanup-start", "agent-cleanup-end"]
    assert _BlockingStartupAgent.cleanup_calls == 1
    assert recorder.cleanup_calls == 1
    assert recorder.events == ["bundle-cleanup"]


@pytest.mark.asyncio
async def test_unmount_awaits_startup_cleanup_before_memory_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_blocking_agent(cleanup_released=False)
    app = _build_blocked_startup_app(tmp_path, monkeypatch)
    events: list[str] = []
    original_close_memory_manager: Callable[[], None] = app._close_memory_manager

    def close_memory_manager() -> None:
        events.append("memory-close")
        original_close_memory_manager()

    monkeypatch.setattr(app, "_close_memory_manager", close_memory_manager)

    async with App.run_test(app, size=(80, 24)):
        await asyncio.wait_for(_BlockingStartupAgent.initialize_started.wait(), timeout=6)

        unmount = asyncio.create_task(app.on_unmount())
        await asyncio.wait_for(_BlockingStartupAgent.cleanup_started.wait(), timeout=6)
        assert events == []

        _BlockingStartupAgent.cleanup_release.set()
        await asyncio.wait_for(unmount, timeout=6)

        assert _BlockingStartupAgent.events == ["initialize-start", "agent-cleanup-start", "agent-cleanup-end"]
        assert events == ["memory-close"]
        assert _BlockingStartupAgent.cleanup_calls == 1
        assert app._startup_complete.is_set()
