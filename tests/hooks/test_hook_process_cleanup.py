import asyncio
import os
import shlex
import signal
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any, Callable

import psutil
import pytest

from kolega_code.hooks import HookCapabilities, HookEvent, HookSpec, LifecycleEvent
from kolega_code.hooks import backends

pytestmark = [pytest.mark.asyncio, pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")]


async def _until(predicate: Callable[[], bool]) -> None:
    async with asyncio.timeout(15):
        while not predicate():
            await asyncio.sleep(0.01)


def _running(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


class HookProcess:
    def __init__(
        self, path: Path, monkeypatch: pytest.MonkeyPatch, *, exit_launcher: bool = False, escape_child: bool = False
    ) -> None:
        self.path = path
        self.proc: asyncio.subprocess.Process | None = None
        self.task: asyncio.Task[Any] | None = None
        self.communication: asyncio.Task[tuple[bytes, bytes]] | None = None
        self.child_pid: int | None = None
        self.ready = asyncio.Event()
        self.communicating = asyncio.Event()
        child = (
            "import os, pathlib, sys, time\n"
            "path = pathlib.Path(sys.argv[1])\n"
            "path.with_suffix('.tmp').write_text(str(os.getpid()))\n"
            "path.with_suffix('.tmp').replace(path)\n"
            "while True: time.sleep(60)\n"
        )
        script = path / "hook.py"
        script.write_text(
            "import pathlib, subprocess, sys, time\n"
            f"subprocess.Popen([sys.executable, '-c', {child!r}, {str(path / 'child-ready')!r}], "
            f"start_new_session={escape_child!r})\n"
            f"while not pathlib.Path({str(path / 'child-ready')!r}).exists(): time.sleep(0.01)\n"
            + ("sys.exit(0)\n" if exit_launcher else "while True: time.sleep(60)\n"),
            encoding="utf-8",
        )
        self.command = shlex.join([sys.executable, str(script)])
        create = asyncio.create_subprocess_exec
        communicate = asyncio.subprocess.Process.communicate

        async def spawn(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
            self.proc = await create(*args, **kwargs)
            await _until(lambda: (path / "child-ready").exists())
            self.child_pid = int((path / "child-ready").read_text())
            if exit_launcher:
                await _until(lambda: self.proc is not None and self.proc.returncode is not None)
            self.ready.set()
            return self.proc

        async def tracked_communicate(proc: asyncio.subprocess.Process, *args: Any) -> tuple[bytes, bytes]:
            task = asyncio.current_task()
            assert task is not None
            self.communication = task
            self.communicating.set()
            return await communicate(proc, *args)

        monkeypatch.setattr(backends.asyncio, "create_subprocess_exec", spawn)
        monkeypatch.setattr(asyncio.subprocess.Process, "communicate", tracked_communicate)

    async def start(self, *, timeout: int = 60) -> None:
        spec = HookSpec(type="command", command=self.command, timeout=timeout, scope="global")
        self.task = asyncio.create_task(
            backends.run_hook(
                LifecycleEvent(name=HookEvent.PRE_TOOL_USE), spec, HookCapabilities(project_path=self.path)
            )
        )
        async with asyncio.timeout(15):
            await self.ready.wait()
            await self.communicating.wait()

    async def assert_stopped(self) -> None:
        assert self.proc is not None
        assert self.child_pid is not None
        pid, child_pid = self.proc.pid, self.child_pid
        await _until(lambda: not _running(pid) and not _running(child_pid))
        assert self.proc.returncode is not None
        assert self.communication is not None and self.communication.done()
        assert not self.communication.cancelled()

    async def close(self) -> None:
        pids: set[int] = set()
        if self.proc is not None:
            pids.add(self.proc.pid)
            descendants: list[psutil.Process] = []
            with suppress(psutil.NoSuchProcess):
                descendants = psutil.Process(self.proc.pid).children(recursive=True)
            with suppress(ProcessLookupError):
                os.killpg(self.proc.pid, signal.SIGKILL)
            with suppress(ProcessLookupError):
                self.proc.kill()
            for descendant in descendants:
                pids.add(descendant.pid)
                with suppress(psutil.NoSuchProcess):
                    descendant.kill()
        ready = self.path / "child-ready"
        if ready.exists():
            child_pid = int(ready.read_text())
            pids.add(child_pid)
            with suppress(ProcessLookupError):
                os.kill(child_pid, signal.SIGKILL)
        if self.task is not None:
            self.task.cancel()
            with suppress(asyncio.CancelledError, backends.HookExecutionError):
                await asyncio.wait_for(self.task, timeout=15)
        if self.proc is not None:
            await asyncio.wait_for(self.proc.wait(), timeout=15)
        await _until(lambda: all(not _running(pid) for pid in pids))


@pytest.mark.parametrize("exit_launcher", [False, True], ids=["running-launcher", "exited-launcher"])
async def test_cancellation_kills_hook_and_descendants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_launcher: bool
) -> None:
    hook = HookProcess(tmp_path, monkeypatch, exit_launcher=exit_launcher)
    try:
        await hook.start()
        assert hook.task is not None
        hook.task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(hook.task, timeout=15)
        await hook.assert_stopped()
    finally:
        await hook.close()


@pytest.mark.parametrize("exit_launcher", [False, True], ids=["running-launcher", "exited-launcher"])
async def test_timeout_kills_hook_and_descendants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_launcher: bool
) -> None:
    hook = HookProcess(tmp_path, monkeypatch, exit_launcher=exit_launcher)
    try:
        await hook.start(timeout=1)
        assert hook.task is not None
        with pytest.raises(backends.HookExecutionError, match="command hook timed out after 1s") as error:
            await asyncio.wait_for(hook.task, timeout=15)
        assert isinstance(error.value.__cause__, asyncio.TimeoutError)
        await hook.assert_stopped()
    finally:
        await hook.close()


@pytest.mark.parametrize("timeout_first", [False, True], ids=["cancellation", "timeout"])
async def test_repeated_cancellation_does_not_interrupt_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, timeout_first: bool
) -> None:
    hook = HookProcess(tmp_path, monkeypatch)
    cleanup_entered = asyncio.Event()
    finish_cleanup = asyncio.Event()
    cleanup = backends._cleanup_command

    async def gated_cleanup(proc: asyncio.subprocess.Process, communication: asyncio.Task[tuple[bytes, bytes]]) -> None:
        cleanup_entered.set()
        await finish_cleanup.wait()
        await cleanup(proc, communication)

    monkeypatch.setattr(backends, "_cleanup_command", gated_cleanup)
    try:
        await hook.start(timeout=1 if timeout_first else 60)
        assert hook.task is not None
        if not timeout_first:
            hook.task.cancel()
        async with asyncio.timeout(15):
            await cleanup_entered.wait()
        for _ in range(3):
            hook.task.cancel()
            await asyncio.sleep(0)
            assert not hook.task.done()
        finish_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(hook.task, timeout=15)
        await hook.assert_stopped()
    finally:
        finish_cleanup.set()
        await hook.close()


@pytest.mark.parametrize("timeout_first", [False, True], ids=["cancellation", "timeout"])
async def test_cleanup_bounds_drain_when_descendant_escapes_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, timeout_first: bool
) -> None:
    hook = HookProcess(tmp_path, monkeypatch, exit_launcher=True, escape_child=True)
    monkeypatch.setattr(backends, "_COMMAND_CLEANUP_TIMEOUT", 0)
    try:
        await hook.start(timeout=1 if timeout_first else 60)
        assert hook.proc is not None and hook.proc.returncode == 0
        assert hook.child_pid is not None and _running(hook.child_pid)
        assert hook.task is not None
        if timeout_first:
            with pytest.raises(backends.HookExecutionError, match="command hook timed out after 1s"):
                await asyncio.wait_for(hook.task, timeout=15)
        else:
            hook.task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(hook.task, timeout=15)
        assert hook.communication is not None and hook.communication.cancelled()
        assert _running(hook.child_pid)
        assert getattr(hook.proc, "_transport").is_closing()
    finally:
        await hook.close()
