import asyncio
from types import SimpleNamespace

import pytest

from kolega_code.sandbox.terminal import SandboxTerminalManager


class FakeCommandHandle:
    def __init__(self, pid: int = 123, exit_code: int = 0):
        self.pid = pid
        self._done = asyncio.Event()
        self._exit_code = exit_code
        self.killed = False

    async def wait(self):
        await self._done.wait()
        return SimpleNamespace(exit_code=self._exit_code)

    def complete(self) -> None:
        self._done.set()

    async def kill(self) -> None:
        self.killed = True
        self._exit_code = 137
        self._done.set()


class FakeCommands:
    def __init__(self):
        self.handle = FakeCommandHandle()
        self.run_calls = []
        self.send_stdin_calls = []
        self.started = asyncio.Event()

    async def run(self, command: str, **kwargs):
        if command.startswith("test -d") or command.startswith("mkdir"):
            return SimpleNamespace(exit_code=0)

        self.run_calls.append((command, kwargs))
        self.started.set()
        on_stdout = kwargs.get("on_stdout")
        if on_stdout:
            await on_stdout("line1\n")
        return self.handle

    async def send_stdin(self, pid: int, data: str):
        self.send_stdin_calls.append((pid, data))


class FakeSandbox:
    def __init__(self):
        self.commands = FakeCommands()


@pytest.mark.asyncio
async def test_exec_command_runs_background_with_stdin():
    sandbox = FakeSandbox()
    manager = SandboxTerminalManager(sandbox, "workspace", "thread")

    result = await manager.exec_command("python prompt.py", yield_time_ms=300)

    assert sandbox.commands.run_calls
    _, kwargs = sandbox.commands.run_calls[0]
    assert kwargs["background"] is True
    assert kwargs["stdin"] is True
    # Still waiting on input -> reported as a running session.
    assert result.status == "running"
    assert result.session_id is not None
    assert "line1" in result.output


@pytest.mark.asyncio
async def test_write_stdin_sends_raw_input_and_reports_exit():
    sandbox = FakeSandbox()
    manager = SandboxTerminalManager(sandbox, "workspace", "thread")

    result = await manager.exec_command("python prompt.py", yield_time_ms=300)
    session_id = result.session_id

    running = await manager.write_stdin(session_id, "Ada\n", yield_time_ms=200)
    assert (123, "Ada\n") in sandbox.commands.send_stdin_calls
    assert running.status == "running"

    sandbox.commands.handle.complete()
    final = await manager.write_stdin(session_id, "", yield_time_ms=2000)
    assert final.status == "exited"
    assert final.exit_code == 0


@pytest.mark.asyncio
async def test_kill_session_kills_handle():
    sandbox = FakeSandbox()
    manager = SandboxTerminalManager(sandbox, "workspace", "thread")

    result = await manager.exec_command("sleep 100", yield_time_ms=200)
    assert result.status == "running"

    killed = await manager.kill_session(result.session_id, "TERM")
    assert sandbox.commands.handle.killed is True
    assert killed.status == "exited"


@pytest.mark.asyncio
async def test_kill_session_interrupt_sends_ctrl_c():
    sandbox = FakeSandbox()
    manager = SandboxTerminalManager(sandbox, "workspace", "thread")

    result = await manager.exec_command("sleep 100", yield_time_ms=200)
    await manager.kill_session(result.session_id, "INT")
    assert (123, "\x03") in sandbox.commands.send_stdin_calls


@pytest.mark.asyncio
async def test_list_sessions_reports_running():
    sandbox = FakeSandbox()
    manager = SandboxTerminalManager(sandbox, "workspace", "thread")

    result = await manager.exec_command("sleep 100", yield_time_ms=200)
    sessions = await manager.list_sessions()
    assert result.session_id in sessions
    assert sessions[result.session_id]["running"] is True


@pytest.mark.asyncio
async def test_write_stdin_unknown_session_raises():
    manager = SandboxTerminalManager(FakeSandbox(), "workspace", "thread")
    with pytest.raises(KeyError):
        await manager.write_stdin("nope", "x")
