import asyncio
import base64
import shlex
from types import SimpleNamespace

import pytest

from kolega_code.sandbox.terminal import BoundedTerminalOutputs, SandboxTerminalManager
from kolega_code.services.base import TerminalCommandCancelled


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
    def __init__(self, stdout: str = "line1\n"):
        self.handle = FakeCommandHandle()
        self.run_calls = []
        self.send_stdin_calls = []
        self.started = asyncio.Event()
        self.stdout = stdout
        self.remote_files = {}

    async def run(self, command: str, **kwargs):
        if command.startswith("test -d") or command.startswith("mkdir"):
            if "&& : >" in command:
                remote_path = shlex.split(command.split("&& : >", 1)[1].strip())[0]
                self.remote_files[remote_path] = bytearray()
            return SimpleNamespace(exit_code=0)
        if command.startswith("printf %s ") and " | base64 -d >> " in command:
            encoded_part, path_part = command.removeprefix("printf %s ").split(
                " | base64 -d >> ",
                1,
            )
            encoded = shlex.split(encoded_part)[0]
            remote_path = shlex.split(path_part)[0]
            self.remote_files.setdefault(remote_path, bytearray()).extend(
                base64.b64decode(encoded),
            )
            return SimpleNamespace(exit_code=0)
        if command.startswith("chmod 0400 "):
            return SimpleNamespace(exit_code=0)

        self.run_calls.append((command, kwargs))
        self.started.set()
        on_stdout = kwargs.get("on_stdout")
        if on_stdout:
            await on_stdout(self.stdout)
        return self.handle

    async def send_stdin(self, pid: int, data: str):
        self.send_stdin_calls.append((pid, data))


class FakeSandbox:
    def __init__(self, stdout: str = "line1\n"):
        self.commands = FakeCommands(stdout)


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
async def test_exec_command_forwards_env_to_sandbox():
    sandbox = FakeSandbox()
    manager = SandboxTerminalManager(sandbox, "workspace", "thread")

    await manager.exec_command("echo $FOO", env={"FOO": "bar"}, yield_time_ms=200)

    assert sandbox.commands.run_calls
    _, kwargs = sandbox.commands.run_calls[0]
    assert kwargs.get("envs") == {"FOO": "bar"}


@pytest.mark.asyncio
async def test_write_stdin_sends_raw_input_and_reports_exit():
    sandbox = FakeSandbox()
    manager = SandboxTerminalManager(sandbox, "workspace", "thread")

    result = await manager.exec_command("python prompt.py", yield_time_ms=300)
    session_id = result.session_id
    assert session_id is not None

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
    assert result.session_id is not None

    killed = await manager.kill_session(result.session_id, "TERM")
    assert sandbox.commands.handle.killed is True
    assert killed.status == "exited"


@pytest.mark.asyncio
async def test_kill_session_interrupt_sends_ctrl_c():
    sandbox = FakeSandbox()
    manager = SandboxTerminalManager(sandbox, "workspace", "thread")

    result = await manager.exec_command("sleep 100", yield_time_ms=200)
    assert result.session_id is not None
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


@pytest.mark.asyncio
async def test_sandbox_spill_is_mirrored_to_recoverable_sandbox_path(tmp_path):
    complete = "BEGIN\n" + ("line-" + "x" * 90 + "\n") * 700 + "END\n"
    sandbox = FakeSandbox(complete)
    manager = SandboxTerminalManager(sandbox, "workspace", "thread")
    manager.configure_output_spill(tmp_path / "terminal-output")

    result = await manager.exec_command(
        "produce lots of output",
        yield_time_ms=300,
        max_output_tokens=1_000_000,
    )

    assert result.status == "running"
    assert result.spill_path is not None
    assert result.spill_path.startswith("/home/user/.kolega/terminal-output/")
    assert bytes(sandbox.commands.remote_files[result.spill_path]).decode("utf-8") == complete
    assert result.spill_bytes == len(complete.encode("utf-8"))
    assert "BEGIN" in result.output
    assert "END" in result.output
    assert len(result.output) <= 40_000

    sandbox.commands.handle.complete()
    assert result.session_id is not None
    final = await manager.write_stdin(result.session_id, "", yield_time_ms=5000)
    assert final.status == "exited"


@pytest.mark.asyncio
async def test_cancelled_sandbox_exec_returns_recoverable_spill_result(tmp_path):
    complete = "BEGIN\n" + ("line-" + "x" * 90 + "\n") * 700 + "END\n"
    sandbox = FakeSandbox(complete)
    manager = SandboxTerminalManager(sandbox, "workspace", "thread")
    manager.configure_output_spill(tmp_path / "terminal-output")

    task = asyncio.create_task(
        manager.exec_command("produce forever", yield_time_ms=30_000),
    )
    await sandbox.commands.started.wait()
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(TerminalCommandCancelled) as cancelled:
        await task

    result = cancelled.value.result
    assert result.status == "cancelled"
    assert result.exit_code == 137
    assert result.spill_path is not None
    assert bytes(sandbox.commands.remote_files[result.spill_path]).decode("utf-8") == complete
    assert result.spill_bytes == len(complete.encode("utf-8"))
    assert sandbox.commands.handle.killed is True
    assert await manager.list_sessions() == {}


def test_sandbox_terminal_history_is_bounded():
    outputs = BoundedTerminalOutputs()
    outputs.append({"type": "stdout", "data": "a" * 600_000})
    outputs.append({"type": "stdout", "data": "b" * 600_000})

    assert sum(len(item["data"].encode("utf-8")) for item in outputs) <= outputs.MAX_BYTES
    assert outputs[-1]["data"].startswith("b")
