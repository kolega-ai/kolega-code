import asyncio
from types import SimpleNamespace

import pytest

from kolega_code.sandbox.terminal import SandboxTerminalManager


class FakeCommandHandle:
    def __init__(self, pid: int = 123):
        self.pid = pid
        self._done = asyncio.Event()

    async def wait(self):
        await self._done.wait()
        return SimpleNamespace(exit_code=0)

    def complete(self) -> None:
        self._done.set()


class FakeCommands:
    def __init__(self):
        self.handle = FakeCommandHandle()
        self.run_calls = []
        self.send_stdin_calls = []
        self.started = asyncio.Event()

    async def run(self, command: str, **kwargs):
        if command.startswith("test -d"):
            return SimpleNamespace(exit_code=0)

        self.run_calls.append((command, kwargs))
        self.started.set()
        return self.handle

    async def send_stdin(self, pid: int, data: str):
        self.send_stdin_calls.append((pid, data))


class FakeSandbox:
    def __init__(self):
        self.commands = FakeCommands()


@pytest.mark.asyncio
async def test_sandbox_terminal_input_uses_active_command_stdin():
    sandbox = FakeSandbox()
    manager = SandboxTerminalManager(sandbox, "workspace", "thread")
    terminal_id = await manager.launch_terminal()

    command_id = await manager.send_command_tracked(terminal_id, "python prompt.py", "Prompt test")
    await asyncio.wait_for(sandbox.commands.started.wait(), timeout=1)

    assert sandbox.commands.run_calls
    _, kwargs = sandbox.commands.run_calls[0]
    assert kwargs["background"] is True
    assert kwargs["stdin"] is True
    assert manager.command_history[command_id]["pid"] == 123

    result = await manager.send_input(terminal_id, "Ada")

    assert result is True
    assert sandbox.commands.send_stdin_calls == [(123, "Ada\n")]
    assert "Ada" not in manager.read_output(terminal_id, num_chars=1000)

    sandbox.commands.handle.complete()
    for _ in range(10):
        if manager.get_command_status(terminal_id, command_id)["status"] == "completed":
            break
        await asyncio.sleep(0.01)

    assert manager.get_command_status(terminal_id, command_id)["status"] == "completed"


@pytest.mark.asyncio
async def test_sandbox_terminal_input_requires_command_id_when_ambiguous():
    manager = SandboxTerminalManager(FakeSandbox(), "workspace", "thread")
    terminal_id = await manager.launch_terminal()
    first_command = {
        "command": "python prompt_one.py",
        "terminal_id": terminal_id,
        "status": "running",
        "pid": 101,
    }
    second_command = {
        "command": "python prompt_two.py",
        "terminal_id": terminal_id,
        "status": "running",
        "pid": 102,
    }
    manager.command_history["cmd_1"] = first_command
    manager.command_history["cmd_2"] = second_command
    manager.terminals[terminal_id]["active_commands"]["cmd_1"] = first_command
    manager.terminals[terminal_id]["active_commands"]["cmd_2"] = second_command

    with pytest.raises(ValueError, match="Multiple active commands"):
        await manager.send_input(terminal_id, "Ada")
