import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from types import SimpleNamespace
from typing import Any

import pytest
import pytest_asyncio

from kolega_code.events import AgentEvent
from kolega_code.sandbox.terminal import SandboxTerminalManager
from kolega_code.services.terminal import TerminalEventContext, terminal_event_scope


class RecordingConnection:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    async def broadcast_event(self, event: AgentEvent, workspace_id: str, thread_id: str) -> None:
        assert (workspace_id, thread_id) == ("workspace", "thread")
        self.events.append(event)


class FakeCommandHandle:
    def __init__(
        self,
        pid: int,
        command: str,
        stdout: Callable[[str], Awaitable[None]],
        stderr: Callable[[str], Awaitable[None]],
    ) -> None:
        self.pid = pid
        self.command = command
        self.stdout = stdout
        self.stderr = stderr
        self.waiting = asyncio.Event()
        self.result: asyncio.Future[SimpleNamespace] = asyncio.get_running_loop().create_future()
        self.killed = False

    async def wait(self) -> SimpleNamespace:
        self.waiting.set()
        return await self.result

    def finish(self, exit_code: int = 0, error: Exception | None = None) -> None:
        if not self.result.done():
            if error is not None:
                self.result.set_exception(error)
            else:
                self.result.set_result(SimpleNamespace(exit_code=exit_code))

    async def kill(self) -> None:
        self.killed = True
        await self.stdout("killed\n")
        self.finish(137)


class FakeCommands:
    def __init__(self) -> None:
        self.launched: asyncio.Queue[FakeCommandHandle] = asyncio.Queue()
        self.handles: dict[int, FakeCommandHandle] = {}
        self.stdin_calls: list[tuple[int, str]] = []

    async def run(self, command: str, **kwargs: Any) -> FakeCommandHandle | SimpleNamespace:
        if command.startswith("test -d "):
            return SimpleNamespace(exit_code=0)
        handle = FakeCommandHandle(len(self.handles) + 100, command, kwargs["on_stdout"], kwargs["on_stderr"])
        self.handles[handle.pid] = handle
        await handle.stdout(f"ready:{command}\n")
        self.launched.put_nowait(handle)
        if kwargs.get("background"):
            return handle
        return await handle.wait()

    async def next_handle(self) -> FakeCommandHandle:
        handle = await asyncio.wait_for(self.launched.get(), timeout=10)
        await asyncio.wait_for(handle.waiting.wait(), timeout=10)
        return handle

    async def send_stdin(self, pid: int, data: str) -> None:
        self.stdin_calls.append((pid, data))
        await self.handles[pid].stdout(f"stdin:{data}")


@pytest_asyncio.fixture
async def sandbox_terminal() -> AsyncIterator[tuple[SandboxTerminalManager, FakeCommands, RecordingConnection]]:
    commands = FakeCommands()
    connection = RecordingConnection()
    manager = SandboxTerminalManager(SimpleNamespace(commands=commands), "workspace", "thread", connection)
    await manager.launch_terminal("shared")
    try:
        yield manager, commands, connection
    finally:
        for handle in commands.handles.values():
            handle.finish()
        tasks = list(manager._command_tasks.values())
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=10)
        await manager.close_all()


def _metadata(agent_id: str | None) -> dict[str, Any] | None:
    if agent_id is None:
        return None
    return {
        "agent_id": agent_id,
        "parent_agent_id": "nested-parent",
        "depth": 3,
        "background": True,
        "effective_routing": {"provider": "supplied", "model": "launch-model", "effort": None},
    }


def _caller(agent_id: str | None, tool_call_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        sub_agent_context=_metadata(agent_id),
        current_tool_execution_id=tool_call_id,
        current_tool_call_id="provider-wire-id",
    )


def _mutate_caller(caller: SimpleNamespace) -> None:
    if caller.sub_agent_context is not None:
        caller.sub_agent_context["effective_routing"]["model"] = "wrong-model"
        caller.sub_agent_context.update(agent_id="wrong-child", depth=99)
    caller.sub_agent_context = {"agent_id": "later-turn"}
    caller.current_tool_execution_id = "later-execution"
    caller.current_tool_call_id = "later-provider-id"


async def _launch(
    manager: SandboxTerminalManager, caller: SimpleNamespace, command: str, tracked: bool = True
) -> str | bool | None:
    with terminal_event_scope(TerminalEventContext.capture(caller)):
        if tracked:
            return await manager.send_command_tracked("shared", command, timeout=0)
        return await manager.send_command("shared", command, timeout=0)


def _assert_events(
    events: list[AgentEvent], session_id: str, agent_id: str | None, tool_call_id: str | None, outputs: list[str]
) -> None:
    assert [event.content["output"] for event in events] == outputs
    for event in events:
        assert event.event_type == "terminal_output"
        assert event.sender == "agent"
        assert event.content["terminal_id"] == "shared"
        assert event.content["thread_id"] == "thread"
        assert event.content["session_id"] == session_id
        assert event.content["tool_call_id"] == tool_call_id
        assert event.sub_agent_info == _metadata(agent_id)
        serialized = event.model_dump(mode="json")
        assert serialized["content"]["tool_call_id"] == tool_call_id
        assert serialized["sub_agent_info"] == _metadata(agent_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_id", [None, "nested-child"], ids=["root-owner", "nested-owner"])
@pytest.mark.parametrize("controller_id", [None, "sibling"], ids=["root-control", "sibling-control"])
@pytest.mark.parametrize("finish", ["exit", "error", "timeout", "TERM", "INT"])
async def test_tracked_command_retains_launch_owner(
    sandbox_terminal: tuple[SandboxTerminalManager, FakeCommands, RecordingConnection],
    owner_id: str | None,
    controller_id: str | None,
    finish: str,
) -> None:
    manager, commands, connection = sandbox_terminal
    caller = _caller(owner_id, "launch-execution")
    launch = asyncio.create_task(_launch(manager, caller, "tracked-command"))
    session_id = await asyncio.wait_for(launch, timeout=10)
    assert launch.done()
    assert isinstance(session_id, str)
    handle = await commands.next_handle()
    worker = manager._command_tasks[session_id]
    _mutate_caller(caller)

    expected = ["$ tracked-command\n", "ready:tracked-command\n", "delayed\n", "warning\n", "stdin:payload\n"]
    with terminal_event_scope(TerminalEventContext.capture(_caller(controller_id, "control-execution"))):
        await handle.stdout("delayed\n")
        await handle.stderr("warning\n")
        assert await manager.send_input("shared", "payload", command_id=session_id)
        if finish in ("TERM", "INT"):
            result = await manager.kill_session(session_id, finish)
            assert handle.killed
            if finish == "INT":
                assert (handle.pid, "\x03") in commands.stdin_calls
                expected.append("stdin:\x03")
            expected.extend(["killed\n", "Process exited with code 137\n"])
            assert result.exit_code == 137
            assert manager.command_history[session_id]["status"] == "terminated"
        else:
            error = RuntimeError("sandbox failure") if finish == "error" else None
            if finish == "timeout":
                error = asyncio.TimeoutError()
            handle.finish(exit_code=7, error=error)
            await asyncio.wait_for(asyncio.shield(worker), timeout=10)
            result = await manager.write_stdin(session_id, "", yield_time_ms=250)
            if finish == "exit":
                expected.append("Process exited with code 7\n")
                assert result.exit_code == 7
            else:
                message = "sandbox failure" if finish == "error" else "Command execution timed out after 5 seconds"
                expected.extend([f"Command failed: {message}", "Process exited with code 1\n"])
                assert result.exit_code == 1
        assert result.status == "exited"
        assert "delayed\nwarning\nstdin:payload\n" in result.output

    await asyncio.wait_for(asyncio.shield(worker), timeout=10)
    _assert_events(connection.events, session_id, owner_id, "launch-execution", expected)
    assert session_id not in manager._command_tasks
    assert await manager.list_sessions() == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_id", [None, "nested-child"], ids=["root-owner", "nested-owner"])
@pytest.mark.parametrize("controller_id", [None, "sibling"], ids=["root-control", "sibling-control"])
@pytest.mark.parametrize("finish", ["exit", "error", "timeout"])
async def test_legacy_command_retains_launch_owner(
    sandbox_terminal: tuple[SandboxTerminalManager, FakeCommands, RecordingConnection],
    owner_id: str | None,
    controller_id: str | None,
    finish: str,
) -> None:
    manager, commands, connection = sandbox_terminal
    caller = _caller(owner_id, "launch-execution")
    launch = asyncio.create_task(_launch(manager, caller, "legacy-command", tracked=False))
    handle = await commands.next_handle()
    try:
        _mutate_caller(caller)
        with terminal_event_scope(TerminalEventContext.capture(_caller(controller_id, "control-execution"))):
            await handle.stdout("delayed\n")
            await handle.stderr("warning\n")
            error = RuntimeError("sandbox failure") if finish == "error" else None
            if finish == "timeout":
                error = asyncio.TimeoutError()
            handle.finish(error=error)
            succeeded = await asyncio.wait_for(asyncio.shield(launch), timeout=10)
        assert succeeded is (finish == "exit")
        expected = ["$ legacy-command\n", "ready:legacy-command\n", "delayed\n", "warning\n"]
        if finish == "exit":
            expected.append("Process exited with code 0\n")
        elif finish == "error":
            expected.extend(["Command failed: sandbox failure", "Process exited with code 1\n"])
        else:
            expected.extend(["Command execution timed out after 5 seconds", "Process exited with code 1\n"])
        session_id = connection.events[0].content.get("session_id")
        assert isinstance(session_id, str) and session_id
        _assert_events(connection.events, session_id, owner_id, "launch-execution", expected)
        assert not manager.command_history
    finally:
        handle.finish()
        await asyncio.wait_for(asyncio.shield(launch), timeout=10)


@pytest.mark.asyncio
@pytest.mark.parametrize("tracked", [True, False], ids=["tracked", "legacy"])
@pytest.mark.parametrize("controller_id", [None, "sibling"], ids=["root-callbacks", "sibling-callbacks"])
async def test_parallel_commands_share_terminal_but_not_owner(
    sandbox_terminal: tuple[SandboxTerminalManager, FakeCommands, RecordingConnection],
    tracked: bool,
    controller_id: str | None,
) -> None:
    manager, commands, connection = sandbox_terminal
    owners = [None, "child-a", "nested-child-b"]
    callers = [_caller(owner, f"execution-{i}") for i, owner in enumerate(owners)]
    launches = [
        asyncio.create_task(_launch(manager, caller, f"parallel-{i}", tracked=tracked))
        for i, caller in enumerate(callers)
    ]
    handles = [await commands.next_handle() for _ in owners]
    workers = list(manager._command_tasks.values())
    try:
        for caller in callers:
            _mutate_caller(caller)
        assert len(commands.handles) == 3
        with terminal_event_scope(TerminalEventContext.capture(_caller(controller_id, "control-execution"))):
            for handle in reversed(handles):
                await handle.stderr(f"warning:{handle.command}\n")
            for handle in handles:
                await handle.stdout(f"delayed:{handle.command}\n")
                handle.finish()
            results = await asyncio.wait_for(asyncio.gather(*launches), timeout=10)
            if workers:
                await asyncio.wait_for(asyncio.gather(*workers), timeout=10)
        session_ids = []
        for i, owner in enumerate(owners):
            command = f"parallel-{i}"
            launch_event = next(event for event in connection.events if event.content["output"] == f"$ {command}\n")
            session_id = launch_event.content.get("session_id")
            assert isinstance(session_id, str) and session_id
            session_ids.append(session_id)
            if tracked:
                assert results[i] == session_id
            else:
                assert results[i] is True
            events = [event for event in connection.events if event.content.get("session_id") == session_id]
            _assert_events(
                events,
                session_id,
                owner,
                f"execution-{i}",
                [
                    f"$ {command}\n",
                    f"ready:{command}\n",
                    f"warning:{command}\n",
                    f"delayed:{command}\n",
                    "Process exited with code 0\n",
                ],
            )
        assert len(set(session_ids)) == 3
        assert len(connection.events) == 15
    finally:
        for handle in handles:
            handle.finish()
        await asyncio.wait_for(asyncio.gather(*launches), timeout=10)


@pytest.mark.asyncio
async def test_default_handlers_snapshot_nested_scope_and_isolate_event_metadata(
    sandbox_terminal: tuple[SandboxTerminalManager, FakeCommands, RecordingConnection],
) -> None:
    manager, _, connection = sandbox_terminal
    outer = _caller("outer", "outer-execution")
    inner = _caller("inner", "inner-execution")
    inner.current_tool_execution_id = None
    inner.current_tool_call_id = "inner-provider-id"
    with terminal_event_scope(TerminalEventContext.capture(outer)):
        outer_handler = await manager._create_output_handler("shared", "stdout")
        with pytest.raises(RuntimeError, match="scope failure"):
            with terminal_event_scope(TerminalEventContext.capture(inner)):
                inner_handler = await manager._create_output_handler("shared", "stderr")
                raise RuntimeError("scope failure")
        await manager._broadcast_output("shared", "outer-notification")
    root_handler = await manager._create_output_handler("shared", "stdout")
    _mutate_caller(outer)
    _mutate_caller(inner)

    with terminal_event_scope(TerminalEventContext.capture(_caller("sibling", "wrong-execution"))):
        await outer_handler("outer-first")
        first = connection.events[-1]
        assert first.sub_agent_info == _metadata("outer")
        assert first.sub_agent_info is not None
        first.sub_agent_info["effective_routing"]["model"] = "consumer-mutated-model"
        await outer_handler("outer-second")
        await inner_handler("inner")
        await root_handler("root")
    await manager._broadcast_output("shared", "unscoped-notification")

    assert [event.content["tool_call_id"] for event in connection.events] == [
        "outer-execution",
        "outer-execution",
        "outer-execution",
        "inner-provider-id",
        None,
        None,
    ]
    assert connection.events[0].sub_agent_info == _metadata("outer")
    assert connection.events[2].sub_agent_info == _metadata("outer")
    assert connection.events[3].sub_agent_info == _metadata("inner")
    assert connection.events[4].sub_agent_info is None
    assert connection.events[5].sub_agent_info is None
