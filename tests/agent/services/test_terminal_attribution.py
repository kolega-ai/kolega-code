import asyncio
import os
import shlex
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from kolega_code.agent.tool_backend.terminal_tool import TerminalTool
from kolega_code.config import AgentConfig
from kolega_code.events import AgentConnectionManager, AgentEvent
from kolega_code.services.terminal import (
    DetachedSession,
    LocalTerminalManager,
    PtySession,
    TerminalEventContext,
    terminal_event_scope,
)


class RecordingConnection(AgentConnectionManager):
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []
        self.changed: asyncio.Condition = asyncio.Condition()

    async def broadcast_event(self, event: AgentEvent, workspace_id: str, thread_id: str) -> None:
        assert workspace_id == "workspace"
        assert thread_id == "thread"
        async with self.changed:
            self.events.append(event)
            self.changed.notify_all()

    async def connect(
        self, websocket: Any, workspace_id: str, thread_id: str, connection_type: str, user_info: Any = None
    ) -> None:
        return None

    def disconnect(self, websocket: Any, workspace_id: str, thread_id: str, connection_type: str) -> None:
        return None

    def get_connection_count(self, workspace_id: str, thread_id: str) -> dict:
        return {}

    async def wait_for_output(self, session_id: str, text: str) -> None:
        async def wait() -> None:
            async with self.changed:
                await self.changed.wait_for(
                    lambda: any(
                        event.content.get("session_id") == session_id and text in event.content.get("output", "")
                        for event in self.events
                    )
                )

        await asyncio.wait_for(wait(), timeout=10)


def _caller(agent_id: str | None, execution_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        agent_name=agent_id or "root",
        sub_agent=agent_id is not None,
        sub_agent_context=(
            {
                "agent_id": agent_id,
                "parent_agent_id": "nested-parent",
                "depth": 3,
                "background": False,
                "effective_routing": {"provider": "supplied", "model": "launch-model", "effort": None},
            }
            if agent_id is not None
            else None
        ),
        current_tool_execution_id=execution_id,
        current_tool_call_id="provider-wire-id",
        scratchpad_dir=None,
        session_recorder=None,
    )


def _tool(path: Path, connection: RecordingConnection, manager: LocalTerminalManager, caller: Any) -> TerminalTool:
    return TerminalTool(
        path,
        "workspace",
        "thread",
        connection,
        AgentConfig(anthropic_api_key="fake-test-key", openai_api_key="fake-test-key"),
        caller,
        terminal_manager=manager,
    )


def _command() -> str:
    script = (
        "import sys\n"
        "print('launch-ready', flush=True)\n"
        "value = sys.stdin.readline()\n"
        "print('delayed-owned:' + value.strip(), flush=True)\n"
        "value = sys.stdin.readline()\n"
        "print('final-owned:' + value.strip(), flush=True)\n"
    )
    return f"{shlex.quote(sys.executable)} -u -c {shlex.quote(script)}"


def _session_id(result: str) -> str:
    return next(line.removeprefix("Session: ") for line in result.splitlines() if line.startswith("Session: "))


def _assert_owner(events: list[AgentEvent], session_id: str, agent_id: str | None, execution_id: str) -> None:
    owned = [event for event in events if event.content.get("session_id") == session_id]
    assert any(event.event_type == "terminal_command" for event in owned)
    assert any(event.event_type == "terminal_output" for event in owned)
    for event in owned:
        assert event.content["tool_call_id"] == execution_id
        if agent_id is None:
            assert event.sub_agent_info is None
        else:
            assert event.sub_agent_info == {
                "agent_id": agent_id,
                "parent_agent_id": "nested-parent",
                "depth": 3,
                "background": False,
                "effective_routing": {"provider": "supplied", "model": "launch-model", "effort": None},
            }


@pytest.mark.asyncio
@pytest.mark.parametrize("background", [False, True], ids=["pty", "detached"])
@pytest.mark.parametrize("owner_id", [None, "nested-child"], ids=["root-owner", "child-owner"])
@pytest.mark.parametrize("controller_id", [None, "other-child"], ids=["root-control", "sibling-control"])
@pytest.mark.parametrize("finish", ["poll", "kill"])
async def test_real_session_retains_owner_after_launch_task_finishes(
    tmp_path: Path, background: bool, owner_id: str | None, controller_id: str | None, finish: str
) -> None:
    connection = RecordingConnection()
    manager = LocalTerminalManager("workspace", "thread", connection, default_workdir=tmp_path)
    caller = _caller(owner_id, "launch-execution")
    launcher = _tool(tmp_path, connection, manager, caller)
    controller = _tool(tmp_path, connection, manager, _caller(controller_id, "control-execution"))
    session = None
    try:
        launch_task = asyncio.create_task(launcher.exec_command(_command(), yield_time_ms=250, background=background))
        launched = await asyncio.wait_for(launch_task, timeout=10)
        assert launch_task.done()
        sid = _session_id(launched)
        session = manager.sessions[sid]
        assert isinstance(session, DetachedSession if background else PtySession)
        assert session.pid is not None
        os.kill(session.pid, 0)
        await connection.wait_for_output(sid, "launch-ready")
        if caller.sub_agent_context:
            caller.sub_agent_context["effective_routing"]["model"] = "wrong-model"
            caller.sub_agent_context.update(agent_id="wrong-child", depth=99)
        caller.sub_agent_context = {"agent_id": "later-turn"}
        caller.current_tool_execution_id = "later-execution"

        if finish == "kill":
            await controller.write_stdin(sid, "release\n", yield_time_ms=250)
            await connection.wait_for_output(sid, "delayed-owned:release")
            assert session.running
            result = await controller.kill_command(sid)
        else:
            assert await session.write("release\nfinish\n")
            await connection.wait_for_output(sid, "delayed-owned:release")
            await asyncio.wait_for(session.exited.wait(), timeout=10)
            result = await controller.write_stdin(sid, "", yield_time_ms=250)
            assert "final-owned:finish" in result

        assert "Status: exited" in result
        assert session.exited.is_set()
        assert sid not in manager.sessions
        if isinstance(session, DetachedSession):
            assert session.log_path is None
            assert session.stdin_path is None
            assert session._pump_task is None
        else:
            assert session.master_fd is None
            assert session._broadcast_task is None
        _assert_owner(connection.events, sid, owner_id, "launch-execution")
        assert any(
            event.content.get("session_id") == sid and "[exited " in event.content.get("output", "")
            for event in connection.events
        )
    finally:
        if session is not None and session.running:
            await session.kill()
        for retained in list(manager.sessions.values()):
            if retained.running:
                await retained.kill()
        await manager.close_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("background", [False, True], ids=["pty", "detached"])
async def test_parallel_real_sessions_do_not_share_worker_attribution(tmp_path: Path, background: bool) -> None:
    connection = RecordingConnection()
    manager = LocalTerminalManager("workspace", "thread", connection, default_workdir=tmp_path)
    callers = [_caller(None, "root-execution"), _caller("child-a", "execution-a"), _caller("child-b", "execution-b")]
    tools = [_tool(tmp_path, connection, manager, caller) for caller in callers]
    sessions = []
    try:
        launches = [
            asyncio.create_task(tool.exec_command(_command(), yield_time_ms=250, background=background))
            for tool in tools
        ]
        results = await asyncio.wait_for(asyncio.gather(*launches), timeout=10)
        assert all(task.done() for task in launches)
        session_ids = [_session_id(result) for result in results]
        sessions = [manager.sessions[sid] for sid in session_ids]
        assert len(set(session_ids)) == 3
        assert len({session.pid for session in sessions}) == 3
        for caller in callers:
            caller.sub_agent_context = {"agent_id": "later-turn"}
            caller.current_tool_execution_id = "later-execution"
        for session in sessions:
            assert await session.write("release\nfinish\n")
        await asyncio.wait_for(asyncio.gather(*(session.exited.wait() for session in sessions)), timeout=10)
        await asyncio.gather(*(tools[0].write_stdin(sid, "", yield_time_ms=250) for sid in session_ids))

        assert not manager.sessions
        for sid, owner_id, execution_id in zip(
            session_ids, [None, "child-a", "child-b"], ["root-execution", "execution-a", "execution-b"], strict=True
        ):
            await connection.wait_for_output(sid, "delayed-owned:release")
            _assert_owner(connection.events, sid, owner_id, execution_id)
    finally:
        for session in sessions:
            if session.running:
                await session.kill()
        for session in list(manager.sessions.values()):
            if session.running:
                await session.kill()
        await manager.close_all()


@pytest.mark.asyncio
async def test_detached_pump_retains_owner_after_registry_cleanup(tmp_path: Path) -> None:
    connection = RecordingConnection()
    manager = LocalTerminalManager("workspace", "thread", connection, default_workdir=tmp_path)
    caller = _caller("detached-owner", "detached-execution")
    tool = _tool(tmp_path, connection, manager, caller)
    session: DetachedSession | None = None
    try:
        launch = asyncio.create_task(tool.exec_command(_command(), yield_time_ms=250, background=True))
        sid = _session_id(await asyncio.wait_for(launch, timeout=10))
        created = manager.sessions[sid]
        assert isinstance(created, DetachedSession)
        session = created
        await connection.wait_for_output(sid, "launch-ready")
        pump = session._pump_task
        assert pump is not None
        log_path, stdin_path = session.log_path, session.stdin_path
        assert log_path and stdin_path

        await manager.cleanup_all()
        assert not manager.sessions
        assert session.running
        assert not pump.done()
        caller.sub_agent_context["agent_id"] = "wrong-child"
        caller.current_tool_execution_id = "later-execution"
        assert await session.write("release\nfinish\n")
        await asyncio.wait_for(asyncio.shield(pump), timeout=10)
        await connection.wait_for_output(sid, "final-owned:finish")

        assert session.exit_code == 0
        assert session._pump_task is None
        assert session.log_path is None
        assert session.stdin_path is None
        assert not os.path.exists(log_path)
        assert not os.path.exists(stdin_path)
        _assert_owner(connection.events, sid, "detached-owner", "detached-execution")
    finally:
        if session is not None:
            if session.running:
                await session.kill()
            if session._pump_task is not None:
                await session._pump_task
            await session.close()
        await manager.close_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("background", [False, True], ids=["pty", "detached"])
@pytest.mark.parametrize("owner_id", [None, "launch-child"], ids=["root-owner", "child-owner"])
async def test_overlapping_final_polls_keep_launch_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, background: bool, owner_id: str | None
) -> None:
    connection = RecordingConnection()
    manager = LocalTerminalManager("workspace", "thread", connection, default_workdir=tmp_path)
    launcher = _tool(tmp_path, connection, manager, _caller(owner_id, "launch-execution"))
    session = None
    try:
        launched = await launcher.exec_command(_command(), yield_time_ms=250, background=background)
        sid = _session_id(launched)
        session = manager.sessions[sid]
        assert await session.write("release\nfinish\n")
        await asyncio.wait_for(session.exited.wait(), timeout=10)
        both_draining = asyncio.Event()
        first_finalized = asyncio.Event()
        entered = 0

        async def drain(_yield_ms: int) -> None:
            nonlocal entered
            entered += 1
            if entered == 1:
                await both_draining.wait()
            else:
                both_draining.set()
                await first_finalized.wait()

        monkeypatch.setattr(session, "drain", drain)

        async def poll(controller_id: str | None, first: bool) -> None:
            owner = TerminalEventContext.capture(_caller(controller_id, "control-execution"))
            with terminal_event_scope(owner):
                result = await manager.write_stdin(sid, "", yield_time_ms=250)
            assert result.status == "exited"
            if first:
                assert sid not in manager.sessions
                first_finalized.set()

        await asyncio.wait_for(asyncio.gather(poll(None, True), poll("other-child", False)), timeout=10)
        _assert_owner(connection.events, sid, owner_id, "launch-execution")
        assert sum("[exited " in event.content.get("output", "") for event in connection.events) == 2
    finally:
        if session is not None:
            if session.running:
                await session.kill()
            await session.close()
        await manager.close_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("message", ["[input notification]\n", "[exited 143]\n", "[error notification]\n"])
async def test_manager_notifications_use_retained_owner_not_control_scope(tmp_path: Path, message: str) -> None:
    connection = RecordingConnection()
    manager = LocalTerminalManager("workspace", "thread", connection)
    owner = TerminalEventContext.capture(_caller("launch-child", "launch-execution"))
    session = PtySession("retained", "unused", str(tmp_path), connection, "workspace", "thread", event_context=owner)
    manager.sessions["retained"] = session
    with terminal_event_scope(TerminalEventContext.capture(_caller("other-child", "control-execution"))):
        await manager._emit_output("retained", message)
    event = connection.events[0]
    assert event.sub_agent_info is not None
    assert event.sub_agent_info["agent_id"] == "launch-child"
    assert event.content["tool_call_id"] == "launch-execution"
    assert event.content["output"] == message
    manager.sessions.clear()


@pytest.mark.asyncio
async def test_terminal_scope_restores_on_exception_and_nested_calls() -> None:
    connection = RecordingConnection()
    manager = LocalTerminalManager("workspace", "thread", connection)
    owner = TerminalEventContext.capture(_caller("launch-child", "launch-execution"))
    with terminal_event_scope(owner):
        await manager._emit_output("no-session", "outer")
        with pytest.raises(RuntimeError, match="deliberate"):
            with terminal_event_scope(TerminalEventContext.capture(_caller("inner-child", "inner-execution"))):
                await manager._emit_output("no-session", "inner")
                raise RuntimeError("deliberate")
        await manager._emit_output("no-session", "outer again")
    await manager._emit_output("no-session", "unscoped")
    assert [event.content["tool_call_id"] for event in connection.events] == [
        "launch-execution",
        "inner-execution",
        "launch-execution",
        None,
    ]
    assert connection.events[-1].sub_agent_info is None


def test_terminal_owner_is_immutable_and_event_metadata_is_not_shared() -> None:
    caller = _caller("launch-child", "launch-execution")
    owner = TerminalEventContext.capture(caller)
    with pytest.raises(FrozenInstanceError):
        setattr(owner, "tool_call_id", "wrong-execution")
    metadata = owner.sub_agent_info
    assert metadata is not None
    metadata["effective_routing"]["model"] = "wrong-model"
    caller.sub_agent_context["agent_id"] = "wrong-child"
    retained = owner.sub_agent_info
    assert retained is not None
    assert retained["agent_id"] == "launch-child"
    assert retained["effective_routing"]["model"] == "launch-model"
