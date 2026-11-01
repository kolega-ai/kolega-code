"""Runtime policy and human-delivery acknowledgements through ACP's real wire."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from acp.agent.connection import AgentSideConnection
from acp.interfaces import Client
from acp.schema import (
    ClientCapabilities,
    ElicitationCapabilities,
    ElicitationFormCapabilities,
    RequestPermissionResponse,
)

from kolega_code.acp.agent_factory import AgentFactory
from kolega_code.acp.server import AcpAgent
from kolega_code.hooks.events import HookEvent
from kolega_code.permissions import (
    PermissionKind,
    PermissionMode,
    PermissionRequest,
    ProjectPermissionStore,
    allow_rule_options,
)
from kolega_code.tools import ToolError

from tests.acp.test_server import StreamingLLM, _FakeConn, _FakeFactory, _make_session


def _permission() -> PermissionRequest:
    return PermissionRequest(
        kind=PermissionKind.COMMAND,
        tool_name="execute_terminal_command",
        inputs={"command": "pytest"},
        command="pytest",
    )


class _Wire:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail: bool = fail
        self.sent: list[dict[str, Any]] = []
        self.incoming: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self.written: asyncio.Event = asyncio.Event()

    async def send(self, message: dict[str, Any]) -> None:
        if self.fail:
            raise OSError("write failed before delivery")
        self.sent.append(message)
        self.written.set()

    async def receive(self) -> dict[str, Any] | None:
        return await self.incoming.get()

    async def close(self) -> None:
        self.incoming.put_nowait(None)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["permission", "question"])
@pytest.mark.parametrize("fail", [False, True])
async def test_wire_delivery_notifies_once_before_reply_and_failed_writes_never_notify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, fail: bool
) -> None:
    session, _ = _make_session(tmp_path, StreamingLLM())
    session.agent.set_permission_mode(PermissionMode.ASK)
    agent = AcpAgent(factory=cast(Any, _FakeFactory(session)))
    await agent.initialize(
        protocol_version=1,
        client_capabilities=ClientCapabilities(elicitation=ElicitationCapabilities(form=ElicitationFormCapabilities())),
    )
    wire = _Wire(fail=fail)
    connection = AgentSideConnection(agent, wire, observers=[agent.on_transport_message])
    hook = AsyncMock()
    monkeypatch.setattr(session.agent, "fire_hook", hook)
    agent._sessions[session.session_id] = session  # noqa: SLF001
    agent._attach_runtime(session)  # noqa: SLF001
    agent._attach_question_elicit(session)  # noqa: SLF001
    assert session.runtime is not None
    task = asyncio.create_task(
        session.runtime.permission_callback(_permission())
        if kind == "permission"
        else session.question_state["elicit"]("Which approach?", ["A", "B"], ["", ""])
    )
    try:
        if fail:
            if kind == "question":
                with pytest.raises(ToolError, match="write failed before delivery"):
                    await asyncio.wait_for(task, 5)
            else:
                assert (await asyncio.wait_for(task, 5)).allowed is False
            hook.assert_not_awaited()
            assert wire.sent == []
        else:
            await asyncio.wait_for(wire.written.wait(), 5)
            async with asyncio.timeout(5):
                while hook.await_count == 0:
                    await asyncio.sleep(0)
            assert not task.done(), "the hook must not wait until the user replies"
            target = "permission_prompt" if kind == "permission" else "question"
            assert hook.await_count == 1
            assert hook.await_args is not None
            assert hook.await_args.args[0] == HookEvent.NOTIFICATION
            assert hook.await_args.kwargs["target"] == target
            message = wire.sent[0]
            assert "kolega_control_request_id" in message["params"]["_meta"]
            result = (
                {"outcome": {"outcome": "selected", "optionId": "allow-always-0"}}
                if kind == "permission"
                else {"action": "accept", "content": {"answer": "B"}}
            )
            wire.incoming.put_nowait({"jsonrpc": "2.0", "id": message["id"], "result": result})
            answer = await asyncio.wait_for(task, 5)
            if kind == "permission":
                assert answer.allowed is True
                assert answer.rule == ProjectPermissionStore(tmp_path).first_match(_permission())
            else:
                assert answer == "B"
            assert hook.await_count == 1, "the reply fallback must not duplicate the write acknowledgement"
            assert session.runtime.control.pending() == []
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await agent._cancel_control_tasks(session)  # noqa: SLF001
        await connection._conn.close()  # noqa: SLF001


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["auto", "saved", "no_controller"])
async def test_acp_auto_saved_and_no_controller_skip_transport_and_notification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    session, _ = _make_session(tmp_path, StreamingLLM())
    session.agent.set_permission_mode(PermissionMode.ASK)
    agent = AcpAgent(factory=cast(Any, _FakeFactory(session)))
    conn = _FakeConn()
    agent.on_connect(cast(Client, conn))
    agent._sessions[session.session_id] = session  # noqa: SLF001
    agent._attach_runtime(session)  # noqa: SLF001
    assert session.runtime is not None
    hook = AsyncMock()
    monkeypatch.setattr(session.agent, "fire_hook", hook)
    if policy == "auto":
        await AgentFactory().set_permission_mode(session, PermissionMode.AUTO)
    elif policy == "saved":
        ProjectPermissionStore(tmp_path).add_rule(allow_rule_options(_permission())[0].rule)
    else:
        session.runtime.control.release(agent._control_client_id(session.session_id))  # noqa: SLF001
    assert session.agent.permission_callback is not None
    decision = await session.agent.permission_callback(_permission())
    assert decision.allowed is (policy != "no_controller")
    assert conn.permission_calls == []
    hook.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("settle", ["cancel", "release", "timeout"])
async def test_nonanswer_resolution_cancels_pending_acp_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settle: str
) -> None:
    session, _ = _make_session(tmp_path, StreamingLLM())
    session.agent.set_permission_mode(PermissionMode.ASK)
    agent = AcpAgent(factory=cast(Any, _FakeFactory(session)))
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class WaitingConn(_FakeConn):
        async def request_permission(self, *args: Any, **kwargs: Any) -> Any:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    agent.on_connect(cast(Client, WaitingConn()))
    agent._sessions[session.session_id] = session  # noqa: SLF001
    agent._attach_runtime(session)  # noqa: SLF001
    assert session.runtime is not None
    hook = AsyncMock()
    monkeypatch.setattr(session.agent, "fire_hook", hook)
    if settle == "timeout":
        session.runtime.control.timeout = 0.05
    task = asyncio.create_task(session.runtime.permission_callback(_permission()))
    await asyncio.wait_for(started.wait(), 5)
    pending = session.runtime.control.pending()[0]
    transport_task = session.control_tasks[pending.request_id]
    if settle == "cancel":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        if settle == "release":
            session.runtime.control.release(agent._control_client_id(session.session_id))  # noqa: SLF001
        assert (await asyncio.wait_for(task, 5)).allowed is False
    await asyncio.wait_for(cancelled.wait(), 5)
    await asyncio.gather(transport_task, return_exceptions=True)
    assert transport_task.cancelled()
    async with asyncio.timeout(5):
        while session.control_tasks:
            await asyncio.sleep(0)
    assert session.control_tasks == {}
    assert await session.runtime.prompt_delivered(pending.request_id) is False
    assert ProjectPermissionStore(tmp_path).load() == []
    hook.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancel_foreground_turn_keeps_independent_background_permission_live(tmp_path: Path) -> None:
    session, _ = _make_session(tmp_path, StreamingLLM())
    session.agent.set_permission_mode(PermissionMode.ASK)
    agent = AcpAgent(factory=cast(Any, _FakeFactory(session)))
    started = asyncio.Event()
    response: asyncio.Future[RequestPermissionResponse] = asyncio.get_running_loop().create_future()

    class WaitingConn(_FakeConn):
        async def request_permission(self, *args: Any, **kwargs: Any) -> RequestPermissionResponse:
            started.set()
            return await response

    agent.on_connect(cast(Client, WaitingConn()))
    agent._sessions[session.session_id] = session  # noqa: SLF001
    agent._attach_runtime(session)  # noqa: SLF001
    assert session.runtime is not None
    background = asyncio.create_task(session.runtime.permission_callback(_permission()))
    foreground = asyncio.create_task(asyncio.Event().wait())
    session.turn_task = cast(Any, foreground)
    try:
        await asyncio.wait_for(started.wait(), 5)
        pending = session.runtime.control.pending()[0]
        transport = session.control_tasks[pending.request_id]
        await agent.cancel(session.session_id)
        with pytest.raises(asyncio.CancelledError):
            await foreground
        assert not transport.done()
        assert not background.done()
        assert len(session.runtime.control.pending()) == 1
        response.set_result(
            RequestPermissionResponse.model_validate({"outcome": {"outcome": "selected", "optionId": "allow-once"}})
        )
        assert (await asyncio.wait_for(background, 5)).allowed is True
        assert session.runtime.control.pending() == []
        # The background callback released the serialized permission lane.
        session.runtime.set_permission_mode(PermissionMode.AUTO)
        assert (await asyncio.wait_for(session.runtime.permission_callback(_permission()), 5)).allowed is True
    finally:
        foreground.cancel()
        background.cancel()
        await asyncio.gather(foreground, background, return_exceptions=True)
        await agent._cancel_control_tasks(session)  # noqa: SLF001


@pytest.mark.asyncio
async def test_factory_rebuild_keeps_runtime_callback_and_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session, manager = _make_session(tmp_path, StreamingLLM())
    server = AcpAgent(factory=cast(Any, _FakeFactory(session)))
    server._sessions[session.session_id] = session  # noqa: SLF001
    server._attach_runtime(session)  # noqa: SLF001
    assert session.runtime is not None
    factory = AgentFactory()
    await factory.set_permission_mode(session, PermissionMode.AUTO)
    monkeypatch.setattr(session.agent, "cleanup", AsyncMock())
    replacement, _ = _make_session(tmp_path, StreamingLLM())
    monkeypatch.setattr(replacement.agent, "connection_manager", manager)
    monkeypatch.setattr(factory, "persist", lambda _session: None)
    construct = AsyncMock(return_value=(replacement.agent, manager))
    monkeypatch.setattr(factory, "_construct_agent", construct)
    await factory._rebuild_agent(session, object())  # noqa: SLF001
    assert session.runtime.agent is replacement.agent
    assert session.agent.permission_mode == PermissionMode.AUTO
    assert session.record.permission_mode == PermissionMode.AUTO.value
    assert session.agent.permission_callback == session.runtime.permission_callback
    await factory.set_permission_mode(session, PermissionMode.ASK)
    assert session.runtime.permission_mode == PermissionMode.ASK
    assert session.agent.permission_mode == PermissionMode.ASK
