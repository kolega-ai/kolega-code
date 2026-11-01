from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from acp.connection import StreamDirection, StreamEvent
from acp.interfaces import Client
from acp.schema import (
    AcceptElicitationResponse,
    ClientCapabilities,
    ElicitationCapabilities,
    ElicitationFormCapabilities,
    RequestPermissionResponse,
)

from kolega_code.acp.server import AcpAgent
from kolega_code.events import AgentEvent, KnownEventType
from kolega_code.hooks.events import HookEvent
from kolega_code.permissions import PermissionMode, ProjectPermissionStore
from kolega_code.session.runtime import serialize_permission_request
from kolega_code.tools import ToolError

from tests.acp.test_control_runtime import _permission
from tests.acp.test_server import StreamingLLM, _FakeConn, _FakeFactory, _make_session


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["permission", "question"])
async def test_embedded_client_response_acknowledges_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    session, _ = _make_session(tmp_path, StreamingLLM())
    session.agent.set_permission_mode(PermissionMode.ASK)
    server = AcpAgent(factory=cast(Any, _FakeFactory(session)))
    await server.initialize(
        protocol_version=1,
        client_capabilities=ClientCapabilities(elicitation=ElicitationCapabilities(form=ElicitationFormCapabilities())),
    )

    class EmbeddedConn(_FakeConn):
        metadata: dict[str, Any]

        async def request_permission(self, *args: Any, **kwargs: Any) -> Any:
            self.metadata = kwargs
            return await super().request_permission(*args, **kwargs)

        async def create_elicitation(self, message: str, mode: Any, **kwargs: Any) -> AcceptElicitationResponse:
            self.metadata = kwargs
            return AcceptElicitationResponse(action="accept", content={"answer": "B"})

    conn = EmbeddedConn()
    server.on_connect(cast(Client, conn))
    server._sessions[session.session_id] = session  # noqa: SLF001
    server._attach_runtime(session)  # noqa: SLF001
    assert session.runtime is not None
    assert session.control_adapter is not None
    hook = AsyncMock()
    monkeypatch.setattr(session.agent, "fire_hook", hook)
    try:
        assert session.runtime.control.controller == session.control_adapter.client_id == f"acp:{session.session_id}"
        if kind == "permission":
            assert (await session.runtime.permission_callback(_permission())).allowed is True
        else:
            assert await session.control_adapter.request_question("Pick one", ["A", "B"], ["", ""]) == "B"
        assert isinstance(conn.metadata["kolega_control_request_id"], str)
        assert hook.await_count == 1
        assert hook.await_args is not None
        assert hook.await_args.kwargs["target"] == ("permission_prompt" if kind == "permission" else "question")
        assert session.runtime.control.pending() == []
    finally:
        await session.control_adapter.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("request_data", [None, [], {"kind": "invalid"}, {"inputs": [1]}])
async def test_malformed_permission_payload_is_denied_without_transport(tmp_path: Path, request_data: Any) -> None:
    session, _ = _make_session(tmp_path, StreamingLLM())
    server = AcpAgent(factory=cast(Any, _FakeFactory(session)))
    conn = _FakeConn()
    server.on_connect(cast(Client, conn))
    server._sessions[session.session_id] = session  # noqa: SLF001
    server._attach_runtime(session)  # noqa: SLF001
    assert session.runtime is not None
    assert session.control_adapter is not None
    try:
        response = await asyncio.wait_for(
            session.runtime.control.request("permission", {"request": request_data}, default={}), 5
        )
        assert response == {"allowed": False, "reason": "The permission request was malformed."}
        assert conn.permission_calls == []
    finally:
        await session.control_adapter.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["settled", "removed", "no_runtime"])
async def test_stale_control_dispatch_and_callbacks_check_live_session(tmp_path: Path, state: str) -> None:
    session, _ = _make_session(tmp_path, StreamingLLM())
    server = AcpAgent(factory=cast(Any, _FakeFactory(session)))
    conn = _FakeConn()
    server.on_connect(cast(Client, conn))
    server._sessions[session.session_id] = session  # noqa: SLF001
    server._attach_runtime(session)  # noqa: SLF001
    assert session.control_adapter is not None
    adapter = session.control_adapter
    runtime = session.runtime
    callback = server._permission_callback_for(session.session_id)  # noqa: SLF001
    if state == "removed":
        server._sessions.pop(session.session_id)  # noqa: SLF001
    elif state == "no_runtime":
        session.runtime = None
    try:
        await adapter.emit(
            AgentEvent(
                sender="system",
                event_type=KnownEventType.CONTROL_REQUESTED,
                session_id=session.session_id,
                content={
                    "request_id": "already-settled",
                    "kind": "permission",
                    "payload": {"request": serialize_permission_request(_permission())},
                    "has_controller": True,
                },
            )
        )
        await asyncio.gather(*adapter.tasks.values())
        assert conn.permission_calls == []
        if state != "settled":
            decision = await callback(_permission())
            assert decision.allowed is False
            assert decision.reason == "Session is no longer active."
            with pytest.raises(ToolError, match=r"^The session is no longer active\.$"):
                await adapter.request_question("Pick one", ["A", "B"], ["", ""])
    finally:
        session.runtime = runtime
        await adapter.shutdown()


@pytest.mark.asyncio
async def test_close_joins_control_before_factory_cleanup_and_preserves_other_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, _ = _make_session(tmp_path, StreamingLLM())
    second, _ = _make_session(tmp_path, StreamingLLM())
    factory = _FakeFactory(first)
    server = AcpAgent(factory=cast(Any, factory))
    order: list[str] = []
    started = {session.session_id: asyncio.Event() for session in (first, second)}

    class WaitingConn(_FakeConn):
        async def request_permission(self, session_id: str, *args: Any, **kwargs: Any) -> Any:
            started[session_id].set()
            try:
                await asyncio.Event().wait()
            finally:
                order.append(f"control:{session_id}")

    server.on_connect(cast(Client, WaitingConn()))
    hooks = [AsyncMock(), AsyncMock()]
    for session, hook in zip((first, second), hooks):
        session.agent.set_permission_mode(PermissionMode.ASK)
        monkeypatch.setattr(session.agent, "fire_hook", hook)
        server._sessions[session.session_id] = session  # noqa: SLF001
        server._attach_runtime(session)  # noqa: SLF001
    assert first.runtime is not None and second.runtime is not None
    assert first.control_adapter is not None and second.control_adapter is not None
    requests = [
        asyncio.create_task(first.runtime.permission_callback(_permission())),
        asyncio.create_task(second.runtime.permission_callback(_permission())),
    ]
    turn_started = asyncio.Event()
    background_started = asyncio.Event()

    async def wait_for_cancel(name: str, ready: asyncio.Event) -> None:
        ready.set()
        try:
            await asyncio.Event().wait()
        finally:
            order.append(name)

    first.turn_task = asyncio.create_task(wait_for_cancel("turn", turn_started))
    first.background_task = asyncio.create_task(wait_for_cancel("background", background_started))

    async def close_factory_session(session_id: str) -> None:
        assert session_id == first.session_id
        assert first.control_adapter is not None and first.runtime is not None
        assert first.control_adapter.tasks == {}
        assert first.runtime.control.controller is None
        order.append("factory")

    monkeypatch.setattr(factory, "close_session", AsyncMock(side_effect=close_factory_session), raising=False)
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in (*started.values(), turn_started, background_started))), 5
        )
        request_id = first.runtime.control.pending()[0].request_id
        event = StreamEvent(
            direction=StreamDirection.OUTGOING,
            message={
                "method": "session/request_permission",
                "params": {"_meta": {"kolega_control_request_id": request_id}},
            },
        )
        await server.on_transport_message(StreamEvent(direction=StreamDirection.INCOMING, message=event.message))
        for message in [
            {**event.message, "method": "session/update"},
            {**event.message, "params": []},
            {**event.message, "params": {}},
            {**event.message, "params": {"_meta": []}},
            {**event.message, "params": {"_meta": {"kolega_control_request_id": 7}}},
            {**event.message, "params": {"_meta": {"kolega_control_request_id": "unknown"}}},
        ]:
            await server.on_transport_message(StreamEvent(direction=StreamDirection.OUTGOING, message=message))
        hooks[0].assert_not_awaited()
        hooks[1].assert_not_awaited()
        assert await second.control_adapter.on_transport_message(event) is False
        await server.on_transport_message(event)
        hooks[0].assert_awaited_once()
        hooks[1].assert_not_awaited()
        await server.close_session(first.session_id)
        assert order == ["turn", "background", f"control:{first.session_id}", "factory"]
        assert (await asyncio.wait_for(requests[0], 5)).allowed is False
        assert first.closed is True
        assert not requests[1].done()
        assert len(second.control_adapter.tasks) == 1
        assert second.runtime.control.controller == f"acp:{second.session_id}"
        assert await first.control_adapter.on_transport_message(event) is False
        hooks[0].assert_awaited_once()
        hooks[1].assert_not_awaited()
    finally:
        for task in requests:
            task.cancel()
        await asyncio.gather(*requests, return_exceptions=True)
        await first.control_adapter.shutdown()
        await second.control_adapter.shutdown()
        for task in (first.turn_task, first.background_task):
            task.cancel()
        await asyncio.gather(first.turn_task, first.background_task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled_kind", ["permission", "question"])
async def test_cancelling_one_control_keeps_the_other_same_session_control_answerable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancelled_kind: str
) -> None:
    session, _ = _make_session(tmp_path, StreamingLLM())
    session.agent.set_permission_mode(PermissionMode.ASK)
    server = AcpAgent(factory=cast(Any, _FakeFactory(session)))
    await server.initialize(
        protocol_version=1,
        client_capabilities=ClientCapabilities(elicitation=ElicitationCapabilities(form=ElicitationFormCapabilities())),
    )
    started = {kind: asyncio.Event() for kind in ("permission", "question")}
    cancelled = {kind: asyncio.Event() for kind in started}
    responses: dict[str, asyncio.Future[Any]] = {kind: asyncio.get_running_loop().create_future() for kind in started}

    class WaitingConn(_FakeConn):
        async def request_permission(self, *args: Any, **kwargs: Any) -> RequestPermissionResponse:
            return await self._wait("permission")

        async def create_elicitation(self, *args: Any, **kwargs: Any) -> AcceptElicitationResponse:
            return await self._wait("question")

        async def _wait(self, kind: str) -> Any:
            started[kind].set()
            try:
                return await responses[kind]
            except asyncio.CancelledError:
                cancelled[kind].set()
                raise

    server.on_connect(cast(Client, WaitingConn()))
    server._sessions[session.session_id] = session  # noqa: SLF001
    server._attach_runtime(session)  # noqa: SLF001
    assert session.runtime is not None and session.control_adapter is not None
    runtime, adapter = session.runtime, session.control_adapter
    hook = AsyncMock()
    monkeypatch.setattr(session.agent, "fire_hook", hook)
    callers = {
        "permission": asyncio.create_task(runtime.permission_callback(_permission())),
        "question": asyncio.create_task(adapter.request_question("Pick one", ["A", "B"], ["", ""])),
    }
    surviving_kind = "question" if cancelled_kind == "permission" else "permission"
    try:
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started.values())), 5)
        request_ids = {request.kind: request.request_id for request in runtime.control.pending()}
        cancelled_transport = adapter.tasks[request_ids[cancelled_kind]]
        surviving_transport = adapter.tasks[request_ids[surviving_kind]]
        callers[cancelled_kind].cancel()
        with pytest.raises(asyncio.CancelledError):
            await callers[cancelled_kind]
        await asyncio.wait_for(cancelled[cancelled_kind].wait(), 5)
        await asyncio.gather(cancelled_transport, return_exceptions=True)
        assert cancelled_transport.cancelled()
        assert not runtime.control.is_pending(request_ids[cancelled_kind])
        assert runtime.control.is_pending(request_ids[surviving_kind])
        assert not surviving_transport.done()
        assert not callers[surviving_kind].done()
        hook.assert_not_awaited()
        responses[surviving_kind].set_result(
            AcceptElicitationResponse(action="accept", content={"answer": "B"})
            if surviving_kind == "question"
            else RequestPermissionResponse.model_validate(
                {"outcome": {"outcome": "selected", "optionId": "allow-once"}}
            )
        )
        answer = await asyncio.wait_for(callers[surviving_kind], 5)
        if surviving_kind == "question":
            assert answer == "B"
        else:
            assert answer.allowed is True
        await surviving_transport
        assert runtime.control.pending() == []
        assert adapter.tasks == {}
        assert ProjectPermissionStore(tmp_path).load() == []
        hook.assert_awaited_once()
        assert hook.await_args is not None
        assert hook.await_args.args[0] == HookEvent.NOTIFICATION
        assert hook.await_args.kwargs["target"] == ("question" if surviving_kind == "question" else "permission_prompt")
        runtime.set_permission_mode(PermissionMode.AUTO)
        assert (await asyncio.wait_for(runtime.permission_callback(_permission()), 5)).allowed is True
    finally:
        for caller in callers.values():
            caller.cancel()
        await asyncio.gather(*callers.values(), return_exceptions=True)
        await adapter.shutdown()


@pytest.mark.asyncio
async def test_adapter_uses_current_connection_and_capabilities_after_attachment(tmp_path: Path) -> None:
    session, _ = _make_session(tmp_path, StreamingLLM())
    session.agent.set_permission_mode(PermissionMode.ASK)
    server = AcpAgent(factory=cast(Any, _FakeFactory(session)))
    server._sessions[session.session_id] = session  # noqa: SLF001
    server._attach_runtime(session)  # noqa: SLF001
    assert session.runtime is not None and session.control_adapter is not None
    adapter = session.control_adapter

    class EmbeddedConn(_FakeConn):
        def __init__(self, answer: str) -> None:
            super().__init__()
            self.answer: str = answer
            self.questions: list[str] = []

        async def create_elicitation(self, message: str, mode: Any, **kwargs: Any) -> AcceptElicitationResponse:
            self.questions.append(message)
            return AcceptElicitationResponse(action="accept", content={"answer": self.answer})

    first, second = EmbeddedConn("A"), EmbeddedConn("B")
    try:
        server.on_connect(cast(Client, first))
        assert (await session.runtime.permission_callback(_permission())).allowed is True
        assert len(first.permission_calls) == 1
        with pytest.raises(ToolError, match="does not support interactive question forms"):
            await adapter.request_question("Unsupported", ["A", "B"], ["", ""])
        await server.initialize(
            protocol_version=1,
            client_capabilities=ClientCapabilities(
                elicitation=ElicitationCapabilities(form=ElicitationFormCapabilities())
            ),
        )
        assert await adapter.request_question("First connection", ["A", "B"], ["", ""]) == "A"
        server.on_connect(cast(Client, second))
        assert await adapter.request_question("Second connection", ["A", "B"], ["", ""]) == "B"
        await server.initialize(protocol_version=1, client_capabilities=ClientCapabilities())
        with pytest.raises(ToolError, match="does not support interactive question forms"):
            await adapter.request_question("Unsupported again", ["A", "B"], ["", ""])
        assert first.questions == ["First connection"]
        assert second.questions == ["Second connection"]
    finally:
        await adapter.shutdown()
