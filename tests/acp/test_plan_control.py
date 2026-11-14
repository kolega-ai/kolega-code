from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from acp.agent.connection import AgentSideConnection
from acp.connection import StreamDirection, StreamEvent
from acp.interfaces import Client
from acp.schema import (
    AcceptElicitationResponse,
    ClientCapabilities,
    ElicitationCapabilities,
    ElicitationFormCapabilities,
    RequestPermissionResponse,
    TextContentBlock,
)

from kolega_code.acp.permissions import CONTROL_REQUEST_ID_META_KEY
from kolega_code.acp.server import AcpAgent
from kolega_code.acp.session import AcpSession
from kolega_code.agent.prompts import build_implement_plan_prompt
from kolega_code.events import AgentEvent, KnownEventType
from kolega_code.llm.models import Message, TextBlock
from kolega_code.permissions import PermissionDecision, PermissionMode, ProjectPermissionStore, allow_rule_options
from kolega_code.session.plans import PLAN_APPROVAL_OPTIONS, plan_approval_payload
from tests.acp.test_control_runtime import _Wire, _permission
from tests.acp.test_server import StreamingLLM, _FakeConn, _FakeFactory, _make_session

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("isolated_cli_env")]

_PLAN = "\n# Exact approval plan\n\n" + ("- [ ] Preserve **all** Markdown and whitespace.\n" * 40) + "\n"
_Harness = tuple[AcpAgent, AcpSession, _FakeConn, list[AgentEvent]]


@pytest_asyncio.fixture
async def plan_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[_Harness]:
    source, _ = _make_session(tmp_path, StreamingLLM())
    factory = _FakeFactory(source)
    server = AcpAgent(factory=cast(Any, factory))
    conn = _FakeConn()
    server.on_connect(cast(Client, conn))
    monkeypatch.setattr(server, "_schedule_open_updates", lambda _session_id: None)
    opened = await server.new_session(str(tmp_path))
    session = server._sessions[opened.session_id]
    assert session is not source
    session.record.interaction_mode = "plan"
    session.record.latest_plan_markdown = _PLAN
    session.record.plan_pending = True
    assert session.runtime is not None and session.control_adapter is not None
    events: list[AgentEvent] = []
    broadcast = session.agent.connection_manager.broadcast_event

    async def record(event: AgentEvent, workspace_id: str, thread_id: str) -> None:
        if event.event_type in {KnownEventType.CONTROL_REQUESTED, KnownEventType.CONTROL_RESOLVED}:
            events.append(event)
        await broadcast(event, workspace_id, thread_id)

    async def close(_session_id: str) -> None:
        await session.agent.cleanup()

    monkeypatch.setattr(session.agent.tool_collection, "cleanup", AsyncMock())
    monkeypatch.setattr(session.agent.connection_manager, "broadcast_event", record)
    monkeypatch.setattr(factory, "close_session", close, raising=False)
    try:
        yield server, session, conn, events
    finally:
        await server.shutdown()


async def _approve(server: AcpAgent, session: AcpSession) -> None:
    async with session.execution_lock:
        await server._request_plan_approval(session, _PLAN)


async def _settled(session: AcpSession, events: list[AgentEvent], *, count: int = 2) -> None:
    assert session.control_adapter is not None
    async with asyncio.timeout(5):
        while len(events) < count or session.control_adapter.tasks:
            await asyncio.sleep(0)


def _response(action: str) -> RequestPermissionResponse:
    return RequestPermissionResponse.model_validate({"outcome": {"outcome": "selected", "optionId": action}})


@pytest.mark.parametrize("action", [option[0] for option in PLAN_APPROVAL_OPTIONS])
@pytest.mark.parametrize("representation", ["native", "dict"])
@pytest.mark.parametrize("policy", ["ask", "auto", "saved"])
async def test_public_plan_actions_use_shared_control_and_exact_acp_content(
    plan_server: _Harness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    representation: str,
    policy: str,
) -> None:
    server, session, conn, events = plan_server
    assert session.runtime is not None and session.control_adapter is not None
    runtime, adapter = session.runtime, session.control_adapter
    runtime.set_permission_mode(PermissionMode.AUTO if policy == "auto" else PermissionMode.ASK)
    store = ProjectPermissionStore(tmp_path)
    if policy == "saved":
        store.add_rule(allow_rule_options(_permission())[0].rule)
    saved = store.load()
    delivery = AsyncMock(wraps=runtime.prompt_delivered)
    monkeypatch.setattr(runtime, "prompt_delivered", delivery)
    seen: list[str | None] = []
    remaining = {"plan": _PLAN}
    monkeypatch.setattr(session.agent, "consume_completed_plan", lambda: remaining.pop("plan", None), raising=False)

    async def drive(current: AcpSession, bridge: Any, text: str | None, attachments: Any) -> str:
        assert current is session
        seen.append(text)
        current.agent.append_user_message([TextBlock(text=text or "")])
        if len(seen) == 1:
            current.agent.history.append(Message(role="assistant", content=[TextBlock(text=_PLAN)]))
        return "end_turn"

    async def answer(session_id: str, tool_call: Any, options: list[Any], **metadata: Any) -> Any:
        request = runtime.control.pending()[0]
        assert session_id == session.session_id
        assert session.turn_task is None and session.execution_lock.locked()
        assert session.plan_approval_task is not None and not session.plan_approval_task.done()
        assert request.kind == "plan_approval" and request.default == {"action": None}
        assert metadata == {CONTROL_REQUEST_ID_META_KEY: request.request_id}
        assert request.request_id in adapter.tasks
        assert tool_call.title == "Approve implementation plan?"
        assert tool_call.tool_call_id == f"plan-approval-{session.session_id[:8]}"
        assert tool_call.kind == "other" and tool_call.status == "pending"
        assert tool_call.content[0].content.text == _PLAN
        assert [(option.option_id, option.name, option.kind) for option in options] == [
            (option_id, label, "reject_once" if option_id == "discuss_plan" else "allow_once")
            for option_id, label in PLAN_APPROVAL_OPTIONS
        ]
        result = _response(action)
        return result if representation == "native" else result.model_dump(by_alias=True)

    monkeypatch.setattr(server, "_drive_turn", drive)
    monkeypatch.setattr(conn, "request_permission", answer)
    result = await server.prompt(session.session_id, [TextContentBlock(type="text", text="Original planning turn")])
    await _settled(session, events)
    assert result.stop_reason == "end_turn"
    requested, resolved = events
    assert requested.content["payload"] == plan_approval_payload(_PLAN)
    assert requested.content["has_controller"] is True
    assert resolved.content["request_id"] == requested.content["request_id"]
    assert resolved.content["kind"] == "plan_approval"
    assert resolved.content["response"] == {"action": action}
    assert resolved.content["reason"] == "answered"
    delivery.assert_awaited_once_with(requested.content["request_id"])
    assert runtime.control.pending() == [] and session.plan_approval_task is None
    assert session.approved_plan is None and session.idle()
    assert session.record.latest_plan_markdown == _PLAN and not session.record.plan_pending
    assert session.record.plan_reofferable is (action == "discuss_plan")
    assert session.record.interaction_mode == ("plan" if action == "discuss_plan" else "build")
    assert len(seen) == (1 if action == "discuss_plan" else 2)
    if len(seen) == 2:
        assert seen[1] == build_implement_plan_prompt(_PLAN)
    assert ("Original planning turn" in str(session.agent.dump_message_history())) is (action != "implement_plan_clear")
    assert store.load() == saved


@pytest.mark.parametrize(
    "response",
    [
        None,
        {},
        {"outcome": {"outcome": "cancelled"}},
        {"outcome": {"outcome": "cancelled", "optionId": "implement_plan"}},
        {"outcome": {"optionId": "implement_plan"}},
        {"outcome": {"outcome": "selected", "optionId": "allow-once"}},
        {"outcome": {"outcome": "selected", "optionId": "implement_plan "}},
        {"outcome": {"outcome": "selected", "optionId": ["implement_plan"]}},
        {"outcome": {"outcome": "selected", "option_id": "implement_plan"}},
        SimpleNamespace(outcome=SimpleNamespace(outcome="selected", option_id="unknown")),
        SimpleNamespace(outcome=SimpleNamespace(outcome="cancelled", option_id="implement_plan_clear")),
    ],
)
async def test_nonapproval_responses_keep_plan_pending_without_continuation(
    plan_server: _Harness, monkeypatch: pytest.MonkeyPatch, response: Any
) -> None:
    server, session, conn, events = plan_server
    monkeypatch.setattr(conn, "request_permission", AsyncMock(return_value=response))
    run = AsyncMock(return_value="end_turn")
    monkeypatch.setattr(server, "_run_turn", run)
    await asyncio.wait_for(_approve(server, session), 5)
    await _settled(session, events)
    run.assert_not_awaited()
    assert session.record.plan_pending and not session.record.plan_reofferable
    assert session.record.interaction_mode == "plan" and session.approved_plan is None
    assert events[-1].content["response"] == {"action": None}
    assert events[-1].content["reason"] == "answered"


@pytest.mark.parametrize("failure", ["no_connection", "transport", "transport_cancelled", "no_controller"])
async def test_unavailable_plan_prompt_defaults_promptly(
    plan_server: _Harness, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    server, session, conn, events = plan_server
    assert session.runtime is not None and session.control_adapter is not None
    if failure == "no_connection":
        del server._conn
    elif failure == "no_controller":
        session.runtime.control.release(session.control_adapter.client_id)
    else:
        monkeypatch.setattr(
            conn,
            "request_permission",
            AsyncMock(side_effect=OSError("transport failed") if failure == "transport" else asyncio.CancelledError()),
        )
    delivery = AsyncMock(wraps=session.runtime.prompt_delivered)
    monkeypatch.setattr(session.runtime, "prompt_delivered", delivery)
    await asyncio.wait_for(_approve(server, session), 5)
    await _settled(session, events)
    assert session.record.plan_pending and session.record.interaction_mode == "plan"
    assert session.approved_plan is None and session.plan_approval_task is None
    assert events[-1].content["response"] == {"action": None}
    assert events[-1].content["reason"] == ("no_controller" if failure == "no_controller" else "answered")
    assert conn.permission_calls == []
    delivery.assert_not_awaited()


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"plan": "", "options": plan_approval_payload(_PLAN)["options"]},
        {"plan": " \n", "options": plan_approval_payload(_PLAN)["options"]},
        {"plan": [_PLAN], "options": plan_approval_payload(_PLAN)["options"]},
        {"plan": _PLAN},
        {"plan": _PLAN, "options": [{"id": "implement_plan", "label": "Incorrect label"}]},
        {"plan": _PLAN, "options": list(reversed(plan_approval_payload(_PLAN)["options"]))},
    ],
)
async def test_malformed_plan_payload_defaults_without_transport(plan_server: _Harness, payload: Any) -> None:
    _, session, conn, events = plan_server
    assert session.runtime is not None
    result = await asyncio.wait_for(
        session.runtime.control.request("plan_approval", payload, default={"action": None}), 5
    )
    await _settled(session, events)
    assert result == {"action": None}
    assert events[-1].content["response"] == result
    assert conn.permission_calls == []


@pytest.mark.parametrize("fail", [False, True])
async def test_plan_wire_delivery_is_observed_before_reply_and_fallback_is_idempotent(
    plan_server: _Harness, monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    server, session, _, events = plan_server
    assert session.runtime is not None
    runtime = session.runtime
    deliveries: list[bool] = []
    original = runtime.prompt_delivered

    async def delivered(request_id: str) -> bool:
        result = await original(request_id)
        deliveries.append(result)
        return result

    monkeypatch.setattr(runtime, "prompt_delivered", delivered)
    wire = _Wire(fail=fail)
    connection = AgentSideConnection(server, wire, observers=[server.on_transport_message])
    caller = asyncio.create_task(runtime.request_plan_approval(_PLAN))
    try:
        if fail:
            assert await asyncio.wait_for(caller, 5) is None
            assert wire.sent == [] and deliveries == []
        else:
            await asyncio.wait_for(wire.written.wait(), 5)
            async with asyncio.timeout(5):
                while not deliveries:
                    await asyncio.sleep(0)
            pending = runtime.control.pending()[0]
            assert pending.delivered and not caller.done()
            message = wire.sent[0]
            assert message["method"] == "session/request_permission"
            assert message["params"]["_meta"] == {CONTROL_REQUEST_ID_META_KEY: pending.request_id}
            assert message["params"]["toolCall"]["content"][0]["content"]["text"] == _PLAN
            assert deliveries == [True]
            wire.incoming.put_nowait(
                {"jsonrpc": "2.0", "id": message["id"], "result": _response("discuss_plan").model_dump(by_alias=True)}
            )
            assert await asyncio.wait_for(caller, 5) == "discuss_plan"
            assert deliveries == [True, False]
        await _settled(session, events)
        assert runtime.control.pending() == []
    finally:
        caller.cancel()
        await asyncio.gather(caller, return_exceptions=True)
        assert session.control_adapter is not None
        await session.control_adapter.shutdown()
        await connection._conn.close()


@pytest.mark.parametrize("background_kind", ["permission", "question"])
async def test_cancel_plan_retains_late_transport_without_cancelling_unrelated_controls(
    plan_server: _Harness, monkeypatch: pytest.MonkeyPatch, background_kind: str
) -> None:
    server, session, conn, events = plan_server
    assert session.runtime is not None and session.control_adapter is not None
    runtime, adapter = session.runtime, session.control_adapter
    runtime.set_permission_mode(PermissionMode.ASK)
    await server.initialize(
        protocol_version=1,
        client_capabilities=ClientCapabilities(elicitation=ElicitationCapabilities(form=ElicitationFormCapabilities())),
    )
    started = {kind: asyncio.Event() for kind in ("plan", "background")}
    cancelled, release = asyncio.Event(), asyncio.Event()
    background_reply: asyncio.Future[Any] = asyncio.get_running_loop().create_future()

    async def permission(_session_id: str, _tool: Any, options: list[Any], **metadata: Any) -> Any:
        if options[0].option_id != "implement_plan":
            started["background"].set()
            return await background_reply
        started["plan"].set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
        return _response("implement_plan_clear")

    async def question(*args: Any, **kwargs: Any) -> Any:
        started["background"].set()
        return await background_reply

    monkeypatch.setattr(conn, "request_permission", permission)
    monkeypatch.setattr(conn, "create_elicitation", question, raising=False)
    run = AsyncMock(return_value="end_turn")
    monkeypatch.setattr(server, "_run_turn", run)
    approving = asyncio.create_task(_approve(server, session))
    background = asyncio.create_task(
        runtime.permission_callback(_permission())
        if background_kind == "permission"
        else adapter.request_question("Pick one", ["A", "B"], ["", ""])
    )
    try:
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started.values())), 5)
        requests = {request.kind: request for request in runtime.control.pending()}
        request_id = requests["plan_approval"].request_id
        transport = adapter.tasks[request_id]
        wait_task = session.plan_approval_task
        assert wait_task is not None and transport is not wait_task
        await server.cancel(session.session_id)
        await asyncio.wait_for(approving, 5)
        await asyncio.wait_for(cancelled.wait(), 5)
        assert session.plan_approval_task is None and wait_task.done()
        assert not transport.done() and adapter.tasks[request_id] is transport
        assert not background.done() and runtime.control.is_pending(requests[background_kind].request_id)
        assert not runtime.control.is_pending(request_id)
        assert await runtime.prompt_delivered(request_id) is False
        release.set()
        await asyncio.wait_for(transport, 5)
        assert runtime.answer_plan_approval(request_id, "implement_plan", client_id=adapter.client_id) is False
        background_reply.set_result(
            RequestPermissionResponse.model_validate({"outcome": {"outcome": "selected", "optionId": "allow-once"}})
            if background_kind == "permission"
            else AcceptElicitationResponse(action="accept", content={"answer": "B"})
        )
        result = await asyncio.wait_for(background, 5)
        if background_kind == "permission":
            assert isinstance(result, PermissionDecision) and result.allowed is True
        else:
            assert result == "B"
        await _settled(session, events, count=4)
        plan_resolutions = [
            event
            for event in events
            if event.event_type == KnownEventType.CONTROL_RESOLVED and event.content["kind"] == "plan_approval"
        ]
        assert len(plan_resolutions) == 1
        assert plan_resolutions[0].content["reason"] == "cancelled"
        assert plan_resolutions[0].content["response"] == {"action": None}
        run.assert_not_awaited()
        assert session.record.plan_pending and session.record.interaction_mode == "plan"
    finally:
        release.set()
        approving.cancel()
        background.cancel()
        await asyncio.gather(approving, background, return_exceptions=True)


@pytest.mark.parametrize("settlement", ["release", "timeout", "answered", "shutdown", "close"])
async def test_settlement_fences_late_plan_reply_and_shutdown_drains_despite_cancellation(
    plan_server: _Harness, monkeypatch: pytest.MonkeyPatch, settlement: str
) -> None:
    server, session, conn, events = plan_server
    assert session.runtime is not None and session.control_adapter is not None
    runtime, adapter = session.runtime, session.control_adapter
    started, cancelled, release = (asyncio.Event() for _ in range(3))
    deliveries = AsyncMock(wraps=runtime.prompt_delivered)
    monkeypatch.setattr(runtime, "prompt_delivered", deliveries)

    async def late(*args: Any, **kwargs: Any) -> RequestPermissionResponse:
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
        return _response("implement_plan_clear")

    monkeypatch.setattr(conn, "request_permission", late)
    run = AsyncMock(return_value="end_turn")
    monkeypatch.setattr(server, "_run_turn", run)
    if settlement == "timeout":
        runtime.control.timeout = 0.05
    approving = asyncio.create_task(_approve(server, session))
    shutting: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(started.wait(), 5)
        request_id = runtime.control.pending()[0].request_id
        transport = adapter.tasks[request_id]
        if settlement == "release":
            runtime.control.release(adapter.client_id)
        elif settlement == "answered":
            assert runtime.answer_plan_approval(request_id, None, client_id=adapter.client_id)
        elif settlement == "shutdown":
            shutting = asyncio.create_task(adapter.shutdown())
        elif settlement == "close":
            shutting = asyncio.create_task(server.close_session(session.session_id))
        await asyncio.wait_for(approving, 5)
        await asyncio.wait_for(cancelled.wait(), 5)
        assert not transport.done() and adapter.tasks[request_id] is transport
        assert runtime.control.pending() == []
        assert not await runtime.prompt_delivered(request_id)
        if shutting is not None:
            async with asyncio.timeout(5):
                while runtime.control.controller is not None:
                    await asyncio.sleep(0)
            shutting.cancel()
            await asyncio.sleep(0)
            shutting.cancel()
            await asyncio.sleep(0)
            assert not shutting.done() and not transport.done()
            assert adapter.tasks[request_id] is transport
            assert runtime.control.controller is None
            if settlement == "close":
                assert session.closed and server._sessions[session.session_id] is session
                assert session.close_task is not None and not session.close_task.done()
        release.set()
        await asyncio.wait_for(transport, 5)
        if shutting is not None:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(shutting, 5)
        await _settled(session, events)
        assert len(events) == 2 and events[-1].content["response"] == {"action": None}
        reasons = {
            "release": "controller_left",
            "timeout": "timeout",
            "answered": "answered",
            "shutdown": "controller_left",
            "close": "cancelled",
        }
        assert events[-1].content["reason"] == reasons[settlement]
        assert deliveries.await_count == 1, "late replies must not invoke the embedded delivery fallback"
        run.assert_not_awaited()
        assert session.record.plan_pending and session.record.interaction_mode == "plan"
    finally:
        release.set()
        await asyncio.gather(approving, *([shutting] if shutting is not None else []), return_exceptions=True)


@pytest.mark.parametrize("cancel", [False, True])
async def test_eager_plan_wait_and_transport_are_owned_before_inline_delivery_or_stop(
    plan_server: _Harness, monkeypatch: pytest.MonkeyPatch, cancel: bool
) -> None:
    eager_factory = getattr(asyncio, "eager_task_factory", None)
    if eager_factory is None:
        pytest.skip("eager task factory requires Python 3.12")
    server, session, conn, events = plan_server
    assert session.runtime is not None and session.control_adapter is not None
    runtime, adapter = session.runtime, session.control_adapter
    run = AsyncMock(return_value="end_turn")
    monkeypatch.setattr(server, "_run_turn", run)
    delivery = AsyncMock(wraps=runtime.prompt_delivered)
    monkeypatch.setattr(runtime, "prompt_delivered", delivery)

    async def inline(*args: Any, **metadata: Any) -> RequestPermissionResponse:
        request_id = metadata[CONTROL_REQUEST_ID_META_KEY]
        assert request_id in adapter.tasks and session.plan_approval_task is not None
        assert session.execution_lock.locked() and session.turn_task is None
        await server.on_transport_message(
            StreamEvent(
                direction=StreamDirection.OUTGOING,
                message={"method": "session/request_permission", "params": {"_meta": metadata}},
            )
        )
        assert runtime.control.pending()[0].delivered
        if cancel:
            await server.cancel(session.session_id)
        return _response("implement_plan")

    monkeypatch.setattr(conn, "request_permission", inline)
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    loop.set_task_factory(eager_factory)
    try:
        await asyncio.wait_for(_approve(server, session), 5)
        await _settled(session, events)
        assert session.plan_approval_task is None and adapter.tasks == {}
        assert runtime.control.pending() == [] and len(events) == 2
        assert delivery.await_count == (1 if cancel else 2)
        assert run.await_count == (0 if cancel else 1)
        assert session.record.plan_pending is cancel
    finally:
        loop.set_task_factory(previous_factory)
