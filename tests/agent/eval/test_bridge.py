"""Tests for the loopback tool bridge (HTTP semantics, dispatch, cancellation)."""

import asyncio
import json

import pytest
import pytest_asyncio

from kolega_code.agent.eval.bridge import LIST_TOOLS_OP, BridgeRegistration, ToolBridge
from kolega_code.tools import ToolError


async def _request(bridge: ToolBridge, body: dict | bytes | None, *, token: str | None = None, path: str = "/v1/tool"):
    reader, writer = await asyncio.open_connection("127.0.0.1", bridge._port)
    if isinstance(body, dict):
        raw = json.dumps(body).encode()
    else:
        raw = body or b""
    auth = f"Bearer {token}" if token is not None else f"Bearer {bridge.token}"
    request = (
        f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
        f"Authorization: {auth}\r\nContent-Length: {len(raw)}\r\nConnection: close\r\n\r\n"
    ).encode() + raw
    writer.write(request)
    await writer.drain()
    response = await reader.read()
    writer.close()
    head, _, payload = response.partition(b"\r\n\r\n")
    status = int(head.split(b" ", 2)[1])
    return status, json.loads(payload) if payload else {}


@pytest_asyncio.fixture
async def bridge():
    instance = ToolBridge()
    await instance._start()
    yield instance
    await instance.stop()


@pytest.mark.asyncio
async def test_rejects_wrong_token(bridge):
    status, payload = await _request(bridge, {"session": "s", "run": "r", "name": "x"}, token="wrong")
    assert status == 403
    assert payload["ok"] is False


@pytest.mark.asyncio
async def test_rejects_unknown_path(bridge):
    status, _ = await _request(bridge, {}, path="/nope")
    assert status == 404


@pytest.mark.asyncio
async def test_rejects_malformed_body(bridge):
    status, payload = await _request(bridge, b"{not json")
    assert status == 400
    assert payload["ok"] is False


@pytest.mark.asyncio
async def test_rejects_missing_fields(bridge):
    status, payload = await _request(bridge, {"session": "s"})
    assert status == 400
    assert "missing" in payload["error"]


@pytest.mark.asyncio
async def test_dispatch_round_trip(bridge):
    seen = []

    async def caller(name, args):
        seen.append((name, args))
        return {"text": f"hello {args['who']}"}

    unregister = bridge.register("sess", "run1", BridgeRegistration(tool_caller=caller))
    try:
        status, payload = await _request(
            bridge, {"session": "sess", "run": "run1", "name": "greet", "args": {"who": "kernel"}}
        )
    finally:
        unregister()
    assert status == 200
    assert payload == {"ok": True, "value": {"text": "hello kernel"}}
    assert seen == [("greet", {"who": "kernel"})]


@pytest.mark.asyncio
async def test_unknown_registration_is_tool_level_error(bridge):
    status, payload = await _request(bridge, {"session": "ghost", "run": "r", "name": "x", "args": {}})
    assert status == 200
    assert payload["ok"] is False
    assert "no active eval cell" in payload["error"]


@pytest.mark.asyncio
async def test_caller_exception_surfaces_as_error(bridge):
    async def caller(name, args):
        raise ToolError(f"{name} exploded")

    unregister = bridge.register("s", "r", BridgeRegistration(tool_caller=caller))
    try:
        _, payload = await _request(bridge, {"session": "s", "run": "r", "name": "edit", "args": {}})
    finally:
        unregister()
    assert payload["ok"] is False
    assert "edit exploded" in payload["error"]


@pytest.mark.asyncio
async def test_deregistration_rejects_late_calls(bridge):
    async def caller(name, args):
        return "late"

    unregister = bridge.register("s", "r", BridgeRegistration(tool_caller=caller))
    unregister()
    _, payload = await _request(bridge, {"session": "s", "run": "r", "name": "x", "args": {}})
    assert payload["ok"] is False
    assert "no active eval cell" in payload["error"]


@pytest.mark.asyncio
async def test_list_tools_op(bridge):
    async def caller(name, args):
        raise AssertionError("must not be called for __list_tools__")

    registration = BridgeRegistration(
        tool_caller=caller,
        tool_lister=lambda: [{"name": "read", "summary": "Read a file"}],
    )
    unregister = bridge.register("s", "r", registration)
    try:
        _, payload = await _request(bridge, {"session": "s", "run": "r", "name": LIST_TOOLS_OP, "args": {}})
    finally:
        unregister()
    assert payload == {"ok": True, "value": [{"name": "read", "summary": "Read a file"}]}


@pytest.mark.asyncio
async def test_interrupt_cancels_in_flight_calls(bridge):
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_caller(name, args):
        started.set()
        await release.wait()
        return "done"

    registration = BridgeRegistration(tool_caller=slow_caller)
    bridge.register("s", "r", registration)

    request_task = asyncio.create_task(_request(bridge, {"session": "s", "run": "r", "name": "slow", "args": {}}))
    await asyncio.wait_for(started.wait(), timeout=5)

    registration.cancel_in_flight()
    _, payload = await asyncio.wait_for(request_task, timeout=5)
    assert payload["ok"] is False
    assert "interrupt" in payload["error"].lower() or "cancel" in payload["error"].lower()

    # New calls are rejected too.
    _, payload = await _request(bridge, {"session": "s", "run": "r", "name": "x", "args": {}})
    assert payload["ok"] is False
    assert "interrupted" in payload["error"]
    release.set()
