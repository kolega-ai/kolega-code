"""Permission bridge tests: broker mapping and the real agent gate."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, cast
from unittest.mock import MagicMock

import pytest
from acp.interfaces import Client
from acp.schema import RequestPermissionResponse

from kolega_code.acp.permissions import OPTION_ALLOW_ONCE, OPTION_REJECT_ONCE, AcpPermissionBroker
from kolega_code.cli.connection import CliConnectionManager
from kolega_code.llm.models import ToolCall
from kolega_code.permissions import (
    PermissionKind,
    PermissionMode,
    PermissionRequest,
    PermissionRule,
    ProjectPermissionStore,
)

from tests.agent.compaction_helpers import build_agent


class _FakeConn:
    def __init__(self, responder: Callable[..., Any] | None = None) -> None:
        self.responder = responder or (lambda session_id, tool_call, options: None)
        self.calls: list[tuple[str, Any, list[Any]]] = []

    async def request_permission(self, session_id: str, tool_call: Any, options: list[Any], **kwargs: Any) -> Any:
        self.calls.append((session_id, tool_call, options))
        result = self.responder(session_id, tool_call, options)
        if asyncio.iscoroutine(result):
            result = await result
        return result


def _selected(option_id: str) -> Any:
    return RequestPermissionResponse.model_validate({"outcome": {"outcome": "selected", "optionId": option_id}})


def _cancelled() -> Any:
    return RequestPermissionResponse.model_validate({"outcome": {"outcome": "cancelled"}})


def _command_request(command: str = "rm -rf /tmp/x") -> PermissionRequest:
    return PermissionRequest(
        kind=PermissionKind.COMMAND,
        tool_name="exec_command",
        inputs={"command": command},
        command=command,
    )


def _broker(conn: _FakeConn, tmp_path: Path, agent: Any = None, **kwargs: Any) -> AcpPermissionBroker:
    return AcpPermissionBroker(cast(Client, conn), "s1", Path(tmp_path), lambda: agent, **kwargs)


@pytest.mark.asyncio
async def test_saved_rule_short_circuits_prompt(tmp_path: Path) -> None:
    request = _command_request()
    ProjectPermissionStore(Path(tmp_path)).add_rule(
        PermissionRule.create(kind=PermissionKind.COMMAND, tool="*", match_type="exact", pattern=request.command),
    )
    conn = _FakeConn(responder=lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not prompt")))

    decision = await _broker(conn, tmp_path)(request)

    assert decision.allowed is True
    assert "saved rule" in decision.reason
    assert conn.calls == []


@pytest.mark.asyncio
async def test_allow_once(tmp_path: Path) -> None:
    conn = _FakeConn(responder=lambda *a, **k: _selected(OPTION_ALLOW_ONCE))
    decision = await _broker(conn, tmp_path)(_command_request())
    assert decision.allowed is True
    assert decision.rule is None
    assert ProjectPermissionStore(Path(tmp_path)).load() == []


@pytest.mark.asyncio
async def test_allow_always_persists_matching_rule(tmp_path: Path) -> None:
    request = _command_request()
    conn = _FakeConn(responder=lambda *a, **k: _selected("allow-always-0"))
    decision = await _broker(conn, tmp_path)(request)

    assert decision.allowed is True
    assert decision.rule is not None
    saved = ProjectPermissionStore(Path(tmp_path)).first_match(request)
    assert saved is not None
    assert saved.matches(request)


@pytest.mark.asyncio
async def test_options_carry_rule_candidates(tmp_path: Path) -> None:
    conn = _FakeConn(responder=lambda *a, **k: _selected(OPTION_ALLOW_ONCE))
    await _broker(conn, tmp_path)(_command_request())
    _, _, options = conn.calls[0]

    kinds = [option.kind for option in options]
    assert kinds[0] == "allow_once"
    assert kinds[-1] == "reject_once"
    assert kinds.count("allow_always") >= 2
    assert options[1].name == "Always allow this exact command"


@pytest.mark.asyncio
async def test_reject_once_denies(tmp_path: Path) -> None:
    conn = _FakeConn(responder=lambda *a, **k: _selected(OPTION_REJECT_ONCE))
    decision = await _broker(conn, tmp_path)(_command_request())
    assert decision.allowed is False


@pytest.mark.asyncio
async def test_cancelled_outcome_denies(tmp_path: Path) -> None:
    conn = _FakeConn(responder=lambda *a, **k: _cancelled())
    decision = await _broker(conn, tmp_path)(_command_request())
    assert decision.allowed is False


@pytest.mark.asyncio
async def test_client_error_denies(tmp_path: Path) -> None:
    def responder(*a: Any, **k: Any) -> Any:
        raise RuntimeError("client exploded")

    conn = _FakeConn(responder=responder)
    decision = await _broker(conn, tmp_path)(_command_request())
    assert decision.allowed is False
    assert "could not be shown" in decision.reason


@pytest.mark.asyncio
async def test_timeout_denies(tmp_path: Path) -> None:
    async def responder(*a: Any, **k: Any) -> Any:
        await asyncio.Event().wait()

    conn = _FakeConn(responder=responder)
    decision = await _broker(conn, tmp_path, timeout=0.05)(_command_request())
    assert decision.allowed is False
    assert "Timed out" in decision.reason


@pytest.mark.asyncio
async def test_pending_tool_call_carries_tool_context(tmp_path: Path) -> None:
    agent = SimpleNamespace(current_tool_call_id="call-7")
    request = _command_request()
    conn = _FakeConn(responder=lambda *a, **k: _selected(OPTION_ALLOW_ONCE))
    await _broker(conn, tmp_path, agent=agent)(request)

    _, tool_call, _ = conn.calls[0]
    assert tool_call.tool_call_id == "call-7"
    assert tool_call.kind == "execute"
    assert tool_call.status == "pending"
    assert tool_call.title == request.command
    assert tool_call.raw_input == request.inputs


class _FakeRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def __contains__(self, item: object) -> bool:
        return True

    def names(self) -> list[str]:
        return ["exec_command"]

    def get(self, name: str) -> object:
        return object()

    async def call(self, name: str, **inputs: Any) -> str:
        self.calls.append((name, inputs))
        return "ok"


def _gated_agent(tmp_path: Path, conn: _FakeConn) -> tuple[Any, _FakeRegistry]:
    manager = CliConnectionManager()
    agent, _cm = build_agent(tmp_path, connection_manager=manager)
    registry = _FakeRegistry()
    assert agent.tool_collection is not None
    agent.tool_collection.registry = MagicMock(return_value=registry)
    broker = _broker(conn, tmp_path, agent=agent)
    agent.set_permission_mode(PermissionMode.ASK)
    agent.set_permission_callback(broker)
    return agent, registry


@pytest.mark.asyncio
async def test_agent_gate_denies_tool_when_client_rejects(tmp_path: Path) -> None:
    conn = _FakeConn(responder=lambda *a, **k: _cancelled())
    agent, registry = _gated_agent(tmp_path, conn)

    result = await agent.execute_single_tool(ToolCall(id="t1", name="exec_command", input={"command": "echo hi"}))

    assert result.is_error is True
    assert "denied" in str(result.content).lower()
    assert registry.calls == []
    assert conn.calls


@pytest.mark.asyncio
async def test_agent_gate_allows_tool_when_client_approves(tmp_path: Path) -> None:
    conn = _FakeConn(responder=lambda *a, **k: _selected(OPTION_ALLOW_ONCE))
    agent, registry = _gated_agent(tmp_path, conn)

    result = await agent.execute_single_tool(ToolCall(id="t1", name="exec_command", input={"command": "echo hi"}))

    assert result.is_error is False
    assert result.content == "ok"
    assert registry.calls == [("exec_command", {"command": "echo hi"})]
