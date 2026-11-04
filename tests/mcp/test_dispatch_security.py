from __future__ import annotations

import re
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from kolega_code.mcp.config import LoadedMCPConfig, MCPServerConfig, server_fingerprint
from kolega_code.mcp.service import MCPService, _suppress_mcp_sdk_terminal_logs, exposed_tool_names_for
from kolega_code.mcp.state import MCPServerStatus, MCPToolStatus
from kolega_code.mcp.tools import build_mcp_tool_extension
from kolega_code.permissions import allow_rule_options, permission_request_for_tool
from kolega_code.tools import ToolError


def _verified_service(tmp_path: Path, inventories: dict[str, list[str]]) -> MCPService:
    servers = {
        server_id: MCPServerConfig(id=server_id, transport="stdio", command="fake-mcp-server")
        for server_id in inventories
    }
    service = MCPService(LoadedMCPConfig(servers=servers), state_dir=tmp_path, project_path=tmp_path)
    for server_id, tool_ids in inventories.items():
        server = servers[server_id]
        service.status_store.update(
            server_id,
            MCPServerStatus.verified(
                fingerprint=server_fingerprint(server),
                transport=server.transport,
                source=server.source,
                tools=[MCPToolStatus(id=tool_id) for tool_id in tool_ids],
            ),
        )
    return service


@pytest.mark.parametrize("original", [True, False])
def test_overlapping_log_suppression_restores_logger_only_after_last_operation(original: bool) -> None:
    logger = logging.getLogger("mcp")
    previous = logger.propagate
    logger.propagate = original
    first = _suppress_mcp_sdk_terminal_logs()
    second = _suppress_mcp_sdk_terminal_logs()
    try:
        first.__enter__()
        second.__enter__()
        first.__exit__(None, None, None)
        assert not logger.propagate
        second.__exit__(None, None, None)
        assert logger.propagate is original
    finally:
        logger.propagate = previous


@pytest.fixture
def transport_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, dict[str, Any]]]:
    calls: list[tuple[str, str, dict[str, Any]]] = []

    @asynccontextmanager
    async def open_session(server: MCPServerConfig, **kwargs: Any) -> AsyncGenerator[SimpleNamespace]:
        async def call_tool(tool_id: str, arguments: dict[str, Any]) -> SimpleNamespace:
            calls.append((server.id, tool_id, arguments))
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")])

        yield SimpleNamespace(call_tool=call_tool)

    monkeypatch.setattr("kolega_code.mcp.service.open_mcp_session", open_session)
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "inputs",
    [
        {"_server_id": "other"},
        {"_tool_id": "delete"},
        {"_server_id": "other", "_tool_id": "delete", "_ordinary": "keep", "query": "hello"},
    ],
)
async def test_callback_inputs_cannot_override_permission_approved_route(
    tmp_path: Path, transport_calls: list[tuple[str, str, dict[str, Any]]], inputs: dict[str, Any]
) -> None:
    service = _verified_service(tmp_path, {"docs": ["search", "delete"], "other": ["search", "delete"]})
    extension = build_mcp_tool_extension(tmp_path, tmp_path, project_trusted=False, loaded_config=service.config)
    assert extension is not None
    wire_name = "mcp__docs__search"
    request = permission_request_for_tool(wire_name, inputs)
    assert request is not None
    assert (request.mcp_server, request.mcp_tool) == ("docs", "search")

    assert await extension.tools[wire_name](**inputs) == "ok"
    assert transport_calls == [("docs", "search", inputs)]
    assert await extension.tools["mcp__other__delete"]() == "ok"
    assert transport_calls[-1] == ("other", "delete", {})


@pytest.mark.asyncio
async def test_empty_tool_name_cannot_bypass_permission_checks(
    tmp_path: Path, transport_calls: list[tuple[str, str, dict[str, Any]]]
) -> None:
    service = _verified_service(tmp_path, {"docs": ["", "_"]})
    extension = build_mcp_tool_extension(tmp_path, tmp_path, project_trusted=False, loaded_config=service.config)
    assert extension is not None
    names = extension.tool_groups["mcp_tools"]
    assert len(set(names)) == 2
    for name, tool_id in zip(names, ["", "_"], strict=True):
        request = permission_request_for_tool(name, {})
        assert request is not None
        assert request.mcp_server == "docs"
        await extension.tools[name]()
        assert transport_calls[-1] == ("docs", tool_id, {})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "inventories",
    [
        {"x": ["y__z"], "x__y": ["z"]},
        {"x": ["y__a.b"], "x__y": ["a_b"]},
        {"x": ["y__" + "z" * 80], "x__y": ["z" * 80]},
    ],
    ids=["ambiguous-separator", "sanitized", "truncated"],
)
async def test_wire_names_are_globally_unique_stable_and_route_to_original_ids(
    tmp_path: Path,
    transport_calls: list[tuple[str, str, dict[str, Any]]],
    inventories: dict[str, list[str]],
) -> None:
    service = _verified_service(tmp_path, {**inventories, "docs": ["search", "get.file", "get_file"]})
    extension = build_mcp_tool_extension(tmp_path, tmp_path, project_trusted=False, loaded_config=service.config)
    assert extension is not None
    routes = [(tool.server.id, tool.tool.id) for tool in service.exposed_tools()]
    names = extension.tool_groups["mcp_tools"]
    assert len(names) == len(set(names)) == len(routes)
    assert list(extension.tools) == names
    assert set(extension.tool_schemas) == set(extension.tool_descriptions) == set(names)
    for name in names:
        assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name)
        assert await extension.tools[name](value=name) == "ok"
    assert transport_calls == [
        (server_id, tool_id, {"value": name}) for (server_id, tool_id), name in zip(routes, names)
    ]
    original_names = dict(zip(routes, names))
    for raw, name in exposed_tool_names_for(
        "docs", [MCPToolStatus(id=tool_id) for tool_id in ["search", "get.file", "get_file"]]
    ):
        tool_id = raw.removeprefix("mcp__docs__")
        assert original_names[("docs", tool_id)] == name
    aliased = original_names[("x__y", inventories["x__y"][0])]
    request = permission_request_for_tool(aliased, {})
    assert request is not None
    assert request.mcp_server != "x"
    assert "__" not in request.mcp_server
    assert f"exposed here as `{aliased}`" in extension.tool_descriptions[aliased]

    reversed_config = LoadedMCPConfig(servers=dict(reversed(list(service.config.servers.items()))))
    rebuilt = build_mcp_tool_extension(tmp_path, tmp_path, project_trusted=False, loaded_config=reversed_config)
    assert rebuilt is not None
    reversed_service = MCPService(reversed_config, state_dir=tmp_path, project_path=tmp_path)
    reversed_routes = [(tool.server.id, tool.tool.id) for tool in reversed_service.exposed_tools()]
    assert dict(zip(reversed_routes, rebuilt.tool_groups["mcp_tools"])) == original_names


@pytest.mark.parametrize("server_id", ["docs__admin", "docs_"])
def test_ambiguous_server_id_cannot_inherit_another_servers_permission(tmp_path: Path, server_id: str) -> None:
    service = _verified_service(tmp_path, {"docs": ["search"], server_id: ["admin__search"]})
    extension = build_mcp_tool_extension(tmp_path, tmp_path, project_trusted=False, loaded_config=service.config)
    assert extension is not None
    routes = [(tool.server.id, tool.tool.id) for tool in service.exposed_tools()]
    names = dict(zip(routes, extension.tool_groups["mcp_tools"]))
    legitimate = permission_request_for_tool(names[("docs", "search")], {})
    ambiguous = permission_request_for_tool(names[(server_id, "admin__search")], {})
    assert legitimate is not None and ambiguous is not None
    assert legitimate.mcp_server == "docs"
    assert ambiguous.mcp_server not in {"docs", server_id}
    server_grant = allow_rule_options(legitimate)[1].rule
    assert server_grant.matches(legitimate)
    assert not server_grant.matches(ambiguous)
    rebuilt = build_mcp_tool_extension(tmp_path, tmp_path, project_trusted=False, loaded_config=service.config)
    assert rebuilt is not None
    assert rebuilt.tool_groups["mcp_tools"] == extension.tool_groups["mcp_tools"]


def test_ambiguous_server_alias_does_not_reuse_permission_across_process_salts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _verified_service(tmp_path, {"docs__admin": ["search"]})
    monkeypatch.setattr("kolega_code.mcp.service._MCP_ALIAS_SALT", b"first-process")
    first = build_mcp_tool_extension(tmp_path, tmp_path, project_trusted=False, loaded_config=service.config)
    monkeypatch.setattr("kolega_code.mcp.service._MCP_ALIAS_SALT", b"second-process")
    second = build_mcp_tool_extension(tmp_path, tmp_path, project_trusted=False, loaded_config=service.config)
    assert first is not None and second is not None
    first_request = permission_request_for_tool(first.tool_groups["mcp_tools"][0], {})
    second_request = permission_request_for_tool(second.tool_groups["mcp_tools"][0], {})
    assert first_request is not None and second_request is not None
    assert first_request.mcp_server != second_request.mcp_server
    previous_grant = allow_rule_options(first_request)[1].rule
    assert not previous_grant.matches(second_request)


@pytest.mark.asyncio
async def test_collision_suffix_cannot_take_an_unchanged_wire_name(
    tmp_path: Path, transport_calls: list[tuple[str, str, dict[str, Any]]]
) -> None:
    service = _verified_service(tmp_path, {"x": ["first.name", "first_name"]})
    extension = build_mcp_tool_extension(tmp_path, tmp_path, project_trusted=False, loaded_config=service.config)
    assert extension is not None
    reserved = extension.tool_groups["mcp_tools"][0]
    natural_tool_id = reserved.removeprefix("mcp__x__")
    status = service.server_status(service.config.servers["x"])
    assert status is not None
    service.status_store.update(
        "x",
        status.model_copy(update={"tools": [*status.tools, MCPToolStatus(id=natural_tool_id)], "tool_count": 3}),
    )

    rebuilt = build_mcp_tool_extension(tmp_path, tmp_path, project_trusted=False, loaded_config=service.config)
    assert rebuilt is not None
    names = rebuilt.tool_groups["mcp_tools"]
    assert len(names) == len(set(names)) == 3
    assert names[2] == reserved
    assert await rebuilt.tools[reserved]() == "ok"
    assert transport_calls == [("x", natural_tool_id, {})]
    assert await rebuilt.tools[names[0]]() == "ok"
    assert transport_calls[-1] == ("x", "first.name", {})


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_ids", [["search"], []], ids=["unknown-tool", "empty-inventory"])
async def test_call_rejects_unknown_tool_before_opening_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool_ids: list[str]
) -> None:
    service = _verified_service(tmp_path, {"docs": tool_ids})

    def unexpected_transport(*args: Any, **kwargs: Any) -> None:
        pytest.fail("unverified tool opened an MCP transport")

    monkeypatch.setattr("kolega_code.mcp.service.open_mcp_session", unexpected_transport)
    with pytest.raises(ToolError, match="verified inventory"):
        await service.call_tool("docs", "delete", {})


@pytest.mark.asyncio
async def test_existing_callback_uses_current_inventory(
    tmp_path: Path, transport_calls: list[tuple[str, str, dict[str, Any]]]
) -> None:
    service = _verified_service(tmp_path, {"docs": ["search"]})
    extension = build_mcp_tool_extension(tmp_path, tmp_path, project_trusted=False, loaded_config=service.config)
    assert extension is not None
    assert await extension.tools["mcp__docs__search"]() == "ok"
    status = service.server_status(service.config.servers["docs"])
    assert status is not None
    service.status_store.update(
        "docs", status.model_copy(update={"tools": [MCPToolStatus(id="replacement")], "tool_count": 1})
    )

    with pytest.raises(ToolError, match="verified inventory"):
        await extension.tools["mcp__docs__search"]()
    assert transport_calls == [("docs", "search", {})]
    assert await service.call_tool("docs", "replacement", {}) == "ok"
    assert transport_calls[-1] == ("docs", "replacement", {})


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["repeated-cursor", "page-limit", "tool-limit"])
async def test_discovery_rejects_unbounded_pagination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    service = _verified_service(tmp_path, {"docs": []})
    monkeypatch.setattr("kolega_code.mcp.service._MCP_MAX_TOOL_PAGES", 2)
    monkeypatch.setattr("kolega_code.mcp.service._MCP_MAX_DISCOVERED_TOOLS", 2)

    class UnboundedSession:
        calls = 0

        async def list_tools(self, **kwargs: Any) -> SimpleNamespace:
            self.calls += 1
            cursor = "same" if failure == "repeated-cursor" else f"page-{self.calls}"
            tools = [object(), object(), object()] if failure == "tool-limit" else []
            return SimpleNamespace(tools=tools, next_cursor=cursor)

    session = UnboundedSession()
    with pytest.raises(ToolError, match="cursor|limit"):
        await service._list_all_tools(session)
    assert session.calls <= 2


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["unknown", "disabled", "missing", "failed", "stale"])
async def test_invalid_server_status_rejects_calls_before_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    service = _verified_service(tmp_path, {"docs": ["search"]})
    server = service.config.servers["docs"]
    if state == "unknown":
        service.config.servers.clear()
    elif state == "disabled":
        server.enabled = False
    elif state == "missing":
        service.status_store.clear("docs")
    else:
        status = service.server_status(server)
        assert status is not None
        changes = {"status": "failed"} if state == "failed" else {"fingerprint": "old-config"}
        service.status_store.update("docs", status.model_copy(update=changes))

    def unexpected_transport(*args: Any, **kwargs: Any) -> None:
        pytest.fail("invalid server status opened an MCP transport")

    monkeypatch.setattr("kolega_code.mcp.service.open_mcp_session", unexpected_transport)
    with pytest.raises(ToolError):
        await service.call_tool("docs", "search", {})
