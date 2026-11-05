from __future__ import annotations

import re
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from acp.exceptions import RequestError
from acp.schema import HttpMcpServer, McpServerStdio, SseMcpServer

from kolega_code.acp.mcp import build_session_mcp_extension, merge_mcp_config, normalize_client_servers
from kolega_code.mcp.config import LoadedMCPConfig, MCPServerConfig, sanitize_mcp_server_id, server_fingerprint
from kolega_code.mcp.service import MCPService, exposed_tool_names
from kolega_code.mcp.state import MCPServerStatus, MCPStatusStore, MCPToolStatus
from kolega_code.mcp.tools import build_mcp_tool_extension
from kolega_code.permissions import ProjectPermissionStore, allow_rule_options, permission_request_for_tool

pytestmark = pytest.mark.usefixtures("isolated_cli_env")


def _client(name: str, transport: str = "stdio") -> dict[str, Any]:
    if transport == "stdio":
        return {
            "name": name,
            "command": "fake-editor-mcp",
            "args": ["--token", "fake-argument-secret"],
            "env": [{"name": "TOKEN", "value": "fake-env-secret"}],
        }
    return {
        "name": name,
        "type": transport,
        "url": "https://example.invalid/mcp?token=fake-url-secret",
        "headers": [{"name": "Authorization", "value": "Bearer fake-header-secret"}],
    }


def _saved(server_id: str) -> MCPServerConfig:
    return MCPServerConfig(id=server_id, transport="stdio", command="fake-saved-mcp")


def _verify(state_dir: Path, server: MCPServerConfig, tool_ids: list[str]) -> None:
    MCPStatusStore(state_dir).update(
        server.id,
        MCPServerStatus.verified(
            fingerprint=server_fingerprint(server),
            transport=server.transport,
            source=server.source,
            tools=[MCPToolStatus(id=tool_id) for tool_id in tool_ids],
        ),
    )


@dataclass
class _Transport:
    inventories: dict[str, list[str]] = field(default_factory=dict)
    discoveries: list[MCPServerConfig] = field(default_factory=list)
    calls: list[tuple[MCPServerConfig, str, dict[str, Any]]] = field(default_factory=list)


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch) -> _Transport:
    transport = _Transport()

    @asynccontextmanager
    async def open_session(server: MCPServerConfig, **kwargs: Any) -> AsyncGenerator[SimpleNamespace, None]:
        async def list_tools() -> SimpleNamespace:
            transport.discoveries.append(server)
            return SimpleNamespace(
                tools=[
                    SimpleNamespace(
                        name=tool_id,
                        description=f"Tool {tool_id}",
                        inputSchema={"type": "object", "properties": {"query": {"type": "string"}}},
                    )
                    for tool_id in transport.inventories.get(server.display_name, ["inspect.file"])
                ]
            )

        async def call_tool(tool_id: str, arguments: dict[str, Any]) -> SimpleNamespace:
            transport.calls.append((server, tool_id, arguments))
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")])

        yield SimpleNamespace(list_tools=list_tools, call_tool=call_tool)

    monkeypatch.setattr("kolega_code.mcp.service.open_mcp_session", open_session)
    return transport


@pytest.mark.parametrize("transport", ["stdio", "http", "sse"])
@pytest.mark.parametrize("sdk_model", [False, True])
def test_arbitrary_editor_names_have_distinct_valid_opaque_identities(
    tmp_path: Path, transport: str, sdk_model: bool
) -> None:
    long_name = "long" * 40
    names = [
        "docs",
        " docs ",
        "docs/a",
        "docs_a",
        "docs.a",
        "docs__admin",
        "docs_",
        "文档 🔧",
        "",
        "line\nbreak",
        long_name,
        sanitize_mcp_server_id(long_name),
    ]
    model = {"stdio": McpServerStdio, "http": HttpMcpServer, "sse": SseMcpServer}[transport]
    payloads = [_client(name, transport) for name in names]
    entries = [model.model_validate(payload) for payload in payloads] if sdk_model else payloads
    clients = normalize_client_servers(entries, tmp_path)
    assert list(clients) == names
    assert len({server.id for server in clients.values()}) == len(names)
    for name, server in clients.items():
        assert server.name == name
        assert re.fullmatch(r"[0-9a-f]{32}", server.id)
        assert sanitize_mcp_server_id(server.id) == server.id
        assert server.id not in names
        assert server.source == "acp"
        assert server.transport == {"stdio": "stdio", "http": "streamable_http", "sse": "sse"}[transport]
        if transport == "stdio":
            assert server.cwd == str(tmp_path)
            assert server.env == {"TOKEN": "fake-env-secret"}
            assert server.args == ["--token", "fake-argument-secret"]
        else:
            assert server.headers == {"Authorization": "Bearer fake-header-secret"}
            assert server.url == payloads[0]["url"]
    other_session = normalize_client_servers(entries, tmp_path)
    assert {server.id for server in clients.values()}.isdisjoint(server.id for server in other_session.values())


def test_duplicate_original_names_replace_without_aliasing_distinct_names(tmp_path: Path) -> None:
    first = _client("docs/a")
    replacement = {**first, "command": "fake-replacement"}
    clients = normalize_client_servers([first, _client("docs_a"), replacement], tmp_path)
    assert list(clients) == ["docs/a", "docs_a"]
    assert clients["docs/a"].command == "fake-replacement"
    assert clients["docs/a"].id != clients["docs_a"].id


def test_random_identity_collision_is_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    identities = iter(["0" * 32, "0" * 32, "1" * 32])
    monkeypatch.setattr("kolega_code.acp.mcp.secrets.token_hex", lambda size: next(identities))
    clients = normalize_client_servers([_client("first"), _client("second")], tmp_path)
    assert [server.id for server in clients.values()] == ["0" * 32, "1" * 32]


@pytest.mark.asyncio
async def test_conflicting_runtime_identity_cannot_replace_a_configured_route(
    tmp_path: Path, transport: _Transport
) -> None:
    clients = normalize_client_servers([_client("editor")], tmp_path)
    saved = _saved(clients["editor"].id)
    configured = LoadedMCPConfig(servers={saved.id: saved})
    merged = merge_mcp_config(configured, clients)
    with pytest.raises(RequestError) as caught:
        await build_session_mcp_extension(merged, tmp_path, tmp_path / "state")
    assert caught.value.data == {"message": "Conflicting ACP MCP server identities."}
    assert configured.servers[saved.id] is saved
    assert not transport.discoveries
    assert not transport.calls


@pytest.mark.asyncio
async def test_colliding_editor_names_discover_and_route_independently(tmp_path: Path, transport: _Transport) -> None:
    long_name = "long" * 40
    names = ["docs/a", "docs_a", " docs ", "docs", long_name, sanitize_mcp_server_id(long_name), "文档"]
    tool_ids = ["get.file", "get_file", "z" * 90 + "a", "z" * 90 + "b"]
    transport.inventories = dict.fromkeys(names, tool_ids)
    clients = normalize_client_servers([_client(name) for name in names], tmp_path)
    state = tmp_path / "state"
    extension = await build_session_mcp_extension(merge_mcp_config(None, clients), tmp_path, state)
    assert extension is not None
    try:
        assert len(extension.tools) == len(names) * len(tool_ids)
        assert len({server.id for server in transport.discoveries}) == len(names)
        routes = set()
        for name, callback in extension.tools.items():
            assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name)
            assert await callback(_server_id="docs", _tool_id="unverified") == "ok"
            server, tool_id, arguments = transport.calls[-1]
            request = permission_request_for_tool(name, arguments)
            assert request is not None
            assert request.mcp_server == server.id
            routes.add((server.id, tool_id))
        assert routes == {(server.id, tool_id) for server in clients.values() for tool_id in tool_ids}
        assert not state.exists()
    finally:
        assert extension.cleanup is not None
        await extension.cleanup()


@pytest.mark.asyncio
async def test_editor_discovery_never_reuses_persisted_status(tmp_path: Path, transport: _Transport) -> None:
    clients = normalize_client_servers([_client("docs")], tmp_path)
    server = clients["docs"]
    state = tmp_path / "state"
    _verify(state, server, ["persisted-only"])
    before = MCPStatusStore(state).path.read_bytes()
    extension = await build_session_mcp_extension(merge_mcp_config(None, clients), tmp_path, state)
    assert extension is not None
    try:
        assert transport.discoveries == [server]
        assert len(extension.tools) == 1
        name = next(iter(extension.tools))
        assert "persisted-only" not in name
        assert await extension.tools[name]() == "ok"
        assert transport.calls == [(server, "inspect.file", {})]
        assert MCPStatusStore(state).path.read_bytes() == before
    finally:
        assert extension.cleanup is not None
        await extension.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("server_id", ["docs", "docs_", "docs__admin", "long" * 40])
@pytest.mark.parametrize("client_transport", ["stdio", "http", "sse"])
async def test_saved_permission_grants_cannot_authorize_editor_override(
    tmp_path: Path, transport: _Transport, server_id: str, client_transport: str
) -> None:
    state = tmp_path / "state"
    saved = _saved(server_id)
    keep = _saved("keep")
    configured = LoadedMCPConfig(servers={saved.id: saved, keep.id: keep}, project_trusted=True)
    _verify(state, saved, ["inspect.file"])
    _verify(state, keep, ["read"])
    before = MCPStatusStore(state).path.read_bytes()
    original = await build_session_mcp_extension(configured, tmp_path, state)
    assert original is not None
    assert not transport.discoveries
    original_name = next(iter(original.tools))
    request = permission_request_for_tool(original_name, {})
    assert request is not None
    grants = ProjectPermissionStore(tmp_path)
    for option in allow_rule_options(request):
        grants.add_rule(option.rule)
    assert grants.first_match(request) is not None
    clients = normalize_client_servers([_client(saved.id, client_transport)], tmp_path)
    merged = merge_mcp_config(configured, clients)
    assert merged.servers[saved.id] is clients[saved.id]
    assert merged.servers["keep"] is keep
    assert configured.servers[saved.id] is saved
    assert merged.project_trusted
    extension = await build_session_mcp_extension(merged, tmp_path, state)
    assert extension is not None
    try:
        assert transport.discoveries == [clients[saved.id]]
        editor_name = next(name for name in extension.tools if name != "mcp__keep__read")
        assert editor_name != original_name
        editor_request = permission_request_for_tool(editor_name, {})
        assert editor_request is not None
        assert editor_request.mcp_server == clients[saved.id].id
        assert grants.first_match(editor_request) is None
        assert await extension.tools[editor_name](query="hello", _server_id="keep", _tool_id="read") == "ok"
        assert transport.calls[-1] == (
            clients[saved.id],
            "inspect.file",
            {"query": "hello", "_server_id": "keep", "_tool_id": "read"},
        )
        assert await extension.tools["mcp__keep__read"]() == "ok"
        assert transport.calls[-1] == (keep, "read", {})
        assert MCPStatusStore(state).path.read_bytes() == before
        assert list(state.iterdir()) == [MCPStatusStore(state).path]
    finally:
        assert original.cleanup is not None and extension.cleanup is not None
        await original.cleanup()
        await extension.cleanup()


@pytest.mark.asyncio
async def test_builds_keep_original_lookup_and_identity_while_replacements_get_fresh_identity(
    tmp_path: Path, transport: _Transport
) -> None:
    saved = _saved("docs")
    state = tmp_path / "state"
    _verify(state, saved, ["saved-only"])
    configured = LoadedMCPConfig(servers={"docs": saved})
    payloads = [_client("docs"), _client("editor/server"), _client("editor_server")]
    clients = normalize_client_servers(payloads, tmp_path)
    merged = merge_mcp_config(configured, clients)
    expected_ids = {name: server.id for name, server in clients.items()}
    extensions = []
    try:
        for config in [merged, merge_mcp_config(configured, clients), merged]:
            extension = await build_session_mcp_extension(config, tmp_path, state)
            assert extension is not None
            extensions.append(extension)
            assert list(config.servers) == list(clients)
            assert all(config.servers[name] is clients[name] for name in clients)
            assert {name: server.id for name, server in clients.items()} == expected_ids
            for callback in extension.tools.values():
                assert await callback(server_id="untrusted", tool_id="unverified") == "ok"
        assert all(list(extension.tools) == list(extensions[0].tools) for extension in extensions)
        assert len(transport.discoveries) == 3 * len(clients)
        assert {server.id for server, _, _ in transport.calls} == set(expected_ids.values())
        old_request = permission_request_for_tool(next(iter(extensions[0].tools)), {})
        assert old_request is not None
        old_grant = allow_rule_options(old_request)[1].rule
        replacement_clients = normalize_client_servers(payloads, tmp_path)
        replacement = await build_session_mcp_extension(
            merge_mcp_config(configured, replacement_clients), tmp_path, state
        )
        assert replacement is not None
        extensions.append(replacement)
        assert set(replacement.tools).isdisjoint(extensions[0].tools)
        for name in replacement.tools:
            request = permission_request_for_tool(name, {})
            assert request is not None
            assert not old_grant.matches(request)
        restored = await build_session_mcp_extension(merge_mcp_config(configured, {}), tmp_path, state)
        assert restored is not None
        extensions.append(restored)
        assert list(restored.tools) == ["mcp__docs__saved-only"]
    finally:
        for extension in extensions:
            assert extension.cleanup is not None
            await extension.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "inventories",
    [
        {"x": ["y__z"], "x__y": ["z"]},
        {"x": ["y__a.b"], "x__y": ["a_b"]},
        {"x": ["y__" + "z" * 80], "x__y": ["z" * 80]},
        {"docs": ["admin__search"], "docs__admin": ["search"], "docs_": ["search"]},
        {"x": ["first.name", "first_name", "z" * 90 + "a", "z" * 90 + "b"]},
    ],
    ids=["separator", "sanitization", "truncation", "permission-ambiguity", "within-server"],
)
async def test_acp_uses_shared_names_and_keeps_callbacks_pinned(
    tmp_path: Path, transport: _Transport, inventories: dict[str, list[str]]
) -> None:
    state = tmp_path / "state"
    servers = {server_id: _saved(server_id) for server_id in inventories}
    for server_id, tool_ids in inventories.items():
        _verify(state, servers[server_id], tool_ids)
    config = LoadedMCPConfig(servers=servers)
    service = MCPService(config, state_dir=state, project_path=tmp_path)
    exposed = service.exposed_tools()
    expected_names = [name for _, name in exposed_tool_names(exposed)]
    shared = build_mcp_tool_extension(tmp_path, state, project_trusted=False, loaded_config=config)
    extension = await build_session_mcp_extension(config, tmp_path, state)
    rebuilt = await build_session_mcp_extension(
        LoadedMCPConfig(servers=dict(reversed(list(servers.items())))), tmp_path, state
    )
    assert shared is not None and extension is not None and rebuilt is not None
    try:
        assert list(extension.tools) == expected_names == list(shared.tools)
        assert extension.tool_groups == shared.tool_groups
        assert extension.tool_descriptions == shared.tool_descriptions
        assert extension.tool_schemas == shared.tool_schemas
        assert extension.propagate_to_sub_agents is False
        assert len(set(expected_names)) == len(exposed)
        assert set(rebuilt.tools) == set(expected_names)
        for item, name in zip(exposed, expected_names, strict=True):
            assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name)
            inputs = {"_server_id": "untrusted", "_tool_id": "delete", "server_id": "other", "query": name}
            assert await extension.tools[name](**inputs) == "ok"
            assert transport.calls[-1] == (item.server, item.tool.id, inputs)
            assert await rebuilt.tools[name](**inputs) == "ok"
            assert transport.calls[-1] == (item.server, item.tool.id, inputs)
            if item.server.id in {"docs_", "docs__admin", "x__y"}:
                request = permission_request_for_tool(name, inputs)
                assert request is not None
                assert request.mcp_server not in {"docs", "x", item.server.id}
        assert not transport.discoveries
    finally:
        for built in [shared, extension, rebuilt]:
            assert built.cleanup is not None
            await built.cleanup()


@pytest.mark.asyncio
async def test_shared_generated_suffix_cannot_take_a_natural_name(tmp_path: Path, transport: _Transport) -> None:
    state = tmp_path / "state"
    server = _saved("docs")
    tools = ["first.name", "first_name"]
    _verify(state, server, tools)
    config = LoadedMCPConfig(servers={server.id: server})
    initial = await build_session_mcp_extension(config, tmp_path, state)
    assert initial is not None
    reserved = next(iter(initial.tools))
    natural_tool_id = reserved.removeprefix("mcp__docs__")
    _verify(state, server, [*tools, natural_tool_id])
    rebuilt = await build_session_mcp_extension(config, tmp_path, state)
    assert rebuilt is not None
    try:
        assert len(set(rebuilt.tools)) == 3
        assert list(rebuilt.tools)[-1] == reserved
        assert await rebuilt.tools[reserved]() == "ok"
        assert transport.calls[-1] == (server, natural_tool_id, {})
    finally:
        assert initial.cleanup is not None and rebuilt.cleanup is not None
        await initial.cleanup()
        await rebuilt.cleanup()


def test_invalid_transport_errors_remain_credential_safe(tmp_path: Path) -> None:
    for payload in [
        {**_client("arbitrary/name"), "command": ""},
        {**_client("arbitrary/name", "http"), "headers": "fake-private-secret"},
        {**_client("arbitrary/name"), "type": "fake-private-secret"},
    ]:
        with pytest.raises(RequestError) as caught:
            normalize_client_servers([payload], tmp_path)
        assert caught.value.data == {"message": "Invalid ACP MCP server configuration."}
        assert "fake-" not in str(caught.value)
        assert caught.value.__suppress_context__
