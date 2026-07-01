from __future__ import annotations

import json
from pathlib import Path

import pytest

from kolega_code.mcp.config import (
    MCPConfigFile,
    MCPServerConfig,
    global_mcp_config_path,
    load_mcp_config,
    mcp_secret_values,
    server_fingerprint,
)
from kolega_code.mcp.service import MCPService, mcp_tool_name
from kolega_code.mcp.state import MCPServerStatus, MCPStatusStore, MCPToolStatus, MCPOAuthTokenStore
from kolega_code.mcp.tools import build_mcp_tool_extension


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_mcp_config_merges_global_and_trusted_project(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    project = tmp_path / "project"
    project.mkdir()
    _write_json(
        global_mcp_config_path(state_dir),
        {
            "schema_version": 1,
            "servers": [
                {
                    "id": "global",
                    "transport": "streamable_http",
                    "url": "https://global.example/mcp",
                    "headers": {"Authorization": "Bearer global-secret"},
                },
                {"id": "shared", "transport": "sse", "url": "https://global.example/sse"},
            ],
        },
    )
    _write_json(
        project / ".kolega" / "mcp_servers.json",
        {
            "schema_version": 1,
            "servers": [
                {"id": "project", "transport": "stdio", "command": "python", "args": ["server.py"]},
                {"id": "shared", "transport": "streamable_http", "url": "https://project.example/mcp"},
            ],
        },
    )

    untrusted = load_mcp_config(project, state_dir, project_trusted=False)
    assert set(untrusted.servers) == {"global", "shared"}
    assert untrusted.project_config_present is True
    assert any("not trusted" in diagnostic for diagnostic in untrusted.diagnostics)

    trusted = load_mcp_config(project, state_dir, project_trusted=True)
    assert set(trusted.servers) == {"global", "project", "shared"}
    assert trusted.servers["shared"].source == "project"
    assert trusted.servers["project"].command == "python"
    assert mcp_secret_values(trusted) == ["Bearer global-secret"]


def test_server_fingerprint_ignores_enabled_and_source_but_not_connection_details() -> None:
    server = MCPServerConfig(
        id="docs",
        transport="streamable_http",
        url="https://docs.example/mcp",
        enabled=True,
        source="global",
    )
    same_connection = server.model_copy(update={"enabled": False, "source": "project"})
    changed_connection = server.model_copy(update={"headers": {"Authorization": "Bearer token"}})

    assert server_fingerprint(server) == server_fingerprint(same_connection)
    assert server_fingerprint(server) != server_fingerprint(changed_connection)


def test_status_and_oauth_token_stores_round_trip_and_redact(tmp_path: Path) -> None:
    status_store = MCPStatusStore(tmp_path)
    status = MCPServerStatus.verified(
        fingerprint="fp",
        transport="streamable_http",
        source="global",
        tools=[MCPToolStatus(id="search", description="Search", input_schema={"type": "object"})],
        oauth=True,
    )
    status_store.update("docs", status)

    loaded = status_store.get("docs")
    assert loaded is not None
    assert loaded.status == "verified"
    assert loaded.tool_count == 1
    assert loaded.tools[0].id == "search"

    token_store = MCPOAuthTokenStore(tmp_path)
    token_store.set_tokens("docs", {"access_token": "access", "refresh_token": "refresh", "id_token": "id"})
    token_store.set_client_info("docs", {"client_id": "client", "client_secret": "client-secret"})
    assert token_store.secret_values() == ["access", "refresh", "id", "client-secret"]

    token_store.clear("docs")
    status_store.clear("docs")
    assert token_store.secret_values() == []
    assert status_store.get("docs") is None


@pytest.mark.asyncio
async def test_build_mcp_tool_extension_exposes_verified_tools_and_uses_schema(tmp_path: Path, monkeypatch) -> None:
    state_dir = tmp_path / "state"
    project = tmp_path / "project"
    project.mkdir()
    server = MCPServerConfig(id="docs", transport="streamable_http", url="https://docs.example/mcp")
    # The extension builder loads from disk, so write the global file after constructing the server.
    _write_json(global_mcp_config_path(state_dir), MCPConfigFile(servers=[server]).to_file_dict())
    config = load_mcp_config(project, state_dir, project_trusted=False)
    tool_status = MCPToolStatus(
        id="search",
        description="Search docs",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    )
    MCPStatusStore(state_dir).update(
        "docs",
        MCPServerStatus.verified(
            fingerprint=server_fingerprint(config.servers["docs"]),
            transport="streamable_http",
            source="global",
            tools=[tool_status],
        ),
    )

    calls = []

    async def fake_call_tool(self, server_id: str, tool_id: str, arguments: dict):
        calls.append((server_id, tool_id, arguments))
        return "ok"

    monkeypatch.setattr(MCPService, "call_tool", fake_call_tool)

    extension = build_mcp_tool_extension(project, state_dir, project_trusted=False)
    assert extension is not None
    assert extension.propagate_to_sub_agents is False
    name = mcp_tool_name("docs", "search")
    assert set(extension.tools) == {name}
    assert extension.tool_schemas[name] == tool_status.input_schema

    assert await extension.tools[name](query="kolega") == "ok"
    assert calls == [("docs", "search", {"query": "kolega"})]
