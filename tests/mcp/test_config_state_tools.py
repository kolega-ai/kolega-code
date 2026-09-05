from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from kolega_code.mcp.config import (
    LoadedMCPConfig,
    MCPConfigFile,
    MCPOAuthConfig,
    MCPServerConfig,
    global_mcp_config_path,
    load_mcp_config,
    mcp_secret_values,
    sanitize_mcp_server_id,
    server_fingerprint,
)
from kolega_code.mcp.service import (
    MCPService,
    exposed_tool_names_for,
    mcp_tool_name,
    mcp_tool_name_adjustment_note,
    mcp_tool_name_adjustments,
    sanitize_mcp_tool_name,
)
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


def test_sanitize_mcp_server_id_rewrites_over_long_ids_deterministically() -> None:
    long_id = "x" * 40
    rewritten = sanitize_mcp_server_id(long_id)
    assert len(rewritten) == 32
    assert rewritten.startswith("x" * 24)
    assert rewritten != long_id
    assert sanitize_mcp_server_id(long_id) == rewritten  # deterministic
    # Distinct long ids sharing a prefix stay distinct.
    other_long = "x" * 24 + "y" * 16
    assert sanitize_mcp_server_id(other_long) != rewritten
    # Within the limit: unchanged, including whitespace trimming.
    assert sanitize_mcp_server_id("short-id") == "short-id"
    assert sanitize_mcp_server_id("  short-id  ") == "short-id"
    # Invalid characters and empty ids still raise.
    with pytest.raises(ValueError):
        sanitize_mcp_server_id("bad.id")
    with pytest.raises(ValueError):
        sanitize_mcp_server_id("")


def test_load_mcp_config_rewrites_over_long_server_ids_with_diagnostic(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    project = tmp_path / "project"
    project.mkdir()
    long_id = "x" * 40
    _write_json(
        global_mcp_config_path(state_dir),
        {
            "schema_version": 1,
            "servers": [{"id": long_id, "transport": "streamable_http", "url": "https://docs.example/mcp"}],
        },
    )

    config = load_mcp_config(project, state_dir, project_trusted=False)
    rewritten = sanitize_mcp_server_id(long_id)
    assert set(config.servers) == {rewritten}
    assert any("exceeds 32 characters" in diagnostic and long_id in diagnostic for diagnostic in config.diagnostics)


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
    state_dir = tmp_path / "state"
    status_store = MCPStatusStore(state_dir)
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

    token_store = MCPOAuthTokenStore(state_dir)
    token_store.set_tokens("docs", {"access_token": "access", "refresh_token": "refresh", "id_token": "id"})
    token_store.set_client_info("docs", {"client_id": "client", "client_secret": "client-secret"})
    assert token_store.secret_values() == ["access", "refresh", "id", "client-secret"]
    assert not token_store.path.with_suffix(token_store.path.suffix + ".tmp").exists()

    if os.name == "posix":
        assert (state_dir.stat().st_mode & 0o777) == 0o700
        assert (status_store.path.stat().st_mode & 0o777) == 0o600
        assert (token_store.path.stat().st_mode & 0o777) == 0o600

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


def test_sanitize_mcp_tool_name_maps_invalid_chars_and_preserves_prefix() -> None:
    assert sanitize_mcp_tool_name("mcp__docs__search") == "mcp__docs__search"
    assert sanitize_mcp_tool_name("mcp__docs__get.file/path") == "mcp__docs__get_file_path"
    assert sanitize_mcp_tool_name("mcp__docs__naïve tool") == "mcp__docs__na_ve_tool"


def test_sanitize_mcp_tool_name_clamps_to_64_preserving_server_prefix() -> None:
    name = sanitize_mcp_tool_name(mcp_tool_name("docs", "x" * 200))
    assert len(name) == 64
    assert name.startswith("mcp__docs__")
    assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name) is not None


def test_sanitize_mcp_tool_name_truncates_whole_name_when_server_id_exhausts_budget() -> None:
    name = sanitize_mcp_tool_name(mcp_tool_name("s" * 100, "tool"))
    assert len(name) == 64
    assert name.startswith("mcp__")
    assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name) is not None


def test_exposed_tool_names_for_dedupes_collisions_deterministically() -> None:
    tools = [
        MCPToolStatus(id="get.file"),
        MCPToolStatus(id="get_file"),
        MCPToolStatus(id="get.file"),
    ]
    first = exposed_tool_names_for("docs", tools)
    second = exposed_tool_names_for("docs", tools)
    assert first == second

    finals = [final for _, final in first]
    assert len(finals) == len(set(finals)) == 3
    for final in finals:
        assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", final) is not None
    # The two distinct raw names hash differently, so their collision suffixes differ.
    assert finals[0] != finals[1]
    assert finals[0] != finals[2]  # raw duplicate falls back to the positional suffix

    raws = [raw for raw, _ in first]
    assert raws[0] == raws[2] == mcp_tool_name("docs", "get.file")
    assert raws[1] == mcp_tool_name("docs", "get_file")


def test_mcp_tool_name_adjustments_and_note() -> None:
    tools = [MCPToolStatus(id="search"), MCPToolStatus(id="get.file")]
    assert mcp_tool_name_adjustments("docs", tools) == [
        (mcp_tool_name("docs", "get.file"), mcp_tool_name("docs", "get_file"))
    ]
    note = mcp_tool_name_adjustment_note("docs", tools)
    assert "1 tool name was adjusted for provider compatibility" in note
    assert "mcp__docs__get.file → mcp__docs__get_file" in note
    assert mcp_tool_name_adjustment_note("docs", [MCPToolStatus(id="search")]) == ""


@pytest.mark.asyncio
async def test_build_mcp_tool_extension_sanitizes_and_dedupes_invalid_tool_names(tmp_path: Path, monkeypatch) -> None:
    state_dir = tmp_path / "state"
    project = tmp_path / "project"
    project.mkdir()
    server = MCPServerConfig(id="docs", transport="streamable_http", url="https://docs.example/mcp")
    _write_json(global_mcp_config_path(state_dir), MCPConfigFile(servers=[server]).to_file_dict())
    config = load_mcp_config(project, state_dir, project_trusted=False)
    tool_statuses = [
        MCPToolStatus(id="get.file", description="Get a file", input_schema={"type": "object"}),
        MCPToolStatus(id="get_file", description="Get a file (underscore)", input_schema={"type": "object"}),
        MCPToolStatus(id="x" * 200, description="Long name", input_schema={"type": "object"}),
    ]
    MCPStatusStore(state_dir).update(
        "docs",
        MCPServerStatus.verified(
            fingerprint=server_fingerprint(config.servers["docs"]),
            transport="streamable_http",
            source="global",
            tools=tool_statuses,
        ),
    )

    calls = []

    async def fake_call_tool(self, server_id: str, tool_id: str, arguments: dict):
        calls.append((server_id, tool_id, arguments))
        return "ok"

    monkeypatch.setattr(MCPService, "call_tool", fake_call_tool)

    extension = build_mcp_tool_extension(project, state_dir, project_trusted=False)
    assert extension is not None
    names = list(extension.tools)
    assert len(names) == len(set(names)) == 3
    for name in names:
        assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name) is not None
    # Every exposed name routes back to its original server/tool ids.
    for name in names:
        assert await extension.tools[name]() == "ok"
    assert {call[1] for call in calls} == {"get.file", "get_file", "x" * 200}
    # Renamed tools tell the model the server-side name.
    assert any("The server-side tool name is `get.file`" in desc for desc in extension.tool_descriptions.values())
    assert any("The server-side tool name is `get_file`" in desc for desc in extension.tool_descriptions.values())
    assert any(
        "The server-side tool name is `" + "x" * 200 + "`" in desc for desc in extension.tool_descriptions.values()
    )


@pytest.mark.asyncio
async def test_verify_server_message_notes_adjusted_tool_names(tmp_path: Path, monkeypatch) -> None:
    state_dir = tmp_path / "state"
    project = tmp_path / "project"
    project.mkdir()
    server = MCPServerConfig(id="docs", transport="streamable_http", url="https://docs.example/mcp")
    _write_json(global_mcp_config_path(state_dir), MCPConfigFile(servers=[server]).to_file_dict())
    config = load_mcp_config(project, state_dir, project_trusted=False)
    service = MCPService(config, state_dir=state_dir, project_path=project)

    class FakeTool:
        name = "get.file"
        title = None
        description = "Get a file"
        input_schema = {"type": "object", "properties": {}}

    class FakeSession:
        async def list_tools(self, params=None):
            return type("Result", (), {"tools": [FakeTool()], "next_cursor": None})()

    class FakeSessionContext:
        async def __aenter__(self):
            return FakeSession()

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr("kolega_code.mcp.service.open_mcp_session", lambda *a, **k: FakeSessionContext())

    result = await service.verify_server(server.id, interactive_oauth=False, open_browser=False)
    assert result.ok
    assert "Verified 1 tool(s)." in result.message
    assert (
        "1 tool name was adjusted for provider compatibility (mcp__docs__get.file → mcp__docs__get_file)."
        in result.message
    )


def test_mcp_oauth_config_defaults_and_validation() -> None:
    # Defaults
    oauth = MCPOAuthConfig()
    assert oauth.enabled is False
    assert oauth.client_id is None
    assert oauth.client_secret is None
    assert oauth.client_secret_env is None
    assert oauth.redirect_uri is None
    assert oauth.token_endpoint_auth_method is None
    assert oauth.resolve_auth_method() == "none"

    # Auto-enabling when credentials are provided
    oauth_with_id = MCPOAuthConfig(client_id="my-client")
    assert oauth_with_id.enabled is True
    assert oauth_with_id.resolve_auth_method() == "none"

    oauth_with_secret = MCPOAuthConfig(client_id="my-client", client_secret="sec")
    assert oauth_with_secret.enabled is True
    assert oauth_with_secret.resolve_auth_method() == "client_secret_post"

    # Explicit auth method
    oauth_basic = MCPOAuthConfig(
        client_id="my-client", client_secret="sec", token_endpoint_auth_method="client_secret_basic"
    )
    assert oauth_basic.resolve_auth_method() == "client_secret_basic"

    # Valid redirect URI
    oauth_valid_uri = MCPOAuthConfig(redirect_uri="http://127.0.0.1:33418/callback")
    assert oauth_valid_uri.redirect_uri == "http://127.0.0.1:33418/callback"
    oauth_valid_localhost = MCPOAuthConfig(redirect_uri="http://localhost:8080/callback")
    assert oauth_valid_localhost.redirect_uri == "http://localhost:8080/callback"

    # Invalid redirect URIs
    with pytest.raises(ValueError, match="localhost http URL"):
        MCPOAuthConfig(redirect_uri="https://example.com/callback")
    with pytest.raises(ValueError, match="localhost http URL"):
        MCPOAuthConfig(redirect_uri="http://example.com:8080/callback")

    # Invalid env var name
    with pytest.raises(ValueError, match="valid environment variable name"):
        MCPOAuthConfig(client_secret_env="invalid-name!")


def test_mcp_oauth_config_secret_resolution_and_redaction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_MCP_SECRET_ENV", "env-secret-val-123")

    oauth_direct = MCPOAuthConfig(client_id="cid", client_secret="direct-secret-456")
    assert oauth_direct.resolve_client_secret() == "direct-secret-456"

    oauth_env = MCPOAuthConfig(client_id="cid", client_secret_env="TEST_MCP_SECRET_ENV")
    assert oauth_env.resolve_client_secret() == "env-secret-val-123"

    # Server sanitization masks direct secret but shows env var name
    server = MCPServerConfig(
        id="hubspot",
        transport="streamable_http",
        url="https://mcp.hubspot.com",
        oauth=MCPOAuthConfig(
            client_id="cid",
            client_secret="direct-secret-456",
            client_secret_env="TEST_MCP_SECRET_ENV",
            redirect_uri="http://127.0.0.1:33418/callback",
        ),
    )
    sanitized = server.sanitized_for_display()
    assert sanitized["oauth"]["client_secret"] == "‹secret›"
    assert sanitized["oauth"]["client_id"] == "cid"
    assert sanitized["oauth"]["client_secret_env"] == "TEST_MCP_SECRET_ENV"

    # mcp_secret_values collects the resolved secrets for diagnostics redaction
    config = LoadedMCPConfig(servers={"hubspot": server})
    secrets = mcp_secret_values(config)
    assert "direct-secret-456" in secrets


def test_server_fingerprint_updates_on_oauth_changes() -> None:
    server = MCPServerConfig(
        id="hubspot",
        transport="streamable_http",
        url="https://mcp.hubspot.com",
        oauth=MCPOAuthConfig(enabled=True),
    )
    fp_base = server_fingerprint(server)

    server_with_client = server.model_copy(update={"oauth": MCPOAuthConfig(enabled=True, client_id="cid-1")})
    assert server_fingerprint(server_with_client) != fp_base

    server_with_redirect = server_with_client.model_copy(
        update={
            "oauth": MCPOAuthConfig(
                enabled=True,
                client_id="cid-1",
                redirect_uri="http://127.0.0.1:33418/callback",
            )
        }
    )
    assert server_fingerprint(server_with_redirect) != server_fingerprint(server_with_client)
