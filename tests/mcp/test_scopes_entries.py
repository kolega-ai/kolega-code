from __future__ import annotations

import contextlib
import io
import json
import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from kolega_code.cli.settings import CliSettings, SettingsStore
from kolega_code.mcp import config as config_module
from kolega_code.mcp.config import (
    LoadedMCPConfig,
    MCPConfigError,
    MCPOAuthConfig,
    MCPServerConfig,
    global_mcp_config_path,
    load_mcp_config,
    load_mcp_config_for,
    project_mcp_config_path,
    remove_server_config,
    remove_standard_server,
    sanitize_mcp_server_id,
    server_from_entry,
    server_fingerprint,
    standard_mcp_config_path,
    upsert_server_config,
    upsert_standard_server,
)
from kolega_code.mcp.entries import (
    MCPEnvReferenceError,
    MCPEntryError,
    UnsupportedMCPTypeError,
    env_references,
    expand_env_references,
    missing_env_references,
    standard_entry_payload,
)
from kolega_code.mcp.failure_messages import (
    MCP_FAILURE_MESSAGE_ENV_MISSING,
    MCP_FAILURE_MESSAGE_GENERIC,
    MCP_FAILURE_MESSAGE_OAUTH_SIGNIN_INCOMPLETE,
    MCP_FAILURE_MESSAGE_OAUTH_UNAUTHORIZED,
    MCP_FAILURE_MESSAGE_TIMEOUT,
)
from kolega_code.mcp.health import HEALTH_LABELS, relative_time, server_health
from kolega_code.mcp.oauth import MCPOAuthCallbackTimeout
from kolega_code.mcp.scopes import (
    find_definitions,
    has_literal_secrets,
    project_server_override,
    project_write_target,
    remove_server,
    reset_project_choices,
    scope_location,
    set_server_enabled_for_scope,
    upsert_server,
)
from kolega_code.mcp.service import MCPService, _mcp_failure_message
from kolega_code.mcp.state import MCPOAuthTokenStore, MCPServerStatus, MCPStatusStore, MCPToolStatus
from kolega_code.mcp.transport import open_mcp_session


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _http_server(server_id: str, url: str = "https://example.com/mcp", **fields: Any) -> MCPServerConfig:
    return MCPServerConfig(id=server_id, transport="streamable_http", url=url, **fields)


def _verified(server: MCPServerConfig, tool_ids: tuple[str, ...] = ("ping",)) -> MCPServerStatus:
    return MCPServerStatus.verified(
        fingerprint=server_fingerprint(server),
        transport=server.transport,
        source=server.source,
        tools=[MCPToolStatus(id=tool_id) for tool_id in tool_ids],
    )


def test_higher_precedence_scopes_win_whole_and_name_what_they_shadow(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    state = tmp_path / "state"
    key = str(project.resolve())
    _write(
        global_mcp_config_path(state),
        {
            "schema_version": 1,
            "servers": [{"id": "shared", "transport": "streamable_http", "url": "https://user.example/mcp"}],
        },
    )
    _write(
        standard_mcp_config_path(project),
        {"mcpServers": {"shared": {"type": "http", "url": "https://standard.example/mcp"}}},
    )
    _write(
        project_mcp_config_path(project),
        {
            "schema_version": 1,
            "servers": [{"id": "shared", "transport": "streamable_http", "url": "https://legacy.example/mcp"}],
        },
    )
    settings = CliSettings()
    settings.trust_mcp_project(project)
    settings.mcp_local_servers[key] = {"shared": {"transport": "streamable_http", "url": "https://local.example/mcp"}}

    config = load_mcp_config_for(project, state, settings)

    assert config.servers["shared"].url == "https://local.example/mcp"
    assert config.servers["shared"].source == "local"
    assert "shared" not in config.server_files
    shadows = [diagnostic for diagnostic in config.diagnostics if "overrides the definition in" in diagnostic]
    assert shadows == [
        "MCP server 'shared' from the project .mcp.json file overrides the definition in the user config.",
        "MCP server 'shared' from the project .kolega/mcp_servers.json file overrides the definition in the project .mcp.json file.",
        "MCP server 'shared' from the local scope overrides the definition in the project .kolega/mcp_servers.json file.",
    ]
    assert len(shadows) == 3

    settings.mcp_local_servers.clear()
    legacy = load_mcp_config_for(project, state, settings).servers["shared"]
    assert legacy.url == "https://legacy.example/mcp"
    assert load_mcp_config_for(project, state, settings).server_files["shared"] == project_mcp_config_path(project)

    project_mcp_config_path(project).unlink()
    config = load_mcp_config_for(project, state, settings)
    assert config.servers["shared"].url == "https://standard.example/mcp"
    assert config.server_files["shared"] == standard_mcp_config_path(project)

    standard_mcp_config_path(project).unlink()
    assert load_mcp_config_for(project, state, settings).servers["shared"].url == "https://user.example/mcp"


@pytest.mark.asyncio
async def test_untrusted_project_servers_are_listed_for_review_but_never_loaded(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    state = tmp_path / "state"
    _write(
        standard_mcp_config_path(project),
        {"mcpServers": {"a": {"command": "npx"}, "b": {"type": "http", "url": "https://b.example/mcp"}}},
    )

    config = load_mcp_config(project, state, project_trusted=False)

    assert [server.id for server in config.untrusted_project_servers] == ["a", "b"]
    assert config.servers == {}
    assert config.enabled_servers == []
    assert config.project_config_present is True
    assert any("not trusted for MCP" in diagnostic for diagnostic in config.diagnostics)

    # A verification left over from when the project was trusted exposes nothing.
    stored = config.untrusted_project_servers[1]
    MCPStatusStore(state).update("b", _verified(stored))
    service = MCPService(config, state_dir=state, project_path=project)
    assert service.exposed_tools() == []
    assert (await service.verify_server("b")).ok is False
    rows = service.list_status_rows()
    assert [row["health"].health for row in rows] == ["pending_trust", "pending_trust"]
    assert [row["scope"] for row in rows] == ["project", "project"]


def test_project_choices_override_only_project_definitions(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    state = tmp_path / "state"
    _write(
        global_mcp_config_path(state),
        {"schema_version": 1, "servers": [_http_server("mine").model_dump(mode="json", exclude={"source"})]},
    )
    _write(
        standard_mcp_config_path(project),
        {
            "mcpServers": {
                "team": {"type": "http", "url": "https://team.example/mcp"},
                "off": {"type": "http", "url": "https://off.example/mcp", "disabled": True},
            }
        },
    )
    settings = CliSettings()
    settings.trust_mcp_project(project)
    settings.mcp_project_server_overrides[str(project.resolve())] = {
        "team": False,
        "mine": False,
        "ghost": False,
        "off": True,
    }

    config = load_mcp_config_for(project, state, settings)

    assert config.servers["team"].enabled is False
    assert config.servers["mine"].enabled is True
    assert config.servers["off"].enabled is True
    assert config.project_overrides == {"team": False, "off": True}
    assert [server.id for server in config.enabled_servers] == ["mine", "off"]
    # The fingerprint ignores the private choice, so a verified server stays verified.
    assert server_fingerprint(config.servers["team"]) == server_fingerprint(
        config.servers["team"].model_copy(update={"enabled": True})
    )


def test_pending_project_files_do_not_shadow_active_file_attribution(tmp_path: Path) -> None:
    project = tmp_path / "project"
    state = (tmp_path / "state").resolve()
    user = global_mcp_config_path(state)
    standard = standard_mcp_config_path(project)
    legacy = project_mcp_config_path(project)
    _write(user, {"mcpServers": {"shared": {"url": "https://user.example/mcp"}}})
    _write(standard, {"mcpServers": {name: {"command": "example"} for name in ("shared", "pending", "local")}})
    _write(legacy, {"servers": [{"id": "shared", "url": "https://legacy.example/mcp"}]})
    local = {"local": {"url": "https://local.example/mcp"}}

    config = load_mcp_config(project, state, project_trusted=False, local_servers=local)

    assert config.servers["shared"].url == "https://user.example/mcp"
    assert config.servers["local"].url == "https://local.example/mcp"
    assert config.server_files == {"shared": user}
    assert config.untrusted_project_server_files == {"shared": legacy, "pending": standard, "local": standard}
    assert {server.id: server.url for server in config.untrusted_project_servers} == {
        "shared": "https://legacy.example/mcp",
        "pending": None,
        "local": None,
    }
    assert len(config.untrusted_project_servers) == 3
    assert load_mcp_config(project, state, project_trusted=True).untrusted_project_server_files == {}


def test_launch_trust_can_be_applied_to_another_workspace(tmp_path: Path) -> None:
    launch = tmp_path / "launch"
    launch.mkdir()
    linked = tmp_path / "linked"
    linked.mkdir()
    state = tmp_path / "state"
    _write(standard_mcp_config_path(linked), {"mcpServers": {"linked": {"command": "npx"}}})
    settings = CliSettings()

    untrusted = load_mcp_config_for(linked, state, settings, trusted=False)
    trusted = load_mcp_config_for(linked, state, settings, trusted=True)

    assert [server.id for server in untrusted.untrusted_project_servers] == ["linked"]
    assert set(trusted.servers) == {"linked"}
    assert load_mcp_config_for(linked, state, settings).project_trusted is False


def test_tolerant_loading_skips_a_bad_entry_but_mutation_refuses_the_file(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    state = tmp_path / "state"
    path = global_mcp_config_path(state)
    _write(
        path,
        {
            "schema_version": 1,
            "servers": [
                {"id": "broken", "transport": "streamable_http"},
                {"id": "fine", "transport": "streamable_http", "url": "https://fine.example/mcp"},
                {"id": "fine", "transport": "sse", "url": "https://dup.example/sse"},
            ],
        },
    )

    config = load_mcp_config(project, state, project_trusted=False)

    assert set(config.servers) == {"fine"}
    assert any(
        "Skipped an entry in global MCP config" in diagnostic and '"broken" is invalid' in diagnostic
        for diagnostic in config.diagnostics
    )
    assert any("Duplicate MCP server id: fine" in diagnostic for diagnostic in config.diagnostics)
    with pytest.raises(MCPConfigError):
        upsert_server_config(path, _http_server("another"), source="global")
    assert set(json.loads(path.read_text())["servers"][0]) == {"id", "transport"}


def test_absent_and_empty_config_files_are_not_load_errors(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    state = tmp_path / "state"
    global_mcp_config_path(state).parent.mkdir(parents=True)
    global_mcp_config_path(state).write_text("  \n", encoding="utf-8")
    standard_mcp_config_path(project).write_text("", encoding="utf-8")

    config = load_mcp_config(project, state, project_trusted=True)

    assert config.servers == {}
    assert config.diagnostics == []
    assert config.project_config_present is True


@pytest.mark.parametrize(
    ("entry", "transport"),
    [
        ({"type": "http", "url": "https://x.example/mcp"}, "streamable_http"),
        ({"type": "streamable-http", "url": "https://x.example/mcp"}, "streamable_http"),
        ({"type": "stdio", "command": "npx"}, "stdio"),
        ({"type": "sse", "url": "https://x.example/sse"}, "sse"),
        ({"url": "https://x.example/mcp"}, "streamable_http"),
        ({"command": "npx", "args": ["-y", "x"]}, "stdio"),
        ({"httpUrl": "https://x.example/sse"}, "sse"),
        ({"transport": "http", "url": "https://x.example/mcp"}, "streamable_http"),
    ],
)
def test_standard_entry_types_and_inference(entry: dict[str, Any], transport: str) -> None:
    assert standard_entry_payload("srv", entry).payload["transport"] == transport
    if "transport" in entry or "type" in entry:
        assert server_from_entry("srv", entry, source="project").transport == transport


def test_server_config_accepts_transport_aliases_directly() -> None:
    for alias in ("http", "streamable-http", "streamable_http", "HTTP"):
        assert MCPServerConfig.model_validate(
            {"id": "srv", "transport": alias, "url": "https://x.example"}
        ).transport == ("streamable_http")


@pytest.mark.parametrize(
    "entry",
    [
        {"type": "ws", "url": "wss://x.example"},
        {"type": "sdk"},
        {"url": "wss://x.example"},
    ],
)
def test_unsupported_entry_types_are_reported_as_skipped(entry: dict[str, Any]) -> None:
    with pytest.raises(UnsupportedMCPTypeError):
        standard_entry_payload("srv", entry)


@pytest.mark.parametrize("entry", [{}, {"type": "ftp"}, {"url": "ftp://x.example"}, {"args": ["x"]}])
def test_uninterpretable_entries_raise_without_echoing_values(entry: dict[str, Any]) -> None:
    with pytest.raises(MCPEntryError) as excinfo:
        standard_entry_payload("srv", entry)
    assert "srv" in str(excinfo.value)


def test_an_entry_that_needs_field_validation_reports_a_config_error() -> None:
    with pytest.raises(MCPConfigError) as excinfo:
        server_from_entry("srv", {"type": "stdio"}, source="project")
    assert "requires command" in str(excinfo.value)


def test_standard_entry_timeout_disabled_flag_and_scalar_coercion() -> None:
    slow = standard_entry_payload(
        "srv", {"command": "npx", "env": {"N": 7, "DROP": None, "LIST": [1]}, "disabled": True, "timeout": 5000}
    )
    assert slow.payload["enabled"] is False
    assert slow.payload["timeout_seconds"] == 5.0
    assert slow.payload["env"] == {"N": "7"}
    assert slow.warnings == [
        'srv: dropped "env.DROP" because it is not a scalar value',
        'srv: dropped "env.LIST" because it is not a scalar value',
    ]

    fast = standard_entry_payload("srv", {"command": "npx", "enabled": False, "timeout": 500})
    assert fast.payload["enabled"] is False
    assert "timeout_seconds" not in fast.payload


def test_standard_entry_maps_claude_shaped_oauth_and_reports_dropped_fields() -> None:
    entry = standard_entry_payload(
        "srv",
        {
            "type": "http",
            "url": "https://x.example/mcp",
            "oauth": {
                "clientId": "cid",
                "callbackPort": 8080,
                "scopes": ["a", "b"],
                "authServerMetadataUrl": "https://x.example/oauth",
            },
            "alwaysLoad": True,
        },
    )
    oauth = MCPOAuthConfig.model_validate(entry.payload["oauth"])

    assert (oauth.enabled, oauth.client_id, oauth.scope) == (True, "cid", "a b")
    assert oauth.redirect_uri == "http://127.0.0.1:8080/callback"
    assert entry.dropped_fields == ["alwaysLoad", "oauth.authServerMetadataUrl"]


def test_kolega_only_fields_round_trip_through_the_portable_file(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    state = tmp_path / "state"
    path = standard_mcp_config_path(project)
    server = MCPServerConfig(
        id="hub",
        name="HubSpot",
        transport="streamable_http",
        url="https://mcp.example.com",
        enabled=False,
        timeout_seconds=45,
        oauth=MCPOAuthConfig(
            enabled=True,
            client_id="cid",
            client_secret_env="HUBSPOT_SECRET",
            redirect_uri="http://127.0.0.1:33418/callback",
            token_endpoint_auth_method="client_secret_post",
        ),
    )

    assert upsert_standard_server(path, server, source="project") is True
    entry = json.loads(path.read_text())["mcpServers"]["hub"]

    assert set(entry) == {"type", "url", "timeout", "x-kolega"}
    assert (entry["type"], entry["url"], entry["timeout"]) == ("http", "https://mcp.example.com", 45000)
    assert entry["x-kolega"]["name"] == "HubSpot"
    assert entry["x-kolega"]["enabled"] is False
    assert entry["x-kolega"]["oauth"]["client_secret_env"] == "HUBSPOT_SECRET"
    restored = load_mcp_config(project, state, project_trusted=True).servers["hub"]
    assert server_fingerprint(restored) == server_fingerprint(server)
    assert restored.display_name == "HubSpot"

    assert upsert_standard_server(path, _http_server("plain"), source="project") is True
    assert set(json.loads(path.read_text())["mcpServers"]["plain"]) == {"type", "url"}


def test_portable_file_edits_preserve_other_entries_unknown_fields_and_order(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    path = standard_mcp_config_path(project)
    original = (
        '{\n  "$schema": "https://example.com/mcp.schema.json",\n  "mcpServers": {\n'
        '    "keep": {\n      "type": "stdio",\n      "command": "npx",\n      "alwaysLoad": true,\n'
        '      "args": ["-y", "kept"],\n      "env": {"A": "1"}\n    }\n  }\n}\n'
    )
    path.write_text(original, encoding="utf-8")

    assert upsert_standard_server(path, _http_server("added"), source="project") is True
    document = json.loads(path.read_text())
    assert list(document) == ["$schema", "mcpServers"]
    assert list(document["mcpServers"]) == ["keep", "added"]
    assert document["mcpServers"]["keep"]["alwaysLoad"] is True

    assert upsert_standard_server(path, _http_server("keep", "https://new.example/mcp"), source="project") is False
    assert json.loads(path.read_text())["mcpServers"]["keep"] == {
        "type": "http",
        "url": "https://new.example/mcp",
        "alwaysLoad": True,
    }

    assert remove_standard_server(path, "keep") is True
    assert remove_standard_server(path, "keep") is False
    assert remove_standard_server(path, "added") is True
    assert json.loads(path.read_text()) == {"$schema": "https://example.com/mcp.schema.json", "mcpServers": {}}


def test_rewriting_an_unchanged_portable_entry_leaves_the_file_byte_identical(tmp_path: Path) -> None:
    path = standard_mcp_config_path(tmp_path)
    upsert_standard_server(path, _http_server("docs"), source="project")
    original = path.read_bytes()

    upsert_standard_server(path, _http_server("docs"), source="project")

    assert path.read_bytes() == original


def test_broken_portable_file_is_a_diagnostic_when_loaded_and_an_error_when_mutated(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    state = tmp_path / "state"
    path = standard_mcp_config_path(project)
    path.write_text("{not json", encoding="utf-8")

    config = load_mcp_config(project, state, project_trusted=True)
    assert any("Could not load project MCP config" in diagnostic for diagnostic in config.diagnostics)
    with pytest.raises(MCPConfigError):
        upsert_standard_server(path, _http_server("docs"), source="project")
    assert path.read_text() == "{not json"


def test_unparseable_portable_entry_survives_an_edit_to_its_neighbour(tmp_path: Path) -> None:
    path = standard_mcp_config_path(tmp_path)
    _write(path, {"mcpServers": {"socket": {"type": "ws", "url": "wss://x.example"}, "fine": {"command": "npx"}}})

    assert upsert_standard_server(path, _http_server("fine", "https://new.example/mcp"), source="project") is False

    assert json.loads(path.read_text())["mcpServers"]["socket"] == {"type": "ws", "url": "wss://x.example"}


def _stdio_server(**env: str) -> MCPServerConfig:
    return MCPServerConfig(id="srv", transport="stdio", command="${BIN:-npx}", args=["${ARG}", "${KEEP}"], env=env)


def test_env_references_resolve_from_the_environment_without_touching_the_definition() -> None:
    server = _stdio_server(TOKEN="${TOKEN}", HOME_DIR="${HOME}")
    environ = {"ARG": "a", "KEEP": "k", "TOKEN": "t", "HOME": "/home/me"}

    resolved = expand_env_references(server, environ)

    assert (resolved.command, resolved.args) == ("npx", ["a", "k"])
    assert resolved.env == {"TOKEN": "t", "HOME_DIR": "/home/me"}
    assert resolved is not server
    assert server.command == "${BIN:-npx}"
    assert server.env == {"TOKEN": "${TOKEN}", "HOME_DIR": "${HOME}"}


def test_unset_reference_takes_its_default_or_names_the_missing_variable() -> None:
    with_defaults = MCPServerConfig(
        id="srv", transport="stdio", command="${BIN:-npx}", args=["${ARG:-}"], env={"FALLBACK": "${TOKEN:-none}"}
    )
    assert expand_env_references(with_defaults, {}).env == {"FALLBACK": "none"}
    assert expand_env_references(with_defaults, {}).args == [""]
    assert missing_env_references(with_defaults, {}) == []

    # An empty value counts as set.
    assert expand_env_references(_stdio_server(TOKEN="${TOKEN}"), {"ARG": "", "KEEP": "", "TOKEN": ""}).args == ["", ""]
    with pytest.raises(MCPEnvReferenceError) as excinfo:
        expand_env_references(_stdio_server(TOKEN="${TOKEN}"), {"ARG": "a"})
    assert excinfo.value.names == ["KEEP", "TOKEN"]
    assert missing_env_references(_stdio_server(TOKEN="${TOKEN}"), {"ARG": "a", "KEEP": "k", "TOKEN": "x"}) == []


def test_only_well_formed_references_expand() -> None:
    literal = "a${UNCLOSED ${input:foo} ${BAD-NAME} $HOME ${OK}"
    server = MCPServerConfig(id="srv", transport="stdio", command="run", args=[literal])

    assert env_references(literal) == ["OK"]
    assert expand_env_references(server, {"OK": "yes"}).args == ["a${UNCLOSED ${input:foo} ${BAD-NAME} $HOME yes"]


def test_http_urls_and_headers_expand_and_a_server_without_references_is_returned_as_is() -> None:
    server = MCPServerConfig(
        id="srv",
        transport="streamable_http",
        url="https://${HOST}/mcp?key=${KEY}",
        headers={"Authorization": "Bearer ${KEY}", "X-Static": "literal"},
    )

    resolved = expand_env_references(server, {"HOST": "h", "KEY": "k"})
    assert resolved.url == "https://h/mcp?key=k"
    assert resolved.headers == {"Authorization": "Bearer k", "X-Static": "literal"}
    with pytest.raises(MCPEnvReferenceError) as excinfo:
        expand_env_references(server, {})
    assert excinfo.value.names == ["HOST", "KEY"]

    plain = _http_server("plain")
    assert expand_env_references(plain, {}) is plain


def test_fingerprint_is_computed_before_any_environment_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _stdio_server(TOKEN="${TOKEN}")
    before = server_fingerprint(server)

    monkeypatch.setenv("TOKEN", "resolved-value")

    assert server_fingerprint(server) == before
    assert server_fingerprint(server) == server_fingerprint(server.model_copy(update={"source": "project"}))


@pytest.mark.asyncio
async def test_open_mcp_session_hands_the_server_resolved_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mcp.client.session
    import mcp.client.stdio

    captured: list[Any] = []

    @contextlib.asynccontextmanager
    async def fake_stdio_client(params, errlog=None):
        captured.append(params)
        yield object(), object()

    class FakeClientSession:
        def __init__(self, read_stream, write_stream, read_timeout_seconds=None) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def initialize(self):
            return None

    monkeypatch.setattr(mcp.client.stdio, "stdio_client", fake_stdio_client)
    monkeypatch.setattr(mcp.client.session, "ClientSession", FakeClientSession)
    server = _stdio_server(TOKEN="${TOKEN}")

    monkeypatch.setenv("ARG", "a")
    monkeypatch.setenv("KEEP", "k")
    monkeypatch.setenv("TOKEN", "from-env")
    async with open_mcp_session(server, project_path=tmp_path, token_store=MCPOAuthTokenStore(tmp_path)):
        pass

    params = captured[0]
    assert params.command == "npx"
    assert params.args == ["a", "k"]
    assert params.env == {"TOKEN": "from-env"}
    assert server.env == {"TOKEN": "${TOKEN}"}

    monkeypatch.delenv("TOKEN")
    captured.clear()
    with pytest.raises(MCPEnvReferenceError):
        async with open_mcp_session(server, project_path=tmp_path, token_store=MCPOAuthTokenStore(tmp_path)):
            pass
    assert captured == []


@pytest.mark.asyncio
async def test_verification_stores_fixed_failure_and_status_checks_current_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("ARG", "KEEP", "TOKEN"):
        monkeypatch.delenv(name, raising=False)
    server = _stdio_server(TOKEN="${TOKEN}")
    service = MCPService(LoadedMCPConfig(servers={"srv": server}), state_dir=tmp_path, project_path=tmp_path)

    result = await service.verify_server("srv")

    assert result.ok is False
    assert result.message == MCP_FAILURE_MESSAGE_ENV_MISSING
    stored = MCPStatusStore(tmp_path).get("srv")
    assert stored is not None
    assert stored.status == "failed"
    row = service.list_status_rows()[0]
    assert row["health"].health == "failed"
    assert row["missing_env"] == ["ARG", "KEEP", "TOKEN"]
    monkeypatch.setenv("TOKEN", "from-env")
    assert service.list_status_rows()[0]["missing_env"] == ["ARG", "KEEP"]


@pytest.mark.asyncio
async def test_abandoned_sign_in_reads_as_needs_auth_not_as_an_unreachable_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _http_server("o", oauth=MCPOAuthConfig(enabled=True, timeout_seconds=0.05))
    service = MCPService(LoadedMCPConfig(servers={"o": server}), state_dir=tmp_path, project_path=tmp_path)

    @contextlib.asynccontextmanager
    async def session_waiting_for_sign_in(config, *, oauth_interaction: Any, **kwargs):
        try:
            await oauth_interaction.callback_handler()
        except Exception as exc:
            # The SDK reports a failed connect through an exception group.
            raise ExceptionGroup("session failed", [exc]) from None
        yield object()  # pragma: no cover - the callback never arrives

    monkeypatch.setattr("kolega_code.mcp.service.open_mcp_session", session_waiting_for_sign_in)

    result = await service.verify_server("o", interactive_oauth=True, open_browser=False, output=io.StringIO())

    assert result.ok is False
    assert result.message == MCP_FAILURE_MESSAGE_OAUTH_SIGNIN_INCOMPLETE
    stored = MCPStatusStore(tmp_path).get("o")
    assert stored is not None and stored.status == "failed"
    health = server_health(server, stored)
    assert (health.health, health.hint) == ("needs_auth", "sign in: kolega-code mcp login o")
    assert service.list_status_rows()[0]["message"] == MCP_FAILURE_MESSAGE_OAUTH_SIGNIN_INCOMPLETE
    # A network timeout is still reported as a timeout; only the callback wait means "sign in".
    assert _mcp_failure_message(server, TimeoutError("timed out")) == MCP_FAILURE_MESSAGE_TIMEOUT
    assert _mcp_failure_message(server, ExceptionGroup("g", [MCPOAuthCallbackTimeout("deadline")])) == (
        MCP_FAILURE_MESSAGE_OAUTH_SIGNIN_INCOMPLETE
    )
    plain = _http_server("p")
    # Without OAuth enabled the callback wait cannot be the cause, so the type alone is not special.
    assert _mcp_failure_message(plain, ExceptionGroup("g", [MCPOAuthCallbackTimeout("deadline")])) == (
        MCP_FAILURE_MESSAGE_TIMEOUT
    )
    # Never signed in: the Settings status block says why, and the row still reads as needing auth.
    fresh = MCPService(LoadedMCPConfig(servers={"o": server}), state_dir=tmp_path / "fresh", project_path=tmp_path)
    fresh_row = fresh.list_status_rows()[0]
    assert (fresh_row["status"], fresh_row["message"], fresh_row["health"].health) == (
        "unverified",
        "Sign-in required before this server can connect.",
        "needs_auth",
    )


def test_config_mutations_lock_the_file_they_rewrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = tmp_path / "state"
    native = global_mcp_config_path(state)
    portable = tmp_path / "project" / ".mcp.json"
    lock_dir = config_module.config_lock_dir(state)
    acquired: list[str] = []
    real_acquire = config_module.FileLock.acquire

    def recording_acquire(self, *args: Any, **kwargs: Any) -> Any:
        acquired.append(str(self.lock_file))
        return real_acquire(self, *args, **kwargs)

    monkeypatch.setattr(config_module.FileLock, "acquire", recording_acquire)

    upsert_server_config(native, _http_server("a"), source="global")
    remove_server_config(native, "a", source="global")
    upsert_standard_server(portable, _http_server("b"), source="project", lock_dir=lock_dir)
    remove_standard_server(portable, "b", lock_dir=lock_dir)

    portable_lock = next(lock_dir.glob("*.lock"))
    assert acquired == [f"{native}.lock", f"{native}.lock", str(portable_lock), str(portable_lock)]
    assert not portable.with_suffix(".json.lock").exists()
    assert not list(portable.parent.glob("*.lock"))


def test_a_lock_timeout_is_reported_as_a_config_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from filelock import Timeout as FileLockTimeout

    def never_acquires(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise FileLockTimeout(str(self.lock_file))

    monkeypatch.setattr(config_module.FileLock, "acquire", never_acquires)
    native = global_mcp_config_path(tmp_path)

    with pytest.raises(MCPConfigError, match="Timed out locking the MCP config"):
        upsert_server_config(native, _http_server("a"), source="global")
    assert not native.exists()


def test_concurrent_processes_adding_servers_do_not_lose_updates(tmp_path: Path) -> None:
    path = global_mcp_config_path(tmp_path / "state")
    workers, per_worker = 4, 25
    program = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        from kolega_code.mcp.config import MCPServerConfig, upsert_server_config

        path, worker = Path(sys.argv[1]), sys.argv[2]
        for index in range(int(sys.argv[3])):
            server = MCPServerConfig(
                id=f"{worker}-{index}", transport="streamable_http", url=f"https://example.com/{index}"
            )
            upsert_server_config(path, server, source="global")
        """
    )
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", program, str(path), str(worker), str(per_worker)],
            cwd=Path(__file__).resolve().parents[2],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for worker in range(workers)
    ]

    for process in processes:
        _, stderr = process.communicate(timeout=120)
        assert process.returncode == 0, stderr

    saved = {server["id"] for server in json.loads(path.read_text())["servers"]}
    assert len(saved) == workers * per_worker


def test_server_health_maps_every_state_and_ignores_unsafe_stored_messages() -> None:
    project_server = MCPServerConfig(id="p", transport="streamable_http", url="https://p.example/mcp", source="project")
    pending = server_health(project_server, _verified(project_server), trusted=False)
    assert (pending.health, pending.glyph, pending.label, pending.hint) == (
        "pending_trust",
        "⏸",
        "Pending approval",
        "project not trusted",
    )

    disabled = _http_server("d", enabled=False)
    assert server_health(disabled, _verified(disabled)).health == "disabled"
    assert server_health(disabled, None).hint == "disabled"
    assert server_health(disabled, None, enabled_override=False).hint == "disabled for this project"

    plain = _http_server("c")
    connected = server_health(plain, _verified(plain, ("a", "b")))
    assert (connected.health, connected.tool_count, connected.hint) == ("connected", 2, "2 tools · verified just now")
    assert server_health(plain, _verified(plain, ("a",))).hint == "1 tool · verified just now"

    oauth = _http_server("o", oauth=MCPOAuthConfig(enabled=True))
    unauthorized = MCPServerStatus.failed(
        fingerprint=server_fingerprint(oauth),
        transport=oauth.transport,
        source="global",
        message=MCP_FAILURE_MESSAGE_OAUTH_UNAUTHORIZED,
    )
    needs_auth = server_health(oauth, unauthorized)
    assert (needs_auth.health, needs_auth.hint) == ("needs_auth", "sign in: kolega-code mcp login o")
    assert (
        server_health(plain, unauthorized.model_copy(update={"fingerprint": server_fingerprint(plain)})).health
        == "failed"
    )

    stale = _verified(plain).model_copy(update={"fingerprint": "an-older-configuration"})
    assert (server_health(plain, stale).health, server_health(plain, stale).hint) == (
        "stale",
        "config changed since the last verification; verify again",
    )

    legacy = MCPServerStatus.failed(
        fingerprint=server_fingerprint(plain),
        transport=plain.transport,
        source="global",
        message="RuntimeError: password=legacy-secret",
    )
    failed = server_health(plain, legacy)
    assert (failed.health, failed.message, failed.hint) == (
        "failed",
        MCP_FAILURE_MESSAGE_GENERIC,
        MCP_FAILURE_MESSAGE_GENERIC,
    )

    assert (server_health(plain, None).health, server_health(plain, None).hint) == ("unverified", "")
    # An OAuth server that was never signed in, or was signed out, waits for the person.
    never = server_health(oauth, None)
    assert (never.health, never.glyph, never.label, never.hint) == (
        "needs_auth",
        "!",
        "Needs auth",
        "sign in: kolega-code mcp login o",
    )
    # A sign-in that was started but not finished is "sign in again", not an unreachable server.
    incomplete = MCPServerStatus.failed(
        fingerprint=server_fingerprint(oauth),
        transport=oauth.transport,
        source="global",
        message=MCP_FAILURE_MESSAGE_OAUTH_SIGNIN_INCOMPLETE,
    )
    assert server_health(oauth, incomplete).health == "needs_auth"
    assert set(HEALTH_LABELS) == {
        "pending_trust",
        "disabled",
        "connected",
        "needs_auth",
        "stale",
        "failed",
        "unverified",
    }


def test_relative_time_handles_missing_and_malformed_timestamps() -> None:
    from datetime import datetime, timedelta, timezone

    assert relative_time(None) == ""
    assert relative_time("not a timestamp") == ""
    assert relative_time("2020-01-01T00:00:00+00:00").endswith("y ago")
    assert relative_time((datetime.now(timezone.utc) - timedelta(seconds=90)).isoformat()) == "1m ago"


def test_find_definitions_lists_every_scope_highest_precedence_first(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    state = tmp_path / "state"
    key = str(project.resolve())
    settings = CliSettings()
    settings.mcp_local_servers[key] = {"shared": {"command": "npx"}}
    _write(
        global_mcp_config_path(state),
        {
            "schema_version": 1,
            "servers": [{"id": "shared", "transport": "streamable_http", "url": "https://u.example/mcp"}],
        },
    )
    _write(
        standard_mcp_config_path(project), {"mcpServers": {"shared": {"type": "http", "url": "https://s.example/mcp"}}}
    )
    _write(
        project_mcp_config_path(project),
        {"schema_version": 1, "servers": [{"id": "shared", "transport": "sse", "url": "https://l.example/sse"}]},
    )

    definitions = find_definitions("shared", project_path=project, state_dir=state, settings=settings)

    assert [(definition.scope, definition.file) for definition in definitions] == [
        ("local", None),
        ("project", project_mcp_config_path(project)),
        ("project", standard_mcp_config_path(project)),
        ("user", global_mcp_config_path(state)),
    ]
    assert find_definitions("missing", project_path=project, state_dir=state, settings=settings) == []


def test_project_scope_writes_the_portable_file_unless_only_a_legacy_one_exists(tmp_path: Path) -> None:
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    legacy = tmp_path / "legacy"
    legacy.mkdir()

    assert project_write_target(fresh) == standard_mcp_config_path(fresh)
    assert project_write_target(legacy) == standard_mcp_config_path(legacy)

    project_mcp_config_path(legacy).parent.mkdir(parents=True)
    project_mcp_config_path(legacy).write_text("{}", encoding="utf-8")

    assert project_write_target(legacy) == project_mcp_config_path(legacy)
    assert scope_location("local", fresh, tmp_path) == "settings.json, this project only"
    assert scope_location("user", fresh, tmp_path) == str(global_mcp_config_path(tmp_path))


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission modes")
@pytest.mark.parametrize("legacy", [False, True])
def test_shared_config_mutations_preserve_modes(tmp_path: Path, legacy: bool) -> None:
    project = tmp_path / "project"
    state = tmp_path / "state"
    path = project_mcp_config_path(project) if legacy else standard_mcp_config_path(project)
    _write(path, {})
    project.chmod(0o755)
    if legacy:
        path.parent.chmod(0o775)
    path.chmod(0o664)
    context = {"project_path": project, "state_dir": state, "settings": CliSettings()}
    previous_umask = os.umask(0o077)
    try:
        assert upsert_server(_http_server("shared"), scope="project", **context) == "added"
        assert stat.S_IMODE(path.stat().st_mode) == 0o664
        assert remove_server("shared", scope="project", **context)
    finally:
        os.umask(previous_umask)
    assert stat.S_IMODE(path.stat().st_mode) == 0o664
    assert stat.S_IMODE(project.stat().st_mode) == 0o755
    assert stat.S_IMODE(path.parent.stat().st_mode) == (0o775 if legacy else 0o755)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission modes")
def test_new_shared_config_honors_umask_but_user_config_stays_private(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir(mode=0o755)
    project.chmod(0o755)
    state = tmp_path / "state"
    context = {"project_path": project, "state_dir": state, "settings": CliSettings()}
    previous_umask = os.umask(0o027)
    try:
        assert upsert_server(_http_server("shared"), scope="project", **context) == "added"
        assert stat.S_IMODE(state.stat().st_mode) == 0o700
        assert upsert_server(_http_server("private"), scope="user", **context) == "added"
    finally:
        os.umask(previous_umask)
    assert stat.S_IMODE(project.stat().st_mode) == 0o755
    assert stat.S_IMODE(standard_mcp_config_path(project).stat().st_mode) == 0o640
    assert stat.S_IMODE(state.stat().st_mode) == 0o700
    assert stat.S_IMODE(global_mcp_config_path(state).stat().st_mode) == 0o600


def test_failed_shared_config_replace_keeps_the_original_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    state = tmp_path / "state"
    path = standard_mcp_config_path(project)
    _write(path, {"mcpServers": {"existing": {"command": "example"}}})
    original = path.read_bytes()

    def fail_replace(temp: Path, target: Path) -> Path:
        assert target == path
        assert temp != target
        assert set(json.loads(temp.read_bytes())["mcpServers"]) == {"existing", "other"}
        assert target.read_bytes() == original
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        upsert_server(
            _http_server("other"), scope="project", project_path=project, state_dir=state, settings=CliSettings()
        )

    assert path.read_bytes() == original
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("legacy", [False, True])
def test_whitespace_only_project_config_can_be_edited(tmp_path: Path, legacy: bool) -> None:
    project = tmp_path / "project"
    path = project_mcp_config_path(project) if legacy else standard_mcp_config_path(project)
    path.parent.mkdir(parents=True)
    path.write_text(" \t\n", encoding="utf-8")
    state = tmp_path / "state"

    assert (
        upsert_server(
            _http_server("shared"), scope="project", project_path=project, state_dir=state, settings=CliSettings()
        )
        == "added"
    )
    assert set(load_mcp_config(project, state, project_trusted=True).servers) == {"shared"}


def test_portable_config_can_update_and_remove_an_id_longer_than_64_characters(tmp_path: Path) -> None:
    project = tmp_path / "project"
    state = tmp_path / "state"
    path = standard_mcp_config_path(project)
    long_id = "long-server-" * 6
    canonical = sanitize_mcp_server_id(long_id)
    _write(path, {"mcpServers": {long_id: {"command": "example", "foreignOption": True}}})
    context = {"project_path": project, "state_dir": state, "settings": CliSettings()}

    assert upsert_server(_http_server(canonical), scope="project", **context) == "updated"
    assert json.loads(path.read_bytes())["mcpServers"] == {
        long_id: {"type": "http", "url": "https://example.com/mcp", "foreignOption": True}
    }
    assert remove_server(canonical, scope="project", **context)
    assert load_mcp_config(project, state, project_trusted=True).servers == {}


def test_local_scope_servers_are_edited_in_place(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    state = tmp_path / "state"
    settings = CliSettings()
    context = {"project_path": project, "state_dir": state, "settings": settings}

    assert upsert_server(_http_server("db"), scope="local", **context) == "added"
    assert upsert_server(_http_server("db", "https://new.example/mcp"), scope="local", **context) == "updated"
    assert set_server_enabled_for_scope("db", False, scope="local", **context) == (True, False)
    assert settings.mcp_local_servers[str(project.resolve())]["db"]["enabled"] is False
    assert set_server_enabled_for_scope("ghost", True, scope="local", **context) == (False, False)

    assert remove_server("db", scope="local", **context) is True
    assert settings.mcp_local_servers == {}
    assert remove_server("db", scope="local", **context) is False


def test_project_enable_disable_records_a_private_choice_and_reset_forgets_it(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    state = tmp_path / "state"
    settings = CliSettings()
    settings.trust_mcp_project(project)
    context = {"project_path": project, "state_dir": state, "settings": settings}
    upsert_server(_http_server("ci"), scope="project", **context)

    assert set_server_enabled_for_scope("ci", False, scope="project", **context) == (True, True)
    assert project_server_override(settings, project, "ci") is False
    assert json.loads(standard_mcp_config_path(project).read_text())["mcpServers"]["ci"] == {
        "type": "http",
        "url": "https://example.com/mcp",
    }

    assert reset_project_choices(settings, project) == (1, True)
    assert settings.mcp_project_server_overrides == {}
    assert not settings.is_mcp_project_trusted(project)
    assert reset_project_choices(settings, project) == (0, False)


def test_literal_secret_detection_treats_references_as_safe() -> None:
    assert has_literal_secrets(_http_server("a", headers={"Authorization": "Bearer ${TOKEN}"})) is False
    assert has_literal_secrets(_http_server("b", headers={"X-Api-Key": "abc123"})) is True
    assert (
        has_literal_secrets(MCPServerConfig(id="c", transport="stdio", command="npx", env={"A": "${A}", "B": "b"}))
        is True
    )
    assert has_literal_secrets(_http_server("d")) is False


def test_settings_coercion_keeps_structurally_valid_mcp_state(tmp_path: Path) -> None:
    settings = CliSettings.from_dict(
        {
            "schema_version": 3,
            "mcp_local_servers": {"/p": {"ok": {"command": "npx"}, "bad": "nope"}, "/q": "nope", "/empty": {}},
            "mcp_project_server_overrides": {"/p": {"a": True, "b": "yes", "c": False}, "/z": 1},
        }
    )

    assert settings.mcp_local_servers == {"/p": {"ok": {"command": "npx"}}}
    assert settings.mcp_project_server_overrides == {"/p": {"a": True, "c": False}}
    restored = CliSettings.from_dict(settings.to_dict())
    assert restored.mcp_local_servers == settings.mcp_local_servers
    assert restored.mcp_project_server_overrides == settings.mcp_project_server_overrides


def test_independent_local_scope_additions_both_survive_the_settings_merge(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SettingsStore(tmp_path)
    first = store.load()
    second = store.load()

    upsert_server(_http_server("a"), scope="local", project_path=project, state_dir=tmp_path, settings=first)
    store.save(first)
    upsert_server(_http_server("b"), scope="local", project_path=project, state_dir=tmp_path, settings=second)
    store.save(second)

    saved = store.load()
    assert set(saved.mcp_local_servers[str(project.resolve())]) == {"a", "b"}


@pytest.mark.parametrize("rename", [False, True])
def test_removing_last_known_local_server_keeps_another_writers_addition(tmp_path: Path, rename: bool) -> None:
    project = tmp_path / "project"
    store = SettingsStore(tmp_path)
    initial = store.load()
    upsert_server(_http_server("old"), scope="local", project_path=project, state_dir=tmp_path, settings=initial)
    store.save(initial)
    stale = store.load()
    other = store.load()
    upsert_server(_http_server("other"), scope="local", project_path=project, state_dir=tmp_path, settings=other)
    store.save(other)

    assert remove_server("old", scope="local", project_path=project, state_dir=tmp_path, settings=stale)
    assert stale.mcp_local_servers == {}
    store.save(stale)
    if rename:
        upsert_server(_http_server("new"), scope="local", project_path=project, state_dir=tmp_path, settings=stale)
        store.save(stale)

    assert set(store.load().mcp_local_servers[str(project.resolve())]) == ({"new", "other"} if rename else {"other"})


def test_the_settings_aware_loader_is_the_only_loader_used_outside_the_config_module() -> None:
    package_root = Path(config_module.__file__).resolve().parents[1]
    allowed = {Path(config_module.__file__).resolve(), package_root / "mcp" / "tools.py"}
    offenders = [
        str(path.relative_to(package_root))
        for path in sorted(package_root.rglob("*.py"))
        if path not in allowed and "load_mcp_config(" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []
