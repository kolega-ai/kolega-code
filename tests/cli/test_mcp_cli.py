from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

from kolega_code.cli.config import API_KEY_ENV, CliConfigOverrides, build_agent_config
from kolega_code.cli.main import main, parse_args
from kolega_code.cli.settings import CliSettings, SettingsStore
from kolega_code.config import ModelProvider
from kolega_code.mcp.config import MCPServerConfig, global_mcp_config_path, server_fingerprint
from kolega_code.mcp.service import MCP_FAILURE_MESSAGE_GENERIC, MCPService, MCPVerificationResult
from kolega_code.mcp.state import MCPServerStatus, MCPStatusStore, MCPOAuthTokenStore, MCPToolStatus


def test_mcp_subcommands_parse() -> None:
    args = parse_args(
        [
            "mcp",
            "--project",
            ".",
            "--state-dir",
            "state",
            "add",
            "docs",
            "--transport",
            "streamable_http",
            "--url",
            "https://docs.example/mcp",
            "--oauth",
        ]
    )

    assert args.command == "mcp"
    assert args.mcp_command == "add"
    assert args.server_id == "docs"
    assert args.transport == "streamable_http"
    assert args.stdio_command is None
    assert args.oauth is True


def test_mcp_add_rewrites_over_long_server_id_and_remove_accepts_it(tmp_path: Path, capsys) -> None:
    project = tmp_path / "project"
    state = tmp_path / "state"
    project.mkdir()
    long_id = "x" * 40

    exit_code = main(
        [
            "mcp",
            "--project",
            str(project),
            "--state-dir",
            str(state),
            "add",
            long_id,
            "--transport",
            "streamable_http",
            "--url",
            "https://docs.example/mcp",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "exceeds 32 characters" in captured.err
    payload = json.loads(global_mcp_config_path(state).read_text(encoding="utf-8"))
    saved_id = payload["servers"][0]["id"]
    assert saved_id != long_id
    assert len(saved_id) == 32

    # The original long id still addresses the server for removal.
    exit_code = main(["mcp", "--project", str(project), "--state-dir", str(state), "remove", long_id])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert f"Removed MCP server {saved_id}" in captured.out
    payload = json.loads(global_mcp_config_path(state).read_text(encoding="utf-8"))
    assert payload["servers"] == []


def test_mcp_add_and_list_use_state_dir(tmp_path: Path, capsys) -> None:
    project = tmp_path / "project"
    state = tmp_path / "state"
    project.mkdir()

    exit_code = main(
        [
            "mcp",
            "--project",
            str(project),
            "--state-dir",
            str(state),
            "add",
            "docs",
            "--transport",
            "streamable_http",
            "--url",
            "https://docs.example/mcp",
            "--header",
            "Authorization=Bearer secret",
            "--oauth",
            "--oauth-scope",
            "read write",
        ]
    )
    assert exit_code == 0
    payload = json.loads(global_mcp_config_path(state).read_text(encoding="utf-8"))
    assert payload["servers"][0]["id"] == "docs"
    assert payload["servers"][0]["headers"] == {"Authorization": "Bearer secret"}
    assert payload["servers"][0]["oauth"]["enabled"] is True

    exit_code = main(["mcp", "--project", str(project), "--state-dir", str(state), "list"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Added MCP server docs to user scope" in captured.out
    # An OAuth server that has not been signed in yet asks for the sign-in, not a plain verify.
    assert "! docs" in captured.out
    assert " user " in captured.out
    assert " http " in captured.out
    assert "Needs auth" in captured.out
    assert "sign in: kolega-code mcp login docs" in captured.out
    assert "Bearer secret" not in captured.out


def test_mcp_list_does_not_print_legacy_failed_status_message(tmp_path: Path, capsys) -> None:
    project = tmp_path / "project"
    state = tmp_path / "state"
    project.mkdir()
    server = MCPServerConfig(id="docs", transport="streamable_http", url="https://docs.example/mcp")

    exit_code = main(
        [
            "mcp",
            "--project",
            str(project),
            "--state-dir",
            str(state),
            "add",
            server.id,
            "--transport",
            server.transport,
            "--url",
            server.url or "",
        ]
    )
    assert exit_code == 0
    MCPStatusStore(state).update(
        server.id,
        MCPServerStatus.failed(
            fingerprint=server_fingerprint(server),
            transport=server.transport,
            source=server.source,
            message="RuntimeError: password=legacy-secret",
        ),
    )

    exit_code = main(["mcp", "--project", str(project), "--state-dir", str(state), "list"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "legacy-secret" not in captured.out + captured.err
    assert "✗ docs" in captured.out
    assert "Failed" in captured.out
    assert MCP_FAILURE_MESSAGE_GENERIC in captured.out


def test_mcp_list_message_column_notes_adjusted_tool_names(tmp_path: Path, capsys) -> None:
    project = tmp_path / "project"
    state = tmp_path / "state"
    project.mkdir()
    server = MCPServerConfig(id="docs", transport="streamable_http", url="https://docs.example/mcp")

    exit_code = main(
        [
            "mcp",
            "--project",
            str(project),
            "--state-dir",
            str(state),
            "add",
            server.id,
            "--transport",
            server.transport,
            "--url",
            server.url or "",
        ]
    )
    assert exit_code == 0
    MCPStatusStore(state).update(
        server.id,
        MCPServerStatus.verified(
            fingerprint=server_fingerprint(server),
            transport=server.transport,
            source="global",
            tools=[MCPToolStatus(id="get.file")],
        ),
    )

    argv = ["mcp", "--project", str(project), "--state-dir", str(state)]

    exit_code = main([*argv, "list"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "● docs" in captured.out
    assert "Connected" in captured.out
    assert "1 tool" in captured.out

    exit_code = main([*argv, "list", "--json"])
    captured = capsys.readouterr()
    assert exit_code == 0
    server = json.loads(captured.out)["servers"][0]
    assert server["id"] == "docs"
    assert server["scope"] == "user"
    assert server["health"] == "connected"
    assert server["tool_count"] == 1
    assert server["file"] == str(global_mcp_config_path(state))
    assert server["verified_at"]


def test_mcp_verify_cli_prints_adjusted_tool_names(tmp_path: Path, capsys, monkeypatch) -> None:
    project = tmp_path / "project"
    state = tmp_path / "state"
    project.mkdir()
    server = MCPServerConfig(id="docs", transport="streamable_http", url="https://docs.example/mcp")

    exit_code = main(
        [
            "mcp",
            "--project",
            str(project),
            "--state-dir",
            str(state),
            "add",
            server.id,
            "--transport",
            server.transport,
            "--url",
            server.url or "",
        ]
    )
    assert exit_code == 0

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

    exit_code = main(["mcp", "--project", str(project), "--state-dir", str(state), "verify", server.id])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert (
        "✓ docs: Verified 1 tool(s). 1 tool name was adjusted for provider compatibility "
        "(mcp__docs__get.file → mcp__docs__get_file)." in captured.out
    )


def test_build_agent_config_loads_trusted_project_mcp(tmp_path: Path, monkeypatch, isolated_cli_env) -> None:
    project = tmp_path / "project"
    state = tmp_path / "state"
    project.mkdir()
    (project / ".kolega").mkdir()
    (project / ".kolega" / "mcp_servers.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "servers": [{"id": "project-docs", "transport": "streamable_http", "url": "https://docs.example/mcp"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(API_KEY_ENV[ModelProvider.ANTHROPIC], "test-key")
    settings = CliSettings(active_provider="anthropic", active_model="claude-opus-4-8")
    settings.trust_mcp_project(project)
    settings_store = SettingsStore(state)
    settings_store.save(settings)

    config = build_agent_config(project, CliConfigOverrides(), settings=settings, settings_store=settings_store)

    assert config.mcp_config is not None
    assert set(config.mcp_config.servers) == {"project-docs"}
    assert config.mcp_config.project_trusted is True


def test_mcp_add_with_pre_registered_oauth(tmp_path: Path, capsys) -> None:
    project = tmp_path / "project"
    state = tmp_path / "state"
    project.mkdir()

    exit_code = main(
        [
            "mcp",
            "--project",
            str(project),
            "--state-dir",
            str(state),
            "add",
            "hubspot",
            "--transport",
            "streamable_http",
            "--url",
            "https://mcp.hubspot.com",
            "--oauth-client-id",
            "hubspot-client-id-123",
            "--oauth-client-secret-env",
            "HUBSPOT_MCP_CLIENT_SECRET",
            "--redirect-uri",
            "http://127.0.0.1:33418/callback",
            "--oauth-auth-method",
            "client_secret_post",
            "--oauth-scope",
            "crm.objects.contacts.read",
        ]
    )
    assert exit_code == 0
    payload = json.loads(global_mcp_config_path(state).read_text(encoding="utf-8"))
    saved = payload["servers"][0]
    assert saved["id"] == "hubspot"
    oauth = saved["oauth"]
    assert oauth["enabled"] is True
    assert oauth["client_id"] == "hubspot-client-id-123"
    assert oauth["client_secret_env"] == "HUBSPOT_MCP_CLIENT_SECRET"
    assert oauth["redirect_uri"] == "http://127.0.0.1:33418/callback"
    assert oauth["token_endpoint_auth_method"] == "client_secret_post"
    assert oauth["scope"] == "crm.objects.contacts.read"


def test_mcp_add_rejects_incomplete_oauth_without_exposing_secret(tmp_path: Path, capsys) -> None:
    state = tmp_path / "state"
    exit_code = main(
        [
            "mcp",
            "--project",
            str(tmp_path),
            "--state-dir",
            str(state),
            "add",
            "example",
            "--transport",
            "streamable_http",
            "--url",
            "https://mcp.example/mcp",
            "--oauth-client-secret",
            "fake-private-secret",
        ]
    )
    assert exit_code == 2
    output = capsys.readouterr()
    assert "client_id" in output.err
    assert "fake-private-secret" not in output.err + output.out
    assert not global_mcp_config_path(state).exists()


def _project(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    project.mkdir()
    return project, tmp_path / "state"


def _mcp(project: Path, state: Path, *args: str) -> list[str]:
    return ["mcp", "--project", str(project), "--state-dir", str(state), *args]


def _user_servers(state: Path) -> dict[str, dict]:
    return {server["id"]: server for server in json.loads(global_mcp_config_path(state).read_text())["servers"]}


def test_mcp_scope_and_hidden_project_alias_parse() -> None:
    args = parse_args(["mcp", "add", "-s", "local", "db", "--", "uvx", "my-db-mcp", "--port", "7000"])
    assert args.mcp_command == "add"
    assert args.scope == "local"
    assert args.server_id == "db"
    assert args.command_or_url == "uvx"
    assert args.server_args == ["my-db-mcp", "--port", "7000"]

    args = parse_args(["mcp", "remove", "db", "--project-config"])
    assert args.project_config is True
    assert args.scope is None

    args = parse_args(["mcp", "list", "--check", "--json", "--scope", "project"])
    assert (args.check, args.json, args.scope) == (True, True, "project")


def test_mcp_every_subcommand_parses() -> None:
    for argv, command in (
        (["mcp", "get", "docs", "--json"], "get"),
        (["mcp", "add-json", "weather", '{"type":"http"}'], "add-json"),
        (["mcp", "add-json", "-", "--client-secret"], "add-json"),
        (["mcp", "verify", "--all", "--yes"], "verify"),
        (["mcp", "login", "docs", "--no-browser"], "login"),
        (["mcp", "logout", "docs"], "logout"),
        (["mcp", "trust"], "trust"),
        (["mcp", "untrust"], "untrust"),
        (["mcp", "reset-project-choices"], "reset-project-choices"),
    ):
        assert parse_args(argv).mcp_command == command


def test_mcp_add_infers_transport_and_passes_server_flags_through(tmp_path: Path, capsys) -> None:
    project, state = _project(tmp_path)

    for argv in (
        ("add", "notion", "https://mcp.notion.com/mcp"),
        ("add", "events", "sse://example.com/sse"),
        ("add", "airtable", "--", "npx", "-y", "airtable-mcp-server", "--port", "8080"),
    ):
        assert main(_mcp(project, state, *argv)) == 0
    capsys.readouterr()

    servers = _user_servers(state)
    assert servers["notion"]["transport"] == "streamable_http"
    assert servers["notion"]["url"] == "https://mcp.notion.com/mcp"
    assert servers["events"]["transport"] == "sse"
    assert servers["airtable"]["transport"] == "stdio"
    assert servers["airtable"]["command"] == "npx"
    assert servers["airtable"]["args"] == ["-y", "airtable-mcp-server", "--port", "8080"]


def test_mcp_add_rejects_a_url_with_command_arguments(tmp_path: Path, capsys) -> None:
    project, state = _project(tmp_path)

    exit_code = main(_mcp(project, state, "add", "docs", "https://docs.example/mcp", "extra"))

    assert exit_code == 2
    assert "URL takes no command arguments" in capsys.readouterr().err


def test_mcp_add_accepts_colon_and_equals_headers(tmp_path: Path) -> None:
    project, state = _project(tmp_path)

    exit_code = main(
        _mcp(
            project,
            state,
            "add",
            "-H",
            "Authorization: Bearer ${TOKEN}",
            "-H",
            "X-Api-Key=abc",
            "-e",
            "A=1",
            "-e",
            "B=2",
            "svc",
            "--url",
            "https://example.com/mcp",
        )
    )

    assert exit_code == 0
    server = _user_servers(state)["svc"]
    assert server["headers"] == {"Authorization": "Bearer ${TOKEN}", "X-Api-Key": "abc"}
    assert server["env"] == {"A": "1", "B": "2"}


def test_mcp_add_again_updates_instead_of_duplicating(tmp_path: Path, capsys) -> None:
    project, state = _project(tmp_path)

    assert main(_mcp(project, state, "add", "docs", "https://first.example/mcp")) == 0
    capsys.readouterr()
    assert main(_mcp(project, state, "add", "docs", "https://second.example/mcp")) == 0

    assert "Updated MCP server docs" in capsys.readouterr().out
    assert [server["url"] for server in _user_servers(state).values()] == ["https://second.example/mcp"]


def test_mcp_add_callback_port_and_masked_secret_from_stdin(tmp_path: Path, capsys, monkeypatch) -> None:
    project, state = _project(tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO("prompted-secret\n"))

    exit_code = main(
        _mcp(
            project,
            state,
            "add",
            "--client-id",
            "client-123",
            "--callback-port",
            "8080",
            "hub",
            "https://mcp.example.com/mcp",
            "--client-secret",
        )
    )

    assert exit_code == 0
    oauth = _user_servers(state)["hub"]["oauth"]
    assert oauth["enabled"] is True
    assert oauth["client_id"] == "client-123"
    assert oauth["client_secret"] == "prompted-secret"
    assert oauth["redirect_uri"] == "http://127.0.0.1:8080/callback"
    assert "prompted-secret" not in capsys.readouterr().out


def test_mcp_add_verify_connects_after_saving(tmp_path: Path, capsys, monkeypatch) -> None:
    project, state = _project(tmp_path)
    calls: list[tuple[str, dict]] = []

    async def fake_verify(self, server_id, **kwargs):
        calls.append((server_id, kwargs))
        return MCPVerificationResult(server_id, True, "Verified 2 tool(s).", 2)

    monkeypatch.setattr(MCPService, "verify_server", fake_verify)

    exit_code = main(_mcp(project, state, "add", "docs", "https://docs.example/mcp", "--verify"))

    assert exit_code == 0
    assert calls == [("docs", {"interactive_oauth": True, "open_browser": True, "output": sys.stderr})]
    assert "✓ docs: Verified 2 tool(s)." in capsys.readouterr().out


def test_mcp_add_json_accepts_entry_wrapper_file_and_stdin(tmp_path: Path, capsys, monkeypatch) -> None:
    project, state = _project(tmp_path)
    block = tmp_path / "block.json"
    block.write_text(
        json.dumps({"mcpServers": {"from-file": {"command": "uvx", "args": ["x"]}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.stdin", io.StringIO('{"mcpServers": {"from-stdin": {"command": "npx"}}}'))

    for argv in (
        (
            "add-json",
            "weather-api",
            '{"type":"http","url":"https://api.weather.com/mcp","headers":{"Authorization":"Bearer ${WEATHER_TOKEN}"}}',
        ),
        (
            "add-json",
            "local-weather",
            '{"mcpServers":{"local-weather":{"type":"stdio","command":"/path/to/weather-cli",'
            '"args":["--api-key","abc123"],"env":{"CACHE_DIR":"/tmp"}}}}',
        ),
        (
            "add-json",
            '{"mcpServers":{"a":{"type":"http","url":"https://a.example/mcp"},'
            '"b":{"url":"https://b.example/mcp","timeout":5000,"headers":{"X-Num":7}}}}',
        ),
        ("add-json", str(block)),
        ("add-json", "-"),
    ):
        assert main(_mcp(project, state, *argv)) == 0
    capsys.readouterr()

    servers = _user_servers(state)
    assert set(servers) == {"weather-api", "local-weather", "a", "b", "from-file", "from-stdin"}
    assert servers["weather-api"]["headers"] == {"Authorization": "Bearer ${WEATHER_TOKEN}"}
    assert servers["local-weather"]["args"] == ["--api-key", "abc123"]
    # A url without a type is inferred, a millisecond timeout becomes seconds, and a
    # numeric header value is stored as a string.
    assert servers["b"]["transport"] == "streamable_http"
    assert servers["b"]["timeout_seconds"] == 5.0
    assert servers["b"]["headers"] == {"X-Num": "7"}


def test_mcp_add_json_rejects_a_name_missing_from_the_block(tmp_path: Path, capsys) -> None:
    project, state = _project(tmp_path)

    exit_code = main(_mcp(project, state, "add-json", "missing", '{"mcpServers":{"other":{"command":"npx"}}}'))

    assert exit_code == 2
    assert "missing" in capsys.readouterr().err
    assert not global_mcp_config_path(state).exists()


def test_mcp_get_masks_secrets_and_strips_control_characters(tmp_path: Path, capsys) -> None:
    project, state = _project(tmp_path)
    (project / ".mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "repo": {
                        "type": "http",
                        "url": "https://example.com/mcp\x1b[2J",
                        "headers": {"Authorization": "Bearer repo-secret"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    assert main(_mcp(project, state, "add", "-e", "TOKEN=sk-live-123", "svc", "--", "npx", "svc")) == 0
    assert main(_mcp(project, state, "trust")) == 0
    capsys.readouterr()

    assert main(_mcp(project, state, "get", "repo", "--json")) == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["scope"] == "project"
    assert data["file"] == str(project / ".mcp.json")
    assert data["endpoint"] == {"url": "https://example.com/mcp[2J"}
    assert data["detail"]["header_keys"] == ["Authorization"]
    assert "repo-secret" not in captured.out + captured.err

    assert main(_mcp(project, state, "get", "svc")) == 0
    captured = capsys.readouterr()
    assert "Env:         TOKEN=‹hidden›" in captured.out
    assert "sk-live-123" not in captured.out + captured.err

    assert main(_mcp(project, state, "list", "--json")) == 0
    captured = capsys.readouterr()
    listed = {server["id"]: server for server in json.loads(captured.out)["servers"]}
    assert listed["repo"]["scope"] == "project"
    assert listed["svc"]["scope"] == "user"
    assert "repo-secret" not in captured.out
    assert "sk-live-123" not in captured.out


def test_mcp_remove_needs_a_scope_when_a_name_is_defined_twice(tmp_path: Path, capsys) -> None:
    project, state = _project(tmp_path)
    assert main(_mcp(project, state, "add", "docs", "https://user.example/mcp")) == 0
    assert main(_mcp(project, state, "add", "-s", "project", "docs", "https://team.example/mcp")) == 0
    capsys.readouterr()

    exit_code = main(_mcp(project, state, "remove", "docs"))

    assert exit_code == 2
    assert "more than one scope (project, user)" in capsys.readouterr().err
    assert main(_mcp(project, state, "remove", "docs", "-s", "project")) == 0
    assert set(_user_servers(state)) == {"docs"}
    assert json.loads((project / ".mcp.json").read_text()) == {"mcpServers": {}}


def test_mcp_project_scope_writes_the_portable_file_unless_a_legacy_one_exists(tmp_path: Path) -> None:
    project, state = _project(tmp_path)

    assert main(_mcp(project, state, "add", "-s", "project", "first", "https://a.example/mcp")) == 0
    assert (project / ".mcp.json").is_file()
    assert not (project / ".kolega" / "mcp_servers.json").exists()

    legacy_project = tmp_path / "legacy"
    (legacy_project / ".kolega").mkdir(parents=True)
    (legacy_project / ".kolega" / "mcp_servers.json").write_text('{"schema_version": 1, "servers": []}')
    assert main(_mcp(legacy_project, state, "add", "--project-config", "old", "https://b.example/mcp")) == 0
    assert [
        server["id"] for server in json.loads((legacy_project / ".kolega" / "mcp_servers.json").read_text())["servers"]
    ] == ["old"]
    assert not (legacy_project / ".mcp.json").exists()


def test_mcp_add_to_project_scope_warns_about_literal_secrets_in_a_shared_file(tmp_path: Path, capsys) -> None:
    project, state = _project(tmp_path)

    assert (
        main(
            _mcp(
                project,
                state,
                "add",
                "-H",
                "Authorization: Bearer literal",
                "-s",
                "project",
                "a",
                "https://a.example/mcp",
            )
        )
        == 0
    )
    assert (
        main(
            _mcp(
                project,
                state,
                "add",
                "-H",
                "Authorization: Bearer ${TOKEN}",
                "-s",
                "project",
                "b",
                "https://b.example/mcp",
            )
        )
        == 0
    )

    err = capsys.readouterr().err
    assert err.count("this file is shared") == 1
    assert "Run `kolega-code mcp trust`" in err


def test_mcp_disable_of_a_project_server_records_a_private_choice(tmp_path: Path, capsys) -> None:
    project, state = _project(tmp_path)
    assert main(_mcp(project, state, "add", "-s", "project", "ci", "https://ci.example/mcp")) == 0
    shared_file = project / ".mcp.json"
    original = shared_file.read_bytes()
    capsys.readouterr()

    assert main(_mcp(project, state, "disable", "ci")) == 0
    assert "Disabled MCP server ci for this project" in capsys.readouterr().out
    assert shared_file.read_bytes() == original
    assert SettingsStore(state).load().mcp_project_server_overrides == {str(project.resolve()): {"ci": False}}

    assert main(_mcp(project, state, "trust")) == 0
    assert main(_mcp(project, state, "list")) == 0
    out = capsys.readouterr().out
    assert "⊘ ci" in out
    assert "disabled for this project" in out

    assert main(_mcp(project, state, "enable", "ci")) == 0
    assert SettingsStore(state).load().mcp_project_server_overrides == {str(project.resolve()): {"ci": True}}
    assert shared_file.read_bytes() == original

    assert main(_mcp(project, state, "reset-project-choices")) == 0
    assert "Reset the trust decision and 1 enable/disable choice" in capsys.readouterr().out
    settings = SettingsStore(state).load()
    assert settings.mcp_project_server_overrides == {}
    assert not settings.is_mcp_project_trusted(project)


def test_mcp_local_scope_is_private_to_one_project(tmp_path: Path, capsys) -> None:
    project, state = _project(tmp_path)
    other = tmp_path / "other"
    other.mkdir()

    assert main(_mcp(project, state, "add", "-s", "local", "db", "--", "uvx", "my-db-mcp")) == 0
    assert "Added MCP server db to local scope" in capsys.readouterr().out
    assert set(SettingsStore(state).load().mcp_local_servers) == {str(project.resolve())}

    assert main(_mcp(project, state, "list")) == 0
    out = capsys.readouterr().out
    assert "○ db" in out
    assert " local " in out

    assert main(_mcp(other, state, "list")) == 0
    assert "db" not in capsys.readouterr().out
    assert main(_mcp(other, state, "get", "db")) == 2
    assert main(_mcp(project, state, "disable", "db")) == 0
    assert main(_mcp(project, state, "list")) == 0
    assert "⊘ db" in capsys.readouterr().out
    assert main(_mcp(project, state, "remove", "db")) == 0
    assert SettingsStore(state).load().mcp_local_servers == {}


def test_mcp_trust_untrust_and_pending_approval_listing(tmp_path: Path, capsys) -> None:
    project, state = _project(tmp_path)
    (project / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"repo": {"command": "npx", "args": ["x"]}}}), encoding="utf-8"
    )

    assert main(_mcp(project, state, "list")) == 0
    out = capsys.readouterr().out
    assert "⏸ repo" in out
    assert "Pending approval" in out
    assert "project not trusted" in out

    assert main(_mcp(project, state, "trust")) == 0
    assert SettingsStore(state).load().is_mcp_project_trusted(project)
    assert main(_mcp(project, state, "list")) == 0
    out = capsys.readouterr().out
    assert "○ repo" in out
    assert "Not connected" in out

    assert main(_mcp(project, state, "untrust")) == 0
    assert not SettingsStore(state).load().is_mcp_project_trusted(project)
    assert main(_mcp(project, state, "untrust")) == 0
    assert "was not trusted" in capsys.readouterr().out


def test_mcp_login_requires_oauth_and_logout_clears_sign_in(tmp_path: Path, capsys, monkeypatch) -> None:
    project, state = _project(tmp_path)
    assert main(_mcp(project, state, "add", "plain", "https://plain.example/mcp")) == 0
    assert "mcp verify plain" in capsys.readouterr().out
    assert main(_mcp(project, state, "add", "hub", "https://hub.example/mcp", "--oauth")) == 0
    added = capsys.readouterr().out
    assert "Run `kolega-code mcp login hub` to sign in and expose its tools." in added
    assert main(_mcp(project, state, "list")) == 0
    never_signed_in = capsys.readouterr().out
    assert "! hub" in never_signed_in and "Needs auth" in never_signed_in
    assert "sign in: kolega-code mcp login hub" in never_signed_in

    assert main(_mcp(project, state, "login", "plain")) == 2
    assert "has no OAuth configured" in capsys.readouterr().err

    calls: list[tuple[str, dict]] = []

    async def fake_verify(self, server_id, **kwargs):
        calls.append((server_id, kwargs))
        return MCPVerificationResult(server_id, True, "Verified 1 tool(s).", 1)

    monkeypatch.setattr(MCPService, "verify_server", fake_verify)
    assert main(_mcp(project, state, "login", "hub", "--no-browser")) == 0
    assert calls == [("hub", {"interactive_oauth": True, "open_browser": False, "output": sys.stderr})]
    assert "✓ hub" in capsys.readouterr().out

    status_store = MCPStatusStore(state)
    token_store = MCPOAuthTokenStore(state)
    status_store.update(
        "hub", MCPServerStatus.verified(fingerprint="fp", transport="streamable_http", source="global", tools=[])
    )
    token_store.set_tokens("hub", {"access_token": "access", "refresh_token": "refresh"})
    assert main(_mcp(project, state, "logout", "hub")) == 0
    assert status_store.get("hub") is None
    assert token_store.secret_values() == []
    signed_out = capsys.readouterr().out
    assert "Signed out of MCP server hub" in signed_out
    assert "Run `kolega-code mcp login hub` to sign in again." in signed_out
    assert main(_mcp(project, state, "logout", "plain")) == 0
    assert "sign in again" not in capsys.readouterr().out


def test_mcp_list_check_verifies_enabled_servers_only(tmp_path: Path, capsys, monkeypatch) -> None:
    project, state = _project(tmp_path)
    assert main(_mcp(project, state, "add", "a", "https://a.example/mcp")) == 0
    assert main(_mcp(project, state, "add", "b", "https://b.example/mcp")) == 0
    assert main(_mcp(project, state, "disable", "b")) == 0
    capsys.readouterr()
    calls: list[tuple[str, dict]] = []

    async def failing_verify(self, server_id, **kwargs):
        calls.append((server_id, kwargs))
        server = self.config.servers[server_id]
        self.status_store.update(
            server_id,
            MCPServerStatus.failed(
                fingerprint=server_fingerprint(server),
                transport=server.transport,
                source=server.source,
                message=MCP_FAILURE_MESSAGE_GENERIC,
            ),
        )
        return MCPVerificationResult(server_id, False, MCP_FAILURE_MESSAGE_GENERIC)

    monkeypatch.setattr(MCPService, "verify_server", failing_verify)

    exit_code = main(_mcp(project, state, "list", "--check", "--json"))
    captured = capsys.readouterr()

    assert exit_code == 1
    assert [server_id for server_id, _ in calls] == ["a"]
    assert calls[0][1] == {"interactive_oauth": False, "open_browser": False, "output": sys.stderr}
    listed = {server["id"]: server for server in json.loads(captured.out)["servers"]}
    assert listed["a"]["health"] == "failed"
    assert listed["b"]["health"] == "disabled"

    async def passing_verify(self, server_id, **kwargs):
        server = self.config.servers[server_id]
        self.status_store.update(
            server_id,
            MCPServerStatus.verified(
                fingerprint=server_fingerprint(server),
                transport=server.transport,
                source=server.source,
                tools=[MCPToolStatus(id="ping")],
            ),
        )
        return MCPVerificationResult(server_id, True, "Verified 1 tool(s).", 1)

    monkeypatch.setattr(MCPService, "verify_server", passing_verify)

    assert main(_mcp(project, state, "list", "--check", "--json")) == 0
    listed = {server["id"]: server for server in json.loads(capsys.readouterr().out)["servers"]}
    assert listed["a"]["health"] == "connected"
    assert listed["a"]["tool_count"] == 1


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_mcp_help_lists_commands_scopes_and_options(flag: str, capsys) -> None:
    def collapsed_help(argv: list[str]) -> str:
        with pytest.raises(SystemExit) as excinfo:
            main(argv)
        assert excinfo.value.code == 0
        return " ".join(capsys.readouterr().out.split())

    command_help = collapsed_help(["mcp", flag])
    for command in ("add-json", "get", "login", "logout", "trust", "reset-project-choices"):
        assert command in command_help
    add_help = collapsed_help(["mcp", "add", flag])
    for option in ("-s SCOPE", "--scope", "--callback-port", "--client-secret", "--transport"):
        assert option in add_help
    assert "--check" in collapsed_help(["mcp", "list", flag])
