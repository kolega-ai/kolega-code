from __future__ import annotations

import json
from pathlib import Path

from kolega_code.cli.config import API_KEY_ENV, CliConfigOverrides, build_agent_config
from kolega_code.cli.main import main, parse_args
from kolega_code.cli.settings import CliSettings, SettingsStore
from kolega_code.config import ModelProvider
from kolega_code.mcp.config import MCPServerConfig, global_mcp_config_path, server_fingerprint
from kolega_code.mcp.service import MCP_FAILURE_MESSAGE_GENERIC
from kolega_code.mcp.state import MCPServerStatus, MCPStatusStore


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
    assert "docs" in captured.out
    assert "streamable_http" in captured.out


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
    assert "legacy-secret" not in captured.out
    assert MCP_FAILURE_MESSAGE_GENERIC in captured.out


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
