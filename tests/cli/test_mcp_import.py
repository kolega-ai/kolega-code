"""``kolega-code mcp import``: the unattended flags and the interactive picker."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from kolega_code.cli import mcp_cli
from kolega_code.cli.main import main
from kolega_code.cli.settings import SettingsStore
from kolega_code.mcp.config import global_mcp_config_path

CODEX = """[mcp_servers.docs]
command = "uvx"
args = ["mcp-docs"]
env_vars = ["DOCS_TOKEN"]

[mcp_servers.remote]
url = "https://example.invalid/mcp"
bearer_token_env_var = "EXAMPLE_TOKEN"
"""


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    for name in ("HOME", "USERPROFILE", "APPDATA", "XDG_CONFIG_HOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    return home


def _project(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    project.mkdir()
    return project, tmp_path / "state"


def _mcp(project: Path, state: Path, *args: str) -> list[str]:
    return ["mcp", "--project", str(project), "--state-dir", str(state), *args]


def _tools(home: Path) -> None:
    _write(
        home / ".cursor" / "mcp.json",
        {
            "mcpServers": {
                "Context Tool": {"command": "npx", "args": ["-y", "ctx"], "env": {"API_KEY": 7}},
                "socket": {"type": "ws", "url": "wss://example.invalid"},
            }
        },
    )
    _write(home / ".codex" / "config.toml", CODEX)
    # A tool whose config exists without servers is never offered.
    _write(home / ".gemini" / "settings.json", {"mcpServers": {}})


def _user_servers(state: Path) -> dict[str, dict]:
    return {server["id"]: server for server in json.loads(global_mcp_config_path(state).read_text())["servers"]}


def test_parse_selection_reads_numbers_ranges_and_words() -> None:
    assert mcp_cli._parse_selection("", 5) == {0, 1, 2, 3, 4}
    assert mcp_cli._parse_selection("ALL", 5) == {0, 1, 2, 3, 4}
    assert mcp_cli._parse_selection("none", 5) == set()
    assert mcp_cli._parse_selection("1,3-4 2", 5) == {0, 1, 2, 3}
    assert mcp_cli._parse_selection("9", 5) == set()
    with pytest.raises(ValueError, match="Could not read the selection 'soon'"):
        mcp_cli._parse_selection("soon", 5)


def test_import_without_any_found_source_says_where_it_looked(tmp_path: Path, home: Path, capsys) -> None:
    project, state = _project(tmp_path)

    exit_code = main(_mcp(project, state, "import", "--scope", "user", "--yes"))

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "No MCP servers found in other tools." in out
    assert "claude-code" in out and "codex" in out and "windsurf" in out
    assert "--file PATH" in out


def test_unattended_import_explains_the_flags_it_needs(tmp_path: Path, home: Path, capsys) -> None:
    project, state = _project(tmp_path)
    _tools(home)

    assert main(_mcp(project, state, "import")) == 2
    assert "needs --from SOURCE or --file PATH, and --scope plus --yes or --dry-run" in capsys.readouterr().err
    assert main(_mcp(project, state, "import", "--from", "codex")) == 2
    assert "needs --scope plus --yes or --dry-run" in capsys.readouterr().err
    assert main(_mcp(project, state, "import", "--from", "codex", "--scope", "user")) == 2
    assert not global_mcp_config_path(state).exists()


def test_import_names_the_sources_it_did_find(tmp_path: Path, home: Path, capsys) -> None:
    project, state = _project(tmp_path)
    _tools(home)

    exit_code = main(_mcp(project, state, "import", "--from", "gemini", "--scope", "user", "--yes"))

    assert exit_code == 2
    assert (
        "No MCP servers were found for 'gemini'. Sources with servers here: cursor, codex." in capsys.readouterr().err
    )


def test_dry_run_reports_what_would_be_added_and_written_nothing(tmp_path: Path, home: Path, capsys) -> None:
    project, state = _project(tmp_path)
    _tools(home)

    exit_code = main(
        _mcp(project, state, "import", "--from", "cursor", "--from", "codex", "--scope", "user", "--dry-run")
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert 'Would add Context-Tool to user scope (npx (+2 args)) · name was "Context Tool"' in out
    assert "Would add docs to user scope (uvx (+1 args))" in out
    assert "Would add remote to user scope (https://example.invalid/mcp)" in out
    assert "Skipped socket: unsupported WebSocket type" in out
    assert not global_mcp_config_path(state).exists()


def test_unattended_import_writes_skips_existing_and_overwrites_on_request(tmp_path: Path, home: Path, capsys) -> None:
    project, state = _project(tmp_path)
    _tools(home)
    flags = ("--from", "cursor", "--from", "codex", "--scope", "user", "--yes")

    assert main(_mcp(project, state, "import", *flags)) == 0

    out = capsys.readouterr().out
    assert "Added MCP server Context-Tool (user scope)" in out
    assert "Added MCP server docs (user scope)" in out
    assert "Skipped socket: unsupported WebSocket type" in out
    assert "kolega-code mcp verify --all" in out
    servers = _user_servers(state)
    assert set(servers) == {"Context-Tool", "docs", "remote"}
    assert servers["Context-Tool"]["env"] == {"API_KEY": "7"}
    assert servers["docs"]["env"] == {"DOCS_TOKEN": "${DOCS_TOKEN}"}
    assert servers["remote"]["headers"] == {"Authorization": "Bearer ${EXAMPLE_TOKEN}"}

    assert main(_mcp(project, state, "import", *flags)) == 0
    out = capsys.readouterr().out
    assert "Added MCP server" not in out
    assert out.count("already defined in user scope") == 3

    _write(home / ".cursor" / "mcp.json", {"mcpServers": {"Context Tool": {"command": "uvx", "args": ["ctx2"]}}})
    assert main(_mcp(project, state, "import", "--from", "cursor", "--scope", "user", "--yes", "--overwrite")) == 0
    assert "Updated MCP server Context-Tool (user scope)" in capsys.readouterr().out
    assert _user_servers(state)["Context-Tool"]["command"] == "uvx"


def test_import_server_narrows_by_original_name(tmp_path: Path, home: Path, capsys) -> None:
    project, state = _project(tmp_path)
    _tools(home)

    exit_code = main(
        _mcp(project, state, "import", "--from", "cursor", "--scope", "local", "--yes", "--server", "Context Tool")
    )

    assert exit_code == 0
    assert "Added MCP server Context-Tool (local scope)" in capsys.readouterr().out
    assert not global_mcp_config_path(state).exists()
    assert set(SettingsStore(state).load().mcp_local_servers) == {str(project.resolve())}


def test_import_from_an_explicit_file(tmp_path: Path, home: Path, capsys) -> None:
    project, state = _project(tmp_path)
    block = tmp_path / "block.json"
    _write(block, {"mcpServers": {"from-file": {"type": "http", "url": "https://file.invalid/mcp"}}})

    assert main(_mcp(project, state, "import", "--file", str(block), "--scope", "project", "--yes")) == 0
    assert "Added MCP server from-file (project scope)" in capsys.readouterr().out
    assert set(json.loads((project / ".mcp.json").read_text())["mcpServers"]) == {"from-file"}

    assert (
        main(_mcp(project, state, "import", "--file", str(tmp_path / "missing.json"), "--scope", "user", "--yes")) == 2
    )
    assert "Import file not found" in capsys.readouterr().err


def test_interactive_picker_offers_found_sources_then_servers(tmp_path: Path, home: Path, capsys, monkeypatch) -> None:
    project, state = _project(tmp_path)
    _tools(home)
    monkeypatch.setattr(mcp_cli, "_stdin_is_interactive", lambda: True)
    monkeypatch.setattr("sys.stdin", io.StringIO("\n\n\n"))

    exit_code = main(_mcp(project, state, "import"))

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Found MCP servers in 2 tools:" in out
    assert "1. Cursor" in out and "2. Codex" in out
    assert "Gemini" not in out
    assert "[x] 1. Context Tool" in out
    assert "unsupported WebSocket type" in out
    assert "Add to scope [user, project, local] (Enter for user):" in out
    assert "Added MCP server Context-Tool (user scope)" in out
    assert set(_user_servers(state)) == {"Context-Tool", "docs", "remote"}


def test_interactive_picker_honours_a_narrower_selection(tmp_path: Path, home: Path, capsys, monkeypatch) -> None:
    project, state = _project(tmp_path)
    _tools(home)
    monkeypatch.setattr(mcp_cli, "_stdin_is_interactive", lambda: True)
    # Cursor only, into the local scope, only its first server.
    monkeypatch.setattr("sys.stdin", io.StringIO("1\nlocal\n1\n"))

    exit_code = main(_mcp(project, state, "import"))

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Servers found in Cursor:" in out
    assert "Added MCP server Context-Tool (local scope)" in out
    assert "Added MCP server docs" not in out
    assert not global_mcp_config_path(state).exists()
    assert list(SettingsStore(state).load().mcp_local_servers[str(project.resolve())]) == ["Context-Tool"]
