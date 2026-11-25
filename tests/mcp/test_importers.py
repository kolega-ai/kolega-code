"""Importers read each tool's real config shape and report what cannot be carried over."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kolega_code.cli.settings import CliSettings, SettingsStore
from kolega_code.mcp.config import MCPServerConfig
from kolega_code.mcp.entries import MCPEntryError
from kolega_code.mcp.importers import (
    IMPORT_SOURCE_IDS,
    ImportCandidate,
    import_file,
    import_servers,
    sanitize_import_name,
)
from kolega_code.mcp.importers import _config_dir as config_dir
from kolega_code.mcp.importers import discover_import_sources


def _write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    return home


def _environment(home: Path) -> dict[str, str]:
    return {
        "HOME": str(home),
        "USERPROFILE": str(home),
        "APPDATA": str(home / "AppData" / "Roaming"),
        "XDG_CONFIG_HOME": str(home / ".config"),
    }


def _found(name: str, url: str, source: str = "cursor") -> ImportCandidate:
    return ImportCandidate(source, source.title(), "user", name, server=MCPServerConfig(id=name, url=url))


def test_sanitize_import_name_keeps_names_recognizable_and_valid() -> None:
    assert sanitize_import_name("My Tool.v2") == "My-Tool-v2"
    assert sanitize_import_name("  __github__  ") == "__github__"
    assert len(sanitize_import_name("x" * 40)) == 32
    with pytest.raises(MCPEntryError):
        sanitize_import_name("!!!")


def test_only_tools_that_define_servers_are_offered(tmp_path: Path, home: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    _write(home / ".cursor" / "mcp.json", {"mcpServers": {"context7": {"command": "npx"}}})
    # A config that exists but holds no servers is not a source.
    _write(home / ".gemini" / "settings.json", {"mcpServers": {}, "theme": "Default"})
    _write(
        home / ".codex" / "config.toml",
        '[mcp_servers.docs]\ncommand = "uvx"\n\n[mcp_servers.remote]\nurl = "https://example.invalid/mcp"\n',
    )

    sources = discover_import_sources(project, _environment(home))

    assert [source.id for source in sources] == ["cursor", "codex"]
    assert sources[0].files == (home / ".cursor" / "mcp.json",)
    assert [candidate.name for candidate in sources[1].candidates] == ["docs", "remote"]
    assert all(source_id in IMPORT_SOURCE_IDS for source_id in ("claude-code", "claude-desktop", "vscode", "windsurf"))


def test_claude_code_reads_user_servers_and_this_projects_private_ones(tmp_path: Path, home: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    _write(
        home / ".claude.json",
        {
            "numStartups": 9,
            "mcpServers": {"shared": {"type": "http", "url": "https://shared.invalid/mcp"}},
            "projects": {
                # Recorded with an unresolved path, as Claude Code stores it.
                str(project): {"allowedTools": [], "mcpServers": {"mine": {"command": "npx", "args": ["local-one"]}}},
                str(tmp_path / "elsewhere"): {"mcpServers": {"theirs": {"command": "npx"}}},
            },
        },
    )

    candidates = discover_import_sources(project, _environment(home))[0].candidates

    assert [(candidate.name, candidate.origin) for candidate in candidates] == [("shared", "user"), ("mine", "local")]


def test_codex_tables_map_onto_portable_entries(tmp_path: Path, home: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    _write(
        home / ".codex" / "config.toml",
        """
model = "gpt"
[mcp_servers.docs]
command = "uvx"
args = ["mcp-docs"]
cwd = "/tmp/docs"
enabled = false
startup_timeout_sec = 20
env_vars = ["GITHUB_TOKEN"]
env = { NODE_ENV = "production" }
tools = { ping = { approval_mode = "auto" } }

[mcp_servers.remote]
url = "https://example.invalid/mcp"
bearer_token_env_var = "EXAMPLE_TOKEN"

[mcp_servers.broken]
startup_timeout_sec = 5
""",
    )

    docs, remote, broken = discover_import_sources(project, _environment(home))[0].candidates

    assert docs.importable and docs.server is not None
    assert (docs.server.transport, docs.server.command, docs.server.args) == ("stdio", "uvx", ["mcp-docs"])
    assert docs.server.cwd == "/tmp/docs"
    assert docs.server.enabled is False
    assert docs.server.timeout_seconds == 20.0
    # Named shell variables are forwarded by reference, so the value never lands in a file.
    assert docs.server.env == {"NODE_ENV": "production", "GITHUB_TOKEN": "${GITHUB_TOKEN}"}
    assert docs.dropped_fields == ("tools",)

    assert remote.importable and remote.server is not None
    assert remote.server.transport == "streamable_http"
    assert remote.server.headers == {"Authorization": "Bearer ${EXAMPLE_TOKEN}"}
    assert remote.dropped_fields == ()

    assert not broken.importable
    assert broken.unsupported == "has neither a command nor a url"


def test_codex_discovery_keeps_valid_servers_when_siblings_have_the_wrong_shape(tmp_path: Path, home: Path) -> None:
    _write(
        home / ".codex" / "config.toml",
        """
[mcp_servers]
not_a_table = "invalid"

[mcp_servers.bad_env]
command = "uvx"
env = "invalid"

[mcp_servers.bad_timeout]
command = "uvx"
startup_timeout_sec = inf

[mcp_servers.good]
command = "uvx"
env = { EXAMPLE = "value" }
""",
    )

    (source,) = discover_import_sources(tmp_path, _environment(home))
    candidates = {candidate.original_name: candidate for candidate in source.candidates}

    assert "table" in candidates["not_a_table"].unsupported
    assert "env" in candidates["bad_env"].unsupported
    assert "startup_timeout_sec" in candidates["bad_timeout"].unsupported
    assert candidates["good"].server is not None
    assert candidates["good"].server.env == {"EXAMPLE": "value"}


def test_codex_wrong_shaped_server_block_is_reported(tmp_path: Path) -> None:
    path = _write(tmp_path / "config.toml", 'mcp_servers = ["invalid"]')

    (candidate,) = import_file(path, tmp_path).candidates

    assert not candidate.importable
    assert candidate.original_name == "mcp_servers"
    assert "table" in candidate.unsupported


def test_windsurf_remote_server_url_is_imported_with_headers(tmp_path: Path, home: Path) -> None:
    path = _write(
        home / ".codeium" / "windsurf" / "mcp_config.json",
        {
            "mcpServers": {
                "remote": {
                    "serverUrl": "https://example.invalid/mcp",
                    "headers": {"Authorization": "Bearer ${EXAMPLE_TOKEN}"},
                }
            }
        },
    )

    (source,) = discover_import_sources(tmp_path, _environment(home))
    for discovered_source in (source, import_file(path, tmp_path)):
        (candidate,) = discovered_source.candidates
        assert candidate.server is not None
        assert candidate.server.transport == "streamable_http"
        assert candidate.server.url == "https://example.invalid/mcp"
        assert candidate.server.headers == {"Authorization": "Bearer ${EXAMPLE_TOKEN}"}
        assert candidate.dropped_fields == ()


def test_json_sources_accept_each_wrapper_and_field_spelling(tmp_path: Path, home: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    _write(
        project / ".vscode" / "mcp.json", {"servers": {"code": {"type": "stdio", "command": "npx", "env": {"N": 7}}}}
    )
    _write(
        home / ".gemini" / "settings.json",
        {"mcpServers": {"gem": {"httpUrl": "https://gem.invalid/sse"}, "site": {"url": "https://site.invalid/mcp"}}},
    )
    _write(home / ".codeium" / "windsurf" / "mcp_config.json", {"mcpServers": {"wind": {"command": "uvx"}}})
    _write(
        config_dir(_environment(home), "Claude") / "claude_desktop_config.json",
        {"mcpServers": {"desk": {"command": "node"}}},
    )
    _write(
        home / ".copilot" / "mcp-config.json",
        {"mcpServers": {"copilot": {"type": "http", "url": "https://copilot.invalid/mcp"}}},
    )

    sources = {source.id: source for source in discover_import_sources(project, _environment(home))}

    assert set(sources) == {"claude-desktop", "vscode", "windsurf", "gemini"}
    vscode = sources["vscode"].candidates
    assert [(candidate.name, candidate.origin) for candidate in vscode] == [("code", "project"), ("copilot", "user")]
    assert vscode[0].server is not None and vscode[0].server.env == {"N": "7"}
    gemini = sources["gemini"].candidates
    assert gemini[0].server is not None and gemini[0].server.transport == "sse"
    assert gemini[1].server is not None and gemini[1].server.transport == "streamable_http"
    assert sources["claude-desktop"].candidates[0].name == "desk"
    assert sources["windsurf"].candidates[0].name == "wind"


def test_unusable_entries_are_reported_with_a_reason_instead_of_raising(tmp_path: Path, home: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    _write(
        home / ".cursor" / "mcp.json",
        {
            "mcpServers": {
                "socket": {"type": "ws", "url": "wss://example.invalid"},
                "in-process": {"type": "sdk"},
                "!!!": {"command": "npx"},
                "headless": {"type": "stdio"},
                "My Tool.v2": {"command": "uvx", "args": ["--token", "abc"], "alwaysLoad": True},
                "remote": {"type": "http", "url": "https://example.invalid/mcp?key=abc#frag", "timeout": 500},
            }
        },
    )

    socket, in_process, unusable, headless, renamed, remote = discover_import_sources(project, _environment(home))[
        0
    ].candidates

    assert (socket.unsupported, socket.importable, socket.endpoint) == ("unsupported WebSocket type", False, "")
    assert in_process.unsupported == "unsupported SDK-hosted type"
    assert unusable.unsupported == "this server's name has no letters, numbers, '_' or '-' in it"
    assert "requires command" in headless.unsupported
    assert renamed.importable and renamed.name == "My-Tool-v2" and renamed.renamed
    assert renamed.notes == 'name was "My Tool.v2" · dropped alwaysLoad'
    # The summary never carries arguments, query strings or fragments.
    assert renamed.endpoint == "uvx (+2 args)"
    assert remote.endpoint == "https://example.invalid/mcp"


def test_import_file_reads_json_and_toml_by_shape(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    block = _write(tmp_path / "block.json", {"mcpServers": {"a": {"command": "npx"}}})
    toml = _write(tmp_path / "custom.toml", '[mcp_servers.b]\ncommand = "uvx"\n')

    assert [candidate.name for candidate in import_file(block, project).candidates] == ["a"]
    assert [candidate.name for candidate in import_file(toml, project).candidates] == ["b"]
    with pytest.raises(ValueError, match="Import file not found"):
        import_file(tmp_path / "missing.json", project)


def test_import_servers_adds_skips_and_names_which_definition_wins(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    state = tmp_path / "state"
    settings = CliSettings()
    settings.mcp_local_servers[str(project.resolve())] = {
        "private": {"transport": "streamable_http", "url": "https://old.invalid/mcp"}
    }
    from kolega_code.mcp.config import save_config_file, MCPConfigFile, global_mcp_config_path

    save_config_file(
        global_mcp_config_path(state),
        MCPConfigFile(servers=[MCPServerConfig(id="docs", url="https://user.invalid/mcp")]),
    )
    new = _found("new", "https://new.invalid/mcp")
    docs = _found("docs", "https://team.invalid/mcp")
    lower = _found("private", "https://lower.invalid/mcp")
    unsupported = ImportCandidate("cursor", "Cursor", "user", "socket", unsupported="unsupported WebSocket type")
    context = {"project_path": project, "state_dir": state, "settings": settings}

    user_outcomes = import_servers([new, docs, lower, unsupported], scope="user", **context)

    assert [(outcome.name, outcome.outcome, outcome.detail) for outcome in user_outcomes] == [
        ("new", "added", ""),
        ("docs", "skipped", "already defined in user scope"),
        ("private", "added", "the local definition takes precedence"),
        ("socket", "skipped", "unsupported WebSocket type"),
    ]
    team_outcomes = import_servers([docs], scope="project", **context)

    assert team_outcomes == [type(team_outcomes[0])("docs", "added", "takes precedence over the user definition")]
    assert (project / ".mcp.json").exists()
    assert import_servers([docs], scope="user", overwrite=True, **context)[0].outcome == "updated"

    local = _found("mine", "https://mine.invalid/mcp")
    import_servers([local], scope="local", **context)
    # A local import edits the in-memory settings; the caller persists them.
    assert set(settings.mcp_local_servers[str(project.resolve())]) == {"private", "mine"}
    assert not SettingsStore(state).path.exists()
