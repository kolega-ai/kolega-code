"""Discover MCP servers configured in other tools and normalize them for import.

Every tool stores servers a little differently. Each source knows where its files live
and how to read them; everything ends in one portable entry shape and then in
``standard_entry_payload``.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

from .config import MCPScope, MCPScopeSettings, MCPServerConfig, sanitize_mcp_server_id, server_from_entry
from .entries import MCPEntryError, UnsupportedMCPTypeError, standard_entry_payload
from .scopes import find_definitions, upsert_server

IMPORT_SOURCE_IDS = ("claude-code", "claude-desktop", "cursor", "vscode", "windsurf", "gemini", "codex")
SOURCE_LABELS = {
    "claude-code": "Claude Code",
    "claude-desktop": "Claude Desktop",
    "cursor": "Cursor",
    "vscode": "VS Code / Copilot",
    "windsurf": "Windsurf",
    "gemini": "Gemini CLI",
    "codex": "Codex",
}
_UNUSABLE_NAME = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass(frozen=True)
class ImportCandidate:
    """One server found in another tool, ready to write or reported as unimportable."""

    source_id: str
    source_label: str
    origin: str
    original_name: str
    server: Optional[MCPServerConfig] = None
    unsupported: str = ""
    dropped_fields: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return self.server.id if self.server is not None else self.original_name

    @property
    def importable(self) -> bool:
        return self.server is not None

    @property
    def renamed(self) -> bool:
        return self.server is not None and self.server.id != self.original_name

    @property
    def endpoint(self) -> str:
        """A secret-free summary: no query strings, arguments or header values."""
        if self.server is None:
            return ""
        if self.server.transport == "stdio":
            command = self.server.command or ""
            if not self.server.args:
                return command
            return f"{command} (+{len(self.server.args)} args)"
        return (self.server.url or "").split("?", 1)[0]

    @property
    def notes(self) -> str:
        parts = []
        if self.unsupported:
            parts.append(self.unsupported)
        if self.renamed:
            parts.append(f'name was "{self.original_name}"')
        if self.dropped_fields:
            parts.append("dropped " + ", ".join(self.dropped_fields))
        parts.extend(self.warnings)
        return " · ".join(parts)


@dataclass(frozen=True)
class ImportSource:
    id: str
    label: str
    candidates: tuple[ImportCandidate, ...] = ()
    files: tuple[Path, ...] = ()

    @property
    def origins(self) -> str:
        return ", ".join(dict.fromkeys(candidate.origin for candidate in self.candidates))


@dataclass(frozen=True)
class ImportLocation:
    """Where a discovered file lives and which origin its servers have."""

    path: Path
    origin: str


def sanitize_import_name(name: str) -> str:
    """Turn another tool's server name into a valid Kolega id, keeping it recognizable."""
    cleaned = _UNUSABLE_NAME.sub("-", name.strip()).strip("-")
    if not cleaned:
        raise MCPEntryError("this server's name has no letters, numbers, '_' or '-' in it")
    return sanitize_mcp_server_id(cleaned)


def _environment(environ: Optional[Mapping[str, str]]) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def _home(environ: Mapping[str, str]) -> Path:
    home = environ.get("HOME") or environ.get("USERPROFILE")
    return Path(home).expanduser() if home else Path.home().expanduser()


def _config_dir(environ: Mapping[str, str], app_name: str) -> Path:
    if sys.platform == "win32":
        base = environ.get("APPDATA") or str(_home(environ) / "AppData" / "Roaming")
        return Path(base) / app_name
    if sys.platform == "darwin":
        return _home(environ) / "Library" / "Application Support" / app_name
    base = environ.get("XDG_CONFIG_HOME") or str(_home(environ) / ".config")
    return Path(base) / app_name


def _locations(source_id: str, project: Path, environ: Mapping[str, str]) -> list[ImportLocation]:
    home = _home(environ)
    by_id: dict[str, list[ImportLocation]] = {
        "claude-code": [ImportLocation(home / ".claude.json", "user")],
        "claude-desktop": [ImportLocation(_config_dir(environ, "Claude") / "claude_desktop_config.json", "user")],
        "cursor": [
            ImportLocation(home / ".cursor" / "mcp.json", "user"),
            ImportLocation(project / ".cursor" / "mcp.json", "project"),
        ],
        "vscode": [
            ImportLocation(project / ".vscode" / "mcp.json", "project"),
            ImportLocation(_config_dir(environ, "Code") / "User" / "mcp.json", "user"),
            ImportLocation(home / ".copilot" / "mcp-config.json", "user"),
        ],
        "windsurf": [ImportLocation(home / ".codeium" / "windsurf" / "mcp_config.json", "user")],
        "gemini": [
            ImportLocation(home / ".gemini" / "settings.json", "user"),
            ImportLocation(project / ".gemini" / "settings.json", "project"),
        ],
        "codex": [ImportLocation(home / ".codex" / "config.toml", "user")],
    }
    return by_id.get(source_id, [])


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _server_block(document: Any) -> dict[str, Any]:
    """The object keyed by server name, whichever wrapper the tool uses."""
    if not isinstance(document, dict):
        return {}
    for key in ("mcpServers", "servers"):
        if isinstance(document.get(key), dict):
            return document[key]
    return {}


def _candidate(source_id: str, origin: str, name: object, raw: Any) -> ImportCandidate:
    """Normalize one raw entry. An entry Kolega cannot represent is reported, not raised."""
    original = str(name)
    label = SOURCE_LABELS.get(source_id, source_id)

    def unsupported(reason: str) -> ImportCandidate:
        return ImportCandidate(source_id, label, origin, original, unsupported=reason)

    try:
        target = sanitize_import_name(original)
        entry = standard_entry_payload(target, raw)
        server = server_from_entry(target, raw, source="global")
    except UnsupportedMCPTypeError as exc:
        return unsupported(f"unsupported {exc.kind} type")
    except (MCPEntryError, ValueError) as exc:
        return unsupported(_reason(exc))
    return ImportCandidate(
        source_id,
        label,
        origin,
        original,
        server=server,
        dropped_fields=tuple(entry.dropped_fields),
        warnings=tuple(entry.warnings),
    )


def _reason(exc: Exception) -> str:
    """The entry error without its server-name prefix; the picker already shows the name."""
    return re.sub(r'^MCP server "[^"]*" ', "", " ".join(str(exc).split()))


def _read_json_source(location: ImportLocation, source_id: str) -> list[ImportCandidate]:
    block = _server_block(_read_json(location.path))
    if source_id in {"windsurf", "file"}:
        for name, raw in block.items():
            if isinstance(raw, dict) and "serverUrl" in raw and "url" not in raw and "httpUrl" not in raw:
                raw = raw.copy()
                raw["url"] = raw.pop("serverUrl")
                block[name] = raw
    return [_candidate(source_id, location.origin, name, raw) for name, raw in block.items()]


def _same_project(recorded: object, project: Path) -> bool:
    if not isinstance(recorded, str) or not recorded:
        return False
    try:
        return Path(recorded).expanduser().resolve() == project
    except OSError:
        return False


def _read_claude_code(location: ImportLocation, project: Path) -> list[ImportCandidate]:
    document = _read_json(location.path)
    if not isinstance(document, dict):
        return []
    candidates: list[ImportCandidate] = []
    user_block = document.get("mcpServers")
    if isinstance(user_block, dict):
        candidates.extend(_candidate("claude-code", "user", name, raw) for name, raw in user_block.items())
    projects = document.get("projects")
    if isinstance(projects, dict):
        # Claude Code keeps a project's private servers under that project's path.
        for recorded, state in projects.items():
            block = state.get("mcpServers") if isinstance(state, dict) else None
            if isinstance(block, dict) and _same_project(recorded, project):
                candidates.extend(_candidate("claude-code", "local", name, raw) for name, raw in block.items())
    return candidates


_CODEX_FIELDS = {
    "command",
    "args",
    "env",
    "cwd",
    "url",
    "enabled",
    "timeout",
    "env_vars",
    "startup_timeout_sec",
    "bearer_token_env_var",
}


def _codex_entry(table: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Rewrite one ``[mcp_servers.<name>]`` table in the portable shape."""
    raw: dict[str, Any] = {}
    dropped = sorted(key for key in table if key not in _CODEX_FIELDS)
    if table.get("command"):
        env = table.get("env", {})
        if not isinstance(env, dict):
            raise MCPEntryError("env must be a table")
        raw.update(type="stdio", command=table["command"], args=table.get("args"), env=env.copy())
        if table.get("cwd"):
            raw["cwd"] = table["cwd"]
    elif table.get("url"):
        raw.update(type="http", url=table["url"])
        bearer = table.get("bearer_token_env_var")
        if isinstance(bearer, str) and bearer:
            raw["headers"] = {"Authorization": "Bearer ${" + bearer + "}"}
    else:
        raise MCPEntryError("has neither a command nor a url")
    if table.get("enabled") is False:
        raw["disabled"] = True
    startup_timeout = table.get("startup_timeout_sec")
    if startup_timeout is not None:
        if (
            isinstance(startup_timeout, bool)
            or not isinstance(startup_timeout, (int, float))
            or not 0 < startup_timeout <= sys.float_info.max
        ):
            raise MCPEntryError("startup_timeout_sec must be a positive, finite number")
        raw["timeout_seconds"] = startup_timeout
    forwarded = table.get("env_vars")
    if isinstance(forwarded, list):
        # Codex forwards named shell variables; the same names become ${NAME} references.
        env = raw.setdefault("env", {})
        for variable in forwarded:
            if isinstance(variable, str) and variable:
                env[variable] = "${" + variable + "}"
    return raw, dropped


def _read_codex(location: ImportLocation, source_id: str) -> list[ImportCandidate]:
    import tomllib

    try:
        document = tomllib.loads(location.path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []

    def unsupported(name: str, reason: str) -> ImportCandidate:
        return ImportCandidate(source_id, SOURCE_LABELS[source_id], location.origin, name, unsupported=reason)

    block = document.get("mcp_servers", {})
    if not isinstance(block, dict):
        return [unsupported("mcp_servers", "must be a table")]
    candidates: list[ImportCandidate] = []
    for name, table in block.items():
        if not isinstance(table, dict):
            candidates.append(unsupported(name, "must be a table"))
            continue
        try:
            raw, dropped = _codex_entry(table)
        except MCPEntryError as exc:
            candidates.append(unsupported(name, str(exc)))
            continue
        candidate = _candidate(source_id, location.origin, name, raw)
        candidates.append(replace(candidate, dropped_fields=tuple(sorted({*candidate.dropped_fields, *dropped}))))
    return candidates


def _read_location(location: ImportLocation, source_id: str, project: Path) -> list[ImportCandidate]:
    if location.path.name == "config.toml":
        return _read_codex(location, source_id)
    if source_id == "claude-code":
        return _read_claude_code(location, project)
    return _read_json_source(location, source_id)


def discover_import_sources(project_path: Path, environ: Optional[Mapping[str, str]] = None) -> list[ImportSource]:
    """Sources on this machine that define at least one server, in a stable order.

    A tool whose config exists but holds no servers is not a source: there is nothing to
    offer, so the picker never lists it.
    """
    project = Path(project_path).expanduser().resolve()
    environment = _environment(environ)
    sources: list[ImportSource] = []
    for source_id in IMPORT_SOURCE_IDS:
        candidates: list[ImportCandidate] = []
        files: list[Path] = []
        for location in _locations(source_id, project, environment):
            if not location.path.is_file():
                continue
            found = _read_location(location, source_id, project)
            if not found:
                continue
            files.append(location.path)
            candidates.extend(found)
        if candidates:
            sources.append(ImportSource(source_id, SOURCE_LABELS[source_id], tuple(candidates), tuple(files)))
    return sources


def import_file(path: Path, project_path: Path) -> ImportSource:
    """Read one explicit file; its shape decides the parser."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise ValueError(f"Import file not found: {path}")
    location = ImportLocation(path, "user")
    project = Path(project_path).expanduser().resolve()
    if path.suffix.lower() == ".toml":
        found = _read_codex(location, "codex")
    elif path.name == ".claude.json":
        found = _read_claude_code(location, project)
    else:
        found = _read_json_source(location, "file")
    return ImportSource("file", path.name, tuple(found), (path,))


_PRECEDENCE = {"user": 1, "project": 2, "local": 3}


@dataclass(frozen=True)
class ImportOutcome:
    name: str
    outcome: str
    detail: str = ""


def import_servers(
    candidates: Sequence[ImportCandidate],
    *,
    scope: MCPScope,
    project_path: Path,
    state_dir: Path,
    settings: MCPScopeSettings,
    overwrite: bool = False,
) -> list[ImportOutcome]:
    """Write each importable candidate into ``scope``. Returns one outcome per candidate.

    An existing definition in the target scope is kept unless ``overwrite`` is set. The
    caller saves settings afterwards: a ``local``-scope import lives in settings.json.
    """
    outcomes: list[ImportOutcome] = []
    context = {"project_path": project_path, "state_dir": state_dir, "settings": settings}
    for candidate in candidates:
        if not candidate.importable:
            outcomes.append(ImportOutcome(candidate.original_name, "skipped", candidate.unsupported))
            continue
        defined = [definition.scope for definition in find_definitions(candidate.name, **context)]
        if scope in defined and not overwrite:
            outcomes.append(ImportOutcome(candidate.name, "skipped", f"already defined in {scope} scope"))
            continue
        assert candidate.server is not None
        outcome = upsert_server(candidate.server, scope=scope, **context)
        others = [other for other in defined if other != scope]
        detail = ""
        if others:
            if _PRECEDENCE[scope] > max(_PRECEDENCE[other] for other in others):
                detail = f"takes precedence over the {', '.join(others)} definition"
            else:
                detail = f"the {others[0]} definition takes precedence"
        outcomes.append(ImportOutcome(candidate.name, outcome, detail))
    return outcomes
