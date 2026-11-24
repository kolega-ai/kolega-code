"""MCP server configuration loading and fingerprinting."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Literal, Mapping, Optional, Protocol
from urllib.parse import urlparse

from filelock import FileLock
from filelock import Timeout as FileLockTimeout
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from kolega_code.local_state import write_private_secret_text

from .entries import (
    STANDARD_CONFIG_FILENAME,
    STANDARD_SERVERS_KEY,
    MCPEntryError,
    server_to_standard_json,
    standard_entry_payload,
)

MCP_CONFIG_SCHEMA_VERSION = 1
MCP_GLOBAL_CONFIG_FILENAME = "mcp_servers.json"
MCP_CONFIG_RELATIVE_PATH = Path(".kolega") / MCP_GLOBAL_CONFIG_FILENAME
MCP_CONFIG_LOCK_TIMEOUT_SECONDS = 10.0
MCPScope = Literal["local", "project", "user"]
_SERVER_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
# Tool wire names are capped at 64 chars total; a 32-char server id leaves at
# least 25 chars of budget for the tool id after the ``mcp__{server}__`` prefix.
_SERVER_ID_MAX_LENGTH = 32
_SERVER_ID_PREFIX_LENGTH = 24
_ENV_VAR_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# Labels used in diagnostics for where a definition lives.
_STANDARD_CONFIG_LABEL = f"the project {STANDARD_CONFIG_FILENAME} file"
_LEGACY_CONFIG_LABEL = f"the project {MCP_CONFIG_RELATIVE_PATH} file"
_SOURCE_LABELS = {"global": "the user config", "local": "the local scope"}


class MCPScopeSettings(Protocol):
    """The settings fields that scope MCP servers to a project."""

    trusted_mcp_projects: list[str]
    mcp_local_servers: dict[str, dict[str, dict[str, Any]]]
    mcp_project_server_overrides: dict[str, dict[str, bool]]


def sanitize_mcp_server_id(value: str) -> str:
    """Validate a server id, rewriting over-long ids so they fit provider budgets.

    Ids longer than ``_SERVER_ID_MAX_LENGTH`` are rewritten deterministically as
    ``{first 24 chars}_{7-char hash of the original}`` so existing config files
    keep loading (instead of failing) and distinct long ids stay distinct. Ids
    within the limit pass through unchanged; invalid characters still raise.
    """
    value = str(value).strip()
    if not value:
        raise ValueError("server id is required")
    if not _SERVER_ID_RE.match(value):
        raise ValueError("server id may contain only letters, numbers, '_' and '-'")
    if len(value) <= _SERVER_ID_MAX_LENGTH:
        return value
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:7]
    return f"{value[:_SERVER_ID_PREFIX_LENGTH]}_{digest}"


def settings_project_key(project_path: Path) -> str:
    """The key a resolved project path uses in settings-scoped MCP state."""
    return str(Path(project_path).expanduser().resolve())


class MCPConfigError(ValueError):
    """Raised when an MCP configuration file cannot be parsed for mutation."""


class MCPOAuthConfig(BaseModel):
    """OAuth settings for an HTTP MCP server."""

    model_config = ConfigDict(hide_input_in_errors=True)

    enabled: bool = False
    redirect_uri: Optional[str] = None
    scope: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    client_secret_env: Optional[str] = None
    token_endpoint_auth_method: Optional[Literal["client_secret_post", "client_secret_basic", "none"]] = None
    client_name: str = "Kolega Code"
    client_uri: Optional[str] = None
    client_metadata_url: Optional[str] = None
    timeout_seconds: float = Field(default=300.0, gt=0)

    @field_validator("redirect_uri")
    @classmethod
    def _validate_redirect_uri(cls, value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        parsed = urlparse(value)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise ValueError(
                "MCP OAuth redirect_uri must be a localhost http URL (e.g. http://127.0.0.1:33418/callback)"
            )
        # Accessing port also rejects malformed and out-of-range values.
        _ = parsed.port
        if parsed.username is not None or parsed.password is not None or parsed.fragment:
            raise ValueError("MCP OAuth redirect_uri must not contain credentials or a fragment")
        return value

    @field_validator("client_secret_env")
    @classmethod
    def _validate_client_secret_env(cls, value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        stripped = value.strip()
        if not _ENV_VAR_RE.match(stripped):
            raise ValueError("client_secret_env must be a valid environment variable name")
        return stripped

    @model_validator(mode="after")
    def _validate_oauth_fields(self) -> "MCPOAuthConfig":
        if (self.client_secret or self.client_secret_env) and not self.client_id:
            raise ValueError("OAuth client_secret and client_secret_env require a client_id")
        if self.token_endpoint_auth_method in {"client_secret_post", "client_secret_basic"}:
            if not self.client_id or not (self.client_secret or self.client_secret_env):
                raise ValueError("OAuth client secret authentication requires a client_id and client secret")
        if "enabled" not in self.model_fields_set and self.client_id:
            self.enabled = True
        return self

    def resolve_client_secret(self, env: Optional[Mapping[str, str]] = None) -> Optional[str]:
        if self.client_secret:
            return self.client_secret
        if self.client_secret_env:
            source_env = os.environ if env is None else env
            return source_env.get(self.client_secret_env) or None
        return None

    def resolve_auth_method(self, env: Optional[Mapping[str, str]] = None) -> str:
        if self.token_endpoint_auth_method:
            return self.token_endpoint_auth_method
        if self.resolve_client_secret(env=env):
            return "client_secret_post"
        return "none"


class MCPServerConfig(BaseModel):
    """One configured MCP server."""

    model_config = ConfigDict(hide_input_in_errors=True)

    id: str
    name: Optional[str] = None
    transport: Literal["streamable_http", "sse", "stdio"] = "streamable_http"
    enabled: bool = True

    # HTTP transports (streamable_http and sse)
    url: Optional[str] = None
    headers: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: float = Field(default=30.0, gt=0)
    sse_read_timeout_seconds: float = Field(default=300.0, gt=0)
    oauth: MCPOAuthConfig = Field(default_factory=MCPOAuthConfig)

    # stdio transport
    command: Optional[str] = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: Optional[str] = None

    # Loader metadata. Kept out of fingerprints and serialized config files.
    # ``global`` is the user scope, ``project`` is a shared project file, ``local`` is a
    # private per-project definition, and ``acp`` comes from the editor.
    source: str = Field(default="global", exclude=True)

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return sanitize_mcp_server_id(value)

    @field_validator("transport", mode="before")
    @classmethod
    def _transport_alias(cls, value: object) -> object:
        # The portable ``mcpServers`` shape spells streamable HTTP ``http`` or ``streamable-http``.
        if isinstance(value, str):
            normalized = value.strip().lower().replace("_", "-")
            return {"http": "streamable_http", "streamable-http": "streamable_http"}.get(normalized, value)
        return value

    @field_validator("headers", "env", mode="before")
    @classmethod
    def _string_mapping(cls, value: object) -> dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("must be an object")
        return {str(k): str(v) for k, v in value.items() if v is not None}

    @field_validator("args", mode="before")
    @classmethod
    def _string_list(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("args must be an array")
        return [str(item) for item in value]

    @model_validator(mode="after")
    def _validate_transport_fields(self) -> "MCPServerConfig":
        if self.transport in {"streamable_http", "sse"}:
            if not self.url:
                raise ValueError(f"{self.transport} server '{self.id}' requires url")
            if self.oauth.enabled and self.transport not in {"streamable_http", "sse"}:
                raise ValueError("OAuth is only supported for HTTP MCP transports")
        elif self.transport == "stdio":
            if not self.command:
                raise ValueError(f"stdio server '{self.id}' requires command")
            if self.oauth.enabled:
                raise ValueError("OAuth is not supported for stdio MCP servers")
        return self

    @property
    def display_name(self) -> str:
        return self.name or self.id

    @property
    def scope(self) -> Optional[MCPScope]:
        """Where the definition is stored, or ``None`` when the editor supplied it."""
        if self.source == "project":
            return "project"
        if self.source == "local":
            return "local"
        if self.source == "global":
            return "user"
        return None

    def sanitized_for_display(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json", exclude={"source"})
        if payload.get("headers"):
            payload["headers"] = {key: "‹secret›" for key in payload["headers"]}
        if payload.get("env"):
            payload["env"] = {key: "‹secret›" for key in payload["env"]}
        if payload.get("oauth") and isinstance(payload["oauth"], dict):
            if payload["oauth"].get("client_secret"):
                payload["oauth"]["client_secret"] = "‹secret›"
        return payload


class MCPConfigFile(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    schema_version: int = MCP_CONFIG_SCHEMA_VERSION
    servers: list[MCPServerConfig] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _accept_portable_wrapper(cls, value: object) -> object:
        if isinstance(value, dict) and "servers" not in value:
            portable = value.get(STANDARD_SERVERS_KEY)
            if isinstance(portable, (dict, list)):
                return {**value, "servers": portable}
        return value

    @field_validator("servers", mode="before")
    @classmethod
    def _coerce_servers(cls, value: object) -> list[object]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            servers: list[object] = []
            for server_id, entry in value.items():
                if isinstance(entry, dict):
                    merged = dict(entry)
                    merged.setdefault("id", str(server_id))
                    servers.append(merged)
            return servers
        raise ValueError("servers must be an array or object")

    @model_validator(mode="after")
    def _validate_schema_version(self) -> "MCPConfigFile":
        if self.schema_version != MCP_CONFIG_SCHEMA_VERSION:
            raise ValueError(f"Unsupported MCP config schema version: {self.schema_version}")
        seen: set[str] = set()
        for server in self.servers:
            if server.id in seen:
                raise ValueError(f"Duplicate MCP server id: {server.id}")
            seen.add(server.id)
        return self

    def to_file_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MCP_CONFIG_SCHEMA_VERSION,
            "servers": [
                server.model_dump(mode="json", exclude={"source"}, exclude_none=True) for server in self.servers
            ],
        }


@dataclass(frozen=True)
class LoadedMCPConfig:
    """Merged user, local, and trusted project MCP configuration."""

    servers: dict[str, MCPServerConfig] = field(default_factory=dict)
    diagnostics: list[str] = field(default_factory=list)
    global_path: Optional[Path] = None
    project_path: Optional[Path] = None
    project_config_path: Optional[Path] = None
    project_trusted: bool = False
    project_config_present: bool = False
    standard_project_path: Optional[Path] = None
    # Definitions in an untrusted project file. Listed so a person can review them
    # before trusting; they are never loaded, verified or exposed.
    untrusted_project_servers: list[MCPServerConfig] = field(default_factory=list)
    # Project-scope servers whose effective enabled state comes from a private choice.
    project_overrides: dict[str, bool] = field(default_factory=dict)
    # The file that defines each loaded server, by id. Absent for local-scope servers
    # (they live in settings) and for servers the editor supplied.
    server_files: dict[str, Path] = field(default_factory=dict)

    @property
    def enabled_servers(self) -> list[MCPServerConfig]:
        return [server for server in self.servers.values() if server.enabled]


def global_mcp_config_path(state_dir: Path) -> Path:
    return Path(state_dir).expanduser() / MCP_GLOBAL_CONFIG_FILENAME


def project_mcp_config_path(project_path: Path) -> Path:
    return Path(project_path).expanduser().resolve() / MCP_CONFIG_RELATIVE_PATH


def standard_mcp_config_path(project_path: Path) -> Path:
    return Path(project_path).expanduser().resolve() / STANDARD_CONFIG_FILENAME


def _raw_entries(data: Any) -> list[tuple[str, dict[str, Any]]]:
    """``(id, entry)`` pairs from ``servers`` or the portable ``mcpServers`` key, in file order."""
    if not isinstance(data, dict):
        raise MCPConfigError("MCP config must be a JSON object")
    servers = data["servers"] if "servers" in data else data.get(STANDARD_SERVERS_KEY)
    if servers is None:
        return []
    entries: list[tuple[str, dict[str, Any]]] = []
    if isinstance(servers, list):
        for item in servers:
            if not isinstance(item, dict) or item.get("id") is None:
                raise MCPConfigError("servers must be an array of objects with an id")
            entries.append((str(item["id"]), item))
    elif isinstance(servers, dict):
        for server_id, item in servers.items():
            if not isinstance(item, dict):
                raise MCPConfigError(f'MCP server "{server_id}" must be an object')
            entries.append((str(server_id), item))
    else:
        raise MCPConfigError("servers must be an array or object")
    return entries


def _server_from_raw(server_id: str, raw: object, *, source: str) -> MCPServerConfig:
    entry = standard_entry_payload(server_id, raw)
    try:
        server = MCPServerConfig.model_validate(entry.payload)
    except ValidationError as exc:
        errors = exc.errors()
        message = str(errors[0].get("msg", "invalid configuration")) if errors else "invalid configuration"
        raise MCPEntryError(f'MCP server "{server_id}" is invalid: {message.removeprefix("Value error, ")}') from exc
    return server.model_copy(update={"source": source})


def _parse_config_entries(
    data: Any,
    *,
    label: str,
    source: str,
    tolerant: bool,
) -> tuple[list[MCPServerConfig], list[str]]:
    """Parse every entry in a config document.

    Tolerant parsing reports a bad entry as a diagnostic and keeps the rest of the
    file; strict parsing raises, so a mutation can never rewrite a file it could not
    fully understand.
    """
    servers: list[MCPServerConfig] = []
    diagnostics: list[str] = []
    try:
        raw_entries = _raw_entries(data)
        version = data.get("schema_version", MCP_CONFIG_SCHEMA_VERSION)
        if version != MCP_CONFIG_SCHEMA_VERSION:
            raise ValueError(f"Unsupported MCP config schema version: {version}")
    except (MCPConfigError, ValueError) as exc:
        if not tolerant:
            raise MCPConfigError(str(exc)) from exc
        return [], [f"Could not load {source} MCP config {label}: {exc}"]
    seen: set[str] = set()
    for raw_id, raw in raw_entries:
        try:
            server = _server_from_raw(raw_id, raw, source=source)
            if server.id in seen:
                raise MCPEntryError(f"Duplicate MCP server id: {server.id}")
        except MCPEntryError as exc:
            if not tolerant:
                raise MCPConfigError(str(exc)) from exc
            diagnostics.append(f"Skipped an entry in {source} MCP config {label}: {exc}")
            continue
        seen.add(server.id)
        if raw_id != server.id:
            diagnostics.append(
                f"MCP server id '{raw_id}' exceeds {_SERVER_ID_MAX_LENGTH} characters; using '{server.id}' instead."
            )
        servers.append(server)
    return servers, diagnostics


def parse_mcp_config_text(text: str, *, source: str, label: str = "<config>") -> MCPConfigFile:
    data = json.loads(text)
    servers, _ = _parse_config_entries(data, label=label, source=source, tolerant=False)
    return MCPConfigFile(servers=servers)


def load_mcp_config_file(path: Path, *, source: str) -> MCPConfigFile:
    path = Path(path)
    return parse_mcp_config_text(path.read_text(encoding="utf-8"), source=source, label=str(path))


def _read_config_file(path: Path, *, source: str) -> tuple[list[MCPServerConfig], list[str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        # A file that was never created is the normal state, not a load error.
        return [], []
    except OSError as exc:
        return [], [f"Could not load {source} MCP config {path}: {exc}"]
    if not text.strip():
        return [], []
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        return [], [f"Could not load {source} MCP config {path}: {exc}"]
    return _parse_config_entries(data, label=str(path), source=source, tolerant=True)


def _entries_from_mapping(
    entries: Mapping[str, Any], *, source: str, label: str
) -> tuple[list[MCPServerConfig], list[str]]:
    return _parse_config_entries({STANDARD_SERVERS_KEY: entries}, label=label, source=source, tolerant=True)


def servers_in_file(path: Path, *, source: str) -> list[MCPServerConfig]:
    """Servers a config file defines, read tolerantly; empty when it is absent or unreadable."""
    servers, _ = _read_config_file(path, source=source)
    return servers


def server_from_entry(server_id: str, raw: object, *, source: str) -> MCPServerConfig:
    """Validate one raw server entry. Raises ``MCPConfigError`` when it cannot be interpreted."""
    try:
        return _server_from_raw(server_id, raw, source=source)
    except MCPEntryError as exc:
        raise MCPConfigError(str(exc)) from exc


def server_ids_in_file(path: Path, *, source: str) -> set[str]:
    """Ids a config file defines, read tolerantly; empty when it is absent or unreadable."""
    return {server.id for server in servers_in_file(path, source=source)}


def load_mcp_config(
    project_path: Path,
    state_dir: Path,
    *,
    project_trusted: bool,
    local_servers: Optional[Mapping[str, Any]] = None,
    project_overrides: Optional[Mapping[str, bool]] = None,
) -> LoadedMCPConfig:
    """Load user, local and trusted project MCP configuration, highest precedence last.

    Invalid files are reported as diagnostics instead of crashing agent startup. The
    explicit ``mcp`` management commands use the stricter mutation helpers below
    when they need to fail loudly.
    """
    global_path = global_mcp_config_path(state_dir)
    resolved_project = Path(project_path).expanduser().resolve()
    legacy_path = project_mcp_config_path(resolved_project)
    standard_path = standard_mcp_config_path(resolved_project)
    project_files = ((standard_path, _STANDARD_CONFIG_LABEL), (legacy_path, _LEGACY_CONFIG_LABEL))
    diagnostics: list[str] = []
    merged: dict[str, MCPServerConfig] = {}
    origin: dict[str, str] = {}
    server_files: dict[str, Path] = {}
    untrusted: list[MCPServerConfig] = []

    def absorb(servers: list[MCPServerConfig], label: str, path: Optional[Path] = None) -> None:
        for server in servers:
            previous = origin.get(server.id)
            if previous is not None:
                diagnostics.append(f"MCP server '{server.id}' from {label} overrides the definition in {previous}.")
            merged[server.id] = server
            origin[server.id] = label
            if path is None:
                server_files.pop(server.id, None)
            else:
                server_files[server.id] = path

    servers, notes = _read_config_file(global_path, source="global")
    diagnostics.extend(notes)
    absorb(servers, _SOURCE_LABELS["global"], global_path)

    present = [path for path, _ in project_files if path.exists()]
    if present and not project_trusted:
        diagnostics.append(
            "Project MCP config exists at "
            + " and ".join(str(path) for path in present)
            + ", but this project is not trusted for MCP; project servers disabled."
        )
    for path, label in project_files:
        if not path.exists():
            continue
        servers, notes = _read_config_file(path, source="project")
        diagnostics.extend(notes)
        if project_trusted:
            absorb(servers, label, path)
        else:
            untrusted.extend(servers)
            for server in servers:
                server_files[server.id] = path

    if local_servers:
        servers, notes = _entries_from_mapping(local_servers, source="local", label="the local scope")
        diagnostics.extend(notes)
        absorb(servers, _SOURCE_LABELS["local"])

    applied_overrides: dict[str, bool] = {}
    for server_id, enabled in (project_overrides or {}).items():
        server = merged.get(server_id)
        if server is not None and server.source == "project":
            merged[server_id] = server.model_copy(update={"enabled": bool(enabled)})
            applied_overrides[server_id] = bool(enabled)

    return LoadedMCPConfig(
        servers=merged,
        diagnostics=diagnostics,
        global_path=global_path,
        project_path=resolved_project,
        project_config_path=legacy_path,
        project_trusted=project_trusted,
        project_config_present=bool(present),
        standard_project_path=standard_path,
        untrusted_project_servers=untrusted,
        project_overrides=applied_overrides,
        server_files=server_files,
    )


def load_mcp_config_for(
    project_path: Path,
    state_dir: Path,
    settings: MCPScopeSettings,
    *,
    trusted: Optional[bool] = None,
) -> LoadedMCPConfig:
    """Load MCP configuration with the trust, local servers and overrides saved in settings.

    ``trusted`` overrides the saved trust decision for ``project_path``; a host whose
    session is anchored to a launch project passes that project's trust.
    """
    key = settings_project_key(project_path)
    return load_mcp_config(
        project_path,
        state_dir,
        project_trusted=(key in getattr(settings, "trusted_mcp_projects", [])) if trusted is None else trusted,
        local_servers=getattr(settings, "mcp_local_servers", {}).get(key),
        project_overrides=getattr(settings, "mcp_project_server_overrides", {}).get(key),
    )


def load_config_file_for_edit(path: Path, *, source: str) -> MCPConfigFile:
    """Load a config file for mutation, returning an empty config if absent."""
    if not path.exists():
        return MCPConfigFile()
    try:
        return load_mcp_config_file(path, source=source)
    except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise MCPConfigError(f"Could not load MCP config {path}: {exc}") from exc


def save_config_file(path: Path, config: MCPConfigFile) -> None:
    payload = json.dumps(config.to_file_dict(), indent=2, sort_keys=True)
    write_private_secret_text(path, payload + "\n")


def config_lock_dir(state_dir: Path) -> Path:
    """Where mutation locks for config files outside the state directory are kept.

    A lock file left beside a shared project file would show up as an untracked file in
    every clone that ran a management command, so project files lock through the
    private state directory instead.
    """
    return Path(state_dir).expanduser() / "locks"


def _lock_path(path: Path, lock_dir: Optional[Path]) -> Path:
    if lock_dir is None:
        return Path(f"{path}.lock")
    digest = hashlib.sha256(str(Path(path).expanduser().resolve()).encode("utf-8")).hexdigest()[:32]
    return Path(lock_dir).expanduser() / f"mcp-config-{digest}.lock"


@contextlib.contextmanager
def _config_lock(path: Path, lock_dir: Optional[Path] = None) -> Generator[None, None, None]:
    """Hold a cross-process lock for one read-modify-write of a config file."""
    lock_file = _lock_path(path, lock_dir)
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(lock_file))
    try:
        lock.acquire(timeout=MCP_CONFIG_LOCK_TIMEOUT_SECONDS)
    except FileLockTimeout as exc:
        raise MCPConfigError(f"Timed out locking the MCP config: {path}") from exc
    try:
        yield
    finally:
        lock.release()


def upsert_server_config(path: Path, server: MCPServerConfig, *, source: str, lock_dir: Optional[Path] = None) -> bool:
    """Add or replace one server. Returns True when the server was added."""
    with _config_lock(path, lock_dir):
        config = load_config_file_for_edit(path, source=source)
        updated: list[MCPServerConfig] = []
        replaced = False
        for existing in config.servers:
            if existing.id == server.id:
                updated.append(server.model_copy(update={"source": source}))
                replaced = True
            else:
                updated.append(existing)
        if not replaced:
            updated.append(server.model_copy(update={"source": source}))
        config.servers = updated
        save_config_file(path, config)
    return not replaced


def remove_server_config(path: Path, server_id: str, *, source: str, lock_dir: Optional[Path] = None) -> bool:
    with _config_lock(path, lock_dir):
        config = load_config_file_for_edit(path, source=source)
        before = len(config.servers)
        config.servers = [server for server in config.servers if server.id != server_id]
        if len(config.servers) == before:
            return False
        save_config_file(path, config)
    return True


def set_server_enabled(
    path: Path, server_id: str, enabled: bool, *, source: str, lock_dir: Optional[Path] = None
) -> bool:
    with _config_lock(path, lock_dir):
        config = load_config_file_for_edit(path, source=source)
        updated: list[MCPServerConfig] = []
        changed = False
        for server in config.servers:
            if server.id == server_id:
                updated.append(server.model_copy(update={"enabled": enabled, "source": source}))
                changed = True
            else:
                updated.append(server)
        if not changed:
            return False
        config.servers = updated
        save_config_file(path, config)
    return True


def _standard_document(path: Path) -> dict[str, Any]:
    """The raw shared ``.mcp.json`` document, or ``{}`` when the file is absent."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise MCPConfigError(f"Could not load project MCP config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise MCPConfigError(f"Could not load project MCP config {path}: MCP config must be a JSON object")
    return data


def _standard_entries(data: dict[str, Any]) -> dict[str, Any]:
    key = "servers" if "servers" in data and STANDARD_SERVERS_KEY not in data else STANDARD_SERVERS_KEY
    if key not in data:
        data[key] = {}
    entries: Any = data[key]
    if not isinstance(entries, dict):
        raise MCPConfigError(f'"{key}" must be an object keyed by server name')
    return entries


def _write_standard_document(path: Path, data: dict[str, Any]) -> None:
    write_private_secret_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def upsert_standard_server(
    path: Path, server: MCPServerConfig, *, source: str, lock_dir: Optional[Path] = None
) -> bool:
    """Add or replace one entry in a shared ``.mcp.json`` without touching anything else.

    Runs on the raw document so fields Kolega does not know, and every other entry,
    survive the rewrite. Returns True when the server was added.
    """
    with _config_lock(path, lock_dir):
        data = _standard_document(path)
        entries = _standard_entries(data)
        existing = entries.get(server.id)
        existing_entry = existing if isinstance(existing, dict) else None
        entry = server_to_standard_json(server.model_copy(update={"source": source}), existing=existing_entry)
        if existing_entry != entry:
            entries[server.id] = entry
            _write_standard_document(path, data)
    return existing_entry is None


def remove_standard_server(path: Path, server_id: str, *, lock_dir: Optional[Path] = None) -> bool:
    with _config_lock(path, lock_dir):
        data = _standard_document(path)
        entries = _standard_entries(data)
        keys = [key for key in entries if key == server_id or _sanitized_id(key) == server_id]
        if not keys:
            return False
        for key in keys:
            del entries[key]
        _write_standard_document(path, data)
    return True


def _sanitized_id(value: object) -> Optional[str]:
    try:
        return sanitize_mcp_server_id(str(value))
    except ValueError:
        return None


def server_fingerprint(server: MCPServerConfig) -> str:
    """Stable fingerprint used to invalidate verification after config changes."""
    payload = server.model_dump(mode="json", exclude={"source", "enabled"}, exclude_none=True)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def mcp_secret_values(config: LoadedMCPConfig) -> list[str]:
    """Exact configured MCP secret values for diagnostics redaction."""
    values: list[str] = []
    for server in config.servers.values():
        values.extend(value for value in server.headers.values() if value)
        values.extend(value for value in server.env.values() if value)
        secret = server.oauth.resolve_client_secret()
        if secret:
            values.append(secret)
    return values
