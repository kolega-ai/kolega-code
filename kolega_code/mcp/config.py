"""MCP server configuration loading and fingerprinting."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from kolega_code.local_state import write_private_text

MCP_CONFIG_SCHEMA_VERSION = 1
MCP_GLOBAL_CONFIG_FILENAME = "mcp_servers.json"
MCP_CONFIG_RELATIVE_PATH = Path(".kolega") / MCP_GLOBAL_CONFIG_FILENAME
MCP_TRANSPORTS = ("streamable_http", "sse", "stdio")
_SERVER_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class MCPConfigError(ValueError):
    """Raised when an MCP configuration file cannot be parsed for mutation."""


class MCPOAuthConfig(BaseModel):
    """OAuth settings for an HTTP MCP server."""

    enabled: bool = False
    redirect_uri: Optional[str] = None
    scope: Optional[str] = None
    client_name: str = "Kolega Code"
    client_uri: Optional[str] = None
    client_metadata_url: Optional[str] = None
    timeout_seconds: float = Field(default=300.0, gt=0)


class MCPServerConfig(BaseModel):
    """One configured MCP server."""

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
    source: str = Field(default="global", exclude=True)

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError("server id is required")
        if not _SERVER_ID_RE.match(value):
            raise ValueError("server id may contain only letters, numbers, '_' and '-'")
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

    def sanitized_for_display(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json", exclude={"source"})
        if payload.get("headers"):
            payload["headers"] = {key: "‹secret›" for key in payload["headers"]}
        if payload.get("env"):
            payload["env"] = {key: "‹secret›" for key in payload["env"]}
        return payload


class MCPConfigFile(BaseModel):
    schema_version: int = MCP_CONFIG_SCHEMA_VERSION
    servers: list[MCPServerConfig] = Field(default_factory=list)

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
    """Merged global + trusted project MCP configuration."""

    servers: dict[str, MCPServerConfig] = field(default_factory=dict)
    diagnostics: list[str] = field(default_factory=list)
    global_path: Optional[Path] = None
    project_path: Optional[Path] = None
    project_config_path: Optional[Path] = None
    project_trusted: bool = False
    project_config_present: bool = False

    @property
    def enabled_servers(self) -> list[MCPServerConfig]:
        return [server for server in self.servers.values() if server.enabled]


def global_mcp_config_path(state_dir: Path) -> Path:
    return Path(state_dir).expanduser() / MCP_GLOBAL_CONFIG_FILENAME


def project_mcp_config_path(project_path: Path) -> Path:
    return Path(project_path).expanduser().resolve() / MCP_CONFIG_RELATIVE_PATH


def parse_mcp_config_text(text: str, *, source: str) -> MCPConfigFile:
    data = json.loads(text)
    parsed = MCPConfigFile.model_validate(data)
    parsed.servers = [server.model_copy(update={"source": source}) for server in parsed.servers]
    return parsed


def load_mcp_config_file(path: Path, *, source: str) -> MCPConfigFile:
    return parse_mcp_config_text(path.read_text(encoding="utf-8"), source=source)


def load_mcp_config(project_path: Path, state_dir: Path, *, project_trusted: bool) -> LoadedMCPConfig:
    """Load global MCP config plus trusted project config.

    Invalid files are reported as diagnostics instead of crashing agent startup.
    The explicit `mcp` management commands use the stricter mutation helpers below
    when they need to fail loudly.
    """
    global_path = global_mcp_config_path(state_dir)
    project_config = project_mcp_config_path(project_path)
    diagnostics: list[str] = []
    merged: dict[str, MCPServerConfig] = {}

    for path, source, trusted in (
        (global_path, "global", True),
        (project_config, "project", project_trusted),
    ):
        if source == "project" and path.exists() and not trusted:
            diagnostics.append(
                f"Project MCP config exists at {path}, but this project is not trusted for MCP; project servers disabled."
            )
            continue
        if not trusted or not path.exists():
            continue
        try:
            config_file = load_mcp_config_file(path, source=source)
        except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
            diagnostics.append(f"Could not load {source} MCP config {path}: {exc}")
            continue
        for server in config_file.servers:
            if server.id in merged:
                diagnostics.append(f"MCP server '{server.id}' from {source} overrides earlier configuration.")
            merged[server.id] = server

    return LoadedMCPConfig(
        servers=merged,
        diagnostics=diagnostics,
        global_path=global_path,
        project_path=Path(project_path).expanduser().resolve(),
        project_config_path=project_config,
        project_trusted=project_trusted,
        project_config_present=project_config.exists(),
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
    write_private_text(path, payload + "\n")


def upsert_server_config(path: Path, server: MCPServerConfig, *, source: str) -> None:
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


def remove_server_config(path: Path, server_id: str, *, source: str) -> bool:
    config = load_config_file_for_edit(path, source=source)
    before = len(config.servers)
    config.servers = [server for server in config.servers if server.id != server_id]
    if len(config.servers) == before:
        return False
    save_config_file(path, config)
    return True


def set_server_enabled(path: Path, server_id: str, enabled: bool, *, source: str) -> bool:
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
    return values
