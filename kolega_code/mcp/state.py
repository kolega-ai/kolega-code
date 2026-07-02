"""Local MCP verification/status state."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from kolega_code.local_state import write_private_secret_text, write_private_text

MCP_STATUS_SCHEMA_VERSION = 1
MCP_STATUS_FILENAME = "mcp_server_status.json"
MCP_OAUTH_TOKENS_FILENAME = "mcp_oauth_tokens.json"


class MCPToolStatus(BaseModel):
    id: str
    name: Optional[str] = None
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class MCPServerStatus(BaseModel):
    status: Literal["unverified", "verified", "failed"] = "unverified"
    fingerprint: Optional[str] = None
    verified_at: Optional[str] = None
    message: str = ""
    transport: Optional[str] = None
    source: Optional[str] = None
    tool_count: int = 0
    tools: list[MCPToolStatus] = Field(default_factory=list)
    oauth: bool = False

    @classmethod
    def verified(
        cls,
        *,
        fingerprint: str,
        transport: str,
        source: str,
        tools: list[MCPToolStatus],
        message: str = "",
        oauth: bool = False,
    ) -> "MCPServerStatus":
        return cls(
            status="verified",
            fingerprint=fingerprint,
            verified_at=_now_iso(),
            message=message or f"Verified {len(tools)} tool(s).",
            transport=transport,
            source=source,
            tool_count=len(tools),
            tools=tools,
            oauth=oauth,
        )

    @classmethod
    def failed(
        cls,
        *,
        fingerprint: str,
        transport: str,
        source: str,
        message: str,
        oauth: bool = False,
    ) -> "MCPServerStatus":
        return cls(
            status="failed",
            fingerprint=fingerprint,
            verified_at=_now_iso(),
            message=message,
            transport=transport,
            source=source,
            tool_count=0,
            tools=[],
            oauth=oauth,
        )


class MCPStatusFile(BaseModel):
    schema_version: int = MCP_STATUS_SCHEMA_VERSION
    servers: dict[str, MCPServerStatus] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_schema_version(self) -> "MCPStatusFile":
        if self.schema_version != MCP_STATUS_SCHEMA_VERSION:
            raise ValueError(f"Unsupported MCP status schema version: {self.schema_version}")
        return self


class MCPOAuthServerTokens(BaseModel):
    tokens: Optional[dict[str, Any]] = None
    client_info: Optional[dict[str, Any]] = None
    updated_at: str = Field(default_factory=lambda: _now_iso())


class MCPOAuthTokenFile(BaseModel):
    schema_version: int = MCP_STATUS_SCHEMA_VERSION
    servers: dict[str, MCPOAuthServerTokens] = Field(default_factory=dict)

    @field_validator("servers", mode="before")
    @classmethod
    def _coerce_servers(cls, value: object) -> dict[str, object]:
        if not isinstance(value, dict):
            return {}
        return {str(k): v for k, v in value.items() if isinstance(v, dict)}

    @model_validator(mode="after")
    def _validate_schema_version(self) -> "MCPOAuthTokenFile":
        if self.schema_version != MCP_STATUS_SCHEMA_VERSION:
            raise ValueError(f"Unsupported MCP OAuth token schema version: {self.schema_version}")
        return self


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MCPStatusStore:
    """Filesystem-backed verification/status store."""

    def __init__(self, state_dir: Path) -> None:
        self.root = Path(state_dir).expanduser()
        self.path = self.root / MCP_STATUS_FILENAME

    def load(self) -> MCPStatusFile:
        if not self.path.exists():
            return MCPStatusFile()
        try:
            return MCPStatusFile.model_validate_json(self.path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, ValueError, json.JSONDecodeError):
            return MCPStatusFile()

    def save(self, data: MCPStatusFile) -> None:
        payload = json.dumps(data.model_dump(mode="json"), indent=2, sort_keys=True)
        write_private_text(self.path, payload + "\n")

    def get(self, server_id: str) -> Optional[MCPServerStatus]:
        return self.load().servers.get(server_id)

    def update(self, server_id: str, status: MCPServerStatus) -> None:
        data = self.load()
        data.servers[server_id] = status
        self.save(data)

    def clear(self, server_id: str) -> None:
        data = self.load()
        if server_id in data.servers:
            data.servers.pop(server_id, None)
            self.save(data)


class MCPOAuthTokenStore:
    """Filesystem-backed OAuth token/client-registration store."""

    def __init__(self, state_dir: Path) -> None:
        self.root = Path(state_dir).expanduser()
        self.path = self.root / MCP_OAUTH_TOKENS_FILENAME

    def load(self) -> MCPOAuthTokenFile:
        if not self.path.exists():
            return MCPOAuthTokenFile()
        try:
            return MCPOAuthTokenFile.model_validate_json(self.path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, ValueError, json.JSONDecodeError):
            return MCPOAuthTokenFile()

    def save(self, data: MCPOAuthTokenFile) -> None:
        payload = json.dumps(data.model_dump(mode="json"), indent=2, sort_keys=True)
        write_private_secret_text(self.path, payload + "\n")

    def get(self, server_id: str) -> MCPOAuthServerTokens:
        return self.load().servers.get(server_id, MCPOAuthServerTokens())

    def set_tokens(self, server_id: str, tokens: Optional[dict[str, Any]]) -> None:
        data = self.load()
        record = data.servers.get(server_id, MCPOAuthServerTokens())
        record.tokens = dict(tokens) if tokens else None
        record.updated_at = _now_iso()
        data.servers[server_id] = record
        self.save(data)

    def set_client_info(self, server_id: str, client_info: Optional[dict[str, Any]]) -> None:
        data = self.load()
        record = data.servers.get(server_id, MCPOAuthServerTokens())
        record.client_info = dict(client_info) if client_info else None
        record.updated_at = _now_iso()
        data.servers[server_id] = record
        self.save(data)

    def clear(self, server_id: str) -> None:
        data = self.load()
        if server_id in data.servers:
            data.servers.pop(server_id, None)
            self.save(data)

    def secret_values(self) -> list[str]:
        values: list[str] = []
        for record in self.load().servers.values():
            tokens = record.tokens or {}
            for key in ("access_token", "refresh_token", "id_token"):
                value = tokens.get(key)
                if value:
                    values.append(str(value))
            client_secret = (record.client_info or {}).get("client_secret")
            if client_secret:
                values.append(str(client_secret))
        return values
