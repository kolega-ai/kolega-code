"""One status vocabulary for MCP servers, shared by the CLI, doctor, Settings and the /mcp panel."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, Optional

from .config import LoadedMCPConfig, MCPServerConfig, server_fingerprint
from .entries import missing_env_references
from .failure_messages import (
    MCP_FAILURE_MESSAGE_OAUTH_SIGNIN_INCOMPLETE,
    MCP_FAILURE_MESSAGE_OAUTH_SIGNIN_REQUIRED,
    MCP_FAILURE_MESSAGE_OAUTH_UNAUTHORIZED,
    MCP_FAILURE_MESSAGE_UNAUTHORIZED,
    safe_failed_status_message,
)
from .state import MCPServerStatus

MCPHealth = Literal["pending_trust", "disabled", "connected", "needs_auth", "stale", "failed", "unverified"]

NEEDS_AUTH_MESSAGES = frozenset(
    {
        MCP_FAILURE_MESSAGE_UNAUTHORIZED,
        MCP_FAILURE_MESSAGE_OAUTH_UNAUTHORIZED,
        MCP_FAILURE_MESSAGE_OAUTH_SIGNIN_INCOMPLETE,
        MCP_FAILURE_MESSAGE_OAUTH_SIGNIN_REQUIRED,
    }
)

HEALTH_GLYPHS: dict[str, str] = {
    "pending_trust": "⏸",
    "disabled": "⊘",
    "connected": "●",
    "needs_auth": "!",
    "stale": "!",
    "failed": "✗",
    "unverified": "○",
}
HEALTH_LABELS: dict[str, str] = {
    "pending_trust": "Pending approval",
    "disabled": "Disabled",
    "connected": "Connected",
    "needs_auth": "Needs auth",
    "stale": "Config changed",
    "failed": "Failed",
    "unverified": "Not connected",
}


@dataclass(frozen=True)
class MCPServerHealth:
    health: MCPHealth
    server_id: str = ""
    message: str = ""
    tool_count: int = 0
    verified_at: Optional[str] = None
    missing_env: list[str] = field(default_factory=list)
    # Set when a private per-project choice, not the definition, decided the state.
    enabled_override: Optional[bool] = None

    @property
    def glyph(self) -> str:
        return HEALTH_GLYPHS[self.health]

    @property
    def label(self) -> str:
        return HEALTH_LABELS[self.health]

    @property
    def hint(self) -> str:
        """A short next step or detail; empty when the label says everything."""
        if self.health == "pending_trust":
            return "project not trusted"
        if self.health == "disabled":
            return "disabled for this project" if self.enabled_override is False else "disabled"
        if self.health == "connected":
            parts = (_tool_count_label(self.tool_count), _verified_label(self.verified_at))
            return " · ".join(part for part in parts if part)
        if self.health == "needs_auth":
            return f"sign in: kolega-code mcp login {self.server_id}"
        if self.health == "stale":
            return "config changed since the last verification; verify again"
        if self.health in {"failed", "unverified"}:
            return self.message
        return ""


def _tool_count_label(count: int) -> str:
    return f"{count} tool" if count == 1 else f"{count} tools"


def health_hint(health: MCPServerHealth, missing_env: Sequence[str] = ()) -> str:
    """The health's own hint, followed by any environment variable the server still needs."""
    parts = [health.hint]
    if missing_env:
        parts.append("missing env: " + ", ".join(missing_env))
    return " · ".join(part for part in parts if part)


def relative_time(iso_timestamp: Optional[str]) -> str:
    """``"2h ago"``-style age of an ISO timestamp, or ``""`` when it cannot be read."""
    if not iso_timestamp:
        return ""
    try:
        moment = datetime.fromisoformat(iso_timestamp)
        now = datetime.now(timezone.utc)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        age = max(0, int((now - moment).total_seconds()))
    except ValueError:
        return ""
    if age < 60:
        return "just now"
    for unit_seconds, unit in ((31536000, "y"), (2592000, "mo"), (86400, "d"), (3600, "h"), (60, "m")):
        if age >= unit_seconds:
            return f"{age // unit_seconds}{unit} ago"
    return "just now"


def _verified_label(verified_at: Optional[str]) -> str:
    age = relative_time(verified_at)
    return f"verified {age}" if age else ""


def server_health(
    server: MCPServerConfig,
    status: Optional[MCPServerStatus],
    *,
    trusted: bool = True,
    enabled_override: Optional[bool] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> MCPServerHealth:
    """Map a stored verification status to the state a person sees. First match wins."""
    missing = missing_env_references(server, environ) if server.enabled and trusted else []
    common: dict[str, Any] = {"server_id": server.id, "missing_env": missing, "enabled_override": enabled_override}
    if server.source == "project" and not trusted:
        return MCPServerHealth("pending_trust", **common)
    if not server.enabled:
        return MCPServerHealth("disabled", **common)
    if status is None:
        # An OAuth server with no stored sign-in is waiting for the user, not for a connection.
        return MCPServerHealth("needs_auth" if server.oauth.enabled else "unverified", **common)
    fingerprint = server_fingerprint(server)
    if status.fingerprint and status.fingerprint != fingerprint:
        return MCPServerHealth("stale", **common)
    if status.status == "verified" and status.fingerprint == fingerprint:
        return MCPServerHealth("connected", tool_count=status.tool_count, verified_at=status.verified_at, **common)
    if status.status == "failed":
        # A stored message is shown again only when it is one of the fixed safe strings.
        message = safe_failed_status_message(status.message)
        needs_auth = server.oauth.enabled and status.message in NEEDS_AUTH_MESSAGES
        health: MCPHealth = "needs_auth" if needs_auth else "failed"
        return MCPServerHealth(health, message=message, verified_at=status.verified_at, **common)
    return MCPServerHealth("needs_auth" if server.oauth.enabled else "unverified", **common)


def health_fields(
    server: MCPServerConfig,
    status: Optional[MCPServerStatus],
    *,
    config: LoadedMCPConfig,
    trusted: bool,
    environ: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """The scope and health columns of a status row."""
    override = config.project_overrides.get(server.id) if trusted and server.source == "project" else None
    health = server_health(server, status, trusted=trusted, enabled_override=override, environ=environ)
    files = config.server_files if trusted else config.untrusted_project_server_files
    path = files.get(server.id)
    return {
        "scope": server.scope,
        "file": str(path) if path is not None else None,
        "health": health,
        "enabled_override": override,
        "missing_env": list(health.missing_env),
    }
