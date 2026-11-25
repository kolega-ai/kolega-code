"""MCP operations shared by the Settings page and the ``/mcp`` panel.

No widget logic lives here. Each function checks the app's preconditions, performs the
change through the scope layer, and rebuilds the agent when its tool set may change.
"""

from __future__ import annotations

import asyncio
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

from kolega_code.cli.settings import SettingsStoreError
from kolega_code.mcp.config import LoadedMCPConfig, MCPConfigError, MCPScope, MCPServerConfig, load_mcp_config_for
from kolega_code.mcp.health import server_health
from kolega_code.mcp.scopes import remove_server, set_server_enabled_for_scope, upsert_server
from kolega_code.mcp.service import MCPService, MCPVerificationResult, exposed_tool_names_for
from kolega_code.mcp.state import MCPStatusStore, MCPOAuthTokenStore

BLOCKED_MESSAGE = "Stop the active turn before changing MCP servers."
BUSY_MESSAGE = "Wait for the current MCP operation to finish."
INCOMPLETE_MESSAGE = "Verification could not complete. Try again."
OPERATION_ERRORS = (MCPConfigError, SettingsStoreError, OSError, UnicodeError)
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def plain_text(value: str) -> str:
    return _CONTROL_CHARS_RE.sub("", value)


def operation_error(exc: Exception) -> str:
    if isinstance(exc, MCPConfigError):
        return "MCP configuration could not be updated. Fix the config file and try again."
    return "Could not access MCP configuration or state. Check file permissions and try again."


def change_blocked(app: Any) -> bool:
    """Whether a turn is running; MCP changes wait until it ends."""
    return bool(app._turn_active or app.agent_worker is not None or app._agent_execution_draining())


def load_config(app: Any, path: Path | None = None) -> LoadedMCPConfig:
    return load_mcp_config_for(
        path if path is not None else app.active_project_path,
        app.settings_store.root,
        app.settings,
        trusted=project_trusted(app),
    )


def project_trusted(app: Any) -> bool:
    # Linked workspaces inherit the launch project's trust decision, not their own.
    return app.settings.is_mcp_project_trusted(app.project_path)


def _service(app: Any, config: Optional[LoadedMCPConfig] = None) -> MCPService:
    return MCPService(
        config if config is not None else load_config(app),
        state_dir=app.settings_store.root,
        project_path=app.active_project_path,
    )


def status_rows(app: Any, config: Optional[LoadedMCPConfig] = None) -> list[dict[str, Any]]:
    return _service(app, config).list_status_rows()


def exposed_tools(
    app: Any, server: MCPServerConfig, config: Optional[LoadedMCPConfig] = None
) -> list[tuple[str, str, str]]:
    """``(wire name, server-side tool id, description)`` for a connected server's tools."""
    service = _service(app, config)
    status = service.server_status(server)
    if status is None or not service.is_verified(server):
        return []
    names = [final for _, final in exposed_tool_names_for(server.id, status.tools)]
    return list(
        zip(names, (tool.id for tool in status.tools), (tool.description for tool in status.tools), strict=True)
    )


def reconnect_targets(app: Any) -> list[str]:
    """Enabled servers that are not connected: what ``/mcp reconnect all`` retries."""
    service = _service(app)
    return [
        server.id
        for server in service.config.enabled_servers
        if server_health(server, service.server_status(server)).health != "connected"
    ]


async def rebuild_agent(app: Any) -> None:
    await app._ensure_agent_from_settings(rebuild=True)


async def verify(app: Any, server_id: str) -> Optional[MCPVerificationResult]:
    """Verify one saved server, then rebuild the agent if the session stayed idle.

    Returns ``None`` when a turn was already running. Transport exceptions are never
    echoed: they can carry credentials.
    """
    if change_blocked(app):
        return None
    try:
        result = await _service(app).verify_server(
            server_id, interactive_oauth=True, open_browser=True, output=app.console
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        result = MCPVerificationResult(server_id, False, INCOMPLETE_MESSAGE)
    if not change_blocked(app):
        # Once replacement starts, let it settle before a cancellation releases the MCP gate.
        rebuild = asyncio.create_task(rebuild_agent(app), name="mcp-agent-rebuild")
        try:
            await asyncio.shield(rebuild)
        except asyncio.CancelledError:
            while not rebuild.done():
                try:
                    await asyncio.gather(asyncio.shield(rebuild), return_exceptions=True)
                except asyncio.CancelledError:
                    continue
            if not rebuild.cancelled():
                rebuild.exception()
            raise
    return result


async def set_enabled(app: Any, server_id: str, scope: MCPScope, enabled: bool) -> bool:
    """Enable or disable a server in one scope. Returns False when blocked or undefined."""
    if change_blocked(app):
        return False
    draft = deepcopy(app.settings) if scope in {"project", "local"} else app.settings
    changed, _ = set_server_enabled_for_scope(
        server_id,
        enabled,
        scope=scope,
        project_path=app.active_project_path,
        state_dir=app.settings_store.root,
        settings=draft,
    )
    if not changed:
        return False
    if scope in {"project", "local"}:
        app.settings_store.save(draft)
        app.settings = draft
    await rebuild_agent(app)
    return True


async def trust_project(app: Any) -> bool:
    """Trust this project's MCP servers so they load. Returns False when blocked."""
    if change_blocked(app):
        return False
    draft = deepcopy(app.settings)
    draft.trust_mcp_project(app.project_path)
    app.settings_store.save(draft)
    app.settings = draft
    await rebuild_agent(app)
    return True


def save_server(app: Any, server: MCPServerConfig, *, scope: MCPScope, replace_id: str | None = None) -> str:
    """Add or replace a definition in one scope. Returns ``"added"`` or ``"updated"``.

    Callers rebuild the agent once their own edit sequence is complete.
    """
    draft = deepcopy(app.settings) if scope == "local" else app.settings
    if replace_id is not None and replace_id != server.id:
        if scope == "local":
            remove_server(
                replace_id,
                scope=scope,
                project_path=app.active_project_path,
                state_dir=app.settings_store.root,
                settings=draft,
            )
        else:
            remove_definition(app, replace_id, scope=scope)
    outcome = upsert_server(
        server,
        scope=scope,
        project_path=app.active_project_path,
        state_dir=app.settings_store.root,
        settings=draft,
    )
    if scope == "local":
        app.settings_store.save(draft)
        app.settings = draft
        if replace_id is not None and replace_id != server.id:
            _clear_server_state(app, replace_id)
    return outcome


def remove_definition(app: Any, server_id: str, *, scope: MCPScope) -> bool:
    """Remove a definition and its saved verification and sign-in. False when undefined there."""
    draft = deepcopy(app.settings) if scope in {"project", "local"} else app.settings
    removed = remove_server(
        server_id,
        scope=scope,
        project_path=app.active_project_path,
        state_dir=app.settings_store.root,
        settings=draft,
    )
    if scope in {"project", "local"}:
        app.settings_store.save(draft)
        app.settings = draft
    if removed:
        _clear_server_state(app, server_id)
    return removed


async def clear_authentication(app: Any, server_id: str) -> None:
    """Forget a server's sign-in and cached tools; it reads as signed out until the next login."""
    _clear_server_state(app, server_id)
    if not change_blocked(app):
        await rebuild_agent(app)


def _clear_server_state(app: Any, server_id: str) -> None:
    MCPOAuthTokenStore(app.settings_store.root).clear(server_id)
    MCPStatusStore(app.settings_store.root).clear(server_id)
