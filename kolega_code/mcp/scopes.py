"""Scope-level MCP server operations shared by the CLI and the TUI.

A definition lives in exactly one scope. ``user`` servers are in the state directory,
``local`` servers are private to one project and kept in settings, and ``project``
servers are in a file the repository shares.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .config import (
    MCPScope,
    MCPScopeSettings,
    MCPServerConfig,
    config_lock_dir,
    global_mcp_config_path,
    project_mcp_config_path,
    remove_server_config,
    remove_standard_server,
    server_from_entry,
    server_ids_in_file,
    servers_in_file,
    settings_project_key,
    standard_mcp_config_path,
    set_server_enabled,
    upsert_server_config,
    upsert_standard_server,
)
from .entries import env_references

SCOPE_LABELS: dict[str, str] = {"user": "user", "project": "project", "local": "local"}


@dataclass(frozen=True)
class ServerDefinition:
    """One place a server id is defined."""

    scope: MCPScope
    file: Optional[Path]


def project_write_target(project_path: Path) -> Path:
    """The project file ``--scope project`` writes to.

    A project that already uses the legacy ``.kolega/mcp_servers.json`` and has no
    ``.mcp.json`` keeps using it; every other project gets the portable file.
    """
    legacy = project_mcp_config_path(project_path)
    standard = standard_mcp_config_path(project_path)
    return legacy if legacy.exists() and not standard.exists() else standard


def scope_file(scope: MCPScope, project_path: Path, state_dir: Path) -> Optional[Path]:
    if scope == "user":
        return global_mcp_config_path(state_dir)
    if scope == "project":
        return project_write_target(project_path)
    return None


def scope_location(scope: MCPScope, project_path: Path, state_dir: Path) -> str:
    """Human-readable description of where a scope is stored."""
    if scope == "local":
        return "settings.json, this project only"
    path = scope_file(scope, project_path, state_dir)
    return str(path) if path is not None else scope


def local_server_entries(settings: MCPScopeSettings, project_path: Path) -> dict[str, dict[str, Any]]:
    return getattr(settings, "mcp_local_servers", {}).get(settings_project_key(project_path), {})


def find_definitions(
    server_id: str, *, project_path: Path, state_dir: Path, settings: MCPScopeSettings
) -> list[ServerDefinition]:
    """Every scope that defines ``server_id``, highest precedence first."""
    definitions: list[ServerDefinition] = []
    if server_id in local_server_entries(settings, project_path):
        definitions.append(ServerDefinition("local", None))
    legacy = project_mcp_config_path(project_path)
    if server_id in server_ids_in_file(legacy, source="project"):
        definitions.append(ServerDefinition("project", legacy))
    standard = standard_mcp_config_path(project_path)
    if server_id in server_ids_in_file(standard, source="project"):
        definitions.append(ServerDefinition("project", standard))
    user = global_mcp_config_path(state_dir)
    if server_id in server_ids_in_file(user, source="global"):
        definitions.append(ServerDefinition("user", user))
    return definitions


def has_literal_secrets(server: MCPServerConfig) -> bool:
    """Whether a header or env value is a literal rather than a ``${NAME}`` reference."""
    values = [*server.headers.values(), *server.env.values()]
    return any(value and not env_references(value) for value in values)


def load_definition(
    server_id: str,
    scope: MCPScope,
    *,
    project_path: Path,
    state_dir: Path,
    settings: MCPScopeSettings,
) -> Optional[MCPServerConfig]:
    """The server as one scope defines it, or ``None`` when that scope does not define it.

    This is the raw definition, not the effective server: a higher-precedence scope
    may shadow it.
    """
    if scope == "local":
        raw = local_server_entries(settings, project_path).get(server_id)
        return server_from_entry(server_id, raw, source="local") if raw is not None else None
    if scope == "user":
        candidates = ((global_mcp_config_path(state_dir), "global"),)
    else:
        candidates = (
            (project_mcp_config_path(project_path), "project"),
            (standard_mcp_config_path(project_path), "project"),
        )
    for path, source in candidates:
        for server in servers_in_file(path, source=source):
            if server.id == server_id:
                return server
    return None


def upsert_local_server(settings: MCPScopeSettings, project_path: Path, server: MCPServerConfig) -> bool:
    """Store a server in the ``local`` scope. Returns True when it was added."""
    project = settings.mcp_local_servers.setdefault(settings_project_key(project_path), {})
    added = server.id not in project
    project[server.id] = server.model_dump(mode="json", exclude={"source"}, exclude_none=True)
    return added


def remove_local_server(settings: MCPScopeSettings, project_path: Path, server_id: str) -> bool:
    project = local_server_entries(settings, project_path)
    if server_id not in project:
        return False
    del project[server_id]
    _prune_empty_local_project(settings, project_path)
    return True


def set_local_server_enabled(settings: MCPScopeSettings, project_path: Path, server_id: str, enabled: bool) -> bool:
    entry = local_server_entries(settings, project_path).get(server_id)
    if entry is None:
        return False
    entry["enabled"] = enabled
    return True


def _prune_empty_local_project(settings: MCPScopeSettings, project_path: Path) -> None:
    key = settings_project_key(project_path)
    local_servers = getattr(settings, "mcp_local_servers", {})
    if key in local_servers and not local_servers[key]:
        del local_servers[key]


def clear_project_server_override(settings: MCPScopeSettings, project_path: Path, server_id: str) -> bool:
    key = settings_project_key(project_path)
    project_choices = getattr(settings, "mcp_project_server_overrides", {})
    overrides = project_choices.get(key)
    if not overrides or server_id not in overrides:
        return False
    del overrides[server_id]
    if not overrides:
        del project_choices[key]
    return True


def untrust_project(settings: MCPScopeSettings, project_path: Path) -> bool:
    key = settings_project_key(project_path)
    if key not in settings.trusted_mcp_projects:
        return False
    settings.trusted_mcp_projects.remove(key)
    return True


def reset_project_choices(settings: MCPScopeSettings, project_path: Path) -> tuple[int, bool]:
    """Forget a project's enable/disable choices and its trust decision.

    Returns ``(choices forgotten, trust was removed)``.
    """
    key = settings_project_key(project_path)
    overrides = getattr(settings, "mcp_project_server_overrides", {}).pop(key, None)
    trusted = untrust_project(settings, project_path)
    return (len(overrides) if overrides else 0), trusted


def project_server_override(settings: MCPScopeSettings, project_path: Path, server_id: str) -> Optional[bool]:
    """The private enable/disable choice recorded for a project server, if any."""
    project_choices = getattr(settings, "mcp_project_server_overrides", {})
    return project_choices.get(settings_project_key(project_path), {}).get(server_id)


def set_project_server_override(settings: MCPScopeSettings, project_path: Path, server_id: str, enabled: bool) -> None:
    """Record a private enable/disable choice for a shared project server."""
    overrides = settings.mcp_project_server_overrides.setdefault(settings_project_key(project_path), {})
    overrides[server_id] = bool(enabled)


def upsert_server(
    server: MCPServerConfig, *, scope: MCPScope, project_path: Path, state_dir: Path, settings: MCPScopeSettings
) -> str:
    """Add or replace a server in one scope. Returns ``"added"`` or ``"updated"``."""
    if scope == "user":
        added = upsert_server_config(global_mcp_config_path(state_dir), server, source="global")
    elif scope == "local":
        added = upsert_local_server(settings, project_path, server)
    else:
        target = project_write_target(project_path)
        lock_dir = config_lock_dir(state_dir)
        if target == project_mcp_config_path(project_path):
            added = upsert_server_config(target, server, source="project", lock_dir=lock_dir)
        else:
            added = upsert_standard_server(target, server, source="project", lock_dir=lock_dir)
    return "added" if added else "updated"


def remove_server(
    server_id: str, *, scope: MCPScope, project_path: Path, state_dir: Path, settings: MCPScopeSettings
) -> bool:
    """Remove a server's definition from one scope. Returns False when it was not defined there."""
    if scope == "user":
        return remove_server_config(global_mcp_config_path(state_dir), server_id, source="global")
    if scope == "local":
        return remove_local_server(settings, project_path, server_id)
    lock_dir = config_lock_dir(state_dir)
    legacy = project_mcp_config_path(project_path)
    standard = standard_mcp_config_path(project_path)
    removed = False
    if legacy.exists():
        removed = remove_server_config(legacy, server_id, source="project", lock_dir=lock_dir) or removed
    if standard.exists():
        removed = remove_standard_server(standard, server_id, lock_dir=lock_dir) or removed
    clear_project_server_override(settings, project_path, server_id)
    return removed


def set_server_enabled_for_scope(
    server_id: str, enabled: bool, *, scope: MCPScope, project_path: Path, state_dir: Path, settings: MCPScopeSettings
) -> tuple[bool, bool]:
    """Enable or disable a server in one scope.

    A shared project definition is never edited for this; the choice is recorded
    privately for the project. Returns ``(changed, recorded as a project choice)``.
    """
    if scope == "user":
        return set_server_enabled(global_mcp_config_path(state_dir), server_id, enabled, source="global"), False
    if scope == "local":
        return set_local_server_enabled(settings, project_path, server_id, enabled), False
    set_project_server_override(settings, project_path, server_id, enabled)
    return True, True
