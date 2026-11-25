"""Translation between the portable ``mcpServers`` entry shape and Kolega server configs.

Standard-library only: ``config.py`` imports this module, so it must not import back.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from .config import MCPServerConfig

STANDARD_CONFIG_FILENAME = ".mcp.json"
STANDARD_SERVERS_KEY = "mcpServers"
KOLEGA_EXTENSION_KEY = "x-kolega"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_SSE_READ_TIMEOUT_SECONDS = 300.0
# Below one second a standard ``timeout`` is noise, not a deliberate override.
_MIN_STANDARD_TIMEOUT_MS = 1000

TRANSPORT_ALIASES = {
    "stdio": "stdio",
    "http": "streamable_http",
    "streamable-http": "streamable_http",
    "streamable_http": "streamable_http",
    "sse": "sse",
}
_STANDARD_TYPES = {"stdio": "stdio", "streamable_http": "http", "sse": "sse"}
_UNSUPPORTED_TYPES = {"ws": "WebSocket", "wss": "WebSocket", "websocket": "WebSocket", "sdk": "SDK-hosted"}
_KNOWN_ENTRY_FIELDS = frozenset(
    {
        "id",
        "type",
        "transport",
        "name",
        "enabled",
        "disabled",
        "url",
        "httpUrl",
        "headers",
        "command",
        "args",
        "env",
        "cwd",
        "timeout",
        "timeout_seconds",
        "sse_read_timeout_seconds",
        "oauth",
        KOLEGA_EXTENSION_KEY,
    }
)
_OAUTH_FIELDS = (
    "enabled",
    "scope",
    "client_id",
    "client_secret",
    "client_secret_env",
    "redirect_uri",
    "token_endpoint_auth_method",
    "client_name",
    "client_uri",
    "client_metadata_url",
    "timeout_seconds",
)
# camelCase spellings other clients write.
_CLAUDE_OAUTH_FIELDS = ("clientId", "callbackPort", "scopes")


class MCPEntryError(ValueError):
    """Raised when a portable MCP server entry cannot be interpreted."""


class UnsupportedMCPTypeError(MCPEntryError):
    """Raised for an entry Kolega cannot connect to, such as WebSocket or SDK-hosted."""

    def __init__(self, server_id: str, kind: str) -> None:
        self.kind = kind
        super().__init__(f'MCP server "{server_id}" uses the unsupported "{kind}" type; it was skipped.')


@dataclass(frozen=True)
class StandardEntry:
    """A Kolega server payload plus what normalization had to leave out."""

    payload: dict[str, Any]
    dropped_fields: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def server_id_label(server_id: object) -> str:
    return str(server_id)[:64]


def _scalar_mapping(raw: object, server_id: str, label: str, warnings: list[str]) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise MCPEntryError(f'MCP server "{server_id}" has "{label}" that is not an object')
    result: dict[str, str] = {}
    for key, value in raw.items():
        if value is None or isinstance(value, (dict, list)):
            warnings.append(f'{server_id}: dropped "{label}.{key}" because it is not a scalar value')
            continue
        result[str(key)] = str(value)
    return result


def _string_list(raw: object, server_id: str, label: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise MCPEntryError(f'MCP server "{server_id}" has "{label}" that is not an array')
    return [str(item) for item in raw if item is not None]


def _oauth_payload(raw: object, server_id: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise MCPEntryError(f'MCP server "{server_id}" has "oauth" that is not an object')
    payload: dict[str, Any] = {key: raw[key] for key in _OAUTH_FIELDS if key in raw}
    client_id = raw.get("clientId")
    if client_id:
        payload.setdefault("client_id", str(client_id))
    port = raw.get("callbackPort")
    if port is not None:
        if isinstance(port, bool) or not isinstance(port, int):
            raise MCPEntryError(f'MCP server "{server_id}" has an "oauth.callbackPort" that is not a port number')
        payload.setdefault("redirect_uri", f"http://127.0.0.1:{port}/callback")
    scopes = raw.get("scopes")
    if scopes is not None:
        payload.setdefault(
            "scope", " ".join(str(scope) for scope in scopes) if isinstance(scopes, list) else str(scopes)
        )
    return payload


def _declared_transport(raw: Mapping[str, Any], server_id: str) -> str:
    declared = raw.get("type") if raw.get("type") is not None else raw.get("transport")
    if declared is None:
        raise MCPEntryError(f'MCP server "{server_id}" must declare a type, a url or a command')
    key = str(declared).strip().lower().replace("_", "-")
    if key in _UNSUPPORTED_TYPES:
        raise UnsupportedMCPTypeError(server_id, _UNSUPPORTED_TYPES[key])
    if key not in TRANSPORT_ALIASES:
        raise MCPEntryError(f'MCP server "{server_id}" has an unknown type "{key[:40]}"')
    return TRANSPORT_ALIASES[key]


def _url_transport(url: object, server_id: str) -> str:
    scheme = url.strip().split("://", 1)[0].lower() if isinstance(url, str) and "://" in url else ""
    if scheme in {"http", "https"}:
        return "streamable_http"
    if scheme == "sse":
        return "sse"
    if scheme in _UNSUPPORTED_TYPES:
        raise UnsupportedMCPTypeError(server_id, _UNSUPPORTED_TYPES[scheme])
    raise MCPEntryError(f'MCP server "{server_id}" has a url with an unrecognized scheme; add a "type"')


def _entry_transport(raw: Mapping[str, Any]) -> tuple[str, object]:
    """Return ``(transport, url)``; an entry without a type is inferred from its url or command."""
    name = _entry_name(raw)
    url = raw.get("url")
    http_url = raw.get("httpUrl")
    if url is None and http_url is not None:
        # Gemini CLI spells its SSE endpoint ``httpUrl``.
        transport = (
            "sse" if raw.get("type") is None and raw.get("transport") is None else _declared_transport(raw, name)
        )
        return transport, http_url
    if raw.get("type") is None and raw.get("transport") is None:
        if url is not None:
            return _url_transport(url, name), url
        if raw.get("command"):
            return "stdio", None
        raise MCPEntryError(f'MCP server "{name}" must declare a type, a url or a command')
    return _declared_transport(raw, name), url


def _entry_name(raw: Mapping[str, Any]) -> str:
    return server_id_label(raw.get("id", ""))


def standard_entry_payload(server_id: str, raw: object) -> StandardEntry:
    """Build the Kolega server payload for one portable ``mcpServers`` entry.

    Raises :class:`MCPEntryError` for an entry that cannot be interpreted. Messages name
    the server and the field, never the field's value.
    """
    label = server_id_label(server_id)
    if not isinstance(raw, dict):
        raise MCPEntryError(f'MCP server "{label}" must be an object')
    merged: dict[str, Any] = {"id": server_id, **raw}
    extension = raw.get(KOLEGA_EXTENSION_KEY)
    if isinstance(extension, dict):
        merged = {**extension, **merged}
    transport, url = _entry_transport(merged)
    warnings: list[str] = []

    payload: dict[str, Any] = {"id": server_id, "transport": transport}
    name = merged.get("name")
    if isinstance(name, str) and name.strip():
        payload["name"] = name.strip()
    payload["enabled"] = not (merged.get("disabled") is True or merged.get("enabled") is False)

    if transport == "stdio":
        payload["command"] = merged.get("command") or ""
        payload["args"] = _string_list(merged.get("args"), label, "args")
        payload["env"] = _scalar_mapping(merged.get("env"), label, "env", warnings)
        cwd = merged.get("cwd")
        if isinstance(cwd, str) and cwd.strip():
            payload["cwd"] = cwd
    else:
        payload["url"] = str(url) if url is not None else None
        payload["headers"] = _scalar_mapping(merged.get("headers"), label, "headers", warnings)

    if merged.get("timeout_seconds") is not None:
        payload["timeout_seconds"] = merged["timeout_seconds"]
    else:
        timeout = merged.get("timeout")
        if isinstance(timeout, (int, float)) and not isinstance(timeout, bool) and timeout >= _MIN_STANDARD_TIMEOUT_MS:
            payload["timeout_seconds"] = timeout / 1000
    if merged.get("sse_read_timeout_seconds") is not None:
        payload["sse_read_timeout_seconds"] = merged["sse_read_timeout_seconds"]
    oauth = _oauth_payload(merged.get("oauth"), label)
    if oauth:
        payload["oauth"] = oauth

    dropped = sorted(key for key in raw if key not in _KNOWN_ENTRY_FIELDS)
    raw_oauth = raw.get("oauth")
    if isinstance(raw_oauth, dict):
        unknown_oauth = sorted(set(raw_oauth) - set(_OAUTH_FIELDS) - set(_CLAUDE_OAUTH_FIELDS))
        dropped.extend(f"oauth.{key}" for key in unknown_oauth)
    return StandardEntry(payload=payload, dropped_fields=dropped, warnings=warnings)


def server_to_standard_json(server: "MCPServerConfig", existing: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Serialize a server as a portable ``mcpServers`` entry.

    Keys in ``existing`` that Kolega does not write are preserved, so rewriting one server
    never strips fields another client owns.
    """
    entry: dict[str, Any] = {"type": _STANDARD_TYPES[server.transport]}
    if server.transport == "stdio":
        entry["command"] = server.command or ""
        if server.args:
            entry["args"] = list(server.args)
        if server.env:
            entry["env"] = dict(server.env)
    else:
        entry["url"] = server.url or ""
        if server.headers:
            entry["headers"] = dict(server.headers)
    if server.timeout_seconds != DEFAULT_TIMEOUT_SECONDS:
        entry["timeout"] = int(round(server.timeout_seconds * 1000))
    for key, value in (existing or {}).items():
        if key not in entry and key not in _KNOWN_ENTRY_FIELDS:
            entry[key] = value

    extension: dict[str, Any] = {}
    if server.name:
        extension["name"] = server.name
    if not server.enabled:
        extension["enabled"] = False
    if server.cwd:
        extension["cwd"] = server.cwd
    if server.sse_read_timeout_seconds != DEFAULT_SSE_READ_TIMEOUT_SECONDS:
        extension["sse_read_timeout_seconds"] = server.sse_read_timeout_seconds
    oauth = server.oauth
    configured = (
        oauth.enabled,
        oauth.client_id,
        oauth.client_secret,
        oauth.client_secret_env,
        oauth.scope,
        oauth.redirect_uri,
        oauth.token_endpoint_auth_method,
        oauth.client_uri,
        oauth.client_metadata_url,
    )
    if any(configured):
        extension["oauth"] = oauth.model_dump(mode="json", exclude_none=True)
    if extension:
        entry[KOLEGA_EXTENSION_KEY] = extension
    return entry


_ENV_REFERENCE_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class MCPEnvReferenceError(RuntimeError):
    """Raised when a server references an environment variable that is unset."""

    def __init__(self, names: list[str]) -> None:
        self.names = list(names)
        super().__init__("MCP server references unset environment variable(s): " + ", ".join(self.names))


def env_references(text: str) -> list[str]:
    """Variable names referenced by ``${NAME}`` or ``${NAME:-default}``, in order of appearance."""
    seen: list[str] = []
    for match in _ENV_REFERENCE_RE.finditer(text):
        name = match.group(1)
        if name not in seen:
            seen.append(name)
    return seen


def _env_values(server: "MCPServerConfig") -> list[str]:
    if server.transport == "stdio":
        return [server.command or "", *server.args, *server.env.values()]
    return [server.url or "", *server.headers.values()]


def _expand_text(text: str, environ: Mapping[str, str], missing: list[str]) -> str:
    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        # An empty value counts as set: only a missing variable falls through.
        if name in environ:
            return environ[name]
        if default is not None:
            return default
        if name not in missing:
            missing.append(name)
        return match.group(0)

    return _ENV_REFERENCE_RE.sub(replace, text)


def missing_env_references(server: "MCPServerConfig", environ: Mapping[str, str] | None = None) -> list[str]:
    """Variables the server needs that are unset and have no ``:-default``."""
    source = os.environ if environ is None else environ
    missing: list[str] = []
    for value in _env_values(server):
        _expand_text(value, source, missing)
    return missing


def expand_env_references(server: "MCPServerConfig", environ: Mapping[str, str] | None = None) -> "MCPServerConfig":
    """Return a copy of ``server`` with ``${NAME}`` references resolved from ``environ``.

    The stored configuration is never modified, so resolved secrets stay out of
    fingerprints, status files and displays. Raises :class:`MCPEnvReferenceError` when a
    referenced variable is unset and has no default.
    """
    source = os.environ if environ is None else environ
    if not any(_ENV_REFERENCE_RE.search(value) for value in _env_values(server)):
        return server
    missing: list[str] = []
    updates: dict[str, Any] = {}
    if server.transport == "stdio":
        updates["command"] = _expand_text(server.command or "", source, missing)
        updates["args"] = [_expand_text(arg, source, missing) for arg in server.args]
        updates["env"] = {key: _expand_text(value, source, missing) for key, value in server.env.items()}
    else:
        updates["url"] = _expand_text(server.url or "", source, missing)
        updates["headers"] = {key: _expand_text(value, source, missing) for key, value in server.headers.items()}
    if missing:
        raise MCPEnvReferenceError(missing)
    return server.model_copy(update=updates)
