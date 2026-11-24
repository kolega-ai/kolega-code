"""The ``kolega-code mcp`` commands: manage MCP servers and their connection state."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, cast

from kolega_code.cli.settings import CliSettings, SettingsStore
from kolega_code.mcp.config import (
    LoadedMCPConfig,
    MCPScope,
    MCPServerConfig,
    load_mcp_config_for,
    sanitize_mcp_server_id,
    server_fingerprint,
)
from kolega_code.mcp.entries import TRANSPORT_ALIASES, standard_entry_payload
from kolega_code.mcp.health import MCPServerHealth, health_fields
from kolega_code.mcp.scopes import (
    ServerDefinition,
    find_definitions,
    has_literal_secrets,
    load_definition,
    remove_server,
    reset_project_choices,
    scope_location,
    set_server_enabled_for_scope,
    untrust_project,
    upsert_server,
)
from kolega_code.mcp.service import (
    MCPService,
    MCPVerificationResult,
    exposed_tool_names_for,
    mcp_tool_name_adjustment_note,
)
from kolega_code.mcp.state import MCPStatusStore, MCPOAuthTokenStore

SCOPES = ("user", "project", "local")
SCOPE_SUMMARY = "Scopes: user = all your projects · local = this project, private · project = .mcp.json (shared)"
TRANSPORT_LABELS = {"streamable_http": "http", "sse": "sse", "stdio": "stdio"}
MASKED_VALUE = "‹hidden›"
_MAX_TOOL_NAMES_SHOWN = 40
# Server names, URLs and commands come from config files a repository may control, so
# control characters are stripped before any of them reaches the terminal.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_SECRET_PROMPT = object()
IMPORT_SOURCE_IDS = ("claude-code", "claude-desktop", "cursor", "vscode", "windsurf", "gemini", "codex")


def add_mcp_parser(subparsers: Any) -> argparse.ArgumentParser:
    mcp = subparsers.add_parser("mcp", help="Manage MCP servers and their connection state.")
    mcp.add_argument("--project", default=".", type=Path, help="Project directory whose MCP scopes apply.")
    mcp.add_argument("--state-dir", type=Path, help="Directory for CLI state and the user MCP config.")
    mcp.add_argument(
        "--trust-mcp",
        action="store_true",
        help="Trust this project's MCP servers (.mcp.json or .kolega/mcp_servers.json) before running the command.",
    )
    commands = mcp.add_subparsers(dest="mcp_command", required=True)

    def with_scope(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "-s",
            "--scope",
            choices=SCOPES,
            metavar="SCOPE",
            help="Where the server is defined: user, project (shared) or local (this project, private).",
        )
        parser.add_argument("--project-config", action="store_true", help=argparse.SUPPRESS)

    listing = commands.add_parser("list", help="List MCP servers with their connection status.")
    with_scope(listing)
    listing.add_argument("--check", action="store_true", help="Connect to each enabled server to refresh its status.")
    listing.add_argument("--yes", action="store_true", help="With --check: start stdio MCP commands without prompting.")
    listing.add_argument("--json", action="store_true", help="Print machine-readable status without secret values.")

    get = commands.add_parser("get", help="Show one server's definition, scope, status and tools.")
    get.add_argument("server_id")
    with_scope(get)
    get.add_argument("--json", action="store_true", help="Print machine-readable details without secret values.")

    add = commands.add_parser(
        "add",
        help="Add or update a server: mcp add [options] <name> <url>, or mcp add [options] <name> -- <command> [args...]",
    )
    add.add_argument("server_id", help="Server name: letters, numbers, '_' and '-'.")
    add.add_argument("command_or_url", nargs="?", help="Remote URL, or the local command that starts the server.")
    add.add_argument("server_args", nargs="*", help="Arguments after the command; put them after '--'.")
    with_scope(add)
    add.add_argument("--name", help="Display name.")
    add.add_argument(
        "-t",
        "--transport",
        help="stdio, http or sse. Inferred when omitted: an http(s):// URL is http, anything else is stdio.",
    )
    add.add_argument("--url", help="Remote endpoint (alternative to the URL argument).")
    add.add_argument(
        "-H",
        "--header",
        action="append",
        default=[],
        metavar="NAME:VALUE",
        help="HTTP header such as 'Authorization: Bearer ${TOKEN}' (repeatable).",
    )
    add.add_argument("--command", dest="stdio_command", help="Local command (alternative to the command argument).")
    add.add_argument("--arg", action="append", default=[], help="Command argument (repeatable).")
    add.add_argument(
        "-e", "--env", action="append", default=[], metavar="KEY=VALUE", help="Environment variable (repeatable)."
    )
    add.add_argument("--cwd", help="Working directory for a stdio command; relative paths resolve under the project.")
    add.add_argument("--oauth", action="store_true", help="Enable OAuth for this HTTP server.")
    add.add_argument("--client-id", "--oauth-client-id", dest="oauth_client_id", help="OAuth client ID.")
    add.add_argument(
        "--client-secret",
        "--oauth-client-secret",
        dest="oauth_client_secret",
        nargs="?",
        const=_SECRET_PROMPT,
        default=None,
        metavar="VALUE",
        help="OAuth client secret. Without a value it prompts with masked input; place it last or before '--'.",
    )
    add.add_argument("--callback-port", type=int, help="Fixed localhost port for the OAuth callback.")
    add.add_argument("--oauth-client-secret-env", help="Environment variable that holds the OAuth client secret.")
    add.add_argument(
        "--oauth-auth-method",
        choices=["client_secret_post", "client_secret_basic", "none"],
        help="Token endpoint authentication method.",
    )
    add.add_argument("--oauth-scope", help="OAuth scopes to request.")
    add.add_argument("--redirect-uri", help="OAuth redirect URI; defaults to an ephemeral localhost callback.")
    add.add_argument("--disabled", action="store_true", help="Add the server disabled.")
    add.add_argument("--verify", action="store_true", help="Connect to the server after saving it.")

    add_json = commands.add_parser("add-json", help="Add a server from a portable mcpServers entry or block.")
    add_json.add_argument("server_id", nargs="?", help="Server name; required unless the JSON holds several servers.")
    add_json.add_argument("json", help="JSON text, a path to a JSON file, or '-' for stdin.")
    with_scope(add_json)
    add_json.add_argument(
        "--client-secret",
        dest="oauth_client_secret",
        nargs="?",
        const=_SECRET_PROMPT,
        default=None,
        metavar="VALUE",
        help="OAuth client secret. Without a value it prompts with masked input.",
    )
    add_json.add_argument("--verify", action="store_true", help="Connect to the server after saving it.")

    remove = commands.add_parser("remove", help="Remove a server's definition.")
    remove.add_argument("server_id")
    with_scope(remove)

    for name in ("enable", "disable"):
        toggle = commands.add_parser(name, help=f"{name.capitalize()} a server without removing its definition.")
        toggle.add_argument("server_id")
        with_scope(toggle)

    verify = commands.add_parser("verify", help="Verify one MCP server, or all enabled servers with --all.")
    verify.add_argument("server_id", nargs="?", help="MCP server id to verify.")
    verify.add_argument("--all", action="store_true", help="Verify all enabled MCP servers.")
    verify.add_argument("--yes", action="store_true", help="Confirm starting stdio MCP commands without prompting.")
    verify.add_argument("--no-browser", action="store_true", help="Print OAuth URL without opening a browser.")
    verify.add_argument("--json", action="store_true", help="Print verification results as JSON.")

    login = commands.add_parser("login", help="Sign in to an OAuth MCP server from the shell.")
    login.add_argument("server_id")
    login.add_argument(
        "--no-browser", action="store_true", help="Print the authorization URL without opening a browser."
    )

    logout = commands.add_parser("logout", help="Clear a server's stored sign-in and cached tools.")
    logout.add_argument("server_id")

    commands.add_parser("trust", help="Trust this project's MCP servers and enable them.")
    commands.add_parser("untrust", help="Stop loading this project's MCP servers.")
    commands.add_parser(
        "reset-project-choices", help="Forget this project's enable/disable choices and its trust decision."
    )
    importer = commands.add_parser(
        "import",
        help="Copy MCP servers from Claude Code, Claude Desktop, Cursor, VS Code, Windsurf, Gemini CLI or Codex.",
    )
    importer.add_argument(
        "--from",
        dest="sources",
        action="append",
        metavar="SOURCE",
        help="Read one source by id: " + ", ".join(IMPORT_SOURCE_IDS) + " (repeatable).",
    )
    importer.add_argument("--file", dest="files", action="append", type=Path, help="Read this file (repeatable).")
    importer.add_argument("--server", dest="servers", action="append", metavar="NAME", help="Only these servers.")
    with_scope(importer)
    importer.add_argument("--dry-run", action="store_true", help="Show what would be added without writing anything.")
    importer.add_argument("--yes", action="store_true", help="Import without the interactive picker.")
    importer.add_argument(
        "--overwrite", action="store_true", help="Replace servers already defined in the target scope."
    )
    return mcp


@dataclass
class McpContext:
    """Resolved project, state and settings for one ``mcp`` command."""

    args: argparse.Namespace
    project_path: Path
    settings_store: SettingsStore
    settings: CliSettings

    def reload(self) -> LoadedMCPConfig:
        return load_mcp_config_for(self.project_path, self.settings_store.root, self.settings)

    def service(self, config: Optional[LoadedMCPConfig] = None) -> MCPService:
        return MCPService(config or self.reload(), state_dir=self.settings_store.root, project_path=self.project_path)

    def save_settings(self) -> None:
        self.settings_store.save(self.settings)

    def definitions(self, server_id: str, scope: Optional[str]) -> list[ServerDefinition]:
        found = find_definitions(
            server_id, project_path=self.project_path, state_dir=self.settings_store.root, settings=self.settings
        )
        return [definition for definition in found if scope is None or definition.scope == scope]


async def run_mcp(args: argparse.Namespace, *, project_path: Path, settings_store: SettingsStore) -> int:
    settings = settings_store.load()
    if getattr(args, "trust_mcp", False):
        settings.trust_mcp_project(project_path)
        settings_store.save(settings)
    ctx = McpContext(args=args, project_path=project_path, settings_store=settings_store, settings=settings)
    handler = _COMMANDS[args.mcp_command]
    return await handler(ctx)


def _clean(value: object) -> str:
    return _CONTROL_CHARS_RE.sub("", str(value))


def _note(message: str) -> None:
    print(f"mcp: {message}", file=sys.stderr)


def _scope_arg(args: argparse.Namespace) -> Optional[str]:
    if getattr(args, "project_config", False):
        return "project"
    return getattr(args, "scope", None)


def _effective_scope(args: argparse.Namespace) -> MCPScope:
    scope = _scope_arg(args)
    if scope is None:
        return "user"
    assert scope in SCOPES
    return cast(MCPScope, scope)


def _transport_label(transport: object) -> str:
    return TRANSPORT_LABELS.get(str(transport), str(transport))


def _one_definition(ctx: McpContext, server_id: str, *, action: str) -> ServerDefinition:
    """The one scope that defines a server, or an error naming why there is not exactly one."""
    scope = _scope_arg(ctx.args)
    definitions = ctx.definitions(server_id, scope)
    if not definitions:
        where = f" in {scope} scope" if scope else ""
        raise ValueError(f"MCP server not found{where}: {server_id}")
    if len(definitions) > 1:
        scopes = ", ".join(definition.scope for definition in definitions)
        raise ValueError(
            f"MCP server '{server_id}' is defined in more than one scope ({scopes}). "
            f"Re-run with --scope to choose the one to {action}."
        )
    return definitions[0]


def _row_hint(row: dict[str, Any]) -> str:
    health: MCPServerHealth = row["health"]
    parts = [health.hint]
    if row.get("missing_env"):
        parts.append("missing env: " + ", ".join(row["missing_env"]))
    return " · ".join(part for part in parts if part)


def _server_json(row: dict[str, Any]) -> dict[str, Any]:
    health: MCPServerHealth = row["health"]
    return {
        "id": _clean(row["id"]),
        "name": _clean(row["name"]),
        "scope": row["scope"],
        "source": row["source"],
        "file": row["file"],
        "transport": row["transport"],
        "enabled": row["enabled"],
        "enabled_override": row["enabled_override"],
        "oauth": bool(row["oauth"]),
        "health": health.health,
        "status": row["status"],
        "message": health.message,
        "tool_count": health.tool_count,
        "verified_at": health.verified_at,
        "missing_env": list(row["missing_env"]),
    }


async def _cmd_list(ctx: McpContext) -> int:
    args = ctx.args
    config = ctx.reload()
    service = ctx.service(config)
    for diagnostic in config.diagnostics:
        _note(diagnostic)

    all_connected = True
    if getattr(args, "check", False):
        scope = _scope_arg(args)
        targets = [server.id for server in config.enabled_servers if scope is None or server.scope == scope]
        if not _confirm_stdio_verification(args, config, targets):
            return 2
        print("Checking MCP server health...", file=sys.stderr)
        results = [
            await service.verify_server(server_id, interactive_oauth=False, open_browser=False, output=sys.stderr)
            for server_id in targets
        ]
        all_connected = all(result.ok for result in results)

    rows = [row for row in service.list_status_rows() if _scope_arg(args) in (None, row["scope"])]
    if getattr(args, "json", False):
        payload = {
            "project": str(ctx.project_path),
            "trusted": bool(config.project_trusted),
            "servers": [_server_json(row) for row in rows],
        }
        print(json.dumps(payload, indent=2))
    else:
        _print_list(ctx, rows)
    return 0 if all_connected else 1


def _print_list(ctx: McpContext, rows: list[dict[str, Any]]) -> None:
    print(f"MCP servers for {ctx.project_path}")
    if not rows:
        print(
            "\nNo MCP servers configured.\n\nAdd one:\n"
            "  kolega-code mcp add --transport http notion https://mcp.notion.com/mcp\n"
            "  kolega-code mcp add airtable -- npx -y airtable-mcp-server\n"
            '  kolega-code mcp add-json weather \'{"type":"http","url":"https://example.com/mcp"}\''
        )
        return
    print()
    cells = []
    for row in rows:
        health: MCPServerHealth = row["health"]
        cells.append(
            (
                health.glyph,
                _clean(row["id"]),
                str(row["scope"]),
                _transport_label(row["transport"]),
                health.label,
                _row_hint(row),
            )
        )
    widths = [max(len(cell[index]) for cell in cells) for index in (1, 2, 3)]
    for glyph, name, scope, transport, label, hint in cells:
        line = f"  {glyph} {name:<{widths[0]}}  {scope:<{widths[1]}}  {transport:<{widths[2]}}  {label}"
        print(f"{line}  {hint}" if hint else line)
    print(f"\n{SCOPE_SUMMARY}")


def _verified_tool_names(server: MCPServerConfig, service: MCPService) -> list[str]:
    status = service.server_status(server)
    if not status or status.status != "verified" or status.fingerprint != server_fingerprint(server):
        return []
    return [final for _, final in exposed_tool_names_for(server.id, status.tools)]


def _shadowed_row(server: MCPServerConfig, definition: ServerDefinition, config: LoadedMCPConfig) -> dict[str, Any]:
    """A status row for a definition that a higher-precedence scope overrides."""
    row: dict[str, Any] = {
        "id": server.id,
        "name": server.display_name,
        "source": server.source,
        "scope": definition.scope,
        "file": str(definition.file) if definition.file is not None else None,
        "transport": server.transport,
        "enabled": server.enabled,
        "oauth": server.oauth.enabled,
        "status": "shadowed",
        "tool_count": 0,
        "message": "",
    }
    row.update(health_fields(server, None, config=config, trusted=True))
    return row


async def _cmd_get(ctx: McpContext) -> int:
    args = ctx.args
    server_id = sanitize_mcp_server_id(args.server_id.strip())
    config = ctx.reload()
    definition = _one_definition(ctx, server_id, action="inspect")
    server = load_definition(
        server_id,
        definition.scope,
        project_path=ctx.project_path,
        state_dir=ctx.settings_store.root,
        settings=ctx.settings,
    )
    if server is None:
        raise ValueError(f"MCP server not found in {definition.scope} scope: {server_id}")
    service = ctx.service(config)
    effective = next((row for row in service.list_status_rows() if row["id"] == server_id), None)
    shadowed_by: Optional[str] = None
    if effective is not None and effective["scope"] == definition.scope:
        row = effective
    else:
        row = _shadowed_row(server, definition, config)
        shadowed_by = effective["scope"] if effective is not None else None
    health: MCPServerHealth = row["health"]
    tools = _verified_tool_names(server, service) if shadowed_by is None else []

    if getattr(args, "json", False):
        data = _server_json(row)
        data.update(
            endpoint=_endpoint_json(server),
            detail=_detail_json(server),
            tools=[_clean(name) for name in tools],
            shadowed_by=shadowed_by,
        )
        print(json.dumps(data, indent=2))
        return 0

    location = scope_location(definition.scope, ctx.project_path, ctx.settings_store.root)
    print(_clean(server.id))
    print(f"  Scope:       {definition.scope} ({_clean(location)})")
    if shadowed_by is not None:
        print(f"  Status:      Shadowed by the {shadowed_by} definition; that one is used")
    else:
        print(f"  Status:      {_status_line(health)}")
    print(f"  Transport:   {_transport_label(server.transport)}")
    for label, value in _endpoint_lines(server):
        print(f"  {label:<13}{value}")
    print(f"  OAuth:       {_sign_in_label(server)}")
    print(f"  Timeout:     {server.timeout_seconds:g}s")
    if row["missing_env"]:
        print(f"  Environment: missing {_clean(', '.join(row['missing_env']))}")
    if tools:
        shown = ", ".join(_clean(name) for name in tools[:_MAX_TOOL_NAMES_SHOWN])
        more = f", ... (+{len(tools) - _MAX_TOOL_NAMES_SHOWN} more)" if len(tools) > _MAX_TOOL_NAMES_SHOWN else ""
        print(f"  Tools:       {shown}{more}")
    return 0


def _status_line(health: MCPServerHealth) -> str:
    return f"{health.label} — {health.hint}" if health.hint else health.label


def _endpoint_lines(server: MCPServerConfig) -> list[tuple[str, str]]:
    lines: list[tuple[str, str]] = []
    if server.transport == "stdio":
        lines.append(("Command:", _clean(server.command or "")))
        if server.args:
            lines.append(("Args:", _clean(" ".join(server.args))))
    else:
        lines.append(("URL:", _clean(server.url or "")))
        if server.headers:
            lines.append(("Headers:", _masked_mapping(server.headers)))
    if server.env:
        lines.append(("Env:", _masked_mapping(server.env)))
    if server.cwd:
        lines.append(("Working dir:", _clean(server.cwd)))
    return lines


def _endpoint_json(server: MCPServerConfig) -> dict[str, Any]:
    if server.transport == "stdio":
        return {"command": _clean(server.command or ""), "args": [_clean(arg) for arg in server.args]}
    return {"url": _clean(server.url or "")}


def _detail_json(server: MCPServerConfig) -> dict[str, Any]:
    return {
        "cwd": server.cwd,
        "timeout_seconds": server.timeout_seconds,
        "sse_read_timeout_seconds": server.sse_read_timeout_seconds,
        "header_keys": sorted(_clean(key) for key in server.headers),
        "env_keys": sorted(_clean(key) for key in server.env),
        "oauth": {
            "enabled": server.oauth.enabled,
            "client_id": server.oauth.client_id,
            "redirect_uri": server.oauth.redirect_uri,
            "client_secret_env": server.oauth.client_secret_env,
            "token_endpoint_auth_method": server.oauth.token_endpoint_auth_method,
            "scope": server.oauth.scope,
        },
    }


def _masked_mapping(values: dict[str, str]) -> str:
    return _clean(", ".join(f"{key}={MASKED_VALUE}" for key in values))


def _sign_in_label(server: MCPServerConfig) -> str:
    oauth = server.oauth
    if not oauth.enabled:
        return "off"
    if not oauth.client_id:
        return "on, dynamic client registration"
    if oauth.client_secret_env:
        return "on, pre-registered client (secret from an environment variable)"
    if oauth.client_secret:
        return "on, pre-registered client (secret stored in the definition)"
    return "on, pre-registered client (public client)"


def _transport_argument(value: str) -> str:
    mapped = TRANSPORT_ALIASES.get(value.strip().lower().replace("_", "-"))
    if mapped is None:
        raise ValueError(f"unknown transport {value!r}; use stdio, http or sse")
    return mapped


def _parse_pairs(values: list[str], flag: str, *, header: bool = False) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in values:
        shape = "Name: value" if header else "NAME=VALUE"
        if header and ":" in item:
            name, value = item.split(":", 1)
            name, value = name.strip(), value.strip()
        elif "=" in item:
            name, value = item.split("=", 1)
            name = name.strip()
        else:
            raise ValueError(f"{flag} values must look like {shape}")
        if not name:
            raise ValueError(f"{flag} values must look like {shape}")
        parsed[name] = value
    return parsed


def _secret_from_args(args: argparse.Namespace) -> Optional[str]:
    value = getattr(args, "oauth_client_secret", None)
    if value is None or value is not _SECRET_PROMPT:
        return value
    import getpass

    isatty = getattr(sys.stdin, "isatty", None)
    if callable(isatty) and isatty():
        return getpass.getpass("OAuth client secret: ").strip() or None
    return sys.stdin.readline().strip() or None


def _validation_message(exc: Exception) -> str:
    try:
        errors = exc.errors()  # pyright: ignore[reportAttributeAccessIssue]
    except AttributeError:
        return " ".join(str(exc).split())
    messages: list[str] = []
    for error in errors:
        message = str(error.get("msg", "")).removeprefix("Value error, ")
        if message and message not in messages:
            messages.append(message)
    return "; ".join(messages) or "Invalid MCP server configuration."


def _validated(payload: dict[str, Any]) -> MCPServerConfig:
    try:
        return MCPServerConfig.model_validate(payload)
    except ValueError as exc:
        raise ValueError(_validation_message(exc)) from exc


def _server_from_add_args(args: argparse.Namespace) -> MCPServerConfig:
    headers = _parse_pairs(getattr(args, "header", None) or [], "--header", header=True)
    env = _parse_pairs(getattr(args, "env", None) or [], "--env")
    positional = (getattr(args, "command_or_url", None) or "").strip()
    url: Optional[str] = getattr(args, "url", None)
    command: Optional[str] = getattr(args, "stdio_command", None)
    server_args = list(getattr(args, "arg", None) or [])
    extra_args = list(getattr(args, "server_args", None) or [])
    target = positional or url or command or ""
    declared = _transport_argument(getattr(args, "transport", None) or "") if getattr(args, "transport", None) else None
    if declared is not None:
        transport = declared
    elif target.startswith(("http://", "https://")):
        transport = "streamable_http"
    elif target.startswith("sse://"):
        transport = "sse"
    else:
        transport = "stdio"

    if transport == "stdio":
        if positional:
            if command is not None:
                raise ValueError("Pass the server's command once, either as the argument or with --command.")
            command = positional
        server_args = [*server_args, *extra_args]
        if url is not None:
            raise ValueError("A stdio server takes a command, not --url; use --transport http for a URL.")
        if not command:
            raise ValueError("A stdio server requires a command. Put it after '--': mcp add <name> -- <command>")
    else:
        if positional:
            if url is not None and url != positional:
                raise ValueError("Pass the server's URL once, either as the argument or with --url.")
            url = positional
        if extra_args:
            raise ValueError("A URL takes no command arguments.")
        if not url:
            raise ValueError(f"An {_transport_label(transport)} server requires a URL.")

    secret = _secret_from_args(args)
    redirect_uri: Optional[str] = getattr(args, "redirect_uri", None)
    callback_port = getattr(args, "callback_port", None)
    if callback_port is not None and redirect_uri is None:
        if not 1 <= callback_port <= 65535:
            raise ValueError("--callback-port must be between 1 and 65535")
        redirect_uri = f"http://127.0.0.1:{callback_port}/callback"
    oauth = {
        "enabled": bool(getattr(args, "oauth", False))
        or bool(getattr(args, "oauth_client_id", None) or secret or getattr(args, "oauth_client_secret_env", None)),
        "scope": getattr(args, "oauth_scope", None),
        "redirect_uri": redirect_uri,
        "client_id": getattr(args, "oauth_client_id", None),
        "client_secret": secret,
        "client_secret_env": getattr(args, "oauth_client_secret_env", None),
        "token_endpoint_auth_method": getattr(args, "oauth_auth_method", None),
    }
    return _validated(
        {
            "id": args.server_id,
            "name": getattr(args, "name", None),
            "transport": transport,
            "enabled": not bool(getattr(args, "disabled", False)),
            "url": url,
            "headers": headers,
            "command": command,
            "args": server_args,
            "env": env,
            "cwd": getattr(args, "cwd", None),
            "oauth": oauth,
        }
    )


async def _verify_after_change(ctx: McpContext, server_ids: list[str]) -> int:
    service = ctx.service()
    all_ok = True
    for server_id in server_ids:
        result = await service.verify_server(server_id, interactive_oauth=True, open_browser=True, output=sys.stderr)
        print(f"{'✓' if result.ok else '✗'} {_clean(server_id)}: {result.message}")
        all_ok = all_ok and result.ok
    return 0 if all_ok else 1


def _save_if_settings_backed(ctx: McpContext, *scopes: str) -> None:
    """``local`` definitions and project enable choices live in settings.json."""
    if "local" in scopes or "project" in scopes:
        ctx.save_settings()


def _report_add(ctx: McpContext, servers: list[tuple[str, MCPServerConfig]], scope: MCPScope) -> None:
    """Print what was saved and the follow-up a person most likely needs."""
    location = scope_location(scope, ctx.project_path, ctx.settings_store.root)
    for outcome, server in servers:
        print(f"{outcome.capitalize()} MCP server {_clean(server.id)} to {scope} scope ({_clean(location)})")
    if scope == "project":
        if any(has_literal_secrets(server) for _, server in servers):
            _note("warning: this file is shared. Prefer ${NAME} references over literal secrets in headers and env.")
        if not ctx.settings.is_mcp_project_trusted(ctx.project_path):
            _note("Project MCP servers are not trusted yet. Run `kolega-code mcp trust` to enable them.")


def _next_step_hint(server: MCPServerConfig) -> str:
    """The command that gets a freshly saved server to "Connected"."""
    if server.oauth.enabled:
        return f"Run `kolega-code mcp login {server.id}` to sign in and expose its tools."
    return f"Run `kolega-code mcp verify {server.id}` to connect and expose its tools."


async def _cmd_add(ctx: McpContext) -> int:
    args = ctx.args
    scope = _effective_scope(args)
    server = _server_from_add_args(args)
    outcome = upsert_server(
        server, scope=scope, project_path=ctx.project_path, state_dir=ctx.settings_store.root, settings=ctx.settings
    )
    _save_if_settings_backed(ctx, scope)
    _report_add(ctx, [(outcome, server)], scope)
    if args.server_id.strip() != server.id:
        _note(f"server id '{_clean(args.server_id.strip())}' exceeds 32 characters; saved as '{server.id}'.")
    if getattr(args, "verify", False):
        return await _verify_after_change(ctx, [server.id])
    print(_next_step_hint(server))
    return 0


def _read_json_payload(text: str) -> Any:
    text = text.strip()
    try:
        if text == "-":
            return json.loads(sys.stdin.read())
        if not text.startswith(("{", "[")):
            path = Path(text).expanduser()
            if not path.is_file():
                raise ValueError("The JSON argument is neither JSON nor a path to an existing file.")
            return json.loads(path.read_text(encoding="utf-8"))
        return json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"The JSON could not be parsed: {exc}") from exc


def _json_entries(payload: Any, name: Optional[str]) -> list[tuple[str, dict[str, Any]]]:
    """``(server name, entry)`` pairs from one entry object or from an ``mcpServers`` block."""
    if not isinstance(payload, dict):
        raise ValueError("The JSON must be a server entry object or an mcpServers block keyed by server name.")
    block_key = "mcpServers" if isinstance(payload.get("mcpServers"), dict) else None
    if block_key is None and isinstance(payload.get("servers"), dict):
        block_key = "servers"
    if block_key is None:
        if not name:
            raise ValueError("add-json needs a server name before a single entry.")
        return [(name, payload)]
    block = payload[block_key]
    if name:
        if name not in block:
            raise ValueError(f'MCP server "{_clean(name)}" is not in the JSON block.')
        return [(name, block[name])]
    return list(block.items())


def _with_client_secret(server: MCPServerConfig, secret: Optional[str]) -> MCPServerConfig:
    if not secret:
        return server
    dumped = server.model_dump(mode="json", exclude={"source"})
    oauth = dict(dumped.get("oauth") or {})
    oauth["client_secret"] = secret
    if oauth.get("client_id"):
        oauth["enabled"] = True
    return _validated({**dumped, "oauth": oauth})


async def _cmd_add_json(ctx: McpContext) -> int:
    args = ctx.args
    scope = _effective_scope(args)
    secret = _secret_from_args(args)
    servers: list[tuple[str, MCPServerConfig]] = []
    for raw_name, raw in _json_entries(_read_json_payload(args.json), getattr(args, "server_id", None)):
        try:
            server_id = sanitize_mcp_server_id(raw_name)
        except ValueError as exc:
            raise ValueError(f'MCP server name "{_clean(raw_name)}" cannot be used here: {exc}') from exc
        server = _with_client_secret(_validated(standard_entry_payload(server_id, raw).payload), secret)
        outcome = upsert_server(
            server, scope=scope, project_path=ctx.project_path, state_dir=ctx.settings_store.root, settings=ctx.settings
        )
        servers.append((outcome, server))
    _save_if_settings_backed(ctx, scope)
    _report_add(ctx, servers, scope)
    if getattr(args, "verify", False):
        return await _verify_after_change(ctx, [server.id for _, server in servers])
    for _, server in servers:
        print(_next_step_hint(server))
    return 0


def _definition_location(definition: ServerDefinition, ctx: McpContext) -> str:
    if definition.file is not None:
        return _clean(str(definition.file))
    return _clean(scope_location(definition.scope, ctx.project_path, ctx.settings_store.root))


def _clear_server_state(ctx: McpContext, server_id: str) -> None:
    MCPStatusStore(ctx.settings_store.root).clear(server_id)
    MCPOAuthTokenStore(ctx.settings_store.root).clear(server_id)


def _not_trusted_note(ctx: McpContext) -> None:
    if not ctx.settings.is_mcp_project_trusted(ctx.project_path):
        print(
            "The project is not trusted for MCP yet; run `kolega-code mcp trust` to load its servers.",
            file=sys.stderr,
        )


async def _cmd_remove(ctx: McpContext) -> int:
    server_id = sanitize_mcp_server_id(ctx.args.server_id.strip())
    definition = _one_definition(ctx, server_id, action="remove")
    removed = remove_server(
        server_id,
        scope=definition.scope,
        project_path=ctx.project_path,
        state_dir=ctx.settings_store.root,
        settings=ctx.settings,
    )
    if not removed:
        raise ValueError(f"MCP server not found in {definition.scope} scope: {server_id}")
    _save_if_settings_backed(ctx, definition.scope)
    _clear_server_state(ctx, server_id)
    print(
        f"Removed MCP server {_clean(server_id)} from {definition.scope} scope ({_definition_location(definition, ctx)})"
    )
    return 0


async def _cmd_set_enabled(ctx: McpContext) -> int:
    args = ctx.args
    enabled = args.mcp_command == "enable"
    server_id = sanitize_mcp_server_id(args.server_id.strip())
    definition = _one_definition(ctx, server_id, action="enable or disable")
    changed, as_project_choice = set_server_enabled_for_scope(
        server_id,
        enabled,
        scope=definition.scope,
        project_path=ctx.project_path,
        state_dir=ctx.settings_store.root,
        settings=ctx.settings,
    )
    if not changed:
        raise ValueError(f"MCP server not found in {definition.scope} scope: {server_id}")
    verb = "Enabled" if enabled else "Disabled"
    _save_if_settings_backed(ctx, definition.scope)
    if as_project_choice:
        print(f"{verb} MCP server {_clean(server_id)} for this project. The shared definition is unchanged.")
        _not_trusted_note(ctx)
    else:
        location = _definition_location(definition, ctx)
        print(f"{verb} MCP server {_clean(server_id)} in {definition.scope} scope ({location})")
    return 0


def _confirm_stdio_verification(args: argparse.Namespace, config: LoadedMCPConfig, server_ids: list[str]) -> bool:
    """Consent before a command starts local stdio servers; they are arbitrary commands."""
    stdio_servers = [
        config.servers[server_id]
        for server_id in server_ids
        if config.servers.get(server_id) and config.servers[server_id].transport == "stdio"
    ]
    if not stdio_servers:
        return True
    if getattr(args, "yes", False):
        return True
    if not sys.stdin.isatty():
        print("Refusing to start stdio MCP server command(s) without --yes in non-interactive mode.", file=sys.stderr)
        return False
    print("This starts local stdio MCP server commands:", file=sys.stderr)
    for server in stdio_servers:
        command = " ".join([server.command or "", *server.args]).strip()
        print(f"  {_clean(server.id)}: {_clean(command)}", file=sys.stderr)
    print("Continue? [y/N] ", end="", file=sys.stderr, flush=True)
    answer = sys.stdin.readline().strip().lower()
    return answer in {"y", "yes"}


def _verify_server_ids(args: argparse.Namespace, config: LoadedMCPConfig) -> list[str]:
    if getattr(args, "all", False) and getattr(args, "server_id", None):
        raise ValueError("Use either `mcp verify SERVER_ID` or `mcp verify --all`, not both.")
    if getattr(args, "all", False):
        return [server.id for server in config.enabled_servers]
    if not getattr(args, "server_id", None):
        raise ValueError("Specify an MCP server id or --all.")
    return [sanitize_mcp_server_id(args.server_id)]


def _print_verification(results: list[MCPVerificationResult], *, json_output: bool) -> int:
    if json_output:
        print(json.dumps([result.__dict__ for result in results], default=str))
    else:
        for result in results:
            print(f"{'✓' if result.ok else '✗'} {_clean(result.server_id)}: {result.message}")
    return 0 if all(result.ok for result in results) else 1


async def _cmd_verify(ctx: McpContext) -> int:
    args = ctx.args
    config = ctx.reload()
    server_ids = _verify_server_ids(args, config)
    if not server_ids:
        raise ValueError("No MCP servers to verify.")
    if not _confirm_stdio_verification(args, config, server_ids):
        return 2
    service = ctx.service(config)
    results = [
        await service.verify_server(
            server_id, interactive_oauth=True, open_browser=not getattr(args, "no_browser", False), output=sys.stderr
        )
        for server_id in server_ids
    ]
    return _print_verification(results, json_output=bool(getattr(args, "json", False)))


async def _cmd_login(ctx: McpContext) -> int:
    args = ctx.args
    server_id = sanitize_mcp_server_id(args.server_id.strip())
    config = ctx.reload()
    server = config.servers.get(server_id)
    if server is None:
        if any(candidate.id == server_id for candidate in config.untrusted_project_servers):
            raise ValueError(
                f"MCP server '{server_id}' belongs to this project, which is not trusted for MCP. "
                "Run `kolega-code mcp trust` first."
            )
        raise ValueError(f"MCP server not found: {server_id}")
    if not server.oauth.enabled:
        raise ValueError(f"MCP server '{server_id}' has no OAuth configured. Use `kolega-code mcp verify {server_id}`.")
    result = await ctx.service(config).verify_server(
        server_id, interactive_oauth=True, open_browser=not getattr(args, "no_browser", False), output=sys.stderr
    )
    return _print_verification([result], json_output=False)


async def _cmd_logout(ctx: McpContext) -> int:
    server_id = sanitize_mcp_server_id(ctx.args.server_id.strip())
    _one_definition(ctx, server_id, action="sign out of")
    _clear_server_state(ctx, server_id)
    print(f"Signed out of MCP server {_clean(server_id)}; its cached tools were cleared.")
    server = ctx.reload().servers.get(server_id)
    if server is not None and server.oauth.enabled:
        print(f"Run `kolega-code mcp login {server_id}` to sign in again.")
    return 0


async def _cmd_trust(ctx: McpContext) -> int:
    ctx.settings.trust_mcp_project(ctx.project_path)
    ctx.save_settings()
    print(f"Trusted MCP servers for {ctx.project_path}.")
    return 0


async def _cmd_untrust(ctx: McpContext) -> int:
    if not untrust_project(ctx.settings, ctx.project_path):
        print(f"This project was not trusted for MCP: {ctx.project_path}")
        return 0
    ctx.save_settings()
    print(f"No longer loading MCP servers from {ctx.project_path}.")
    return 0


async def _cmd_reset_project_choices(ctx: McpContext) -> int:
    choices, trusted = reset_project_choices(ctx.settings, ctx.project_path)
    ctx.save_settings()
    reset: list[str] = []
    if trusted:
        reset.append("the trust decision")
    if choices:
        reset.append(f"{choices} enable/disable choice{'s' if choices != 1 else ''}")
    if not reset:
        print(f"Nothing to reset for {ctx.project_path}.")
        return 0
    print(f"Reset {' and '.join(reset)} for {ctx.project_path}.")
    return 0


def _stdin_is_interactive() -> bool:
    isatty = getattr(sys.stdin, "isatty", None)
    return bool(isatty and isatty())


def _ask(prompt: str) -> str:
    print(prompt, end="", flush=True)
    return sys.stdin.readline().strip()


def _parse_selection(answer: str, count: int) -> set[int]:
    """Zero-based indexes chosen by a ``1,3-5`` or ``all`` answer; empty means every entry."""
    answer = answer.strip().lower()
    if answer in {"", "all"}:
        return set(range(count))
    if answer == "none":
        return set()
    chosen: set[int] = set()
    for token in answer.replace(",", " ").split():
        bounds = token.split("-", 1) if "-" in token else (token, token)
        if not all(part.isdigit() for part in bounds):
            raise ValueError(f"Could not read the selection '{token}'; use numbers, ranges, all or none.")
        chosen.update(range(int(bounds[0]) - 1, int(bounds[1])))
    return {index for index in chosen if 0 <= index < count}


def _sources_with_servers(found: list[Any]) -> str:
    if not found:
        return " None of " + ", ".join(IMPORT_SOURCE_IDS) + " defines a server on this machine."
    return " Sources with servers here: " + ", ".join(source.id for source in found) + "."


async def _cmd_import(ctx: McpContext) -> int:
    from kolega_code.mcp.importers import discover_import_sources, import_file, import_servers

    args = ctx.args
    overwrite = bool(getattr(args, "overwrite", False))
    interactive = _stdin_is_interactive() and not getattr(args, "yes", False)
    found = discover_import_sources(ctx.project_path)
    by_id = {source.id: source for source in found}
    sources = []
    for source_id in getattr(args, "sources", None) or []:
        if source_id not in by_id:
            raise ValueError(f"No MCP servers were found for '{source_id}'." + _sources_with_servers(found))
        sources.append(by_id[source_id])
    sources.extend(import_file(path, ctx.project_path) for path in getattr(args, "files", None) or [])
    explicit = bool(getattr(args, "sources", None) or getattr(args, "files", None))
    unattended_needs = "--scope plus --yes or --dry-run"
    if not explicit:
        if not found:
            print("No MCP servers found in other tools." + _sources_with_servers(found))
            print("Import a specific file with `kolega-code mcp import --file PATH`.")
            return 1
        if not interactive:
            raise ValueError(f"Import without a terminal needs --from SOURCE or --file PATH, and {unattended_needs}.")
        print(f"Found MCP servers in {len(found)} tool{'s' if len(found) != 1 else ''}:")
        print()
        for index, source in enumerate(found, start=1):
            count = len(source.candidates)
            print(f"  {index}. {source.label:<18} {count} server{'s' if count != 1 else ''}  ({source.origins})")
        print()
        chosen = _parse_selection(_ask("Import from [numbers, ranges, all] (Enter for all): "), len(found))
        sources = [source for index, source in enumerate(found) if index in chosen]
    if not interactive and (
        _scope_arg(args) is None or not (getattr(args, "yes", False) or getattr(args, "dry_run", False))
    ):
        raise ValueError(f"Import without a terminal needs {unattended_needs}.")

    scope = _effective_scope(args)
    if interactive and _scope_arg(args) is None:
        answer = _ask(f"Add to scope [user, project, local] (Enter for {scope}): ").lower()
        if answer and answer not in SCOPES:
            raise ValueError(f"Unknown scope '{answer}'; use user, project or local.")
        scope = cast(MCPScope, answer or scope)

    wanted = getattr(args, "servers", None) or None
    candidates = [
        candidate
        for source in sources
        for candidate in source.candidates
        if wanted is None or candidate.original_name in wanted or candidate.name in wanted
    ]
    if not candidates:
        print("No MCP servers to import.")
        return 1

    def blocked(candidate: Any) -> str:
        if not candidate.importable:
            return candidate.unsupported
        if not overwrite and scope in [definition.scope for definition in ctx.definitions(candidate.name, None)]:
            return f"already defined in {scope} scope"
        return ""

    selected = [candidate for candidate in candidates if not blocked(candidate)]
    if interactive and wanted is None:
        labels = ", ".join(source.label for source in sources)
        print(f"\nServers found in {labels}:\n")
        for index, candidate in enumerate(candidates, start=1):
            mark = " " if blocked(candidate) else "x"
            transport = _transport_label(candidate.server.transport) if candidate.server else "?"
            line = (
                f"  [{mark}] {index}. {_clean(candidate.original_name):<24} {transport:<6} {candidate.origin:<8} "
                f"{_clean(candidate.endpoint):<40} {candidate.source_label}"
            )
            note = blocked(candidate) or candidate.notes
            print(f"{line}\n         · {_clean(note)}" if note else line)
        print()
        chosen = _parse_selection(
            _ask("Import [numbers, ranges, all, none] (Enter for every unblocked server): "), len(candidates)
        )
        selected = [
            candidate for index, candidate in enumerate(candidates) if index in chosen and not blocked(candidate)
        ]
    # An unattended run never shows the picker, so it reports what it left out and why.
    if not interactive:
        for candidate in candidates:
            reason = blocked(candidate)
            if reason:
                print(f"Skipped {_clean(candidate.original_name)}: {_clean(reason)}")
    if not selected:
        print("No servers selected; nothing was changed.")
        return 0
    if getattr(args, "dry_run", False):
        for candidate in selected:
            notes = f" · {_clean(candidate.notes)}" if candidate.notes else ""
            print(f"Would add {_clean(candidate.name)} to {scope} scope ({_clean(candidate.endpoint)}){notes}")
        return 0

    outcomes = import_servers(
        selected,
        scope=scope,
        project_path=ctx.project_path,
        state_dir=ctx.settings_store.root,
        settings=ctx.settings,
        overwrite=overwrite,
    )
    _save_if_settings_backed(ctx, scope)
    for outcome in outcomes:
        if outcome.outcome == "skipped":
            print(f"Skipped {_clean(outcome.name)}: {outcome.detail}")
            continue
        line = f"{outcome.outcome.capitalize()} MCP server {_clean(outcome.name)} ({scope} scope)"
        print(f"{line}; {outcome.detail}" if outcome.detail else line)
    print("Run `kolega-code mcp verify --all` to connect them and expose their tools.")
    return 0


_HEALTH_TONES = {
    "connected": "success",
    "failed": "error",
    "needs_auth": "warning",
    "stale": "warning",
    "pending_trust": "warning",
    "disabled": "muted",
    "unverified": "muted",
}


def doctor_mcp_lines(project_path: Path, state_dir: Path, settings: CliSettings) -> list[tuple[str, str, str]]:
    """``(label, value, tone)`` rows for ``kolega-code doctor``. Offline: never connects."""
    config = load_mcp_config_for(project_path, state_dir, settings)
    service = MCPService(config, state_dir=state_dir, project_path=project_path)
    for diagnostic in config.diagnostics:
        _note(diagnostic)
    rows = service.list_status_rows()
    if not rows:
        return [("MCP servers", "none configured", "muted")]
    lines: list[tuple[str, str, str]] = []
    for row in rows:
        health: MCPServerHealth = row["health"]
        server_id = str(row["id"])
        state = "disabled" if not row["enabled"] else "enabled"
        detail = f"{_transport_label(row['transport'])} · {row['scope']} · {state} · {health.label}"
        if health.tool_count:
            detail += f" · {health.tool_count} tool(s)"
        lines.append((f"MCP {_clean(server_id)}", detail, _HEALTH_TONES[health.health]))
        note = _tool_name_note(config, service, server_id)
        if note:
            lines.append(
                (f"MCP {_clean(server_id)} tool names", f"{note} See `kolega-code mcp list` for detail.", "warning")
            )
    return lines


def _tool_name_note(config: LoadedMCPConfig, service: MCPService, server_id: str) -> str:
    server = config.servers.get(server_id)
    if server is None:
        return ""
    status = service.server_status(server)
    if not status or status.status != "verified" or status.fingerprint != server_fingerprint(server):
        return ""
    return mcp_tool_name_adjustment_note(server.id, status.tools)


_COMMANDS: dict[str, Any] = {
    "import": _cmd_import,
    "list": _cmd_list,
    "get": _cmd_get,
    "add": _cmd_add,
    "add-json": _cmd_add_json,
    "remove": _cmd_remove,
    "enable": _cmd_set_enabled,
    "disable": _cmd_set_enabled,
    "verify": _cmd_verify,
    "login": _cmd_login,
    "logout": _cmd_logout,
    "trust": _cmd_trust,
    "untrust": _cmd_untrust,
    "reset-project-choices": _cmd_reset_project_choices,
}
