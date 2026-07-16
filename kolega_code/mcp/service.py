"""High-level MCP verification, status, and tool-call service."""

from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

from kolega_code.llm.models import ImageBlock, TextBlock
from kolega_code.tools import ToolError

from .config import LoadedMCPConfig, MCPServerConfig, server_fingerprint
from .oauth import interactive_oauth_interaction
from .state import MCPServerStatus, MCPStatusStore, MCPToolStatus, MCPOAuthTokenStore
from .transport import open_mcp_session

MCP_TOOL_PREFIX = "mcp__"
MCP_TOOL_SEPARATOR = "__"

MCP_FAILURE_MESSAGE_GENERIC = "Verification failed. Check the MCP server configuration and try again."
MCP_FAILURE_MESSAGE_GITHUB_COPILOT_OAUTH = (
    "GitHub remote MCP OAuth failed: this endpoint does not support generic MCP dynamic client registration. "
    "Use a PAT Authorization header for this endpoint, or a client with GitHub-provided OAuth support."
)
MCP_FAILURE_MESSAGE_OAUTH_REGISTRATION = (
    "OAuth dynamic client registration failed. This server may require a pre-registered OAuth client or "
    "bearer-token header."
)
MCP_FAILURE_MESSAGE_OAUTH_UNAUTHORIZED = (
    "MCP OAuth authorization failed. Check the server credentials or bearer-token header."
)
MCP_FAILURE_MESSAGE_UNAUTHORIZED = (
    "MCP authentication failed. Check the configured credentials or Authorization header."
)
MCP_FAILURE_MESSAGE_NOT_FOUND = "MCP server endpoint was not found. Check the configured URL."
MCP_FAILURE_MESSAGE_TIMEOUT = "MCP server verification timed out. Check the server URL and network connectivity."
MCP_FAILURE_MESSAGE_CONNECTION = (
    "Could not connect to the MCP server. Check the command, URL, and network connectivity."
)
MCP_FAILURE_MESSAGE_COMMAND = "Could not start the MCP server command. Check the configured command and arguments."
MCP_FAILURE_MESSAGE_PERMISSION = "Permission denied while starting or contacting the MCP server."
MCP_TOOL_FAILURE_MESSAGE_GENERIC = "MCP tool call failed."
MCP_TOOL_FAILURE_MESSAGE_TIMEOUT = "MCP tool call timed out. Check the server URL and network connectivity."

_SAFE_FAILED_STATUS_MESSAGES = (
    MCP_FAILURE_MESSAGE_GENERIC,
    MCP_FAILURE_MESSAGE_GITHUB_COPILOT_OAUTH,
    MCP_FAILURE_MESSAGE_OAUTH_REGISTRATION,
    MCP_FAILURE_MESSAGE_OAUTH_UNAUTHORIZED,
    MCP_FAILURE_MESSAGE_UNAUTHORIZED,
    MCP_FAILURE_MESSAGE_NOT_FOUND,
    MCP_FAILURE_MESSAGE_TIMEOUT,
    MCP_FAILURE_MESSAGE_CONNECTION,
    MCP_FAILURE_MESSAGE_COMMAND,
    MCP_FAILURE_MESSAGE_PERMISSION,
)
_UNAUTHORIZED_MARKERS = ("unauthorized", "401", "forbidden", "403", "invalid token", "invalid_token")
_NOT_FOUND_MARKERS = ("404", "not found")
_TIMEOUT_MARKERS = ("timeout", "timed out", "readtimeout", "connecttimeout")
_COMMAND_FAILURE_MARKERS = (
    "command not found",
    "executable file not found",
    "no such file or directory",
    "filenotfounderror",
)
_CONNECTION_FAILURE_MARKERS = (
    "connection refused",
    "connection reset",
    "connection aborted",
    "connecterror",
    "connect error",
    "clientconnectorerror",
    "network is unreachable",
    "nodename nor servname",
    "name or service not known",
    "gaierror",
    "could not connect",
    "connect call failed",
)
_PERMISSION_MARKERS = ("permission denied", "permissionerror")


@dataclass(frozen=True)
class MCPVerificationResult:
    server_id: str
    ok: bool
    message: str
    tool_count: int = 0
    status: Optional[MCPServerStatus] = None


@dataclass(frozen=True)
class MCPExposedTool:
    server: MCPServerConfig
    tool: MCPToolStatus

    @property
    def name(self) -> str:
        return mcp_tool_name(self.server.id, self.tool.id)

    @property
    def description(self) -> str:
        base = self.tool.description or f"MCP tool `{self.tool.id}` from server `{self.server.display_name}`."
        return f"MCP server: {self.server.display_name} ({self.server.id}).\n\n{base}"


def mcp_tool_name(server_id: str, tool_id: str) -> str:
    return f"{MCP_TOOL_PREFIX}{server_id}{MCP_TOOL_SEPARATOR}{tool_id}"


def parse_mcp_tool_name(name: str) -> Optional[tuple[str, str]]:
    if not name.startswith(MCP_TOOL_PREFIX):
        return None
    rest = name[len(MCP_TOOL_PREFIX) :]
    if MCP_TOOL_SEPARATOR not in rest:
        return None
    server_id, tool_id = rest.split(MCP_TOOL_SEPARATOR, 1)
    if not server_id or not tool_id:
        return None
    return server_id, tool_id


@contextlib.contextmanager
def _suppress_mcp_sdk_terminal_logs():
    """Prevent caught MCP SDK errors from also printing raw tracebacks to stderr.

    Some SDK transport/auth tasks log exceptions before re-raising them. The
    service catches those errors and writes a concise MCP status, so the raw
    logger output is duplicate noise and can corrupt the Textual UI.
    """
    logger = logging.getLogger("mcp")
    null_handler = logging.NullHandler()
    old_propagate = logger.propagate
    logger.addHandler(null_handler)
    logger.propagate = False
    try:
        yield
    finally:
        logger.propagate = old_propagate
        logger.removeHandler(null_handler)


def _mcp_failure_message(server: MCPServerConfig, exc: BaseException) -> str:
    """Return a fixed, credential-safe status message for a caught MCP failure."""
    return _safe_mcp_failure_message(server, _exception_text_for_matching(exc))


def _mcp_tool_failure_message(server: MCPServerConfig, exc: BaseException) -> str:
    """Return a credential-safe tool error without exposing exception messages."""
    safe_message = _safe_mcp_failure_message(server, _exception_text_for_matching(exc))
    if safe_message == MCP_FAILURE_MESSAGE_TIMEOUT:
        return MCP_TOOL_FAILURE_MESSAGE_TIMEOUT
    if safe_message != MCP_FAILURE_MESSAGE_GENERIC:
        return safe_message

    exception_types = _dedupe_messages(_leaf_exception_type_names(exc))[:3]
    if not exception_types:
        return MCP_TOOL_FAILURE_MESSAGE_GENERIC
    return f"{MCP_TOOL_FAILURE_MESSAGE_GENERIC[:-1]} ({', '.join(exception_types)})."


def _exception_text_for_matching(exc: BaseException) -> str:
    messages = _dedupe_messages(_leaf_exception_messages(exc))
    if not messages:
        messages = [_single_exception_message(exc)]
    return " ".join(messages).lower()


def _safe_mcp_failure_message(server: MCPServerConfig, lower_message: str) -> str:
    if server.oauth.enabled and _contains_any(lower_message, ("registration failed", "dynamic client")):
        if _is_github_copilot_api_url(server.url):
            return MCP_FAILURE_MESSAGE_GITHUB_COPILOT_OAUTH
        return MCP_FAILURE_MESSAGE_OAUTH_REGISTRATION
    if _contains_any(lower_message, _UNAUTHORIZED_MARKERS):
        return MCP_FAILURE_MESSAGE_OAUTH_UNAUTHORIZED if server.oauth.enabled else MCP_FAILURE_MESSAGE_UNAUTHORIZED
    if _contains_any(lower_message, _NOT_FOUND_MARKERS):
        return MCP_FAILURE_MESSAGE_NOT_FOUND
    if _contains_any(lower_message, _TIMEOUT_MARKERS):
        return MCP_FAILURE_MESSAGE_TIMEOUT
    if _contains_any(lower_message, _COMMAND_FAILURE_MARKERS):
        return MCP_FAILURE_MESSAGE_COMMAND
    if _contains_any(lower_message, _PERMISSION_MARKERS):
        return MCP_FAILURE_MESSAGE_PERMISSION
    if _contains_any(lower_message, _CONNECTION_FAILURE_MARKERS):
        return MCP_FAILURE_MESSAGE_CONNECTION
    return MCP_FAILURE_MESSAGE_GENERIC


def _safe_failed_status_message(message: str) -> str:
    for safe_message in _SAFE_FAILED_STATUS_MESSAGES:
        if message == safe_message:
            return safe_message
    return MCP_FAILURE_MESSAGE_GENERIC


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _is_github_copilot_api_url(url: Optional[str]) -> bool:
    if not url:
        return False
    try:
        hostname = urlsplit(url).hostname
    except ValueError:
        return False
    return bool(hostname and hostname.rstrip(".").lower() == "api.githubcopilot.com")


def _leaf_exception_messages(exc: BaseException, seen: Optional[set[int]] = None) -> list[str]:
    seen = seen or set()
    if id(exc) in seen:
        return []
    seen.add(id(exc))

    if isinstance(exc, BaseExceptionGroup):
        messages: list[str] = []
        for child in exc.exceptions:
            messages.extend(_leaf_exception_messages(child, seen))
        if messages:
            return messages

    messages = [_single_exception_message(exc)]
    cause = exc.__cause__ or (None if exc.__suppress_context__ else exc.__context__)
    if cause is not None:
        messages.extend(_leaf_exception_messages(cause, seen))
    return messages


def _leaf_exception_type_names(exc: BaseException, seen: Optional[set[int]] = None) -> list[str]:
    seen = seen or set()
    if id(exc) in seen:
        return []
    seen.add(id(exc))

    if isinstance(exc, BaseExceptionGroup):
        names: list[str] = []
        for child in exc.exceptions:
            names.extend(_leaf_exception_type_names(child, seen))
        if names:
            return names

    name = "".join(character for character in exc.__class__.__name__ if character.isalnum() or character in "._")
    names = [name[:64] or "Exception"]
    cause = exc.__cause__ or (None if exc.__suppress_context__ else exc.__context__)
    if cause is not None:
        names.extend(_leaf_exception_type_names(cause, seen))
    return names


def _single_exception_message(exc: BaseException) -> str:
    name = exc.__class__.__name__
    text = str(exc).strip()
    if not text:
        return name
    if text.startswith(f"{name}:"):
        return text
    return f"{name}: {text}"


def _dedupe_messages(messages: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for message in messages:
        normalized = " ".join(message.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def _status_row_message(status: Optional[MCPServerStatus], *, verified: bool, stale: bool) -> str:
    if status is None:
        return "Not verified."
    if stale:
        return "Configuration changed since last verification. Verify again."
    if verified:
        return f"Verified {status.tool_count} tool(s)."
    if status.status == "failed":
        return _safe_failed_status_message(status.message)
    return "Not verified."


class MCPService:
    """Coordinates MCP config, verification state, OAuth tokens, and calls."""

    def __init__(self, config: LoadedMCPConfig, state_dir: Path, project_path: Path) -> None:
        self.config = config
        self.state_dir = Path(state_dir).expanduser()
        self.project_path = Path(project_path).expanduser().resolve()
        self.status_store = MCPStatusStore(self.state_dir)
        self.oauth_store = MCPOAuthTokenStore(self.state_dir)

    def server_status(self, server: MCPServerConfig) -> Optional[MCPServerStatus]:
        return self.status_store.get(server.id)

    def is_verified(self, server: MCPServerConfig) -> bool:
        status = self.server_status(server)
        return bool(status and status.status == "verified" and status.fingerprint == server_fingerprint(server))

    def exposed_tools(self) -> list[MCPExposedTool]:
        tools: list[MCPExposedTool] = []
        for server in self.config.enabled_servers:
            status = self.server_status(server)
            if not status or status.status != "verified":
                continue
            if status.fingerprint != server_fingerprint(server):
                continue
            for tool in status.tools:
                tools.append(MCPExposedTool(server=server, tool=tool))
        return tools

    def list_status_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for server in self.config.servers.values():
            status = self.server_status(server)
            current_fingerprint = server_fingerprint(server)
            verified = bool(status and status.status == "verified" and status.fingerprint == current_fingerprint)
            stale = bool(status and status.fingerprint and status.fingerprint != current_fingerprint)
            rows.append(
                {
                    "id": server.id,
                    "name": server.display_name,
                    "source": server.source,
                    "transport": server.transport,
                    "enabled": server.enabled,
                    "oauth": server.oauth.enabled,
                    "status": "verified"
                    if verified
                    else ("stale" if stale else (status.status if status else "unverified")),
                    "tool_count": status.tool_count if verified and status else 0,
                    "message": _status_row_message(status, verified=verified, stale=stale),
                }
            )
        return rows

    async def verify_server(
        self,
        server_id: str,
        *,
        interactive_oauth: bool = False,
        open_browser: bool = True,
        output=None,
    ) -> MCPVerificationResult:
        server = self.config.servers.get(server_id)
        if server is None:
            return MCPVerificationResult(server_id=server_id, ok=False, message=f"Unknown MCP server: {server_id}")
        fingerprint = server_fingerprint(server)
        try:
            with _suppress_mcp_sdk_terminal_logs():
                async with self._maybe_interactive_oauth(
                    server, interactive_oauth, open_browser, output
                ) as interaction:
                    async with open_mcp_session(
                        server,
                        project_path=self.project_path,
                        token_store=self.oauth_store,
                        oauth_interaction=interaction,
                    ) as session:
                        tools = await self._list_all_tools(session)
            tool_statuses = [_tool_status_from_mcp_tool(tool) for tool in tools]
            status = MCPServerStatus.verified(
                fingerprint=fingerprint,
                transport=server.transport,
                source=server.source,
                tools=tool_statuses,
                oauth=server.oauth.enabled,
            )
            self.status_store.update(server.id, status)
            return MCPVerificationResult(
                server_id=server.id,
                ok=True,
                message=status.message,
                tool_count=len(tool_statuses),
                status=status,
            )
        except Exception as exc:  # noqa: BLE001 - verification failures are reported in status
            message = _mcp_failure_message(server, exc)
            status = MCPServerStatus.failed(
                fingerprint=fingerprint,
                transport=server.transport,
                source=server.source,
                message=message,
                oauth=server.oauth.enabled,
            )
            self.status_store.update(server.id, status)
            return MCPVerificationResult(server_id=server.id, ok=False, message=message, status=status)

    async def verify_all(
        self,
        *,
        interactive_oauth: bool = False,
        open_browser: bool = True,
        output=None,
    ) -> list[MCPVerificationResult]:
        results: list[MCPVerificationResult] = []
        for server in self.config.enabled_servers:
            results.append(
                await self.verify_server(
                    server.id,
                    interactive_oauth=interactive_oauth,
                    open_browser=open_browser,
                    output=output,
                )
            )
        return results

    async def call_tool(self, server_id: str, tool_id: str, arguments: dict[str, Any]) -> Any:
        server = self.config.servers.get(server_id)
        if server is None:
            raise ToolError(f"Unknown MCP server: {server_id}")
        if not server.enabled:
            raise ToolError(f"MCP server '{server_id}' is disabled")
        if not self.is_verified(server):
            raise ToolError(f"MCP server '{server_id}' is not verified for its current configuration")
        try:
            with _suppress_mcp_sdk_terminal_logs():
                async with open_mcp_session(
                    server, project_path=self.project_path, token_store=self.oauth_store
                ) as session:
                    result = await session.call_tool(tool_id, arguments or {})
        except Exception as exc:  # noqa: BLE001 - MCP failures are returned as safe tool errors
            raise ToolError(_mcp_tool_failure_message(server, exc)) from exc
        output = _tool_result_to_agent_output(result)
        if bool(getattr(result, "isError", getattr(result, "is_error", False))):
            if isinstance(output, list):
                text = "\n\n".join(block.to_markdown() for block in output)
            else:
                text = str(output)
            raise ToolError(text or f"MCP tool '{tool_id}' failed")
        return output

    async def cleanup(self) -> None:
        """Hook for ToolExtension cleanup; sessions are per-call today."""
        return None

    @contextlib.asynccontextmanager
    async def _maybe_interactive_oauth(self, server: MCPServerConfig, interactive: bool, open_browser: bool, output):
        if server.oauth.enabled and interactive:
            async with interactive_oauth_interaction(server, open_browser=open_browser, output=output) as interaction:
                yield interaction
        else:
            yield None

    async def _list_all_tools(self, session) -> list[Any]:
        try:
            from mcp.types import PaginatedRequestParams
        except Exception:  # pragma: no cover - older SDK fallback
            PaginatedRequestParams = None

        tools: list[Any] = []
        cursor: Optional[str] = None
        while True:
            if cursor and PaginatedRequestParams is not None:
                result = await session.list_tools(params=PaginatedRequestParams(cursor=cursor))
            else:
                result = await session.list_tools()
            tools.extend(list(getattr(result, "tools", []) or []))
            cursor = getattr(result, "next_cursor", None) or getattr(result, "nextCursor", None)
            if not cursor:
                break
        return tools


def _tool_status_from_mcp_tool(tool: Any) -> MCPToolStatus:
    tool_id = str(getattr(tool, "name", ""))
    name = getattr(tool, "title", None) or tool_id
    description = getattr(tool, "description", None) or ""
    schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {}
    if not isinstance(schema, dict):
        schema = {"type": "object", "properties": {}}
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    return MCPToolStatus(id=tool_id, name=name, description=description, input_schema=schema)


def _tool_result_to_agent_output(result: Any) -> Any:
    blocks: list[Any] = []
    for item in getattr(result, "content", []) or []:
        block = _content_item_to_block(item)
        if block is not None:
            blocks.append(block)
    structured = getattr(result, "structuredContent", None)
    if structured is None:
        structured = getattr(result, "structured_content", None)
    if structured is not None:
        blocks.append(TextBlock(text="Structured content:\n" + json.dumps(structured, indent=2, default=str)))

    if any(isinstance(block, ImageBlock) for block in blocks):
        return blocks
    text = "\n\n".join(block.text for block in blocks if isinstance(block, TextBlock)).strip()
    return text if text else ""


def _content_item_to_block(item: Any) -> Optional[Any]:
    item_type = getattr(item, "type", None)
    if item_type == "text" or hasattr(item, "text"):
        text = getattr(item, "text", None)
        return TextBlock(text=str(text or ""))
    if item_type == "image" or (hasattr(item, "data") and hasattr(item, "mime_type")):
        data = getattr(item, "data", "")
        mime_type = getattr(item, "mime_type", None) or getattr(item, "mimeType", None) or "image/png"
        return ImageBlock(image_type="base64", media_type=str(mime_type), data=str(data))
    if item_type == "resource" and hasattr(item, "resource"):
        resource = getattr(item, "resource")
        text = getattr(resource, "text", None)
        uri = getattr(resource, "uri", "resource")
        if text is not None:
            return TextBlock(text=f"Resource {uri}:\n{text}")
        blob = getattr(resource, "blob", None)
        mime_type = getattr(resource, "mime_type", None) or getattr(resource, "mimeType", "application/octet-stream")
        if blob is not None:
            return TextBlock(text=f"Resource {uri}: <{mime_type} blob, {len(str(blob))} base64 chars>")
    if item_type == "audio":
        mime_type = getattr(item, "mime_type", None) or getattr(item, "mimeType", "audio/*")
        data = getattr(item, "data", "")
        return TextBlock(text=f"Audio content ({mime_type}, {len(str(data))} base64 chars) returned by MCP tool.")
    if item is not None:
        return TextBlock(text=str(item))
    return None
