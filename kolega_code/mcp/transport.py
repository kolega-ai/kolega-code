"""MCP transport/session adapters."""

from __future__ import annotations

import contextlib
import inspect
from pathlib import Path
from typing import Optional

import httpx

from .config import MCPServerConfig
from .oauth import OAuthInteraction, build_oauth_provider
from .state import MCPOAuthTokenStore


class MCPTransportError(RuntimeError):
    """Raised when an MCP transport cannot be opened."""


def _resolve_cwd(project_path: Path, cwd: Optional[str]) -> Optional[str]:
    if not cwd:
        return None
    path = Path(cwd).expanduser()
    if not path.is_absolute():
        path = project_path / path
    return str(path.resolve())


@contextlib.asynccontextmanager
async def open_mcp_session(
    server: MCPServerConfig,
    *,
    project_path: Path,
    token_store: MCPOAuthTokenStore,
    oauth_interaction: Optional[OAuthInteraction] = None,
):
    """Open and initialize a ClientSession for a configured MCP server."""
    from mcp.client.session import ClientSession

    if server.transport == "stdio":
        from mcp.client.stdio import StdioServerParameters, stdio_client

        params = StdioServerParameters(
            command=server.command or "",
            args=server.args,
            env=server.env or None,
            cwd=_resolve_cwd(project_path, server.cwd),
        )
        async with stdio_client(params) as streams:
            read_stream, write_stream = streams
            async with ClientSession(read_stream, write_stream, read_timeout_seconds=server.timeout_seconds) as session:
                await session.initialize()
                yield session
        return

    auth = await build_oauth_provider(server, token_store, interaction=oauth_interaction)
    if server.transport == "sse":
        from mcp.client.sse import sse_client

        async with sse_client(
            server.url or "",
            headers=server.headers or None,
            timeout=server.timeout_seconds,
            sse_read_timeout=server.sse_read_timeout_seconds,
            auth=auth,
        ) as streams:
            read_stream, write_stream = streams
            async with ClientSession(read_stream, write_stream, read_timeout_seconds=server.timeout_seconds) as session:
                await session.initialize()
                yield session
        return

    if server.transport == "streamable_http":
        async with _streamable_http_client(server, auth=auth) as streams:
            read_stream, write_stream = streams
            async with ClientSession(read_stream, write_stream, read_timeout_seconds=server.timeout_seconds) as session:
                await session.initialize()
                yield session
        return

    raise MCPTransportError(f"Unsupported MCP transport: {server.transport}")


@contextlib.asynccontextmanager
async def _streamable_http_client(server: MCPServerConfig, *, auth=None):
    """Compatibility wrapper for MCP SDK streamable HTTP client naming/signatures."""
    try:
        from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
    except ImportError:  # pragma: no cover - older SDK compatibility
        from mcp.client.streamable_http import create_mcp_http_client, streamablehttp_client as streamable_http_client

    signature = inspect.signature(streamable_http_client)
    params = signature.parameters
    timeout = httpx.Timeout(server.timeout_seconds)

    if "http_client" in params:
        client = create_mcp_http_client(headers=server.headers or None, timeout=timeout, auth=auth)
        async with client:
            async with streamable_http_client(server.url or "", http_client=client) as streams:
                yield streams
        return

    kwargs = {}
    if "headers" in params:
        kwargs["headers"] = server.headers or None
    if "timeout" in params:
        kwargs["timeout"] = server.timeout_seconds
    if "auth" in params:
        kwargs["auth"] = auth
    async with streamable_http_client(server.url or "", **kwargs) as streams:  # pragma: no cover - older SDK
        yield streams
