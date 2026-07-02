"""OAuth helpers for MCP HTTP transports."""

from __future__ import annotations

import asyncio
import contextlib
import sys
import webbrowser
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional
from urllib.parse import parse_qs, urlparse

from .config import MCPServerConfig
from .state import MCPOAuthTokenStore


class MCPOAuthError(RuntimeError):
    """Raised when MCP OAuth setup or callback handling fails."""


class MCPFileTokenStorage:
    """Adapter from Kolega's token store to the MCP SDK TokenStorage protocol."""

    def __init__(self, server_id: str, token_store: MCPOAuthTokenStore) -> None:
        self.server_id = server_id
        self.token_store = token_store

    async def get_tokens(self):
        from mcp.shared.auth import OAuthToken

        raw = self.token_store.get(self.server_id).tokens
        if not raw:
            return None
        return OAuthToken.model_validate(raw)

    async def set_tokens(self, tokens) -> None:
        if tokens is None:
            self.token_store.set_tokens(self.server_id, None)
            return
        self.token_store.set_tokens(self.server_id, tokens.model_dump(mode="json"))

    async def get_client_info(self):
        from mcp.shared.auth import OAuthClientInformationFull

        raw = self.token_store.get(self.server_id).client_info
        if not raw:
            return None
        return OAuthClientInformationFull.model_validate(raw)

    async def set_client_info(self, client_info) -> None:
        if client_info is None:
            self.token_store.set_client_info(self.server_id, None)
            return
        self.token_store.set_client_info(self.server_id, client_info.model_dump(mode="json"))


@dataclass
class OAuthInteraction:
    """Handlers used by the MCP SDK during an interactive OAuth flow."""

    redirect_uri: str
    redirect_handler: Callable[[str], Awaitable[None]]
    callback_handler: Callable[[], Awaitable[tuple[str, Optional[str]]]]
    close: Callable[[], Awaitable[None]]


class LocalOAuthCallbackServer:
    """Tiny one-shot loopback HTTP server for OAuth authorization-code redirects."""

    def __init__(self, redirect_uri: Optional[str] = None, *, timeout_seconds: float = 300.0) -> None:
        self._configured_redirect_uri = redirect_uri
        self.timeout_seconds = timeout_seconds
        self.redirect_uri = redirect_uri or ""
        self._server: Optional[asyncio.AbstractServer] = None
        self._future: Optional[asyncio.Future[tuple[str, Optional[str]]]] = None
        self._path = "/callback"

    async def __aenter__(self) -> "LocalOAuthCallbackServer":
        parsed = urlparse(self._configured_redirect_uri or "http://127.0.0.1:0/callback")
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise MCPOAuthError("MCP OAuth redirect_uri must be a localhost http URL")
        self._path = parsed.path or "/callback"
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 0
        self._future = asyncio.get_running_loop().create_future()
        self._server = await asyncio.start_server(self._handle_client, host=host, port=port)
        socket = self._server.sockets[0]
        bound_host, bound_port = socket.getsockname()[:2]
        # Prefer 127.0.0.1 in metadata even if the OS reports localhost/::1.
        if bound_host in {"0.0.0.0", "::", "::1"}:
            bound_host = "127.0.0.1"
        self.redirect_uri = f"http://{bound_host}:{bound_port}{self._path}"
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=10)
            request_line = line.decode("latin-1", errors="replace").strip()
            parts = request_line.split()
            target = parts[1] if len(parts) >= 2 else "/"
            # Drain headers.
            while True:
                header = await asyncio.wait_for(reader.readline(), timeout=10)
                if header in {b"\r\n", b"\n", b""}:
                    break
            parsed = urlparse(target)
            params = parse_qs(parsed.query)
            code = (params.get("code") or [""])[0]
            state = (params.get("state") or [None])[0]
            error = (params.get("error") or [""])[0]
            if parsed.path != self._path:
                await self._write_response(writer, 404, "Not found")
                return
            if error:
                if self._future and not self._future.done():
                    self._future.set_exception(MCPOAuthError(f"OAuth authorization failed: {error}"))
                await self._write_response(writer, 400, "Authorization failed. You can close this tab.")
                return
            if code:
                if self._future and not self._future.done():
                    self._future.set_result((code, state))
                await self._write_response(writer, 200, "Authorization complete. You can close this tab.")
                return
            await self._write_response(writer, 400, "Missing authorization code. You can close this tab.")
        except Exception as exc:  # noqa: BLE001 - never let HTTP callback exceptions leak
            if self._future and not self._future.done():
                self._future.set_exception(exc)
        finally:
            with contextlib.suppress(Exception):
                writer.close()
                await writer.wait_closed()

    async def _write_response(self, writer: asyncio.StreamWriter, status: int, body: str) -> None:
        reason = {200: "OK", 400: "Bad Request", 404: "Not Found"}.get(status, "OK")
        html = f"<html><body><p>{body}</p></body></html>".encode("utf-8")
        writer.write(
            f"HTTP/1.1 {status} {reason}\r\n"
            "Content-Type: text/html; charset=utf-8\r\n"
            f"Content-Length: {len(html)}\r\n"
            "Connection: close\r\n\r\n".encode("ascii")
            + html
        )
        await writer.drain()

    async def wait_for_callback(self) -> tuple[str, Optional[str]]:
        if self._future is None:
            raise MCPOAuthError("OAuth callback server is not running")
        return await asyncio.wait_for(self._future, timeout=self.timeout_seconds)


async def default_redirect_handler(url: str, *, open_browser: bool = True, output=None) -> None:
    stream = output or sys.stderr
    print("MCP OAuth authorization required:", file=stream)
    print(url, file=stream)
    if open_browser:
        try:
            await asyncio.to_thread(webbrowser.open, url)
        except Exception:
            print("Could not open a browser automatically; open the URL above manually.", file=stream)


async def build_oauth_provider(
    server: MCPServerConfig,
    token_store: MCPOAuthTokenStore,
    *,
    interaction: Optional[OAuthInteraction] = None,
):
    """Build an MCP SDK OAuthClientProvider for a server."""
    if not server.oauth.enabled:
        return None
    if not server.url:
        raise MCPOAuthError(f"MCP server '{server.id}' has OAuth enabled but no URL")

    from mcp.client.auth import OAuthClientProvider
    from mcp.shared.auth import OAuthClientMetadata

    redirect_uri = interaction.redirect_uri if interaction else server.oauth.redirect_uri
    if not redirect_uri:
        # Non-interactive agent paths should not invent a callback URL. They can
        # use existing/refreshable tokens, but a full browser flow is only allowed
        # during explicit verification.
        redirect_uri = "http://127.0.0.1:1/callback"

    metadata_kwargs = {
        "redirect_uris": [redirect_uri],
        "client_name": server.oauth.client_name or "Kolega Code",
    }
    if server.oauth.scope:
        metadata_kwargs["scope"] = server.oauth.scope
    if server.oauth.client_uri:
        metadata_kwargs["client_uri"] = server.oauth.client_uri

    return OAuthClientProvider(
        server_url=server.url,
        client_metadata=OAuthClientMetadata(**metadata_kwargs),
        storage=MCPFileTokenStorage(server.id, token_store),
        redirect_handler=interaction.redirect_handler if interaction else None,
        callback_handler=interaction.callback_handler if interaction else None,
        timeout=server.oauth.timeout_seconds,
        client_metadata_url=server.oauth.client_metadata_url,
    )


@contextlib.asynccontextmanager
async def interactive_oauth_interaction(
    server: MCPServerConfig,
    *,
    open_browser: bool = True,
    output=None,
):
    """Create handlers for an explicit interactive verification OAuth flow."""
    async with LocalOAuthCallbackServer(
        server.oauth.redirect_uri,
        timeout_seconds=server.oauth.timeout_seconds,
    ) as callback_server:

        async def redirect_handler(url: str) -> None:
            await default_redirect_handler(url, open_browser=open_browser, output=output)

        async def callback_handler() -> tuple[str, Optional[str]]:
            return await callback_server.wait_for_callback()

        yield OAuthInteraction(
            redirect_uri=callback_server.redirect_uri,
            redirect_handler=redirect_handler,
            callback_handler=callback_handler,
            close=callback_server.close,
        )


def oauth_secret_values(token_store: MCPOAuthTokenStore) -> list[str]:
    return token_store.secret_values()
