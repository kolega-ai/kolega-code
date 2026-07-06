"""Minimal async JSON-RPC 2.0 client for communicating with language servers over stdio.

Uses asyncio subprocess + Content-Length-prefixed message framing (LSP transport).
No external dependencies.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# wire types
# ---------------------------------------------------------------------------


@dataclass
class LspDiagnostic:
    """Mirrors the LSP ``Diagnostic`` struct.

    https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#diagnostic
    """

    range: dict[str, Any]
    """``{start: {line, character}, end: {line, character}}``."""

    severity: Optional[int] = None
    """1 = Error, 2 = Warning, 3 = Information, 4 = Hint."""

    code: Optional[str] = None
    message: str = ""
    source: Optional[str] = None


@dataclass
class PublishDiagnosticsParams:
    """Payload of ``textDocument/publishDiagnostics``."""

    uri: str
    diagnostics: list[LspDiagnostic]


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------


class LspClientError(RuntimeError):
    """An error from the LSP client or the language server."""


class LspClient:
    """Async JSON-RPC 2.0 client for a single language server stdio subprocess."""

    def __init__(self, command: list[str], env: Optional[dict[str, str]] = None) -> None:
        self._command = command
        self._env = env
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._request_id: int = 0
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._notification_handlers: dict[str, list[Callable[[dict[str, Any]], None]]] = {}
        self._reader_task: Optional[asyncio.Task[None]] = None
        self._lock = asyncio.Lock()
        self._running = False

    # -- lifecycle ----------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """Launch the language server subprocess and begin reading messages."""
        if self._running:
            return

        env = {**dict(self._env or {}), **self._build_env()}

        self._proc = await asyncio.create_subprocess_exec(
            *self._command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env or None,
        )
        self._running = True
        self._reader_task = asyncio.create_task(self._read_loop())
        logger.debug("LSP client started: %s (pid=%s)", self._command, self._proc.pid)

    async def stop(self) -> None:
        """Terminate the subprocess and cancel the reader task."""
        self._running = False
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None

        # Fail outstanding requests
        for rid, fut in self._pending.items():
            if not fut.done():
                fut.set_exception(LspClientError("Language server stopped"))
        self._pending.clear()

        if self._proc and self._proc.returncode is None:
            try:
                self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), timeout=3)
            except (ProcessLookupError, asyncio.TimeoutError):
                self._proc.kill()
                await self._proc.wait()
        self._proc = None
        logger.debug("LSP client stopped: %s", self._command)

    # -- request / notification --------------------------------------------

    async def request(self, method: str, params: Any = None) -> dict[str, Any]:
        """Send a JSON-RPC request and await the response."""
        if not self._running:
            raise LspClientError("Client not running")
        async with self._lock:
            rid = self._request_id = self._request_id + 1
            payload = {"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}
            fut: asyncio.Future[dict[str, Any]] = asyncio.get_event_loop().create_future()
            self._pending[rid] = fut
            await self._send(payload)
        try:
            return await asyncio.wait_for(fut, timeout=30)
        except asyncio.TimeoutError:
            self._pending.pop(rid, None)
            raise LspClientError(f"Request '{method}' timed out after 30s")

    async def notify(self, method: str, params: Any = None) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if not self._running:
            raise LspClientError("Client not running")
        payload = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        await self._send(payload)

    def on_notification(self, method: str, handler: Callable[[dict[str, Any]], None]) -> None:
        """Register a handler for server→client notifications."""
        self._notification_handlers.setdefault(method, []).append(handler)

    # -- internals ----------------------------------------------------------

    async def _send(self, payload: dict[str, Any]) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        body = json.dumps(payload)
        header = f"Content-Length: {len(body)}\r\n\r\n"
        self._proc.stdin.write((header + body).encode("utf-8"))
        await self._proc.stdin.drain()

    async def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        buffer = b""
        try:
            while self._running:
                chunk = await self._proc.stdout.read(4096)
                if not chunk:
                    break  # EOF
                buffer += chunk
                buffer = await self._process_buffer(buffer)
        except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
            pass
        except Exception:
            logger.exception("LSP reader loop error")
        finally:
            self._running = False

    async def _process_buffer(self, buffer: bytes) -> bytes:
        header_end = buffer.find(b"\r\n\r\n")
        if header_end == -1:
            return buffer

        header_text = buffer[:header_end].decode("utf-8", errors="replace")
        content_length = self._parse_content_length(header_text)
        if content_length is None:
            logger.warning("LSP: could not parse Content-Length from header: %r", header_text[:200])
            # Discard the header and try again
            remaining = buffer[header_end + 4 :]
            if remaining:
                return await self._process_buffer(remaining)
            return b""

        body_start = header_end + 4
        if len(buffer) < body_start + content_length:
            return buffer  # Wait for more data

        body = buffer[body_start : body_start + content_length]
        remaining = buffer[body_start + content_length :]

        try:
            message = json.loads(body.decode("utf-8"))
            await self._dispatch(message)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning("LSP: malformed message: %s", exc)

        if remaining:
            return await self._process_buffer(remaining)
        return b""

    @staticmethod
    def _parse_content_length(header: str) -> Optional[int]:
        for line in header.split("\r\n"):
            if line.lower().startswith("content-length:"):
                try:
                    return int(line.split(":", 1)[1].strip())
                except ValueError:
                    return None
        return None

    async def _dispatch(self, message: dict[str, Any]) -> None:
        if "id" in message and "method" not in message:
            # Response
            rid = message["id"]
            fut = self._pending.pop(rid, None)
            if fut and not fut.done():
                if "error" in message:
                    fut.set_exception(
                        LspClientError(
                            f"LSP error {message['error'].get('code', '?')}: "
                            f"{message['error'].get('message', 'unknown')}"
                        )
                    )
                else:
                    fut.set_result(message.get("result") or {})
        elif "method" in message:
            # Notification
            method = message["method"]
            params = message.get("params", {})
            for handler in self._notification_handlers.get(method, []):
                try:
                    handler(params)
                except Exception:
                    logger.exception("LSP notification handler error for %s", method)

    @staticmethod
    def _build_env() -> dict[str, str]:
        """Environment variables to pass to the LS subprocess."""
        import os

        env = {}
        # Inherit PATH but strip PYTHON* to avoid confusing Python-based servers
        path = os.environ.get("PATH", "")
        if path:
            env["PATH"] = path
        return env


# ---------------------------------------------------------------------------
# diagnostic helpers
# ---------------------------------------------------------------------------


def parse_publish_diagnostics(params: dict[str, Any]) -> PublishDiagnosticsParams:
    """Parse a ``textDocument/publishDiagnostics`` notification payload."""
    raw_diags = params.get("diagnostics", [])
    diagnostics = [
        LspDiagnostic(
            range=d.get("range", {}),
            severity=d.get("severity"),
            code=str(d.get("code")) if d.get("code") else None,
            message=d.get("message", ""),
            source=d.get("source"),
        )
        for d in raw_diags
    ]
    return PublishDiagnosticsParams(uri=params.get("uri", ""), diagnostics=diagnostics)
