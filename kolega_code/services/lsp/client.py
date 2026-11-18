"""Minimal async JSON-RPC 2.0 client for communicating with language servers over stdio.

Uses asyncio subprocess + Content-Length-prefixed message framing (LSP transport).
No external dependencies.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_SEND_TIMEOUT_SECONDS = 30.0
_TERMINATE_TIMEOUT_SECONDS = 3.0
_KILL_TIMEOUT_SECONDS = 1.0


async def _settle(task: asyncio.Task[None], *, cancelled: bool = False) -> None:
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            break
    if cancelled:
        if not task.cancelled():
            task.exception()
        raise asyncio.CancelledError
    task.result()


# ---------------------------------------------------------------------------
# subprocess environment policy
# ---------------------------------------------------------------------------

# Minimal allowlist of environment variables forwarded to language-server
# subprocesses. Everything else (including provider API keys / OAuth tokens) is
# withheld so a compromised or untrusted server cannot exfiltrate secrets.
_ENV_ALLOWLIST = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TERM",
        "SHELL",
        "TMPDIR",
        "TMP",
        "TEMP",
        "SYSTEMROOT",
        "COMSPEC",
        "APPDATA",
        "LOCALAPPDATA",
        "PROGRAMFILES",
        "USERPROFILE",
        "WINDIR",
        "PATHEXT",
    }
)

# Loader-injection variables that are stripped even from a server's declared
# ``env`` so a custom_servers entry cannot hijack a legitimate binary.
_ENV_DENYLIST = frozenset(
    {
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "PYTHONPATH",
        "PYTHONHOME",
        "NODE_OPTIONS",
        "NODE_PATH",
    }
)

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
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._notification_handlers: dict[str, list[Callable[[dict[str, Any]], None]]] = {}
        self._request_handlers: dict[str, Callable[[dict[str, Any]], Any]] = {}
        self._reader_task: Optional[asyncio.Task[None]] = None
        self._start_task: Optional[asyncio.Task[None]] = None
        self._stop_task: Optional[asyncio.Task[None]] = None
        self._lock = asyncio.Lock()
        self._running = False

        # -- state tracking ---------------------------------------
        self.server_capabilities: Optional[dict[str, Any]] = None
        self.server_pid: Optional[int] = None
        self.last_error: Optional[str] = None
        self.status: str = "stopped"
        self.active_root: Optional[str] = None

    # -- lifecycle ----------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """Launch the language server subprocess and begin reading messages."""
        while self._stop_task is not None:
            cleanup = self._stop_task
            await _settle(cleanup)
            if self._stop_task is cleanup:
                self._stop_task = None
        if self._running:
            return
        if self._start_task is None or self._start_task.done():
            self._start_task = asyncio.create_task(self._start_process())
        try:
            await _settle(self._start_task)
        except asyncio.CancelledError:
            await _settle(self._begin_stop(), cancelled=True)
            raise

    async def _start_process(self) -> None:
        env = self._subprocess_env()
        self._proc = await asyncio.create_subprocess_exec(
            *self._command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=env or None,
        )
        self._running = self._stop_task is None
        self.server_pid = self._proc.pid
        self.status = "starting"
        self._reader_task = asyncio.create_task(self._read_loop())
        logger.debug("LSP client started: %s (pid=%s)", self._command, self.server_pid)

    async def stop(self) -> None:
        await _settle(self._begin_stop())

    def _begin_stop(self, error: Optional[str] = None) -> asyncio.Task[None]:
        self._running = False
        if error is not None:
            self.last_error = error
            self.status = "error"
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(LspClientError(error or "Language server stopped"))
        self._pending.clear()
        if self._stop_task is None or (self._stop_task.done() and self._proc is not None):
            self._stop_task = asyncio.create_task(self._stop_process())
            self._stop_task.add_done_callback(self._report_stop_failure)
        return self._stop_task

    @staticmethod
    def _report_stop_failure(task: asyncio.Task[None]) -> None:
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.error("LSP cleanup failed: %s", error)

    async def _stop_process(self) -> None:
        if self._start_task is not None:
            try:
                await _settle(self._start_task)
            except Exception:
                pass
        self._running = False
        if self._reader_task is not None:
            self._reader_task.cancel()
            await asyncio.gather(self._reader_task, return_exceptions=True)

        proc = self._proc
        if proc is not None:
            drain = asyncio.create_task(self._drain_stdout(proc))
            transport: asyncio.SubprocessTransport = getattr(proc, "_transport")
            try:
                if proc.returncode is None:
                    try:
                        proc.terminate()
                    except ProcessLookupError:
                        pass
                try:
                    await asyncio.wait_for(proc.wait(), timeout=_TERMINATE_TIMEOUT_SECONDS)
                except asyncio.TimeoutError:
                    if proc.returncode is None:
                        try:
                            proc.kill()
                        except ProcessLookupError:
                            pass
                    # transport.close() polls the child, which can reap the SIGKILL
                    # status before asyncio's watcher and report 255 instead.
                    await asyncio.wait_for(proc.wait(), timeout=_KILL_TIMEOUT_SECONDS)
                    transport.close()
            except Exception as exc:
                self.status = "error"
                self.last_error = f"Language server cleanup failed: {exc}"
                raise LspClientError(self.last_error) from exc
            finally:
                transport.close()
                drain.cancel()
                await asyncio.gather(drain, return_exceptions=True)
        self._proc = None
        self._reader_task = None
        self.status = "stopped"
        logger.debug("LSP client stopped: %s", self._command)

    @staticmethod
    async def _drain_stdout(proc: asyncio.subprocess.Process) -> None:
        if proc.stdout is not None:
            while await proc.stdout.read(65536):
                pass

    # -- request / notification --------------------------------------------

    async def request(self, method: str, params: Any = None, *, timeout: float = 30) -> Any:
        """Send a JSON-RPC request and await the response.

        Args:
            method: LSP method name.
            params: Request parameters (sent as ``params``).
            timeout: Maximum seconds for lock acquisition, send, and response (default 30).
        """
        if not self._running:
            raise LspClientError("Client not running")
        rid: Optional[int] = None
        fut: Optional[asyncio.Future[Any]] = None
        try:
            async with asyncio.timeout(timeout):
                async with self._lock:
                    if not self._running:
                        raise LspClientError("Client not running")
                    rid = self._request_id = self._request_id + 1
                    payload = {
                        "jsonrpc": "2.0",
                        "id": rid,
                        "method": method,
                        "params": params if params is not None else {},
                    }
                    fut = asyncio.get_running_loop().create_future()
                    self._pending[rid] = fut
                    await self._send(payload)
                return await fut
        except asyncio.TimeoutError:
            raise LspClientError(f"Request '{method}' timed out after {timeout}s") from None
        finally:
            if rid is not None:
                self._pending.pop(rid, None)
            if fut is not None:
                if not fut.done():
                    fut.cancel()
                elif not fut.cancelled():
                    fut.exception()

    async def notify(self, method: str, params: Any = None, *, timeout: float = 30) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if not self._running:
            raise LspClientError("Client not running")
        payload = {"jsonrpc": "2.0", "method": method, "params": params if params is not None else {}}
        try:
            async with asyncio.timeout(timeout):
                await self._send(payload)
        except asyncio.TimeoutError:
            raise LspClientError(f"Notification '{method}' timed out after {timeout}s") from None

    def on_notification(self, method: str, handler: Callable[[dict[str, Any]], None]) -> None:
        """Register a handler for server→client notifications."""
        self._notification_handlers.setdefault(method, []).append(handler)

    def on_request(self, method: str, handler: Callable[[dict[str, Any]], Any]) -> None:
        """Register a handler for server→client requests.

        The handler receives the request ``params`` and must return a JSON-serialisable
        result (or raise, which sends an error response).  Only one handler per method
        is supported (last registration wins).
        """
        self._request_handlers[method] = handler

    # -- internals ----------------------------------------------------------

    async def _send(self, payload: dict[str, Any]) -> None:
        proc = self._proc
        if not self._running or proc is None or proc.stdin is None:
            raise LspClientError("Client not running")
        body = json.dumps(payload)
        header = f"Content-Length: {len(body)}\r\n\r\n"
        try:
            async with asyncio.timeout(_SEND_TIMEOUT_SECONDS):
                proc.stdin.write((header + body).encode("utf-8"))
                await proc.stdin.drain()
        except (asyncio.CancelledError, OSError, asyncio.TimeoutError) as exc:
            error = "LSP send interrupted" if isinstance(exc, asyncio.CancelledError) else f"LSP send failed: {exc}"
            cleanup = self._begin_stop(error if self._running else None)
            if asyncio.current_task() is not self._reader_task:
                await _settle(cleanup, cancelled=isinstance(exc, asyncio.CancelledError))
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise LspClientError(error) from exc

    async def _send_response(self, rid: Any, result: Any = None, error: Optional[dict[str, Any]] = None) -> None:
        """Send a JSON-RPC response to a server→client request."""
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": rid}
        if error is not None:
            payload["error"] = error
        else:
            payload["result"] = result
        await self._send(payload)

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
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self._begin_stop(f"LSP reader failed: {exc}")
            logger.exception("LSP reader loop error")
        finally:
            if self._running:
                self._begin_stop("Language server closed stdout")

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
        has_id = "id" in message
        has_method = "method" in message

        if has_id and not has_method:
            # Response to our request
            rid = message["id"]
            fut = self._pending.pop(rid, None)
            if fut and not fut.done():
                if "error" in message:
                    err = message["error"]
                    fut.set_exception(
                        LspClientError(f"LSP error {err.get('code', '?')}: {err.get('message', 'unknown')}")
                    )
                else:
                    fut.set_result(message["result"] if "result" in message else None)
            # If fut is None or done: late-arriving response to a timed-out/cancelled
            # request — silently drop it.
        elif has_id and has_method:
            # Server→client request — must send a response back
            rid = message["id"]
            method = message["method"]
            params = message.get("params", {})
            handler = self._request_handlers.get(method)
            if handler is not None:
                try:
                    result = handler(params)
                except Exception as exc:
                    logger.exception("LSP request handler error for %s", method)
                    await self._send_response(
                        rid,
                        error={"code": -32603, "message": f"Internal error: {exc}"},
                    )
                else:
                    await self._send_response(rid, result=result)
            else:
                # Method not found — respond so the server doesn't hang
                await self._send_response(
                    rid,
                    error={"code": -32601, "message": f"Method not found: {method}"},
                )
        elif has_method:
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
        """Return the allowlisted base environment for an LS subprocess.

        Only a minimal set of safe variables (PATH, HOME, ...) is forwarded; all
        other variables -- including provider API keys and OAuth tokens -- are
        withheld so a compromised or untrusted server cannot exfiltrate secrets.
        """
        return {key: value for key, value in os.environ.items() if key in _ENV_ALLOWLIST}

    def _subprocess_env(self) -> dict[str, str]:
        """Build the final subprocess env.

        Starts from the allowlisted base, then applies the server's declared
        ``env`` overrides -- but always strips loader-injection variables
        (``LD_PRELOAD``, ``PYTHONPATH``, ``NODE_OPTIONS``, ...) so a
        ``custom_servers`` entry cannot hijack a legitimate binary.
        """
        env = self._build_env()
        for key, value in (self._env or {}).items():
            if key.upper() in _ENV_DENYLIST:
                continue
            env[key] = value
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
