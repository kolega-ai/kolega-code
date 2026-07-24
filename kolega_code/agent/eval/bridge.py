"""HTTP loopback bridge that lets eval kernels call back into agent tools.

Kernels are separate processes; from inside a cell, ``tool.<name>(args)`` POSTs
JSON to ``127.0.0.1`` on an ephemeral port with a per-process bearer token. The
bridge looks up the registration for the cell's ``(session, run)`` pair and
forwards the call to the executing agent's standard tool-execution path, so
bridge-originated calls get the same permission prompts, hooks, and TUI
visibility as model-issued tool calls.

Registrations live only for the duration of one cell; in-flight bridge calls
are cancelled when the owning cell is interrupted so nothing is orphaned.
"""

from __future__ import annotations

import asyncio
import json
import secrets
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

# Special bridge op (not a session tool): list the executing agent's registry.
LIST_TOOLS_OP = "__list_tools__"

_MAX_HEADER_BYTES = 64 * 1024
_MAX_BODY_BYTES = 64 * 1024 * 1024

# (session_id, run_id) -> JSON-safe value returned to the kernel.
ToolCaller = Callable[[str, Any], Awaitable[Any]]
# () -> [{"name": ..., "summary": ...}]
ToolLister = Callable[[], Any]


@dataclass
class BridgeRegistration:
    """Routing entry for one executing cell."""

    tool_caller: ToolCaller
    tool_lister: Optional[ToolLister] = None
    tasks: set[asyncio.Task] = field(default_factory=set)
    cancelled: bool = False

    def cancel_in_flight(self) -> None:
        """Reject new calls and cancel in-flight ones (cell was interrupted)."""
        self.cancelled = True
        for task in list(self.tasks):
            task.cancel()


class ToolBridge:
    """Asyncio loopback HTTP server dispatching kernel tool calls."""

    _instances: Dict[int, "ToolBridge"] = {}

    def __init__(self) -> None:
        self._token = secrets.token_hex(24)
        self._server: Optional[asyncio.AbstractServer] = None
        self._registrations: Dict[Tuple[str, str], BridgeRegistration] = {}
        self._port = 0
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    @property
    def token(self) -> str:
        return self._token

    @classmethod
    async def ensure(cls) -> "ToolBridge":
        """Return the bridge singleton for the current event loop, starting it lazily."""
        loop = asyncio.get_running_loop()
        key = id(loop)
        instance = cls._instances.get(key)
        if instance is None or instance._server is None:
            instance = cls()
            instance._loop = loop  # keep the loop alive for the id() key
            await instance._start()
            cls._instances[key] = instance
        return instance

    async def _start(self) -> None:
        self._server = await asyncio.start_server(self._handle_client, host="127.0.0.1", port=0)
        sock = self._server.sockets[0]
        self._port = int(sock.getsockname()[1])

    async def stop(self) -> None:
        """Stop the server and cancel every in-flight call. Mainly for tests."""
        for registration in self._registrations.values():
            registration.cancel_in_flight()
        self._registrations.clear()
        if self._server is not None:
            server = self._server
            self._server = None
            server.close()
            await server.wait_closed()

    def register(self, session_id: str, run_id: str, registration: BridgeRegistration) -> Callable[[], None]:
        """Route calls for (session_id, run_id) for the duration of one cell."""
        key = (session_id, run_id)
        self._registrations[key] = registration

        def unregister() -> None:
            if self._registrations.get(key) is registration:
                registration.cancel_in_flight()
                del self._registrations[key]

        return unregister

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            status, payload = await self._dispatch_request(reader)
        except Exception as exc:  # never leak a stack into a kernel response
            status, payload = 500, {"ok": False, "error": f"bridge error: {exc}"}
        try:
            body = json.dumps(payload).encode("utf-8", "replace")
            reason = {200: "OK", 400: "Bad Request", 403: "Forbidden", 404: "Not Found", 413: "Too Large"}.get(
                status, "Error"
            )
            head = (
                f"HTTP/1.1 {status} {reason}\r\n"
                "Content-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\n"
                "Connection: close\r\n\r\n"
            )
            writer.write(head.encode("ascii") + body)
            await writer.drain()
        except (ConnectionError, RuntimeError):
            pass
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def _dispatch_request(self, reader: asyncio.StreamReader) -> Tuple[int, Dict[str, Any]]:
        request_line = await reader.readline()
        if len(request_line) > _MAX_HEADER_BYTES:
            return 400, {"ok": False, "error": "request line too long"}
        try:
            method, target, _version = request_line.decode("ascii", "replace").strip().split(" ", 2)
        except ValueError:
            return 400, {"ok": False, "error": "malformed request line"}

        headers: Dict[str, str] = {}
        header_bytes = len(request_line)
        while True:
            line = await reader.readline()
            header_bytes += len(line)
            if header_bytes > _MAX_HEADER_BYTES:
                return 400, {"ok": False, "error": "headers too large"}
            if line in (b"\r\n", b"\n", b""):
                break
            name, sep, value = line.decode("ascii", "replace").partition(":")
            if sep:
                headers[name.strip().lower()] = value.strip()

        if method != "POST" or target != "/v1/tool":
            return 404, {"ok": False, "error": "not found"}
        if headers.get("authorization") != f"Bearer {self._token}":
            return 403, {"ok": False, "error": "forbidden"}

        try:
            length = int(headers.get("content-length") or "0")
        except ValueError:
            return 400, {"ok": False, "error": "invalid content-length"}
        if length <= 0:
            return 400, {"ok": False, "error": "missing body"}
        if length > _MAX_BODY_BYTES:
            return 413, {"ok": False, "error": "body too large"}
        raw = await reader.readexactly(length)
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            return 400, {"ok": False, "error": "invalid JSON body"}
        if not isinstance(body, dict):
            return 400, {"ok": False, "error": "body must be a JSON object"}

        session_id = body.get("session")
        run_id = body.get("run")
        name = body.get("name")
        if not isinstance(session_id, str) or not isinstance(run_id, str) or not isinstance(name, str) or not name:
            return 400, {"ok": False, "error": "missing session/run/name"}
        args = body.get("args")

        registration = self._registrations.get((session_id, run_id))
        if registration is None:
            return 200, {"ok": False, "error": f"no active eval cell for session/run {session_id}/{run_id}"}
        if registration.cancelled:
            return 200, {"ok": False, "error": f"bridge call {name!r} rejected: eval cell was interrupted"}

        if name == LIST_TOOLS_OP:
            try:
                tools = registration.tool_lister() if registration.tool_lister else []
                return 200, {"ok": True, "value": tools}
            except Exception as exc:
                return 200, {"ok": False, "error": str(exc)}

        task = asyncio.current_task()
        if task is not None:
            registration.tasks.add(task)
        try:
            value = await registration.tool_caller(name, args)
            return 200, {"ok": True, "value": value}
        except asyncio.CancelledError:
            return 200, {"ok": False, "error": f"bridge call {name!r} cancelled: eval cell was interrupted"}
        except Exception as exc:
            return 200, {"ok": False, "error": str(exc) or exc.__class__.__name__}
        finally:
            if task is not None:
                registration.tasks.discard(task)


async def ensure_tool_bridge() -> ToolBridge:
    """Module-level convenience matching ToolBridge.ensure()."""
    return await ToolBridge.ensure()
