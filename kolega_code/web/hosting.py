"""Run the session server inside a host process.

``kolega-code serve`` owns its process and blocks, which is right for a terminal
but useless to a running TUI: sharing a session should not mean opening a second
terminal, finding the session id, and constructing a URL by hand. This runs the
same ASGI app as a task on the caller's event loop, on a port the OS picks, and
hands back the link.

Two things differ from the standalone command:

* **A token is always minted.** The standalone server defaults to loopback with
  no token because it is aimed at the machine's owner. A share link exists to be
  given away, so it must carry proof of access even before anyone decides to
  expose it beyond loopback.
* **Signal handlers are left alone.** ``uvicorn.Server.serve`` installs its own
  SIGINT/SIGTERM handlers, which would take the host's Ctrl-C out from under it.
"""

from __future__ import annotations

import asyncio
import contextlib
import secrets
import socket
from dataclasses import dataclass
from typing import Any, Optional

from kolega_code.cli.session_store import SessionStore

from .server import ServerConfig, create_app

#: How long to wait for uvicorn to bind before giving up.
_STARTUP_TIMEOUT_S = 10.0
_STARTUP_POLL_S = 0.02

LOOPBACK = "127.0.0.1"
#: Bind address that accepts connections from other machines.
ALL_INTERFACES = "0.0.0.0"
#: uvicorn's default "auto" still selects the deprecated legacy websockets
#: implementation, which warns on import. Naming the sans-io one keeps a host
#: process's log clean and is the implementation uvicorn is moving to.
WS_IMPL = "websockets-sansio"


class ShareServerError(RuntimeError):
    """Raised when the share server cannot be started."""


@dataclass(frozen=True)
class ShareHandle:
    """Where a running share server is reachable."""

    url: str
    host: str
    port: int
    token: str
    #: True when bound beyond loopback, so callers can warn appropriately.
    exposed: bool


def local_network_address() -> Optional[str]:
    """Best guess at this machine's address on its local network.

    Opens a UDP socket toward a public address and reads back the local end. No
    packet is sent and nothing needs to be reachable; this is only a way to ask
    the routing table which interface would be used.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 53))
        address = probe.getsockname()[0]
    except OSError:
        return None
    finally:
        probe.close()
    return str(address) if address and not str(address).startswith("127.") else None


def _build_server(config: Any) -> Any:
    """A uvicorn server that leaves the host process's signal handlers alone.

    ``Server.serve`` wraps itself in ``capture_signals``, which points
    SIGINT/SIGTERM at uvicorn's own shutdown. This server is a guest on someone
    else's process, so Ctrl-C has to keep reaching the host. The subclass is
    defined lazily to keep uvicorn off the CLI's import path.

    uvicorn has renamed this hook before (it was ``install_signal_handlers``
    until 0.29), so ``test_does_not_touch_the_host_signal_handlers`` asserts the
    outcome rather than trusting the override to still apply.
    """
    import uvicorn

    class _GuestServer(uvicorn.Server):
        @contextlib.contextmanager
        def capture_signals(self):
            yield

    return _GuestServer(config)


class ShareServer:
    """The session server, running on the caller's event loop."""

    def __init__(
        self,
        store: SessionStore,
        *,
        bind: str = LOOPBACK,
        port: int = 0,
        token: Optional[str] = None,
    ) -> None:
        self._store = store
        self._bind = bind
        self._requested_port = port
        self._token = token or secrets.token_urlsafe(16)
        self._server: Any = None
        self._task: Optional[asyncio.Task[Any]] = None
        self._handle: Optional[ShareHandle] = None

    @property
    def handle(self) -> Optional[ShareHandle]:
        return self._handle

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def exposed(self) -> bool:
        """Whether this is bound beyond loopback. Known before it starts."""
        return self._bind != LOOPBACK

    async def start(self) -> ShareHandle:
        """Bind and begin serving, returning once the port is accepting."""
        if self.running and self._handle is not None:
            return self._handle

        import uvicorn

        app = create_app(ServerConfig(store=self._store, token=self._token))
        config = uvicorn.Config(
            app,
            host=self._bind,
            port=self._requested_port,
            log_level="warning",
            ws=WS_IMPL,
            # Nothing here uses ASGI lifespan, and leaving it on makes startup
            # failures surface as a hang rather than an error.
            lifespan="off",
        )
        self._server = _build_server(config)
        self._task = asyncio.create_task(self._server.serve())

        try:
            await self._await_started()
        except Exception:
            await self.stop()
            raise

        host = self._bind if self._bind != ALL_INTERFACES else (local_network_address() or LOOPBACK)
        self._handle = ShareHandle(
            url=f"http://{host}:{self._bound_port()}",
            host=host,
            port=self._bound_port(),
            token=self._token,
            exposed=self._bind != LOOPBACK,
        )
        return self._handle

    def session_url(self, session_id: str) -> str:
        """Player link for one session, carrying the access token."""
        if self._handle is None:
            raise ShareServerError("The share server is not running")
        return f"{self._handle.url}/s/{session_id}?token={self._handle.token}"

    def request_stop(self) -> None:
        """Ask the server to stop without awaiting it.

        For teardown paths that are not async, such as Textual's ``on_unmount``.
        The socket closes when the task next runs; the process exiting closes it
        regardless.
        """
        server, task = self._server, self._task
        self._server = self._task = self._handle = None
        if server is not None:
            server.should_exit = True
            # Close the listeners now rather than waiting for the task to notice.
            # Cancellation alone leaves the port bound until the coroutine next
            # runs, and on a teardown path it may never get another turn.
            for listener in getattr(server, "servers", None) or []:
                with contextlib.suppress(Exception):
                    listener.close()
        if task is not None and not task.done():
            task.cancel()

    async def stop(self) -> None:
        """Stop serving and wait for the task to unwind."""
        server, task = self._server, self._task
        self._server = self._task = self._handle = None
        if server is not None:
            server.should_exit = True
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(asyncio.shield(task), timeout=5)
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

    async def _await_started(self) -> None:
        waited = 0.0
        while waited < _STARTUP_TIMEOUT_S:
            task = self._task
            if task is not None and task.done():
                # serve() returning this early means the bind failed; re-raise it.
                task.result()
                raise ShareServerError("The share server stopped before it began serving")
            if getattr(self._server, "started", False) and self._sockets():
                return
            await asyncio.sleep(_STARTUP_POLL_S)
            waited += _STARTUP_POLL_S
        raise ShareServerError(f"The share server did not start within {_STARTUP_TIMEOUT_S:.0f}s")

    def _sockets(self) -> list[Any]:
        servers = getattr(self._server, "servers", None) or []
        return [sock for entry in servers for sock in (getattr(entry, "sockets", None) or [])]

    def _bound_port(self) -> int:
        """The port actually bound, which is what port 0 was asked to discover."""
        for sock in self._sockets():
            with contextlib.suppress(OSError):
                return int(sock.getsockname()[1])
        raise ShareServerError("The share server is not bound to a port")
