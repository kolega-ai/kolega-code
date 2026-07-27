"""Run the session server inside a host process.

Sharing a session should not mean opening a second terminal, finding the session
id, and constructing a URL by hand. This runs the session ASGI app as a task on
the caller's event loop and hands back the link.

Three things matter for a link that is meant to be given away:

* **A token is always minted.** ``create_app`` allows no token at all, which is
  fine for a server aimed at the machine's owner. A share link exists to be
  handed to someone else, so it must carry proof of access even before anyone
  decides to expose it beyond loopback.
* **The link is scoped to one session.** The token gates routes, not sessions,
  so an unscoped server would let the person you shared one session with read
  every session in the store. Pass ``session_id`` and everything else 404s.
* **Signal handlers are left alone.** ``uvicorn.Server.serve`` installs its own
  SIGINT/SIGTERM handlers, which would take the host's Ctrl-C out from under it.

What a link exposes is *not* redacted. An exported bundle runs every string
through :mod:`kolega_code.web.redaction`; this serves the recorded events as
they are, so anything the agent printed — credentials included — is visible to
whoever holds the link.
"""

from __future__ import annotations

import asyncio
import contextlib
import secrets
import socket
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from kolega_code.cli.session_store import SessionStore

# ``.server`` pulls in FastAPI and ``.session_store`` pulls in the CLI, so both
# are imported where they are used. This module's constants are read while the
# argument parser is being built, on every invocation of every subcommand.

#: How long to wait for uvicorn to bind before giving up.
_STARTUP_TIMEOUT_S = 10.0
_STARTUP_POLL_S = 0.02

LOOPBACK = "127.0.0.1"
#: Bind address that accepts connections from other machines.
ALL_INTERFACES = "0.0.0.0"
#: The port both ``kolega-code serve`` and ``/share`` use unless told otherwise.
#: Sharing through a tunnel means forwarding a port, and a forwarding rule is
#: worth writing down only if the port is the same next time.
DEFAULT_PORT = 8765
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
        store: "SessionStore",
        *,
        bind: str = LOOPBACK,
        port: int = 0,
        token: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self._store = store
        self._bind = bind
        self._requested_port = port
        self._token = token or secrets.token_urlsafe(16)
        # None keeps every session reachable, which only makes sense for a host
        # serving its own. A share link should always name its session.
        self._session_ids = None if session_id is None else frozenset({session_id})
        self._server: Any = None
        self._task: Optional[asyncio.Task[Any]] = None
        self._handle: Optional[ShareHandle] = None
        self._socket: Optional[socket.socket] = None

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

        from .server import ServerConfig, create_app

        app = create_app(ServerConfig(store=self._store, token=self._token, session_ids=self._session_ids))
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
        # Bind before handing the socket over. Left to itself uvicorn binds during
        # startup and reports failure with sys.exit, and a SystemExit raised
        # inside a task is re-raised into the event loop -- taking the host down
        # over a port collision. Binding here turns that into an ordinary error.
        self._socket = self._bind_socket()
        self._server = _build_server(config)
        self._task = asyncio.create_task(self._server.serve(sockets=[self._socket]))

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

    def _bind_socket(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((self._bind, self._requested_port))
        except OSError as exc:
            sock.close()
            raise ShareServerError(
                f"could not bind {self._bind}:{self._requested_port} ({exc.strerror or exc})"
            ) from exc
        return sock

    def request_stop(self) -> None:
        """Ask the server to stop without awaiting it.

        For teardown paths that are not async, such as Textual's ``on_unmount``.
        The socket closes when the task next runs; the process exiting closes it
        regardless.
        """
        server, task, sock = self._server, self._task, self._socket
        self._server = self._task = self._handle = self._socket = None
        if server is not None:
            server.should_exit = True
            # Close the listeners now rather than waiting for the task to notice.
            # Cancellation alone leaves the port bound until the coroutine next
            # runs, and on a teardown path it may never get another turn.
            for listener in getattr(server, "servers", None) or []:
                with contextlib.suppress(Exception):
                    listener.close()
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.close()
        if task is not None and not task.done():
            task.cancel()

    async def stop(self) -> None:
        """Stop serving and wait for the task to unwind."""
        server, task, sock = self._server, self._task, self._socket
        self._server = self._task = self._handle = self._socket = None
        if server is not None:
            server.should_exit = True
        if task is not None:
            # SystemExit from a failed bind is a BaseException, so suppressing
            # Exception alone would let teardown raise.
            with contextlib.suppress(asyncio.CancelledError, SystemExit, Exception):
                await asyncio.wait_for(asyncio.shield(task), timeout=5)
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, SystemExit, Exception):
                    await task
        if sock is not None:
            # uvicorn closes the sockets it was handed, but not if it never got
            # as far as serving them.
            with contextlib.suppress(OSError):
                sock.close()

    async def _await_started(self) -> None:
        waited = 0.0
        while waited < _STARTUP_TIMEOUT_S:
            task = self._task
            if task is not None and task.done():
                # Returning this early means the bind failed. uvicorn reports that
                # by calling sys.exit, so the failure arrives as SystemExit --
                # which is not an Exception and would sail past every caller's
                # handler and out of the host's command loop.
                try:
                    task.result()
                except (SystemExit, OSError) as exc:
                    raise ShareServerError(
                        f"could not bind {self._bind}:{self._requested_port} (is it already in use?)"
                    ) from exc
                raise ShareServerError("the share server stopped before it began serving")
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
        for sock in ([self._socket] if self._socket is not None else []) + self._sockets():
            with contextlib.suppress(OSError):
                return int(sock.getsockname()[1])
        raise ShareServerError("the share server is not bound to a port")
