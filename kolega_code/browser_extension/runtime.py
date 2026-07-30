"""Authenticated local channels over owner-private macOS Unix sockets."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import inspect
import os
import secrets
import stat
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from .framing import MAX_NATIVE_INBOUND_MESSAGE_BYTES, FramingError, read_message_async, write_message_async
from .protocol import PROTOCOL_VERSION, Envelope, MessageDirection
from .registry import (
    RuntimeDescriptor,
    RuntimeDescriptorRegistry,
    RuntimeTransport,
    validate_extension_origin,
    validate_identifier,
)

if TYPE_CHECKING:
    from .multiplex import EventHandler, MultiplexedPeer, RequestHandler

AUTH_TIMEOUT_SECONDS = 5.0
RUNTIME_WRITE_TIMEOUT_SECONDS = 5.0
DEFAULT_RUNTIME_TTL_MS = 12 * 60 * 60 * 1000
_AUTH_KEYS = frozenset({"kind", "protocol_version", "runtime_id", "token", "extension_origin"})
_AUTH_OK_KEYS = frozenset({"kind", "protocol_version", "runtime_id"})


class RuntimeTransportError(RuntimeError):
    """A safe-to-report local transport failure."""


class UnsupportedRuntimeTransportError(RuntimeTransportError):
    """The selected secure platform transport cannot execute."""


class RuntimeAuthenticationError(RuntimeTransportError):
    """Runtime-channel authentication failed."""


class RuntimeDisconnectedError(RuntimeTransportError):
    """An authenticated runtime channel disconnected."""


def selected_runtime_transport(*, platform: str | None = None) -> RuntimeTransport:
    """Select the macOS transport without a network-socket fallback."""
    if (platform or sys.platform) != "darwin":
        raise UnsupportedRuntimeTransportError("Chrome browser integration is supported only on macOS")
    return "unix"


def ensure_runtime_transport_supported(*, platform: str | None = None) -> RuntimeTransport:
    """Verify that the selected transport can execute in this interpreter."""
    transport = selected_runtime_transport(platform=platform)
    if not hasattr(asyncio, "start_unix_server"):
        raise UnsupportedRuntimeTransportError("Unix-domain-socket runtime transport is unavailable")
    return transport


class _Listener(Protocol):
    def close(self) -> None: ...

    async def wait_closed(self) -> None: ...


class RuntimeChannel:
    """A framed, authenticated protocol-envelope channel."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        runtime_id: str,
        session_id: str,
        outbound_direction: MessageDirection,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self.runtime_id = runtime_id
        self.session_id = session_id
        self.outbound_direction = outbound_direction
        self._write_lock = asyncio.Lock()
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed or self._writer.is_closing()

    async def read(self) -> Envelope | None:
        if self._closed:
            return None
        # This is our own local socket, not Chrome's stdio, so Chrome's 1 MB
        # message limit does not apply to it. Screenshot responses cross here.
        raw = await read_message_async(self._reader, max_bytes=MAX_NATIVE_INBOUND_MESSAGE_BYTES)
        if raw is None:
            return None
        envelope = Envelope.from_mapping(raw)
        if envelope.runtime_id != self.runtime_id or envelope.session_id != self.session_id:
            raise RuntimeTransportError("Runtime channel received a mismatched envelope")
        if envelope.direction is not self.outbound_direction.opposite:
            raise RuntimeTransportError("Runtime channel received an invalid direction")
        return envelope

    async def write(self, envelope: Envelope) -> None:
        if self.closed:
            raise RuntimeDisconnectedError("Runtime channel is disconnected")
        if envelope.runtime_id != self.runtime_id or envelope.session_id != self.session_id:
            raise RuntimeTransportError("Runtime channel cannot send a mismatched envelope")
        if envelope.direction is not self.outbound_direction:
            raise RuntimeTransportError("Runtime channel cannot send an invalid direction")
        try:
            async with asyncio.timeout(RUNTIME_WRITE_TIMEOUT_SECONDS):
                async with self._write_lock:
                    if self.closed:
                        raise RuntimeDisconnectedError("Runtime channel is disconnected")
                    await write_message_async(
                        self._writer,
                        envelope.to_mapping(),
                        max_bytes=MAX_NATIVE_INBOUND_MESSAGE_BYTES,
                    )
        except (TimeoutError, ConnectionError, BrokenPipeError, FramingError):
            raise RuntimeDisconnectedError("Runtime channel is disconnected") from None

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._writer.close()
        with contextlib.suppress(ConnectionError, OSError):
            await self._writer.wait_closed()


def _validate_auth_request(raw: object) -> tuple[str, str, str]:
    if not isinstance(raw, dict) or set(raw) != _AUTH_KEYS:
        raise RuntimeAuthenticationError("Runtime authentication failed")
    if raw["kind"] != "authenticate" or raw["protocol_version"] != PROTOCOL_VERSION:
        raise RuntimeAuthenticationError("Runtime authentication failed")
    try:
        runtime_id = validate_identifier(raw["runtime_id"], "runtime ID")
        origin = validate_extension_origin(raw["extension_origin"])
    except Exception:
        raise RuntimeAuthenticationError("Runtime authentication failed") from None
    token = raw["token"]
    if not isinstance(token, str):
        raise RuntimeAuthenticationError("Runtime authentication failed")
    return runtime_id, token, origin


async def _open_local_connection(
    descriptor: RuntimeDescriptor,
    *,
    platform: str | None = None,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    selected = selected_runtime_transport(platform=platform)
    if descriptor.transport != selected:
        raise UnsupportedRuntimeTransportError("Runtime descriptor uses a different platform transport")
    try:
        return await asyncio.open_unix_connection(descriptor.endpoint)
    except OSError:
        raise RuntimeDisconnectedError("Runtime endpoint is unavailable") from None


async def connect_runtime_channel(
    descriptor: RuntimeDescriptor,
    *,
    extension_origin: str,
    timeout: float = AUTH_TIMEOUT_SECONDS,
    platform: str | None = None,
) -> RuntimeChannel:
    """Connect and authenticate to one runtime as the native-host side."""
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    ensure_runtime_transport_supported(platform=platform)
    origin = validate_extension_origin(extension_origin)
    if not descriptor.accepts_origin(origin):
        raise RuntimeAuthenticationError("Runtime authentication failed")
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    try:
        async with asyncio.timeout_at(deadline):
            reader, writer = await _open_local_connection(descriptor, platform=platform)
    except TimeoutError:
        raise RuntimeDisconnectedError("Runtime endpoint is unavailable") from None
    try:
        async with asyncio.timeout_at(deadline):
            await write_message_async(
                writer,
                {
                    "kind": "authenticate",
                    "protocol_version": PROTOCOL_VERSION,
                    "runtime_id": descriptor.runtime_id,
                    "token": descriptor.token,
                    "extension_origin": origin,
                },
            )
            response = await read_message_async(reader)
        if (
            not isinstance(response, dict)
            or set(response) != _AUTH_OK_KEYS
            or response["kind"] != "authenticated"
            or response["protocol_version"] != PROTOCOL_VERSION
            or response["runtime_id"] != descriptor.runtime_id
        ):
            raise RuntimeAuthenticationError("Runtime authentication failed")
    except (TimeoutError, ConnectionError, FramingError, RuntimeAuthenticationError):
        writer.close()
        with contextlib.suppress(ConnectionError, OSError):
            await writer.wait_closed()
        raise RuntimeAuthenticationError("Runtime authentication failed") from None
    return RuntimeChannel(
        reader,
        writer,
        runtime_id=descriptor.runtime_id,
        session_id=descriptor.session_id,
        outbound_direction=MessageDirection.EXTENSION_TO_RUNTIME,
    )


PeerCallback = Callable[["MultiplexedPeer"], Awaitable[None] | None]


class RuntimeServer:
    """Authenticated per-runtime server selected for the current platform."""

    def __init__(
        self,
        registry: RuntimeDescriptorRegistry,
        *,
        session_id: str,
        extension_origin: str,
        request_handler: RequestHandler | None = None,
        event_handler: EventHandler | None = None,
        on_peer: PeerCallback | None = None,
        runtime_id: str | None = None,
        ttl_ms: int = DEFAULT_RUNTIME_TTL_MS,
        max_pending_requests: int = 128,
        max_inflight_requests: int = 32,
        platform: str | None = None,
    ) -> None:
        self.transport: RuntimeTransport = ensure_runtime_transport_supported(platform=platform)
        if ttl_ms <= 0:
            raise ValueError("ttl_ms must be positive")
        self.registry = registry
        self.platform = platform
        self.session_id = validate_identifier(session_id, "session ID")
        self.extension_origin = validate_extension_origin(extension_origin)
        self.runtime_id = validate_identifier(
            runtime_id or f"runtime_{secrets.token_urlsafe(18)}",
            "runtime ID",
        )
        self.token = secrets.token_urlsafe(32)
        self.ttl_ms = ttl_ms
        self.request_handler = request_handler
        self.event_handler = event_handler
        self.on_peer = on_peer
        self.max_pending_requests = max_pending_requests
        self.max_inflight_requests = max_inflight_requests
        self.endpoint = self._select_endpoint()
        self._listener: _Listener | None = None
        self._descriptor: RuntimeDescriptor | None = None
        self._socket_identity: tuple[int, int] | None = None
        self._peers: set[MultiplexedPeer] = set()
        self._peer_queue: asyncio.Queue[MultiplexedPeer] = asyncio.Queue(maxsize=8)
        self._refresh_task: asyncio.Task[None] | None = None
        self._close_lock = asyncio.Lock()
        self._closed = False
        self._published = False

    def _select_endpoint(self) -> str:
        preferred = self.registry.root / f"{self.runtime_id}.sock"
        if len(os.fsencode(preferred)) <= 100:
            return str(preferred)
        getuid = getattr(os, "getuid", None)
        if getuid is None:
            raise UnsupportedRuntimeTransportError("Owner-private Unix runtime endpoints are unavailable")
        short_root = Path("/tmp") / f"kb-{getuid()}"
        try:
            short_root.mkdir(mode=0o700, exist_ok=False)
        except FileExistsError:
            pass
        root_stat = short_root.lstat()
        if (
            not stat.S_ISDIR(root_stat.st_mode)
            or root_stat.st_uid != getuid()
            or stat.S_IMODE(root_stat.st_mode) & 0o077
        ):
            raise RuntimeTransportError("Short runtime socket directory is unsafe")
        digest = hashlib.sha256(f"{self.registry.root}\0{self.runtime_id}".encode()).hexdigest()[:16]
        return str(short_root / f"{digest}.sock")

    @property
    def descriptor(self) -> RuntimeDescriptor:
        if self._descriptor is None:
            raise RuntimeTransportError("Runtime server has not started")
        return self._descriptor

    @property
    def published(self) -> bool:
        """Whether this runtime is currently advertised to the extension picker."""
        return self._published

    def publish(self) -> None:
        """Advertise this runtime, meaning "this session wants the browser now".

        An advertisement is a claim on the browser, not a fact about the process:
        the extension will not guess between several claims, so a session that has
        finished browsing must stop making one. Re-advertising is also how a session
        asks for the browser back — the native host watches the advertised set and
        tells Chrome to re-enumerate whenever it changes, which is the only channel
        a runtime has for speaking first.
        """
        descriptor = self._descriptor
        if descriptor is None:
            raise RuntimeTransportError("Runtime server has not started")
        if self._closed:
            raise RuntimeTransportError("Runtime server is closed")
        now_ms = int(time.time() * 1000)
        refreshed = replace(descriptor, expires_at_ms=now_ms + self.ttl_ms)
        self.registry.register(refreshed)
        self._descriptor = refreshed
        self._published = True

    def withdraw(self) -> None:
        """Stop advertising, without giving up the socket or its connections.

        Keeping the listener means an existing relay connection stays usable, so a
        session that detaches and then browses again resumes immediately instead of
        waiting out a fresh discovery.
        """
        if not self._published:
            return
        self._published = False
        with contextlib.suppress(Exception):
            self.registry.unregister(self.runtime_id, token=self.token)

    async def start(self) -> RuntimeDescriptor:
        if self._listener is not None:
            return self.descriptor
        if self._closed:
            raise RuntimeTransportError("Runtime server is closed")
        if Path(self.endpoint).exists() or Path(self.endpoint).is_symlink():
            raise RuntimeTransportError("Runtime endpoint already exists")
        try:
            listener = await asyncio.start_unix_server(self._accept_client, path=self.endpoint)
            os.chmod(self.endpoint, 0o600)
            socket_stat = Path(self.endpoint).lstat()
            if not stat.S_ISSOCK(socket_stat.st_mode):
                raise RuntimeTransportError("Runtime endpoint is not a Unix socket")
            self._socket_identity = (socket_stat.st_dev, socket_stat.st_ino)
            self._listener = listener
            now_ms = int(time.time() * 1000)
            self._descriptor = RuntimeDescriptor(
                runtime_id=self.runtime_id,
                session_id=self.session_id,
                transport=self.transport,
                endpoint=self.endpoint,
                token=self.token,
                pid=os.getpid(),
                created_at_ms=now_ms,
                expires_at_ms=now_ms + self.ttl_ms,
                extension_origin=self.extension_origin,
            )
            self.registry.register(self._descriptor)
            self._published = True
            self._refresh_task = asyncio.create_task(
                self._refresh_descriptor(),
                name=f"browser-extension-runtime-refresh-{self.runtime_id}",
            )
            return self._descriptor
        except Exception:
            await self._close_listener()
            self._unlink_owned_socket()
            raise

    async def wait_for_connection(self, *, timeout: float | None = None) -> MultiplexedPeer:
        async def next_open_peer() -> MultiplexedPeer:
            while True:
                peer = await self._peer_queue.get()
                if not peer.closed:
                    return peer

        if timeout is None:
            return await next_open_peer()
        return await asyncio.wait_for(next_open_peer(), timeout=timeout)

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self._published = False
            refresh = self._refresh_task
            self._refresh_task = None
            if refresh is not None:
                refresh.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await refresh
            with contextlib.suppress(Exception):
                await self._close_listener()
            peers = tuple(self._peers)
            self._peers.clear()
            if peers:
                await asyncio.gather(
                    *(peer.close("Runtime server closed") for peer in peers),
                    return_exceptions=True,
                )
            with contextlib.suppress(Exception):
                self.registry.unregister(self.runtime_id, token=self.token)
            self._unlink_owned_socket()

    async def _refresh_descriptor(self) -> None:
        interval = max(0.05, self.ttl_ms / 3_000)
        while True:
            await asyncio.sleep(interval)
            if self._descriptor is None or self._closed:
                return
            # Never resurrect a withdrawn advertisement: the lease refresh would
            # otherwise reinstate a claim the session has explicitly given up.
            if self._published:
                self.publish()

    async def _close_listener(self) -> None:
        listener = self._listener
        self._listener = None
        if listener is not None:
            listener.close()
            await listener.wait_closed()

    def _unlink_owned_socket(self) -> None:
        if self._socket_identity is None:
            return
        try:
            endpoint_stat = Path(self.endpoint).lstat()
            if (endpoint_stat.st_dev, endpoint_stat.st_ino) == self._socket_identity and stat.S_ISSOCK(
                endpoint_stat.st_mode
            ):
                Path(self.endpoint).unlink()
        except FileNotFoundError:
            pass
        self._socket_identity = None

    async def _accept_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        from .multiplex import MultiplexedPeer

        peer: MultiplexedPeer | None = None
        try:
            raw = await asyncio.wait_for(read_message_async(reader), timeout=AUTH_TIMEOUT_SECONDS)
            runtime_id, token, origin = _validate_auth_request(raw)
            if (
                runtime_id != self.runtime_id
                or not hmac.compare_digest(token, self.token)
                or not hmac.compare_digest(origin, self.extension_origin)
            ):
                raise RuntimeAuthenticationError("Runtime authentication failed")
            await write_message_async(
                writer,
                {
                    "kind": "authenticated",
                    "protocol_version": PROTOCOL_VERSION,
                    "runtime_id": self.runtime_id,
                },
            )
            channel = RuntimeChannel(
                reader,
                writer,
                runtime_id=self.runtime_id,
                session_id=self.session_id,
                outbound_direction=MessageDirection.RUNTIME_TO_EXTENSION,
            )
            peer = MultiplexedPeer(
                channel,
                request_handler=self.request_handler,
                event_handler=self.event_handler,
                max_pending_requests=self.max_pending_requests,
                max_inflight_requests=self.max_inflight_requests,
            )
            assert peer is not None
            self._peers.add(peer)
            await peer.start()
            if self.on_peer is None:
                try:
                    self._peer_queue.put_nowait(peer)
                except asyncio.QueueFull:
                    await peer.close("Runtime connection limit reached")
                    return
            else:
                callback_result = self.on_peer(peer)
                if inspect.isawaitable(callback_result):
                    await callback_result
            await peer.wait_closed()
        except (TimeoutError, FramingError, RuntimeAuthenticationError, ConnectionError):
            writer.close()
            with contextlib.suppress(ConnectionError, OSError):
                await writer.wait_closed()
        finally:
            if peer is not None:
                self._peers.discard(peer)


async def connect_runtime_peer(
    descriptor: RuntimeDescriptor,
    *,
    extension_origin: str,
    request_handler: RequestHandler | None = None,
    event_handler: EventHandler | None = None,
    max_pending_requests: int = 128,
    max_inflight_requests: int = 32,
    platform: str | None = None,
) -> MultiplexedPeer:
    """Connect a multiplexed peer as the extension/native-host side."""
    from .multiplex import MultiplexedPeer

    channel = await connect_runtime_channel(
        descriptor,
        extension_origin=extension_origin,
        platform=platform,
    )
    peer = MultiplexedPeer(
        channel,
        request_handler=request_handler,
        event_handler=event_handler,
        max_pending_requests=max_pending_requests,
        max_inflight_requests=max_inflight_requests,
    )
    await peer.start()
    return peer
