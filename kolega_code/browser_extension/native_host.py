"""Chrome Native Messaging broker for multiple authenticated runtimes."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import os
import queue
import stat
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal, Mapping, Sequence, TextIO, cast

from .framing import MAX_NATIVE_MESSAGE_BYTES, FramingError, encode_message, read_message
from .protocol import (
    MAX_DEADLINE_AHEAD_MS,
    PROTOCOL_VERSION,
    Envelope,
    MessageDirection,
    MessageType,
    ProtocolValidationError,
    runtimes_changed_notification,
    validate_discovery_request,
    validate_operation_request,
)
from .registry import DescriptorError, RuntimeDescriptor, RuntimeDescriptorRegistry, validate_extension_origin
from .runtime import (
    RuntimeChannel,
    RuntimeTransportError,
    UnsupportedRuntimeTransportError,
    connect_runtime_channel,
    ensure_runtime_transport_supported,
)

MAX_HOST_CONFIG_BYTES = 65_536
DEFAULT_MAX_RUNTIMES = 16
DEFAULT_MAX_RELAY_PENDING = 256
RUNTIME_WATCH_INTERVAL_SECONDS = 1.0
DEFAULT_MAX_NATIVE_PENDING_WRITES = 256
DEFAULT_MAX_NATIVE_PENDING_WRITE_BYTES = 8 * MAX_NATIVE_MESSAGE_BYTES
DEFAULT_NATIVE_WRITE_SHUTDOWN_SECONDS = 1.0
_HOST_CONFIG_KEYS = frozenset({"extension_origin", "registry_dir", "max_runtimes", "max_pending_requests"})


class NativeHostError(RuntimeError):
    """A safe-to-report native-host failure."""


class NativeHostConfigurationError(NativeHostError):
    """Native-host configuration is absent, unsafe, or invalid."""


class NativeMessageEndpointClosedError(NativeHostError):
    """Native stdout is closing or unavailable."""


@dataclass(frozen=True, slots=True)
class NativeHostConfig:
    """Non-secret native-host configuration."""

    extension_origin: str
    registry_dir: Path
    max_runtimes: int = DEFAULT_MAX_RUNTIMES
    max_pending_requests: int = DEFAULT_MAX_RELAY_PENDING

    def __post_init__(self) -> None:
        try:
            origin = validate_extension_origin(self.extension_origin)
        except DescriptorError:
            raise NativeHostConfigurationError("Native-host extension origin is invalid") from None
        if not self.registry_dir.is_absolute():
            raise NativeHostConfigurationError("Native-host registry path must be absolute")
        if not 0 < self.max_runtimes <= 64:
            raise NativeHostConfigurationError("Native-host runtime limit is invalid")
        if not 0 < self.max_pending_requests <= 4096:
            raise NativeHostConfigurationError("Native-host pending-request limit is invalid")
        object.__setattr__(self, "extension_origin", origin)

    @classmethod
    def from_mapping(cls, raw: object) -> NativeHostConfig:
        if not isinstance(raw, dict) or set(raw) != _HOST_CONFIG_KEYS:
            raise NativeHostConfigurationError("Native-host configuration has an invalid schema")
        origin = raw["extension_origin"]
        registry_dir = raw["registry_dir"]
        max_runtimes = raw["max_runtimes"]
        max_pending = raw["max_pending_requests"]
        if not isinstance(origin, str):
            raise NativeHostConfigurationError("Native-host extension origin is invalid")
        if not isinstance(registry_dir, str) or "\0" in registry_dir:
            raise NativeHostConfigurationError("Native-host registry path is invalid")
        if (
            not isinstance(max_runtimes, int)
            or isinstance(max_runtimes, bool)
            or not isinstance(max_pending, int)
            or isinstance(max_pending, bool)
        ):
            raise NativeHostConfigurationError("Native-host limits are invalid")
        return cls(
            extension_origin=origin,
            registry_dir=Path(registry_dir).expanduser(),
            max_runtimes=max_runtimes,
            max_pending_requests=max_pending,
        )

    def accepts_origin(self, origin: str) -> bool:
        return self.extension_origin == origin


def default_host_config_path(*, platform: str | None = None) -> Path:
    if (platform or sys.platform) != "darwin":
        raise UnsupportedRuntimeTransportError("Chrome browser integration is supported only on macOS")
    configured = os.environ.get("KOLEGA_BROWSER_HOST_CONFIG")
    if configured:
        return Path(configured).expanduser()
    root = Path.home() / "Library" / "Application Support"
    return root / "kolega-code" / "browser-extension-host.json"


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise NativeHostConfigurationError("Native-host configuration is malformed")
        result[key] = value
    return result


def _private_file_stat(file_stat: os.stat_result) -> bool:
    if not stat.S_ISREG(file_stat.st_mode):
        return False
    getuid = getattr(os, "getuid", None)
    if getuid is None:
        return False
    return file_stat.st_uid == getuid() and not stat.S_IMODE(file_stat.st_mode) & 0o077


def load_host_config(path: Path) -> NativeHostConfig:
    """Load a bounded per-user host configuration without following symlinks."""
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError:
        raise NativeHostConfigurationError("Native-host configuration is unavailable") from None
    try:
        file_stat = os.fstat(fd)
        if not _private_file_stat(file_stat):
            raise NativeHostConfigurationError("Native-host configuration permissions are unsafe")
        with os.fdopen(fd, "rb", closefd=True) as file:
            fd = -1
            payload = file.read(MAX_HOST_CONFIG_BYTES + 1)
    finally:
        if fd >= 0:
            os.close(fd)
    if len(payload) > MAX_HOST_CONFIG_BYTES:
        raise NativeHostConfigurationError("Native-host configuration exceeds the size limit")
    try:
        raw = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda _: (_ for _ in ()).throw(
                NativeHostConfigurationError("Native-host configuration is malformed")
            ),
        )
    except NativeHostConfigurationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise NativeHostConfigurationError("Native-host configuration is malformed") from None
    return NativeHostConfig.from_mapping(raw)


@dataclass(frozen=True, slots=True)
class _NativeWrite:
    frame: bytes
    acknowledgement: concurrent.futures.Future[None]


class NativeMessageEndpoint:
    """Async endpoint with one bounded, uncancellable stdout owner."""

    def __init__(
        self,
        input_stream: BinaryIO,
        output_stream: BinaryIO,
        *,
        max_message_bytes: int = MAX_NATIVE_MESSAGE_BYTES,
        max_pending_writes: int = DEFAULT_MAX_NATIVE_PENDING_WRITES,
        max_pending_write_bytes: int = DEFAULT_MAX_NATIVE_PENDING_WRITE_BYTES,
        shutdown_timeout: float = DEFAULT_NATIVE_WRITE_SHUTDOWN_SECONDS,
    ) -> None:
        if max_pending_writes <= 0 or max_pending_write_bytes <= 0 or shutdown_timeout <= 0:
            raise ValueError("native endpoint bounds must be positive")
        self.input_stream = input_stream
        self.output_stream = output_stream
        self.max_message_bytes = max_message_bytes
        self.max_pending_writes = max_pending_writes
        self.max_pending_write_bytes = max_pending_write_bytes
        self.shutdown_timeout = shutdown_timeout
        self._write_queue: queue.Queue[_NativeWrite] = queue.Queue(maxsize=max_pending_writes)
        self._writer_thread: threading.Thread | None = None
        self._writer_finished: concurrent.futures.Future[None] = concurrent.futures.Future()
        self._writer_stop = threading.Event()
        self._owner_loop: asyncio.AbstractEventLoop | None = None
        self._slot_available: asyncio.Event | None = None
        self._outstanding_writes = 0
        self._outstanding_write_bytes = 0
        self._closing = False
        self._write_failure: BaseException | None = None

    @property
    def outstanding_write_count(self) -> int:
        return self._outstanding_writes

    @property
    def outstanding_write_bytes(self) -> int:
        return self._outstanding_write_bytes

    async def read(self) -> dict[str, Any] | None:
        return await asyncio.to_thread(read_message, self.input_stream, max_bytes=self.max_message_bytes)

    async def write(self, envelope: Envelope) -> None:
        await self.write_mapping(envelope.to_mapping())

    async def write_mapping(self, message: Mapping[str, object]) -> None:
        self._raise_if_unwritable()
        frame = encode_message(message, max_bytes=self.max_message_bytes)
        await self._reserve_write_slot(len(frame))
        acknowledgement: concurrent.futures.Future[None] = concurrent.futures.Future()
        item = _NativeWrite(frame=frame, acknowledgement=acknowledgement)
        try:
            self._ensure_writer()
            self._write_queue.put_nowait(item)
        except BaseException:
            self._release_write_slot(len(frame))
            raise
        wrapped = asyncio.wrap_future(acknowledgement)
        wrapped.add_done_callback(self._consume_abandoned_write_exception)
        await asyncio.shield(wrapped)

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._writer_stop.set()
        if self._slot_available is not None:
            self._slot_available.set()
        if self._writer_thread is None:
            if not self._writer_finished.done():
                self._writer_finished.set_result(None)
            return
        try:
            await asyncio.wait_for(
                asyncio.shield(asyncio.wrap_future(self._writer_finished)),
                timeout=self.shutdown_timeout,
            )
        except TimeoutError:
            self._write_failure = NativeMessageEndpointClosedError(
                "Native stdout did not drain before the shutdown deadline"
            )

    def _raise_if_unwritable(self) -> None:
        if self._write_failure is not None:
            raise NativeMessageEndpointClosedError("Native stdout is unavailable") from self._write_failure
        if self._closing:
            raise NativeMessageEndpointClosedError("Native stdout is closed")

    async def _reserve_write_slot(self, frame_bytes: int) -> None:
        if frame_bytes > self.max_pending_write_bytes:
            raise NativeMessageEndpointClosedError("Native message exceeds the pending stdout byte limit")
        loop = asyncio.get_running_loop()
        if self._owner_loop is None:
            self._owner_loop = loop
            self._slot_available = asyncio.Event()
            self._slot_available.set()
        elif self._owner_loop is not loop:
            raise NativeMessageEndpointClosedError("Native stdout cannot be shared across event loops")
        assert self._slot_available is not None
        while (
            self._outstanding_writes >= self.max_pending_writes
            or self._outstanding_write_bytes + frame_bytes > self.max_pending_write_bytes
        ):
            self._raise_if_unwritable()
            self._slot_available.clear()
            if (
                self._outstanding_writes < self.max_pending_writes
                and self._outstanding_write_bytes + frame_bytes <= self.max_pending_write_bytes
            ):
                self._slot_available.set()
                continue
            await self._slot_available.wait()
        self._raise_if_unwritable()
        self._outstanding_writes += 1
        self._outstanding_write_bytes += frame_bytes

    def _release_write_slot(self, frame_bytes: int) -> None:
        self._outstanding_writes = max(0, self._outstanding_writes - 1)
        self._outstanding_write_bytes = max(0, self._outstanding_write_bytes - frame_bytes)
        if self._slot_available is not None:
            self._slot_available.set()

    def _release_write_slot_from_thread(self, frame_bytes: int) -> None:
        loop = self._owner_loop
        if loop is not None:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(self._release_write_slot, frame_bytes)

    def _ensure_writer(self) -> None:
        if self._writer_thread is not None:
            return
        self._writer_thread = threading.Thread(
            target=self._writer_main,
            name="kolega-native-message-writer",
            daemon=True,
        )
        self._writer_thread.start()

    def _writer_main(self) -> None:
        try:
            while True:
                if self._writer_stop.is_set() and self._write_queue.empty():
                    return
                try:
                    item = self._write_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                try:
                    self._write_frame(item.frame)
                except BaseException as exc:
                    self._write_failure = exc
                    if not item.acknowledgement.done():
                        item.acknowledgement.set_exception(exc)
                    self._drain_queued_writes(exc)
                    return
                else:
                    if not item.acknowledgement.done():
                        item.acknowledgement.set_result(None)
                finally:
                    self._write_queue.task_done()
                    self._release_write_slot_from_thread(len(item.frame))
        finally:
            if not self._writer_finished.done():
                self._writer_finished.set_result(None)

    def _write_frame(self, frame: bytes) -> None:
        view = memoryview(frame)
        written = 0
        while written < len(view):
            count = self.output_stream.write(view[written:])
            if count is None:
                written = len(view)
            elif count <= 0:
                raise FramingError("Native message could not be written")
            else:
                written += count
        flush = getattr(self.output_stream, "flush", None)
        if flush is not None:
            flush()

    def _drain_queued_writes(self, error: BaseException) -> None:
        while True:
            try:
                item = self._write_queue.get_nowait()
            except queue.Empty:
                return
            if not item.acknowledgement.done():
                item.acknowledgement.set_exception(error)
            self._write_queue.task_done()
            self._release_write_slot_from_thread(len(item.frame))

    @staticmethod
    def _consume_abandoned_write_exception(future: asyncio.Future[None]) -> None:
        if not future.cancelled():
            with contextlib.suppress(BaseException):
                future.exception()


@dataclass(slots=True)
class _RuntimeRelay:
    descriptor: RuntimeDescriptor
    channel: RuntimeChannel
    reader_task: asyncio.Task[None] | None = None


@dataclass(slots=True)
class _RelayPending:
    request: Envelope
    timeout_task: asyncio.Task[None]


class NativeHostBroker:
    """Relay one Chrome native port across active runtimes."""

    def __init__(
        self,
        *,
        endpoint: NativeMessageEndpoint,
        registry: RuntimeDescriptorRegistry,
        extension_origin: str,
        max_runtimes: int = DEFAULT_MAX_RUNTIMES,
        max_pending_requests: int = DEFAULT_MAX_RELAY_PENDING,
    ) -> None:
        if max_runtimes <= 0 or max_pending_requests <= 0:
            raise ValueError("broker bounds must be positive")
        self.endpoint = endpoint
        self.registry = registry
        self.extension_origin = validate_extension_origin(extension_origin)
        self.max_runtimes = max_runtimes
        self.max_pending_requests = max_pending_requests
        self._runtimes: dict[str, _RuntimeRelay] = {}
        self._pending: dict[tuple[MessageDirection, str, str], _RelayPending] = {}
        self._close_lock = asyncio.Lock()
        self._closed = False
        self._watch_task: asyncio.Task[None] | None = None
        self._advertised: frozenset[str] | None = None

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def _advertised_runtime_ids(self) -> frozenset[str]:
        try:
            return frozenset(
                descriptor.runtime_id
                for descriptor in self.registry.list_active()
                if descriptor.accepts_origin(self.extension_origin)
            )
        except OSError:
            return self._advertised or frozenset()

    async def _watch_runtimes(self) -> None:
        """Tell Chrome to re-enumerate runtimes whenever the live set changes.

        Compares runtime ids rather than file mtimes: each runtime rewrites its
        descriptor every TTL/3 to extend the lease, so mtime-based watching would
        notify continuously.
        """
        while True:
            await asyncio.sleep(RUNTIME_WATCH_INTERVAL_SECONDS)
            if self._closed:
                return
            current = self._advertised_runtime_ids()
            if current == self._advertised:
                continue
            self._advertised = current
            try:
                await self.endpoint.write_mapping(runtimes_changed_notification())
            except (NativeHostError, RuntimeTransportError, OSError):
                return

    async def run(self) -> None:
        self._advertised = self._advertised_runtime_ids()
        self._watch_task = asyncio.create_task(self._watch_runtimes(), name="kolega-browser-runtime-watch")
        try:
            while True:
                raw = await self.endpoint.read()
                if raw is None:
                    break
                if raw.get("kind") == "list_runtimes":
                    await self._list_runtimes(raw)
                    continue
                envelope = Envelope.from_mapping(raw)
                if envelope.direction is not MessageDirection.EXTENSION_TO_RUNTIME:
                    raise NativeHostError("Chrome sent an envelope with an invalid direction")
                self._validate_request(envelope)
                await self._route_from_extension(envelope)
        finally:
            await self.close()

    @staticmethod
    def _validate_request(envelope: Envelope) -> None:
        if envelope.type is MessageType.REQUEST:
            validate_operation_request(envelope.payload["operation"], envelope.payload["params"])

    async def _list_runtimes(self, raw: Mapping[str, object]) -> None:
        try:
            request = validate_discovery_request(dict(raw))
        except ProtocolValidationError as exc:
            raise NativeHostError("Chrome sent an invalid runtime discovery request") from exc
        request_id = cast(str, request["request_id"])
        matching = [
            descriptor for descriptor in self.registry.list_active() if descriptor.accepts_origin(self.extension_origin)
        ]
        runtimes = [
            {
                "runtime_id": descriptor.runtime_id,
                "session_id": descriptor.session_id,
                "created_at_ms": descriptor.created_at_ms,
                "expires_at_ms": descriptor.expires_at_ms,
            }
            for descriptor in matching[: self.max_runtimes]
        ]
        await self.endpoint.write_mapping(
            {
                "kind": "runtimes",
                "protocol_version": PROTOCOL_VERSION,
                "request_id": request_id,
                "runtimes": runtimes,
            }
        )

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            watch = self._watch_task
            self._watch_task = None
            if watch is not None and watch is not asyncio.current_task():
                watch.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await watch
            pending = tuple(self._pending.values())
            self._pending.clear()
            for item in pending:
                item.timeout_task.cancel()
                relay = self._runtimes.get(item.request.runtime_id)
                if relay is None:
                    continue
                with contextlib.suppress(RuntimeTransportError):
                    if item.request.direction is MessageDirection.RUNTIME_TO_EXTENSION:
                        await relay.channel.write(
                            Envelope.error_for(
                                item.request,
                                code="extension_disconnected",
                                message="Chrome extension disconnected",
                                retryable=True,
                            )
                        )
                    else:
                        await relay.channel.write(Envelope.cancel_for(item.request, reason="extension_disconnected"))

            relays = tuple(self._runtimes.values())
            self._runtimes.clear()
            current = asyncio.current_task()
            for relay in relays:
                if relay.reader_task is not None and relay.reader_task is not current:
                    relay.reader_task.cancel()
            await asyncio.gather(*(relay.channel.close() for relay in relays), return_exceptions=True)
            readers = [
                relay.reader_task
                for relay in relays
                if relay.reader_task is not None and relay.reader_task is not current
            ]
            if readers:
                await asyncio.gather(*readers, return_exceptions=True)

    async def _route_from_extension(self, envelope: Envelope) -> None:
        if envelope.type is MessageType.REQUEST:
            now_ms = int(time.time() * 1000)
            if envelope.deadline_ms > now_ms + MAX_DEADLINE_AHEAD_MS:
                await self.endpoint.write(
                    Envelope.error_for(
                        envelope,
                        code="invalid_deadline",
                        message="Request deadline exceeds the protocol limit",
                    )
                )
                return
            if envelope.is_expired(now_ms=now_ms):
                await self.endpoint.write(
                    Envelope.error_for(
                        envelope,
                        code="deadline_exceeded",
                        message="Request deadline exceeded",
                        retryable=True,
                    )
                )
                return
        try:
            relay = await self._runtime_for(envelope)
        except (DescriptorError, RuntimeTransportError):
            if envelope.type is MessageType.REQUEST:
                expired = envelope.is_expired()
                await self.endpoint.write(
                    Envelope.error_for(
                        envelope,
                        code="deadline_exceeded" if expired else "runtime_unavailable",
                        message="Request deadline exceeded" if expired else "Requested runtime is unavailable",
                        retryable=True,
                    )
                )
            return

        if envelope.type is MessageType.REQUEST:
            if envelope.is_expired():
                await self.endpoint.write(
                    Envelope.error_for(
                        envelope,
                        code="deadline_exceeded",
                        message="Request deadline exceeded",
                        retryable=True,
                    )
                )
                return
            registration = self._register_pending(envelope)
            if registration == "duplicate":
                return
            if registration == "full":
                await self.endpoint.write(
                    Envelope.error_for(
                        envelope,
                        code="request_limit",
                        message="Native-host request limit reached",
                        retryable=True,
                    )
                )
                return
        elif envelope.type in {MessageType.RESPONSE, MessageType.ERROR}:
            pending = self._settle_pending(envelope.direction.opposite, envelope.runtime_id, envelope.request_id)
            if pending is None:
                return
            if not self._response_matches(pending.request, envelope):
                await relay.channel.write(
                    Envelope.error_for(
                        pending.request,
                        code="invalid_response",
                        message="Extension response metadata is invalid",
                    )
                )
                return
            if pending.request.is_expired():
                await relay.channel.write(
                    Envelope.error_for(
                        pending.request,
                        code="deadline_exceeded",
                        message="Request deadline exceeded",
                        retryable=True,
                    )
                )
                return
        elif envelope.type is MessageType.CANCEL:
            pending = self._settle_pending(envelope.direction, envelope.runtime_id, envelope.request_id)
            if pending is None or not self._response_matches(pending.request, envelope):
                return
        try:
            await relay.channel.write(envelope)
        except RuntimeTransportError:
            await self._runtime_disconnected(relay)

    async def _runtime_for(self, envelope: Envelope) -> _RuntimeRelay:
        existing = self._runtimes.get(envelope.runtime_id)
        if existing is not None:
            if existing.descriptor.session_id != envelope.session_id:
                raise DescriptorError("Runtime session does not match")
            return existing
        if len(self._runtimes) >= self.max_runtimes:
            raise RuntimeTransportError("Native-host runtime limit reached")
        descriptor = self.registry.read(envelope.runtime_id)
        if descriptor.session_id != envelope.session_id or not descriptor.accepts_origin(self.extension_origin):
            raise DescriptorError("Runtime identity or extension origin does not match")
        remaining = max(0.001, (envelope.deadline_ms - int(time.time() * 1000)) / 1000)
        channel = await connect_runtime_channel(
            descriptor,
            extension_origin=self.extension_origin,
            timeout=remaining,
        )
        relay = _RuntimeRelay(descriptor=descriptor, channel=channel)
        self._runtimes[descriptor.runtime_id] = relay
        relay.reader_task = asyncio.create_task(
            self._read_runtime(relay),
            name=f"browser-extension-broker-{descriptor.runtime_id}",
        )
        return relay

    async def _read_runtime(self, relay: _RuntimeRelay) -> None:
        try:
            while True:
                envelope = await relay.channel.read()
                if envelope is None:
                    break
                self._validate_request(envelope)
                await self._route_from_runtime(relay, envelope)
        except (RuntimeTransportError, ProtocolValidationError, ConnectionError):
            pass
        finally:
            await self._runtime_disconnected(relay)

    async def _route_from_runtime(self, relay: _RuntimeRelay, envelope: Envelope) -> None:
        if envelope.type is MessageType.REQUEST:
            now_ms = int(time.time() * 1000)
            if envelope.deadline_ms > now_ms + MAX_DEADLINE_AHEAD_MS:
                await relay.channel.write(
                    Envelope.error_for(
                        envelope,
                        code="invalid_deadline",
                        message="Request deadline exceeds the protocol limit",
                    )
                )
                return
            if envelope.is_expired(now_ms=now_ms):
                await relay.channel.write(
                    Envelope.error_for(
                        envelope,
                        code="deadline_exceeded",
                        message="Request deadline exceeded",
                        retryable=True,
                    )
                )
                return
            registration = self._register_pending(envelope)
            if registration == "duplicate":
                return
            if registration == "full":
                await relay.channel.write(
                    Envelope.error_for(
                        envelope,
                        code="request_limit",
                        message="Native-host request limit reached",
                        retryable=True,
                    )
                )
                return
        elif envelope.type in {MessageType.RESPONSE, MessageType.ERROR}:
            pending = self._settle_pending(envelope.direction.opposite, envelope.runtime_id, envelope.request_id)
            if pending is None:
                return
            if not self._response_matches(pending.request, envelope):
                await self.endpoint.write(
                    Envelope.error_for(
                        pending.request,
                        code="invalid_response",
                        message="Runtime response metadata is invalid",
                    )
                )
                return
            if pending.request.is_expired():
                await self.endpoint.write(
                    Envelope.error_for(
                        pending.request,
                        code="deadline_exceeded",
                        message="Request deadline exceeded",
                        retryable=True,
                    )
                )
                return
        elif envelope.type is MessageType.CANCEL:
            pending = self._settle_pending(envelope.direction, envelope.runtime_id, envelope.request_id)
            if pending is None or not self._response_matches(pending.request, envelope):
                return
        await self.endpoint.write(envelope)

    def _register_pending(self, request: Envelope) -> Literal["added", "duplicate", "full"]:
        key = (request.direction, request.runtime_id, request.request_id)
        if key in self._pending:
            return "duplicate"
        if len(self._pending) >= self.max_pending_requests:
            return "full"
        delay = max(0.0, (request.deadline_ms - int(time.time() * 1000)) / 1000)
        timeout_task = asyncio.create_task(
            self._expire_pending(key, delay),
            name="browser-extension-broker-deadline",
        )
        self._pending[key] = _RelayPending(request=request, timeout_task=timeout_task)
        return "added"

    def _settle_pending(
        self,
        requester: MessageDirection,
        runtime_id: str,
        request_id: str,
    ) -> _RelayPending | None:
        pending = self._pending.pop((requester, runtime_id, request_id), None)
        if pending is not None:
            pending.timeout_task.cancel()
        return pending

    @staticmethod
    def _response_matches(request: Envelope, response: Envelope) -> bool:
        return (
            response.runtime_id == request.runtime_id
            and response.session_id == request.session_id
            and response.request_id == request.request_id
            and response.deadline_ms == request.deadline_ms
        )

    async def _expire_pending(self, key: tuple[MessageDirection, str, str], delay: float) -> None:
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        pending = self._pending.pop(key, None)
        if pending is None:
            return
        relay = self._runtimes.get(pending.request.runtime_id)
        if pending.request.direction is MessageDirection.EXTENSION_TO_RUNTIME:
            with contextlib.suppress(FramingError, ConnectionError):
                await self.endpoint.write(
                    Envelope.error_for(
                        pending.request,
                        code="deadline_exceeded",
                        message="Request deadline exceeded",
                        retryable=True,
                    )
                )
            if relay is not None:
                with contextlib.suppress(RuntimeTransportError):
                    await relay.channel.write(Envelope.cancel_for(pending.request, reason="deadline_exceeded"))
        elif relay is not None:
            with contextlib.suppress(RuntimeTransportError):
                await relay.channel.write(
                    Envelope.error_for(
                        pending.request,
                        code="deadline_exceeded",
                        message="Request deadline exceeded",
                        retryable=True,
                    )
                )
            with contextlib.suppress(FramingError, ConnectionError):
                await self.endpoint.write(Envelope.cancel_for(pending.request, reason="deadline_exceeded"))

    async def _runtime_disconnected(self, relay: _RuntimeRelay) -> None:
        if self._runtimes.get(relay.descriptor.runtime_id) is not relay:
            return
        self._runtimes.pop(relay.descriptor.runtime_id, None)
        disconnected = [
            (key, pending)
            for key, pending in self._pending.items()
            if pending.request.runtime_id == relay.descriptor.runtime_id
        ]
        try:
            for key, pending in disconnected:
                settled = self._settle_pending(*key)
                if settled is not pending or self._closed:
                    continue
                if pending.request.direction is MessageDirection.EXTENSION_TO_RUNTIME:
                    with contextlib.suppress(FramingError, ConnectionError):
                        await self.endpoint.write(
                            Envelope.error_for(
                                pending.request,
                                code="runtime_disconnected",
                                message="Runtime disconnected",
                                retryable=True,
                            )
                        )
                else:
                    with contextlib.suppress(FramingError, ConnectionError):
                        await self.endpoint.write(Envelope.cancel_for(pending.request, reason="runtime_disconnected"))
        finally:
            await relay.channel.close()


async def run_native_host(
    *,
    config: NativeHostConfig,
    extension_origin: str,
    input_stream: BinaryIO,
    output_stream: BinaryIO,
) -> None:
    origin = validate_extension_origin(extension_origin)
    if not config.accepts_origin(origin):
        raise NativeHostConfigurationError("Chrome extension origin is not configured")
    ensure_runtime_transport_supported()
    registry = RuntimeDescriptorRegistry(config.registry_dir)
    endpoint = NativeMessageEndpoint(
        input_stream,
        output_stream,
        max_pending_writes=config.max_pending_requests,
    )
    broker = NativeHostBroker(
        endpoint=endpoint,
        registry=registry,
        extension_origin=origin,
        max_runtimes=config.max_runtimes,
        max_pending_requests=config.max_pending_requests,
    )
    try:
        await broker.run()
    finally:
        await endpoint.close()


def _binary_stream(stream: object, label: str) -> BinaryIO:
    binary = getattr(stream, "buffer", stream)
    if label == "input" and not hasattr(binary, "read"):
        raise NativeHostError("Native-host input stream is unavailable")
    if label == "output" and not hasattr(binary, "write"):
        raise NativeHostError("Native-host output stream is unavailable")
    return cast(BinaryIO, binary)


def main(
    argv: Sequence[str] | None = None,
    *,
    config_path: Path | None = None,
    input_stream: BinaryIO | None = None,
    output_stream: BinaryIO | None = None,
    error_stream: TextIO | None = None,
) -> int:
    """Console entry point launched by Chrome."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    errors = sys.stderr if error_stream is None else error_stream
    if not arguments:
        print("kolega browser host: Chrome extension origin is missing", file=errors)
        return 2
    try:
        try:
            origin = validate_extension_origin(arguments[0])
        except DescriptorError:
            raise NativeHostConfigurationError("Chrome extension origin is invalid") from None
        config = load_host_config(config_path or default_host_config_path())
        if not config.accepts_origin(origin):
            raise NativeHostConfigurationError("Chrome extension origin is not configured")
        native_input = _binary_stream(sys.stdin if input_stream is None else input_stream, "input")
        native_output = _binary_stream(sys.stdout if output_stream is None else output_stream, "output")
        asyncio.run(
            run_native_host(
                config=config,
                extension_origin=origin,
                input_stream=native_input,
                output_stream=native_output,
            )
        )
        return 0
    except NativeHostConfigurationError:
        print("kolega browser host: configuration or extension origin rejected", file=errors)
        return 2
    except UnsupportedRuntimeTransportError:
        print("kolega browser host: secure platform transport is unavailable", file=errors)
        return 3
    except (NativeHostError, DescriptorError, RuntimeTransportError, FramingError, ProtocolValidationError, OSError):
        print("kolega browser host: transport terminated", file=errors)
        return 1
    except KeyboardInterrupt:
        return 130
