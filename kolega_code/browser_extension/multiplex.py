"""Bounded request correlation, deadlines, cancellation, and disconnects."""

from __future__ import annotations

import asyncio
import secrets
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import cast

from .protocol import JSONValue, Envelope, MessageType, ProtocolValidationError
from .runtime import RuntimeChannel, RuntimeTransportError

DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0
MAX_REQUEST_TIMEOUT_SECONDS = 300.0

RequestHandler = Callable[[Envelope], Awaitable[JSONValue]]
EventHandler = Callable[[Envelope], Awaitable[None]]


class MultiplexError(RuntimeError):
    """A safe-to-report multiplexing failure."""


class ConnectionClosedError(MultiplexError):
    """The peer disconnected before a request settled."""


class PendingRequestLimitError(MultiplexError):
    """The bounded outgoing request table is full."""


class RequestTimeoutError(MultiplexError):
    """A request deadline elapsed."""


class RemoteRequestError(MultiplexError):
    """A validated bounded error returned by the other endpoint."""

    def __init__(self, code: str, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


@dataclass(slots=True)
class _PendingRequest:
    future: asyncio.Future[JSONValue]
    request: Envelope
    timeout_handle: asyncio.TimerHandle


@dataclass(slots=True)
class _InflightRequest:
    request: Envelope
    task: asyncio.Task[None]


class MultiplexedPeer:
    """Multiplex requests and events over one authenticated runtime channel."""

    def __init__(
        self,
        channel: RuntimeChannel,
        *,
        request_handler: RequestHandler | None = None,
        event_handler: EventHandler | None = None,
        max_pending_requests: int = 128,
        max_inflight_requests: int = 32,
    ) -> None:
        if max_pending_requests <= 0 or max_inflight_requests <= 0:
            raise ValueError("request bounds must be positive")
        self.channel = channel
        self.request_handler = request_handler
        self.event_handler = event_handler
        self.max_pending_requests = max_pending_requests
        self.max_inflight_requests = max_inflight_requests
        self._pending: dict[str, _PendingRequest] = {}
        self._inflight: dict[str, _InflightRequest] = {}
        self._event_tasks: set[asyncio.Task[None]] = set()
        self._reader_task: asyncio.Task[None] | None = None
        self._closed_event = asyncio.Event()
        self._closed_reason = "Runtime connection closed"
        self._closing = False
        self._close_lock = asyncio.Lock()

    @property
    def closed(self) -> bool:
        return self._closing or self._closed_event.is_set() or self.channel.closed

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def inflight_count(self) -> int:
        return len(self._inflight)

    async def start(self) -> None:
        if self.closed:
            raise ConnectionClosedError("Runtime connection is closed")
        if self._reader_task is None:
            self._reader_task = asyncio.create_task(
                self._read_loop(),
                name="browser-extension-runtime-reader",
            )

    async def wait_closed(self) -> None:
        await self._closed_event.wait()

    async def drain_events(self) -> None:
        """Wait for event callbacks already admitted by the read loop."""
        tasks = tuple(self._event_tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def request(
        self,
        operation: str,
        params: Mapping[str, JSONValue],
        *,
        timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        request_id: str | None = None,
    ) -> JSONValue:
        if not 0 < timeout <= MAX_REQUEST_TIMEOUT_SECONDS:
            raise ValueError(f"timeout must be between 0 and {MAX_REQUEST_TIMEOUT_SECONDS} seconds")
        if self.closed:
            raise ConnectionClosedError(self._closed_reason)
        if self._reader_task is None:
            await self.start()
        if len(self._pending) >= self.max_pending_requests:
            raise PendingRequestLimitError("Pending request limit reached")

        selected_id = request_id or f"request_{secrets.token_urlsafe(18)}"
        if selected_id in self._pending:
            raise MultiplexError("Request ID is already pending")
        loop = asyncio.get_running_loop()
        deadline_at = loop.time() + timeout
        deadline_ms = int(time.time() * 1000 + timeout * 1000)
        envelope = Envelope.request(
            direction=self.channel.outbound_direction,
            request_id=selected_id,
            runtime_id=self.channel.runtime_id,
            session_id=self.channel.session_id,
            deadline_ms=deadline_ms,
            operation=operation,
            params=params,
        )
        future: asyncio.Future[JSONValue] = loop.create_future()
        timeout_handle = loop.call_later(timeout, self._expire_request, selected_id)
        self._pending[selected_id] = _PendingRequest(
            future=future,
            request=envelope,
            timeout_handle=timeout_handle,
        )
        try:
            async with asyncio.timeout_at(deadline_at):
                await self.channel.write(envelope)
        except TimeoutError:
            pending = self._pop_pending(selected_id)
            self._settle_abandoned_future(pending.future if pending is not None else future)
            await self.close("Runtime request write exceeded its deadline")
            raise RequestTimeoutError("Request deadline exceeded") from None
        except asyncio.CancelledError:
            pending = self._pop_pending(selected_id)
            self._settle_abandoned_future(pending.future if pending is not None else future)
            if pending is not None:
                await self._send_cancel(pending.request, "caller_cancelled")
            raise
        except Exception:
            pending = self._pop_pending(selected_id)
            self._settle_abandoned_future(pending.future if pending is not None else future)
            await self.close("Runtime connection is closed")
            raise ConnectionClosedError("Runtime connection is closed") from None

        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            pending = self._pop_pending(selected_id)
            if pending is not None:
                pending.future.cancel()
                await self._send_cancel(pending.request, "caller_cancelled")
            raise

    async def notify(
        self,
        event: str,
        data: Mapping[str, JSONValue],
        *,
        deadline_ms: int | None = None,
    ) -> None:
        if self.closed:
            raise ConnectionClosedError(self._closed_reason)
        now_ms = int(time.time() * 1000)
        envelope = Envelope.event(
            direction=self.channel.outbound_direction,
            request_id=f"event_{secrets.token_urlsafe(18)}",
            runtime_id=self.channel.runtime_id,
            session_id=self.channel.session_id,
            deadline_ms=deadline_ms or now_ms + int(DEFAULT_REQUEST_TIMEOUT_SECONDS * 1000),
            event=event,
            data=data,
        )
        try:
            await self.channel.write(envelope)
        except RuntimeTransportError:
            await self.close("Runtime connection is closed")
            raise ConnectionClosedError("Runtime connection is closed") from None

    async def close(self, reason: str = "Runtime connection closed") -> None:
        async with self._close_lock:
            if self._closed_event.is_set():
                return
            self._closing = True
            self._closed_reason = reason
            current = asyncio.current_task()
            reader = self._reader_task
            if reader is not None and reader is not current and not reader.done():
                reader.cancel()

            pending_items = tuple(self._pending.values())
            self._pending.clear()
            for pending in pending_items:
                pending.timeout_handle.cancel()
                if not pending.future.done():
                    pending.future.set_exception(ConnectionClosedError(reason))

            inflight = tuple(item.task for item in self._inflight.values())
            self._inflight.clear()
            events = tuple(self._event_tasks)
            self._event_tasks.clear()
            for task in (*inflight, *events):
                if task is not current:
                    task.cancel()

            await self.channel.close()
            waits = [
                task
                for task in (reader, *inflight, *events)
                if task is not None and task is not current and not task.done()
            ]
            if waits:
                await asyncio.gather(*waits, return_exceptions=True)
            self._closed_event.set()

    def _expire_request(self, request_id: str) -> None:
        pending = self._pop_pending(request_id)
        if pending is None:
            return
        if not pending.future.done():
            pending.future.set_exception(RequestTimeoutError("Request deadline exceeded"))
        asyncio.create_task(self._send_cancel(pending.request, "deadline_exceeded"))

    def _pop_pending(self, request_id: str) -> _PendingRequest | None:
        pending = self._pending.pop(request_id, None)
        if pending is not None:
            pending.timeout_handle.cancel()
        return pending

    @staticmethod
    def _settle_abandoned_future(future: asyncio.Future[JSONValue]) -> None:
        if not future.done():
            future.cancel()
        elif not future.cancelled():
            future.exception()

    async def _send_cancel(self, request: Envelope, reason: str) -> None:
        if self.closed:
            return
        try:
            await self.channel.write(Envelope.cancel_for(request, reason=reason))
        except asyncio.CancelledError:
            try:
                await self.close("Runtime connection is closed")
            finally:
                raise
        except Exception:
            await self.close("Runtime connection is closed")

    async def _read_loop(self) -> None:
        reason = "Runtime connection closed"
        try:
            while True:
                envelope = await self.channel.read()
                if envelope is None:
                    break
                await self._dispatch(envelope)
        except asyncio.CancelledError:
            reason = self._closed_reason
        except (ProtocolValidationError, RuntimeTransportError, ConnectionError):
            reason = "Runtime connection protocol failed"
        finally:
            if not self._closing:
                await self.close(reason)

    async def _dispatch(self, envelope: Envelope) -> None:
        if envelope.type is MessageType.RESPONSE:
            self._handle_response(envelope)
        elif envelope.type is MessageType.ERROR:
            self._handle_error(envelope)
        elif envelope.type is MessageType.REQUEST:
            await self._start_incoming_request(envelope)
        elif envelope.type is MessageType.CANCEL:
            self._handle_cancel(envelope)
        else:
            self._handle_event(envelope)

    def _handle_response(self, envelope: Envelope) -> None:
        pending = self._pop_pending(envelope.request_id)
        if pending is None:
            return
        if not self._response_matches(pending.request, envelope):
            if not pending.future.done():
                pending.future.set_exception(MultiplexError("Response metadata does not match the request"))
            return
        if pending.request.is_expired():
            if not pending.future.done():
                pending.future.set_exception(RequestTimeoutError("Request deadline exceeded"))
            return
        if not pending.future.done():
            pending.future.set_result(envelope.payload["result"])

    def _handle_error(self, envelope: Envelope) -> None:
        pending = self._pop_pending(envelope.request_id)
        if pending is None:
            return
        if not self._response_matches(pending.request, envelope):
            if not pending.future.done():
                pending.future.set_exception(MultiplexError("Response metadata does not match the request"))
            return
        if pending.request.is_expired():
            if not pending.future.done():
                pending.future.set_exception(RequestTimeoutError("Request deadline exceeded"))
            return
        payload = envelope.payload
        if not pending.future.done():
            pending.future.set_exception(
                RemoteRequestError(
                    cast(str, payload["code"]),
                    cast(str, payload["message"]),
                    retryable=cast(bool, payload["retryable"]),
                )
            )

    @staticmethod
    def _response_matches(request: Envelope, response: Envelope) -> bool:
        return (
            response.direction is request.direction.opposite
            and response.runtime_id == request.runtime_id
            and response.session_id == request.session_id
            and response.request_id == request.request_id
            and response.deadline_ms == request.deadline_ms
        )

    async def _start_incoming_request(self, envelope: Envelope) -> None:
        if envelope.is_expired():
            await self._send_error(envelope, "deadline_exceeded", "Request deadline exceeded", retryable=True)
            return
        if envelope.request_id in self._inflight:
            return
        if len(self._inflight) >= self.max_inflight_requests:
            await self._send_error(envelope, "server_busy", "Runtime request limit reached", retryable=True)
            return
        task = asyncio.create_task(
            self._run_handler(envelope),
            name=f"browser-extension-request-{envelope.request_id}",
        )
        self._inflight[envelope.request_id] = _InflightRequest(request=envelope, task=task)

    async def _run_handler(self, envelope: Envelope) -> None:
        current = asyncio.current_task()
        try:
            if self.request_handler is None:
                await self._send_error(envelope, "unsupported_operation", "Operation is not supported")
                return
            remaining = max(0.0, (envelope.deadline_ms - int(time.time() * 1000)) / 1000)
            if remaining <= 0:
                await self._send_error(envelope, "deadline_exceeded", "Request deadline exceeded", retryable=True)
                return
            try:
                async with asyncio.timeout(remaining):
                    result = await self.request_handler(envelope)
            except TimeoutError:
                await self._send_error(envelope, "deadline_exceeded", "Request deadline exceeded", retryable=True)
                return
            except asyncio.CancelledError:
                return
            except Exception:
                await self._send_error(envelope, "internal_error", "Runtime operation failed")
                return
            item = self._inflight.get(envelope.request_id)
            if item is not None and item.task is current:
                if envelope.is_expired():
                    await self._send_error(envelope, "deadline_exceeded", "Request deadline exceeded", retryable=True)
                    return
                try:
                    response = Envelope.response_for(envelope, result)
                except ProtocolValidationError:
                    await self._send_error(envelope, "internal_error", "Runtime operation failed")
                    return
                remaining = max(0.0, (envelope.deadline_ms - int(time.time() * 1000)) / 1000)
                if remaining <= 0:
                    await self._send_error(envelope, "deadline_exceeded", "Request deadline exceeded", retryable=True)
                    return
                try:
                    async with asyncio.timeout(remaining):
                        await self.channel.write(response)
                except TimeoutError:
                    await self.close("Runtime response write exceeded its deadline")
        except RuntimeTransportError:
            await self.close("Runtime connection is closed")
        finally:
            item = self._inflight.get(envelope.request_id)
            if item is not None and item.task is current:
                self._inflight.pop(envelope.request_id, None)

    async def _send_error(
        self,
        request: Envelope,
        code: str,
        message: str,
        *,
        retryable: bool = False,
    ) -> None:
        if self.closed:
            return
        try:
            await self.channel.write(Envelope.error_for(request, code=code, message=message, retryable=retryable))
        except RuntimeTransportError:
            await self.close("Runtime connection is closed")

    def _handle_cancel(self, envelope: Envelope) -> None:
        item = self._inflight.get(envelope.request_id)
        if item is None:
            return
        request = item.request
        if (
            envelope.direction is request.direction
            and envelope.runtime_id == request.runtime_id
            and envelope.session_id == request.session_id
            and envelope.deadline_ms == request.deadline_ms
        ):
            self._inflight.pop(envelope.request_id, None)
            item.task.cancel()

    def _handle_event(self, envelope: Envelope) -> None:
        if self.event_handler is None or envelope.is_expired():
            return
        if len(self._event_tasks) >= self.max_inflight_requests:
            return
        task = asyncio.create_task(self._run_event_handler(envelope), name="browser-extension-event")
        self._event_tasks.add(task)
        task.add_done_callback(self._event_tasks.discard)

    async def _run_event_handler(self, envelope: Envelope) -> None:
        assert self.event_handler is not None
        try:
            await self.event_handler(envelope)
        except (asyncio.CancelledError, Exception):
            return
