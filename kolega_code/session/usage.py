from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextvars import copy_context
from copy import deepcopy
import logging
from typing import Any, TypeVar

from kolega_code.cli.session_journal import SessionEvent, SessionJournal
from kolega_code.cli.session_usage import derive_session_usage
from kolega_code.events import AgentEvent

logger = logging.getLogger(__name__)

_RELEVANT_EVENTS = frozenset(
    {
        "llm.run_started",
        "llm.request_started",
        "llm.request_settled",
        "llm.usage_observed",
        "llm.run_finished",
        "llm.message",
        "llm.request_failed",
        "assistant.message",
        "turn.started",
        "context.epoch_started",
    }
)
_COALESCE_SECONDS = 0.1
_T = TypeVar("_T")


class UsagePresentation:
    def __init__(
        self,
        journal: SessionJournal,
        broadcast: Callable[[AgentEvent, str, str], Awaitable[None]],
        *,
        workspace_id: str,
        thread_id: str,
        live_run_id: Callable[[], str | None] | None = None,
    ) -> None:
        self._journal = journal
        self._broadcast = broadcast
        self._workspace_id = workspace_id
        self._thread_id = thread_id
        self._live_run_id = live_run_id
        self._loop: asyncio.AbstractEventLoop | None = None
        self._worker: asyncio.Task[None] | None = None
        self._unsubscribe: Callable[[], None] | None = None
        self._wake = asyncio.Event()
        self._waiters: list[asyncio.Future[None]] = []
        self._immediate = False
        self._closing = False
        self._closed = False
        self._last_usage: dict[str, Any] | None = None

    async def start(self) -> None:
        # Startup cancellation must not finalize before the host drains accounting.
        operation = asyncio.create_task(self.flush())
        cancelled = False
        while not operation.done():
            try:
                await asyncio.shield(operation)
            except asyncio.CancelledError:
                cancelled = True
        operation.result()
        if cancelled:
            raise asyncio.CancelledError

    async def flush(self) -> None:
        if self._closing:
            await self.aclose()
            return
        self._ensure_worker()
        assert self._loop is not None
        waiter = self._loop.create_future()
        self._waiters.append(waiter)
        self._immediate = True
        self._wake.set()
        cancelled = False
        try:
            await asyncio.shield(waiter)
        except asyncio.CancelledError:
            cancelled = True
            self._begin_close()
        if self._closing:
            cancelled = await self._drain_worker() or cancelled
        if cancelled:
            raise asyncio.CancelledError

    async def aclose(self) -> None:
        self._begin_close()
        if await self._drain_worker():
            raise asyncio.CancelledError

    def _ensure_worker(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not loop:
            raise RuntimeError("UsagePresentation belongs to another event loop")
        self._loop = loop
        if self._worker is None and not self._closed:
            self._worker = loop.create_task(self._run(), name="session-usage-presentation")
            self._worker.add_done_callback(self._worker_done)

    def _begin_close(self) -> None:
        self._closing = True
        self._wake.set()

    def _on_event(self, event: SessionEvent) -> None:
        if event.event_type not in _RELEVANT_EVENTS or self._closing:
            return
        loop = self._loop
        if loop is not None:
            try:
                loop.call_soon_threadsafe(self._notify)
            except RuntimeError:
                logger.debug("Session usage publisher event loop is closed")

    def _notify(self) -> None:
        if not self._closing:
            self._wake.set()

    def _settle_waiters(self, waiters: list[asyncio.Future[None]]) -> None:
        for waiter in waiters:
            if not waiter.done():
                waiter.set_result(None)

    def _worker_done(self, worker: asyncio.Task[None]) -> None:
        if worker.cancelled():
            self._begin_close()
        elif worker.exception() is not None:
            logger.error("Session usage publisher stopped", exc_info=worker.exception())
            self._begin_close()
        self._settle_waiters(self._waiters)
        self._waiters.clear()

    async def _drain_worker(self) -> bool:
        cancelled = False
        while not self._closed:
            self._ensure_worker()
            assert self._worker is not None
            worker = self._worker
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancelled = True
                self._begin_close()
                if worker.cancelled():
                    self._worker = None
                continue
            if not self._closed:
                self._worker = None
        return cancelled

    async def _wait_owned(self, future: asyncio.Future[_T]) -> _T:
        while not future.done():
            try:
                await asyncio.shield(future)
            except asyncio.CancelledError:
                self._begin_close()
        return future.result()

    async def _io(self, function: Callable[..., _T], *args: Any) -> _T:
        context = copy_context()

        def run() -> _T:
            return context.run(function, *args)

        future = asyncio.get_running_loop().run_in_executor(None, run)
        return await self._wait_owned(future)

    async def _run(self) -> None:
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            self._begin_close()
        if not self._closing:
            try:
                self._unsubscribe = await self._io(self._journal.add_listener, self._on_event)
            except Exception:
                logger.exception("Could not subscribe to session usage updates")
        while True:
            try:
                await self._wake.wait()
                if not self._immediate and not self._closing:
                    await asyncio.sleep(_COALESCE_SECONDS)
            except asyncio.CancelledError:
                self._begin_close()
            self._wake.clear()
            self._immediate = False
            final = self._closing
            if final and self._unsubscribe is not None:
                await self._io(self._unsubscribe)
                self._unsubscribe = None
            waiters, self._waiters = self._waiters, []
            await self._refresh()
            self._settle_waiters(waiters)
            if final:
                self._closed = True
                return

    def _read_snapshot(self, live_run_id: str | None) -> tuple[int, dict[str, Any]]:
        events = [
            event for event in self._journal.read_events(repair_tail=False) if event.event_type in _RELEVANT_EVENTS
        ]
        return max((event.seq for event in events), default=0), derive_session_usage(events, live_run_id=live_run_id)

    async def _refresh(self) -> None:
        try:
            live_run_id = self._live_run_id() if self._live_run_id is not None and not self._closing else None
            source_seq, usage = await self._io(self._read_snapshot, live_run_id)
            if usage == self._last_usage:
                return
            last_usage = deepcopy(usage)
            event = AgentEvent(
                event_type="usage_updated",
                sender="agent",
                session_id=self._journal.session_id,
                workspace_id=self._workspace_id,
                thread_id=self._thread_id,
                sub_agent_info=None,
                content={"scope": "session", "semantics": "snapshot", "source_seq": source_seq, "usage": usage},
            )
            publication = asyncio.ensure_future(self._broadcast(event, self._workspace_id, self._thread_id))
            await self._wait_owned(publication)
            self._last_usage = last_usage
        except asyncio.CancelledError:
            self._begin_close()
            logger.warning("Session usage publication was cancelled")
        except Exception:
            logger.exception("Could not publish session usage; continuing")
