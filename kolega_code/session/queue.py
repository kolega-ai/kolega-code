"""Read-only observations of host-owned, in-memory input queues.

Snapshots never admit, acknowledge, recover, or deliver an input. The host
captures them synchronously after its existing mutations; one owned worker
publishes them in order independently of the turn's cancellation owner.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from copy import deepcopy
import logging
from typing import Any, Literal
from uuid import uuid4

logger = logging.getLogger(__name__)

QueueReason = Literal["snapshot", "enqueued", "dequeued", "cleared", "removed"]


def queue_preview(text: str) -> str:
    preview = " ".join(text.split()) or "(empty)"
    return preview if len(preview) <= 120 else preview[:117] + "…"


def queue_item(
    queue_id: str, text: str, *, image_count: int = 0, origin: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Project only human-facing text and peer identity, never attachment data."""
    peer = None
    if origin is not None and origin.get("kind") == "peer":
        peer = {key: origin[key] for key in ("kind", "session_id", "title")}
    return {"queue_id": queue_id, "text": queue_preview(text), "image_count": image_count, "origin": peer}


class QueuePresentation:
    def __init__(
        self,
        emit: Callable[[dict[str, Any]], Awaitable[None]],
        *,
        source: Literal["tui", "ask"],
    ) -> None:
        self._emit = emit
        self._source = source
        self._queue_id = uuid4().hex
        self._revision = -1
        self._pending: deque[dict[str, Any]] = deque()
        self._worker: asyncio.Task[None] | None = None
        self._closed = False

    def publish(self, items: list[dict[str, Any]], *, reason: QueueReason) -> None:
        if self._closed:
            return
        self._revision += 1
        self._pending.append(
            {
                "queue_id": self._queue_id,
                "revision": self._revision,
                "source": self._source,
                "reason": reason,
                "items": deepcopy(items),
            }
        )
        self._ensure_worker()

    def _ensure_worker(self) -> None:
        if self._worker is None and self._pending:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # Synchronous host setup can capture observations before its
                # event loop exists. Startup/close will drain those as well.
                return
            self._worker = loop.create_task(self._run(), name="session-queue-presentation")

    async def _run(self) -> None:
        # Eager launch must assign the owner before an immediate emitter finishes.
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass
        try:
            while self._pending:
                snapshot = self._pending.popleft()
                try:
                    await self._emit(snapshot)
                except (Exception, asyncio.CancelledError):
                    # A failed observation must neither undo a queue mutation
                    # nor prevent subsequent observations from being attempted.
                    logger.debug("Could not publish queue observation", exc_info=True)
        finally:
            self._worker = None

    async def flush(self) -> None:
        cancelled = False
        while self._pending or self._worker is not None:
            self._ensure_worker()
            worker = self._worker
            assert worker is not None
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancelled = True
                if worker.cancelled() and self._worker is worker:
                    self._worker = None
        if cancelled:
            raise asyncio.CancelledError

    async def aclose(self) -> None:
        self._closed = True
        await self.flush()
