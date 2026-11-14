"""Recorded observations of host-owned goals, never an execution authority."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from copy import deepcopy
import logging
from typing import Any, Literal
from uuid import uuid4

logger = logging.getLogger(__name__)

GoalReason = Literal["snapshot", "started", "replaced", "evaluated", "paused", "resumed", "completed", "cleared"]


class GoalPresentation:
    def __init__(self, emit: Callable[[dict[str, Any]], Awaitable[None]], *, source: Literal["tui", "ask"]) -> None:
        self._emit = emit
        self._source = source
        self._generation_id = uuid4().hex
        self._revision = -1
        self._evaluation_met: bool | None = None
        self._pending: deque[dict[str, Any]] = deque()
        self._worker: asyncio.Task[None] | None = None
        self._closed = False

    def publish(self, goal: dict[str, Any] | None, *, reason: GoalReason, evaluation_met: bool | None = None) -> None:
        if self._closed:
            return
        if reason in ("snapshot", "started", "replaced", "cleared"):
            self._evaluation_met = None
        elif reason == "evaluated":
            self._evaluation_met = evaluation_met
        self._revision += 1
        self._pending.append(
            {
                "generation_id": self._generation_id,
                "revision": self._revision,
                "source": self._source,
                "reason": reason,
                "goal": deepcopy(goal),
                "evaluation_met": self._evaluation_met,
            }
        )
        self._ensure_worker()

    def _ensure_worker(self) -> None:
        if self._worker is None and self._pending:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            self._worker = loop.create_task(self._run(), name="session-goal-presentation")

    async def _run(self) -> None:
        # Eager tasks must publish their owner before an immediate emitter finishes.
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass
        try:
            while self._pending:
                snapshot = self._pending.popleft()
                try:
                    operation = asyncio.ensure_future(self._emit(snapshot))
                    while not operation.done():
                        try:
                            await asyncio.shield(operation)
                        except asyncio.CancelledError:
                            pass
                    operation.result()
                except (Exception, asyncio.CancelledError):
                    logger.debug("Could not publish goal observation", exc_info=True)
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
