"""Route one session's event stream to per-kind subscribers.

The agent's connection manager is a single queue, but the gateway runs two
concurrent consumers per session: the turn renderer (``chat_message`` tool
activity) and the control relay (``control_requested``/``control_resolved``).
Shared queues with type-filtered consumers would let one steal the other's
events, so one router task owns the source and fans each event out to the
subscriber queues that asked for its kind.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from kolega_code.events import AgentEvent


class EventRouter:
    """Fan one ``AgentEvent`` source out to per-kind subscriber queues."""

    def __init__(self, source: asyncio.Queue[AgentEvent]) -> None:
        self._source = source
        self._subscribers: dict[str, list[asyncio.Queue[AgentEvent]]] = {}
        self._task: Optional[asyncio.Task[None]] = None

    def subscribe(self, *event_types: str) -> asyncio.Queue[AgentEvent]:
        """Return a queue receiving every event whose type is in ``event_types``."""
        queue: asyncio.Queue[AgentEvent] = asyncio.Queue()
        for event_type in event_types:
            self._subscribers.setdefault(event_type, []).append(queue)
        return queue

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="gateway-event-router")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _run(self) -> None:
        try:
            while True:
                event = await self._source.get()
                for queue in self._subscribers.get(event.event_type, []):
                    queue.put_nowait(event)
        except asyncio.CancelledError:
            raise
