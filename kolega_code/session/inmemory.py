"""In-process implementations of the session store contracts.

These are the reference implementations: small enough to read as documentation
of the protocol, and the substrate for tests, ephemeral sessions, and hosts that
have not yet wired a database.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import AsyncIterator, Collection, Optional

from kolega_code.events import AgentEvent, ArtifactRef

from .store import ArtifactStore, SessionEventMeta, SessionEventStore, SessionStoreError


class InMemorySessionEventStore(SessionEventStore):
    """Reference store holding a session's events in a list.

    Ordering comes from an ``asyncio.Lock`` around sequence assignment plus the
    append itself, so concurrent emitters cannot interleave and produce a gap.
    """

    def __init__(self) -> None:
        self._events: dict[str, list[AgentEvent]] = {}
        self._lock = asyncio.Lock()
        # One condition per session so a tailing consumer wakes on append
        # without polling. Created lazily because sessions appear over time.
        self._waiters: dict[str, asyncio.Condition] = {}
        self._closed = False

    def _condition(self, session_id: str) -> asyncio.Condition:
        condition = self._waiters.get(session_id)
        if condition is None:
            condition = asyncio.Condition()
            self._waiters[session_id] = condition
        return condition

    async def append(self, event: AgentEvent) -> int:
        if not event.session_id:
            raise SessionStoreError("Cannot store an event without a session_id")
        async with self._lock:
            events = self._events.setdefault(event.session_id, [])
            event.seq = len(events) + 1
            events.append(event)
        condition = self._condition(event.session_id)
        async with condition:
            condition.notify_all()
        return event.seq

    async def read(
        self,
        session_id: str,
        *,
        from_seq: int = 1,
        to_seq: Optional[int] = None,
        types: Optional[Collection[str]] = None,
    ) -> list[AgentEvent]:
        events = self._events.get(session_id, [])
        selected = [
            event
            for event in events
            if event.seq is not None
            and event.seq >= from_seq
            and (to_seq is None or event.seq <= to_seq)
            and (types is None or event.event_type in types)
        ]
        return selected

    async def tail(self, session_id: str, *, from_seq: int = 1) -> AsyncIterator[AgentEvent]:
        next_seq = max(1, from_seq)
        condition = self._condition(session_id)
        while not self._closed:
            backlog = await self.read(session_id, from_seq=next_seq)
            if backlog:
                for event in backlog:
                    assert event.seq is not None
                    next_seq = event.seq + 1
                    yield event
                # Re-check for events appended while we were yielding before
                # sleeping, otherwise a fast producer can outrun the waiter.
                continue
            async with condition:
                try:
                    await asyncio.wait_for(condition.wait(), timeout=0.25)
                except asyncio.TimeoutError:
                    pass

    async def head(self, session_id: str) -> SessionEventMeta:
        events = self._events.get(session_id, [])
        if not events:
            return SessionEventMeta(
                session_id=session_id,
                last_seq=0,
                event_count=0,
                started_at=None,
                updated_at=None,
                duration_ms=0,
                status="empty",
            )
        last = events[-1]
        open_turns = 0
        for event in events:
            if event.event_type == "turn_started":
                open_turns += 1
            elif event.event_type == "turn_ended":
                open_turns = max(0, open_turns - 1)
        return SessionEventMeta(
            session_id=session_id,
            last_seq=last.seq or len(events),
            event_count=len(events),
            started_at=events[0].timestamp,
            updated_at=last.timestamp,
            duration_ms=last.elapsed_ms,
            status="open" if open_turns else "idle",
        )

    def close(self) -> None:
        """Stop all tailing iterators."""
        self._closed = True


class InMemoryArtifactStore(ArtifactStore):
    """Reference artifact store keyed by content hash."""

    def __init__(self) -> None:
        self._blobs: dict[str, bytes] = {}

    async def put(
        self,
        data: bytes,
        *,
        media_type: str,
        purpose: str,
        encoding: str,
        chars: Optional[int] = None,
    ) -> ArtifactRef:
        digest = hashlib.sha256(data).hexdigest()
        self._blobs[digest] = data
        return ArtifactRef(
            sha256=digest,
            bytes=len(data),
            media_type=media_type,
            purpose=purpose,
            encoding=encoding,
            chars=chars,
        )

    async def open(self, ref: ArtifactRef) -> bytes:
        data = self._blobs.get(ref.sha256)
        if data is None:
            raise SessionStoreError(f"Missing session artifact: {ref.sha256}")
        if hashlib.sha256(data).hexdigest() != ref.sha256:
            raise SessionStoreError(f"Session artifact failed integrity check: {ref.sha256}")
        return data
