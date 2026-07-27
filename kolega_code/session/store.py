"""Contracts for durable, ordered session event storage.

Two concerns that were previously conflated are separated here:

* ``AgentConnectionManager`` (in :mod:`kolega_code.events`) is *live fan-out* —
  it delivers an event to whoever is currently connected and forgets it.
* ``SessionEventStore`` is *durability and ordering* — it assigns each event a
  monotonic ``seq`` and can replay any range later.

A frontend needs both: ``read``/``tail`` to catch up on history, then live
delivery to stay current. ``tail`` combines the two so no event falls into the
gap between "end of backlog" and "start of live".

Ordering is the store's responsibility, never the emitter's: a session may be
driven by several processes or workers, so a correct implementation assigns
``seq`` with an atomic operation (an append under a lock, a database increment)
rather than from a local counter.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import AsyncIterator, Collection, Optional

from kolega_code.events import AgentEvent, ArtifactRef


class SessionStoreError(RuntimeError):
    """Raised when stored events or artifacts are missing or untrustworthy."""


@dataclass(frozen=True)
class SessionEventMeta:
    """Summary of one session's event stream, cheap enough to list many at once."""

    session_id: str
    #: Highest assigned sequence number; 0 for a session with no events.
    last_seq: int
    event_count: int
    #: Timestamp of the first event, or None when empty.
    started_at: Optional[str]
    #: Timestamp of the most recent event, or None when empty.
    updated_at: Optional[str]
    #: ``elapsed_ms`` of the last event: the replay duration of the session.
    duration_ms: int
    #: "empty" | "open" (a turn is in progress) | "idle".
    status: str


class SessionEventStore(abc.ABC):
    """Append-only, ordered, replayable storage for a session's events."""

    @abc.abstractmethod
    async def append(self, event: AgentEvent) -> int:
        """Persist ``event``, assigning and returning its ``seq``.

        Must be atomic with respect to concurrent callers: two appends never
        receive the same ``seq``, and later appends always receive a higher one.

        Sequence numbers are *strictly increasing*, not necessarily contiguous.
        A store may share one sequence space with other records — the filesystem
        implementation shares the session journal's, so that a UI event and the
        provider message it accompanied can be ordered against each other — which
        leaves gaps in the numbers this store returns. Consumers must therefore
        treat ``seq`` as an opaque ordering key and never infer "the next event is
        seq + 1".

        Implementations set ``seq`` on the event in place so the caller can
        broadcast the same object it stored.
        """

    @abc.abstractmethod
    async def read(
        self,
        session_id: str,
        *,
        from_seq: int = 1,
        to_seq: Optional[int] = None,
        types: Optional[Collection[str]] = None,
    ) -> list[AgentEvent]:
        """Return events with ``from_seq <= seq <= to_seq`` in ascending order.

        ``types`` filters by ``event_type`` when given. A range beyond the end of
        the stream returns an empty list rather than raising.
        """

    @abc.abstractmethod
    def tail(self, session_id: str, *, from_seq: int = 1) -> AsyncIterator[AgentEvent]:
        """Yield the backlog from ``from_seq``, then follow live appends.

        The contract is *every seq exactly once, ascending*. An implementation
        must not drop an event that is appended while the backlog is being read,
        nor deliver one twice. Iteration ends only when the consumer stops or the
        store is closed, which is what lets a client reconnect at its last seen
        ``seq`` and lose nothing.
        """

    @abc.abstractmethod
    async def head(self, session_id: str) -> SessionEventMeta:
        """Return the stream summary, without reading every event where possible."""


class ArtifactStore(abc.ABC):
    """Content-addressed storage for payloads too large to inline in an event."""

    @abc.abstractmethod
    async def put(
        self,
        data: bytes,
        *,
        media_type: str,
        purpose: str,
        encoding: str,
        chars: Optional[int] = None,
    ) -> ArtifactRef:
        """Store ``data`` and return a reference addressed by its content hash.

        Storing identical bytes twice must yield the same reference; callers rely
        on that to deduplicate repeated payloads.
        """

    @abc.abstractmethod
    async def open(self, ref: ArtifactRef) -> bytes:
        """Return the referenced bytes, verifying them against ``ref.sha256``.

        Raises :class:`SessionStoreError` when the payload is missing or fails
        the integrity check; a silently wrong payload is worse than an error.
        """
