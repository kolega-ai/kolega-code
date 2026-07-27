"""Filesystem-backed session event store for the CLI.

Presentation events share the session journal's single sequence space rather than
living in a parallel log. That is what makes the two halves of a session
correlatable: a tool result recorded for the model and the UI event that rendered
it are ordered against each other by construction, with no join to reconstruct.

Events are namespaced with a ``ui.`` prefix on the way in. The journal's message
reconstruction selects the four provider-facing types by name and ignores
everything else, so presentation events are invisible to history replay while
still occupying the same timeline.

Reads and tails parse ``events.jsonl`` incrementally rather than going through
``SessionJournal.read_events``, which re-reads and re-validates the whole file on
every call — fine for loading a session once, far too expensive for a follower
polling several times a second.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, AsyncIterator, Collection, Optional

from kolega_code.events import AgentEvent, ArtifactRef
from kolega_code.session.store import (
    ArtifactStore,
    SessionEventMeta,
    SessionEventStore,
    SessionStoreError,
)

from .session_journal import SessionJournal, SessionJournalError

#: Prefix marking a journal record as a presentation event.
UI_EVENT_PREFIX = "ui."

#: How often a follower checks for new records. Low enough to feel live, high
#: enough that idle followers cost nothing measurable.
TAIL_POLL_SECONDS = 0.15


class FileArtifactStore(ArtifactStore):
    """Content-addressed artifact storage backed by the session's artifact dir."""

    def __init__(self, journal: SessionJournal) -> None:
        self._journal = journal

    async def put(
        self,
        data: bytes,
        *,
        media_type: str,
        purpose: str,
        encoding: str,
        chars: Optional[int] = None,
    ) -> ArtifactRef:
        try:
            ref = await asyncio.to_thread(
                self._journal.put_artifact,
                data,
                media_type=media_type,
                purpose=purpose,
                encoding=encoding,
                chars=chars,
            )
        except SessionJournalError as exc:
            raise SessionStoreError(str(exc)) from exc
        return ArtifactRef(**ref)

    async def open(self, ref: ArtifactRef) -> bytes:
        try:
            return await asyncio.to_thread(self._journal.read_artifact, ref.model_dump(exclude_none=True))
        except SessionJournalError as exc:
            raise SessionStoreError(str(exc)) from exc


class FileSessionEventStore(SessionEventStore):
    """Session event store over the journal's append-only JSONL log."""

    def __init__(self, journal: SessionJournal) -> None:
        self._journal = journal
        self._session_id = journal.session_id
        self._events_path: Path = journal.events_path

    # -- Writing -----------------------------------------------------------

    async def append(self, event: AgentEvent) -> int:
        seq = await asyncio.to_thread(self._append_sync, event)
        event.seq = seq
        return seq

    def _append_sync(self, event: AgentEvent) -> int:
        try:
            stored = self._journal.append(
                UI_EVENT_PREFIX + event.event_type,
                actor=event.sender or "system",
                payload={"event": event.model_dump(mode="json")},
                artifacts=[ref.model_dump(exclude_none=True) for ref in event.artifacts],
            )
        except SessionJournalError as exc:
            raise SessionStoreError(str(exc)) from exc
        return stored.seq

    # -- Reading -----------------------------------------------------------

    async def read(
        self,
        session_id: str,
        *,
        from_seq: int = 1,
        to_seq: Optional[int] = None,
        types: Optional[Collection[str]] = None,
    ) -> list[AgentEvent]:
        self._require_session(session_id)
        records, _ = await asyncio.to_thread(self._scan, 0)
        return [
            event
            for seq, event in records
            if seq >= from_seq and (to_seq is None or seq <= to_seq) and (types is None or event.event_type in types)
        ]

    async def tail(self, session_id: str, *, from_seq: int = 1) -> AsyncIterator[AgentEvent]:
        self._require_session(session_id)
        offset = 0
        next_seq = max(1, from_seq)
        while True:
            records, offset = await asyncio.to_thread(self._scan, offset)
            delivered = False
            for seq, event in records:
                if seq < next_seq:
                    continue
                next_seq = seq + 1
                delivered = True
                yield event
            if not delivered:
                await asyncio.sleep(TAIL_POLL_SECONDS)

    async def head(self, session_id: str) -> SessionEventMeta:
        self._require_session(session_id)
        records, _ = await asyncio.to_thread(self._scan, 0)
        if not records:
            return SessionEventMeta(
                session_id=session_id,
                last_seq=0,
                event_count=0,
                started_at=None,
                updated_at=None,
                duration_ms=0,
                status="empty",
            )
        open_turns = 0
        for _, event in records:
            if event.event_type == "turn_started":
                open_turns += 1
            elif event.event_type == "turn_ended":
                open_turns = max(0, open_turns - 1)
        last_seq, last_event = records[-1]
        return SessionEventMeta(
            session_id=session_id,
            last_seq=last_seq,
            event_count=len(records),
            started_at=records[0][1].timestamp,
            updated_at=last_event.timestamp,
            duration_ms=last_event.elapsed_ms,
            status="open" if open_turns else "idle",
        )

    # -- Internals ---------------------------------------------------------

    def _require_session(self, session_id: str) -> None:
        if session_id != self._session_id:
            raise SessionStoreError(
                f"This store is bound to session {self._session_id}, not {session_id}",
            )

    def _scan(self, offset: int) -> tuple[list[tuple[int, AgentEvent]], int]:
        """Parse presentation events from ``offset``, returning them and the new offset.

        Only whole lines are consumed. A partially written final record is left
        for the next scan rather than treated as corruption, which is what lets a
        follower read a log while another process is appending to it.
        """
        try:
            size = os.path.getsize(self._events_path)
        except OSError:
            return [], offset
        if size <= offset:
            # Truncation means a different log (a new session dir reusing the
            # path); restart rather than silently misreport sequence numbers.
            return ([], 0) if size < offset else ([], offset)
        try:
            with open(self._events_path, "rb") as handle:
                handle.seek(offset)
                raw = handle.read(size - offset)
        except OSError as exc:
            raise SessionStoreError(f"Could not read session events: {self._events_path}") from exc

        consumed = raw.rfind(b"\n") + 1
        if consumed <= 0:
            return [], offset
        events: list[tuple[int, AgentEvent]] = []
        for line in raw[:consumed].splitlines():
            if not line.strip():
                continue
            event = self._decode(line)
            if event is not None:
                events.append(event)
        return events, offset + consumed

    def _decode(self, line: bytes) -> Optional[tuple[int, AgentEvent]]:
        try:
            record: Any = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # The journal validates its own log on load and repairs a torn tail.
            # A follower's job is to keep following, not to adjudicate integrity.
            return None
        if not isinstance(record, dict):
            return None
        event_type = record.get("type")
        if not isinstance(event_type, str) or not event_type.startswith(UI_EVENT_PREFIX):
            return None
        payload = record.get("payload")
        if not isinstance(payload, dict) or not isinstance(payload.get("event"), dict):
            return None
        seq = record.get("seq")
        if not isinstance(seq, int):
            return None
        try:
            event = AgentEvent.model_validate(payload["event"])
        except Exception:
            return None
        # The journal's sequence is authoritative; the embedded copy was written
        # before a number was assigned.
        event.seq = seq
        return seq, event
