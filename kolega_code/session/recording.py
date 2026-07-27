"""Recording transport: gives live fan-out a durable, ordered counterpart.

Every event in the agent reaches the outside world through one method,
``AgentConnectionManager.broadcast_event``. Wrapping that single method is what
lets sequencing and persistence arrive without editing any of the ~20 places
that construct and emit events.

Persistence happens *before* fan-out. A client that sees ``seq`` 5 live must be
able to find ``seq`` 5 in the store afterwards, otherwise reconnecting at "the
last seq I saw" would skip an event.
"""

from __future__ import annotations

import asyncio
import base64
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from kolega_code.events import (
    AgentConnectionManager,
    AgentEvent,
    ArtifactPurpose,
    ArtifactRef,
    KnownEventType,
)

from .store import ArtifactStore, SessionEventStore

#: Content keys that may hold large text. Checked for artifact offload.
_TEXT_KEYS = ("text", "output", "content", "message", "summary", "diff")

#: Content key carrying images a tool produced, as ``{"media_type", "data"}``
#: entries with base64 ``data``. Consumed here and never persisted inline.
IMAGE_PAYLOAD_KEY = "images"


@dataclass(frozen=True)
class RetentionPolicy:
    """Bounds on what reaches durable storage.

    Streaming deltas arrive far too fast to persist one-for-one: a single verbose
    command can emit thousands of terminal chunks. Coalescing keeps the stored
    stream a faithful but much coarser recording — replaying it reproduces the
    same text, just in fewer, larger steps.
    """

    #: Flush a coalesced streaming checkpoint once this much text has accumulated.
    stream_checkpoint_chars: int = 8_000
    #: ...or once this long has passed, so slow streams still make progress.
    stream_checkpoint_ms: int = 2_000
    #: Event types coalesced even when not flagged ``is_streaming``. Terminal
    #: output is the motivating case: it arrives as thousands of independent
    #: non-streaming events, each with its own uuid, so nothing about an
    #: individual event marks it as part of a high-volume run.
    coalesce_types: frozenset[str] = frozenset(
        {
            KnownEventType.TERMINAL_OUTPUT,
            KnownEventType.TOOL_STREAMING_UPDATE,
        }
    )
    #: Offload a single text value larger than this to the artifact store.
    inline_payload_chars: int = 16_000
    #: Hard ceilings. On breach, recording stops and one truncation event is
    #: written, so a replay shows an explicit gap rather than silently ending.
    max_events: int = 200_000
    max_bytes: int = 256 * 1024 * 1024


@dataclass
class _StreamState:
    """Coalescing state for one in-flight run of related events."""

    #: The event most recently folded into this buffer, used as the template for
    #: the checkpoint record so it keeps the run's type, sender, and metadata.
    template: AgentEvent
    #: Pending text for append-mode runs. Dropping these would lose content, so
    #: they are concatenated into the next checkpoint instead.
    pending: list[str] = field(default_factory=list)
    #: Latest full text for replace-mode streams, where intermediates are
    #: genuinely redundant and can be discarded outright.
    latest: Optional[str] = None
    last_flush_ms: int = 0
    #: Whether this run began with an ``is_streaming`` event. Only then does a
    #: non-streaming event terminate the run; otherwise (terminal output) every
    #: event is non-streaming and would end the run immediately.
    streaming: bool = False
    #: Monotonic order in which this buffer was opened. A consumer places a
    #: segment in the transcript where its *first* record lands, so records have
    #: to be written in the order their runs began; see ``_flush_older_than``.
    open_index: int = 0


class RecordingConnectionManager(AgentConnectionManager):
    """Wrap a connection manager so everything it broadcasts is also recorded."""

    def __init__(
        self,
        inner: AgentConnectionManager,
        store: SessionEventStore,
        *,
        session_id: str,
        artifact_store: Optional[ArtifactStore] = None,
        policy: Optional[RetentionPolicy] = None,
    ) -> None:
        self.inner = inner
        self.store = store
        self.session_id = session_id
        self.artifact_store = artifact_store
        self.policy = policy or RetentionPolicy()
        self._streams: dict[tuple[str, str], _StreamState] = {}
        self._origin_monotonic: Optional[float] = None
        self._origin_offset_ms = 0
        self._events_written = 0
        self._bytes_written = 0
        self._open_counter = 0
        self._last_persisted_elapsed_ms = 0
        # Buffers are mutated by the broadcast path and by the idle checkpoint
        # ticker, and two writers interleaving would reorder or duplicate a
        # record. Everything that touches _streams or appends holds this.
        self._write_lock = asyncio.Lock()
        self._ticker: Optional[asyncio.Task[None]] = None
        self._retention_exceeded = False

    # -- AgentConnectionManager passthrough --------------------------------

    async def connect(
        self,
        websocket: Any,
        workspace_id: str,
        thread_id: str,
        connection_type: str,
        user_info: Any = None,
    ) -> None:
        await self.inner.connect(websocket, workspace_id, thread_id, connection_type, user_info)

    def disconnect(self, websocket: Any, workspace_id: str, thread_id: str, connection_type: str) -> None:
        self.inner.disconnect(websocket, workspace_id, thread_id, connection_type)

    def get_connection_count(self, workspace_id: str, thread_id: str) -> dict:
        return self.inner.get_connection_count(workspace_id, thread_id)

    # -- Recording ---------------------------------------------------------

    async def broadcast_event(self, event: AgentEvent, workspace_id: str, thread_id: str) -> None:
        self._stamp(event, workspace_id, thread_id)
        try:
            async with self._write_lock:
                await self._record(event)
        except Exception:
            # Recording is an enhancement to live delivery, never a precondition
            # for it. A failed store must not silence the running session's UI.
            pass
        await self.inner.broadcast_event(event, workspace_id, thread_id)

    def _stamp(self, event: AgentEvent, workspace_id: str, thread_id: str) -> None:
        event.session_id = event.session_id or self.session_id
        event.workspace_id = event.workspace_id or workspace_id
        event.thread_id = event.thread_id or thread_id
        event.elapsed_ms = self._elapsed_ms()

    def _elapsed_ms(self) -> int:
        """Milliseconds since this session's stream began.

        Uses a monotonic clock so a wall-clock adjustment cannot make a replay
        run backwards. On a resumed session the offset continues from the stored
        duration, which also collapses the idle gap between sittings instead of
        replaying it.
        """
        now = time.monotonic()
        if self._origin_monotonic is None:
            self._origin_monotonic = now
        return self._origin_offset_ms + int((now - self._origin_monotonic) * 1000)

    async def prime_elapsed_offset(self) -> None:
        """Continue ``elapsed_ms`` from an existing recording, for resumed sessions."""
        try:
            meta = await self.store.head(self.session_id)
        except Exception:
            return
        self._origin_offset_ms = meta.duration_ms
        self._origin_monotonic = time.monotonic()

    async def _record(self, event: AgentEvent) -> None:
        if self._retention_exceeded:
            return
        key = self._coalesce_key(event)
        if key is None:
            # Not part of a high-volume run. Flush what is buffered first so the
            # recording keeps the order in which things actually happened — this
            # is also what flushes a stream's tail, since every turn ends with a
            # non-coalescable turn_ended event.
            await self._flush_all()
            await self._persist(event)
            return
        await self._accumulate(key, event)

    def _coalesce_key(self, event: AgentEvent) -> Optional[tuple[str, str]]:
        """Return the buffer key grouping ``event`` with its run, or None if it stands alone.

        Choosing the grouping identity is the subtle part. Streaming segments
        share one ``uuid``, so that is their key. High-volume non-streaming types
        do not: every terminal output event carries a *fresh* uuid, so keying on
        it would open one buffer per event and coalesce nothing. Those group by an
        identifier from the payload instead, falling back to the type itself.
        """
        scope = ""
        if event.sub_agent_info:
            scope = str(
                event.sub_agent_info.get("dispatch_id") or event.sub_agent_info.get("agent_name") or "sub",
            )
        if event.event_type in self.policy.coalesce_types:
            for candidate in ("tool_call_id", "terminal_id", "browser_id"):
                value = event.content.get(candidate)
                if isinstance(value, str) and value:
                    scope = f"{scope}|{value}"
                    break
            return (event.event_type, scope)

        key = (event.event_type, f"{scope}|{event.uuid}")
        if event.is_streaming:
            return key
        # The final, non-streaming delta of an open streaming segment belongs
        # with the run it completes.
        return key if key in self._streams else None

    async def _accumulate(self, key: tuple[str, str], event: AgentEvent) -> None:
        state = self._streams.get(key)
        if state is None:
            self._open_counter += 1
            state = _StreamState(
                template=event,
                last_flush_ms=event.elapsed_ms,
                streaming=event.is_streaming,
                open_index=self._open_counter,
            )
            self._streams[key] = state
            self._ensure_ticker()
        else:
            state.template = event

        replace_mode = str(event.content.get("stream_mode") or "") == "replace"
        text = self._event_text(event)
        if replace_mode:
            state.latest = text
        elif text:
            state.pending.append(text)

        if state.streaming and not event.is_streaming:
            # End of a streaming segment: emit whatever is buffered as one record.
            await self._flush(key)
            return

        pending_chars = len(state.latest or "") if replace_mode else sum(len(part) for part in state.pending)
        if pending_chars == 0:
            return
        due_by_size = pending_chars >= self.policy.stream_checkpoint_chars
        due_by_time = event.elapsed_ms - state.last_flush_ms >= self.policy.stream_checkpoint_ms
        if due_by_size or due_by_time:
            await self._flush(key, keep_open=True)

    async def _flush(self, key: tuple[str, str], *, keep_open: bool = False) -> None:
        """Persist one buffer's accumulated text, after anything that began earlier.

        Reasoning and prose stream concurrently and the prose segment reliably
        ends first, so writing this buffer straight away would put the reply
        ahead of the thinking that led to it: a consumer positions a segment
        where its *first* record lands, and both the exported replay and any live
        client then show the two swapped.
        """
        state = self._streams.get(key)
        if state is None:
            return
        for older_key in self._keys_opened_before(state.open_index):
            # keep_open matters: closing a run here would end the segment as far
            # as a consumer is concerned, and its remaining deltas would open a
            # second transcript entry instead of continuing the first.
            await self._write_checkpoint(older_key, keep_open=True)
        await self._write_checkpoint(key, keep_open=keep_open)

    def _keys_opened_before(self, open_index: int) -> list[tuple[str, str]]:
        """Buffer keys for runs that began before ``open_index``, oldest first."""
        return [
            key
            for _, key in sorted(
                (state.open_index, key) for key, state in self._streams.items() if state.open_index < open_index
            )
        ]

    async def _write_checkpoint(self, key: tuple[str, str], *, keep_open: bool) -> None:
        """Write one buffer's accumulated text, with no regard for ordering."""
        state = self._streams.get(key)
        if state is None:
            return
        checkpoint = state.latest if state.latest is not None else "".join(state.pending)
        state.pending.clear()
        state.latest = None
        if not keep_open:
            del self._streams[key]
        if not checkpoint:
            return
        state.last_flush_ms = state.template.elapsed_ms
        await self._persist(self._with_text(state.template, checkpoint))

    async def _flush_all(self) -> None:
        # Insertion order is open order, so a run that began first is written
        # first; _write_checkpoint skips the per-flush ordering pass that would
        # otherwise re-walk the whole map for every buffer.
        for key in list(self._streams):
            await self._write_checkpoint(key, keep_open=False)

    def _ensure_ticker(self) -> None:
        """Start the idle checkpoint loop if it is not already running.

        ``due_by_time`` is only ever evaluated when the *next* delta of a run
        arrives, so a segment that stalls — a model thinking for twenty seconds,
        a command producing nothing — stays buffered and invisible. Anyone
        watching a shared link sees a frozen transcript for that whole time.
        """
        if self._ticker is not None and not self._ticker.done():
            return
        try:
            self._ticker = asyncio.get_running_loop().create_task(self._checkpoint_idle_buffers())
        except RuntimeError:
            # No loop (synchronous test harness): checkpoints still happen on
            # the next delta, which is the behaviour without this loop at all.
            self._ticker = None

    async def _checkpoint_idle_buffers(self) -> None:
        interval = max(self.policy.stream_checkpoint_ms, 1) / 1000
        try:
            while True:
                await asyncio.sleep(interval)
                async with self._write_lock:
                    if not self._streams:
                        return
                    now = self._elapsed_ms()
                    stale = [
                        key
                        for key, state in self._streams.items()
                        if (state.pending or state.latest is not None)
                        and now - state.last_flush_ms >= self.policy.stream_checkpoint_ms
                    ]
                    # _flush, not _write_checkpoint: a stalled younger run must
                    # not overtake an older one that simply has not gone stale.
                    for key in stale:
                        await self._flush(key, keep_open=True)
        except asyncio.CancelledError:
            raise
        except Exception:
            # A recording loop must never take the session down with it.
            return

    def _stop_ticker(self) -> None:
        ticker = self._ticker
        self._ticker = None
        if ticker is not None and not ticker.done():
            ticker.cancel()

    async def flush(self) -> None:
        """Persist every buffered run. Call before shutdown so no tail is lost."""
        self._stop_ticker()
        try:
            async with self._write_lock:
                await self._flush_all()
        except Exception:
            pass

    def _event_text(self, event: AgentEvent) -> str:
        for candidate in _TEXT_KEYS:
            value = event.content.get(candidate)
            if isinstance(value, str):
                return value
        return ""

    def _with_text(self, event: AgentEvent, text: str) -> AgentEvent:
        """Copy ``event`` with its text payload replaced by ``text``."""
        clone = event.model_copy(deep=True)
        for candidate in _TEXT_KEYS:
            if isinstance(event.content.get(candidate), str):
                clone.content[candidate] = text
                break
        # The stored copy is a distinct record; it must not inherit the live
        # event's identity or a consumer would treat them as the same delta.
        clone.seq = None
        return clone

    async def _persist(self, event: AgentEvent) -> None:
        stored = await self._offload_large_payloads(event)
        # A coalesced record carries the elapsed_ms of the delta it was built
        # from, which can predate a record already written — buffers do not end
        # in the order they began. Every consumer treats elapsed_ms as a
        # non-decreasing timeline (the player binary-searches it), so clamp
        # rather than hand them a clock that runs backwards.
        if stored.elapsed_ms < self._last_persisted_elapsed_ms:
            stored.elapsed_ms = self._last_persisted_elapsed_ms
        self._last_persisted_elapsed_ms = stored.elapsed_ms
        size = len(stored.model_dump_json())
        if self._events_written + 1 > self.policy.max_events or self._bytes_written + size > self.policy.max_bytes:
            await self._record_truncation()
            return
        await self.store.append(stored)
        self._events_written += 1
        self._bytes_written += size
        # The live event carries the seq of its stored counterpart when they are
        # the same record, so a client can correlate what it saw with the store.
        if stored is event:
            return
        if event.seq is None and stored.seq is not None and not event.is_streaming:
            event.seq = stored.seq

    async def _record_truncation(self) -> None:
        self._retention_exceeded = True
        notice = AgentEvent(
            sender="system",
            event_type=KnownEventType.STREAM_TRUNCATED,
            session_id=self.session_id,
            elapsed_ms=self._elapsed_ms(),
            content={
                "reason": "retention_limit",
                "events_written": self._events_written,
                "bytes_written": self._bytes_written,
                "max_events": self.policy.max_events,
                "max_bytes": self.policy.max_bytes,
            },
        )
        try:
            await self.store.append(notice)
        except Exception:
            pass

    async def _offload_large_payloads(self, event: AgentEvent) -> AgentEvent:
        """Move oversized text and image payloads out of the event body.

        Images are why a replay can show a screenshot at all. They reach the
        model as content blocks on a history record, which no presentation
        client reads, so a tool that returns one has to hand the bytes over here
        as well; this is where they become an artifact the event can point at.
        """
        if self.artifact_store is None:
            return event
        images = event.content.get(IMAGE_PAYLOAD_KEY)
        oversized = {
            key: value
            for key, value in event.content.items()
            if isinstance(value, str) and len(value) > self.policy.inline_payload_chars
        }
        if not oversized and not images:
            return event
        clone = event.model_copy(deep=True)
        if images:
            await self._offload_images(clone)
        refs: list[ArtifactRef] = list(clone.artifacts)
        for key, value in oversized.items():
            try:
                ref = await self.artifact_store.put(
                    value.encode("utf-8"),
                    media_type="text/plain; charset=utf-8",
                    purpose="tool_result",
                    encoding="utf-8",
                    chars=len(value),
                )
            except Exception:
                continue
            clone.content[key] = _preview(value, self.policy.inline_payload_chars, ref)
            refs.append(ref)
        clone.artifacts = refs
        return clone

    async def _offload_images(self, clone: AgentEvent) -> None:
        """Turn ``content['images']`` into image artifacts and drop the payload.

        The base64 never reaches the store inline: it would bloat every read of
        the event stream for bytes that only the artifact endpoint serves.
        """
        assert self.artifact_store is not None
        payload = clone.content.pop(IMAGE_PAYLOAD_KEY, None)
        if not isinstance(payload, list):
            return
        refs: list[ArtifactRef] = list(clone.artifacts)
        for item in payload:
            if not isinstance(item, dict):
                continue
            encoded = item.get("data")
            if not isinstance(encoded, str) or not encoded:
                continue
            try:
                data = base64.b64decode(encoded, validate=True)
            except Exception:
                # A malformed image must not cost the event its text.
                continue
            try:
                refs.append(
                    await self.artifact_store.put(
                        data,
                        media_type=str(item.get("media_type") or "application/octet-stream"),
                        purpose=ArtifactPurpose.IMAGE,
                        encoding="base64",
                    )
                )
            except Exception:
                continue
        clone.artifacts = refs


def _preview(value: str, budget: int, ref: ArtifactRef) -> str:
    """Head+tail preview naming the artifact that holds the full text."""
    marker = f"\n\n[… {ref.chars:,} characters elided; full payload sha256 {ref.sha256} …]\n\n"
    available = max(0, budget - len(marker))
    head = available // 2
    tail = available - head
    return value[:head] + marker + (value[-tail:] if tail else "")
