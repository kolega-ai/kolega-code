"""Journal persistence for LLM usage: internal messages and the lifetime aggregate.

Two halves, both keyed to the append-only session journal:

- :class:`SessionUsageSink` — a :class:`~kolega_code.llm.ledger.UsageLedger`
  observer that persists every settled non-history response as an
  ``llm.message`` event (full prepared Message + request/run ids + origin
  attribution) and every failed invocation as ``llm.request_failed``. Top-level
  main-loop responses are skipped: the session recorder already journals them
  as ``assistant.message`` (with ``Message.usage`` inside), and the HISTORY
  origin marks exactly those calls — so no response is ever written twice.
  An ``llm.run_started`` marker written at attach time is the accounting-start
  boundary: turns journaled before the first marker predate accounting and are
  reported as uncovered history rather than as zero usage.

- :func:`derive_session_usage` — the load-time fold that turns journaled
  messages and failure events into the lifetime ``SessionRecord.usage``
  aggregate. It scans every epoch and ignores rewinds: the journal is
  append-only and provider billing is never reversed.

Like the presentation stream (see kolega_code/session/recording.py), usage
persistence is an enhancement, never a precondition: sink failures are counted
on the ledger (``persist_failures``) and can never fail or re-issue a paid
request. History-bearing events keep their stricter, terminal failure
semantics through the session recorder.

The ``llm.*`` events share the journal's single sequence space and are invisible
to history replay by construction (``SessionStore._replay`` selects provider
history by type) and to share export (which reads only ``ui.*`` events).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from kolega_code.llm.ledger import SettledRequest, UsageLedger
from kolega_code.llm.usage import NormalizedUsage

from .session_journal import SessionEvent, SessionJournal, SessionRecorder

logger = logging.getLogger(__name__)

LLM_MESSAGE_EVENT = "llm.message"
LLM_REQUEST_FAILED_EVENT = "llm.request_failed"
LLM_RUN_STARTED_EVENT = "llm.run_started"

_CLOSE_TIMEOUT_SECONDS = 10.0

_TOKEN_SUM_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cache_read_input_tokens",
    "cache_write_input_tokens",
    "reasoning_output_tokens",
)


@dataclass(frozen=True, slots=True)
class _PendingEvent:
    kind: str  # "message" | "failure"
    turn_id: Optional[str]
    payload: dict[str, Any]
    message_dict: Optional[dict[str, Any]]


class SessionUsageSink:
    """Ledger observer that journals settled LLM requests off the event loop.

    Notification is synchronous on the event loop and does only cheap work: a
    ``message.to_dict()`` snapshot (the agent mutates the Message object after
    settlement) plus a queue put. A single drainer task performs the blocking
    journal writes via ``asyncio.to_thread`` in FIFO order, so journal order
    matches settlement order and the journal's lock serializes us against the
    session recorder.
    """

    def __init__(self, journal: SessionJournal, recorder: SessionRecorder, ledger: UsageLedger, *, mode: str) -> None:
        self._journal = journal
        self._recorder = recorder
        self._ledger = ledger
        self._mode = mode
        self._queue: asyncio.Queue[Optional[_PendingEvent]] = asyncio.Queue()
        self._drainer: Optional[asyncio.Task[None]] = None
        self._closed = False

    async def start(self) -> None:
        """Write the accounting-start marker and begin draining.

        The marker is written synchronously (awaited) so it lands before any
        turn this run can journal — coverage boundaries are exact. A marker
        failure is counted but does not prevent attaching: missing coverage
        only ever under-reports, never fabricates.
        """
        try:
            await asyncio.to_thread(
                self._journal.append,
                LLM_RUN_STARTED_EVENT,
                actor="system",
                payload={"run_id": self._ledger.run_id, "mode": self._mode},
            )
        except Exception:
            self._ledger.note_persist_failure()
            logger.exception("Failed to write the usage accounting-start marker")
        self._drainer = asyncio.create_task(self._drain())

    # -- ledger observer protocol (synchronous, event loop) -----------------

    def on_response(self, settled: SettledRequest, message: Any) -> None:
        if self._closed:
            return
        if settled.origin is not None and settled.origin.kind == "history":
            return  # journaled as assistant.message by the session recorder
        message_dict: Optional[dict[str, Any]] = None
        if message is not None:
            to_dict = getattr(message, "to_dict", None)
            if callable(to_dict):
                result = to_dict()
                if isinstance(result, dict):
                    message_dict = result
        self._queue.put_nowait(
            _PendingEvent(
                kind="message",
                turn_id=self._recorder.current_turn_id,
                payload=self._base_payload(settled),
                message_dict=message_dict,
            )
        )

    def on_failure(self, settled: SettledRequest) -> None:
        if self._closed:
            return
        # HISTORY failures are journaled too: a failed invocation never produces
        # an assistant.message, so there is nothing to duplicate.
        payload = self._base_payload(settled)
        payload["error"] = settled.error or "unknown error"
        self._queue.put_nowait(
            _PendingEvent(kind="failure", turn_id=self._recorder.current_turn_id, payload=payload, message_dict=None)
        )

    async def aclose(self) -> None:
        """Drain queued events and stop; bounded so quit never hangs on I/O."""
        if self._closed:
            return
        self._closed = True
        self._queue.put_nowait(None)
        if self._drainer is None:
            return
        try:
            await asyncio.wait_for(self._drainer, timeout=_CLOSE_TIMEOUT_SECONDS)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            self._drainer.cancel()

    # -- internals ----------------------------------------------------------

    def _base_payload(self, settled: SettledRequest) -> dict[str, Any]:
        origin = settled.origin.to_payload() if settled.origin is not None else {"kind": "unknown"}
        return {
            "request_id": settled.request_id,
            "run_id": settled.run_id,
            "provider": settled.provider,
            "model": settled.model,
            "origin": origin,
        }

    async def _drain(self) -> None:
        while True:
            item = await self._queue.get()
            if item is None:
                return
            try:
                await asyncio.to_thread(self._write, item)
            except Exception:
                self._ledger.note_persist_failure()
                logger.exception("Failed to journal a usage event; continuing")

    def _write(self, item: _PendingEvent) -> None:
        if item.kind == "message":
            prepared: Optional[dict[str, Any]] = None
            refs: list[dict[str, Any]] = []
            if item.message_dict is not None:
                prepared, refs = self._journal.prepare_message(item.message_dict)
            self._journal.append(
                LLM_MESSAGE_EVENT,
                actor="assistant",
                payload={**item.payload, "message": prepared},
                turn_id=item.turn_id,
                artifacts=refs,
            )
        else:
            self._journal.append(
                LLM_REQUEST_FAILED_EVENT,
                actor="system",
                payload=item.payload,
                turn_id=item.turn_id,
            )


def derive_session_usage(events: Iterable[SessionEvent]) -> dict[str, Any]:
    """Fold journaled usage into the lifetime session aggregate.

    Scans every epoch and ignores rewinds — provider billing is never reversed
    by editing history. Reads payload dicts directly (never Message.from_dict,
    which rejects unknown content blocks) and parses usage only through the
    tolerant NormalizedUsage.from_dict.

    Coverage: turns journaled before the first ``llm.run_started`` marker
    predate accounting; their top-level usage is deliberately excluded from the
    sums (their nested calls are unrecoverable, and a blended total would read
    as authoritative) and surfaced as ``coverage.pre_accounting_turns``.
    """
    event_list = list(events)
    marker_seqs = [event.seq for event in event_list if event.event_type == LLM_RUN_STARTED_EVENT]
    first_marker = min(marker_seqs) if marker_seqs else None

    counts = {"requests": 0, "responses": 0, "reported": 0, "unreported": 0, "failed": 0}
    sums = {name: 0 for name in _TOKEN_SUM_FIELDS}
    detail = {
        "reported_missing_cache_read": 0,
        "reported_missing_cache_write": 0,
        "reported_missing_reasoning": 0,
    }
    rows: dict[tuple[str, str], dict[str, int]] = {}
    pre_accounting_turns = 0

    def row_for(provider: str, model: str) -> dict[str, int]:
        return rows.setdefault((provider, model), {**{k: 0 for k in counts}, **{k: 0 for k in sums}})

    def fold_response(payload: dict[str, Any]) -> None:
        message = payload.get("message")
        usage_dict = message.get("usage") if isinstance(message, dict) else None
        usage = NormalizedUsage.from_dict(usage_dict)
        if usage is not None and usage.reported:
            key = (usage.provider, usage.model or "unknown")
            row = row_for(*key)
            for target in (counts, row):
                target["responses"] += 1
                target["reported"] += 1
            for name in _TOKEN_SUM_FIELDS:
                value = getattr(usage, name) or 0
                sums[name] += value
                row[name] += value
            if usage.cache_read_input_tokens is None:
                detail["reported_missing_cache_read"] += 1
            if usage.cache_write_input_tokens is None:
                detail["reported_missing_cache_write"] += 1
            if usage.reasoning_output_tokens is None:
                detail["reported_missing_reasoning"] += 1
        else:
            key = (str(payload.get("provider") or "unknown"), str(payload.get("model") or "unknown"))
            row = row_for(*key)
            for target in (counts, row):
                target["responses"] += 1
                target["unreported"] += 1

    for event in event_list:
        if event.event_type == LLM_MESSAGE_EVENT:
            fold_response(event.payload)
        elif event.event_type == LLM_REQUEST_FAILED_EVENT:
            key = (str(event.payload.get("provider") or "unknown"), str(event.payload.get("model") or "unknown"))
            row = row_for(*key)
            for target in (counts, row):
                target["failed"] += 1
        elif event.event_type == "assistant.message":
            if first_marker is not None and event.seq > first_marker:
                fold_response(event.payload)
        elif event.event_type == "turn.started":
            if first_marker is None or event.seq < first_marker:
                pre_accounting_turns += 1

    counts["requests"] = counts["responses"] + counts["failed"]
    for row in rows.values():
        row["requests"] = row["responses"] + row["failed"]

    breakdown = [{"provider": provider, "model": model, **row} for (provider, model), row in sorted(rows.items())]
    return {
        **counts,
        **sums,
        "breakdown": breakdown,
        "coverage": {
            "accounted_runs": len(marker_seqs),
            "pre_accounting_turns": pre_accounting_turns,
            "full": bool(marker_seqs) and pre_accounting_turns == 0,
        },
        "detail": detail,
    }
