"""Durable request facts and a deduplicated, all-epochs lifetime usage fold."""

from __future__ import annotations

import asyncio
import copy
from contextlib import contextmanager
from contextvars import ContextVar
from decimal import Decimal, InvalidOperation
import logging
from dataclasses import dataclass
from typing import Any, Generator, Iterable, Optional

from kolega_code.llm.ledger import SettledRequest, UsageLedger

from .session_journal import SessionEvent, SessionJournal, SessionRecorder

logger = logging.getLogger(__name__)

LLM_MESSAGE_EVENT = "llm.message"
LLM_REQUEST_FAILED_EVENT = "llm.request_failed"
LLM_RUN_STARTED_EVENT = "llm.run_started"
LLM_REQUEST_STARTED_EVENT = "llm.request_started"
LLM_REQUEST_SETTLED_EVENT = "llm.request_settled"
LLM_USAGE_OBSERVED_EVENT = "llm.usage_observed"
LLM_RUN_FINISHED_EVENT = "llm.run_finished"

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
    kind: str
    recorder: SessionRecorder
    turn_id: Optional[str]
    epoch_id: str
    payload: dict[str, Any]
    message_dict: Optional[dict[str, Any]] = None


@dataclass(frozen=True, slots=True)
class _UsageEventScope:
    recorder: SessionRecorder
    turn_id: Optional[str]


@dataclass(frozen=True, slots=True)
class _RequestOwner:
    recorder: SessionRecorder
    turn_id: Optional[str]
    epoch_id: str


_usage_event_scope: ContextVar[Optional[_UsageEventScope]] = ContextVar("session_usage_scope", default=None)


def current_session_usage_scope() -> Optional[_UsageEventScope]:
    """Return the task-local journaling override for usage events, if any."""
    return _usage_event_scope.get()


@contextmanager
def session_usage_scope(recorder: SessionRecorder, *, turn_id: Optional[str] = None) -> Generator[None]:
    """Override usage-event ownership for work running in this task tree.

    ``turn_id`` defaults to the current scope's captured turn when nested, so a
    background child can preserve the parent's launch turn while narrower child
    scopes replace only the owning agent stamp.
    """
    previous = _usage_event_scope.get()
    resolved_turn_id = turn_id if turn_id is not None else (previous.turn_id if previous is not None else None)
    token = _usage_event_scope.set(_UsageEventScope(recorder=recorder, turn_id=resolved_turn_id))
    try:
        yield
    finally:
        try:
            _usage_event_scope.reset(token)
        except ValueError:
            _usage_event_scope.set(previous)


class SessionUsageSink:
    """Ledger observer that journals settled LLM requests off the event loop.

    Notification is synchronous on the event loop and does only cheap work: a
    ``message.to_dict()`` snapshot (the agent mutates the Message object after
    settlement) plus a queue put. A single drainer task performs the blocking
    journal writes via ``asyncio.to_thread`` in FIFO order, so journal order
    matches settlement order and the journal's lock serializes us against the
    session recorder.
    """

    def __init__(
        self,
        journal: SessionJournal,
        recorder: SessionRecorder,
        ledger: UsageLedger,
        *,
        mode: str,
        run_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self._journal = journal
        self._recorder = recorder
        self._ledger = ledger
        self._mode = mode
        self._run_metadata = run_metadata
        self._queue: asyncio.Queue[Optional[_PendingEvent]] = asyncio.Queue()
        self._drainer: Optional[asyncio.Task[None]] = None
        self._closed = False
        self._owners: dict[tuple[str, str], _RequestOwner] = {}
        self._revisions: dict[tuple[str, str], int] = {}
        self._evidence: dict[tuple[str, str], dict[str, Any]] = {}
        self._settled: set[tuple[str, str]] = set()
        self._flush_timeouts = 0

    async def start(self) -> None:
        """Write the accounting-start marker and begin draining.

        The marker is written synchronously (awaited) so it lands before any
        turn this run can journal — coverage boundaries are exact. A marker
        failure is counted but does not prevent attaching: missing coverage
        only ever under-reports, never fabricates.
        """
        if self._drainer is not None or self._closed:
            return
        try:
            await asyncio.to_thread(
                self._journal.append,
                LLM_RUN_STARTED_EVENT,
                actor="system",
                payload={
                    **(self._run_metadata or {}),
                    "run_id": self._ledger.run_id,
                    "mode": self._mode,
                    "accounting_version": 2,
                },
            )
        except Exception:
            self._ledger.note_persist_failure()
            logger.exception("Failed to write the usage accounting-start marker")
        self._drainer = asyncio.create_task(self._drain())

    def on_begin(self, request: SettledRequest) -> None:
        if self._closed:
            return
        key = (request.run_id, request.request_id)
        if key in self._owners:
            return
        self._owners[key] = self._capture_owner()
        self._enqueue(LLM_REQUEST_STARTED_EVENT, request, self._base_payload(request))

    def on_response(self, settled: SettledRequest, message: Any) -> None:
        if self._closed or not self._settle(settled, "responded"):
            return
        if settled.origin is not None and settled.origin.kind == "history":
            return
        message_dict: Optional[dict[str, Any]] = None
        if message is not None:
            to_dict = getattr(message, "to_dict", None)
            if callable(to_dict):
                result = to_dict()
                if isinstance(result, dict):
                    message_dict = result
        self._enqueue(LLM_MESSAGE_EVENT, settled, self._base_payload(settled), message_dict)

    def on_failure(self, settled: SettledRequest) -> None:
        if self._closed or not self._settle(settled, "failed"):
            return
        payload = self._base_payload(settled)
        payload["error"] = settled.error or "unknown error"
        self._enqueue(LLM_REQUEST_FAILED_EVENT, settled, payload)

    def on_observation(self, request: SettledRequest) -> None:
        if self._closed:
            return
        key = (request.run_id, request.request_id)
        if key not in self._settled:
            return
        payload = self._base_payload(request)
        revision = _revision(payload)
        if revision <= self._revisions.get(key, -1):
            return
        self._revisions[key] = revision
        evidence = {name: payload.get(name) for name in ("usage", "observation", "cost")}
        if evidence == self._evidence.get(key):
            return
        self._evidence[key] = copy.deepcopy(evidence)
        self._enqueue(LLM_USAGE_OBSERVED_EVENT, request, payload)

    async def flush(self, *, timeout: float = _CLOSE_TIMEOUT_SECONDS) -> bool:
        if self._drainer is None:
            return self._queue.empty()
        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            self._flush_timeouts += 1
            self._ledger.note_persist_failure()
            return False

    async def aclose(self) -> None:
        """Call after request producers drain; leave an unclosed marker on timeout."""
        if self._closed:
            return
        self._closed = True
        if self._drainer is None:
            return
        try:
            await self._finish_run()
        finally:
            if not self._drainer.done():
                self._drainer.cancel()

    async def _finish_run(self) -> None:
        assert self._drainer is not None
        if not await self.flush():
            self._drainer.cancel()
            return
        snapshot = self._ledger.snapshot()
        payload = {
            "run_id": self._ledger.run_id,
            "accounting_version": 2,
            "mode": self._mode,
            "counts": {
                name: getattr(snapshot, name, 0)
                for name in ("requests", "responses", "failed", "open", "abandoned", "partial", "missing")
            },
            "persist_failures": self._ledger.persist_failures,
            "flush_timeouts": self._flush_timeouts,
        }
        self._queue.put_nowait(
            _PendingEvent(
                LLM_RUN_FINISHED_EVENT,
                self._recorder,
                None,
                self._journal.epoch_id,
                payload,
            )
        )
        if not await self.flush():
            self._drainer.cancel()
            return
        self._queue.put_nowait(None)
        try:
            await asyncio.wait_for(asyncio.shield(self._drainer), timeout=_CLOSE_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            self._drainer.cancel()

    def _capture_owner(self) -> _RequestOwner:
        scope = current_session_usage_scope()
        recorder = scope.recorder if scope is not None else self._recorder
        turn_id = scope.turn_id if scope is not None else recorder.current_turn_id
        return _RequestOwner(recorder, turn_id, recorder.journal.epoch_id)

    def _enqueue(
        self,
        kind: str,
        request: SettledRequest,
        payload: dict[str, Any],
        message_dict: Optional[dict[str, Any]] = None,
    ) -> None:
        key = (request.run_id, request.request_id)
        owner = self._owners.setdefault(key, self._capture_owner())
        self._queue.put_nowait(
            _PendingEvent(
                kind, owner.recorder, owner.turn_id, owner.epoch_id, copy.deepcopy(payload), copy.deepcopy(message_dict)
            )
        )

    def _settle(self, request: SettledRequest, state: str) -> bool:
        key = (request.run_id, request.request_id)
        if key in self._settled:
            return False
        self._settled.add(key)
        payload = self._base_payload(request)
        payload["state"] = _state(payload.get("state"), state)
        self._revisions[key] = _revision(payload)
        self._evidence[key] = copy.deepcopy({name: payload.get(name) for name in ("usage", "observation", "cost")})
        self._enqueue(LLM_REQUEST_SETTLED_EVENT, request, payload)
        return True

    def _base_payload(self, settled: SettledRequest) -> dict[str, Any]:
        origin = settled.origin.to_payload() if settled.origin is not None else {"kind": "unknown"}
        to_payload = getattr(settled, "to_payload", None)
        serialized = to_payload() if callable(to_payload) else None
        payload: dict[str, Any] = dict(serialized) if isinstance(serialized, dict) else {}
        payload.update(
            {
                "request_id": settled.request_id,
                "run_id": settled.run_id,
                "provider": settled.provider,
                "model": settled.model,
                "origin": origin,
            }
        )
        for name in ("prompt_id", "state", "revision", "observation", "cost", "pinned_pricing", "started_at"):
            value = getattr(settled, name, None)
            if name not in payload and value is not None:
                serializer = getattr(value, "to_payload", None) or getattr(value, "to_dict", None)
                payload[name] = serializer() if callable(serializer) else getattr(value, "value", value)
        payload.setdefault("revision", 0)
        if "usage" not in payload:
            payload["usage"] = settled.usage.to_dict() if settled.usage is not None else None
        return payload

    async def _drain(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                if item is None:
                    return
                await asyncio.to_thread(self._write, item)
            except Exception:
                self._ledger.note_persist_failure()
                logger.exception("Failed to journal a usage event; continuing")
            finally:
                self._queue.task_done()

    def _write(self, item: _PendingEvent) -> None:
        if item.kind == LLM_MESSAGE_EVENT:
            prepared: Optional[dict[str, Any]] = None
            refs: list[dict[str, Any]] = []
            if item.message_dict is not None:
                prepared, refs = item.recorder.journal.prepare_message(item.message_dict)
            if prepared is not None:
                # The ledger's call-identity stamp duplicates this event's own
                # top-level request_id; keep the journaled message clean.
                metadata = prepared.get("usage_metadata")
                if isinstance(metadata, dict):
                    metadata.pop("llm_call", None)
            item.recorder.journal.append(
                LLM_MESSAGE_EVENT,
                actor="assistant",
                payload={**item.payload, "message": prepared},
                turn_id=item.turn_id,
                epoch_id=item.epoch_id,
                artifacts=refs,
                agent=item.recorder.agent_stamp,
            )
        else:
            item.recorder.journal.append(
                item.kind,
                actor="system",
                payload=item.payload,
                turn_id=item.turn_id,
                epoch_id=item.epoch_id,
                agent=item.recorder.agent_stamp,
            )


def _revision(payload: dict[str, Any]) -> int:
    value = payload.get("revision", 0)
    return value if type(value) is int and value >= 0 else -1


def _state(value: Any, default: str = "open") -> str:
    value = getattr(value, "value", value)
    return value if value in ("open", "responded", "failed", "cancelled", "abandoned") else default


def _identity(payload: dict[str, Any]) -> Optional[tuple[str, str]]:
    message = payload.get("message")
    metadata = message.get("usage_metadata") if isinstance(message, dict) else None
    call = metadata.get("llm_call") if isinstance(metadata, dict) else None
    call = call if isinstance(call, dict) else {}
    request_id = payload.get("request_id") or payload.get("llm_call_id") or call.get("llm_call_id")
    run_id = payload.get("run_id") or call.get("run_id") or ""
    if not isinstance(request_id, str) or not request_id or not isinstance(run_id, str):
        return None
    return run_id, request_id


def _token_evidence(payload: dict[str, Any]) -> tuple[dict[str, Optional[int]], dict[str, Any]]:
    observation = payload.get("observation")
    message = payload.get("message")
    usage = payload.get("usage")
    if isinstance(observation, dict):
        raw = observation.get("usage", observation)
    elif isinstance(usage, dict):
        raw = usage
    else:
        raw = message.get("usage") if isinstance(message, dict) else {}
    raw = raw if isinstance(raw, dict) else {}
    if raw.get("reported") is False:
        raw = {}
    values = {name: value if type(value := raw.get(name)) is int and value >= 0 else None for name in _TOKEN_SUM_FIELDS}
    inp, out = values["input_tokens"], values["output_tokens"]
    if inp is not None and out is not None:
        values["total_tokens"] = inp + out
    return values, raw


def _amount(value: Any) -> Optional[Decimal]:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        return None
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        return None
    return result if result.is_finite() and result >= 0 else None


@dataclass
class _FoldedRequest:
    payload: dict[str, Any]
    state: str = "open"
    revision: int = -1
    terminal: bool = False


def derive_session_usage(events: Iterable[SessionEvent], *, live_run_id: Optional[str] = None) -> dict[str, Any]:
    """Fold immutable recorded evidence, never current prices or replayed history."""
    event_list = list({event.event_id: event for event in events}.values())
    facts: dict[tuple[str, str], _FoldedRequest] = {}
    runs: dict[str, dict[str, Any]] = {}
    finished: dict[str, dict[str, Any]] = {}
    malformed = 0
    fact_types = (LLM_REQUEST_STARTED_EVENT, LLM_REQUEST_SETTLED_EVENT, LLM_USAGE_OBSERVED_EVENT)
    fact_turns: set[tuple[str, Optional[str]]] = set()
    for event in event_list:
        payload = event.payload
        if event.event_type in (LLM_RUN_STARTED_EVENT, LLM_RUN_FINISHED_EVENT):
            run_id = payload.get("run_id")
            if not isinstance(run_id, str) or not run_id:
                malformed += 1
                continue
            if event.event_type == LLM_RUN_FINISHED_EVENT:
                run_counts = payload.get("counts")
                expected = run_counts.get("requests") if isinstance(run_counts, dict) else None
                failures = payload.get("persist_failures", 0)
                if type(expected) is not int or expected < 0 or type(failures) is not int or failures < 0:
                    malformed += 1
                    continue
            target = runs if event.event_type == LLM_RUN_STARTED_EVENT else finished
            target[run_id] = payload
        elif event.event_type in fact_types:
            key = _identity(payload)
            revision = _revision(payload)
            if key is None or not key[0] or revision < 0:
                malformed += 1
                continue
            fact_turns.add((event.epoch_id, event.turn_id))
            request = facts.setdefault(key, _FoldedRequest(payload))
            if event.event_type == LLM_REQUEST_SETTLED_EVENT and not request.terminal:
                request.state = _state(payload.get("state"), "abandoned")
                if request.state == "open" or payload.get("state") not in (
                    "responded",
                    "failed",
                    "cancelled",
                    "abandoned",
                ):
                    malformed += 1
                    request.state = "abandoned"
                request.terminal = True
            if revision >= request.revision:
                if "usage" in payload and "observation" not in payload:
                    request.payload = {**request.payload, "observation": None}
                request.payload = {**request.payload, **payload}
                request.revision = revision

    counts = {
        name: 0
        for name in (
            "requests",
            "responses",
            "reported",
            "unreported",
            "failed",
            "open",
            "abandoned",
            "partial",
            "missing",
        )
    }
    sums = {name: 0 for name in _TOKEN_SUM_FIELDS}
    known = {name: 0 for name in _TOKEN_SUM_FIELDS}
    detail = {
        "reported_missing_cache_read": 0,
        "reported_missing_cache_write": 0,
        "reported_missing_reasoning": 0,
    }
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    money: dict[str, dict[str, Any]] = {}
    credits: dict[str, dict[str, Any]] = {}
    cost_counts = {"reported": 0, "estimated": 0, "missing": 0, "partial": 0}

    def fold_cost(
        payload: dict[str, Any],
        cost_counts: dict[str, int],
        money: dict[str, dict[str, Any]],
        credits: dict[str, dict[str, Any]],
    ) -> None:
        cost = payload.get("cost")
        if not isinstance(cost, dict):
            observation = payload.get("observation")
            cost = observation.get("cost") if isinstance(observation, dict) else None
        if not isinstance(cost, dict):
            cost_counts["missing"] += 1
            return
        source = cost.get("source")
        amount = _amount(cost.get("amount"))
        subtotal_only = amount is None and source == "estimated"
        if subtotal_only:
            amount = _amount(cost.get("known_subtotal"))
        denomination = cost.get("currency") or cost.get("denomination")
        if (
            source not in ("provider_reported", "estimated")
            or amount is None
            or not isinstance(denomination, str)
            or not denomination
        ):
            cost_counts["missing"] += 1
            return
        non_monetary = bool(
            cost.get("kind") in ("credits", "credit")
            or cost.get("unit") in ("credits", "credit")
            or cost.get("denomination_kind") in ("credits", "credit")
            or cost.get("denomination") in ("credits", "credit", "provider_credits")
            or not cost.get("currency")
            or denomination.lower() in ("credits", "credit")
        )
        if non_monetary and isinstance(cost.get("denomination"), str):
            denomination = cost["denomination"]
        target = credits if non_monetary else money
        row = target.setdefault(
            denomination,
            {
                "reported": Decimal(0),
                "estimated": Decimal(0),
                "requests": 0,
                "reported_requests": 0,
                "estimated_requests": 0,
            },
        )
        row["reported" if source == "provider_reported" else "estimated"] += amount
        row["requests"] += 1
        row["reported_requests" if source == "provider_reported" else "estimated_requests"] += 1
        token_subtotal = _amount(cost.get("token_cost_subtotal"))
        if token_subtotal is not None:
            row["token_cost_subtotal"] = row.get("token_cost_subtotal", Decimal(0)) + token_subtotal
        cost_counts["reported" if source == "provider_reported" else "estimated"] += 1
        if (
            subtotal_only
            or cost.get("complete") is not True
            or cost.get("missing_components")
            or cost.get("unavailable_reasons")
        ):
            cost_counts["partial"] += 1

    def fold(payload: dict[str, Any], state: str) -> None:
        values, raw = _token_evidence(payload)
        provider = str(raw.get("provider") or payload.get("provider") or "unknown")
        model = str(raw.get("model") or payload.get("model") or "unknown")
        row = rows.setdefault(
            (provider, model),
            {
                **{k: 0 for k in counts},
                **{k: 0 for k in sums},
                "known_requests": {name: 0 for name in _TOKEN_SUM_FIELDS},
                "cost_counts": {name: 0 for name in cost_counts},
                "money": {},
                "credits": {},
            },
        )
        complete = raw.get("complete") is not False and all(
            values[name] is not None for name in ("input_tokens", "output_tokens", "total_tokens")
        )
        partial = not complete and any(value is not None for value in values.values())
        outcome = {
            "responded": "responses",
            "failed": "failed",
            "cancelled": "failed",
            "open": "open",
            "abandoned": "abandoned",
        }[state]
        for target in (counts, row):
            target["requests"] += 1
            target[outcome] += 1
            target["reported"] += int(complete)
            target["unreported"] += int(state == "responded" and not complete)
            target["partial"] += int(partial)
            target["missing"] += int(not complete and not partial)
        for name, value in values.items():
            if value is not None:
                sums[name] += value
                row[name] += value
                known[name] += 1
                row["known_requests"][name] += 1
        if complete:
            for name, counter in (
                ("cache_read_input_tokens", "reported_missing_cache_read"),
                ("cache_write_input_tokens", "reported_missing_cache_write"),
                ("reasoning_output_tokens", "reported_missing_reasoning"),
            ):
                detail[counter] += int(values[name] is None)
        fold_cost(payload, cost_counts, money, credits)
        fold_cost(payload, row["cost_counts"], row["money"], row["credits"])

    for (run_id, _), request in facts.items():
        state = request.state if request.terminal else ("open" if run_id == live_run_id else "abandoned")
        fold(request.payload, state)

    legacy: dict[tuple[str, str], _FoldedRequest] = {}
    legacy_identities: set[tuple[str, str]] = set()
    legacy_events = 0
    uncovered_assistant_turns: set[tuple[str, Optional[str]]] = set()
    fact_ids = {request_id for _, request_id in facts}
    call_runs: dict[str, set[str]] = {}
    for event in event_list:
        if (identity := _identity(event.payload)) is not None and identity[0]:
            call_runs.setdefault(identity[1], set()).add(identity[0])
    ambiguous_legacy_ids: set[str] = set()
    for event in event_list:
        if event.event_type not in (LLM_MESSAGE_EVENT, LLM_REQUEST_FAILED_EVENT, "assistant.message"):
            continue
        payload = event.payload
        if payload.get("origin_type") == "synthetic":
            continue
        key = _identity(payload)
        if key is not None and not key[0]:
            candidates = call_runs.get(key[1], set())
            if len(candidates) == 1:
                key = (next(iter(candidates)), key[1])
            elif len(candidates) > 1:
                ambiguous_legacy_ids.add(key[1])
        if key in facts or (key is not None and not key[0] and key[1] in fact_ids):
            continue
        legacy_events += 1
        if event.event_type == "assistant.message":
            uncovered_assistant_turns.add((event.epoch_id, event.turn_id))
        if key is None:
            message_id = payload.get("message_id")
            key = ("legacy_message" if message_id else "legacy_event", str(message_id or event.event_id))
            legacy_identities.add(key)
        state = "failed" if event.event_type == LLM_REQUEST_FAILED_EVENT else "responded"
        previous = legacy.get(key)
        if previous is None:
            legacy[key] = _FoldedRequest(payload, state)
        else:
            values, _ = _token_evidence(payload)
            if any(value is not None for value in values.values()):
                previous.payload = payload
            if state == "failed":
                previous.state = state
    for request in legacy.values():
        fold(request.payload, request.state)

    active_run: Optional[str] = None
    pre_accounting_turns = 0
    uninstrumented_turns = 0
    legacy_turns = 0
    seen_marker = False
    for event in event_list:
        if event.event_type == LLM_RUN_STARTED_EVENT:
            run_id = event.payload.get("run_id")
            active_run = run_id if isinstance(run_id, str) and run_id in runs else None
            seen_marker = True
        elif (
            event.event_type == LLM_RUN_FINISHED_EVENT
            and active_run in finished
            and event.payload.get("run_id") == active_run
        ):
            active_run = None
        elif event.event_type == "turn.started":
            if not seen_marker:
                pre_accounting_turns += 1
            if active_run is None:
                uninstrumented_turns += 1
            elif runs[active_run].get("accounting_version") != 2:
                legacy_turns += 1
    unclosed_runs = sum(
        payload.get("accounting_version") == 2 and run_id not in finished and run_id != live_run_id
        for run_id, payload in runs.items()
    )
    orphan_runs = len({run_id for run_id, _ in facts}.difference(runs))
    persist_failures = sum(
        value for payload in finished.values() if type(value := payload.get("persist_failures")) is int and value > 0
    )
    lost_requests = 0
    for run_id, payload in finished.items():
        run_counts = payload.get("counts")
        expected = run_counts.get("requests") if isinstance(run_counts, dict) else None
        if type(expected) is int:
            actual = sum(key[0] == run_id for key in facts)
            lost_requests += max(0, expected - actual)
    coverage = {
        "accounted_runs": len(runs),
        "pre_accounting_turns": pre_accounting_turns,
        "uninstrumented_turns": uninstrumented_turns,
        "legacy_turns": legacy_turns,
        "legacy_requests": len(legacy),
        "legacy_identity_requests": len(legacy_identities),
        "legacy_events": legacy_events,
        "ambiguous_legacy_ids": len(ambiguous_legacy_ids),
        "uncovered_assistant_turns": len(uncovered_assistant_turns.difference(fact_turns)),
        "unclosed_runs": unclosed_runs,
        "orphan_runs": orphan_runs,
        "malformed_facts": malformed,
        "lost_requests": lost_requests,
    }
    coverage["full"] = (
        (bool(runs) or counts["requests"] == 0)
        and not persist_failures
        and not any(value for name, value in coverage.items() if name != "accounted_runs")
    )
    for payload in runs.values():
        if payload.get("accounting_version") != 2:
            coverage["full"] = False

    def cost_rows(source: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {
            denomination: {
                "reported": str(row["reported"]),
                "estimated": str(row["estimated"]),
                "total": str(row["reported"] + row["estimated"]),
                "requests": row["requests"],
                "reported_requests": row["reported_requests"],
                "estimated_requests": row["estimated_requests"],
                **({"token_cost_subtotal": str(row["token_cost_subtotal"])} if "token_cost_subtotal" in row else {}),
            }
            for denomination, row in sorted(source.items())
        }

    def cost_payload(
        cost_counts: dict[str, int],
        money: dict[str, dict[str, Any]],
        credits: dict[str, dict[str, Any]],
        request_counts: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            **cost_counts,
            "missing_requests": cost_counts["missing"],
            "partial_requests": cost_counts["partial"],
            "currencies": cost_rows(money),
            "non_monetary": cost_rows(credits),
            "complete": (
                bool(coverage["full"])
                and not (
                    cost_counts["missing"]
                    or cost_counts["partial"]
                    or request_counts["open"]
                    or request_counts["abandoned"]
                )
            ),
        }

    costs = cost_payload(cost_counts, money, credits, counts)
    breakdown = []
    for (provider, model), row in sorted(rows.items()):
        row_costs = cost_payload(row.pop("cost_counts"), row.pop("money"), row.pop("credits"), row)
        breakdown.append({"provider": provider, "model": model, **row, "costs": row_costs})
    persistence = {
        "persist_failures": persist_failures,
        "unclosed_runs": unclosed_runs,
        "lost_requests": lost_requests,
        "healthy": not (persist_failures or malformed or lost_requests or unclosed_runs or orphan_runs),
    }
    return {
        **counts,
        **sums,
        "breakdown": breakdown,
        "coverage": coverage,
        "detail": detail,
        "known_fields": known,
        "complete": bool(coverage["full"])
        and not (counts["partial"] or counts["missing"] or counts["open"] or counts["abandoned"]),
        "known_requests": known,
        "detail_complete": {name: known[name] == counts["requests"] for name in _TOKEN_SUM_FIELDS[3:]},
        "costs": costs,
        "persist_failures": persist_failures,
        "persistence": persistence,
    }
