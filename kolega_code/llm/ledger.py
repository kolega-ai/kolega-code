"""Request-owned usage facts, captured on the single asyncio event loop.

Terminal outcomes settle once; later evidence refines the same identity.
SDK-internal retries remain invisible. Accounting failures never retry inference.
"""

from __future__ import annotations

import logging
import uuid
import weakref
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Awaitable, Callable, Generator, Literal, Optional, Protocol, cast

from .accounting import TOKEN_FIELDS, UsageObservation, freeze_json, json_payload, nonnegative_decimal, observe_usage

from .usage import NormalizedUsage

logger = logging.getLogger(__name__)

_ERROR_TRUNCATION = 500

_TOKEN_SUM_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cache_read_input_tokens",
    "cache_write_input_tokens",
    "reasoning_output_tokens",
)

_COUNT_FIELDS = ("requests", "responses", "reported", "unreported", "failed", "open")


def describe_error(error: BaseException | type[BaseException] | None) -> str:
    """One-line error description for a failed invocation record."""
    if error is None:
        return "unknown error"
    if isinstance(error, type):
        return error.__name__
    return f"{type(error).__name__}: {error}"[:_ERROR_TRUNCATION]


@dataclass(frozen=True, slots=True)
class LlmCallOrigin:
    """Who made an LLM call, for persistence attribution.

    ``kind`` values: "history" (the top-level agent's main loop, whose response
    is recorded as ``assistant.message`` by the session recorder), "sub_agent",
    "helper" (a named internal helper client), and "primary" (a top-level loop
    with no session recorder, e.g. unsaved ask runs).
    """

    kind: str
    agent_name: Optional[str] = None
    agent_id: Optional[str] = None
    parent_tool_call_id: Optional[str] = None
    depth: Optional[int] = None
    helper: Optional[str] = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"kind": self.kind}
        for name in ("agent_name", "agent_id", "parent_tool_call_id", "depth", "helper"):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        return payload


HISTORY_ORIGIN = LlmCallOrigin(kind="history")


def helper_origin(name: str) -> LlmCallOrigin:
    return LlmCallOrigin(kind="helper", helper=name)


_llm_call_origin: ContextVar[Optional[LlmCallOrigin]] = ContextVar("llm_call_origin", default=None)


def current_llm_call_origin() -> Optional[LlmCallOrigin]:
    return _llm_call_origin.get()


@contextmanager
def llm_call_origin(origin: LlmCallOrigin) -> Generator[None]:
    """Attribute LLM calls begun within this scope (and child tasks) to ``origin``.

    The scope may span the yields of an async generator (the main agent loop
    holds it across streaming). Two consequences: consumers observe the origin
    between chunks (any code that makes its own LLM calls mid-stream must set
    its own shadowing origin, as every current helper does), and teardown can
    run in a different Context than entry — ``aclose()`` driven by another task
    or the loop's ``shutdown_asyncgens`` — where ``reset(token)`` raises, so the
    reset falls back to restoring the previously observed value.
    """
    previous = _llm_call_origin.get()
    token = _llm_call_origin.set(origin)
    try:
        yield
    finally:
        try:
            _llm_call_origin.reset(token)
        except ValueError:
            _llm_call_origin.set(previous)


_logical_prompt_id: ContextVar[str | None] = ContextVar("logical_prompt_id", default=None)


def current_logical_prompt_id() -> str | None:
    return _logical_prompt_id.get()


@contextmanager
def logical_prompt_scope(prompt_id: str | None) -> Generator[None, None, None]:
    previous = _logical_prompt_id.get()
    token = _logical_prompt_id.set(prompt_id)
    try:
        yield
    finally:
        try:
            _logical_prompt_id.reset(token)
        except ValueError:
            _logical_prompt_id.set(previous)


class _RequestState(Enum):
    OPEN = "open"
    RESPONDED = "responded"
    FAILED = "failed"
    ABANDONED = "abandoned"


@dataclass(slots=True)
class _RequestRecord:
    provider: str
    model: Optional[str]
    origin: Optional[LlmCallOrigin] = None
    state: _RequestState = _RequestState.OPEN
    usage: Optional[NormalizedUsage] = None
    error: Optional[str] = None
    prompt_id: str | None = None
    revision: int = 0
    observation: UsageObservation | None = None
    cost: dict[str, Any] = field(default_factory=dict)
    pricing_context: dict[str, Any] = field(default_factory=dict)
    pinned_pricing: dict[str, Any] = field(default_factory=dict)
    started_at: str = ""


@dataclass(frozen=True, slots=True)
class SettledRequest:
    """Immutable handoff to a persistence observer at settlement time."""

    request_id: str
    run_id: str
    provider: str
    model: Optional[str]
    origin: Optional[LlmCallOrigin]
    usage: Optional[NormalizedUsage]
    error: Optional[str] = None
    prompt_id: str | None = None
    state: str = "responded"
    revision: int = 0
    observation: UsageObservation | None = None
    cost: dict[str, Any] = field(default_factory=dict)
    pricing_context: dict[str, Any] = field(default_factory=dict)
    pinned_pricing: dict[str, Any] = field(default_factory=dict)
    started_at: str = ""

    def __post_init__(self) -> None:
        for name in ("cost", "pricing_context", "pinned_pricing"):
            object.__setattr__(self, name, freeze_json(getattr(self, name)))

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "request_id": self.request_id,
            "run_id": self.run_id,
            "provider": self.provider,
            "model": self.model,
            "origin": self.origin.to_payload() if self.origin else None,
            "usage": self.usage.to_dict() if self.usage else None,
            "error": self.error,
            "prompt_id": self.prompt_id,
            "state": self.state,
            "revision": self.revision,
            "observation": self.observation.to_dict() if self.observation else None,
            "cost": json_payload(self.cost),
            "pricing_context": json_payload(self.pricing_context),
            "pinned_pricing": json_payload(self.pinned_pricing),
            "started_at": self.started_at,
        }


class RequestBeginObserver(Protocol):
    """Optional persistence hook; legacy response/failure sinks need not implement it."""

    def on_begin(self, request: SettledRequest) -> None: ...


class RequestObservationObserver(Protocol):
    """Optional hook for genuine evidence refinement after a terminal outcome."""

    def on_observation(self, request: SettledRequest) -> None: ...


@dataclass(frozen=True, slots=True)
class ProviderModelUsage:
    """Aggregate for one (provider, model) key within a snapshot."""

    provider: str
    model: str
    requests: int = 0
    responses: int = 0
    reported: int = 0
    unreported: int = 0
    failed: int = 0
    open: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    reasoning_output_tokens: int = 0
    abandoned: int = 0
    missing: int = 0
    partial: int = 0
    known_requests: dict[str, int] = field(default_factory=dict)
    costs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "known_requests", freeze_json(self.known_requests))
        object.__setattr__(self, "costs", freeze_json(self.costs))

    @property
    def complete(self) -> bool:
        return (
            self.open == 0
            and self.missing == 0
            and self.partial == 0
            and (self.requests == 0 or self.known_requests.get("total_tokens", 0) == self.requests)
        )


@dataclass(frozen=True, slots=True)
class UsageSnapshot:
    """Inclusive known subtotals; ``known_requests`` distinguishes unknown from zero."""

    run_id: str
    requests: int = 0
    responses: int = 0
    reported: int = 0
    unreported: int = 0
    failed: int = 0
    open: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    reasoning_output_tokens: int = 0
    # Failed persistence-observer notifications/writes. Health of the journal
    # sink, not of accounting itself — deliberately outside `complete`.
    persist_failures: int = 0
    breakdown: tuple[ProviderModelUsage, ...] = field(default=())
    revision: int = 0
    prompt_id: str | None = None
    abandoned: int = 0
    missing: int = 0
    partial: int = 0
    known_requests: dict[str, int] = field(default_factory=dict)
    costs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "known_requests", freeze_json(self.known_requests))
        object.__setattr__(self, "costs", freeze_json(self.costs))

    @property
    def complete(self) -> bool:
        """True iff every begun request settled with authoritative usage."""
        return (
            self.open == 0
            and self.missing == 0
            and self.partial == 0
            and (self.requests == 0 or self.known_requests.get("total_tokens", 0) == self.requests)
        )

    @property
    def detail_complete(self) -> dict[str, bool]:
        return {name: self.known_requests.get(name, 0) == self.requests for name in _TOKEN_SUM_FIELDS[3:]}

    def to_payload(self) -> dict[str, Any]:
        def tokens(value: Any) -> dict[str, Any]:
            return {
                name: getattr(value, name) if value.requests == 0 or value.known_requests.get(name, 0) else None
                for name in _TOKEN_SUM_FIELDS
            }

        return {
            "run_id": self.run_id,
            "revision": self.revision,
            "prompt_id": self.prompt_id,
            **{name: getattr(self, name) for name in _COUNT_FIELDS + ("abandoned", "missing", "partial")},
            **tokens(self),
            "complete": self.complete,
            "known_requests": json_payload(self.known_requests),
            "detail_complete": self.detail_complete,
            "costs": json_payload(self.costs),
            "persist_failures": self.persist_failures,
            "breakdown": [
                {
                    "provider": row.provider,
                    "model": row.model,
                    **{name: getattr(row, name) for name in _COUNT_FIELDS + ("abandoned", "missing", "partial")},
                    **tokens(row),
                    "complete": row.complete,
                    "known_requests": json_payload(row.known_requests),
                    "costs": json_payload(row.costs),
                }
                for row in self.breakdown
            ],
        }

    def since(self, earlier: Optional["UsageSnapshot"]) -> "UsageSnapshot":
        """Checkpoint delta relative to an earlier snapshot of the same run.

        Monotonic counts and sums subtract; ``open`` is the CURRENT outstanding
        count (not a difference — a request open at both points is still open).
        ``earlier`` of None or from another run (run_id mismatch) yields self.

        This legacy arithmetic view is not a request-owned interval: late usage
        can refine requests already present at the checkpoint. Costs therefore
        remain absent here. Use ``snapshot_for_prompt`` for spend/attribution.
        """
        if earlier is None or earlier.run_id != self.run_id:
            return self

        deltas = {name: getattr(self, name) - getattr(earlier, name) for name in _COUNT_FIELDS}
        deltas["open"] = self.open
        deltas["persist_failures"] = self.persist_failures - earlier.persist_failures
        sums = {name: getattr(self, name) - getattr(earlier, name) for name in _TOKEN_SUM_FIELDS}

        earlier_rows = {(row.provider, row.model): row for row in earlier.breakdown}
        rows = []
        for row in self.breakdown:
            base = earlier_rows.get((row.provider, row.model))
            if base is None:
                rows.append(row)
                continue
            delta_row = ProviderModelUsage(
                provider=row.provider,
                model=row.model,
                **{name: getattr(row, name) - getattr(base, name) for name in _COUNT_FIELDS if name != "open"},
                open=row.open,
                **{name: getattr(row, name) - getattr(base, name) for name in _TOKEN_SUM_FIELDS},
                abandoned=row.abandoned - base.abandoned,
                missing=max(row.missing - base.missing, 0),
                partial=max(row.partial - base.partial, 0),
                known_requests={
                    name: row.known_requests.get(name, 0) - base.known_requests.get(name, 0) for name in TOKEN_FIELDS
                },
            )
            if any(
                getattr(delta_row, name)
                for name in _COUNT_FIELDS + _TOKEN_SUM_FIELDS + ("abandoned", "missing", "partial")
            ) or any(delta_row.known_requests.values()):
                rows.append(delta_row)

        return UsageSnapshot(
            run_id=self.run_id,
            **deltas,
            **sums,
            breakdown=tuple(rows),
            revision=self.revision,
            prompt_id=self.prompt_id,
            abandoned=self.abandoned - earlier.abandoned,
            missing=max(self.missing - earlier.missing, 0),
            partial=max(self.partial - earlier.partial, 0),
            known_requests=freeze_json(
                {name: self.known_requests.get(name, 0) - earlier.known_requests.get(name, 0) for name in TOKEN_FIELDS}
            ),
        )


class UsageLedger:
    """Process-wide LLM usage accounting. See the module docstring for the
    concurrency and failure contracts."""

    def __init__(self, pricing_resolver: Any = None) -> None:
        if pricing_resolver is None:
            from .pricing import PricingResolver

            pricing_resolver = PricingResolver()
        self.pricing_resolver = pricing_resolver
        self.run_id: str = uuid.uuid4().hex
        self._records: dict[str, _RequestRecord] = {}
        # Optional persistence observer, duck-typed: on_response(settled, message)
        # / on_failure(settled). Notified exactly once per request, inside the
        # first-wins state transition. Observer errors are counted, never raised.
        self.observer: Optional[Any] = None
        self.persist_failures: int = 0
        self.revision: int = 0
        self._listeners: dict[str, Callable[[], None]] = {}

    def begin(self, provider: str, model: Optional[str], context: dict[str, Any] | None = None) -> str:
        """Register one invocation and return its request_id.

        Always returns a fresh id, even if internal bookkeeping fails — a later
        settle for an unknown id is a logged no-op, never an error. The ambient
        LlmCallOrigin (if any) is captured here so attribution survives to
        settlement regardless of which task finalizes the request.
        """
        request_id = uuid.uuid4().hex
        try:
            started_at = datetime.now(timezone.utc).isoformat()
            pricing_context = freeze_json({**(context or {}), "requested_at": started_at})
            record = _RequestRecord(
                provider=str(provider),
                model=model,
                origin=current_llm_call_origin(),
                prompt_id=current_logical_prompt_id(),
                pricing_context=pricing_context,
                started_at=started_at,
            )
            self._records[request_id] = record
            try:
                record.pinned_pricing = freeze_json(self.pricing_resolver.pin(str(provider), model, pricing_context))
            except Exception:
                logger.exception("Could not pin request pricing")
            self._price(record)
            self._changed(record)
            self._notify_optional("on_begin", self._settled(request_id, record))
        except Exception:
            logger.exception("UsageLedger.begin failed; request %s will be untracked", request_id)
        return request_id

    def record_response(self, request_id: str, usage: Optional[NormalizedUsage], message: Optional[Any] = None) -> None:
        """Settle a completed response. First settle wins; later calls no-op."""
        try:
            record = self._records.get(request_id)
            if record is None:
                logger.debug("UsageLedger: response for unknown request %s", request_id)
                return
            self._stamp_message(request_id, record, message)
            if record.state is not _RequestState.OPEN:
                return
            record.state = _RequestState.RESPONDED
            if (
                record.observation is None
                or not record.observation.complete
                or (isinstance(usage, NormalizedUsage) and usage.reported)
            ):
                record.usage = usage if isinstance(usage, NormalizedUsage) else None
            observation = getattr(message, "usage_observation", None) if message is not None else None
            if isinstance(observation, UsageObservation):
                self._merge_observation(record, observation)
            elif record.usage is not None:
                self._merge_observation(record, UsageObservation.from_usage(record.usage))
            self._price(record)
            self._changed(record)
            self._notify(lambda observer: observer.on_response(self._settled(request_id, record), message))
        except Exception:
            logger.exception("UsageLedger.record_response failed for request %s", request_id)

    def _stamp_message(self, request_id: str, record: _RequestRecord, message: Any) -> None:
        # Late final messages retain identity without settling again. History
        # consumers must deduplicate these against the original terminal fact.
        if message is None:
            return
        try:
            metadata = getattr(message, "usage_metadata", None)
            if not isinstance(metadata, dict):
                metadata = {}
                message.usage_metadata = metadata
            metadata["llm_call"] = {
                "llm_call_id": request_id,
                "run_id": self.run_id,
                "provider": record.provider,
                "model": record.model,
            }
        except Exception:
            logger.debug("UsageLedger: could not stamp llm_call onto the message", exc_info=True)

    def record_failure(self, request_id: str, error: str) -> None:
        """Settle a failed invocation. First settle wins; later calls no-op."""
        try:
            record = self._records.get(request_id)
            if record is None:
                logger.debug("UsageLedger: failure for unknown request %s", request_id)
                return
            if record.state is not _RequestState.OPEN:
                return
            record.state = _RequestState.FAILED
            record.error = str(error)[:_ERROR_TRUNCATION]
            self._price(record)
            self._changed(record)
            self._notify(lambda observer: observer.on_failure(self._settled(request_id, record)))
        except Exception:
            logger.exception("UsageLedger.record_failure failed for request %s", request_id)

    def _settled(self, request_id: str, record: _RequestRecord) -> SettledRequest:
        return SettledRequest(
            request_id=request_id,
            run_id=self.run_id,
            provider=record.provider,
            model=record.model,
            origin=record.origin,
            usage=record.usage,
            error=record.error,
            prompt_id=record.prompt_id,
            state=record.state.value,
            revision=record.revision,
            observation=record.observation,
            cost=freeze_json(record.cost),
            pricing_context=freeze_json(record.pricing_context),
            pinned_pricing=freeze_json(record.pinned_pricing),
            started_at=record.started_at,
        )

    def get_request(self, request_id: str) -> SettledRequest | None:
        try:
            record = self._records.get(request_id)
            return self._settled(request_id, record) if record is not None else None
        except Exception:
            logger.exception("Could not read usage request")
            return None

    def _merge_observation(self, record: _RequestRecord, observation: UsageObservation) -> bool:
        previous = record.observation
        merged = previous.refined_by(observation) if previous else observation
        if merged == previous:
            return False
        record.observation = merged
        if merged.complete:
            record.usage = NormalizedUsage(
                provider=merged.provider,
                model=merged.model,
                reported=True,
                **{name: getattr(merged, name) for name in TOKEN_FIELDS},
                unavailable_reason=None,
            )
        return True

    def record_observation(self, request_id: str, observation: UsageObservation) -> None:
        try:
            record = self._records.get(request_id)
            if record is None or not isinstance(observation, UsageObservation):
                return
            if not self._merge_observation(record, observation):
                return
            self._price(record)
            self._changed(record)
            if record.state is not _RequestState.OPEN:
                self._notify_optional("on_observation", self._settled(request_id, record))
        except Exception:
            logger.exception("Could not record request observation")

    def record_abandonment(self, request_id: str) -> None:
        try:
            record = self._records.get(request_id)
            if record is None or record.state is not _RequestState.OPEN:
                return
            record.state = _RequestState.ABANDONED
            record.error = "stream abandoned before completion"
            self._price(record)
            self._changed(record)
            self._notify(lambda observer: observer.on_failure(self._settled(request_id, record)))
        except Exception:
            logger.exception("Could not record abandoned request")

    def _price(self, record: _RequestRecord) -> None:
        try:
            cost = self.pricing_resolver.calculate(record.pinned_pricing, record.usage, record.observation)
            if cost.get("source") == "estimated" and record.observation is not None and not record.observation.complete:
                cost = {
                    **cost,
                    "complete": False,
                    "missing_components": [*cost.get("missing_components", []), "token_usage_incomplete"],
                }
            record.cost = freeze_json(cost)
        except Exception:
            logger.exception("Could not calculate request cost")
            record.cost = freeze_json(
                record.observation.cost
                if record.observation is not None and record.observation.cost is not None
                else {
                    "source": "unavailable",
                    "amount": None,
                    "currency": None,
                    "denomination": None,
                    "complete": False,
                    "missing_components": ["pricing_calculation_failed"],
                    "provenance": {},
                }
            )

    def subscribe(self, listener: Callable[[], None]) -> Callable[[], None]:
        key = uuid.uuid4().hex
        self._listeners[key] = listener

        def unsubscribe() -> None:
            self._listeners.pop(key, None)

        return unsubscribe

    def _changed(self, record: _RequestRecord | None = None) -> None:
        self.revision += 1
        if record is not None:
            record.revision += 1
        for listener in tuple(self._listeners.values()):
            try:
                listener()
            except Exception:
                logger.exception("Usage change listener failed")

    def _notify_optional(self, name: Literal["on_begin", "on_observation"], request: SettledRequest) -> None:
        callback = getattr(self.observer, name, None)
        if callable(callback):
            if name == "on_begin":
                self._notify(lambda observer: cast(RequestBeginObserver, observer).on_begin(request))
            else:
                self._notify(lambda observer: cast(RequestObservationObserver, observer).on_observation(request))

    def _notify(self, call: Any) -> None:
        observer = self.observer
        if observer is None:
            return
        try:
            call(observer)
        except Exception:
            self.persist_failures += 1
            self._changed()
            logger.exception("Usage observer notification failed")

    def note_persist_failure(self) -> None:
        """Record an asynchronous persistence failure (exception-proof)."""
        try:
            self.persist_failures += 1
            self._changed()
        except Exception:
            pass

    def snapshot(self, prompt_id: str | None = None) -> UsageSnapshot:
        """Aggregate all records into an immutable snapshot."""
        try:

            def empty() -> dict[str, Any]:
                return {
                    **{name: 0 for name in _COUNT_FIELDS + _TOKEN_SUM_FIELDS + ("abandoned", "missing", "partial")},
                    "known_requests": {name: 0 for name in TOKEN_FIELDS},
                    "costs": {
                        "currencies": {},
                        "non_monetary": {},
                        "complete": True,
                        "missing_requests": 0,
                        "partial_requests": 0,
                    },
                }

            totals = empty()
            rows: dict[tuple[str, str], dict[str, Any]] = {}

            for record in self._records.values():
                if prompt_id is not None and record.prompt_id != prompt_id:
                    continue
                usage = record.usage
                if usage is not None and usage.reported and usage.model is not None:
                    key = (usage.provider, usage.model)
                else:
                    key = (record.provider, record.model or "unknown")
                row = rows.setdefault(key, empty())
                observation = record.observation

                for target in (totals, row):
                    target["requests"] += 1
                    if record.state is _RequestState.FAILED:
                        target["failed"] += 1
                    elif record.state is _RequestState.ABANDONED:
                        target["abandoned"] += 1
                    elif record.state is _RequestState.OPEN:
                        target["open"] += 1
                    else:
                        target["responses"] += 1
                        if usage is not None and usage.reported:
                            target["reported"] += 1
                        else:
                            target["unreported"] += 1
                    if observation is None or not any(getattr(observation, name) is not None for name in TOKEN_FIELDS):
                        target["missing"] += 1
                    elif not observation.complete:
                        target["partial"] += 1
                    if observation is not None:
                        for name in TOKEN_FIELDS:
                            value = getattr(observation, name)
                            if value is not None:
                                target[name] += value
                                target["known_requests"][name] += 1
                    self._sum_cost(target["costs"], record.cost)

            breakdown = tuple(
                ProviderModelUsage(
                    provider=provider,
                    model=model,
                    **{
                        **counts,
                        "known_requests": freeze_json(counts["known_requests"]),
                        "costs": freeze_json(counts["costs"]),
                    },
                )
                for (provider, model), counts in sorted(rows.items())
            )
            return UsageSnapshot(
                run_id=self.run_id,
                persist_failures=self.persist_failures,
                breakdown=breakdown,
                revision=self.revision,
                prompt_id=prompt_id,
                **{
                    **totals,
                    "known_requests": freeze_json(totals["known_requests"]),
                    "costs": freeze_json(totals["costs"]),
                },
            )
        except Exception:
            logger.exception("UsageLedger.snapshot failed; returning empty snapshot")
            return UsageSnapshot(run_id=self.run_id)

    def snapshot_for_prompt(self, prompt_id: str) -> UsageSnapshot:
        return self.snapshot(prompt_id)

    @staticmethod
    def _sum_cost(aggregate: dict[str, Any], cost: dict[str, Any]) -> None:
        amount = nonnegative_decimal(cost.get("amount"))
        complete = cost.get("complete") is True and not cost.get("missing_components")
        if amount is None and cost.get("source") == "estimated":
            amount = nonnegative_decimal(cost.get("known_subtotal"))
            complete = False
        if not complete:
            aggregate["complete"] = False
            aggregate["missing_requests" if amount is None else "partial_requests"] += 1
        if amount is None:
            aggregate["complete"] = False
            return
        currency = cost.get("currency")
        denomination = cost.get("denomination")
        monetary = (
            isinstance(currency, str) and currency != "credits" and denomination not in ("credits", "provider_credits")
        )
        buckets = aggregate["currencies" if monetary else "non_monetary"]
        key = currency if monetary else str(denomination or "unknown")
        bucket = buckets.setdefault(
            key,
            {
                "reported": "0",
                "estimated": "0",
                "total": "0",
                "requests": 0,
                "reported_requests": 0,
                "estimated_requests": 0,
            },
        )
        source = "reported" if cost.get("source") == "provider_reported" else "estimated"
        bucket[source] = str(Decimal(bucket[source]) + amount)
        bucket["total"] = str(Decimal(bucket["total"]) + amount)
        bucket["requests"] += 1
        bucket[f"{source}_requests"] += 1
        token_subtotal = nonnegative_decimal(cost.get("token_cost_subtotal"))
        if token_subtotal is not None:
            bucket["token_cost_subtotal"] = str(Decimal(bucket.get("token_cost_subtotal", "0")) + token_subtotal)


class LedgerStreamAdapter:
    """Observe consumed wire evidence without draining or finalizing the provider."""

    def __init__(self, inner: Any, ledger: UsageLedger, request_id: str) -> None:
        self._inner = inner
        self._ledger = ledger
        self._request_id = request_id
        self._exhausted = False
        self._abandonment = weakref.finalize(self, ledger.record_abandonment, request_id)

    def _observe(self, observation: UsageObservation) -> None:
        self._ledger.record_observation(self._request_id, observation)

    def _capture(self) -> None:
        try:
            snapshot = getattr(self._inner, "get_usage_observation", None)
            if callable(snapshot):
                observation = snapshot()
                if isinstance(observation, UsageObservation):
                    self._observe(observation)
        except Exception:
            logger.exception("Could not capture consumed stream usage")

    async def __aenter__(self) -> "LedgerStreamAdapter":
        try:
            with observe_usage(self._observe):
                await self._inner.__aenter__()
        except BaseException as exc:
            self._capture()
            self._ledger.record_failure(self._request_id, describe_error(exc))
            raise
        self._capture()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            with observe_usage(self._observe):
                return await self._inner.__aexit__(exc_type, exc_val, exc_tb)
        except BaseException as exc:
            self._capture()
            self._ledger.record_failure(self._request_id, describe_error(exc_val or exc))
            raise
        finally:
            self._capture()
            if exc_type is not None:
                self._ledger.record_failure(self._request_id, describe_error(exc_val or exc_type))
            elif not self._exhausted:
                self._ledger.record_abandonment(self._request_id)

    def __aiter__(self) -> "LedgerStreamAdapter":
        return self

    async def __anext__(self):
        try:
            with observe_usage(self._observe):
                chunk = await self._inner.__anext__()
            self._capture()
            return chunk
        except StopAsyncIteration:
            self._exhausted = True
            self._capture()
            raise
        except BaseException as exc:
            self._capture()
            self._ledger.record_failure(self._request_id, describe_error(exc))
            raise

    async def get_final_message(self):
        try:
            with observe_usage(self._observe):
                message = await self._inner.get_final_message()
        except BaseException as exc:
            self._capture()
            self._ledger.record_failure(self._request_id, describe_error(exc))
            raise
        self._capture()
        observation = getattr(message, "usage_observation", None)
        if isinstance(observation, UsageObservation):
            self._observe(observation)
        else:
            usage = getattr(message, "usage", None)
            if isinstance(usage, NormalizedUsage):
                self._observe(UsageObservation.from_usage(usage))
        self._ledger.record_response(self._request_id, getattr(message, "usage", None), message=message)
        return message

    async def aclose(self) -> None:
        try:
            close: Any = getattr(self._inner, "aclose", None)
            if callable(close):
                with observe_usage(self._observe):
                    await cast(Callable[[], Awaitable[Any]], close)()
            else:
                await self.__aexit__(None, None, None)
        except BaseException as exc:
            self._capture()
            self._ledger.record_failure(self._request_id, describe_error(exc))
            raise
        finally:
            self._capture()
            self._ledger.record_abandonment(self._request_id)

    def __getattr__(self, name: str):
        return getattr(self._inner, name)
