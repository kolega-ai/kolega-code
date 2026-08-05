"""Runtime usage ledger: exactly-once accounting for every LLM invocation.

One :class:`UsageLedger` exists per CLI process, owned by the host (the TUI app
or the ``ask`` runner) and threaded into every :class:`~kolega_code.llm.client.LLMClient`
in the command tree — the top-level agent, sub-agents, goal verifiers, history
compression, web-fetch answering, hook prompts, and terminal
security checks. The client layer settles each invocation exactly once from the
normalized ``Message.usage`` (see kolega_code/llm/usage.py); repeated stream
finalization (the Langfuse wrapper calls ``get_final_message()`` up to three
times per turn) is deduplicated by a first-settle-wins state machine.

Accounting is per client invocation: SDK-internal retries (a 429 retried inside
the vendor SDK) are invisible to the process and therefore indistinguishable
from a single attempt. Agent-loop retries issue new client calls and are
recorded as separate requests.

Concurrency contract: every LLM call in this codebase runs on the single asyncio
event loop (``asyncio.to_thread`` is used only for payload preparation), and all
ledger methods are synchronous with no awaits, so gathered sub-agent tasks
cannot interleave inside a method — no locks are needed.

Failure contract: a ledger bug must never break a paid request or trigger
another one. Every public method catches its own internal errors, logs, and
returns; nothing raises into the request path. A stream that is abandoned
without finalization stays OPEN and marks the snapshot incomplete — visible,
never silently dropped.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generator, Optional

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
    """Attribute LLM calls begun within this scope (and child tasks) to ``origin``."""
    token = _llm_call_origin.set(origin)
    try:
        yield
    finally:
        _llm_call_origin.reset(token)


class _RequestState(Enum):
    OPEN = "open"
    RESPONDED = "responded"
    FAILED = "failed"


@dataclass(slots=True)
class _RequestRecord:
    provider: str
    model: Optional[str]
    origin: Optional[LlmCallOrigin] = None
    state: _RequestState = _RequestState.OPEN
    usage: Optional[NormalizedUsage] = None
    error: Optional[str] = None


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


@dataclass(frozen=True, slots=True)
class UsageSnapshot:
    """Immutable aggregate of the ledger at one moment.

    Token sums cover reported usages only; unreported and failed invocations
    contribute to their counts (and to ``complete``) but never fabricate zeros
    into the totals. ``requests == responses + failed + open``.
    """

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

    @property
    def complete(self) -> bool:
        """True iff every begun request settled with authoritative usage."""
        return self.failed == 0 and self.unreported == 0 and self.open == 0

    def since(self, earlier: Optional["UsageSnapshot"]) -> "UsageSnapshot":
        """Checkpoint delta relative to an earlier snapshot of the same run.

        Monotonic counts and sums subtract; ``open`` is the CURRENT outstanding
        count (not a difference — a request open at both points is still open).
        ``earlier`` of None or from another run (run_id mismatch) yields self.
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
            )
            if any(getattr(delta_row, name) for name in _COUNT_FIELDS + _TOKEN_SUM_FIELDS):
                rows.append(delta_row)

        return UsageSnapshot(run_id=self.run_id, **deltas, **sums, breakdown=tuple(rows))


class UsageLedger:
    """Process-wide LLM usage accounting. See the module docstring for the
    concurrency and failure contracts."""

    def __init__(self) -> None:
        self.run_id: str = uuid.uuid4().hex
        self._records: dict[str, _RequestRecord] = {}
        # Optional persistence observer, duck-typed: on_response(settled, message)
        # / on_failure(settled). Notified exactly once per request, inside the
        # first-wins state transition. Observer errors are counted, never raised.
        self.observer: Optional[Any] = None
        self.persist_failures: int = 0

    def begin(self, provider: str, model: Optional[str]) -> str:
        """Register one invocation and return its request_id.

        Always returns a fresh id, even if internal bookkeeping fails — a later
        settle for an unknown id is a logged no-op, never an error. The ambient
        LlmCallOrigin (if any) is captured here so attribution survives to
        settlement regardless of which task finalizes the request.
        """
        request_id = uuid.uuid4().hex
        try:
            self._records[request_id] = _RequestRecord(
                provider=str(provider), model=model, origin=current_llm_call_origin()
            )
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
            if record.state is not _RequestState.OPEN:
                return
            record.state = _RequestState.RESPONDED
            record.usage = usage if isinstance(usage, NormalizedUsage) else None
            self._notify(lambda observer: observer.on_response(self._settled(request_id, record), message))
        except Exception:
            logger.exception("UsageLedger.record_response failed for request %s", request_id)

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
        )

    def _notify(self, call: Any) -> None:
        observer = self.observer
        if observer is None:
            return
        try:
            call(observer)
        except Exception:
            self.persist_failures += 1
            logger.exception("Usage observer notification failed")

    def note_persist_failure(self) -> None:
        """Record an asynchronous persistence failure (exception-proof)."""
        try:
            self.persist_failures += 1
        except Exception:
            pass

    def snapshot(self) -> UsageSnapshot:
        """Aggregate all records into an immutable snapshot."""
        try:
            totals: dict[str, int] = {name: 0 for name in _COUNT_FIELDS + _TOKEN_SUM_FIELDS}
            rows: dict[tuple[str, str], dict[str, int]] = {}

            for record in self._records.values():
                usage = record.usage
                if usage is not None and usage.reported and usage.model is not None:
                    key = (usage.provider, usage.model)
                else:
                    key = (record.provider, record.model or "unknown")
                row = rows.setdefault(key, {name: 0 for name in _COUNT_FIELDS + _TOKEN_SUM_FIELDS})

                for target in (totals, row):
                    target["requests"] += 1
                    if record.state is _RequestState.FAILED:
                        target["failed"] += 1
                    elif record.state is _RequestState.OPEN:
                        target["open"] += 1
                    else:
                        target["responses"] += 1
                        if usage is not None and usage.reported:
                            target["reported"] += 1
                            for name in _TOKEN_SUM_FIELDS:
                                target[name] += getattr(usage, name) or 0
                        else:
                            target["unreported"] += 1

            breakdown = tuple(
                ProviderModelUsage(provider=provider, model=model, **counts)
                for (provider, model), counts in sorted(rows.items())
            )
            return UsageSnapshot(
                run_id=self.run_id, persist_failures=self.persist_failures, breakdown=breakdown, **totals
            )
        except Exception:
            logger.exception("UsageLedger.snapshot failed; returning empty snapshot")
            return UsageSnapshot(run_id=self.run_id)


class LedgerStreamAdapter:
    """Transparent proxy over a provider stream wrapper that settles its ledger
    request exactly once.

    Settlement events (first wins): a successful ``get_final_message`` records
    the response; a raising ``get_final_message``, a raising ``__anext__``, or
    an exception unwinding the ``async with`` block records a failure. A clean
    exit settles nothing — every in-tree consumer finalizes afterwards (or
    already did inside the block), and an abandoned stream deliberately stays
    OPEN so the snapshot reports it as incomplete.
    """

    def __init__(self, inner: Any, ledger: UsageLedger, request_id: str) -> None:
        self._inner = inner
        self._ledger = ledger
        self._request_id = request_id

    async def __aenter__(self) -> "LedgerStreamAdapter":
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            return await self._inner.__aexit__(exc_type, exc_val, exc_tb)
        finally:
            if exc_type is not None:
                self._ledger.record_failure(self._request_id, describe_error(exc_val or exc_type))

    def __aiter__(self) -> "LedgerStreamAdapter":
        return self

    async def __anext__(self):
        try:
            return await self._inner.__anext__()
        except StopAsyncIteration:
            raise
        except BaseException as exc:
            self._ledger.record_failure(self._request_id, describe_error(exc))
            raise

    async def get_final_message(self):
        try:
            message = await self._inner.get_final_message()
        except BaseException as exc:
            self._ledger.record_failure(self._request_id, describe_error(exc))
            raise
        self._ledger.record_response(self._request_id, getattr(message, "usage", None), message=message)
        return message

    def __getattr__(self, name: str):
        return getattr(self._inner, name)
