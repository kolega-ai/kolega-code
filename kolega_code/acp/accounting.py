from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from decimal import Decimal, InvalidOperation
import logging
from typing import Any, Awaitable, Callable, TypedDict

from acp.schema import Cost, SessionInfoUpdate, Usage, UsageUpdate

from kolega_code.cli.session_usage import SessionUsageSink
from kolega_code.llm.ledger import UsageLedger

from .usage import build_context_usage_update, context_window_for

logger = logging.getLogger(__name__)

USAGE_META_KEY = "dev.kolega/usage"
_FLUSH_TIMEOUT = 10.0
_TOKEN_FIELDS = {
    "inputTokens": "input_tokens",
    "outputTokens": "output_tokens",
    "totalTokens": "total_tokens",
    "cachedReadTokens": "cache_read_input_tokens",
    "cachedWriteTokens": "cache_write_input_tokens",
    "thoughtTokens": "reasoning_output_tokens",
}
_COUNTS = ("requests", "responses", "reported", "unreported", "failed", "open", "abandoned", "partial", "missing")


class UsageExtension(TypedDict):
    schemaVersion: int
    session: dict[str, Any]
    prompt: dict[str, Any] | None


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return deepcopy(value)
    to_payload = getattr(value, "to_payload", None)
    if callable(to_payload):
        payload = to_payload()
        return payload if isinstance(payload, dict) else {}
    if is_dataclass(value) and not isinstance(value, type):
        result = asdict(value)
        return result if isinstance(result, dict) else {}
    return {}


def _count(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _add_costs(parts: list[dict[str, Any]]) -> dict[str, Any]:
    monetary: dict[str, dict[str, Any]] = {}
    credits: dict[str, dict[str, Any]] = {}
    missing = 0
    partial = 0
    complete = True
    for part in parts:
        costs = part.get("costs", part.get("cost", {}))
        if not isinstance(costs, dict) or not costs:
            missing += _count(part.get("requests"))
            continue
        missing += _count(costs.get("missing", costs.get("missing_requests", 0)))
        partial += _count(costs.get("partial", costs.get("partial_requests", 0)))
        complete = complete and costs.get("complete", True)
        for section, target in (("monetary", monetary), ("non_monetary", credits)):
            rows = costs.get(section, costs.get("currencies" if section == "monetary" else "credits", []))
            if isinstance(rows, dict):
                rows = [
                    {"currency" if section == "monetary" else "denomination": key, **value}
                    for key, value in rows.items()
                    if isinstance(value, dict)
                ]
            if not isinstance(rows, (list, tuple)):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                key = row.get("currency" if section == "monetary" else "denomination")
                if not isinstance(key, str):
                    continue
                dest = target.setdefault(
                    key,
                    {
                        "currency" if section == "monetary" else "denomination": key,
                        "reported": "0",
                        "estimated": "0",
                        "amount": "0",
                        "requests": 0,
                        "reported_requests": 0,
                        "estimated_requests": 0,
                        "_reported_present": False,
                        "_estimated_present": False,
                        "_source_counts_known": True,
                    },
                )
                dest["requests"] += _count(row.get("requests"))
                reported_requests = _count(row.get("reported_requests"))
                estimated_requests = _count(row.get("estimated_requests"))
                dest["_source_counts_known"] = dest["_source_counts_known"] and all(
                    type(row.get(name)) is int and row[name] >= 0
                    for name in ("reported_requests", "estimated_requests")
                )
                source = row.get("source")
                dest["_reported_present"] |= bool(reported_requests) or source in (
                    "reported",
                    "provider_reported",
                    "mixed",
                )
                dest["_estimated_present"] |= bool(estimated_requests) or source in ("estimated", "mixed")
                for name in ("reported", "estimated"):
                    try:
                        subtotal = Decimal(str(row.get(name, "0")))
                        if subtotal.is_finite() and subtotal > 0:
                            dest[f"_{name}_present"] = True
                    except (InvalidOperation, ValueError):
                        pass
                dest["reported_requests"] += reported_requests
                dest["estimated_requests"] += estimated_requests
                for name in ("reported", "estimated", "amount"):
                    try:
                        amount = Decimal(str(row.get(name, row.get("total", "0") if name == "amount" else "0")))
                        if amount.is_finite() and amount >= 0:
                            dest[name] = str(Decimal(dest[name]) + amount)
                    except (InvalidOperation, ValueError):
                        missing += 1
                if "token_cost_subtotal" in row:
                    try:
                        subtotal = Decimal(str(row["token_cost_subtotal"]))
                        if subtotal.is_finite() and subtotal >= 0:
                            dest["token_cost_subtotal"] = str(Decimal(dest.get("token_cost_subtotal", "0")) + subtotal)
                    except (InvalidOperation, ValueError):
                        missing += 1
    for row in (*monetary.values(), *credits.values()):
        reported, estimated = row.pop("_reported_present"), row.pop("_estimated_present")
        if not row.pop("_source_counts_known"):
            row["reported_requests"] = row["estimated_requests"] = None
        row["source"] = (
            "mixed"
            if reported and estimated
            else ("estimated" if estimated else ("reported" if reported else "unknown"))
        )
    return {
        "monetary": list(monetary.values()),
        "non_monetary": list(credits.values()),
        "missing": missing,
        "partial": partial,
        "complete": complete and missing == partial == 0,
        "basis": "published_list_value_before_account_credits_discounts_and_taxes",
    }


def _combine(parts: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {name: sum(_count(part.get(name)) for part in parts) for name in _COUNTS}
    for field in _TOKEN_FIELDS.values():
        result[field] = sum(_count(part.get(field)) for part in parts)
    result["persist_failures"] = sum(_count(part.get("persist_failures")) for part in parts)
    result["io_errors"] = sum(
        _count(part.get("io_errors", part.get("persistence", {}).get("io_errors"))) for part in parts
    )
    statuses = [part.get("close_status") for part in parts]
    result["close_status"] = next(
        (status for status in ("unknown", "not_committed", "pending", "committed") if status in statuses), None
    )
    result["known_requests"] = {
        name: sum(_count(part.get("known_requests", part.get("known_fields", {})).get(name)) for part in parts)
        for name in _TOKEN_FIELDS.values()
    }
    result["costs"] = _add_costs(parts)
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    row_parts: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for part in parts:
        for row in part.get("breakdown", ()):
            row = _mapping(row)
            key = (str(row.get("provider", "unknown")), str(row.get("model", "unknown")))
            dest = rows.setdefault(key, {"provider": key[0], "model": key[1]})
            row_parts.setdefault(key, []).append(row)
            for name in (*_COUNTS, *_TOKEN_FIELDS.values()):
                dest[name] = _count(dest.get(name)) + _count(row.get(name))
    for key, dest in rows.items():
        dest["costs"] = _add_costs(row_parts[key])
    result["breakdown"] = list(rows.values())
    result["detail"] = {
        key: sum(_count(part.get("detail", {}).get(key)) for part in parts)
        for key in ("reported_missing_cache_read", "reported_missing_cache_write", "reported_missing_reasoning")
    }
    for field, key in (
        ("cache_read_input_tokens", "reported_missing_cache_read"),
        ("cache_write_input_tokens", "reported_missing_cache_write"),
        ("reasoning_output_tokens", "reported_missing_reasoning"),
    ):
        result["detail"][key] = max(result["detail"][key], result["requests"] - result["known_requests"][field])
    result["coverage"] = {
        "full": all(part.get("coverage", {}).get("full", True) for part in parts),
        "pre_accounting_turns": sum(_count(part.get("coverage", {}).get("pre_accounting_turns")) for part in parts),
        **{
            name: sum(_count(part.get("coverage", {}).get(name)) for part in parts)
            for name in (
                "uncertain_runs",
                "unclosed_runs",
                "uninstrumented_turns",
                "legacy_turns",
                "legacy_requests",
                "legacy_identity_requests",
                "uncovered_assistant_turns",
                "orphan_runs",
                "malformed_facts",
                "lost_requests",
            )
        },
    }
    return result


def _public_totals(payload: dict[str, Any]) -> dict[str, Any]:
    counts = {name: _count(payload.get(name)) for name in _COUNTS}
    close_status = payload.get("close_status")
    close_settled = close_status in (None, "committed")
    known_requests = payload.get("known_requests", {})
    tokens = {
        name: _count(payload.get(field)) if known_requests.get(field) or counts["requests"] == 0 else None
        for name, field in _TOKEN_FIELDS.items()
    }
    detail = payload.get("detail", {})
    for name, missing in (
        ("cachedReadTokens", "reported_missing_cache_read"),
        ("cachedWriteTokens", "reported_missing_cache_write"),
        ("thoughtTokens", "reported_missing_reasoning"),
    ):
        if _count(detail.get(missing)):
            tokens[name] = None
    complete = (
        counts["open"] == counts["partial"] == counts["missing"] == 0
        and (counts["requests"] == known_requests.get("total_tokens", 0))
        and payload.get("coverage", {}).get("full", True)
        and close_settled
    )
    persistence_gaps = {
        name: _count(payload.get("coverage", {}).get(name))
        for name in ("unclosed_runs", "orphan_runs", "malformed_facts", "lost_requests")
    }
    return {
        **tokens,
        "knownSubtotals": {
            name: _count(payload.get(field)) if known_requests.get(field) or not counts["requests"] else None
            for name, field in _TOKEN_FIELDS.items()
        },
        "requests": counts,
        "complete": complete,
        "coverage": payload.get("coverage", {}),
        "persistence": {
            "failures": _count(payload.get("persist_failures")),
            "ioErrors": _count(payload.get("io_errors")),
            "closeStatus": close_status,
            **persistence_gaps,
            "healthy": close_settled and not payload.get("persist_failures") and not any(persistence_gaps.values()),
        },
        "detail": detail,
        "breakdown": payload.get("breakdown", []),
        "cost": payload.get("costs", {}),
    }


class AcpUsageAccounting:
    def __init__(
        self,
        session_id: str,
        *,
        baseline: dict[str, Any] | None = None,
        ledger: UsageLedger | None = None,
    ) -> None:
        self.session_id = session_id
        self.ledger = ledger if ledger is not None else UsageLedger()
        self._baseline = deepcopy(baseline or {})
        self.sink: SessionUsageSink | None = None
        self._agent: Any = None
        self._generation: str | None = None
        self._context: int | None = None
        self._revision = 0
        self._wake = asyncio.Event()
        self._dirty = True
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._close_status: str | None = None
        self._publisher: asyncio.Task[None] | None = None
        self._send: Callable[[Any], Awaitable[None]] | None = None
        self._publish_lock = asyncio.Lock()
        self._flush_failures = 0
        self._unsubscribe: Callable[[], None] | None = self.ledger.subscribe(self.changed)

    def changed(self, *_: Any) -> None:
        self._revision += 1
        self._dirty = True
        self._wake.set()

    def rebind(self, agent: Any) -> None:
        if agent is self._agent:
            return
        self._agent = agent
        self._generation = getattr(agent, "_acp_generation", None)
        self.invalidate_context()

    def invalidate_context(self) -> None:
        self._context = None
        self.changed()

    def observe_context(self, tokens: int, generation: str | None = None) -> None:
        if self._generation is not None and generation != self._generation:
            return
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            return
        if tokens != self._context:
            self._context = tokens
            self.changed()

    def start_publisher(self, send: Callable[[Any], Awaitable[None]]) -> None:
        self._send = send
        if self._publisher is None and self._close_task is None and not self._closed:
            self._publisher = asyncio.create_task(self._publish_loop(), name=f"acp-usage-{self.session_id}")
        self._wake.set()

    def _live(self, prompt_id: str | None = None) -> dict[str, Any]:
        snapshot = self.ledger.snapshot() if prompt_id is None else self.ledger.snapshot(prompt_id=prompt_id)
        payload = _mapping(snapshot)
        # Runtime detail completeness is carried separately from numeric subset subtotals.
        if "detail" not in payload:
            payload["detail"] = {
                name: _count(payload.get(name))
                for name in (
                    "reported_missing_cache_read",
                    "reported_missing_cache_write",
                    "reported_missing_reasoning",
                )
            }
        if prompt_id is None and self.sink is not None:
            payload["persist_failures"] = max(
                _count(payload.get("persist_failures")), _count(getattr(self.sink, "persistence_failures", 0))
            )
            payload["io_errors"] = _count(getattr(self.sink, "io_errors", 0))
            result = getattr(self.sink, "close_result", None)
            payload["close_status"] = getattr(result, "status", None) or self._close_status
        return payload

    def extension(self, prompt_id: str | None = None) -> UsageExtension:
        payload = _combine([self._baseline, self._live()])
        payload["persist_failures"] += self._flush_failures
        session = {
            "sessionId": self.session_id,
            "runId": self.ledger.run_id,
            "revision": self._revision,
            "scope": "session",
            "semantics": "snapshot",
            **_public_totals(payload),
            "context": {
                "used": self._context,
                "size": context_window_for(self._agent),
                "usableInputBudget": getattr(self._agent, "model_max_input_tokens", None),
                "basis": "latest_pre_request_measurement_or_estimate",
            },
            "limitations": [
                "SDK-internal retry attempts are not observable",
                "External tool and MCP bills are excluded",
            ],
        }
        prompt = None
        if prompt_id is not None:
            prompt = {
                "promptId": prompt_id,
                "runId": self.ledger.run_id,
                "revision": self._revision,
                "scope": "prompt",
                "semantics": "snapshot",
                **_public_totals(_combine([self._live(prompt_id)])),
            }
        return UsageExtension(schemaVersion=1, session=session, prompt=prompt)

    def update(self) -> UsageUpdate | SessionInfoUpdate:
        extension = self.extension()
        update = build_context_usage_update(self._agent, self._context)
        if update is None:
            return SessionInfoUpdate(session_update="session_info_update", field_meta={USAGE_META_KEY: extension})
        update.field_meta = {USAGE_META_KEY: extension}
        cost = extension["session"]["cost"]
        rows = cost.get("monetary", [])
        if (
            extension["session"]["coverage"].get("full", True)
            and extension["session"]["requests"]["open"] == 0
            and cost.get("missing", 0) == 0
            and cost.get("partial", 0) == 0
            and cost.get("complete") is True
            and not cost.get("non_monetary")
            and len(rows) == 1
        ):
            try:
                amount = Decimal(str(rows[0]["amount"]))
                if amount.is_finite() and amount >= 0:
                    native = float(amount)
                    if native != float("inf"):
                        update.cost = Cost(amount=native, currency=rows[0]["currency"])
            except (ValueError, InvalidOperation, KeyError):
                pass
        return update

    async def publish(self, *, force: bool = False) -> None:
        async with self._publish_lock:
            if self._send is None or (not self._dirty and not force):
                return
            revision = self._revision
            try:
                async with asyncio.timeout(_FLUSH_TIMEOUT):
                    await self._send(self.update())
            except Exception:
                logger.warning("ACP usage notification failed (session=%s)", self.session_id, exc_info=True)
                return
            if revision == self._revision:
                self._dirty = False

    async def _publish_loop(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            await asyncio.sleep(0)
            await self.publish()

    async def flush(self) -> bool:
        if self.sink is None:
            return True
        try:
            async with asyncio.timeout(_FLUSH_TIMEOUT):
                result = await self.sink.flush()
            if result is False:
                self._flush_failures += 1
                return False
            return True
        except Exception:
            self._flush_failures += 1
            logger.warning("ACP usage flush failed (session=%s)", self.session_id, exc_info=True)
            return False

    async def finish_prompt(self, prompt_id: str) -> UsageExtension:
        await self.flush()
        self.changed()
        await self.publish(force=True)
        return self.extension(prompt_id)

    async def aclose(self) -> None:
        # Returning transfers ownership to callers that unregister the session.
        while not self._closed:
            if self._close_task is None or self._close_task.cancelled():
                self._close_status = "pending"
                self._close_task = asyncio.create_task(self._close(), name=f"acp-usage-close-{self.session_id}")
            try:
                await asyncio.shield(self._close_task)
            except asyncio.CancelledError:
                continue
            self._close_task.result()

    async def _close(self) -> None:
        try:
            try:
                if self._publisher is not None:
                    self._publisher.cancel()
                    await asyncio.gather(self._publisher, return_exceptions=True)
                    self._publisher = None
            finally:
                if self.sink is not None:
                    try:
                        if callable(getattr(self.sink, "wait_closed", None)):
                            result = await self.sink.wait_closed()
                        else:
                            result = await self.sink.aclose()
                        self._close_status = getattr(result, "status", "committed")
                    except Exception:
                        self._close_status = "unknown"
                        self._flush_failures += 1
                        logger.warning("ACP usage close failed (session=%s)", self.session_id, exc_info=True)
                else:
                    self._close_status = None
            self.changed()
            try:
                async with asyncio.timeout(_FLUSH_TIMEOUT):
                    await self.publish(force=True)
            except TimeoutError:
                self._flush_failures += 1
        finally:
            if self._unsubscribe is not None:
                self._unsubscribe()
                self._unsubscribe = None
            if self._publisher is not None:
                self._publisher.cancel()
                await asyncio.gather(self._publisher, return_exceptions=True)
                self._publisher = None
            self._closed = True


def native_prompt_usage(extension: UsageExtension, *, enabled: bool) -> Usage | None:
    prompt = extension["prompt"]
    if not enabled or prompt is None or not prompt["complete"]:
        return None
    values = [prompt[name] for name in _TOKEN_FIELDS]
    if any(value is None for value in values):
        return None
    input_tokens, output_tokens, total, read, write, thoughts = values
    if read + write > input_tokens or thoughts > output_tokens or total != input_tokens + output_tokens:
        return None
    return Usage(
        input_tokens=input_tokens - read - write,
        output_tokens=output_tokens - thoughts,
        total_tokens=total,
        cached_read_tokens=read,
        cached_write_tokens=write,
        thought_tokens=thoughts,
    )
