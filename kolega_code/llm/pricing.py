from __future__ import annotations

import json
import logging
import copy
from datetime import datetime, timezone, timedelta
from decimal import Decimal, localcontext
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from .specs.accessors import get_model_pricing
from .specs.pricing import (
    RATE_UNITS,
    decimal_string,
    decimal_value,
    pricing_datetime,
    pricing_to_json,
)
from .usage import NormalizedUsage

logger = logging.getLogger(__name__)
_CONTEXT_FIELDS = {
    "requested_at",
    "timestamp",
    "service_tier",
    "region",
    "modality",
    "cache_ttl",
    "billing_model",
    "base_model",
    "resolved_model",
    "billing_surface",
    "route",
    "byok",
    "tools",
    "pricing_quantities",
    "billing_time_uncertainty_seconds",
}
_CONDITION_FIELDS = {
    "service_tier",
    "region",
    "modality",
    "cache_ttl",
    "billing_surface",
    "route",
    "byok",
    "effective_from",
    "effective_to",
    "min_input_tokens",
    "max_input_tokens",
    "min_tokens",
    "max_tokens",
    "quantity",
    "timezone",
    "weekdays",
    "start_time",
    "end_time",
    "dates",
    "excluded_dates",
    "holidays",
    "working_dates",
    "calendar_complete_through",
}
_TOOL_ALIASES = {
    "web_search": {"web_search", "web_search_fast", "google_search"},
    "web_search_fast": {"web_search_fast"},
    "google_search": {"web_search", "google_search", "google_search_retrieval"},
    "google_maps": {"google_maps", "maps"},
    "file_search": {"file_search"},
    "code_execution": {"code_execution", "code_interpreter"},
    "server_search": {"search", "web_search"},
    "server_fetch": {"fetch", "web_fetch"},
    "fetch_url": {"fetch_url", "web_fetch"},
    "finance_search": {"finance_search"},
    "people_search": {"people_search"},
}


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _quantity(value: Any) -> Decimal | None:
    try:
        return decimal_value(value)
    except ValueError:
        return None


def _tool_disabled(component: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
    tools = context.get("tools")
    if not isinstance(tools, (list, tuple)):
        return False
    aliases = {component["tool"]} if component.get("tool") else _TOOL_ALIASES.get(component["name"])
    return aliases is not None and not any(tool in aliases for tool in tools)


def _ttl_write_rate(card: dict[str, Any], context: dict[str, Any], quantities: dict[str, Any], ttl: str) -> str | None:
    selection = dict(context, cache_ttl=ttl)
    found: list[tuple[int, str]] = []
    for schedule in card.get("schedules", []):
        if "cache_ttl" not in schedule.get("when", schedule.get("conditions", {})):
            continue
        selected, _ = _match(schedule.get("when", schedule.get("conditions", {})), selection, quantities)
        if selected is True and "cache_write" in schedule.get("rates", {}) and schedule.get("mode") != "marginal":
            found.append((schedule.get("priority", 0), schedule["rates"]["cache_write"]))
    if not found:
        return None
    highest = max(priority for priority, _ in found)
    rates = {rate for priority, rate in found if priority == highest}
    return rates.pop() if len(rates) == 1 else None


def _safe_context(context: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in _CONTEXT_FIELDS:
        value = context.get(name)
        if value is None:
            continue
        if name in {"tools", "pricing_quantities"} and isinstance(value, Mapping):
            result[name] = {
                key: decimal_string(item) if isinstance(item, Decimal) else item
                for key, item in value.items()
                if isinstance(key, str) and _quantity(item) is not None
            }
        elif name == "tools" and isinstance(value, (list, tuple)):
            result[name] = [item for item in value if isinstance(item, str) and "://" not in item]
        elif isinstance(value, (str, int, bool)) and (not isinstance(value, str) or "://" not in value):
            result[name] = value
    return result


def _unavailable(reason: str, pinned: dict[str, Any], *, assumptions: list[str] | None = None) -> dict[str, Any]:
    return {
        "source": "unavailable",
        "amount": None,
        "currency": None,
        "denomination": None,
        "complete": False,
        "subtotals": [],
        "missing_components": [reason],
        "assumptions": list(assumptions or []),
        "pricing_source": pinned.get("pricing_source"),
        "pinned_pricing": pinned,
        "provenance": pinned.get("pricing", {}).get("sources", []),
    }


def _reported_cost(observation: Any, pinned: dict[str, Any]) -> dict[str, Any] | None:
    raw = _field(observation, "cost")
    if not isinstance(raw, Mapping):
        return None
    if raw.get("source", "provider_reported") != "provider_reported":
        return None
    try:
        amount = decimal_value(raw.get("amount"))
    except ValueError:
        return None
    currency, denomination = raw.get("currency"), raw.get("denomination")
    if denomination == "currency" and currency:
        denomination = None
    if bool(currency) == bool(denomination):
        return None
    if currency and (
        not isinstance(currency, str)
        or len(currency) != 3
        or not currency.isascii()
        or not currency.isalpha()
        or not currency.isupper()
    ):
        return None
    if denomination and not isinstance(denomination, str):
        return None
    result = _unavailable("", pinned)
    result.update(
        source="provider_reported",
        amount=decimal_string(amount),
        currency=currency,
        denomination=denomination,
        complete=raw.get("complete", True) is True,
        missing_components=list(raw.get("missing_components", [])),
        subtotals=list(raw.get("subtotals", [])),
        assumptions=list(raw.get("assumptions", [])),
        provenance=raw.get("provenance", []),
        scope=raw.get("scope", "request_inference"),
    )
    return result


def _match(
    conditions: Mapping[str, Any], context: Mapping[str, Any], quantities: Mapping[str, Any]
) -> tuple[bool | None, str | None]:
    unknown = set(conditions) - _CONDITION_FIELDS
    if unknown:
        return None, "unsupported pricing condition: " + ", ".join(sorted(unknown))
    missing: list[str] = []
    for name in ("service_tier", "region", "modality", "cache_ttl", "billing_surface", "route", "byok"):
        if name not in conditions:
            continue
        actual = context.get(name, quantities.get(name))
        expected = conditions[name]
        if actual is None:
            missing.append(name)
        elif actual not in (expected if isinstance(expected, list) else [expected]):
            return False, None
    quantity_name = conditions.get("quantity", "input")
    length = _count(quantities.get(quantity_name))
    for name, compare in (
        ("min_input_tokens", "min"),
        ("max_input_tokens", "max"),
        ("min_tokens", "min"),
        ("max_tokens", "max"),
    ):
        if name not in conditions:
            continue
        if length is None:
            missing.append(quantity_name)
        elif (compare == "min" and length < conditions[name]) or (compare == "max" and length > conditions[name]):
            return False, None
    date_fields = {
        "effective_from",
        "effective_to",
        "weekdays",
        "start_time",
        "end_time",
        "dates",
        "excluded_dates",
        "holidays",
        "working_dates",
        "calendar_complete_through",
    }
    if date_fields.intersection(conditions):
        try:
            stamp = pricing_datetime(context.get("requested_at"))
        except ValueError:
            return None, "request timestamp unavailable"
        if conditions.get("effective_from") and stamp < pricing_datetime(conditions["effective_from"]):
            return False, None
        if conditions.get("effective_to") and stamp >= pricing_datetime(conditions["effective_to"]):
            return False, None
        local = stamp.astimezone(ZoneInfo(conditions.get("timezone", "UTC")))
        date, clock = local.date().isoformat(), local.strftime("%H:%M")
        if conditions.get("calendar_complete_through") and date > conditions["calendar_complete_through"]:
            return None, "holiday calendar coverage expired"
        if date in conditions.get("excluded_dates", []):
            return False, None
        if "dates" in conditions and date not in conditions["dates"]:
            return False, None
        if "weekdays" in conditions:
            day = local.weekday()
            if date in conditions.get("holidays", []):
                day = 6
            elif date in conditions.get("working_dates", []):
                day = 0
            if day not in conditions["weekdays"]:
                return False, None
        start, end = conditions.get("start_time"), conditions.get("end_time")
        if start and end:
            inside = start <= clock < end if start < end else clock >= start or clock < end
            if not inside:
                return False, None
        elif (start and clock < start) or (end and clock >= end):
            return False, None
    return (None, "missing pricing context: " + ", ".join(sorted(set(missing)))) if missing else (True, None)


def _quantities(usage: Any, observation: Any, context: dict[str, Any]) -> dict[str, Any]:
    values = dict(context.get("pricing_quantities", {}))
    observed = _field(observation, "pricing_quantities", {})
    if isinstance(observed, Mapping):
        values.update(observed)
    for name, field in (
        ("input", "input_tokens"),
        ("output", "output_tokens"),
        ("cache_read", "cache_read_input_tokens"),
        ("cache_write", "cache_write_input_tokens"),
    ):
        value = _field(observation, field)
        if value is None:
            value = _field(usage, field)
        if value is not None:
            values[name] = value
    if "pricing_tier_input_tokens" in values:
        values["tier_input"] = values["pricing_tier_input_tokens"]
    else:
        values["tier_input"] = values.get("input")
    return values


def _override_for(overrides: Mapping[Any, Any], provider: str, model: str | None) -> Any:
    if model is None:
        return None
    if (provider, model) in overrides:
        return overrides[(provider, model)]
    if f"{provider}/{model}" in overrides:
        return overrides[f"{provider}/{model}"]
    nested = overrides.get(provider)
    return nested.get(model) if isinstance(nested, Mapping) else None


class PricingResolver:
    def __init__(self, overrides: Mapping[Any, Any] | None = None) -> None:
        self.overrides = copy.deepcopy(dict(overrides)) if isinstance(overrides, Mapping) else {}

    def pin(self, provider: str, model: str | None, context: dict[str, Any] | None = None) -> dict[str, Any]:
        provider = getattr(provider, "value", provider)
        selection = _safe_context(context or {})
        selection["requested_at"] = (
            selection.get("requested_at", selection.get("timestamp")) or datetime.now(timezone.utc).isoformat()
        )
        pinned: dict[str, Any] = {
            "schema_version": 1,
            "provider": provider,
            "model": model,
            "context": selection,
            "pricing_source": None,
            "pricing": {},
            "warnings": [],
        }
        override = _override_for(self.overrides, provider, model)
        if override is False or (isinstance(override, Mapping) and override.get("enabled") is False):
            pinned["reason"] = "estimation disabled by exact-route override"
            return pinned
        if override is not None:
            try:
                card = dict(override)
                card.pop("enabled", None)
                card.setdefault("schema_version", 1)
                card.setdefault("pricing_id", f"override:{provider}/{model}")
                card.setdefault("provider", provider)
                card.setdefault("model", model)
                card.setdefault("status", "priced")
                card.setdefault("billing_mode", "operator_configured")
                card.setdefault("checked_at", selection["requested_at"])
                card.setdefault(
                    "sources",
                    [{"kind": "operator", "model_id": model}],
                )
                card = pricing_to_json(card)
                if (card["provider"], card["model"]) != (provider, model):
                    raise ValueError("override route does not match request")
                pinned.update(pricing=card, pricing_source="override")
            except (TypeError, ValueError, KeyError):
                warning = f"Invalid usage_pricing override for {provider}/{model}; using catalogue pricing."
                pinned["warnings"].append(warning)
                logger.warning(warning)
        checkpoint = provider == "tinker" and bool(model and model.startswith("tinker://"))
        if checkpoint and not pinned["pricing"]:
            pinned["reason"] = "saved Tinker checkpoints have no automatic catalogue pricing"
            return pinned
        if not pinned["pricing"]:
            card = get_model_pricing(provider, model) if model else None
            if card:
                pinned.update(pricing=card, pricing_source="catalogue", billing_model=model)
        elif pinned["pricing_source"] == "override" and not checkpoint:
            fallback = get_model_pricing(provider, model) if model else None
            if fallback:
                pinned["catalogue_fallback"] = fallback
        if not pinned["pricing"]:
            if provider.startswith("custom:"):
                reason = "custom endpoint requires an exact operator pricing override"
            elif provider == "llama":
                reason = "self-hosted inference has no universal per-request monetary rate"
            else:
                reason = "no applicable exact-route catalogue pricing"
            pinned["reason"] = reason
            return pinned
        card = pinned["pricing"]
        try:
            now = pricing_datetime(selection["requested_at"])
            pinned["review_due"] = (now - pricing_datetime(card["checked_at"])).days > 30
        except ValueError:
            pinned["reason"] = "invalid request pricing timestamp"
        return json.loads(json.dumps(pinned, allow_nan=False))

    def calculate(
        self, pinned: dict[str, Any], usage: NormalizedUsage | None, observation: Any = None
    ) -> dict[str, Any]:
        snapshot = json.loads(json.dumps(pinned, allow_nan=False))
        reported = _reported_cost(observation, snapshot)
        if reported is not None:
            return reported
        with localcontext() as precision:
            precision.prec = 60
            return self._estimate(snapshot, usage, observation)

    def _estimate(self, pinned: dict[str, Any], usage: Any, observation: Any) -> dict[str, Any]:
        assumptions = list(pinned.get("warnings", []))
        if pinned.get("reason"):
            return _unavailable(pinned["reason"], pinned, assumptions=assumptions)
        card = pinned.get("pricing", {})
        try:
            card = pricing_to_json(card)
        except (ValueError, TypeError, KeyError):
            return _unavailable("invalid pinned pricing card", pinned)
        if card["status"] != "priced":
            return _unavailable(card.get("reason", card["status"]), pinned)
        context = {**card.get("default_context", {}), **pinned.get("context", {})}
        quantities = _quantities(usage, observation, context)
        if card["provider"] == "anthropic":
            for wire, name in (("web_search_requests", "web_search_calls"), ("web_fetch_requests", "web_fetch_calls")):
                if wire in quantities:
                    quantities.setdefault(name, quantities[wire])
        conditions = dict(card.get("conditions", {}))
        for name in ("effective_from", "effective_to"):
            if card.get(name):
                conditions[name] = card[name]
        applies, reason = _match(conditions, context, quantities)
        if applies is not True:
            if pinned.get("pricing_source") == "override" and pinned.get("catalogue_fallback"):
                revised = dict(pinned, pricing=pinned["catalogue_fallback"], pricing_source="catalogue")
                revised.pop("catalogue_fallback", None)
                return self._estimate(revised, usage, observation)
            regular = card.get("regular_pricing")
            if regular is not None:
                revised = dict(pinned, pricing=regular)
                return self._estimate(revised, usage, observation)
            return _unavailable(reason or "pricing card not effective for this request", pinned)
        rates = dict(card.get("rates", {}))
        components = list(card.get("components", []))
        missing: list[str] = []
        marginal: list[dict[str, Any]] = []
        applied: list[str] = []
        uncertain_rates: set[str] = set()
        selected_rates: dict[str, tuple[str, int | None]] = {}
        selection_quantities = dict(quantities, input=quantities.get("tier_input"))
        for index, schedule in enumerate(card.get("schedules", [])):
            selected, why = _match(schedule.get("when", schedule.get("conditions", {})), context, selection_quantities)
            if selected is None:
                affected = ", ".join(schedule.get("rates", {})) or "components"
                missing.append(f"schedule {index} ({affected}): {why}")
                uncertain_rates.update(schedule.get("rates", {}))
                continue
            if not selected:
                continue
            applied.append(str(schedule.get("id", index)))
            if schedule.get("mode") == "marginal":
                marginal.append(schedule)
            else:
                for name, rate in schedule.get("rates", {}).items():
                    priority = schedule.get("priority")
                    previous = selected_rates.get(name)
                    if previous and previous[0] != rate:
                        if priority is None or previous[1] is None or priority == previous[1]:
                            missing.append(f"{name}: ambiguous overlapping schedules")
                            uncertain_rates.add(name)
                        elif priority < previous[1]:
                            continue
                    rates[name] = rate
                    selected_rates[name] = (rate, priority)
            components.extend(schedule.get("components", []))
        if card.get("requires_schedule") and not applied:
            if pinned.get("pricing_source") == "override" and pinned.get("catalogue_fallback"):
                revised = dict(pinned, pricing=pinned["catalogue_fallback"], pricing_source="catalogue")
                revised.pop("catalogue_fallback", None)
                return self._estimate(revised, usage, observation)
            missing.append("no applicable required rate schedule")
        assumptions.extend(card.get("assumptions", []))
        defaults_used = set(card.get("default_context", {})) - set(pinned.get("context", {}))
        if defaults_used:
            assumptions.append("Documented default billing context: " + ", ".join(sorted(defaults_used)) + ".")
        assumptions.append("Published resource/list value before account allowances, taxes and negotiated discounts.")
        if pinned.get("review_due"):
            assumptions.append("Pricing source is older than 30 days and due for review.")
        if card.get("schedules"):
            assumptions.append("Schedule selected using request-start timestamp; provider billing time may differ.")
            uncertainty = _count(context.get("billing_time_uncertainty_seconds"))
            if uncertainty:
                stamp = pricing_datetime(context["requested_at"])
                for direction in (-1, 1):
                    adjacent = dict(
                        context, requested_at=(stamp + timedelta(seconds=direction * uncertainty)).isoformat()
                    )
                    if any(
                        _match(schedule.get("when", schedule.get("conditions", {})), context, selection_quantities)[0]
                        != _match(schedule.get("when", schedule.get("conditions", {})), adjacent, selection_quantities)[
                            0
                        ]
                        for schedule in card["schedules"]
                    ):
                        missing.append("billing-time uncertainty may cross a schedule boundary")
                        break
        subtotals: list[dict[str, Any]] = []
        unit = Decimal(RATE_UNITS[card.get("unit", "per_million_tokens")])
        counts = {name: _count(quantities.get(name)) for name in ("input", "output", "cache_read", "cache_write")}
        policies = card.get("cache_policy", {})
        for name in ("cache_read", "cache_write"):
            if policies.get(name) in {"unsupported", "included"}:
                if policies[name] == "unsupported" and counts[name] not in (0, None):
                    missing.append(f"{name}: observed on an unsupported billing component")
                counts[name] = 0
            elif name not in rates and not any(name in band.get("rates", {}) for band in marginal):
                if counts[name] not in (None, 0):
                    missing.append(f"{name}: no documented rate or included policy")
                if counts[name] is None:
                    missing.append(f"{name}: count and billing policy unavailable")
            elif counts[name] is None:
                missing.append(f"{name}: count unavailable")
        inp, read, write = counts["input"], counts["cache_read"], counts["cache_write"]
        if inp is not None and read is not None and write is not None and read + write <= inp:
            counts["input"] = (
                inp
                - (0 if policies.get("cache_read") == "supplemental" else read)
                - (0 if policies.get("cache_write") == "supplemental" else write)
            )
        elif inp is not None:
            counts["input"] = None
            missing.append("input: cache partitions unavailable or exceed inclusive input")

        covering = {
            component["covers"]
            for component in components
            if component.get("covers") and _match(component.get("when", {}), context, quantities)[0] is not False
        }
        for name in ("input", "output", "cache_read", "cache_write"):
            if name in uncertain_rates:
                continue
            if name in covering:
                continue
            count = counts[name]
            if count is None:
                missing.append(f"{name}: count unavailable")
                continue
            if count == 0:
                continue
            bands = [band for band in marginal if name in band.get("rates", {})]
            if bands:
                disjoint = any(band.get("quantities") for band in bands)
                if not disjoint and (name not in {"input", "output"} or (name == "input" and (read or write))):
                    missing.append(f"{name}: marginal allocation requires disjoint band quantities")
                    continue
                covered = 0
                prior_end = 0
                total = Decimal(0)
                for band in sorted(bands, key=lambda value: value.get("min_tokens", 0)):
                    low, high = band.get("min_tokens", 0), band.get("max_tokens", count)
                    if not disjoint and low < prior_end:
                        missing.append(f"{name}: overlapping marginal bands")
                        break
                    if disjoint:
                        part = _count(quantities.get(band.get("quantities", {}).get(name)))
                        if part is None:
                            missing.append(f"{name}: marginal band quantity unavailable")
                            break
                    else:
                        part = max(0, min(count, high) - low)
                    total += part * decimal_value(band["rates"][name]) / unit
                    covered += part
                    prior_end = high
                if covered != count:
                    missing.append(f"{name}: marginal bands do not cover quantity")
                else:
                    subtotals.append({"component": name, "amount": decimal_string(total), "quantity": count})
            elif name in rates:
                total = Decimal(count) * decimal_value(rates[name]) / unit
                subtotals.append({"component": name, "amount": decimal_string(total), "quantity": count})
            else:
                missing.append(f"{name}: rate unavailable")
        ttl_rates = {key: rate for key, rate in rates.items() if key.startswith("cache_write_")}
        if ttl_rates and write:
            subtotals = [part for part in subtotals if part["component"] != "cache_write"]
            covered = 0
            for name, rate in ttl_rates.items():
                count = _count(quantities.get(name))
                if count is None:
                    missing.append(f"{name}: TTL-specific quantity unavailable")
                    continue
                covered += count
                subtotals.append(
                    {
                        "component": name,
                        "amount": decimal_string(Decimal(count) * decimal_value(rate) / unit),
                        "quantity": count,
                    }
                )
            if covered != write:
                missing.append("cache_write: TTL quantities do not cover cache writes")
            else:
                missing = [item for item in missing if not item.startswith("cache_write:")]
        ttl_counts = {
            name: _count(value)
            for name, value in quantities.items()
            if name.startswith("cache_write_") and _count(value) is not None
        }
        if ttl_counts and not ttl_rates and write:
            subtotals = [part for part in subtotals if part["component"] != "cache_write"]
            covered = 0
            for name, count in ttl_counts.items():
                assert count is not None
                rate = _ttl_write_rate(card, context, selection_quantities, name.removeprefix("cache_write_"))
                if rate is None:
                    missing.append(f"{name}: no unambiguous TTL-specific rate")
                    continue
                covered += count
                subtotals.append(
                    {
                        "component": name,
                        "amount": decimal_string(Decimal(count) * decimal_value(rate) / unit),
                        "quantity": count,
                    }
                )
            if covered != write:
                missing.append("cache_write: TTL quantities do not cover cache writes")
            else:
                missing = [item for item in missing if not item.startswith("cache_write:")]
        partition_counts: dict[str, int] = {}
        token_component_names: set[str] = {"input", "output", "cache_read", "cache_write", *ttl_rates, *ttl_counts}
        for component in components:
            name = component["name"]
            selected, why = _match(component.get("when", {}), context, quantities)
            if selected is False:
                continue
            if selected is None:
                missing.append(f"{name}: {why}")
                continue
            quantity = component.get("quantity", name)
            count = _quantity(quantities.get(quantity))
            if count is None and quantity == "request":
                count = Decimal(1)
            if count is None and quantity in context.get("tools", {}):
                tools = context["tools"]
                if isinstance(tools, Mapping):
                    count = _quantity(tools[quantity])
            if count is None and _tool_disabled(component, context):
                count = Decimal(0)
            if count is None:
                missing.append(f"{name}: quantity {quantity} unavailable")
                continue
            if component.get("covers"):
                partition = component["covers"]
                if count != int(count):
                    missing.append(f"{name}: token quantity must be an integer")
                    continue
                partition_counts[partition] = partition_counts.get(partition, 0) + int(count)
                token_component_names.add(name)
            total = (
                Decimal(count)
                * decimal_value(component["rate"])
                / Decimal(RATE_UNITS[component.get("unit", "per_request")])
            )
            subtotals.append(
                {
                    "component": name,
                    "amount": decimal_string(total),
                    "quantity": int(count) if count == int(count) else decimal_string(count),
                }
            )
        for partition in covering:
            if counts[partition] is None or partition_counts.get(partition, 0) != counts[partition]:
                missing.append(f"{partition}: component quantities do not cover normalized partition")
        missing.extend(card.get("missing_components", []))
        total = sum((decimal_value(part["amount"]) for part in subtotals), Decimal(0))
        missing = list(dict.fromkeys(missing))
        complete = not missing
        result = _unavailable("pricing quantities unavailable", pinned)
        result.update(
            source="estimated" if subtotals or complete else "unavailable",
            amount=decimal_string(total) if complete else None,
            currency=card.get("currency"),
            denomination=card.get("denomination"),
            complete=complete,
            subtotals=subtotals,
            known_subtotal=decimal_string(total) if subtotals else None,
            token_cost_subtotal=decimal_string(
                sum(
                    (decimal_value(part["amount"]) for part in subtotals if part["component"] in token_component_names),
                    Decimal(0),
                )
            )
            if subtotals
            else None,
            missing_components=missing,
            assumptions=assumptions,
            scope=card.get("scope", "request_inference"),
            basis="published_list_price",
            applied_schedules=applied,
        )
        return result


def usage_cost_summary(baseline: Mapping[str, Any], live: Any) -> str:
    """Format baseline plus live cost without converting currencies or credits."""
    totals: dict[str, Decimal] = {}
    credits: dict[str, Decimal] = {}
    incomplete = False
    estimated = False
    seen = False
    for source in (baseline, live):
        if not source:
            continue
        seen = True
        incomplete = incomplete or _field(source, "complete", False) is not True
        estimated = estimated or bool(_field(source, "estimated", {}))
        for key, destination in (("currencies", totals), ("non_monetary", credits), ("denominations", credits)):
            entries = _field(source, key, {})
            if not isinstance(entries, Mapping):
                continue
            for denomination, entry in entries.items():
                raw = entry.get("amount", entry.get("total")) if isinstance(entry, Mapping) else entry
                if isinstance(entry, Mapping):
                    estimated = estimated or _count(entry.get("estimated_requests")) not in (0, None)
                    try:
                        estimated = estimated or decimal_value(entry.get("estimated", "0")) > 0
                    except ValueError:
                        incomplete = True
                try:
                    destination[denomination] = destination.get(denomination, Decimal(0)) + decimal_value(raw)
                except ValueError:
                    incomplete = True
    pieces = [f"{key} {decimal_string(value)}" for key, value in sorted(totals.items())]
    pieces.extend(f"{decimal_string(value)} {key} (non-monetary)" for key, value in sorted(credits.items()))
    if not pieces:
        return "Cost: unavailable" if seen else ""
    qualifier = "partial subtotal" if incomplete else ("includes estimates" if estimated else "recorded")
    return "Cost: " + " · ".join(pieces) + f" ({qualifier})"
