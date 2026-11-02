from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

PRICING_SCHEMA_VERSION = 1
PRICING_STATUSES = frozenset(
    {"priced", "subscription", "self_hosted", "requires_configuration", "unpublished", "retired", "unverified"}
)
RATE_UNITS = {"per_token": "1", "per_million_tokens": "1000000", "per_thousand_tokens": "1000", "per_request": "1"}


def decimal_value(value: Any) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, Decimal)):
        raise ValueError("a finite nonnegative decimal string is required")
    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("invalid decimal") from exc
    if not parsed.is_finite() or parsed < 0:
        raise ValueError("a finite nonnegative decimal is required")
    return parsed


def decimal_string(value: Decimal) -> str:
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def pricing_datetime(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("an ISO timestamp is required")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("invalid ISO timestamp") from exc
    if parsed.tzinfo is None:
        if len(value) != 10:
            raise ValueError("timestamps must include a timezone")
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def public_source_url(value: str) -> str:
    parts = urlsplit(value)
    if parts.scheme not in {"https", "http"} or not parts.hostname or parts.username or parts.password:
        raise ValueError("pricing sources must be public HTTP(S) URLs without credentials")
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _rates(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("rates must be a mapping")
    for name, rate in value.items():
        if not isinstance(name, str) or not isinstance(rate, str):
            raise ValueError("rates must have string names and decimal-string values")
        decimal_value(rate)


def _conditions(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("conditions must be a mapping")
    if "timezone" in value:
        try:
            ZoneInfo(value["timezone"])
        except (ZoneInfoNotFoundError, TypeError, ValueError) as exc:
            raise ValueError("invalid schedule timezone") from exc
    for name in ("effective_from", "effective_to"):
        if name in value:
            pricing_datetime(value[name])
    for name in ("min_input_tokens", "max_input_tokens", "min_tokens", "max_tokens"):
        if name in value and (type(value[name]) is not int or value[name] < 0):
            raise ValueError(f"{name} must be a nonnegative integer")
    if "weekdays" in value:
        days = value["weekdays"]
        if not isinstance(days, list) or any(type(day) is not int or not 0 <= day <= 6 for day in days):
            raise ValueError("weekdays must be integers from Monday=0 to Sunday=6")
    for name in ("start_time", "end_time"):
        if name in value:
            if not isinstance(value[name], str) or not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value[name]):
                raise ValueError(f"{name} must be HH:MM")
    for name in ("dates", "excluded_dates", "holidays", "working_dates"):
        if name in value:
            if not isinstance(value[name], list):
                raise ValueError(f"{name} must be a list")
            for date in value[name]:
                if not isinstance(date, str) or len(date) != 10:
                    raise ValueError(f"{name} must contain ISO dates")
                pricing_datetime(date)


def _component(value: Any) -> None:
    if not isinstance(value, Mapping) or not isinstance(value.get("name"), str):
        raise ValueError("components require a name")
    if not isinstance(value.get("rate"), str):
        raise ValueError("component rate must be a decimal string")
    decimal_value(value["rate"])
    if value.get("unit", "per_request") not in RATE_UNITS:
        raise ValueError("unsupported component unit")
    if not isinstance(value.get("quantity", value["name"]), str):
        raise ValueError("component quantity must be a field name")
    if value.get("covers") is not None and value["covers"] not in {"input", "output", "cache_read", "cache_write"}:
        raise ValueError("component covers must identify a normalized token partition")
    _conditions(value.get("when", {}))


def validate_pricing(value: Any) -> None:
    if not isinstance(value, Mapping) or type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("unsupported pricing schema_version")
    for name in ("pricing_id", "provider", "model", "billing_mode", "checked_at"):
        if not isinstance(value.get(name), str) or not value[name]:
            raise ValueError(f"pricing requires {name}")
    if value.get("status") not in PRICING_STATUSES:
        raise ValueError("invalid pricing status")
    if value["status"] != "priced" and not value.get("reason"):
        raise ValueError("non-priced records require a precise reason")
    pricing_datetime(value["checked_at"])
    for name in ("effective_from", "effective_to"):
        if value.get(name) is not None:
            pricing_datetime(value[name])
    if value.get("effective_from") and value.get("effective_to"):
        if pricing_datetime(value["effective_from"]) >= pricing_datetime(value["effective_to"]):
            raise ValueError("effective_to must follow effective_from")
    currency, denomination = value.get("currency"), value.get("denomination")
    if currency is not None and (not isinstance(currency, str) or not re.fullmatch("[A-Z]{3}", currency)):
        raise ValueError("currency must be an ISO currency code")
    if denomination is not None and (not isinstance(denomination, str) or not denomination):
        raise ValueError("denomination must be a nonempty string")
    if currency is not None and denomination is not None:
        raise ValueError("currency and provider-credit denomination are mutually exclusive")
    if value["status"] == "priced" and not (currency or denomination):
        raise ValueError("priced records require currency or denomination")
    if value.get("unit", "per_million_tokens") not in RATE_UNITS:
        raise ValueError("unsupported rate unit")
    _rates(value.get("rates", {}))
    _conditions(value.get("conditions", {}))
    _conditions(value.get("default_context", {}))
    if not isinstance(value.get("cache_policy", {}), Mapping):
        raise ValueError("cache_policy must be a mapping")
    sources = value.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("pricing requires source references")
    for source in sources:
        if not isinstance(source, Mapping):
            raise ValueError("source requires url")
        if source.get("kind") != "operator":
            if not isinstance(source.get("url"), str):
                raise ValueError("source requires url")
            public_source_url(source["url"])
        if not isinstance(source.get("model_id"), str) or not source["model_id"]:
            raise ValueError("source requires exact model_id")
        if source.get("checked_at") is not None:
            pricing_datetime(source["checked_at"])
    for name in ("schedules", "components", "exclusions", "assumptions"):
        if name in value and not isinstance(value[name], list):
            raise ValueError(f"{name} must be a list")
    for name in ("exclusions", "assumptions", "missing_components"):
        if name in value and (
            not isinstance(value[name], list) or any(not isinstance(item, str) for item in value[name])
        ):
            raise ValueError(f"{name} must be a list of strings")
    if value["status"] == "priced" and not any(value.get(name) for name in ("rates", "schedules", "components")):
        raise ValueError("priced records require rates, schedules, or components")
    for schedule in value.get("schedules", []):
        if not isinstance(schedule, Mapping):
            raise ValueError("schedules must contain mappings")
        _conditions(schedule.get("when", schedule.get("conditions", {})))
        _rates(schedule.get("rates", {}))
        if "components" in schedule and not isinstance(schedule["components"], list):
            raise ValueError("schedule components must be a list")
        if "priority" in schedule and type(schedule["priority"]) is not int:
            raise ValueError("schedule priority must be an integer")
        if "quantities" in schedule and (
            not isinstance(schedule["quantities"], Mapping)
            or any(
                not isinstance(key, str) or not isinstance(item, str) for key, item in schedule["quantities"].items()
            )
        ):
            raise ValueError("schedule quantities must map component names to observed field names")
        if schedule.get("mode", "whole_request") not in {"whole_request", "marginal"}:
            raise ValueError("schedule mode must be whole_request or marginal")
        if schedule.get("mode") == "marginal":
            for name in ("min_tokens", "max_tokens"):
                if name in schedule and (type(schedule[name]) is not int or schedule[name] < 0):
                    raise ValueError(f"{name} must be a nonnegative integer")
            if schedule.get("max_tokens") is not None and schedule["max_tokens"] <= schedule.get("min_tokens", 0):
                raise ValueError("marginal bands must have positive width")
        for component in schedule.get("components", []):
            _component(component)
    for component in value.get("components", []):
        _component(component)
    for name in ("cache_read", "cache_write"):
        policy = value.get("cache_policy", {}).get(name)
        if policy is not None and policy not in {"unsupported", "included", "separate", "supplemental"}:
            raise ValueError("invalid cache billing policy")
    json.dumps(dict(value), allow_nan=False)


@dataclass(frozen=True, slots=True)
class PricingCard:
    serialized: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PricingCard:
        validate_pricing(value)
        copied = json.loads(json.dumps(dict(value), allow_nan=False))
        for source in copied["sources"]:
            if source.get("url"):
                source["url"] = public_source_url(source["url"])
        return cls(json.dumps(copied, sort_keys=True, separators=(",", ":"), allow_nan=False))

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.serialized)


def pricing_to_json(value: Any) -> dict[str, Any]:
    return value.to_dict() if isinstance(value, PricingCard) else PricingCard.from_dict(value).to_dict()


def pricing_is_newer(candidate: Any, current: Any) -> bool:
    try:
        new = pricing_to_json(candidate)
        if current is None:
            return True
        old = pricing_to_json(current)
        return (new["provider"], new["model"]) == (old["provider"], old["model"]) and pricing_datetime(
            new["checked_at"]
        ) > pricing_datetime(old["checked_at"])
    except (ValueError, TypeError, KeyError):
        return False
