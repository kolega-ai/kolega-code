from __future__ import annotations

import logging
import math
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any, Callable, Generator, Mapping

if TYPE_CHECKING:
    from .usage import NormalizedUsage

logger = logging.getLogger(__name__)

TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cache_read_input_tokens",
    "cache_write_input_tokens",
    "reasoning_output_tokens",
)


class FrozenDict(dict[str, Any]):
    def _immutable(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("accounting facts are immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = __ior__ = _immutable

    def setdefault(self, key: str, default: Any = None) -> Any:
        raise TypeError("accounting facts are immutable")

    def update(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("accounting facts are immutable")

    def __copy__(self) -> FrozenDict:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> FrozenDict:
        return self


def freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        return FrozenDict({key: freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("accounting facts must contain finite JSON values")


def json_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: json_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_payload(item) for item in value]
    return value


def nonnegative_decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        return None
    if isinstance(value, str) and not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value):
        return None
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return amount if amount.is_finite() and amount >= 0 else None


def read_wire(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    result = getattr(value, key, None)
    if result is None:
        extras = getattr(value, "model_extra", None)
        if isinstance(extras, Mapping):
            result = extras.get(key)
    return result


def reported_cost(provider: str, usage: Any) -> dict[str, Any] | None:
    raw: Any = None
    scope = "request"
    provenance: dict[str, Any] = {}
    if provider == "perplexity_agent":
        breakdown = read_wire(usage, "cost")
        raw = read_wire(breakdown, "total_cost")
        provenance["field"] = "usage.cost.total_cost"
        if isinstance(breakdown, Mapping):
            provenance["breakdown"] = json_payload(breakdown)
    elif provider == "openrouter":
        raw = read_wire(usage, "cost")
        scope = "gateway_request_charge"
        provenance["field"] = "usage.cost"
        details = read_wire(usage, "cost_details")
        if isinstance(details, Mapping):
            provenance["cost_details"] = json_payload(details)
    elif provider == "xai":
        ticks = read_wire(usage, "cost_in_usd_ticks")
        if type(ticks) is not int or ticks < 0:
            return None
        raw = Decimal(f"{ticks}e-10")
        provenance = {"field": "usage.cost_in_usd_ticks", "cost_in_usd_ticks": ticks, "ticks_per_usd": 10**10}
    amount = nonnegative_decimal(raw)
    if amount is None:
        return None
    try:
        return freeze_json(
            {
                "source": "provider_reported",
                "amount": str(amount),
                "currency": "USD",
                "denomination": "currency",
                "scope": scope,
                "complete": True,
                "missing_components": [],
                "provenance": provenance,
            }
        )
    except ValueError:
        # An invalid optional breakdown must not discard a valid total.
        provenance = {"field": provenance["field"]}
        return freeze_json(
            {
                "source": "provider_reported",
                "amount": str(amount),
                "currency": "USD",
                "denomination": "currency",
                "scope": scope,
                "complete": True,
                "missing_components": [],
                "provenance": provenance,
            }
        )


@dataclass(frozen=True, slots=True)
class UsageObservation:
    provider: str
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_write_input_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    complete: bool = False
    cost: dict[str, Any] | None = None
    pricing_quantities: dict[str, Any] = field(default_factory=dict)
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str) or not self.provider:
            raise ValueError("provider is required")
        if self.model is not None and not isinstance(self.model, str):
            raise ValueError("invalid model")
        if type(self.complete) is not bool:
            raise ValueError("invalid completeness")
        if self.unavailable_reason is not None and not isinstance(self.unavailable_reason, str):
            raise ValueError("invalid unavailable reason")
        for name in TOKEN_FIELDS:
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"invalid {name}")
        if self.input_tokens is not None and self.output_tokens is not None:
            total = self.input_tokens + self.output_tokens
            if self.total_tokens is None:
                object.__setattr__(self, "total_tokens", total)
            elif self.total_tokens != total:
                raise ValueError("total must equal inclusive input plus output")
        if self.complete and any(getattr(self, name) is None for name in TOKEN_FIELDS[:3]):
            raise ValueError("complete observation needs core token counts")
        if self.cost is not None:
            if not isinstance(self.cost, Mapping) or nonnegative_decimal(self.cost.get("amount")) is None:
                raise ValueError("invalid observed cost")
            object.__setattr__(self, "cost", freeze_json(self.cost))
        if not isinstance(self.pricing_quantities, Mapping):
            raise ValueError("invalid pricing quantities")
        object.__setattr__(self, "pricing_quantities", freeze_json(self.pricing_quantities))

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            **{name: getattr(self, name) for name in TOKEN_FIELDS},
            "complete": self.complete,
            "cost": json_payload(self.cost),
            "pricing_quantities": json_payload(self.pricing_quantities),
            "unavailable_reason": self.unavailable_reason,
        }

    @classmethod
    def from_dict(cls, data: Any) -> UsageObservation | None:
        if not isinstance(data, dict):
            return None
        try:
            return cls(**data)
        except (TypeError, ValueError):
            return None

    @classmethod
    def from_usage(cls, usage: NormalizedUsage) -> UsageObservation:
        return cls(
            provider=usage.provider,
            model=usage.model,
            **{name: getattr(usage, name) for name in TOKEN_FIELDS},
            complete=usage.reported,
            unavailable_reason=usage.unavailable_reason,
        )

    def refined_by(self, newer: UsageObservation) -> UsageObservation:
        values = self.to_dict()
        incoming = newer.to_dict()
        for name in TOKEN_FIELDS + ("model", "cost", "unavailable_reason"):
            if self.complete and not newer.complete and name in TOKEN_FIELDS[:3] + ("unavailable_reason",):
                continue
            if incoming[name] is not None:
                values[name] = incoming[name]
        values["pricing_quantities"].update(incoming["pricing_quantities"])
        values["complete"] = self.complete or newer.complete
        if newer.complete:
            values["unavailable_reason"] = newer.unavailable_reason
        if (newer.complete or not self.complete) and (
            newer.input_tokens is not None or newer.output_tokens is not None
        ):
            values["total_tokens"] = None
        return UsageObservation(**values)


_observation_listener: ContextVar[Callable[[UsageObservation], None] | None] = ContextVar(
    "usage_observation_listener", default=None
)


@contextmanager
def observe_usage(listener: Callable[[UsageObservation], None]) -> Generator[None, None, None]:
    token = _observation_listener.set(listener)
    try:
        yield
    finally:
        _observation_listener.reset(token)


def publish_observation(observation: UsageObservation) -> None:
    listener = _observation_listener.get()
    if listener is not None:
        try:
            listener(observation)
        except Exception:
            logger.exception("Usage observation listener failed")
