from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import date
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any, Mapping

from .pricing import pricing_to_json

CHECKED_AT = "2026-11-01"
OPENROUTER_URL = "https://openrouter.ai/api/v1/models"
TINKER_URL = "https://tinker-docs.thinkingmachines.ai/tinker/models.json"
PERPLEXITY_URL = "https://docs.perplexity.ai/docs/agent-api/models"
CLOUDFLARE_URL = "https://developers.cloudflare.com/workers-ai/platform/pricing/"
OLLAMA_URL = "https://ollama.com/pricing"
_TINKER_RETIRED = {
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "Qwen/Qwen3.6-27B",
    "deepseek-ai/DeepSeek-V3.1",
}


def _decimal(value: Any, *, dollar: bool = False, factor: str = "1") -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError("invalid source rate")
    text = str(value)
    if len(text) > 128:
        raise ValueError("invalid source rate")
    if dollar:
        if not isinstance(value, str) or not text.startswith("$"):
            raise ValueError("expected a dollar-denominated source rate")
        text = text[1:]
    try:
        number = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError("invalid source rate") from exc
    if not number.is_finite() or number < 0 or abs(number.adjusted()) > 100:
        raise ValueError("invalid source rate")
    with localcontext() as context:
        context.prec = max(28, len(number.as_tuple().digits) + len(Decimal(factor).as_tuple().digits))
        return format(number * Decimal(factor), "f")


_OPENROUTER_RATE_FIELDS = {
    "prompt": "input",
    "completion": "output",
    "input_cache_read": "cache_read",
    "input_cache_write": "cache_write",
    "input_cache_write_1h": "cache_write_1h",
}
_OPENROUTER_CONDITION_FIELDS = {"min_prompt_tokens", "max_prompt_tokens", "utc_days", "utc_start", "utc_end"}


def _openrouter_schedules(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    overrides = raw.get("overrides", [])
    if not isinstance(overrides, list):
        raise ValueError("invalid upstream overrides")
    schedules: list[dict[str, Any]] = []
    days = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
    for index, override in enumerate(overrides):
        if not isinstance(override, Mapping):
            raise ValueError("invalid upstream override")
        when: dict[str, Any] = {}
        for src, dest in (("min_prompt_tokens", "min_input_tokens"), ("max_prompt_tokens", "max_input_tokens")):
            if src in override:
                if type(override[src]) is not int or override[src] < 0:
                    raise ValueError("invalid upstream token boundary")
                when[dest] = override[src] + (1 if src == "min_prompt_tokens" else 0)
        if "utc_days" in override:
            selected_days = override["utc_days"]
            if not isinstance(selected_days, list) or any(day not in days for day in selected_days):
                raise ValueError("invalid upstream weekdays")
            when["weekdays"] = [days.index(day) for day in selected_days]
        for src, dest in (("utc_start", "start_time"), ("utc_end", "end_time")):
            if src in override:
                clock = override[src]
                if type(clock) is not int or not 0 <= clock <= 2400 or clock % 100 >= 60:
                    raise ValueError("invalid upstream UTC time")
                when[dest] = f"{clock // 100 % 24:02d}:{clock % 100:02d}"
        if any(key.startswith("utc_") for key in override):
            when["timezone"] = "UTC"
        if when.get("start_time") == when.get("end_time") and "start_time" in when:
            when.pop("start_time")
            when.pop("end_time")
        schedules.append(
            {
                "id": f"upstream_override_{index}",
                "priority": index,
                "when": when,
                "rates": {
                    dest: _decimal(override[src], factor="1000000")
                    for src, dest in _OPENROUTER_RATE_FIELDS.items()
                    if src in override
                },
            }
        )
    return schedules


def _record(
    provider: str,
    model: str,
    url: str,
    rates: Mapping[str, str] | None = None,
    *,
    status: str = "priced",
    reason: str | None = None,
    source_model: str | None = None,
    checked_at: str = CHECKED_AT,
    billing_mode: str = "pay_as_you_go",
    **extra: Any,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": 1,
        "pricing_id": f"{provider}:{model}:{checked_at}",
        "provider": provider,
        "model": model,
        "status": status,
        "billing_mode": billing_mode,
        "currency": "USD",
        "unit": "per_million_tokens",
        "rates": dict(rates or {}),
        "checked_at": checked_at,
        "sources": [{"url": url, "model_id": source_model or model, "checked_at": checked_at}],
        "conditions": {},
        "schedules": [],
        "exclusions": ["Taxes, account credits/free allowances, and negotiated discounts."],
    }
    if reason:
        record["reason"] = reason
    record.update(extra)
    return record


def _fingerprint(record: dict[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    digest = hashlib.sha256(json.dumps(dict(row), sort_keys=True, allow_nan=False).encode()).hexdigest()
    record["source_payload_sha256"] = digest
    record["pricing_id"] += f":{digest[:16]}"
    return record


def _bundled_pricing(provider: str, model: str) -> dict[str, Any] | None:
    from . import catalog

    specs = getattr(catalog, f"{provider.upper()}_SPECS", {})
    if provider == "tinker" and model == "tinker://*":
        specs = catalog.TINKER_WILDCARD_SPECS
    record = specs.get((provider, model), {}).get("pricing")
    return deepcopy(record) if record is not None else None


def feed_pricing(provider: str, model: str, row: Mapping[str, Any], *, checked_at: str | None = None) -> dict[str, Any]:
    stamp = checked_at or date.today().isoformat()
    urls = {"openrouter": OPENROUTER_URL, "tinker": TINKER_URL, "perplexity_agent": PERPLEXITY_URL}
    url = urls[provider]
    try:
        if provider == "openrouter":
            raw = row.get("pricing")
            if not isinstance(raw, Mapping):
                raise ValueError("missing pricing object")
            rates = {
                dest: _decimal(raw[src], factor="1000000")
                for src, dest in _OPENROUTER_RATE_FIELDS.items()
                if src in raw
            }
            if not {"input", "output"} <= rates.keys():
                raise ValueError("missing input/output rates")
            schedules = _openrouter_schedules(raw)
            free = (
                model.endswith(":free")
                and all(Decimal(rate) == 0 for rate in rates.values())
                and all(Decimal(rate) == 0 for schedule in schedules for rate in schedule["rates"].values())
                and all(
                    Decimal(_decimal(value)) == 0
                    for prices in (raw, *raw.get("overrides", []))
                    for key, value in prices.items()
                    if key not in {*_OPENROUTER_RATE_FIELDS, *_OPENROUTER_CONDITION_FIELDS, "overrides"}
                )
            )
            record = _record(
                provider,
                model,
                url,
                rates,
                checked_at=stamp,
                billing_mode="gateway",
                scope="gateway_model_list_rates",
                source_scope="Model-level advertised rates; not an endpoint-pinned request charge.",
                upstream_pricing=deepcopy(dict(raw)),
                routing_variant=model.partition(":")[2] or "standard",
                verified_free=free,
                schedules=schedules,
                assumptions=[
                    "Published OpenRouter model-list rates estimate resource/list value; "
                    "the routed endpoint, service tier and BYOK arrangement may change the actual charge. "
                    "Provider-reported usage.cost takes precedence."
                ],
            )
            if free:
                record["cache_policy"] = {"cache_read": "included", "cache_write": "included"}
            record["components"] = [
                {"name": name, "quantity": quantity, "rate": _decimal(raw[name]), "unit": "per_request"}
                for name, quantity in (("request", "request"), ("web_search", "web_search_calls"))
                if name in raw and Decimal(_decimal(raw[name])) > 0
            ]
            record["missing_components"] = []
            for index, prices in enumerate((raw, *raw.get("overrides", []))):
                for key, value in prices.items():
                    if key in {*_OPENROUTER_RATE_FIELDS, *_OPENROUTER_CONDITION_FIELDS, "overrides"}:
                        continue
                    if index == 0 and key in {"request", "web_search"}:
                        continue
                    if key == "internal_reasoning" and Decimal(_decimal(value)) == Decimal(
                        _decimal(prices.get("completion", raw["completion"]))
                    ):
                        continue
                    if Decimal(_decimal(value)) > 0:
                        missing = f"{key}: published gateway pricing requires a separately observed billing quantity"
                        if missing not in record["missing_components"]:
                            record["missing_components"].append(missing)
        elif provider == "tinker":
            rates = {
                dest: _decimal(row[src], dollar=True)
                for src, dest in (("prefill", "input"), ("cached_prefill", "cache_read"), ("sample", "output"))
            }
            record = _record(
                provider,
                model,
                url,
                rates,
                checked_at=stamp,
                billing_mode="native_sampling",
                cache_policy={"cache_read": "separate", "cache_write": "included"},
            )
            record["exclusions"].extend(["Training/forward passes billed at training rates.", "Checkpoint storage."])
            if row.get("note") and "discount" in str(row["note"]).lower():
                record["promotion"] = {"description": str(row["note"]), "expiration": "not published"}
                record["regular_rates"] = {
                    dest: _decimal(row[src], dollar=True)
                    for src, dest in (
                        ("original_prefill", "input"),
                        ("original_cached_prefill", "cache_read"),
                        ("original_sample", "output"),
                    )
                    if src in row
                }
                record["assumptions"] = ["Published temporary discount; no end date is supplied by the source."]
            if model in _TINKER_RETIRED and stamp >= "2026-10-23":
                record.update(
                    status="retired",
                    reason=f"{model}: official deprecation notice retires this native base model on October 23, 2026; "
                    "listed rates are historical evidence, not a current request price.",
                    effective_to="2026-10-23",
                )
                record["sources"].append(
                    {
                        "url": "https://tinker-docs.thinkingmachines.ai/tinker/model-deprecations/",
                        "model_id": model,
                        "checked_at": CHECKED_AT,
                    }
                )
        else:
            raw = row.get("pricing")
            if not isinstance(raw, Mapping):
                raise ValueError("missing pricing object")
            if isinstance(raw.get("usd_per_1m_tokens"), Mapping):
                prices = raw["usd_per_1m_tokens"]
            elif raw.get("unit") == "usd_per_1m_tokens":
                prices = raw
            else:
                raise ValueError("unrecognized pricing denomination or unit")
            rates = {
                key: _decimal(prices[key]) for key in ("input", "output", "cache_read", "cache_write") if key in prices
            }
            if not {"input", "output"} <= rates.keys():
                raise ValueError("missing input/output rates")
            record = _record(
                provider,
                model,
                "https://api.perplexity.ai/v1/models",
                rates,
                checked_at=stamp,
                billing_mode="agent_api",
            )
            record["missing_components"] = [
                "Hosted tool charges require tool-specific quantities; prefer usage.cost.total_cost.",
                "The model feed alone does not establish context-length and service-tier applicability.",
            ]
        return pricing_to_json(_fingerprint(record, row))
    except (TypeError, ValueError, KeyError, InvalidOperation, OverflowError):
        reviewed = _bundled_pricing(provider, model)
        if reviewed is not None:
            return reviewed
        return _record(
            provider,
            model,
            url,
            status="unverified",
            reason=f"{model}: upstream pricing is absent, malformed, negative/nonfinite, or has unsupported units.",
            checked_at=stamp,
        )


def sanitize_spec_pricing(spec: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(spec)
    if "pricing" in result:
        try:
            result["pricing"] = pricing_to_json(result["pricing"])
        except (TypeError, ValueError, KeyError, OverflowError):
            result.pop("pricing")
    return result


def reviewed_pricing(provider: str, model: str) -> dict[str, Any]:
    record = _bundled_pricing(provider, model)
    if record is not None:
        return record
    urls = {"ollama_cloud": OLLAMA_URL, "cloudflare_workers_ai": CLOUDFLARE_URL}
    return _record(
        provider,
        model,
        urls[provider],
        status="unverified",
        reason=f"{model}: this newly discovered exact route has no reviewed pricing row in the bundled rate card.",
        checked_at=CHECKED_AT,
    )
