from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kolega_code.config import ModelProvider  # noqa: E402
from kolega_code.llm.specs.catalog import MODEL_SPECS, WILDCARD_MODEL_SPECS  # noqa: E402
from kolega_code.llm.specs.catalog.pricing_sources import PROVIDER_SOURCES  # noqa: E402
from kolega_code.llm.specs.pricing import pricing_datetime, validate_pricing  # noqa: E402


def pricing_coverage(*, now: datetime | None = None) -> dict[str, Any]:
    checked = now or datetime.now(timezone.utc)
    providers = {provider.value: {"models": 0, "statuses": Counter()} for provider in ModelProvider}
    exceptions: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    stale: list[dict[str, Any]] = []
    for provider in (*providers, "custom:*", "tinker://*"):
        policy = PROVIDER_SOURCES.get(provider)
        if not isinstance(policy, dict) or any(
            not isinstance(policy.get(name), str) or not policy[name].strip()
            for name in ("url", "checked_at", "billing_surface", "policy")
        ):
            errors.append({"provider": provider, "model": "*", "error": "missing documented provider billing policy"})
        elif provider in providers:
            providers[provider]["policy"] = policy
    for (provider, model), spec in {**MODEL_SPECS, **WILDCARD_MODEL_SPECS}.items():
        pricing = spec.get("pricing")
        row = {"provider": provider, "model": model}
        if provider not in providers:
            errors.append({**row, "error": "bundled model has no built-in provider policy"})
            continue
        providers[provider]["models"] += 1
        try:
            if not isinstance(pricing, dict):
                raise ValueError("missing structured pricing policy")
            validate_pricing(pricing)
            if pricing["provider"] != provider or pricing["model"] != model:
                raise ValueError("pricing identity differs from the billed catalogue route")
        except (TypeError, KeyError, ValueError) as exc:
            errors.append({**row, "error": str(exc)})
            providers[provider]["statuses"]["missing_or_invalid"] += 1
            continue
        status = pricing["status"]
        providers[provider]["statuses"][status] += 1
        if status != "priced":
            reason = pricing["reason"].lower()
            unfinished = ("todo", "not researched", "research pending", "pending research", "unverified default")
            if any(marker in reason for marker in unfinished):
                errors.append({**row, "error": "unfinished pricing research is not a sourced model exception"})
        age = (checked - pricing_datetime(pricing["checked_at"])).days
        if age > 30:
            stale.append({**row, "age_days": age, "checked_at": pricing["checked_at"]})
        if status != "priced":
            exceptions.append(
                {
                    **row,
                    "status": status,
                    "reason": pricing["reason"],
                    "sources": pricing["sources"],
                    "pricing_id": pricing["pricing_id"],
                }
            )
    return {
        "checked_at": checked.isoformat(),
        "provider_count": len(providers),
        "model_count": len(MODEL_SPECS),
        "wildcard_count": len(WILDCARD_MODEL_SPECS),
        "providers": providers,
        "exceptions": exceptions,
        "errors": errors,
        "due_for_review": stale,
        "valid": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit exact-route bundled pricing and list sourced exceptions.")
    parser.add_argument("--json", action="store_true", help="Print the complete machine-readable coverage report.")
    args = parser.parse_args()
    report = pricing_coverage()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for provider, row in report["providers"].items():
            statuses = ", ".join(f"{status}={count}" for status, count in sorted(row["statuses"].items()))
            print(f"{provider}: {row['models']} models; {statuses or 'provider policy only'}")
        for row in report["errors"]:
            print(f"ERROR {row['provider']}/{row['model']}: {row['error']}")
        for row in report["exceptions"]:
            print(f"{row['status'].upper()} {row['provider']}/{row['model']}: {row['reason']}")
        print(
            f"{report['provider_count']} providers; {report['model_count']} bundled models; "
            f"{len(report['exceptions'])} sourced exceptions; {len(report['errors'])} errors; "
            f"{len(report['due_for_review'])} cards older than 30 days."
        )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
