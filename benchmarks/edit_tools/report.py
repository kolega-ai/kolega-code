"""Deterministic benchmark aggregation and paired protocol comparisons."""

from __future__ import annotations

from collections import defaultdict
import csv
from io import StringIO
import math
from pathlib import Path
import random
import statistics
from typing import Any, Iterable

from .artifacts import atomic_write_text, write_json
from .models import TrialRecord
from .protocols import get_protocol


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    spread = z * math.sqrt((proportion * (1 - proportion) + z * z / (4 * total)) / total) / denominator
    return max(0.0, center - spread), min(1.0, center + spread)


def _percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return float(ordered[index])


def aggregate(records: Iterable[TrialRecord]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[TrialRecord]] = defaultdict(list)
    for record in records:
        groups[(record.provider, record.model, record.protocol, record.lane)].append(record)
    rows: list[dict[str, Any]] = []
    for (provider, model, protocol, lane), items in sorted(groups.items()):
        scored = [item for item in items if item.scored]
        successes = sum(item.task_success for item in scored)
        low, high = wilson_interval(successes, len(scored))
        edit_names = set(get_protocol(protocol).tool_names)
        edit_attempts = [attempt for item in scored for attempt in item.tool_attempts if attempt.name in edit_names]
        first_edit_attempts = [
            attempts[0]
            for item in scored
            if (attempts := [attempt for attempt in item.tool_attempts if attempt.name in edit_names])
        ]
        first_edit_attempt_successes = sum(attempt.apply_ok for attempt in first_edit_attempts)
        latency = [item.elapsed_ms for item in scored]
        family_groups: dict[str, list[TrialRecord]] = defaultdict(list)
        for item in scored:
            family_groups[item.family or "unknown"].append(item)
        macro_family_success = (
            sum(sum(child.task_success for child in family) / len(family) for family in family_groups.values())
            / len(family_groups)
            if family_groups
            else None
        )
        rows.append(
            {
                "provider": provider,
                "model": model,
                "protocol": protocol,
                "lane": lane,
                "planned_or_recorded": len(items),
                "scored": len(scored),
                "passed": successes,
                "success_rate": successes / len(scored) if scored else None,
                "macro_family_success_rate": macro_family_success,
                "functional_success_rate": (
                    sum(item.functional_success for item in scored) / len(scored) if scored else None
                ),
                "instruction_success_rate": (
                    sum(item.instruction_success for item in scored) / len(scored) if scored else None
                ),
                "exact_match_rate": sum(item.exact_match for item in scored) / len(scored) if scored else None,
                "collateral_success_rate": (
                    sum(item.collateral_success for item in scored) / len(scored) if scored else None
                ),
                "operation_success_rate": (
                    sum(item.completed_operations for item in scored) / sum(item.total_operations for item in scored)
                    if sum(item.total_operations for item in scored)
                    else None
                ),
                "success_ci_low": low if scored else None,
                "success_ci_high": high if scored else None,
                # Retained for report consumers created before the metric was
                # given an explicit edit-tool name. Its denominator is all
                # scored trials, so no-edit trials count as unsuccessful.
                "first_attempt_rate": (
                    sum(item.first_attempt_success for item in scored) / len(scored) if scored else None
                ),
                "first_edit_attempt_successes": first_edit_attempt_successes,
                "first_edit_attempts": len(first_edit_attempts),
                "first_edit_attempt_success_rate": (
                    first_edit_attempt_successes / len(first_edit_attempts) if first_edit_attempts else None
                ),
                "parse_success_rate": (
                    sum(attempt.parse_ok for attempt in edit_attempts) / len(edit_attempts) if edit_attempts else None
                ),
                "apply_success_rate": (
                    sum(attempt.apply_ok for attempt in edit_attempts) / len(edit_attempts) if edit_attempts else None
                ),
                "no_edit_rate": (
                    sum(not any(attempt.name in edit_names for attempt in item.tool_attempts) for item in scored)
                    / len(scored)
                    if scored
                    else None
                ),
                "collateral_rate": (
                    sum(bool(item.collateral_paths) for item in scored) / len(scored) if scored else None
                ),
                "avg_edit_attempts": len(edit_attempts) / len(scored) if scored else None,
                "recovery_rate": (
                    sum(
                        any(not attempt.apply_ok for attempt in item.tool_attempts if attempt.name in edit_names)
                        and any(attempt.apply_ok for attempt in item.tool_attempts if attempt.name in edit_names)
                        for item in scored
                    )
                    / len(scored)
                    if scored
                    else None
                ),
                "iteration_exhaustion_rate": (
                    sum(bool(item.metadata.get("iteration_exhausted")) for item in scored) / len(scored)
                    if scored
                    else None
                ),
                "median_latency_ms": statistics.median(latency) if latency else None,
                "p95_latency_ms": _percentile(latency, 0.95) if latency else None,
                "avg_input_tokens": (sum(item.usage.input_tokens for item in scored) / len(scored) if scored else None),
                "avg_output_tokens": (
                    sum(item.usage.output_tokens for item in scored) / len(scored) if scored else None
                ),
                "provider_errors": sum(item.status == "provider_error" for item in items),
                "harness_errors": sum(item.status == "harness_error" for item in items),
                "not_run": sum(item.status == "not_run" for item in items),
                "unsupported": sum(item.status == "unsupported" for item in items),
            }
        )
    return rows


def breakdown(records: Iterable[TrialRecord], dimension: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[TrialRecord]] = defaultdict(list)
    for record in records:
        value = str(getattr(record, dimension, None) or "unknown")
        groups[(record.provider, record.model, record.protocol, record.lane, value)].append(record)
    rows: list[dict[str, Any]] = []
    for (provider, model, protocol, lane, value), items in sorted(groups.items()):
        scored = [item for item in items if item.scored]
        edit_names = set(get_protocol(protocol).tool_names)
        first_edit_attempts = [
            attempts[0]
            for item in scored
            if (attempts := [attempt for attempt in item.tool_attempts if attempt.name in edit_names])
        ]
        first_edit_attempt_successes = sum(attempt.apply_ok for attempt in first_edit_attempts)
        rows.append(
            {
                "provider": provider,
                "model": model,
                "protocol": protocol,
                "lane": lane,
                dimension: value,
                "scored": len(scored),
                "success_rate": sum(item.task_success for item in scored) / len(scored) if scored else None,
                "functional_success_rate": (
                    sum(item.functional_success for item in scored) / len(scored) if scored else None
                ),
                "exact_match_rate": sum(item.exact_match for item in scored) / len(scored) if scored else None,
                "first_edit_attempt_successes": first_edit_attempt_successes,
                "first_edit_attempts": len(first_edit_attempts),
                "first_edit_attempt_success_rate": (
                    first_edit_attempt_successes / len(first_edit_attempts) if first_edit_attempts else None
                ),
                "operation_success_rate": (
                    sum(item.completed_operations for item in scored) / sum(item.total_operations for item in scored)
                    if sum(item.total_operations for item in scored)
                    else None
                ),
            }
        )
    return rows


def _paired_bootstrap(
    left: dict[tuple[str, int], bool],
    right: dict[tuple[str, int], bool],
    *,
    samples: int = 10_000,
    seed: int = 1,
) -> tuple[float, float, float, int]:
    common = sorted(set(left) & set(right))
    task_deltas: dict[str, list[float]] = defaultdict(list)
    for task_id, repetition in common:
        task_deltas[task_id].append(float(left[(task_id, repetition)]) - float(right[(task_id, repetition)]))
    grouped = [sum(values) / len(values) for _, values in sorted(task_deltas.items())]
    if not grouped:
        return 0.0, 0.0, 0.0, 0
    observed = sum(grouped) / len(grouped)
    rng = random.Random(seed)
    bootstrapped = sorted(sum(rng.choice(grouped) for _ in grouped) / len(grouped) for _ in range(samples))
    low = bootstrapped[int(0.025 * (samples - 1))]
    high = bootstrapped[int(0.975 * (samples - 1))]
    return observed, low, high, len(common)


def paired_comparisons(records: Iterable[TrialRecord]) -> list[dict[str, Any]]:
    by_model: dict[tuple[str, str, str], list[TrialRecord]] = defaultdict(list)
    for record in records:
        if record.scored:
            by_model[(record.provider, record.model, record.lane)].append(record)
    comparisons: list[dict[str, Any]] = []
    for (provider, model, lane), items in sorted(by_model.items()):
        protocols = sorted({item.protocol for item in items})
        for left_index, left_protocol in enumerate(protocols):
            for right_protocol in protocols[left_index + 1 :]:
                left = {
                    (item.task_id, item.repetition): item.task_success
                    for item in items
                    if item.protocol == left_protocol
                }
                right = {
                    (item.task_id, item.repetition): item.task_success
                    for item in items
                    if item.protocol == right_protocol
                }
                delta, low, high, pairs = _paired_bootstrap(left, right)
                leader = "tie"
                if low > 0:
                    leader = left_protocol
                elif high < 0:
                    leader = right_protocol
                comparisons.append(
                    {
                        "provider": provider,
                        "model": model,
                        "lane": lane,
                        "left_protocol": left_protocol,
                        "right_protocol": right_protocol,
                        "paired_trials": pairs,
                        "success_delta": delta,
                        "delta_ci_low": low,
                        "delta_ci_high": high,
                        "leader": leader,
                    }
                )
    return comparisons


def _format_rate(value: Any) -> str:
    return "—" if value is None else f"{100 * float(value):.1f}%"


def _format_counted_rate(value: Any, successes: int, attempts: int) -> str:
    if value is None:
        return "—"
    return f"{_format_rate(value)} ({successes}/{attempts})"


def markdown_report(
    rows: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    breakdowns: dict[str, list[dict[str, Any]]],
) -> str:
    lines = [
        "# Edit-tool benchmark report",
        "",
        "Provider and harness failures are not included in scored success-rate denominators.",
        "Task success means all configured oracle checks passed; exact match means the resulting workspace "
        "matched the expected workspace exactly. First edit attempt measures successful first edit-tool "
        "applications among trials that made an edit attempt.",
        "",
        "| Provider/model | Lane | Protocol | Scored | Task success | Exact match | Operations | Family macro | 95% CI | First edit attempt | Apply | Infra/not run |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        ci = (
            "—"
            if row["success_ci_low"] is None
            else f"{_format_rate(row['success_ci_low'])}–{_format_rate(row['success_ci_high'])}"
        )
        infra = row["provider_errors"] + row["harness_errors"] + row["not_run"]
        lines.append(
            f"| {row['provider']}/{row['model']} | {row['lane']} | {row['protocol']} | {row['scored']} | "
            f"{_format_rate(row['success_rate'])} | {_format_rate(row['exact_match_rate'])} | "
            f"{_format_rate(row['operation_success_rate'])} | "
            f"{_format_rate(row['macro_family_success_rate'])} | {ci} | "
            f"{_format_counted_rate(row['first_edit_attempt_success_rate'], row['first_edit_attempt_successes'], row['first_edit_attempts'])} | "
            f"{_format_rate(row['apply_success_rate'])} | {infra} |"
        )
    for dimension in ("language", "family", "target_length", "payload_size", "target_file_count"):
        lines.extend(
            [
                "",
                f"## By {dimension}",
                "",
                f"| Provider/model | Lane | Protocol | {dimension.replace('_', ' ').title()} | Scored | Task success | Exact match | First edit attempt | Operations |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in breakdowns[dimension]:
            lines.append(
                f"| {item['provider']}/{item['model']} | {item['lane']} | {item['protocol']} | "
                f"{item[dimension]} | {item['scored']} | {_format_rate(item['success_rate'])} | "
                f"{_format_rate(item['exact_match_rate'])} | "
                f"{_format_counted_rate(item['first_edit_attempt_success_rate'], item['first_edit_attempt_successes'], item['first_edit_attempts'])} | "
                f"{_format_rate(item['operation_success_rate'])} |"
            )
    lines.extend(
        [
            "",
            "## Paired protocol comparisons",
            "",
            "A leader is named only when the task-level paired bootstrap interval excludes zero.",
            "",
            "| Provider/model | Lane | Comparison | Pairs | Delta | 95% CI | Leader |",
            "| --- | --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for item in comparisons:
        lines.append(
            f"| {item['provider']}/{item['model']} | {item['lane']} | "
            f"{item['left_protocol']} − {item['right_protocol']} | {item['paired_trials']} | "
            f"{item['success_delta']:+.3f} | {item['delta_ci_low']:+.3f}–{item['delta_ci_high']:+.3f} | "
            f"{item['leader']} |"
        )
    return "\n".join(lines) + "\n"


def write_report(run_dir: Path, records: list[TrialRecord]) -> dict[str, Any]:
    rows = aggregate(records)
    comparisons = paired_comparisons(records)
    breakdowns = {
        dimension: breakdown(records, dimension)
        for dimension in ("language", "family", "target_length", "payload_size", "target_file_count")
    }
    summary = {
        "schema_version": 1,
        "groups": rows,
        "breakdowns": breakdowns,
        "paired_comparisons": comparisons,
    }
    write_json(run_dir / "summary.json", summary)
    atomic_write_text(run_dir / "summary.md", markdown_report(rows, comparisons, breakdowns))
    output = StringIO()
    if rows:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_write_text(run_dir / "summary.csv", output.getvalue())
    return summary
