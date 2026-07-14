from benchmarks.edit_tools.models import TrialRecord
from benchmarks.edit_tools.report import aggregate, breakdown, paired_comparisons, wilson_interval


def record(protocol: str, repetition: int, success: bool) -> TrialRecord:
    return TrialRecord(
        trial_id=f"{protocol}-{repetition}",
        run_id="run",
        suite_id="suite",
        task_id="task",
        task_digest="digest",
        lane="controlled",
        provider="anthropic",
        model="model",
        protocol=protocol,
        protocol_version="1",
        repetition=repetition,
        seed=repetition,
        status="passed" if success else "failed",
        task_success=success,
        oracle_success=success,
        completed_operations=2 if success else 1,
        total_operations=2,
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        elapsed_ms=1000,
        language="python",
        family="localized-replacement",
        difficulty="easy",
        shape="mechanical",
        target_length="medium",
        payload_size="small",
        target_file_count=1,
        artifact_dir="artifacts/test",
    )


def test_aggregate_and_paired_comparison_do_not_count_infrastructure() -> None:
    records = [
        record("search_replace", 1, False),
        record("codex_apply_patch", 1, True),
        record("search_replace", 2, False),
        record("codex_apply_patch", 2, True),
    ]

    rows = aggregate(records)
    comparisons = paired_comparisons(records)

    assert {row["protocol"]: row["success_rate"] for row in rows} == {
        "codex_apply_patch": 1.0,
        "search_replace": 0.0,
    }
    assert comparisons[0]["paired_trials"] == 2
    assert comparisons[0]["leader"] == "codex_apply_patch"
    assert {row["language"] for row in breakdown(records, "language")} == {"python"}
    assert all(row["exact_match_rate"] in {0.0, 1.0} for row in rows)
    assert {row["protocol"]: row["operation_success_rate"] for row in rows} == {
        "codex_apply_patch": 1.0,
        "search_replace": 0.5,
    }


def test_wilson_interval_is_bounded() -> None:
    low, high = wilson_interval(8, 10)
    assert 0 < low < 0.8 < high < 1
