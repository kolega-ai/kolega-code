from benchmarks.edit_tools.models import ToolAttempt, TrialRecord
from benchmarks.edit_tools.report import (
    aggregate,
    breakdown,
    markdown_report,
    paired_comparisons,
    wilson_interval,
)


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


def test_report_distinguishes_task_exact_and_first_edit_success() -> None:
    recovered = record("hashline_v2", 1, True).model_copy(
        update={
            "exact_match": False,
            "first_attempt_success": False,
            "tool_attempts": [
                ToolAttempt(
                    iteration=1,
                    name="edit",
                    input_kind="json",
                    raw_input={"path": "example.py", "edits": []},
                    apply_ok=False,
                    is_error=True,
                    error="invalid anchor",
                ),
                ToolAttempt(
                    iteration=2,
                    name="edit",
                    input_kind="json",
                    raw_input={"path": "example.py", "edits": []},
                    apply_ok=True,
                ),
            ],
        }
    )
    first_try = record("hashline_v2", 2, True).model_copy(
        update={
            "exact_match": True,
            "first_attempt_success": True,
            "tool_attempts": [
                ToolAttempt(
                    iteration=1,
                    name="edit",
                    input_kind="json",
                    raw_input={"path": "example.py", "edits": []},
                    apply_ok=True,
                )
            ],
        }
    )
    records = [recovered, first_try]

    rows = aggregate(records)
    row = rows[0]
    assert row["success_rate"] == 1.0
    assert row["exact_match_rate"] == 0.5
    assert row["first_edit_attempt_successes"] == 1
    assert row["first_edit_attempts"] == 2
    assert row["first_edit_attempt_success_rate"] == 0.5

    breakdowns = {
        dimension: breakdown(records, dimension)
        for dimension in ("language", "family", "target_length", "payload_size", "target_file_count")
    }
    markdown = markdown_report(rows, [], breakdowns)
    assert "| Scored | Task success | Exact match |" in markdown
    assert "| 2 | 100.0% | 50.0% |" in markdown
    assert "First attempt" in markdown
    assert "50.0% (1/2)" in markdown


def test_end_to_end_first_attempt_counts_no_edit_as_failure() -> None:
    no_edit = record("codex_apply_patch", 1, False)
    first_try = record("codex_apply_patch", 2, True).model_copy(
        update={
            "first_attempt_success": True,
            "tool_attempts": [
                ToolAttempt(
                    iteration=1,
                    name="apply_patch",
                    input_kind="freeform",
                    raw_input="*** Begin Patch\n*** Add File: a\n+x\n*** End Patch",
                    apply_ok=True,
                )
            ],
        }
    )

    row = aggregate([no_edit, first_try])[0]

    assert row["first_attempt_rate"] == 0.5
    assert row["first_edit_attempt_success_rate"] == 1.0
    assert row["no_edit_rate"] == 0.5


def test_first_attempt_rate_is_weighted_per_target_file() -> None:
    partial = record("claude_code", 1, True).model_copy(
        update={
            "first_attempt_file_successes": 1,
            "first_attempt_files": 2,
            "first_attempt_success": False,
        }
    )
    complete = record("claude_code", 2, True).model_copy(
        update={
            "first_attempt_file_successes": 2,
            "first_attempt_files": 2,
            "first_attempt_success": True,
        }
    )

    row = aggregate([partial, complete])[0]

    assert row["first_attempt_file_success_rate"] == 0.75
    assert row["first_attempt_file_successes"] == 3
    assert row["first_attempt_files"] == 4
    assert row["all_files_first_attempt_success_rate"] == 0.5
