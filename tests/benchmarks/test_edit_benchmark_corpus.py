from pathlib import Path

import pytest

from benchmarks.edit_tools.corpus import load_suite
from benchmarks.edit_tools.models import FileContent, TaskSpec


ROOT = Path(__file__).resolve().parents[2]
SUITES = ROOT / "benchmarks" / "edit_tools" / "suites"


def test_core_suite_expands_deterministically() -> None:
    first_suite, first = load_suite(SUITES / "core.yaml")
    second_suite, second = load_suite(SUITES / "core.yaml")

    assert first_suite.id == "edit-core"
    assert len(first_suite.curated_tasks) == 12
    assert len(first) == 48
    assert [task.digest for task in first] == [task.digest for task in second]
    assert len({task.id for task in first}) == 48
    assert sum(task.provenance == "synthetic" for task in first) == 36


def test_smoke_suite_selects_four_curated_cases() -> None:
    suite, tasks = load_suite(SUITES / "smoke.yaml")

    assert suite.id == "edit-smoke"
    assert [task.id for task in tasks] == [
        "python-literal",
        "python-insert-function",
        "create-changelog",
        "coordinated-version",
    ]


def test_task_rejects_workspace_escape() -> None:
    with pytest.raises(ValueError, match="project-relative"):
        TaskSpec(
            id="escape",
            prompt="escape",
            before_files={"../outside": FileContent(text="old")},
            expected_files={"safe": FileContent(text="new")},
        )
