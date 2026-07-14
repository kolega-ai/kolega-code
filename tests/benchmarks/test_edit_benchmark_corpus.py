from collections import Counter
from hashlib import sha256
from pathlib import Path

import pytest
import yaml

from benchmarks.edit_tools.corpus import load_suite
from benchmarks.edit_tools.models import FileContent, TaskSpec
from benchmarks.edit_tools.recipes import line_count, target_length_bucket
from benchmarks.edit_tools.workspace import materialize_task, materialize_tree, verify_task


ROOT = Path(__file__).resolve().parents[2]
SUITES = ROOT / "benchmarks" / "edit_tools" / "suites"
SNAPSHOTS = ROOT / "benchmarks" / "edit_tools" / "fixtures" / "snapshots"


def test_core_suite_expands_deterministically() -> None:
    first_suite, first = load_suite(SUITES / "core.yaml")
    second_suite, second = load_suite(SUITES / "core.yaml")

    assert first_suite.id == "edit-core"
    assert len(first_suite.curated_tasks) == 100
    assert len(first) == 100
    assert [task.digest for task in first] == [task.digest for task in second]
    assert len({task.id for task in first}) == 100
    assert Counter(task.provenance for task in first) == {"synthetic": 100}
    assert Counter(task.shape for task in first) == {"mechanical": 100}
    assert Counter(task.language for task in first) == {
        "python": 9,
        "typescript": 9,
        "javascript": 8,
        "go": 9,
        "rust": 9,
        "java": 8,
        "cpp": 8,
        "csharp": 8,
        "ruby": 8,
        "php": 8,
        "swift": 8,
        "kotlin": 8,
    }
    assert Counter(task.target_length for task in first) == {
        "short": 10,
        "normal": 25,
        "medium": 35,
        "long": 20,
        "oversized": 10,
    }
    assert Counter(task.family for task in first) == {
        "localized-replacement": 10,
        "block-insertion": 12,
        "targeted-removal": 10,
        "nested-file-creation": 8,
        "same-file-multi-hunk": 12,
        "signature-callsite": 14,
        "coordinated-multi-file": 16,
        "ambiguous-context": 10,
        "structured-data": 8,
    }
    assert Counter(_scope(len(task.recipe.operations)) for task in first if task.recipe) == {
        "one": 25,
        "few": 35,
        "several": 30,
        "many": 10,
    }
    assert Counter(
        _file_scope(len({operation.path for operation in task.recipe.operations})) for task in first if task.recipe
    ) == {"one": 50, "few": 35, "many": 15}
    assert len({(task.snapshot_id, task.primary_target) for task in first}) == 100
    assert len({task.snapshot_id for task in first}) == 12
    assert all(task.recipe and task.authoring and task.snapshot_id for task in first)
    for task in first:
        assert task.primary_target is not None
        assert task.recipe is not None
        assert target_length_bucket(line_count(task.before_files[task.primary_target].text)) == task.target_length
        assert all(operation.new_text in task.prompt for operation in task.recipe.operations if operation.new_text)
        kinds = Counter(operation.kind for operation in task.recipe.operations)
        if task.family == "localized-replacement":
            assert kinds == {"replace": 1}
        elif task.family == "block-insertion":
            assert kinds["insert"]
        elif task.family == "targeted-removal":
            assert kinds["delete"]
        elif task.family == "nested-file-creation":
            assert kinds["create"] == 1


def _scope(count: int) -> str:
    if count == 1:
        return "one"
    if count <= 3:
        return "few"
    if count <= 8:
        return "several"
    return "many"


def _file_scope(count: int) -> str:
    if count == 1:
        return "one"
    if count <= 3:
        return "few"
    return "many"


@pytest.mark.asyncio
async def test_every_core_before_fails_and_expected_tree_passes_without_external_commands(tmp_path: Path) -> None:
    _, tasks = load_suite(SUITES / "core.yaml")

    for task in tasks:
        before = tmp_path / task.id / "before"
        expected = tmp_path / task.id / "expected"
        materialize_task(before, task)
        materialize_tree(expected, task.expected_files)

        assert not (await verify_task(before, task, run_commands=False)).success, task.id
        assert (await verify_task(expected, task, run_commands=False)).success, task.id


def test_snapshot_cases_are_compact_in_run_artifacts(tmp_path: Path) -> None:
    from benchmarks.edit_tools.artifacts import write_materialized_cases

    _, tasks = load_suite(SUITES / "core.yaml")
    write_materialized_cases(tmp_path, tasks[:1])

    case = tmp_path / "cases" / tasks[0].id
    assert (case / "task.json").is_file()
    assert not (case / "before").exists()
    assert not (case / "expected").exists()


def test_snapshot_manifests_pin_provenance_licenses_and_file_hashes() -> None:
    manifests = sorted(SNAPSHOTS.glob("*/manifest.yaml"))

    assert len(manifests) == 12
    for manifest_path in manifests:
        root = manifest_path.parent
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        assert manifest["repository"].startswith("https://github.com/")
        assert len(manifest["commit"]) == 40
        assert manifest["license"]
        assert manifest["license_files"]
        assert all((root / relative).is_file() for relative in manifest["license_files"])
        for relative, metadata in manifest["files"].items():
            data = (root / "tree" / relative).read_bytes()
            assert sha256(data).hexdigest() == metadata["sha256"]
            assert len(data) == metadata["bytes"]
            assert metadata["git_blob"]


def test_smoke_suite_selects_four_curated_cases() -> None:
    suite, tasks = load_suite(SUITES / "smoke.yaml")

    assert suite.id == "edit-smoke"
    assert [task.id for task in tasks] == [
        "python-literal",
        "python-insert-function",
        "create-changelog",
        "coordinated-version",
    ]


def test_real_corpus_and_tiny_integration_fixtures_are_separate() -> None:
    core_suite, core_tasks = load_suite(SUITES / "core.yaml")
    smoke_suite, smoke_tasks = load_suite(SUITES / "smoke.yaml")
    coder_suite, coder_tasks = load_suite(SUITES / "coder-agent.yaml")

    assert core_suite.curated_sources == ["corpora/edit-core.yaml"]
    assert smoke_suite.curated_sources == ["fixtures/smoke-tasks.yaml"]
    assert coder_suite.curated_sources == ["fixtures/smoke-tasks.yaml"]
    assert all(task.shape == "mechanical" and task.snapshot_id for task in core_tasks)
    assert all(task.shape != "mechanical" and not task.snapshot_id for task in smoke_tasks)
    assert {task.id for task in core_tasks}.isdisjoint(task.id for task in coder_tasks)
    assert min(line_count(file.text) for task in core_tasks for file in task.before_files.values()) >= 20


def test_task_rejects_workspace_escape() -> None:
    with pytest.raises(ValueError, match="project-relative"):
        TaskSpec(
            id="escape",
            prompt="escape",
            before_files={"../outside": FileContent(text="old")},
            expected_files={"safe": FileContent(text="new")},
        )
