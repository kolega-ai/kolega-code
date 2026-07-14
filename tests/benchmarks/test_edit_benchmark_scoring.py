from pathlib import Path

from benchmarks.edit_tools.models import FileContent, TaskSpec, ToolAttempt
from benchmarks.edit_tools.scoring import attempt_file_paths, score_first_attempts_by_file


def task() -> TaskSpec:
    return TaskSpec(
        id="two-files",
        prompt="Update both files.",
        before_files={
            "a.py": FileContent(text="old a\n"),
            "nested/b.py": FileContent(text="old b\n"),
        },
        expected_files={
            "a.py": FileContent(text="new a\n"),
            "nested/b.py": FileContent(text="new b\n"),
        },
    )


def attempt(path: str, *, apply_ok: bool) -> ToolAttempt:
    return ToolAttempt(
        iteration=1,
        name="edit",
        input_kind="json",
        raw_input={"path": path},
        parse_ok=True,
        apply_ok=apply_ok,
        is_error=not apply_ok,
    )


def test_first_failure_for_a_file_is_not_erased_by_recovery() -> None:
    attempts = [
        attempt("a.py", apply_ok=False),
        attempt("a.py", apply_ok=True),
        attempt("nested/b.py", apply_ok=True),
    ]

    assert score_first_attempts_by_file(task(), attempts, {"edit"}) == (1, 2)


def test_target_file_without_an_edit_call_counts_as_a_failure() -> None:
    assert score_first_attempts_by_file(task(), [attempt("a.py", apply_ok=True)], {"edit"}) == (1, 2)


def test_apply_patch_call_attributes_outcome_to_every_file() -> None:
    patch = """*** Begin Patch
*** Update File: a.py
@@
-old a
+new a
*** Update File: nested/b.py
@@
-old b
+new b
*** End Patch"""
    patch_attempt = ToolAttempt(
        iteration=1,
        name="apply_patch",
        input_kind="freeform",
        raw_input=patch,
        apply_ok=True,
    )

    assert attempt_file_paths(patch_attempt) == {"a.py", "nested/b.py"}
    assert score_first_attempts_by_file(task(), [patch_attempt], {"apply_patch"}) == (2, 2)


def test_rename_call_targets_source_and_destination() -> None:
    rename_attempt = ToolAttempt(
        iteration=1,
        name="edit",
        input_kind="json",
        raw_input={"path": "old.py", "rename": "new.py", "edits": []},
        apply_ok=True,
    )

    assert attempt_file_paths(rename_attempt) == {"old.py", "new.py"}


def test_absolute_workspace_path_maps_to_target_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    assert score_first_attempts_by_file(
        task(),
        [attempt(str(workspace / "a.py"), apply_ok=True)],
        {"edit"},
        workspace=workspace,
    ) == (1, 2)
