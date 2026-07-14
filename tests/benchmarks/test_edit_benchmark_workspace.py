from pathlib import Path

import pytest

from benchmarks.edit_tools import workspace as workspace_module
from benchmarks.edit_tools.models import (
    AssertionSpec,
    CommandSpec,
    EditOperationSpec,
    EditRecipeSpec,
    FileContent,
    OracleSpec,
    TaskSpec,
)
from benchmarks.edit_tools.recipes import apply_recipe, text_sha256
from benchmarks.edit_tools.workspace import CommandResult, materialize_task, verify_task, workspace_diff


@pytest.mark.asyncio
async def test_byte_exact_oracle_preserves_crlf_and_detects_collateral(tmp_path: Path) -> None:
    task = TaskSpec(
        id="crlf",
        prompt="edit",
        before_files={"a.txt": FileContent(text="old\r\n")},
        expected_files={"a.txt": FileContent(text="new\r\n")},
    )
    materialize_task(tmp_path, task)
    (tmp_path / "a.txt").write_bytes(b"new\r\n")
    assert (await verify_task(tmp_path, task)).success

    (tmp_path / "extra.txt").write_text("extra")
    result = await verify_task(tmp_path, task)
    assert not result.success
    assert result.unexpected_paths == ["extra.txt"]
    assert "extra.txt" in workspace_diff(task, tmp_path)


@pytest.mark.asyncio
async def test_verifier_uses_argv_without_shell_and_records_failure(tmp_path: Path) -> None:
    task = TaskSpec(
        id="command",
        prompt="edit",
        before_files={"a.txt": FileContent(text="old\n")},
        expected_files={"a.txt": FileContent(text="new\n")},
        oracle=OracleSpec(commands=[CommandSpec(argv=["sh", "-c", "test -f a.txt"])]),
    )
    materialize_task(tmp_path, task)
    (tmp_path / "a.txt").write_text("new\n")

    result = await verify_task(tmp_path, task)

    assert result.success
    assert result.command_results[0].exit_code == 0


@pytest.mark.asyncio
async def test_semantic_success_is_separate_from_exact_fidelity(tmp_path: Path) -> None:
    task = TaskSpec(
        id="semantic",
        prompt="add function",
        before_files={"math.py": FileContent(text="def double(value):\n    return value * 2\n")},
        expected_files={
            "math.py": FileContent(
                text="def is_even(value):\n    return value % 2 == 0\n\ndef double(value):\n    return value * 2\n"
            )
        },
        oracle=OracleSpec(
            exact_tree=False,
            functional_assertions=[AssertionSpec(kind="contains", path="math.py", value="return value % 2 == 0")],
            instruction_assertions=[AssertionSpec(kind="contains", path="math.py", value="return value * 2")],
        ),
    )
    materialize_task(tmp_path, task)
    (tmp_path / "math.py").write_text(
        "def is_even(value):\n    return value % 2 == 0\n\n\ndef double(value):\n    return value * 2\n"
    )

    result = await verify_task(tmp_path, task)

    assert result.success
    assert result.functional_success
    assert result.instruction_success
    assert result.collateral_success
    assert not result.exact_match


@pytest.mark.asyncio
async def test_explicit_line_ending_assertion_remains_primary(tmp_path: Path) -> None:
    task = TaskSpec(
        id="line-endings",
        prompt="preserve CRLF",
        before_files={"settings.ini": FileContent(text="enabled=false\r\n")},
        expected_files={"settings.ini": FileContent(text="enabled=true\r\n")},
        oracle=OracleSpec(
            exact_tree=False,
            functional_assertions=[AssertionSpec(kind="contains", path="settings.ini", value="enabled=true")],
            instruction_assertions=[AssertionSpec(kind="line_endings", path="settings.ini", value="crlf")],
        ),
    )
    materialize_task(tmp_path, task)
    (tmp_path / "settings.ini").write_bytes(b"enabled=true\n")

    result = await verify_task(tmp_path, task)

    assert result.functional_success
    assert not result.instruction_success
    assert not result.success


@pytest.mark.asyncio
async def test_operation_diagnostics_measure_partial_exact_recipe_completion(tmp_path: Path) -> None:
    before = {"a.txt": FileContent(text="alpha\nbeta\ngamma\n")}
    recipe = EditRecipeSpec(
        operations=[
            EditOperationSpec(
                id="replace-alpha",
                kind="replace",
                path="a.txt",
                start_line=1,
                end_line=1,
                before_sha256=text_sha256("alpha\n"),
                new_text="ALPHA\n",
            ),
            EditOperationSpec(
                id="remove-gamma",
                kind="delete",
                path="a.txt",
                start_line=3,
                end_line=3,
                before_sha256=text_sha256("gamma\n"),
            ),
        ]
    )
    task = TaskSpec(
        id="partial-recipe",
        prompt="apply exact edits",
        before_files=before,
        expected_files=apply_recipe(before, recipe),
        recipe=recipe,
    )
    materialize_task(tmp_path, task)
    (tmp_path / "a.txt").write_text("ALPHA\nbeta\ngamma\n")

    result = await verify_task(tmp_path, task)

    assert not result.success
    assert result.completed_operations == 1
    assert result.total_operations == 2
    assert result.operation_success_rate == 0.5
    assert [(item.id, item.success) for item in result.operation_results] == [
        ("replace-alpha", True),
        ("remove-gamma", False),
    ]


@pytest.mark.asyncio
async def test_container_verifier_files_are_not_materialized_in_model_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    async def fake_container(root: Path, spec: CommandSpec, verifier_files: dict[str, FileContent]) -> CommandResult:
        observed["workspace_has_verifier"] = (root / "hidden_test.py").exists()
        observed["verifier_files"] = verifier_files
        return CommandResult(argv=spec.argv, cwd=spec.cwd, exit_code=0)

    monkeypatch.setattr(workspace_module, "_run_container_command", fake_container)
    task = TaskSpec(
        id="hidden-verifier",
        prompt="edit",
        before_files={"a.txt": FileContent(text="old\n")},
        expected_files={"a.txt": FileContent(text="new\n")},
        verifier_files={"hidden_test.py": FileContent(text="assert True\n")},
        oracle=OracleSpec(commands=[CommandSpec(argv=["python", "/verifier/hidden_test.py"], runtime="container")]),
    )
    materialize_task(tmp_path, task)
    (tmp_path / "a.txt").write_text("new\n")

    assert (await verify_task(tmp_path, task)).success
    assert observed["workspace_has_verifier"] is False
    assert observed["verifier_files"] == {"hidden_test.py": FileContent(text="assert True\n")}
