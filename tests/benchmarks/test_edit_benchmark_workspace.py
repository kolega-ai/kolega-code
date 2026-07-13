from pathlib import Path

import pytest

from benchmarks.edit_tools.models import CommandSpec, FileContent, OracleSpec, TaskSpec
from benchmarks.edit_tools.workspace import materialize_task, verify_task, workspace_diff


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
