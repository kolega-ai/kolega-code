from pathlib import Path
from unittest.mock import AsyncMock, Mock
import uuid

import pytest

from kolega_code.agent.tool_backend.codex_patch import CodexPatchError, parse_codex_patch
from kolega_code.agent.tool_backend.edit_tool import EditTool
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider
from kolega_code.services.snapshots import SnapshotService


@pytest.fixture
def caller() -> Mock:
    value = Mock()
    value.agent_name = "test-agent"
    value.agent_mode = AgentMode.CODE.value
    value.current_tool_execution_id = "call-1"
    value.sub_agent = False
    return value


@pytest.fixture
def config() -> AgentConfig:
    model = ModelConfig(provider=ModelProvider.ANTHROPIC, model="test-model")
    return AgentConfig(
        anthropic_api_key="test",
        long_context_config=model,
        fast_config=model,
        thinking_config=model,
    )


@pytest.fixture
def edit_tool(tmp_path: Path, caller: Mock, config: AgentConfig) -> EditTool:
    return EditTool(tmp_path, "workspace", str(uuid.uuid4()), AsyncMock(), config, caller)


def make_edit_tool(project_path: Path, caller: Mock, config: AgentConfig) -> EditTool:
    return EditTool(project_path, "workspace", str(uuid.uuid4()), AsyncMock(), config, caller)


def patch(*lines: str) -> str:
    return "\n".join(("*** Begin Patch", *lines, "*** End Patch")) + "\n"


def test_parser_rejects_missing_markers() -> None:
    with pytest.raises(CodexPatchError, match="start"):
        parse_codex_patch("*** Add File: a.txt\n+x\n")


@pytest.mark.asyncio
async def test_apply_patch_add_update_delete_and_missing_parents(edit_tool: EditTool, tmp_path: Path) -> None:
    (tmp_path / "update.txt").write_text("alpha\nbeta\ngamma\n")
    (tmp_path / "delete.txt").write_text("gone\n")

    result = await edit_tool.apply_patch(
        patch(
            "*** Add File: nested/new.txt",
            "+hello",
            "+world",
            "*** Update File: update.txt",
            "@@",
            " alpha",
            "-beta",
            "+BETA",
            " gamma",
            "*** Delete File: delete.txt",
        )
    )

    assert (tmp_path / "nested/new.txt").read_text() == "hello\nworld\n"
    assert (tmp_path / "update.txt").read_text() == "alpha\nBETA\ngamma\n"
    assert not (tmp_path / "delete.txt").exists()
    assert result.splitlines()[:4] == [
        "Success. Updated the following files:",
        "A nested/new.txt",
        "M update.txt",
        "D delete.txt",
    ]


@pytest.mark.asyncio
async def test_apply_patch_move_overwrites_destination(edit_tool: EditTool, tmp_path: Path) -> None:
    (tmp_path / "old.txt").write_text("old\n")
    (tmp_path / "new.txt").write_text("destination\n")

    result = await edit_tool.apply_patch(
        patch(
            "*** Update File: old.txt",
            "*** Move to: new.txt",
            "@@",
            "-old",
            "+moved",
        )
    )

    assert not (tmp_path / "old.txt").exists()
    assert (tmp_path / "new.txt").read_text() == "moved\n"
    assert "M old.txt -> new.txt" in result


@pytest.mark.asyncio
async def test_apply_patch_add_overwrites_and_preserves_crlf(edit_tool: EditTool, tmp_path: Path) -> None:
    target = tmp_path / "windows.txt"
    target.write_bytes(b"old\r\n")

    await edit_tool.apply_patch(patch("*** Add File: windows.txt", "+new", "+line"))

    assert target.read_bytes() == b"new\r\nline\r\n"


@pytest.mark.asyncio
async def test_apply_patch_multiple_chunks_context_eof_and_fuzzy_match(edit_tool: EditTool, tmp_path: Path) -> None:
    target = tmp_path / "code.py"
    target.write_text("def one():\n    return ‘one’\n\ndef two():\n    return 2   \n")

    await edit_tool.apply_patch(
        patch(
            "*** Update File: code.py",
            "@@ def one():",
            "-    return 'one'",
            "+    return 'ONE'",
            "@@ def two():",
            "-    return 2",
            "+    return 22",
            "*** End of File",
        )
    )

    assert target.read_text() == "def one():\n    return 'ONE'\n\ndef two():\n    return 22\n"


@pytest.mark.asyncio
async def test_apply_patch_validation_failure_writes_nothing(edit_tool: EditTool, tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("one\n")
    second.write_text("two\n")

    with pytest.raises(CodexPatchError, match="does not match"):
        await edit_tool.apply_patch(
            patch(
                "*** Update File: first.txt",
                "@@",
                "-one",
                "+ONE",
                "*** Update File: second.txt",
                "@@",
                "-missing",
                "+TWO",
            )
        )

    assert first.read_text() == "one\n"
    assert second.read_text() == "two\n"


@pytest.mark.asyncio
async def test_apply_patch_supports_absolute_parent_traversal_and_external_moves(
    tmp_path: Path, caller: Mock, config: AgentConfig
) -> None:
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    (project / "entry").mkdir()
    absolute_update = outside / "absolute-update.txt"
    absolute_delete = outside / "absolute-delete.txt"
    absolute_destination = outside / "absolute-moved.txt"
    absolute_update.write_text("before\n")
    absolute_delete.write_text("delete me\n")
    (outside / "move-source.txt").write_text("move me\n")
    tool = make_edit_tool(project, caller, config)

    result = await tool.apply_patch(
        patch(
            "*** Add File: ../outside/created.txt",
            "+created",
            "*** Add File: entry/../../outside/repeated-parent.txt",
            "+repeated",
            f"*** Update File: {absolute_update}",
            "@@",
            "-before",
            "+after",
            f"*** Delete File: {absolute_delete}",
            "*** Update File: ../outside/move-source.txt",
            f"*** Move to: {absolute_destination}",
            "@@",
            "-move me",
            "+moved",
        )
    )

    assert (outside / "created.txt").read_text() == "created\n"
    assert (outside / "repeated-parent.txt").read_text() == "repeated\n"
    assert absolute_update.read_text() == "after\n"
    assert not absolute_delete.exists()
    assert not (outside / "move-source.txt").exists()
    assert absolute_destination.read_text() == "moved\n"
    assert f"M ../outside/move-source.txt -> {absolute_destination}" in result


@pytest.mark.asyncio
async def test_apply_patch_rejects_directories(edit_tool: EditTool, tmp_path: Path) -> None:
    (tmp_path / "folder").mkdir()
    with pytest.raises(IsADirectoryError):
        await edit_tool.apply_patch(patch("*** Add File: folder", "+no"))


@pytest.mark.asyncio
async def test_apply_patch_checks_external_vibe_protected_path_before_mixed_mutation(
    tmp_path: Path, caller: Mock, config: AgentConfig
) -> None:
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    edit_tool = make_edit_tool(project, caller, config)
    caller.agent_mode = AgentMode.VIBE.value
    caller.protected_files = {"package.json"}

    result = await edit_tool.apply_patch(
        patch("*** Add File: safe.txt", "+safe", "*** Add File: ../outside/package.json", "+{}")
    )

    assert "not allowed" in result
    assert not (project / "safe.txt").exists()
    assert not (outside / "package.json").exists()


@pytest.mark.asyncio
async def test_apply_patch_snapshot_can_restore_multi_file_change(
    tmp_path: Path, caller: Mock, config: AgentConfig
) -> None:
    original = tmp_path / "original.txt"
    original.write_text("before\n")
    filesystem_tool = EditTool(tmp_path, "workspace", "thread", AsyncMock(), config, caller)
    snapshots = SnapshotService(
        tmp_path,
        "workspace",
        "thread",
        "session",
        filesystem_tool.filesystem,
        root=tmp_path / "state",
    )
    filesystem_tool._snapshot_service = snapshots

    await filesystem_tool.apply_patch(
        patch(
            "*** Update File: original.txt",
            "@@",
            "-before",
            "+after",
            "*** Add File: created.txt",
            "+created",
        )
    )
    record = snapshots.latest_snapshot()
    assert record is not None

    snapshots.restore_snapshot(record.snapshot_id)

    assert original.read_text() == "before\n"
    assert not (tmp_path / "created.txt").exists()
