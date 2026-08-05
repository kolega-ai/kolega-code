from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.agent.tool_backend.edit_tool import EditTool
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider
from kolega_code.services.snapshots import SnapshotService


@pytest.fixture
def edit_tool(tmp_path: Path) -> EditTool:
    model = ModelConfig(provider=ModelProvider.ANTHROPIC, model="test-model")
    config = AgentConfig(
        anthropic_api_key="test",
        long_context_config=model,
        fast_config=model,
    )
    caller = Mock()
    caller.agent_name = "test-agent"
    caller.agent_mode = AgentMode.CODE.value
    caller.current_tool_execution_id = "call-1"
    caller.sub_agent = False
    return EditTool(tmp_path, "workspace", "thread", AsyncMock(), config, caller)


def nested_edit_tool(tmp_path: Path, template: EditTool) -> tuple[EditTool, Path]:
    project = tmp_path / "project"
    project.mkdir()
    return (
        EditTool(project, "workspace", "thread", AsyncMock(), template.config, template.caller),
        project,
    )


@pytest.mark.asyncio
async def test_claude_edit_requires_read_and_replaces_unique_match(edit_tool: EditTool, tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("before\nmiddle\nafter\n")

    with pytest.raises(ValueError, match="has not been read"):
        await edit_tool.claude_edit("a.txt", "middle", "changed")

    edit_tool.observe_read("a.txt")
    result = await edit_tool.claude_edit("a.txt", "middle", "changed")

    assert result == "Edited a.txt"
    assert target.read_text() == "before\nchanged\nafter\n"


@pytest.mark.asyncio
async def test_claude_edit_rejects_ambiguous_match_and_supports_replace_all(
    edit_tool: EditTool, tmp_path: Path
) -> None:
    target = tmp_path / "a.txt"
    target.write_text("old old old\n")
    edit_tool.observe_read("a.txt")

    with pytest.raises(ValueError, match="Found 3 matches"):
        await edit_tool.claude_edit("a.txt", "old", "new")

    await edit_tool.claude_edit("a.txt", "old", "new", replace_all=True)
    assert target.read_text() == "new new new\n"


@pytest.mark.asyncio
async def test_claude_edit_can_create_but_empty_search_cannot_overwrite(edit_tool: EditTool, tmp_path: Path) -> None:
    await edit_tool.claude_edit("nested/new.txt", "", "created\n")
    assert (tmp_path / "nested/new.txt").read_text() == "created\n"

    edit_tool.observe_read("nested/new.txt")
    with pytest.raises(ValueError, match="cannot be empty"):
        await edit_tool.claude_edit("nested/new.txt", "", "overwritten\n")


@pytest.mark.asyncio
async def test_claude_edit_preserves_bom_and_crlf(edit_tool: EditTool, tmp_path: Path) -> None:
    target = tmp_path / "windows.txt"
    target.write_bytes("\ufeffone\r\ntwo\r\n".encode())
    edit_tool.observe_read("windows.txt")

    await edit_tool.claude_edit("windows.txt", "one\ntwo", "ONE\nTWO")
    await edit_tool.claude_edit("windows.txt", "ONE", "Uno")

    assert target.read_bytes() == "\ufeffUno\r\nTWO\r\n".encode()


@pytest.mark.asyncio
async def test_claude_write_requires_read_for_overwrite_and_detects_stale_file(
    edit_tool: EditTool, tmp_path: Path
) -> None:
    target = tmp_path / "a.txt"
    target.write_text("before\n")

    with pytest.raises(ValueError, match="has not been read"):
        await edit_tool.claude_write("a.txt", "after\n")

    edit_tool.observe_read("a.txt")
    target.write_text("external\n")
    with pytest.raises(ValueError, match="has changed"):
        await edit_tool.claude_write("a.txt", "after\n")

    edit_tool.observe_read("a.txt")
    await edit_tool.claude_write("a.txt", "after\n")
    assert target.read_text() == "after\n"


@pytest.mark.asyncio
async def test_claude_write_preserves_existing_bom_and_line_endings(edit_tool: EditTool, tmp_path: Path) -> None:
    target = tmp_path / "windows.txt"
    target.write_bytes("\ufeffbefore\r\n".encode())
    edit_tool.observe_read("windows.txt")

    await edit_tool.claude_write("windows.txt", "after\nline\n")

    assert target.read_bytes() == "\ufeffafter\r\nline\r\n".encode()


@pytest.mark.asyncio
async def test_claude_external_absolute_and_parent_paths_support_full_write_and_update(
    edit_tool: EditTool, tmp_path: Path
) -> None:
    tool, _project = nested_edit_tool(tmp_path, edit_tool)
    outside = tmp_path / "outside"
    outside.mkdir()
    absolute = outside / "absolute.txt"
    absolute.write_text("before\n")

    tool.observe_read(str(absolute))
    assert await tool.claude_write(str(absolute), "full write\n") == f"Wrote {absolute}"
    tool.observe_read(str(absolute))
    assert await tool.claude_edit(str(absolute), "full write", "updated") == f"Edited {absolute}"

    relative = "../outside/relative.txt"
    assert await tool.claude_write(relative, "created\n") == f"Wrote {relative}"
    tool.observe_read(relative)
    assert await tool.claude_edit(relative, "created", "changed") == f"Edited {relative}"

    assert absolute.read_text() == "updated\n"
    assert (outside / "relative.txt").read_text() == "changed\n"


@pytest.mark.asyncio
async def test_claude_write_preserves_dotdot_after_symlink_component(edit_tool: EditTool, tmp_path: Path) -> None:
    tool, project = nested_edit_tool(tmp_path, edit_tool)
    symlink_target = tmp_path / "real" / "child"
    symlink_target.mkdir(parents=True)
    (project / "link").symlink_to(symlink_target, target_is_directory=True)
    raw_path = "link/../symlink-sensitive.txt"

    assert await tool.claude_write(raw_path, "through symlink\n") == f"Wrote {raw_path}"

    assert (tmp_path / "real" / "symlink-sensitive.txt").read_text() == "through symlink\n"
    assert not (project / "symlink-sensitive.txt").exists()


@pytest.mark.asyncio
async def test_claude_edit_snapshot_can_restore_change(edit_tool: EditTool, tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("before\n")
    snapshots = SnapshotService(
        tmp_path,
        "workspace",
        "thread",
        "session",
        edit_tool.filesystem,
        root=tmp_path / "state",
    )
    edit_tool._snapshot_service = snapshots
    edit_tool.observe_read("a.txt")

    await edit_tool.claude_edit("a.txt", "before", "after")
    record = snapshots.latest_snapshot()
    assert record is not None
    snapshots.restore_snapshot(record.snapshot_id)

    assert target.read_text() == "before\n"


@pytest.mark.asyncio
async def test_write_then_edit_without_read_succeeds(edit_tool: EditTool, tmp_path: Path) -> None:
    """A successful write counts as read: the guard accepts a subsequent edit."""
    target = tmp_path / "a.txt"

    result = await edit_tool.write("a.txt", "before\nmiddle\nafter\n")
    assert result == "Wrote a.txt"

    assert await edit_tool.claude_edit("a.txt", "middle", "changed") == "Edited a.txt"
    assert target.read_text() == "before\nchanged\nafter\n"


@pytest.mark.asyncio
async def test_write_overwrite_then_edit_without_read_succeeds(edit_tool: EditTool, tmp_path: Path) -> None:
    """Overwriting an existing file with write also records the new contents."""
    target = tmp_path / "a.txt"
    target.write_text("original\n")

    await edit_tool.write("a.txt", "new content\n")

    assert await edit_tool.claude_edit("a.txt", "new content", "updated") == "Edited a.txt"
    assert target.read_text() == "updated\n"


@pytest.mark.asyncio
async def test_write_then_external_modification_then_edit_is_blocked(edit_tool: EditTool, tmp_path: Path) -> None:
    """Write grants no more than a read: external changes still go stale."""
    target = tmp_path / "a.txt"
    await edit_tool.write("a.txt", "before\nmiddle\nafter\n")

    target.write_text("before\nEXTERNALLY CHANGED\nafter\n")

    with pytest.raises(ValueError, match="has changed since it was read"):
        await edit_tool.claude_edit("a.txt", "middle", "changed")


@pytest.mark.asyncio
async def test_hashline_write_then_edit_without_read_succeeds(edit_tool: EditTool, tmp_path: Path) -> None:
    """hashline_write funnels through the same write path and records too."""
    target = tmp_path / "a.txt"

    assert await edit_tool.hashline_write("a.txt", "hello\n") == "Wrote a.txt"

    assert await edit_tool.claude_edit("a.txt", "hello", "world") == "Edited a.txt"
    assert target.read_text() == "world\n"


@pytest.mark.asyncio
async def test_apply_patch_add_then_edit_without_read_succeeds(edit_tool: EditTool, tmp_path: Path) -> None:
    """apply_patch's Add File writes through the shared path and records too."""
    target = tmp_path / "a.txt"

    result = await edit_tool.apply_patch("*** Begin Patch\n*** Add File: a.txt\n+hello\n*** End Patch\n")
    assert "A a.txt" in result

    assert await edit_tool.claude_edit("a.txt", "hello", "world") == "Edited a.txt"
    assert target.read_text() == "world\n"


@pytest.mark.asyncio
async def test_search_replace_edit_then_claude_edit_without_read_succeeds(edit_tool: EditTool, tmp_path: Path) -> None:
    """Search/replace edits write through the shared path, so their result is known."""
    target = tmp_path / "a.txt"
    target.write_text("before\nmiddle\nafter\n")

    result = await edit_tool.edit("a.txt", "<<<<<<< SEARCH\nmiddle\n=======\nchanged\n>>>>>>> REPLACE")
    assert result == "Edited a.txt"

    assert await edit_tool.claude_edit("a.txt", "changed", "changed2") == "Edited a.txt"
    assert target.read_text() == "before\nchanged2\nafter\n"
