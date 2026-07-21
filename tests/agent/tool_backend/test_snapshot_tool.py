from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, Mock

import pytest

from kolega_code.agent.tool_backend.edit_tool import EditTool
from kolega_code.agent.tool_backend.lsp_tool import LspEditTool
from kolega_code.agent.tool_backend.snapshot_tool import SnapshotTool
from kolega_code.services.file_system import LocalFileSystem
from kolega_code.services.snapshots import SnapshotService
from kolega_code.tools import ToolError


def _block(search: str, replace: str) -> str:
    return f"<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE"


def _caller() -> Mock:
    caller = Mock()
    caller.agent_name = "test_agent"
    caller.sub_agent = False
    caller.current_tool_execution_id = "tool-call-1"
    return caller


def _snapshot_service(project_path, filesystem) -> SnapshotService:
    return SnapshotService(
        project_path,
        "workspace",
        "thread",
        f"session-{uuid.uuid4().hex}",
        filesystem,
        root=project_path.parent / "state",
    )


def _manager_with_rename_edit(workspace_edit) -> Mock:
    manager = Mock()
    manager.enabled = True
    manager._initialized = True
    manager.server_for_path.return_value = "pyright"
    manager._resolve_position.return_value = (0, 0)
    manager.get_rename = AsyncMock(return_value=workspace_edit)
    manager._config = Mock(auto_diagnostics_on_edit=False)
    return manager


@pytest.mark.asyncio
async def test_edit_snapshot_can_be_restored(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    path = project / "a.txt"
    path.write_text("one\ntwo\n", encoding="utf-8")
    filesystem = LocalFileSystem(project)
    service = _snapshot_service(project, filesystem)
    caller = _caller()
    connection = AsyncMock()
    edit_tool = EditTool(
        project,
        "workspace",
        "thread",
        connection,
        Mock(),
        caller,
        filesystem,
        snapshot_service=service,
    )
    snapshot_tool = SnapshotTool(
        project,
        "workspace",
        "thread",
        connection,
        Mock(),
        caller,
        filesystem,
        snapshot_service=service,
    )

    result = await edit_tool.edit("a.txt", _block("two", "three"))

    assert result == "Edited a.txt"
    assert path.read_text(encoding="utf-8") == "one\nthree\n"
    records = service.list_snapshots()
    assert len(records) == 1

    restore_result = await snapshot_tool.snapshot(action="restore", snapshot_id="latest")

    assert "Restored snapshot" in restore_result
    assert path.read_text(encoding="utf-8") == "one\ntwo\n"


@pytest.mark.asyncio
async def test_out_of_project_write_skips_snapshot_and_succeeds(tmp_path):
    project = tmp_path / "project"
    outside_dir = tmp_path / "outside"
    project.mkdir()
    outside_dir.mkdir()
    outside = outside_dir / "outside.txt"
    filesystem = LocalFileSystem(project)
    service = _snapshot_service(project, filesystem)
    tool = EditTool(
        project,
        "workspace",
        "thread",
        AsyncMock(),
        Mock(),
        _caller(),
        filesystem,
        snapshot_service=service,
    )

    result = await tool.write(str(outside), "outside\n")

    assert result == f"Wrote {outside}"
    assert outside.read_text(encoding="utf-8") == "outside\n"
    assert service.list_snapshots() == []


@pytest.mark.asyncio
async def test_out_of_project_edit_skips_snapshot_and_succeeds(tmp_path):
    project = tmp_path / "project"
    outside_dir = tmp_path / "outside"
    project.mkdir()
    outside_dir.mkdir()
    outside = outside_dir / "outside.txt"
    outside.write_text("alpha\n", encoding="utf-8")
    filesystem = LocalFileSystem(project)
    service = _snapshot_service(project, filesystem)
    tool = EditTool(
        project,
        "workspace",
        "thread",
        AsyncMock(),
        Mock(),
        _caller(),
        filesystem,
        snapshot_service=service,
    )

    result = await tool.edit(str(outside), _block("alpha", "beta"))

    assert result == f"Edited {outside}"
    assert outside.read_text(encoding="utf-8") == "beta\n"
    assert service.list_snapshots() == []


@pytest.mark.asyncio
async def test_external_lsp_apply_succeeds_without_snapshot(tmp_path):
    project = tmp_path / "project"
    outside_dir = tmp_path / "outside"
    project.mkdir()
    outside_dir.mkdir()
    outside = outside_dir / "external.py"
    outside.write_text("old = 1\n", encoding="utf-8")
    filesystem = LocalFileSystem(project)
    service = _snapshot_service(project, filesystem)
    manager = _manager_with_rename_edit(
        {
            "changes": {
                outside.as_uri(): [
                    {
                        "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 3}},
                        "newText": "new",
                    }
                ]
            }
        }
    )
    tool = LspEditTool(
        project,
        "workspace",
        "thread",
        AsyncMock(),
        Mock(),
        _caller(),
        filesystem,
        lsp_manager=manager,
        snapshot_service=service,
    )

    result = await tool.lsp_edit(
        operation="rename",
        path=str(outside),
        line=1,
        symbol="old",
        new_name="new",
        apply=True,
    )

    assert result.startswith("Applied LSP edit `rename`.")
    assert outside.read_text(encoding="utf-8") == "new = 1\n"
    assert service.list_snapshots() == []


@pytest.mark.asyncio
async def test_mixed_lsp_apply_changes_every_file_without_snapshot(tmp_path):
    project = tmp_path / "project"
    outside_dir = tmp_path / "outside"
    project.mkdir()
    outside_dir.mkdir()
    internal = project / "internal.py"
    external = outside_dir / "external.py"
    internal.write_text("old = 1\n", encoding="utf-8")
    external.write_text("old = 2\n", encoding="utf-8")
    replacement = [
        {
            "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 3}},
            "newText": "new",
        }
    ]
    filesystem = LocalFileSystem(project)
    service = _snapshot_service(project, filesystem)
    manager = _manager_with_rename_edit(
        {
            "changes": {
                internal.as_uri(): replacement,
                external.as_uri(): replacement,
            }
        }
    )
    tool = LspEditTool(
        project,
        "workspace",
        "thread",
        AsyncMock(),
        Mock(),
        _caller(),
        filesystem,
        lsp_manager=manager,
        snapshot_service=service,
    )

    result = await tool.lsp_edit(
        operation="rename",
        path="internal.py",
        line=1,
        symbol="old",
        new_name="new",
        apply=True,
    )

    assert result.startswith("Applied LSP edit `rename`.")
    assert internal.read_text(encoding="utf-8") == "new = 1\n"
    assert external.read_text(encoding="utf-8") == "new = 2\n"
    assert service.list_snapshots() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mixed", [False, True], ids=["external", "mixed"])
async def test_external_lsp_preview_writes_nothing_and_creates_no_pending_action(tmp_path, mixed):
    project = tmp_path / "project"
    outside_dir = tmp_path / "outside"
    project.mkdir()
    outside_dir.mkdir()
    internal = project / "internal.py"
    external = outside_dir / "external.py"
    internal.write_text("old = 1\n", encoding="utf-8")
    external.write_text("old = 2\n", encoding="utf-8")
    replacement = [
        {
            "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 3}},
            "newText": "new",
        }
    ]
    changes = {external.as_uri(): replacement}
    if mixed:
        changes[internal.as_uri()] = replacement
    filesystem = LocalFileSystem(project)
    service = _snapshot_service(project, filesystem)
    manager = _manager_with_rename_edit({"changes": changes})
    tool = LspEditTool(
        project,
        "workspace",
        "thread",
        AsyncMock(),
        Mock(),
        _caller(),
        filesystem,
        lsp_manager=manager,
        snapshot_service=service,
    )

    result = await tool.lsp_edit(
        operation="rename",
        path="internal.py" if mixed else str(external),
        line=1,
        symbol="old",
        new_name="new",
        apply=False,
    )

    assert result.startswith("Preview LSP edit `rename`.")
    assert "No pending action was created; rerun with `apply=True`" in result
    assert internal.read_text(encoding="utf-8") == "old = 1\n"
    assert external.read_text(encoding="utf-8") == "old = 2\n"
    assert service.list_pending_actions() == []
    assert service.list_snapshots() == []


@pytest.mark.asyncio
async def test_mixed_apply_patch_bypasses_snapshot_and_external_is_never_restored(tmp_path):
    project = tmp_path / "project"
    outside_dir = tmp_path / "outside"
    project.mkdir()
    outside_dir.mkdir()
    internal = project / "internal.txt"
    external = outside_dir / "external.txt"
    internal.write_text("one\n", encoding="utf-8")
    external.write_text("alpha\n", encoding="utf-8")
    filesystem = LocalFileSystem(project)
    service = _snapshot_service(project, filesystem)
    caller = _caller()
    connection = AsyncMock()
    edit_tool = EditTool(
        project,
        "workspace",
        "thread",
        connection,
        Mock(),
        caller,
        filesystem,
        snapshot_service=service,
    )
    snapshot_tool = SnapshotTool(
        project,
        "workspace",
        "thread",
        connection,
        Mock(),
        caller,
        filesystem,
        snapshot_service=service,
    )
    patch = (
        "*** Begin Patch\n"
        "*** Update File: internal.txt\n"
        "@@\n"
        "-one\n"
        "+two\n"
        f"*** Update File: {external}\n"
        "@@\n"
        "-alpha\n"
        "+beta\n"
        "*** End Patch\n"
    )

    await edit_tool.apply_patch(patch)

    assert internal.read_text(encoding="utf-8") == "two\n"
    assert external.read_text(encoding="utf-8") == "beta\n"
    assert service.list_snapshots() == []

    await edit_tool.edit("internal.txt", _block("two", "three"))
    record = service.latest_snapshot()
    assert record is not None
    external_text = str(external)
    assert external_text not in record.touched_paths
    assert external_text not in record.before
    assert external_text not in record.after

    await snapshot_tool.snapshot(action="restore", snapshot_id=record.snapshot_id)

    assert internal.read_text(encoding="utf-8") == "two\n"
    assert external.read_text(encoding="utf-8") == "beta\n"


@pytest.mark.asyncio
async def test_manual_checkpoint_restores_after_change_without_force(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    path = project / "a.txt"
    path.write_text("checkpoint\n", encoding="utf-8")
    filesystem = LocalFileSystem(project)
    service = _snapshot_service(project, filesystem)
    tool = SnapshotTool(
        project,
        "workspace",
        "thread",
        AsyncMock(),
        Mock(),
        _caller(),
        filesystem,
        snapshot_service=service,
    )
    await tool.snapshot(action="create", paths=["a.txt"])
    record = service.latest_snapshot()
    assert record is not None
    path.write_text("changed\n", encoding="utf-8")

    result = await tool.snapshot(action="restore", snapshot_id=record.snapshot_id)

    assert "Restored snapshot" in result
    assert path.read_text(encoding="utf-8") == "checkpoint\n"


@pytest.mark.asyncio
async def test_snapshot_expected_failures_raise_tool_error(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    filesystem = LocalFileSystem(project)
    service = _snapshot_service(project, filesystem)
    tool = SnapshotTool(
        project,
        "workspace",
        "thread",
        AsyncMock(),
        Mock(),
        _caller(),
        filesystem,
        snapshot_service=service,
    )

    with pytest.raises(ToolError, match="Snapshot not found"):
        await tool.snapshot(action="show", snapshot_id="snap_missing")


@pytest.mark.asyncio
async def test_resolve_expected_failures_raise_tool_error(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    path = project / "a.txt"
    path.write_text("one\n", encoding="utf-8")
    filesystem = LocalFileSystem(project)
    service = _snapshot_service(project, filesystem)
    tool = SnapshotTool(
        project,
        "workspace",
        "thread",
        AsyncMock(),
        Mock(),
        _caller(),
        filesystem,
        snapshot_service=service,
    )
    action = service.create_pending_workspace_edit(
        tool_name="lsp_edit",
        tool_call_id="tool-call-1",
        operation="format_document",
        workspace_edit={"changes": {}},
        touched_paths=["a.txt"],
    )

    with pytest.raises(ToolError, match="decision must be apply or discard"):
        await tool.resolve(action.action_id, decision="wait")


@pytest.mark.asyncio
async def test_resolve_applies_pending_workspace_edit(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    path = project / "a.txt"
    path.write_text("one\n", encoding="utf-8")
    filesystem = LocalFileSystem(project)
    service = _snapshot_service(project, filesystem)
    caller = _caller()
    tool = SnapshotTool(
        project,
        "workspace",
        "thread",
        AsyncMock(),
        Mock(),
        caller,
        filesystem,
        snapshot_service=service,
    )
    workspace_edit = {
        "changes": {
            path.resolve().as_uri(): [
                {
                    "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 3}},
                    "newText": "two",
                }
            ]
        }
    }
    action = service.create_pending_workspace_edit(
        tool_name="lsp_edit",
        tool_call_id="tool-call-1",
        operation="format_document",
        workspace_edit=workspace_edit,
        touched_paths=["a.txt"],
        summaries=["updated a.txt"],
    )

    result = await tool.resolve(action.action_id, decision="apply")

    assert f"Applied pending action `{action.action_id}`" in result
    assert path.read_text(encoding="utf-8") == "two\n"
    assert service.load_pending_action(action.action_id).status == "applied"
    assert len(service.list_snapshots()) == 1


@pytest.mark.asyncio
async def test_resolve_discards_pending_workspace_edit(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    path = project / "a.txt"
    path.write_text("one\n", encoding="utf-8")
    filesystem = LocalFileSystem(project)
    service = _snapshot_service(project, filesystem)
    caller = _caller()
    tool = SnapshotTool(
        project,
        "workspace",
        "thread",
        AsyncMock(),
        Mock(),
        caller,
        filesystem,
        snapshot_service=service,
    )
    action = service.create_pending_workspace_edit(
        tool_name="lsp_edit",
        tool_call_id="tool-call-1",
        operation="format_document",
        workspace_edit={"changes": {}},
        touched_paths=["a.txt"],
    )

    result = await tool.resolve(action.action_id, decision="discard")

    assert f"Discarded pending action `{action.action_id}`" in result
    assert path.read_text(encoding="utf-8") == "one\n"
    assert service.load_pending_action(action.action_id).status == "discarded"


@pytest.mark.asyncio
async def test_internal_lsp_preview_creates_and_resolves_pending_action(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    path = project / "a.py"
    path.write_text("old = 1\n", encoding="utf-8")
    filesystem = LocalFileSystem(project)
    service = _snapshot_service(project, filesystem)
    manager = Mock()
    manager.enabled = True
    manager._initialized = True
    manager.server_for_path.return_value = "pyright"
    manager._resolve_position.return_value = (0, 0)
    manager.get_rename = AsyncMock(
        return_value={
            "changes": {
                path.resolve().as_uri(): [
                    {
                        "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 3}},
                        "newText": "new",
                    }
                ]
            }
        }
    )
    tool = LspEditTool(
        project,
        "workspace",
        "thread",
        AsyncMock(),
        Mock(),
        _caller(),
        filesystem,
        lsp_manager=manager,
        snapshot_service=service,
    )

    result = await tool.lsp_edit(
        operation="rename",
        path="a.py",
        line=1,
        symbol="old",
        new_name="new",
        apply=False,
    )

    assert result.startswith("Preview LSP edit `rename`.")
    assert "Pending action:" in result
    assert path.read_text(encoding="utf-8") == "old = 1\n"
    pending = service.list_pending_actions()
    assert len(pending) == 1
    assert pending[0].operation == "rename"

    snapshot_tool = SnapshotTool(
        project,
        "workspace",
        "thread",
        AsyncMock(),
        Mock(),
        _caller(),
        filesystem,
        snapshot_service=service,
    )
    resolve_result = await snapshot_tool.resolve(pending[0].action_id, decision="apply")

    assert f"Applied pending action `{pending[0].action_id}`" in resolve_result
    assert path.read_text(encoding="utf-8") == "new = 1\n"
    assert service.load_pending_action(pending[0].action_id).status == "applied"
    assert len(service.list_snapshots()) == 1
