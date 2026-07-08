from __future__ import annotations

import pytest

from kolega_code.services.file_system import LocalFileSystem
from kolega_code.services.snapshots import SnapshotError, SnapshotService


@pytest.fixture
def snapshot_service(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    state = tmp_path / "state"
    return SnapshotService(
        project,
        "workspace",
        "thread",
        "session",
        LocalFileSystem(project),
        root=state,
    )


def test_record_mutation_and_restore(snapshot_service, tmp_path):
    project = tmp_path / "project"
    path = project / "a.txt"
    path.write_text("before\n", encoding="utf-8")

    result = snapshot_service.record_mutation(
        tool_name="edit",
        tool_call_id="call-1",
        reason="test edit",
        paths=["a.txt"],
        mutate=lambda: path.write_text("after\n", encoding="utf-8"),
    )

    assert result.snapshot is not None
    assert path.read_text(encoding="utf-8") == "after\n"

    restored = snapshot_service.restore_snapshot(result.snapshot.snapshot_id)

    assert restored.snapshot_id == result.snapshot.snapshot_id
    assert path.read_text(encoding="utf-8") == "before\n"


def test_restore_refuses_when_current_state_changed(snapshot_service, tmp_path):
    project = tmp_path / "project"
    path = project / "a.txt"
    path.write_text("before\n", encoding="utf-8")
    result = snapshot_service.record_mutation(
        tool_name="edit",
        tool_call_id="call-1",
        reason="test edit",
        paths=["a.txt"],
        mutate=lambda: path.write_text("after\n", encoding="utf-8"),
    )
    assert result.snapshot is not None
    path.write_text("user change\n", encoding="utf-8")

    with pytest.raises(SnapshotError) as exc_info:
        snapshot_service.restore_snapshot(result.snapshot.snapshot_id)

    assert "tracked files changed" in str(exc_info.value)
    assert path.read_text(encoding="utf-8") == "user change\n"


def test_force_restore_overwrites_changed_current_state(snapshot_service, tmp_path):
    project = tmp_path / "project"
    path = project / "a.txt"
    path.write_text("before\n", encoding="utf-8")
    result = snapshot_service.record_mutation(
        tool_name="edit",
        tool_call_id="call-1",
        reason="test edit",
        paths=["a.txt"],
        mutate=lambda: path.write_text("after\n", encoding="utf-8"),
    )
    assert result.snapshot is not None
    path.write_text("user change\n", encoding="utf-8")

    snapshot_service.restore_snapshot(result.snapshot.snapshot_id, force=True)

    assert path.read_text(encoding="utf-8") == "before\n"


def test_manual_snapshot_restores_after_change_without_force(snapshot_service, tmp_path):
    project = tmp_path / "project"
    path = project / "a.txt"
    path.write_text("checkpoint\n", encoding="utf-8")
    record = snapshot_service.create_manual_snapshot(paths=["a.txt"])
    path.write_text("changed\n", encoding="utf-8")

    snapshot_service.restore_snapshot(record.snapshot_id)

    assert path.read_text(encoding="utf-8") == "checkpoint\n"


def test_write_snapshot_restores_new_file_and_created_parent(snapshot_service, tmp_path):
    project = tmp_path / "project"
    filesystem = LocalFileSystem(project)

    result = snapshot_service.record_mutation(
        tool_name="write",
        tool_call_id="call-1",
        reason="create nested file",
        paths=["new/child.txt", "new"],
        mutate=lambda: (project / "new").mkdir() or (project / "new" / "child.txt").write_text("created\n"),
    )
    assert result.snapshot is not None
    assert (project / "new" / "child.txt").exists()

    snapshot_service.restore_snapshot(result.snapshot.snapshot_id)

    assert not filesystem.exists("new")


def test_create_and_update_pending_workspace_edit(snapshot_service, tmp_path):
    project = tmp_path / "project"
    (project / "a.txt").write_text("one\n", encoding="utf-8")
    edit = {"changes": {"file:///tmp/unused": []}}

    action = snapshot_service.create_pending_workspace_edit(
        tool_name="lsp_edit",
        tool_call_id="call-1",
        operation="format_document",
        workspace_edit=edit,
        touched_paths=["a.txt"],
        summaries=["updated a.txt"],
        previews=[{"kind": "diff", "path": "a.txt", "lines": [], "more": 0, "adds": 0, "dels": 0}],
    )

    loaded = snapshot_service.load_pending_action(action.action_id)
    assert loaded.status == "pending"
    assert loaded.source["a.txt"].kind == "file"

    updated = snapshot_service.update_pending_action(loaded, status="discarded", message="discarded")
    assert snapshot_service.load_pending_action(action.action_id).status == "discarded"
    assert updated.message == "discarded"
