from pathlib import Path
import json
import os
import stat

import pytest

from kolega_code.cli.session_store import SessionStore, SessionStoreError, default_state_dir


def test_default_state_dir_honors_env() -> None:
    assert default_state_dir({"KOLEGA_CODE_STATE_DIR": "/tmp/kolega-test"}) == Path("/tmp/kolega-test")


def test_session_store_writes_private_files_and_directories(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")

    old_umask = os.umask(0)
    try:
        record = store.create(project, "code", {"api_key": "secret"})
    finally:
        os.umask(old_umask)

    if os.name != "nt":
        assert stat.S_IMODE(store.root.stat().st_mode) == 0o700
        assert stat.S_IMODE(store.sessions_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(store.path_for(record.session_id).stat().st_mode) == 0o600


def test_session_store_create_load_list_export_delete(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")

    record = store.create(project, "code", {"long_model": "claude-opus-4-7"}, title="Project")
    record.history = [{"role": "user", "content": []}]
    record.task_list_markdown = "- [ ] inspect\n- [x] plan"
    record.latest_plan_markdown = "# Plan\n\nImplement it."
    record.plan_pending = True
    record.plan_reofferable = True
    record.interaction_mode = "plan"
    record.permission_mode = "auto"
    record.gigacode_enabled = True
    store.save(record)

    loaded = store.load(record.session_id)
    assert loaded.project_path == str(project.resolve())
    assert loaded.history == [{"role": "user", "content": []}]
    assert loaded.task_list_markdown == "- [ ] inspect\n- [x] plan"
    assert loaded.latest_plan_markdown == "# Plan\n\nImplement it."
    assert loaded.plan_pending is True
    assert loaded.plan_reofferable is True
    assert loaded.interaction_mode == "plan"
    assert loaded.permission_mode == "auto"
    assert loaded.gigacode_enabled is True
    assert store.latest_for_project(project).session_id == record.session_id
    exported = store.export(record.session_id)
    assert record.session_id in exported
    assert "task_list_markdown" in exported
    assert "latest_plan_markdown" in exported
    assert "plan_pending" in exported
    assert "plan_reofferable" in exported
    assert "interaction_mode" in exported
    assert "permission_mode" in exported
    assert "gigacode_enabled" in exported

    store.delete(record.session_id)
    with pytest.raises(SessionStoreError):
        store.load(record.session_id)


def test_session_store_loads_old_sessions_without_planning_state(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")

    record = store.create(project, "code", {})
    payload = record.to_dict()
    payload.pop("task_list_markdown")
    payload.pop("latest_plan_markdown")
    payload.pop("plan_pending")
    payload.pop("plan_reofferable")
    payload.pop("interaction_mode")
    payload.pop("permission_mode")
    payload.pop("gigacode_enabled")
    store.path_for(record.session_id).write_text(json.dumps(payload), encoding="utf-8")

    loaded = store.load(record.session_id)

    assert loaded.task_list_markdown == ""
    assert loaded.latest_plan_markdown == ""
    assert loaded.plan_pending is False
    assert loaded.plan_reofferable is False
    assert loaded.interaction_mode == "build"
    assert loaded.permission_mode == "ask"
    assert loaded.gigacode_enabled is False


def test_session_store_old_pending_plan_is_reofferable(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")

    record = store.create(project, "code", {})
    payload = record.to_dict()
    payload["latest_plan_markdown"] = "# Plan\n\nImplement it."
    payload["plan_pending"] = True
    payload.pop("plan_reofferable")
    store.path_for(record.session_id).write_text(json.dumps(payload), encoding="utf-8")

    loaded = store.load(record.session_id)

    assert loaded.latest_plan_markdown == "# Plan\n\nImplement it."
    assert loaded.plan_pending is True
    assert loaded.plan_reofferable is True


def test_session_store_round_trips_compaction(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")

    record = store.create(project, "code", {})
    record.compaction = {"summary": "## Goal\nShip it", "compacted_through": 7}
    store.save(record)

    loaded = store.load(record.session_id)
    assert loaded.compaction == {"summary": "## Goal\nShip it", "compacted_through": 7}


def test_session_store_loads_old_sessions_without_compaction(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")

    record = store.create(project, "code", {})
    payload = record.to_dict()
    payload.pop("compaction")
    store.path_for(record.session_id).write_text(json.dumps(payload), encoding="utf-8")

    loaded = store.load(record.session_id)
    assert loaded.compaction == {}


def test_session_store_ignores_corrupt_files_when_listing(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "state")
    store.ensure_dirs()
    (store.sessions_dir / "bad.json").write_text("{not json", encoding="utf-8")

    assert store.list() == []
