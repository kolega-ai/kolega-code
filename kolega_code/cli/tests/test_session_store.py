from pathlib import Path

import pytest

from kolega_code.cli.session_store import SessionStore, SessionStoreError, default_state_dir


def test_default_state_dir_honors_env() -> None:
    assert default_state_dir({"KOLEGA_CODE_STATE_DIR": "/tmp/kolega-test"}) == Path("/tmp/kolega-test")


def test_session_store_create_load_list_export_delete(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")

    record = store.create(project, "code", {"long_model": "claude-opus-4-7"}, title="Project")
    record.history = [{"role": "user", "content": []}]
    store.save(record)

    loaded = store.load(record.session_id)
    assert loaded.project_path == str(project.resolve())
    assert loaded.history == [{"role": "user", "content": []}]
    assert store.latest_for_project(project).session_id == record.session_id
    assert record.session_id in store.export(record.session_id)

    store.delete(record.session_id)
    with pytest.raises(SessionStoreError):
        store.load(record.session_id)


def test_session_store_ignores_corrupt_files_when_listing(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "state")
    store.ensure_dirs()
    (store.sessions_dir / "bad.json").write_text("{not json", encoding="utf-8")

    assert store.list() == []
