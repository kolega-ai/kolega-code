import json
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from kolega_code.cli.session_store import SessionRecord, SessionStore, SessionStoreError


def _create_session(tmp_path: Path) -> tuple[Path, Path, SessionStore, str]:
    project = tmp_path / "project"
    project.mkdir()
    state_dir = tmp_path / "state"
    store = SessionStore(state_dir)
    record = store.create(project, "code", {})
    return project, state_dir, store, record.session_id


def _assert_claim_refused(state_dir: Path, session_id: str, project: Path) -> None:
    with pytest.raises(SessionStoreError, match="already open"):
        SessionStore(state_dir).claim_for_resume(session_id, project)


def test_claim_reserves_lock_until_same_store_recorder_initializes_and_releases(tmp_path: Path) -> None:
    project, state_dir, store, session_id = _create_session(tmp_path)

    claimed = store.claim_for_resume(session_id, project)
    assert claimed.session_id == session_id
    _assert_claim_refused(state_dir, session_id, project)

    recorder = store.recorder(session_id)
    assert recorder.journal is store.journal(session_id)
    _assert_claim_refused(state_dir, session_id, project)

    store.release_session_lock(session_id)
    retry_store = SessionStore(state_dir)
    assert retry_store.claim_for_resume(session_id, project).session_id == session_id
    retry_store.release_session_lock(session_id)


def test_subprocess_claim_contention_does_not_load_or_repair_and_retry_succeeds_after_release(
    tmp_path: Path,
) -> None:
    project, state_dir, store, session_id = _create_session(tmp_path)
    marker = tmp_path / "claimed.marker"
    child_script = tmp_path / "hold_claim.py"
    child_script.write_text(
        "import sys\n"
        "import time\n"
        "from pathlib import Path\n"
        "from kolega_code.cli.session_store import SessionStore\n"
        "state_dir, session_id, project, marker = sys.argv[1:]\n"
        "store = SessionStore(Path(state_dir))\n"
        "store.claim_for_resume(session_id, Path(project))\n"
        "Path(marker).write_text('claimed', encoding='utf-8')\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )

    proc = subprocess.Popen(
        [sys.executable, str(child_script), str(state_dir), session_id, str(project), str(marker)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if marker.exists():
                break
            if proc.poll() is not None:
                child_error = proc.stderr.read() if proc.stderr is not None else ""
                raise AssertionError(f"child exited before claiming the session: {child_error}")
            time.sleep(0.05)
        assert marker.exists(), "child never acquired the resume claim"

        metadata_path = store.path_for(session_id)
        events_path = store.events_path_for(session_id)
        events_path.write_bytes(events_path.read_bytes() + b'{"partial":')
        metadata_before = metadata_path.read_bytes()
        events_before = events_path.read_bytes()

        contender = SessionStore(state_dir)
        with pytest.raises(SessionStoreError, match="already open"):
            contender.claim_for_resume(session_id, project)

        assert metadata_path.read_bytes() == metadata_before
        assert events_path.read_bytes() == events_before
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    retried = contender.claim_for_resume(session_id, project)
    assert retried.session_id == session_id
    assert events_path.read_bytes().endswith(b"\n")
    contender.release_session_lock(session_id)


def test_release_old_session_does_not_release_reserved_resume_target(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    state_dir = tmp_path / "state"
    store = SessionStore(state_dir)
    old = store.create(project, "code", {})
    target = store.create(project, "code", {})

    assert store.claim_for_resume(target.session_id, project).session_id == target.session_id
    store.recorder(old.session_id)
    store.release_session_lock(old.session_id)

    _assert_claim_refused(state_dir, target.session_id, project)

    store.release_session_lock(target.session_id)
    retry_store = SessionStore(state_dir)
    assert retry_store.claim_for_resume(target.session_id, project).session_id == target.session_id
    retry_store.release_session_lock(target.session_id)


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("missing", "Session not found"),
        ("mismatched_project", "different project"),
        ("corrupt_metadata", "Unsupported session schema version"),
        ("corrupt_journal", "Invalid JSON in session event log"),
    ],
)
def test_rejected_claims_do_not_leak_locks(tmp_path: Path, case: str, match: str) -> None:
    project = tmp_path / "project"
    project.mkdir()
    other_project = tmp_path / "other-project"
    other_project.mkdir()
    state_dir = tmp_path / "state"
    store = SessionStore(state_dir)

    if case == "missing":
        session_id = "missing-session"
        with pytest.raises(SessionStoreError, match=match):
            store.claim_for_resume(session_id, project)
        record = store.create(project, "code", {}, session_id=session_id)
    else:
        record = store.create(project, "code", {})
        metadata_path = store.path_for(record.session_id)
        events_path = store.events_path_for(record.session_id)
        original_metadata = metadata_path.read_bytes()
        original_events = events_path.read_bytes()
        claim_project = project

        if case == "mismatched_project":
            claim_project = other_project
        elif case == "corrupt_metadata":
            metadata = json.loads(original_metadata.decode("utf-8"))
            metadata["schema_version"] = 999
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        elif case == "corrupt_journal":
            events_path.write_text("{not-json\n", encoding="utf-8")

        with pytest.raises(SessionStoreError, match=match):
            store.claim_for_resume(record.session_id, claim_project)

        if case == "corrupt_metadata":
            metadata_path.write_bytes(original_metadata)
        elif case == "corrupt_journal":
            events_path.write_bytes(original_events)

    retry_store = SessionStore(state_dir)
    assert retry_store.claim_for_resume(record.session_id, project).session_id == record.session_id
    retry_store.release_session_lock(record.session_id)


def test_existing_same_store_ownership_survives_failed_nested_claim(tmp_path: Path) -> None:
    project, state_dir, store, session_id = _create_session(tmp_path)
    wrong_project = tmp_path / "wrong-project"
    wrong_project.mkdir()

    assert store.claim_for_resume(session_id, project).session_id == session_id
    with pytest.raises(SessionStoreError, match="different project"):
        store.claim_for_resume(session_id, wrong_project)

    _assert_claim_refused(state_dir, session_id, project)

    store.release_session_lock(session_id)
    retry_store = SessionStore(state_dir)
    assert retry_store.claim_for_resume(session_id, project).session_id == session_id
    retry_store.release_session_lock(session_id)


def test_malformed_metadata_session_id_rejects_without_holding_target_lock(tmp_path: Path) -> None:
    project, state_dir, store, session_id = _create_session(tmp_path)
    metadata_path = store.path_for(session_id)
    original_metadata = metadata_path.read_bytes()
    metadata = json.loads(original_metadata.decode("utf-8"))
    metadata["session_id"] = "wrong-session-id"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(SessionStoreError, match="saved session ID"):
        store.claim_for_resume(session_id, project)

    metadata_path.write_bytes(original_metadata)
    retry_store = SessionStore(state_dir)
    assert retry_store.claim_for_resume(session_id, project).session_id == session_id
    retry_store.release_session_lock(session_id)


def _session_tree_snapshot(root: Path) -> dict[Path, tuple[int, bytes | None]]:
    return {
        path.relative_to(root): (path.stat().st_mtime_ns, path.read_bytes() if path.is_file() else None)
        for path in (root, *root.rglob("*"))
    }


@pytest.mark.parametrize("damage", ["none", "corrupt", "missing", "non_object", "missing_field", "unsupported_schema"])
@pytest.mark.parametrize("other_project", [False, True])
def test_metadata_listing_never_repairs_locked_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str, other_project: bool
) -> None:
    project, state_dir, owner, session_id = _create_session(tmp_path)
    listing_project = tmp_path / "other-project" if other_project else project
    owner.claim_for_resume(session_id, project)
    try:
        metadata_path = owner.path_for(session_id)
        if damage == "corrupt":
            metadata_path.write_text('{"partial":', encoding="utf-8")
        elif damage == "missing":
            metadata_path.unlink()
        elif damage == "non_object":
            metadata_path.write_text("[]", encoding="utf-8")
        elif damage in {"missing_field", "unsupported_schema"}:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if damage == "missing_field":
                del metadata["updated_at"]
            else:
                metadata["schema_version"] = 999
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        events_path = owner.events_path_for(session_id)
        events_path.write_bytes(events_path.read_bytes() + b'{"partial":')
        before = _session_tree_snapshot(state_dir)

        reader = SessionStore(state_dir)
        journal = Mock(side_effect=AssertionError("Metadata listing must not open journals"))
        monkeypatch.setattr(reader, "journal", journal)
        records = reader.list(listing_project, recover=False)

        expected_ids = [session_id] if damage == "none" and not other_project else []
        assert [record.session_id for record in records] == expected_ids
        journal.assert_not_called()
        assert _session_tree_snapshot(state_dir) == before
        _assert_claim_refused(state_dir, session_id, project)
    finally:
        owner.release_session_lock(session_id)


def test_metadata_listing_does_not_create_state_directories(tmp_path: Path) -> None:
    state_dir = tmp_path / "absent-state"
    assert SessionStore(state_dir).list(tmp_path, recover=False) == []
    assert not state_dir.exists()


def test_metadata_listing_reads_legacy_without_migrating_and_keeps_filter_and_sort(tmp_path: Path) -> None:
    project, state_dir, store, session_id = _create_session(tmp_path)
    metadata_path = store.path_for(session_id)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["updated_at"] = "2026-01-02T00:00:00+00:00"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    # A leftover legacy file must not be cleaned up or override the directory.
    store.legacy_path_for(session_id).write_text(
        json.dumps({**metadata, "updated_at": "2026-01-04T00:00:00+00:00"}), encoding="utf-8"
    )
    legacy = SessionRecord.create(project, "code", {}, session_id="legacy")
    legacy.updated_at = "2026-01-03T00:00:00+00:00"
    legacy.history = [{"role": "user", "content": "Keep this in the legacy file"}]
    legacy.compaction = {"summary": "Keep this too"}
    store.legacy_path_for(legacy.session_id).write_text(json.dumps(legacy.to_dict()), encoding="utf-8")
    older = SessionRecord.create(project, "code", {}, session_id="older-legacy")
    older.updated_at = "2026-01-01T00:00:00+00:00"
    store.legacy_path_for(older.session_id).write_text(json.dumps(older.to_dict()), encoding="utf-8")
    other = SessionRecord.create(tmp_path / "other-project", "code", {}, session_id="other-legacy")
    store.legacy_path_for(other.session_id).write_text(json.dumps(other.to_dict()), encoding="utf-8")
    before = _session_tree_snapshot(state_dir)

    records = SessionStore(state_dir).list(project, recover=False)

    assert [record.session_id for record in records] == ["legacy", session_id, "older-legacy"]
    assert all(record.history == [] and record.compaction == {} for record in records)
    assert _session_tree_snapshot(state_dir) == before
    assert not store.session_dir_for("legacy").exists()
    assert not store.session_dir_for("other-legacy").exists()


@pytest.mark.parametrize("payload", ['{"partial":', "[]", "null", "{}"])
def test_metadata_listing_skips_malformed_legacy_without_migrating(tmp_path: Path, payload: str) -> None:
    project, state_dir, store, session_id = _create_session(tmp_path)
    store.legacy_path_for("malformed").write_text(payload, encoding="utf-8")
    before = _session_tree_snapshot(state_dir)

    assert [record.session_id for record in SessionStore(state_dir).list(project, recover=False)] == [session_id]
    assert _session_tree_snapshot(state_dir) == before


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("recover", [False, True])
def test_listing_rejects_metadata_ids_that_point_to_other_sessions(tmp_path: Path, legacy: bool, recover: bool) -> None:
    project, state_dir, store, session_id = _create_session(tmp_path)
    target = store.create(project, "code", {}, title="Actual target title")
    forged = SessionRecord.create(project, "code", {}, session_id=target.session_id, title="Misleading title")
    if legacy:
        store.legacy_path_for("forged-legacy").write_text(json.dumps(forged.to_dict()), encoding="utf-8")
        expected_ids = {session_id, target.session_id}
    else:
        store.path_for(session_id).write_text(json.dumps(forged.to_metadata_dict()), encoding="utf-8")
        expected_ids = {target.session_id}

    records = SessionStore(state_dir).list(project, recover=recover)

    assert {record.session_id for record in records} == expected_ids
    assert len(records) == len(expected_ids)
    assert all(record.title != "Misleading title" for record in records)
    assert next(record for record in records if record.session_id == target.session_id).title == "Actual target title"


@pytest.mark.parametrize("damage", ["corrupt", "missing"])
def test_default_listing_still_recovers_metadata_and_partial_journal(tmp_path: Path, damage: str) -> None:
    project, state_dir, store, session_id = _create_session(tmp_path)
    metadata_path = store.path_for(session_id)
    if damage == "corrupt":
        metadata_path.write_text('{"partial":', encoding="utf-8")
    else:
        metadata_path.unlink()
    events_path = store.events_path_for(session_id)
    original_events = events_path.read_bytes()
    events_path.write_bytes(original_events + b'{"partial":')

    assert [record.session_id for record in SessionStore(state_dir).list(project)] == [session_id]
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["session_id"] == session_id
    assert events_path.read_bytes() == original_events
