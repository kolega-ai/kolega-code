from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from kolega_code.cli.tui.session_diff import (
    REWIND_MAX_CHECKPOINTS,
    GitSessionDiffTracker,
    RewindDriftError,
    SnapshotLedgerDiffTracker,
)
from kolega_code.services.file_system import LocalFileSystem
from kolega_code.services.snapshots import SnapshotService


def _git(project: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=project, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _init_repo(project: Path) -> None:
    _git(project, "init")
    _git(project, "config", "user.email", "test@example.com")
    _git(project, "config", "user.name", "Test User")


def _commit_all(project: Path, message: str = "initial") -> None:
    _git(project, "add", ".")
    _git(project, "commit", "-m", message)


def _head_sha(project: Path) -> str:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project, check=True, stdout=subprocess.PIPE, text=True)
    return completed.stdout.strip()


def _by_path(changes):
    return {change.path: change for change in changes}


def _git_tracker(project: Path) -> GitSessionDiffTracker:
    tracker = GitSessionDiffTracker.create(project)
    assert tracker is not None
    tracker.capture_baseline()
    return tracker


# ---- git tracker: checkpoints ------------------------------------------------


def test_checkpoint_scopes_diff_to_turn(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    _init_repo(project)
    (project / "a.py").write_text("one\n", encoding="utf-8")
    _commit_all(project)

    tracker = _git_tracker(project)
    (project / "a.py").write_text("two\n", encoding="utf-8")
    turn = tracker.capture_checkpoint("turn 2")
    (project / "b.py").write_text("new\n", encoding="utf-8")

    since_turn = _by_path(tracker.refresh(checkpoint_id=turn.checkpoint_id))
    assert set(since_turn) == {"b.py"}
    assert since_turn["b.py"].status == "added"

    since_start = _by_path(tracker.refresh())
    assert set(since_start) == {"a.py", "b.py"}
    assert since_start["a.py"].status == "modified"


def test_checkpoint_eviction_keeps_session_start(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    _init_repo(project)
    (project / "a.py").write_text("one\n", encoding="utf-8")
    _commit_all(project)

    tracker = _git_tracker(project)
    for index in range(REWIND_MAX_CHECKPOINTS + 5):
        tracker.capture_checkpoint(f"turn {index}")

    checkpoints = tracker.checkpoints()
    assert len(checkpoints) == REWIND_MAX_CHECKPOINTS
    assert checkpoints[0].checkpoint_id == 0
    assert checkpoints[-1].label == f"turn {REWIND_MAX_CHECKPOINTS + 4}"


def test_checkpoints_intern_identical_dirty_content(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    _init_repo(project)
    (project / "a.py").write_text("committed\n", encoding="utf-8")
    _commit_all(project)
    (project / "a.py").write_text("dirty\n", encoding="utf-8")

    tracker = _git_tracker(project)
    first = tracker.capture_checkpoint("turn 1")
    second = tracker.capture_checkpoint("turn 2")

    base = tracker.checkpoints()[0].dirty["a.py"]
    assert first.dirty["a.py"].data is base.data
    assert second.dirty["a.py"].data is base.data


# ---- git tracker: restore ----------------------------------------------------


def test_restore_plan_reverts_modified_added_and_deleted(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    _init_repo(project)
    (project / "a.py").write_text("old\n", encoding="utf-8")
    (project / "d.py").write_text("keep\n", encoding="utf-8")
    _commit_all(project)
    (project / "c.txt").write_text("pre\n", encoding="utf-8")  # dirty at session start

    tracker = _git_tracker(project)
    (project / "a.py").write_text("new\n", encoding="utf-8")
    (project / "c.txt").write_text("changed\n", encoding="utf-8")
    (project / "d.py").unlink()
    (project / "sub").mkdir()
    (project / "sub" / "new.py").write_text("created\n", encoding="utf-8")

    plan = tracker.build_restore_plan()
    assert {item.display_path for item in plan} == {"a.py", "c.txt", "d.py", "sub/new.py"}

    result = tracker.apply_restore_plan(plan)
    assert result.errors == []
    assert result.skipped == []
    assert sorted(result.restored) == ["a.py", "c.txt", "d.py"]
    assert result.deleted == ["sub/new.py"]

    assert (project / "a.py").read_text(encoding="utf-8") == "old\n"
    assert (project / "c.txt").read_text(encoding="utf-8") == "pre\n"
    assert (project / "d.py").read_text(encoding="utf-8") == "keep\n"
    assert not (project / "sub" / "new.py").exists()
    assert tracker.refresh() == []


def test_restore_preserves_non_utf8_and_binary_bytes(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    _init_repo(project)
    (project / "keep.txt").write_text("x\n", encoding="utf-8")
    _commit_all(project)
    latin = b"caf\xe9\n"
    binary = b"a\x00b"
    (project / "latin.txt").write_bytes(latin)
    (project / "bin.dat").write_bytes(binary)

    tracker = _git_tracker(project)
    (project / "latin.txt").write_bytes(b"caf\xc3\xa9 changed\n")
    (project / "bin.dat").write_bytes(b"c\x00d\x00e")

    result = tracker.apply_restore_plan(tracker.build_restore_plan())
    assert result.errors == []
    assert (project / "latin.txt").read_bytes() == latin
    assert (project / "bin.dat").read_bytes() == binary


def test_restore_recreates_deleted_nested_file(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    _init_repo(project)
    (project / "pkg" / "inner").mkdir(parents=True)
    (project / "pkg" / "inner" / "mod.py").write_text("content\n", encoding="utf-8")
    _commit_all(project)

    tracker = _git_tracker(project)
    (project / "pkg" / "inner" / "mod.py").unlink()
    (project / "pkg" / "inner").rmdir()
    (project / "pkg").rmdir()

    result = tracker.apply_restore_plan(tracker.build_restore_plan())
    assert result.errors == []
    assert (project / "pkg" / "inner" / "mod.py").read_text(encoding="utf-8") == "content\n"


def test_restore_drift_aborts_unless_forced(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    _init_repo(project)
    (project / "a.py").write_text("old\n", encoding="utf-8")
    _commit_all(project)

    tracker = _git_tracker(project)
    (project / "a.py").write_text("new\n", encoding="utf-8")
    plan = tracker.build_restore_plan()

    (project / "a.py").write_text("drifted after planning\n", encoding="utf-8")
    with pytest.raises(RewindDriftError) as excinfo:
        tracker.apply_restore_plan(plan)
    assert excinfo.value.paths == ["a.py"]

    result = tracker.apply_restore_plan(plan, force=True)
    assert result.restored == ["a.py"]
    assert (project / "a.py").read_text(encoding="utf-8") == "old\n"


def test_gitignored_pre_session_file_is_not_deleted(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    _init_repo(project)
    (project / ".gitignore").write_text("secret.txt\n", encoding="utf-8")
    _commit_all(project)
    (project / "secret.txt").write_text("pre-session\n", encoding="utf-8")

    tracker = _git_tracker(project)
    (project / "secret.txt").write_text("edited by agent\n", encoding="utf-8")

    plan = tracker.build_restore_plan(event_paths=["secret.txt"])
    assert [item.action for item in plan] == ["delete"]

    result = tracker.apply_restore_plan(plan)
    assert result.skipped == [("secret.txt", "gitignored file kept")]
    assert result.deleted == []
    assert (project / "secret.txt").read_text(encoding="utf-8") == "edited by agent\n"


def test_committed_changes_stay_restorable_without_touching_history(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    _init_repo(project)
    (project / "a.py").write_text("old\n", encoding="utf-8")
    _commit_all(project)

    tracker = _git_tracker(project)
    (project / "a.py").write_text("new\n", encoding="utf-8")
    _commit_all(project, "mid-session commit")
    committed_sha = _head_sha(project)

    change = _by_path(tracker.refresh())["a.py"]
    assert change.status == "modified"

    result = tracker.apply_restore_plan(tracker.build_restore_plan())
    assert result.restored == ["a.py"]
    assert (project / "a.py").read_text(encoding="utf-8") == "old\n"
    assert _head_sha(project) == committed_sha


def test_per_file_restore_filter(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    _init_repo(project)
    (project / "a.py").write_text("a old\n", encoding="utf-8")
    (project / "b.py").write_text("b old\n", encoding="utf-8")
    _commit_all(project)

    tracker = _git_tracker(project)
    (project / "a.py").write_text("a new\n", encoding="utf-8")
    (project / "b.py").write_text("b new\n", encoding="utf-8")

    plan = tracker.build_restore_plan(paths={"a.py"})
    assert [item.display_path for item in plan] == ["a.py"]

    tracker.apply_restore_plan(plan)
    assert (project / "a.py").read_text(encoding="utf-8") == "a old\n"
    assert (project / "b.py").read_text(encoding="utf-8") == "b new\n"


# ---- snapshot-ledger tracker (non-git) ---------------------------------------


def _ledger_setup(tmp_path: Path) -> tuple[Path, SnapshotService, SnapshotLedgerDiffTracker]:
    project = tmp_path / "project"
    project.mkdir()
    service = SnapshotService(
        project,
        "workspace",
        "thread",
        "session",
        LocalFileSystem(project),
        root=tmp_path / "state",
    )
    tracker = SnapshotLedgerDiffTracker(project, service)
    tracker.capture_baseline()
    return project, service, tracker


def _mutate(service: SnapshotService, project: Path, path: str, content: str | None) -> None:
    # Snapshot record timestamps order the ledger; keep them strictly after the
    # most recent checkpoint.
    time.sleep(0.01)

    def apply() -> None:
        target = project / path
        if content is None:
            target.unlink()
        else:
            target.write_text(content, encoding="utf-8")

    service.record_mutation(tool_name="edit", tool_call_id=None, reason="test", paths=[path], mutate=apply)


def test_ledger_diff_and_restore_roundtrip(tmp_path: Path) -> None:
    project, service, tracker = _ledger_setup(tmp_path)
    (project / "a.txt").write_text("before\n", encoding="utf-8")

    _mutate(service, project, "a.txt", "after\n")
    _mutate(service, project, "new.txt", "created\n")

    changes = _by_path(tracker.refresh())
    assert changes["a.txt"].status == "modified"
    assert changes["new.txt"].status == "added"

    result = tracker.apply_restore_plan(tracker.build_restore_plan())
    assert result.errors == []
    assert (project / "a.txt").read_text(encoding="utf-8") == "before\n"
    assert not (project / "new.txt").exists()
    assert tracker.refresh() == []


def test_ledger_checkpoint_scoping_and_restore(tmp_path: Path) -> None:
    project, service, tracker = _ledger_setup(tmp_path)
    (project / "a.txt").write_text("v1\n", encoding="utf-8")

    _mutate(service, project, "a.txt", "v2\n")
    time.sleep(0.01)
    turn = tracker.capture_checkpoint("turn 2")
    _mutate(service, project, "a.txt", "v3\n")

    since_turn = _by_path(tracker.refresh(checkpoint_id=turn.checkpoint_id))
    preview = since_turn["a.txt"].preview
    assert preview is not None
    lines = [row[1] for row in preview["lines"] if row[0] in {"add", "del"}]
    assert lines == ["-v2", "+v3"]

    tracker.apply_restore_plan(tracker.build_restore_plan(checkpoint_id=turn.checkpoint_id))
    assert (project / "a.txt").read_text(encoding="utf-8") == "v2\n"

    since_start = _by_path(tracker.refresh())
    preview = since_start["a.txt"].preview
    assert preview is not None
    lines = [row[1] for row in preview["lines"] if row[0] in {"add", "del"}]
    assert lines == ["-v1", "+v2"]


def test_ledger_ignores_manual_snapshots(tmp_path: Path) -> None:
    project, service, tracker = _ledger_setup(tmp_path)
    (project / "a.txt").write_text("stable\n", encoding="utf-8")

    time.sleep(0.01)
    service.create_manual_snapshot(paths=["a.txt"], reason="pre-rewind safety")

    assert tracker.refresh() == []
