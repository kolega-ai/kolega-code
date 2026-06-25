from __future__ import annotations

import subprocess
from pathlib import Path

from kolega_code.cli.tui.session_diff import GitSessionDiffTracker


def _git(project: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=project, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _init_repo(project: Path) -> None:
    _git(project, "init")
    _git(project, "config", "user.email", "test@example.com")
    _git(project, "config", "user.name", "Test User")


def _commit_all(project: Path, message: str = "initial") -> None:
    _git(project, "add", ".")
    _git(project, "commit", "-m", message)


def _by_path(changes):
    return {change.path: change for change in changes}


def test_tracked_file_modified_after_baseline(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    _init_repo(project)
    (project / "a.py").write_text("old\n", encoding="utf-8")
    _commit_all(project)

    tracker = GitSessionDiffTracker.create(project)
    assert tracker is not None
    tracker.capture_baseline()

    (project / "a.py").write_text("new\n", encoding="utf-8")
    change = _by_path(tracker.refresh())["a.py"]

    assert change.status == "modified"
    assert change.adds == 1
    assert change.dels == 1
    assert [row[1] for row in change.preview["lines"] if row[0] in {"add", "del"}] == ["-old", "+new"]


def test_tracked_file_deleted_after_baseline(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    _init_repo(project)
    (project / "a.py").write_text("old\n", encoding="utf-8")
    _commit_all(project)

    tracker = GitSessionDiffTracker.create(project)
    assert tracker is not None
    tracker.capture_baseline()

    (project / "a.py").unlink()
    change = _by_path(tracker.refresh())["a.py"]

    assert change.status == "deleted"
    assert change.dels == 1
    assert any(row[1] == "-old" for row in change.preview["lines"])


def test_new_untracked_file_after_baseline(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    _init_repo(project)
    (project / "README.md").write_text("# Repo\n", encoding="utf-8")
    _commit_all(project)

    tracker = GitSessionDiffTracker.create(project)
    assert tracker is not None
    tracker.capture_baseline()

    (project / "new.py").write_text("print('new')\n", encoding="utf-8")
    change = _by_path(tracker.refresh())["new.py"]

    assert change.status == "added"
    assert change.adds == 1
    assert change.preview["kind"] == "diff"
    assert any(row[1] == "+print('new')" for row in change.preview["lines"])


def test_pre_existing_dirty_file_is_session_baseline(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    _init_repo(project)
    (project / "a.py").write_text("head\n", encoding="utf-8")
    _commit_all(project)
    (project / "a.py").write_text("dirty\n", encoding="utf-8")

    tracker = GitSessionDiffTracker.create(project)
    assert tracker is not None
    tracker.capture_baseline()

    assert tracker.refresh() == []

    (project / "a.py").write_text("session\n", encoding="utf-8")
    change = _by_path(tracker.refresh())["a.py"]

    assert change.status == "modified"
    lines = [row[1] for row in change.preview["lines"] if row[0] in {"add", "del"}]
    assert lines == ["-dirty", "+session"]


def test_reverted_to_session_start_state_disappears(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    _init_repo(project)
    (project / "a.py").write_text("old\n", encoding="utf-8")
    _commit_all(project)

    tracker = GitSessionDiffTracker.create(project)
    assert tracker is not None
    tracker.capture_baseline()

    (project / "a.py").write_text("new\n", encoding="utf-8")
    assert _by_path(tracker.refresh())["a.py"].status == "modified"

    (project / "a.py").write_text("old\n", encoding="utf-8")
    assert tracker.refresh() == []


def test_non_git_create_returns_none(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()

    assert GitSessionDiffTracker.create(project) is None
