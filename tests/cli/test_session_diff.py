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


def _count_method_calls(monkeypatch, target, name: str):
    original = getattr(target, name)
    calls = {"count": 0}

    def wrapper(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(target, name, wrapper)
    return calls


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
    assert change.preview is not None
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
    assert change.preview is not None
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
    assert change.preview is not None
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
    assert change.preview is not None
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


def test_second_refresh_reuses_cached_diff_without_file_work(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    _init_repo(project)
    (project / "a.py").write_text("old\n", encoding="utf-8")
    _commit_all(project)

    tracker = GitSessionDiffTracker.create(project)
    assert tracker is not None
    tracker.capture_baseline()
    (project / "a.py").write_text("new content\n", encoding="utf-8")

    head_calls = _count_method_calls(monkeypatch, tracker, "_commit_baseline")
    snapshot_calls = _count_method_calls(monkeypatch, tracker, "_snapshot_repo_path")
    first = tracker.refresh()
    assert first

    head_calls["count"] = 0
    snapshot_calls["count"] = 0
    second = tracker.refresh()

    assert second == first
    assert head_calls["count"] == 0
    assert snapshot_calls["count"] == 0


def test_refresh_recomputes_when_file_stat_signature_changes(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    _init_repo(project)
    (project / "a.py").write_text("old\n", encoding="utf-8")
    _commit_all(project)

    tracker = GitSessionDiffTracker.create(project)
    assert tracker is not None
    tracker.capture_baseline()
    (project / "a.py").write_text("one\n", encoding="utf-8")
    first_change = _by_path(tracker.refresh())["a.py"]
    assert first_change.preview is not None
    assert any(row[1] == "+one" for row in first_change.preview["lines"])

    snapshot_calls = _count_method_calls(monkeypatch, tracker, "_snapshot_repo_path")
    (project / "a.py").write_text("second content\n", encoding="utf-8")
    change = _by_path(tracker.refresh())["a.py"]

    assert snapshot_calls["count"] == 1
    assert change.preview is not None
    assert any(row[1] == "+second content" for row in change.preview["lines"])


def test_committing_mid_session_keeps_changes_visible(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    _init_repo(project)
    (project / "a.py").write_text("old\n", encoding="utf-8")
    _commit_all(project)

    tracker = GitSessionDiffTracker.create(project)
    assert tracker is not None
    tracker.capture_baseline()
    (project / "a.py").write_text("new content\n", encoding="utf-8")
    assert _by_path(tracker.refresh(["a.py"]))["a.py"].status == "modified"

    _commit_all(project, "commit modified file")

    # The checkpoint pins the session-start HEAD, so committing does not hide
    # the change; it stays visible (and restorable) via the committed-paths scan.
    change = _by_path(tracker.refresh(["a.py"]))["a.py"]
    assert change.status == "modified"
    assert change.preview is not None
    lines = [row[1] for row in change.preview["lines"] if row[0] in {"add", "del"}]
    assert lines == ["-old", "+new content"]


def test_deleted_file_second_refresh_uses_cached_diff(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    _init_repo(project)
    (project / "a.py").write_text("old\n", encoding="utf-8")
    _commit_all(project)

    tracker = GitSessionDiffTracker.create(project)
    assert tracker is not None
    tracker.capture_baseline()
    (project / "a.py").unlink()

    head_calls = _count_method_calls(monkeypatch, tracker, "_commit_baseline")
    first = _by_path(tracker.refresh())["a.py"]
    assert first.status == "deleted"

    head_calls["count"] = 0
    second = _by_path(tracker.refresh())["a.py"]

    assert second == first
    assert head_calls["count"] == 0


def test_added_binary_file_second_refresh_uses_cached_diff(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    _init_repo(project)
    (project / "README.md").write_text("# Repo\n", encoding="utf-8")
    _commit_all(project)

    tracker = GitSessionDiffTracker.create(project)
    assert tracker is not None
    tracker.capture_baseline()
    (project / "bin.dat").write_bytes(b"abc\x00def")

    head_calls = _count_method_calls(monkeypatch, tracker, "_commit_baseline")
    first = _by_path(tracker.refresh())["bin.dat"]
    assert first.status == "added"
    assert first.preview is None
    assert first.message == "Binary or unreadable file changed; textual diff unavailable."

    head_calls["count"] = 0
    second = _by_path(tracker.refresh())["bin.dat"]

    assert second == first
    assert head_calls["count"] == 0


def test_repo_with_no_commits_reports_added_files(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    _init_repo(project)

    tracker = GitSessionDiffTracker.create(project)
    assert tracker is not None
    tracker.capture_baseline()
    (project / "a.py").write_text("new\n", encoding="utf-8")

    change = _by_path(tracker.refresh())["a.py"]

    assert change.status == "added"
    assert change.adds == 1
    assert change.preview is not None
    assert any(row[1] == "+new" for row in change.preview["lines"])


def test_clean_event_paths_second_refresh_skips_head_baseline_calls(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    _init_repo(project)
    event_paths = [f"file_{index}.txt" for index in range(50)]
    for path in event_paths:
        (project / path).write_text(f"{path}\n", encoding="utf-8")
    _commit_all(project)

    tracker = GitSessionDiffTracker.create(project)
    assert tracker is not None
    tracker.capture_baseline()

    head_calls = _count_method_calls(monkeypatch, tracker, "_commit_baseline")
    assert tracker.refresh(event_paths) == []
    assert head_calls["count"] == len(event_paths)

    head_calls["count"] = 0
    assert tracker.refresh(event_paths) == []
    assert head_calls["count"] == 0


def test_non_git_create_returns_none(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()

    assert GitSessionDiffTracker.create(project) is None
