from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from kolega_code.cli.tui.session_diff import GitSessionDiffTracker


# These tests name the initial branch, merge it, and rebase onto it, so they must
# not inherit whatever the developer's ~/.gitconfig says about git's behaviour.
pytestmark = pytest.mark.usefixtures("hermetic_git_config")


def _git(project: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=project, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _init_repo(project: Path) -> None:
    # Pin the initial branch: without a global init.defaultBranch (as on CI) git
    # would name it "master" and every reference to "main" below would fail.
    _git(project, "init", "-b", "main")
    _git(project, "config", "user.email", "test@example.com")
    _git(project, "config", "user.name", "Test User")


def _commit_all(project: Path, message: str = "initial") -> None:
    _git(project, "add", ".")
    _git(project, "commit", "-m", message)


def _add_worktree(project: Path, name: str, branch: str, *, ignore: bool = True) -> Path:
    """Create a linked worktree under ``.kolega/worktrees/<name>`` and return its root."""
    if ignore:
        exclude = project / ".git" / "info" / "exclude"
        exclude.parent.mkdir(parents=True, exist_ok=True)
        with exclude.open("a", encoding="utf-8") as handle:
            handle.write("/.kolega/worktrees/\n")
    relative = f".kolega/worktrees/{name}"
    _git(project, "worktree", "add", "-b", branch, relative)
    return project / relative


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


# ---- worktree and history scoping -------------------------------------------
#
# Several Kolega Code instances routinely work in sibling worktrees of one
# repository. Whatever another instance committed must never be attributed to
# this session: it would be shown as our change and, worse, a rewind would
# revert it. Only commits created in this worktree during this session count.


def _repo_with_sibling_work(tmp_path: Path) -> tuple[Path, Path]:
    """A repo whose `feat-b` branch holds another agent's committed work.

    Returns (main checkout, worktree tracked by this session).
    """
    project = tmp_path / "repo"
    project.mkdir()
    _init_repo(project)
    (project / "shared.txt").write_text("base\n", encoding="utf-8")
    _commit_all(project)

    worktree_b = _add_worktree(project, "b", "feat-b")
    (worktree_b / "b_only.py").write_text("b\n", encoding="utf-8")
    (worktree_b / "shared.txt").write_text("base\nfrom b\n", encoding="utf-8")
    _commit_all(worktree_b, "b work")

    worktree_a = _add_worktree(project, "a", "feat-a", ignore=False)
    return project, worktree_a


def test_merged_branch_files_are_not_session_changes(tmp_path: Path) -> None:
    project, worktree_a = _repo_with_sibling_work(tmp_path)
    tracker = GitSessionDiffTracker.create(worktree_a)
    assert tracker is not None
    tracker.capture_baseline()

    (worktree_a / "a_only.py").write_text("a\n", encoding="utf-8")
    _commit_all(worktree_a, "a work")
    _git(project, "merge", "--no-edit", "feat-b")
    _git(worktree_a, "merge", "--no-edit", "main")

    changes = _by_path(tracker.refresh())

    assert "a_only.py" in changes  # our own committed work stays visible
    assert "b_only.py" not in changes  # the other agent's file must not appear
    assert "shared.txt" not in changes  # nor their edit to a shared file
    assert tracker.scope().history_moved is True


def test_fast_forward_merge_brings_no_session_changes(tmp_path: Path) -> None:
    project, worktree_a = _repo_with_sibling_work(tmp_path)
    tracker = GitSessionDiffTracker.create(worktree_a)
    assert tracker is not None
    tracker.capture_baseline()

    # This session changed nothing; it only brought the branch up to date.
    _git(project, "merge", "--no-edit", "feat-b")
    _git(worktree_a, "merge", "--no-edit", "main")

    assert tracker.refresh() == []
    assert tracker.scope().history_moved is True


def test_rebase_onto_updated_base_keeps_only_own_commits(tmp_path: Path) -> None:
    project, worktree_a = _repo_with_sibling_work(tmp_path)
    tracker = GitSessionDiffTracker.create(worktree_a)
    assert tracker is not None
    tracker.capture_baseline()

    (worktree_a / "a_only.py").write_text("a\n", encoding="utf-8")
    _commit_all(worktree_a, "a work")
    _git(project, "merge", "--no-edit", "feat-b")
    _git(worktree_a, "rebase", "main")

    changes = _by_path(tracker.refresh())

    assert set(changes) == {"a_only.py"}


def test_conflicted_merge_reports_only_resolved_files(tmp_path: Path) -> None:
    project, worktree_a = _repo_with_sibling_work(tmp_path)
    tracker = GitSessionDiffTracker.create(worktree_a)
    assert tracker is not None
    tracker.capture_baseline()

    (worktree_a / "shared.txt").write_text("base\nfrom a\n", encoding="utf-8")
    _commit_all(worktree_a, "a work")
    _git(project, "merge", "--no-edit", "feat-b")
    conflict = subprocess.run(
        ["git", "merge", "--no-edit", "main"],
        cwd=worktree_a,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert conflict.returncode != 0  # the merge really did conflict
    (worktree_a / "shared.txt").write_text("base\nfrom a\nfrom b\n", encoding="utf-8")
    _commit_all(worktree_a, "resolve")

    changes = _by_path(tracker.refresh())

    # The conflict resolution was authored here; b_only.py merely arrived.
    assert set(changes) == {"shared.txt"}


def test_restore_plan_excludes_merged_in_files(tmp_path: Path) -> None:
    project, worktree_a = _repo_with_sibling_work(tmp_path)
    tracker = GitSessionDiffTracker.create(worktree_a)
    assert tracker is not None
    tracker.capture_baseline()

    (worktree_a / "a_only.py").write_text("a\n", encoding="utf-8")
    _commit_all(worktree_a, "a work")
    _git(project, "merge", "--no-edit", "feat-b")
    _git(worktree_a, "merge", "--no-edit", "main")

    plan = tracker.build_restore_plan()

    # A rewind must never delete or revert another agent's committed work.
    assert {item.display_path for item in plan} == {"a_only.py"}


def test_amend_and_cherry_pick_stay_visible(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    _init_repo(project)
    (project / "base.txt").write_text("base\n", encoding="utf-8")
    _commit_all(project)
    # A commit on a side branch that this session will cherry-pick.
    _git(project, "checkout", "-q", "-b", "side")
    (project / "picked.py").write_text("picked\n", encoding="utf-8")
    _commit_all(project, "side work")
    _git(project, "checkout", "-q", "-")

    tracker = GitSessionDiffTracker.create(project)
    assert tracker is not None
    tracker.capture_baseline()

    (project / "amended.py").write_text("one\n", encoding="utf-8")
    _commit_all(project, "amend me")
    (project / "amended.py").write_text("one\ntwo\n", encoding="utf-8")
    _git(project, "add", ".")
    _git(project, "commit", "--amend", "--no-edit")
    _git(project, "cherry-pick", "side")

    changes = _by_path(tracker.refresh())

    assert set(changes) == {"amended.py", "picked.py"}
    assert changes["amended.py"].adds == 2


def test_initial_commit_mid_session_is_visible(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    _init_repo(project)

    tracker = GitSessionDiffTracker.create(project)
    assert tracker is not None
    tracker.capture_baseline()

    (project / "first.py").write_text("first\n", encoding="utf-8")
    _commit_all(project, "initial commit")

    changes = _by_path(tracker.refresh())

    assert set(changes) == {"first.py"}
    # Regression guard: `git diff-tree --root` without --no-commit-id emits the
    # commit sha as a bogus path.
    assert not any(len(path) == 40 and all(c in "0123456789abcdef" for c in path) for path in changes)


def test_git_dir_env_does_not_retarget_tracker(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    _init_repo(project)
    (project / "shared.txt").write_text("base\n", encoding="utf-8")
    _commit_all(project)
    worktree_a = _add_worktree(project, "a", "feat-a")

    # An exported GIT_DIR/GIT_WORK_TREE makes git ignore the working directory.
    monkeypatch.setenv("GIT_DIR", str(project / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(project))

    tracker = GitSessionDiffTracker.create(worktree_a)
    assert tracker is not None
    assert tracker.git_root == worktree_a.resolve()
    tracker.capture_baseline()

    (project / "main_only.py").write_text("main\n", encoding="utf-8")
    (worktree_a / "a_only.py").write_text("a\n", encoding="utf-8")

    assert set(_by_path(tracker.refresh())) == {"a_only.py"}


def test_sibling_worktree_paths_are_never_candidates(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    _init_repo(project)
    (project / "shared.txt").write_text("base\n", encoding="utf-8")
    _commit_all(project)
    # No exclude rule: the nested worktree is visible to the parent's git status.
    worktree_b = _add_worktree(project, "b", "feat-b", ignore=False)

    tracker = GitSessionDiffTracker.create(project)
    assert tracker is not None
    tracker.capture_baseline()

    (worktree_b / "b_only.py").write_text("b\n", encoding="utf-8")
    (worktree_b / "shared.txt").write_text("base\nfrom b\n", encoding="utf-8")
    (project / "mine.py").write_text("mine\n", encoding="utf-8")

    changes = _by_path(tracker.refresh(event_paths=[".kolega/worktrees/b/b_only.py"]))

    assert set(changes) == {"mine.py"}


def test_scope_reports_worktree_branch_and_history(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    _init_repo(project)
    (project / "shared.txt").write_text("base\n", encoding="utf-8")
    _commit_all(project)
    worktree_a = _add_worktree(project, "a", "feat-a")

    main_tracker = GitSessionDiffTracker.create(project)
    assert main_tracker is not None
    main_tracker.capture_baseline()
    main_scope = main_tracker.scope()

    assert main_scope.linked_worktree is False
    assert main_scope.root_label == ""
    assert main_scope.branch == "main"
    assert main_scope.history_moved is False
    assert main_scope.history_tracked is True

    tracker = GitSessionDiffTracker.create(worktree_a)
    assert tracker is not None
    tracker.capture_baseline()
    scope = tracker.scope()

    assert scope.linked_worktree is True
    assert scope.root_label == "a"
    assert scope.branch == "feat-a"
    assert scope.root_path == str(worktree_a.resolve())
    assert scope.history_moved is False

    # A plain commit is our own work, so the history did not "move".
    (worktree_a / "a_only.py").write_text("a\n", encoding="utf-8")
    _commit_all(worktree_a, "a work")
    assert tracker.scope().history_moved is False

    # Checking out somebody else's commit did not come from this session.
    (project / "foreign.py").write_text("foreign\n", encoding="utf-8")
    _commit_all(project, "foreign work")
    _git(worktree_a, "checkout", "-q", "--detach", "main")
    scope = tracker.scope()

    assert scope.history_moved is True
    assert scope.branch.startswith("detached at ")
    assert "foreign.py" not in _by_path(tracker.refresh())


def test_returning_head_to_the_baseline_hides_nothing(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    _init_repo(project)
    (project / "shared.txt").write_text("base\n", encoding="utf-8")
    _commit_all(project)

    tracker = GitSessionDiffTracker.create(project)
    assert tracker is not None
    tracker.capture_baseline()

    (project / "scratch.py").write_text("scratch\n", encoding="utf-8")
    _commit_all(project, "work")
    _git(project, "reset", "--hard", "HEAD~1")

    # HEAD and the tree are back at the baseline, so nothing is omitted and the
    # UI must not claim otherwise.
    assert tracker.refresh() == []
    assert tracker.scope().history_moved is False
