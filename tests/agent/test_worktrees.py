"""Tests for project-local agent worktree support."""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from kolega_code.memory.identity import resolve_git_common_dir
from kolega_code.worktrees import WORKTREE_EXCLUDE_RULE, ensure_worktree_dir_ignored, list_worktrees


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _git_init(repo: Path) -> None:
    repo.mkdir()
    _git(repo, "init", "-q")


def _exclude_path(repo: Path) -> Path:
    common_dir = resolve_git_common_dir(repo)
    assert common_dir is not None
    return common_dir / "info" / "exclude"


def test_setup_preserves_existing_content_and_adds_exact_rule(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _git_init(repo)
    exclude = _exclude_path(repo)
    exclude.write_bytes(b"*.local")

    assert ensure_worktree_dir_ignored(repo) is True

    content = exclude.read_text(encoding="utf-8")
    assert content.startswith("*.local\n")
    assert "# kolega-code-runtime\n" in content
    assert content.endswith(f"{WORKTREE_EXCLUDE_RULE}\n")
    assert content.splitlines().count(WORKTREE_EXCLUDE_RULE) == 1


def test_repeated_and_concurrent_setup_is_idempotent(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _git_init(repo)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: ensure_worktree_dir_ignored(repo), range(16)))

    assert all(results)
    lines = _exclude_path(repo).read_text(encoding="utf-8").splitlines()
    assert lines.count("# kolega-code-runtime") == 1
    assert lines.count(WORKTREE_EXCLUDE_RULE) == 1
    assert ensure_worktree_dir_ignored(repo) is True
    assert _exclude_path(repo).read_text(encoding="utf-8").splitlines().count(WORKTREE_EXCLUDE_RULE) == 1


def test_only_managed_worktree_subtree_is_ignored(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _git_init(repo)

    assert ensure_worktree_dir_ignored(repo) is True

    ignored = _git(repo, "check-ignore", "--no-index", "-q", ".kolega/worktrees/probe/file", check=False)
    trackable = _git(repo, "check-ignore", "--no-index", "-q", ".kolega/lsp.json", check=False)
    assert ignored.returncode == 0
    assert trackable.returncode == 1


def test_linked_worktree_updates_shared_common_exclude(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    linked = tmp_path / "linked"
    _git_init(repo)
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test User")
    (repo / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-qm", "initial")
    _git(repo, "worktree", "add", "-qb", "test-linked", str(linked))

    assert resolve_git_common_dir(linked) == resolve_git_common_dir(repo)
    assert ensure_worktree_dir_ignored(linked) is True
    assert WORKTREE_EXCLUDE_RULE in _exclude_path(repo).read_text(encoding="utf-8").splitlines()
    assert _git(linked, "check-ignore", "--no-index", "-q", ".kolega/worktrees/probe", check=False).returncode == 0


def test_non_git_path_is_unchanged(tmp_path: Path) -> None:
    assert ensure_worktree_dir_ignored(tmp_path) is False
    assert list(tmp_path.iterdir()) == []


def test_list_worktrees_reports_main_checkout_first_with_branches(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _git_init(repo)
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test User")
    (repo / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-qm", "initial")
    nested = repo / ".kolega" / "worktrees" / "a"
    _git(repo, "worktree", "add", "-qb", "feat-a", str(nested))

    from_main = list_worktrees(repo)
    from_linked = list_worktrees(nested)

    assert [info.path for info in from_main] == [repo.resolve(), nested.resolve()]
    assert [info.branch for info in from_main] == ["main", "feat-a"] or [info.branch for info in from_main] == [
        "master",
        "feat-a",
    ]
    assert all(info.head for info in from_main)
    # Every worktree of a repository sees the same list.
    assert from_linked == from_main


def test_list_worktrees_handles_detached_heads(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _git_init(repo)
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test User")
    (repo / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-qm", "initial")
    detached = tmp_path / "detached"
    _git(repo, "worktree", "add", "-q", "--detach", str(detached))

    infos = {info.path: info for info in list_worktrees(repo)}

    assert infos[detached.resolve()].branch == ""
    assert infos[detached.resolve()].head


def test_list_worktrees_on_a_non_repository_is_empty(tmp_path: Path) -> None:
    assert list_worktrees(tmp_path) == []


def test_git_common_directory_resolution_failure_is_non_fatal(tmp_path: Path, monkeypatch) -> None:
    def fail_resolution(path: Path | str) -> Path | None:
        del path
        raise OSError("git metadata unavailable")

    monkeypatch.setattr("kolega_code.worktrees.resolve_git_common_dir", fail_resolution)

    assert ensure_worktree_dir_ignored(tmp_path) is False


def test_git_metadata_write_failure_is_non_fatal(tmp_path: Path, monkeypatch) -> None:
    not_a_directory = tmp_path / "common"
    not_a_directory.write_text("blocked", encoding="utf-8")

    def resolve_to_file(path: Path | str) -> Path:
        del path
        return not_a_directory

    monkeypatch.setattr("kolega_code.worktrees.resolve_git_common_dir", resolve_to_file)

    assert ensure_worktree_dir_ignored(tmp_path) is False
