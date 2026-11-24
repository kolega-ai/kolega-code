"""Repository discovery and version lookups the CLI performs on every startup.

Both helpers are on the per-mount path: the TUI resolves its project identity while
building the agent, and renders the package version in the startup title. Neither
may reach a subprocess or rescan installed metadata when the answer is already
knowable locally.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from kolega_code.cli import updater
from kolega_code.git_env import git_env
from kolega_code.memory.identity import (
    resolve_git_common_dir,
    resolve_git_worktree_root,
    resolve_project_identity,
)


pytestmark = pytest.mark.usefixtures("hermetic_git_config")


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
        env=git_env(),
    )


def _git_init(repo: Path) -> None:
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")


def _forbid_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("repository discovery spawned git for a non-repository path")

    monkeypatch.setattr("kolega_code.memory.identity.run_subprocess", fail)


def test_plain_directory_needs_no_git_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _forbid_subprocess(monkeypatch)
    project = tmp_path / "project"
    project.mkdir()

    assert resolve_git_common_dir(project) is None
    assert resolve_project_identity(project).kind == "path"
    assert resolve_git_worktree_root(project) is None


def test_nonexistent_and_file_paths_need_no_git_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _forbid_subprocess(monkeypatch)
    plain_file = tmp_path / "notes.txt"
    plain_file.write_text("hello\n", encoding="utf-8")

    # `git -C` fails for both shapes, so the answer is "no repository" either way.
    assert resolve_git_common_dir(tmp_path / "missing") is None
    assert resolve_git_common_dir(plain_file) is None


def test_repository_still_resolves_through_git(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _git_init(repo)
    nested = repo / "src" / "pkg"
    nested.mkdir(parents=True)

    common_dir = resolve_git_common_dir(nested)

    assert common_dir is not None
    assert common_dir.resolve() == (repo / ".git").resolve()
    assert resolve_project_identity(nested).kind == "git-common-dir"
    assert resolve_git_worktree_root(nested) == repo.resolve()


def test_bare_repository_still_resolves_through_git(tmp_path: Path) -> None:
    bare = tmp_path / "bare.git"
    subprocess.run(
        ["git", "init", "-q", "--bare", str(bare)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
        env=git_env(),
    )

    common_dir = resolve_git_common_dir(bare)

    assert common_dir is not None
    assert common_dir.resolve() == bare.resolve()


def test_linked_worktree_shares_the_primary_common_dir(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _git_init(repo)
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test User")
    (repo / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-qm", "initial")
    linked = tmp_path / "linked"
    _git(repo, "worktree", "add", "-q", "-b", "feature", str(linked))

    assert resolve_git_common_dir(linked) == resolve_git_common_dir(repo)


def test_current_version_is_read_once_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def version(name: str) -> str:
        calls.append(name)
        return "9.9.9"

    updater.current_version.cache_clear()
    monkeypatch.setattr(updater.metadata, "version", version)
    try:
        assert updater.current_version() == "9.9.9"
        assert updater.current_version() == "9.9.9"
        assert calls == [updater.PACKAGE_NAME]
    finally:
        updater.current_version.cache_clear()
