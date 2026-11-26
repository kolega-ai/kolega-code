"""Tests for Git-backed project identity and worktree path resolution."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

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


def _init_repo_with_commit(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test User")
    (repo / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-qm", "initial")


def _make_pre_231_git_runner() -> Any:
    """Emulate Git < 2.31 echoing unknown ``--path-format=...`` flags with exit 0."""
    real_run = subprocess.run

    def emulate(command: list[str], *args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if "rev-parse" in command:
            unknown = [arg for arg in command if arg.startswith("--path-format")]
            if unknown:
                stripped = [arg for arg in command if not arg.startswith("--path-format")]
                completed = real_run(stripped, *args, **kwargs)
                echoed = "\n".join(unknown) + "\n" + completed.stdout
                return subprocess.CompletedProcess(
                    command,
                    completed.returncode,
                    stdout=echoed,
                    stderr=completed.stderr,
                )
        return real_run(command, *args, **kwargs)

    return emulate


def test_rev_parse_never_passes_path_format_and_runs_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    _init_repo_with_commit(repo)
    commands: list[list[str]] = []
    real_run = subprocess.run

    def spy(command: list[str], *args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(list(command))
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr("kolega_code.memory.identity.run_subprocess", spy)

    assert resolve_git_common_dir(repo) == (repo / ".git").resolve()
    assert len(commands) == 1
    assert not any(arg.startswith("--path-format") for arg in commands[0])

    commands.clear()
    assert resolve_git_worktree_root(repo) == repo.resolve()
    assert len(commands) == 1
    assert not any(arg.startswith("--path-format") for arg in commands[0])


def test_pre_231_git_emulator_resolves_repo_subdir_and_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    subdir = repo / "src" / "pkg"
    linked = tmp_path / "linked"
    _init_repo_with_commit(repo)
    subdir.mkdir(parents=True)
    _git(repo, "worktree", "add", "-qb", "topic", str(linked))

    monkeypatch.setattr("kolega_code.memory.identity.run_subprocess", _make_pre_231_git_runner())

    expected_common = (repo / ".git").resolve()
    assert resolve_git_common_dir(repo) == expected_common
    assert resolve_git_common_dir(subdir) == expected_common
    assert resolve_git_common_dir(linked) == expected_common
    assert resolve_git_worktree_root(repo) == repo.resolve()
    assert resolve_git_worktree_root(subdir) == repo.resolve()
    assert resolve_git_worktree_root(linked) == linked.resolve()

    repo_identity = resolve_project_identity(repo)
    assert resolve_project_identity(subdir).identity == repo_identity.identity
    assert resolve_project_identity(linked).identity == repo_identity.identity


@pytest.mark.parametrize(
    "stdout",
    [
        "--path-format=absolute\n.git\n",
        "line-one\nline-two\n",
        "",
        "\n",
        "   \n",
        "--show-toplevel\n",
    ],
)
def test_malformed_or_echoed_rev_parse_output_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stdout: str
) -> None:
    def fake_run(command: list[str], *args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("kolega_code.memory.identity.run_subprocess", fake_run)

    assert resolve_git_worktree_root(tmp_path) is None
    if stdout != "--show-toplevel\n":
        assert resolve_git_common_dir(tmp_path) is None


@pytest.mark.parametrize(
    ("subdir_parts", "git_dir_stdout"),
    [
        ((), ".git\n"),
        (("a", "b"), "../../.git\n"),
    ],
)
def test_git_common_dir_falls_back_to_git_dir_when_flag_is_echoed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    subdir_parts: tuple[str, ...],
    git_dir_stdout: str,
) -> None:
    repo = tmp_path / "repo"
    target = repo.joinpath(*subdir_parts)
    target.mkdir(parents=True)
    seen_flags: list[str] = []

    def fake_run(command: list[str], *args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        flag = command[-1]
        seen_flags.append(flag)
        if flag == "--git-common-dir":
            return subprocess.CompletedProcess(command, 0, stdout="--git-common-dir\n", stderr="")
        if flag == "--git-dir":
            return subprocess.CompletedProcess(command, 0, stdout=git_dir_stdout, stderr="")
        raise AssertionError(f"Unexpected flag: {flag}")

    monkeypatch.setattr("kolega_code.memory.identity.run_subprocess", fake_run)

    assert resolve_git_common_dir(target) == (repo / ".git").resolve(strict=False)
    assert seen_flags == ["--git-common-dir", "--git-dir"]


@pytest.mark.parametrize(
    ("git_dir_returncode", "git_dir_stdout"),
    [
        (0, "--git-dir\n"),
        (128, ""),
    ],
)
def test_git_common_dir_returns_none_when_fallback_echoes_or_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    git_dir_returncode: int,
    git_dir_stdout: str,
) -> None:
    def fake_run(command: list[str], *args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        if command[-1] == "--git-common-dir":
            return subprocess.CompletedProcess(command, 0, stdout="--git-common-dir\n", stderr="")
        return subprocess.CompletedProcess(command, git_dir_returncode, stdout=git_dir_stdout, stderr="")

    monkeypatch.setattr("kolega_code.memory.identity.run_subprocess", fake_run)

    assert resolve_git_common_dir(tmp_path) is None


def test_real_git_layouts_resolve_canonical_paths_and_identities(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    subdir = repo / "a" / "b"
    linked = tmp_path / "linked"
    linked_subdir = linked / "a" / "b"
    repo_link = tmp_path / "repo-link"
    subdir_link = tmp_path / "subdir-link"
    bare = tmp_path / "bare.git"
    outside = tmp_path / "outside"

    _init_repo_with_commit(repo)
    subdir.mkdir(parents=True)
    _git(repo, "worktree", "add", "-qb", "linked-branch", str(linked))
    linked_subdir.mkdir(parents=True)
    repo_link.symlink_to(repo, target_is_directory=True)
    subdir_link.symlink_to(subdir, target_is_directory=True)
    subprocess.run(["git", "init", "--bare", "-q", str(bare)], check=True, env=git_env())
    outside.mkdir()

    expected_common = (repo / ".git").resolve()
    expected_identity = resolve_project_identity(repo)
    for probe in (repo, subdir, repo_link, subdir_link, linked, linked_subdir):
        common = resolve_git_common_dir(probe)
        assert common is not None
        assert common == expected_common
        assert ".." not in common.parts
        assert resolve_project_identity(probe).identity == expected_identity.identity
        assert resolve_project_identity(probe).directory_key == expected_identity.directory_key

    assert resolve_git_common_dir(bare) == bare.resolve()
    assert resolve_git_worktree_root(bare) is None

    assert resolve_git_common_dir(outside) is None
    assert resolve_git_worktree_root(outside) is None

    assert resolve_git_worktree_root(repo) == repo.resolve()
    assert resolve_git_worktree_root(subdir) == repo.resolve()
    assert resolve_git_worktree_root(repo / "tracked.txt") == repo.resolve()
    assert resolve_git_worktree_root(repo / "missing" / "nested" / "file.txt") == repo.resolve()
    assert resolve_git_worktree_root(linked_subdir) == linked.resolve()


@pytest.mark.parametrize(
    "error",
    [
        OSError("git not found"),
        subprocess.TimeoutExpired(cmd=["git"], timeout=3),
    ],
)
def test_subprocess_errors_return_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    def fail(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        raise error

    monkeypatch.setattr("kolega_code.memory.identity.run_subprocess", fail)

    assert resolve_git_common_dir(tmp_path) is None
    assert resolve_git_worktree_root(tmp_path) is None
