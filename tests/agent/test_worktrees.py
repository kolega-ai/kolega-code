"""Tests for project-local agent worktree support."""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from kolega_code.git_env import git_env
from kolega_code.memory.identity import resolve_git_common_dir
from kolega_code.worktrees import (
    WORKTREE_EXCLUDE_RULE,
    WorktreeError,
    WorktreeErrorCode,
    branch_worktree_slug,
    create_worktree,
    ensure_worktree_dir_ignored,
    list_worktrees,
    resolve_commit_ref,
    resolve_worktree,
    validate_branch_name,
)


pytestmark = pytest.mark.usefixtures("hermetic_git_config")


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check,
        capture_output=True,
        text=True,
        timeout=10,
        env=git_env(),
    )


def _git_init(repo: Path) -> None:
    repo.mkdir()
    # Pin the initial branch: without a global init.defaultBranch (as on CI) git
    # would name it "master" and branch assertions would depend on the machine.
    _git(repo, "init", "-q", "-b", "main")


def _commit_initial(repo: Path) -> str:
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test User")
    (repo / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-qm", "initial")
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


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
    assert [info.branch for info in from_main] == ["main", "feat-a"]
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


def test_resolve_registered_worktree_by_absolute_relative_nested_and_branch(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    linked = tmp_path / "linked"
    _git_init(repo)
    _commit_initial(repo)
    _git(repo, "worktree", "add", "-qb", "fix/image-history", str(linked))
    nested = linked / "src" / "package"
    nested.mkdir(parents=True)

    assert resolve_worktree(repo, linked).path == linked.resolve()
    assert resolve_worktree(repo, Path("..") / "linked").path == linked.resolve()
    assert resolve_worktree(repo, Path("..") / "linked" / "src" / "package").path == linked.resolve()
    assert resolve_worktree(repo, "fix/image-history").path == linked.resolve()
    assert resolve_worktree(nested, ".").path == linked.resolve()


def test_resolve_rejects_unknown_detached_and_different_repository_targets(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    detached = tmp_path / "detached"
    other = tmp_path / "other"
    nested_other = repo / "nested-other"
    _git_init(repo)
    _commit_initial(repo)
    _git(repo, "worktree", "add", "-q", "--detach", str(detached))
    _git_init(other)
    _git_init(nested_other)

    with pytest.raises(WorktreeError) as unknown:
        resolve_worktree(repo, "missing")
    assert unknown.value.code is WorktreeErrorCode.UNKNOWN_WORKTREE
    assert str(repo.resolve()) in str(unknown.value)
    assert "main" in str(unknown.value)

    with pytest.raises(WorktreeError) as detached_by_name:
        resolve_worktree(repo, "detached")
    assert detached_by_name.value.code is WorktreeErrorCode.UNKNOWN_WORKTREE

    with pytest.raises(WorktreeError) as different:
        resolve_worktree(repo, other)
    assert different.value.code is WorktreeErrorCode.DIFFERENT_REPOSITORY

    with pytest.raises(WorktreeError) as nested_different:
        resolve_worktree(repo, nested_other)
    assert nested_different.value.code is WorktreeErrorCode.DIFFERENT_REPOSITORY


def test_resolve_rejects_registered_worktree_whose_checkout_is_missing(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    linked = tmp_path / "linked"
    moved = tmp_path / "moved"
    _git_init(repo)
    _commit_initial(repo)
    _git(repo, "worktree", "add", "-qb", "stale-branch", str(linked))
    linked.rename(moved)

    with pytest.raises(WorktreeError) as error:
        resolve_worktree(repo, "stale-branch")

    assert error.value.code is WorktreeErrorCode.UNKNOWN_WORKTREE
    assert "missing or invalid" in str(error.value)


def test_resolve_rejects_ambiguous_branch_and_path(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    linked = tmp_path / "linked"
    _git_init(repo)
    _commit_initial(repo)
    _git(repo, "worktree", "add", "-qb", "feature", str(linked))
    (repo / "feature").mkdir()

    with pytest.raises(WorktreeError) as error:
        resolve_worktree(repo, "feature")

    assert error.value.code is WorktreeErrorCode.AMBIGUOUS_WORKTREE


def test_validation_uses_git_branch_and_commit_rules(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _git_init(repo)
    initial = _commit_initial(repo)

    assert validate_branch_name(repo, "fix/nested-name") == "fix/nested-name"
    assert resolve_commit_ref(repo, "HEAD") == initial
    with pytest.raises(WorktreeError) as invalid_branch:
        validate_branch_name(repo, "bad..branch")
    assert invalid_branch.value.code is WorktreeErrorCode.INVALID_BRANCH
    with pytest.raises(WorktreeError) as invalid_ref:
        resolve_commit_ref(repo, "refs/heads/not-present")
    assert invalid_ref.value.code is WorktreeErrorCode.INVALID_REF


def test_create_new_branch_from_head_at_managed_destination(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _git_init(repo)
    initial = _commit_initial(repo)

    created = create_worktree(repo, "fix/image-history")

    expected = repo / ".kolega" / "worktrees" / "fix-image-history"
    assert created.path == expected.resolve()
    assert created.branch == "fix/image-history"
    assert created.head == initial
    assert _git(created.path, "branch", "--show-current").stdout.strip() == "fix/image-history"
    assert WORKTREE_EXCLUDE_RULE in _exclude_path(repo).read_text(encoding="utf-8").splitlines()


def test_create_new_branch_from_explicit_start_ref(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    destination = tmp_path / "from-initial"
    _git_init(repo)
    initial = _commit_initial(repo)
    (repo / "tracked.txt").write_text("second\n", encoding="utf-8")
    _git(repo, "commit", "-qam", "second")

    created = create_worktree(repo, "topic/from-initial", start_ref=initial, destination=destination)

    assert created.path == destination.resolve()
    assert created.head == initial
    assert (destination / "tracked.txt").read_text(encoding="utf-8") == "tracked\n"


def test_create_checks_out_existing_unoccupied_local_branch(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    destination = tmp_path / "existing"
    _git_init(repo)
    initial = _commit_initial(repo)
    _git(repo, "branch", "ready")

    created = create_worktree(repo, "ready", destination=destination)

    assert created.branch == "ready"
    assert created.head == initial
    assert _git(destination, "branch", "--show-current").stdout.strip() == "ready"


def test_create_accepts_an_existing_empty_destination(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    destination = tmp_path / "empty"
    _git_init(repo)
    _commit_initial(repo)
    destination.mkdir()

    created = create_worktree(repo, "empty-destination", destination=destination)

    assert created.path == destination.resolve()
    assert (destination / ".git").is_file()


def test_create_rejects_checked_out_branch(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _git_init(repo)
    _commit_initial(repo)

    with pytest.raises(WorktreeError) as checked_out:
        create_worktree(repo, "main")
    assert checked_out.value.code is WorktreeErrorCode.BRANCH_CHECKED_OUT
    assert "existing worktree" in str(checked_out.value)


def test_existing_branch_does_not_move_when_start_ref_is_supplied(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _git_init(repo)
    initial = _commit_initial(repo)
    _git(repo, "branch", "ready")
    (repo / "tracked.txt").write_text("second\n", encoding="utf-8")
    _git(repo, "commit", "-qam", "second")

    created = create_worktree(repo, "ready", start_ref="HEAD")

    assert created.head == initial


def test_create_validates_before_mutating_worktree_state(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _git_init(repo)
    _commit_initial(repo)
    before = list_worktrees(repo)

    with pytest.raises(WorktreeError) as invalid_name:
        create_worktree(repo, "invalid name")
    assert invalid_name.value.code is WorktreeErrorCode.INVALID_BRANCH

    with pytest.raises(WorktreeError) as invalid_ref:
        create_worktree(repo, "new-branch", start_ref="not-a-ref")
    assert invalid_ref.value.code is WorktreeErrorCode.INVALID_REF
    assert list_worktrees(repo) == before
    assert _git(repo, "show-ref", "--verify", "--quiet", "refs/heads/new-branch", check=False).returncode == 1


def test_create_rejects_occupied_registered_and_slug_collision_destinations(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _git_init(repo)
    _commit_initial(repo)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("keep\n", encoding="utf-8")

    with pytest.raises(WorktreeError) as occupied_error:
        create_worktree(repo, "occupied-target", destination=occupied)
    assert occupied_error.value.code is WorktreeErrorCode.DESTINATION_OCCUPIED
    assert (occupied / "keep.txt").read_text(encoding="utf-8") == "keep\n"

    first = create_worktree(repo, "feat-a")
    with pytest.raises(WorktreeError) as registered:
        create_worktree(repo, "other", destination=first.path)
    assert registered.value.code is WorktreeErrorCode.DESTINATION_REGISTERED

    with pytest.raises(WorktreeError) as collision:
        create_worktree(repo, "feat/a")
    assert collision.value.code is WorktreeErrorCode.DESTINATION_REGISTERED
    assert branch_worktree_slug("feat/a") == branch_worktree_slug("feat-a") == "feat-a"
    assert _git(repo, "show-ref", "--verify", "--quiet", "refs/heads/feat/a", check=False).returncode == 1


def test_default_creation_from_linked_checkout_uses_main_checkout(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    linked = tmp_path / "linked"
    _git_init(repo)
    _commit_initial(repo)
    _git(repo, "worktree", "add", "-qb", "linked-source", str(linked))

    created = create_worktree(linked, "created/from-linked")

    assert created.path == (repo / ".kolega" / "worktrees" / "created-from-linked").resolve()


def test_create_rejects_destination_inside_another_repository(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    other = tmp_path / "other"
    _git_init(repo)
    _commit_initial(repo)
    _git_init(other)
    destination = other / "nested-worktree"

    with pytest.raises(WorktreeError) as error:
        create_worktree(repo, "other-repo-target", destination=destination)

    assert error.value.code is WorktreeErrorCode.DIFFERENT_REPOSITORY
    assert not destination.exists()
