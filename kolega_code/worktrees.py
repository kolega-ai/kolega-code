"""Project-local paths and clone-local Git exclusion for agent worktrees."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from subprocess import run as run_subprocess
from typing import Sequence

from filelock import FileLock, Timeout

from kolega_code.git_env import git_env
from kolega_code.memory.identity import resolve_git_common_dir

WORKTREE_RELATIVE_DIR = Path(".kolega") / "worktrees"
WORKTREE_EXCLUDE_RULE = "/.kolega/worktrees/"

_EXCLUDE_MARKER = "# kolega-code-runtime"
_LOCK_TIMEOUT_SECONDS = 3
_LIST_TIMEOUT_SECONDS = 5
_GIT_TIMEOUT_SECONDS = 10
_CREATE_TIMEOUT_SECONDS = 60


class WorktreeErrorCode(str, Enum):
    """Machine-readable reason for a worktree operation failure."""

    NOT_A_REPOSITORY = "not_a_repository"
    UNKNOWN_WORKTREE = "unknown_worktree"
    DIFFERENT_REPOSITORY = "different_repository"
    AMBIGUOUS_WORKTREE = "ambiguous_worktree"
    INVALID_BRANCH = "invalid_branch"
    INVALID_REF = "invalid_ref"
    BRANCH_CHECKED_OUT = "branch_checked_out"
    DESTINATION_REGISTERED = "destination_registered"
    DESTINATION_OCCUPIED = "destination_occupied"
    IGNORE_SETUP_FAILED = "ignore_setup_failed"
    GIT_FAILED = "git_failed"


class WorktreeError(RuntimeError):
    """Structured failure raised by strict worktree helpers."""

    def __init__(
        self,
        code: WorktreeErrorCode,
        message: str,
        *,
        worktrees: Sequence[WorktreeInfo] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.worktrees = tuple(worktrees)


def ensure_worktree_dir_ignored(project_path: Path | str) -> bool:
    """Best-effort ensure the managed worktree subtree is ignored by Git.

    The rule is stored in the clone-local shared ``info/exclude`` file rather
    than a tracked ``.gitignore``. Linked worktrees therefore share the rule
    while other ``.kolega`` project configuration remains trackable.

    Returns ``True`` when the rule is present or was added, and ``False`` when
    the path is not a Git repository or setup could not be completed.
    """
    try:
        common_dir = resolve_git_common_dir(project_path)
    except (OSError, RuntimeError):
        return False
    if common_dir is None:
        return False

    exclude_path = common_dir / "info" / "exclude"
    lock_path = exclude_path.with_name("exclude.kolega-code.lock")
    try:
        exclude_path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(lock_path), timeout=_LOCK_TIMEOUT_SECONDS, mode=0o600):
            try:
                content = exclude_path.read_bytes()
            except FileNotFoundError:
                content = b""

            rule = WORKTREE_EXCLUDE_RULE.encode("utf-8")
            if rule in content.splitlines():
                return True

            marker = _EXCLUDE_MARKER.encode("utf-8")
            addition = bytearray()
            if content and not content.endswith((b"\n", b"\r")):
                addition.extend(b"\n")
            if marker not in content.splitlines():
                addition.extend(marker)
                addition.extend(b"\n")
            addition.extend(rule)
            addition.extend(b"\n")

            with exclude_path.open("ab") as exclude_file:
                exclude_file.write(addition)
            return True
    except (OSError, Timeout):
        return False


@dataclass(frozen=True)
class WorktreeInfo:
    """One checkout registered with ``git worktree``."""

    path: Path  # absolute, resolved
    branch: str  # short branch name, "" when detached
    head: str  # commit sha, "" when unborn


def resolve_worktree(source_path: Path | str, target: Path | str) -> WorktreeInfo:
    """Resolve ``target`` to a registered worktree in ``source_path``'s repository.

    A target may be an exact short local branch name or a canonical absolute /
    source-worktree-relative path. Existing nested paths resolve to their
    containing registered worktree.
    """
    worktrees = _list_worktrees_strict(source_path)
    source = _canonical_existing_path(source_path)
    source_worktree = _containing_worktree(source, worktrees)
    if source_worktree is None:
        raise WorktreeError(
            WorktreeErrorCode.UNKNOWN_WORKTREE,
            _with_registered(f"Source path is not inside a registered worktree: {source}", worktrees),
            worktrees=worktrees,
        )

    target_text = str(target)
    matches: dict[Path, WorktreeInfo] = {}
    for info in worktrees:
        if info.branch and info.branch == target_text:
            matches[info.path] = info

    raw_target = Path(target).expanduser()
    path_target = raw_target if raw_target.is_absolute() else source_worktree.path / raw_target
    if path_target.exists():
        canonical_target = path_target.resolve(strict=True)
        repository_probe = canonical_target.parent if canonical_target.is_file() else canonical_target
        target_common_dir = resolve_git_common_dir(repository_probe)
        source_common_dir = resolve_git_common_dir(source_worktree.path)
        if (
            target_common_dir is not None
            and source_common_dir is not None
            and target_common_dir.resolve(strict=False) != source_common_dir.resolve(strict=False)
        ):
            raise WorktreeError(
                WorktreeErrorCode.DIFFERENT_REPOSITORY,
                _with_registered(
                    f"Worktree target belongs to a different repository: {canonical_target}",
                    worktrees,
                ),
                worktrees=worktrees,
            )
        path_match = _containing_worktree(canonical_target, worktrees)
        if path_match is not None:
            matches[path_match.path] = path_match

    if len(matches) == 1:
        match = next(iter(matches.values()))
        if not _worktree_is_available(match, source_worktree):
            raise WorktreeError(
                WorktreeErrorCode.UNKNOWN_WORKTREE,
                _with_registered(f"Registered worktree is missing or invalid: {match.path}", worktrees),
                worktrees=worktrees,
            )
        return match
    if len(matches) > 1:
        raise WorktreeError(
            WorktreeErrorCode.AMBIGUOUS_WORKTREE,
            _with_registered(f"Worktree target is ambiguous: {target_text}", worktrees),
            worktrees=worktrees,
        )
    raise WorktreeError(
        WorktreeErrorCode.UNKNOWN_WORKTREE,
        _with_registered(f"No registered worktree matches: {target_text}", worktrees),
        worktrees=worktrees,
    )


def validate_branch_name(source_path: Path | str, branch: str) -> str:
    """Validate and return an exact short local branch name."""
    if not branch or branch != branch.strip() or branch.startswith("-"):
        raise WorktreeError(WorktreeErrorCode.INVALID_BRANCH, f"Invalid branch name: {branch!r}")
    completed = _run_git(source_path, ["check-ref-format", "--branch", branch])
    if completed.returncode:
        detail = _git_error_detail(completed)
        raise WorktreeError(
            WorktreeErrorCode.INVALID_BRANCH,
            f"Invalid branch name {branch!r}: {detail}",
        )
    return branch


def resolve_commit_ref(source_path: Path | str, ref: str) -> str:
    """Resolve ``ref`` to a commit SHA or raise ``INVALID_REF``."""
    if not ref or ref != ref.strip():
        raise WorktreeError(WorktreeErrorCode.INVALID_REF, f"Invalid start ref: {ref!r}")
    completed = _run_git(
        source_path,
        ["rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"],
    )
    if completed.returncode:
        detail = _git_error_detail(completed)
        raise WorktreeError(WorktreeErrorCode.INVALID_REF, f"Invalid start ref {ref!r}: {detail}")
    commit = completed.stdout.strip()
    if not commit:
        raise WorktreeError(WorktreeErrorCode.INVALID_REF, f"Invalid start ref: {ref!r}")
    return commit


def branch_worktree_slug(branch: str) -> str:
    """Return a filesystem-safe slug derived from the complete branch name."""
    slug = "".join(character if character.isalnum() or character in "._-" else "-" for character in branch)
    slug = re.sub(r"-+", "-", slug).strip("._-")
    return slug or "branch"


def create_worktree(
    source_path: Path | str,
    branch: str,
    *,
    start_ref: str | None = None,
    destination: Path | str | None = None,
) -> WorktreeInfo:
    """Create and return a registered worktree without force or reuse behavior.

    Existing unoccupied local branches are checked out as-is. A missing branch
    is created from ``start_ref``, or from the source checkout's ``HEAD`` when
    no start ref is supplied.
    """
    worktrees = _list_worktrees_strict(source_path)
    source = _canonical_existing_path(source_path)
    source_worktree = _containing_worktree(source, worktrees)
    if source_worktree is None:
        raise WorktreeError(
            WorktreeErrorCode.UNKNOWN_WORKTREE,
            _with_registered(f"Source path is not inside a registered worktree: {source}", worktrees),
            worktrees=worktrees,
        )
    branch = validate_branch_name(source_worktree.path, branch)

    checked_out = [info for info in worktrees if info.branch == branch]
    if checked_out:
        checked_out_at = ", ".join(str(info.path) for info in checked_out)
        raise WorktreeError(
            WorktreeErrorCode.BRANCH_CHECKED_OUT,
            f"Branch {branch!r} is already checked out at {checked_out_at}; select it as an existing worktree.",
            worktrees=worktrees,
        )

    branch_ref = f"refs/heads/{branch}"
    branch_probe = _run_git(source_worktree.path, ["show-ref", "--verify", "--quiet", branch_ref])
    if branch_probe.returncode not in (0, 1):
        raise WorktreeError(
            WorktreeErrorCode.GIT_FAILED,
            f"Could not inspect local branch {branch!r}: {_git_error_detail(branch_probe)}",
            worktrees=worktrees,
        )
    branch_exists = branch_probe.returncode == 0
    start_commit = ""
    if not branch_exists:
        start_commit = resolve_commit_ref(source_worktree.path, start_ref or "HEAD")

    main_worktree = worktrees[0]
    if destination is None:
        target = main_worktree.path / WORKTREE_RELATIVE_DIR / branch_worktree_slug(branch)
    else:
        raw_destination = Path(destination).expanduser()
        target = raw_destination if raw_destination.is_absolute() else source_worktree.path / raw_destination
    target = target.resolve(strict=False)

    registered_target = next((info for info in worktrees if info.path == target), None)
    if registered_target is not None:
        raise WorktreeError(
            WorktreeErrorCode.DESTINATION_REGISTERED,
            f"Worktree destination is already registered: {target}",
            worktrees=worktrees,
        )
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise WorktreeError(
            WorktreeErrorCode.DESTINATION_OCCUPIED,
            f"Worktree destination must be absent or an empty directory: {target}",
            worktrees=worktrees,
        )
    _reject_other_repository_destination(source_worktree.path, target, worktrees)

    if not ensure_worktree_dir_ignored(source_worktree.path):
        raise WorktreeError(
            WorktreeErrorCode.IGNORE_SETUP_FAILED,
            "Could not configure the clone-local ignore rule for managed worktrees.",
            worktrees=worktrees,
        )

    if branch_exists:
        arguments = ["worktree", "add", "--", str(target), branch]
    else:
        arguments = ["worktree", "add", "-b", branch, "--", str(target), start_commit]
    completed = _run_git(source_worktree.path, arguments, timeout=_CREATE_TIMEOUT_SECONDS)
    if completed.returncode:
        raise WorktreeError(
            WorktreeErrorCode.GIT_FAILED,
            f"Git could not create worktree {target}: {_git_error_detail(completed)}",
            worktrees=worktrees,
        )

    created = _list_worktrees_strict(source_worktree.path)
    result = next((info for info in created if info.path == target and info.branch == branch), None)
    if result is None:
        raise WorktreeError(
            WorktreeErrorCode.GIT_FAILED,
            f"Git completed but did not register the expected worktree at {target}.",
            worktrees=created,
        )
    return result


def list_worktrees(cwd: Path | str) -> list[WorktreeInfo]:
    """Return every worktree of the repository containing ``cwd``, main checkout first.

    Returns an empty list when ``cwd`` is not a Git repository or Git cannot be
    run; callers treat the empty list as "no sibling worktrees are known".
    """
    try:
        completed = run_subprocess(
            ["git", "worktree", "list", "--porcelain"],
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
            timeout=_LIST_TIMEOUT_SECONDS,
            env=git_env(),
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return []
    if completed.returncode:
        return []
    return _parse_worktree_porcelain(completed.stdout)


def _list_worktrees_strict(cwd: Path | str) -> list[WorktreeInfo]:
    completed = _run_git(cwd, ["worktree", "list", "--porcelain"], timeout=_LIST_TIMEOUT_SECONDS)
    if completed.returncode:
        raise WorktreeError(
            WorktreeErrorCode.NOT_A_REPOSITORY,
            f"Path is not in a Git worktree: {Path(cwd).expanduser().resolve(strict=False)} "
            f"({_git_error_detail(completed)})",
        )
    worktrees = _parse_worktree_porcelain(completed.stdout)
    if not worktrees:
        raise WorktreeError(
            WorktreeErrorCode.NOT_A_REPOSITORY,
            f"Git reported no registered worktrees for: {Path(cwd).expanduser().resolve(strict=False)}",
        )
    return worktrees


def _run_git(
    cwd: Path | str,
    arguments: Sequence[str],
    *,
    timeout: int = _GIT_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    try:
        return run_subprocess(
            ["git", *arguments],
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=git_env(),
        )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise WorktreeError(WorktreeErrorCode.GIT_FAILED, f"Could not run Git: {exc}") from exc


def _canonical_existing_path(path: Path | str) -> Path:
    try:
        return Path(path).expanduser().resolve(strict=True)
    except OSError as exc:
        raise WorktreeError(
            WorktreeErrorCode.UNKNOWN_WORKTREE,
            f"Worktree path does not exist: {Path(path).expanduser()}",
        ) from exc


def _containing_worktree(path: Path, worktrees: Sequence[WorktreeInfo]) -> WorktreeInfo | None:
    matches = [info for info in worktrees if path == info.path or path.is_relative_to(info.path)]
    if not matches:
        return None
    return max(matches, key=lambda info: len(info.path.parts))


def _worktree_is_available(worktree: WorktreeInfo, source_worktree: WorktreeInfo) -> bool:
    if not worktree.path.is_dir():
        return False
    completed = _run_git(worktree.path, ["rev-parse", "--show-toplevel"])
    if completed.returncode or not completed.stdout.strip():
        return False
    reported_root = Path(completed.stdout.strip()).resolve(strict=False)
    if reported_root != worktree.path:
        return False
    target_common_dir = resolve_git_common_dir(worktree.path)
    source_common_dir = resolve_git_common_dir(source_worktree.path)
    return (
        target_common_dir is not None
        and source_common_dir is not None
        and target_common_dir.resolve(strict=False) == source_common_dir.resolve(strict=False)
    )


def _reject_other_repository_destination(
    source_worktree: Path,
    target: Path,
    worktrees: Sequence[WorktreeInfo],
) -> None:
    probe = target
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if not probe.exists():
        return
    target_common_dir = resolve_git_common_dir(probe)
    source_common_dir = resolve_git_common_dir(source_worktree)
    if (
        target_common_dir is not None
        and source_common_dir is not None
        and target_common_dir.resolve(strict=False) != source_common_dir.resolve(strict=False)
    ):
        raise WorktreeError(
            WorktreeErrorCode.DIFFERENT_REPOSITORY,
            f"Worktree destination is inside a different repository: {target}",
            worktrees=worktrees,
        )


def _git_error_detail(completed: subprocess.CompletedProcess[str]) -> str:
    return completed.stderr.strip() or completed.stdout.strip() or f"Git exited with status {completed.returncode}"


def _with_registered(message: str, worktrees: Sequence[WorktreeInfo]) -> str:
    entries = ", ".join(f"{info.path} [{info.branch or 'detached'}]" for info in worktrees)
    return f"{message}. Registered worktrees: {entries}"


def _parse_worktree_porcelain(output: str) -> list[WorktreeInfo]:
    """Parse ``git worktree list --porcelain`` blocks separated by blank lines."""
    worktrees: list[WorktreeInfo] = []
    path = ""
    branch = ""
    head = ""

    def flush() -> None:
        nonlocal path, branch, head
        if path:
            worktrees.append(
                WorktreeInfo(path=Path(path).resolve(strict=False), branch=branch, head=head),
            )
        path, branch, head = "", "", ""

    for raw_line in output.splitlines():
        line = raw_line.rstrip("\n")
        if not line.strip():
            flush()
            continue
        key, _, value = line.partition(" ")
        if key == "worktree":
            flush()
            path = value
        elif key == "HEAD":
            head = value
        elif key == "branch":
            branch = value.removeprefix("refs/heads/")
    flush()
    return worktrees
