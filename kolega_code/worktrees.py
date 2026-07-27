"""Project-local paths and clone-local Git exclusion for agent worktrees."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from subprocess import run as run_subprocess

from filelock import FileLock, Timeout

from kolega_code.git_env import git_env
from kolega_code.memory.identity import resolve_git_common_dir

WORKTREE_RELATIVE_DIR = Path(".kolega") / "worktrees"
WORKTREE_EXCLUDE_RULE = "/.kolega/worktrees/"

_EXCLUDE_MARKER = "# kolega-code-runtime"
_LOCK_TIMEOUT_SECONDS = 3
_LIST_TIMEOUT_SECONDS = 5


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
