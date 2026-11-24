"""Stable project identity derivation."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from subprocess import run as run_subprocess

from kolega_code.git_env import git_env


@dataclass(frozen=True, slots=True)
class ProjectIdentity:
    kind: str
    identity: str
    display_path: str

    @property
    def directory_key(self) -> str:
        digest = hashlib.sha256(f"project-memory-v1\0{self.identity}".encode()).hexdigest()[:24]
        identity_path = Path(self.identity.partition(":")[2])
        if self.kind == "git-common-dir" and identity_path.name == ".git":
            identity_path = identity_path.parent
        name = identity_path.name or "project"
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-")[:40] or "project"
        return f"{slug}-{digest}"


def resolve_project_identity(project_path: Path | str) -> ProjectIdentity:
    project = Path(project_path).expanduser().resolve()
    common_dir = resolve_git_common_dir(project)
    if common_dir is not None:
        canonical = common_dir.resolve()
        return ProjectIdentity("git-common-dir", f"git:{canonical}", str(project))
    return ProjectIdentity("path", f"path:{project}", str(project))


def _could_discover_git_repository(path: Path) -> bool:
    """Whether Git's discovery walk could find a repository at or above ``path``.

    Git walks up to the first ``.git`` entry, stopping at a filesystem boundary
    unless ``GIT_DISCOVERY_ACROSS_FILESYSTEM`` allows crossing. Repeating that walk
    with ``stat`` calls lets a negative answer skip a process spawn; a positive
    one still asks Git, because only Git resolves worktrees, ``.git`` files and bare
    repositories.
    """
    try:
        anchor = path if path.is_dir() else path.parent
        device = anchor.stat().st_dev
    except OSError:
        return False
    across_filesystems = bool(os.environ.get("GIT_DISCOVERY_ACROSS_FILESYSTEM", "").strip())
    for candidate in (anchor, *anchor.parents):
        try:
            if (candidate / ".git").exists():
                return True
            # A bare repository has no `.git` entry, so only Git can answer for it.
            if candidate is anchor and (candidate / "HEAD").is_file() and (candidate / "objects").is_dir():
                return True
            candidate_device = candidate.stat().st_dev
        except OSError:
            return False
        if not across_filesystems and candidate_device != device:
            return False
        device = candidate_device
    return False


def _run_rev_parse(directory: Path, flag: str) -> str | None:
    try:
        result = run_subprocess(
            ["git", "-C", str(directory), "rev-parse", flag],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
            env=git_env(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode:
        return None
    value = result.stdout.removesuffix("\n").removesuffix("\r")
    if not value.strip() or "\n" in value or "\r" in value:
        return None
    return value


def _resolve_rev_parse_path(directory: Path, value: str) -> Path:
    resolved = Path(value)
    if not resolved.is_absolute():
        resolved = directory / resolved
    return resolved.resolve(strict=False)


def resolve_git_worktree_root(path: Path | str) -> Path | None:
    """Return the containing Git worktree root for an existing path or ancestor."""
    candidate = Path(path).expanduser().resolve(strict=False)
    while not candidate.exists() and candidate.parent != candidate:
        candidate = candidate.parent
    if candidate.is_file():
        candidate = candidate.parent
    if not _could_discover_git_repository(candidate):
        return None
    value = _run_rev_parse(candidate, "--show-toplevel")
    if value is None or value == "--show-toplevel":
        return None
    return _resolve_rev_parse_path(candidate, value)


def resolve_git_common_dir(path: Path | str) -> Path | None:
    """Return the containing repository's shared Git directory, if any.

    Linked worktrees resolve to the same common directory as their primary
    checkout. The returned path is absolute but is not required to exist.
    """
    project = Path(path).expanduser().resolve()
    if not project.is_dir() or not _could_discover_git_repository(project):
        return None
    value = _run_rev_parse(project, "--git-common-dir")
    if value == "--git-common-dir":
        # Older Git echoes unknown rev-parse flags verbatim with exit code 0.
        value = _run_rev_parse(project, "--git-dir")
        if value == "--git-dir":
            return None
    if value is None:
        return None
    return _resolve_rev_parse_path(project, value)
