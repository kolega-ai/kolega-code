"""Stable project identity derivation."""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from subprocess import run as run_subprocess


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
    common_dir = _git_common_dir(project)
    if common_dir is not None:
        canonical = common_dir.resolve()
        return ProjectIdentity("git-common-dir", f"git:{canonical}", str(project))
    return ProjectIdentity("path", f"path:{project}", str(project))


def resolve_git_worktree_root(path: Path | str) -> Path | None:
    """Return the containing Git worktree root for an existing path or ancestor."""
    candidate = Path(path).expanduser().resolve(strict=False)
    while not candidate.exists() and candidate.parent != candidate:
        candidate = candidate.parent
    if candidate.is_file():
        candidate = candidate.parent
    commands = (
        ["git", "-C", str(candidate), "rev-parse", "--path-format=absolute", "--show-toplevel"],
        ["git", "-C", str(candidate), "rev-parse", "--show-toplevel"],
    )
    for command in commands:
        try:
            result = run_subprocess(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode:
            continue
        value = result.stdout.strip()
        if value:
            return Path(value).resolve(strict=False)
    return None


def _git_common_dir(project: Path) -> Path | None:
    commands = (
        ["git", "-C", str(project), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        ["git", "-C", str(project), "rev-parse", "--git-common-dir"],
    )
    for command in commands:
        try:
            result = run_subprocess(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode:
            continue
        value = result.stdout.strip()
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = project / path
        return path
    return None
