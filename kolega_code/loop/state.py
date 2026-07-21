"""WorkLog — persistent state for autonomous loop engineering.

Manages work-log.json with atomic saves, attempt limit enforcement,
branch-based safe revert, and anti-pattern recording.
"""

import copy
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


try:
    from shlex import quote as shlex_quote
except ImportError:

    def shlex_quote(s: str) -> str:
        if " " in s:
            return f"'{s}'"
        return s


class LoopLimitExceeded(Exception):
    """Raised when attempt counter exceeds max_attempts."""

    def __init__(self, attempts_made: int, max_attempts: int):
        self.attempts_made = attempts_made
        self.max_attempts = max_attempts
        super().__init__(f"Attempt limit exceeded: {attempts_made}/{max_attempts}")


DEFAULT_TEMPLATE: dict = {
    "version": "1.0",
    "task_id": None,
    "loop_type": None,
    "attempts_made": 0,
    "max_attempts": 3,
    "original_branch": None,
    "last_green_commit": None,
    "last_green_backup": None,
    "touched_files": [],
    "history": [],
    "anti_patterns": [],
}


def _hash_project_path(project_path: str | Path) -> str:
    """Create a stable hash of the project path for state directory naming."""
    return hashlib.sha256(str(project_path).encode()).hexdigest()[:16]


def _get_state_dir() -> Path:
    """Return the kolega-code state directory (deferred import to avoid cycles)."""
    from kolega_code.cli.session_store import default_state_dir

    return default_state_dir()


class WorkLog:
    """Manages the work-log.json file for loop state tracking.

    All filesystem operations use ``project_path``, not the process
    working directory, so the loop operates on the correct project
    regardless of where the agent has ``cd``'d.
    """

    def __init__(self, path: str = "work-log.json", project_path: str | Path | None = None):
        self._path = Path(path)
        self._project_path = Path(project_path) if project_path else Path.cwd()
        # Defensive initialization — prevents KeyError on pre-load access.
        self._data: dict = copy.deepcopy(DEFAULT_TEMPLATE)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def for_task(cls, project_path: str | Path, task_id: str) -> "WorkLog":
        """Create a WorkLog stored in kolega-code's state directory.

        Returns a fully initialized WorkLog (loaded from disk or fresh template).
        """
        project_hash = _hash_project_path(project_path)
        state_root = _get_state_dir()
        path = state_root / "projects" / project_hash / "loops" / task_id / "work-log.json"
        return cls.load(str(path), project_path=project_path)

    @classmethod
    def load(cls, path: str = "work-log.json", project_path: str | Path | None = None) -> "WorkLog":
        """Load work-log.json, creating from template if missing or corrupted."""
        wl = cls(path, project_path=project_path)
        p = wl._path

        if p.exists():
            try:
                raw = p.read_text(encoding="utf-8")
                data = json.loads(raw)
                for key, default in DEFAULT_TEMPLATE.items():
                    if key not in data:
                        data[key] = default
                wl._data = data
                wl._validate()
            except (json.JSONDecodeError, ValueError):
                import sys

                print(
                    "[loop-state] WARNING: work-log.json corrupted. Reinitializing.",
                    file=sys.stderr,
                )
                wl._data = copy.deepcopy(DEFAULT_TEMPLATE)
                wl.save()
        else:
            wl._data = copy.deepcopy(DEFAULT_TEMPLATE)
            wl.save()

        return wl

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        if not isinstance(self._data, dict):
            raise ValueError("work-log.json must be a JSON object")
        if self._data.get("version") != "1.0":
            raise ValueError(f"Unsupported work-log version: {self._data.get('version')}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Atomically write work-log.json (temp file + os.replace)."""
        p = self._path
        p.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".work-log-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
                f.write("\n")
            os.replace(tmp, str(p))
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def to_dict(self) -> dict:
        return dict(self._data)

    # ------------------------------------------------------------------
    # Attempt tracking
    # ------------------------------------------------------------------

    @property
    def attempts_made(self) -> int:
        return self._data["attempts_made"]

    @property
    def max_attempts(self) -> int:
        return self._data["max_attempts"]

    def inc_attempt(self) -> int:
        """Increment attempts_made. Raises LoopLimitExceeded if over limit."""
        self._data["attempts_made"] += 1
        self.save()

        if self._data["attempts_made"] > self._data["max_attempts"]:
            raise LoopLimitExceeded(self._data["attempts_made"], self._data["max_attempts"])

        return self._data["attempts_made"]

    def record_attempt(
        self,
        status: str,
        summary: str,
        sub_agent_ids: Optional[list] = None,
        phase: str = "",
    ) -> None:
        """Append a history entry. If 'kept', update last_green_commit."""
        entry = {
            "attempt": self._data["attempts_made"],
            "phase": phase,
            "status": status,
            "sub_agent_ids": sub_agent_ids or [],
            "summary": summary,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._data["history"].append(entry)

        if status == "kept":
            self._data["last_green_commit"] = self._git_head()

        self.save()

    # ------------------------------------------------------------------
    # Revert / Backup (branch-based, safe)
    # ------------------------------------------------------------------

    def _git(self, *args: str) -> subprocess.CompletedProcess:
        """Run a git command in the project directory."""
        return subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            cwd=str(self._project_path),
        )

    def _is_git_repo(self) -> bool:
        try:
            self._git("rev-parse", "--git-dir").check_returncode()
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _git_head(self) -> Optional[str]:
        try:
            result = self._git("rev-parse", "HEAD")
            result.check_returncode()
            return result.stdout.strip() or None
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def _current_branch(self) -> Optional[str]:
        try:
            result = self._git("rev-parse", "--abbrev-ref", "HEAD")
            result.check_returncode()
            return result.stdout.strip() or None
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def _loop_branches(self) -> list[str]:
        """List branches matching loop/<task-id>-*."""
        task_id = self._data.get("task_id", "")
        if not task_id:
            return []
        prefix = f"loop/{task_id}-"
        try:
            result = self._git("branch", "--list", f"{prefix}*")
            return [b.strip().lstrip("* ") for b in result.stdout.splitlines() if b.strip()]
        except subprocess.CalledProcessError:
            return []

    def record_original_branch(self) -> str:
        """Record the starting branch before loop work begins."""
        branch = self._current_branch() or "main"
        self._data["original_branch"] = branch
        self.save()
        return branch

    def revert(self) -> str:
        """Return the shell command to revert to the pre-loop state.

        Uses branch-based strategy: switches back to the original branch
        and deletes all loop branches. Only loop-created changes are
        discarded; pre-existing user work is untouched.

        Falls back to rsync for non-git projects.
        """
        original_branch = self._data.get("original_branch")
        if original_branch and self._is_git_repo():
            loop_branches = self._loop_branches()
            delete_cmds = " ".join(f"-D {shlex_quote(b)}" for b in loop_branches) if loop_branches else ""
            cmds = [
                f"cd {shlex_quote(str(self._project_path))}",
                f"git checkout {shlex_quote(original_branch)}",
            ]
            if delete_cmds:
                cmds.append(f"git branch {delete_cmds}")
            return " && ".join(cmds)

        backup = self._data.get("last_green_backup")
        if backup and os.path.isdir(backup):
            proj = shlex_quote(str(self._project_path))
            bak = shlex_quote(backup)
            # Only restore files the loop touched, not the entire tree
            touched = self._data.get("touched_files", [])
            if touched:
                restore_cmds = " && ".join(
                    f"cp {bak}/{shlex_quote(f)} {proj}/{shlex_quote(f)}" for f in touched
                )
                return f"cd {proj} && {restore_cmds}"
            return f"rsync -a --delete {bak}/ {proj}/"

        return "echo '[loop-state] No revert point available. Nothing to do.'"

    def backup_current(self) -> str:
        """Snapshot current working tree in the project directory.

        Records the current git HEAD and branch as the revert point.
        For non-git repos, creates a filesystem backup.
        """
        if self._is_git_repo():
            commit = self._git_head()
            branch = self._current_branch()
            self._data["last_green_commit"] = commit
            if branch and not self._data.get("original_branch"):
                self._data["original_branch"] = branch
            self.save()
            return commit or ""

        proj = str(self._project_path)
        backup_dir = os.path.join(
            proj,
            f".loop-backup-{self._data.get('task_id', 'unknown')}",
        )
        if os.path.isdir(backup_dir):
            shutil.rmtree(backup_dir)

        ignore = shutil.ignore_patterns(
            ".loop-backup-*",
            "work-log.json",
            ".git",
            "node_modules",
            "__pycache__",
            "*.pyc",
            ".venv",
            "venv",
        )
        shutil.copytree(proj, backup_dir, ignore=ignore, symlinks=True)

        self._data["last_green_backup"] = backup_dir
        self.save()
        return backup_dir

    def record_touched_file(self, filepath: str) -> None:
        """Track a file the loop has modified for safe revert."""
        fp = Path(filepath)
        # If path is relative, resolve against project path
        if not fp.is_absolute():
            fp = self._project_path / fp
        try:
            rel = fp.relative_to(self._project_path).as_posix()
        except ValueError:
            # File is not under project path — store as-is
            rel = filepath
        if rel not in self._data["touched_files"]:
            self._data["touched_files"].append(rel)
            self.save()

    # ------------------------------------------------------------------
    # Anti-patterns
    # ------------------------------------------------------------------

    def record_anti_pattern(
        self,
        pattern: str,
        root_cause: str,
        file: str,
        line: int,
        prevention_rule: str,
    ) -> None:
        """Record or increment an anti-pattern entry."""
        existing = None
        for ap in self._data["anti_patterns"]:
            if ap["pattern"] == pattern:
                existing = ap
                break

        if existing:
            existing["occurrence_count"] += 1
        else:
            self._data["anti_patterns"].append(
                {
                    "pattern": pattern,
                    "root_cause": root_cause,
                    "file": file,
                    "line": line,
                    "prevention_rule": prevention_rule,
                    "first_seen": datetime.now(timezone.utc).isoformat(),
                    "occurrence_count": 1,
                }
            )

        self.save()

    def get_anti_patterns(self, for_module: Optional[str] = None) -> list:
        """Return anti-patterns, optionally filtered by module/file substring."""
        aps = self._data["anti_patterns"]
        if for_module:
            aps = [ap for ap in aps if for_module.lower() in ap.get("file", "").lower()]
        return aps
