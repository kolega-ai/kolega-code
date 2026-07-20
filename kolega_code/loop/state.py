"""WorkLog — persistent state for autonomous loop engineering.

Manages work-log.json with atomic saves, attempt limit enforcement,
git-based revert (with rsync fallback), and anti-pattern recording.
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

from kolega_code.local_state import get_state_dir

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
    "last_green_commit": None,
    "last_green_backup": None,
    "history": [],
    "anti_patterns": [],
}


def _hash_project_path(project_path: str) -> str:
    """Create a stable hash of the project path for state directory naming."""
    return hashlib.sha256(project_path.encode()).hexdigest()[:16]


class WorkLog:
    """Manages the work-log.json file for loop state tracking."""

    def __init__(self, path: str = "work-log.json"):
        self._path = Path(path)
        self._data: dict = {}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def for_task(cls, project_path: str, task_id: str) -> "WorkLog":
        """Create a WorkLog stored in kolega-code's state directory."""
        project_hash = hashlib.sha256(project_path.encode()).hexdigest()[:16]
        state_root = get_state_dir()
        path = state_root / "projects" / project_hash / "loops" / task_id / "work-log.json"
        return cls(str(path))

    @classmethod
    def load(cls, path: str = "work-log.json") -> "WorkLog":
        """Load work-log.json, creating from template if missing or corrupted."""
        wl = cls(path)
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
                    f"[loop-state] WARNING: work-log.json corrupted. Reinitializing.",
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
    # Revert / Backup
    # ------------------------------------------------------------------

    def revert(self) -> str:
        """Return the shell command to revert to last known-good state."""
        commit = self._data.get("last_green_commit")
        if commit and self._is_git_repo():
            return f"git reset --hard {commit}"

        backup = self._data.get("last_green_backup")
        if backup and os.path.isdir(backup):
            cwd = os.getcwd()
            return f"rsync -a --delete {shlex_quote(backup)}/ {shlex_quote(cwd)}/"

        return "echo '[loop-state] No revert point available. Nothing to do.'"

    def backup_current(self) -> str:
        """Snapshot current working tree. Returns the backup path."""
        if self._is_git_repo():
            commit = self._git_head()
            self._data["last_green_commit"] = commit
            self.save()
            return commit

        backup_dir = os.path.join(
            os.getcwd(),
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
        shutil.copytree(os.getcwd(), backup_dir, ignore=ignore, symlinks=True)

        self._data["last_green_backup"] = backup_dir
        self.save()
        return backup_dir

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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_git_repo() -> bool:
        try:
            subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                capture_output=True,
                check=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    @staticmethod
    def _git_head() -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip() or None
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
