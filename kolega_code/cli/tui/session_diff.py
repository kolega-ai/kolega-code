"""Git-backed net session diff tracking for the CLI TUI Changes inspector."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from subprocess import DEVNULL, PIPE, run as subprocess_run
from typing import Iterable, Literal, Optional

from kolega_code.agent.tool_backend.edit_preview import build_diff_preview


@dataclass
class FileBaseline:
    """A session-start baseline for one git path."""

    path: str  # git-root-relative, posix-style path
    exists: bool
    content: Optional[str] = None
    binary_or_unreadable: bool = False


@dataclass
class SessionDiffFile:
    """One file whose current state differs from the TUI session baseline."""

    path: str  # project-relative display path
    status: Literal["modified", "added", "deleted"]
    preview: Optional[dict]
    adds: int = 0
    dels: int = 0
    message: str = ""


class GitSessionDiffTracker:
    """Compute net file changes since a TUI session began, for git repos only.

    The tracker avoids whole-repo snapshots. It snapshots only files that are dirty
    at session start, then uses HEAD as the baseline for files that were clean.
    """

    def __init__(self, project_path: Path, git_root: Path) -> None:
        self.project_path = project_path.resolve()
        self.git_root = git_root.resolve()
        self._baseline: dict[str, FileBaseline] = {}
        self._baseline_paths: set[str] = set()
        self._head_sha: Optional[str] = None
        self._head_cache: dict[str, FileBaseline] = {}
        self._diff_cache: dict[str, tuple[Optional[tuple[int, int]], Optional[SessionDiffFile]]] = {}

    @classmethod
    def create(cls, project_path: Path) -> "GitSessionDiffTracker | None":
        git_root = cls._detect_git_root(project_path)
        if git_root is None:
            return None
        return cls(project_path, git_root)

    @staticmethod
    def _detect_git_root(project_path: Path) -> Path | None:
        try:
            completed = subprocess_run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=str(project_path),
                stdout=PIPE,
                stderr=DEVNULL,
                text=True,
                check=False,
            )
        except (OSError, ValueError):
            return None
        if completed.returncode != 0:
            return None
        root = completed.stdout.strip()
        return Path(root).resolve() if root else None

    def capture_baseline(self) -> None:
        """Snapshot files that are already dirty at session start."""
        self._baseline = {}
        for repo_path in self._git_status_paths():
            self._baseline[repo_path] = self._snapshot_repo_path(repo_path)
        self._baseline_paths = set(self._baseline)

    def refresh(self, event_paths: Iterable[str] = ()) -> list[SessionDiffFile]:
        """Return current net changes relative to the session-start baseline.

        Not safe for concurrent calls; the TUI caller serializes refreshes.
        """
        head_sha = self._current_head_sha()
        if head_sha != self._head_sha:
            self._head_cache.clear()
            self._diff_cache = {
                repo_path: cached for repo_path, cached in self._diff_cache.items() if repo_path in self._baseline
            }
            self._head_sha = head_sha

        candidates = set(self._git_status_paths()) | set(self._baseline_paths)
        candidates.update(self._repo_paths_from_event_paths(event_paths))

        diffs: list[SessionDiffFile] = []
        for repo_path in sorted(candidates):
            if not self._is_under_project(repo_path):
                continue
            sig = self._stat_signature(repo_path)
            cached = self._diff_cache.get(repo_path)
            if cached is not None and cached[0] == sig:
                if cached[1] is not None:
                    diffs.append(cached[1])
                continue
            baseline = self._baseline.get(repo_path)
            if baseline is None:
                baseline = self._cached_head_baseline(repo_path)
            current = self._snapshot_repo_path(repo_path)
            diff = self._build_diff(repo_path, baseline, current)
            self._diff_cache[repo_path] = (sig, diff)
            if diff is not None:
                diffs.append(diff)
        return diffs

    # ---- git/status helpers --------------------------------------------------

    def _git_status_paths(self) -> set[str]:
        try:
            completed = subprocess_run(
                ["git", "-c", "core.quotePath=false", "status", "--porcelain", "-z"],
                cwd=str(self.git_root),
                stdout=PIPE,
                stderr=DEVNULL,
                check=False,
            )
        except (OSError, ValueError):
            return set()
        if completed.returncode != 0:
            return set()
        return self._parse_porcelain_z(completed.stdout)

    @staticmethod
    def _parse_porcelain_z(output: bytes) -> set[str]:
        paths: set[str] = set()
        entries = [entry for entry in output.decode("utf-8", errors="surrogateescape").split("\0") if entry]
        index = 0
        while index < len(entries):
            entry = entries[index]
            if len(entry) < 4:
                index += 1
                continue
            status = entry[:2]
            path = entry[3:]
            if path:
                paths.add(path)
            # In porcelain v1 -z, rename/copy records are followed by the other path.
            if status[0] in {"R", "C"} or status[1] in {"R", "C"}:
                index += 1
                if index < len(entries) and entries[index]:
                    paths.add(entries[index])
            index += 1
        return paths

    def _head_baseline(self, repo_path: str) -> FileBaseline:
        try:
            completed = subprocess_run(
                ["git", "show", f"HEAD:{repo_path}"],
                cwd=str(self.git_root),
                stdout=PIPE,
                stderr=DEVNULL,
                check=False,
            )
        except (OSError, ValueError):
            return FileBaseline(path=repo_path, exists=False)
        if completed.returncode != 0:
            return FileBaseline(path=repo_path, exists=False)
        return self._baseline_from_bytes(repo_path, completed.stdout, exists=True)

    def _current_head_sha(self) -> str:
        try:
            completed = subprocess_run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(self.git_root),
                stdout=PIPE,
                stderr=DEVNULL,
                text=True,
                check=False,
            )
        except (OSError, ValueError):
            return ""
        if completed.returncode != 0:
            return ""
        return completed.stdout.strip()

    def _cached_head_baseline(self, repo_path: str) -> FileBaseline:
        baseline = self._head_cache.get(repo_path)
        if baseline is None:
            baseline = self._head_baseline(repo_path)
            self._head_cache[repo_path] = baseline
        return baseline

    # ---- path/content helpers ------------------------------------------------

    def _stat_signature(self, repo_path: str) -> Optional[tuple[int, int]]:
        try:
            stat = os.stat(self.git_root / repo_path)
        except OSError:
            return None
        return (stat.st_mtime_ns, stat.st_size)

    def _snapshot_repo_path(self, repo_path: str) -> FileBaseline:
        abs_path = self.git_root / repo_path
        if not abs_path.exists() or not abs_path.is_file():
            return FileBaseline(path=repo_path, exists=False)
        try:
            data = abs_path.read_bytes()
        except OSError:
            return FileBaseline(path=repo_path, exists=True, binary_or_unreadable=True)
        return self._baseline_from_bytes(repo_path, data, exists=True)

    @staticmethod
    def _baseline_from_bytes(repo_path: str, data: bytes, *, exists: bool) -> FileBaseline:
        if b"\x00" in data:
            return FileBaseline(path=repo_path, exists=exists, binary_or_unreadable=True)
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError:
            content = data.decode("utf-8", errors="replace")
        return FileBaseline(path=repo_path, exists=exists, content=content)

    def _repo_paths_from_event_paths(self, event_paths: Iterable[str]) -> set[str]:
        paths: set[str] = set()
        for event_path in event_paths:
            repo_path = self._event_path_to_repo_path(event_path)
            if repo_path:
                paths.add(repo_path)
        return paths

    def _event_path_to_repo_path(self, event_path: str) -> str:
        if not event_path:
            return ""
        path = Path(event_path)
        abs_path = path if path.is_absolute() else self.project_path / path
        try:
            resolved = abs_path.resolve(strict=False)
            rel = resolved.relative_to(self.git_root)
        except ValueError:
            return ""
        return self._posix(rel)

    def _is_under_project(self, repo_path: str) -> bool:
        try:
            (self.git_root / repo_path).resolve(strict=False).relative_to(self.project_path)
            return True
        except ValueError:
            return False

    def _display_path(self, repo_path: str) -> str:
        try:
            rel = (self.git_root / repo_path).resolve(strict=False).relative_to(self.project_path)
            return self._posix(rel)
        except ValueError:
            return repo_path

    @staticmethod
    def _posix(path: Path) -> str:
        rel = str(PurePosixPath(path.as_posix()))
        return "" if rel == "." else rel

    # ---- diff construction ---------------------------------------------------

    def _build_diff(
        self,
        repo_path: str,
        baseline: FileBaseline,
        current: FileBaseline,
    ) -> SessionDiffFile | None:
        if self._same_state(baseline, current):
            return None

        display_path = self._display_path(repo_path)
        if not baseline.exists and current.exists:
            status: Literal["modified", "added", "deleted"] = "added"
        elif baseline.exists and not current.exists:
            status = "deleted"
        else:
            status = "modified"

        if baseline.binary_or_unreadable or current.binary_or_unreadable:
            return SessionDiffFile(
                path=display_path,
                status=status,
                preview=None,
                message="Binary or unreadable file changed; textual diff unavailable.",
            )

        old = baseline.content or ""
        new = current.content or ""
        preview = build_diff_preview(old, new, display_path, max_lines=0)
        if preview is None and old != new:
            message = "File changed; textual diff unavailable."
        else:
            message = ""
        return SessionDiffFile(
            path=display_path,
            status=status,
            preview=preview,
            adds=int((preview or {}).get("adds") or 0),
            dels=int((preview or {}).get("dels") or 0),
            message=message,
        )

    @staticmethod
    def _same_state(left: FileBaseline, right: FileBaseline) -> bool:
        if left.exists != right.exists:
            return False
        if not left.exists and not right.exists:
            return True
        if left.binary_or_unreadable or right.binary_or_unreadable:
            return left.binary_or_unreadable == right.binary_or_unreadable and left.content == right.content
        return (left.content or "") == (right.content or "")
