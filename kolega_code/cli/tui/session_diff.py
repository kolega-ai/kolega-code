"""Net session diff tracking and turn checkpoints for the CLI TUI Changes inspector.

Two trackers share one currency (checkpoints, ``SessionDiffFile`` diffs, restore
plans): ``GitSessionDiffTracker`` for git repos (sees every working-tree change,
including shell-driven ones) and ``SnapshotLedgerDiffTracker`` for non-git
projects (sees only edits recorded by the agent's snapshot service). The diff a
checkpoint displays and the restore plan for that checkpoint are built from the
same collected change records, so rewind always reverts exactly what is shown.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from subprocess import DEVNULL, PIPE, run as subprocess_run
from typing import TYPE_CHECKING, Iterable, Literal, Optional

from kolega_code.agent.tool_backend.edit_preview import build_diff_preview

if TYPE_CHECKING:
    from kolega_code.services.snapshots import FileState, SnapshotService

# Checkpoint baseline bytes above this size are not retained for restore (the
# diff preview still renders from the decoded content).
MAX_RETAINED_BYTES = 8 * 1024 * 1024
# Oldest turn checkpoints are evicted past this cap; checkpoint 0 never is.
REWIND_MAX_CHECKPOINTS = 50
# Committed-paths candidates per refresh are truncated to this many entries.
# The Changes screen renders from the same collection, so a truncated view
# still matches what a rewind would restore.
MAX_COMMITTED_CANDIDATES = 500


@dataclass
class FileBaseline:
    """A checkpoint baseline for one tracked path."""

    path: str  # tracker-relative, posix-style path
    exists: bool
    content: Optional[str] = None
    binary_or_unreadable: bool = False
    data: Optional[bytes] = None  # raw bytes for restore; None when not retained
    sha: Optional[str] = None  # sha256 of the file bytes, when read


@dataclass
class TurnCheckpoint:
    """The tracked state boundary captured when a turn starts.

    ``checkpoint_id`` 0 is the session start; ids stay stable across eviction.
    """

    checkpoint_id: int
    label: str
    created_at: float
    head_sha: str = ""  # git tracker only; pinned at capture
    dirty: dict[str, FileBaseline] = field(default_factory=dict)


@dataclass
class SessionDiffFile:
    """One file whose current state differs from the selected checkpoint."""

    path: str  # project-relative display path
    status: Literal["modified", "added", "deleted"]
    preview: Optional[dict]
    adds: int = 0
    dels: int = 0
    message: str = ""


@dataclass
class ChangeRecord:
    """A collected change: the shared source for display and restore."""

    repo_path: str
    baseline: FileBaseline
    sig: Optional[tuple[int, int]]
    diff: SessionDiffFile


@dataclass
class RewindPlanItem:
    repo_path: str
    display_path: str
    action: Literal["write", "delete"]
    baseline: FileBaseline
    current_sig: Optional[tuple[int, int]]


@dataclass
class RewindResult:
    restored: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (path, reason)
    errors: list[tuple[str, str]] = field(default_factory=list)  # (path, reason)


class RewindDriftError(Exception):
    """Files changed between planning a rewind and applying it."""

    def __init__(self, paths: list[str]) -> None:
        self.paths = paths
        super().__init__(f"Files changed since the rewind was planned: {', '.join(paths)}")


class SessionDiffTrackerBase:
    """Checkpoint ladder, diff collection, and restore shared by both trackers.

    Not safe for concurrent calls; the TUI caller serializes refreshes and
    rewinds. ``capture_baseline()`` must run before ``refresh()``.
    """

    def __init__(self, project_path: Path) -> None:
        self.project_path = project_path.resolve()
        self._checkpoints: list[TurnCheckpoint] = []
        self._next_checkpoint_id = 0
        # sha256 -> (data, content): shared payloads so identical file versions
        # across checkpoints cost one copy.
        self._content_intern: dict[str, tuple[bytes, Optional[str]]] = {}
        self._diff_cache: dict[str, tuple[Optional[tuple[int, int]], Optional[SessionDiffFile]]] = {}
        self._diff_cache_checkpoint_id: Optional[int] = None

    # ---- checkpoints ---------------------------------------------------------

    def capture_baseline(self) -> None:
        """Capture checkpoint 0 (session start), resetting the ladder."""
        self._checkpoints = []
        self._next_checkpoint_id = 0
        self._content_intern = {}
        self._diff_cache = {}
        self._diff_cache_checkpoint_id = None
        self.capture_checkpoint("")

    def capture_checkpoint(self, label: str) -> TurnCheckpoint:
        checkpoint = TurnCheckpoint(
            checkpoint_id=self._next_checkpoint_id,
            label=label,
            created_at=time.time(),
            head_sha=self._current_head_sha(),
            dirty=self._capture_dirty_baselines(),
        )
        self._next_checkpoint_id += 1
        self._checkpoints.append(checkpoint)
        while len(self._checkpoints) > REWIND_MAX_CHECKPOINTS:
            del self._checkpoints[1]  # keep checkpoint 0, evict the oldest turn
            self._rebuild_intern()
        return checkpoint

    def checkpoints(self) -> list[TurnCheckpoint]:
        return list(self._checkpoints)

    def checkpoint_for_id(self, checkpoint_id: int) -> Optional[TurnCheckpoint]:
        for checkpoint in self._checkpoints:
            if checkpoint.checkpoint_id == checkpoint_id:
                return checkpoint
        return None

    def _resolve_checkpoint(self, checkpoint_id: Optional[int]) -> TurnCheckpoint:
        if checkpoint_id is None:
            return self._checkpoints[0]
        checkpoint = self.checkpoint_for_id(checkpoint_id)
        if checkpoint is None:
            raise ValueError(f"Unknown checkpoint id: {checkpoint_id}")
        return checkpoint

    def _rebuild_intern(self) -> None:
        self._content_intern = {}
        for checkpoint in self._checkpoints:
            for baseline in checkpoint.dirty.values():
                if baseline.sha is not None and baseline.data is not None:
                    self._content_intern.setdefault(baseline.sha, (baseline.data, baseline.content))

    def _intern_baseline(self, baseline: FileBaseline) -> FileBaseline:
        if baseline.sha is None or baseline.data is None:
            return baseline
        cached = self._content_intern.get(baseline.sha)
        if cached is None:
            self._content_intern[baseline.sha] = (baseline.data, baseline.content)
            return baseline
        data, content = cached
        return FileBaseline(
            path=baseline.path,
            exists=baseline.exists,
            content=content,
            binary_or_unreadable=baseline.binary_or_unreadable,
            data=data,
            sha=baseline.sha,
        )

    # ---- diff collection -----------------------------------------------------

    def refresh(self, event_paths: Iterable[str] = (), *, checkpoint_id: Optional[int] = None) -> list[SessionDiffFile]:
        """Return current net changes relative to the selected checkpoint."""
        checkpoint = self._resolve_checkpoint(checkpoint_id)
        return [record.diff for record in self._collect_changes(checkpoint, event_paths)]

    def _collect_changes(self, checkpoint: TurnCheckpoint, event_paths: Iterable[str]) -> list[ChangeRecord]:
        if checkpoint.checkpoint_id != self._diff_cache_checkpoint_id:
            self._diff_cache = {}
            self._diff_cache_checkpoint_id = checkpoint.checkpoint_id

        records: list[ChangeRecord] = []
        for repo_path in sorted(self._candidate_paths(checkpoint, event_paths)):
            if not self._is_under_project(repo_path):
                continue
            sig = self._stat_signature(repo_path)
            cached = self._diff_cache.get(repo_path)
            if cached is not None and cached[0] == sig:
                diff = cached[1]
                if diff is None:
                    continue
                baseline = self._checkpoint_baseline(checkpoint, repo_path)
            else:
                baseline = self._checkpoint_baseline(checkpoint, repo_path)
                current = self._snapshot_repo_path(repo_path)
                diff = self._build_diff(repo_path, baseline, current)
                self._diff_cache[repo_path] = (sig, diff)
                if diff is None:
                    continue
            records.append(ChangeRecord(repo_path=repo_path, baseline=baseline, sig=sig, diff=diff))
        return records

    def _checkpoint_baseline(self, checkpoint: TurnCheckpoint, repo_path: str) -> FileBaseline:
        baseline = checkpoint.dirty.get(repo_path)
        if baseline is not None:
            return baseline
        return self._fallback_baseline(checkpoint, repo_path)

    # ---- restore -------------------------------------------------------------

    def build_restore_plan(
        self,
        *,
        checkpoint_id: Optional[int] = None,
        event_paths: Iterable[str] = (),
        paths: Optional[set[str]] = None,
    ) -> list[RewindPlanItem]:
        """Plan a restore to the selected checkpoint.

        ``paths`` filters by display path for per-file restore.
        """
        checkpoint = self._resolve_checkpoint(checkpoint_id)
        plan: list[RewindPlanItem] = []
        for record in self._collect_changes(checkpoint, event_paths):
            if paths is not None and record.diff.path not in paths:
                continue
            action: Literal["write", "delete"] = "delete" if record.diff.status == "added" else "write"
            plan.append(
                RewindPlanItem(
                    repo_path=record.repo_path,
                    display_path=record.diff.path,
                    action=action,
                    baseline=record.baseline,
                    current_sig=record.sig,
                )
            )
        return plan

    def apply_restore_plan(self, plan: list[RewindPlanItem], *, force: bool = False) -> RewindResult:
        """Write baselines back to the working tree. Never touches git history."""
        if not force:
            drifted = [item.display_path for item in plan if self._stat_signature(item.repo_path) != item.current_sig]
            if drifted:
                raise RewindDriftError(drifted)

        result = RewindResult()
        for item in (item for item in plan if item.action == "delete"):
            reason = self._delete_guard_reason(item.repo_path)
            if reason:
                result.skipped.append((item.display_path, reason))
                continue
            try:
                self._abs_path(item.repo_path).unlink(missing_ok=True)
                result.deleted.append(item.display_path)
            except OSError as exc:
                result.errors.append((item.display_path, str(exc)))
        for item in (item for item in plan if item.action == "write"):
            if item.baseline.data is None:
                result.skipped.append((item.display_path, "baseline content was not retained"))
                continue
            abs_path = self._abs_path(item.repo_path)
            try:
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                abs_path.write_bytes(item.baseline.data)
                result.restored.append(item.display_path)
            except OSError as exc:
                result.errors.append((item.display_path, str(exc)))
        return result

    # ---- provider hooks ------------------------------------------------------

    def _capture_dirty_baselines(self) -> dict[str, FileBaseline]:
        raise NotImplementedError

    def _candidate_paths(self, checkpoint: TurnCheckpoint, event_paths: Iterable[str]) -> set[str]:
        raise NotImplementedError

    def _fallback_baseline(self, checkpoint: TurnCheckpoint, repo_path: str) -> FileBaseline:
        raise NotImplementedError

    def _current_head_sha(self) -> str:
        return ""

    def _delete_guard_reason(self, repo_path: str) -> Optional[str]:
        return None

    def _abs_path(self, repo_path: str) -> Path:
        raise NotImplementedError

    def _display_path(self, repo_path: str) -> str:
        raise NotImplementedError

    def _is_under_project(self, repo_path: str) -> bool:
        raise NotImplementedError

    # ---- content helpers -----------------------------------------------------

    def _stat_signature(self, repo_path: str) -> Optional[tuple[int, int]]:
        try:
            stat = os.stat(self._abs_path(repo_path))
        except OSError:
            return None
        return (stat.st_mtime_ns, stat.st_size)

    def _snapshot_repo_path(self, repo_path: str) -> FileBaseline:
        abs_path = self._abs_path(repo_path)
        if not abs_path.exists() or not abs_path.is_file():
            return FileBaseline(path=repo_path, exists=False)
        try:
            data = abs_path.read_bytes()
        except OSError:
            return FileBaseline(path=repo_path, exists=True, binary_or_unreadable=True)
        return self._baseline_from_bytes(repo_path, data, exists=True)

    @staticmethod
    def _baseline_from_bytes(repo_path: str, data: bytes, *, exists: bool) -> FileBaseline:
        sha = hashlib.sha256(data).hexdigest()
        retained = data if len(data) <= MAX_RETAINED_BYTES else None
        if b"\x00" in data:
            return FileBaseline(path=repo_path, exists=exists, binary_or_unreadable=True, data=retained, sha=sha)
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError:
            content = data.decode("utf-8", errors="replace")
        return FileBaseline(path=repo_path, exists=exists, content=content, data=retained, sha=sha)

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
        if left.sha is not None and right.sha is not None:
            return left.sha == right.sha
        if left.binary_or_unreadable or right.binary_or_unreadable:
            return left.binary_or_unreadable == right.binary_or_unreadable and left.content == right.content
        return (left.content or "") == (right.content or "")


class GitSessionDiffTracker(SessionDiffTrackerBase):
    """Track net changes against per-turn checkpoints, for git repos only.

    The tracker avoids whole-repo snapshots. Each checkpoint snapshots only the
    files that are dirty at capture time and pins the HEAD sha; clean files
    baseline against that pinned commit. Pinned commits stay resolvable for the
    tracker's lifetime because they were HEAD during this session and remain
    reflog-reachable.
    """

    def __init__(self, project_path: Path, git_root: Path) -> None:
        super().__init__(project_path)
        self.git_root = git_root.resolve()
        self._commit_cache: dict[tuple[str, str], FileBaseline] = {}
        self._committed_paths_cache: dict[tuple[str, str], set[str]] = {}

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

    # ---- provider hooks ------------------------------------------------------

    def _capture_dirty_baselines(self) -> dict[str, FileBaseline]:
        return {
            repo_path: self._intern_baseline(self._snapshot_repo_path(repo_path))
            for repo_path in self._git_status_paths()
        }

    def _candidate_paths(self, checkpoint: TurnCheckpoint, event_paths: Iterable[str]) -> set[str]:
        candidates = self._git_status_paths() | set(checkpoint.dirty)
        candidates.update(self._repo_paths_from_event_paths(event_paths))
        candidates.update(self._committed_paths_since(checkpoint.head_sha))
        return candidates

    def _fallback_baseline(self, checkpoint: TurnCheckpoint, repo_path: str) -> FileBaseline:
        key = (checkpoint.head_sha, repo_path)
        baseline = self._commit_cache.get(key)
        if baseline is None:
            baseline = self._commit_baseline(checkpoint.head_sha, repo_path)
            self._commit_cache[key] = baseline
        return baseline

    def _delete_guard_reason(self, repo_path: str) -> Optional[str]:
        # A gitignored file that predates the session never appears in git
        # status, so its baseline resolves to "absent at checkpoint" and a
        # restore would delete it. Keep ignored files instead.
        if self._is_ignored(repo_path):
            return "gitignored file kept"
        return None

    def _abs_path(self, repo_path: str) -> Path:
        return self.git_root / repo_path

    def _display_path(self, repo_path: str) -> str:
        try:
            rel = (self.git_root / repo_path).resolve(strict=False).relative_to(self.project_path)
            return self._posix(rel)
        except ValueError:
            return repo_path

    def _is_under_project(self, repo_path: str) -> bool:
        try:
            (self.git_root / repo_path).resolve(strict=False).relative_to(self.project_path)
            return True
        except ValueError:
            return False

    # ---- git helpers ---------------------------------------------------------

    def _git_status_paths(self) -> set[str]:
        try:
            completed = subprocess_run(
                # -uall lists files inside untracked directories individually;
                # the default would report only "dir/" and hide the files.
                ["git", "-c", "core.quotePath=false", "status", "--porcelain", "-z", "-uall"],
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

    def _commit_baseline(self, commit_sha: str, repo_path: str) -> FileBaseline:
        if not commit_sha:
            return FileBaseline(path=repo_path, exists=False)
        try:
            completed = subprocess_run(
                ["git", "show", f"{commit_sha}:{repo_path}"],
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

    def _committed_paths_since(self, checkpoint_sha: str) -> set[str]:
        """Paths changed by commits made after the checkpoint.

        Without this, a change that gets committed (and is then clean in git
        status) would vanish from both the diff view and the restore plan.
        A repo that gained its first commit mid-session has no checkpoint sha
        to diff from and is not covered.
        """
        head = self._current_head_sha()
        if not checkpoint_sha or not head or head == checkpoint_sha:
            return set()
        key = (checkpoint_sha, head)
        cached = self._committed_paths_cache.get(key)
        if cached is not None:
            return cached
        try:
            completed = subprocess_run(
                ["git", "diff", "--name-only", "-z", checkpoint_sha, head],
                cwd=str(self.git_root),
                stdout=PIPE,
                stderr=DEVNULL,
                check=False,
            )
        except (OSError, ValueError):
            return set()
        if completed.returncode != 0:
            return set()
        names = [name for name in completed.stdout.decode("utf-8", errors="surrogateescape").split("\0") if name]
        paths = set(sorted(names)[:MAX_COMMITTED_CANDIDATES])
        self._committed_paths_cache[key] = paths
        return paths

    def _is_ignored(self, repo_path: str) -> bool:
        try:
            completed = subprocess_run(
                ["git", "check-ignore", "-q", "--", repo_path],
                cwd=str(self.git_root),
                stdout=DEVNULL,
                stderr=DEVNULL,
                check=False,
            )
        except (OSError, ValueError):
            return False
        return completed.returncode == 0

    # ---- path helpers --------------------------------------------------------

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

    @staticmethod
    def _posix(path: Path) -> str:
        rel = str(PurePosixPath(path.as_posix()))
        return "" if rel == "." else rel


class SnapshotLedgerDiffTracker(SessionDiffTrackerBase):
    """Track net changes from the agent's snapshot records, for non-git projects.

    Baselines come from the earliest mutation record at or after the selected
    checkpoint: its ``before`` state is the file's content at the checkpoint.
    Changes made outside snapshotted tools (e.g. shell commands) are invisible.
    """

    def __init__(self, project_path: Path, snapshot_service: "SnapshotService") -> None:
        super().__init__(project_path)
        self._service = snapshot_service
        # (checkpoint_id, path) -> baseline reconstructed from the ledger
        self._ledger_baseline_cache: dict[tuple[int, str], FileBaseline] = {}

    # ---- provider hooks ------------------------------------------------------

    def _capture_dirty_baselines(self) -> dict[str, FileBaseline]:
        return {}

    def _candidate_paths(self, checkpoint: TurnCheckpoint, event_paths: Iterable[str]) -> set[str]:
        del event_paths  # ledger records are authoritative for tracked edits
        candidates: set[str] = set()
        for record in self._records_since(checkpoint):
            candidates.update(record.before)
        return candidates

    def _fallback_baseline(self, checkpoint: TurnCheckpoint, repo_path: str) -> FileBaseline:
        key = (checkpoint.checkpoint_id, repo_path)
        baseline = self._ledger_baseline_cache.get(key)
        if baseline is not None:
            return baseline
        state: Optional["FileState"] = None
        for record in self._records_since(checkpoint):
            if repo_path in record.before:
                state = record.before[repo_path]
                break
        baseline = self._baseline_from_state(repo_path, state)
        self._ledger_baseline_cache[key] = baseline
        return baseline

    def _abs_path(self, repo_path: str) -> Path:
        return self.project_path / repo_path

    def _display_path(self, repo_path: str) -> str:
        return repo_path

    def _is_under_project(self, repo_path: str) -> bool:
        return True  # ledger paths are project-relative by construction

    # ---- ledger helpers ------------------------------------------------------

    def _records_since(self, checkpoint: TurnCheckpoint):
        """Mutation records at/after the checkpoint, oldest first."""
        boundary = datetime.fromtimestamp(checkpoint.created_at, timezone.utc).isoformat()
        records = [
            record
            for record in self._service.list_snapshots(limit=10_000)
            if not record.manual and record.created_at >= boundary
        ]
        records.sort(key=lambda record: record.created_at)
        return records

    def _baseline_from_state(self, repo_path: str, state: Optional["FileState"]) -> FileBaseline:
        from kolega_code.services.snapshots import SnapshotError

        if state is None or state.kind == "missing":
            return FileBaseline(path=repo_path, exists=False)
        if state.kind != "file":
            return FileBaseline(path=repo_path, exists=True, binary_or_unreadable=True)
        try:
            data = self._service.read_blob(state)
        except SnapshotError:
            return FileBaseline(path=repo_path, exists=True, binary_or_unreadable=True)
        return self._intern_baseline(self._baseline_from_bytes(repo_path, data, exists=True))
