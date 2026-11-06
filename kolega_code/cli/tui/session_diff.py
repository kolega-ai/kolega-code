"""Net session diff tracking and turn checkpoints for the CLI TUI Changes inspector.

Two trackers share one currency (checkpoints, ``SessionDiffFile`` diffs, restore
plans): ``GitSessionDiffTracker`` for git repos (sees every working-tree change
in the tracked worktree, including shell-driven ones) and
``SnapshotLedgerDiffTracker`` for non-git projects (sees only edits recorded by
the agent's snapshot service). The diff a checkpoint displays and the restore
plan for that checkpoint are built from the same collected change records, so
rewind always reverts exactly what is shown. File contents are never held in
memory: checkpoint copies live in an on-disk spool (``checkpoint_store``) and
everything else is re-read from git, the working tree, or the snapshot ledger.

Scope is per-worktree and per-session. Content that merely *arrived* in HEAD —
merge second parents, fast-forwarded commits, rebase base moves, checkouts —
belongs to whoever authored it (often another agent working in a sibling
worktree), so it is neither shown nor restorable. See
``GitSessionDiffTracker._session_committed_paths``.
"""

from __future__ import annotations

import bisect
import enum
import hashlib
import os
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from stat import S_ISREG
from subprocess import DEVNULL, PIPE, CompletedProcess, run as subprocess_run
from typing import TYPE_CHECKING, Iterable, Literal, Optional, Sequence, Union

from kolega_code.agent.tool_backend.edit_preview import MAX_DIFF_INPUT_BYTES, build_diff_preview
from kolega_code.git_env import git_env
from kolega_code.worktrees import list_worktrees

from .checkpoint_store import CheckpointBlobStore

if TYPE_CHECKING:
    from kolega_code.services.snapshots import FileState, SnapshotService

# Files above this size are never read: they get a stat-signature baseline,
# show as changed without a textual diff, and cannot be restored.
MAX_RETAINED_BYTES = 8 * 1024 * 1024
# Disk the checkpoint copies of one tracker may use.
CHECKPOINT_STORE_BUDGET = 10 * 1024**3
# Share of the budget the session-start capture may fill, so files changed
# later in the session still get copies.
BASELINE_STORE_SHARE = 0.8
SPOOL_DIRNAME = "checkpoint-spool"
# A side larger than this cannot fit build_diff_preview's character cap, so it
# is never decoded (UTF-8 is at most 4 bytes per character).
MAX_DIFF_DECODE_BYTES = 4 * MAX_DIFF_INPUT_BYTES
# Head shown on the Changes screen when a diff input is too large to diff.
CHANGES_HEAD_MAX_LINES = 400
# Oldest turn checkpoints are evicted past this cap; checkpoint 0 never is.
REWIND_MAX_CHECKPOINTS = 50
# Committed-paths candidates per refresh are truncated to this many entries.
# The Changes screen renders from the same collection, so a truncated view
# still matches what a rewind would restore.
MAX_COMMITTED_CANDIDATES = 500
# HEAD reflog entries inspected per refresh when attributing commits.
MAX_REFLOG_ENTRIES = 200

# Reflog subjects for transitions that created a commit in this worktree. Git
# writes these unlocalised, and anything unrecognised is treated as
# non-attributable, so an unknown subject hides changes rather than leaking
# another worktree's work.
_COMMIT_REFLOG_KINDS = frozenset(
    {
        "commit",
        "commit (initial)",
        "commit (amend)",
        "cherry-pick",
        "revert",
        "am",
        "apply",
        "rebase (pick)",
        "rebase (reword)",
        "rebase (squash)",
        "rebase (fixup)",
        "rebase (continue)",
    }
)
# Reflog subjects for transitions that can bring in outside history. Only the
# combined diff of a real merge commit (the conflict resolution authored here)
# is attributed; a fast-forward contributes nothing.
_MERGE_REFLOG_KINDS = frozenset({"merge", "pull", "commit (merge)", "rebase (merge)"})


NOT_RETAINED = "baseline content was not retained"
TOO_LARGE_TO_COPY = "too large to keep a copy"
DAMAGED_COPY = "checkpoint copy is damaged"

BaselineSource = Literal["none", "store", "commit", "ledger"]


class _Unset(enum.Enum):
    TOKEN = 0


@dataclass(frozen=True, slots=True)
class FileBaseline:
    """One path's state at a checkpoint. Bytes are loaded on demand from ``source``."""

    path: str  # tracker-relative, posix-style path
    exists: bool
    binary_or_unreadable: bool = False
    sha: Optional[str] = None  # sha256 of the file bytes, when read
    sig: Optional[tuple[int, int]] = None  # (mtime_ns, size) of a working-tree snapshot
    size: int = 0
    source: BaselineSource = "none"  # where the bytes live; "none" when not retained
    ref: Union[str, "FileState", None] = None  # commit sha ("commit") or FileState ("ledger")
    unavailable: str = ""  # why the bytes were not retained


@dataclass
class TurnCheckpoint:
    """The tracked state boundary captured when a turn starts.

    ``checkpoint_id`` 0 is the session start; ids stay stable across eviction.
    """

    checkpoint_id: int
    label: str
    created_at: float
    head_sha: str = ""  # git tracker only; pinned at capture


@dataclass
class SessionDiffFile:
    """One file whose current state differs from the selected checkpoint."""

    path: str  # project-relative display path
    status: Literal["modified", "added", "deleted"]
    preview: Optional[dict]
    adds: int = 0
    dels: int = 0
    message: str = ""
    unrestorable: str = ""  # why a rewind would skip this file, "" when it can restore it


@dataclass
class ChangeRecord:
    """A collected change: the shared source for display and restore."""

    repo_path: str
    baseline: FileBaseline
    sig: Optional[tuple[int, int]]
    diff: SessionDiffFile


@dataclass(frozen=True)
class _ReflogEntry:
    """One HEAD reflog record for the tracked worktree."""

    sha: str  # the commit HEAD moved to
    parents: tuple[str, ...]
    timestamp: int  # unix seconds, 0 when unparsable
    subject: str  # reflog message, e.g. "commit: fix parser"


@dataclass(frozen=True)
class DiffScope:
    """What the current diff view covers, for display and rewind warnings."""

    root_label: str = ""  # "" for the main checkout, else the worktree directory name
    root_path: str = ""  # absolute tracker root
    branch: str = ""  # short branch name, "detached at <sha>", or ""
    linked_worktree: bool = False
    history_moved: bool = False  # a non-attributed HEAD move happened since the baseline
    history_tracked: bool = True  # the HEAD reflog could be read


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


_BINARY_MESSAGE = "Binary or unreadable file changed; textual diff unavailable."


class SessionDiffTrackerBase:
    """Checkpoint ladder, diff collection, and restore shared by both trackers.

    Not safe for concurrent calls; the TUI caller serializes refreshes and
    rewinds. ``capture_baseline()`` must run before ``refresh()``.
    """

    def __init__(self, project_path: Path, *, store: Optional[CheckpointBlobStore] = None) -> None:
        self.project_path = project_path.resolve()
        self._checkpoints: list[TurnCheckpoint] = []
        self._next_checkpoint_id = 0
        # Every "store" baseline held by a checkpoint or by _snapshot_cache owns
        # one reference to its blob.
        self._store = store
        self._store_limit = CHECKPOINT_STORE_BUDGET
        # repo_path -> [(checkpoint_id, baseline or None)], oldest first: from
        # that checkpoint on, the path was dirty with that baseline, or clean
        # (None, so it baselines against the checkpoint's HEAD). A capture
        # appends only for paths that changed, so unchanged paths cost nothing.
        self._history: dict[str, list[tuple[int, Optional[FileBaseline]]]] = {}
        self._dirty_now: set[str] = set()  # dirty at the latest checkpoint
        self._multi_version: set[str] = set()  # paths with more than one history entry
        self._dirty_at_cache: Optional[tuple[int, frozenset[str]]] = None
        self._diff_cache: dict[str, tuple[Optional[tuple[int, int]], Optional[SessionDiffFile]]] = {}
        self._diff_cache_checkpoint_id: Optional[int] = None
        # repo_path -> latest snapshot: shared by capture and refresh so a file
        # is read and hashed once per content version, then stat-only.
        self._snapshot_cache: dict[str, FileBaseline] = {}
        self._streamed_sha_cache: dict[tuple[str, tuple[int, int]], Optional[str]] = {}
        self._abs_root = str(self.project_path)  # the directory _abs_path resolves repo paths against

    def close(self) -> None:
        """Delete the checkpoint copies. The tracker must not be used afterwards."""
        if self._store is not None:
            self._store.close()

    # ---- checkpoints ---------------------------------------------------------

    def capture_baseline(self, label: str = "") -> None:
        """Capture checkpoint 0 (session start), resetting the ladder."""
        self._checkpoints = []
        self._next_checkpoint_id = 0
        self._history = {}
        self._dirty_now = set()
        self._multi_version = set()
        self._dirty_at_cache = None
        self._diff_cache = {}
        self._diff_cache_checkpoint_id = None
        self._snapshot_cache = {}
        self._streamed_sha_cache = {}
        if self._store is not None:
            self._store.clear()
        self._store_limit = int(CHECKPOINT_STORE_BUDGET * BASELINE_STORE_SHARE)
        try:
            self.capture_checkpoint(label)
        finally:
            self._store_limit = CHECKPOINT_STORE_BUDGET

    def capture_checkpoint(self, label: str) -> TurnCheckpoint:
        checkpoint = TurnCheckpoint(
            checkpoint_id=self._next_checkpoint_id,
            label=label,
            created_at=time.time(),
            head_sha=self._current_head_sha(),
        )
        self._record_dirty(checkpoint.checkpoint_id, self._capture_dirty_baselines())
        self._next_checkpoint_id += 1
        self._checkpoints.append(checkpoint)
        while len(self._checkpoints) > REWIND_MAX_CHECKPOINTS:
            del self._checkpoints[1]  # keep checkpoint 0, evict the oldest turn
            self._compact_history(self._checkpoints[1].checkpoint_id)
        return checkpoint

    def dirty_paths_at(self, checkpoint_id: int) -> frozenset[str]:
        """Paths that had a dirty baseline when the checkpoint was captured."""
        cached = self._dirty_at_cache
        if cached is not None and cached[0] == checkpoint_id:
            return cached[1]
        paths = frozenset(
            path for path, history in self._history.items() if self._entry_at(history, checkpoint_id) is not None
        )
        self._dirty_at_cache = (checkpoint_id, paths)
        return paths

    def _record_dirty(self, checkpoint_id: int, dirty: dict[str, FileBaseline]) -> None:
        for path, baseline in dirty.items():
            history = self._history.get(path)
            if history is None:
                self._history[path] = [(checkpoint_id, baseline)]
            elif history[-1][1] is not baseline:
                history.append((checkpoint_id, baseline))
                self._multi_version.add(path)
            else:
                continue
            self._acquire(baseline)
        for path in self._dirty_now.difference(dirty):
            self._history[path].append((checkpoint_id, None))
            self._multi_version.add(path)
        self._dirty_now = set(dirty)
        self._dirty_at_cache = None

    def _compact_history(self, floor: int) -> None:
        """Drop history no retained checkpoint can see once turns before ``floor`` are evicted."""
        for path in list(self._multi_version):
            history = self._history[path]
            between = [index for index, (checkpoint_id, _) in enumerate(history) if 0 < checkpoint_id < floor]
            if not between:
                continue
            # The newest entry before the floor is what the floor checkpoint
            # sees, unless the floor checkpoint has its own entry.
            shadowed = any(checkpoint_id == floor for checkpoint_id, _ in history)
            dead = set(between if shadowed else between[:-1])
            if not dead:
                continue
            for index in dead:
                baseline = history[index][1]
                if baseline is not None:
                    self._release(baseline)
            kept = [entry for index, entry in enumerate(history) if index not in dead]
            if all(baseline is None for _, baseline in kept):
                del self._history[path]
                self._multi_version.discard(path)
                continue
            self._history[path] = kept
            if len(kept) == 1:
                self._multi_version.discard(path)
        self._dirty_at_cache = None

    @staticmethod
    def _entry_at(history: list[tuple[int, Optional[FileBaseline]]], checkpoint_id: int) -> Optional[FileBaseline]:
        if len(history) == 1:
            first_id, baseline = history[0]
            return baseline if first_id <= checkpoint_id else None
        index = bisect.bisect_right(history, checkpoint_id, key=lambda entry: entry[0]) - 1
        return history[index][1] if index >= 0 else None

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

    def _acquire(self, baseline: FileBaseline) -> None:
        if self._store is not None and baseline.source == "store" and baseline.sha is not None:
            self._store.acquire(baseline.sha)

    def _release(self, baseline: FileBaseline) -> None:
        if self._store is not None and baseline.source == "store" and baseline.sha is not None:
            self._store.release(baseline.sha)

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
            st = self._stat_path(repo_path)
            sig = None if st is None else (st.st_mtime_ns, st.st_size)
            cached = self._diff_cache.get(repo_path)
            if cached is not None and cached[0] == sig:
                diff = cached[1]
                if diff is None:
                    continue
                baseline = self._checkpoint_baseline(checkpoint, repo_path)
            else:
                baseline = self._checkpoint_baseline(checkpoint, repo_path)
                current = self._snapshot_repo_path(repo_path, st)
                diff = self._build_diff(repo_path, baseline, current)
                self._diff_cache[repo_path] = (sig, diff)
                if diff is None:
                    continue
            records.append(ChangeRecord(repo_path=repo_path, baseline=baseline, sig=sig, diff=diff))
        return records

    def _checkpoint_baseline(self, checkpoint: TurnCheckpoint, repo_path: str) -> FileBaseline:
        history = self._history.get(repo_path)
        baseline = None if history is None else self._entry_at(history, checkpoint.checkpoint_id)
        if baseline is not None:
            return baseline
        return self._fallback_baseline(checkpoint, repo_path)

    # ---- scope ---------------------------------------------------------------

    def scope(self, *, checkpoint_id: Optional[int] = None) -> DiffScope:
        """Describe what the diff covers. May shell out; call it off the UI thread."""
        del checkpoint_id
        return DiffScope(
            root_label=self.project_path.name,
            root_path=str(self.project_path),
            history_tracked=False,
        )

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
                self._forget_cached_state(item.repo_path)
                result.deleted.append(item.display_path)
            except OSError as exc:
                result.errors.append((item.display_path, str(exc)))
        for item in (item for item in plan if item.action == "write"):
            data = self._load_bytes(item.baseline)
            if data is None:
                result.skipped.append((item.display_path, self._unrestorable_reason(item.baseline) or NOT_RETAINED))
                continue
            if item.baseline.sha is not None and hashlib.sha256(data).hexdigest() != item.baseline.sha:
                result.skipped.append((item.display_path, DAMAGED_COPY))
                continue
            abs_path = self._abs_path(item.repo_path)
            try:
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                abs_path.write_bytes(data)
                self._forget_cached_state(item.repo_path)
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

    def _load_source_bytes(self, baseline: FileBaseline) -> Optional[bytes]:
        """Bytes of a baseline whose source is not the store."""
        return None

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

    def _forget_cached_state(self, repo_path: str) -> None:
        """Drop the stat-keyed caches for a path we just rewrote.

        A restore writes bytes of the same length as the file it replaces, so on
        a filesystem whose timestamp tick is coarser than the rewrite the new
        stat signature can match the pre-restore one and the cached content
        would shadow what is now on disk.
        """
        self._drop_snapshot(repo_path)
        self._diff_cache.pop(repo_path, None)

    def _drop_snapshot(self, repo_path: str) -> None:
        cached = self._snapshot_cache.pop(repo_path, None)
        if cached is not None:
            self._release(cached)

    def _stat_path(self, repo_path: str) -> Optional[os.stat_result]:
        # A plain string join: building a Path per file cost 3x the stat
        # itself across tens of thousands of untracked files.
        try:
            return os.stat(f"{self._abs_root}{os.sep}{repo_path}")
        except OSError:
            return None

    def _snapshot_is_current(self, repo_path: str, st: Optional[os.stat_result]) -> bool:
        cached = self._snapshot_cache.get(repo_path)
        return (
            cached is not None and st is not None and S_ISREG(st.st_mode) and cached.sig == (st.st_mtime_ns, st.st_size)
        )

    def _snapshot_repo_path(
        self, repo_path: str, st: Union[os.stat_result, None, _Unset] = _Unset.TOKEN
    ) -> FileBaseline:
        if st is _Unset.TOKEN:
            st = self._stat_path(repo_path)
        if st is None or not S_ISREG(st.st_mode):
            self._drop_snapshot(repo_path)
            return FileBaseline(path=repo_path, exists=False)
        sig = (st.st_mtime_ns, st.st_size)
        cached = self._snapshot_cache.get(repo_path)
        if cached is not None and cached.sig == sig:
            return cached
        if st.st_size > MAX_RETAINED_BYTES:
            baseline = FileBaseline(
                path=repo_path, exists=True, sig=sig, size=st.st_size, unavailable=TOO_LARGE_TO_COPY
            )
        else:
            try:
                data = self._abs_path(repo_path).read_bytes()
            except OSError:
                return FileBaseline(path=repo_path, exists=True, binary_or_unreadable=True)
            baseline = self._retain(self._baseline_from_bytes(repo_path, data, exists=True, sig=sig), data)
        # Retain before dropping: an unchanged rewrite shares the old blob.
        self._drop_snapshot(repo_path)
        self._snapshot_cache[repo_path] = baseline
        return baseline

    def _retain(self, baseline: FileBaseline, data: bytes) -> FileBaseline:
        """Copy a working-tree snapshot into the store, within the budget."""
        if self._store is None or baseline.sha is None:
            return baseline
        refused = self._store.put(baseline.sha, data, limit=self._store_limit)
        if refused:
            return replace(baseline, unavailable=refused)
        return replace(baseline, source="store")

    @staticmethod
    def _baseline_from_bytes(
        repo_path: str, data: bytes, *, exists: bool, sig: Optional[tuple[int, int]] = None
    ) -> FileBaseline:
        return FileBaseline(
            path=repo_path,
            exists=exists,
            binary_or_unreadable=b"\x00" in data,
            sha=hashlib.sha256(data).hexdigest(),
            sig=sig,
            size=len(data),
        )

    def _load_bytes(self, baseline: FileBaseline) -> Optional[bytes]:
        """The bytes a baseline stands for, or None when they were not retained."""
        if not baseline.exists:
            return None
        if baseline.source == "store":
            if self._store is None or baseline.sha is None:
                return None
            return self._store.read(baseline.sha)
        return self._load_source_bytes(baseline)

    def _current_bytes(self, current: FileBaseline) -> Optional[bytes]:
        if current.source == "store":
            data = self._load_bytes(current)
            if data is not None:
                return data
        try:
            return self._abs_path(current.path).read_bytes()
        except OSError:
            return None

    @staticmethod
    def _unrestorable_reason(baseline: FileBaseline) -> str:
        if not baseline.exists or baseline.source != "none":
            return ""
        return baseline.unavailable or NOT_RETAINED

    def _streamed_sha(self, baseline: FileBaseline) -> Optional[str]:
        """sha256 of a large working-tree file, read in chunks rather than whole."""
        if baseline.sig is None:
            return None
        key = (baseline.path, baseline.sig)
        if key in self._streamed_sha_cache:
            return self._streamed_sha_cache[key]
        digest: Optional[str] = None
        try:
            with self._abs_path(baseline.path).open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
        except OSError:
            digest = None
        if self._stat_signature(baseline.path) != baseline.sig:
            digest = None  # the file moved on while it was hashed
        self._streamed_sha_cache[key] = digest
        return digest

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
        unrestorable = self._unrestorable_reason(baseline)

        def without_text(message: str) -> SessionDiffFile:
            return SessionDiffFile(
                path=display_path, status=status, preview=None, message=message, unrestorable=unrestorable
            )

        if baseline.binary_or_unreadable or current.binary_or_unreadable:
            return without_text(_BINARY_MESSAGE)
        if max(baseline.size, current.size) > MAX_DIFF_DECODE_BYTES:
            return without_text("File too large for a textual diff.")
        old = self._load_bytes(baseline) if baseline.exists else b""
        if old is None:
            return without_text("File changed; no checkpoint copy was kept, so a textual diff is unavailable.")
        new = self._current_bytes(current) if current.exists else b""
        if new is None:
            return without_text("File changed; textual diff unavailable.")
        if b"\x00" in old or b"\x00" in new:
            return without_text(_BINARY_MESSAGE)

        old_text = old.decode("utf-8", errors="replace")
        new_text = new.decode("utf-8", errors="replace")
        preview = build_diff_preview(
            old_text, new_text, display_path, max_lines=0, head_max_lines=CHANGES_HEAD_MAX_LINES
        )
        message = "File changed; textual diff unavailable." if preview is None and old_text != new_text else ""
        return SessionDiffFile(
            path=display_path,
            status=status,
            preview=preview,
            adds=int((preview or {}).get("adds") or 0),
            dels=int((preview or {}).get("dels") or 0),
            message=message,
            unrestorable=unrestorable,
        )

    def _same_state(self, left: FileBaseline, right: FileBaseline) -> bool:
        if left.exists != right.exists:
            return False
        if not left.exists:
            return True
        if left.sha is not None and right.sha is not None:
            return left.sha == right.sha
        if left.sig is not None and right.sig is not None:
            return left.sig == right.sig
        left_known = left.sha is not None or left.sig is not None
        right_known = right.sha is not None or right.sig is not None
        if not left_known or not right_known:
            return not left_known and not right_known  # both unreadable
        if left.size != right.size:
            return False
        # A sha-only baseline (commit or ledger) against a large working-tree
        # file that was never hashed.
        sig_side, sha_side = (left, right) if left.sha is None else (right, left)
        return self._streamed_sha(sig_side) == sha_side.sha


class GitSessionDiffTracker(SessionDiffTrackerBase):
    """Track net changes against per-turn checkpoints, for git repos only.

    The tracker avoids whole-repo snapshots. Each checkpoint snapshots only the
    files that are dirty at capture time and pins the HEAD sha; clean files
    baseline against that pinned commit. Pinned commits stay resolvable for the
    tracker's lifetime because they were HEAD during this session and remain
    reflog-reachable.
    """

    def __init__(self, project_path: Path, git_root: Path, *, spool_root: Optional[Path] = None) -> None:
        if spool_root is None:
            from kolega_code.cli.session_store import default_state_dir

            spool_root = default_state_dir() / SPOOL_DIRNAME
        super().__init__(project_path, store=CheckpointBlobStore(spool_root))
        self.git_root = git_root.resolve()
        self._abs_root = str(self.git_root)
        # repo_path -> verdict: _is_under_project resolves symlinks, which is
        # too slow to repeat for tens of thousands of status paths per capture.
        self._under_project_cache: dict[str, bool] = {}
        self._commit_cache: dict[tuple[str, str], FileBaseline] = {}
        # (old_sha, new_sha) -> paths that transition changed. Immutable by
        # construction, so entries never need invalidating.
        self._transition_paths_cache: dict[tuple[str, str], frozenset[str]] = {}
        # Recorded by the last _session_committed_paths walk, for scope().
        self._history_moved = False
        self._history_tracked = True
        self._history_state: Optional[tuple[int, str]] = None  # (checkpoint_id, head) the flags describe

    @classmethod
    def create(cls, project_path: Path, *, spool_root: Optional[Path] = None) -> "GitSessionDiffTracker | None":
        git_root = cls._detect_git_root(project_path)
        if git_root is None:
            return None
        return cls(project_path, git_root, spool_root=spool_root)

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
                # Without this, an exported GIT_DIR/GIT_WORK_TREE resolves to
                # another worktree and the whole tracker targets the wrong tree.
                env=git_env(),
            )
        except (OSError, ValueError):
            return None
        if completed.returncode != 0:
            return None
        root = completed.stdout.strip()
        return Path(root).resolve() if root else None

    # ---- git plumbing --------------------------------------------------------

    def _git(self, args: Sequence[str], *, text: bool = False) -> Optional[CompletedProcess]:
        """Run git in the tracked worktree with a sanitized environment.

        Returns ``None`` when git could not be run or exited non-zero, so every
        caller degrades to "no information" instead of raising.
        """
        try:
            completed = subprocess_run(
                ["git", *args],
                cwd=str(self.git_root),
                stdout=PIPE,
                stderr=DEVNULL,
                text=text,
                check=False,
                env=git_env(),
            )
        except (OSError, ValueError):
            return None
        return completed if completed.returncode == 0 else None

    def _git_names(self, args: Sequence[str]) -> set[str]:
        """Run a NUL-separated name-only git command and return its paths."""
        completed = self._git(args)
        if completed is None:
            return set()
        raw: bytes = completed.stdout
        return {name for name in raw.decode("utf-8", errors="surrogateescape").split("\0") if name}

    # ---- provider hooks ------------------------------------------------------

    def _dirty_entries(self) -> dict[str, bool]:
        """Status paths the session can ever display or restore, mapped to "untracked".

        Sibling-worktree and out-of-project paths are dropped before any file
        is read: they are unreachable from diff and restore output anyway.
        """
        entries = self._git_status_entries()
        return {
            path: entries[path]
            for path in self._without_sibling_worktrees(set(entries))
            if self._is_under_project(path)
        }

    def _capture_dirty_baselines(self) -> dict[str, FileBaseline]:
        entries = self._dirty_entries()
        stats = {path: self._stat_path(path) for path in entries}
        baselines: dict[str, FileBaseline] = {}
        unread: list[str] = []
        for path, st in stats.items():
            if self._snapshot_is_current(path, st):
                baselines[path] = self._snapshot_repo_path(path, st)
            else:
                unread.append(path)

        def priority(path: str) -> tuple[bool, int, str]:
            st = stats[path]
            return (entries[path], st.st_size if st is not None else 0, path)

        # The store budget rations copies: uncommitted edits to tracked files
        # first, then untracked files smallest-first.
        for path in sorted(unread, key=priority):
            baselines[path] = self._snapshot_repo_path(path, stats[path])
        return baselines

    def _candidate_paths(self, checkpoint: TurnCheckpoint, event_paths: Iterable[str]) -> set[str]:
        candidates = self._git_status_paths() | self.dirty_paths_at(checkpoint.checkpoint_id)
        candidates.update(self._repo_paths_from_event_paths(event_paths))
        candidates.update(self._session_committed_paths(checkpoint))
        return self._without_sibling_worktrees(candidates)

    def _without_sibling_worktrees(self, candidates: set[str]) -> set[str]:
        """Drop paths that live inside another worktree nested in this one.

        Sibling worktrees are normally kept out of git status by the shared
        ``info/exclude`` rule, but that rule is best-effort. Filtering by the
        registered worktree roots makes the isolation guarantee unconditional.
        """
        prefixes = self._nested_worktree_prefixes()
        if not prefixes:
            return candidates
        return {
            path
            for path in candidates
            if not any(path == prefix or path.startswith(f"{prefix}/") for prefix in prefixes)
        }

    def _nested_worktree_prefixes(self) -> list[str]:
        """Repo-relative prefixes of other worktrees nested inside this one."""
        prefixes: list[str] = []
        for worktree in list_worktrees(self.git_root):
            if worktree.path == self.git_root:
                continue
            try:
                rel = worktree.path.relative_to(self.git_root)
            except ValueError:
                continue  # a sibling elsewhere on disk cannot collide with our paths
            prefix = self._posix(rel)
            if prefix:
                prefixes.append(prefix)
        return prefixes

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

    def scope(self, *, checkpoint_id: Optional[int] = None) -> DiffScope:
        """Describe the tracked worktree. Shells out; call it off the UI thread."""
        checkpoint = self._resolve_checkpoint(checkpoint_id)
        # The history flags come from the reflog walk. Reuse the ones a refresh
        # for this same checkpoint and HEAD already produced.
        if self._history_state != (checkpoint.checkpoint_id, self._current_head_sha()):
            self._session_committed_paths(checkpoint)
        worktrees = list_worktrees(self.git_root)
        main_root = worktrees[0].path if worktrees else self.git_root
        linked = bool(worktrees) and self.git_root != main_root
        # A repository that has never had a commit has no reflog to read; that
        # is not a degraded state worth reporting.
        history_tracked = self._history_tracked or not (checkpoint.head_sha or self._current_head_sha())
        return DiffScope(
            root_label=self.git_root.name if linked else "",
            root_path=str(self.git_root),
            branch=self._branch_label(),
            linked_worktree=linked,
            history_moved=self._history_moved,
            history_tracked=history_tracked,
        )

    def _branch_label(self) -> str:
        completed = self._git(["symbolic-ref", "--quiet", "--short", "HEAD"], text=True)
        if completed is not None:
            branch = str(completed.stdout).strip()
            if branch:
                return branch
        head = self._current_head_sha()
        return f"detached at {head[:7]}" if head else ""

    def _abs_path(self, repo_path: str) -> Path:
        return self.git_root / repo_path

    def _display_path(self, repo_path: str) -> str:
        try:
            rel = (self.git_root / repo_path).resolve(strict=False).relative_to(self.project_path)
            return self._posix(rel)
        except ValueError:
            return repo_path

    def _is_under_project(self, repo_path: str) -> bool:
        cached = self._under_project_cache.get(repo_path)
        if cached is not None:
            return cached
        try:
            (self.git_root / repo_path).resolve(strict=False).relative_to(self.project_path)
            verdict = True
        except ValueError:
            verdict = False
        self._under_project_cache[repo_path] = verdict
        return verdict

    # ---- git helpers ---------------------------------------------------------

    def _git_status_paths(self) -> set[str]:
        return set(self._git_status_entries())

    def _git_status_entries(self) -> dict[str, bool]:
        """Status paths mapped to whether git reports them untracked."""
        completed = self._git(
            [
                "--no-optional-locks",  # never contend on index.lock with the agent's own git
                "-c",
                "core.quotePath=false",
                "status",
                "--porcelain",
                "-z",
                # -uall lists files inside untracked directories individually;
                # the default would report only "dir/" and hide the files.
                "-uall",
            ]
        )
        if completed is None:
            return {}
        return self._parse_porcelain_z(completed.stdout)

    @staticmethod
    def _parse_porcelain_z(output: bytes) -> dict[str, bool]:
        paths: dict[str, bool] = {}
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
                paths[path] = status == "??"
            # In porcelain v1 -z, rename/copy records are followed by the other path.
            if status[0] in {"R", "C"} or status[1] in {"R", "C"}:
                index += 1
                if index < len(entries) and entries[index]:
                    paths.setdefault(entries[index], False)
            index += 1
        return paths

    def _commit_baseline(self, commit_sha: str, repo_path: str) -> FileBaseline:
        if not commit_sha:
            return FileBaseline(path=repo_path, exists=False)
        completed = self._git(["show", f"{commit_sha}:{repo_path}"])
        if completed is None:
            return FileBaseline(path=repo_path, exists=False)
        # Only the digest is kept; the bytes are read back from the commit on demand.
        baseline = self._baseline_from_bytes(repo_path, completed.stdout, exists=True)
        return replace(baseline, source="commit", ref=commit_sha)

    def _load_source_bytes(self, baseline: FileBaseline) -> Optional[bytes]:
        if baseline.source != "commit" or not isinstance(baseline.ref, str):
            return None
        completed = self._git(["show", f"{baseline.ref}:{baseline.path}"])
        return None if completed is None else completed.stdout

    def _current_head_sha(self) -> str:
        completed = self._git(["rev-parse", "HEAD"], text=True)
        if completed is None:
            return ""
        return str(completed.stdout).strip()

    # ---- committed-change attribution ----------------------------------------

    def _session_committed_paths(self, checkpoint: TurnCheckpoint) -> set[str]:
        """Paths changed by commits this session created in this worktree.

        Without this, a change that gets committed (and is then clean in git
        status) would vanish from both the diff view and the restore plan.

        A plain ``git diff checkpoint..HEAD`` cannot be used: any merge, pull,
        or rebase would attribute every incoming file — typically the work of
        another agent in a sibling worktree — to this session, and the restore
        plan would then offer to revert it. Instead this walks the worktree's
        own HEAD reflog (``logs/HEAD`` lives in the linked worktree's gitdir)
        and credits only transitions that created a commit here.
        """
        self._history_moved = False
        self._history_tracked = True

        head = self._current_head_sha()
        self._history_state = (checkpoint.checkpoint_id, head)
        if checkpoint.head_sha and head and head == checkpoint.head_sha:
            return set()

        entries = self._reflog_entries()
        if entries is None:
            self._history_tracked = False
            return set()

        paths: set[str] = set()
        floor = int(checkpoint.created_at) - 1  # reflog timestamps have 1s granularity
        for index, entry in enumerate(entries):
            if entry.sha == checkpoint.head_sha:
                break  # HEAD was in this state at capture; everything older is pre-session
            if entry.timestamp and entry.timestamp < floor:
                break
            old = entries[index + 1].sha if index + 1 < len(entries) else checkpoint.head_sha
            if old == entry.sha:
                continue  # a no-op move such as "rebase (finish)"
            paths.update(self._attributed_paths(entry, old))
        return set(sorted(paths)[:MAX_COMMITTED_CANDIDATES])

    def _attributed_paths(self, entry: _ReflogEntry, old: str) -> frozenset[str]:
        """Paths this reflog transition contributed, crediting only local work."""
        kind = entry.subject.split(":", 1)[0].strip()
        merge_commit = len(entry.parents) >= 2
        if kind in _COMMIT_REFLOG_KINDS:
            if merge_commit:
                # A merge finished by `git commit`: credit only the conflict
                # resolution authored here, not the incoming branch.
                self._history_moved = True
                return self._combined_diff_names(entry.sha)
            return self._transition_diff_names(old, entry.sha)
        if kind in _MERGE_REFLOG_KINDS or kind.startswith(("merge ", "pull ")):
            self._history_moved = True
            return self._combined_diff_names(entry.sha) if merge_commit else frozenset()
        # checkout, reset, rebase (start)/(finish), clone, branch, unknown: the
        # content came from elsewhere, so it is not this session's change.
        self._history_moved = True
        return frozenset()

    def _reflog_entries(self) -> Optional[list[_ReflogEntry]]:
        """This worktree's HEAD reflog, newest first, or None when unreadable."""
        completed = self._git(
            [
                "log",
                "-g",
                "-z",
                "--date=unix",
                "-n",
                str(MAX_REFLOG_ENTRIES),
                "--format=%H%x1f%P%x1f%gd%x1f%gs",
                "HEAD",
            ]
        )
        if completed is None:
            return None
        raw: bytes = completed.stdout
        entries: list[_ReflogEntry] = []
        for record in raw.decode("utf-8", errors="surrogateescape").split("\0"):
            record = record.strip("\n")
            if not record:
                continue
            fields = record.split("\x1f")
            fields.extend([""] * (4 - len(fields)))
            entries.append(
                _ReflogEntry(
                    sha=fields[0],
                    parents=tuple(fields[1].split()),
                    timestamp=self._reflog_timestamp(fields[2]),
                    subject=fields[3],
                )
            )
        return entries

    @staticmethod
    def _reflog_timestamp(selector: str) -> int:
        """Seconds from a ``HEAD@{<unix>}`` selector; 0 when it cannot be parsed."""
        start = selector.find("{")
        if start < 0 or not selector.endswith("}"):
            return 0
        try:
            return int(selector[start + 1 : -1])
        except ValueError:
            return 0

    def _transition_diff_names(self, old: str, new: str) -> frozenset[str]:
        key = (old, new)
        cached = self._transition_paths_cache.get(key)
        if cached is not None:
            return cached
        if old:
            names = self._git_names(["diff", "--name-only", "-z", old, new])
        else:
            # An initial commit made mid-session has no predecessor to diff
            # against. --no-commit-id is required or the sha is emitted as a path.
            names = self._git_names(["diff-tree", "--root", "--no-commit-id", "--name-only", "-z", "-r", new])
        result = frozenset(names)
        self._transition_paths_cache[key] = result
        return result

    def _combined_diff_names(self, merge_sha: str) -> frozenset[str]:
        """Files a merge commit changed relative to *every* parent.

        That is exactly the conflict resolution authored in this worktree; a
        clean automatic merge contributes nothing.
        """
        key = ("--cc", merge_sha)
        cached = self._transition_paths_cache.get(key)
        if cached is not None:
            return cached
        result = frozenset(
            self._git_names(["diff-tree", "--cc", "--no-commit-id", "--name-only", "-z", "-r", merge_sha])
        )
        self._transition_paths_cache[key] = result
        return result

    def _is_ignored(self, repo_path: str) -> bool:
        try:
            completed = subprocess_run(
                ["git", "check-ignore", "-q", "--", repo_path],
                cwd=str(self.git_root),
                stdout=DEVNULL,
                stderr=DEVNULL,
                check=False,
                env=git_env(),
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
        if state is None or state.kind == "missing":
            return FileBaseline(path=repo_path, exists=False)
        if state.kind != "file" or state.sha256 is None:
            return FileBaseline(path=repo_path, exists=True, binary_or_unreadable=True)
        return FileBaseline(path=repo_path, exists=True, sha=state.sha256, size=state.size, source="ledger", ref=state)

    def _load_source_bytes(self, baseline: FileBaseline) -> Optional[bytes]:
        from kolega_code.services.snapshots import SnapshotError

        if baseline.source != "ledger" or baseline.ref is None or isinstance(baseline.ref, str):
            return None
        try:
            return self._service.read_blob(baseline.ref)
        except SnapshotError:
            return None
