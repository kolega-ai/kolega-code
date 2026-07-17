"""Private workspace snapshots and pending preview actions.

The snapshot store is deliberately outside the user's project tree. It records
pre/post file states for trusted mutating tools so an agent edit can be undone
without creating commits, refs, or repo-local metadata.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Generic, Iterable, Literal, TypeVar

from kolega_code.cli.session_store import default_state_dir
from kolega_code.local_state import PRIVATE_FILE_MODE, ensure_private_dir, write_private_text
from kolega_code.services.file_system import FileSystem


SnapshotKind = Literal["missing", "file", "directory"]
PendingStatus = Literal["pending", "applied", "discarded", "stale"]

T = TypeVar("T")


class SnapshotError(RuntimeError):
    """Raised when snapshot, pending-action, or restore operations fail."""


@dataclass(frozen=True)
class FileState:
    path: str
    kind: SnapshotKind
    sha256: str | None = None
    size: int = 0
    blob_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "sha256": self.sha256,
            "size": self.size,
            "blob_id": self.blob_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileState":
        return cls(
            path=str(data["path"]),
            kind=data["kind"],
            sha256=data.get("sha256"),
            size=int(data.get("size") or 0),
            blob_id=data.get("blob_id"),
        )


@dataclass(frozen=True)
class SnapshotRecord:
    snapshot_id: str
    session_id: str
    workspace_id: str
    thread_id: str
    project_path: str
    tool_name: str
    tool_call_id: str | None
    reason: str
    created_at: str
    touched_paths: tuple[str, ...]
    before: dict[str, FileState]
    after: dict[str, FileState]
    manual: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "snapshot_id": self.snapshot_id,
            "session_id": self.session_id,
            "workspace_id": self.workspace_id,
            "thread_id": self.thread_id,
            "project_path": self.project_path,
            "tool_name": self.tool_name,
            "tool_call_id": self.tool_call_id,
            "reason": self.reason,
            "created_at": self.created_at,
            "touched_paths": list(self.touched_paths),
            "before": {path: state.to_dict() for path, state in self.before.items()},
            "after": {path: state.to_dict() for path, state in self.after.items()},
            "manual": self.manual,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SnapshotRecord":
        if data.get("schema_version") != 1:
            raise SnapshotError(f"Unsupported snapshot schema version: {data.get('schema_version')}")
        return cls(
            snapshot_id=str(data["snapshot_id"]),
            session_id=str(data["session_id"]),
            workspace_id=str(data.get("workspace_id") or ""),
            thread_id=str(data.get("thread_id") or ""),
            project_path=str(data.get("project_path") or ""),
            tool_name=str(data.get("tool_name") or ""),
            tool_call_id=data.get("tool_call_id"),
            reason=str(data.get("reason") or ""),
            created_at=str(data["created_at"]),
            touched_paths=tuple(str(path) for path in data.get("touched_paths") or ()),
            before={path: FileState.from_dict(state) for path, state in (data.get("before") or {}).items()},
            after={path: FileState.from_dict(state) for path, state in (data.get("after") or {}).items()},
            manual=bool(data.get("manual", False)),
        )


@dataclass(frozen=True)
class PendingAction:
    action_id: str
    session_id: str
    workspace_id: str
    thread_id: str
    project_path: str
    tool_name: str
    tool_call_id: str | None
    operation: str
    created_at: str
    updated_at: str
    status: PendingStatus
    touched_paths: tuple[str, ...]
    source: dict[str, FileState]
    workspace_edit: dict[str, Any]
    summaries: tuple[str, ...] = ()
    previews: tuple[dict[str, Any], ...] = ()
    snapshot_id: str | None = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "action_id": self.action_id,
            "session_id": self.session_id,
            "workspace_id": self.workspace_id,
            "thread_id": self.thread_id,
            "project_path": self.project_path,
            "tool_name": self.tool_name,
            "tool_call_id": self.tool_call_id,
            "operation": self.operation,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "touched_paths": list(self.touched_paths),
            "source": {path: state.to_dict() for path, state in self.source.items()},
            "workspace_edit": self.workspace_edit,
            "summaries": list(self.summaries),
            "previews": list(self.previews),
            "snapshot_id": self.snapshot_id,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PendingAction":
        if data.get("schema_version") != 1:
            raise SnapshotError(f"Unsupported pending action schema version: {data.get('schema_version')}")
        return cls(
            action_id=str(data["action_id"]),
            session_id=str(data["session_id"]),
            workspace_id=str(data.get("workspace_id") or ""),
            thread_id=str(data.get("thread_id") or ""),
            project_path=str(data.get("project_path") or ""),
            tool_name=str(data.get("tool_name") or ""),
            tool_call_id=data.get("tool_call_id"),
            operation=str(data.get("operation") or ""),
            created_at=str(data["created_at"]),
            updated_at=str(data.get("updated_at") or data["created_at"]),
            status=data.get("status") or "pending",
            touched_paths=tuple(str(path) for path in data.get("touched_paths") or ()),
            source={path: FileState.from_dict(state) for path, state in (data.get("source") or {}).items()},
            workspace_edit=data.get("workspace_edit") or {},
            summaries=tuple(str(item) for item in data.get("summaries") or ()),
            previews=tuple(dict(item) for item in data.get("previews") or ()),
            snapshot_id=data.get("snapshot_id"),
            message=str(data.get("message") or ""),
        )


@dataclass(frozen=True)
class RestoreResult:
    snapshot_id: str
    restored_paths: tuple[str, ...]
    forced: bool


@dataclass(frozen=True)
class MutationRecordResult(Generic[T]):
    result: T
    snapshot: SnapshotRecord | None


@dataclass(frozen=True)
class StateMismatch:
    path: str
    expected: FileState
    actual: FileState


class SnapshotService:
    """Session-scoped snapshot and pending-action store."""

    def __init__(
        self,
        project_path: Path,
        workspace_id: str,
        thread_id: str,
        session_id: str,
        filesystem: FileSystem,
        *,
        root: Path | None = None,
    ) -> None:
        self.project_path = project_path.resolve()
        self.workspace_id = workspace_id
        self.thread_id = thread_id
        self.session_id = session_id
        self.filesystem = filesystem
        self.root = (root or default_state_dir()).expanduser() / "snapshots" / session_id
        self.blobs_dir = self.root / "blobs"
        self.snapshots_dir = self.root / "records"
        self.pending_dir = self.root / "pending"

    def ensure_dirs(self) -> None:
        ensure_private_dir(self.root)
        ensure_private_dir(self.blobs_dir)
        ensure_private_dir(self.snapshots_dir)
        ensure_private_dir(self.pending_dir)

    def capture_paths(self, paths: Iterable[str]) -> dict[str, FileState]:
        states: dict[str, FileState] = {}
        for raw_path in paths:
            path = self.normalize_path(raw_path)
            self._capture_path(path, states)
        return dict(sorted(states.items()))

    def can_snapshot_paths(self, paths: Iterable[str]) -> bool:
        try:
            for path in paths:
                self.normalize_path(path)
        except SnapshotError:
            return False
        return True

    def record_mutation(
        self,
        *,
        tool_name: str,
        tool_call_id: str | None,
        reason: str,
        paths: Iterable[str],
        mutate: Callable[[], T],
    ) -> MutationRecordResult[T]:
        touched_paths = tuple(dict.fromkeys(self.normalize_path(path) for path in paths))
        before = self.capture_paths(touched_paths)
        result = mutate()
        after = self.capture_paths(touched_paths)
        snapshot = None
        if not self._states_equal(before, after):
            snapshot = self._write_snapshot(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                reason=reason,
                touched_paths=touched_paths,
                before=before,
                after=after,
                manual=False,
            )
        return MutationRecordResult(result=result, snapshot=snapshot)

    def create_manual_snapshot(
        self,
        *,
        paths: Iterable[str],
        reason: str = "manual snapshot",
        tool_call_id: str | None = None,
    ) -> SnapshotRecord:
        touched_paths = tuple(dict.fromkeys(self.normalize_path(path) for path in paths))
        if not touched_paths:
            raise SnapshotError("At least one path is required to create a manual snapshot.")
        current = self.capture_paths(touched_paths)
        return self._write_snapshot(
            tool_name="snapshot",
            tool_call_id=tool_call_id,
            reason=reason,
            touched_paths=touched_paths,
            before=current,
            after=current,
            manual=True,
        )

    def restore_snapshot(self, snapshot_id: str, *, force: bool = False) -> RestoreResult:
        record = self.load_snapshot(snapshot_id)
        expected = record.after
        if not force and not record.manual:
            mismatches = self.diff_expected_current(expected)
            if mismatches:
                raise SnapshotError(self._format_mismatch_error(record.snapshot_id, mismatches))
        self._apply_states(record.before)
        return RestoreResult(
            snapshot_id=record.snapshot_id,
            restored_paths=record.touched_paths,
            forced=force,
        )

    def diff_expected_current(self, expected: dict[str, FileState]) -> list[StateMismatch]:
        if not expected:
            return []
        current = self.capture_paths(expected.keys())
        mismatches: list[StateMismatch] = []
        for path in sorted(set(expected) | set(current)):
            expected_state = expected.get(path) or FileState(path=path, kind="missing")
            actual_state = current.get(path) or FileState(path=path, kind="missing")
            if expected_state != actual_state:
                mismatches.append(StateMismatch(path=path, expected=expected_state, actual=actual_state))
        return mismatches

    def list_snapshots(self, *, limit: int = 20) -> list[SnapshotRecord]:
        self.ensure_dirs()
        records: list[SnapshotRecord] = []
        for path in self.snapshots_dir.glob("*.json"):
            try:
                records.append(SnapshotRecord.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except Exception:
                continue
        records.sort(key=lambda item: item.created_at, reverse=True)
        return records[: max(1, limit)]

    def latest_snapshot(self) -> SnapshotRecord | None:
        records = self.list_snapshots(limit=1)
        return records[0] if records else None

    def load_snapshot(self, snapshot_id: str) -> SnapshotRecord:
        if snapshot_id == "latest":
            latest = self.latest_snapshot()
            if latest is None:
                raise SnapshotError("No snapshots exist for this session.")
            return latest
        path = self.snapshots_dir / f"{snapshot_id}.json"
        if not path.exists():
            raise SnapshotError(f"Snapshot not found: {snapshot_id}")
        return SnapshotRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def read_blob(self, state: FileState) -> bytes:
        """Return the stored bytes for a captured file state."""
        return self._read_blob(state)

    def create_pending_workspace_edit(
        self,
        *,
        tool_name: str,
        tool_call_id: str | None,
        operation: str,
        workspace_edit: dict[str, Any],
        touched_paths: Iterable[str],
        summaries: Iterable[str] = (),
        previews: Iterable[dict[str, Any]] = (),
    ) -> PendingAction:
        normalized_paths = tuple(dict.fromkeys(self.normalize_path(path) for path in touched_paths))
        source = self.capture_paths(normalized_paths)
        now = _now()
        action = PendingAction(
            action_id=_new_id("act"),
            session_id=self.session_id,
            workspace_id=self.workspace_id,
            thread_id=self.thread_id,
            project_path=str(self.project_path),
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            operation=operation,
            created_at=now,
            updated_at=now,
            status="pending",
            touched_paths=normalized_paths,
            source=source,
            workspace_edit=workspace_edit,
            summaries=tuple(summaries),
            previews=tuple(previews),
        )
        self._write_pending(action)
        return action

    def list_pending_actions(self, *, include_resolved: bool = False, limit: int = 20) -> list[PendingAction]:
        self.ensure_dirs()
        actions: list[PendingAction] = []
        for path in self.pending_dir.glob("*.json"):
            try:
                action = PendingAction.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
            if include_resolved or action.status == "pending":
                actions.append(action)
        actions.sort(key=lambda item: item.updated_at, reverse=True)
        return actions[: max(1, limit)]

    def load_pending_action(self, action_id: str) -> PendingAction:
        path = self.pending_dir / f"{action_id}.json"
        if not path.exists():
            raise SnapshotError(f"Pending action not found: {action_id}")
        return PendingAction.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def update_pending_action(
        self,
        action: PendingAction,
        *,
        status: PendingStatus,
        snapshot_id: str | None = None,
        message: str = "",
    ) -> PendingAction:
        updated = PendingAction(
            action_id=action.action_id,
            session_id=action.session_id,
            workspace_id=action.workspace_id,
            thread_id=action.thread_id,
            project_path=action.project_path,
            tool_name=action.tool_name,
            tool_call_id=action.tool_call_id,
            operation=action.operation,
            created_at=action.created_at,
            updated_at=_now(),
            status=status,
            touched_paths=action.touched_paths,
            source=action.source,
            workspace_edit=action.workspace_edit,
            summaries=action.summaries,
            previews=action.previews,
            snapshot_id=snapshot_id if snapshot_id is not None else action.snapshot_id,
            message=message,
        )
        self._write_pending(updated)
        return updated

    def normalize_path(self, path: str) -> str:
        if not path:
            raise SnapshotError("Snapshot path must not be empty.")
        candidate = Path(path)
        if candidate.is_absolute():
            try:
                rel = candidate.resolve(strict=False).relative_to(self.project_path)
            except ValueError as exc:
                raise SnapshotError(f"Snapshot path is outside the project: {path}") from exc
        else:
            try:
                rel = (self.project_path / candidate).resolve(strict=False).relative_to(self.project_path)
            except ValueError as exc:
                raise SnapshotError(f"Snapshot path is outside the project: {path}") from exc
        normalized = PurePosixPath(rel.as_posix()).as_posix()
        if not normalized or normalized == ".":
            raise SnapshotError("Snapshot path must not be the project root.")
        return normalized

    def _capture_path(self, path: str, states: dict[str, FileState]) -> None:
        if path in states:
            return
        if not self.filesystem.exists(path):
            states[path] = FileState(path=path, kind="missing")
            return
        if self.filesystem.is_file(path):
            data = self.filesystem.read_bytes(path)
            digest = sha256(data).hexdigest()
            blob_id = self._write_blob(digest, data)
            states[path] = FileState(path=path, kind="file", sha256=digest, size=len(data), blob_id=blob_id)
            return
        if self.filesystem.is_dir(path):
            states[path] = FileState(path=path, kind="directory")
            for name in sorted(self.filesystem.listdir(path)):
                child = f"{path.rstrip('/')}/{name}"
                self._capture_path(child, states)
            return
        raise SnapshotError(f"Cannot snapshot unsupported filesystem entry: {path}")

    def _apply_states(self, states: dict[str, FileState]) -> None:
        for state in sorted(
            (s for s in states.values() if s.kind == "missing"), key=lambda item: _depth(item.path), reverse=True
        ):
            self._remove_existing(state.path)
        for state in sorted((s for s in states.values() if s.kind == "directory"), key=lambda item: _depth(item.path)):
            if self.filesystem.exists(state.path) and not self.filesystem.is_dir(state.path):
                self._remove_existing(state.path)
            if not self.filesystem.exists(state.path):
                self.filesystem.mkdir(state.path, parents=True, exist_ok=True)
        for state in sorted((s for s in states.values() if s.kind == "file"), key=lambda item: _depth(item.path)):
            parent = self.filesystem.get_parent(state.path)
            if parent and parent != "." and not self.filesystem.exists(parent):
                self.filesystem.mkdir(parent, parents=True, exist_ok=True)
            if self.filesystem.exists(state.path) and self.filesystem.is_dir(state.path):
                self.filesystem.rmtree(state.path)
            self.filesystem.write_bytes(state.path, self._read_blob(state))

    def _remove_existing(self, path: str) -> None:
        if not self.filesystem.exists(path):
            return
        if self.filesystem.is_dir(path):
            self.filesystem.rmtree(path)
        else:
            self.filesystem.remove(path, missing_ok=True)

    def _write_snapshot(
        self,
        *,
        tool_name: str,
        tool_call_id: str | None,
        reason: str,
        touched_paths: tuple[str, ...],
        before: dict[str, FileState],
        after: dict[str, FileState],
        manual: bool,
    ) -> SnapshotRecord:
        record = SnapshotRecord(
            snapshot_id=_new_id("snap"),
            session_id=self.session_id,
            workspace_id=self.workspace_id,
            thread_id=self.thread_id,
            project_path=str(self.project_path),
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            reason=reason,
            created_at=_now(),
            touched_paths=touched_paths,
            before=before,
            after=after,
            manual=manual,
        )
        self.ensure_dirs()
        write_private_text(
            self.snapshots_dir / f"{record.snapshot_id}.json", json.dumps(record.to_dict(), indent=2) + "\n"
        )
        return record

    def _write_pending(self, action: PendingAction) -> None:
        self.ensure_dirs()
        write_private_text(self.pending_dir / f"{action.action_id}.json", json.dumps(action.to_dict(), indent=2) + "\n")

    def _write_blob(self, digest: str, data: bytes) -> str:
        self.ensure_dirs()
        path = self.blobs_dir / digest
        if not path.exists():
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, PRIVATE_FILE_MODE)
            try:
                with os.fdopen(fd, "wb", closefd=False) as handle:
                    handle.write(data)
            except Exception:
                try:
                    path.unlink(missing_ok=True)
                finally:
                    raise
            finally:
                os.close(fd)
            try:
                os.chmod(path, PRIVATE_FILE_MODE)
            except OSError:
                pass
        return digest

    def _read_blob(self, state: FileState) -> bytes:
        if not state.blob_id:
            raise SnapshotError(f"Missing blob id for file state: {state.path}")
        path = self.blobs_dir / state.blob_id
        if not path.exists():
            raise SnapshotError(f"Snapshot blob is missing for {state.path}: {state.blob_id}")
        data = path.read_bytes()
        digest = sha256(data).hexdigest()
        if digest != state.sha256:
            raise SnapshotError(f"Snapshot blob hash mismatch for {state.path}")
        return data

    @staticmethod
    def _states_equal(left: dict[str, FileState], right: dict[str, FileState]) -> bool:
        return left == right

    @staticmethod
    def _format_mismatch_error(snapshot_id: str, mismatches: list[StateMismatch]) -> str:
        lines = [
            f"Snapshot {snapshot_id} cannot be restored because tracked files changed after the snapshot.",
            "Use force=true only if you intentionally want to overwrite current changes.",
        ]
        for mismatch in mismatches[:8]:
            lines.append(
                "- "
                + mismatch.path
                + f": expected {mismatch.expected.kind}:{mismatch.expected.sha256 or '-'}"
                + f", found {mismatch.actual.kind}:{mismatch.actual.sha256 or '-'}"
            )
        if len(mismatches) > 8:
            lines.append(f"- ... {len(mismatches) - 8} more mismatch(es)")
        return "\n".join(lines)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _depth(path: str) -> int:
    return len(PurePosixPath(path).parts)
