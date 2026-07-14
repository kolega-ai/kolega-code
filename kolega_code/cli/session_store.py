"""Local resumable session storage for the Kolega Code CLI."""

from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from kolega_code.local_state import ensure_private_dir, write_private_text

from .session_journal import (
    SessionEvent,
    SessionJournal,
    SessionJournalError,
    SessionRecorder,
)

SCHEMA_VERSION = 1


class SessionStoreError(RuntimeError):
    """Raised when a CLI session cannot be loaded or saved."""


@dataclass
class SessionRecord:
    session_id: str
    project_path: str
    workspace_id: str
    thread_id: str
    mode: str
    title: str
    created_at: str
    updated_at: str
    config: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    compaction: dict[str, Any] = field(default_factory=dict)
    task_list_markdown: str = ""
    latest_plan_markdown: str = ""
    plan_pending: bool = False
    plan_reofferable: bool = False
    interaction_mode: str = "build"
    permission_mode: str = "ask"
    gigacode_enabled: bool = False
    goal: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        project_path: Path,
        mode: str,
        config: dict[str, Any],
        session_id: Optional[str] = None,
        title: Optional[str] = None,
    ) -> "SessionRecord":
        now = _now()
        resolved_project = str(project_path.resolve())
        session_id = session_id or uuid.uuid4().hex
        return cls(
            schema_version=SCHEMA_VERSION,
            session_id=session_id,
            project_path=resolved_project,
            workspace_id=f"cli-{uuid.uuid4().hex}",
            thread_id=uuid.uuid4().hex,
            mode=mode,
            title=title or Path(resolved_project).name or "Kolega Code",
            created_at=now,
            updated_at=now,
            config=config,
            history=[],
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionRecord":
        if data.get("schema_version") != SCHEMA_VERSION:
            raise SessionStoreError(f"Unsupported session schema version: {data.get('schema_version')}")
        latest_plan_markdown = data.get("latest_plan_markdown") or ""
        plan_pending = bool(latest_plan_markdown and data.get("plan_pending", False))
        return cls(
            schema_version=data["schema_version"],
            session_id=data["session_id"],
            project_path=data["project_path"],
            workspace_id=data["workspace_id"],
            thread_id=data["thread_id"],
            mode=data["mode"],
            title=data.get("title") or "Kolega Code",
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            config=data.get("config") or {},
            history=data.get("history") or [],
            compaction=data.get("compaction") or {},
            task_list_markdown=data.get("task_list_markdown") or "",
            latest_plan_markdown=latest_plan_markdown,
            plan_pending=plan_pending,
            plan_reofferable=bool(latest_plan_markdown and data.get("plan_reofferable", plan_pending)),
            interaction_mode=data.get("interaction_mode") or "build",
            permission_mode=data.get("permission_mode") or "ask",
            gigacode_enabled=bool(data.get("gigacode_enabled", False)),
            goal=data.get("goal") or {},
        )

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "project_path": self.project_path,
            "workspace_id": self.workspace_id,
            "thread_id": self.thread_id,
            "mode": self.mode,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "config": self.config,
            "task_list_markdown": self.task_list_markdown,
            "latest_plan_markdown": self.latest_plan_markdown,
            "plan_pending": self.plan_pending,
            "plan_reofferable": self.plan_reofferable,
            "interaction_mode": self.interaction_mode,
            "permission_mode": self.permission_mode,
            "gigacode_enabled": self.gigacode_enabled,
            "goal": self.goal,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.to_metadata_dict(),
            "history": self.history,
            "compaction": self.compaction,
        }


@dataclass(frozen=True)
class SessionBugExport:
    session_json: str
    events_jsonl: str
    artifact_manifest_json: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_session_id(session_id: str) -> str:
    if not session_id or session_id in {".", ".."} or "/" in session_id or "\\" in session_id or "\0" in session_id:
        raise SessionStoreError("Session id must be a single non-empty path component")
    return session_id


def default_state_dir(env: Optional[dict[str, str]] = None) -> Path:
    env = env or dict(os.environ)
    if env.get("KOLEGA_CODE_STATE_DIR"):
        return Path(env["KOLEGA_CODE_STATE_DIR"]).expanduser()

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "kolega-code"
    if sys.platform.startswith("win"):
        base = Path(env.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return base / "kolega-code"
    return Path(env.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "kolega-code"


class SessionStore:
    """Filesystem-backed sessions with JSONL as the canonical history."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = (root or default_state_dir()).expanduser()
        self.sessions_dir = self.root / "sessions"
        self._locks_guard = threading.RLock()
        self._locks: dict[str, threading.RLock] = {}
        self._journals: dict[str, SessionJournal] = {}
        self._recorders: dict[str, SessionRecorder] = {}

    def ensure_dirs(self) -> None:
        ensure_private_dir(self.root)
        ensure_private_dir(self.sessions_dir)

    def session_dir_for(self, session_id: str) -> Path:
        return self.sessions_dir / _validate_session_id(session_id)

    def path_for(self, session_id: str) -> Path:
        """Return the small metadata projection for a directory-format session."""
        return self.session_dir_for(session_id) / "metadata.json"

    def events_path_for(self, session_id: str) -> Path:
        return self.session_dir_for(session_id) / "events.jsonl"

    def legacy_path_for(self, session_id: str) -> Path:
        return self.sessions_dir / f"{_validate_session_id(session_id)}.json"

    def create(
        self,
        project_path: Path,
        mode: str,
        config: dict[str, Any],
        session_id: Optional[str] = None,
        title: Optional[str] = None,
    ) -> SessionRecord:
        record = SessionRecord.create(project_path, mode, config, session_id=session_id, title=title)
        self.ensure_dirs()
        session_dir = self.session_dir_for(record.session_id)
        if session_dir.exists() or self.legacy_path_for(record.session_id).exists():
            raise SessionStoreError(f"Session already exists: {record.session_id}")
        ensure_private_dir(session_dir)
        ensure_private_dir(session_dir / "artifacts")
        journal = self.journal(record.session_id)
        epoch_id = str(uuid.uuid4())
        try:
            journal.append(
                "session.created",
                actor="system",
                payload={"metadata": record.to_metadata_dict()},
                epoch_id=epoch_id,
            )
            journal.append(
                "context.epoch_started",
                actor="system",
                payload={"reason": "session_created"},
                epoch_id=epoch_id,
            )
            self._write_metadata(record.to_metadata_dict())
        except Exception:
            shutil.rmtree(session_dir, ignore_errors=True)
            with self._locks_guard:
                self._journals.pop(record.session_id, None)
                self._recorders.pop(record.session_id, None)
            raise
        return record

    def save(self, record: SessionRecord) -> None:
        """Persist metadata only; history is written through ``SessionRecorder``."""
        self.ensure_dirs()
        self._ensure_migrated(record.session_id)
        if not self.session_dir_for(record.session_id).exists():
            raise SessionStoreError(f"Session not found: {record.session_id}")
        with self._lock_for(record.session_id):
            try:
                current = self._read_metadata(record.session_id)
            except SessionStoreError:
                events = self.journal(record.session_id).read_events(repair_tail=True)
                current = self._metadata_projection(record.session_id, events)
            record.updated_at = _now()
            updated = record.to_metadata_dict()
            patch = {key: copy.deepcopy(value) for key, value in updated.items() if current.get(key) != value}
            if patch:
                self.journal(record.session_id).append(
                    "session.metadata_updated",
                    actor="system",
                    payload={"patch": patch},
                )
            self._write_metadata(updated)

    def load(self, session_id: str) -> SessionRecord:
        self._ensure_migrated(session_id)
        if not self.session_dir_for(session_id).exists():
            raise SessionStoreError(f"Session not found: {session_id}")
        try:
            events = self.journal(session_id).read_events(repair_tail=True)
            metadata = self._metadata_projection(session_id, events)
            history, compaction = self._replay(events, self.journal(session_id))
            return SessionRecord.from_dict({**metadata, "history": history, "compaction": compaction})
        except SessionStoreError:
            raise
        except SessionJournalError as exc:
            raise SessionStoreError(str(exc)) from exc

    def recorder(self, session_id: str, *, recover: bool = True) -> SessionRecorder:
        self._ensure_migrated(session_id)
        if not self.session_dir_for(session_id).exists():
            raise SessionStoreError(f"Session not found: {session_id}")
        with self._lock_for(session_id):
            with self._locks_guard:
                recorder = self._recorders.get(session_id)
            if recorder is not None:
                return recorder
            try:
                recorder = SessionRecorder(self.journal(session_id), recover=recover)
            except SessionJournalError as exc:
                raise SessionStoreError(str(exc)) from exc
            with self._locks_guard:
                self._recorders[session_id] = recorder
            return recorder

    def journal(self, session_id: str) -> SessionJournal:
        with self._locks_guard:
            journal = self._journals.get(session_id)
            if journal is None:
                journal = SessionJournal(session_id, self.session_dir_for(session_id), self._lock_for(session_id))
                self._journals[session_id] = journal
            return journal

    def start_epoch(self, session_id: str, reason: str) -> str:
        return self.recorder(session_id).start_epoch(reason)

    def load_by_thread_id(self, thread_id: str) -> SessionRecord:
        for record in self._iter_records():
            if record.thread_id == thread_id:
                return record
        raise SessionStoreError(f"Thread not found: {thread_id}")

    def load_session_or_thread(self, identifier: str) -> SessionRecord:
        if self.session_dir_for(identifier).exists() or self.legacy_path_for(identifier).exists():
            return self.load(identifier)
        return self.load_by_thread_id(identifier)

    def delete(self, session_id: str) -> None:
        self._ensure_migrated(session_id)
        path = self.session_dir_for(session_id)
        if not path.exists():
            raise SessionStoreError(f"Session not found: {session_id}")
        with self._locks_guard:
            self._recorders.pop(session_id, None)
            self._journals.pop(session_id, None)
        shutil.rmtree(path)

    def list(self, project_path: Optional[Path] = None) -> list[SessionRecord]:
        records = list(self._iter_records())
        if project_path is not None:
            resolved = str(project_path.resolve())
            records = [record for record in records if record.project_path == resolved]
        return sorted(records, key=lambda record: record.updated_at, reverse=True)

    def latest_for_project(self, project_path: Path) -> Optional[SessionRecord]:
        records = self.list(project_path=project_path)
        return records[0] if records else None

    def export(self, session_id: str) -> str:
        return json.dumps(self.load(session_id).to_dict(), indent=2, sort_keys=True) + "\n"

    def bug_export(self, session_id: str) -> SessionBugExport:
        self._ensure_migrated(session_id)
        journal = self.journal(session_id)
        try:
            events = journal.read_events(repair_tail=True)
            metadata = self._metadata_projection(session_id, events)
            history, compaction = self._replay(events, journal, hydrate_artifacts=False)
            projection = SessionRecord.from_dict({**metadata, "history": history, "compaction": compaction}).to_dict()
            events_jsonl = journal.raw_events()
        except SessionJournalError as exc:
            raise SessionStoreError(str(exc)) from exc
        refs: dict[str, dict[str, Any]] = {}
        for event in events:
            for ref in event.artifacts:
                digest = str(ref.get("sha256") or "")
                if digest:
                    refs[digest] = {
                        key: value
                        for key, value in ref.items()
                        if key in {"sha256", "bytes", "chars", "media_type", "purpose", "encoding"}
                    }
        manifest = {
            "session_id": session_id,
            "artifacts_included": False,
            "artifacts": list(refs.values()),
        }
        return SessionBugExport(
            session_json=json.dumps(projection, indent=2, sort_keys=True) + "\n",
            events_jsonl=events_jsonl,
            artifact_manifest_json=json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )

    def _iter_records(self) -> Iterable[SessionRecord]:
        if not self.sessions_dir.exists():
            return []
        identifiers = {
            path.name for path in self.sessions_dir.iterdir() if path.is_dir() and not path.name.startswith(".")
        }
        identifiers.update(path.stem for path in self.sessions_dir.glob("*.json"))
        records = []
        for session_id in identifiers:
            try:
                self._ensure_migrated(session_id)
                try:
                    metadata = self._read_metadata(session_id)
                except SessionStoreError:
                    events = self.journal(session_id).read_events(repair_tail=True)
                    metadata = self._metadata_projection(session_id, events)
                records.append(SessionRecord.from_dict({**metadata, "history": [], "compaction": {}}))
            except Exception:
                continue
        return records

    def _lock_for(self, session_id: str) -> threading.RLock:
        with self._locks_guard:
            return self._locks.setdefault(session_id, threading.RLock())

    def _read_metadata(self, session_id: str) -> dict[str, Any]:
        path = self.path_for(session_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SessionStoreError(f"Session metadata is not valid JSON: {path}") from exc
        except OSError as exc:
            raise SessionStoreError(f"Could not read session metadata: {path}") from exc
        if not isinstance(data, dict):
            raise SessionStoreError(f"Session metadata is not an object: {path}")
        return data

    def _write_metadata(self, metadata: dict[str, Any]) -> None:
        session_id = str(metadata["session_id"])
        payload = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        write_private_text(self.path_for(session_id), payload)

    def _metadata_with_event_patches(self, metadata: dict[str, Any], events: Iterable[SessionEvent]) -> dict[str, Any]:
        result = copy.deepcopy(metadata)
        for event in events:
            if event.event_type == "session.metadata_updated" and isinstance(event.payload.get("patch"), dict):
                result.update(copy.deepcopy(event.payload["patch"]))
        return result

    def _metadata_projection(self, session_id: str, events: list[SessionEvent]) -> dict[str, Any]:
        """Read metadata or rebuild it from canonical events, then apply newer patches."""
        original: Optional[dict[str, Any]]
        try:
            original = self._read_metadata(session_id)
        except SessionStoreError:
            original = None

        if original is None:
            created = next(
                (
                    event.payload.get("metadata")
                    for event in events
                    if event.event_type == "session.created" and isinstance(event.payload.get("metadata"), dict)
                ),
                None,
            )
            if not isinstance(created, dict):
                raise SessionStoreError(f"Session {session_id} has no recoverable metadata event")
            metadata = copy.deepcopy(created)
        else:
            metadata = original

        metadata = self._metadata_with_event_patches(metadata, events)
        # Validate the projection before publishing a repair.
        SessionRecord.from_dict({**metadata, "history": [], "compaction": {}})
        if original != metadata:
            try:
                self._write_metadata(metadata)
            except OSError:
                # Events are canonical. A read remains usable even when the small
                # rebuildable projection cannot currently be refreshed.
                pass
        return metadata

    def _replay(
        self,
        events: Iterable[SessionEvent],
        journal: SessionJournal,
        *,
        hydrate_artifacts: bool = True,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        history: list[dict[str, Any]] = []
        compaction: dict[str, Any] = {}
        current_epoch: Optional[str] = None
        message_events = {"turn.started", "assistant.message", "tool.results", "context.message"}
        for event in events:
            if event.event_type == "context.epoch_started":
                current_epoch = event.epoch_id
                history = []
                compaction = {}
                continue
            if current_epoch is None or event.epoch_id != current_epoch:
                continue
            if event.event_type in message_events:
                message = event.payload.get("message")
                if not isinstance(message, dict):
                    raise SessionStoreError(f"Session event {event.seq} is missing a message")
                if hydrate_artifacts:
                    try:
                        history.append(journal.hydrate_message(message))
                    except SessionJournalError as exc:
                        raise SessionStoreError(str(exc)) from exc
                else:
                    history.append(copy.deepcopy(message))
            elif event.event_type == "context.compacted":
                value = event.payload.get("compaction")
                if not isinstance(value, dict):
                    raise SessionStoreError(f"Session compaction event {event.seq} is invalid")
                compaction = copy.deepcopy(value)
        return history, compaction

    def _ensure_migrated(self, session_id: str) -> None:
        destination = self.session_dir_for(session_id)
        legacy = self.legacy_path_for(session_id)
        if destination.exists():
            if legacy.exists():
                try:
                    legacy.unlink()
                except OSError:
                    pass
            return
        if not legacy.exists():
            return
        with self._lock_for(session_id):
            if destination.exists():
                return
            try:
                data = json.loads(legacy.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise SessionStoreError(f"Legacy session is not an object: {legacy}")
                record = SessionRecord.from_dict(data)
            except json.JSONDecodeError as exc:
                raise SessionStoreError(f"Session file is not valid JSON: {legacy}") from exc

            staging = self.sessions_dir / f".{session_id}.migrating-{uuid.uuid4().hex}"
            ensure_private_dir(staging)
            ensure_private_dir(staging / "artifacts")
            journal = SessionJournal(
                session_id,
                staging,
                artifact_reference_dir=destination / "artifacts",
            )
            epoch_id = str(uuid.uuid4())
            try:
                journal.append(
                    "session.created",
                    actor="system",
                    payload={"metadata": record.to_metadata_dict(), "migrated_from": "monolithic_json"},
                    epoch_id=epoch_id,
                )
                journal.append(
                    "context.epoch_started",
                    actor="system",
                    payload={"reason": "legacy_migration"},
                    epoch_id=epoch_id,
                )
                self._migrate_history(record.history, journal)
                SessionRecorder(journal, recover=True)
                if record.compaction:
                    journal.append(
                        "context.compacted",
                        actor="system",
                        payload={"compaction": copy.deepcopy(record.compaction)},
                    )
                payload = json.dumps(record.to_metadata_dict(), indent=2, sort_keys=True) + "\n"
                write_private_text(staging / "metadata.json", payload)
                staging.replace(destination)
                try:
                    legacy.unlink()
                except OSError:
                    pass
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise
            finally:
                with self._locks_guard:
                    self._journals.pop(session_id, None)
                    self._recorders.pop(session_id, None)

    def _migrate_history(self, history: list[dict[str, Any]], journal: SessionJournal) -> None:
        turn_id: Optional[str] = None
        for message in history:
            if not isinstance(message, dict):
                raise SessionStoreError("Legacy session history contains a non-message value")
            stored, artifacts = journal.prepare_message(message)
            role = str(message.get("role") or "system")
            raw_content = message.get("content")
            blocks: list[Any] = raw_content if isinstance(raw_content, list) else []
            has_results = any(isinstance(block, dict) and block.get("type") == "tool_result" for block in blocks)
            if role == "user" and not has_results:
                if turn_id is not None:
                    journal.append("turn.completed", actor="system", payload={"migrated": True}, turn_id=turn_id)
                turn_id = str(uuid.uuid4())
                journal.append(
                    "turn.started",
                    actor="user",
                    payload={"message": stored},
                    turn_id=turn_id,
                    artifacts=artifacts,
                )
            elif role == "assistant" and turn_id is not None:
                journal.append(
                    "assistant.message",
                    actor="assistant",
                    payload={"message": stored},
                    turn_id=turn_id,
                    artifacts=artifacts,
                )
            elif has_results and turn_id is not None:
                journal.append(
                    "tool.results",
                    actor="tool",
                    payload={"message": stored},
                    turn_id=turn_id,
                    artifacts=artifacts,
                )
            else:
                journal.append(
                    "context.message",
                    actor=role,
                    payload={"message": stored},
                    turn_id=turn_id,
                    artifacts=artifacts,
                )
