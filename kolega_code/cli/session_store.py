"""Local resumable session storage for the Kolega Code CLI."""

from __future__ import annotations

import json
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from kolega_code.local_state import ensure_private_dir, write_private_text

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

    def to_dict(self) -> dict[str, Any]:
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
            "history": self.history,
            "compaction": self.compaction,
            "task_list_markdown": self.task_list_markdown,
            "latest_plan_markdown": self.latest_plan_markdown,
            "plan_pending": self.plan_pending,
            "plan_reofferable": self.plan_reofferable,
            "interaction_mode": self.interaction_mode,
            "permission_mode": self.permission_mode,
            "gigacode_enabled": self.gigacode_enabled,
            "goal": self.goal,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_state_dir(env: Optional[dict[str, str]] = None) -> Path:
    env = env or os.environ
    if env.get("KOLEGA_CODE_STATE_DIR"):
        return Path(env["KOLEGA_CODE_STATE_DIR"]).expanduser()

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "kolega-code"
    if sys.platform.startswith("win"):
        base = Path(env.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return base / "kolega-code"
    return Path(env.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "kolega-code"


class SessionStore:
    """Filesystem-backed session store."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = (root or default_state_dir()).expanduser()
        self.sessions_dir = self.root / "sessions"

    def ensure_dirs(self) -> None:
        ensure_private_dir(self.root)
        ensure_private_dir(self.sessions_dir)

    def path_for(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.json"

    def create(
        self,
        project_path: Path,
        mode: str,
        config: dict[str, Any],
        session_id: Optional[str] = None,
        title: Optional[str] = None,
    ) -> SessionRecord:
        record = SessionRecord.create(project_path, mode, config, session_id=session_id, title=title)
        self.save(record)
        return record

    def save(self, record: SessionRecord) -> None:
        self.ensure_dirs()
        record.updated_at = _now()
        payload = json.dumps(record.to_dict(), indent=2, sort_keys=True)
        write_private_text(self.path_for(record.session_id), payload + "\n")

    def load(self, session_id: str) -> SessionRecord:
        path = self.path_for(session_id)
        if not path.exists():
            raise SessionStoreError(f"Session not found: {session_id}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SessionStoreError(f"Session file is not valid JSON: {path}") from exc
        return SessionRecord.from_dict(data)

    def load_by_thread_id(self, thread_id: str) -> SessionRecord:
        for record in self._iter_records():
            if record.thread_id == thread_id:
                return record
        raise SessionStoreError(f"Thread not found: {thread_id}")

    def load_session_or_thread(self, identifier: str) -> SessionRecord:
        path = self.path_for(identifier)
        if path.exists():
            return self.load(identifier)
        return self.load_by_thread_id(identifier)

    def delete(self, session_id: str) -> None:
        path = self.path_for(session_id)
        if not path.exists():
            raise SessionStoreError(f"Session not found: {session_id}")
        path.unlink()

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

    def _iter_records(self) -> Iterable[SessionRecord]:
        if not self.sessions_dir.exists():
            return []

        records = []
        for path in self.sessions_dir.glob("*.json"):
            try:
                records.append(SessionRecord.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except Exception:
                continue
        return records
