"""Append-only durable session events and large-payload artifacts."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import logging
import os
import threading
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, cast

from filelock import BaseFileLock, Timeout

from kolega_code.llm.models import ContentBlock, Message, ToolResult
from kolega_code.local_state import ensure_private_dir, ensure_private_file, write_private_bytes

logger = logging.getLogger(__name__)

EVENT_SCHEMA_VERSION = 2
SUPPORTED_EVENT_VERSIONS = frozenset({1, 2})
PUBLIC_EVENT_SCHEMA = "kolega.session.event"
DEFAULT_ROOT_AGENT_NAME = "main"
TOOL_RESULT_PREVIEW_CHARS = 100_000
TERMINAL_TURN_EVENTS = {"turn.completed", "turn.failed", "turn.cancelled"}

_AGENT_ID_NAMESPACE = uuid.UUID("6f9c4d47-1a52-4ef1-9c11-7f1b8f7a3a9e")


def derive_root_agent_id(session_id: str) -> str:
    """Stable root-agent identity, shared with v1 export fallbacks."""
    return str(uuid.uuid5(_AGENT_ID_NAMESPACE, f"{session_id}:root"))


@dataclass(frozen=True)
class AgentStamp:
    """Agent lineage stamped on every v2 session event."""

    agent_id: str
    agent_name: str
    parent_agent_id: Optional[str]
    parent_tool_call_id: Optional[str]
    depth: int


class SessionJournalError(RuntimeError):
    """Raised when an event log or referenced artifact is not trustworthy."""

    session_persistence_error = True


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class SessionEvent:
    version: int
    event_id: str
    session_id: str
    seq: int
    epoch_id: str
    turn_id: Optional[str]
    timestamp: str
    actor: str
    event_type: str
    payload: dict[str, Any]
    artifacts: list[dict[str, Any]]
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    parent_agent_id: Optional[str] = None
    parent_tool_call_id: Optional[str] = None
    depth: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "version": self.version,
            "id": self.event_id,
            "session_id": self.session_id,
            "seq": self.seq,
            "epoch_id": self.epoch_id,
            "turn_id": self.turn_id,
            "timestamp": self.timestamp,
            "actor": self.actor,
            "type": self.event_type,
            "payload": self.payload,
            "artifacts": self.artifacts,
        }
        if self.version >= 2:
            data["schema"] = PUBLIC_EVENT_SCHEMA
            data["agent_id"] = self.agent_id
            data["agent_name"] = self.agent_name
            data["parent_agent_id"] = self.parent_agent_id
            data["parent_tool_call_id"] = self.parent_tool_call_id
            data["depth"] = self.depth
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionEvent":
        required = {
            "version",
            "id",
            "session_id",
            "seq",
            "epoch_id",
            "timestamp",
            "actor",
            "type",
            "payload",
        }
        missing = required.difference(data)
        if missing:
            raise SessionJournalError(f"Session event is missing fields: {', '.join(sorted(missing))}")
        version = data["version"]
        if version not in SUPPORTED_EVENT_VERSIONS:
            raise SessionJournalError(f"Unsupported session event version: {version}")
        if not isinstance(data["seq"], int) or data["seq"] < 1:
            raise SessionJournalError("Session event sequence must be a positive integer")
        if not isinstance(data["payload"], dict):
            raise SessionJournalError("Session event payload must be an object")
        artifacts = data.get("artifacts") or []
        if not isinstance(artifacts, list) or not all(isinstance(ref, dict) for ref in artifacts):
            raise SessionJournalError("Session event artifacts must be a list of objects")
        for field in ("id", "session_id", "epoch_id", "timestamp", "actor", "type"):
            if not isinstance(data[field], str) or not data[field]:
                raise SessionJournalError(f"Session event field {field} must be a non-empty string")
        agent_id: Optional[str] = None
        agent_name: Optional[str] = None
        parent_agent_id: Optional[str] = None
        parent_tool_call_id: Optional[str] = None
        depth: Optional[int] = None
        if version >= 2:
            for field in ("agent_id", "agent_name"):
                if not isinstance(data.get(field), str) or not data[field]:
                    raise SessionJournalError(f"Session event field {field} must be a non-empty string")
            depth = data.get("depth")
            if not isinstance(depth, int) or depth < 0:
                raise SessionJournalError("Session event depth must be a non-negative integer")
            for field in ("parent_agent_id", "parent_tool_call_id"):
                value = data.get(field)
                if value is not None and (not isinstance(value, str) or not value):
                    raise SessionJournalError(f"Session event field {field} must be null or a non-empty string")
            agent_id = str(data["agent_id"])
            agent_name = str(data["agent_name"])
            parent_agent_id = data.get("parent_agent_id")
            parent_tool_call_id = data.get("parent_tool_call_id")
        return cls(
            version=version,
            event_id=str(data["id"]),
            session_id=str(data["session_id"]),
            seq=data["seq"],
            epoch_id=str(data["epoch_id"]),
            turn_id=str(data["turn_id"]) if data.get("turn_id") is not None else None,
            timestamp=str(data["timestamp"]),
            actor=str(data["actor"]),
            event_type=str(data["type"]),
            payload=data["payload"],
            artifacts=artifacts,
            agent_id=agent_id,
            agent_name=agent_name,
            parent_agent_id=parent_agent_id,
            parent_tool_call_id=parent_tool_call_id,
            depth=depth,
        )


@dataclass(frozen=True)
class TurnSummary:
    """One recorded turn in the current context epoch."""

    turn_id: str
    boundary_seq: int  # seq of the turn.started event; the rewind truncation key
    started_at: str
    status: str  # completed | failed | cancelled | open
    user_text: str


@dataclass(frozen=True)
class RewindOutcome:
    target_turn_id: str
    boundary_seq: int
    rewound_turn_ids: list[str]
    user_message_text: str


def collect_epoch_turns(events: Iterable[SessionEvent], epoch_id: str) -> list[TurnSummary]:
    """Fold turn events within an epoch, applying prior rewinds."""
    turns: list[TurnSummary] = []
    for event in events:
        if event.epoch_id != epoch_id:
            continue
        if event.depth:
            # Subagent turns live in the same journal but are never part of the
            # root agent's rewindable history.
            continue
        if event.event_type == "turn.started":
            turns.append(
                TurnSummary(
                    turn_id=event.turn_id or "",
                    boundary_seq=event.seq,
                    started_at=event.timestamp,
                    status="open",
                    user_text=_turn_user_text(event.payload.get("message")),
                )
            )
        elif event.event_type in TERMINAL_TURN_EVENTS:
            for index in range(len(turns) - 1, -1, -1):
                if turns[index].turn_id == event.turn_id:
                    turns[index] = replace(turns[index], status=event.event_type.split(".", 1)[1])
                    break
        elif event.event_type == "context.rewound":
            boundary = event.payload.get("boundary_seq")
            if isinstance(boundary, int):
                turns = [turn for turn in turns if turn.boundary_seq < boundary]
    return turns


def _turn_user_text(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    parts = [str(block.get("text") or "") for block in _message_blocks(message) if block.get("type") == "text"]
    return "\n".join(part for part in parts if part)


class SessionJournal:
    """Single-session JSONL writer and artifact store.

    Each event is serialized with its trailing newline and appended in one
    ``os.write`` call. That makes completed events visible after a process crash;
    it intentionally does not claim power-loss durability.
    """

    def __init__(
        self,
        session_id: str,
        session_dir: Path,
        lock: Optional[threading.RLock] = None,
        artifact_reference_dir: Optional[Path] = None,
        cross_process_lock: Optional[BaseFileLock] = None,
    ) -> None:
        self.session_id = session_id
        self.session_dir = session_dir
        self.events_path = session_dir / "events.jsonl"
        self.artifacts_dir = session_dir / "artifacts"
        self.artifact_reference_dir = artifact_reference_dir or self.artifacts_dir
        self._lock = lock or threading.RLock()
        self._cross_process_lock: Optional[BaseFileLock] = cross_process_lock
        self._loaded = False
        self._next_seq = 1
        self._epoch_id: Optional[str] = None
        self._root_stamp = AgentStamp(
            agent_id=derive_root_agent_id(session_id),
            agent_name=DEFAULT_ROOT_AGENT_NAME,
            parent_agent_id=None,
            parent_tool_call_id=None,
            depth=0,
        )
        self._listeners: list[Callable[[SessionEvent], None]] = []

    @property
    def root_stamp(self) -> AgentStamp:
        return self._root_stamp

    def add_listener(self, listener: Callable[[SessionEvent], None]) -> None:
        """Tee every successfully appended event to ``listener``.

        Listeners receive the exact persisted ``SessionEvent`` object, so a live
        consumer and the saved record share id, seq, and timestamp. Listener
        failures are logged and never fail the append.
        """
        with self._lock:
            self._listeners.append(listener)

    @property
    def epoch_id(self) -> str:
        with self._lock:
            self._ensure_loaded_locked()
            if self._epoch_id is None:
                raise SessionJournalError(f"Session {self.session_id} has no context epoch")
            return self._epoch_id

    def read_events(self, *, repair_tail: bool = True) -> list[SessionEvent]:
        with self._lock:
            events = self._read_events_locked(repair_tail=repair_tail)
            self._set_state_from_events_locked(events)
            return events

    def raw_events(self) -> str:
        self.read_events(repair_tail=True)
        try:
            return self.events_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise SessionJournalError(f"Could not read session events: {self.events_path}") from exc

    def append(
        self,
        event_type: str,
        *,
        actor: str,
        payload: Optional[dict[str, Any]] = None,
        turn_id: Optional[str] = None,
        epoch_id: Optional[str] = None,
        artifacts: Optional[list[dict[str, Any]]] = None,
        agent: Optional[AgentStamp] = None,
    ) -> SessionEvent:
        with self._lock:
            if self._cross_process_lock is not None:
                try:
                    with self._cross_process_lock:
                        return self._append_locked(
                            event_type,
                            actor=actor,
                            payload=payload,
                            turn_id=turn_id,
                            epoch_id=epoch_id,
                            artifacts=artifacts,
                            agent=agent,
                        )
                except Timeout as exc:
                    raise SessionJournalError(
                        f"Session event log is locked by another kolega-code instance: {self.events_path}"
                    ) from exc
            return self._append_locked(
                event_type,
                actor=actor,
                payload=payload,
                turn_id=turn_id,
                epoch_id=epoch_id,
                artifacts=artifacts,
                agent=agent,
            )

    def _append_locked(
        self,
        event_type: str,
        *,
        actor: str,
        payload: Optional[dict[str, Any]] = None,
        turn_id: Optional[str] = None,
        epoch_id: Optional[str] = None,
        artifacts: Optional[list[dict[str, Any]]] = None,
        agent: Optional[AgentStamp] = None,
    ) -> SessionEvent:
        """Allocate and persist one event; callers hold the process lock and, when configured, the cross-process lock."""
        self._ensure_loaded_locked()
        if self._cross_process_lock is not None:
            self._resync_seq_from_tail_locked()
        resolved_epoch = epoch_id or self._epoch_id
        if not resolved_epoch:
            raise SessionJournalError("An epoch id is required for every session event")
        stamp = agent or self._root_stamp
        event = SessionEvent(
            version=EVENT_SCHEMA_VERSION,
            event_id=str(uuid.uuid4()),
            session_id=self.session_id,
            seq=self._next_seq,
            epoch_id=resolved_epoch,
            turn_id=turn_id,
            timestamp=_now(),
            actor=actor,
            event_type=event_type,
            payload=payload or {},
            artifacts=artifacts or [],
            agent_id=stamp.agent_id,
            agent_name=stamp.agent_name,
            parent_agent_id=stamp.parent_agent_id,
            parent_tool_call_id=stamp.parent_tool_call_id,
            depth=stamp.depth,
        )
        self._write_event_locked(event)
        self._next_seq += 1
        if event_type == "context.epoch_started":
            self._epoch_id = resolved_epoch
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                logger.exception("Session event listener failed for %s", event_type)
        return event

    def _resync_seq_from_tail_locked(self) -> None:
        """Reconcile ``_next_seq`` with the durable tail after another writer appended.

        Called with the cross-process lock held, so the tail read cannot race
        with another writer's allocate+write. A missing or unreadable tail
        (e.g. a crashed partial write) falls back to a full repair-and-reload.
        """
        tail_seq = self._read_tail_seq_locked()
        if tail_seq is None:
            events = self._read_events_locked(repair_tail=True)
            self._set_state_from_events_locked(events)
            return
        if tail_seq + 1 != self._next_seq:
            self._next_seq = tail_seq + 1

    def _read_tail_seq_locked(self) -> Optional[int]:
        """Return the seq of the last complete event line, or None when absent or unreadable."""
        if not self.events_path.exists():
            return None
        try:
            size = self.events_path.stat().st_size
        except OSError:
            return None
        if size == 0:
            return None
        try:
            with open(self.events_path, "rb") as fh:
                fh.seek(max(0, size - 65536))
                chunk = fh.read()
        except OSError:
            return None
        if chunk.endswith(b"\n"):
            lines = chunk.split(b"\n")
            last = lines[-2] if len(lines) >= 2 else b""
        else:
            last = chunk.split(b"\n")[-1]
        if not last.strip():
            return None
        try:
            data = json.loads(last)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        seq = data.get("seq") if isinstance(data, dict) else None
        return seq if isinstance(seq, int) else None

    def repair_sequence(self) -> int:
        """Renumber every event 1..N in file order and return the event count.

        Rewrites the journal atomically only when the sequence is broken. The
        store holds the cross-process lock around calls that may race with
        another instance; the lock is acquired here too for direct callers.
        """
        with self._lock:
            if self._cross_process_lock is not None:
                with self._cross_process_lock:
                    return self._repair_sequence_locked()
            return self._repair_sequence_locked()

    def _repair_sequence_locked(self) -> int:
        events = self._read_events_locked(repair_tail=True, strict_seq=False)
        if not events:
            self._set_state_from_events_locked(events)
            return 0
        needs_rewrite = any(event.seq != index for index, event in enumerate(events, start=1))
        if needs_rewrite:
            payload = b"".join(
                (
                    json.dumps(replace(event, seq=index).to_dict(), separators=(",", ":"), ensure_ascii=False) + "\n"
                ).encode("utf-8")
                for index, event in enumerate(events, start=1)
            )
            tmp_path = self.events_path.with_name(f".{self.events_path.name}.repair-{uuid.uuid4().hex}")
            try:
                fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                try:
                    os.fchmod(fd, 0o600)
                    with os.fdopen(fd, "wb") as fh:
                        fh.write(payload)
                        fh.flush()
                        os.fsync(fh.fileno())
                except BaseException:
                    os.close(fd)
                    raise
                os.replace(tmp_path, self.events_path)
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        self._set_state_from_events_locked(events)
        return len(events)

    def _write_event_locked(self, event: SessionEvent) -> None:
        """Append one complete event, truncating a failed partial write."""
        data = (json.dumps(event.to_dict(), separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
        fd: Optional[int] = None
        original_size = 0
        try:
            ensure_private_dir(self.session_dir)
            fd = os.open(self.events_path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
            ensure_private_file(self.events_path)
            original_size = os.fstat(fd).st_size
            written = os.write(fd, data)
            if written != len(data):
                raise SessionJournalError(
                    f"Short event write for session {self.session_id}: wrote {written} of {len(data)} bytes"
                )
        except Exception as exc:
            if fd is not None:
                try:
                    os.ftruncate(fd, original_size)
                except OSError:
                    # Force the next operation to validate and repair any tail.
                    pass
            self._loaded = False
            if isinstance(exc, SessionJournalError):
                raise
            raise SessionJournalError(f"Could not append session event: {self.events_path}") from exc
        finally:
            if fd is not None:
                os.close(fd)

    def start_epoch(self, reason: str) -> str:
        epoch_id = str(uuid.uuid4())
        self.append(
            "context.epoch_started",
            actor="system",
            payload={"reason": reason},
            epoch_id=epoch_id,
        )
        return epoch_id

    def prepare_message(self, message: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Externalize large/opaque fields and return the event-safe message."""
        prepared = copy.deepcopy(message)
        refs: list[dict[str, Any]] = []
        content = prepared.get("content")
        if isinstance(content, list):
            self._prepare_blocks(content, refs)
        return prepared, _dedupe_refs(refs)

    def hydrate_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Restore provider-required opaque fields while retaining tool previews."""
        hydrated = copy.deepcopy(message)
        content = hydrated.get("content")
        if isinstance(content, list):
            self._hydrate_blocks(content)
        return hydrated

    def put_artifact(
        self,
        data: bytes,
        *,
        media_type: str,
        purpose: str,
        encoding: str,
        chars: Optional[int] = None,
    ) -> dict[str, Any]:
        digest = hashlib.sha256(data).hexdigest()
        with self._lock:
            path = self.artifacts_dir / digest
            try:
                ensure_private_dir(self.artifacts_dir)
                if not path.exists():
                    write_private_bytes(path, data)
                else:
                    ensure_private_file(path)
            except OSError as exc:
                raise SessionJournalError(f"Could not persist session artifact: {digest}") from exc
        ref: dict[str, Any] = {
            "sha256": digest,
            "bytes": len(data),
            "media_type": media_type,
            "purpose": purpose,
            "encoding": encoding,
            "path": str(self.artifact_reference_dir / digest),
        }
        if chars is not None:
            ref["chars"] = chars
        return ref

    def read_artifact(self, ref: dict[str, Any]) -> bytes:
        digest = str(ref.get("sha256") or "")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise SessionJournalError("Invalid session artifact digest")
        path = self.artifacts_dir / digest
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise SessionJournalError(f"Missing session artifact: {digest}") from exc
        if hashlib.sha256(data).hexdigest() != digest:
            raise SessionJournalError(f"Session artifact failed integrity check: {digest}")
        return data

    def _prepare_blocks(self, blocks: list[Any], refs: list[dict[str, Any]]) -> None:
        for raw in blocks:
            if not isinstance(raw, dict):
                continue
            block_type = raw.get("type")
            if block_type == "tool_result":
                content = raw.get("content")
                if isinstance(content, str) and len(content) > TOOL_RESULT_PREVIEW_CHARS:
                    data = content.encode("utf-8")
                    ref = self.put_artifact(
                        data,
                        media_type="text/plain; charset=utf-8",
                        purpose="tool_result",
                        encoding="utf-8",
                        chars=len(content),
                    )
                    raw["content"] = _tool_result_preview(content, ref)
                    raw["content_artifact"] = ref
                    refs.append(ref)
                elif isinstance(content, list):
                    self._prepare_blocks(content, refs)

            artifact_fields: dict[str, dict[str, Any]] = {}
            if block_type == "image_url" and raw.get("image_type") == "base64" and raw.get("data"):
                encoded = str(raw["data"])
                try:
                    data = base64.b64decode(encoded, validate=True)
                except Exception as exc:
                    raise SessionJournalError("Image block contains invalid base64 data") from exc
                ref = self.put_artifact(
                    data,
                    media_type=str(raw.get("media_type") or "application/octet-stream"),
                    purpose="image",
                    encoding="base64",
                )
                artifact_fields["data"] = ref
                raw["data"] = ""
                refs.append(ref)

            opaque_fields: tuple[tuple[str, str], ...] = ()
            if block_type == "thinking":
                opaque_fields = (("signature", "provider_signature"),)
            elif block_type == "redacted_thinking":
                opaque_fields = (("data", "redacted_reasoning"),)
            elif block_type == "responses_reasoning":
                opaque_fields = (("encrypted_content", "encrypted_reasoning"),)
            elif block_type == "tool_call":
                opaque_fields = (("thought_signature", "thought_signature"),)

            for field, purpose in opaque_fields:
                value = raw.get(field)
                if not value:
                    continue
                encoding = "base64" if field == "thought_signature" else "utf-8"
                try:
                    data = base64.b64decode(str(value), validate=True) if encoding == "base64" else str(value).encode()
                except Exception as exc:
                    raise SessionJournalError(f"Invalid encoded provider field: {field}") from exc
                ref = self.put_artifact(
                    data,
                    media_type="application/octet-stream",
                    purpose=purpose,
                    encoding=encoding,
                )
                artifact_fields[field] = ref
                raw[field] = ""
                refs.append(ref)

            if artifact_fields:
                raw["artifact_fields"] = artifact_fields

    def _hydrate_blocks(self, blocks: list[Any]) -> None:
        for raw in blocks:
            if not isinstance(raw, dict):
                continue
            artifact_fields = raw.pop("artifact_fields", {})
            if isinstance(artifact_fields, dict):
                for field, ref in artifact_fields.items():
                    if not isinstance(ref, dict):
                        raise SessionJournalError(f"Invalid artifact reference for field {field}")
                    data = self.read_artifact(ref)
                    raw[field] = (
                        base64.b64encode(data).decode("ascii") if ref.get("encoding") == "base64" else data.decode()
                    )
            if raw.get("type") == "tool_result" and isinstance(raw.get("content"), list):
                self._hydrate_blocks(raw["content"])

    def _ensure_loaded_locked(self) -> None:
        if self._loaded:
            return
        events = self._read_events_locked(repair_tail=True)
        self._set_state_from_events_locked(events)

    def _set_state_from_events_locked(self, events: Iterable[SessionEvent]) -> None:
        events = list(events)
        self._next_seq = events[-1].seq + 1 if events else 1
        self._epoch_id = None
        for event in events:
            if event.event_type == "context.epoch_started":
                self._epoch_id = event.epoch_id
        self._loaded = True

    def _read_events_locked(self, *, repair_tail: bool, strict_seq: bool = True) -> list[SessionEvent]:
        if not self.events_path.exists():
            return []
        try:
            raw = self.events_path.read_bytes()
        except OSError as exc:
            raise SessionJournalError(f"Could not read session events: {self.events_path}") from exc

        if raw and not raw.endswith(b"\n"):
            if not repair_tail:
                raise SessionJournalError(f"Session event log has an incomplete final record: {self.events_path}")
            valid_length = raw.rfind(b"\n") + 1
            try:
                fd = os.open(self.events_path, os.O_WRONLY)
                try:
                    os.ftruncate(fd, valid_length)
                finally:
                    os.close(fd)
            except OSError as exc:
                raise SessionJournalError(f"Could not repair session event tail: {self.events_path}") from exc
            raw = raw[:valid_length]

        events: list[SessionEvent] = []
        expected_seq = 1
        for line_number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                raise SessionJournalError(f"Blank line in session event log at line {line_number}")
            try:
                data = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise SessionJournalError(
                    f"Invalid JSON in session event log at line {line_number}: {self.events_path}"
                ) from exc
            if not isinstance(data, dict):
                raise SessionJournalError(f"Session event at line {line_number} is not an object")
            event = SessionEvent.from_dict(data)
            if event.session_id != self.session_id:
                raise SessionJournalError(
                    f"Session event at line {line_number} belongs to {event.session_id}, expected {self.session_id}"
                )
            if strict_seq and event.seq != expected_seq:
                raise SessionJournalError(
                    f"Session event sequence gap at line {line_number}: expected {expected_seq}, got {event.seq}"
                )
            events.append(event)
            expected_seq += 1
        return events


class SessionRecorder:
    """Semantic recorder for one agent's events in a shared session journal.

    The top-level agent uses a recorder carrying the journal's root stamp;
    subagents record through ``scoped_child`` recorders whose stamp carries
    their lineage. All recorders append to the same journal and share its
    session-wide sequence.
    """

    def __init__(
        self,
        journal: SessionJournal,
        *,
        recover: bool = True,
        agent_stamp: Optional[AgentStamp] = None,
    ) -> None:
        self.journal = journal
        self._lock = threading.RLock()
        self.current_turn_id: Optional[str] = None
        self._agent_stamp = agent_stamp
        self._last_system_fingerprint: Optional[str] = None
        self._last_tools_fingerprint: Optional[str] = None
        self._turn_started_at: Optional[str] = None
        self._assistant_count = 0
        self._tool_batches = 0
        self._run_terminal_recorded = False
        if recover:
            self.recover_interrupted_turn()

    @property
    def agent_stamp(self) -> AgentStamp:
        return self._agent_stamp or self.journal.root_stamp

    def scoped_child(
        self,
        *,
        agent_id: str,
        agent_name: str,
        parent_tool_call_id: Optional[str],
        depth: int,
    ) -> "SessionRecorder":
        """A recorder for one subagent, stamping its lineage on every event."""
        stamp = AgentStamp(
            agent_id=agent_id,
            agent_name=agent_name,
            parent_agent_id=self.agent_stamp.agent_id,
            parent_tool_call_id=parent_tool_call_id,
            depth=depth,
        )
        return SessionRecorder(self.journal, recover=False, agent_stamp=stamp)

    def _append(
        self,
        event_type: str,
        *,
        actor: str,
        payload: Optional[dict[str, Any]] = None,
        turn_id: Optional[str] = None,
        epoch_id: Optional[str] = None,
        artifacts: Optional[list[dict[str, Any]]] = None,
    ) -> SessionEvent:
        return self.journal.append(
            event_type,
            actor=actor,
            payload=payload,
            turn_id=turn_id,
            epoch_id=epoch_id,
            artifacts=artifacts,
            agent=self.agent_stamp,
        )

    def _require_root(self, operation: str) -> None:
        if self.agent_stamp.depth:
            raise SessionJournalError(f"{operation} is only valid for the root agent recorder")

    def start_turn(self, message: Message) -> str:
        with self._lock:
            if self.current_turn_id is not None:
                raise SessionJournalError("Cannot start a session turn while another turn is open")
            stored, artifacts = self.journal.prepare_message(message.to_dict())
            return self._open_turn_locked(actor="user", payload={"message": stored}, artifacts=artifacts)

    def start_continuation_turn(self) -> str:
        """Open a turn that continues from restored history with no user message.

        The ``turn.started`` payload carries ``{"continuation": True}`` and no
        ``message`` key: it contributes no history message on replay, and export
        consumers must not synthesize a user step for it.
        """
        with self._lock:
            if self.current_turn_id is not None:
                raise SessionJournalError("Cannot start a session turn while another turn is open")
            return self._open_turn_locked(actor="system", payload={"continuation": True}, artifacts=None)

    def _open_turn_locked(
        self, *, actor: str, payload: dict[str, Any], artifacts: Optional[list[dict[str, Any]]]
    ) -> str:
        turn_id = str(uuid.uuid4())
        event = self._append(
            "turn.started",
            actor=actor,
            payload=payload,
            turn_id=turn_id,
            artifacts=artifacts,
        )
        self.current_turn_id = turn_id
        self._turn_started_at = event.timestamp
        self._assistant_count = 0
        self._tool_batches = 0
        return turn_id

    def record_assistant(self, message: Message, *, reasoning_effort: Optional[str] = None) -> None:
        with self._lock:
            turn_id = self._require_turn()
            stored, artifacts = self.journal.prepare_message(message.to_dict())
            llm_call = None
            metadata = stored.get("usage_metadata")
            if isinstance(metadata, dict):
                llm_call = metadata.pop("llm_call", None)
                if not metadata:
                    stored.pop("usage_metadata", None)
            payload: dict[str, Any] = {
                "message": stored,
                "origin_type": "llm",
                "llm_call_id": None,
                "provider": None,
                "model": None,
                "reasoning_effort": reasoning_effort,
                "llm_call_count": 1,
            }
            if isinstance(llm_call, dict):
                payload["llm_call_id"] = llm_call.get("llm_call_id")
                payload["run_id"] = llm_call.get("run_id")
                payload["provider"] = llm_call.get("provider")
                payload["model"] = llm_call.get("model")
            self._append(
                "assistant.message",
                actor="assistant",
                payload=payload,
                turn_id=turn_id,
                artifacts=artifacts,
            )
            self._assistant_count += 1

    def record_synthetic_assistant(self, text: str, *, notice_code: str) -> None:
        """A deterministic assistant notice that did not come from an LLM call."""
        with self._lock:
            payload = {
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": text}],
                    "stop_reason": "end_turn",
                },
                "origin_type": "synthetic",
                "llm_call_id": None,
                "llm_call_count": 0,
                "notice_code": notice_code,
            }
            self._append(
                "assistant.message",
                actor="assistant",
                payload=payload,
                turn_id=self.current_turn_id,
            )
            if self.current_turn_id is not None:
                self._assistant_count += 1

    def record_system_context(self, text: str) -> bool:
        """Record the rendered provider-facing system context when it changes."""
        fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self._lock:
            if fingerprint == self._last_system_fingerprint:
                return False
            message = {"role": "system", "content": [{"type": "text", "text": text}]}
            stored, artifacts = self.journal.prepare_message(message)
            self._append(
                "context.system",
                actor="system",
                payload={"message": stored, "sha256": fingerprint},
                turn_id=self.current_turn_id,
                artifacts=artifacts,
            )
            self._last_system_fingerprint = fingerprint
            return True

    def record_tool_definitions(self, tools: list[dict[str, Any]]) -> bool:
        """Record the public schemas of the tools available to this agent.

        Fingerprinted like the system context so only changes are journaled;
        an empty toolset records nothing.
        """
        if not tools:
            return False
        canonical = json.dumps(tools, sort_keys=True, separators=(",", ":"), default=str)
        fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        with self._lock:
            if fingerprint == self._last_tools_fingerprint:
                return False
            self._append(
                "context.tools",
                actor="system",
                payload={"tools": copy.deepcopy(tools), "sha256": fingerprint},
                turn_id=self.current_turn_id,
            )
            self._last_tools_fingerprint = fingerprint
            return True

    def record_tool_results(self, results: list[ToolResult]) -> list[ToolResult]:
        with self._lock:
            turn_id = self._require_turn()
            message = Message(role="user", content=cast(list[ContentBlock], results))
            stored, artifacts = self.journal.prepare_message(message.to_dict())
            self._append(
                "tool.results",
                actor="tool",
                payload={"message": stored},
                turn_id=turn_id,
                artifacts=artifacts,
            )
            self._tool_batches += 1
            replayed = Message.from_dict(self.journal.hydrate_message(stored))
            if not isinstance(replayed.content, list) or not all(
                isinstance(item, ToolResult) for item in replayed.content
            ):
                raise SessionJournalError("Prepared tool result event did not replay as tool results")
            return cast(list[ToolResult], list(replayed.content))

    def record_context_message(self, message: Message, *, actor: Optional[str] = None) -> None:
        with self._lock:
            stored, artifacts = self.journal.prepare_message(message.to_dict())
            self._append(
                "context.message",
                actor=actor or message.role,
                payload={"message": stored},
                turn_id=self.current_turn_id,
                artifacts=artifacts,
            )

    def record_workspace_switched(
        self,
        *,
        old_root: str,
        new_root: str,
        old_label: str = "",
        new_label: str = "",
        old_branch: str = "",
        new_branch: str = "",
    ) -> None:
        """Record a durable, additive active-workspace boundary."""
        with self._lock:
            self._append(
                "session.workspace_switched",
                actor="system",
                payload={
                    "old_root": old_root,
                    "new_root": new_root,
                    "old_label": old_label,
                    "new_label": new_label,
                    "old_branch": old_branch,
                    "new_branch": new_branch,
                },
                turn_id=self.current_turn_id,
            )

    def record_compaction(self, compaction: dict[str, Any], *, info: Optional[dict[str, Any]] = None) -> None:
        with self._lock:
            payload: dict[str, Any] = {"compaction": copy.deepcopy(compaction)}
            if info:
                payload.update(copy.deepcopy(info))
            self._append(
                "context.compacted",
                actor="system",
                payload=payload,
                turn_id=self.current_turn_id,
            )

    def finish_turn(self, status: str, *, error: Optional[str] = None, reason: Optional[str] = None) -> None:
        if status not in {"completed", "failed", "cancelled"}:
            raise ValueError(f"Unknown terminal turn status: {status}")
        with self._lock:
            turn_id = self._require_turn()
            payload: dict[str, Any] = {
                "assistant_messages": self._assistant_count,
                "tool_result_batches": self._tool_batches,
            }
            if self._turn_started_at is not None:
                payload["started_at"] = self._turn_started_at
                try:
                    started = datetime.fromisoformat(self._turn_started_at)
                    payload["duration_ms"] = max(0, int((datetime.now(timezone.utc) - started).total_seconds() * 1000))
                except ValueError:
                    pass
            if error:
                payload["error"] = error[:2000]
            if reason:
                payload["reason"] = reason
            self._append(
                f"turn.{status}",
                actor="system",
                payload=payload,
                turn_id=turn_id,
            )
            self.current_turn_id = None
            self._turn_started_at = None

    def record_run_terminal(self, status: str, payload: dict[str, Any]) -> None:
        """Record the run's terminal state; at most one per recorder, root only."""
        if status not in {"completed", "failed", "cancelled"}:
            raise ValueError(f"Unknown terminal run status: {status}")
        self._require_root("A run terminal record")
        with self._lock:
            if self._run_terminal_recorded:
                return
            self._append(f"run.{status}", actor="system", payload=copy.deepcopy(payload))
            self._run_terminal_recorded = True

    def record_skill_activated(self, *, name: str, source: str, already_active: bool = False) -> None:
        with self._lock:
            self._append(
                "skill.activated",
                actor="system",
                payload={"name": name, "source": source, "already_active": already_active},
                turn_id=self.current_turn_id,
            )

    def record_goal_evaluated(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._append("goal.evaluated", actor="system", payload=copy.deepcopy(payload))

    def record_goal_completed(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._append("goal.completed", actor="system", payload=copy.deepcopy(payload))

    def record_loop_iteration_started(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._append("loop.iteration_started", actor="system", payload=copy.deepcopy(payload))

    def record_loop_sleeping(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._append("loop.sleeping", actor="system", payload=copy.deepcopy(payload))

    def record_loop_completed(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._append("loop.completed", actor="system", payload=copy.deepcopy(payload))

    def record_agent_started(self, payload: dict[str, Any], *, turn_id: Optional[str] = None) -> None:
        """Record a subagent's start under its own stamp, tied to the parent turn."""
        with self._lock:
            self._append("agent.started", actor="system", payload=copy.deepcopy(payload), turn_id=turn_id)

    def record_agent_terminal(self, status: str, payload: dict[str, Any]) -> None:
        if status not in {"completed", "failed"}:
            raise ValueError(f"Unknown terminal agent status: {status}")
        with self._lock:
            self._append(f"agent.{status}", actor="system", payload=copy.deepcopy(payload))

    def start_epoch(self, reason: str) -> str:
        self._require_root("A context epoch reset")
        with self._lock:
            if self.current_turn_id is not None:
                raise SessionJournalError("Cannot reset context while a session turn is open")
            self._last_system_fingerprint = None
            self._last_tools_fingerprint = None
            return self.journal.start_epoch(reason)

    def list_rewindable_turns(self) -> list[TurnSummary]:
        with self._lock:
            events = self.journal.read_events(repair_tail=True)
            return collect_epoch_turns(events, self.journal.epoch_id)

    def record_rewind(self, target_turn_id: str) -> RewindOutcome:
        """Append the rewind event truncating the target turn and everything after it."""
        self._require_root("A context rewind")
        with self._lock:
            if self.current_turn_id is not None:
                raise SessionJournalError("Cannot rewind while a session turn is open")
            events = self.journal.read_events(repair_tail=True)
            turns = collect_epoch_turns(events, self.journal.epoch_id)
            target = next((turn for turn in turns if turn.turn_id == target_turn_id), None)
            if target is None:
                raise SessionJournalError(f"Turn {target_turn_id} is not rewindable in the current context epoch")
            rewound = [turn.turn_id for turn in turns if turn.boundary_seq >= target.boundary_seq]
            self._append(
                "context.rewound",
                actor="user",
                payload={
                    "target_turn_id": target.turn_id,
                    "boundary_seq": target.boundary_seq,
                    "rewound_turn_ids": rewound,
                },
            )
            return RewindOutcome(
                target_turn_id=target.turn_id,
                boundary_seq=target.boundary_seq,
                rewound_turn_ids=rewound,
                user_message_text=target.user_text,
            )

    def recover_interrupted_turn(self) -> bool:
        """Close one interrupted turn without re-running tools or continuing it."""
        with self._lock:
            events = self.journal.read_events(repair_tail=True)
            if not events:
                return False
            current_epoch = self.journal.epoch_id
            open_turn: Optional[str] = None
            turn_events: list[SessionEvent] = []
            for event in events:
                if event.epoch_id != current_epoch:
                    continue
                if event.depth:
                    # Recovery closes root turns only; a subagent's interrupted
                    # turn is surfaced by export, not auto-closed here.
                    continue
                if event.event_type == "turn.started":
                    open_turn = event.turn_id
                    turn_events = [event]
                elif open_turn and event.turn_id == open_turn:
                    turn_events.append(event)
                    if event.event_type in TERMINAL_TURN_EVENTS:
                        open_turn = None
                        turn_events = []
            if not open_turn:
                return False

            assistant_messages = [
                event.payload.get("message")
                for event in turn_events
                if event.event_type == "assistant.message" and isinstance(event.payload.get("message"), dict)
            ]
            result_ids: set[str] = set()
            for event in turn_events:
                if event.event_type != "tool.results":
                    continue
                message = event.payload.get("message")
                for block in _message_blocks(message):
                    if block.get("type") == "tool_result" and block.get("tool_use_id"):
                        result_ids.add(str(block["tool_use_id"]))

            last_assistant = assistant_messages[-1] if assistant_messages else None
            missing: list[ToolResult] = []
            for block in _message_blocks(last_assistant):
                if block.get("type") != "tool_call" or not block.get("id") or str(block["id"]) in result_ids:
                    continue
                missing.append(
                    ToolResult(
                        tool_use_id=str(block["id"]),
                        content=(
                            "Tool execution was interrupted before a durable result was recorded. "
                            "The tool was not re-run."
                        ),
                        name=str(block.get("name") or "unknown_tool"),
                        is_error=True,
                        execution_id=block.get("execution_id"),
                        input_kind=block.get("input_kind", "json"),
                    )
                )
            self.current_turn_id = open_turn
            if missing:
                self.record_tool_results(missing)

            last_blocks = _message_blocks(last_assistant)
            has_tool_calls = any(block.get("type") == "tool_call" for block in last_blocks)
            if last_assistant is not None and not has_tool_calls:
                self.finish_turn("completed")
            else:
                self.finish_turn("failed", error="Process exited before the turn reached a durable terminal marker")
            return True

    def _require_turn(self) -> str:
        if self.current_turn_id is None:
            raise SessionJournalError("Session event requires an active turn")
        return self.current_turn_id


def _message_blocks(message: Any) -> list[dict[str, Any]]:
    if not isinstance(message, dict) or not isinstance(message.get("content"), list):
        return []
    return [block for block in message["content"] if isinstance(block, dict)]


def _tool_result_preview(content: str, ref: dict[str, Any]) -> str:
    location = ref.get("path") or "the session artifact store"
    marker = (
        "\n\n[Middle of tool result omitted from model history. "
        f"Full output: {location} (sha256 {ref['sha256']}, {ref['chars']:,} characters).]\n\n"
    )
    available = max(0, TOOL_RESULT_PREVIEW_CHARS - len(marker))
    head = available // 2
    tail = available - head
    return content[:head] + marker + (content[-tail:] if tail else "")


def _dedupe_refs(refs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ref in refs:
        digest = str(ref.get("sha256") or "")
        if digest in seen:
            continue
        seen.add(digest)
        result.append(ref)
    return result
