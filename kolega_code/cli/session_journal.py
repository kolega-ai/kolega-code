"""Append-only durable session events and large-payload artifacts."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, cast

from kolega_code.llm.models import ContentBlock, Message, ToolResult
from kolega_code.local_state import ensure_private_dir, ensure_private_file, write_private_bytes

EVENT_SCHEMA_VERSION = 1
TOOL_RESULT_PREVIEW_CHARS = 100_000
TERMINAL_TURN_EVENTS = {"turn.completed", "turn.failed", "turn.cancelled"}


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

    def to_dict(self) -> dict[str, Any]:
        return {
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
        if data["version"] != EVENT_SCHEMA_VERSION:
            raise SessionJournalError(f"Unsupported session event version: {data['version']}")
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
        return cls(
            version=data["version"],
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
        )


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
    ) -> None:
        self.session_id = session_id
        self.session_dir = session_dir
        self.events_path = session_dir / "events.jsonl"
        self.artifacts_dir = session_dir / "artifacts"
        self.artifact_reference_dir = artifact_reference_dir or self.artifacts_dir
        self._lock = lock or threading.RLock()
        self._loaded = False
        self._next_seq = 1
        self._epoch_id: Optional[str] = None

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
    ) -> SessionEvent:
        with self._lock:
            self._ensure_loaded_locked()
            resolved_epoch = epoch_id or self._epoch_id
            if not resolved_epoch:
                raise SessionJournalError("An epoch id is required for every session event")
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
            )
            line = (json.dumps(event.to_dict(), separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
            try:
                ensure_private_dir(self.session_dir)
                fd = os.open(self.events_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                try:
                    written = os.write(fd, line)
                    if written != len(line):
                        raise SessionJournalError(
                            f"Short event write for session {self.session_id}: wrote {written} of {len(line)} bytes"
                        )
                finally:
                    os.close(fd)
                ensure_private_file(self.events_path)
            except Exception as exc:
                # A short/failed append may have left a trailing fragment. Force the
                # next operation to validate and repair it before assigning a sequence.
                self._loaded = False
                if isinstance(exc, SessionJournalError):
                    raise
                raise SessionJournalError(f"Could not append session event: {self.events_path}") from exc
            self._next_seq += 1
            if event_type == "context.epoch_started":
                self._epoch_id = resolved_epoch
            return event

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

    def _read_events_locked(self, *, repair_tail: bool) -> list[SessionEvent]:
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
            if event.seq != expected_seq:
                raise SessionJournalError(
                    f"Session event sequence gap at line {line_number}: expected {expected_seq}, got {event.seq}"
                )
            events.append(event)
            expected_seq += 1
        return events


class SessionRecorder:
    """Semantic turn recorder used by the top-level agent."""

    def __init__(self, journal: SessionJournal, *, recover: bool = True) -> None:
        self.journal = journal
        self._lock = threading.RLock()
        self.current_turn_id: Optional[str] = None
        if recover:
            self.recover_interrupted_turn()

    def start_turn(self, message: Message) -> str:
        with self._lock:
            if self.current_turn_id is not None:
                raise SessionJournalError("Cannot start a session turn while another turn is open")
            turn_id = str(uuid.uuid4())
            stored, artifacts = self.journal.prepare_message(message.to_dict())
            self.journal.append(
                "turn.started",
                actor="user",
                payload={"message": stored},
                turn_id=turn_id,
                artifacts=artifacts,
            )
            self.current_turn_id = turn_id
            return turn_id

    def record_assistant(self, message: Message) -> None:
        with self._lock:
            turn_id = self._require_turn()
            stored, artifacts = self.journal.prepare_message(message.to_dict())
            self.journal.append(
                "assistant.message",
                actor="assistant",
                payload={"message": stored},
                turn_id=turn_id,
                artifacts=artifacts,
            )

    def record_tool_results(self, results: list[ToolResult]) -> list[ToolResult]:
        with self._lock:
            turn_id = self._require_turn()
            message = Message(role="user", content=cast(list[ContentBlock], results))
            stored, artifacts = self.journal.prepare_message(message.to_dict())
            self.journal.append(
                "tool.results",
                actor="tool",
                payload={"message": stored},
                turn_id=turn_id,
                artifacts=artifacts,
            )
            replayed = Message.from_dict(self.journal.hydrate_message(stored))
            if not isinstance(replayed.content, list) or not all(
                isinstance(item, ToolResult) for item in replayed.content
            ):
                raise SessionJournalError("Prepared tool result event did not replay as tool results")
            return cast(list[ToolResult], list(replayed.content))

    def record_context_message(self, message: Message, *, actor: Optional[str] = None) -> None:
        with self._lock:
            stored, artifacts = self.journal.prepare_message(message.to_dict())
            self.journal.append(
                "context.message",
                actor=actor or message.role,
                payload={"message": stored},
                turn_id=self.current_turn_id,
                artifacts=artifacts,
            )

    def record_compaction(self, compaction: dict[str, Any]) -> None:
        with self._lock:
            self.journal.append(
                "context.compacted",
                actor="system",
                payload={"compaction": copy.deepcopy(compaction)},
                turn_id=self.current_turn_id,
            )

    def finish_turn(self, status: str, *, error: Optional[str] = None) -> None:
        if status not in {"completed", "failed", "cancelled"}:
            raise ValueError(f"Unknown terminal turn status: {status}")
        with self._lock:
            turn_id = self._require_turn()
            payload: dict[str, Any] = {}
            if error:
                payload["error"] = error[:2000]
            self.journal.append(
                f"turn.{status}",
                actor="system",
                payload=payload,
                turn_id=turn_id,
            )
            self.current_turn_id = None

    def start_epoch(self, reason: str) -> str:
        with self._lock:
            if self.current_turn_id is not None:
                raise SessionJournalError("Cannot reset context while a session turn is open")
            return self.journal.start_epoch(reason)

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
    marker = (
        "\n\n[Middle of tool result omitted from model history. "
        f"Full output: {ref['path']} (sha256 {ref['sha256']}, {ref['chars']:,} characters).]\n\n"
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
