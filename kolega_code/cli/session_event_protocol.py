"""Public semantic session-event protocol (v2).

One projection serves every public surface: the live ``ask --json`` stream, the
``sessions export --format events-jsonl`` output, and the ATIF converter all
consume ``to_public_event``. The journal remains the single assigner of event
identity (id/seq/timestamp); this module only removes replay-private state and
serializes.

The public/private split is allowlist-shaped where it matters: artifact
references leave this module only for the shareable purposes (tool results,
images) and never carry local filesystem paths. Opaque provider fields
(signatures, encrypted reasoning) are structurally removed from message
payloads, matching what the TUI shows users. Configured secret values and the
machine's home directory are scrubbed from every string leaf.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
from pathlib import Path
from typing import IO, Any, Iterable, Optional, Sequence

from kolega_code.cli.session_journal import (
    DEFAULT_ROOT_AGENT_NAME,
    PUBLIC_EVENT_SCHEMA,
    SessionEvent,
    SessionJournal,
    SessionJournalError,
    derive_root_agent_id,
)
from kolega_code.security import redact_secrets_in_obj
from kolega_code.web.redaction import scrub_home_paths

PUBLIC_EVENT_VERSION = 2
UI_EVENT_PREFIX = "ui."

#: The only artifact purposes permitted in public output (kolega_code.events.ArtifactPurpose.SHAREABLE).
SHAREABLE_ARTIFACT_PURPOSES = frozenset({"tool_result", "image"})

#: Public artifact-reference keys; local ``path`` is deliberately absent.
_PUBLIC_ARTIFACT_KEYS = ("sha256", "bytes", "media_type", "purpose", "encoding", "chars")

# Content-block state that exists purely to be replayed to the provider.
OPAQUE_BLOCK_TYPES = {"redacted_thinking"}
OPAQUE_BLOCK_FIELDS = {
    "thinking": ("signature",),
    "tool_call": ("thought_signature",),
    "responses_reasoning": ("encrypted_content", "item_id"),
}

SYNTHETIC_ORIGIN = "synthetic"
LLM_ORIGIN = "llm"

#: Event payload keys whose value is a prepared Message dict.
_MESSAGE_BEARING_TYPES = {
    "turn.started",
    "assistant.message",
    "tool.results",
    "context.message",
    "context.session",
    "context.system",
    "llm.message",
}


def public_artifact_ref(ref: dict[str, Any]) -> dict[str, Any]:
    """Portable artifact reference: content address and metadata, no local path."""
    return {key: ref[key] for key in _PUBLIC_ARTIFACT_KEYS if key in ref}


def public_tool_call(block: dict[str, Any]) -> dict[str, Any]:
    """Public shape for one tool call: canonical id first, provider id retained.

    ``arguments`` is always a JSON object usable for ATIF conversion; the
    original ``input``/``input_kind`` are kept so free-form input is lossless.
    """
    input_kind = str(block.get("input_kind") or "json")
    original = block.get("input")
    if input_kind == "freeform" or not isinstance(original, dict):
        arguments: dict[str, Any] = {"input": original if isinstance(original, str) else str(original)}
    else:
        arguments = original
    return {
        "type": "tool_call",
        "tool_call_id": block.get("execution_id") or block.get("id"),
        "provider_call_id": block.get("id"),
        "name": block.get("name"),
        "input_kind": input_kind,
        "input": original,
        "arguments": arguments,
    }


def public_tool_result(block: dict[str, Any]) -> dict[str, Any]:
    """Public shape for one tool result, correlated by the same canonical id.

    v1 records may lack ``execution_id``; the provider ``tool_use_id`` is the
    documented fallback so a result always pairs with its call.
    """
    result: dict[str, Any] = {
        "type": "tool_result",
        "tool_call_id": block.get("execution_id") or block.get("tool_use_id"),
        "provider_call_id": block.get("tool_use_id"),
        "name": block.get("name"),
        "is_error": bool(block.get("is_error")),
        "input_kind": block.get("input_kind", "json"),
        "content": block.get("content"),
    }
    artifact = block.get("content_artifact")
    if isinstance(artifact, dict):
        result["content_artifact"] = public_artifact_ref(artifact)
    return result


def _public_blocks(blocks: list[Any]) -> list[Any]:
    cleaned: list[Any] = []
    for block in blocks:
        if not isinstance(block, dict):
            cleaned.append(block)
            continue
        block_type = block.get("type")
        if block_type in OPAQUE_BLOCK_TYPES:
            continue
        if block_type == "responses_reasoning" and not block.get("content"):
            # Encrypted-only reasoning (OpenAI/ChatGPT backends): nothing
            # readable remains once the opaque blob is stripped. DeepSeek
            # flash blocks carry readable text in `content` and are kept.
            continue
        if block_type == "tool_call":
            cleaned.append(public_tool_call(block))
            continue
        if block_type == "tool_result":
            public = public_tool_result(block)
            if isinstance(public.get("content"), list):
                public["content"] = _public_blocks(public["content"])
            cleaned.append(public)
            continue
        block = dict(block)
        for field in OPAQUE_BLOCK_FIELDS.get(block_type or "", ()):
            block.pop(field, None)
        artifact_fields = block.pop("artifact_fields", None)
        if isinstance(artifact_fields, dict):
            for field, ref in artifact_fields.items():
                if isinstance(ref, dict) and ref.get("purpose") in SHAREABLE_ARTIFACT_PURPOSES:
                    block[f"{field}_artifact"] = public_artifact_ref(ref)
        cleaned.append(block)
    return cleaned


def _public_message(message: dict[str, Any]) -> dict[str, Any]:
    public = dict(message)
    public.pop("usage_metadata", None)  # internal provider-shaped dict; `usage` is the contract
    content = public.get("content")
    if isinstance(content, list):
        public["content"] = _public_blocks(content)
    return public


def _scrub_home_in_obj(obj: Any) -> Any:
    if isinstance(obj, str):
        return scrub_home_paths(obj)
    if isinstance(obj, list):
        return [_scrub_home_in_obj(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _scrub_home_in_obj(value) for key, value in obj.items()}
    return obj


def to_public_event(event: SessionEvent, *, secret_values: Sequence[str] = ()) -> Optional[dict[str, Any]]:
    """Project one journal event to its public form, or None for `ui.*` events."""
    if event.event_type.startswith(UI_EVENT_PREFIX):
        return None
    payload = dict(event.payload)
    if event.event_type in _MESSAGE_BEARING_TYPES and isinstance(payload.get("message"), dict):
        payload["message"] = _public_message(payload["message"])
    payload = redact_secrets_in_obj(payload, list(secret_values), include_environment=True)
    payload = _scrub_home_in_obj(payload)
    return {
        "schema": PUBLIC_EVENT_SCHEMA,
        "version": PUBLIC_EVENT_VERSION,
        "id": event.event_id,
        "session_id": event.session_id,
        "seq": event.seq,
        "epoch_id": event.epoch_id,
        "turn_id": event.turn_id,
        "timestamp": event.timestamp,
        "actor": event.actor,
        "agent_id": event.agent_id or derive_root_agent_id(event.session_id),
        "agent_name": event.agent_name or DEFAULT_ROOT_AGENT_NAME,
        "parent_agent_id": event.parent_agent_id,
        "parent_tool_call_id": event.parent_tool_call_id,
        "depth": event.depth if event.depth is not None else 0,
        "type": event.event_type,
        "payload": payload,
        "artifacts": [
            public_artifact_ref(ref)
            for ref in event.artifacts
            if isinstance(ref, dict) and ref.get("purpose") in SHAREABLE_ARTIFACT_PURPOSES
        ],
    }


class SemanticStdoutPrinter:
    """Journal listener that streams public events as JSON lines.

    Registered via ``SessionJournal.add_listener``, so it receives the exact
    persisted event objects: a live line and the saved record share id, seq,
    and timestamp by construction. The only writer of protocol lines.
    """

    def __init__(self, *, secret_values: Sequence[str] = (), stream: Optional[IO[str]] = None) -> None:
        self._secret_values = list(secret_values)
        self._stream = stream if stream is not None else sys.stdout
        self._lock = threading.Lock()
        self._emitted_ids: set[str] = set()
        self.emitted_total = 0

    def __call__(self, event: SessionEvent) -> None:
        record = to_public_event(event, secret_values=self._secret_values)
        if record is None:
            return
        line = json.dumps(record, separators=(",", ":"), ensure_ascii=False, default=str)
        with self._lock:
            if event.event_id in self._emitted_ids:
                return
            self._emitted_ids.add(event.event_id)
            self._stream.write(line + "\n")
            self._stream.flush()
            self.emitted_total += 1

    def emit_backlog(self, events: Iterable[SessionEvent]) -> None:
        """Print events persisted before this printer was registered (e.g. the
        session bootstrap records of a session created by this invocation)."""
        for event in events:
            self(event)


class InMemorySessionJournal(SessionJournal):
    """Journal for unsaved runs: same id/seq/schema rules, nothing on disk.

    Creates no directories and never appears in ``sessions list``. Artifacts
    live in memory keyed by content hash; their references carry no ``path``.
    """

    def __init__(self, session_id: str) -> None:
        super().__init__(session_id, Path("<in-memory>") / session_id)
        self._records: list[SessionEvent] = []
        self._artifact_data: dict[str, bytes] = {}

    def _write_event_locked(self, event: SessionEvent) -> None:
        self._records.append(event)

    def _read_events_locked(self, *, repair_tail: bool) -> list[SessionEvent]:
        return list(self._records)

    def raw_events(self) -> str:
        with self._lock:
            return "".join(
                json.dumps(event.to_dict(), separators=(",", ":"), ensure_ascii=False) + "\n" for event in self._records
            )

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
            self._artifact_data[digest] = data
        ref: dict[str, Any] = {
            "sha256": digest,
            "bytes": len(data),
            "media_type": media_type,
            "purpose": purpose,
            "encoding": encoding,
        }
        if chars is not None:
            ref["chars"] = chars
        return ref

    def read_artifact(self, ref: dict[str, Any]) -> bytes:
        digest = str(ref.get("sha256") or "")
        with self._lock:
            data = self._artifact_data.get(digest)
        if data is None:
            raise SessionJournalError(f"Missing session artifact: {digest}")
        if hashlib.sha256(data).hexdigest() != digest:
            raise SessionJournalError(f"Session artifact failed integrity check: {digest}")
        return data
