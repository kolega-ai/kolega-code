"""ATIF v1.7 export: a deterministic, validated projection of the semantic journal.

The converter consumes the *public* event projection (``to_public_event``) —
never raw journal lines — so scrubbing, opaque-state removal, and canonical
tool identity are identical to every other public surface. Conversion is a
pure pipeline::

    TrajectorySource -> public events -> grouped trajectory drafts
        -> ATIF document dicts (+ asset plan) -> validate -> write

Everything up to the write is side-effect-free. The document is built as plain
dicts and validated with ``atif.Trajectory.model_validate`` before any output
exists; a failed conversion or validation writes nothing. File export is
atomic: document and assets are staged as temp siblings and renamed into
place, so a failure leaves any previous export intact.
"""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol, Sequence

from kolega_code.cli.session_event_protocol import to_public_event
from kolega_code.cli.session_journal import SessionEvent

ATIF_SCHEMA_VERSION = "ATIF-v1.7"
ATIF_AGENT_NAME = "kolega-code"
STATE_DIR_PLACEHOLDER = "<state-dir>"

_UUID_NS = uuid.uuid5(uuid.NAMESPACE_URL, "kolega-code:atif")
_IMAGE_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
}

#: Event types that fold into root ``extra.kolega`` rather than steps.
_EXTRA_ONLY_TYPES = {
    "session.created",
    "session.metadata_updated",
    "turn.completed",
    "turn.failed",
    "turn.cancelled",
    "run.completed",
    "run.failed",
    "run.cancelled",
    "goal.completed",
    "loop.sleeping",
    "loop.completed",
    "llm.run_started",
    "llm.request_failed",
    "agent.started",
    "agent.completed",
    "agent.failed",
}


class AtifExportError(RuntimeError):
    """Conversion or validation failed; no partial output was written."""


class AtifImagesNeedOutputError(AtifExportError):
    """Stdout export refused: the trajectory contains images, so assets need a
    file destination (``--output``) to hold portable relative paths."""


class TrajectorySource(Protocol):
    session_id: str

    def read_events(self, *, repair_tail: bool = True) -> list[SessionEvent]: ...

    def read_artifact(self, ref: dict[str, Any]) -> bytes: ...


@dataclass
class ConversionWarnings:
    entries: list[dict[str, Any]] = field(default_factory=list)
    _seen: set[tuple[str, str]] = field(default_factory=set)

    def add(self, code: str, message: str, *, seq: Optional[int] = None) -> None:
        key = (code, message)
        if key in self._seen:
            return
        self._seen.add(key)
        entry: dict[str, Any] = {"code": code, "message": message}
        if seq is not None:
            entry["seq"] = seq
        self.entries.append(entry)

    def to_extra(self) -> list[dict[str, Any]]:
        return list(self.entries)

    def to_notes(self) -> Optional[str]:
        if not self.entries:
            return None
        lines = [f"- {entry['code']}: {entry['message']}" for entry in self.entries]
        return "kolega-code conversion notes:\n" + "\n".join(lines)


class _AssetPlan:
    """Collects shareable image bytes for the ``<output-stem>.assets/`` dir."""

    def __init__(self, *, assets_dir_name: Optional[str], source: TrajectorySource) -> None:
        self._assets_dir_name = assets_dir_name
        self._source = source
        self._files: dict[str, bytes] = {}
        self.has_images = False

    def add_image_ref(self, ref: dict[str, Any], *, warnings: ConversionWarnings, seq: int) -> Optional[dict[str, Any]]:
        if ref.get("purpose") != "image":
            raise AtifExportError(f"Refusing to materialize a non-image artifact: {ref.get('purpose')}")
        media_type = str(ref.get("media_type") or "")
        extension = _IMAGE_EXTENSIONS.get(media_type)
        if extension is None:
            warnings.add(
                "unsupported_image_media_type",
                f"Image with media type {media_type or 'unknown'} is retained by reference only.",
                seq=seq,
            )
            return None
        self.has_images = True
        if self._assets_dir_name is None:
            # Building for stdout: remember that images exist so the caller can
            # refuse before emitting anything; bytes are not materialized.
            return None
        data = self._source.read_artifact(ref)
        digest = str(ref["sha256"])
        filename = f"{digest}{extension}"
        self._files[filename] = data
        return {"media_type": media_type, "path": f"{self._assets_dir_name}/{filename}"}

    def files(self) -> list[tuple[str, bytes]]:
        return sorted(self._files.items())


@dataclass
class _StepDraft:
    seq: int
    event_id: str
    timestamp: Optional[str]
    source: str  # system | user | agent
    message: Any = ""
    model_name: Optional[str] = None
    reasoning_effort: Optional[Any] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None
    observation_results: list[dict[str, Any]] = field(default_factory=list)
    metrics: Optional[dict[str, Any]] = None
    llm_call_count: Optional[int] = None
    is_copied_context: Optional[bool] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class _TrajectoryDraft:
    agent_id: str
    agent_name: str
    depth: int
    parent_agent_id: Optional[str] = None
    parent_tool_call_id: Optional[str] = None
    task: Optional[str] = None
    status: Optional[str] = None
    summary: Optional[str] = None
    model_name: Optional[str] = None
    tool_definitions: Optional[list[dict[str, Any]]] = None
    steps: list[_StepDraft] = field(default_factory=list)
    call_owner: dict[str, int] = field(default_factory=dict)  # tool_call_id -> steps index
    extra: dict[str, Any] = field(default_factory=dict)


def _derived_id(kind: str, *parts: Any) -> str:
    return str(uuid.uuid5(_UUID_NS, ":".join([kind, *[str(part) for part in parts]])))


def _message_blocks(message: Any) -> list[dict[str, Any]]:
    if not isinstance(message, dict) or not isinstance(message.get("content"), list):
        return []
    return [block for block in message["content"] if isinstance(block, dict)]


def _step_message_and_reasoning(
    message: dict[str, Any],
    *,
    event: dict[str, Any],
    assets: _AssetPlan,
    warnings: ConversionWarnings,
) -> tuple[Any, Optional[str], Optional[list[dict[str, Any]]], list[dict[str, Any]]]:
    """Project one prepared message: (message, reasoning, tool_calls, content_blocks)."""
    text_parts: list[str] = []
    content_parts: list[dict[str, Any]] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    content_blocks: list[dict[str, Any]] = []
    has_image = False
    seq = int(event["seq"])

    for index, block in enumerate(_message_blocks(message)):
        block_type = block.get("type")
        record: dict[str, Any] = {"index": index, "type": block_type}
        if block_type == "text":
            text = str(block.get("text") or "")
            text_parts.append(text)
            content_parts.append({"type": "text", "text": text})
            record["disposition"] = "message"
        elif block_type == "thinking":
            reasoning_parts.append(str(block.get("thinking") or ""))
            record["disposition"] = "reasoning"
        elif block_type == "responses_reasoning":
            texts = [str(item) for item in block.get("content") or []]
            if texts:
                reasoning_parts.append("\n".join(texts))
            record["disposition"] = "reasoning"
        elif block_type == "tool_call":
            call_id = block.get("tool_call_id") or _derived_id("tool", event["id"], index)
            if not block.get("tool_call_id"):
                warnings.add(
                    "v1_derived_tool_call_id",
                    "A tool call without a recorded id was assigned a derived stable id.",
                    seq=seq,
                )
            call = {
                "tool_call_id": str(call_id),
                "function_name": str(block.get("name") or "unknown_tool"),
                "arguments": block.get("arguments") if isinstance(block.get("arguments"), dict) else {},
                "extra": {
                    "kolega": {
                        "provider_call_id": block.get("provider_call_id"),
                        "input_kind": block.get("input_kind"),
                    }
                },
            }
            if block.get("input_kind") == "freeform":
                call["extra"]["kolega"]["input"] = block.get("input")
            tool_calls.append(call)
            record["disposition"] = "tool_call"
            record["tool_call_id"] = call["tool_call_id"]
        elif block_type == "image_url":
            ref = block.get("data_artifact")
            source = None
            if isinstance(ref, dict):
                source = assets.add_image_ref(ref, warnings=warnings, seq=seq)
            if source is not None:
                has_image = True
                content_parts.append({"type": "image", "source": source})
                record["disposition"] = "message"
            else:
                record["disposition"] = "retained"
                record["block"] = {k: v for k, v in block.items() if k != "data"}
        else:
            # Unsupported public block (web_search_call, future types): retained
            # losslessly, never silently discarded.
            warnings.add(
                "unsupported_content_block",
                f"Content blocks of type {block_type} are retained in step extra, not mapped.",
                seq=seq,
            )
            record["disposition"] = "retained"
            record["block"] = dict(block)
        content_blocks.append(record)

    step_message: Any
    if has_image:
        step_message = content_parts
    else:
        step_message = "\n\n".join(part for part in text_parts if part) if text_parts else ""
    reasoning = "\n\n".join(part for part in reasoning_parts if part) or None
    return step_message, reasoning, tool_calls or None, content_blocks


def _hydrated_result_content(
    block: dict[str, Any],
    *,
    source: TrajectorySource,
    assets: _AssetPlan,
    event: dict[str, Any],
    warnings: ConversionWarnings,
) -> Any:
    """Full tool-result content: artifact-backed text is hydrated, images copied."""
    artifact = block.get("content_artifact")
    if isinstance(artifact, dict):
        try:
            return source.read_artifact(artifact).decode("utf-8")
        except Exception as exc:
            raise AtifExportError(f"Could not hydrate an oversized tool result: {exc}") from exc
    content = block.get("content")
    if isinstance(content, list):
        parts: list[dict[str, Any]] = []
        for nested in content:
            if not isinstance(nested, dict):
                continue
            if nested.get("type") == "text":
                parts.append({"type": "text", "text": str(nested.get("text") or "")})
            elif nested.get("type") == "image_url" and isinstance(nested.get("data_artifact"), dict):
                image = assets.add_image_ref(nested["data_artifact"], warnings=warnings, seq=int(event["seq"]))
                if image is not None:
                    parts.append({"type": "image", "source": image})
            else:
                warnings.add(
                    "unsupported_content_block",
                    f"Tool-result blocks of type {nested.get('type')} are not mapped.",
                    seq=int(event["seq"]),
                )
        return parts
    return str(content or "")


def _metrics_from_usage(usage: Any, *, warnings: ConversionWarnings, seq: int) -> Optional[dict[str, Any]]:
    if not isinstance(usage, dict) or not usage.get("reported"):
        warnings.add(
            "usage_missing_for_llm_step",
            "One or more LLM steps have no reported usage; totals cover reported steps only.",
            seq=seq,
        )
        return None
    metrics: dict[str, Any] = {}
    mapping = {
        "prompt_tokens": "input_tokens",
        "completion_tokens": "output_tokens",
        "cached_tokens": "cache_read_input_tokens",
    }
    for target, key in mapping.items():
        value = usage.get(key)
        if isinstance(value, int):
            metrics[target] = value
    extra: dict[str, Any] = {}
    for key in ("reasoning_output_tokens", "cache_write_input_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            extra[key if key != "reasoning_output_tokens" else "reasoning_tokens"] = value
    if extra:
        metrics["extra"] = extra
    return metrics or None


def _final_metrics(steps: list[_StepDraft], *, warnings: ConversionWarnings) -> Optional[dict[str, Any]]:
    llm_steps = [step for step in steps if step.llm_call_count == 1]
    with_usage = [step for step in llm_steps if step.metrics]
    if llm_steps and len(with_usage) < len(llm_steps):
        warnings.add(
            "usage_totals_partial",
            f"{len(llm_steps) - len(with_usage)} of {len(llm_steps)} LLM steps carry no usage; "
            "final metrics sum reported values only.",
        )
    totals: dict[str, Any] = {"total_steps": len(steps)}
    for target, source_key in (
        ("total_prompt_tokens", "prompt_tokens"),
        ("total_completion_tokens", "completion_tokens"),
        ("total_cached_tokens", "cached_tokens"),
    ):
        values = [step.metrics[source_key] for step in with_usage if step.metrics and source_key in step.metrics]
        if values:
            totals[target] = sum(values)
    extra: dict[str, Any] = {"llm_steps": len(llm_steps), "llm_steps_with_usage": len(with_usage)}
    for key in ("reasoning_tokens", "cache_write_input_tokens"):
        values = [
            step.metrics["extra"][key]
            for step in with_usage
            if step.metrics and isinstance(step.metrics.get("extra"), dict) and key in step.metrics["extra"]
        ]
        if values:
            extra[key] = sum(values)
    totals["extra"] = extra
    return totals


class _Grouper:
    """One seq-ordered pass over public events -> trajectory drafts."""

    def __init__(self, session_id: str, *, source: TrajectorySource, assets: _AssetPlan, warnings: ConversionWarnings):
        self.session_id = session_id
        self.source = source
        self.assets = assets
        self.warnings = warnings
        self.root: Optional[_TrajectoryDraft] = None
        self.subagents: dict[str, _TrajectoryDraft] = {}
        self.root_extra: dict[str, Any] = {
            "session": {},
            "turns": [],
            "epochs": [],
            "unknown_events": [],
        }
        self._saw_v1 = False

    # -- draft resolution ---------------------------------------------------

    def _draft_for(self, event: dict[str, Any]) -> _TrajectoryDraft:
        depth = int(event.get("depth") or 0)
        agent_id = str(event["agent_id"])
        if depth == 0:
            if self.root is None:
                self.root = _TrajectoryDraft(agent_id=agent_id, agent_name=str(event["agent_name"]), depth=0)
            return self.root
        draft = self.subagents.get(agent_id)
        if draft is None:
            draft = _TrajectoryDraft(
                agent_id=agent_id,
                agent_name=str(event["agent_name"]),
                depth=depth,
                parent_agent_id=event.get("parent_agent_id"),
                parent_tool_call_id=event.get("parent_tool_call_id"),
            )
            self.subagents[agent_id] = draft
        return draft

    def _append_step(self, draft: _TrajectoryDraft, step: _StepDraft) -> None:
        step.extra.setdefault("kolega", {}).update({"event_id": step.event_id, "seq": step.seq})
        draft.steps.append(step)

    def _system_step(self, event: dict[str, Any], text: str, *, extra: Optional[dict[str, Any]] = None) -> _StepDraft:
        step = _StepDraft(
            seq=int(event["seq"]),
            event_id=str(event["id"]),
            timestamp=event.get("timestamp"),
            source="system",
            message=text,
        )
        if extra:
            step.extra.update(extra)
        return step

    # -- event handlers -----------------------------------------------------

    def feed(self, event: dict[str, Any]) -> None:
        event_type = str(event["type"])
        draft = self._draft_for(event)
        payload = event.get("payload") or {}
        handler = {
            "turn.started": self._on_turn_started,
            "context.system": self._on_context_system,
            "context.session": self._on_context_session,
            "context.tools": self._on_context_tools,
            "context.message": self._on_context_message,
            "assistant.message": self._on_assistant_message,
            "llm.message": self._on_llm_message,
            "tool.results": self._on_tool_results,
            "context.compacted": self._on_compacted,
            "context.rewound": self._on_rewound,
            "context.epoch_started": self._on_epoch_started,
            "session.workspace_switched": self._on_workspace_switched,
            "goal.evaluated": self._on_goal_evaluated,
            "loop.iteration_started": self._on_loop_iteration,
            "skill.activated": self._on_skill_activated,
        }.get(event_type)
        if handler is not None:
            handler(draft, event, payload)
            return
        if event_type in _EXTRA_ONLY_TYPES:
            self._fold_extra(draft, event, payload)
            return
        self.root_extra["unknown_events"].append({"type": event_type, "seq": event["seq"]})
        self.warnings.add("unknown_event_type", f"Unknown event type {event_type} kept in extra only.")

    def _on_turn_started(self, draft: _TrajectoryDraft, event: dict[str, Any], payload: dict[str, Any]) -> None:
        if payload.get("continuation"):
            # A continuation turn carries no user utterance; keep the boundary
            # in extra instead of fabricating an empty user step. Root drafts
            # surface extra via root_extra; subagent drafts via their own extra.
            record = {"turn_id": event.get("turn_id"), "seq": int(event["seq"])}
            if draft is self.root:
                self.root_extra.setdefault("continuation_turns", []).append(record)
            else:
                draft.extra.setdefault("continuation_turns", []).append(record)
            return
        message, _, _, content_blocks = _step_message_and_reasoning(
            payload.get("message") or {}, event=event, assets=self.assets, warnings=self.warnings
        )
        step = _StepDraft(
            seq=int(event["seq"]),
            event_id=str(event["id"]),
            timestamp=event.get("timestamp"),
            source="user",
            message=message,
        )
        step.extra["kolega"] = {"turn_id": event.get("turn_id"), "content_blocks": content_blocks}
        self._append_step(draft, step)

    def _on_context_system(self, draft: _TrajectoryDraft, event: dict[str, Any], payload: dict[str, Any]) -> None:
        message, _, _, _ = _step_message_and_reasoning(
            payload.get("message") or {}, event=event, assets=self.assets, warnings=self.warnings
        )
        self._append_step(draft, self._system_step(event, message if isinstance(message, str) else ""))

    def _on_context_session(self, draft: _TrajectoryDraft, event: dict[str, Any], payload: dict[str, Any]) -> None:
        message, _, _, content_blocks = _step_message_and_reasoning(
            payload.get("message") or {}, event=event, assets=self.assets, warnings=self.warnings
        )
        step = self._system_step(event, message if isinstance(message, str) else "")
        step.extra["kolega"] = {"origin": "context.session", "content_blocks": content_blocks}
        self._append_step(draft, step)

    def _on_context_tools(self, draft: _TrajectoryDraft, event: dict[str, Any], payload: dict[str, Any]) -> None:
        # Not a step: the toolset describes the agent, not a context change the
        # model observed as content. Latest recording wins.
        tools = payload.get("tools")
        if isinstance(tools, list) and tools:
            draft.tool_definitions = tools

    def _on_context_message(self, draft: _TrajectoryDraft, event: dict[str, Any], payload: dict[str, Any]) -> None:
        message, _, _, content_blocks = _step_message_and_reasoning(
            payload.get("message") or {}, event=event, assets=self.assets, warnings=self.warnings
        )
        source = "user" if event.get("actor") == "user" else "system"
        step = _StepDraft(
            seq=int(event["seq"]),
            event_id=str(event["id"]),
            timestamp=event.get("timestamp"),
            source=source,
            message=message,
        )
        step.extra["kolega"] = {"origin": "context.message", "content_blocks": content_blocks}
        self._append_step(draft, step)

    def _assistant_step(
        self, draft: _TrajectoryDraft, event: dict[str, Any], payload: dict[str, Any], *, llm_call_id: Optional[str]
    ) -> None:
        message, reasoning, tool_calls, content_blocks = _step_message_and_reasoning(
            payload.get("message") or {}, event=event, assets=self.assets, warnings=self.warnings
        )
        synthetic = payload.get("origin_type") == "synthetic"
        step = _StepDraft(
            seq=int(event["seq"]),
            event_id=str(event["id"]),
            timestamp=event.get("timestamp"),
            source="agent",
            message=message,
            llm_call_count=0 if synthetic else 1,
        )
        if not synthetic:
            step.model_name = payload.get("model")
            step.reasoning_effort = payload.get("reasoning_effort")
            step.reasoning_content = reasoning
            step.tool_calls = tool_calls
            usage = (payload.get("message") or {}).get("usage")
            step.metrics = _metrics_from_usage(usage, warnings=self.warnings, seq=int(event["seq"]))
            if draft.model_name is None and step.model_name:
                draft.model_name = step.model_name
        kolega_extra: dict[str, Any] = {"content_blocks": content_blocks}
        if llm_call_id:
            kolega_extra["llm_call_id"] = llm_call_id
        if synthetic:
            kolega_extra["origin_type"] = "synthetic"
            if payload.get("notice_code"):
                kolega_extra["notice_code"] = payload["notice_code"]
            if tool_calls:
                # A synthetic notice never carries tool calls today; retain
                # defensively rather than violating the ATIF gating.
                kolega_extra["tool_calls"] = tool_calls
        step.extra["kolega"] = kolega_extra
        self._append_step(draft, step)
        for call in step.tool_calls or []:
            draft.call_owner[call["tool_call_id"]] = len(draft.steps) - 1

    def _on_assistant_message(self, draft: _TrajectoryDraft, event: dict[str, Any], payload: dict[str, Any]) -> None:
        llm_call_id = payload.get("llm_call_id")
        if not llm_call_id and payload.get("origin_type") != "synthetic":
            llm_call_id = _derived_id("llm", event["id"])
            self.warnings.add(
                "v1_derived_llm_call_id",
                "Assistant messages recorded before call-identity stamping use ids derived from their event ids.",
                seq=int(event["seq"]),
            )
        self._assistant_step(draft, event, payload, llm_call_id=llm_call_id)

    def _on_llm_message(self, draft: _TrajectoryDraft, event: dict[str, Any], payload: dict[str, Any]) -> None:
        origin = payload.get("origin") or {}
        kind = origin.get("kind")
        if kind == "sub_agent" and origin.get("agent_id"):
            # v1 sessions: the only durable record of subagent inference.
            sub = self.subagents.get(str(origin["agent_id"]))
            if sub is None:
                sub = _TrajectoryDraft(
                    agent_id=str(origin["agent_id"]),
                    agent_name=str(origin.get("agent_name") or "subagent"),
                    depth=int(origin.get("depth") or 1),
                    parent_agent_id=self.root.agent_id if self.root else None,
                    parent_tool_call_id=origin.get("parent_tool_call_id"),
                )
                self.subagents[sub.agent_id] = sub
                self.warnings.add(
                    "v1_subagent_tools_unrecorded",
                    "Subagent trajectories recovered from llm.message events contain LLM steps only; "
                    "their tool results and turn boundaries were never durably recorded.",
                )
            enriched = dict(payload)
            enriched.setdefault("llm_call_id", payload.get("request_id"))
            enriched.setdefault("origin_type", "llm")
            self._assistant_step(sub, event, enriched, llm_call_id=enriched.get("llm_call_id"))
            return
        # Helper calls (compaction summarizer, goal verifier, hooks): real
        # spend that belongs to no trajectory step.
        usage = (payload.get("message") or {}).get("usage") or {}
        bucket = self.root_extra.setdefault(
            "unattributed_usage", {"requests": 0, "input_tokens": 0, "output_tokens": 0}
        )
        bucket["requests"] += 1
        for key in ("input_tokens", "output_tokens"):
            value = usage.get(key)
            if isinstance(value, int):
                bucket[key] += value
        self.warnings.add(
            "unattributed_helper_usage",
            "Helper LLM calls (compaction, goal verification, hooks) are summed in extra.kolega.unattributed_usage.",
        )

    def _on_tool_results(self, draft: _TrajectoryDraft, event: dict[str, Any], payload: dict[str, Any]) -> None:
        for block in _message_blocks(payload.get("message")):
            if block.get("type") != "tool_result":
                continue
            call_id = block.get("tool_call_id")
            content = _hydrated_result_content(
                block, source=self.source, assets=self.assets, event=event, warnings=self.warnings
            )
            result: dict[str, Any] = {
                "content": content,
                "extra": {
                    "kolega": {
                        "is_error": bool(block.get("is_error")),
                        "name": block.get("name"),
                        "provider_call_id": block.get("provider_call_id"),
                        "event_id": event["id"],
                        "seq": event["seq"],
                    }
                },
            }
            artifact = block.get("content_artifact")
            if isinstance(artifact, dict):
                result["extra"]["kolega"]["hydrated_from_artifact"] = {
                    key: artifact[key] for key in ("sha256", "bytes", "chars") if key in artifact
                }
            owner = draft.call_owner.get(str(call_id)) if call_id else None
            if owner is not None:
                result["source_call_id"] = str(call_id)
                draft.steps[owner].observation_results.append(result)
            else:
                self.warnings.add(
                    "unmatched_tool_result",
                    "A tool result could not be correlated to a recorded call; kept as a system observation.",
                    seq=int(event["seq"]),
                )
                result["extra"]["kolega"]["uncorrelated_call_id"] = call_id
                step = self._system_step(event, "")
                step.observation_results.append(result)
                self._append_step(draft, step)

    def _on_compacted(self, draft: _TrajectoryDraft, event: dict[str, Any], payload: dict[str, Any]) -> None:
        compaction = payload.get("compaction") or {}
        summary = ""
        summary_message = compaction.get("summary")
        if isinstance(summary_message, dict):
            summary = "\n".join(
                str(block.get("text") or "")
                for block in _message_blocks(summary_message)
                if block.get("type") == "text"
            )
        provenance = {key: payload[key] for key in payload if key not in ("compaction",)}
        if not provenance:
            self.warnings.add(
                "v1_compaction_missing_token_info",
                "Compactions recorded before provenance enrichment carry no trigger/token information.",
            )
        step = self._system_step(
            event,
            "",
            extra={
                "context_management": {"type": "compaction", "boundary": "replace"},
                "kolega": {"compaction": provenance or None},
            },
        )
        step.observation_results.append({"content": summary})
        self._append_step(draft, step)

    def _on_rewound(self, draft: _TrajectoryDraft, event: dict[str, Any], payload: dict[str, Any]) -> None:
        step = self._system_step(
            event,
            f"Context rewound to turn {payload.get('target_turn_id')}",
            extra={"kolega": {"rewind": dict(payload)}},
        )
        self._append_step(draft, step)

    def _on_epoch_started(self, draft: _TrajectoryDraft, event: dict[str, Any], payload: dict[str, Any]) -> None:
        self.root_extra["epochs"].append(
            {"epoch_id": event.get("epoch_id"), "reason": payload.get("reason"), "seq": event["seq"]}
        )
        if len(self.root_extra["epochs"]) > 1:
            step = self._system_step(
                event,
                f"Context epoch started: {payload.get('reason')}",
                extra={"kolega": {"epoch": {"epoch_id": event.get("epoch_id"), "reason": payload.get("reason")}}},
            )
            self._append_step(draft, step)

    def _on_workspace_switched(self, draft: _TrajectoryDraft, event: dict[str, Any], payload: dict[str, Any]) -> None:
        old = payload.get("new_label") or payload.get("new_branch") or "workspace"
        step = self._system_step(
            event,
            f"Workspace switched to {old}",
            extra={
                "kolega": {
                    "workspace_switch": {
                        key: payload.get(key) for key in ("old_label", "new_label", "old_branch", "new_branch")
                    }
                }
            },
        )
        self._append_step(draft, step)

    def _on_goal_evaluated(self, draft: _TrajectoryDraft, event: dict[str, Any], payload: dict[str, Any]) -> None:
        met = payload.get("met")
        step = self._system_step(
            event,
            f"Goal evaluated: {'met' if met else 'not met'} — {payload.get('reason')}",
            extra={"kolega": {"goal_evaluated": dict(payload)}},
        )
        self._append_step(draft, step)

    def _on_loop_iteration(self, draft: _TrajectoryDraft, event: dict[str, Any], payload: dict[str, Any]) -> None:
        step = self._system_step(
            event,
            f"Loop iteration {payload.get('iteration')} started",
            extra={"kolega": {"loop_iteration": dict(payload)}},
        )
        self._append_step(draft, step)

    def _on_skill_activated(self, draft: _TrajectoryDraft, event: dict[str, Any], payload: dict[str, Any]) -> None:
        step = self._system_step(
            event,
            f"Skill activated: {payload.get('name')}",
            extra={"kolega": {"skill_activated": dict(payload)}},
        )
        self._append_step(draft, step)

    def _fold_extra(self, draft: _TrajectoryDraft, event: dict[str, Any], payload: dict[str, Any]) -> None:
        event_type = str(event["type"])
        if event_type == "session.created":
            metadata = payload.get("metadata") or {}
            config = metadata.get("config") or {}
            self.root_extra["session"] = {
                "kolega_version": payload.get("kolega_version"),
                "mode": metadata.get("mode"),
                "title": metadata.get("title"),
                "project": Path(str(metadata.get("project_path") or "")).name or None,
                "created_at": metadata.get("created_at"),
                "config": {
                    key: config.get(key)
                    for key in ("long_provider", "long_model", "thinking_effort")
                    if isinstance(config, dict) and key in config
                },
            }
        elif event_type.startswith("turn."):
            self.root_extra["turns"].append(
                {
                    "turn_id": event.get("turn_id"),
                    "status": event_type.split(".", 1)[1],
                    "agent_id": event.get("agent_id"),
                    **{k: payload[k] for k in ("error", "duration_ms") if k in payload},
                }
            )
        elif event_type.startswith("run."):
            self.root_extra["status"] = {"outcome": event_type.split(".", 1)[1], **payload}
        elif event_type == "goal.completed":
            self.root_extra["goal"] = dict(payload)
        elif event_type in ("loop.sleeping", "loop.completed"):
            if event_type == "loop.completed":
                self.root_extra["loop"] = dict(payload)
        elif event_type == "llm.request_failed":
            self.root_extra.setdefault("llm_failures", []).append(
                {key: payload.get(key) for key in ("provider", "model", "error")} | {"seq": event["seq"]}
            )
        elif event_type == "agent.started":
            draft.task = payload.get("task")
            if payload.get("agent_name"):
                draft.agent_name = str(payload["agent_name"])
            draft.extra["started"] = {
                key: payload.get(key) for key in ("class_name", "requested_routing", "effective_routing")
            }
        elif event_type in ("agent.completed", "agent.failed"):
            draft.status = event_type.split(".", 1)[1]
            draft.summary = payload.get("summary")
            draft.extra["terminal"] = dict(payload)
        # session.metadata_updated and llm.run_started carry nothing ATIF needs.


def _project_trajectory(
    draft: _TrajectoryDraft,
    *,
    warnings: ConversionWarnings,
    agent: dict[str, Any],
    session_id: Optional[str],
    extra: Optional[dict[str, Any]],
    subagent_refs: dict[str, str],
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    for index, step_draft in enumerate(sorted(draft.steps, key=lambda item: item.seq), start=1):
        step: dict[str, Any] = {
            "step_id": index,
            "source": step_draft.source,
            "message": step_draft.message,
        }
        if step_draft.timestamp:
            step["timestamp"] = step_draft.timestamp
        if step_draft.model_name:
            step["model_name"] = step_draft.model_name
        if step_draft.reasoning_effort is not None:
            step["reasoning_effort"] = step_draft.reasoning_effort
        if step_draft.reasoning_content:
            step["reasoning_content"] = step_draft.reasoning_content
        if step_draft.tool_calls:
            step["tool_calls"] = step_draft.tool_calls
        if step_draft.llm_call_count is not None:
            step["llm_call_count"] = step_draft.llm_call_count
        results = list(step_draft.observation_results)
        # Attach embedded-subagent refs to the dispatch call's observation.
        for call in step_draft.tool_calls or []:
            child_id = subagent_refs.get(call["tool_call_id"])
            if child_id is None:
                continue
            ref = {"trajectory_id": child_id}
            existing = next((r for r in results if r.get("source_call_id") == call["tool_call_id"]), None)
            if existing is not None:
                existing.setdefault("subagent_trajectory_ref", []).append(ref)
            else:
                results.append({"source_call_id": call["tool_call_id"], "subagent_trajectory_ref": [ref]})
        if results:
            step["observation"] = {"results": results}
        if step_draft.metrics:
            step["metrics"] = step_draft.metrics
        if step_draft.is_copied_context is not None:
            step["is_copied_context"] = step_draft.is_copied_context
        if step_draft.extra:
            step["extra"] = step_draft.extra
        steps.append(step)

    document: dict[str, Any] = {
        "schema_version": ATIF_SCHEMA_VERSION,
        "trajectory_id": draft.agent_id,
        "agent": agent,
        "steps": steps,
    }
    if session_id:
        document["session_id"] = session_id
    metrics = _final_metrics(sorted(draft.steps, key=lambda item: item.seq), warnings=warnings)
    if metrics:
        document["final_metrics"] = metrics
    if extra:
        document["extra"] = extra
    return document


def build_atif_document(
    source: TrajectorySource,
    *,
    session_metadata: Optional[dict[str, Any]] = None,
    kolega_version: str,
    secret_values: Sequence[str] = (),
    state_dirs: Sequence[Path] = (),
    assets_dir_name: Optional[str] = None,
) -> tuple[dict[str, Any], _AssetPlan]:
    """Convert one session's journal into an unvalidated ATIF document dict."""
    warnings = ConversionWarnings()
    assets = _AssetPlan(assets_dir_name=assets_dir_name, source=source)
    grouper = _Grouper(source.session_id, source=source, assets=assets, warnings=warnings)

    raw_events = source.read_events(repair_tail=True)
    if not raw_events:
        raise AtifExportError(f"Session {source.session_id} has no recorded events to convert")
    if any(event.version == 1 for event in raw_events):
        warnings.add(
            "v1_source_journal",
            "This session was recorded with journal schema v1; deterministic fallbacks were applied.",
        )
        warnings.add(
            "v1_derived_agent_id",
            "The root agent identity is derived from the session id.",
        )
    for event in raw_events:
        public = to_public_event(event, secret_values=secret_values)
        if public is not None:
            grouper.feed(public)

    if grouper.root is None:
        raise AtifExportError(f"Session {source.session_id} contains no root-agent events")

    session_info = grouper.root_extra.get("session") or {}
    config = session_info.get("config") or {}
    if session_metadata:
        metadata_config = session_metadata.get("config") or {}
        for key in ("long_provider", "long_model", "thinking_effort"):
            if key in metadata_config and key not in config:
                config[key] = metadata_config[key]
    root_agent = {
        "name": ATIF_AGENT_NAME,
        "version": kolega_version,
        "model_name": grouper.root.model_name or config.get("long_model"),
        "extra": {
            "provider": config.get("long_provider"),
            "reasoning_effort": config.get("thinking_effort"),
            "mode": session_info.get("mode"),
            "profile": grouper.root.agent_name,
        },
    }
    if grouper.root.tool_definitions:
        root_agent["tool_definitions"] = grouper.root.tool_definitions
    else:
        warnings.add(
            "tool_definitions_unavailable",
            "This session recorded no tool schemas (pre-context.tools journal); agent.tool_definitions is omitted.",
        )

    subagent_refs = {
        draft.parent_tool_call_id: draft.agent_id for draft in grouper.subagents.values() if draft.parent_tool_call_id
    }

    subagent_documents = []
    for child in grouper.subagents.values():
        child_agent: dict[str, Any] = {
            "name": ATIF_AGENT_NAME,
            "version": kolega_version,
            "model_name": child.model_name,
            "extra": {
                "agent_name": child.agent_name,
                "task": child.task,
                "depth": child.depth,
                "parent_agent_id": child.parent_agent_id,
                "parent_tool_call_id": child.parent_tool_call_id,
                "status": child.status,
                **({"summary": child.summary} if child.summary else {}),
            },
        }
        if child.tool_definitions:
            child_agent["tool_definitions"] = child.tool_definitions
        subagent_documents.append(
            _project_trajectory(
                child,
                warnings=warnings,
                agent=child_agent,
                session_id=None,
                extra={"kolega": child.extra} if child.extra else None,
                subagent_refs=subagent_refs,
            )
        )

    root_extra_kolega = {key: value for key, value in grouper.root_extra.items() if value}
    document = _project_trajectory(
        grouper.root,
        warnings=warnings,
        agent=root_agent,
        session_id=source.session_id,
        extra=None,
        subagent_refs=subagent_refs,
    )
    if subagent_documents:
        document["subagent_trajectories"] = subagent_documents
    notes = warnings.to_notes()
    if notes:
        document["notes"] = notes
    document["extra"] = {"conversion_warnings": warnings.to_extra(), "kolega": root_extra_kolega}

    document = _scrub_state_dirs(document, state_dirs)
    return document, assets


def _scrub_state_dirs(document: dict[str, Any], state_dirs: Sequence[Path]) -> dict[str, Any]:
    """Replace any state-directory prefix with a placeholder, everywhere.

    Secrets and home paths were already scrubbed by the public projection; the
    state dir gets its own, more specific placeholder (it usually lives under
    home, so this runs against both raw and home-rewritten spellings).
    """
    prefixes: set[str] = set()
    home = Path.home()
    for state_dir in state_dirs:
        for form in (state_dir, state_dir.expanduser(), state_dir.resolve()):
            text = str(form).rstrip("/")
            if len(text) > 1:
                prefixes.add(text)
            try:
                prefixes.add("~/" + str(form.relative_to(home)))
            except ValueError:
                pass
    if not prefixes:
        return document
    ordered = sorted(prefixes, key=len, reverse=True)

    def scrub(value: Any) -> Any:
        if isinstance(value, str):
            for prefix in ordered:
                if prefix in value:
                    value = value.replace(prefix, STATE_DIR_PLACEHOLDER)
            return value
        if isinstance(value, list):
            return [scrub(item) for item in value]
        if isinstance(value, dict):
            return {key: scrub(item) for key, item in value.items()}
        return value

    return scrub(document)


def validate_atif_document(document: dict[str, Any]) -> Any:
    import atif

    try:
        return atif.Trajectory.model_validate(document)
    except Exception as exc:
        raise AtifExportError(f"Generated ATIF document failed validation (kolega-code bug): {exc}") from exc


def render_atif_text(document: dict[str, Any]) -> str:
    validated = validate_atif_document(document)
    return json.dumps(validated.to_json_dict(exclude_none=True), indent=2, ensure_ascii=False, default=str) + "\n"


def export_atif_to_text(source: TrajectorySource, **kwargs: Any) -> str:
    """Build, validate, and render for stdout. Refuses multimodal output."""
    kwargs.pop("assets_dir_name", None)
    document, assets = build_atif_document(source, assets_dir_name=None, **kwargs)
    if assets.has_images:
        raise AtifImagesNeedOutputError(
            "This trajectory contains image content; pass --output FILE so assets "
            "can be written to FILE-stem.assets/ with portable relative paths."
        )
    return render_atif_text(document)


def export_atif_to_path(source: TrajectorySource, *, output: Path, **kwargs: Any) -> Path:
    """Atomic file export: temp siblings, hash-verified assets, rename into place."""
    output = output.expanduser()
    assets_dir = output.parent / f"{output.stem}.assets"
    kwargs.pop("assets_dir_name", None)
    document, assets = build_atif_document(source, assets_dir_name=assets_dir.name, **kwargs)
    rendered = render_atif_text(document)
    files = assets.files()

    token = uuid.uuid4().hex
    tmp_doc = output.parent / f".{output.name}.tmp-atif-{token}"
    tmp_assets = output.parent / f".{assets_dir.name}.tmp-atif-{token}"
    old_aside = output.parent / f".{assets_dir.name}.old-{token}"
    swapped = False
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        if files:
            tmp_assets.mkdir()
            import hashlib

            for filename, data in files:
                path = tmp_assets / filename
                path.write_bytes(data)
                digest = filename.split(".", 1)[0]
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise AtifExportError(f"Asset failed hash verification after write: {filename}")
        tmp_doc.write_text(rendered, encoding="utf-8")

        if files and assets_dir.exists():
            assets_dir.rename(old_aside)
        if files:
            tmp_assets.rename(assets_dir)
        tmp_doc.replace(output)
        swapped = True
        if old_aside.exists():
            shutil.rmtree(old_aside, ignore_errors=True)
        return output
    finally:
        if not swapped:
            for leftover in (tmp_doc, tmp_assets):
                if leftover.is_dir():
                    shutil.rmtree(leftover, ignore_errors=True)
                elif leftover.exists():
                    leftover.unlink(missing_ok=True)
            if old_aside.exists() and not assets_dir.exists():
                old_aside.rename(assets_dir)


def iter_public_events_for_source(
    source: TrajectorySource, *, secret_values: Sequence[str] = ()
) -> Iterable[dict[str, Any]]:
    """Public events for an arbitrary source (used by tests and tooling)."""
    for event in source.read_events(repair_tail=True):
        public = to_public_event(event, secret_values=secret_values)
        if public is not None:
            yield public
