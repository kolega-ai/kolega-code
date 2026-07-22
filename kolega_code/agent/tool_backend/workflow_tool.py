"""WorkflowTool — the ``run_workflow`` tool backing gigacode.

It turns the model-authored Python orchestration script into a real multi-agent
run: it persists artifacts under the CLI state dir, builds a
:class:`WorkflowRuntime` whose ``agent()`` primitive dispatches genuine sub-agents
(via :meth:`AgentTool.dispatch_workflow_agent`), streams phase/log progress to the
UI, and supports journal-based resume.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from kolega_code.events import AgentEvent
from kolega_code.llm.specs import supports_vision

from ..model_routing import model_routing_fingerprint, resolve_subagent_model
from ..orchestration import (
    AgentRunResult,
    AgentRunSpec,
    Budget,
    DEFAULT_AGENT_CAP,
    DispatchFn,
    EmitFn,
    RunJournal,
    WorkflowRuntime,
    WorkflowScriptError,
    extract_meta,
    saved_workflows_dir,
)
from ..orchestration.accounting import WorkflowRunAccounting, get_current_agent_reservation
from .agent_tool import AgentTool
from .base_tool import BaseTool

# agent_type name -> agent class import path. None / unknown falls back to GeneralAgent.
_AGENT_TYPE_IMPORTS = {
    "general": "kolega_code.agent.generalagent.GeneralAgent",
    "general-purpose": "kolega_code.agent.generalagent.GeneralAgent",
    "investigation": "kolega_code.agent.investigationagent.InvestigationAgent",
    "explore": "kolega_code.agent.investigationagent.InvestigationAgent",
    "browser": "kolega_code.agent.browseragent.BrowserAgent",
    "coder": "kolega_code.agent.coder.CoderAgent",
    "coding": "kolega_code.agent.coder.CoderAgent",
}
_DEFAULT_AGENT_IMPORT = "kolega_code.agent.generalagent.GeneralAgent"


# Explicit input schema: `args` is free-form JSON (no `type`), which the signature
# introspector cannot express, so it is supplied verbatim via extension_schemas.
RUN_WORKFLOW_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "script": {
            "type": "string",
            "description": "The Python orchestration script source (must define a top-level `meta` literal).",
        },
        "args": {
            "description": "Free-form JSON value exposed to the script as the global `args`.",
        },
        "token_budget": {
            "type": "integer",
            "description": "Optional output-token ceiling for the whole run (0 = unbounded).",
        },
        "script_path": {
            "type": "string",
            "description": "Path to a script file on disk; takes precedence over `script`.",
        },
        "resume_from_run_id": {
            "type": "string",
            "description": "Resume from a prior run id, replaying cached agent() results for the unchanged prefix.",
        },
    },
    "required": [],
}


def _import_agent_class(import_path: str) -> type[Any]:
    module_path, class_name = import_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


class WorkflowTool(BaseTool):
    """Executes a gigacode workflow script and returns a compact run summary."""

    def __init__(self, *args, langfuse_client=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.langfuse_client = langfuse_client
        # Dedicated dispatcher bound to the same caller, so workflow sub-agents
        # inherit config/permissions/hooks exactly like normal dispatches.
        self._agent_tool = AgentTool(
            self.project_path,
            self.workspace_id,
            self.thread_id,
            self.connection_manager,
            self.config,
            self.caller,
            self.filesystem,
            terminal_manager=self.terminal_manager,
            browser_manager=self.browser_manager,
            langfuse_client=langfuse_client,
            memory_manager=getattr(self.caller, "memory_manager", None),
        )

    def _workflow_is_read_only(self) -> bool:
        """Whether every direct workflow worker must use read-only tooling."""
        return bool(getattr(getattr(self.caller, "tool_collection", None), "read_only", False))

    @staticmethod
    def _agent_import_path(agent_type: Optional[str], *, read_only: bool) -> str:
        if read_only:
            return _AGENT_TYPE_IMPORTS["investigation"]
        return _AGENT_TYPE_IMPORTS.get((agent_type or "").lower(), _DEFAULT_AGENT_IMPORT)

    # ------------------------------------------------------------------ tool
    async def run_workflow(
        self,
        script: str = "",
        args: Any = None,
        token_budget: int = 0,
        script_path: str = "",
        resume_from_run_id: str = "",
    ) -> str:
        """Run a gigacode orchestration script (see the gigacode authoring guide).

        Args:
            script: The Python orchestration script source. Omit when re-running an
                edited script via script_path, or resuming via resume_from_run_id.
            args: Free-form JSON value exposed to the script as the global `args`.
            token_budget: Optional output-token ceiling for the whole run (0 = unbounded).
            script_path: Path to a script file on disk; takes precedence over `script`.
            resume_from_run_id: Resume from a prior run, replaying cached agent() results
                for the unchanged prefix and running new/changed calls live.

        Returns:
            A compact artifact manifest including runId, scriptPath, token count,
            resultPath, and transcriptPath. Results are persisted in the main
            result/transcript files instead of being returned inline. Per-agent
            artifacts are saved for debugging but are intentionally not advertised
            to the model-facing tool result.
        """
        from kolega_code.cli.session_store import default_state_dir

        state_dir = default_state_dir()
        source = self._resolve_source(script, script_path, resume_from_run_id, state_dir)

        meta = extract_meta(source)

        run_id = uuid.uuid4().hex
        journal = RunJournal.for_run(state_dir, run_id)
        journal.write_script(source)

        resume_cache = None
        if resume_from_run_id:
            resume_cache = RunJournal.for_run(state_dir, resume_from_run_id).load_cache()

        budget = Budget(total=token_budget or None)
        accounting = WorkflowRunAccounting(budget, DEFAULT_AGENT_CAP)
        started = time.time()
        journal.write_meta(
            {
                "run_id": run_id,
                "name": meta.get("name"),
                "description": meta.get("description"),
                "max_agent_depth": meta["max_agent_depth"],
                "status": "running",
                "args": args,
                "resumed_from": resume_from_run_id or None,
            }
        )

        # Emitted (and awaited) before execute() so the workflow_start event is on
        # the broadcast queue ahead of any phase/log the runtime fires — those go
        # through fire-and-forget _emit_soon and can't run until we next await.
        emit = self._make_emit(run_id, journal)
        await emit(
            "workflow_start",
            {
                "name": meta.get("name"),
                "description": meta.get("description"),
                "phases": meta.get("phases") or [],
                "max_agent_depth": meta["max_agent_depth"],
            },
        )

        read_only = self._workflow_is_read_only()
        runtime = WorkflowRuntime(
            dispatch=self._make_dispatch(run_id, journal, accounting),
            emit=emit,
            journal=journal,
            budget=budget,
            accounting=accounting,
            max_agent_depth=meta["max_agent_depth"],
            resume_cache=resume_cache,
            workflow_resolver=self._make_resolver(state_dir),
            routing_fingerprint=model_routing_fingerprint(self.config),
            actual_agent_type_resolver=lambda agent_type: self._agent_import_path(
                agent_type, read_only=read_only
            ).rsplit(".", 1)[1],
            read_only_mode=read_only,
        )

        status = "completed"
        error: Optional[str] = None
        result: Any = None
        try:
            result = await runtime.execute(source, args)
        except WorkflowScriptError as exc:
            status = "failed"
            error = f"workflow script error: {exc}"
        except Exception as exc:  # noqa: BLE001 - report any orchestration failure to the model
            status = "failed"
            error = f"workflow failed: {exc}"

        duration_seconds = round(time.time() - started, 2)
        rendered_result = self._result_json_text(result)
        journal.write_result_artifacts(result, self._render_result_markdown(meta, run_id, status, error, result))

        await emit("workflow_end", {"status": status, "error": error})
        journal.write_transcript_markdown(
            self._render_workflow_transcript(meta, run_id, journal, budget, status, error, duration_seconds)
        )

        journal.update_meta(
            status=status,
            error=error,
            duration_seconds=duration_seconds,
            total_tokens=budget.spent(),
            result_size_chars=len(rendered_result),
            artifacts={
                "scriptPath": str(journal.script_path),
                "resultPath": str(journal.result_md_path),
                "transcriptPath": str(journal.transcript_md_path),
            },
        )

        return self._summarize(meta, run_id, journal, budget, status, error, result)

    # -------------------------------------------------------------- internals
    def _resolve_source(self, script: str, script_path: str, resume_from_run_id: str, state_dir: Path) -> str:
        if script_path:
            try:
                return open(script_path, encoding="utf-8").read()
            except OSError as exc:
                raise WorkflowScriptError(f"could not read script_path {script_path!r}: {exc}") from exc
        if script:
            return script
        if resume_from_run_id:
            try:
                return RunJournal.for_run(state_dir, resume_from_run_id).read_script()
            except OSError as exc:
                raise WorkflowScriptError(
                    f"could not read script for resume run {resume_from_run_id!r}: {exc}"
                ) from exc
        raise WorkflowScriptError("run_workflow requires one of: script, script_path, or resume_from_run_id")

    def _make_dispatch(
        self,
        run_id: str,
        journal: RunJournal,
        accounting: WorkflowRunAccounting,
    ) -> DispatchFn:
        # When the orchestrating agent is itself read-only (e.g. plan mode's
        # PlanningAgent), every workflow sub-agent is forced to a read-only
        # investigation agent so the read-only contract is preserved — the
        # workflow can research in parallel (including running investigative
        # commands) but cannot edit files, since read_only agents have no
        # file-edit tools.
        read_only = self._workflow_is_read_only()

        async def dispatch(spec: AgentRunSpec) -> AgentRunResult:
            reservation = get_current_agent_reservation()
            if reservation is None:
                raise RuntimeError("workflow dispatch requires an agent reservation")
            import_path = self._agent_import_path(spec.agent_type, read_only=read_only)
            agent_class = _import_agent_class(import_path)
            actual_agent_type = agent_class.__name__
            agent_name = getattr(agent_class, "agent_name", actual_agent_type)
            requested_routing = spec.model_override.as_dict() if spec.model_override is not None else None

            try:
                routing = resolve_subagent_model(
                    self.config,
                    agent_name,
                    spec.model_override.as_dict() if spec.model_override is not None else None,
                    effort_key="effort",
                )
                if (
                    spec.model_override is not None
                    and agent_name == "browser-agent"
                    and not supports_vision(routing.model_config.provider, routing.model_config.model)
                ):
                    raise ValueError(
                        "BrowserAgent model_override requires a vision-capable model; "
                        f"{routing.model_config.provider.value}/{routing.model_config.model} "
                        "does not support image input."
                    )
            except (TypeError, ValueError) as exc:
                return AgentRunResult(
                    status="failed",
                    error=f"invalid model_override: {exc}",
                    requested_routing=requested_routing,
                    actual_agent_type=actual_agent_type,
                )

            effective_routing = {
                "provider": routing.model_config.provider.value,
                "model": routing.model_config.model,
                "effort": routing.model_config.thinking_effort,
            }
            config = routing.config if spec.model_override is not None else None
            sub_info_extra = {
                "workflow_run_id": run_id,
                "phase": spec.phase,
                "label": spec.label,
                "call_index": spec.call_index,
                "depth": 1,
                "max_agent_depth": spec.max_agent_depth,
                "requested_routing": requested_routing,
                "effective_routing": effective_routing,
                "actual_agent_type": actual_agent_type,
            }
            label_for_path = spec.label or spec.agent_type or agent_class.__name__
            artifact_paths = journal.agent_artifact_paths(spec.call_index, label_for_path)
            artifact_metadata = {
                "call_index": spec.call_index,
                "label": spec.label,
                "phase": spec.phase,
                "agent_type": actual_agent_type,
                "requested_agent_type": spec.agent_type,
                "actual_agent_type": actual_agent_type,
                "agent_name": agent_name,
                "max_agent_depth": spec.max_agent_depth,
                "requested_routing": requested_routing,
                "effective_routing": effective_routing,
            }
            try:
                recap, tokens, structured = await self._agent_tool.dispatch_workflow_agent(
                    agent_class,
                    spec.prompt,
                    workflow_accounting=accounting,
                    reservation=reservation,
                    config=config,
                    schema=spec.schema,
                    sub_agent_info_extra=sub_info_extra,
                    artifact_paths=artifact_paths,
                    artifact_metadata=artifact_metadata,
                )
            except Exception as exc:  # noqa: BLE001 - a dead agent becomes a None result, not a crash
                return AgentRunResult(
                    tokens=reservation.reported_tokens,
                    status="failed",
                    error=str(exc),
                    transcript_path=str(artifact_paths["jsonl"]),
                    transcript_markdown_path=str(artifact_paths["markdown"]),
                    requested_routing=requested_routing,
                    effective_routing=effective_routing,
                    actual_agent_type=actual_agent_type,
                )
            return AgentRunResult(
                text=recap,
                structured=structured,
                tokens=tokens or 0,
                status="completed",
                transcript_path=str(artifact_paths["jsonl"]),
                transcript_markdown_path=str(artifact_paths["markdown"]),
                requested_routing=requested_routing,
                effective_routing=effective_routing,
                actual_agent_type=actual_agent_type,
            )

        return dispatch

    def _make_emit(self, run_id: str, journal: RunJournal) -> EmitFn:
        sender = getattr(self.caller, "agent_name", None) or "gigacode"

        async def emit(kind: str, content: dict) -> None:
            # workflow_run_id keys every event to its card in the TUI so phase/log
            # updates land on the right workflow when several run in a turn.
            payload: dict = {"workflow_run_id": run_id, "text": ""}
            if kind == "workflow_phase":
                payload.update(message_type="workflow_phase", text=content.get("title", ""))
            elif kind == "workflow_log":
                payload.update(message_type="workflow_log", text=content.get("message", ""))
            elif kind == "workflow_agent_cached":
                payload.update(message_type="workflow_log", text=f"cached: {content.get('label', '')}")
            elif kind == "workflow_start":
                payload.update(
                    message_type="workflow_start",
                    name=content.get("name"),
                    description=content.get("description"),
                    phases=content.get("phases") or [],
                    max_agent_depth=content.get("max_agent_depth"),
                )
            elif kind == "workflow_end":
                payload.update(
                    message_type="workflow_end",
                    status=content.get("status"),
                    error=content.get("error"),
                )
            else:
                payload.update(message_type="workflow_log", text=str(content))
            journal.append_transcript_event({"type": kind, "content": content})
            event = AgentEvent(
                event_type="chat_message",
                sender=sender,
                content=payload,
            )
            await self.connection_manager.broadcast_event(event, self.workspace_id, self.thread_id)

        return emit

    def _make_resolver(self, state_dir):
        def resolve(name_or_ref: Any) -> str:
            if isinstance(name_or_ref, dict) and name_or_ref.get("script_path"):
                path = name_or_ref["script_path"]
            else:
                path = saved_workflows_dir(state_dir) / f"{name_or_ref}.py"
            try:
                return open(path, encoding="utf-8").read()
            except OSError as exc:
                raise WorkflowScriptError(f"could not resolve nested workflow {name_or_ref!r}: {exc}") from exc

        return resolve

    def _result_json_text(self, result: Any) -> str:
        try:
            return json.dumps(result, indent=2, default=str)
        except (TypeError, ValueError):
            return json.dumps(str(result), indent=2)

    def _render_value_markdown(self, value: Any) -> str:
        if value is None:
            return "None"
        if isinstance(value, str):
            return value
        return "```json\n" + self._result_json_text(value) + "\n```"

    def _render_result_markdown(self, meta, run_id: str, status: str, error: Optional[str], result: Any) -> str:
        lines = [
            f"# Workflow result: {meta.get('name') or 'workflow'}",
            "",
            f"- Run id: `{run_id}`",
            f"- Status: {status}",
        ]
        if error:
            lines.append(f"- Error: {error}")
        lines.extend(["", "## Full return value", "", self._render_value_markdown(result)])
        return "\n".join(lines).rstrip() + "\n"

    def _read_jsonl(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        entries: list[dict] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            try:
                entries.append(json.loads(raw))
            except json.JSONDecodeError:
                entries.append({"type": "invalid_jsonl", "raw": raw})
        return entries

    def _render_workflow_transcript(
        self,
        meta,
        run_id: str,
        journal: RunJournal,
        budget: Budget,
        status: str,
        error: Optional[str],
        duration_seconds: float,
    ) -> str:
        journal_entries = self._read_jsonl(journal.journal_path)
        raw_events = self._read_jsonl(journal.transcript_jsonl_path)
        lines = [
            f"# Workflow transcript: {meta.get('name') or 'workflow'}",
            "",
            f"- Run id: `{run_id}`",
            f"- Status: {status}",
            f"- Duration: {duration_seconds}s",
            f"- Tokens: {budget.spent()}",
            f"- Max agent depth: {meta['max_agent_depth']}",
            f"- Script: `{journal.script_path}`",
            f"- Result: `{journal.result_md_path}`",
            "",
            (
                "> For normal workflow output, read only this main transcript and `resultPath`. "
                "Avoid reading individual sub-agent transcripts unless you are explicitly debugging "
                "workflow execution."
            ),
        ]
        if error:
            lines.extend(["", f"**Error:** {error}"])

        lines.extend(["", "## Agent call index", ""])
        if journal_entries:
            lines.append("| # | Label | Phase | Actual agent type | Status | Tokens |")
            lines.append("| --- | --- | --- | --- | --- | ---: |")
            for entry in journal_entries:
                index = entry.get("index", "")
                label = str(entry.get("label") or "")
                phase = str(entry.get("phase") or "")
                agent_type = str(entry.get("actual_agent_type") or entry.get("agent_type") or "")
                status_text = str(entry.get("status") or "")
                tokens = entry.get("tokens", "")
                lines.append(f"| {index} | {label} | {phase} | {agent_type} | {status_text} | {tokens} |")
        else:
            lines.append("No agent calls were recorded.")

        lines.extend(["", "## Agent results", ""])
        if journal_entries:
            for entry in journal_entries:
                index = entry.get("index", "")
                label = entry.get("label") or f"agent {index}"
                lines.extend([f"### Call {index}: {label}", ""])
                for key in (
                    "phase",
                    "requested_agent_type",
                    "agent_type",
                    "actual_agent_type",
                    "requested_routing",
                    "effective_routing",
                    "status",
                    "tokens",
                    "error",
                ):
                    if key == "requested_routing" and key in entry:
                        requested = entry[key]
                        rendered = "inherited (null)" if requested is None else requested
                        lines.append(f"- {key}: `{rendered}`")
                    elif entry.get(key) is not None:
                        lines.append(f"- {key}: `{entry.get(key)}`")
                value_text = self._render_value_markdown(entry.get("value"))
                if len(value_text) <= 2000:
                    lines.extend(["", "Returned value:", "", value_text, ""])
                else:
                    lines.extend(
                        [
                            "",
                            (
                                f"Returned value is {len(value_text):,} chars; "
                                f"read `{journal.result_md_path}` for the full workflow result."
                            ),
                            "",
                        ]
                    )
        else:
            lines.append("No returned agent values were recorded.")

        cached_events = [event for event in raw_events if event.get("type") == "agent_cached"]
        if cached_events:
            lines.extend(["", "## Cached resume calls", ""])
            for event in cached_events:
                lines.append(f"- Call {event.get('index')}: {event.get('label')} (served from resume cache)")

        return "\n".join(str(line) for line in lines).rstrip() + "\n"

    def _summarize(self, meta, run_id, journal: RunJournal, budget: Budget, status, error, result) -> str:
        lines = [
            f"Workflow {meta.get('name')!r} {status}.",
            f"runId: {run_id}",
            f"scriptPath: {journal.script_path}",
            f"tokens: {budget.spent()}",
            f"resultPath: {journal.result_md_path}",
            f"transcriptPath: {journal.transcript_md_path}",
        ]
        if error:
            lines.append(f"error: {error}")
        lines.append(
            "result: written to resultPath. Read resultPath for the workflow result, "
            "or transcriptPath for execution details."
        )
        lines.append(
            "IMPORTANT: The workflow already ran. Do not re-run it to recover output; "
            "read resultPath or transcriptPath first."
        )
        return "\n".join(lines)
