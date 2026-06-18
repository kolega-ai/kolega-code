"""WorkflowTool — the ``run_workflow`` tool backing gigacode.

It turns the model-authored Python orchestration script into a real multi-agent
run: it persists artifacts under the CLI state dir, builds a
:class:`WorkflowRuntime` whose ``agent()`` primitive dispatches genuine sub-agents
(via :meth:`AgentTool.dispatch_workflow_agent`), streams phase/log progress to the
UI, and supports journal-based resume.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Optional

from kolega_code.events import AgentEvent

from ..orchestration import (
    AgentRunResult,
    AgentRunSpec,
    Budget,
    RunJournal,
    WorkflowRuntime,
    WorkflowScriptError,
    extract_meta,
    saved_workflows_dir,
)
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


def _import_agent_class(import_path: str):
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
        )

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
            A compact summary including the runId, the persisted scriptPath, agent and
            token counts, and the workflow's return value.
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
        started = time.time()
        journal.write_meta(
            {
                "run_id": run_id,
                "name": meta.get("name"),
                "description": meta.get("description"),
                "status": "running",
                "args": args,
                "resumed_from": resume_from_run_id or None,
            }
        )

        runtime = WorkflowRuntime(
            dispatch=self._make_dispatch(run_id),
            emit=self._make_emit(),
            journal=journal,
            budget=budget,
            resume_cache=resume_cache,
            workflow_resolver=self._make_resolver(state_dir),
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

        journal.update_meta(
            status=status,
            error=error,
            duration_seconds=round(time.time() - started, 2),
            total_tokens=budget.spent(),
        )

        return self._summarize(meta, run_id, journal, budget, status, error, result)

    # -------------------------------------------------------------- internals
    def _resolve_source(self, script: str, script_path: str, resume_from_run_id: str, state_dir) -> str:
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

    def _make_dispatch(self, run_id: str):
        # When the orchestrating agent is itself read-only (e.g. plan mode's
        # PlanningAgent), every workflow sub-agent is forced to a read-only
        # investigation agent so the read-only contract is preserved — the
        # workflow can research in parallel but never write.
        read_only = bool(getattr(getattr(self.caller, "tool_collection", None), "read_only", False))

        async def dispatch(spec: AgentRunSpec) -> AgentRunResult:
            if read_only:
                import_path = _AGENT_TYPE_IMPORTS["investigation"]
            else:
                import_path = _AGENT_TYPE_IMPORTS.get((spec.agent_type or "").lower(), _DEFAULT_AGENT_IMPORT)
            agent_class = _import_agent_class(import_path)
            config = self._config_override(spec.model, spec.effort)
            sub_info_extra = {"workflow_run_id": run_id, "phase": spec.phase, "label": spec.label}
            try:
                recap, tokens, structured = await self._agent_tool.dispatch_workflow_agent(
                    agent_class,
                    spec.prompt,
                    config=config,
                    schema=spec.schema,
                    sub_agent_info_extra=sub_info_extra,
                )
            except Exception as exc:  # noqa: BLE001 - a dead agent becomes a None result, not a crash
                return AgentRunResult(status="failed", error=str(exc))
            return AgentRunResult(text=recap, structured=structured, tokens=tokens or 0, status="completed")

        return dispatch

    def _make_emit(self):
        sender = getattr(self.caller, "agent_name", None) or "gigacode"

        async def emit(kind: str, content: dict) -> None:
            if kind == "workflow_phase":
                message_type, text = "workflow_phase", content.get("title", "")
            elif kind == "workflow_log":
                message_type, text = "workflow_log", content.get("message", "")
            elif kind == "workflow_agent_cached":
                message_type, text = "workflow_log", f"cached: {content.get('label', '')}"
            else:
                message_type, text = "workflow_log", str(content)
            event = AgentEvent(
                event_type="chat_message",
                sender=sender,
                content={"message_type": message_type, "text": text},
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

    def _config_override(self, model: Optional[str], effort: Optional[str]):
        """Clone the agent config with per-call model/effort overrides, or None."""
        if not model and not effort:
            return None
        lc_update: dict = {}
        th_update: dict = {}
        if model:
            lc_update["model"] = model
            th_update["model"] = model
        if effort:
            lc_update["thinking_effort"] = effort
            th_update["thinking_effort"] = effort
        new_long = self.config.long_context_config.model_copy(update=lc_update)
        new_thinking = self.config.thinking_config.model_copy(update=th_update)
        return self.config.model_copy(
            update={"long_context_config": new_long, "thinking_config": new_thinking}
        )

    def _summarize(self, meta, run_id, journal: RunJournal, budget: Budget, status, error, result) -> str:
        import json

        lines = [
            f"Workflow {meta.get('name')!r} {status}.",
            f"runId: {run_id}",
            f"scriptPath: {journal.script_path}",
            f"tokens: {budget.spent()}",
        ]
        if error:
            lines.append(f"error: {error}")
        if result is not None:
            try:
                rendered = json.dumps(result, default=str)
            except (TypeError, ValueError):
                rendered = str(result)
            if len(rendered) > 4000:
                rendered = rendered[:4000] + "… (truncated)"
            lines.append(f"result: {rendered}")
        return "\n".join(lines)
