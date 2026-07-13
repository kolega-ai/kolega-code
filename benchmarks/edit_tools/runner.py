"""Controlled and CoderAgent benchmark execution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, cast

from kolega_code.agent.coder import CoderAgent
from kolega_code.agent.errors import MaxAgentIterationsExceeded
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.cli.config import CliConfigError, CliConfigOverrides, build_agent_config
from kolega_code.cli.settings import SettingsStore
from kolega_code.config import AgentConfig, ModelProvider
from kolega_code.llm.client import LLMClient
from kolega_code.llm.exceptions import LLMError
from kolega_code.llm.models import (
    ContentBlock,
    Message,
    MessageHistory,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from kolega_code.llm.specs import default_thinking_effort, get_model_specs
from kolega_code.permissions import PermissionMode
from kolega_code.services.lsp.config import LspConfig
from kolega_code.services.snapshots import SnapshotService

from .artifacts import (
    append_jsonl,
    create_manifest,
    load_trial_records,
    make_run_id,
    trial_id,
    utc_now,
    write_json,
    write_materialized_cases,
)
from .models import MatrixSpec, ModelRunSpec, SuiteSpec, TaskSpec, ToolAttempt, TrialRecord, UsageTotals
from .protocols import (
    READ_TOOL_NAMES,
    RecordingConnectionManager,
    create_tool_collection,
    execute_call,
    get_protocol,
)
from .usage import add_usage
from .workspace import collateral_paths, materialize_task, oracle_to_dict, verify_task, workspace_diff


CONTROLLED_SYSTEM_PROMPT = Message(
    role="system",
    content=[
        TextBlock(
            text=(
                "You are completing a deterministic file-edit task in an isolated workspace. "
                "Inspect files with the supplied read tools, make the requested change with the supplied edit tools, "
                "and stop when the workspace is correct. Do not merely describe an edit. Do not make unrelated changes."
            )
        )
    ],
)


@dataclass(frozen=True)
class PlannedTrial:
    trial_id: str
    task: TaskSpec
    lane: str
    model: ModelRunSpec
    protocol: str
    protocol_version: str
    repetition: int
    seed: int
    model_parameters: dict[str, Any]

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "task_id": self.task.id,
            "lane": self.lane,
            "provider": self.model.provider,
            "model": self.model.model,
            "protocol": self.protocol,
            "protocol_version": self.protocol_version,
            "repetition": self.repetition,
            "seed": self.seed,
            "model_parameters": self.model_parameters,
        }


@dataclass
class ExecutionResult:
    transcript: list[dict[str, Any]]
    events: list[dict[str, Any]]
    attempts: list[ToolAttempt]
    usage: UsageTotals
    terminal_stop: bool
    iteration_exhausted: bool = False


def _model_parameters(model: ModelRunSpec) -> dict[str, Any]:
    specs = get_model_specs(model.provider, model.model)
    return {
        "temperature": model.temperature if model.temperature is not None else specs.get("default_temperature", 1.0),
        "thinking_effort": (
            model.thinking_effort
            if model.thinking_effort is not None
            else default_thinking_effort(model.provider, model.model)
        ),
        "max_output_tokens": min(model.max_output_tokens, int(specs["max_completion_tokens"])),
    }


def plan_trials(
    suite: SuiteSpec,
    tasks: Iterable[TaskSpec],
    matrix: MatrixSpec,
    *,
    task_ids: set[str] | None = None,
    tags: set[str] | None = None,
    providers: set[str] | None = None,
    protocols: set[str] | None = None,
    lanes: set[str] | None = None,
) -> list[PlannedTrial]:
    repetitions = matrix.repetitions or suite.default_repetitions
    selected_tasks = [
        task for task in tasks if (not task_ids or task.id in task_ids) and (not tags or tags.intersection(task.tags))
    ]
    planned: list[PlannedTrial] = []
    for model in matrix.models:
        if not model.enabled or (providers and model.provider not in providers):
            continue
        model_parameters = _model_parameters(model)
        for protocol_id in model.protocols:
            if protocols and protocol_id not in protocols:
                continue
            adapter = get_protocol(protocol_id)
            for lane in matrix.lanes:
                if lanes and lane not in lanes:
                    continue
                for task in selected_tasks:
                    for repetition in range(1, repetitions + 1):
                        seed = (task.seed or 0) + repetition
                        identifier = trial_id(
                            suite=suite,
                            task=task,
                            lane=lane,
                            provider=model.provider,
                            model=model.model,
                            protocol=protocol_id,
                            protocol_version=adapter.version,
                            repetition=repetition,
                            seed=seed,
                            model_parameters=model_parameters,
                        )
                        planned.append(
                            PlannedTrial(
                                trial_id=identifier,
                                task=task,
                                lane=lane,
                                model=model,
                                protocol=protocol_id,
                                protocol_version=adapter.version,
                                repetition=repetition,
                                seed=seed,
                                model_parameters=model_parameters,
                            )
                        )
    return planned


def _build_config(project_path: Path, spec: PlannedTrial) -> AgentConfig:
    settings_store = SettingsStore()
    settings = settings_store.load()
    override = CliConfigOverrides(
        provider=spec.model.provider,
        model=spec.model.model,
        fast_provider=spec.model.provider,
        fast_model=spec.model.model,
        thinking_provider=spec.model.provider,
        thinking_model=spec.model.model,
        thinking_effort=spec.model_parameters["thinking_effort"],
        environment="benchmark",
        edit_protocol=spec.protocol,
    )
    config = build_agent_config(
        project_path,
        override,
        env=os.environ,
        settings=settings,
        settings_store=settings_store,
    )
    primary = config.long_context_config.model_copy(
        update={"thinking_effort": spec.model_parameters["thinking_effort"]}
    )
    return config.model_copy(
        update={
            "long_context_config": primary,
            "fast_config": primary,
            "thinking_config": primary,
            "agent_models": {},
            "lsp": LspConfig(enabled=False),
        }
    )


def _client(config: AgentConfig, provider: str) -> LLMClient:
    model_config = config.long_context_config
    return LLMClient(
        provider=provider,
        api_key=config.get_api_key(ModelProvider(provider)) or "",
        max_retries=model_config.rate_limits.max_retries,
        requests_per_minute=model_config.rate_limits.requests_per_minute,
        tokens_per_minute=model_config.rate_limits.tokens_per_minute,
        token_manager=config.get_chatgpt_token_manager(),
    )


async def _controlled_execution(
    workspace: Path,
    artifact_dir: Path,
    config: AgentConfig,
    trial: PlannedTrial,
) -> ExecutionResult:
    adapter = get_protocol(trial.protocol)
    collection, connection, caller = create_tool_collection(workspace, config, adapter, artifact_dir)
    definitions = adapter.definitions(collection)
    definition_map: dict[str, ToolDefinition] = {item.name: item for item in definitions}
    history = MessageHistory([Message(role="user", content=[TextBlock(text=trial.task.prompt)])])
    client = _client(config, trial.model.provider)
    usage = UsageTotals()
    attempts: list[ToolAttempt] = []
    terminal_stop = False
    exhausted = False
    try:
        for iteration in range(1, 7):
            response = await client.generate(
                messages=history,
                system=CONTROLLED_SYSTEM_PROMPT,
                temperature=float(trial.model_parameters["temperature"]),
                max_completion_tokens=int(trial.model_parameters["max_output_tokens"]),
                tools=definitions,
                thinking=trial.model_parameters["thinking_effort"],
                model=trial.model.model,
            )
            add_usage(usage, response.usage_metadata, trial.model.provider)
            history.append(response)
            calls = list(response.tool_calls)
            if not calls and isinstance(response.content, list):
                calls = [block for block in response.content if isinstance(block, ToolCall)]
            if not calls:
                terminal_stop = True
                break
            results: list[ToolResult] = []
            for call in calls:
                result, attempt = await execute_call(collection, caller, call, definition_map, iteration)
                results.append(result)
                attempts.append(attempt)
            history.append(Message(role="user", content=cast(list[ContentBlock], results)))
        else:
            exhausted = True
    finally:
        await collection.cleanup()
    return ExecutionResult(
        transcript=[message.to_dict() for message in history],
        events=[event.model_dump(mode="json") for event in connection.events],
        attempts=attempts,
        usage=usage,
        terminal_stop=terminal_stop,
        iteration_exhausted=exhausted,
    )


def _attempts_from_history(history: Iterable[Message], edit_names: set[str]) -> list[ToolAttempt]:
    results: dict[str, ToolResult] = {}
    calls: list[ToolCall] = []
    for message in history:
        if not isinstance(message.content, list):
            continue
        for block in message.content:
            if isinstance(block, ToolCall):
                calls.append(block)
            elif isinstance(block, ToolResult):
                results[block.tool_use_id] = block
    attempts: list[ToolAttempt] = []
    iteration = 0
    for call in calls:
        if call.name not in edit_names:
            continue
        iteration += 1
        result = results.get(call.id)
        is_error = result is None or result.is_error
        attempts.append(
            ToolAttempt(
                iteration=iteration,
                name=call.name,
                input_kind=call.input_kind,
                raw_input=call.input,
                parse_ok=not is_error,
                apply_ok=not is_error,
                is_error=is_error,
                error=str(result.content) if result is not None and result.is_error else None,
            )
        )
    return attempts


async def _coder_agent_execution(
    workspace: Path,
    artifact_dir: Path,
    config: AgentConfig,
    trial: PlannedTrial,
) -> ExecutionResult:
    adapter = get_protocol(trial.protocol)
    if adapter.production_protocol is None:
        raise ValueError(f"research protocol {adapter.id} is not available in the CoderAgent lane")
    connection = RecordingConnectionManager()
    agent = CoderAgent(
        project_path=workspace,
        workspace_id="benchmark-workspace",
        thread_id=trial.trial_id,
        connection_manager=connection,
        config=config,
        agent_mode=AgentMode.CLI,
        permission_mode=PermissionMode.AUTO,
        max_iterations=12,
    )
    assert agent.tool_collection is not None
    agent.tool_collection.tool_config.allowed_tools = [*READ_TOOL_NAMES, *adapter.tool_names]
    snapshot = SnapshotService(
        workspace,
        "benchmark-workspace",
        trial.trial_id,
        trial.trial_id,
        agent.tool_collection.filesystem,
        root=artifact_dir / "private-state",
    )
    agent.tool_collection.snapshot_service = snapshot
    agent.tool_collection.edit_tool._snapshot_service = snapshot
    agent.model_completion_tokens = int(trial.model_parameters["max_output_tokens"])
    agent.model_default_temperature = float(trial.model_parameters["temperature"])
    exhausted = False
    terminal_stop = False
    try:
        try:
            async for _ in agent.process_message_stream(trial.task.prompt):
                pass
            terminal_stop = True
        except MaxAgentIterationsExceeded:
            exhausted = True
        history = list(agent.history)
        usage = UsageTotals()
        for message in history:
            if message.role == "assistant":
                add_usage(usage, message.usage_metadata, trial.model.provider)
        return ExecutionResult(
            transcript=[message.to_dict() for message in history],
            events=[event.model_dump(mode="json") for event in connection.events],
            attempts=_attempts_from_history(history, set(adapter.tool_names)),
            usage=usage,
            terminal_stop=terminal_stop,
            iteration_exhausted=exhausted,
        )
    finally:
        await agent.cleanup()


def _write_execution_artifacts(artifact_dir: Path, execution: ExecutionResult) -> None:
    write_json(artifact_dir / "transcript.json", execution.transcript)
    for message in execution.transcript:
        append_jsonl(artifact_dir / "transcript.jsonl", message)
    for event in execution.events:
        append_jsonl(artifact_dir / "events.jsonl", event)


def _base_record(
    trial: PlannedTrial,
    run_id: str,
    artifact_dir: Path,
    started_at: str,
    started: float,
    **updates: Any,
) -> TrialRecord:
    finished_at = utc_now()
    values: dict[str, Any] = {
        "trial_id": trial.trial_id,
        "run_id": run_id,
        "suite_id": updates.pop("suite_id"),
        "task_id": trial.task.id,
        "task_digest": trial.task.digest,
        "lane": trial.lane,
        "provider": trial.model.provider,
        "model": trial.model.model,
        "protocol": trial.protocol,
        "protocol_version": trial.protocol_version,
        "repetition": trial.repetition,
        "seed": trial.seed,
        "status": "harness_error",
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_ms": round((time.monotonic() - started) * 1000),
        "artifact_dir": artifact_dir.as_posix(),
        "metadata": {"model_parameters": trial.model_parameters},
    }
    values.update(updates)
    return TrialRecord.model_validate(values)


async def run_trial(
    *,
    run_id: str,
    suite: SuiteSpec,
    trial: PlannedTrial,
    run_dir: Path,
    timeout_seconds: float,
) -> TrialRecord:
    started_at = utc_now()
    started = time.monotonic()
    artifact_dir = run_dir / "artifacts" / trial.trial_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_json(artifact_dir / "trial.json", trial.manifest_entry())
    adapter = get_protocol(trial.protocol)

    if not adapter.supports(trial.task.required_capabilities):
        return _base_record(
            trial,
            run_id,
            artifact_dir,
            started_at,
            started,
            suite_id=suite.id,
            status="unsupported",
            failure_stage="capability",
            error=f"requires {sorted(trial.task.required_capabilities)}; protocol supports {sorted(adapter.capabilities)}",
        )

    with tempfile.TemporaryDirectory(prefix=f"kolega-edit-bench-{trial.trial_id}-") as temporary:
        workspace = Path(temporary) / "workspace"
        materialize_task(workspace, trial.task)
        try:
            config = _build_config(workspace, trial)
        except CliConfigError as exc:
            return _base_record(
                trial,
                run_id,
                artifact_dir,
                started_at,
                started,
                suite_id=suite.id,
                status="not_run",
                failure_stage="credentials",
                error=str(exc),
            )

        execution: ExecutionResult | None = None
        execution_error: Exception | None = None
        provider_error = False
        try:
            operation = (
                _controlled_execution(workspace, artifact_dir, config, trial)
                if trial.lane == "controlled"
                else _coder_agent_execution(workspace, artifact_dir, config, trial)
            )
            execution = await asyncio.wait_for(operation, timeout=timeout_seconds)
        except TimeoutError as exc:
            execution_error = exc
            provider_error = True
        except LLMError as exc:
            execution_error = exc
            provider_error = True
        except Exception as exc:  # noqa: BLE001 - persisted as a harness failure with artifacts
            execution_error = exc

        oracle = await verify_task(workspace, trial.task)
        (artifact_dir / "workspace.diff").write_text(workspace_diff(trial.task, workspace), encoding="utf-8")
        write_json(artifact_dir / "oracle.json", oracle_to_dict(oracle))
        if execution is not None:
            _write_execution_artifacts(artifact_dir, execution)

        if execution_error is not None:
            return _base_record(
                trial,
                run_id,
                artifact_dir,
                started_at,
                started,
                suite_id=suite.id,
                status="provider_error" if provider_error else "harness_error",
                task_success=oracle.success,
                oracle_success=oracle.success,
                failure_stage="timeout"
                if isinstance(execution_error, TimeoutError)
                else ("provider" if provider_error else "harness"),
                error=str(execution_error) or type(execution_error).__name__,
            )

        assert execution is not None
        edit_attempts = [attempt for attempt in execution.attempts if attempt.name in adapter.tool_names]
        first_attempt_success = bool(edit_attempts and edit_attempts[0].apply_ok)
        status = "passed" if oracle.success else "failed"
        failure_stage = None
        if not oracle.success:
            if not edit_attempts:
                failure_stage = "tool_selection"
            elif not any(attempt.parse_ok for attempt in edit_attempts):
                failure_stage = "parse"
            elif not any(attempt.apply_ok for attempt in edit_attempts):
                failure_stage = "apply"
            else:
                failure_stage = "oracle"
        collateral = collateral_paths(trial.task, workspace)
        return _base_record(
            trial,
            run_id,
            artifact_dir,
            started_at,
            started,
            suite_id=suite.id,
            status=status,
            task_success=oracle.success,
            first_attempt_success=first_attempt_success,
            oracle_success=oracle.success,
            terminal_stop=execution.terminal_stop,
            failure_stage=failure_stage,
            usage=execution.usage,
            tool_attempts=execution.attempts,
            collateral_paths=collateral,
            metadata={
                "model_parameters": trial.model_parameters,
                "iteration_exhausted": execution.iteration_exhausted,
                "transport_kind": (
                    "native_freeform"
                    if trial.protocol == "codex_apply_patch" and trial.model.provider in {"openai", "openai_chatgpt"}
                    else "json_envelope"
                    if trial.protocol == "codex_apply_patch"
                    else "json"
                ),
            },
        )


async def run_benchmark(
    *,
    repo_root: Path,
    suite: SuiteSpec,
    tasks: list[TaskSpec],
    matrix: MatrixSpec,
    output_root: Path,
    resume_dir: Path | None = None,
    rerun_infrastructure_failures: bool = False,
    max_trials: int | None = None,
    task_ids: set[str] | None = None,
    tags: set[str] | None = None,
    providers: set[str] | None = None,
    protocols: set[str] | None = None,
    lanes: set[str] | None = None,
) -> tuple[Path, list[TrialRecord]]:
    planned = plan_trials(
        suite,
        tasks,
        matrix,
        task_ids=task_ids,
        tags=tags,
        providers=providers,
        protocols=protocols,
        lanes=lanes,
    )
    if max_trials is not None:
        planned = planned[:max_trials]
    if not planned:
        raise ValueError("filters selected zero benchmark trials")

    if resume_dir is None:
        run_id = make_run_id(suite.id, matrix.id)
        run_dir = output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest = create_manifest(
            run_id=run_id,
            repo_root=repo_root,
            suite=suite,
            tasks=tasks,
            matrix=matrix,
            planned_trials=[item.manifest_entry() for item in planned],
        )
        write_json(run_dir / "manifest.json", manifest)
        write_materialized_cases(run_dir, tasks)
    else:
        run_dir = resume_dir
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        run_id = str(manifest["run_id"])
        expected_ids = {item["trial_id"] for item in manifest.get("planned_trials", [])}
        actual_ids = {item.trial_id for item in planned}
        if expected_ids != actual_ids:
            raise ValueError("resume suite/matrix/filters do not match the saved run manifest")

    journal = run_dir / "trials.jsonl"
    existing = {record.trial_id: record for record in load_trial_records(journal)}
    skip_statuses = {"passed", "failed", "unsupported"}
    if not rerun_infrastructure_failures:
        skip_statuses.update({"provider_error", "harness_error", "not_run", "cancelled"})

    pending: list[tuple[int, PlannedTrial]] = []
    for index, item in enumerate(planned, 1):
        previous = existing.get(item.trial_id)
        if previous is not None and previous.status in skip_statuses:
            continue
        pending.append((index, item))

    queue: asyncio.Queue[tuple[int, PlannedTrial]] = asyncio.Queue()
    for queued in pending:
        queue.put_nowait(queued)
    journal_lock = asyncio.Lock()
    provider_locks: dict[str, asyncio.Lock] = {
        provider: asyncio.Lock() for provider in {item.model.provider for _, item in pending}
    }

    async def worker() -> None:
        while True:
            try:
                index, item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                print(
                    f"[{index}/{len(planned)}] {item.model.provider}/{item.model.model} "
                    f"{item.protocol} {item.lane} {item.task.id} r{item.repetition}",
                    flush=True,
                )
                # Provider SDK clients carry their own request limiters. Keeping one
                # trial per provider in flight avoids bypassing those limits while still
                # allowing independent providers to run concurrently.
                async with provider_locks[item.model.provider]:
                    record = await run_trial(
                        run_id=run_id,
                        suite=suite,
                        trial=item,
                        run_dir=run_dir,
                        timeout_seconds=matrix.trial_timeout_seconds,
                    )
                async with journal_lock:
                    append_jsonl(journal, record.model_dump(mode="json"))
                    existing[item.trial_id] = record
                print(
                    f"  -> {record.status}{f' ({record.failure_stage})' if record.failure_stage else ''}",
                    flush=True,
                )
            finally:
                queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(min(matrix.concurrency, max(1, len(pending))))]
    if workers:
        await asyncio.gather(*workers)

    records = load_trial_records(journal)
    return run_dir, records
