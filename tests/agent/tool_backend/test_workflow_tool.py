"""Integration checks for WorkflowTool.run_workflow: artifacts, dispatch, resume.

The sub-agent dispatch is stubbed, so these run without the LLM stack but exercise
the real run_workflow code path (state-dir resolution, journal, emit, summary).
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from kolega_code.agent.tool_backend.workflow_tool import WorkflowTool
from kolega_code.agent.orchestration.accounting import AgentReservation, WorkflowRunAccounting
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider
from kolega_code.events import AgentConnectionManager


@pytest.fixture
def connection_manager():
    manager = Mock(spec=AgentConnectionManager)
    manager.broadcast_event = AsyncMock()
    return manager


@pytest.fixture
def caller():
    c = Mock()
    c.agent_name = "coder-agent"
    c.sub_agent = False
    c.current_tool_execution_id = "exec-1"
    c.current_tool_call_id = "exec-1"
    c.sub_agent_recorder = None
    c.prompt_extensions = None
    c.tool_extensions = None
    c.tool_collection = Mock(read_only=False)  # build-mode (writable) caller by default
    return c


@pytest.fixture
def workflow_tool(tmp_path, connection_manager, caller, monkeypatch):
    monkeypatch.setenv("KOLEGA_CODE_STATE_DIR", str(tmp_path))
    config = AgentConfig(
        anthropic_api_key="anthropic-key",
        deepseek_api_key="deepseek-key",
    )
    tool = WorkflowTool(
        str(tmp_path / "project"),
        "ws",
        "thread",
        connection_manager,
        config,
        caller,
        None,
    )
    (tmp_path / "project").mkdir(parents=True, exist_ok=True)
    return tool, tmp_path


def _stub_dispatch() -> tuple[Any, list[tuple[Any, ...]]]:
    """A dispatch_workflow_agent stub returning (recap, tokens, structured)."""
    calls = []

    async def dispatch_workflow_agent(
        agent_class: type[Any],
        task: str,
        *,
        workflow_accounting: WorkflowRunAccounting,
        reservation: AgentReservation,
        config: Any = None,
        schema: Any = None,
        sub_agent_info_extra: Any = None,
        artifact_paths: Any = None,
        artifact_metadata: Any = None,
    ) -> tuple[str, int, Any]:
        calls.append((task, schema, sub_agent_info_extra, artifact_paths, artifact_metadata))
        if artifact_paths:
            artifact_paths["jsonl"].write_text(json.dumps({"role": "assistant", "content": task}) + "\n")
            artifact_paths["markdown"].write_text(f"# Agent artifact\n\n{task}\n")
        if schema:
            return (f"recap:{task}", 7, {"task": task})
        return (f"recap:{task}", 3, None)

    return dispatch_workflow_agent, calls


SCRIPT = """\
meta = {"name": "demo", "description": "demo workflow", "phases": [{"title": "Find"}]}
phase("Find")
log("starting")
res = await parallel([(lambda i=i: agent(f"task {i}")) for i in range(3)])
one = await agent("structured", schema={"type": "object"})
return {"res": res, "one": one, "spent": budget.spent()}
"""


@pytest.mark.asyncio
async def test_run_workflow_writes_artifacts_and_summary(workflow_tool):
    tool, state_dir = workflow_tool
    stub, calls = _stub_dispatch()
    tool._agent_tool.dispatch_workflow_agent = stub

    summary = await tool.run_workflow(script=SCRIPT)

    # 4 dispatches: 3 parallel + 1 schema.
    assert len(calls) == 4
    assert "Workflow 'demo' completed." in summary
    assert "runId:" in summary and "scriptPath:" in summary
    assert "resultPath:" in summary
    assert "transcriptPath:" in summary

    # Artifacts under <state_dir>/workflows/<run_id>/
    run_id = next(line.split("runId:")[1].strip() for line in summary.splitlines() if "runId:" in line)
    run_dir = Path(state_dir) / "workflows" / run_id
    assert (run_dir / "script.py").read_text() == SCRIPT
    assert (run_dir / "result.md").is_file()
    assert (run_dir / "result.json").is_file()
    assert (run_dir / "transcript.md").is_file()
    assert (run_dir / "transcript.jsonl").is_file()
    meta = json.loads((run_dir / "run.json").read_text())
    assert meta["status"] == "completed"
    assert meta["name"] == "demo"
    assert meta["max_agent_depth"] == 1
    assert meta["artifacts"]["resultPath"] == str(run_dir / "result.md")
    # token total = 3*3 (parallel) + 7 (schema) = 16
    assert meta["total_tokens"] == 16
    journal_lines = (run_dir / "journal.jsonl").read_text().splitlines()
    assert len(journal_lines) == 4

    # phase + log + start/end were broadcast as chat_message events.
    event_types = [
        c.args[0].content.get("message_type") for c in tool.connection_manager.broadcast_event.call_args_list
    ]
    assert "workflow_phase" in event_types
    assert "workflow_log" in event_types
    assert "workflow_start" in event_types
    assert "workflow_end" in event_types

    # Every workflow event is keyed to its run so the TUI updates the right card.
    workflow_events = [
        c.args[0].content
        for c in tool.connection_manager.broadcast_event.call_args_list
        if str(c.args[0].content.get("message_type", "")).startswith("workflow_")
    ]
    assert all(e.get("workflow_run_id") == run_id for e in workflow_events)

    # workflow_start carries the plan; workflow_end carries the final status.
    start = next(e for e in workflow_events if e["message_type"] == "workflow_start")
    assert start["name"] == "demo"
    assert start["description"] == "demo workflow"
    assert start["phases"] == [{"title": "Find"}]
    assert start["max_agent_depth"] == 1
    end = next(e for e in workflow_events if e["message_type"] == "workflow_end")
    assert end["status"] == "completed"


@pytest.mark.asyncio
async def test_long_result_is_persisted_not_truncated_inline(workflow_tool):
    tool, state_dir = workflow_tool
    script = 'meta = {"name": "big", "description": "big result"}\nreturn {"blob": "x" * 5001}\n'

    summary = await tool.run_workflow(script=script)

    assert "resultPath:" in summary
    assert "transcriptPath:" in summary
    assert "result: written to resultPath" in summary
    assert "IMPORTANT: The workflow already ran" in summary
    assert "resultPreview:" not in summary
    assert "… (truncated)" not in summary
    assert len(summary) < 100_000

    run_id = next(line.split("runId:")[1].strip() for line in summary.splitlines() if "runId:" in line)
    run_dir = Path(state_dir) / "workflows" / run_id
    result_json = json.loads((run_dir / "result.json").read_text())
    assert result_json["blob"] == "x" * 5001
    assert "x" * 5001 in (run_dir / "result.md").read_text()


@pytest.mark.asyncio
async def test_readable_transcript_indexes_agent_calls(workflow_tool):
    tool, state_dir = workflow_tool
    stub, calls = _stub_dispatch()
    tool._agent_tool.dispatch_workflow_agent = stub
    script = (
        'meta = {"name": "transcript", "description": "d", "phases": [{"title": "Find"}]}\n'
        'phase("Find")\n'
        'await agent("alpha", label="alpha-label")\n'
        'await agent("beta", label="beta-label", phase="Check")\n'
        'return "done"\n'
    )

    summary = await tool.run_workflow(script=script)

    assert len(calls) == 2
    run_id = next(line.split("runId:")[1].strip() for line in summary.splitlines() if "runId:" in line)
    run_dir = Path(state_dir) / "workflows" / run_id
    transcript = (run_dir / "transcript.md").read_text()
    assert "# Workflow transcript: transcript" in transcript
    assert "Max agent depth: 1" in transcript
    assert "alpha-label" in transcript
    assert "beta-label" in transcript
    assert "Find" in transcript
    assert "Check" in transcript
    # Main workflow transcript should not advertise per-agent transcript paths;
    # agents should use the main result/transcript and avoid individual sub-agent transcripts.
    assert "agent-000-alpha-label.md" not in transcript
    raw_events = [json.loads(line) for line in (run_dir / "transcript.jsonl").read_text().splitlines() if line]
    assert any(event["type"] == "agent_call" for event in raw_events)


@pytest.mark.asyncio
async def test_failed_agent_call_is_recorded_in_transcript(
    workflow_tool: tuple[WorkflowTool, Path],
) -> None:
    tool, state_dir = workflow_tool

    async def failing_dispatch(
        agent_class: type[Any],
        task: str,
        *,
        reservation: AgentReservation,
        **kwargs: Any,
    ) -> tuple[str, int, Any]:
        assert agent_class is not None
        assert task == "boom"
        reservation.report_total(9)
        raise RuntimeError("agent exploded")

    tool._agent_tool.dispatch_workflow_agent = failing_dispatch
    script = 'meta = {"name": "failure", "description": "d"}\nvalue = await agent("boom", label="boom")\nreturn {"value": value}\n'

    summary = await tool.run_workflow(script=script)

    assert "transcriptPath:" in summary
    run_id = next(line.split("runId:")[1].strip() for line in summary.splitlines() if "runId:" in line)
    transcript = (Path(state_dir) / "workflows" / run_id / "transcript.md").read_text()
    assert "failed" in transcript
    assert "agent exploded" in transcript
    assert "- tokens: `9`" in transcript
    assert json.loads((Path(state_dir) / "workflows" / run_id / "run.json").read_text())["total_tokens"] == 9


@pytest.mark.asyncio
async def test_run_workflow_carries_phase_and_label_to_dispatch(workflow_tool):
    tool, state_dir = workflow_tool
    stub, calls = _stub_dispatch()
    tool._agent_tool.dispatch_workflow_agent = stub

    script = 'meta = {"name": "p", "description": "d"}\nphase("Build")\nawait agent("go", label="my-label")\nreturn 1\n'
    summary = await tool.run_workflow(script=script)
    _task, _schema, extra, _artifact_paths, _artifact_metadata = calls[0]
    assert extra["phase"] == "Build"
    assert extra["label"] == "my-label"
    assert "workflow_run_id" in extra
    assert extra["depth"] == 1
    assert extra["max_agent_depth"] == 1
    assert extra["effective_routing"]["provider"] == "anthropic"
    assert extra["requested_routing"] is None
    assert extra["actual_agent_type"] == "GeneralAgent"
    run_id = next(line.split("runId:")[1].strip() for line in summary.splitlines() if "runId:" in line)
    journal_entry = json.loads((state_dir / "workflows" / run_id / "journal.jsonl").read_text().splitlines()[0])
    transcript = (state_dir / "workflows" / run_id / "transcript.md").read_text()
    assert "requested_routing" in journal_entry
    assert journal_entry["requested_routing"] is None
    assert "requested_routing: `inherited (null)`" in transcript


@pytest.mark.asyncio
async def test_resume_cache_distinguishes_plan_mode_forced_worker(workflow_tool) -> None:
    tool, _ = workflow_tool
    stub, calls = _stub_dispatch()
    tool._agent_tool.dispatch_workflow_agent = stub
    script = 'meta = {"name": "cache-mode", "description": "d"}\nreturn await agent("same task", agent_type="coder")\n'

    first_summary = await tool.run_workflow(script=script)
    first_run_id = next(line.split("runId:")[1].strip() for line in first_summary.splitlines() if "runId:" in line)
    assert len(calls) == 1
    assert calls[0][2]["actual_agent_type"] == "CoderAgent"

    calls.clear()
    tool.caller.tool_collection.read_only = True
    await tool.run_workflow(resume_from_run_id=first_run_id)

    assert len(calls) == 1
    assert calls[0][2]["actual_agent_type"] == "InvestigationAgent"


@pytest.mark.asyncio
async def test_run_workflow_propagates_explicit_max_agent_depth(
    workflow_tool: tuple[WorkflowTool, Path],
) -> None:
    tool, _ = workflow_tool
    stub, calls = _stub_dispatch()
    tool._agent_tool.dispatch_workflow_agent = stub

    script = (
        'meta = {"name": "nested", "description": "d", "max_agent_depth": 2}\n'
        'await agent("go", label="coordinator")\n'
        "return 1\n"
    )
    await tool.run_workflow(script=script)

    _task, _schema, extra, _artifact_paths, artifact_metadata = calls[0]
    assert extra["depth"] == 1
    assert extra["max_agent_depth"] == 2
    assert artifact_metadata["max_agent_depth"] == 2


@pytest.mark.asyncio
async def test_per_agent_markdown_and_jsonl_are_written(workflow_tool):
    tool, state_dir = workflow_tool

    class ArtifactAgent:
        agent_name = "artifact-agent"

        def __init__(self, **kwargs):
            self.total_tokens_used = 11
            self._history = [
                {"role": "user", "content": [{"type": "text", "text": "artifact task"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "artifact answer"}]},
            ]

        async def process_message_stream(self, task):
            yield {"type": "response", "content": "artifact answer", "complete": True, "uuid": "u1"}

        def dump_message_history(self):
            return self._history

        async def recap_agent_outcome(self):
            return "final artifact recap"

    script = 'meta = {"name": "agent-artifacts", "description": "d"}\nreturn await agent("artifact task", label="artifact")\n'

    with patch("kolega_code.agent.generalagent.GeneralAgent", ArtifactAgent):
        summary = await tool.run_workflow(script=script)

    run_id = next(line.split("runId:")[1].strip() for line in summary.splitlines() if "runId:" in line)
    agents_dir = Path(state_dir) / "workflows" / run_id / "agents"
    markdown = agents_dir / "agent-000-artifact.md"
    raw = agents_dir / "agent-000-artifact.jsonl"
    assert markdown.is_file()
    assert raw.is_file()
    markdown_text = markdown.read_text()
    assert "artifact task" in markdown_text
    assert "final artifact recap" in markdown_text
    assert "- Requested routing: null" in markdown_text
    assert "Tokens: 11" in markdown_text
    assert "Actual agent type: ArtifactAgent" in markdown_text
    assert "Requested routing: null" in markdown_text
    assert '"provider": "anthropic"' in markdown_text
    assert "artifact answer" in raw.read_text()
    assert json.loads((Path(state_dir) / "workflows" / run_id / "run.json").read_text())["total_tokens"] == 11


@pytest.mark.asyncio
async def test_resume_replays_without_redispatch(workflow_tool):
    tool, state_dir = workflow_tool
    stub, calls = _stub_dispatch()
    tool._agent_tool.dispatch_workflow_agent = stub

    summary = await tool.run_workflow(script=SCRIPT)
    run_id = next(line.split("runId:")[1].strip() for line in summary.splitlines() if "runId:" in line)
    assert len(calls) == 4

    calls.clear()
    summary2 = await tool.run_workflow(script=SCRIPT, resume_from_run_id=run_id)
    assert len(calls) == 0  # fully replayed from journal
    assert "completed" in summary2
    run_id2 = next(line.split("runId:")[1].strip() for line in summary2.splitlines() if "runId:" in line)
    transcript2 = (Path(state_dir) / "workflows" / run_id2 / "transcript.md").read_text()
    assert "Cached resume calls" in transcript2
    assert "served from resume cache" in transcript2


@pytest.mark.asyncio
async def test_read_only_caller_forces_investigation_agents(workflow_tool, caller):
    """A read-only orchestrator (plan mode) forces every sub-agent to investigation,
    regardless of the agent_type the script asked for."""
    tool, _ = workflow_tool
    caller.tool_collection.read_only = True

    seen = []

    async def stub(agent_class, task, *, config=None, schema=None, sub_agent_info_extra=None, **kwargs):
        seen.append((agent_class.__name__, config, sub_agent_info_extra))
        return (f"recap:{task}", 1, None)

    tool._agent_tool.dispatch_workflow_agent = stub
    script = (
        'meta = {"name": "p", "description": "d"}\n'
        'await agent("research", agent_type="general")\n'
        'await agent("more", agent_type="coder")\n'
        "return 1\n"
    )
    await tool.run_workflow(script=script)
    assert [entry[0] for entry in seen] == ["InvestigationAgent", "InvestigationAgent"]
    assert all(entry[2]["actual_agent_type"] == "InvestigationAgent" for entry in seen)


@pytest.mark.asyncio
async def test_invalid_meta_reports_failure(workflow_tool):
    tool, _ = workflow_tool
    stub, _ = _stub_dispatch()
    tool._agent_tool.dispatch_workflow_agent = stub
    # Missing meta entirely -> WorkflowScriptError surfaced before any run dir.
    from kolega_code.agent.orchestration import WorkflowScriptError

    with pytest.raises(WorkflowScriptError):
        await tool.run_workflow(script="return 1")


def _config_with_investigation_override() -> AgentConfig:
    return AgentConfig(
        anthropic_api_key="anthropic-key",
        deepseek_api_key="deepseek-key",
        agent_models={
            "investigation": ModelConfig(provider=ModelProvider.DEEPSEEK, model="deepseek-v4-flash"),
        },
    )


@pytest.mark.asyncio
async def test_atomic_override_replaces_only_actual_worker_role(tmp_path, connection_manager, caller):
    config = _config_with_investigation_override()
    tool = WorkflowTool(str(tmp_path / "project"), "ws", "thread", connection_manager, config, caller, None)
    original_fast = config.fast_config
    original_thinking = config.thinking_config
    original_agent_models = dict(config.agent_models)
    captured: list[AgentConfig] = []

    async def stub(
        agent_class: type[Any],
        task: str,
        *,
        config: AgentConfig | None = None,
        **kwargs: Any,
    ):
        assert agent_class.__name__ == "InvestigationAgent"
        assert config is not None
        captured.append(config)
        return ("ok", 1, None)

    tool._agent_tool.dispatch_workflow_agent = stub
    script = (
        'meta = {"name": "override", "description": "d"}\n'
        'return await agent("route", agent_type="investigation", '
        'model_override={"provider": "anthropic", "model": "claude-opus-4-7", "effort": "high"})\n'
    )
    await tool.run_workflow(script=script)

    overridden = captured[0]
    selected = overridden.model_config_for_agent("investigation-agent")
    assert selected.provider == ModelProvider.ANTHROPIC
    assert selected.model == "claude-opus-4-7"
    assert selected.thinking_effort == "high"
    assert overridden.fast_config == original_fast
    assert overridden.thinking_config == original_thinking
    assert config.agent_models == original_agent_models


@pytest.mark.asyncio
async def test_atomic_override_accepts_explicit_null_for_no_effort_model(workflow_tool):
    tool, _ = workflow_tool
    captured: list[AgentConfig] = []

    async def stub(
        agent_class: type[Any],
        task: str,
        *,
        config: AgentConfig | None = None,
        **kwargs: Any,
    ):
        assert config is not None
        captured.append(config)
        return ("ok", 1, None)

    tool._agent_tool.dispatch_workflow_agent = stub
    script = (
        'meta = {"name": "no-effort", "description": "d"}\n'
        'return await agent("route", model_override={"provider": "anthropic", '
        '"model": "claude-sonnet-4-5-20250929", "effort": None})\n'
    )
    await tool.run_workflow(script=script)

    selected = captured[0].model_config_for_agent("general-agent")
    assert selected.model == "claude-sonnet-4-5-20250929"
    assert selected.thinking_effort is None


@pytest.mark.asyncio
async def test_no_override_passes_through_role_configuration(tmp_path, connection_manager, caller):
    config = _config_with_investigation_override()
    tool = WorkflowTool(str(tmp_path / "project"), "ws", "thread", connection_manager, config, caller, None)
    captured: list[Any] = []

    async def stub(agent_class, task, *, config=None, sub_agent_info_extra=None, **kwargs):
        captured.append((config, sub_agent_info_extra))
        return ("ok", 1, None)

    tool._agent_tool.dispatch_workflow_agent = stub
    script = 'meta = {"name": "inherit", "description": "d"}\nreturn await agent("route", agent_type="investigation")\n'
    await tool.run_workflow(script=script)

    assert captured[0][0] is None
    assert captured[0][1]["effective_routing"] == {
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "effort": None,
    }
    assert config.model_config_for_agent("investigation-agent").model == "deepseek-v4-flash"


@pytest.mark.asyncio
async def test_invalid_override_is_a_failed_agent_result_without_dispatch(workflow_tool):
    tool, state_dir = workflow_tool
    stub = AsyncMock()
    tool._agent_tool.dispatch_workflow_agent = stub
    script = (
        'meta = {"name": "invalid-route", "description": "d"}\n'
        'return await agent("route", model_override={'
        '"provider": "anthropic", "model": "claude-opus-4-8", "effort": None})\n'
    )

    summary = await tool.run_workflow(script=script)

    stub.assert_not_awaited()
    run_id = next(line.split("runId:")[1].strip() for line in summary.splitlines() if "runId:" in line)
    run_dir = Path(state_dir) / "workflows" / run_id
    journal_entry = json.loads((run_dir / "journal.jsonl").read_text().splitlines()[0])
    assert journal_entry["status"] == "failed"
    assert "effort must be a string" in journal_entry["error"]
    assert journal_entry["actual_agent_type"] == "GeneralAgent"


@pytest.mark.asyncio
async def test_browser_override_requires_vision_after_actual_class_selection(workflow_tool):
    tool, state_dir = workflow_tool
    stub = AsyncMock()
    tool._agent_tool.dispatch_workflow_agent = stub
    script = (
        'meta = {"name": "browser-route", "description": "d"}\n'
        'return await agent("browse", agent_type="browser", model_override={'
        '"provider": "deepseek", "model": "deepseek-v4-flash", "effort": "high"})\n'
    )

    summary = await tool.run_workflow(script=script)

    stub.assert_not_awaited()
    run_id = next(line.split("runId:")[1].strip() for line in summary.splitlines() if "runId:" in line)
    entry = json.loads((Path(state_dir) / "workflows" / run_id / "journal.jsonl").read_text().splitlines()[0])
    assert entry["status"] == "failed"
    assert "vision-capable" in entry["error"]
    assert entry["actual_agent_type"] == "BrowserAgent"


@pytest.mark.asyncio
async def test_plan_mode_resolves_override_for_forced_investigation_agent(workflow_tool, caller):
    tool, state_dir = workflow_tool
    caller.tool_collection.read_only = True
    captured: list[tuple[type[Any], AgentConfig, dict[str, Any], dict[str, Any]]] = []

    async def stub(
        agent_class: type[Any],
        task: str,
        *,
        config: AgentConfig | None = None,
        sub_agent_info_extra: dict[str, Any] | None = None,
        artifact_metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        assert config is not None
        assert sub_agent_info_extra is not None
        assert artifact_metadata is not None
        captured.append((agent_class, config, sub_agent_info_extra, artifact_metadata))
        return ("ok", 1, None)

    tool._agent_tool.dispatch_workflow_agent = stub
    script = (
        'meta = {"name": "plan-route", "description": "d"}\n'
        'return await agent("route", agent_type="browser", model_override={'
        '"provider": "deepseek", "model": "deepseek-v4-flash", "effort": "high"})\n'
    )
    summary = await tool.run_workflow(script=script)

    agent_class, config, sub_info, artifact_metadata = captured[0]
    assert agent_class.__name__ == "InvestigationAgent"
    assert config.model_config_for_agent("investigation-agent").provider == ModelProvider.DEEPSEEK
    assert sub_info["actual_agent_type"] == "InvestigationAgent"
    assert sub_info["requested_routing"]["provider"] == "deepseek"
    assert artifact_metadata["actual_agent_type"] == "InvestigationAgent"

    run_id = next(line.split("runId:")[1].strip() for line in summary.splitlines() if "runId:" in line)
    run_dir = Path(state_dir) / "workflows" / run_id
    journal_entry = json.loads((run_dir / "journal.jsonl").read_text().splitlines()[0])
    assert journal_entry["actual_agent_type"] == "InvestigationAgent"
    assert journal_entry["effective_routing"]["provider"] == "deepseek"
    transcript = (run_dir / "transcript.md").read_text()
    assert "InvestigationAgent" in transcript
    assert "deepseek-v4-flash" in transcript


@pytest.mark.asyncio
async def test_changed_inherited_routing_invalidates_resume(
    tmp_path: Path,
    connection_manager,
    caller,
    monkeypatch,
):
    monkeypatch.setenv("KOLEGA_CODE_STATE_DIR", str(tmp_path))
    first_config = _config_with_investigation_override()
    first = WorkflowTool(str(tmp_path / "project"), "ws", "thread", connection_manager, first_config, caller, None)
    first_stub, first_calls = _stub_dispatch()
    first._agent_tool.dispatch_workflow_agent = first_stub
    script = (
        'meta = {"name": "fingerprint", "description": "d"}\nreturn await agent("route", agent_type="investigation")\n'
    )
    first_summary = await first.run_workflow(script=script)
    first_run_id = next(line.split("runId:")[1].strip() for line in first_summary.splitlines() if "runId:" in line)
    assert len(first_calls) == 1

    second_config = first_config.model_copy(
        update={
            "agent_models": {
                "investigation": ModelConfig(
                    provider=ModelProvider.ANTHROPIC,
                    model="claude-opus-4-8",
                    thinking_effort="high",
                )
            }
        }
    )
    second = WorkflowTool(str(tmp_path / "project"), "ws", "thread", connection_manager, second_config, caller, None)
    second_stub, second_calls = _stub_dispatch()
    second._agent_tool.dispatch_workflow_agent = second_stub

    await second.run_workflow(script=script, resume_from_run_id=first_run_id)
    assert len(second_calls) == 1
