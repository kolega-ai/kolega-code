"""Integration checks for WorkflowTool.run_workflow: artifacts, dispatch, resume.

The sub-agent dispatch is stubbed, so these run without the LLM stack but exercise
the real run_workflow code path (state-dir resolution, journal, emit, summary).
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from kolega_code.agent.tool_backend.workflow_tool import WorkflowTool
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
    config = Mock(spec=AgentConfig)
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


def _stub_dispatch():
    """A dispatch_workflow_agent stub returning (recap, tokens, structured)."""
    calls = []

    async def dispatch_workflow_agent(
        agent_class,
        task,
        *,
        config=None,
        schema=None,
        sub_agent_info_extra=None,
        artifact_paths=None,
        artifact_metadata=None,
    ):
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
async def test_failed_agent_call_is_recorded_in_transcript(workflow_tool):
    tool, state_dir = workflow_tool

    async def failing_dispatch(*args, **kwargs):
        raise RuntimeError("agent exploded")

    tool._agent_tool.dispatch_workflow_agent = failing_dispatch
    script = 'meta = {"name": "failure", "description": "d"}\nvalue = await agent("boom", label="boom")\nreturn {"value": value}\n'

    summary = await tool.run_workflow(script=script)

    assert "transcriptPath:" in summary
    run_id = next(line.split("runId:")[1].strip() for line in summary.splitlines() if "runId:" in line)
    transcript = (Path(state_dir) / "workflows" / run_id / "transcript.md").read_text()
    assert "failed" in transcript
    assert "agent exploded" in transcript


@pytest.mark.asyncio
async def test_run_workflow_carries_phase_and_label_to_dispatch(workflow_tool):
    tool, _ = workflow_tool
    stub, calls = _stub_dispatch()
    tool._agent_tool.dispatch_workflow_agent = stub

    script = 'meta = {"name": "p", "description": "d"}\nphase("Build")\nawait agent("go", label="my-label")\nreturn 1\n'
    await tool.run_workflow(script=script)
    _task, _schema, extra, _artifact_paths, _artifact_metadata = calls[0]
    assert extra["phase"] == "Build"
    assert extra["label"] == "my-label"
    assert "workflow_run_id" in extra


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
    assert "artifact answer" in raw.read_text()


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
        seen.append(agent_class.__name__)
        return (f"recap:{task}", 1, None)

    tool._agent_tool.dispatch_workflow_agent = stub
    script = (
        'meta = {"name": "p", "description": "d"}\n'
        'await agent("research", agent_type="general")\n'
        'await agent("more", agent_type="coder")\n'
        "return 1\n"
    )
    await tool.run_workflow(script=script)
    assert seen == ["InvestigationAgent", "InvestigationAgent"]


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


def test_config_override_clears_agent_models_for_explicit_model(tmp_path, connection_manager, caller):
    config = _config_with_investigation_override()
    tool = WorkflowTool(str(tmp_path / "project"), "ws", "thread", connection_manager, config, caller, None)

    overridden = tool._config_override("claude-opus-4-7", "high")

    assert overridden.long_context_config.model == "claude-opus-4-7"
    assert overridden.long_context_config.thinking_effort == "high"
    # The workflow author's explicit model wins over the role override.
    assert overridden.agent_models == {}
    assert overridden.model_config_for_agent("investigation-agent").model == "claude-opus-4-7"


def test_config_override_none_preserves_role_overrides(tmp_path, connection_manager, caller):
    config = _config_with_investigation_override()
    tool = WorkflowTool(str(tmp_path / "project"), "ws", "thread", connection_manager, config, caller, None)

    # No explicit model/effort -> no clone, role overrides still flow to sub-agents.
    assert tool._config_override(None, None) is None
    assert config.model_config_for_agent("investigation-agent").model == "deepseek-v4-flash"
