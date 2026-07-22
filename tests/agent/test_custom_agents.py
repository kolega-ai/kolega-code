from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from kolega_code.agent.coder import CoderAgent
from kolega_code.agent.custom_agents import (
    MAX_CUSTOM_AGENT_FILE_BYTES,
    CustomAgent,
    CustomAgentDefinition,
    discover_custom_agents,
    validate_custom_agent_models,
)
from kolega_code.agent.planningagent import PlanningAgent
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.llm.models import TextBlock
from kolega_code.config import ModelConfig, ModelProvider
from kolega_code.permissions import PermissionMode


def _write_agent(
    root: Path, relative_path: str, frontmatter: str, body: str = "Follow the custom instructions."
) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\n{frontmatter.strip()}\n---\n\n{body}\n", encoding="utf-8")
    return path


def test_discovery_scans_both_recursive_roots_and_project_overrides_user(tmp_path: Path) -> None:
    project = tmp_path / "project"
    state = tmp_path / "state"
    project.mkdir()
    _write_agent(
        state,
        "agents/nested/reviewer.md",
        "name: reviewer\ndescription: User reviewer",
        "User prompt",
    )
    _write_agent(
        project,
        ".kolega/agents/team/reviewer.md",
        "name: reviewer\ndescription: Project reviewer",
        "Project prompt",
    )
    _write_agent(
        state,
        "agents/tester.md",
        "name: tester\ndescription: Runs focused tests",
    )

    catalog = discover_custom_agents(project, state)

    assert catalog.names() == ["reviewer", "tester"]
    reviewer = catalog.get("reviewer")
    assert reviewer is not None
    assert reviewer.scope == "project"
    assert reviewer.prompt == "Project prompt"
    assert any("overrides user definition" in diagnostic.message for diagnostic in catalog.diagnostics)


@pytest.mark.parametrize(
    ("frontmatter", "body", "error_text"),
    [
        ("name: Bad_Name\ndescription: invalid name", "Prompt", "lowercase kebab-case"),
        ("name: coder\ndescription: reserved", "Prompt", "reserved"),
        ("name: demo\ndescription: okay\nunknown: value", "Prompt", "unknown frontmatter"),
        ("name: demo\ndescription: okay\ntools: [read_entire_file, read_entire_file]", "Prompt", "duplicate"),
        ("name: demo\ndescription: okay\nmax_iterations: 0", "Prompt", "positive integer"),
        ("name: demo\ndescription: okay\nmode: sometimes", "Prompt", "build, plan, all"),
        ("name: demo\ndescription: okay", "", "must not be empty"),
        ("name: demo\ndescription: okay\nmodel: no-such/model", "Prompt", "not a valid ModelProvider"),
    ],
)
def test_discovery_skips_invalid_definitions(
    tmp_path: Path,
    frontmatter: str,
    body: str,
    error_text: str,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    _write_agent(project, ".kolega/agents/demo.md", frontmatter, body)

    catalog = discover_custom_agents(project, tmp_path / "state")

    assert not catalog.has_agents()
    assert catalog.has_errors()
    assert error_text in catalog.diagnostics[0].message


def test_discovery_reports_missing_frontmatter_and_oversized_files(tmp_path: Path) -> None:
    project = tmp_path / "project"
    agent_dir = project / ".kolega" / "agents"
    agent_dir.mkdir(parents=True)
    (agent_dir / "plain.md").write_text("No frontmatter", encoding="utf-8")
    (agent_dir / "large.md").write_text("x" * (MAX_CUSTOM_AGENT_FILE_BYTES + 1), encoding="utf-8")

    catalog = discover_custom_agents(project, tmp_path / "state")

    assert not catalog.has_agents()
    messages = [diagnostic.message for diagnostic in catalog.diagnostics]
    assert any("missing YAML frontmatter" in message for message in messages)
    assert any("exceeds 128 KiB" in message for message in messages)


def test_discovery_parses_focused_schema_and_warns_on_filename_mismatch(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    _write_agent(
        project,
        ".kolega/agents/not-the-name.md",
        """
name: reviewer
description: Reviews code
mode: all
tools: [read_entire_file, search_codebase]
model: anthropic/claude-opus-4-8
thinking_effort: high
max_iterations: 17
""",
    )

    catalog = discover_custom_agents(project, tmp_path / "state")
    definition = catalog.get("reviewer")

    assert definition is not None
    assert definition.tools == ("read_entire_file", "search_codebase")
    assert definition.mode == "all"
    assert definition.model == "anthropic/claude-opus-4-8"
    assert definition.thinking_effort == "high"
    assert definition.max_iterations == 17
    assert any("does not match filename" in diagnostic.message for diagnostic in catalog.diagnostics)


def test_session_model_validation_removes_effort_incompatible_with_inherited_model(
    tmp_path: Path,
    agent_config,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    _write_agent(
        project,
        ".kolega/agents/reviewer.md",
        "name: reviewer\ndescription: Reviews code\nthinking_effort: high",
    )

    catalog = validate_custom_agent_models(discover_custom_agents(project, tmp_path / "state"), agent_config)

    assert not catalog.has_agents()
    assert any("does not support thinking effort" in diagnostic.message for diagnostic in catalog.diagnostics)


def test_definition_without_model_inherits_general_agent_override(agent_config, tmp_path: Path) -> None:
    general_model = ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model="claude-opus-4-8",
        thinking_effort="high",
    )
    config = agent_config.model_copy(update={"agent_models": {"general": general_model}})
    definition = CustomAgentDefinition(
        name="reviewer",
        description="Reviews code",
        prompt="Review.",
        source_path=tmp_path / "reviewer.md",
        scope="project",
    )

    resolved = definition.resolve_model_config(config)

    assert resolved.model == "claude-opus-4-8"
    assert resolved.thinking_effort == "high"


def test_custom_agent_applies_prompt_model_tools_permissions_and_dynamic_context(
    tmp_path: Path,
    mock_connection_manager,
    agent_config,
) -> None:
    (tmp_path / "AGENTS.md").write_text("Always run the focused tests.", encoding="utf-8")
    permission_callback = AsyncMock()
    definition = CustomAgentDefinition(
        name="reviewer",
        description="Reviews code",
        prompt="You are the project reviewer.",
        source_path=tmp_path / ".kolega/agents/reviewer.md",
        scope="project",
        model="anthropic/claude-opus-4-8",
        thinking_effort="high",
        max_iterations=17,
    )

    agent = CustomAgent(
        project_path=tmp_path,
        workspace_id="workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
        definition=definition,
        allowed_tools=["read_entire_file"],
        agent_mode=AgentMode.CLI,
        permission_mode=PermissionMode.ASK,
        permission_callback=permission_callback,
    )

    assert agent.agent_name == "reviewer"
    assert agent.primary_model_config.model == "claude-opus-4-8"
    assert agent.primary_model_config.thinking_effort == "high"
    assert agent.max_iterations == 17
    assert agent.permission_mode == PermissionMode.ASK
    assert agent.permission_callback is permission_callback
    system_block = agent.system_prompt.content[0]
    assert isinstance(system_block, TextBlock)
    assert "You are the project reviewer." in system_block.text
    assert "Always run the focused tests." in system_block.text
    assert agent.tool_collection is not None
    tool_names = {tool.name for tool in agent.tool_collection.get_tool_list()}
    assert tool_names == {"read_entire_file"}
    assert not tool_names.intersection(agent.tool_collection.agent_dispatch_tools)


def test_custom_agent_runtime_resolved_model_wins_definition(
    tmp_path: Path,
    mock_connection_manager,
    agent_config,
) -> None:
    definition = CustomAgentDefinition(
        name="reviewer",
        description="Reviews code",
        prompt="Review.",
        source_path=tmp_path / "reviewer.md",
        scope="project",
        model="anthropic/claude-opus-4-8",
        thinking_effort="high",
    )
    runtime_model = ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model="claude-sonnet-4-5-20250929",
        thinking_effort=None,
    )

    agent = CustomAgent(
        project_path=tmp_path,
        workspace_id="workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
        definition=definition,
        allowed_tools=[],
        resolved_model=runtime_model,
    )

    assert agent.primary_model_config.provider == ModelProvider.ANTHROPIC
    assert agent.primary_model_config.model == "claude-sonnet-4-5-20250929"
    assert agent.primary_model_config.thinking_effort is None


def test_custom_agent_catalog_filters_build_plan_and_all_modes(
    tmp_path: Path,
    mock_connection_manager,
    agent_config,
) -> None:
    _write_agent(
        tmp_path,
        ".kolega/agents/reviewer.md",
        "name: reviewer\ndescription: Reviews code for correctness",
    )
    _write_agent(
        tmp_path,
        ".kolega/agents/planner.md",
        "name: planner\ndescription: Produces focused plans\nmode: plan",
    )
    _write_agent(
        tmp_path,
        ".kolega/agents/shared.md",
        "name: shared\ndescription: Works in either mode\nmode: all",
    )
    catalog = discover_custom_agents(tmp_path, tmp_path / "state")

    coder = CoderAgent(
        project_path=tmp_path,
        workspace_id="workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
        agent_mode=AgentMode.CLI,
        custom_agent_catalog=catalog.for_mode("build"),
    )
    planning = PlanningAgent(
        project_path=tmp_path,
        workspace_id="workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
        agent_mode=AgentMode.CLI,
        custom_agent_catalog=catalog.for_mode("plan"),
    )

    coder_tool = next(tool for tool in coder.tool_collection.get_tool_list() if tool.name == "dispatch_custom_agent")
    planning_tool = next(
        tool for tool in planning.tool_collection.get_tool_list() if tool.name == "dispatch_custom_agent"
    )
    assert coder_tool.input_schema["properties"]["agent"]["enum"] == ["reviewer", "shared"]
    assert planning_tool.input_schema["properties"]["agent"]["enum"] == ["planner", "shared"]
    assert planning_tool.input_schema["required"] == ["agent", "task"]
    override_schema = planning_tool.input_schema["properties"]["model_override"]
    assert override_schema["required"] == ["provider", "model", "thinking_effort"]
    assert {"type": "null"} in override_schema["properties"]["thinking_effort"]["anyOf"]
    assert "Reviews code for correctness" in coder_tool.description
    assert "Produces focused plans" in planning_tool.description


def test_planning_agent_hides_dispatch_when_no_definition_opts_into_plan(
    tmp_path: Path,
    mock_connection_manager,
    agent_config,
) -> None:
    _write_agent(
        tmp_path,
        ".kolega/agents/reviewer.md",
        "name: reviewer\ndescription: Build-only reviewer",
    )
    catalog = discover_custom_agents(tmp_path, tmp_path / "state").for_mode("plan")
    planning = PlanningAgent(
        project_path=tmp_path,
        workspace_id="workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
        agent_mode=AgentMode.CLI,
        custom_agent_catalog=catalog,
    )

    assert "dispatch_custom_agent" not in {tool.name for tool in planning.tool_collection.get_tool_list()}


@pytest.mark.asyncio
async def test_dispatch_rejects_tools_outside_callers_capability_ceiling(
    tmp_path: Path,
    mock_connection_manager,
    agent_config,
) -> None:
    _write_agent(
        tmp_path,
        ".kolega/agents/editor.md",
        "name: editor\ndescription: Edits files\nmode: plan\ntools: [write]",
    )
    catalog = discover_custom_agents(tmp_path, tmp_path / "state").for_mode("plan")
    planning = PlanningAgent(
        project_path=tmp_path,
        workspace_id="workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
        agent_mode=AgentMode.CLI,
        custom_agent_catalog=catalog,
    )

    with pytest.raises(ValueError, match="requests unavailable tool.*write"):
        await planning.tool_collection.agent_tool.dispatch_custom_agent("editor", "Edit the file")


@pytest.mark.asyncio
async def test_dispatch_passes_resolved_definition_and_narrowed_tools_to_fresh_agent(
    tmp_path: Path,
    mock_connection_manager,
    agent_config,
) -> None:
    _write_agent(
        tmp_path,
        ".kolega/agents/reviewer.md",
        "name: reviewer\ndescription: Reviews code\ntools: [read_entire_file]",
    )
    catalog = discover_custom_agents(tmp_path, tmp_path / "state")
    coder = CoderAgent(
        project_path=tmp_path,
        workspace_id="workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
        agent_mode=AgentMode.CLI,
        custom_agent_catalog=catalog,
    )

    class StubCustomAgent:
        last_kwargs: dict[str, object] | None = None

        def __init__(self, *args, **kwargs):
            StubCustomAgent.last_kwargs = kwargs
            self.total_tokens_used = 0
            self.parent_tool_call_id = None
            self.conversation_id = None
            self.sub_agent_context = None

        async def process_message_stream(self, task):
            if False:
                yield task

        def dump_message_history(self):
            return []

        async def recap_agent_outcome(self):
            return "Review complete"

    with patch("kolega_code.agent.custom_agents.CustomAgent", StubCustomAgent):
        result = await coder.tool_collection.agent_tool.dispatch_custom_agent("reviewer", "Review src/app.py")

    assert result == "Review complete"
    captured = StubCustomAgent.last_kwargs
    assert captured is not None
    captured_definition = captured["definition"]
    assert isinstance(captured_definition, CustomAgentDefinition)
    assert captured_definition.name == "reviewer"
    assert captured["allowed_tools"] == ["read_entire_file"]
    assert captured["permission_mode"] == coder.permission_mode
    assert captured["permission_callback"] is coder.permission_callback
    assert "resolved_model" not in captured


@pytest.mark.asyncio
async def test_dispatch_runtime_override_replaces_custom_frontmatter_route(
    tmp_path: Path,
    mock_connection_manager,
    agent_config,
) -> None:
    _write_agent(
        tmp_path,
        ".kolega/agents/reviewer.md",
        ("name: reviewer\ndescription: Reviews code\nmodel: anthropic/claude-opus-4-8\nthinking_effort: high"),
    )
    catalog = discover_custom_agents(tmp_path, tmp_path / "state")
    coder = CoderAgent(
        project_path=tmp_path,
        workspace_id="workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
        agent_mode=AgentMode.CLI,
        custom_agent_catalog=catalog,
    )

    class StubCustomAgent:
        last_kwargs: dict[str, object] | None = None

        def __init__(self, *args, **kwargs):
            StubCustomAgent.last_kwargs = kwargs
            self.total_tokens_used = 0
            self.parent_tool_call_id = None
            self.conversation_id = None
            self.sub_agent_context = None

        async def process_message_stream(self, task):
            if False:
                yield task

        def dump_message_history(self):
            return []

        async def recap_agent_outcome(self):
            return "Review complete"

    with patch("kolega_code.agent.custom_agents.CustomAgent", StubCustomAgent):
        await coder.tool_collection.agent_tool.dispatch_custom_agent(
            "reviewer",
            "Review",
            {
                "provider": "anthropic",
                "model": "claude-sonnet-4-5-20250929",
                "thinking_effort": None,
            },
        )

    captured = StubCustomAgent.last_kwargs
    assert captured is not None
    resolved_model = captured["resolved_model"]
    assert isinstance(resolved_model, ModelConfig)
    assert resolved_model.model == "claude-sonnet-4-5-20250929"
    assert resolved_model.thinking_effort is None


@pytest.mark.asyncio
async def test_dispatch_with_omitted_tools_inherits_non_recursive_caller_surface(
    tmp_path: Path,
    mock_connection_manager,
    agent_config,
) -> None:
    _write_agent(
        tmp_path,
        ".kolega/agents/helper.md",
        "name: helper\ndescription: General build helper",
    )
    catalog = discover_custom_agents(tmp_path, tmp_path / "state")
    coder = CoderAgent(
        project_path=tmp_path,
        workspace_id="workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
        agent_mode=AgentMode.CLI,
        custom_agent_catalog=catalog,
    )

    class StubCustomAgent:
        allowed_tools: list[str] = []

        def __init__(self, *args, **kwargs):
            StubCustomAgent.allowed_tools = kwargs["allowed_tools"]
            self.total_tokens_used = 0
            self.parent_tool_call_id = None
            self.conversation_id = None
            self.sub_agent_context = None

        async def process_message_stream(self, task):
            if False:
                yield task

        def dump_message_history(self):
            return []

        async def recap_agent_outcome(self):
            return "Done"

    with patch("kolega_code.agent.custom_agents.CustomAgent", StubCustomAgent):
        await coder.tool_collection.agent_tool.dispatch_custom_agent("helper", "Help")

    assert "read_entire_file" in StubCustomAgent.allowed_tools
    assert "write" in StubCustomAgent.allowed_tools
    assert "run_workflow" not in StubCustomAgent.allowed_tools
    assert not set(StubCustomAgent.allowed_tools).intersection(coder.tool_collection.agent_dispatch_tools)
