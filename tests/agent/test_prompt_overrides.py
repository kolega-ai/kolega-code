from datetime import datetime
from unittest.mock import AsyncMock, Mock

from kolega_code.agent.coder import CoderAgent
from kolega_code.agent.planningagent import PlanningAgent
from kolega_code.agent.prompt_overrides import ProjectPromptOverrides
from kolega_code.agent.prompt_provider import AgentMode, AgentType, PromptExtension
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.services.file_system import LocalFileSystem


def make_config() -> AgentConfig:
    def cfg():
        return ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model="claude-haiku-4-5-20251001",
            rate_limits=RateLimitConfig(),
        )

    return AgentConfig(
        anthropic_api_key="test-key",
        openai_api_key="test-key",
        long_context_config=cfg(),
        fast_config=cfg(),
        thinking_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model="claude-haiku-4-5-20251001",
            rate_limits=RateLimitConfig(),
            thinking_effort="medium",
        ),
    )


def make_connection_manager():
    connection_manager = Mock()
    connection_manager.broadcast_event = AsyncMock()
    return connection_manager


def test_project_prompt_overrides_load_uppercase_files_only(tmp_path):
    prompt_dir = tmp_path / ".kolega" / "prompts"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "CODER.md").write_text("Coder override", encoding="utf-8")
    (prompt_dir / "PLANNING.md").write_text("Planning override", encoding="utf-8")
    (prompt_dir / "GENERAL.md").write_text("General override", encoding="utf-8")
    (prompt_dir / "INVESTIGATION.md").write_text("Investigation override", encoding="utf-8")
    (prompt_dir / "BROWSER.md").write_text("Browser override", encoding="utf-8")
    (prompt_dir / "COMPACTION.md").write_text("Compaction override", encoding="utf-8")

    overrides = ProjectPromptOverrides(LocalFileSystem(tmp_path))

    coder_override = overrides.load_agent_system_prompt(AgentType.CODER)
    assert coder_override is not None
    assert coder_override.content == "Coder override"
    planning_override = overrides.load_agent_system_prompt(AgentType.PLANNING)
    assert planning_override is not None
    assert planning_override.content == "Planning override"
    general_override = overrides.load_agent_system_prompt(AgentType.GENERAL)
    assert general_override is not None
    assert general_override.content == "General override"
    investigation_override = overrides.load_agent_system_prompt(AgentType.INVESTIGATION)
    assert investigation_override is not None
    assert investigation_override.content == "Investigation override"
    browser_override = overrides.load_agent_system_prompt(AgentType.BROWSER)
    assert browser_override is not None
    assert browser_override.content == "Browser override"
    compaction_override = overrides.load_compaction_system_prompt()
    assert compaction_override is not None
    assert compaction_override.content == "Compaction override"

    (prompt_dir / "CODER.md").unlink()
    (prompt_dir / "coder.md").write_text("lowercase ignored", encoding="utf-8")
    assert overrides.load_agent_system_prompt(AgentType.CODER) is None


def test_project_prompt_overrides_skip_oversized_files(tmp_path):
    prompt_dir = tmp_path / ".kolega" / "prompts"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "CODER.md").write_text("x" * (128 * 1024 + 1), encoding="utf-8")

    overrides = ProjectPromptOverrides(LocalFileSystem(tmp_path))

    assert overrides.load_agent_system_prompt(AgentType.CODER) is None


def test_coder_override_replaces_base_prompt_and_keeps_dynamic_sections(tmp_path):
    prompt_dir = tmp_path / ".kolega" / "prompts"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "CODER.md").write_text("# Custom coder prompt", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("Project guidance", encoding="utf-8")
    (tmp_path / "AGENT_MEMORY.md").write_text("Remember release steps", encoding="utf-8")

    agent = CoderAgent(
        project_path=tmp_path,
        workspace_id="workspace-123",
        thread_id="thread-123",
        connection_manager=make_connection_manager(),
        config=make_config(),
        agent_mode=AgentMode.CLI,
        workspace_memories=["Workspace fact"],
        prompt_extensions=[
            PromptExtension(
                id="matching",
                title="Matching Extension",
                markdown="Extra matching context.",
                agent_types=[AgentType.CODER],
            ),
            PromptExtension(
                id="browser-only",
                title="Browser Only",
                markdown="Should not render.",
                agent_types=[AgentType.BROWSER],
            ),
        ],
    )

    prompt = agent.system_prompt.content[0].text
    assert "# Custom coder prompt" in prompt
    assert "powerful AI coding assistant" not in prompt
    assert "Matching Extension" in prompt
    assert "Extra matching context." in prompt
    assert "Browser Only" not in prompt
    assert "Project guidance" in prompt
    assert "Remember release steps" in prompt
    assert "Workspace fact" in prompt


def test_coder_override_renders_jinja_context_and_aliases(tmp_path):
    prompt_dir = tmp_path / ".kolega" / "prompts"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "CODER.md").write_text(
        "\n".join(
            [
                "Project via context: {{ context.project_path }}",
                "Project via alias: {{ project_path }}",
                "Today: {{ context.date_today }}",
                "Model: {{ context.model_name }}",
                "Mode: {{ mode }}",
            ]
        ),
        encoding="utf-8",
    )

    agent = CoderAgent(
        project_path=tmp_path,
        workspace_id="workspace-123",
        thread_id="thread-123",
        connection_manager=make_connection_manager(),
        config=make_config(),
        agent_mode=AgentMode.CLI,
    )

    prompt = agent.system_prompt.content[0].text
    assert f"Project via context: {tmp_path}" in prompt
    assert f"Project via alias: {tmp_path}" in prompt
    assert f"Today: {datetime.now().strftime('%Y-%m-%d')}" in prompt
    assert "Model: claude-haiku-4-5-20251001" in prompt
    assert "Mode: cli" in prompt


def test_malformed_coder_override_falls_back_to_default_prompt(tmp_path, caplog, capsys):
    prompt_dir = tmp_path / ".kolega" / "prompts"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "CODER.md").write_text("{{ missing_variable }}", encoding="utf-8")

    agent = CoderAgent(
        project_path=tmp_path,
        workspace_id="workspace-123",
        thread_id="thread-123",
        connection_manager=make_connection_manager(),
        config=make_config(),
        agent_mode=AgentMode.CLI,
    )

    prompt = agent.system_prompt.content[0].text
    captured = capsys.readouterr()
    assert "powerful AI coding assistant" in prompt
    assert "missing_variable" in caplog.text
    assert agent.prompt_override_errors
    assert len(agent.prompt_override_errors) == 1
    assert "Could not render prompt override .kolega/prompts/CODER.md" in agent.prompt_override_errors[0]
    assert "Could not render prompt override .kolega/prompts/CODER.md" in captured.err
    assert "Falling back to the default prompt" in captured.err


def test_inactive_prompt_override_syntax_error_is_reported_on_startup(tmp_path):
    prompt_dir = tmp_path / ".kolega" / "prompts"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "GENERAL.md").write_text("{% if true %}missing endif", encoding="utf-8")

    agent = CoderAgent(
        project_path=tmp_path,
        workspace_id="workspace-123",
        thread_id="thread-123",
        connection_manager=make_connection_manager(),
        config=make_config(),
        agent_mode=AgentMode.CLI,
    )

    prompt = agent.system_prompt.content[0].text
    assert "powerful AI coding assistant" in prompt
    assert len(agent.prompt_override_errors) == 1
    assert "Could not render prompt override .kolega/prompts/GENERAL.md" in agent.prompt_override_errors[0]
    assert "endif" in agent.prompt_override_errors[0]
    assert "Falling back to the default prompt" in agent.prompt_override_errors[0]


def test_inactive_prompt_override_unknown_variable_is_reported_on_startup(tmp_path):
    prompt_dir = tmp_path / ".kolega" / "prompts"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "INVESTIGATION.md").write_text("{{ missing_variable }}", encoding="utf-8")

    agent = CoderAgent(
        project_path=tmp_path,
        workspace_id="workspace-123",
        thread_id="thread-123",
        connection_manager=make_connection_manager(),
        config=make_config(),
        agent_mode=AgentMode.CLI,
    )

    assert len(agent.prompt_override_errors) == 1
    assert "Could not render prompt override .kolega/prompts/INVESTIGATION.md" in agent.prompt_override_errors[0]
    assert "missing_variable" in agent.prompt_override_errors[0]


def test_literal_non_jinja_mangling_renders_without_validation_error(tmp_path):
    prompt_dir = tmp_path / ".kolega" / "prompts"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "CODER.md").write_text("Project literal: { { context.project_path } }", encoding="utf-8")

    agent = CoderAgent(
        project_path=tmp_path,
        workspace_id="workspace-123",
        thread_id="thread-123",
        connection_manager=make_connection_manager(),
        config=make_config(),
        agent_mode=AgentMode.CLI,
    )

    prompt = agent.system_prompt.content[0].text
    assert "Project literal: { { context.project_path } }" in prompt
    assert agent.prompt_override_errors == []


def test_coder_override_satisfies_hosted_mode_without_private_template(tmp_path):
    prompt_dir = tmp_path / ".kolega" / "prompts"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "CODER.md").write_text("Hosted-mode coder override", encoding="utf-8")

    agent = CoderAgent(
        project_path=tmp_path,
        workspace_id="workspace-123",
        thread_id="thread-123",
        connection_manager=make_connection_manager(),
        config=make_config(),
        agent_mode=AgentMode.CODE,
    )

    assert "Hosted-mode coder override" in agent.system_prompt.content[0].text


def test_planning_override_uses_dynamic_sections(tmp_path):
    prompt_dir = tmp_path / ".kolega" / "prompts"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "PLANNING.md").write_text(
        "# Custom planning prompt for {{ context.project_path }} on {{ platform }}", encoding="utf-8"
    )
    (tmp_path / "AGENTS.md").write_text("Planning guidance", encoding="utf-8")

    agent = PlanningAgent(
        project_path=tmp_path,
        workspace_id="workspace-123",
        thread_id="thread-123",
        connection_manager=make_connection_manager(),
        config=make_config(),
        agent_mode=AgentMode.CLI,
        prompt_extensions=[
            PromptExtension(
                id="planning-context",
                title="Planning Context",
                markdown="Plan carefully.",
                agent_types=[AgentType.PLANNING],
            )
        ],
    )

    prompt = agent.system_prompt.content[0].text
    assert f"# Custom planning prompt for {tmp_path}" in prompt
    assert "Planning Context" in prompt
    assert "Plan carefully." in prompt
    assert "Planning guidance" in prompt
