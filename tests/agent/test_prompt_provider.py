"""
Unit tests for the PromptProvider class.
"""

from pathlib import Path

import pytest

from kolega_code.agent.prompt_provider import (
    AgentMode,
    AgentType,
    MissingPromptTemplateError,
    PromptContext,
    PromptExtension,
    PromptProvider,
)


class TestPromptProvider:
    """Test suite for PromptProvider functionality."""

    @pytest.fixture
    def prompt_provider(self):
        """Create a PromptProvider instance."""
        return PromptProvider()

    @pytest.fixture
    def prompt_context(self):
        """Create a test prompt context."""
        return PromptContext(
            system_name="Kolega Code",
            project_path="/test/project",
            is_git_repo=True,
            platform="Darwin",
            date_today="2024-01-15",
            model_name="claude-opus-4-8",
            available_ports="9001-9999",
            project_guidance="Test project documentation",
            project_guidance_file="AGENTS.md",
            workspace_id="test-workspace-123",
        )

    def test_coder_agent_cli_mode(self, prompt_provider, prompt_context):
        """Test coder agent with public CLI mode."""
        prompt = prompt_provider.get_system_prompt(
            agent_type=AgentType.CODER, mode=AgentMode.CLI, context=prompt_context
        )

        assert prompt is not None
        assert "local developer CLI" in prompt
        assert "Kolega Code" in prompt
        assert "/test/project" in prompt
        assert "Test project documentation" in prompt
        assert "AGENTS.md" in prompt
        assert len(prompt) > 0

    @pytest.mark.parametrize("mode", [AgentMode.CODE, AgentMode.VIBE, AgentMode.FIX])
    def test_hosted_coder_modes_require_host_template(self, prompt_provider, prompt_context, mode):
        """Hosted coder modes are private and require host-owned prompt templates."""
        with pytest.raises(MissingPromptTemplateError) as exc_info:
            prompt_provider.get_system_prompt(agent_type=AgentType.CODER, mode=mode, context=prompt_context)

        assert mode.value in str(exc_info.value)
        assert "host-owned template_dirs" in str(exc_info.value)

    def test_cli_mode_prompt_omits_platform_memory_bank_instructions(self, prompt_provider, prompt_context):
        """CLI mode should not include platform memory-bank or hosted-agent instructions."""
        prompt = prompt_provider.get_system_prompt(
            agent_type=AgentType.CODER, mode=AgentMode.CLI, context=prompt_context
        )
        prompt_lower = prompt.lower()

        assert "memory bank" not in prompt_lower
        assert "kolega-memory-bank" not in prompt
        assert "dispatch_investigation_agent" not in prompt
        assert "Test Task Detection" not in prompt
        assert "Scope Boundary Management" not in prompt

    def test_investigation_agent_prompt_generation(self, prompt_provider, prompt_context):
        """Test that investigation agent prompts can be generated."""
        prompt = prompt_provider.get_system_prompt(agent_type=AgentType.INVESTIGATION, context=prompt_context)

        assert prompt is not None
        assert len(prompt) > 0
        assert "code investigation agent" in prompt
        assert "explaining a codebase" in prompt
        assert "/test/project" in prompt

    def test_browser_agent_prompt_generation(self, prompt_provider, prompt_context):
        """Test that browser agent prompts can be generated."""
        prompt = prompt_provider.get_system_prompt(agent_type=AgentType.BROWSER, context=prompt_context)

        assert prompt is not None
        assert len(prompt) > 0
        assert "web browser agent" in prompt
        assert "QA on a web application" in prompt
        assert "URL Navigation Guidelines" in prompt

    def test_cli_prompt_with_matching_prompt_extension(self, prompt_provider, prompt_context):
        """CLI mode should render prompt extensions that target CLI mode."""
        prompt = prompt_provider.get_system_prompt(
            agent_type=AgentType.CODER,
            mode=AgentMode.CLI,
            prompt_extensions=[
                PromptExtension(
                    id="cli-example",
                    title="CLI Extension",
                    markdown="Extra CLI context.",
                    agent_types=[AgentType.CODER],
                    modes=[AgentMode.CLI],
                )
            ],
            context=prompt_context,
        )

        assert prompt is not None
        assert "CLI Extension" in prompt
        assert "Extra CLI context." in prompt

    def test_prompt_filters_non_matching_prompt_extension(self, prompt_provider, prompt_context):
        """Prompt extensions should only render for matching agent types and modes."""
        prompt = prompt_provider.get_system_prompt(
            agent_type=AgentType.CODER,
            mode=AgentMode.CLI,
            prompt_extensions=[
                PromptExtension(
                    id="browser-only",
                    title="Browser Only",
                    markdown="This should not render.",
                    agent_types=[AgentType.BROWSER],
                )
            ],
            context=prompt_context,
        )

        assert prompt is not None
        assert "Browser Only" not in prompt
        assert "This should not render." not in prompt

    def test_cli_mode_prompt_includes_workspace_environment_variables(self, prompt_provider, prompt_context):
        """Coder CLI mode should list workspace environment variable descriptions when provided."""
        prompt_context.workspace_environment_variables = {
            "STRIPE_API_KEY": "Stripe API key for billing",
        }

        prompt = prompt_provider.get_system_prompt(
            agent_type=AgentType.CODER, mode=AgentMode.CLI, context=prompt_context
        )

        assert "STRIPE_API_KEY" in prompt
        assert "Stripe API key for billing" in prompt

    def test_host_template_dir_supplies_hosted_mode_prompt(self, tmp_path, prompt_context):
        """Host template dirs can provide private hosted-mode prompts."""
        agents_dir = tmp_path / "system" / "agents"
        agents_dir.mkdir(parents=True)
        (agents_dir / "coder_code.md.j2").write_text(
            "Private {{ context.system_name }} prompt for {{ mode }} at {{ context.project_path }}",
            encoding="utf-8",
        )

        prompt = PromptProvider(template_dirs=[tmp_path]).get_system_prompt(
            agent_type=AgentType.CODER,
            mode=AgentMode.CODE,
            context=prompt_context,
        )

        assert prompt == "Private Kolega Code prompt for code at /test/project"

    def test_host_template_dir_can_use_builtin_includes(self, tmp_path, prompt_context):
        """Private templates can still include bundled generic snippets."""
        agents_dir = tmp_path / "system" / "agents"
        agents_dir.mkdir(parents=True)
        (agents_dir / "coder_vibe.md.j2").write_text(
            "{% include 'system/includes/environment_variables/workspace_env_vars.md' %}",
            encoding="utf-8",
        )
        prompt_context.workspace_environment_variables = {"PAYMENTS_REGION": "Region for payment processor"}

        prompt = PromptProvider(template_dirs=[tmp_path]).get_system_prompt(
            agent_type=AgentType.CODER,
            mode=AgentMode.VIBE,
            context=prompt_context,
        )

        assert "PAYMENTS_REGION" in prompt
        assert "Region for payment processor" in prompt

    def test_hosted_prompt_with_matching_prompt_extension(self, tmp_path, prompt_context):
        """Private hosted prompts still receive filtered prompt extensions."""
        agents_dir = tmp_path / "system" / "agents"
        agents_dir.mkdir(parents=True)
        (agents_dir / "coder_fix.md.j2").write_text(
            "{% for extension in prompt_extensions %}{{ extension.title }}: {{ extension.markdown }}{% endfor %}",
            encoding="utf-8",
        )

        prompt = PromptProvider(template_dirs=[tmp_path]).get_system_prompt(
            agent_type=AgentType.CODER,
            mode=AgentMode.FIX,
            prompt_extensions=[
                PromptExtension(
                    id="fix-example",
                    title="Fix Extension",
                    markdown="Extra fix context.",
                    agent_types=[AgentType.CODER],
                    modes=[AgentMode.FIX],
                )
            ],
            context=prompt_context,
        )

        assert "Fix Extension: Extra fix context." in prompt

    def test_prompt_with_template_slug_in_cli_mode(self, prompt_provider, prompt_context):
        """Template guidance is still available to public CLI mode."""
        prompt = prompt_provider.get_system_prompt(
            agent_type=AgentType.CODER,
            mode=AgentMode.CLI,
            template_slug="mern-stack-template",
            context=prompt_context,
        )

        assert prompt is not None
        assert "Project Starter Template" in prompt

    def test_minimal_context(self, prompt_provider):
        """Test prompt generation with minimal context."""
        prompt = prompt_provider.get_system_prompt(agent_type=AgentType.CODER, mode=AgentMode.CLI)

        assert prompt is not None
        assert "Kolega Code" in prompt

    def test_templates_can_be_loaded(self, prompt_provider):
        """Test that public templates load successfully from the template directory."""
        prompt = prompt_provider.get_system_prompt(agent_type=AgentType.CODER, mode=AgentMode.CLI)
        assert prompt is not None
        assert len(prompt) > 0

    def test_all_public_agent_types(self, prompt_provider, prompt_context):
        """Test that all public agent types can generate prompts."""
        for agent_type in AgentType:
            mode = AgentMode.CLI if agent_type == AgentType.CODER else None
            prompt = prompt_provider.get_system_prompt(agent_type=agent_type, mode=mode, context=prompt_context)

            assert prompt is not None
            assert len(prompt) > 0
            if agent_type == AgentType.CODER:
                assert "powerful AI coding assistant" in prompt
            elif agent_type == AgentType.INVESTIGATION:
                assert "code investigation agent" in prompt
            elif agent_type == AgentType.BROWSER:
                assert "web browser agent" in prompt


def test_public_package_does_not_contain_private_hosted_prompt_markers():
    package_root = Path(__file__).resolve().parents[2] / "kolega_code"
    forbidden_markers = [
        "app.kolega.studio/kolega-error-reporter.js",
        "kolega-memory-bank",
        "Test Task Detection",
        "Scope Boundary Management",
        "You are operating in **fix mode**",
    ]

    for path in package_root.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".j2", ".md"}:
            continue
        if "__pycache__" in path.parts or path.name == "test_prompt_provider.py":
            continue
        text = path.read_text(encoding="utf-8")
        for marker in forbidden_markers:
            assert marker not in text, f"{marker!r} leaked into {path}"
