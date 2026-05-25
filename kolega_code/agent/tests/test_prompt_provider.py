"""
Unit tests for the PromptProvider class.
"""

import pytest

from kolega_code.agent.prompt_provider import PromptProvider, AgentType, AgentMode, PromptContext, PromptExtension


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
            model_name="claude-3-5-sonnet",
            available_ports="9001-9999",
            kolega_md="Test project documentation",
            workspace_id="test-workspace-123",
        )

    @pytest.fixture
    def template_slug(self):
        """Create a test template slug."""
        return "mern-stack-template"

    def test_coder_agent_prompt_generation(self, prompt_provider, prompt_context):
        """Test that coder agent prompts can be generated."""
        prompt = prompt_provider.get_system_prompt(
            agent_type=AgentType.CODER, mode=AgentMode.CODE, context=prompt_context
        )

        assert prompt is not None
        assert len(prompt) > 0
        assert "Kolega Code" in prompt
        assert "/test/project" in prompt

    def test_coder_agent_vibe_mode(self, prompt_provider, prompt_context):
        """Test coder agent with vibe mode."""
        prompt = prompt_provider.get_system_prompt(
            agent_type=AgentType.CODER, mode=AgentMode.VIBE, context=prompt_context
        )

        assert prompt is not None
        # The coder template doesn't have special mode sections yet
        assert "Kolega Code" in prompt
        assert len(prompt) > 0

    def test_coder_agent_code_mode(self, prompt_provider, prompt_context):
        """Test coder agent with code mode."""
        prompt = prompt_provider.get_system_prompt(
            agent_type=AgentType.CODER, mode=AgentMode.CODE, context=prompt_context
        )

        assert prompt is not None
        # The coder template doesn't have special mode sections yet
        assert "Kolega Code" in prompt
        assert len(prompt) > 0

    def test_coder_agent_cli_mode(self, prompt_provider, prompt_context):
        """Test coder agent with CLI mode."""
        prompt = prompt_provider.get_system_prompt(
            agent_type=AgentType.CODER, mode=AgentMode.CLI, context=prompt_context
        )

        assert prompt is not None
        assert "local developer CLI" in prompt
        assert "Kolega Code" in prompt
        assert "/test/project" in prompt
        assert "Test project documentation" in prompt
        assert len(prompt) > 0

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

    def test_prompt_with_matching_prompt_extension(self, prompt_provider, prompt_context):
        """Test prompt generation with a host-provided prompt extension."""
        prompt = prompt_provider.get_system_prompt(
            agent_type=AgentType.CODER,
            mode=AgentMode.CODE,
            prompt_extensions=[
                PromptExtension(
                    id="example",
                    title="Example Extension",
                    markdown="Extra host context.",
                    agent_types=[AgentType.CODER],
                    modes=[AgentMode.CODE],
                )
            ],
            context=prompt_context,
        )

        assert prompt is not None
        assert "Example Extension" in prompt
        assert "Extra host context." in prompt

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
            mode=AgentMode.CODE,
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

    def test_code_mode_prompt_includes_workspace_environment_variables(self, prompt_provider, prompt_context):
        """Coder code mode should list workspace environment variable descriptions when provided."""
        prompt_context.workspace_environment_variables = {
            "STRIPE_API_KEY": "Stripe API key for billing",
            "DEBUG_MODE": "Toggle verbose logging",
        }

        prompt = prompt_provider.get_system_prompt(
            agent_type=AgentType.CODER, mode=AgentMode.CODE, context=prompt_context
        )

        assert "STRIPE_API_KEY" in prompt
        assert "Stripe API key for billing" in prompt
        assert "DEBUG_MODE" in prompt
        assert "Toggle verbose logging" in prompt

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

    def test_vibe_mode_prompt_includes_workspace_environment_variables(self, prompt_provider, prompt_context):
        """Coder vibe mode should list workspace environment variables when provided."""
        prompt_context.workspace_environment_variables = {
            "PAYMENTS_REGION": "Region for payment processor",
        }

        prompt = prompt_provider.get_system_prompt(
            agent_type=AgentType.CODER, mode=AgentMode.VIBE, context=prompt_context
        )

        assert "PAYMENTS_REGION" in prompt
        assert "Region for payment processor" in prompt

    def test_vibe_mode_prompt_preserves_kolega_error_reporter_script(self, prompt_provider, prompt_context):
        """Coder vibe mode should preserve the Kolega error reporter script in index.html."""
        prompt = prompt_provider.get_system_prompt(
            agent_type=AgentType.CODER, mode=AgentMode.VIBE, context=prompt_context
        )

        assert "https://app.kolega.studio/kolega-error-reporter.js" in prompt
        assert "index.html" in prompt
        assert "never" in prompt.lower()
        assert "remove" in prompt.lower()

    def test_prompt_with_template_slug(self, prompt_provider, prompt_context, template_slug):
        """Test prompt generation with a template slug."""
        prompt = prompt_provider.get_system_prompt(
            agent_type=AgentType.CODER,
            mode=AgentMode.CODE,
            template_slug=template_slug,
            context=prompt_context,
        )

        assert prompt is not None
        assert len(prompt) > 0

    def test_prompt_with_different_template_slugs(self, prompt_provider, prompt_context):
        """Test prompt generation with different template slugs."""
        template_slugs = [
            "mern-stack-template",
            "html-website-template",
            "react-vite-shadcdn-template",
            "non-existent-template",  # Should not cause error
        ]

        for slug in template_slugs:
            prompt = prompt_provider.get_system_prompt(
                agent_type=AgentType.CODER,
                mode=AgentMode.CODE,
                template_slug=slug,
                context=prompt_context,
            )
            assert prompt is not None
            assert len(prompt) > 0
            # For supported templates, it should include template section
            if slug in ["mern-stack-template", "html-website-template", "react-vite-shadcdn-template"]:
                assert "Project Starter Template" in prompt

    def test_minimal_context(self, prompt_provider):
        """Test prompt generation with minimal context."""
        # Should use default PromptContext values
        prompt = prompt_provider.get_system_prompt(agent_type=AgentType.CODER, mode=AgentMode.CODE)

        assert prompt is not None
        assert "Kolega Code" in prompt  # Default system name

    def test_templates_can_be_loaded(self, prompt_provider):
        """Test that templates load successfully from the template directory."""
        # This implicitly verifies that the template directory exists and is accessible
        prompt = prompt_provider.get_system_prompt(agent_type=AgentType.CODER, mode=AgentMode.CODE)
        assert prompt is not None
        assert len(prompt) > 0

    def test_all_agent_types(self, prompt_provider, prompt_context):
        """Test that all defined agent types can generate prompts."""
        for agent_type in AgentType:
            # CODER requires a mode
            mode = AgentMode.CODE if agent_type == AgentType.CODER else None
            prompt = prompt_provider.get_system_prompt(agent_type=agent_type, mode=mode, context=prompt_context)

            assert prompt is not None
            assert len(prompt) > 0
            # Each agent type has its own unique prompt content
            # Just verify it contains some expected keywords
            if agent_type == AgentType.CODER:
                assert "powerful AI coding assistant" in prompt
            elif agent_type == AgentType.INVESTIGATION:
                assert "code investigation agent" in prompt
            elif agent_type == AgentType.BROWSER:
                assert "web browser agent" in prompt
