from unittest.mock import AsyncMock, Mock

from kolega_code.agent.coder import CoderAgent
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.agent.prompt_provider import AgentMode, AgentType, PromptExtension, PromptProvider


def test_coder_agent_includes_matching_prompt_extensions(tmp_path):
    config = AgentConfig(
        anthropic_api_key="test-key",
        openai_api_key="test-key",
        long_context_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model="claude-opus-4-8",
            rate_limits=RateLimitConfig(),
        ),
        fast_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model="claude-haiku-4-5-20251001",
            rate_limits=RateLimitConfig(),
        ),
        thinking_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model="claude-opus-4-8",
            rate_limits=RateLimitConfig(),
            thinking_effort="medium",
        ),
    )

    connection_manager = Mock()
    connection_manager.broadcast_event = AsyncMock()
    template_dir = tmp_path / "prompt_templates"
    agents_dir = template_dir / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "coder_code_mode.j2").write_text(
        "{% for extension in prompt_extensions %}{{ extension.title }}\n{{ extension.markdown }}{% endfor %}",
        encoding="utf-8",
    )

    agent = CoderAgent(
        project_path=tmp_path,
        workspace_id="workspace-123",
        thread_id="thread-123",
        connection_manager=connection_manager,
        config=config,
        agent_mode=AgentMode.CODE,
        prompt_provider=PromptProvider(template_dirs=[template_dir]),
        prompt_extensions=[
            PromptExtension(
                id="host-context",
                title="Host Context",
                markdown="Injected host-specific context.",
                agent_types=[AgentType.CODER],
                modes=[AgentMode.CODE],
            )
        ],
    )

    prompt = agent.system_prompt.content[0].text
    assert "Host Context" in prompt
    assert "Injected host-specific context." in prompt
