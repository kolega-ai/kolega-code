from unittest.mock import AsyncMock, Mock

from kolega_code.agent.coder import CoderAgent
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.agent.prompt_provider import AgentMode, AgentType, PromptExtension


def test_coder_agent_includes_matching_prompt_extensions(tmp_path):
    config = AgentConfig(
        anthropic_api_key="test-key",
        openai_api_key="test-key",
        long_context_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model="claude-sonnet-4-20250514",
            rate_limits=RateLimitConfig(),
        ),
        fast_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model="claude-3-haiku-20240307",
            rate_limits=RateLimitConfig(),
        ),
        thinking_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model="claude-3-7-sonnet-20250219",
            rate_limits=RateLimitConfig(),
            thinking_tokens=1024,
        ),
    )

    connection_manager = Mock()
    connection_manager.broadcast_event = AsyncMock()

    agent = CoderAgent(
        project_path=tmp_path,
        workspace_id="workspace-123",
        thread_id="thread-123",
        connection_manager=connection_manager,
        config=config,
        agent_mode=AgentMode.CODE,
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
