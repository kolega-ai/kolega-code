# ruff: noqa: F401,F811,E402
import os
import uuid
from unittest.mock import AsyncMock

import pytest

from kolega_code.agent.baseagent import BaseAgent
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.events import AgentConnectionManager


@pytest.fixture
def agent_config():
    return AgentConfig(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "test_key"),  # Use actual API key from environment
        openai_api_key="test-key",
        long_context_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model="claude-haiku-4-5-20251001",  # Using a valid model name
            rate_limits=RateLimitConfig(),
        ),
        fast_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model="claude-haiku-4-5-20251001",  # Using a valid model name
            rate_limits=RateLimitConfig(),
        ),
        thinking_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model="claude-haiku-4-5-20251001",  # Using a valid model name
            rate_limits=RateLimitConfig(),
            thinking_effort="medium",
        ),
    )

@pytest.fixture
def mock_connection_manager():
    return AsyncMock(spec=AgentConnectionManager)

@pytest.fixture
def base_agent(tmp_path, mock_connection_manager, agent_config):
    return BaseAgent(
        project_path=tmp_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),  # Add thread_id
        connection_manager=mock_connection_manager,
        config=agent_config,
    )

