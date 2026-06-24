# ruff: noqa: F401,F811,E402
import os
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dotenv import load_dotenv

from kolega_code.agent.baseagent import BaseAgent
from kolega_code.agent.errors import MaxAgentIterationsExceeded
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.events import AgentConnectionManager
from kolega_code.llm.exceptions import (
    LLMBillingError,
    LLMAuthenticationError,
    LLMContextWindowExceededError,
    LLMInternalServerError,
    LLMRateLimitError,
)
from kolega_code.llm.models import (
    ImageBlock,
    Message,
    MessageHistory,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResult,
)

from .compaction_helpers import FakeLLM

# Load environment variables
load_dotenv()


class _InvestigationRoleAgent(BaseAgent):
    """Minimal BaseAgent carrying the investigation role for resolution tests."""

    agent_name = "investigation-agent"


class _BuildingRoleAgent(BaseAgent):
    agent_name = "coder"


def _role_config():
    return AgentConfig(
        anthropic_api_key="anthropic-key",
        deepseek_api_key="deepseek-key",
        long_context_config=ModelConfig(provider=ModelProvider.ANTHROPIC, model="claude-haiku-4-5-20251001"),
        fast_config=ModelConfig(provider=ModelProvider.ANTHROPIC, model="claude-haiku-4-5-20251001"),
        thinking_config=ModelConfig(provider=ModelProvider.ANTHROPIC, model="claude-haiku-4-5-20251001"),
        agent_models={
            "investigation": ModelConfig(
                provider=ModelProvider.DEEPSEEK, model="deepseek-v4-flash", thinking_effort="high"
            )
        },
    )


def test_agent_uses_role_override_for_primary_model(tmp_path, mock_connection_manager):
    agent = _InvestigationRoleAgent(
        project_path=tmp_path,
        workspace_id="w",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=_role_config(),
    )

    assert agent.primary_model_config.provider == ModelProvider.DEEPSEEK
    assert agent.primary_model_config.model == "deepseek-v4-flash"
    assert agent.primary_model_config.thinking_effort == "high"
    # Model specs and the LLM client both follow the resolved role model.
    assert agent.model_context_length == 1_000_000
    assert agent.llm.provider_name == "deepseek"


def test_agent_without_override_inherits_long_context(tmp_path, mock_connection_manager):
    agent = _BuildingRoleAgent(
        project_path=tmp_path,
        workspace_id="w",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=_role_config(),
    )

    assert agent.primary_model_config.model == "claude-haiku-4-5-20251001"
    assert agent.model_context_length == 200_000
    assert agent.llm.provider_name == "anthropic"
