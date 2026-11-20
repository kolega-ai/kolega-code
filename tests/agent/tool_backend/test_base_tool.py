from pathlib import Path
from unittest.mock import Mock
import uuid

import pytest

from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.agent.tool_backend.base_tool import BaseTool


@pytest.fixture
def mock_connection_manager():
    return Mock()


@pytest.fixture
def project_path(tmp_path):
    return tmp_path


@pytest.fixture
def agent_config():
    return AgentConfig(
        anthropic_api_key="test_key",
        openai_api_key="test-key",
        long_context_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC, model="test-model", rate_limits=RateLimitConfig()
        ),
        fast_config=ModelConfig(provider=ModelProvider.ANTHROPIC, model="test-model", rate_limits=RateLimitConfig()),
    )


@pytest.fixture
def mock_base_agent():
    return Mock()


@pytest.fixture
def base_tool(project_path, mock_connection_manager, agent_config, mock_base_agent):
    return BaseTool(
        project_path, "test_workspace", str(uuid.uuid4()), mock_connection_manager, agent_config, mock_base_agent
    )


class TestBaseTool:
    def test_initialization(self, base_tool, project_path):
        assert base_tool.workspace_id == "test_workspace"
        assert base_tool.project_path == project_path
        assert base_tool.connection_manager is not None
        assert base_tool.caller is not None

    def test_initialization_with_string_path(self, mock_connection_manager, agent_config, mock_base_agent):
        tool = BaseTool(
            "/test/path", "test_workspace", str(uuid.uuid4()), mock_connection_manager, agent_config, mock_base_agent
        )
        assert isinstance(tool.project_path, Path)
        assert str(tool.project_path) == "/test/path"
        assert tool.caller is not None
