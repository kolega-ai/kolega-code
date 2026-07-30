# ruff: noqa: F401,F811,E402
from pathlib import Path
from unittest.mock import AsyncMock, Mock
import uuid

import pytest

from kolega_code.agent.baseagent import BaseAgent
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.events import AgentConnectionManager
from kolega_code.agent.tool_backend.memory_tool import MemoryTool
from kolega_code.agent.tools import ToolCollection, ToolDefinition, ToolCollectionConfig, ToolExtension


INTERNAL_TOOL_NAMES = {
    "registry",
    "has_tool",
    "call",
    "cleanup",
    "initialize",
    "get_tool_list",
    "log_error",
    "log_warning",
    "log_info",
}


@pytest.fixture
def mock_connection_manager() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def project_path(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def agent_config() -> AgentConfig:
    return AgentConfig(
        anthropic_api_key="test_key",
        openai_api_key="test-key",
        long_context_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC, model="test-model", rate_limits=RateLimitConfig()
        ),
        fast_config=ModelConfig(provider=ModelProvider.ANTHROPIC, model="test-model", rate_limits=RateLimitConfig()),
        thinking_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model="test-model",
            rate_limits=RateLimitConfig(),
            thinking_effort="medium",
        ),
    )


@pytest.fixture
def mock_base_agent() -> Mock:
    mock = Mock()
    mock.agent_name = "test_agent"
    # Default: non-vision mock so the read_image tool gate excludes it.
    mock.supports_vision = False
    return mock


@pytest.fixture
def tool_collection(
    project_path: Path,
    mock_connection_manager: AgentConnectionManager,
    agent_config: AgentConfig,
    mock_base_agent: BaseAgent,
) -> ToolCollection:
    # Create a ToolCollection with mocked tools
    collection = ToolCollection(
        project_path, "test_workspace", str(uuid.uuid4()), mock_connection_manager, agent_config, mock_base_agent
    )

    # Mock all tool methods
    collection.think_hard_tool.think_hard = AsyncMock()
    collection.edit_tool.edit = AsyncMock()
    collection.edit_tool.multi_edit = AsyncMock()
    collection.edit_tool.write = AsyncMock()
    collection.list_directory_tool.list_directory = AsyncMock()
    collection.terminal_tool.execute_terminal_command = AsyncMock()
    collection.read_file_tool.read_entire_file = AsyncMock()
    collection.read_file_tool.read_file_section = AsyncMock()
    collection.memory_tool.read_memory = AsyncMock()
    collection.memory_tool.write_memory = AsyncMock()
    collection.search_codebase_tool.search_codebase = AsyncMock()
    collection.glob_tool.find_files_by_pattern = AsyncMock()
    collection.web_fetch_tool.web_fetch = AsyncMock()
    collection.terminal_tool.write_stdin = AsyncMock()

    return collection


@pytest.mark.asyncio
class TestToolCollection:
    async def test_get_tool_list(self, tool_collection: AsyncMock) -> None:
        tool_list = tool_collection.get_tool_list()
        assert isinstance(tool_list, list)
        assert len(tool_list) > 0

        # Check that each tool has required fields
        for tool in tool_list:
            assert isinstance(tool, ToolDefinition)
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")
            assert hasattr(tool, "parameters")
            assert isinstance(tool.parameters, list)
            for param in tool.parameters:
                assert hasattr(param, "name")
                assert hasattr(param, "type")
                assert hasattr(param, "description")
                assert hasattr(param, "required")

        # Check that excluded/internal tools are not in the list
        excluded_tools = tool_collection.tool_exclusions
        tool_names = [tool.name for tool in tool_list]
        assert "exec_command" in tool_names
        assert "write_stdin" in tool_names
        assert "initialize" in excluded_tools
        assert "initialize" not in tool_names
        assert INTERNAL_TOOL_NAMES.isdisjoint(tool_names)
        for excluded_tool in excluded_tools:
            assert excluded_tool not in tool_names

    async def test_tool_collection_config_read_only(
        self,
        project_path: Path,
        mock_connection_manager: AgentConnectionManager,
        agent_config: AgentConfig,
        mock_base_agent: BaseAgent,
    ) -> None:
        """Test that read_only configuration properly filters tools."""
        config = ToolCollectionConfig(read_only=True)
        tool_collection = ToolCollection(
            project_path,
            "test_workspace",
            str(uuid.uuid4()),
            mock_connection_manager,
            agent_config,
            mock_base_agent,
            tool_config=config,
        )

        tool_list = tool_collection.get_tool_list()
        tool_names = [tool.name for tool in tool_list]

        # Should only include read-only tools
        for tool_name in tool_names:
            assert tool_name in ToolCollection.read_only_tools

        # Should not include write tools
        write_tools = ["edit", "lsp_edit", "multi_edit", "write"]
        for write_tool in write_tools:
            assert write_tool not in tool_names

    async def test_tool_collection_config_browser_only(
        self,
        project_path: Path,
        mock_connection_manager: AgentConnectionManager,
        agent_config: AgentConfig,
        mock_base_agent: BaseAgent,
    ) -> None:
        """Test that browser_only configuration properly filters tools."""
        config = ToolCollectionConfig(browser_only=True)
        tool_collection = ToolCollection(
            project_path,
            "test_workspace",
            str(uuid.uuid4()),
            mock_connection_manager,
            agent_config,
            mock_base_agent,
            tool_config=config,
        )

        tool_list = tool_collection.get_tool_list()
        tool_names = [tool.name for tool in tool_list]

        # Should only include browser tools
        for tool_name in tool_names:
            assert tool_name in ToolCollection.browser_tools

        # Should not include file tools
        file_tools = ["read_entire_file", "write", "list_directory"]
        for file_tool in file_tools:
            assert file_tool not in tool_names

    async def test_tool_collection_config_mixed_options(
        self,
        project_path: Path,
        mock_connection_manager: AgentConnectionManager,
        agent_config: AgentConfig,
        mock_base_agent: BaseAgent,
    ) -> None:
        """Test combinations of configuration options work correctly."""
        config = ToolCollectionConfig(include_agent_dispatch_tools=True, tool_exclusions=["think_hard"])
        tool_collection = ToolCollection(
            project_path,
            "test_workspace",
            str(uuid.uuid4()),
            mock_connection_manager,
            agent_config,
            mock_base_agent,
            tool_config=config,
        )

        tool_list = tool_collection.get_tool_list()
        tool_names = [tool.name for tool in tool_list]

        # Should exclude explicitly excluded tools
        assert "think_hard" not in tool_names

        # Should include investigation tools
        assert "dispatch_investigation_agent" in tool_names
        assert "dispatch_browser_agent" in tool_names
        assert "list_subagent_models" in tool_names

        dispatch_names = [
            "dispatch_investigation_agent",
            "dispatch_browser_agent",
            "dispatch_coding_agent",
            "dispatch_general_agent",
        ]
        definitions = {tool.name: tool for tool in tool_list}
        for name in dispatch_names:
            schema = definitions[name].input_schema
            assert schema is not None
            assert schema["required"] == ["task"]
            override = schema["properties"]["model_override"]
            assert override["required"] == ["provider", "model", "thinking_effort"]
            assert override["additionalProperties"] is False
            assert override["properties"]["provider"]["minLength"] == 1
            assert override["properties"]["model"]["minLength"] == 1
            effort_options = override["properties"]["thinking_effort"]["anyOf"]
            assert {"type": "string", "minLength": 1} in effort_options
            assert {"type": "null"} in effort_options

        discovery = tool_collection.registry().get("list_subagent_models")
        assert discovery.parallel_safe is True

    async def test_workflow_depth_gate_cannot_be_bypassed_by_dispatch_extension_group(
        self,
        project_path: Path,
        mock_connection_manager: AgentConnectionManager,
        agent_config: AgentConfig,
        mock_base_agent: BaseAgent,
    ) -> None:
        async def dispatch_host_agent(task: str) -> str:
            return task

        extension = ToolExtension(
            name="host-agent-dispatch",
            tools={"dispatch_host_agent": dispatch_host_agent},
            tool_groups={"agent_dispatch_tools": ["dispatch_host_agent"]},
        )
        config = ToolCollectionConfig(include_agent_dispatch_tools=True)
        tool_collection = ToolCollection(
            project_path,
            "test_workspace",
            str(uuid.uuid4()),
            mock_connection_manager,
            agent_config,
            mock_base_agent,
            tool_config=config,
            tool_extensions=[extension],
        )

        assert "dispatch_host_agent" in tool_collection.registry()

        setattr(
            mock_base_agent,
            "sub_agent_context",
            {"workflow_run_id": "run-1", "depth": 1, "max_agent_depth": 1},
        )
        names_at_limit = set(tool_collection.registry().names())
        assert not names_at_limit.intersection(ToolCollection.agent_dispatch_tools)

    async def test_model_discovery_is_exposed_to_workflow_authors_but_hidden_from_leaves(
        self,
        project_path: Path,
        mock_connection_manager: AgentConnectionManager,
        agent_config: AgentConfig,
        mock_base_agent: BaseAgent,
    ) -> None:
        mock_base_agent.sub_agent = False
        mock_base_agent.gigacode_enabled = False
        leaf_collection = ToolCollection(
            project_path,
            "test_workspace",
            str(uuid.uuid4()),
            mock_connection_manager,
            agent_config,
            mock_base_agent,
        )
        assert "list_subagent_models" not in leaf_collection.registry()

        mock_base_agent.gigacode_enabled = True
        workflow_collection = ToolCollection(
            project_path,
            "test_workspace",
            str(uuid.uuid4()),
            mock_connection_manager,
            agent_config,
            mock_base_agent,
        )
        discovery = workflow_collection.registry().get("list_subagent_models")
        assert discovery.parallel_safe is True

        mock_base_agent.sub_agent = True
        setattr(
            mock_base_agent,
            "sub_agent_context",
            {
                "workflow_run_id": "run-1",
                "depth": 1,
                "max_agent_depth": 1,
            },
        )
        assert "list_subagent_models" not in workflow_collection.registry()

    async def test_backward_compatibility_legacy_parameters(
        self,
        project_path: Path,
        mock_connection_manager: AgentConnectionManager,
        agent_config: AgentConfig,
        mock_base_agent: BaseAgent,
    ) -> None:
        """Test that legacy read_only and browser_only parameters still work."""
        # Test legacy read_only parameter
        tool_collection_ro = ToolCollection(
            project_path,
            "test_workspace",
            str(uuid.uuid4()),
            mock_connection_manager,
            agent_config,
            mock_base_agent,
            read_only=True,
        )

        tool_list_ro = tool_collection_ro.get_tool_list()
        tool_names_ro = [tool.name for tool in tool_list_ro]

        for tool_name in tool_names_ro:
            assert tool_name in ToolCollection.read_only_tools

        # Test legacy browser_only parameter
        tool_collection_browser = ToolCollection(
            project_path,
            "test_workspace",
            str(uuid.uuid4()),
            mock_connection_manager,
            agent_config,
            mock_base_agent,
            browser_only=True,
        )

        tool_list_browser = tool_collection_browser.get_tool_list()
        tool_names_browser = [tool.name for tool in tool_list_browser]

        for tool_name in tool_names_browser:
            assert tool_name in ToolCollection.browser_tools
