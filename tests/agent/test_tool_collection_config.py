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


class _StubSandbox:
    def get_host(self, port: int) -> str:
        return f"stub-sandbox:{port}"


class _SandboxCarryingTerminalManager:
    """Terminal-manager stand-in that carries a ``sandbox`` host provider.

    Deliberately not an instance of ``SandboxTerminalManager``: the gate must
    duck-type on the attribute (getattr), not isinstance, so kolega-code-e2b's
    injected manager — possibly subclassed from a pinned older version — keeps
    qualifying when they bump.
    """

    def __init__(self) -> None:
        self.sandbox = _StubSandbox()


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
            provider=ModelProvider.ANTHROPIC, model="claude-sonnet-4-5-20250929", rate_limits=RateLimitConfig()
        ),
        fast_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC, model="claude-sonnet-4-5-20250929", rate_limits=RateLimitConfig()
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
    collection.edit_tool.edit = AsyncMock()
    collection.edit_tool.multi_edit = AsyncMock()
    collection.edit_tool.write = AsyncMock()
    collection.terminal_tool.execute_terminal_command = AsyncMock()
    collection.read_file_tool.read = AsyncMock()
    collection.read_file_tool.read = AsyncMock()
    collection.memory_tool.read_memory = AsyncMock()
    collection.memory_tool.write_memory = AsyncMock()
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

    async def test_get_host_absent_without_sandbox_terminal_manager(
        self,
        project_path: Path,
        mock_connection_manager: AgentConnectionManager,
        agent_config: AgentConfig,
        mock_base_agent: BaseAgent,
    ) -> None:
        """get_host needs a sandbox host provider; the default local manager drops it."""
        tool_collection = ToolCollection(
            project_path,
            "test_workspace",
            str(uuid.uuid4()),
            mock_connection_manager,
            agent_config,
            mock_base_agent,
        )

        assert "get_host" not in [tool.name for tool in tool_collection.get_tool_list()]
        assert tool_collection.has_tool("get_host") is False

    async def test_get_host_present_with_sandbox_terminal_manager(
        self,
        project_path: Path,
        mock_connection_manager: AgentConnectionManager,
        agent_config: AgentConfig,
        mock_base_agent: BaseAgent,
    ) -> None:
        """A terminal manager carrying a ``sandbox`` attribute unlocks get_host.

        The gate is registration-time and duck-typed: the stub is not an
        instance of ``SandboxTerminalManager``, mirroring kolega-code-e2b
        injecting its own (possibly subclassed) manager.
        """
        tool_collection = ToolCollection(
            project_path,
            "test_workspace",
            str(uuid.uuid4()),
            mock_connection_manager,
            agent_config,
            mock_base_agent,
            # The stub is deliberately not a TerminalManager subclass: the gate
            # duck-types on the `sandbox` attribute, so pyright can't see it.
            terminal_manager=_SandboxCarryingTerminalManager(),  # pyright: ignore[reportArgumentType]
        )

        definitions = {tool.name: tool for tool in tool_collection.get_tool_list()}
        assert "get_host" in definitions
        assert definitions["get_host"].description == (
            "Get the externally reachable hostname for a service listening on the given\n"
            "port in this sandbox. Use it to construct URLs instead of localhost."
        )

        host = await tool_collection.call("get_host", port=8000)
        assert host == "stub-sandbox:8000"

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
        file_tools = ["read", "write", "lsp"]
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
        config = ToolCollectionConfig(include_agent_dispatch_tools=True, tool_exclusions=["web_search"])
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
        assert "web_search" not in tool_names

        # Should include the dispatch tool with the full built-in agent_type enum
        assert "dispatch_agent" in tool_names
        assert "list_subagent_models" in tool_names

        definitions = {tool.name: tool for tool in tool_list}
        schema = definitions["dispatch_agent"].input_schema
        assert schema is not None
        assert schema["required"] == ["agent_type", "task"]
        assert schema["properties"]["agent_type"]["enum"] == ["general", "investigation", "browser", "coding"]
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

    async def test_dispatch_agent_rejects_unavailable_agent_type(
        self,
        project_path: Path,
        mock_connection_manager: AgentConnectionManager,
        agent_config: AgentConfig,
        mock_base_agent: BaseAgent,
    ) -> None:
        """An unknown or gated-off agent_type fails with the valid values listed."""
        config = ToolCollectionConfig(include_agent_dispatch_tools=True, excluded_agent_types=["coding"])
        tool_collection = ToolCollection(
            project_path,
            "test_workspace",
            str(uuid.uuid4()),
            mock_connection_manager,
            agent_config,
            mock_base_agent,
            tool_config=config,
        )

        for agent_type in ("nonexistent", "coding"):
            with pytest.raises(ValueError) as exc_info:
                await tool_collection.dispatch_agent(agent_type, "do something")
            assert agent_type in str(exc_info.value)
            assert "general, investigation, browser" in str(exc_info.value)

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

    async def test_registration_collects_exclusive_tools_from_extensions(
        self,
        project_path: Path,
        mock_connection_manager: AgentConnectionManager,
        agent_config: AgentConfig,
        mock_base_agent: BaseAgent,
    ) -> None:
        """The batch-rejection guard reads ``exclusive_tools`` off the live
        collection; registration must derive it from extension declarations."""

        async def switch_workspace(path: str) -> str:
            return path

        async def plain_tool(task: str) -> str:
            return task

        extension = ToolExtension(
            name="host-session-control",
            tools={"switch_workspace": switch_workspace, "plain_tool": plain_tool},
            exclusive_tools=frozenset({"switch_workspace"}),
        )
        tool_collection = ToolCollection(
            project_path,
            "test_workspace",
            str(uuid.uuid4()),
            mock_connection_manager,
            agent_config,
            mock_base_agent,
            tool_extensions=[extension],
        )

        assert tool_collection.exclusive_tools == frozenset({"switch_workspace"})
        assert "switch_workspace" in tool_collection.registry()

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
