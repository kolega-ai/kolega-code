"""Tests for the unified AgentTool dispatch mechanism."""

import pytest
from typing import Any, ClassVar, Optional
from unittest.mock import AsyncMock, Mock, MagicMock, patch
import uuid
import builtins

from kolega_code.agent.tool_backend.agent_tool import AgentTool
from kolega_code.agent.orchestration.accounting import WorkflowRunAccounting
from kolega_code.agent.orchestration.budget import Budget
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig


@pytest.fixture
def mock_config():
    """Create a mock agent configuration."""
    return AgentConfig(
        anthropic_api_key="test-key",
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
def mock_connection_manager():
    """Create a mock connection manager."""
    manager = AsyncMock()
    manager.broadcast_event = AsyncMock()
    return manager


class MockSubAgentRecorder:
    def __init__(self):
        self.start_conversation = AsyncMock(return_value="test-conversation-id")
        self.record_message = AsyncMock()
        self.complete_conversation = AsyncMock()
        self.fail_conversation = AsyncMock()
        self.interrupt_conversation = AsyncMock()


@pytest.fixture
def mock_caller():
    """Create a mock caller agent."""
    caller = Mock()
    caller.agent_name = "test-caller"
    caller.log_info = AsyncMock()
    caller.current_tool_call_id = "test-tool-call-id"  # Set a test tool call ID for sub-agent creation
    caller.sub_agent = False  # Caller is not a sub-agent
    caller.protected_files = ["custom.lock"]
    caller.workspace_env_var_descriptions = {"API_TOKEN": "Token for external API"}
    caller.workspace_memories = []
    caller.prompt_extensions = []
    caller.tool_extensions = []
    caller.usage_recorder = None
    caller.sub_agent_recorder = MockSubAgentRecorder()
    caller.max_iterations = None
    return caller


@pytest.fixture
def agent_tool(tmp_path, mock_connection_manager, mock_config, mock_caller):
    """Create an AgentTool instance for testing."""
    return AgentTool(
        project_path=tmp_path,
        workspace_id="test-workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=mock_config,
        caller=mock_caller,
    )


class MockAgent:
    """Mock agent class for testing."""

    agent_name = "mock-agent"
    default_stream_messages = []
    last_instance: ClassVar[Optional["MockAgent"]] = None

    def __init__(self, *args, **kwargs):
        self.init_kwargs = kwargs
        self.recap_agent_outcome = AsyncMock(return_value="Mock agent completed")
        self._stream_messages = list(self.default_stream_messages)
        self._message_history = []
        self.total_tokens_used = 0  # Add token count attribute
        MockAgent.last_instance = self

    def setup_streaming(self, messages):
        """Setup the process_message_stream to return an async generator."""
        self._stream_messages = messages

    @classmethod
    def configure_streaming(cls, messages):
        """Configure default messages for the next instantiated agent."""
        cls.default_stream_messages = messages

    async def process_message_stream(self, task):
        """Mock implementation of process_message_stream that returns an async generator."""
        # Add the task as a user message to history
        self._message_history.append({"role": "user", "content": [{"type": "text", "text": task}]})

        # Simulate assistant responses
        for msg in self._stream_messages:
            if msg.get("complete", False):
                # Add completed message to history
                self._message_history.append(
                    {"role": "assistant", "content": [{"type": "text", "text": msg.get("content", "")}]}
                )
            yield msg

    def dump_message_history(self):
        """Mock implementation of dump_message_history."""
        return self._message_history

    async def cleanup(self):
        """Mock implementation of cleanup."""
        pass


class MockInvestigationAgent(MockAgent):
    """Mock investigation agent."""

    agent_name = "investigation-agent"


class MockBrowserAgent(MockAgent):
    """Mock browser agent."""

    agent_name = "browser-agent"


class MockCodingAgent(MockAgent):
    """Mock coding agent."""

    agent_name = "coding-agent"


@pytest.mark.asyncio
class TestAgentTool:
    """Test suite for AgentTool."""

    async def test_dispatch_agent_standard_flow(self, agent_tool, mock_connection_manager, mock_caller):
        """Test standard agent dispatch flow with consistent status messages."""
        # Save original import before patching
        original_import = builtins.__import__

        # Mock the dynamic import
        with patch.object(builtins, "__import__") as mock_import:
            mock_module = MagicMock()
            mock_module.MockAgent = MockAgent

            def mock_import_func(name, *args, **kwargs):
                if name == "test.module":
                    return mock_module
                return original_import(name, *args, **kwargs)

            mock_import.side_effect = mock_import_func

            # Configure mock agent with streaming response
            MockAgent.configure_streaming(
                [
                    {"content": "Processing...", "complete": False, "uuid": str(uuid.uuid4())},
                    {"content": "Done.", "complete": True, "uuid": str(uuid.uuid4())},
                ]
            )
            MockAgent.last_instance = None

            # Dispatch the agent
            result = await agent_tool._dispatch_agent(agent_class_import="test.module.MockAgent", task="Test task")

            # Verify result
            assert result == "Mock agent completed"
            assert MockAgent.last_instance is not None
            assert MockAgent.last_instance.init_kwargs["max_iterations"] is None

            # Verify start status was sent
            start_event_calls = [
                call
                for call in mock_connection_manager.broadcast_event.call_args_list
                if call[0][0].content.get("status") == "GENERATING"
            ]
            assert len(start_event_calls) == 1
            assert start_event_calls[0][0][0].content["message"] == "Starting mock-agent task"

            # Verify completion status was sent
            end_event_calls = [
                call
                for call in mock_connection_manager.broadcast_event.call_args_list
                if call[0][0].content.get("status") == "STOPPED"
            ]
            assert len(end_event_calls) == 1
            assert end_event_calls[0][0][0].content["message"] == "Completed mock-agent task"

            assert mock_caller.sub_agent_recorder.start_conversation.await_count == 1
            assert mock_caller.sub_agent_recorder.record_message.await_count >= 1
            assert mock_caller.sub_agent_recorder.complete_conversation.await_count == 1

            # Verify agent was created and cleaned up
            assert str(uuid.uuid4()) not in agent_tool.agents
            # Ensure protected files were forwarded to the sub-agent
            assert MockAgent.last_instance is not None
            assert MockAgent.last_instance.init_kwargs.get("protected_files") == mock_caller.protected_files
            assert (
                MockAgent.last_instance.init_kwargs.get("workspace_env_var_descriptions")
                == mock_caller.workspace_env_var_descriptions
            )
            MockAgent.configure_streaming([])

    async def test_dispatch_agent_inherits_parent_max_iterations(self, agent_tool, mock_caller):
        original_import = builtins.__import__
        mock_caller.max_iterations = 3

        with patch.object(builtins, "__import__") as mock_import:
            mock_module = MagicMock()
            mock_module.MockAgent = MockAgent

            def mock_import_func(name, *args, **kwargs):
                if name == "test.module":
                    return mock_module
                return original_import(name, *args, **kwargs)

            mock_import.side_effect = mock_import_func
            MockAgent.configure_streaming(
                [{"content": "Done.", "complete": True, "uuid": str(uuid.uuid4()), "type": "response"}]
            )
            MockAgent.last_instance = None

            await agent_tool._dispatch_agent(agent_class_import="test.module.MockAgent", task="Test task")

        assert MockAgent.last_instance is not None
        assert MockAgent.last_instance.init_kwargs["max_iterations"] == 3
        MockAgent.configure_streaming([])

    async def test_nested_dispatch_inherits_workflow_depth_and_grouping(
        self,
        agent_tool: AgentTool,
        mock_connection_manager: AsyncMock,
        mock_caller: Mock,
    ) -> None:
        original_import = builtins.__import__
        mock_caller.sub_agent = True
        mock_caller.sub_agent_context = {
            "agent_id": "parent-agent",
            "workflow_run_id": "workflow-run",
            "phase": "Verify",
            "depth": 1,
            "max_agent_depth": 2,
        }
        accounting = WorkflowRunAccounting(Budget(), agent_cap=2)
        parent_reservation = accounting.reserve_agent()
        mock_caller._workflow_accounting = accounting
        mock_caller._accounting_reservation = parent_reservation

        with patch.object(builtins, "__import__") as mock_import:
            mock_module = MagicMock()
            mock_module.MockAgent = MockAgent

            def mock_import_func(name: str, *args: Any, **kwargs: Any) -> Any:
                if name == "test.module":
                    return mock_module
                return original_import(name, *args, **kwargs)

            mock_import.side_effect = mock_import_func
            MockAgent.configure_streaming(
                [{"content": "Done.", "complete": True, "uuid": str(uuid.uuid4()), "type": "response"}]
            )
            MockAgent.last_instance = None

            await agent_tool._dispatch_agent(agent_class_import="test.module.MockAgent", task="Nested task")

        assert MockAgent.last_instance is not None
        context = MockAgent.last_instance.sub_agent_context
        assert context["workflow_run_id"] == "workflow-run"
        assert context["phase"] == "Verify"
        assert context["parent_agent_id"] == "parent-agent"
        assert context["depth"] == 2
        assert context["max_agent_depth"] == 2
        assert MockAgent.last_instance._workflow_accounting is accounting
        assert MockAgent.last_instance._accounting_reservation is not parent_reservation
        assert accounting.agent_count == 2

        start_event = next(
            call.args[0]
            for call in mock_connection_manager.broadcast_event.call_args_list
            if call.args[0].content.get("status") == "GENERATING"
        )
        assert start_event.sub_agent_info["depth"] == 2
        assert start_event.sub_agent_info["workflow_run_id"] == "workflow-run"
        MockAgent.configure_streaming([])

    async def test_nested_dispatch_rejects_cap_after_routing_but_before_conversation_or_events(
        self,
        agent_tool: AgentTool,
        mock_connection_manager: AsyncMock,
        mock_caller: Mock,
    ) -> None:
        accounting = WorkflowRunAccounting(Budget(), agent_cap=1)
        accounting.reserve_agent()
        mock_caller.sub_agent = True
        mock_caller.sub_agent_context = {
            "workflow_run_id": "workflow-run",
            "depth": 1,
            "max_agent_depth": 2,
        }
        mock_caller._workflow_accounting = accounting

        with patch.object(builtins, "__import__") as mock_import:
            with pytest.raises(Exception, match="lifetime agent cap"):
                await agent_tool._dispatch_agent(agent_class_import="test.module.MockAgent", task="Nested task")

        # The class/role must be known to validate routing before the reservation.
        mock_import.assert_called_once_with("test.module", fromlist=["MockAgent"])
        assert accounting.agent_count == 1
        assert mock_caller.sub_agent_recorder.start_conversation.await_count == 0
        assert not any(
            call.args[0].content.get("status") == "GENERATING"
            for call in mock_connection_manager.broadcast_event.call_args_list
        )

    @pytest.mark.parametrize(
        "context",
        [
            {"workflow_run_id": "run", "depth": True, "max_agent_depth": 2},
            {"workflow_run_id": "run", "depth": 1, "max_agent_depth": "2"},
            {"workflow_run_id": "run", "depth": 0, "max_agent_depth": 2},
            {"workflow_run_id": "run", "depth": 1, "max_agent_depth": 3},
            {"workflow_run_id": "run", "depth": 2, "max_agent_depth": 2},
        ],
    )
    async def test_nested_dispatch_rejects_invalid_or_over_depth_context_before_import(
        self,
        context: dict[str, Any],
        agent_tool: AgentTool,
        mock_caller: Mock,
    ) -> None:
        accounting = WorkflowRunAccounting(Budget(), agent_cap=3)
        mock_caller.sub_agent = True
        mock_caller.sub_agent_context = context
        mock_caller._workflow_accounting = accounting

        with patch.object(builtins, "__import__") as mock_import:
            with pytest.raises(RuntimeError, match="workflow"):
                await agent_tool._dispatch_agent(agent_class_import="test.module.MockAgent", task="Nested task")

        mock_import.assert_not_called()
        assert accounting.agent_count == 0

    async def test_dispatch_agent_uses_execution_id_for_sub_agent_conversation(
        self, agent_tool, mock_connection_manager, mock_caller
    ):
        """Sub-agent records must use internal execution IDs, not provider tool IDs."""
        original_import = builtins.__import__
        mock_caller.current_provider_tool_call_id = "dispatch_investigation_agent_0"
        mock_caller.current_tool_execution_id = "tool_exec_unique_123"
        mock_caller.current_tool_call_id = "tool_exec_unique_123"

        with patch.object(builtins, "__import__") as mock_import:
            mock_module = MagicMock()
            mock_module.MockAgent = MockAgent

            def mock_import_func(name, *args, **kwargs):
                if name == "test.module":
                    return mock_module
                return original_import(name, *args, **kwargs)

            mock_import.side_effect = mock_import_func
            MockAgent.configure_streaming(
                [{"content": "Done.", "complete": True, "uuid": str(uuid.uuid4()), "type": "response"}]
            )
            MockAgent.last_instance = None

            await agent_tool._dispatch_agent(agent_class_import="test.module.MockAgent", task="Test task")

        conversation_payload = mock_caller.sub_agent_recorder.start_conversation.call_args.args[0]
        assert conversation_payload["parent_tool_call_id"] == "tool_exec_unique_123"
        assert conversation_payload["parent_tool_call_id"] != mock_caller.current_provider_tool_call_id
        assert MockAgent.last_instance is not None
        assert MockAgent.last_instance.parent_tool_call_id == "tool_exec_unique_123"

        sub_agent_events = [
            call.args[0]
            for call in mock_connection_manager.broadcast_event.call_args_list
            if call.args[0].sub_agent_info
        ]
        assert sub_agent_events
        assert sub_agent_events[0].sub_agent_info["parent_tool_call_id"] == "tool_exec_unique_123"
        MockAgent.configure_streaming([])

    async def test_all_agents_get_consistent_status_messages(self, agent_tool):
        """Test that all agent types get consistent start/end/error messages."""
        # This is a meta-test to ensure our simplification goal is achieved

        # Test data for different agent types
        test_agents = [
            ("investigation", agent_tool.dispatch_investigation_agent, "Investigate code"),
            ("browser", agent_tool.dispatch_browser_agent, "Browse website"),
            ("coding", agent_tool.dispatch_coding_agent, "Write code"),
        ]

        for agent_type, dispatch_method, task in test_agents:
            # Mock the underlying dispatch to verify it gets called with consistent pattern
            with patch.object(agent_tool, "_dispatch_agent") as mock_dispatch:
                mock_dispatch.return_value = f"{agent_type} completed"

                await dispatch_method(task)

                # Verify the dispatch was called with the task
                assert mock_dispatch.called
                call_args = mock_dispatch.call_args
                assert call_args[1]["task"] == task

    async def test_dispatch_investigation_agent(self, agent_tool):
        """Test investigation agent dispatch."""
        with patch.object(agent_tool, "_dispatch_agent") as mock_dispatch:
            mock_dispatch.return_value = "Investigation completed"

            result = await agent_tool.dispatch_investigation_agent("Investigate this code")

            mock_dispatch.assert_called_once_with(
                agent_class_import="kolega_code.agent.investigationagent.InvestigationAgent",
                task="Investigate this code",
                model_override=None,
            )
            assert result == "Investigation completed"

    async def test_dispatch_browser_agent(self, agent_tool):
        """Test browser agent dispatch."""
        with patch.object(agent_tool, "_dispatch_agent") as mock_dispatch:
            mock_dispatch.return_value = "Browser task completed"

            result = await agent_tool.dispatch_browser_agent("Browse the web")

            mock_dispatch.assert_called_once_with(
                agent_class_import="kolega_code.agent.browseragent.BrowserAgent",
                task="Browse the web",
                model_override=None,
            )
            assert result == "Browser task completed"

    async def test_dispatch_coding_agent(self, agent_tool):
        """Test coding agent dispatch."""
        with patch.object(agent_tool, "_dispatch_agent") as mock_dispatch:
            mock_dispatch.return_value = "Coding completed"

            result = await agent_tool.dispatch_coding_agent("Write some code")

            mock_dispatch.assert_called_once_with(
                agent_class_import="kolega_code.agent.coder.CoderAgent",
                task="Write some code",
                model_override=None,
            )
            assert result == "Coding completed"

    async def test_atomic_override_is_applied_and_reported_without_mutating_parent(
        self,
        agent_tool: AgentTool,
        mock_connection_manager: AsyncMock,
    ) -> None:
        original_import = builtins.__import__
        parent_general = agent_tool.config.model_config_for_agent("investigation-agent")

        with patch.object(builtins, "__import__") as mock_import:
            mock_module = MagicMock()
            mock_module.InvestigationAgent = MockInvestigationAgent

            def mock_import_func(name: str, *args: Any, **kwargs: Any) -> Any:
                if name == "kolega_code.agent.investigationagent":
                    return mock_module
                return original_import(name, *args, **kwargs)

            mock_import.side_effect = mock_import_func
            MockInvestigationAgent.configure_streaming([])
            MockAgent.last_instance = None

            await agent_tool.dispatch_investigation_agent(
                "Investigate",
                {
                    "provider": "openai",
                    "model": "gpt-5.4",
                    "thinking_effort": "high",
                },
            )

        assert MockAgent.last_instance is not None
        child_config = MockAgent.last_instance.init_kwargs["config"]
        selected = child_config.model_config_for_agent("investigation-agent")
        assert selected.provider == ModelProvider.OPENAI
        assert selected.model == "gpt-5.4"
        assert selected.thinking_effort == "high"
        assert agent_tool.config.model_config_for_agent("investigation-agent") is parent_general

        start_event = next(
            call.args[0]
            for call in mock_connection_manager.broadcast_event.call_args_list
            if call.args[0].content.get("status") == "GENERATING"
        )
        assert start_event.sub_agent_info["requested_routing"] == {
            "provider": "openai",
            "model": "gpt-5.4",
            "thinking_effort": "high",
        }
        assert start_event.sub_agent_info["effective_routing"] == {
            "provider": "openai",
            "model": "gpt-5.4",
            "thinking_effort": "high",
        }
        MockInvestigationAgent.configure_streaming([])

    async def test_override_is_scoped_to_direct_worker_for_same_role_child(
        self,
        agent_tool: AgentTool,
        mock_connection_manager: AsyncMock,
    ) -> None:
        """A depth-2 same-role child starts from the inherited role default."""
        original_import = builtins.__import__
        mock_module = MagicMock()
        mock_module.InvestigationAgent = MockInvestigationAgent

        def mock_import_func(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "kolega_code.agent.investigationagent":
                return mock_module
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import_func):
            MockInvestigationAgent.configure_streaming([])
            await agent_tool.dispatch_investigation_agent(
                "Direct worker",
                {"provider": "openai", "model": "gpt-5.4", "thinking_effort": "high"},
            )
            direct_worker = MockAgent.last_instance
            assert direct_worker is not None
            assert getattr(direct_worker, "_subagent_dispatch_config") is agent_tool.config

            direct_config = direct_worker.init_kwargs["config"]
            assert direct_config.model_config_for_agent("investigation-agent").model == "gpt-5.4"

            nested_tool = AgentTool(
                project_path=agent_tool.project_path,
                workspace_id=agent_tool.workspace_id,
                thread_id=agent_tool.thread_id,
                connection_manager=mock_connection_manager,
                config=direct_config,
                caller=direct_worker,
            )
            await nested_tool.dispatch_investigation_agent("Same-role child")

        nested_worker = MockAgent.last_instance
        assert nested_worker is not None
        nested_route = nested_worker.init_kwargs["config"].model_config_for_agent("investigation-agent")
        assert nested_route.provider == ModelProvider.ANTHROPIC
        assert nested_route.model == "test-model"

    async def test_invalid_override_precedes_conversation_events_and_workflow_reservation(
        self,
        agent_tool: AgentTool,
        mock_connection_manager: AsyncMock,
        mock_caller: Mock,
    ) -> None:
        original_import = builtins.__import__
        accounting = WorkflowRunAccounting(Budget(), agent_cap=2)
        mock_caller.sub_agent = True
        mock_caller.sub_agent_context = {
            "workflow_run_id": "workflow-run",
            "depth": 1,
            "max_agent_depth": 2,
        }
        mock_caller._workflow_accounting = accounting

        with patch.object(builtins, "__import__") as mock_import:
            mock_module = MagicMock()
            mock_module.InvestigationAgent = MockInvestigationAgent

            def mock_import_func(name: str, *args: Any, **kwargs: Any) -> Any:
                if name == "kolega_code.agent.investigationagent":
                    return mock_module
                return original_import(name, *args, **kwargs)

            mock_import.side_effect = mock_import_func
            with pytest.raises(ValueError, match="missing required field.*thinking_effort"):
                await agent_tool.dispatch_investigation_agent(
                    "Investigate",
                    {"provider": "openai", "model": "gpt-5.4"},
                )

        assert accounting.agent_count == 0
        assert mock_caller.sub_agent_recorder.start_conversation.await_count == 0
        assert mock_connection_manager.broadcast_event.await_count == 0

    async def test_browser_override_rejects_nonvision_route_before_lifecycle(
        self,
        agent_tool: AgentTool,
        mock_connection_manager: AsyncMock,
        mock_caller: Mock,
    ) -> None:
        agent_tool.config = agent_tool.config.model_copy(update={"deepseek_api_key": "test-key"})

        with pytest.raises(ValueError, match="vision-capable"):
            await agent_tool.dispatch_browser_agent(
                "Browse",
                {
                    "provider": "deepseek",
                    "model": "deepseek-v4-pro",
                    "thinking_effort": "high",
                },
            )

        assert mock_caller.sub_agent_recorder.start_conversation.await_count == 0
        assert mock_connection_manager.broadcast_event.await_count == 0

    async def test_workflow_markdown_renders_supplied_routing_metadata(self, agent_tool: AgentTool, tmp_path) -> None:
        markdown = tmp_path / "agent.md"
        agent_tool._write_workflow_agent_markdown(
            {"markdown": markdown},
            metadata={
                "actual_agent_type": "InvestigationAgent",
                "requested_routing": {"provider": "openai", "model": "gpt-5.4", "effort": "high"},
                "effective_routing": {"provider": "openai", "model": "gpt-5.4", "effort": "high"},
            },
            prompt="Investigate",
            status="completed",
        )

        rendered = markdown.read_text(encoding="utf-8")
        assert "- Actual agent type: InvestigationAgent" in rendered
        assert '- Requested routing: {"effort": "high", "model": "gpt-5.4", "provider": "openai"}' in rendered
        assert '- Effective routing: {"effort": "high", "model": "gpt-5.4", "provider": "openai"}' in rendered

    async def test_agent_name_consistency(self, agent_tool):
        """Test that all agent names follow kebab-case convention."""
        # This test verifies our agent name assumptions
        test_agents = [
            ("kolega_code.agent.investigationagent.InvestigationAgent", "investigation-agent"),
            ("kolega_code.agent.browseragent.BrowserAgent", "browser-agent"),
            ("kolega_code.agent.coder.CoderAgent", "coding-agent"),
        ]

        original_import = builtins.__import__

        for class_import, expected_name in test_agents:
            with patch.object(builtins, "__import__") as mock_import:
                # Create appropriate mock class based on the import
                if "InvestigationAgent" in class_import:
                    mock_class = MockInvestigationAgent
                elif "BrowserAgent" in class_import:
                    mock_class = MockBrowserAgent
                elif "CoderAgent" in class_import:
                    mock_class = MockCodingAgent
                else:
                    mock_class = MockAgent

                mock_module = MagicMock()
                setattr(mock_module, class_import.split(".")[-1], mock_class)

                def mock_import_func(name, *args, **kwargs):
                    module_base = class_import.rsplit(".", 1)[0]
                    if name == module_base:
                        return mock_module
                    return original_import(name, *args, **kwargs)

                mock_import.side_effect = mock_import_func

                # Just verify we can get the agent name
                module_path, class_name = class_import.rsplit(".", 1)
                agent_class = getattr(mock_module, class_name)
                assert agent_class.agent_name == expected_name

    async def _create_mock_stream(self, messages):
        """Helper to create async generator for mock message stream."""
        for msg in messages:
            yield msg
