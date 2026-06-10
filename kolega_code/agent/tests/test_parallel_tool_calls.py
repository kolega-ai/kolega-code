"""Tests for parallel tool-call execution and task-local tool-call IDs."""

import asyncio
import os
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from dotenv import load_dotenv

from kolega_code.agent.baseagent import BaseAgent
from kolega_code.agent.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.agent.connection_manager import AgentConnectionManager
from kolega_code.agent.llm.models import ToolCall

# Load environment variables
load_dotenv()


@pytest.fixture
def agent_config():
    return AgentConfig(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "test_key"),
        openai_api_key="test-key",
        long_context_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model="claude-haiku-4-5-20251001",
            rate_limits=RateLimitConfig(),
        ),
        fast_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model="claude-haiku-4-5-20251001",
            rate_limits=RateLimitConfig(),
        ),
        thinking_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model="claude-haiku-4-5-20251001",
            rate_limits=RateLimitConfig(),
            thinking_tokens=1024,
        ),
    )


@pytest.fixture
def mock_connection_manager():
    return AsyncMock(spec=AgentConnectionManager)


@pytest.fixture
def base_agent(tmp_path, mock_connection_manager, agent_config):
    agent = BaseAgent(
        project_path=tmp_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
    )
    agent.send_chat_message = AsyncMock()
    agent.log_info = AsyncMock()
    agent.log_error = AsyncMock()
    return agent


def make_tool_call(name: str, index: int) -> ToolCall:
    return ToolCall(
        id=f"{name}_{index}",
        name=name,
        input={"task": f"task {index}"},
        execution_id=f"exec_{name}_{index}",
    )


class ConcurrencyTracker:
    """Tracks how many fake tool executions overlap."""

    def __init__(self):
        self.active = 0
        self.max_active = 0

    async def run(self, duration: float = 0.01):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(duration)
        self.active -= 1


class TestParallelToolCalls:
    @pytest.mark.asyncio
    async def test_dispatch_tools_run_concurrently(self, base_agent):
        first_started = asyncio.Event()
        second_started = asyncio.Event()

        class Tools:
            def __init__(self):
                self.calls = 0

            def get_tool_list(self):
                return [SimpleNamespace(name="dispatch_general_agent")]

            async def dispatch_general_agent(self, task: str):
                if "task 0" in task:
                    first_started.set()
                    # Deadlocks unless the second call runs concurrently
                    await asyncio.wait_for(second_started.wait(), timeout=2)
                    return "first done"
                second_started.set()
                await asyncio.wait_for(first_started.wait(), timeout=2)
                return "second done"

        base_agent.tool_collection = Tools()
        blocks = [make_tool_call("dispatch_general_agent", i) for i in range(2)]

        results = await asyncio.wait_for(base_agent.process_tool_calls(blocks), timeout=5)

        assert [r.content for r in results] == ["first done", "second done"]
        assert [r.tool_use_id for r in results] == ["dispatch_general_agent_0", "dispatch_general_agent_1"]
        assert not any(r.is_error for r in results)

    @pytest.mark.asyncio
    async def test_mixed_read_only_and_dispatch_parallelize(self, base_agent):
        tracker = ConcurrencyTracker()

        class Tools:
            def get_tool_list(self):
                return [
                    SimpleNamespace(name="dispatch_general_agent"),
                    SimpleNamespace(name="read_entire_file"),
                ]

            async def dispatch_general_agent(self, task: str):
                await tracker.run()
                return "dispatched"

            async def read_entire_file(self, task: str):
                await tracker.run()
                return "file contents"

        base_agent.tool_collection = Tools()
        blocks = [
            make_tool_call("read_entire_file", 0),
            make_tool_call("dispatch_general_agent", 1),
            make_tool_call("read_entire_file", 2),
        ]

        results = await base_agent.process_tool_calls(blocks)

        assert tracker.max_active > 1
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_write_tool_forces_sequential(self, base_agent):
        tracker = ConcurrencyTracker()

        class Tools:
            def get_tool_list(self):
                return [
                    SimpleNamespace(name="dispatch_general_agent"),
                    SimpleNamespace(name="create_file"),
                ]

            async def dispatch_general_agent(self, task: str):
                await tracker.run()
                return "dispatched"

            async def create_file(self, task: str):
                await tracker.run()
                return "created"

        base_agent.tool_collection = Tools()
        blocks = [
            make_tool_call("dispatch_general_agent", 0),
            make_tool_call("create_file", 1),
        ]

        results = await base_agent.process_tool_calls(blocks)

        assert tracker.max_active == 1
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_semaphore_caps_concurrency(self, base_agent):
        tracker = ConcurrencyTracker()
        total = BaseAgent.PARALLEL_TOOL_LIMIT + 4

        class Tools:
            def get_tool_list(self):
                return [SimpleNamespace(name="dispatch_general_agent")]

            async def dispatch_general_agent(self, task: str):
                await tracker.run()
                return "done"

        base_agent.tool_collection = Tools()
        blocks = [make_tool_call("dispatch_general_agent", i) for i in range(total)]

        results = await base_agent.process_tool_calls(blocks)

        assert len(results) == total
        assert tracker.max_active > 1
        assert tracker.max_active <= BaseAgent.PARALLEL_TOOL_LIMIT

    @pytest.mark.asyncio
    async def test_contextvar_isolation_under_gather(self, base_agent):
        captured: dict[str, str] = {}

        class Tools:
            def get_tool_list(self):
                return [SimpleNamespace(name="dispatch_general_agent")]

            async def dispatch_general_agent(self, task: str):
                # Yield so concurrent executions interleave before reading the ID
                await asyncio.sleep(0.01)
                captured[task] = base_agent.current_tool_execution_id
                return "done"

        base_agent.tool_collection = Tools()
        blocks = [make_tool_call("dispatch_general_agent", i) for i in range(3)]

        await base_agent.process_tool_calls(blocks)

        assert captured == {
            "task 0": "exec_dispatch_general_agent_0",
            "task 1": "exec_dispatch_general_agent_1",
            "task 2": "exec_dispatch_general_agent_2",
        }
        assert base_agent.current_tool_call_id is None
        assert base_agent.current_tool_execution_id is None
        assert base_agent.current_provider_tool_call_id is None

    @pytest.mark.asyncio
    async def test_nested_agent_does_not_clobber_parent_ids(
        self, tmp_path, mock_connection_manager, agent_config, base_agent
    ):
        nested_agent = BaseAgent(
            project_path=tmp_path,
            workspace_id="test_workspace",
            thread_id=str(uuid.uuid4()),
            connection_manager=mock_connection_manager,
            config=agent_config,
        )
        nested_agent.send_chat_message = AsyncMock()
        nested_agent.log_info = AsyncMock()

        class NestedTools:
            def get_tool_list(self):
                return [SimpleNamespace(name="read_entire_file")]

            async def read_entire_file(self, task: str):
                return "nested contents"

        nested_agent.tool_collection = NestedTools()
        observed = {}

        class ParentTools:
            def get_tool_list(self):
                return [SimpleNamespace(name="dispatch_general_agent")]

            async def dispatch_general_agent(self, task: str):
                # Simulate a sub-agent running its own tool within the same asyncio task
                await nested_agent.execute_single_tool(make_tool_call("read_entire_file", 99))
                observed["parent_id"] = base_agent.current_tool_execution_id
                return "done"

        base_agent.tool_collection = ParentTools()

        await base_agent.process_tool_calls([make_tool_call("dispatch_general_agent", 0)])

        assert observed["parent_id"] == "exec_dispatch_general_agent_0"
