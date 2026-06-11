"""Tests for GeneralAgent dispatch and the enriched sub-agent event contract."""

import asyncio
import os
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from dotenv import load_dotenv

from kolega_code.agent.baseagent import BaseAgent
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.events import AgentConnectionManager
from kolega_code.llm.models import ToolCall
from kolega_code.agent.prompt_provider import AgentType, PromptContext, PromptProvider
from kolega_code.agent.tool_backend.agent_tool import AgentTool

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


class StubGeneralAgent:
    """Stands in for GeneralAgent inside AgentTool._dispatch_agent."""

    agent_name = "general-agent"

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.parent_tool_call_id = None
        self.conversation_id = None
        self.sub_agent_context = None

    async def process_message_stream(self, task):
        yield {"type": "response", "content": "working on it", "complete": False, "uuid": "stream-1"}
        await asyncio.sleep(0)
        yield {"type": "response", "content": " done", "complete": True, "uuid": "stream-1"}

    def dump_message_history(self):
        return []

    async def recap_agent_outcome(self):
        return "final report"


class RecordingRecorder:
    def __init__(self):
        self.started = []

    async def start_conversation(self, payload):
        self.started.append(payload)
        return f"conv-{len(self.started)}"


def make_agent_tool(tmp_path, mock_connection_manager, agent_config, caller):
    return AgentTool(
        project_path=tmp_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=mock_connection_manager,
        config=agent_config,
        caller=caller,
    )


class TestDispatchGeneralAgent:
    @pytest.mark.asyncio
    async def test_dispatch_general_agent_happy_path(self, tmp_path, mock_connection_manager, agent_config, base_agent):
        agent_tool = make_agent_tool(tmp_path, mock_connection_manager, agent_config, base_agent)

        with patch("kolega_code.agent.generalagent.GeneralAgent", StubGeneralAgent):
            base_agent.current_tool_execution_id = "exec_123"
            result = await agent_tool.dispatch_general_agent("write a haiku")

        assert result == "final report"
        assert agent_tool.agents == {}  # cleaned up

        # Every broadcast event carries enriched sub_agent_info
        events = [call.args[0] for call in mock_connection_manager.broadcast_event.call_args_list]
        assert events, "expected broadcast events"
        infos = [e.sub_agent_info for e in events if e.sub_agent_info]
        assert infos, "expected sub_agent_info on dispatch events"
        for info in infos:
            assert info["agent_name"] == "general-agent"
            assert info["task"] == "write a haiku"
            assert info["agent_id"]
            assert info["parent_tool_call_id"] == "exec_123"

        # Lifecycle events: GENERATING then STOPPED, both with sub_agent_info
        statuses = [e.content.get("status") for e in events if e.content.get("status")]
        assert statuses == ["GENERATING", "STOPPED"]
        for event in events:
            if event.content.get("status"):
                assert event.sub_agent_info is not None

    @pytest.mark.asyncio
    async def test_dispatch_failure_emits_error_status(
        self, tmp_path, mock_connection_manager, agent_config, base_agent
    ):
        class ExplodingAgent(StubGeneralAgent):
            async def process_message_stream(self, task):
                raise RuntimeError("boom")
                yield  # pragma: no cover

        agent_tool = make_agent_tool(tmp_path, mock_connection_manager, agent_config, base_agent)

        with patch("kolega_code.agent.generalagent.GeneralAgent", ExplodingAgent):
            with pytest.raises(RuntimeError, match="boom"):
                await agent_tool.dispatch_general_agent("explode")

        events = [call.args[0] for call in mock_connection_manager.broadcast_event.call_args_list]
        statuses = [e.content.get("status") for e in events if e.content.get("status")]
        assert statuses == ["GENERATING", "ERROR"]
        assert agent_tool.agents == {}

    @pytest.mark.asyncio
    async def test_parallel_dispatches_record_distinct_parent_tool_call_ids(
        self, tmp_path, mock_connection_manager, agent_config, base_agent
    ):
        recorder = RecordingRecorder()
        base_agent.sub_agent_recorder = recorder
        agent_tool = make_agent_tool(tmp_path, mock_connection_manager, agent_config, base_agent)

        class Tools:
            def get_tool_list(self):
                return [SimpleNamespace(name="dispatch_general_agent")]

            async def dispatch_general_agent(self, task: str):
                return await agent_tool.dispatch_general_agent(task)

        base_agent.tool_collection = Tools()

        blocks = [
            ToolCall(
                id=f"call_{i}",
                name="dispatch_general_agent",
                input={"task": f"task {i}"},
                execution_id=f"exec_{i}",
            )
            for i in range(2)
        ]

        with patch("kolega_code.agent.generalagent.GeneralAgent", StubGeneralAgent):
            results = await base_agent.process_tool_calls(blocks)

        assert [r.content for r in results] == ["final report", "final report"]
        recorded_ids = {payload["parent_tool_call_id"] for payload in recorder.started}
        assert recorded_ids == {"exec_0", "exec_1"}
        recorded_tasks = {payload["initial_task"] for payload in recorder.started}
        assert recorded_tasks == {"task 0", "task 1"}

    @pytest.mark.asyncio
    async def test_dispatch_sets_sub_agent_context_on_agent(
        self, tmp_path, mock_connection_manager, agent_config, base_agent
    ):
        created = {}

        class CapturingAgent(StubGeneralAgent):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                created["agent"] = self

        agent_tool = make_agent_tool(tmp_path, mock_connection_manager, agent_config, base_agent)

        with patch("kolega_code.agent.generalagent.GeneralAgent", CapturingAgent):
            base_agent.current_tool_execution_id = "exec_ctx"
            await agent_tool.dispatch_general_agent("capture context")

        context = created["agent"].sub_agent_context
        assert context["agent_name"] == "general-agent"
        assert context["task"] == "capture context"
        assert context["parent_tool_call_id"] == "exec_ctx"
        assert context["agent_id"]


def test_general_agent_prompt_renders():
    prompt = PromptProvider().get_system_prompt(agent_type=AgentType.GENERAL, context=PromptContext())
    assert prompt
    assert "FINAL message" in prompt
    assert "general-purpose sub-agent" in prompt
