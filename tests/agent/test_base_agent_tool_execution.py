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


class TestBaseAgent:
    @pytest.mark.asyncio
    async def test_execute_single_tool_uses_execution_id_for_app_events_and_provider_id_for_result(self, base_agent):
        class TestTools:
            def get_tool_list(self):
                return [SimpleNamespace(name="dispatch_investigation_agent")]

            def registry(self):
                from kolega_code.agent.tools import ToolCollection
                from kolega_code.llm.models import ToolDefinition
                from kolega_code.tools import Tool, ToolRegistry

                parallel = set(ToolCollection.read_only_tools) | set(ToolCollection.agent_dispatch_tools)
                registry = ToolRegistry()
                for spec in self.get_tool_list():
                    registry.add(
                        Tool(
                            name=spec.name,
                            definition=ToolDefinition(name=spec.name, description="", parameters=[]),
                            handler=getattr(self, spec.name),
                            parallel_safe=spec.name in parallel,
                        )
                    )
                return registry

            async def dispatch_investigation_agent(self, **_inputs):
                return "investigation complete"

        tool_call = ToolCall(
            id="dispatch_investigation_agent_0",
            name="dispatch_investigation_agent",
            input={"task": "check this"},
            execution_id="tool_exec_unique_123",
        )
        base_agent.tool_collection = TestTools()
        base_agent.send_chat_message = AsyncMock()
        base_agent.log_info = AsyncMock()

        result = await base_agent.execute_single_tool(tool_call)

        assert result.tool_use_id == "dispatch_investigation_agent_0"
        assert result.execution_id == "tool_exec_unique_123"
        assert base_agent.send_chat_message.call_args_list[0].kwargs["tool_call_id"] == "tool_exec_unique_123"
        assert base_agent.send_chat_message.call_args_list[1].kwargs["tool_call_id"] == "tool_exec_unique_123"
        assert base_agent.current_tool_call_id is None
        assert base_agent.current_tool_execution_id is None
        assert base_agent.current_provider_tool_call_id is None
