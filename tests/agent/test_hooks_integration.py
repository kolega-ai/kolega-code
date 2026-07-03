"""Integration tests: lifecycle hooks fired through BaseAgent's real loop and tool path."""

import os
import uuid
from unittest.mock import AsyncMock, Mock

import pytest

from kolega_code.agent.baseagent import BaseAgent
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.events import AgentConnectionManager
from kolega_code.hooks import HookConfig, HookDispatcher, HookEvent, HookMatcher, HookOutcome, HookSpec
from kolega_code.llm.models import Message, TextBlock, ToolCall, ToolDefinition
from kolega_code.llm.providers.models import TokenCount
from kolega_code.tools import Tool, ToolRegistry


@pytest.fixture
def agent_config():
    return AgentConfig(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "test_key"),
        long_context_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC, model="claude-haiku-4-5-20251001", rate_limits=RateLimitConfig()
        ),
        fast_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC, model="claude-haiku-4-5-20251001", rate_limits=RateLimitConfig()
        ),
        thinking_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model="claude-haiku-4-5-20251001",
            rate_limits=RateLimitConfig(),
            thinking_effort="medium",
        ),
    )


# --- module-level python hooks referenced from configs ---------------------- #

_STOP_BLOCKS_REMAINING = [0]


def deny_hook(event):
    return HookOutcome.deny("blocked by test")


def modify_input_hook(event):
    return HookOutcome(updated_input={"command": "echo safe"})


def modify_output_hook(event):
    return HookOutcome(updated_output="REDACTED")


def deny_boom_hook(event):
    if event.payload.get("tool_input", {}).get("task") == "boom":
        return HookOutcome.deny("boom is not allowed")
    return HookOutcome.empty()


def deny_prompt_hook(event):
    return HookOutcome.deny("prompt rejected")


def stop_until_flag_hook(event):
    if _STOP_BLOCKS_REMAINING[0] > 0:
        _STOP_BLOCKS_REMAINING[0] -= 1
        return HookOutcome.deny("keep going")
    return HookOutcome.empty()


def _dispatcher(event: HookEvent, func: str, matcher: str = "*") -> HookDispatcher:
    spec = HookSpec(type="python", timeout=5, scope="global", callable=f"{__name__}:{func}")
    return HookDispatcher(HookConfig(entries={event: [(HookMatcher(matcher), [spec])]}))


def _make_agent(tmp_path, agent_config, dispatcher):
    agent = BaseAgent(
        project_path=tmp_path,
        workspace_id="test_workspace",
        thread_id=str(uuid.uuid4()),
        connection_manager=AsyncMock(spec=AgentConnectionManager),
        config=agent_config,
        hook_dispatcher=dispatcher,
    )
    agent.send_chat_message = AsyncMock()
    agent.log_info = AsyncMock()
    agent.log_warning = AsyncMock()
    agent.log_error = AsyncMock()
    return agent


def _tools(handler, name="do_thing", parallel_safe=False):
    class Tools:
        def registry(self):
            return ToolRegistry(
                [
                    Tool(
                        name=name,
                        definition=ToolDefinition(name=name, description="", parameters=[]),
                        handler=handler,
                        parallel_safe=parallel_safe,
                    )
                ]
            )

    return Tools()


def _call(name="do_thing", index=1, **inputs):
    return ToolCall(id=f"{name}_{index}", name=name, input=inputs, execution_id=f"exec_{index}")


# --- PreToolUse -------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_pre_tool_use_deny_blocks_and_skips_handler(tmp_path, agent_config, monkeypatch):
    handler = AsyncMock(return_value="ran")
    agent = _make_agent(tmp_path, agent_config, _dispatcher(HookEvent.PRE_TOOL_USE, "deny_hook"))
    monkeypatch.setattr(agent, "tool_collection", _tools(handler))

    result = await agent.execute_single_tool(_call(command="rm -rf /"))

    assert result.is_error is True
    assert "Permission denied for do_thing" in result.content
    assert "blocked by test" in result.content
    handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_pre_tool_use_updated_input_is_applied(tmp_path, agent_config, monkeypatch):
    handler = AsyncMock(return_value="ran")
    agent = _make_agent(tmp_path, agent_config, _dispatcher(HookEvent.PRE_TOOL_USE, "modify_input_hook"))
    monkeypatch.setattr(agent, "tool_collection", _tools(handler))

    result = await agent.execute_single_tool(_call(command="rm -rf /"))

    assert result.is_error is False
    handler.assert_awaited_once_with(command="echo safe")


# --- PostToolUse ------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_post_tool_use_updated_output_replaces_content(tmp_path, agent_config, monkeypatch):
    handler = AsyncMock(return_value="secret value")
    agent = _make_agent(tmp_path, agent_config, _dispatcher(HookEvent.POST_TOOL_USE, "modify_output_hook"))
    monkeypatch.setattr(agent, "tool_collection", _tools(handler))

    result = await agent.execute_single_tool(_call())

    assert result.is_error is False
    assert result.content == "REDACTED"


# --- concurrency: deny exactly one tool in a parallel batch ------------------ #


@pytest.mark.asyncio
async def test_parallel_batch_denies_only_matching_tool(tmp_path, agent_config, monkeypatch):
    handler = AsyncMock(return_value="ran")
    agent = _make_agent(tmp_path, agent_config, _dispatcher(HookEvent.PRE_TOOL_USE, "deny_boom_hook"))
    # search_codebase is parallel-safe, so the batch runs via asyncio.gather.
    monkeypatch.setattr(agent, "tool_collection", _tools(handler, name="search_codebase", parallel_safe=True))

    results = await agent.process_tool_calls(
        [_call(name="search_codebase", index=1, task="boom"), _call(name="search_codebase", index=2, task="ok")]
    )

    assert results[0].is_error is True and "boom is not allowed" in results[0].content
    assert results[1].is_error is False
    # The non-denied call still reached the handler exactly once.
    handler.assert_awaited_once_with(task="ok")


# --- UserPromptSubmit -------------------------------------------------------- #


@pytest.mark.asyncio
async def test_user_prompt_submit_block_ends_turn_without_llm_call(tmp_path, agent_config):
    agent = _make_agent(tmp_path, agent_config, _dispatcher(HookEvent.USER_PROMPT_SUBMIT, "deny_prompt_hook"))
    agent.llm = Mock()

    chunks = [chunk async for chunk in agent.process_message_stream("do the thing")]

    assert len(chunks) == 1
    assert chunks[0]["content"] == "prompt rejected"
    assert chunks[0]["complete"] is True
    assert agent.history == []
    agent.llm.stream.assert_not_called()


# --- Stop keep-working ------------------------------------------------------- #


class _EndTurnStream:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def get_final_message(self):
        return Message("assistant", [TextBlock("done")], stop_reason="end_turn")


@pytest.mark.asyncio
async def test_stop_hook_keeps_working_then_completes(tmp_path, agent_config):
    _STOP_BLOCKS_REMAINING[0] = 1  # block the first stop, allow the second
    agent = _make_agent(tmp_path, agent_config, _dispatcher(HookEvent.STOP, "stop_until_flag_hook"))
    agent.system_prompt = Message("system", [TextBlock("test agent")])  # set by subclasses in real use
    agent.tool_collection = Mock()
    agent.tool_collection.get_tool_list = Mock(return_value=[])
    agent.count_current_context = AsyncMock(return_value=TokenCount(input_tokens=10))
    agent.llm = Mock()
    agent.llm.stream = AsyncMock(return_value=_EndTurnStream())

    chunks = [chunk async for chunk in agent.process_message_stream("hello")]

    # The Stop hook forced one extra loop, so the LLM was streamed twice.
    assert agent.llm.stream.await_count == 2
    assert chunks[-1]["complete"] is True
    # The keep-working reason was appended to history as a user message.
    assert any("keep going" in m.get_text_content() for m in agent.history if m.role == "user")
