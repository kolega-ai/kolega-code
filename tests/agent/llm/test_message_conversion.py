# ruff: noqa: F401,F811,E402
import asyncio
import os
from unittest.mock import AsyncMock, patch

import pytest

from kolega_code.llm.client import (
    GenerationParams,
    LLMClient,
    TokenCount,
)
from kolega_code.llm.models import (
    Message,
    MessageChunk,
    MessageHistory,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResult,
)
from kolega_code.llm.providers.anthropic import AnthropicProvider, AnthropicStreamWrapper
from kolega_code.llm.providers.openai import OpenAIProvider

TEST_MESSAGES = MessageHistory([Message("user", [TextBlock("Hello, how are you?")])])
TEST_SYSTEM = Message("system", [TextBlock("You are a helpful assistant.")])


def test_anthropic_synthetic_thinking_chunk_conversion():
    class Chunk:
        type = "thinking"
        thinking = "working through the problem"

    chunk = MessageChunk.from_anthropic(Chunk())

    assert chunk.type == "thinking"
    assert chunk.thinking == "working through the problem"
def test_anthropic_raw_thinking_delta_chunk_is_ignored():
    class Delta:
        type = "thinking_delta"
        thinking = "working through the problem"

    class Chunk:
        type = "content_block_delta"
        delta = Delta()

    chunk = MessageChunk.from_anthropic(Chunk())

    assert chunk.type == "ignore"
def test_anthropic_thinking_blocks_round_trip_to_anthropic_shape():
    class ThinkingContent:
        type = "thinking"
        thinking = "provider reasoning"
        signature = "provider-signature"

    class RedactedThinkingContent:
        type = "redacted_thinking"
        data = "encrypted-redacted-reasoning"

    class AnthropicMessage:
        role = "assistant"
        content = [
            ThinkingContent(),
            RedactedThinkingContent(),
            type("TextContent", (), {"type": "text", "text": "done"})(),
        ]

    message = Message.from_anthropic(AnthropicMessage())

    assert isinstance(message.content[0], ThinkingBlock)
    assert message.content[0].thinking == "provider reasoning"
    assert message.content[0].signature == "provider-signature"
    assert isinstance(message.content[1], RedactedThinkingBlock)
    assert message.content[1].data == "encrypted-redacted-reasoning"
    assert message.to_anthropic()["content"][:2] == [
        {"type": "thinking", "thinking": "provider reasoning", "signature": "provider-signature"},
        {"type": "redacted_thinking", "data": "encrypted-redacted-reasoning"},
    ]
def test_tool_call_execution_id_is_internal_and_provider_id_is_preserved():
    first = ToolCall(id="dispatch_investigation_agent_0", name="dispatch_investigation_agent", input={})
    second = ToolCall(id="dispatch_investigation_agent_0", name="dispatch_investigation_agent", input={})

    assert first.id == second.id == "dispatch_investigation_agent_0"
    assert first.execution_id != second.execution_id
    assert first.to_anthropic()["id"] == "dispatch_investigation_agent_0"
    assert first.to_openai()["id"] == "dispatch_investigation_agent_0"
    tool_result = ToolResult(
        tool_use_id=first.id,
        content="done",
        name="dispatch_investigation_agent",
        is_error=False,
        execution_id=first.execution_id,
    )
    assert tool_result.tool_use_id == "dispatch_investigation_agent_0"
    assert tool_result.execution_id == first.execution_id
    assert tool_result.to_anthropic()["tool_use_id"] == "dispatch_investigation_agent_0"
    assert "execution_id" not in tool_result.to_anthropic()
    assert ToolResult.from_dict(tool_result.to_dict()).execution_id == first.execution_id

    restored = ToolCall.from_dict(first.to_dict())
    assert restored.id == first.id
    assert restored.execution_id == first.execution_id
def test_local_anthropic_token_counting_includes_tool_result_content():
    provider = AnthropicProvider(api_key="test_key", provider_name="moonshot")
    large_tool_output = "unique_token " * 20_000
    messages = MessageHistory(
        [
            Message(
                role="user",
                content=[
                    ToolResult(
                        tool_use_id="tool_1",
                        content=large_tool_output,
                        name="read_entire_file",
                        is_error=False,
                    )
                ],
            )
        ]
    )

    token_count = provider._count_tokens_local(messages)

    assert token_count.input_tokens > 20_000
@pytest.mark.asyncio
async def test_anthropic_stream_tool_use_start_execution_id_matches_final_tool_call():
    class ContentBlock:
        type = "tool_use"
        id = "toolu_create_file"
        name = "create_file"
        input = {"path": "hello.txt", "content": "hello"}

    class StartChunk:
        type = "content_block_start"
        index = 0
        content_block = ContentBlock()

    class FinalMessage:
        role = "assistant"
        stop_reason = "tool_use"
        content = [ContentBlock()]

    class FakeGenerator:
        def __init__(self):
            self.chunks = iter([StartChunk()])

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.chunks)
            except StopIteration:
                raise StopAsyncIteration

        async def get_final_message(self):
            return FinalMessage()

    class FakeAnthropicStream:
        async def __aenter__(self):
            return FakeGenerator()

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return False

    async with AnthropicStreamWrapper(FakeAnthropicStream()) as stream:
        start_chunk = await stream.__anext__()
        final_message = await stream.get_final_message()

    execution_id = start_chunk.tool_call_delta["execution_id"]

    assert start_chunk.tool_call_delta["id"] == "toolu_create_file"
    assert execution_id.startswith("tool_exec_")
    assert final_message.tool_calls[0].id == "toolu_create_file"
    assert final_message.tool_calls[0].execution_id == execution_id
    assert final_message.content[0].execution_id == execution_id
