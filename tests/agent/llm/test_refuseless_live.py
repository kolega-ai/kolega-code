from __future__ import annotations

import inspect
import os
from collections.abc import AsyncIterator, Callable
from copy import deepcopy
from pathlib import Path

import pytest
import pytest_asyncio
from dotenv import dotenv_values

from kolega_code.llm.client import LLMClient
from kolega_code.llm.models import (
    Message,
    MessageHistory,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)
from kolega_code.llm.providers.openai import OpenAIProvider

pytestmark = [pytest.mark.integration, pytest.mark.live_api]

MODELS = (
    "glm-5.3-abliterated",
    "glm-5.3-flash-abliterated",
    "deepseek-v4.1-flash-abliterated",
    "deepseek-v4.1-flash-fast",
    "minimax-m3.1-flash",
)


def _api_key() -> str:
    if os.environ.get("KOLEGA_TEST_LIVE_API") != "1":
        pytest.skip("Set KOLEGA_TEST_LIVE_API=1 to run Refuseless live acceptance.")
    key = os.environ.get("REFUSELESS_API_KEY")
    if not key:
        key = dotenv_values(Path(__file__).resolve().parents[3] / ".env").get("REFUSELESS_API_KEY")
    if not key:
        pytest.fail("Explicit Refuseless live acceptance requires REFUSELESS_API_KEY.")
    return key


@pytest_asyncio.fixture
async def client_factory() -> AsyncIterator[Callable[[str], LLMClient]]:
    clients: list[LLMClient] = []

    def create(model: str) -> LLMClient:
        client = LLMClient(provider="refuseless", api_key=_api_key(), model=model, max_retries=0)
        clients.append(client)
        return client

    try:
        yield create
    finally:
        for client in clients:
            assert isinstance(client.provider, OpenAIProvider)
            await client.provider.async_client.close()


def _usage(message: Message) -> None:
    assert message.usage is not None
    assert message.usage.reported
    assert message.usage.provider == "refuseless"
    assert message.usage.input_tokens is not None and message.usage.input_tokens > 0
    assert message.usage.output_tokens is not None and message.usage.output_tokens > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
async def test_live_text_stream_and_tool_round_trip(model: str, client_factory: Callable[[str], LLMClient]) -> None:
    client = client_factory(model)
    history = MessageHistory([Message("user", [TextBlock("Reply with exactly OK.")])])
    response = await client.generate(messages=history, model=model, max_completion_tokens=2048)
    assert response.get_text_content().strip() == "OK"
    _usage(response)

    stream_context = client.stream(messages=history, model=model, max_completion_tokens=2048)
    if inspect.isawaitable(stream_context):
        stream_context = await stream_context
    async with stream_context as stream:
        async for _chunk in stream:
            pass
        response = await stream.get_final_message()
        final_snapshot = deepcopy(response.to_dict())
        repeated_response = await stream.get_final_message()
        assert repeated_response.to_dict() == final_snapshot
        assert response.to_dict() == final_snapshot
    assert response.get_text_content().strip() == "OK", response.to_dict()
    _usage(response)

    tool = ToolDefinition(
        name="add_numbers",
        description="Add two integers.",
        parameters=[
            ToolParameter(name="a", type="integer", description="First integer", required=True),
            ToolParameter(name="b", type="integer", description="Second integer", required=True),
        ],
    )
    history = MessageHistory(
        [Message("user", [TextBlock("Use add_numbers to add 37 and 58, then tell me the result.")])]
    )
    response = await client.generate(
        messages=history,
        model=model,
        tools=[tool],
        max_completion_tokens=2048,
        tool_choice={"type": "function", "function": {"name": "add_numbers"}},
    )
    _usage(response)
    calls = [block for block in response.content if isinstance(block, ToolCall)]
    assert len(calls) == 1
    call = calls[0]
    assert call.name == "add_numbers"
    assert call.input == {"a": 37, "b": 58}
    history.append(response)
    history.append(Message("user", [ToolResult(tool_use_id=call.id, content="95", name="add_numbers", is_error=False)]))
    final = await client.generate(messages=history, model=model, tools=[tool], max_completion_tokens=2048)
    assert "95" in final.get_text_content()
    _usage(final)
