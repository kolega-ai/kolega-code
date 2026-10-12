"""Credential-gated live check that DeepSeek enforces strict tool schemas. Run with:

./run_tests.sh --all tests/agent/llm/test_deepseek_strict_tools_live.py
"""

import os

import pytest

from kolega_code.llm.client import LLMClient
from kolega_code.llm.exceptions import LLMError
from kolega_code.llm.models import Message, MessageHistory, TextBlock, ToolCall, ToolDefinition

pytestmark = [pytest.mark.slow, pytest.mark.integration]

MODEL = "deepseek-flash"
PROMPT = "Call save_note with count 1 and the text hello."
CONFORMING = {
    "type": "object",
    "properties": {
        "count": {"type": "integer", "description": "How many times to save it."},
        "text": {"type": "string", "description": "The note text."},
    },
    "required": ["count", "text"],
    "additionalProperties": False,
}
# Breaks the strict rules: ``text`` is not required and extra properties are allowed.
NON_CONFORMING = {**CONFORMING, "required": ["count"], "additionalProperties": True}


@pytest.fixture
def client() -> LLMClient:
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        pytest.skip("DEEPSEEK_API_KEY not set")
    return LLMClient(provider="deepseek", api_key=key, model=MODEL)


def _tool(schema: dict, *, strict: bool) -> ToolDefinition:
    return ToolDefinition(
        name="save_note", description="Save a note.", parameters=[], input_schema=schema, strict=strict
    )


async def _generate(client: LLMClient, tool: ToolDefinition) -> Message:
    history = MessageHistory([Message("user", [TextBlock(PROMPT)])])
    response = await client.generate(
        messages=history, system=None, model=MODEL, tools=[tool], thinking="high", max_completion_tokens=2048
    )
    assert isinstance(response, Message)
    return response


@pytest.mark.asyncio
async def test_strict_tool_call_has_every_required_field(client: LLMClient) -> None:
    response = await _generate(client, _tool(CONFORMING, strict=True))
    calls = [block for block in response.content if isinstance(block, ToolCall)]
    assert len(calls) == 1
    call_input = calls[0].input
    assert isinstance(call_input, dict)
    assert set(call_input) == {"count", "text"}
    assert call_input["count"] == 1


@pytest.mark.asyncio
async def test_strict_tool_with_a_non_conforming_schema_is_rejected(client: LLMClient) -> None:
    with pytest.raises(LLMError) as raised:
        await _generate(client, _tool(NON_CONFORMING, strict=True))
    assert "Required properties must match" in str(raised.value)


@pytest.mark.asyncio
async def test_same_schema_without_strict_is_accepted(client: LLMClient) -> None:
    response = await _generate(client, _tool(NON_CONFORMING, strict=False))
    assert any(isinstance(block, ToolCall) for block in response.content)
