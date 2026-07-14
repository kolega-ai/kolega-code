"""Credential-gated OpenAI smoke for the native freeform edit protocol."""

import os
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from kolega_code.agent.tool_backend.codex_patch import CODEX_APPLY_PATCH_GRAMMAR
from kolega_code.agent.tool_backend.edit_tool import EditTool
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider
from kolega_code.llm.client import LLMClient
from kolega_code.llm.models import Message, MessageHistory, TextBlock, ToolCall, ToolDefinition, ToolResult


pytestmark = [pytest.mark.slow, pytest.mark.integration]

_API_KEY_ENV = "OPENAI_API_KEY"
_MODEL = "gpt-5.5"


def _enabled_key() -> str:
    value = os.getenv(_API_KEY_ENV)
    if not value:
        pytest.skip(f"{_API_KEY_ENV} not set")
    return value


def _definition() -> ToolDefinition:
    return ToolDefinition(
        name="apply_patch",
        description="Use apply_patch to edit files. Supply the raw patch, not JSON.",
        parameters=[],
        input_kind="freeform",
        freeform_format={
            "type": "grammar",
            "syntax": "lark",
            "definition": CODEX_APPLY_PATCH_GRAMMAR,
        },
    )


def _normalize(call: ToolCall) -> ToolCall:
    if isinstance(call.input, dict):
        call.input = call.input["input"]
    call.input_kind = "freeform"
    return call


async def _execute(tmp_path: Path, call: ToolCall, api_key: str) -> ToolResult:
    model = ModelConfig(provider=ModelProvider.OPENAI, model=_MODEL)
    config = AgentConfig(
        openai_api_key=api_key,
        long_context_config=model,
        fast_config=model,
        thinking_config=model,
    )
    caller = Mock(agent_name="live-smoke", agent_mode="normal", current_tool_execution_id="live-call")
    tool = EditTool(tmp_path, "workspace", "thread", AsyncMock(), config, caller)
    output = await tool.apply_patch(str(call.input))
    return ToolResult(
        tool_use_id=call.id,
        name=call.name,
        content=output,
        is_error=False,
        input_kind="freeform",
    )


@pytest.mark.asyncio
async def test_live_openai_generates_executes_and_consumes_patch(tmp_path: Path) -> None:
    api_key = _enabled_key()
    client = LLMClient("openai", api_key)
    system = Message("system", [TextBlock("Use the supplied edit tool exactly when requested. Be concise.")])
    user = Message(
        "user",
        [TextBlock("Call apply_patch to create live-smoke.txt containing exactly `provider-ok` and a newline.")],
    )
    response = await client.generate(
        messages=MessageHistory([user]),
        system=system,
        model=_MODEL,
        tools=[_definition()],
        max_completion_tokens=1024,
        temperature=0,
    )
    assert response.tool_calls
    call = _normalize(response.tool_calls[0])
    result = await _execute(tmp_path, call, api_key)
    assert (tmp_path / "live-smoke.txt").read_text() == "provider-ok\n"

    follow_up = Message("user", [result, TextBlock("Reply with exactly live-edit-ok.")])
    continued = await client.generate(
        messages=MessageHistory([user, response, follow_up]),
        system=system,
        model=_MODEL,
        tools=[],
        max_completion_tokens=128,
        temperature=0,
    )
    assert "live-edit-ok" in continued.get_text_content().lower()
