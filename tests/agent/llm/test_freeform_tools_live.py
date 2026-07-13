"""Credential- and opt-in-gated live provider smokes for the edit protocol."""

import os
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from kolega_code.agent.conversation import adapt_history_for_provider
from kolega_code.agent.tool_backend.codex_patch import CODEX_APPLY_PATCH_GRAMMAR
from kolega_code.agent.tool_backend.edit_tool import EditTool
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider
from kolega_code.llm.client import LLMClient
from kolega_code.llm.models import Message, MessageHistory, TextBlock, ToolCall, ToolDefinition, ToolResult


pytestmark = [pytest.mark.slow, pytest.mark.integration]

_PROVIDERS = {
    "openai": ("OPENAI_API_KEY", "gpt-5.5"),
    "anthropic": ("ANTHROPIC_API_KEY", "claude-haiku-4-5-20251001"),
    "google": ("GOOGLE_API_KEY", "gemini-3.1-pro-preview"),
}


def _enabled_key(provider: str) -> str:
    if os.getenv("KOLEGA_RUN_LIVE_EDIT_TOOL_TESTS") != "1":
        pytest.skip("Set KOLEGA_RUN_LIVE_EDIT_TOOL_TESTS=1 to run live edit-tool smokes.")
    env_name, _ = _PROVIDERS[provider]
    value = os.getenv(env_name)
    if not value:
        pytest.skip(f"{env_name} not set")
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


async def _execute(tmp_path: Path, call: ToolCall, provider: str, api_key: str) -> ToolResult:
    model = ModelConfig(provider=ModelProvider(provider), model=_PROVIDERS[provider][1])
    config = AgentConfig(
        openai_api_key=api_key if provider == "openai" else None,
        anthropic_api_key=api_key if provider == "anthropic" else None,
        google_api_key=api_key if provider == "google" else None,
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


@pytest.mark.parametrize("provider", ["openai", "anthropic", "google"])
@pytest.mark.asyncio
async def test_live_provider_generates_executes_and_consumes_patch(provider: str, tmp_path: Path) -> None:
    api_key = _enabled_key(provider)
    _, model = _PROVIDERS[provider]
    client = LLMClient(provider, api_key)
    system = Message("system", [TextBlock("Use the supplied edit tool exactly when requested. Be concise.")])
    user = Message(
        "user",
        [TextBlock("Call apply_patch to create live-smoke.txt containing exactly `provider-ok` and a newline.")],
    )
    response = await client.generate(
        messages=MessageHistory([user]),
        system=system,
        model=model,
        tools=[_definition()],
        max_completion_tokens=1024,
        temperature=0,
    )
    assert response.tool_calls
    call = _normalize(response.tool_calls[0])
    result = await _execute(tmp_path, call, provider, api_key)
    assert (tmp_path / "live-smoke.txt").read_text() == "provider-ok\n"

    follow_up = Message("user", [result, TextBlock("Reply with exactly live-edit-ok.")])
    continued = await client.generate(
        messages=MessageHistory([user, response, follow_up]),
        system=system,
        model=model,
        tools=[],
        max_completion_tokens=128,
        temperature=0,
    )
    assert "live-edit-ok" in continued.get_text_content().lower()


@pytest.mark.asyncio
async def test_live_openai_to_anthropic_cross_provider_continuation(tmp_path: Path) -> None:
    openai_key = _enabled_key("openai")
    anthropic_key = _enabled_key("anthropic")
    definition = _definition()
    system = Message("system", [TextBlock("Use the supplied edit tool exactly when requested. Be concise.")])
    user = Message("user", [TextBlock("Call apply_patch to create switched.txt containing `switched`.")])
    openai = LLMClient("openai", openai_key)
    response = await openai.generate(
        messages=MessageHistory([user]),
        system=system,
        model=_PROVIDERS["openai"][1],
        tools=[definition],
        max_completion_tokens=1024,
        temperature=0,
    )
    call = _normalize(response.tool_calls[0])
    result = await _execute(tmp_path, call, "openai", openai_key)
    stored = [user, response, Message("user", [result]), Message("user", [TextBlock("Reply switched-ok.")])]
    adapted = adapt_history_for_provider(
        stored,
        target_provider="anthropic",
        target_model=_PROVIDERS["anthropic"][1],
        supports_vision=True,
        target_edit_protocol="codex_apply_patch",
    )

    anthropic = LLMClient("anthropic", anthropic_key)
    continued = await anthropic.generate(
        messages=MessageHistory(adapted),
        system=system,
        model=_PROVIDERS["anthropic"][1],
        tools=[],
        max_completion_tokens=128,
        temperature=0,
    )
    assert "switched-ok" in continued.get_text_content().lower()
