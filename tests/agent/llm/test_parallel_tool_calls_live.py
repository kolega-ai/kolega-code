"""Credential-gated live coverage for parallel Responses tool calls.

These tests prove that each Responses backend accepts ``parallel_tool_calls=true``
and can return two independent function calls in one assistant message. They are
skipped in CI and when the corresponding local credential is unavailable.

Run with:

    pytest -m integration tests/agent/llm/test_parallel_tool_calls_live.py -v
"""

from __future__ import annotations

import json
import os

import pytest

from kolega_code.auth import constants as chatgpt_constants
from kolega_code.auth.tokens import ChatGPTTokenManager, OAuthTokens
from kolega_code.cli.settings import SettingsStore
from kolega_code.llm.client import LLMClient
from kolega_code.llm.exceptions import LLMBillingError, LLMRateLimitError
from kolega_code.llm.models import Message, MessageHistory, TextBlock, ToolCall, ToolDefinition, ToolParameter

pytestmark = pytest.mark.integration

SKIP_IN_CI = bool(os.getenv("CI")) or bool(os.getenv("GITLAB_CI"))

PROVIDERS = [
    ("openai", "gpt-5.6-sol", "medium"),
    ("openai_chatgpt", chatgpt_constants.DEFAULT_MODEL, "medium"),
    ("deepseek", "deepseek-v4-flash", "high"),
]

TOOLS = [
    ToolDefinition(
        name="lookup_alpha",
        description="Record the alpha probe. Call this whenever the user requests the alpha probe.",
        parameters=[ToolParameter(name="token", type="string", description="Must be alpha", required=True)],
    ),
    ToolDefinition(
        name="lookup_beta",
        description="Record the beta probe. Call this whenever the user requests the beta probe.",
        parameters=[ToolParameter(name="token", type="string", description="Must be beta", required=True)],
    ),
]

SYSTEM = Message(
    role="system",
    content=[
        TextBlock(
            text=(
                "When the user requests independent tool calls, emit every requested function call together "
                "in the same assistant response before waiting for any result."
            )
        )
    ],
)


def _chatgpt_tokens() -> OAuthTokens | None:
    raw = os.getenv("KOLEGA_CODE_CHATGPT_TOKENS")
    if raw:
        try:
            return OAuthTokens.model_validate(json.loads(raw))
        except (TypeError, ValueError):
            return None
    stored = SettingsStore().load().get_oauth_token(chatgpt_constants.PROVIDER_KEY)
    if stored:
        try:
            return OAuthTokens.model_validate(stored)
        except (TypeError, ValueError):
            return None
    return None


def _client_for(provider: str, model: str) -> LLMClient:
    if SKIP_IN_CI:
        pytest.skip("Skipping live provider call in CI")
    if provider == chatgpt_constants.PROVIDER_KEY:
        tokens = _chatgpt_tokens()
        if tokens is None:
            pytest.skip("No ChatGPT OAuth tokens available")
        return LLMClient(
            provider=provider,
            api_key="unused",
            token_manager=ChatGPTTokenManager(tokens),
            model=model,
        )

    env_name = "OPENAI_API_KEY" if provider == "openai" else "DEEPSEEK_API_KEY"
    api_key = os.getenv(env_name)
    if not api_key:
        pytest.skip(f"{env_name} not set")
    return LLMClient(provider=provider, api_key=api_key, model=model)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider,model,thinking", PROVIDERS, ids=[provider for provider, _, _ in PROVIDERS])
async def test_live_responses_provider_returns_parallel_tool_calls(
    provider: str,
    model: str,
    thinking: str,
) -> None:
    client = _client_for(provider, model)
    try:
        response = await client.generate(
            messages=MessageHistory(
                [
                    Message(
                        role="user",
                        content=[
                            TextBlock(
                                text=(
                                    "Call lookup_alpha with token alpha and lookup_beta with token beta now. "
                                    "The calls are independent. Emit both calls in this response and no prose."
                                )
                            )
                        ],
                    )
                ]
            ),
            system=SYSTEM,
            model=model,
            tools=TOOLS,
            thinking=thinking,
            temperature=1.0,
        )
    except (LLMRateLimitError, LLMBillingError) as exc:
        pytest.skip(f"provider quota exhausted for this key: {exc}")

    tool_calls = [block for block in response.content if isinstance(block, ToolCall)]
    names = {call.name for call in tool_calls}
    assert len(tool_calls) >= 2, (
        f"{provider}/{model} returned fewer than two calls in one response: "
        f"names={sorted(names)!r}, text={response.get_text_content()!r}"
    )
    assert names == {"lookup_alpha", "lookup_beta"}, (
        f"{provider}/{model} did not return both calls in one response: "
        f"names={sorted(names)!r}, text={response.get_text_content()!r}"
    )
    assert response.stop_reason == "tool_use"
