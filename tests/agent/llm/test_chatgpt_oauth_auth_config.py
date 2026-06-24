# ruff: noqa: F401,F811,E402
"""Tests for the ChatGPT-subscription Responses provider and its wiring."""

import types

import httpx
import pytest

from kolega_code.auth import constants as chatgpt_constants
from kolega_code.auth.tokens import ChatGPTTokenManager, OAuthTokens
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider
from kolega_code.llm.client import LLMClient
from kolega_code.llm.models import (
    ImageBlock,
    Message,
    MessageHistory,
    ResponsesReasoningBlock,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)
from kolega_code.llm.providers.chatgpt_oauth import (
    ChatGPTAuth,
    ChatGPTOAuthProvider,
    ResponsesStreamWrapper,
    instructions_from,
    responses_tools,
    to_responses_input,
)
from kolega_code.llm.providers.models import GenerationParams

def _ns(**kwargs):
    return types.SimpleNamespace(**kwargs)

def _tokens():
    return OAuthTokens(access_token="at", refresh_token="rt", expires_at=10**12, account_id="acct_1", plan_type="pro")

@pytest.mark.asyncio
async def test_chatgpt_auth_sets_headers():
    auth = ChatGPTAuth(ChatGPTTokenManager(_tokens()))
    request = httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses")
    flow = auth.async_auth_flow(request)
    sent = await flow.__anext__()
    assert sent.headers["Authorization"] == "Bearer at"
    assert sent.headers[chatgpt_constants.ACCOUNT_ID_HEADER] == "acct_1"
    await flow.aclose()
def test_llmclient_routes_to_chatgpt_provider():
    client = LLMClient(
        provider="openai_chatgpt",
        api_key="unused",
        token_manager=ChatGPTTokenManager(_tokens()),
    )
    assert isinstance(client.provider, ChatGPTOAuthProvider)
def test_llmclient_chatgpt_without_manager_raises():
    with pytest.raises(Exception):
        LLMClient(provider="openai_chatgpt", api_key="unused")
def test_agent_config_validates_with_chatgpt_tokens():
    config = AgentConfig(
        openai_chatgpt_tokens=_tokens(),
        long_context_config=ModelConfig(provider=ModelProvider.OPENAI_CHATGPT, model="gpt-5.5"),
        fast_config=ModelConfig(provider=ModelProvider.OPENAI_CHATGPT, model="gpt-5.5"),
        thinking_config=ModelConfig(provider=ModelProvider.OPENAI_CHATGPT, model="gpt-5.5"),
    )
    assert config.get_api_key(ModelProvider.OPENAI_CHATGPT) == "at"
    manager = config.get_chatgpt_token_manager()
    assert manager is not None
def test_agent_config_without_tokens_rejects_chatgpt_provider():
    with pytest.raises(ValueError, match="signed in"):
        AgentConfig(
            long_context_config=ModelConfig(provider=ModelProvider.OPENAI_CHATGPT, model="gpt-5.5"),
            fast_config=ModelConfig(provider=ModelProvider.OPENAI_CHATGPT, model="gpt-5.5"),
            thinking_config=ModelConfig(provider=ModelProvider.OPENAI_CHATGPT, model="gpt-5.5"),
        )
