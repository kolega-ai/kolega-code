from __future__ import annotations

import pytest

from kolega_code import AgentConfig, LLMClient, ModelConfig, ModelProvider
from kolega_code.agent.context import AgentContext, Telemetry
from kolega_code.llm.providers.openai import OpenAIProvider

FAKE_KEY = "abl_FAKE_refuseless_credential_for_tests"
MODEL = "glm-5.3-abliterated"
BASE_URL = "https://api.refuseless.com/v1"


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["refuseless", "REFUSELESS"])
async def test_public_client_uses_refuseless_chat_completions_endpoint(provider: str) -> None:
    client = LLMClient(provider=provider, api_key=FAKE_KEY, max_retries=0)

    assert isinstance(client.provider, OpenAIProvider)
    try:
        assert client.provider_name == "refuseless"
        assert client.provider.provider_name == "refuseless"
        assert str(client.provider.async_client.base_url).rstrip("/") == BASE_URL
        assert client.provider.async_client.api_key == FAKE_KEY
        assert client.provider.async_client.max_retries == 0
        assert LLMClient._provider_class(provider) is OpenAIProvider
    finally:
        await client.provider.async_client.close()


@pytest.mark.asyncio
async def test_refuseless_accepts_chat_completions_default_headers() -> None:
    client = LLMClient(provider="refuseless", api_key=FAKE_KEY, default_headers={"X-Test-Header": "fake-header-value"})

    assert isinstance(client.provider, OpenAIProvider)
    try:
        assert client.provider.async_client.default_headers["X-Test-Header"] == "fake-header-value"
        assert str(client.provider.async_client.base_url).rstrip("/") == BASE_URL
    finally:
        await client.provider.async_client.close()


@pytest.mark.asyncio
async def test_agent_context_creates_refuseless_client_for_role_override() -> None:
    context = AgentContext.__new__(AgentContext)
    route = ModelConfig(provider=ModelProvider.REFUSELESS, model=MODEL)
    context.config = AgentConfig(
        anthropic_api_key="fake-anthropic-key",
        refuseless_api_key=FAKE_KEY,
        agent_models={"investigation": route},
    )
    context.telemetry = Telemetry()

    client = context.create_llm_client("investigation-agent")

    assert isinstance(client.provider, OpenAIProvider)
    try:
        assert client.provider_name == "refuseless"
        assert client.provider.provider_name == "refuseless"
        assert client.provider.async_client.api_key == FAKE_KEY
        assert str(client.provider.async_client.base_url).rstrip("/") == BASE_URL
        assert context.config.long_context_config.provider is ModelProvider.ANTHROPIC
    finally:
        await client.provider.async_client.close()
