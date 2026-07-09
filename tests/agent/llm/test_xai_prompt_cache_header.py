"""xAI prompt-cache routing for Chat Completions.

xAI highly recommends setting the ``x-grok-conv-id`` header on Chat Completions
so consecutive turns of a conversation stick to the same cache-warm server. The
Responses API equivalent is body ``prompt_cache_key`` (already handled by the
Responses providers). See https://docs.x.ai/developers/advanced-api-usage/prompt-caching
"""

from kolega_code.llm.client import LLMClient
from kolega_code.llm.providers.openai import OpenAIProvider


def test_xai_provider_sets_stable_grok_conv_id_header():
    provider = OpenAIProvider(
        api_key="sk-test",
        base_url="https://api.x.ai/v1",
        provider_name="xai",
    )

    assert provider._session_id
    assert provider.async_client.default_headers.get("x-grok-conv-id") == provider._session_id


def test_non_xai_openai_compatible_provider_omits_grok_conv_id():
    provider = OpenAIProvider(
        api_key="sk-test",
        base_url="https://api.fireworks.ai/inference/v1",
        provider_name="fireworks",
    )

    assert provider.async_client.default_headers.get("x-grok-conv-id") is None


def test_llmclient_xai_route_carries_grok_conv_id():
    client = LLMClient(provider="xai", api_key="sk-test")
    assert isinstance(client.provider, OpenAIProvider)
    assert client.provider.provider_name == "xai"
    assert client.provider.async_client.default_headers.get("x-grok-conv-id") == client.provider._session_id
