from kolega_code.llm.client import LLMClient
from kolega_code.llm.providers.anthropic import AnthropicProvider
from kolega_code.llm.providers.openai import OpenAIProvider


# TODO: Fix after qwen-3-coder-plus PR is merged - needs dashscope provider mapping
def test_llm_client_maps_dashscope_to_openai_provider():
    client = LLMClient(provider='dashscope', api_key='sk-test')
    assert isinstance(client.provider, OpenAIProvider)
    assert client.provider.base_url == 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'


def test_llm_client_maps_moonshot_to_anthropic_provider():
    client = LLMClient(provider='moonshot', api_key='sk-test')
    assert isinstance(client.provider, AnthropicProvider)
    assert client.provider.base_url == 'https://api.moonshot.ai/anthropic'
    assert client.provider.provider_name == 'moonshot'
    assert client.provider.use_local_token_counting is True

    thinking = client._prepare_thinking_param(8192)
    assert thinking.budget_tokens == 8192


def test_llm_client_maps_deepseek_to_anthropic_provider():
    client = LLMClient(provider='deepseek', api_key='sk-test')
    assert isinstance(client.provider, AnthropicProvider)
    assert client.provider.base_url == 'https://api.deepseek.com/anthropic'
    assert client.provider.provider_name == 'deepseek'
    assert client.provider.use_local_token_counting is True

    thinking = client._prepare_thinking_param(8192)
    assert thinking.budget_tokens == 8192
