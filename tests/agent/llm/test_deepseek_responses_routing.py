# ruff: noqa: E402
"""Provider-level API routing for DeepSeek.

The whole ``deepseek`` provider (``deepseek-v4-pro`` and ``deepseek-v4-flash``)
speaks the Responses API via :class:`DeepSeekResponsesProvider`. DeepSeek models
hosted on other providers (Fireworks/OpenRouter/Ollama-Cloud) still use Chat
Completions. See ``LLMClient._provider_class``.
"""

from kolega_code.llm.client import LLMClient
from kolega_code.llm.providers.deepseek_responses import DeepSeekResponsesProvider
from kolega_code.llm.providers.openai import OpenAIProvider
from kolega_code.llm.specs.thinking import build_thinking_request_params, reasoning_replay_field


class TestDeepSeekModelRouting:
    def test_flash_routes_to_responses_provider(self):
        provider = LLMClient(provider="deepseek", api_key="sk-test", model="deepseek-v4-flash").provider
        assert isinstance(provider, DeepSeekResponsesProvider)
        assert provider.base_url == "https://api.deepseek.com"
        assert provider.provider_name == "deepseek"

    def test_pro_routes_to_responses_provider(self):
        provider = LLMClient(provider="deepseek", api_key="sk-test", model="deepseek-v4-pro").provider
        assert isinstance(provider, DeepSeekResponsesProvider)
        assert provider.base_url == "https://api.deepseek.com"
        assert provider.provider_name == "deepseek"

    def test_missing_model_defaults_to_responses(self):
        provider = LLMClient(provider="deepseek", api_key="sk-test").provider
        assert isinstance(provider, DeepSeekResponsesProvider)

    def test_other_provider_uses_chat_completions(self):
        provider = LLMClient(provider="openrouter", api_key="sk-test", model="deepseek-v4-flash").provider
        assert isinstance(provider, OpenAIProvider)
        assert not isinstance(provider, DeepSeekResponsesProvider)


class TestDeepSeekReasoningShape:
    def test_flash_emits_responses_reasoning_block(self):
        for effort in ("none", "low", "high", "max"):
            params = build_thinking_request_params("deepseek", "deepseek-v4-flash", effort)
            assert params == {"reasoning": {"effort": effort, "summary": "auto"}}

    def test_pro_emits_responses_reasoning_block(self):
        for effort in ("none", "low", "high", "max"):
            params = build_thinking_request_params("deepseek", "deepseek-v4-pro", effort)
            assert params == {"reasoning": {"effort": effort, "summary": "auto"}}

    def test_flash_and_pro_excluded_from_flat_replay(self):
        # Both carry reasoning as Responses reasoning ITEMS, not a flat field.
        assert reasoning_replay_field("deepseek", "deepseek-v4-flash") is None
        assert reasoning_replay_field("deepseek", "deepseek-v4-pro") is None
