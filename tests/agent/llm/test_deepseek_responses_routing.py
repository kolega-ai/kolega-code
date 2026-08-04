# ruff: noqa: E402
"""Model-level API routing for DeepSeek.

`deepseek-v4-flash` speaks the Responses API (its own base URL), while every other
DeepSeek model stays on the OpenAI-compatible Chat Completions path. This is a
*model*-level decision made at client construction — the rest of the client is
provider-scoped. See DeepSeekResponsesProvider and LLMClient._initialize_provider.
"""

from kolega_code.llm.client import LLMClient
from kolega_code.llm.providers.deepseek_responses import DeepSeekResponsesProvider
from kolega_code.llm.providers.openai import OpenAIProvider
from kolega_code.llm.specs.thinking import build_thinking_request_params, reasoning_replay_field


class TestDeepSeekModelRouting:
    def test_flash_routes_to_responses_provider(self):
        provider = LLMClient(provider="deepseek", api_key="sk-test", model="deepseek-v4-flash").provider
        assert isinstance(provider, DeepSeekResponsesProvider)
        # Responses guide uses the bare host, not the /v1 Chat Completions base.
        assert provider.base_url == "https://api.deepseek.com"
        assert provider.provider_name == "deepseek"

    def test_pro_stays_on_chat_completions(self):
        provider = LLMClient(provider="deepseek", api_key="sk-test", model="deepseek-v4-pro").provider
        assert isinstance(provider, OpenAIProvider)
        assert not isinstance(provider, DeepSeekResponsesProvider)
        assert provider.base_url == "https://api.deepseek.com/v1"

    def test_missing_model_defaults_to_chat_completions(self):
        # No model at construction => safe default (Chat Completions), never the
        # Responses special-case.
        provider = LLMClient(provider="deepseek", api_key="sk-test").provider
        assert isinstance(provider, OpenAIProvider)
        assert not isinstance(provider, DeepSeekResponsesProvider)

    def test_other_provider_ignores_flash_model_name(self):
        # The special-case is guarded on provider == "deepseek"; a same-named model on a
        # different provider must not be hijacked into the DeepSeek Responses provider.
        provider = LLMClient(provider="openrouter", api_key="sk-test", model="deepseek-v4-flash").provider
        assert not isinstance(provider, DeepSeekResponsesProvider)


class TestDeepSeekReasoningShape:
    def test_flash_emits_responses_reasoning_block(self):
        # The shared Responses request builder only acts on a `reasoning` key; the flat
        # `reasoning_effort` the Chat path uses would be silently dropped.
        for effort in ("high", "max"):
            params = build_thinking_request_params("deepseek", "deepseek-v4-flash", effort)
            assert params == {"reasoning": {"effort": effort, "summary": "auto"}}

    def test_pro_keeps_flat_reasoning_effort(self):
        params = build_thinking_request_params("deepseek", "deepseek-v4-pro", "high")
        assert params == {"reasoning_effort": "high"}

    def test_flash_excluded_from_flat_replay_but_pro_included(self):
        # Flash replays reasoning as Responses reasoning ITEMS (plain reasoning_text
        # retained by the stream wrapper; the backend also restores CoT server-side
        # keyed on its own call_ids), so flash must not replay reasoning via the
        # flat Chat-Completions reasoning_content field; pro still does.
        assert reasoning_replay_field("deepseek", "deepseek-v4-flash") is None
        assert reasoning_replay_field("deepseek", "deepseek-v4-pro") == "reasoning_content"
