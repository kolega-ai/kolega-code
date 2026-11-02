from collections import Counter

import pytest

from kolega_code.cli.config import API_KEY_ENV, OAUTH_PROVIDERS
from kolega_code.cli.provider_registry import default_model_for_provider
from kolega_code.config import ModelProvider
from kolega_code.llm.client import LLMClient
from kolega_code.llm.models import MessageHistory
from kolega_code.llm.providers.deepseek_responses import DeepSeekResponsesProvider
from kolega_code.llm.providers.models import GenerationParams
from kolega_code.llm.providers.openai import OpenAIProvider
from kolega_code.llm.providers.responses_common import ResponsesProviderBase
from kolega_code.llm.specs import (
    MODEL_SPECS,
    get_model_specs,
    get_thinking_effort_spec,
    supports_hosted_web_search,
    thinking_effort_options,
)

from . import (
    test_context_residual_live as residual,
    test_deepseek_flash_context_gauge_live as context_gauge,
    test_deepseek_output_cap_live as output_cap,
    test_deepseek_responses_live as deepseek_responses,
    test_hosted_web_search_live as hosted_search,
    test_live_providers as smoke,
    test_normalized_usage_live as usage,
)
from ._live_models import smoke_generation_settings


@pytest.mark.parametrize(
    "model", [residual.MODEL, context_gauge.MODEL, output_cap._FLASH, deepseek_responses._VISION_MODEL]
)
def test_flash_fixtures_use_a_catalog_model_with_reasoning(model: str) -> None:
    specs = get_model_specs("deepseek", model)
    assert specs["supports_vision"] is True
    effort = get_thinking_effort_spec("deepseek", model)
    assert effort is not None
    assert effort.mode == "openai_responses_reasoning"
    assert "high" in effort.options


@pytest.mark.parametrize("name", list(hosted_search.PROVIDERS))
def test_hosted_search_fixtures_support_the_required_tool_and_transport(name: str) -> None:
    provider_cls, provider, model, thinking = hosted_search.PROVIDERS[name]
    assert (provider.value, model) in MODEL_SPECS
    assert supports_hosted_web_search(provider.value, model)
    assert thinking in thinking_effort_options(provider.value, model)
    client = LLMClient(provider=provider.value, api_key="test-placeholder", model=model)
    assert isinstance(client.provider, provider_cls)
    assert isinstance(client.provider, ResponsesProviderBase)
    request = client.provider._build_request(
        MessageHistory([]),
        None,
        GenerationParams(hosted_web_search=True, thinking=thinking),
        {"model": model},
    )
    assert {"type": "web_search"} in request["tools"]


def test_smoke_and_usage_suites_share_unique_keyed_provider_coverage() -> None:
    expected = {provider for provider, _ in MODEL_SPECS if ModelProvider(provider) not in OAUTH_PROVIDERS}
    for pairs in (smoke.PROVIDER_SMOKE_MODELS, usage.PROVIDER_MODELS):
        assert Counter(provider for provider, _ in pairs) == Counter(dict.fromkeys(expected, 1))
        assert len(pairs) == len(set(pairs))
        for provider, model in pairs:
            assert ModelProvider(provider) in API_KEY_ENV
            if provider == "together":
                assert model == "openai/gpt-oss-120b"
                assert smoke_generation_settings(provider, model) == {"temperature": 1.0, "thinking": None}
            else:
                assert (provider, model) in MODEL_SPECS
    assert smoke.PROVIDER_SMOKE_MODELS == usage.PROVIDER_MODELS


def test_together_smoke_uses_a_current_serverless_model() -> None:
    assert dict(smoke.PROVIDER_SMOKE_MODELS)["together"] == "openai/gpt-oss-120b"
    assert dict(usage.PROVIDER_MODELS)["together"] == "openai/gpt-oss-120b"
    assert default_model_for_provider(ModelProvider.TOGETHER) == "moonshotai/Kimi-K2.7-Code"
    together_specs = {model: specs for (provider, model), specs in MODEL_SPECS.items() if provider == "together"}
    assert set(together_specs) == {"moonshotai/Kimi-K2.7-Code", "zai-org/GLM-5.1"}
    assert all(specs["pricing"]["status"] == "retired" for specs in together_specs.values())


@pytest.mark.parametrize("provider", ["together", "fireworks", "openrouter"])
def test_smoke_settings_do_not_accept_arbitrary_unknown_models(provider: str) -> None:
    with pytest.raises(ValueError, match="not supported"):
        smoke_generation_settings(provider, "not-a-smoke-model")


def test_smoke_thinking_coverage_preserves_catalog_effort_controls() -> None:
    expected = {
        (provider, model)
        for provider, model in smoke.PROVIDER_SMOKE_MODELS
        if provider != "ollama_cloud"
        and (provider, model) in MODEL_SPECS
        and get_thinking_effort_spec(provider, model) is not None
    }
    assert set(smoke.PROVIDER_THINKING_MODELS) == expected


def test_chat_cap_fixtures_use_current_deepseek_gateway_routes() -> None:
    assert output_cap.CHAT_HOSTS == [
        ("openrouter", "deepseek/deepseek-v4-pro"),
        ("openrouter", "deepseek/deepseek-v4-flash"),
    ]
    for provider, model in output_cap.CHAT_HOSTS:
        assert get_model_specs(provider, model)["pricing"]["status"] == "priced"


@pytest.mark.parametrize(
    "provider,model",
    [
        ("fireworks", "accounts/fireworks/models/ember-1"),
        ("dashscope", "qwen3-coder-flash"),
    ],
)
def test_smoke_overrides_use_catalog_models_with_current_prices(provider: str, model: str) -> None:
    assert dict(smoke.PROVIDER_SMOKE_MODELS)[provider] == model
    assert dict(usage.PROVIDER_MODELS)[provider] == model
    specs = get_model_specs(provider, model)
    assert specs["max_completion_tokens"] >= 8192
    assert specs["pricing"]["status"] == "priced"


@pytest.mark.parametrize("provider,model", output_cap.CHAT_HOSTS)
def test_raw_chat_cap_fixtures_actually_use_chat_completions(provider: str, model: str) -> None:
    assert (provider, model) in MODEL_SPECS
    client = LLMClient(provider=provider, api_key="test-placeholder", model=model)
    assert type(client.provider) is OpenAIProvider


@pytest.mark.parametrize("model", output_cap.RESPONSES_MODELS)
def test_responses_cap_fixtures_use_the_catalog_responses_transport(model: str) -> None:
    assert ("deepseek", model) in MODEL_SPECS
    client = LLMClient(provider="deepseek", api_key="test-placeholder", model=model)
    assert isinstance(client.provider, DeepSeekResponsesProvider)
    request = client.provider._build_request(MessageHistory([]), None, None, {"model": model})
    assert request["model"] == model
    assert request["stream"] is True
    assert "max_output_tokens" in request


def test_deepseek_search_does_not_use_the_flash_architecture() -> None:
    assert hosted_search.PROVIDERS["deepseek"][2] == deepseek_responses._PRO_MODEL
    assert supports_hosted_web_search("deepseek", deepseek_responses._PRO_MODEL)
    assert not supports_hosted_web_search("deepseek", deepseek_responses._VISION_MODEL)
