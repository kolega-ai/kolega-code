import pytest

from kolega_code.llm.client import LLMClient
from kolega_code.llm.providers.anthropic import AnthropicProvider
from kolega_code.llm.providers.google import GoogleProvider
from kolega_code.llm.providers.models import GenerationParams
from kolega_code.llm.providers.openai import OpenAIProvider
from kolega_code.llm.specs import default_thinking_effort, thinking_effort_options


def test_model_specs_expose_provider_specific_thinking_efforts() -> None:
    assert thinking_effort_options("moonshot", "kimi-k2.6") == ("auto", "none")
    assert default_thinking_effort("moonshot", "kimi-k2.6") == "auto"
    assert thinking_effort_options("deepseek", "deepseek-v4-pro") == ("none", "high", "max")
    assert default_thinking_effort("deepseek", "deepseek-v4-pro") == "high"
    assert thinking_effort_options("google", "gemini-2.5-pro") == ("auto", "low", "medium", "high")
    assert default_thinking_effort("google", "gemini-2.5-pro") == "medium"


def test_anthropic_opus_effort_uses_adaptive_thinking_without_budget_tokens() -> None:
    provider = AnthropicProvider(api_key="test-key", provider_name="anthropic")
    generation_params = provider._prepare_generation_params(GenerationParams(thinking="xhigh"))
    generation_params["model"] = "claude-opus-4-7"

    provider._apply_thinking_params(generation_params, GenerationParams(thinking="xhigh"))

    assert generation_params["thinking"] == {"type": "adaptive"}
    assert generation_params["output_config"] == {"effort": "xhigh"}
    assert "budget_tokens" not in generation_params["thinking"]


def test_deepseek_thinking_effort_serialization() -> None:
    provider = AnthropicProvider(api_key="test-key", provider_name="deepseek")

    disabled_params = {"model": "deepseek-v4-pro"}
    provider._apply_thinking_params(disabled_params, GenerationParams(thinking="none"))
    assert disabled_params == {"model": "deepseek-v4-pro", "thinking": {"type": "disabled"}}

    high_params = {"model": "deepseek-v4-pro"}
    provider._apply_thinking_params(high_params, GenerationParams(thinking="high"))
    assert high_params["thinking"] == {"type": "enabled"}
    assert high_params["output_config"] == {"effort": "high"}

    max_params = {"model": "deepseek-v4-pro"}
    provider._apply_thinking_params(max_params, GenerationParams(thinking="max"))
    assert max_params["output_config"] == {"effort": "max"}


def test_moonshot_kimi_thinking_toggle_serialization() -> None:
    provider = AnthropicProvider(api_key="test-key", provider_name="moonshot")

    auto_params = {"model": "kimi-k2.6"}
    provider._apply_thinking_params(auto_params, GenerationParams(thinking="auto"))
    assert auto_params == {"model": "kimi-k2.6", "thinking": {"type": "enabled"}}

    disabled_params = {"model": "kimi-k2.6"}
    provider._apply_thinking_params(disabled_params, GenerationParams(thinking="none"))
    assert disabled_params == {"model": "kimi-k2.6", "thinking": {"type": "disabled"}}


def test_google_gemini_25_pro_uses_thinking_budget_mapping() -> None:
    provider = GoogleProvider(api_key="test-key")

    auto_config = provider._prepare_thinking_config("gemini-2.5-pro", GenerationParams(thinking="auto"))
    medium_config = provider._prepare_thinking_config("gemini-2.5-pro", GenerationParams(thinking="medium"))
    high_config = provider._prepare_thinking_config("gemini-2.5-pro", GenerationParams(thinking="high"))

    assert auto_config.thinking_budget == -1
    assert medium_config.thinking_budget == 8192
    assert high_config.thinking_budget == 24576


def test_openai_reasoning_effort_is_sent_for_reasoning_models() -> None:
    provider = OpenAIProvider(api_key="test-key")
    generation_params = provider._prepare_generation_params(GenerationParams(thinking="high"))
    generation_params["model"] = "o3"

    provider._apply_thinking_params(generation_params, GenerationParams(thinking="high"))

    assert generation_params["reasoning_effort"] == "high"


def test_numeric_thinking_budget_is_rejected() -> None:
    client = LLMClient(provider="moonshot", api_key="test-key")

    with pytest.raises(ValueError, match="named thinking effort"):
        client._prepare_thinking_param(1024, model="kimi-k2.6")
