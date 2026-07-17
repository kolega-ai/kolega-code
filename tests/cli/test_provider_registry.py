from kolega_code.cli.provider_registry import (
    default_model_for_provider,
    ui_model_options,
    ui_thinking_effort_options,
)
from kolega_code.config import ModelProvider
from kolega_code.llm.specs import get_model_specs


def test_kimi_k3_is_first_and_default_for_moonshot():
    assert ui_model_options(ModelProvider.MOONSHOT.value)[0] == ("Kimi K3", "kimi-k3")
    assert default_model_for_provider(ModelProvider.MOONSHOT) == "kimi-k3"
    assert ui_thinking_effort_options("moonshot", "kimi-k3") == [("Max", "max")]


def test_kimi_coding_exposes_plan_specific_k3_models():
    assert ui_model_options(ModelProvider.KIMI_CODING.value) == [
        ("Kimi K3 (256K)", "k3"),
        ("Kimi K3 (1M)", "k3[1m]"),
        ("Kimi for Coding", "kimi-for-coding"),
    ]
    assert default_model_for_provider(ModelProvider.KIMI_CODING) == "kimi-for-coding"
    assert ui_thinking_effort_options("kimi_coding", "k3") == [("Max", "max")]


def test_gpt56_models_are_first_and_sol_is_default_for_openai_providers():
    expected = [
        ("GPT-5.6 Sol", "gpt-5.6-sol"),
        ("GPT-5.6 Terra", "gpt-5.6-terra"),
        ("GPT-5.6 Luna", "gpt-5.6-luna"),
    ]

    for provider in (ModelProvider.OPENAI, ModelProvider.OPENAI_CHATGPT):
        assert ui_model_options(provider.value)[:3] == expected
        assert default_model_for_provider(provider) == "gpt-5.6-sol"

    assert ui_thinking_effort_options("openai", "gpt-5.6-sol") == [
        ("None", "none"),
        ("Low", "low"),
        ("Medium", "medium"),
        ("High", "high"),
        ("Extra high", "xhigh"),
        ("Max", "max"),
    ]


def test_fireworks_ui_model_options_include_serverless_catalog():
    options = dict(ui_model_options("fireworks"))

    assert options["GLM-5.2"] == "accounts/fireworks/models/glm-5p2"
    assert options["GLM-5.1"] == "accounts/fireworks/models/glm-5p1"
    assert options["Kimi K2.7 Code"] == "accounts/fireworks/models/kimi-k2p7-code"
    assert options["DeepSeek V4 Pro"] == "accounts/fireworks/models/deepseek-v4-pro"
    assert options["DeepSeek V4 Flash"] == "accounts/fireworks/models/deepseek-v4-flash"
    assert options["MiniMax M3"] == "accounts/fireworks/models/minimax-m3"
    assert options["Qwen 3.7 Plus"] == "accounts/fireworks/models/qwen3p7-plus"
    assert "Gemma 4 31B IT" not in options


def test_fireworks_default_model_is_glm_52():
    assert default_model_for_provider(ModelProvider.FIREWORKS) == "accounts/fireworks/models/glm-5p2"


def test_vision_only_model_options_follow_catalog_capabilities():
    fireworks = dict(ui_model_options("fireworks", vision_only=True))

    assert fireworks == {
        "Kimi K2.7 Code": "accounts/fireworks/models/kimi-k2p7-code",
        "MiniMax M3": "accounts/fireworks/models/minimax-m3",
    }
    assert ui_model_options("deepseek", vision_only=True) == []


def test_ollama_cloud_smoke_model_is_available_without_live_call():
    options = dict(ui_model_options("ollama_cloud"))

    assert options["GPT-OSS 20B"] == "gpt-oss:20b"
    assert default_model_for_provider(ModelProvider.OLLAMA_CLOUD) == "gpt-oss:20b"
    assert get_model_specs(ModelProvider.OLLAMA_CLOUD.value, "gpt-oss:20b")["max_completion_tokens"] > 0
