from kolega_code.cli.provider_registry import (
    UI_DEFAULT_MODEL,
    UI_DEFAULT_PROVIDER,
    default_model_for_provider,
    get_ui_model,
    ui_model_options,
    ui_thinking_effort_options,
)
from kolega_code.config import ModelProvider
from kolega_code.llm.specs import get_model_specs


def test_kimi_k3_is_first_and_default_for_moonshot():
    assert ui_model_options(ModelProvider.MOONSHOT.value)[0] == ("Kimi K3", "kimi-k3")
    assert default_model_for_provider(ModelProvider.MOONSHOT) == "kimi-k3"
    assert ui_thinking_effort_options("moonshot", "kimi-k3") == [("Max", "max")]
    assert (UI_DEFAULT_PROVIDER, UI_DEFAULT_MODEL) == ("moonshot", "kimi-k3")


def test_opus_5_is_selectable_and_default_for_anthropic() -> None:
    options = ui_model_options(ModelProvider.ANTHROPIC.value)

    assert options[:3] == [
        ("Claude Fable 5", "claude-fable-5"),
        ("Claude Opus 5", "claude-opus-5"),
        ("Claude Opus 4.8", "claude-opus-4-8"),
    ]
    assert default_model_for_provider(ModelProvider.ANTHROPIC) == "claude-opus-5"
    assert ui_thinking_effort_options("anthropic", "claude-opus-5") == [
        ("Low", "low"),
        ("Medium", "medium"),
        ("High", "high"),
        ("Extra high", "xhigh"),
        ("Max", "max"),
    ]


def test_kimi_coding_exposes_plan_specific_k3_models():
    assert ui_model_options(ModelProvider.KIMI_CODING.value) == [
        ("Kimi K3 (256K)", "k3"),
        ("Kimi K3-256K", "k3-256k"),
        ("Kimi for Coding", "kimi-for-coding"),
        ("Kimi for Coding (HighSpeed)", "kimi-for-coding-highspeed"),
    ]
    assert default_model_for_provider(ModelProvider.KIMI_CODING) == "kimi-for-coding"
    assert ui_thinking_effort_options("kimi_coding", "k3") == [("Max", "max")]


def test_zai_exposes_glm_models_and_defaults_to_glm_53() -> None:
    assert ui_model_options(ModelProvider.ZAI.value) == [
        ("GLM-5.3", "glm-5.3"),
        ("GLM-5.3 Flash", "glm-5.3-flash"),
        ("GLM-5.2", "glm-5.2"),
        ("GLM-5.1", "glm-5.1"),
    ]
    assert default_model_for_provider(ModelProvider.ZAI) == "glm-5.3"
    assert ui_thinking_effort_options("zai", "glm-5.3") == [("High", "high"), ("Max", "max")]


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


def test_google_models_include_37_and_use_it_as_default() -> None:
    options = ui_model_options(ModelProvider.GOOGLE.value)

    assert options[:3] == [
        ("Gemini 3.7 Flash", "gemini-3.7-flash"),
        ("Gemini 3.6 Flash", "gemini-3.6-flash"),
        ("Gemini 3.5 Flash-Lite", "gemini-3.5-flash-lite"),
    ]
    assert default_model_for_provider(ModelProvider.GOOGLE) == "gemini-3.7-flash"
    assert ui_thinking_effort_options("google", "gemini-3.7-flash") == [
        ("Low", "low"),
        ("Medium", "medium"),
        ("High", "high"),
    ]
    assert ui_thinking_effort_options("google", "gemini-3.6-flash") == [
        ("Minimal", "minimal"),
        ("Low", "low"),
        ("Medium", "medium"),
        ("High", "high"),
    ]
    assert ui_thinking_effort_options("google", "gemini-3.5-flash-lite") == [
        ("Minimal", "minimal"),
        ("Low", "low"),
        ("Medium", "medium"),
        ("High", "high"),
    ]

    vision_options = dict(ui_model_options(ModelProvider.GOOGLE.value, vision_only=True))
    assert vision_options["Gemini 3.7 Flash"] == "gemini-3.7-flash"
    assert vision_options["Gemini 3.6 Flash"] == "gemini-3.6-flash"
    assert vision_options["Gemini 3.5 Flash-Lite"] == "gemini-3.5-flash-lite"


def test_fireworks_ui_model_options_include_serverless_catalog():
    options = dict(ui_model_options("fireworks"))

    assert options["Kimi K3"] == "accounts/fireworks/models/kimi-k3"
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
        "Kimi K3": "accounts/fireworks/models/kimi-k3",
        "Kimi K2.7 Code": "accounts/fireworks/models/kimi-k2p7-code",
        "MiniMax M3": "accounts/fireworks/models/minimax-m3",
    }
    assert dict(ui_model_options("deepseek", vision_only=True)) == {
        "DeepSeek V4 Flash Vision (Exp)": "deepseek-v4-flash-vision-exp"
    }


def test_ollama_cloud_smoke_model_is_available_without_live_call():
    options = ui_model_options("ollama_cloud")
    default = default_model_for_provider(ModelProvider.OLLAMA_CLOUD)

    assert default in {model for _label, model in options}
    assert get_model_specs(ModelProvider.OLLAMA_CLOUD.value, default)["max_completion_tokens"] > 0


def test_thinking_machines_models_are_selectable_and_inkling_is_default():
    options = ui_model_options(ModelProvider.THINKING_MACHINES.value)

    assert options == [
        ("Inkling", "thinkingmachines/Inkling"),
        ("Inkling Small", "thinkingmachines/Inkling-Small"),
    ]
    assert default_model_for_provider(ModelProvider.THINKING_MACHINES) == "thinkingmachines/Inkling"
    assert ui_thinking_effort_options("thinking_machines", "thinkingmachines/Inkling") == [
        ("None", "none"),
        ("Low", "low"),
        ("Medium", "medium"),
        ("High", "high"),
        ("Extra high", "xhigh"),
        ("Max", "max"),
    ]
    assert ui_thinking_effort_options("thinking_machines", "thinkingmachines/Inkling-Small") == [
        ("None", "none"),
        ("Low", "low"),
        ("Medium", "medium"),
        ("High", "high"),
        ("Extra high", "xhigh"),
        ("Max", "max"),
    ]


def test_thinking_machines_models_support_vision():
    assert ui_model_options("thinking_machines", vision_only=True) == [
        ("Inkling", "thinkingmachines/Inkling"),
        ("Inkling Small", "thinkingmachines/Inkling-Small"),
    ]


def test_tinker_provider_lists_catalogued_bases_and_defaults_to_qwen3_8b():
    options = ui_model_options(ModelProvider.TINKER.value)

    assert any(model == "Qwen/Qwen3-8B" for _label, model in options)
    assert default_model_for_provider(ModelProvider.TINKER) == "Qwen/Qwen3-8B"
    # Inkling bases expose the named effort set through the native renderer.
    assert ui_thinking_effort_options("tinker", "thinkingmachines/Inkling") == [
        ("None", "none"),
        ("Low", "low"),
        ("Medium", "medium"),
        ("High", "high"),
        ("Extra high", "xhigh"),
        ("Max", "max"),
    ]


def test_tinker_checkpoint_paths_resolve_through_wildcard_and_other_entry():
    from kolega_code.cli.tui.custom_model import CUSTOM_MODEL_SENTINEL, settings_model_options

    option = get_ui_model("tinker", "tinker://run-1:train:0/sampler_weights/000080")
    assert option is not None
    assert option.model == "tinker://run-1:train:0/sampler_weights/000080"
    assert option.api_key_env == "TINKER_API_KEY"
    assert option.context_length == 65536
    assert option.thinking_efforts == ()
    # The Settings picker offers the typed-id "Other…" entry for tinker.
    assert settings_model_options("tinker")[-1] == ("Other…", CUSTOM_MODEL_SENTINEL)
