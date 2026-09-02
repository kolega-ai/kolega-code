import pytest

from kolega_code.llm.specs import (
    build_thinking_request_params,
    get_model_specs,
    resolve_max_input_tokens,
    thinking_effort_options,
)


@pytest.mark.parametrize(
    "model,thinking_options,default_thinking",
    [
        ("gemini-3.7-flash", ("low", "medium", "high"), "medium"),
        ("gemini-3.6-flash", ("minimal", "low", "medium", "high"), "medium"),
        ("gemini-3.5-flash-lite", ("minimal", "low", "medium", "high"), "minimal"),
    ],
)
def test_new_google_model_specs(
    model: str,
    thinking_options: tuple[str, ...],
    default_thinking: str,
) -> None:
    specs = get_model_specs("google", model)

    assert specs["context_length"] == 1048576
    assert specs["max_completion_tokens"] == 65536
    assert specs["input_budget"] == "separate_output_limit"
    assert specs["default_temperature"] == 1.0
    assert specs["supports_temperature"] is False
    assert specs["supports_vision"] is True
    assert specs["thinking_effort"].options == thinking_options
    assert specs["thinking_effort"].default == default_thinking
    assert specs["thinking_effort"].mode == "google_thinking_level"


@pytest.mark.parametrize("model", ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"])
@pytest.mark.parametrize("provider,context_length", [("openai", 1050000), ("openai_chatgpt", 400000)])
def test_gpt56_model_specs(provider, context_length, model):
    specs = get_model_specs(provider, model)

    assert specs["context_length"] == context_length
    assert specs["max_completion_tokens"] == 128000
    assert specs["default_temperature"] == 1.0
    assert specs["supports_temperature"] is False
    assert specs["supports_vision"] is True
    assert specs["thinking_effort"].options == ("none", "low", "medium", "high", "xhigh", "max")
    assert specs["thinking_effort"].default == "medium"
    assert specs["thinking_effort"].mode == "openai_responses_reasoning"


def test_openai_chatgpt_gpt55_context_length():
    """GPT-5.5 on ChatGPT subscription is capped at 400K context by the Codex
    backend, with 128K reserved for output → effective max input is ~272K."""
    specs = get_model_specs("openai_chatgpt", "gpt-5.5")

    assert specs["context_length"] == 400000
    assert specs["max_completion_tokens"] == 128000
    assert resolve_max_input_tokens(specs) == 272000
    assert specs["thinking_effort"].default == "medium"


@pytest.mark.parametrize("provider", ["openai", "openai_chatgpt"])
def test_gpt54_mini_thinking_efforts(provider):
    specs = get_model_specs(provider, "gpt-5.4-mini")

    assert specs["thinking_effort"].options == ("none", "low", "medium", "high", "xhigh")
    assert specs["thinking_effort"].default == "medium"


def test_grok_46_model_specs():
    specs = get_model_specs("xai", "grok-4.6")

    assert specs["context_length"] == 500000
    assert specs["max_completion_tokens"] == 32768
    assert specs["default_temperature"] == 0.7
    assert specs["supports_vision"] is True
    assert specs["thinking_effort"].options == ("low", "medium", "high", "xhigh")
    assert specs["thinking_effort"].default == "medium"
    assert specs["thinking_effort"].mode == "openai_reasoning_effort"


def test_grok_45_model_specs():
    specs = get_model_specs("xai", "grok-4.5")

    assert specs["context_length"] == 500000
    assert specs["max_completion_tokens"] == 32768
    assert specs["default_temperature"] == 0.7
    assert specs["supports_vision"] is True
    assert specs["thinking_effort"].options == ("low", "medium", "high")
    assert specs["thinking_effort"].default == "medium"
    assert specs["thinking_effort"].mode == "openai_reasoning_effort"


def test_kimi_k3_model_specs():
    specs = get_model_specs("moonshot", "kimi-k3")

    assert specs["context_length"] == 1048576
    assert specs["max_completion_tokens"] == 131072
    assert specs["default_temperature"] == 1.0
    assert specs["supports_temperature"] is False
    assert specs["supports_vision"] is True
    assert specs["thinking_effort"].options == ("max",)
    assert specs["thinking_effort"].default == "max"
    assert specs["thinking_effort"].mode == "moonshot_reasoning_effort"


def test_kimi_k27_code_model_specs():
    specs = get_model_specs("moonshot", "kimi-k2.7-code")

    assert specs["context_length"] == 262144
    assert specs["max_completion_tokens"] == 32768
    assert specs["default_temperature"] == 1.0
    assert specs["thinking_effort"].options == ("auto",)
    assert specs["thinking_effort"].default == "auto"


def test_kimi_k26_model_specs():
    specs = get_model_specs("moonshot", "kimi-k2.6")

    assert specs["context_length"] == 262144
    assert specs["max_completion_tokens"] == 32768
    assert specs["default_temperature"] == 1.0
    assert specs["thinking_effort"].options == ("auto", "none")
    assert specs["thinking_effort"].default == "auto"


@pytest.mark.parametrize(
    "model,context_length",
    [
        ("k3", 262144),
        ("k3-256k", 262144),
    ],
)
def test_kimi_coding_k3_model_specs(model, context_length):
    specs = get_model_specs("kimi_coding", model)

    assert specs["context_length"] == context_length
    assert specs["max_completion_tokens"] == 131072
    assert specs["default_temperature"] == 1.0
    assert specs["supports_temperature"] is False
    assert specs["supports_vision"] is True
    assert specs["thinking_effort"].options == ("max",)
    assert specs["thinking_effort"].default == "max"
    assert specs["thinking_effort"].mode == "kimi_coding_effort"


def test_kimi_coding_highspeed_model_specs():
    specs = get_model_specs("kimi_coding", "kimi-for-coding-highspeed")

    assert specs["context_length"] == 262144
    assert specs["max_completion_tokens"] == 32768
    assert specs["default_temperature"] == 1.0
    assert specs["supports_vision"] is True
    assert specs["thinking_effort"].options == ("auto", "none")
    assert specs["thinking_effort"].default == "auto"
    assert specs["thinking_effort"].mode == "moonshot_toggle"


@pytest.mark.parametrize(
    "model,max_completion_tokens",
    [
        # pro: the MEASURED chat ceiling — the server cuts at exactly 65536.
        ("deepseek-v4-pro", 65536),
        # flash: no such cut; measured running to 112990 in one call.
        ("deepseek-v4-flash", 384000),
    ],
)
def test_deepseek_model_specs(model: str, max_completion_tokens: int):
    specs = get_model_specs("deepseek", model)

    assert specs["context_length"] == 1000000
    assert specs["max_completion_tokens"] == max_completion_tokens
    assert specs["default_temperature"] == 1.0
    assert specs["thinking_effort"].options == ("none", "low", "high", "max")
    assert specs["thinking_effort"].default == "high"


def test_zai_model_specs() -> None:
    for model in ("glm-5.3", "glm-5.3-flash", "glm-5.2"):
        specs = get_model_specs("zai", model)
        assert specs["context_length"] == 1000000
        assert specs["max_completion_tokens"] == 131072
        assert specs["default_temperature"] == 0.6
        assert specs["thinking_effort"].options == ("high", "max")
        assert specs["thinking_effort"].default == "max"

    # GLM-5.3-Flash is the first native-multimodal GLM-5 series model.
    assert get_model_specs("zai", "glm-5.3-flash")["supports_vision"] is True
    assert get_model_specs("zai", "glm-5.3")["supports_vision"] is False


def test_fireworks_serverless_model_specs():
    expected = {
        "accounts/fireworks/models/glm-5p2": (1048576, 131072),
        "accounts/fireworks/models/glm-5p1": (202800, 131072),
        "accounts/fireworks/models/kimi-k2p7-code": (262144, 262144),
        "accounts/fireworks/models/deepseek-v4-pro": (1048576, 65536),
        "accounts/fireworks/models/deepseek-v4-flash": (1048576, 65536),
        "accounts/fireworks/models/minimax-m3": (512000, 512000),
        "accounts/fireworks/models/qwen3p7-plus": (262144, 65536),
    }

    for model, (context_length, max_completion_tokens) in expected.items():
        specs = get_model_specs("fireworks", model)
        assert specs["context_length"] == context_length
        assert specs["max_completion_tokens"] == max_completion_tokens
        assert specs["default_temperature"] == 1.0
        assert thinking_effort_options("fireworks", model) == ("none", "low", "medium", "high", "max")
        assert specs["thinking_effort"].default == "medium"


def test_fireworks_kimi_k3_model_specs():
    specs = get_model_specs("fireworks", "accounts/fireworks/models/kimi-k3")

    assert specs["context_length"] == 1048576
    assert specs["max_completion_tokens"] == 131072
    assert specs["default_temperature"] == 1.0
    assert specs["supports_vision"] is True
    assert specs["thinking_effort"].options == ("max",)
    assert specs["thinking_effort"].default == "max"
    assert specs["thinking_effort"].mode == "openai_reasoning_effort"


def test_claude_fable_5_model_specs():
    specs = get_model_specs("anthropic", "claude-fable-5")

    assert specs["context_length"] == 1000000
    assert specs["max_completion_tokens"] == 128000
    assert specs["default_temperature"] == 1.0
    assert specs["supports_temperature"] is False
    assert specs["supports_vision"] is True
    assert specs["thinking_effort"].options == ("low", "medium", "high", "xhigh", "max")
    assert specs["thinking_effort"].default == "medium"
    assert specs["thinking_effort"].mode == "anthropic_adaptive_effort"


def test_claude_opus_5_model_specs() -> None:
    specs = get_model_specs("anthropic", "claude-opus-5")

    assert specs["context_length"] == 1000000
    assert specs["max_completion_tokens"] == 128000
    assert specs["default_temperature"] == 1.0
    assert specs["supports_temperature"] is False
    assert specs["supports_vision"] is True
    assert specs["thinking_effort"].options == ("low", "medium", "high", "xhigh", "max")
    assert specs["thinking_effort"].default == "medium"
    assert specs["thinking_effort"].mode == "anthropic_adaptive_effort"


def test_claude_sonnet_5_model_specs():
    specs = get_model_specs("anthropic", "claude-sonnet-5")

    assert specs["context_length"] == 1000000
    assert specs["max_completion_tokens"] == 128000
    assert specs["default_temperature"] == 1.0
    assert specs["supports_temperature"] is False
    assert specs["supports_vision"] is True
    assert specs["thinking_effort"].options == ("low", "medium", "high", "xhigh", "max")
    assert specs["thinking_effort"].default == "medium"
    assert specs["thinking_effort"].mode == "anthropic_adaptive_effort"


def test_thinking_machines_model_specs():
    for model in ("thinkingmachines/Inkling", "thinkingmachines/Inkling-Small"):
        specs = get_model_specs("thinking_machines", model)

        assert specs["context_length"] == 1000000
        assert specs["max_completion_tokens"] == 32768
        assert specs["default_temperature"] == 1.0
        assert specs["supports_vision"] is True
        assert thinking_effort_options("thinking_machines", model) == (
            "none",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        )
        assert specs["thinking_effort"].default == "high"
        assert specs["thinking_effort"].mode == "tinker_effort"


def test_tinker_effort_request_params():
    # The Tinker Anthropic-compatible endpoint controls depth via the
    # Tinker-specific output_config.effort field; "none" turns reasoning off
    # with the standard Anthropic thinking={"type": "disabled"}.
    assert build_thinking_request_params("thinking_machines", "thinkingmachines/Inkling", "none") == {
        "thinking": {"type": "disabled"}
    }
    assert build_thinking_request_params("thinking_machines", "thinkingmachines/Inkling", "low") == {
        "output_config": {"effort": "low"}
    }
    assert build_thinking_request_params("thinking_machines", "thinkingmachines/Inkling", "max") == {
        "output_config": {"effort": "max"}
    }
    # Effort values outside the documented Tinker set are rejected.
    with pytest.raises(ValueError, match="Unsupported thinking effort"):
        build_thinking_request_params("thinking_machines", "thinkingmachines/Inkling", "turbo")
