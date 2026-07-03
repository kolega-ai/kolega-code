from kolega_code.llm.specs import get_model_specs, thinking_effort_options


def test_openai_chatgpt_gpt55_context_length():
    """GPT-5.5 on ChatGPT subscription is capped at 400K context by the Codex
    backend, with 128K reserved for output → effective max input is ~272K."""
    specs = get_model_specs("openai_chatgpt", "gpt-5.5")

    assert specs["context_length"] == 272000
    assert specs["max_completion_tokens"] == 128000
    assert specs["thinking_effort"].default == "medium"


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


def test_deepseek_v4_pro_model_specs():
    specs = get_model_specs("deepseek", "deepseek-v4-pro")

    assert specs["context_length"] == 1000000
    assert specs["max_completion_tokens"] == 384000
    assert specs["default_temperature"] == 1.0
    assert specs["thinking_effort"].options == ("none", "high", "max")
    assert specs["thinking_effort"].default == "high"


def test_fireworks_serverless_model_specs():
    expected = {
        "accounts/fireworks/models/glm-5p2": (1048576, 131072),
        "accounts/fireworks/models/glm-5p1": (202800, 131072),
        "accounts/fireworks/models/kimi-k2p7-code": (262144, 262144),
        "accounts/fireworks/models/deepseek-v4-pro": (1048576, 384000),
        "accounts/fireworks/models/deepseek-v4-flash": (1048576, 384000),
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
