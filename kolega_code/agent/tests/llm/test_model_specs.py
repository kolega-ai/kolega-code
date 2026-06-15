from kolega_code.llm.specs import get_model_specs


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
