from kolega_code.cli.provider_registry import default_model_for_provider, ui_model_options
from kolega_code.config import ModelProvider


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
