"""Tests for the ``supports_vision`` capability flag and accessor."""

import pytest

from kolega_code.llm.specs import MODEL_SPECS, get_model_specs, supports_vision


def test_supports_vision_flag_present_on_every_entry():
    missing = [key for key, spec in MODEL_SPECS.items() if "supports_vision" not in spec]
    assert not missing, f"entries missing supports_vision: {missing}"


@pytest.mark.parametrize(
    "provider,model,expected",
    [
        ("anthropic", "claude-fable-5", True),
        ("anthropic", "claude-sonnet-5", True),
        ("anthropic", "claude-opus-4-8", True),
        ("anthropic", "claude-haiku-4-5-20251001", True),
        ("openai", "gpt-5.6-sol", True),
        ("openai", "gpt-5.6-terra", True),
        ("openai", "gpt-5.6-luna", True),
        ("openai", "gpt-5.5", True),
        ("openai", "gpt-5.4-mini", True),
        ("openai_chatgpt", "gpt-5.6-sol", True),
        ("openai_chatgpt", "gpt-5.6-terra", True),
        ("openai_chatgpt", "gpt-5.6-luna", True),
        ("openai_chatgpt", "gpt-5.5", True),
        ("openai_chatgpt", "gpt-5.3-codex-spark", True),
        ("google", "gemini-3.5-flash", True),
        ("google", "gemini-3.1-pro-preview", True),
        ("moonshot", "kimi-k3", True),
        ("moonshot", "kimi-k2.7-code", True),
        ("moonshot", "kimi-k2.6", True),
        ("kimi_coding", "k3", True),
        ("kimi_coding", "k3[1m]", True),
        ("kimi_coding", "kimi-for-coding", True),
        ("xai", "grok-4.5", True),
        ("xai", "grok-4.3", True),
        ("fireworks", "accounts/fireworks/models/minimax-m3", True),
        ("fireworks", "accounts/fireworks/models/kimi-k2p7-code", True),
        ("together", "moonshotai/Kimi-K2.7-Code", True),
        # Non-vision models
        ("deepseek", "deepseek-v4-pro", False),
        ("deepseek", "deepseek-v4-flash", False),
        ("fireworks", "accounts/fireworks/models/deepseek-v4-pro", False),
        ("fireworks", "accounts/fireworks/models/deepseek-v4-flash", False),
        ("fireworks", "accounts/fireworks/models/glm-5p2", False),
        ("fireworks", "accounts/fireworks/models/glm-5p1", False),
        ("fireworks", "accounts/fireworks/models/qwen3p7-plus", False),
        ("dashscope", "qwen3-coder-plus", False),
        ("dashscope", "qwen3-coder-flash", False),
        ("zai", "glm-5.2", False),
        ("zai", "glm-5.1", False),
        ("xai", "grok-build-0.1", False),
        ("together", "zai-org/GLM-5.1", False),
    ],
)
def test_supports_vision_values(provider, model, expected):
    assert supports_vision(provider, model) is expected


def test_supports_vision_defaults_false_when_key_absent():
    # All real entries have the flag, but the accessor must default to False
    # so a future entry that omits it is safely non-vision.
    spec = get_model_specs("anthropic", "claude-opus-4-8")
    assert spec.get("supports_vision", False) is True
    assert {}.get("supports_vision", False) is False
