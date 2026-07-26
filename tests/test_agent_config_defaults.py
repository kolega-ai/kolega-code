from kolega_code.config import AgentConfig, ModelProvider


def test_agent_config_defaults_to_opus_5_for_long_and_thinking_slots() -> None:
    config = AgentConfig(anthropic_api_key="test-key")

    assert config.long_context_config.provider == ModelProvider.ANTHROPIC
    assert config.long_context_config.model == "claude-opus-5"
    assert config.long_context_config.thinking_effort == "medium"
    assert config.thinking_config.provider == ModelProvider.ANTHROPIC
    assert config.thinking_config.model == "claude-opus-5"
    assert config.thinking_config.thinking_effort == "medium"


def test_agent_config_keeps_haiku_as_the_fast_default() -> None:
    config = AgentConfig(anthropic_api_key="test-key")

    assert config.fast_config.provider == ModelProvider.ANTHROPIC
    assert config.fast_config.model == "claude-haiku-4-5-20251001"
