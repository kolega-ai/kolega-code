from pathlib import Path

import pytest

from kolega_code.config import ModelProvider
from kolega_code.cli.config import (
    DEFAULT_EDIT_MODEL,
    DEFAULT_FAST_MODEL,
    DEFAULT_LONG_MODEL,
    DEFAULT_THINKING_MODEL,
    CliConfigError,
    CliConfigOverrides,
    build_agent_config,
    config_summary,
)
from kolega_code.cli.provider_registry import DEEPSEEK_DEFAULT_MODEL, UI_DEFAULT_MODEL, UI_DEFAULT_PROVIDER
from kolega_code.cli.settings import CliSettings


def test_build_agent_config_defaults_to_latest_anthropic_models(tmp_path: Path) -> None:
    config = build_agent_config(tmp_path, env={"ANTHROPIC_API_KEY": "test-key"})

    assert config.long_context_config.provider == ModelProvider.ANTHROPIC
    assert config.long_context_config.model == DEFAULT_LONG_MODEL
    assert config.fast_config.model == DEFAULT_FAST_MODEL
    assert config.edit_model_config.model == DEFAULT_EDIT_MODEL
    assert config.thinking_config.model == DEFAULT_THINKING_MODEL
    assert config.thinking_config.thinking_tokens == 1024


def test_build_agent_config_env_overrides(tmp_path: Path) -> None:
    config = build_agent_config(
        tmp_path,
        env={
            "ANTHROPIC_API_KEY": "test-key",
            "KOLEGA_CODE_MODEL": "claude-sonnet-4-6",
            "KOLEGA_CODE_THINKING_TOKENS": "2048",
        },
    )

    assert config.long_context_config.model == "claude-sonnet-4-6"
    assert config.thinking_config.thinking_tokens == 2048


def test_build_agent_config_flags_override_env(tmp_path: Path) -> None:
    config = build_agent_config(
        tmp_path,
        CliConfigOverrides(model="claude-opus-4-7", thinking_tokens=4096),
        env={
            "ANTHROPIC_API_KEY": "test-key",
            "KOLEGA_CODE_MODEL": "claude-sonnet-4-6",
            "KOLEGA_CODE_THINKING_TOKENS": "2048",
        },
    )

    assert config.long_context_config.model == "claude-opus-4-7"
    assert config.thinking_config.thinking_tokens == 4096


def test_build_agent_config_requires_api_key(tmp_path: Path) -> None:
    with pytest.raises(CliConfigError, match="ANTHROPIC_API_KEY"):
        build_agent_config(tmp_path, env={})


def test_build_agent_config_rejects_unknown_model(tmp_path: Path) -> None:
    with pytest.raises(CliConfigError, match="not supported"):
        build_agent_config(tmp_path, CliConfigOverrides(model="claude-not-real"), env={"ANTHROPIC_API_KEY": "key"})


def test_config_summary_excludes_api_keys(tmp_path: Path) -> None:
    config = build_agent_config(tmp_path, env={"ANTHROPIC_API_KEY": "secret-value"})

    summary = config_summary(config)

    assert summary["long_model"] == DEFAULT_LONG_MODEL
    assert "secret-value" not in str(summary)
    assert "api_key" not in str(summary).lower()


def test_build_agent_config_uses_stored_kimi_for_all_model_slots(tmp_path: Path) -> None:
    settings = CliSettings(active_provider=UI_DEFAULT_PROVIDER, active_model=UI_DEFAULT_MODEL)
    settings.set_api_key(UI_DEFAULT_PROVIDER, "moonshot-key")

    config = build_agent_config(tmp_path, settings=settings, env={})

    assert config.long_context_config.provider == ModelProvider.MOONSHOT
    assert config.long_context_config.model == UI_DEFAULT_MODEL
    assert config.fast_config.provider == ModelProvider.MOONSHOT
    assert config.fast_config.model == UI_DEFAULT_MODEL
    assert config.edit_model_config.provider == ModelProvider.MOONSHOT
    assert config.edit_model_config.model == UI_DEFAULT_MODEL
    assert config.thinking_config.provider == ModelProvider.MOONSHOT
    assert config.thinking_config.model == UI_DEFAULT_MODEL
    assert config.moonshot_api_key == "moonshot-key"


def test_build_agent_config_uses_stored_deepseek_for_all_model_slots(tmp_path: Path) -> None:
    settings = CliSettings(active_provider=ModelProvider.DEEPSEEK.value, active_model=DEEPSEEK_DEFAULT_MODEL)
    settings.set_api_key(ModelProvider.DEEPSEEK.value, "deepseek-key")

    config = build_agent_config(tmp_path, settings=settings, env={})

    assert config.long_context_config.provider == ModelProvider.DEEPSEEK
    assert config.long_context_config.model == DEEPSEEK_DEFAULT_MODEL
    assert config.fast_config.provider == ModelProvider.DEEPSEEK
    assert config.fast_config.model == DEEPSEEK_DEFAULT_MODEL
    assert config.edit_model_config.provider == ModelProvider.DEEPSEEK
    assert config.edit_model_config.model == DEEPSEEK_DEFAULT_MODEL
    assert config.thinking_config.provider == ModelProvider.DEEPSEEK
    assert config.thinking_config.model == DEEPSEEK_DEFAULT_MODEL
    assert config.deepseek_api_key == "deepseek-key"


def test_build_agent_config_accepts_moonshot_cli_active_model(tmp_path: Path) -> None:
    config = build_agent_config(
        tmp_path,
        CliConfigOverrides(provider=UI_DEFAULT_PROVIDER, model=UI_DEFAULT_MODEL),
        env={"MOONSHOT_API_KEY": "moonshot-key"},
    )

    assert config.long_context_config.provider == ModelProvider.MOONSHOT
    assert config.fast_config.provider == ModelProvider.MOONSHOT
    assert config.edit_model_config.provider == ModelProvider.MOONSHOT
    assert config.thinking_config.provider == ModelProvider.MOONSHOT


def test_build_agent_config_accepts_deepseek_cli_active_model(tmp_path: Path) -> None:
    config = build_agent_config(
        tmp_path,
        CliConfigOverrides(provider=ModelProvider.DEEPSEEK.value, model=DEEPSEEK_DEFAULT_MODEL),
        env={"DEEPSEEK_API_KEY": "deepseek-key"},
    )

    assert config.long_context_config.provider == ModelProvider.DEEPSEEK
    assert config.fast_config.provider == ModelProvider.DEEPSEEK
    assert config.edit_model_config.provider == ModelProvider.DEEPSEEK
    assert config.thinking_config.provider == ModelProvider.DEEPSEEK


def test_env_provider_model_overrides_stored_settings(tmp_path: Path) -> None:
    settings = CliSettings(active_provider=UI_DEFAULT_PROVIDER, active_model=UI_DEFAULT_MODEL)
    settings.set_api_key(UI_DEFAULT_PROVIDER, "moonshot-key")

    config = build_agent_config(
        tmp_path,
        settings=settings,
        env={
            "ANTHROPIC_API_KEY": "anthropic-key",
            "KOLEGA_CODE_PROVIDER": "anthropic",
            "KOLEGA_CODE_MODEL": "claude-sonnet-4-6",
        },
    )

    assert config.long_context_config.provider == ModelProvider.ANTHROPIC
    assert config.long_context_config.model == "claude-sonnet-4-6"
    assert config.fast_config.provider == ModelProvider.ANTHROPIC
    assert config.fast_config.model == "claude-sonnet-4-6"


def test_stored_kimi_settings_require_moonshot_key(tmp_path: Path) -> None:
    settings = CliSettings(active_provider=UI_DEFAULT_PROVIDER, active_model=UI_DEFAULT_MODEL)

    with pytest.raises(CliConfigError, match="MOONSHOT_API_KEY"):
        build_agent_config(tmp_path, settings=settings, env={})


def test_stored_deepseek_settings_require_deepseek_key(tmp_path: Path) -> None:
    settings = CliSettings(active_provider=ModelProvider.DEEPSEEK.value, active_model=DEEPSEEK_DEFAULT_MODEL)

    with pytest.raises(CliConfigError, match="DEEPSEEK_API_KEY"):
        build_agent_config(tmp_path, settings=settings, env={})
