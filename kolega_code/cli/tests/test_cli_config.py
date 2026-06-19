from pathlib import Path

import pytest

from kolega_code.config import ModelProvider
from kolega_code.cli.config import (
    DEFAULT_LONG_MODEL,
    CliConfigError,
    CliConfigOverrides,
    build_agent_config,
    config_summary,
)
from kolega_code.cli.provider_registry import (
    DEEPSEEK_DEFAULT_MODEL,
    MOONSHOT_K26_MODEL,
    UI_DEFAULT_MODEL,
    UI_DEFAULT_PROVIDER,
)
from kolega_code.cli.settings import CliSettings


@pytest.mark.parametrize(
    ("api_key_env", "api_key"),
    [
        ("ANTHROPIC_API_KEY", "anthropic-key"),
        ("MOONSHOT_API_KEY", "moonshot-key"),
        ("DEEPSEEK_API_KEY", "deepseek-key"),
    ],
)
def test_build_agent_config_requires_model_selection_even_with_api_key(
    tmp_path: Path, api_key_env: str, api_key: str
) -> None:
    with pytest.raises(CliConfigError, match="No provider/model configured"):
        build_agent_config(tmp_path, env={api_key_env: api_key})


def test_build_agent_config_explicit_provider_uses_provider_default_model(tmp_path: Path) -> None:
    config = build_agent_config(
        tmp_path,
        env={
            "ANTHROPIC_API_KEY": "test-key",
            "KOLEGA_CODE_PROVIDER": "anthropic",
        },
    )

    assert config.long_context_config.provider == ModelProvider.ANTHROPIC
    assert config.long_context_config.model == DEFAULT_LONG_MODEL
    assert config.fast_config.model == DEFAULT_LONG_MODEL
    assert config.thinking_config.model == DEFAULT_LONG_MODEL
    assert config.long_context_config.thinking_effort == "medium"
    assert config.thinking_config.thinking_effort == "medium"


def test_build_agent_config_env_overrides(tmp_path: Path) -> None:
    config = build_agent_config(
        tmp_path,
        env={
            "ANTHROPIC_API_KEY": "test-key",
            "KOLEGA_CODE_MODEL": "claude-sonnet-4-6",
            "KOLEGA_CODE_THINKING_EFFORT": "high",
        },
    )

    assert config.long_context_config.model == "claude-sonnet-4-6"
    assert config.long_context_config.thinking_effort == "high"
    assert config.thinking_config.thinking_effort == "high"


def test_build_agent_config_flags_override_env(tmp_path: Path) -> None:
    config = build_agent_config(
        tmp_path,
        CliConfigOverrides(model="claude-opus-4-7", thinking_effort="xhigh"),
        env={
            "ANTHROPIC_API_KEY": "test-key",
            "KOLEGA_CODE_MODEL": "claude-sonnet-4-6",
            "KOLEGA_CODE_THINKING_EFFORT": "high",
        },
    )

    assert config.long_context_config.model == "claude-opus-4-7"
    assert config.long_context_config.thinking_effort == "xhigh"
    assert config.thinking_config.thinking_effort == "xhigh"


def test_build_agent_config_rejects_deprecated_thinking_tokens_env(tmp_path: Path) -> None:
    with pytest.raises(CliConfigError, match="named effort"):
        build_agent_config(
            tmp_path,
            env={
                "ANTHROPIC_API_KEY": "test-key",
                "KOLEGA_CODE_THINKING_TOKENS": "2048",
            },
        )


def test_build_agent_config_rejects_invalid_thinking_effort(tmp_path: Path) -> None:
    with pytest.raises(CliConfigError, match="Unsupported thinking effort"):
        build_agent_config(
            tmp_path,
            CliConfigOverrides(model="claude-opus-4-7", thinking_effort="auto"),
            env={"ANTHROPIC_API_KEY": "test-key"},
        )


def test_build_agent_config_requires_api_key(tmp_path: Path) -> None:
    with pytest.raises(CliConfigError, match="ANTHROPIC_API_KEY"):
        build_agent_config(tmp_path, CliConfigOverrides(provider="anthropic"), env={})


def test_build_agent_config_rejects_unknown_model(tmp_path: Path) -> None:
    with pytest.raises(CliConfigError, match="not supported"):
        build_agent_config(tmp_path, CliConfigOverrides(model="claude-not-real"), env={"ANTHROPIC_API_KEY": "key"})


def test_config_summary_excludes_api_keys(tmp_path: Path) -> None:
    config = build_agent_config(
        tmp_path,
        env={
            "ANTHROPIC_API_KEY": "secret-value",
            "KOLEGA_CODE_PROVIDER": "anthropic",
        },
    )

    summary = config_summary(config)

    assert summary["long_model"] == DEFAULT_LONG_MODEL
    assert "secret-value" not in str(summary)
    assert "api_key" not in str(summary).lower()


def test_build_agent_config_uses_stored_kimi_for_model_slots(tmp_path: Path) -> None:
    settings = CliSettings(active_provider=UI_DEFAULT_PROVIDER, active_model=UI_DEFAULT_MODEL)
    settings.set_api_key(UI_DEFAULT_PROVIDER, "moonshot-key")

    config = build_agent_config(tmp_path, settings=settings, env={})

    assert config.long_context_config.provider == ModelProvider.MOONSHOT
    assert config.long_context_config.model == UI_DEFAULT_MODEL
    assert config.fast_config.provider == ModelProvider.MOONSHOT
    assert config.fast_config.model == UI_DEFAULT_MODEL
    assert config.thinking_config.provider == ModelProvider.MOONSHOT
    assert config.thinking_config.model == UI_DEFAULT_MODEL
    assert config.long_context_config.thinking_effort == "auto"
    assert config.thinking_config.thinking_effort == "auto"
    assert config.moonshot_api_key == "moonshot-key"


def test_build_agent_config_uses_stored_deepseek_for_model_slots(tmp_path: Path) -> None:
    settings = CliSettings(active_provider=ModelProvider.DEEPSEEK.value, active_model=DEEPSEEK_DEFAULT_MODEL)
    settings.set_api_key(ModelProvider.DEEPSEEK.value, "deepseek-key")

    config = build_agent_config(tmp_path, settings=settings, env={})

    assert config.long_context_config.provider == ModelProvider.DEEPSEEK
    assert config.long_context_config.model == DEEPSEEK_DEFAULT_MODEL
    assert config.fast_config.provider == ModelProvider.DEEPSEEK
    assert config.fast_config.model == DEEPSEEK_DEFAULT_MODEL
    assert config.thinking_config.provider == ModelProvider.DEEPSEEK
    assert config.thinking_config.model == DEEPSEEK_DEFAULT_MODEL
    assert config.long_context_config.thinking_effort == "high"
    assert config.thinking_config.thinking_effort == "high"
    assert config.deepseek_api_key == "deepseek-key"


def test_build_agent_config_accepts_moonshot_cli_active_model(tmp_path: Path) -> None:
    config = build_agent_config(
        tmp_path,
        CliConfigOverrides(provider=UI_DEFAULT_PROVIDER, model=UI_DEFAULT_MODEL),
        env={"MOONSHOT_API_KEY": "moonshot-key"},
    )

    assert config.long_context_config.provider == ModelProvider.MOONSHOT
    assert config.fast_config.provider == ModelProvider.MOONSHOT
    assert config.thinking_config.provider == ModelProvider.MOONSHOT


def test_build_agent_config_accepts_moonshot_k26_model(tmp_path: Path) -> None:
    config = build_agent_config(
        tmp_path,
        CliConfigOverrides(provider=UI_DEFAULT_PROVIDER, model=MOONSHOT_K26_MODEL),
        env={"MOONSHOT_API_KEY": "moonshot-key"},
    )

    assert config.long_context_config.provider == ModelProvider.MOONSHOT
    assert config.long_context_config.model == MOONSHOT_K26_MODEL
    assert config.long_context_config.thinking_effort == "auto"


def test_build_agent_config_accepts_deepseek_cli_active_model(tmp_path: Path) -> None:
    config = build_agent_config(
        tmp_path,
        CliConfigOverrides(provider=ModelProvider.DEEPSEEK.value, model=DEEPSEEK_DEFAULT_MODEL),
        env={"DEEPSEEK_API_KEY": "deepseek-key"},
    )

    assert config.long_context_config.provider == ModelProvider.DEEPSEEK
    assert config.fast_config.provider == ModelProvider.DEEPSEEK
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


def _anthropic_settings_with_deepseek_key() -> CliSettings:
    settings = CliSettings(active_provider="anthropic", active_model=DEFAULT_LONG_MODEL)
    settings.set_api_key("anthropic", "anthropic-key")
    settings.set_api_key("deepseek", "deepseek-key")
    return settings


def test_build_agent_config_applies_settings_agent_model_override(tmp_path: Path) -> None:
    settings = _anthropic_settings_with_deepseek_key()
    settings.set_agent_model("investigation", "deepseek", "deepseek-v4-flash", "high")

    config = build_agent_config(tmp_path, settings=settings, env={})

    investigation = config.model_config_for_agent("investigation-agent")
    assert investigation.provider == ModelProvider.DEEPSEEK
    assert investigation.model == "deepseek-v4-flash"
    assert investigation.thinking_effort == "high"
    # Roles with no override inherit the active (long-context) model.
    assert config.model_config_for_agent("coder").model == DEFAULT_LONG_MODEL


def test_env_overrides_settings_agent_model(tmp_path: Path) -> None:
    settings = CliSettings(active_provider="anthropic", active_model=DEFAULT_LONG_MODEL)
    settings.set_agent_model("investigation", "deepseek", "deepseek-v4-flash")

    config = build_agent_config(
        tmp_path,
        settings=settings,
        env={
            "ANTHROPIC_API_KEY": "anthropic-key",
            "DEEPSEEK_API_KEY": "deepseek-key",
            "KOLEGA_CODE_INVESTIGATION_MODEL": "deepseek-v4-pro",
        },
    )

    assert config.model_config_for_agent("investigation-agent").model == "deepseek-v4-pro"


def test_env_only_agent_model_override(tmp_path: Path) -> None:
    config = build_agent_config(
        tmp_path,
        env={
            "ANTHROPIC_API_KEY": "anthropic-key",
            "DEEPSEEK_API_KEY": "deepseek-key",
            "KOLEGA_CODE_PROVIDER": "anthropic",
            "KOLEGA_CODE_INVESTIGATION_PROVIDER": "deepseek",
            "KOLEGA_CODE_INVESTIGATION_MODEL": "deepseek-v4-flash",
        },
    )

    assert config.model_config_for_agent("investigation-agent").provider == ModelProvider.DEEPSEEK
    assert config.model_config_for_agent("investigation-agent").model == "deepseek-v4-flash"
    assert config.model_config_for_agent("coder").model == DEFAULT_LONG_MODEL


def test_agent_model_override_requires_api_key(tmp_path: Path) -> None:
    settings = CliSettings(active_provider="anthropic", active_model=DEFAULT_LONG_MODEL)
    settings.set_api_key("anthropic", "anthropic-key")
    settings.set_agent_model("investigation", "deepseek", "deepseek-v4-flash")

    with pytest.raises(CliConfigError, match="DEEPSEEK_API_KEY"):
        build_agent_config(tmp_path, settings=settings, env={})


def test_config_summary_includes_agent_models(tmp_path: Path) -> None:
    settings = _anthropic_settings_with_deepseek_key()
    settings.set_agent_model("investigation", "deepseek", "deepseek-v4-flash")

    summary = config_summary(build_agent_config(tmp_path, settings=settings, env={}))

    assert summary["agent_models"] == {"investigation": "deepseek/deepseek-v4-flash"}


def test_build_agent_config_no_agent_model_overrides_by_default(tmp_path: Path) -> None:
    config = build_agent_config(
        tmp_path, env={"ANTHROPIC_API_KEY": "anthropic-key", "KOLEGA_CODE_PROVIDER": "anthropic"}
    )

    assert config.agent_models == {}
