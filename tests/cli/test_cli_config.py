from pathlib import Path

import pytest

from kolega_code.config import EditProtocol, ModelProvider
from kolega_code.cli.config import (
    CliConfigError,
    CliConfigOverrides,
    build_agent_config,
    config_summary,
    key_status,
)
from kolega_code.cli.provider_registry import (
    DEEPSEEK_DEFAULT_MODEL,
    MOONSHOT_K26_MODEL,
    UI_DEFAULT_MODEL,
    UI_DEFAULT_PROVIDER,
    default_model_for_provider,
)
from kolega_code.cli.settings import CliSettings, SettingsStore
from kolega_code.llm.specs import MODEL_SPECS, default_thinking_effort, is_featured_model

# Anthropic's registry default, used throughout as a convenient known-good model.
ANTHROPIC_DEFAULT_MODEL = default_model_for_provider(ModelProvider.ANTHROPIC)
OLLAMA_CLOUD_DEFAULT_MODEL = default_model_for_provider(ModelProvider.OLLAMA_CLOUD)
OLLAMA_CLOUD_THINKING_MODEL = next(
    model
    for (provider, model), specs in MODEL_SPECS.items()
    if provider == ModelProvider.OLLAMA_CLOUD.value and "thinking_effort" in specs
)


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


def test_build_agent_config_coerces_stale_active_model(tmp_path: Path) -> None:
    # A settings.json pointing at a model that has since been removed (e.g. an old
    # ChatGPT slug) must not brick startup — it falls back to the provider default.
    settings = CliSettings(
        active_provider="openai_chatgpt",
        active_model="gpt-5-codex",  # no longer in MODEL_SPECS
        active_thinking_effort="medium",
    )
    settings.set_oauth_token(
        "openai_chatgpt",
        {"access_token": "at", "refresh_token": "rt", "expires_at": 10**12, "account_id": "a", "plan_type": "pro"},
    )

    config = build_agent_config(tmp_path, env={}, settings=settings)

    assert config.long_context_config.provider == ModelProvider.OPENAI_CHATGPT
    assert config.long_context_config.model == "gpt-5.6-sol"  # coerced from the removed slug


def test_build_agent_config_unknown_saved_provider_is_unconfigured(tmp_path: Path) -> None:
    settings = CliSettings(active_provider="nonexistent-provider", active_model="whatever")
    with pytest.raises(CliConfigError, match="No provider/model configured"):
        build_agent_config(tmp_path, env={}, settings=settings)


def test_build_agent_config_accepts_tinker_checkpoint_paths(tmp_path: Path) -> None:
    checkpoint = "tinker://0034d8c9-0a88-52a9-b2b7-bce7cb1e6fef:train:0/sampler_weights/000080"
    config = build_agent_config(
        tmp_path,
        env={
            "TINKER_API_KEY": "test-key",
            "KOLEGA_CODE_PROVIDER": "tinker",
            "KOLEGA_CODE_MODEL": checkpoint,
        },
    )

    assert config.long_context_config.provider == ModelProvider.TINKER
    assert config.long_context_config.model == checkpoint
    assert config.long_context_config.thinking_effort is None  # wildcard spec has no effort


def test_build_agent_config_rejects_unknown_tinker_base_with_refresh_hint(tmp_path: Path) -> None:
    with pytest.raises(CliConfigError, match="models refresh"):
        build_agent_config(
            tmp_path,
            env={
                "TINKER_API_KEY": "test-key",
                "KOLEGA_CODE_PROVIDER": "tinker",
                "KOLEGA_CODE_MODEL": "Qwen/Qwen9.9-99B",  # not in the bundled catalog
            },
        )


def test_build_agent_config_explicit_provider_uses_provider_default_model(tmp_path: Path) -> None:
    config = build_agent_config(
        tmp_path,
        env={
            "ANTHROPIC_API_KEY": "test-key",
            "KOLEGA_CODE_PROVIDER": "anthropic",
            "KOLEGA_CODE_MODEL": "claude-opus-5",
        },
    )

    # A provider named without a model resolves through the provider registry, and
    # every slot then inherits that active model.
    anthropic_default = default_model_for_provider(ModelProvider.ANTHROPIC)
    assert config.long_context_config.provider == ModelProvider.ANTHROPIC
    assert anthropic_default == "claude-opus-5"
    assert config.long_context_config.model == anthropic_default
    assert config.fast_config.model == anthropic_default
    assert config.long_context_config.thinking_effort == "medium"


def test_build_agent_config_env_overrides(tmp_path: Path) -> None:
    config = build_agent_config(
        tmp_path,
        env={
            "ANTHROPIC_API_KEY": "test-key",
            "KOLEGA_CODE_PROVIDER": "anthropic",
            "KOLEGA_CODE_MODEL": "claude-sonnet-4-6",
            "KOLEGA_CODE_THINKING_EFFORT": "high",
        },
    )

    assert config.long_context_config.model == "claude-sonnet-4-6"
    assert config.long_context_config.thinking_effort == "high"


def test_build_agent_config_flags_override_env(tmp_path: Path) -> None:
    config = build_agent_config(
        tmp_path,
        CliConfigOverrides(provider="anthropic", model="claude-opus-4-7", thinking_effort="xhigh"),
        env={
            "ANTHROPIC_API_KEY": "test-key",
            "KOLEGA_CODE_MODEL": "claude-sonnet-4-6",
            "KOLEGA_CODE_THINKING_EFFORT": "high",
        },
    )

    assert config.long_context_config.model == "claude-opus-4-7"
    assert config.long_context_config.thinking_effort == "xhigh"


def test_build_agent_config_edit_protocol_flag_overrides_env(tmp_path: Path) -> None:
    config = build_agent_config(
        tmp_path,
        CliConfigOverrides(provider="anthropic", model="claude-opus-4-7", edit_protocol="codex_apply_patch"),
        env={
            "ANTHROPIC_API_KEY": "test-key",
            "KOLEGA_CODE_EDIT_PROTOCOL": "search_replace",
        },
    )

    assert config.edit_protocol == EditProtocol.CODEX_APPLY_PATCH


def test_build_agent_config_leaves_protocol_unset_for_catalog_resolution(tmp_path: Path) -> None:
    config = build_agent_config(
        tmp_path,
        CliConfigOverrides(provider="anthropic", model="claude-opus-4-7"),
        env={"ANTHROPIC_API_KEY": "test-key"},
    )

    summary = config_summary(config)

    assert config.edit_protocol is None
    assert summary["edit_protocol"] == "claude_code"
    assert summary["edit_protocol_source"] == "default"


def test_build_agent_config_rejects_unknown_edit_protocol(tmp_path: Path) -> None:
    with pytest.raises(CliConfigError, match="Unsupported edit protocol"):
        build_agent_config(
            tmp_path,
            CliConfigOverrides(provider="anthropic", model="claude-opus-4-7", edit_protocol="not-real"),
            env={"ANTHROPIC_API_KEY": "test-key"},
        )


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
            CliConfigOverrides(provider="anthropic", model="claude-opus-4-7", thinking_effort="auto"),
            env={"ANTHROPIC_API_KEY": "test-key"},
        )


def test_build_agent_config_requires_api_key(tmp_path: Path) -> None:
    with pytest.raises(CliConfigError, match="ANTHROPIC_API_KEY"):
        build_agent_config(tmp_path, CliConfigOverrides(provider="anthropic", model=ANTHROPIC_DEFAULT_MODEL), env={})


def test_build_agent_config_rejects_unknown_model(tmp_path: Path) -> None:
    with pytest.raises(CliConfigError, match="not available"):
        build_agent_config(
            tmp_path,
            CliConfigOverrides(provider="anthropic", model="claude-not-real"),
            env={"ANTHROPIC_API_KEY": "key"},
        )


def test_model_without_a_provider_is_rejected(tmp_path: Path) -> None:
    """Provider and model are both required; neither is inferred nor defaulted."""
    with pytest.raises(CliConfigError, match="also needs a provider"):
        build_agent_config(tmp_path, CliConfigOverrides(model="claude-sonnet-5"), env={"ANTHROPIC_API_KEY": "key"})

    # Unambiguous ids get no special treatment — the error names the candidates instead.
    with pytest.raises(CliConfigError, match="offered by: anthropic"):
        build_agent_config(tmp_path, env={"ANTHROPIC_API_KEY": "key", "KOLEGA_CODE_MODEL": "claude-sonnet-5"})

    config = build_agent_config(
        tmp_path,
        env={"ANTHROPIC_API_KEY": "key", "KOLEGA_CODE_PROVIDER": "anthropic", "KOLEGA_CODE_MODEL": "claude-sonnet-5"},
    )
    assert config.long_context_config.provider == ModelProvider.ANTHROPIC
    assert config.long_context_config.model == "claude-sonnet-5"


def test_provider_without_a_model_is_rejected(tmp_path: Path) -> None:
    """Provider and model go together everywhere: active model, slots, and agent roles."""
    with pytest.raises(CliConfigError, match="also needs a model"):
        build_agent_config(tmp_path, CliConfigOverrides(provider="anthropic"), env={"ANTHROPIC_API_KEY": "key"})

    settings = CliSettings(active_provider="anthropic", active_model=ANTHROPIC_DEFAULT_MODEL)
    settings.set_api_key("anthropic", "key")
    settings.set_api_key("deepseek", "key")

    with pytest.raises(CliConfigError, match=r"--fast-provider=deepseek also needs a model"):
        build_agent_config(tmp_path, CliConfigOverrides(fast_provider="deepseek"), settings=settings, env={})

    with pytest.raises(CliConfigError, match=r"KOLEGA_CODE_PLANNING_PROVIDER=deepseek also needs a model"):
        build_agent_config(tmp_path, settings=settings, env={"KOLEGA_CODE_PLANNING_PROVIDER": "deepseek"})


def test_config_summary_excludes_api_keys(tmp_path: Path) -> None:
    config = build_agent_config(
        tmp_path,
        env={
            "ANTHROPIC_API_KEY": "secret-value",
            "KOLEGA_CODE_PROVIDER": "anthropic",
            "KOLEGA_CODE_MODEL": "claude-opus-5",
        },
    )

    summary = config_summary(config)

    assert summary["long_model"] == ANTHROPIC_DEFAULT_MODEL
    assert "secret-value" not in str(summary)
    assert "api_key" not in str(summary).lower()


def test_project_dotenv_openai_key_is_ignored_when_settings_choose_moonshot(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("OPENAI_API_KEY=project-openai-key\n", encoding="utf-8")
    settings = CliSettings(active_provider=UI_DEFAULT_PROVIDER, active_model=UI_DEFAULT_MODEL)
    settings.set_api_key(UI_DEFAULT_PROVIDER, "moonshot-key")

    config = build_agent_config(tmp_path, settings=settings, env={})

    assert config.long_context_config.provider == ModelProvider.MOONSHOT
    assert config.long_context_config.model == UI_DEFAULT_MODEL
    assert config.openai_api_key is None
    assert config.moonshot_api_key == "moonshot-key"


def test_project_dotenv_api_key_alone_does_not_configure_model(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("OPENAI_API_KEY=project-openai-key\n", encoding="utf-8")

    with pytest.raises(CliConfigError, match="No provider/model configured"):
        build_agent_config(tmp_path, env={})


def test_project_dotenv_model_selection_is_ignored(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=project-openai-key\nKOLEGA_CODE_PROVIDER=openai\nKOLEGA_CODE_MODEL=gpt-5.5\n",
        encoding="utf-8",
    )
    settings = CliSettings(active_provider=UI_DEFAULT_PROVIDER, active_model=UI_DEFAULT_MODEL)
    settings.set_api_key(UI_DEFAULT_PROVIDER, "moonshot-key")

    config = build_agent_config(tmp_path, settings=settings, env={})

    assert config.long_context_config.provider == ModelProvider.MOONSHOT
    assert config.long_context_config.model == UI_DEFAULT_MODEL
    assert config.openai_api_key is None


def test_project_dotenv_model_selection_without_settings_is_unconfigured(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=project-openai-key\nKOLEGA_CODE_PROVIDER=openai\nKOLEGA_CODE_MODEL=gpt-5.5\n",
        encoding="utf-8",
    )

    with pytest.raises(CliConfigError, match="No provider/model configured"):
        build_agent_config(tmp_path, env={})


def test_process_openai_key_and_model_selection_still_work(tmp_path: Path) -> None:
    config = build_agent_config(
        tmp_path,
        env={
            "OPENAI_API_KEY": "process-openai-key",
            "KOLEGA_CODE_PROVIDER": ModelProvider.OPENAI.value,
            "KOLEGA_CODE_MODEL": "gpt-5.5",
        },
    )

    assert config.long_context_config.provider == ModelProvider.OPENAI
    assert config.long_context_config.model == "gpt-5.5"
    assert config.openai_api_key == "process-openai-key"


def test_explicit_model_override_rejects_mismatched_provider(tmp_path: Path) -> None:
    settings = CliSettings(active_provider=UI_DEFAULT_PROVIDER, active_model=UI_DEFAULT_MODEL)
    settings.set_api_key(UI_DEFAULT_PROVIDER, "moonshot-key")

    with pytest.raises(CliConfigError, match=r"KOLEGA_CODE_MODEL=gpt-5\.5 is not available"):
        build_agent_config(
            tmp_path,
            settings=settings,
            env={"KOLEGA_CODE_MODEL": "gpt-5.5", "KOLEGA_CODE_PROVIDER": UI_DEFAULT_PROVIDER},
        )


@pytest.mark.parametrize(
    ("provider_key", "model_key"),
    [
        ("KOLEGA_CODE_FAST_PROVIDER", "KOLEGA_CODE_FAST_MODEL"),
    ],
)
def test_explicit_slot_model_override_rejects_mismatched_provider(
    tmp_path: Path, provider_key: str, model_key: str
) -> None:
    settings = CliSettings(active_provider=UI_DEFAULT_PROVIDER, active_model=UI_DEFAULT_MODEL)
    settings.set_api_key(UI_DEFAULT_PROVIDER, "moonshot-key")

    with pytest.raises(CliConfigError, match=rf"{model_key}=gpt-5\.5 is not available"):
        build_agent_config(tmp_path, settings=settings, env={provider_key: UI_DEFAULT_PROVIDER, model_key: "gpt-5.5"})


def test_explicit_agent_model_override_rejects_mismatched_provider(tmp_path: Path) -> None:
    settings = CliSettings(active_provider=UI_DEFAULT_PROVIDER, active_model=UI_DEFAULT_MODEL)
    settings.set_api_key(UI_DEFAULT_PROVIDER, "moonshot-key")

    with pytest.raises(CliConfigError, match=r"KOLEGA_CODE_INVESTIGATION_MODEL=gpt-5\.5 is not available"):
        build_agent_config(
            tmp_path,
            settings=settings,
            env={
                "KOLEGA_CODE_INVESTIGATION_PROVIDER": UI_DEFAULT_PROVIDER,
                "KOLEGA_CODE_INVESTIGATION_MODEL": "gpt-5.5",
            },
        )


def test_build_agent_config_uses_stored_kimi_for_model_slots(tmp_path: Path) -> None:
    settings = CliSettings(active_provider=UI_DEFAULT_PROVIDER, active_model=UI_DEFAULT_MODEL)
    settings.set_api_key(UI_DEFAULT_PROVIDER, "moonshot-key")

    config = build_agent_config(tmp_path, settings=settings, env={})

    assert config.long_context_config.provider == ModelProvider.MOONSHOT
    assert config.long_context_config.model == UI_DEFAULT_MODEL
    assert config.fast_config.provider == ModelProvider.MOONSHOT
    assert config.fast_config.model == UI_DEFAULT_MODEL
    assert config.long_context_config.thinking_effort == "max"
    assert config.moonshot_api_key == "moonshot-key"


def test_build_agent_config_uses_stored_deepseek_for_model_slots(tmp_path: Path) -> None:
    settings = CliSettings(active_provider=ModelProvider.DEEPSEEK.value, active_model=DEEPSEEK_DEFAULT_MODEL)
    settings.set_api_key(ModelProvider.DEEPSEEK.value, "deepseek-key")

    config = build_agent_config(tmp_path, settings=settings, env={})

    assert config.long_context_config.provider == ModelProvider.DEEPSEEK
    assert config.long_context_config.model == DEEPSEEK_DEFAULT_MODEL
    assert config.fast_config.provider == ModelProvider.DEEPSEEK
    assert config.fast_config.model == DEEPSEEK_DEFAULT_MODEL
    assert config.long_context_config.thinking_effort == "high"
    assert config.deepseek_api_key == "deepseek-key"


def test_build_agent_config_accepts_moonshot_cli_active_model(tmp_path: Path) -> None:
    config = build_agent_config(
        tmp_path,
        CliConfigOverrides(provider=UI_DEFAULT_PROVIDER, model=UI_DEFAULT_MODEL),
        env={"MOONSHOT_API_KEY": "moonshot-key"},
    )

    assert config.long_context_config.provider == ModelProvider.MOONSHOT
    assert config.fast_config.provider == ModelProvider.MOONSHOT


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


def test_build_agent_config_accepts_ollama_cloud_cli_active_model(tmp_path: Path) -> None:
    config = build_agent_config(
        tmp_path,
        CliConfigOverrides(provider=ModelProvider.OLLAMA_CLOUD.value, model=OLLAMA_CLOUD_THINKING_MODEL),
        env={"OLLAMA_API_KEY": "ollama-key"},
    )

    assert config.long_context_config.provider == ModelProvider.OLLAMA_CLOUD
    assert config.long_context_config.model == OLLAMA_CLOUD_THINKING_MODEL
    assert config.long_context_config.thinking_effort == default_thinking_effort(
        ModelProvider.OLLAMA_CLOUD.value,
        OLLAMA_CLOUD_THINKING_MODEL,
    )
    assert config.fast_config.provider == ModelProvider.OLLAMA_CLOUD
    assert config.ollama_cloud_api_key == "ollama-key"


def test_build_agent_config_ollama_cloud_provider_default_is_accessible_model(tmp_path: Path) -> None:
    # The CLI never defaults a model, but the registry default still has to be a model
    # that actually resolves — it backs stale-model recovery and the onboarding picker.
    assert (ModelProvider.OLLAMA_CLOUD.value, OLLAMA_CLOUD_DEFAULT_MODEL) in MODEL_SPECS
    config = build_agent_config(
        tmp_path,
        CliConfigOverrides(provider=ModelProvider.OLLAMA_CLOUD.value, model=OLLAMA_CLOUD_DEFAULT_MODEL),
        env={"OLLAMA_API_KEY": "ollama-key"},
    )

    assert config.long_context_config.provider == ModelProvider.OLLAMA_CLOUD
    assert config.long_context_config.model == OLLAMA_CLOUD_DEFAULT_MODEL
    assert config.fast_config.model == OLLAMA_CLOUD_DEFAULT_MODEL


def test_ollama_cloud_requires_ollama_api_key(tmp_path: Path) -> None:
    with pytest.raises(CliConfigError, match="OLLAMA_API_KEY"):
        build_agent_config(
            tmp_path,
            CliConfigOverrides(provider=ModelProvider.OLLAMA_CLOUD.value, model=OLLAMA_CLOUD_DEFAULT_MODEL),
            env={},
        )


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
    settings = CliSettings(active_provider="anthropic", active_model=ANTHROPIC_DEFAULT_MODEL)
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
    assert config.model_config_for_agent("coder").model == ANTHROPIC_DEFAULT_MODEL


def test_env_overrides_settings_agent_model(tmp_path: Path) -> None:
    settings = CliSettings(active_provider="anthropic", active_model=ANTHROPIC_DEFAULT_MODEL)
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
            "KOLEGA_CODE_MODEL": "claude-opus-5",
            "KOLEGA_CODE_INVESTIGATION_PROVIDER": "deepseek",
            "KOLEGA_CODE_INVESTIGATION_MODEL": "deepseek-v4-flash",
        },
    )

    assert config.model_config_for_agent("investigation-agent").provider == ModelProvider.DEEPSEEK
    assert config.model_config_for_agent("investigation-agent").model == "deepseek-v4-flash"
    assert config.model_config_for_agent("coder").model == ANTHROPIC_DEFAULT_MODEL


def test_agent_model_override_requires_api_key(tmp_path: Path) -> None:
    settings = CliSettings(active_provider="anthropic", active_model=ANTHROPIC_DEFAULT_MODEL)
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
        tmp_path,
        env={
            "ANTHROPIC_API_KEY": "anthropic-key",
            "KOLEGA_CODE_PROVIDER": "anthropic",
            "KOLEGA_CODE_MODEL": ANTHROPIC_DEFAULT_MODEL,
        },
    )

    assert config.agent_models == {}


def test_web_search_defaults_to_keyless_duckduckgo(tmp_path: Path) -> None:
    config = build_agent_config(
        tmp_path,
        env={
            "ANTHROPIC_API_KEY": "anthropic-key",
            "KOLEGA_CODE_PROVIDER": "anthropic",
            "KOLEGA_CODE_MODEL": ANTHROPIC_DEFAULT_MODEL,
        },
    )

    assert config.web_search_backend == "duckduckgo"
    assert config.web_search_api_key is None
    assert config.web_search_base_url is None


def test_web_search_backend_key_from_settings(tmp_path: Path) -> None:
    settings = CliSettings(
        active_provider="anthropic", active_model=ANTHROPIC_DEFAULT_MODEL, web_search_backend="firecrawl"
    )
    settings.set_api_key("anthropic", "anthropic-key")
    settings.set_api_key("firecrawl", "fc-from-settings")

    config = build_agent_config(tmp_path, settings=settings, env={"ANTHROPIC_API_KEY": "anthropic-key"})

    assert config.web_search_backend == "firecrawl"
    assert config.web_search_api_key == "fc-from-settings"


def test_web_search_key_env_overrides_settings(tmp_path: Path) -> None:
    settings = CliSettings(
        active_provider="anthropic", active_model=ANTHROPIC_DEFAULT_MODEL, web_search_backend="firecrawl"
    )
    settings.set_api_key("anthropic", "anthropic-key")
    settings.set_api_key("firecrawl", "fc-from-settings")

    config = build_agent_config(
        tmp_path,
        settings=settings,
        env={"ANTHROPIC_API_KEY": "anthropic-key", "FIRECRAWL_API_KEY": "fc-from-env"},
    )

    assert config.web_search_api_key == "fc-from-env"


def test_web_search_cloud_backend_without_key_does_not_block_startup(tmp_path: Path) -> None:
    settings = CliSettings(
        active_provider="anthropic", active_model=ANTHROPIC_DEFAULT_MODEL, web_search_backend="tavily"
    )
    settings.set_api_key("anthropic", "anthropic-key")

    # Selecting a cloud backend without its key must NOT raise (keyless-default promise).
    config = build_agent_config(tmp_path, settings=settings, env={"ANTHROPIC_API_KEY": "anthropic-key"})

    assert config.web_search_backend == "tavily"
    assert config.web_search_api_key is None


def test_web_search_backend_and_base_url_from_env(tmp_path: Path) -> None:
    config = build_agent_config(
        tmp_path,
        env={
            "ANTHROPIC_API_KEY": "anthropic-key",
            "KOLEGA_CODE_PROVIDER": "anthropic",
            "KOLEGA_CODE_MODEL": "claude-opus-5",
            "KOLEGA_CODE_WEB_SEARCH_BACKEND": "searxng",
            "SEARXNG_BASE_URL": "https://searx.example",
        },
    )

    assert config.web_search_backend == "searxng"
    assert config.web_search_base_url == "https://searx.example"


# --- openrouter gateway ---------------------------------------------------


def test_build_agent_config_openrouter_provider_default(tmp_path: Path) -> None:
    assert default_model_for_provider(ModelProvider.OPENROUTER) == "moonshotai/kimi-k3"
    config = build_agent_config(
        tmp_path,
        CliConfigOverrides(provider=ModelProvider.OPENROUTER.value, model="moonshotai/kimi-k3"),
        env={"OPENROUTER_API_KEY": "sk-or-key"},
    )

    assert config.long_context_config.provider == ModelProvider.OPENROUTER
    assert config.long_context_config.model == "moonshotai/kimi-k3"
    assert config.fast_config.provider == ModelProvider.OPENROUTER
    assert config.openrouter_api_key == "sk-or-key"


def test_build_agent_config_accepts_a_non_featured_openrouter_model(tmp_path: Path) -> None:
    # The picker lists only the most-used models, but any catalogued id resolves.
    model = "deepseek/deepseek-v3.2"
    assert not is_featured_model(ModelProvider.OPENROUTER.value, model)

    config = build_agent_config(
        tmp_path,
        CliConfigOverrides(provider=ModelProvider.OPENROUTER.value, model=model),
        env={"OPENROUTER_API_KEY": "sk-or-key"},
    )

    assert config.long_context_config.model == model


def test_openrouter_model_paired_with_the_wrong_provider_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(CliConfigError, match="openrouter"):
        build_agent_config(
            tmp_path,
            CliConfigOverrides(provider=ModelProvider.ANTHROPIC.value, model="z-ai/glm-5.2"),
            env={"ANTHROPIC_API_KEY": "anthropic-key"},
        )


def test_openrouter_requires_openrouter_api_key(tmp_path: Path) -> None:
    with pytest.raises(CliConfigError, match="OPENROUTER_API_KEY"):
        build_agent_config(
            tmp_path,
            CliConfigOverrides(provider=ModelProvider.OPENROUTER.value, model="moonshotai/kimi-k3"),
            env={},
        )


def test_saved_non_featured_openrouter_model_survives_config_building(tmp_path: Path) -> None:
    # _coerce_known_model must not "repair" a valid-but-unlisted gateway model.
    settings = CliSettings(
        active_provider=ModelProvider.OPENROUTER.value,
        active_model="deepseek/deepseek-v3.2",
    )

    config = build_agent_config(tmp_path, env={"OPENROUTER_API_KEY": "sk-or-key"}, settings=settings)

    assert config.long_context_config.model == "deepseek/deepseek-v3.2"


def test_openrouter_edit_protocol_defaults_to_claude_code_except_openai_models(tmp_path: Path) -> None:
    config = build_agent_config(
        tmp_path,
        CliConfigOverrides(provider=ModelProvider.OPENROUTER.value, model="z-ai/glm-5.2"),
        env={"OPENROUTER_API_KEY": "sk-or-key"},
    )
    assert config.resolve_edit_protocol() == EditProtocol.CLAUDE_CODE

    config = build_agent_config(
        tmp_path,
        CliConfigOverrides(provider=ModelProvider.OPENROUTER.value, model="openai/gpt-5.6-sol"),
        env={"OPENROUTER_API_KEY": "sk-or-key"},
    )
    assert config.resolve_edit_protocol() == EditProtocol.CODEX_APPLY_PATCH


def test_build_agent_config_applies_saved_model_slots(tmp_path: Path) -> None:
    settings = _anthropic_settings_with_deepseek_key()
    settings.set_model_slot("fast", "deepseek", "deepseek-v4-flash")

    config = build_agent_config(tmp_path, settings=settings, env={})

    assert config.fast_config.provider == ModelProvider.DEEPSEEK
    assert config.fast_config.model == "deepseek-v4-flash"
    # The main model is untouched by a slot override.
    assert config.long_context_config.provider == ModelProvider.ANTHROPIC
    assert config.long_context_config.model == ANTHROPIC_DEFAULT_MODEL


def test_model_slots_inherit_the_active_model_when_unset(tmp_path: Path) -> None:
    settings = CliSettings(active_provider="anthropic", active_model=ANTHROPIC_DEFAULT_MODEL)
    settings.set_api_key("anthropic", "anthropic-key")

    config = build_agent_config(tmp_path, settings=settings, env={})

    assert config.fast_config.model == ANTHROPIC_DEFAULT_MODEL


def test_env_fast_model_beats_a_saved_model_slot(tmp_path: Path) -> None:
    settings = _anthropic_settings_with_deepseek_key()
    settings.set_model_slot("fast", "deepseek", "deepseek-v4-flash")

    config = build_agent_config(
        tmp_path,
        settings=settings,
        env={"KOLEGA_CODE_FAST_PROVIDER": "anthropic", "KOLEGA_CODE_FAST_MODEL": "claude-haiku-4-5-20251001"},
    )

    assert config.fast_config.provider == ModelProvider.ANTHROPIC
    assert config.fast_config.model == "claude-haiku-4-5-20251001"


def test_fast_model_flag_beats_env_and_saved_model_slot(tmp_path: Path) -> None:
    settings = _anthropic_settings_with_deepseek_key()
    settings.set_model_slot("fast", "deepseek", "deepseek-v4-flash")

    config = build_agent_config(
        tmp_path,
        CliConfigOverrides(fast_provider="anthropic", fast_model="claude-haiku-4-5-20251001"),
        settings=settings,
        env={"KOLEGA_CODE_FAST_PROVIDER": "deepseek", "KOLEGA_CODE_FAST_MODEL": "deepseek-v4-pro"},
    )

    assert config.fast_config.provider == ModelProvider.ANTHROPIC
    assert config.fast_config.model == "claude-haiku-4-5-20251001"


def test_saved_model_slot_with_an_uncatalogued_model_falls_back_to_the_provider_default(tmp_path: Path) -> None:
    # A model that leaves the catalog must not lock the user out of Settings.
    settings = _anthropic_settings_with_deepseek_key()
    settings.set_model_slot("fast", "deepseek", "retired-from-the-catalog")

    config = build_agent_config(tmp_path, settings=settings, env={})

    assert config.fast_config.provider == ModelProvider.DEEPSEEK
    assert config.fast_config.model == DEEPSEEK_DEFAULT_MODEL


def test_model_slot_override_requires_the_slot_providers_api_key(tmp_path: Path) -> None:
    settings = CliSettings(active_provider="anthropic", active_model=ANTHROPIC_DEFAULT_MODEL)
    settings.set_api_key("anthropic", "anthropic-key")
    settings.set_model_slot("fast", "deepseek", "deepseek-v4-flash")

    with pytest.raises(CliConfigError, match="DEEPSEEK_API_KEY"):
        build_agent_config(tmp_path, settings=settings, env={})


# --- custom endpoints -----------------------------------------------------


def _saved_endpoint_settings() -> CliSettings:
    settings = CliSettings(active_provider="custom:lmstudio", active_model="qwen2.5")
    settings.custom_endpoints = {"lmstudio": {"api_style": "openai_chat", "base_url": "http://localhost:1234/v1"}}
    return settings


def test_build_agent_config_with_custom_endpoint(tmp_path: Path) -> None:
    config = build_agent_config(tmp_path, env={}, settings=_saved_endpoint_settings())

    assert config.long_context_config.provider.value == "custom:lmstudio"
    assert config.long_context_config.model == "qwen2.5"
    endpoint = config.custom_endpoint_for(config.long_context_config)
    assert endpoint is not None and endpoint.base_url == "http://localhost:1234/v1"


def test_build_agent_config_custom_endpoint_keyless_passes_key_check(tmp_path: Path) -> None:
    config = build_agent_config(tmp_path, env={}, settings=_saved_endpoint_settings())
    assert config.get_api_key(config.long_context_config.provider) is None


def test_build_agent_config_custom_endpoint_api_key_resolves(tmp_path: Path) -> None:
    settings = _saved_endpoint_settings()
    settings.custom_endpoints["lmstudio"]["api_key"] = "secret"
    config = build_agent_config(tmp_path, env={}, settings=settings)
    assert config.get_api_key(config.long_context_config.provider) == "secret"


def test_build_agent_config_custom_endpoint_agent_and_slot_overrides(tmp_path: Path) -> None:
    settings = _saved_endpoint_settings()
    settings.set_agent_model("investigation", "custom:lmstudio", "qwen2.5")
    settings.set_model_slot("fast", "custom:lmstudio", "qwen2.5")

    config = build_agent_config(tmp_path, env={}, settings=settings)
    assert config.agent_models["investigation"].provider.value == "custom:lmstudio"
    assert config.fast_config.provider.value == "custom:lmstudio"
    assert config.fast_config.model == "qwen2.5"


def test_build_agent_config_deleted_endpoint_degrades_to_unconfigured(tmp_path: Path) -> None:
    settings = CliSettings(active_provider="custom:gone", active_model="m")
    with pytest.raises(CliConfigError, match="No provider/model configured"):
        build_agent_config(tmp_path, env={}, settings=settings)


def test_build_agent_config_deleted_endpoint_slot_inherits(tmp_path: Path) -> None:
    settings = _saved_endpoint_settings()
    settings.set_model_slot("fast", "custom:gone", "m")
    config = build_agent_config(tmp_path, env={}, settings=settings)
    assert config.fast_config.provider.value == "custom:lmstudio"
    assert config.fast_config.model == "qwen2.5"


def test_build_agent_config_explicit_undefined_endpoint_targeted_error(tmp_path: Path) -> None:
    with pytest.raises(CliConfigError, match="Custom endpoint 'nope' is not defined"):
        build_agent_config(
            tmp_path,
            env={},
            settings=CliSettings(),
            overrides=CliConfigOverrides(provider="custom:nope", model="m"),
        )


def test_build_agent_config_endpoint_flags_define_custom_cli(tmp_path: Path) -> None:
    config = build_agent_config(
        tmp_path,
        env={},
        settings=CliSettings(),
        overrides=CliConfigOverrides(
            endpoint_url="http://localhost:11434/v1",
            endpoint_style="openai_chat",
            endpoint_context="32768",
            endpoint_max_output="8192",
            endpoint_thinking="thinking_toggle",
            endpoint_reasoning="auto",
            model="qwen2.5",
        ),
    )

    assert config.long_context_config.provider.value == "custom:cli"
    assert config.long_context_config.model == "qwen2.5"
    endpoint = config.custom_endpoint_for(config.long_context_config)
    assert endpoint is not None
    assert endpoint.base_url == "http://localhost:11434/v1"
    assert endpoint.context_length == 32768
    assert endpoint.max_output_tokens == 8192
    assert endpoint.thinking == {"mode": "thinking_toggle", "options": ["none", "enabled"], "default": "enabled"}


def test_build_agent_config_explicit_provider_wins_over_custom_cli_default(tmp_path: Path) -> None:
    config = build_agent_config(
        tmp_path,
        env={},
        settings=CliSettings(),
        overrides=CliConfigOverrides(endpoint_url="http://localhost:11434/v1", provider="custom:cli", model="qwen2.5"),
    )
    assert config.long_context_config.provider.value == "custom:cli"


def test_build_agent_config_env_json_overrides_saved_endpoint(tmp_path: Path) -> None:
    settings = _saved_endpoint_settings()
    config = build_agent_config(
        tmp_path,
        env={
            "KOLEGA_CODE_CUSTOM_ENDPOINTS": '{"lmstudio": {"api_style": "openai_chat", "base_url": "http://new:2/v1"}}'
        },
        settings=settings,
    )
    endpoint = config.custom_endpoint_for(config.long_context_config)
    assert endpoint is not None and endpoint.base_url == "http://new:2/v1"


def test_build_agent_config_custom_endpoints_never_persisted(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path)
    settings = CliSettings(active_provider="custom:lmstudio", active_model="qwen2.5")
    store.save(settings)

    build_agent_config(
        tmp_path,
        env={},
        settings=store.load(),
        settings_store=store,
        overrides=CliConfigOverrides(endpoint_url="http://localhost:11434/v1", model="qwen2.5"),
    )

    reloaded = store.load()
    assert reloaded.custom_endpoints == {}
    assert reloaded.active_provider == "custom:lmstudio"


def test_build_agent_config_endpoint_flags_require_url(tmp_path: Path) -> None:
    with pytest.raises(CliConfigError, match="require --endpoint-url"):
        build_agent_config(
            tmp_path,
            env={},
            settings=CliSettings(),
            overrides=CliConfigOverrides(endpoint_context="100"),
        )


def test_build_agent_config_invalid_endpoint_style_and_thinking(tmp_path: Path) -> None:
    with pytest.raises(CliConfigError, match="endpoint style"):
        build_agent_config(
            tmp_path,
            env={},
            settings=CliSettings(),
            overrides=CliConfigOverrides(endpoint_url="http://x/v1", endpoint_style="grpc", model="m"),
        )
    with pytest.raises(CliConfigError, match="endpoint-thinking"):
        build_agent_config(
            tmp_path,
            env={},
            settings=CliSettings(),
            overrides=CliConfigOverrides(endpoint_url="http://x/v1", endpoint_thinking="nope", model="m"),
        )


def test_build_agent_config_anthropic_budget_cap_validated(tmp_path: Path) -> None:
    settings = CliSettings(active_provider="custom:ap", active_model="m")
    settings.custom_endpoints = {
        "ap": {
            "api_style": "anthropic",
            "base_url": "http://x:1",
            "max_output_tokens": 8192,
            "thinking": {
                "mode": "anthropic_budget",
                "options": ["none", "low"],
                "default": "low",
                "budgets": {"low": 9000},
            },
        }
    }
    with pytest.raises(CliConfigError, match="budgets must stay below max_output_tokens"):
        build_agent_config(tmp_path, env={}, settings=settings)


def test_build_agent_config_custom_endpoint_requires_defined_endpoint_for_saved_active(tmp_path: Path) -> None:
    settings = CliSettings(active_provider="custom:lmstudio", active_model="qwen2.5")
    with pytest.raises(CliConfigError, match="Custom endpoint 'lmstudio' is not defined"):
        build_agent_config(
            tmp_path,
            env={},
            settings=settings,
            overrides=CliConfigOverrides(provider="custom:lmstudio", model="qwen2.5"),
        )


def test_key_status_reports_custom_endpoint_keys(tmp_path: Path) -> None:
    settings = _saved_endpoint_settings()
    assert key_status("custom:lmstudio", tmp_path, settings) == "key not set (optional)"

    settings.custom_endpoints["lmstudio"]["api_key"] = "secret"
    assert key_status("custom:lmstudio", tmp_path, settings) == "present in local settings"

    assert key_status("custom:gone", tmp_path, settings) == "endpoint not defined"


def test_build_agent_config_endpoint_temperature_flag(tmp_path: Path) -> None:
    config = build_agent_config(
        tmp_path,
        env={},
        settings=CliSettings(),
        overrides=CliConfigOverrides(endpoint_url="http://localhost:11434/v1", endpoint_temperature="0.3", model="m"),
    )
    endpoint = config.custom_endpoint_for(config.long_context_config)
    assert endpoint is not None and endpoint.temperature == 0.3
    from kolega_code.llm.specs import get_model_specs

    assert get_model_specs("custom:cli", "m")["default_temperature"] == 0.3


def test_build_agent_config_endpoint_temperature_validation(tmp_path: Path) -> None:
    for bad in ("0", "2.5", "hot"):
        with pytest.raises(CliConfigError, match="endpoint-temperature"):
            build_agent_config(
                tmp_path,
                env={},
                settings=CliSettings(),
                overrides=CliConfigOverrides(
                    endpoint_url="http://localhost:11434/v1", endpoint_temperature=bad, model="m"
                ),
            )
