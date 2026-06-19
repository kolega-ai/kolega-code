import os
import stat
from pathlib import Path

import pytest

from kolega_code.cli.config import API_KEY_ENV
from kolega_code.cli.provider_registry import (
    DEEPSEEK_DEFAULT_MODEL,
    MOONSHOT_K26_MODEL,
    UI_DEFAULT_MODEL,
    UI_DEFAULT_PROVIDER,
    get_ui_model,
    ui_model_options,
    ui_provider_options,
    ui_thinking_effort_options,
)
from kolega_code.cli.settings import (
    SETTINGS_SCHEMA_VERSION,
    CliSettings,
    SettingsStore,
    SettingsStoreError,
)
from kolega_code.config import ModelProvider
from kolega_code.llm.specs import MODEL_SPECS


def test_settings_store_round_trip_and_file_permissions(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path)
    settings = CliSettings(
        active_provider=UI_DEFAULT_PROVIDER,
        active_model=UI_DEFAULT_MODEL,
        active_thinking_effort="auto",
    )
    settings.set_api_key(UI_DEFAULT_PROVIDER, "secret-key")

    store.save(settings)

    loaded = store.load()
    assert loaded.active_provider == UI_DEFAULT_PROVIDER
    assert loaded.active_model == UI_DEFAULT_MODEL
    assert loaded.active_thinking_effort == "auto"
    assert loaded.get_api_key(UI_DEFAULT_PROVIDER) == "secret-key"

    if os.name != "nt":
        assert stat.S_IMODE(store.path.stat().st_mode) == 0o600


def test_settings_store_missing_file_returns_empty_settings(tmp_path: Path) -> None:
    settings = SettingsStore(tmp_path).load()

    assert settings.active_provider is None
    assert settings.active_model is None
    assert settings.active_thinking_effort is None
    assert settings.api_keys == {}
    assert settings.agent_models == {}


def test_settings_store_round_trips_agent_models(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path)
    settings = CliSettings(active_provider=UI_DEFAULT_PROVIDER, active_model=UI_DEFAULT_MODEL)
    settings.set_agent_model("investigation", "deepseek", "deepseek-v4-flash", "high")
    settings.set_agent_model("building", "anthropic", "claude-opus-4-8")

    store.save(settings)
    loaded = store.load()

    assert loaded.get_agent_model("investigation") == {
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "thinking_effort": "high",
    }
    assert loaded.get_agent_model("building") == {"provider": "anthropic", "model": "claude-opus-4-8"}


def test_clear_agent_model_makes_role_inherit() -> None:
    settings = CliSettings()
    settings.set_agent_model("investigation", "deepseek", "deepseek-v4-flash")
    settings.clear_agent_model("investigation")

    assert settings.get_agent_model("investigation") is None
    assert settings.agent_models == {}


def test_from_dict_drops_incomplete_agent_model_entries() -> None:
    data = {
        "schema_version": SETTINGS_SCHEMA_VERSION,
        "agent_models": {
            "investigation": {"provider": "deepseek", "model": "deepseek-v4-flash"},
            "building": {"provider": "anthropic"},  # missing model -> dropped
            "general": "not-a-dict",  # malformed -> dropped
        },
    }

    settings = CliSettings.from_dict(data)

    assert set(settings.agent_models) == {"investigation"}


def test_settings_store_migrates_v1_settings(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path)
    store.root.mkdir(parents=True, exist_ok=True)
    store.path.write_text(
        '{"schema_version": 1, "active_provider": "moonshot", "active_model": "kimi-k2.6", "api_keys": {"moonshot": "key"}}',
        encoding="utf-8",
    )

    settings = store.load()

    assert settings.schema_version == SETTINGS_SCHEMA_VERSION
    assert settings.active_provider == UI_DEFAULT_PROVIDER
    assert settings.active_model == MOONSHOT_K26_MODEL
    assert settings.active_thinking_effort is None
    assert settings.agent_models == {}


def test_settings_store_rejects_corrupt_json(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path)
    store.root.mkdir(parents=True, exist_ok=True)
    store.path.write_text("{bad json", encoding="utf-8")

    with pytest.raises(SettingsStoreError):
        store.load()


def test_ui_provider_registry_is_derived_from_model_specs() -> None:
    # Every model in the central catalog is exposed by the UI registry, with the
    # right API-key env var derived for each one.
    for provider_value, model in MODEL_SPECS:
        option = get_ui_model(provider_value, model)
        assert option is not None, (provider_value, model)
        assert option.api_key_env == API_KEY_ENV[ModelProvider(provider_value)]

    # Every provider that has specs appears in the provider dropdown.
    spec_providers = {provider_value for provider_value, _ in MODEL_SPECS}
    assert {value for _, value in ui_provider_options()} == spec_providers

    # The Moonshot default and its models are present with friendly labels.
    assert ("Moonshot AI", UI_DEFAULT_PROVIDER) in ui_provider_options()
    moonshot_models = dict(ui_model_options(UI_DEFAULT_PROVIDER))
    assert moonshot_models["Kimi K2.7 Code"] == UI_DEFAULT_MODEL
    assert moonshot_models["Kimi K2.6"] == MOONSHOT_K26_MODEL

    # Thinking-effort options still come through from the specs unchanged.
    assert ui_thinking_effort_options(UI_DEFAULT_PROVIDER, UI_DEFAULT_MODEL) == [("Auto", "auto")]
    assert ui_thinking_effort_options(UI_DEFAULT_PROVIDER, MOONSHOT_K26_MODEL) == [
        ("Auto", "auto"),
        ("None", "none"),
    ]
    assert ui_thinking_effort_options("deepseek", DEEPSEEK_DEFAULT_MODEL) == [
        ("None", "none"),
        ("High", "high"),
        ("Max", "max"),
    ]

    default = get_ui_model(UI_DEFAULT_PROVIDER, UI_DEFAULT_MODEL)
    assert default is not None
    assert default.api_key_env == "MOONSHOT_API_KEY"
    assert default.default_thinking_effort == "auto"

    deepseek_model = get_ui_model("deepseek", DEEPSEEK_DEFAULT_MODEL)
    assert deepseek_model is not None
    assert deepseek_model.api_key_env == "DEEPSEEK_API_KEY"
    assert deepseek_model.default_thinking_effort == "high"
