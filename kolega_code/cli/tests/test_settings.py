import os
import stat
from pathlib import Path

import pytest

from kolega_code.cli.provider_registry import (
    DEEPSEEK_DEFAULT_MODEL,
    UI_DEFAULT_MODEL,
    UI_DEFAULT_PROVIDER,
    get_ui_model,
    ui_model_options,
    ui_provider_options,
)
from kolega_code.cli.settings import CliSettings, SettingsStore, SettingsStoreError


def test_settings_store_round_trip_and_file_permissions(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path)
    settings = CliSettings(active_provider=UI_DEFAULT_PROVIDER, active_model=UI_DEFAULT_MODEL)
    settings.set_api_key(UI_DEFAULT_PROVIDER, "secret-key")

    store.save(settings)

    loaded = store.load()
    assert loaded.active_provider == UI_DEFAULT_PROVIDER
    assert loaded.active_model == UI_DEFAULT_MODEL
    assert loaded.get_api_key(UI_DEFAULT_PROVIDER) == "secret-key"

    if os.name != "nt":
        assert stat.S_IMODE(store.path.stat().st_mode) == 0o600


def test_settings_store_missing_file_returns_empty_settings(tmp_path: Path) -> None:
    settings = SettingsStore(tmp_path).load()

    assert settings.active_provider is None
    assert settings.active_model is None
    assert settings.api_keys == {}


def test_settings_store_rejects_corrupt_json(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path)
    store.root.mkdir(parents=True, exist_ok=True)
    store.path.write_text("{bad json", encoding="utf-8")

    with pytest.raises(SettingsStoreError):
        store.load()


def test_ui_provider_registry_supports_kimi_and_deepseek() -> None:
    assert ui_provider_options() == [("Moonshot AI", UI_DEFAULT_PROVIDER), ("DeepSeek AI", "deepseek")]
    assert ui_model_options(UI_DEFAULT_PROVIDER) == [("Kimi K2.6", UI_DEFAULT_MODEL)]
    assert ui_model_options("deepseek") == [("DeepSeek V4 Pro", DEEPSEEK_DEFAULT_MODEL)]

    model = get_ui_model(UI_DEFAULT_PROVIDER, UI_DEFAULT_MODEL)
    assert model is not None
    assert model.api_key_env == "MOONSHOT_API_KEY"

    deepseek_model = get_ui_model("deepseek", DEEPSEEK_DEFAULT_MODEL)
    assert deepseek_model is not None
    assert deepseek_model.api_key_env == "DEEPSEEK_API_KEY"
