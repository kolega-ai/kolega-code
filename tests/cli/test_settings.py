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
        permission_mode="auto",
    )
    settings.set_api_key(UI_DEFAULT_PROVIDER, "secret-key")

    store.save(settings)

    loaded = store.load()
    assert loaded.active_provider == UI_DEFAULT_PROVIDER
    assert loaded.active_model == UI_DEFAULT_MODEL
    assert loaded.active_thinking_effort == "auto"
    assert loaded.permission_mode == "auto"
    assert loaded.get_api_key(UI_DEFAULT_PROVIDER) == "secret-key"

    if os.name != "nt":
        assert stat.S_IMODE(store.path.stat().st_mode) == 0o600


def test_settings_store_missing_file_returns_empty_settings(tmp_path: Path) -> None:
    settings = SettingsStore(tmp_path).load()

    assert settings.active_provider is None
    assert settings.active_model is None
    assert settings.active_thinking_effort is None
    assert settings.permission_mode == "ask"
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
        provider = ModelProvider(provider_value)
        # OAuth providers (ChatGPT subscription) authenticate via sign-in, not an
        # API-key env var, so they carry no api_key_env.
        if provider in API_KEY_ENV:
            assert option.api_key_env == API_KEY_ENV[provider]
        else:
            assert option.api_key_env == ""

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

    ollama_model = get_ui_model("ollama_cloud", "gpt-oss:20b")
    assert ollama_model is not None
    assert ollama_model.provider_label == "Ollama Cloud"
    assert ollama_model.model_label == "GPT-OSS 20B"
    assert ollama_model.api_key_env == "OLLAMA_API_KEY"
    assert ollama_model.default_thinking_effort == "medium"

    ollama_glm_model = get_ui_model("ollama_cloud", "glm-5.2")
    assert ollama_glm_model is not None
    assert ollama_glm_model.model_label == "GLM-5.2"
    assert ollama_glm_model.default_thinking_effort == "medium"
    assert ui_thinking_effort_options("ollama_cloud", "gpt-oss:120b") == [
        ("None", "none"),
        ("Low", "low"),
        ("Medium", "medium"),
        ("High", "high"),
        ("Max", "max"),
    ]


def test_web_search_settings_round_trip(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path)
    settings = CliSettings(
        active_provider=UI_DEFAULT_PROVIDER,
        active_model=UI_DEFAULT_MODEL,
        web_search_backend="tavily",
        web_search_base_url="https://searx.example",
    )
    settings.set_api_key("tavily", "tvly-secret")

    store.save(settings)
    loaded = store.load()

    assert loaded.web_search_backend == "tavily"
    assert loaded.web_search_base_url == "https://searx.example"
    assert loaded.get_api_key("tavily") == "tvly-secret"


def test_oauth_tokens_round_trip(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path)
    settings = CliSettings(active_provider=UI_DEFAULT_PROVIDER, active_model=UI_DEFAULT_MODEL)
    settings.set_oauth_token(
        "openai_chatgpt",
        {
            "access_token": "at",
            "refresh_token": "rt",
            "id_token": "it",
            "expires_at": 4600.0,
            "account_id": "acct_1",
            "plan_type": "pro",
            "email": "u@example.com",
        },
    )

    store.save(settings)
    loaded = store.load()

    assert loaded.has_oauth_token("openai_chatgpt")
    token = loaded.get_oauth_token("openai_chatgpt")
    assert token is not None
    assert token["access_token"] == "at"
    assert token["plan_type"] == "pro"


def test_clear_oauth_token_signs_out() -> None:
    settings = CliSettings()
    settings.set_oauth_token("openai_chatgpt", {"access_token": "a", "refresh_token": "r"})
    settings.clear_oauth_token("openai_chatgpt")

    assert not settings.has_oauth_token("openai_chatgpt")
    assert settings.oauth_tokens == {}


def test_from_dict_drops_incomplete_oauth_token_entries() -> None:
    data = {
        "schema_version": SETTINGS_SCHEMA_VERSION,
        "oauth_tokens": {
            "openai_chatgpt": {"access_token": "a", "refresh_token": "r"},
            "broken": {"access_token": "a"},  # missing refresh_token -> dropped
            "garbage": "not-a-dict",  # malformed -> dropped
        },
    }

    settings = CliSettings.from_dict(data)

    assert set(settings.oauth_tokens) == {"openai_chatgpt"}


def test_oauth_tokens_absent_in_old_file_default_to_empty() -> None:
    settings = CliSettings.from_dict(
        {
            "schema_version": 3,
            "active_provider": UI_DEFAULT_PROVIDER,
            "active_model": UI_DEFAULT_MODEL,
            "api_keys": {UI_DEFAULT_PROVIDER: "k"},
        }
    )

    assert settings.oauth_tokens == {}


def test_permission_mode_absent_in_old_file_defaults_to_ask() -> None:
    settings = CliSettings.from_dict(
        {
            "schema_version": 3,
            "active_provider": UI_DEFAULT_PROVIDER,
            "active_model": UI_DEFAULT_MODEL,
            "api_keys": {UI_DEFAULT_PROVIDER: "k"},
        }
    )

    assert settings.permission_mode == "ask"


def test_invalid_permission_mode_defaults_to_ask() -> None:
    settings = CliSettings.from_dict(
        {
            "schema_version": 3,
            "active_provider": UI_DEFAULT_PROVIDER,
            "active_model": UI_DEFAULT_MODEL,
            "permission_mode": "dangerously-yolo",
        }
    )

    assert settings.permission_mode == "ask"


def test_web_search_settings_absent_in_old_file_default_to_none() -> None:
    # A v3 file written before web search existed: keys absent -> None (active_theme
    # precedent), and additive fields ship without a schema bump.
    settings = CliSettings.from_dict(
        {
            "schema_version": 3,
            "active_provider": UI_DEFAULT_PROVIDER,
            "active_model": UI_DEFAULT_MODEL,
            "api_keys": {UI_DEFAULT_PROVIDER: "k"},
        }
    )

    assert settings.web_search_backend is None
    assert settings.web_search_base_url is None
    assert SETTINGS_SCHEMA_VERSION == 3
