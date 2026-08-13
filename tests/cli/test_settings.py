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
    PROVIDER_DEFAULT_MODEL,
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
from kolega_code.llm.specs import MODEL_SPECS, is_featured_model


def test_settings_store_round_trip_and_file_permissions(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path)
    settings = CliSettings(
        active_provider=UI_DEFAULT_PROVIDER,
        active_model=UI_DEFAULT_MODEL,
        active_thinking_effort="max",
        permission_mode="auto",
    )
    settings.set_api_key(UI_DEFAULT_PROVIDER, "secret-key")

    store.save(settings)

    loaded = store.load()
    assert loaded.active_provider == UI_DEFAULT_PROVIDER
    assert loaded.active_model == UI_DEFAULT_MODEL
    assert loaded.active_thinking_effort == "max"
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
    assert settings.eval_enabled is None
    assert settings.subagents_enabled is None
    assert settings.skills_enabled is None


def test_settings_store_round_trips_eval_enabled(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path)
    store.save(CliSettings(eval_enabled=False))

    loaded = store.load()
    assert loaded.eval_enabled is False
    assert loaded.to_dict()["eval_enabled"] is False

    # Absent in older files -> None -> the eval tool stays enabled downstream.
    store.save(CliSettings())
    assert SettingsStore(tmp_path).load().eval_enabled is None


def test_settings_store_round_trips_subagents_enabled(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path)
    store.save(CliSettings(subagents_enabled=False))

    loaded = store.load()
    assert loaded.subagents_enabled is False
    assert loaded.to_dict()["subagents_enabled"] is False

    # Absent in older files -> None -> sub-agent dispatch stays enabled downstream.
    store.save(CliSettings())
    assert SettingsStore(tmp_path).load().subagents_enabled is None


def test_settings_store_round_trips_skills_enabled(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path)
    store.save(CliSettings(skills_enabled=False))

    loaded = store.load()
    assert loaded.skills_enabled is False
    assert loaded.to_dict()["skills_enabled"] is False

    # Absent in older files -> None -> Agent Skills stay enabled downstream.
    data = CliSettings().to_dict()
    del data["skills_enabled"]
    assert CliSettings.from_dict(data).skills_enabled is None


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


def test_settings_store_round_trips_model_slots(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path)
    settings = CliSettings(active_provider=UI_DEFAULT_PROVIDER, active_model=UI_DEFAULT_MODEL)
    settings.set_model_slot("fast", "deepseek", "deepseek-v4-flash")

    store.save(settings)
    loaded = store.load()

    assert loaded.get_model_slot("fast") == {"provider": "deepseek", "model": "deepseek-v4-flash"}


def test_clear_model_slot_makes_slot_inherit() -> None:
    settings = CliSettings()
    settings.set_model_slot("fast", "deepseek", "deepseek-v4-flash")
    settings.clear_model_slot("fast")

    assert settings.get_model_slot("fast") is None
    assert settings.model_slots == {}


def test_from_dict_drops_incomplete_model_slot_entries() -> None:
    data = {
        "schema_version": SETTINGS_SCHEMA_VERSION,
        "model_slots": {
            "fast": {"provider": "deepseek", "model": "deepseek-v4-flash"},
            "fast2": {"provider": "anthropic"},  # missing model -> dropped
        },
    }

    settings = CliSettings.from_dict(data)

    assert set(settings.model_slots) == {"fast"}


def test_from_dict_drops_legacy_thinking_slot() -> None:
    """The thinking model slot was removed; a saved entry is ignored on load."""
    data = {
        "schema_version": SETTINGS_SCHEMA_VERSION,
        "model_slots": {
            "fast": {"provider": "deepseek", "model": "deepseek-v4-flash"},
            "thinking": {"provider": "anthropic", "model": "claude-opus-4-8"},
        },
    }

    settings = CliSettings.from_dict(data)

    assert set(settings.model_slots) == {"fast"}


def test_settings_written_before_model_slots_load_with_none(tmp_path: Path) -> None:
    # Additive optional field: an older file simply has no slot overrides.
    store = SettingsStore(tmp_path)
    store.root.mkdir(parents=True, exist_ok=True)
    store.path.write_text(
        f'{{"schema_version": {SETTINGS_SCHEMA_VERSION}, "active_provider": "anthropic", "active_model": "claude-opus-5"}}',
        encoding="utf-8",
    )

    assert store.load().model_slots == {}


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

    # Every provider that has specs appears in the provider dropdown, plus any
    # custom endpoints registered for this process (via CUSTOM_PROVIDER_LABELS).
    spec_providers = {provider_value for provider_value, _ in MODEL_SPECS}
    option_providers = {value for _, value in ui_provider_options()}
    assert spec_providers <= option_providers
    for provider_value in option_providers - spec_providers:
        assert provider_value.startswith("custom:")

    # The Moonshot default and its models are present with friendly labels.
    assert ("Moonshot AI", UI_DEFAULT_PROVIDER) in ui_provider_options()
    moonshot_models = dict(ui_model_options(UI_DEFAULT_PROVIDER))
    assert moonshot_models["Kimi K3"] == UI_DEFAULT_MODEL
    assert moonshot_models["Kimi K2.7 Code"] == "kimi-k2.7-code"
    assert moonshot_models["Kimi K2.6"] == MOONSHOT_K26_MODEL

    # Thinking-effort options still come through from the specs unchanged.
    assert ui_thinking_effort_options(UI_DEFAULT_PROVIDER, UI_DEFAULT_MODEL) == [("Max", "max")]
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
    assert default.default_thinking_effort == "max"

    deepseek_model = get_ui_model("deepseek", DEEPSEEK_DEFAULT_MODEL)
    assert deepseek_model is not None
    assert deepseek_model.api_key_env == "DEEPSEEK_API_KEY"
    assert deepseek_model.default_thinking_effort == "high"

    ollama_provider = ModelProvider.OLLAMA_CLOUD.value
    ollama_default = PROVIDER_DEFAULT_MODEL[ModelProvider.OLLAMA_CLOUD]
    ollama_model = get_ui_model(ollama_provider, ollama_default)
    assert ollama_model is not None
    assert ollama_model.provider_label == "Ollama Cloud"
    assert ollama_model.api_key_env == "OLLAMA_API_KEY"
    default_effort = MODEL_SPECS[(ollama_provider, ollama_default)].get("thinking_effort")
    assert ollama_model.default_thinking_effort == (default_effort.default if default_effort else None)

    thinking_model, thinking_spec = next(
        (model, specs["thinking_effort"])
        for (provider, model), specs in MODEL_SPECS.items()
        if provider == ollama_provider and "thinking_effort" in specs
    )
    assert ui_thinking_effort_options(ollama_provider, thinking_model) == [
        (option.title(), option) for option in thinking_spec.options
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


def test_settings_store_round_trips_compression_threshold(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path)
    store.save(CliSettings(compression_threshold=85.0))

    loaded = store.load()
    assert loaded.compression_threshold == 85.0
    assert loaded.to_dict()["compression_threshold"] == 85.0

    # Absent in older files -> None -> the agent's built-in default (95%) applies.
    store.save(CliSettings())
    assert SettingsStore(tmp_path).load().compression_threshold is None


def test_compression_threshold_absent_in_old_file_defaults_to_none() -> None:
    settings = CliSettings.from_dict(
        {
            "schema_version": 3,
            "active_provider": UI_DEFAULT_PROVIDER,
            "active_model": UI_DEFAULT_MODEL,
            "api_keys": {UI_DEFAULT_PROVIDER: "k"},
        }
    )

    assert settings.compression_threshold is None


@pytest.mark.parametrize("raw", [200, 5, 0, -10, 3.5, "90", "high", True, [80], {"percent": 80}])
def test_invalid_compression_threshold_coerced_to_none(raw: object) -> None:
    """Hand-edited settings must degrade to the default, never crash startup."""
    settings = CliSettings.from_dict(
        {
            "schema_version": 3,
            "active_provider": UI_DEFAULT_PROVIDER,
            "active_model": UI_DEFAULT_MODEL,
            "compression_threshold": raw,
        }
    )

    assert settings.compression_threshold is None


@pytest.mark.parametrize("raw,expected", [(10, 10.0), (80, 80.0), (100, 100.0), (65.5, 65.5)])
def test_valid_compression_threshold_loads(raw: object, expected: float) -> None:
    settings = CliSettings.from_dict(
        {
            "schema_version": 3,
            "active_provider": UI_DEFAULT_PROVIDER,
            "active_model": UI_DEFAULT_MODEL,
            "compression_threshold": raw,
        }
    )

    assert settings.compression_threshold == expected


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


# ---------------------------------------------------------------------------
# F1: LSP project trust round-trip
# ---------------------------------------------------------------------------


def test_lsp_project_trust_round_trip(tmp_path: Path) -> None:
    """F1: trusted_lsp_projects persists and resolves via is_lsp_project_trusted."""
    project = tmp_path / "repo"
    project.mkdir()

    settings = CliSettings()
    assert settings.is_lsp_project_trusted(project) is False

    settings.trust_lsp_project(project)
    assert settings.is_lsp_project_trusted(project) is True

    # Round-trip through to_dict / from_dict.
    restored = CliSettings.from_dict(settings.to_dict())
    assert restored.is_lsp_project_trusted(project) is True
    assert str(project.resolve()) in restored.trusted_lsp_projects


def test_lsp_project_trust_independent_of_mcp(tmp_path: Path) -> None:
    """F1: trusting LSP does not implicitly trust MCP (and vice versa)."""
    project = tmp_path / "repo"
    project.mkdir()

    settings = CliSettings()
    settings.trust_lsp_project(project)

    assert settings.is_lsp_project_trusted(project) is True
    assert settings.is_mcp_project_trusted(project) is False


def test_gateway_providers_list_featured_models_but_resolve_every_catalogued_id() -> None:
    # OpenRouter catalogs hundreds of models. The picker shows only the featured
    # (most-used) ones so the dropdown and sub-agent prompts stay bounded, while
    # get_ui_model still resolves anything in the catalog.
    catalogued = [model for provider_value, model in MODEL_SPECS if provider_value == "openrouter"]
    featured = [model for model in catalogued if is_featured_model("openrouter", model)]

    assert featured, "expected the generated catalog to mark its most-used models"
    assert len(catalogued) > len(featured)
    assert [model for _label, model in ui_model_options("openrouter")] == featured

    unlisted = next(model for model in catalogued if model not in set(featured))
    option = get_ui_model("openrouter", unlisted)
    assert option is not None
    assert option.api_key_env == "OPENROUTER_API_KEY"
    assert option.featured is False


def test_non_gateway_providers_still_list_every_model() -> None:
    anthropic_models = [model for provider_value, model in MODEL_SPECS if provider_value == "anthropic"]
    assert [model for _label, model in ui_model_options("anthropic")] == anthropic_models


# --- custom endpoints -----------------------------------------------------


def test_settings_store_round_trips_custom_endpoints(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path)
    settings = CliSettings()
    settings.custom_endpoints = {
        "lmstudio": {
            "api_style": "openai_chat",
            "base_url": "http://localhost:1234/v1",
            "api_key": "k",
            "label": "LM Studio",
            "context_length": 32768,
            "max_output_tokens": 8192,
            "supports_vision": False,
            "thinking": {"mode": "thinking_toggle", "options": ["none", "enabled"], "default": "enabled"},
            "reasoning_replay": "auto",
            "models": {"big": {"context_length": 65536}},
        }
    }
    store.save(settings)

    loaded = store.load()
    assert loaded.custom_endpoints == settings.custom_endpoints
    assert loaded.to_dict()["custom_endpoints"] == settings.custom_endpoints
    store.save(CliSettings())
    assert SettingsStore(tmp_path).load().custom_endpoints == {}


def test_custom_endpoints_coercion_tolerates_hand_edits() -> None:
    settings = CliSettings.from_dict(
        {
            "schema_version": 3,
            "custom_endpoints": {
                "good": {"api_style": "openai_chat", "base_url": "http://x/v1"},
                "Bad Slug!": {"api_style": "openai_chat", "base_url": "http://x/v1"},
                "no-style": {"base_url": "http://x/v1"},
                "bad-style": {"api_style": "grpc", "base_url": "http://x/v1"},
                "no-url": {"api_style": "openai_chat"},
                "not-dict": "nope",
                "bad-int": {"api_style": "openai_chat", "base_url": "http://x/v1", "context_length": "huge"},
                "bad-thinking": {
                    "api_style": "openai_chat",
                    "base_url": "http://x/v1",
                    "thinking": {"mode": "not-a-mode"},
                },
                "bad-replay": {"api_style": "openai_chat", "base_url": "http://x/v1", "reasoning_replay": "sideways"},
                "bad-models": {
                    "api_style": "openai_chat",
                    "base_url": "http://x/v1",
                    "models": {"a": {"context_length": 1}, "b": "not-dict"},
                },
            },
        }
    )

    assert set(settings.custom_endpoints) == {"good", "bad-int", "bad-thinking", "bad-replay", "bad-models"}
    assert "context_length" not in settings.custom_endpoints["bad-int"]
    assert "thinking" not in settings.custom_endpoints["bad-thinking"]
    assert settings.custom_endpoints["bad-replay"]["reasoning_replay"] == "auto"
    assert settings.custom_endpoints["bad-models"]["models"] == {"a": {"context_length": 1}}
