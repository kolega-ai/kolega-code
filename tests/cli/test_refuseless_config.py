from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from kolega_code import AgentConfig, ModelConfig, ModelProvider
from kolega_code.agent.model_routing import provider_is_configured
from kolega_code.cli import config as cli_config
from kolega_code.cli.config import (
    API_KEY_ENV,
    CliConfigError,
    CliConfigOverrides,
    build_agent_config,
    config_summary,
    key_status,
    resolved_api_key,
)
from kolega_code.cli.provider_registry import (
    MODEL_LABELS,
    PROVIDER_DEFAULT_MODEL,
    PROVIDER_LABELS,
    UI_DEFAULT_MODEL,
    UI_DEFAULT_PROVIDER,
    default_model_for_provider,
    _thinking_effort_label,
)
from kolega_code.cli.settings import CliSettings, SettingsStore
from kolega_code.security.secrets import SECRET_PLACEHOLDER, redact_secrets

pytestmark = pytest.mark.usefixtures("isolated_cli_env")

PROVIDER = ModelProvider.REFUSELESS
MODEL = "glm-5.3-abliterated"
FAKE_KEY = "abl_FAKE_refuseless_credential_for_tests"
SAVED_KEY = "abl_FAKE_saved_refuseless_credential"


def _route() -> ModelConfig:
    return ModelConfig(provider=PROVIDER, model=MODEL)


@pytest.fixture
def refuseless_config_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate config plumbing from separately verified model capabilities."""
    original_known = cli_config.model_is_known
    original_specs = cli_config.get_model_specs
    original_effort = cli_config.normalize_thinking_effort

    def model_is_known(provider: str, model: str) -> bool:
        return model == MODEL if provider == PROVIDER else original_known(provider, model)

    def get_model_specs(provider: str, model: str) -> dict[str, Any]:
        return {} if provider == PROVIDER else original_specs(provider, model)

    def normalize_thinking_effort(provider: str, model: str, effort: Any) -> str | None:
        return None if provider == PROVIDER else original_effort(provider, model, effort)

    monkeypatch.setattr(cli_config, "model_is_known", model_is_known)
    monkeypatch.setattr(cli_config, "get_model_specs", get_model_specs)
    monkeypatch.setattr(cli_config, "normalize_thinking_effort", normalize_thinking_effort)


def test_public_config_accepts_refuseless_and_keeps_existing_defaults() -> None:
    config = AgentConfig(refuseless_api_key=FAKE_KEY, long_context_config=_route(), fast_config=_route())

    assert ModelProvider("refuseless") is PROVIDER
    assert list(ModelProvider)[-1] is PROVIDER
    assert config.get_api_key(PROVIDER) == FAKE_KEY
    assert config.llm_client_kwargs(_route()) == {}
    defaults = AgentConfig(anthropic_api_key="fake-anthropic-key")
    assert defaults.refuseless_api_key is None
    assert defaults.long_context_config.provider == ModelProvider.ANTHROPIC
    assert defaults.long_context_config.model == "claude-opus-5"
    assert defaults.fast_config.model == "claude-haiku-4-5-20251001"


@pytest.mark.parametrize("slot", ["active", "fast", "role"])
def test_public_config_requires_refuseless_key_for_every_route(slot: str) -> None:
    values: dict[str, Any] = {"anthropic_api_key": "fake-anthropic-key"}
    if slot == "active":
        values["long_context_config"] = _route()
    elif slot == "fast":
        values["fast_config"] = _route()
    else:
        values["agent_models"] = {"investigation": _route()}

    with pytest.raises(ValidationError, match="Missing API key for .* provider 'refuseless'"):
        AgentConfig(**values)


@pytest.mark.parametrize("env_key", [None, "", FAKE_KEY])
def test_cli_key_environment_over_settings(
    refuseless_config_catalog: None, tmp_path: Path, env_key: str | None
) -> None:
    settings = CliSettings(api_keys={PROVIDER.value: SAVED_KEY})
    env = {} if env_key is None else {"REFUSELESS_API_KEY": env_key}

    config = build_agent_config(
        tmp_path, CliConfigOverrides(provider=PROVIDER.value, model=MODEL), env=env, settings=settings
    )

    assert API_KEY_ENV[PROVIDER] == "REFUSELESS_API_KEY"
    assert config.refuseless_api_key == (env_key or SAVED_KEY)
    assert config.long_context_config.provider is PROVIDER
    assert config.long_context_config.model == MODEL
    assert config.fast_config.provider is PROVIDER
    assert config.fast_config.model == MODEL
    assert settings.get_api_key(PROVIDER.value) == SAVED_KEY


def test_refuseless_key_alone_does_not_select_a_model(tmp_path: Path) -> None:
    with pytest.raises(CliConfigError, match="No provider/model configured"):
        build_agent_config(tmp_path, env={"REFUSELESS_API_KEY": FAKE_KEY})


@pytest.mark.parametrize("slot", ["active", "fast", "role"])
def test_cli_requires_refuseless_key_for_every_route(
    refuseless_config_catalog: None, tmp_path: Path, slot: str
) -> None:
    overrides = CliConfigOverrides(provider="anthropic", model="claude-opus-5")
    settings = CliSettings()
    if slot == "active":
        overrides = CliConfigOverrides(provider=PROVIDER.value, model=MODEL)
    elif slot == "fast":
        overrides = CliConfigOverrides(
            provider="anthropic", model="claude-opus-5", fast_provider=PROVIDER.value, fast_model=MODEL
        )
    else:
        settings.agent_models = {"investigation": {"provider": PROVIDER.value, "model": MODEL}}

    with pytest.raises(CliConfigError, match="Missing required API key environment variable.*REFUSELESS_API_KEY"):
        build_agent_config(tmp_path, overrides, env={"ANTHROPIC_API_KEY": "fake-anthropic-key"}, settings=settings)


def test_saved_key_and_routes_round_trip(refuseless_config_catalog: None, tmp_path: Path) -> None:
    store = SettingsStore()
    settings = store.load()
    settings.active_provider = PROVIDER.value
    settings.active_model = MODEL
    settings.model_slots = {"fast": {"provider": PROVIDER.value, "model": MODEL}}
    settings.agent_models = {"investigation": {"provider": PROVIDER.value, "model": MODEL}}
    settings.set_api_key(PROVIDER.value, SAVED_KEY)
    store.save(settings)

    loaded = store.load()
    config = build_agent_config(tmp_path, env={}, settings=loaded)

    assert loaded.get_api_key(PROVIDER.value) == SAVED_KEY
    assert loaded.active_provider == PROVIDER.value
    assert loaded.active_model == MODEL
    assert loaded.model_slots == settings.model_slots
    assert loaded.agent_models == settings.agent_models
    assert config.refuseless_api_key == SAVED_KEY
    assert config.fast_config.provider is PROVIDER
    assert config.agent_models["investigation"].provider is PROVIDER
    assert config.model_config_for_agent("investigation-agent").model == MODEL
    assert SAVED_KEY not in json.dumps(config_summary(config))
    assert "refuseless_api_key" not in config_summary(config)


def test_refuseless_configuration_is_independent_of_other_provider_keys() -> None:
    unconfigured = AgentConfig(anthropic_api_key="fake-anthropic-key", openai_api_key="fake-openai-key")
    configured = unconfigured.model_copy(update={"refuseless_api_key": FAKE_KEY})

    assert provider_is_configured(unconfigured, PROVIDER) is False
    assert provider_is_configured(configured, PROVIDER) is True
    assert configured.long_context_config == unconfigured.long_context_config


def test_key_status_and_resolution_follow_environment_over_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = CliSettings()
    assert key_status(PROVIDER.value, tmp_path, settings) == "missing"
    assert resolved_api_key(PROVIDER.value, tmp_path, settings) is None
    settings.set_api_key(PROVIDER.value, SAVED_KEY)
    assert key_status(PROVIDER.value, tmp_path, settings) == "present in local settings"
    assert resolved_api_key(PROVIDER.value, tmp_path, settings) == SAVED_KEY

    monkeypatch.setenv("REFUSELESS_API_KEY", FAKE_KEY)
    assert key_status(PROVIDER.value, tmp_path, settings) == "present via REFUSELESS_API_KEY"
    assert resolved_api_key(PROVIDER.value, tmp_path, settings) == FAKE_KEY


def test_configured_refuseless_secret_is_discovered_and_redacted() -> None:
    from kolega_code.cli.main import known_secret_values

    settings = CliSettings(api_keys={PROVIDER.value: FAKE_KEY})
    values = known_secret_values(settings, SettingsStore())

    assert FAKE_KEY in values
    assert redact_secrets(f"credential was {FAKE_KEY}", values) == f"credential was {SECRET_PLACEHOLDER}"


def test_refuseless_environment_secret_is_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REFUSELESS_API_KEY", FAKE_KEY)
    assert redact_secrets(FAKE_KEY, include_environment=True) == SECRET_PLACEHOLDER


def test_registry_appends_refuseless_without_changing_existing_ui_defaults() -> None:
    assert list(PROVIDER_LABELS)[-1] is PROVIDER
    assert PROVIDER_LABELS[PROVIDER] == "Refuseless"
    assert list(PROVIDER_DEFAULT_MODEL)[-1] is PROVIDER
    assert default_model_for_provider(PROVIDER) == MODEL
    assert UI_DEFAULT_PROVIDER == "moonshot"
    assert UI_DEFAULT_MODEL == "kimi-k3"
    assert _thinking_effort_label("off") == "Off"


@pytest.mark.parametrize(
    ("model", "label"),
    [
        ("glm-5.3-abliterated", "GLM-5.3 Abliterated"),
        ("glm-5.3-flash-abliterated", "GLM-5.3 Flash Abliterated"),
        ("deepseek-v4.1-flash-abliterated", "DeepSeek V4.1 Flash Abliterated"),
        ("deepseek-v4.1-flash-fast", "DeepSeek V4.1 Flash Fast"),
        ("minimax-m3.1-flash", "MiniMax M3.1 Flash"),
    ],
)
def test_model_labels_keep_variant_names(model: str, label: str) -> None:
    assert MODEL_LABELS[model] == label
