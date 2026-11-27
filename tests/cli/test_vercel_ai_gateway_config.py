"""CLI config, credential precedence, ACP discovery, and secret redaction for Vercel AI Gateway."""

from __future__ import annotations

from pathlib import Path

import pytest

from kolega_code.acp.agent_factory import AgentFactory
from kolega_code.cli.config import (
    API_KEY_ENV,
    API_KEY_ENV_ALIASES,
    CliConfigError,
    CliConfigOverrides,
    build_agent_config,
    key_status,
    provider_api_key_env_names,
    resolved_api_key,
)
from kolega_code.cli.settings import CliSettings, SettingsStore
from kolega_code.config import ModelProvider
from kolega_code.security.secrets import SECRET_PLACEHOLDER, redact_secrets

PROVIDER = ModelProvider.VERCEL_AI_GATEWAY
PRIMARY_KEY = "vck_primary_test_token_0000000000000000"
ALIAS_KEY = "vck_alias_test_token_111111111111111111"
SAVED_KEY = "vck_saved_test_token_222222222222222222"
MODEL = "moonshotai/kimi-k2.7-code"


def test_env_var_registry_and_precedence(
    isolated_cli_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert API_KEY_ENV[PROVIDER] == "VERCEL_API_KEY"
    assert API_KEY_ENV_ALIASES[PROVIDER] == ("AI_GATEWAY_API_KEY",)
    assert provider_api_key_env_names(PROVIDER) == ("VERCEL_API_KEY", "AI_GATEWAY_API_KEY")

    empty_settings = CliSettings()
    assert key_status(PROVIDER.value, tmp_path, empty_settings) == "missing"
    assert resolved_api_key(PROVIDER.value, tmp_path, empty_settings) is None

    saved_settings = CliSettings(api_keys={PROVIDER.value: SAVED_KEY})
    assert key_status(PROVIDER.value, tmp_path, saved_settings) == "present in local settings"
    assert resolved_api_key(PROVIDER.value, tmp_path, saved_settings) == SAVED_KEY

    monkeypatch.setenv("AI_GATEWAY_API_KEY", ALIAS_KEY)
    assert key_status(PROVIDER.value, tmp_path, saved_settings) == "present via AI_GATEWAY_API_KEY"
    assert resolved_api_key(PROVIDER.value, tmp_path, saved_settings) == ALIAS_KEY

    monkeypatch.setenv("VERCEL_API_KEY", PRIMARY_KEY)
    assert key_status(PROVIDER.value, tmp_path, saved_settings) == "present via VERCEL_API_KEY"
    assert resolved_api_key(PROVIDER.value, tmp_path, saved_settings) == PRIMARY_KEY


def test_build_agent_config_accepts_primary_and_alias_keys(
    isolated_cli_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AI_GATEWAY_API_KEY", ALIAS_KEY)
    config = build_agent_config(
        project_path=tmp_path,
        overrides=CliConfigOverrides(provider=PROVIDER.value, model=MODEL),
        settings=CliSettings(),
    )
    assert config.long_context_config.provider is PROVIDER
    assert config.long_context_config.model == MODEL
    assert config.get_api_key(PROVIDER) == ALIAS_KEY

    monkeypatch.setenv("VERCEL_API_KEY", PRIMARY_KEY)
    config_primary = build_agent_config(
        project_path=tmp_path,
        overrides=CliConfigOverrides(provider=PROVIDER.value, model=MODEL),
        settings=CliSettings(),
    )
    assert config_primary.get_api_key(PROVIDER) == PRIMARY_KEY


def test_build_agent_config_rejects_unknown_model(
    isolated_cli_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("VERCEL_API_KEY", PRIMARY_KEY)
    with pytest.raises(CliConfigError, match="not available"):
        build_agent_config(
            project_path=tmp_path,
            overrides=CliConfigOverrides(provider=PROVIDER.value, model="nonexistent/model"),
            settings=CliSettings(),
        )


def test_acp_discovers_provider_via_alias_and_primary_env(
    isolated_cli_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory = AgentFactory()
    factory._settings = CliSettings()
    assert PROVIDER.value not in factory._configured_providers()

    monkeypatch.setenv("AI_GATEWAY_API_KEY", ALIAS_KEY)
    assert PROVIDER.value in factory._configured_providers()

    monkeypatch.delenv("AI_GATEWAY_API_KEY")
    monkeypatch.setenv("VERCEL_API_KEY", PRIMARY_KEY)
    assert PROVIDER.value in factory._configured_providers()


def test_vck_token_redaction() -> None:
    from kolega_code.cli.main import known_secret_values

    assert redact_secrets(f"Authorization: {PRIMARY_KEY}") == f"Authorization: {SECRET_PLACEHOLDER}"
    settings = CliSettings(api_keys={PROVIDER.value: SAVED_KEY})
    values = known_secret_values(settings, SettingsStore())
    assert SAVED_KEY in values
    assert redact_secrets(f"token={SAVED_KEY}", values) == f"token={SECRET_PLACEHOLDER}"
