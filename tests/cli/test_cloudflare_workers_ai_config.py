"""Cloudflare Workers AI credentials, account scoping, and connection kwargs across the config layers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from kolega_code.agent.model_routing import provider_is_configured
from kolega_code.cli.config import (
    CliConfigError,
    CliConfigOverrides,
    build_agent_config,
    key_status,
    provider_connection_kwargs,
    resolved_cloudflare_account_id,
)
from kolega_code.cli.model_connection import test_model_connection as probe_model_connection
from kolega_code.cli.settings import CliSettings, SettingsStore
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider
from kolega_code.llm import cloudflare_workers_ai
from kolega_code.security.secrets import redact_secrets

PROVIDER = ModelProvider.CLOUDFLARE_WORKERS_AI
MODEL = "@cf/zai-org/glm-5.3"
ACCOUNT_ID = "0123456789abcdef0123456789abcdef"
OTHER_ACCOUNT_ID = "fedcba9876543210fedcba9876543210"
BASE_URL = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/v1"
TOKEN_ENV = {"CLOUDFLARE_API_TOKEN": "cf-test-token"}


def _route() -> ModelConfig:
    return ModelConfig(provider=PROVIDER, model=MODEL)


def _agent_config(**kwargs: Any) -> AgentConfig:
    values: dict[str, Any] = {
        "cloudflare_api_token": "cf-test-token",
        "long_context_config": _route(),
        "fast_config": _route(),
    }
    values.update(kwargs)
    return AgentConfig(**values)


def _cli_config(tmp_path: Path, env: dict[str, str], settings: CliSettings | None = None) -> AgentConfig:
    return build_agent_config(
        tmp_path,
        CliConfigOverrides(provider=PROVIDER.value, model=MODEL),
        env=env,
        settings=settings,
    )


# --- AgentConfig -----------------------------------------------------------


def test_agent_config_requires_an_account_id_for_cloudflare_routes() -> None:
    with pytest.raises(ValidationError, match="CLOUDFLARE_ACCOUNT_ID"):
        _agent_config()


@pytest.mark.parametrize("value", ["../../accounts/x", "not-hex", "0123456789abcdef"])
def test_agent_config_rejects_malformed_account_ids(value: str) -> None:
    with pytest.raises(ValidationError, match="32-character hex"):
        _agent_config(cloudflare_account_id=value)


def test_agent_config_normalizes_ids_and_builds_client_kwargs() -> None:
    config = _agent_config(cloudflare_account_id=f"  {ACCOUNT_ID.upper()} ", cloudflare_ai_gateway_id="my-gateway")

    assert config.cloudflare_account_id == ACCOUNT_ID
    assert config.get_api_key(PROVIDER) == "cf-test-token"
    assert config.llm_client_kwargs(_route()) == {
        "base_url": BASE_URL,
        "default_headers": {"cf-aig-gateway-id": "my-gateway"},
    }


def test_agent_config_omits_the_gateway_header_when_unset() -> None:
    assert _agent_config(cloudflare_account_id=ACCOUNT_ID).llm_client_kwargs(_route()) == {"base_url": BASE_URL}


def test_agent_config_rejects_malformed_gateway_ids() -> None:
    with pytest.raises(ValidationError, match="Gateway ID"):
        _agent_config(cloudflare_account_id=ACCOUNT_ID, cloudflare_ai_gateway_id="bad id\n")


def test_llm_client_kwargs_are_empty_for_fixed_endpoint_providers() -> None:
    config = AgentConfig(anthropic_api_key="sk-ant-test")
    assert config.llm_client_kwargs(config.long_context_config) == {}


def test_created_llm_client_targets_the_account() -> None:
    from kolega_code.agent.context import AgentContext, Telemetry
    from kolega_code.llm.providers.openai import OpenAIProvider

    context = AgentContext.__new__(AgentContext)
    context.config = _agent_config(cloudflare_account_id=ACCOUNT_ID, cloudflare_ai_gateway_id="gw")
    context.telemetry = Telemetry()

    client = context.create_llm_client("coder")

    assert isinstance(client.provider, OpenAIProvider)
    assert str(client.provider.async_client.base_url).rstrip("/") == BASE_URL
    assert client.provider.async_client._custom_headers["cf-aig-gateway-id"] == "gw"


def test_sub_agent_routing_needs_both_token_and_account() -> None:
    configured = _agent_config(cloudflare_account_id=ACCOUNT_ID)
    token_only = AgentConfig(anthropic_api_key="sk-ant-test", cloudflare_api_token="cf-test-token")

    assert provider_is_configured(configured, PROVIDER) is True
    assert provider_is_configured(token_only, PROVIDER) is False


# --- build_agent_config ----------------------------------------------------


def test_cli_resolves_token_account_and_gateway_from_the_environment(tmp_path: Path) -> None:
    env = {**TOKEN_ENV, "CLOUDFLARE_ACCOUNT_ID": ACCOUNT_ID, "CLOUDFLARE_AI_GATEWAY_ID": "gw"}

    config = _cli_config(tmp_path, env)

    assert config.long_context_config.provider == PROVIDER
    assert config.cloudflare_api_token == "cf-test-token"
    assert config.cloudflare_account_id == ACCOUNT_ID
    assert config.cloudflare_ai_gateway_id == "gw"


def test_cli_environment_account_beats_settings(tmp_path: Path) -> None:
    settings = CliSettings(cloudflare_account_id=OTHER_ACCOUNT_ID)

    config = _cli_config(tmp_path, {**TOKEN_ENV, "CLOUDFLARE_ACCOUNT_ID": ACCOUNT_ID}, settings)

    assert config.cloudflare_account_id == ACCOUNT_ID


def test_cli_falls_back_to_settings_for_token_and_account(tmp_path: Path) -> None:
    settings = CliSettings(cloudflare_account_id=ACCOUNT_ID)
    settings.set_api_key(PROVIDER.value, "saved-token")

    config = _cli_config(tmp_path, {}, settings)

    assert config.cloudflare_api_token == "saved-token"
    assert config.cloudflare_account_id == ACCOUNT_ID


def test_cli_missing_token_is_reported_by_its_env_name(tmp_path: Path) -> None:
    with pytest.raises(CliConfigError, match="CLOUDFLARE_API_TOKEN"):
        _cli_config(tmp_path, {"CLOUDFLARE_ACCOUNT_ID": ACCOUNT_ID})


def test_cli_missing_account_id_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(CliConfigError, match=r"Missing Cloudflare account ID: set CLOUDFLARE_ACCOUNT_ID"):
        _cli_config(tmp_path, TOKEN_ENV)


def test_cli_invalid_account_id_names_its_source(tmp_path: Path) -> None:
    with pytest.raises(CliConfigError, match="from CLOUDFLARE_ACCOUNT_ID"):
        _cli_config(tmp_path, {**TOKEN_ENV, "CLOUDFLARE_ACCOUNT_ID": "nope"})
    with pytest.raises(CliConfigError, match="saved in Settings"):
        _cli_config(tmp_path, TOKEN_ENV, CliSettings(cloudflare_account_id="nope"))


def test_cli_invalid_gateway_id_is_fatal_only_for_cloudflare_routes(tmp_path: Path) -> None:
    with pytest.raises(CliConfigError, match="CLOUDFLARE_AI_GATEWAY_ID"):
        _cli_config(tmp_path, {**TOKEN_ENV, "CLOUDFLARE_ACCOUNT_ID": ACCOUNT_ID, "CLOUDFLARE_AI_GATEWAY_ID": "a b"})


def test_stale_cloudflare_values_never_lock_out_other_providers(tmp_path: Path) -> None:
    config = build_agent_config(
        tmp_path,
        CliConfigOverrides(provider="anthropic", model="claude-opus-5"),
        env={"ANTHROPIC_API_KEY": "sk-ant-test", "CLOUDFLARE_AI_GATEWAY_ID": "a b"},
        settings=CliSettings(cloudflare_account_id="nope"),
    )

    assert config.cloudflare_account_id is None
    assert config.cloudflare_ai_gateway_id is None


def test_cloudflare_as_the_fast_slot_also_requires_the_account(tmp_path: Path) -> None:
    with pytest.raises(CliConfigError, match="Missing Cloudflare account ID"):
        build_agent_config(
            tmp_path,
            CliConfigOverrides(
                provider="anthropic", model="claude-opus-5", fast_provider=PROVIDER.value, fast_model=MODEL
            ),
            env={"ANTHROPIC_API_KEY": "sk-ant-test", **TOKEN_ENV},
        )


# --- status and probe helpers ----------------------------------------------


@pytest.mark.parametrize(
    ("env", "settings", "expected"),
    [
        ({}, CliSettings(), "missing"),
        (TOKEN_ENV, CliSettings(), "present via CLOUDFLARE_API_TOKEN; account ID missing"),
        ({**TOKEN_ENV, "CLOUDFLARE_ACCOUNT_ID": ACCOUNT_ID}, CliSettings(), "present via CLOUDFLARE_API_TOKEN"),
        (TOKEN_ENV, CliSettings(cloudflare_account_id="nope"), "present via CLOUDFLARE_API_TOKEN; account ID invalid"),
        (
            {},
            CliSettings(api_keys={PROVIDER.value: "k"}, cloudflare_account_id=ACCOUNT_ID),
            "present in local settings",
        ),
    ],
)
def test_key_status_reports_the_account_alongside_the_token(
    isolated_cli_env: None, tmp_path: Path, monkeypatch, env: dict[str, str], settings: CliSettings, expected: str
) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    assert key_status(PROVIDER.value, tmp_path, settings) == expected


def test_probe_kwargs_follow_env_then_draft(isolated_cli_env: None, tmp_path: Path, monkeypatch) -> None:
    draft = CliSettings(cloudflare_account_id=OTHER_ACCOUNT_ID)
    assert provider_connection_kwargs(PROVIDER.value, tmp_path, draft) == {
        "base_url": cloudflare_workers_ai.base_url(OTHER_ACCOUNT_ID)
    }
    assert resolved_cloudflare_account_id(tmp_path, draft) == OTHER_ACCOUNT_ID

    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", ACCOUNT_ID)
    monkeypatch.setenv("CLOUDFLARE_AI_GATEWAY_ID", "gw")
    assert provider_connection_kwargs(PROVIDER.value, tmp_path, draft) == {
        "base_url": BASE_URL,
        "default_headers": {"cf-aig-gateway-id": "gw"},
    }
    assert provider_connection_kwargs("groq", tmp_path, draft) == {}


def test_probe_kwargs_explain_a_missing_account(isolated_cli_env: None, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="CLOUDFLARE_ACCOUNT_ID"):
        provider_connection_kwargs(PROVIDER.value, tmp_path, CliSettings())
    assert resolved_cloudflare_account_id(tmp_path, CliSettings(cloudflare_account_id="nope")) is None


@pytest.mark.asyncio
async def test_connection_probe_forwards_default_headers() -> None:
    captured: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        async def generate(self, **_kwargs: Any) -> None:
            return None

    result = await probe_model_connection(
        PROVIDER,
        MODEL,
        api_key="cf-test-token",
        client_factory=FakeClient,
        base_url=BASE_URL,
        default_headers={"cf-aig-gateway-id": "gw"},
    )

    assert result.ok
    assert captured["base_url"] == BASE_URL
    assert captured["default_headers"] == {"cf-aig-gateway-id": "gw"}


# --- settings + redaction --------------------------------------------------


def test_settings_round_trip_the_account_id(isolated_cli_env: None) -> None:
    store = SettingsStore()
    settings = store.load()
    assert settings.cloudflare_account_id is None
    settings.cloudflare_account_id = ACCOUNT_ID
    store.save(settings)

    assert store.load().cloudflare_account_id == ACCOUNT_ID
    assert CliSettings.from_dict({"schema_version": 3, "cloudflare_account_id": "  "}).cloudflare_account_id is None
    assert CliSettings.from_dict({"schema_version": 3, "cloudflare_account_id": 7}).cloudflare_account_id is None


@pytest.mark.parametrize("prefix", ["cfut_", "cfat_", "cfk_"])
def test_scannable_cloudflare_tokens_are_redacted(prefix: str) -> None:
    token = prefix + "FAKE" * 10 + "abc123"
    assert token not in redact_secrets(f"token was {token} in the log")


def test_account_ids_are_not_treated_as_secrets() -> None:
    assert ACCOUNT_ID in redact_secrets(f"account {ACCOUNT_ID}")
