from __future__ import annotations

import json

import pytest

from kolega_code.agent.model_routing import (
    model_routing_fingerprint,
    resolve_subagent_model,
    subagent_model_catalog,
)
from kolega_code.auth.tokens import OAuthTokens
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider


def _config() -> AgentConfig:
    return AgentConfig(
        anthropic_api_key="secret-anthropic-key",
        deepseek_api_key="secret-deepseek-key",
        openai_chatgpt_tokens=OAuthTokens(
            access_token="secret-access-token",
            refresh_token="secret-refresh-token",
            email="private@example.com",
            account_id="acct-private",
        ),
    )


def test_complete_override_replaces_only_target_role() -> None:
    config = _config()
    original_thinking = config.thinking_config

    resolved = resolve_subagent_model(
        config,
        "investigation-agent",
        {"provider": "deepseek", "model": "deepseek-v4-flash", "thinking_effort": "HIGH"},
        effort_key="thinking_effort",
    )

    selected = resolved.config.model_config_for_agent("investigation-agent")
    assert selected.provider == ModelProvider.DEEPSEEK
    assert selected.model == "deepseek-v4-flash"
    assert selected.thinking_effort == "high"
    assert resolved.config.model_config_for_agent("general-agent") == config.long_context_config
    assert resolved.config.thinking_config == original_thinking
    assert config.agent_models == {}


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"provider": "anthropic", "model": "claude-opus-4-8"}, "missing required"),
        (
            {
                "provider": "anthropic",
                "model": "claude-opus-4-8",
                "thinking_effort": "high",
                "extra": True,
            },
            "unsupported field",
        ),
        ({"provider": "", "model": "claude-opus-4-8", "thinking_effort": "high"}, "non-empty"),
        (["anthropic", "claude-opus-4-8", "high"], "must be an object"),
    ],
)
def test_override_shape_is_atomic(override: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_subagent_model(
            _config(),
            "general-agent",
            override,
            effort_key="thinking_effort",
        )


def test_no_effort_model_requires_explicit_null() -> None:
    config = _config()
    resolved = resolve_subagent_model(
        config,
        "general-agent",
        {
            "provider": "anthropic",
            "model": "claude-sonnet-4-5-20250929",
            "thinking_effort": None,
        },
        effort_key="thinking_effort",
    )
    assert resolved.model_config.thinking_effort is None

    with pytest.raises(ValueError, match="does not support thinking effort"):
        resolve_subagent_model(
            config,
            "general-agent",
            {
                "provider": "anthropic",
                "model": "claude-sonnet-4-5-20250929",
                "thinking_effort": "none",
            },
            effort_key="thinking_effort",
        )


def test_effort_model_rejects_null() -> None:
    with pytest.raises(ValueError, match="effort must be a string"):
        resolve_subagent_model(
            _config(),
            "general-agent",
            {"provider": "anthropic", "model": "claude-opus-4-8", "thinking_effort": None},
            effort_key="thinking_effort",
        )


def test_unconfigured_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="not configured"):
        resolve_subagent_model(
            _config(),
            "general-agent",
            {"provider": "google", "model": "gemini-3.1-pro-preview", "thinking_effort": "high"},
            effort_key="thinking_effort",
        )


def test_catalog_is_filtered_and_contains_no_credentials() -> None:
    catalog = subagent_model_catalog(_config())
    providers = {entry["provider"] for entry in catalog["providers"]}
    assert {"anthropic", "deepseek", "openai_chatgpt"} <= providers
    assert "google" not in providers

    rendered = json.dumps(catalog)
    for secret in (
        "secret-anthropic-key",
        "secret-deepseek-key",
        "secret-access-token",
        "secret-refresh-token",
        "private@example.com",
        "acct-private",
    ):
        assert secret not in rendered


def test_catalog_provider_filter_and_nullable_effort_marker() -> None:
    catalog = subagent_model_catalog(_config(), "anthropic")
    assert [entry["provider"] for entry in catalog["providers"]] == ["anthropic"]
    no_effort = next(
        model for model in catalog["providers"][0]["models"] if model["model"] == "claude-sonnet-4-5-20250929"
    )
    assert no_effort["thinking_efforts"] == []
    assert no_effort["override_effort"] == "null"


def test_routing_fingerprint_changes_without_including_secrets() -> None:
    first = _config()
    second = first.model_copy(
        update={
            "agent_models": {
                "general": ModelConfig(
                    provider=ModelProvider.DEEPSEEK,
                    model="deepseek-v4-flash",
                    thinking_effort="high",
                )
            }
        }
    )
    assert model_routing_fingerprint(first) != model_routing_fingerprint(second)
    assert "secret" not in model_routing_fingerprint(first)
