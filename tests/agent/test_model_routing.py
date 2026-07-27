from __future__ import annotations

import json

import pytest

from kolega_code.agent.model_routing import (
    configured_provider_names,
    model_routing_fingerprint,
    render_subagent_model_catalog,
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


@pytest.mark.parametrize("provider", [None, "", "   "])
def test_catalog_omitted_or_blank_provider_lists_all_configured_models(provider: str | None) -> None:
    catalog = subagent_model_catalog(_config(), provider)
    providers = {entry["provider"] for entry in catalog["providers"]}
    assert {"anthropic", "deepseek", "openai_chatgpt"} <= providers


def test_catalog_markdown_is_compact_readable_and_credential_free() -> None:
    catalog = subagent_model_catalog(_config())
    rendered = render_subagent_model_catalog(catalog)
    compact_json = json.dumps(catalog, separators=(",", ":"))

    assert rendered.startswith("# Available sub-agent models")
    assert "| Role | Provider/model | Effort |" in rendered
    assert "`anthropic/claude-opus-4-8`" in rendered
    assert "`null`" in rendered
    assert "Vision" in rendered
    assert not rendered.lstrip().startswith("{")
    assert len(rendered) < len(compact_json) * 0.7
    assert "secret-" not in rendered
    assert "private@example.com" not in rendered


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


# ---------------------------------------------------------------------------
# Failed overrides must name a way forward
#
# A real session produced 10 consecutive failed dispatches: the caller guessed
# `openai` while running on `openai_chatgpt`, retried byte-identical input five
# times, then invented `provider: "default"`. The old messages listed the
# supported enum rather than what was configured, which just invited the next
# unusable guess. It never called list_subagent_models.
# ---------------------------------------------------------------------------


def _override_error(config: AgentConfig, override: dict[str, object]) -> str:
    with pytest.raises(ValueError) as excinfo:
        resolve_subagent_model(config, "general-agent", override, effort_key="thinking_effort")
    return str(excinfo.value)


def test_unconfigured_provider_error_lists_configured_providers_not_the_enum() -> None:
    message = _override_error(
        _config(), {"provider": "google", "model": "gemini-3.1-pro-preview", "thinking_effort": "high"}
    )

    assert "Provider 'google' is not configured." in message
    assert "Configured providers:" in message
    assert "anthropic" in message and "openai_chatgpt" in message
    # google is unconfigured here, so it must not be advertised back as a choice.
    assert "Configured providers: " in message
    listed = message.split("Configured providers: ", 1)[1].split(".", 1)[0]
    assert "google" not in listed


def test_unconfigured_provider_error_names_the_configured_sibling() -> None:
    """The exact shape that failed five times in a row."""
    message = _override_error(_config(), {"provider": "openai", "model": "gpt-5.4-mini", "thinking_effort": "low"})

    assert "serve the same models" in message
    assert "prefer 'openai_chatgpt' when it is configured" in message


def test_sibling_advice_is_absent_for_unrelated_providers() -> None:
    message = _override_error(
        _config(), {"provider": "google", "model": "gemini-3.1-pro-preview", "thinking_effort": "high"}
    )

    assert "serve the same models" not in message


def test_sibling_advice_is_absent_when_the_sibling_is_not_configured() -> None:
    config = AgentConfig(anthropic_api_key="secret-anthropic-key")

    message = _override_error(config, {"provider": "openai", "model": "gpt-5.4-mini", "thinking_effort": "low"})

    assert "not configured" in message
    assert "serve the same models" not in message


def test_unsupported_provider_error_lists_configured_providers() -> None:
    message = _override_error(_config(), {"provider": "default", "model": "default", "thinking_effort": None})

    assert "Unsupported model_override provider 'default'." in message
    assert "Configured providers:" in message
    # The supported-enum dump used to advertise providers that were not usable.
    assert "groq" not in message
    assert "fireworks" not in message


def test_unsupported_model_error_still_points_at_the_default() -> None:
    message = _override_error(
        _config(), {"provider": "anthropic", "model": "claude-imaginary-9", "thinking_effort": "high"}
    )

    assert "claude-imaginary-9" in message
    assert "Omit model_override entirely" in message


@pytest.mark.parametrize(
    "override",
    [
        {"provider": "google", "model": "gemini-3.1-pro-preview", "thinking_effort": "high"},
        {"provider": "openai", "model": "gpt-5.4-mini", "thinking_effort": "low"},
        {"provider": "default", "model": "default", "thinking_effort": None},
    ],
)
def test_every_override_failure_offers_the_inherited_default(override: dict[str, object]) -> None:
    """In all 10 real failures the caller did not need a specific model."""
    config = _config()
    inherited = config.model_config_for_agent("general-agent")

    message = _override_error(config, override)

    assert "Omit model_override entirely to run with the default:" in message
    assert f"{inherited.provider.value}/{inherited.model}" in message


def test_configured_provider_names_match_the_discovery_catalog() -> None:
    """An error and list_subagent_models must never disagree about availability."""
    config = _config()

    names = configured_provider_names(config)
    catalog = {entry["provider"] for entry in subagent_model_catalog(config)["providers"]}

    assert set(names) == catalog


def test_catalog_notes_the_sibling_preference_only_when_both_are_configured() -> None:
    both = AgentConfig(
        # anthropic keys the default role models so the config validates.
        anthropic_api_key="secret-anthropic-key",
        openai_api_key="secret-openai-key",
        openai_chatgpt_tokens=OAuthTokens(
            access_token="secret-access-token",
            refresh_token="secret-refresh-token",
            email="private@example.com",
            account_id="acct-private",
        ),
    )

    with_both = render_subagent_model_catalog(subagent_model_catalog(both))
    with_one = render_subagent_model_catalog(subagent_model_catalog(_config()))

    assert "Prefer `openai_chatgpt` when configured." in with_both
    assert "Prefer `openai_chatgpt` when configured." not in with_one
