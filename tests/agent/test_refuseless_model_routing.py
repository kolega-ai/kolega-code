from __future__ import annotations

import json
from typing import Any

import pytest

from kolega_code.agent.model_routing import (
    configured_provider_names,
    model_routing_fingerprint,
    provider_is_configured,
    render_subagent_model_catalog,
    resolve_subagent_model,
    subagent_model_catalog,
)
from kolega_code.config import AGENT_ROLE_BY_NAME, AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.llm.specs import MODEL_SPECS, ThinkingEffortSpec
from kolega_code.llm.specs.validation import validate_model_spec

pytestmark = pytest.mark.usefixtures("isolated_cli_env")

REFUSELESS_MODELS = (
    "glm-5.3-abliterated",
    "glm-5.3-flash-abliterated",
    "deepseek-v4.1-flash-abliterated",
    "deepseek-v4.1-flash-fast",
    "minimax-m3.1-flash",
)
SYNTHETIC_ROUTING_MODEL = "synthetic-refuseless-routing-fixture-not-a-public-model"
SYNTHETIC_ROUTING_EFFORTS = ("off", "low", "high")
FAKE_REFUSELESS_KEY = "abl_FAKE_refuseless_routing_not_a_credential"
FAKE_OTHER_KEYS = {
    "anthropic_api_key": "FAKE_anthropic_routing_not_a_credential",
    "openai_api_key": "FAKE_openai_routing_not_a_credential",
    "openrouter_api_key": "FAKE_openrouter_routing_not_a_credential",
    "fireworks_api_key": "FAKE_fireworks_routing_not_a_credential",
    "ollama_cloud_api_key": "FAKE_ollama_routing_not_a_credential",
    "deepseek_api_key": "FAKE_deepseek_routing_not_a_credential",
}


@pytest.fixture
def synthetic_refuseless_routing_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    specs: dict[str, Any] = {
        "context_length": 4096,
        "max_completion_tokens": 512,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": False,
        "thinking_effort": ThinkingEffortSpec(
            options=SYNTHETIC_ROUTING_EFFORTS, default="low", mode="openai_reasoning_effort"
        ),
    }
    validate_model_spec(specs)
    monkeypatch.setitem(MODEL_SPECS, ("refuseless", SYNTHETIC_ROUTING_MODEL), specs)


@pytest.fixture
def parent_config() -> AgentConfig:
    return AgentConfig(
        anthropic_api_key=FAKE_OTHER_KEYS["anthropic_api_key"],
        openai_api_key=FAKE_OTHER_KEYS["openai_api_key"],
        openrouter_api_key=FAKE_OTHER_KEYS["openrouter_api_key"],
        fireworks_api_key=FAKE_OTHER_KEYS["fireworks_api_key"],
        ollama_cloud_api_key=FAKE_OTHER_KEYS["ollama_cloud_api_key"],
        deepseek_api_key=FAKE_OTHER_KEYS["deepseek_api_key"],
        refuseless_api_key=FAKE_REFUSELESS_KEY,
        agent_models={
            "general": ModelConfig(
                provider=ModelProvider.ANTHROPIC,
                model="claude-opus-5",
                thinking_effort="high",
                rate_limits=RateLimitConfig(requests_per_minute=17, tokens_per_minute=12345),
            )
        },
    )


@pytest.mark.parametrize("missing_key", [None, ""])
@pytest.mark.parametrize("effort_key", ["thinking_effort", "effort"], ids=["ordinary-dispatch", "gigacode"])
@pytest.mark.usefixtures("synthetic_refuseless_routing_catalog")
def test_synthetic_routing_discovery_and_override_require_refuseless_own_key(
    parent_config: AgentConfig, missing_key: str | None, effort_key: str
) -> None:
    config = parent_config.model_copy(update={"refuseless_api_key": missing_key})
    assert config.get_api_key(ModelProvider.OPENAI)
    assert config.get_api_key(ModelProvider.OPENROUTER)
    assert config.get_api_key(ModelProvider.FIREWORKS)
    assert config.get_api_key(ModelProvider.OLLAMA_CLOUD)
    assert config.get_api_key(ModelProvider.DEEPSEEK)
    assert not provider_is_configured(config, ModelProvider.REFUSELESS)
    assert "refuseless" not in configured_provider_names(config)
    catalog = subagent_model_catalog(config)
    assert "refuseless" not in {entry["provider"] for entry in catalog["providers"]}
    assert SYNTHETIC_ROUTING_MODEL not in render_subagent_model_catalog(catalog)
    with pytest.raises(ValueError, match="Provider 'refuseless' is not configured"):
        subagent_model_catalog(config, "refuseless")
    with pytest.raises(ValueError, match="Provider 'refuseless' is not configured") as caught:
        resolve_subagent_model(
            config,
            "investigation-agent",
            {"provider": "refuseless", "model": SYNTHETIC_ROUTING_MODEL, effort_key: "high"},
            effort_key=effort_key,
        )
    for secret in FAKE_OTHER_KEYS.values():
        assert secret not in str(caught.value)


@pytest.mark.usefixtures("synthetic_refuseless_routing_catalog")
def test_synthetic_routing_configured_discovery_and_rendering_are_credential_free(parent_config: AgentConfig) -> None:
    assert provider_is_configured(parent_config, ModelProvider.REFUSELESS)
    catalog = subagent_model_catalog(parent_config)
    assert set(configured_provider_names(parent_config)) == {entry["provider"] for entry in catalog["providers"]}
    filtered = subagent_model_catalog(parent_config, " REFUSELESS ")
    assert [entry["provider"] for entry in filtered["providers"]] == ["refuseless"]
    synthetic = next(model for model in filtered["providers"][0]["models"] if model["model"] == SYNTHETIC_ROUTING_MODEL)
    assert synthetic == {
        "model": SYNTHETIC_ROUTING_MODEL,
        "thinking_efforts": list(SYNTHETIC_ROUTING_EFFORTS),
        "default_thinking_effort": "low",
        "override_effort": "string",
        "supports_vision": False,
    }
    assert filtered["override_contract"]["ordinary_dispatch"]["effort_field"] == "thinking_effort"
    assert filtered["override_contract"]["gigacode"]["effort_field"] == "effort"
    rendered = render_subagent_model_catalog(filtered)
    assert f"| `refuseless/{SYNTHETIC_ROUTING_MODEL}` | `off, low, high` | `low` | no |" in rendered
    for secret in (FAKE_REFUSELESS_KEY, *FAKE_OTHER_KEYS.values()):
        assert secret not in json.dumps(catalog)
        assert secret not in json.dumps(filtered)
        assert secret not in rendered


@pytest.mark.parametrize("agent_name", AGENT_ROLE_BY_NAME)
@pytest.mark.parametrize("effort_key", ["thinking_effort", "effort"], ids=["ordinary-dispatch", "gigacode"])
@pytest.mark.parametrize("effort", SYNTHETIC_ROUTING_EFFORTS)
@pytest.mark.usefixtures("synthetic_refuseless_routing_catalog")
def test_synthetic_routing_complete_override_changes_only_target_role(
    parent_config: AgentConfig, agent_name: str, effort_key: str, effort: str
) -> None:
    before = parent_config.model_dump()
    defaults_before = subagent_model_catalog(parent_config)["agent_defaults"]
    fingerprint_before = model_routing_fingerprint(parent_config)
    inherited = parent_config.model_config_for_agent(agent_name)
    resolved = resolve_subagent_model(
        parent_config,
        agent_name,
        {"provider": " REFUSELESS ", "model": f" {SYNTHETIC_ROUTING_MODEL} ", effort_key: f" {effort.upper()} "},
        effort_key=effort_key,
    )

    selected = resolved.model_config
    assert resolved.config is not parent_config
    assert selected.provider is ModelProvider.REFUSELESS
    assert selected.model == SYNTHETIC_ROUTING_MODEL
    assert selected.thinking_effort == effort
    assert resolved.requested is not None
    assert resolved.requested.as_dict(effort_key=effort_key) == {
        "provider": "refuseless",
        "model": SYNTHETIC_ROUTING_MODEL,
        effort_key: effort,
    }
    assert resolved.metadata == {
        "provider": "refuseless",
        "model": SYNTHETIC_ROUTING_MODEL,
        "thinking_effort": effort,
    }
    assert resolved.config.model_config_for_agent(agent_name) is selected
    assert selected.rate_limits == inherited.rate_limits
    assert selected.rate_limits is not inherited.rate_limits
    assert resolved.config.get_api_key(selected.provider) == FAKE_REFUSELESS_KEY
    assert resolved.config.model_dump(exclude={"agent_models"}) == parent_config.model_dump(exclude={"agent_models"})
    for other_agent in AGENT_ROLE_BY_NAME:
        if other_agent != agent_name:
            assert resolved.config.model_config_for_agent(other_agent) == parent_config.model_config_for_agent(
                other_agent
            )
    assert set(resolved.config.agent_models) == set(parent_config.agent_models) | {AGENT_ROLE_BY_NAME[agent_name].value}
    assert model_routing_fingerprint(resolved.config) != fingerprint_before
    for secret in (FAKE_REFUSELESS_KEY, *FAKE_OTHER_KEYS.values()):
        assert secret not in json.dumps(resolved.metadata)
    selected.rate_limits.requests_per_minute += 1
    assert parent_config.model_dump() == before
    assert subagent_model_catalog(parent_config)["agent_defaults"] == defaults_before
    assert model_routing_fingerprint(parent_config) == fingerprint_before


@pytest.mark.parametrize("agent_name", AGENT_ROLE_BY_NAME)
@pytest.mark.parametrize("effort_key", ["thinking_effort", "effort"], ids=["ordinary-dispatch", "gigacode"])
def test_synthetic_routing_omitted_override_preserves_inherited_defaults(
    parent_config: AgentConfig, agent_name: str, effort_key: str
) -> None:
    resolved = resolve_subagent_model(parent_config, agent_name, None, effort_key=effort_key)
    assert resolved.config is parent_config
    assert resolved.model_config is parent_config.model_config_for_agent(agent_name)
    assert resolved.requested is None
    assert resolved.model_config.provider is ModelProvider.ANTHROPIC


@pytest.mark.parametrize("effort_key", ["thinking_effort", "effort"], ids=["ordinary-dispatch", "gigacode"])
def test_synthetic_routing_unknown_model_is_rejected_without_changing_parent(
    parent_config: AgentConfig, effort_key: str
) -> None:
    before = parent_config.model_dump()
    unknown_model = f"{SYNTHETIC_ROUTING_MODEL}-unknown"
    with pytest.raises(ValueError, match=f"Model {unknown_model} from provider refuseless is not supported") as caught:
        resolve_subagent_model(
            parent_config,
            "coder",
            {"provider": "refuseless", "model": unknown_model, effort_key: "high"},
            effort_key=effort_key,
        )
    assert "Omit model_override entirely" in str(caught.value)
    assert FAKE_REFUSELESS_KEY not in str(caught.value)
    assert parent_config.model_dump() == before


@pytest.mark.parametrize("effort_key", ["thinking_effort", "effort"], ids=["ordinary-dispatch", "gigacode"])
@pytest.mark.parametrize(
    ("effort", "message"),
    [
        ("medium", "Unsupported thinking effort"),
        ("minimal", "Unsupported thinking effort"),
        ("xhigh", "Unsupported thinking effort"),
        ("none", "Unsupported thinking effort"),
        (None, "effort must be a string"),
        ("", "must be a non-empty string or null"),
        (42, "must be a non-empty string or null"),
    ],
)
@pytest.mark.usefixtures("synthetic_refuseless_routing_catalog")
def test_synthetic_routing_invalid_effort_and_null_are_rejected(
    parent_config: AgentConfig, effort_key: str, effort: object, message: str
) -> None:
    before = parent_config.model_dump()
    with pytest.raises(ValueError, match=message):
        resolve_subagent_model(
            parent_config,
            "planning-agent",
            {"provider": "refuseless", "model": SYNTHETIC_ROUTING_MODEL, effort_key: effort},
            effort_key=effort_key,
        )
    assert parent_config.model_dump() == before


@pytest.mark.parametrize("effort_key", ["thinking_effort", "effort"], ids=["ordinary-dispatch", "gigacode"])
@pytest.mark.parametrize("missing_field", ["provider", "model", "effort_field"])
def test_synthetic_routing_override_requires_all_three_contract_fields(
    parent_config: AgentConfig, effort_key: str, missing_field: str
) -> None:
    override = {"provider": "refuseless", "model": SYNTHETIC_ROUTING_MODEL, effort_key: "high"}
    del override[effort_key if missing_field == "effort_field" else missing_field]
    with pytest.raises(ValueError, match="missing required field"):
        resolve_subagent_model(parent_config, "general-agent", override, effort_key=effort_key)


def _catalog_efforts(model: str) -> tuple[str, ...]:
    return ("off", "low", "high", "max") if model.startswith("deepseek-") else ("low", "high", "max")


def test_real_refuseless_catalog_discovery_exposes_all_five_routes(parent_config: AgentConfig) -> None:
    catalog = subagent_model_catalog(parent_config)
    filtered = subagent_model_catalog(parent_config, " REFUSELESS ")
    assert "refuseless" in configured_provider_names(parent_config)
    assert [entry["provider"] for entry in filtered["providers"]] == ["refuseless"]
    models = filtered["providers"][0]["models"]
    assert tuple(entry["model"] for entry in models) == REFUSELESS_MODELS
    assert models == next(entry["models"] for entry in catalog["providers"] if entry["provider"] == "refuseless")
    assert SYNTHETIC_ROUTING_MODEL not in {model for provider, model in MODEL_SPECS if provider == "refuseless"}
    rendered = render_subagent_model_catalog(filtered)
    for entry in models:
        model = entry["model"]
        vision = model != REFUSELESS_MODELS[0]
        assert entry == {
            "model": model,
            "thinking_efforts": list(_catalog_efforts(model)),
            "default_thinking_effort": "low",
            "override_effort": "string",
            "supports_vision": vision,
        }
        assert (
            f"| `refuseless/{model}` | `{', '.join(_catalog_efforts(model))}` | `low` | {'yes' if vision else 'no'} |"
            in rendered
        )
    assert filtered["override_contract"]["ordinary_dispatch"]["effort_field"] == "thinking_effort"
    assert filtered["override_contract"]["gigacode"]["effort_field"] == "effort"
    for secret in (FAKE_REFUSELESS_KEY, *FAKE_OTHER_KEYS.values()):
        assert secret not in json.dumps(catalog)
        assert secret not in rendered


@pytest.mark.parametrize("missing_key", (None, ""))
@pytest.mark.parametrize("model", REFUSELESS_MODELS)
@pytest.mark.parametrize("effort_key", ("thinking_effort", "effort"), ids=("ordinary-dispatch", "gigacode"))
def test_real_refuseless_catalog_routes_require_own_key(
    parent_config: AgentConfig, missing_key: str | None, model: str, effort_key: str
) -> None:
    assert ("refuseless", model) in MODEL_SPECS
    config = parent_config.model_copy(update={"refuseless_api_key": missing_key})
    assert not provider_is_configured(config, ModelProvider.REFUSELESS)
    assert "refuseless" not in configured_provider_names(config)
    catalog = subagent_model_catalog(config)
    assert "refuseless" not in {entry["provider"] for entry in catalog["providers"]}
    assert not any(
        f"`refuseless/{candidate}`" in render_subagent_model_catalog(catalog) for candidate in REFUSELESS_MODELS
    )
    with pytest.raises(ValueError, match="Provider 'refuseless' is not configured"):
        subagent_model_catalog(config, "refuseless")
    before = config.model_dump()
    with pytest.raises(ValueError, match="Provider 'refuseless' is not configured") as caught:
        resolve_subagent_model(
            config,
            "investigation-agent",
            {"provider": "refuseless", "model": model, effort_key: "low"},
            effort_key=effort_key,
        )
    assert config.model_dump() == before
    for secret in FAKE_OTHER_KEYS.values():
        assert secret not in str(caught.value)


@pytest.mark.parametrize("agent_name", AGENT_ROLE_BY_NAME)
@pytest.mark.parametrize("effort_key", ("thinking_effort", "effort"), ids=("ordinary-dispatch", "gigacode"))
@pytest.mark.parametrize(
    ("model", "effort"),
    [(model, effort) for model in REFUSELESS_MODELS for effort in _catalog_efforts(model)],
)
def test_real_refuseless_catalog_atomic_override_routes_every_native_effort(
    parent_config: AgentConfig, agent_name: str, effort_key: str, model: str, effort: str
) -> None:
    before = parent_config.model_dump()
    fingerprint = model_routing_fingerprint(parent_config)
    inherited = parent_config.model_config_for_agent(agent_name)
    resolved = resolve_subagent_model(
        parent_config,
        agent_name,
        {"provider": " REFUSELESS ", "model": f" {model} ", effort_key: f" {effort.upper()} "},
        effort_key=effort_key,
    )
    selected = resolved.model_config
    assert selected.provider is ModelProvider.REFUSELESS
    assert selected.model == model
    assert selected.thinking_effort == effort
    assert selected.rate_limits == inherited.rate_limits
    assert selected.rate_limits is not inherited.rate_limits
    assert resolved.config is not parent_config
    assert resolved.config.model_config_for_agent(agent_name) is selected
    assert resolved.config.get_api_key(selected.provider) == FAKE_REFUSELESS_KEY
    assert resolved.requested is not None
    assert resolved.requested.as_dict(effort_key=effort_key) == {
        "provider": "refuseless",
        "model": model,
        effort_key: effort,
    }
    assert resolved.metadata == {"provider": "refuseless", "model": model, "thinking_effort": effort}
    assert resolved.config.model_dump(exclude={"agent_models"}) == parent_config.model_dump(exclude={"agent_models"})
    for other_agent in AGENT_ROLE_BY_NAME:
        if other_agent != agent_name:
            assert resolved.config.model_config_for_agent(other_agent) == parent_config.model_config_for_agent(
                other_agent
            )
    assert model_routing_fingerprint(resolved.config) != fingerprint
    for secret in (FAKE_REFUSELESS_KEY, *FAKE_OTHER_KEYS.values()):
        assert secret not in json.dumps(resolved.metadata)
    selected.rate_limits.requests_per_minute += 1
    assert parent_config.model_dump() == before
    assert model_routing_fingerprint(parent_config) == fingerprint


@pytest.mark.parametrize("model", REFUSELESS_MODELS)
@pytest.mark.parametrize("effort_key", ("thinking_effort", "effort"), ids=("ordinary-dispatch", "gigacode"))
@pytest.mark.parametrize("effort", ("none", "medium", "minimal", "xhigh", None))
def test_real_refuseless_catalog_rejects_unknown_effort_or_null(
    parent_config: AgentConfig, model: str, effort_key: str, effort: str | None
) -> None:
    before = parent_config.model_dump()
    message = "effort must be a string" if effort is None else "Unsupported thinking effort"
    with pytest.raises(ValueError, match=message):
        resolve_subagent_model(
            parent_config,
            "planning-agent",
            {"provider": "refuseless", "model": model, effort_key: effort},
            effort_key=effort_key,
        )
    assert parent_config.model_dump() == before


@pytest.mark.parametrize("model", (REFUSELESS_MODELS[0], REFUSELESS_MODELS[1], REFUSELESS_MODELS[-1]))
@pytest.mark.parametrize("effort_key", ("thinking_effort", "effort"), ids=("ordinary-dispatch", "gigacode"))
def test_real_refuseless_glm_and_minimax_reject_native_off(
    parent_config: AgentConfig, model: str, effort_key: str
) -> None:
    before = parent_config.model_dump()
    with pytest.raises(ValueError, match="Unsupported thinking effort"):
        resolve_subagent_model(
            parent_config,
            "coder",
            {"provider": "refuseless", "model": model, effort_key: "off"},
            effort_key=effort_key,
        )
    assert parent_config.model_dump() == before


@pytest.mark.parametrize("model", ("glm-5.3-unknown", "deepseek-v4.1-flash", "private/not-a-public-model"))
@pytest.mark.parametrize("effort_key", ("thinking_effort", "effort"), ids=("ordinary-dispatch", "gigacode"))
def test_real_refuseless_catalog_rejects_unknown_models(
    parent_config: AgentConfig, model: str, effort_key: str
) -> None:
    before = parent_config.model_dump()
    with pytest.raises(ValueError, match="from provider refuseless is not supported") as caught:
        resolve_subagent_model(
            parent_config,
            "general-agent",
            {"provider": "refuseless", "model": model, effort_key: "low"},
            effort_key=effort_key,
        )
    assert "Omit model_override entirely" in str(caught.value)
    assert FAKE_REFUSELESS_KEY not in str(caught.value)
    assert parent_config.model_dump() == before
