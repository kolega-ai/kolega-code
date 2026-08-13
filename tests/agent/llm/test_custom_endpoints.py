"""Custom endpoints: identity, spec sync, client routing, thinking, and reasoning replay."""

from __future__ import annotations

from typing import Any, cast

import pytest

from kolega_code.config import AgentConfig, CustomEndpointConfig, ModelConfig, ModelProvider
from kolega_code.llm.client import LLMClient
from kolega_code.llm.models import Message, MessageHistory, TextBlock, ThinkingBlock
from kolega_code.llm.providers.models import GenerationParams
from kolega_code.llm.specs import (
    MODEL_SPECS,
    get_model_specs,
    model_is_known,
    resolve_max_input_tokens,
    supports_vision,
    sync_custom_endpoint_specs,
    thinking_effort_options,
)
from kolega_code.llm.specs.catalog import WILDCARD_MODEL_SPECS
from kolega_code.llm.specs.custom_endpoints import CUSTOM_PROVIDER_PREFIX, custom_endpoint_id, is_custom_provider
from kolega_code.llm.specs.thinking import (
    build_thinking_request_params,
    default_thinking_effort,
    reasoning_replay_field,
)

CHAT_ENDPOINT = {
    "api_style": "openai_chat",
    "base_url": "http://localhost:1234/v1",
    "context_length": 32768,
    "max_output_tokens": 8192,
    "thinking": {"mode": "thinking_toggle"},
    "models": {"big-model": {"context_length": 65536, "supports_vision": True}},
}

ANTHROPIC_ENDPOINT = {
    "api_style": "anthropic",
    "base_url": "http://localhost:8080",
    "max_output_tokens": 16384,
    "thinking": {
        "mode": "anthropic_budget",
        "options": ["none", "low", "medium", "high"],
        "default": "medium",
        "budgets": {"low": 2048, "medium": 8192, "high": 12288},
    },
}


@pytest.fixture(autouse=True)
def _clean_custom_specs():
    yield
    sync_custom_endpoint_specs({})


def _config(endpoints, *, provider: str, model: str) -> AgentConfig:
    return AgentConfig(
        anthropic_api_key="sk-test",
        long_context_config=ModelConfig(provider=ModelProvider(provider), model=model),
        custom_endpoints={
            endpoint_id: CustomEndpointConfig.model_validate(entry) for endpoint_id, entry in endpoints.items()
        },
    )


# --- provider identity ----------------------------------------------------


def test_model_provider_missing_creates_dynamic_members():
    first = ModelProvider("custom:lmstudio")
    second = ModelProvider("custom:lmstudio")
    assert first is second
    assert first.value == "custom:lmstudio"
    assert str(first) == "ModelProvider.custom:lmstudio"
    assert "custom:lmstudio" not in ModelProvider.__members__
    with pytest.raises(ValueError):
        ModelProvider("custom:Bad Slug")
    with pytest.raises(ValueError):
        ModelProvider("bogus")


def test_model_config_round_trips_custom_provider():
    config = ModelConfig(provider=ModelProvider("custom:lmstudio"), model="qwen2.5")
    assert config.provider.value == "custom:lmstudio"
    dumped = config.model_dump()
    assert dumped["provider"].value == "custom:lmstudio"
    reloaded = ModelConfig.model_validate(config.model_dump(mode="json"))
    assert reloaded.provider.value == "custom:lmstudio"
    assert ModelConfig(provider=ModelProvider("custom:lmstudio"), model="x").provider.value == "custom:lmstudio"


def test_custom_endpoint_helpers():
    assert is_custom_provider(ModelProvider("custom:lmstudio"))
    assert not is_custom_provider(ModelProvider.OPENAI)
    assert custom_endpoint_id(ModelProvider("custom:lmstudio")) == "lmstudio"
    assert custom_endpoint_id(ModelProvider.OPENAI) is None


# --- spec sync ------------------------------------------------------------


def test_sync_registers_wildcard_and_exact_specs():
    sync_custom_endpoint_specs({"lmstudio": CHAT_ENDPOINT})
    provider = f"{CUSTOM_PROVIDER_PREFIX}lmstudio"

    wildcard = get_model_specs(provider, "any-model")
    assert wildcard["context_length"] == 32768
    assert wildcard["max_completion_tokens"] == 8192
    assert wildcard["input_budget"] == "window_minus_output"
    assert wildcard["supports_vision"] is False
    assert wildcard["reasoning_replay"] == "auto"
    assert wildcard["thinking_effort"].mode == "thinking_toggle"

    exact = get_model_specs(provider, "big-model")
    assert exact["context_length"] == 65536
    assert exact["supports_vision"] is True
    assert exact["reasoning_replay"] == "auto"

    assert model_is_known(provider, "anything-else")
    assert supports_vision(provider, "big-model")
    assert not supports_vision(provider, "plain-model")
    assert resolve_max_input_tokens(wildcard) == 32768 - 8192


def test_sync_is_idempotent_and_removes_stale_entries():
    sync_custom_endpoint_specs({"a": CHAT_ENDPOINT, "b": CHAT_ENDPOINT})
    assert ("custom:a", "*") in WILDCARD_MODEL_SPECS
    assert ("custom:b", "*") in WILDCARD_MODEL_SPECS

    sync_custom_endpoint_specs({"a": CHAT_ENDPOINT})
    assert ("custom:a", "*") in WILDCARD_MODEL_SPECS
    assert ("custom:b", "*") not in WILDCARD_MODEL_SPECS

    sync_custom_endpoint_specs({"a": CHAT_ENDPOINT})
    assert ("custom:a", "*") in WILDCARD_MODEL_SPECS

    sync_custom_endpoint_specs({})
    assert not any(key[0].startswith(CUSTOM_PROVIDER_PREFIX) for key in WILDCARD_MODEL_SPECS)
    assert not any(key[0].startswith(CUSTOM_PROVIDER_PREFIX) for key in MODEL_SPECS)


def test_sync_skips_invalid_ids_and_drops_invalid_thinking_blocks():
    sync_custom_endpoint_specs(
        {
            "bad-id!": CHAT_ENDPOINT,
            "bad-thinking": {**CHAT_ENDPOINT, "thinking": {"mode": "nope"}},
            "bad-budgets": {**ANTHROPIC_ENDPOINT, "thinking": {"mode": "anthropic_budget", "budgets": {}}},
            "good": CHAT_ENDPOINT,
        }
    )
    assert ("custom:bad-id!", "*") not in WILDCARD_MODEL_SPECS
    assert get_model_specs("custom:bad-thinking", "m").get("thinking_effort") is None
    assert get_model_specs("custom:bad-budgets", "m").get("thinking_effort") is None
    assert get_model_specs("custom:good", "m")["thinking_effort"].mode == "thinking_toggle"


def test_reasoning_replay_only_declared_for_chat_style():
    sync_custom_endpoint_specs(
        {
            "chat": CHAT_ENDPOINT,
            "anthropic": ANTHROPIC_ENDPOINT,
            "responses": {"api_style": "openai_responses", "base_url": "http://localhost:8000/v1"},
        }
    )
    assert "reasoning_replay" in get_model_specs("custom:chat", "m")
    assert "reasoning_replay" not in get_model_specs("custom:anthropic", "m")
    assert "reasoning_replay" not in get_model_specs("custom:responses", "m")


# --- thinking -------------------------------------------------------------


def test_thinking_toggle_and_anthropic_budget_params():
    sync_custom_endpoint_specs({"chat": CHAT_ENDPOINT, "anthropic": ANTHROPIC_ENDPOINT})
    assert build_thinking_request_params("custom:chat", "m", "enabled") == {
        "extra_body": {"thinking": {"type": "enabled"}}
    }
    assert build_thinking_request_params("custom:chat", "m", "none") == {
        "extra_body": {"thinking": {"type": "disabled"}}
    }
    assert build_thinking_request_params("custom:anthropic", "m", "low") == {
        "thinking": {"type": "enabled", "budget_tokens": 2048}
    }
    assert build_thinking_request_params("custom:anthropic", "m", "none") == {"thinking": {"type": "disabled"}}
    assert thinking_effort_options("custom:chat", "m") == ("none", "enabled")
    assert default_thinking_effort("custom:chat", "m") == "enabled"
    assert thinking_effort_options("custom:anthropic", "m") == ("none", "low", "medium", "high")
    assert default_thinking_effort("custom:anthropic", "m") == "medium"
    assert build_thinking_request_params("custom:chat", "m", None) == {}


# --- reasoning replay -----------------------------------------------------


def test_reasoning_replay_field_spec_declared_values():
    sync_custom_endpoint_specs(
        {
            "auto": CHAT_ENDPOINT,
            "content": {**CHAT_ENDPOINT, "reasoning_replay": "reasoning_content"},
            "field": {**CHAT_ENDPOINT, "reasoning_replay": "reasoning"},
            "off": {**CHAT_ENDPOINT, "reasoning_replay": "off"},
            "anthropic": ANTHROPIC_ENDPOINT,
        }
    )
    assert reasoning_replay_field("custom:auto", "m") == "auto"
    assert reasoning_replay_field("custom:content", "m") == "reasoning_content"
    assert reasoning_replay_field("custom:field", "m") == "reasoning"
    assert reasoning_replay_field("custom:off", "m") is None
    assert reasoning_replay_field("custom:anthropic", "m") is None
    assert reasoning_replay_field("custom:missing", "m") is None


def _assistant(provider: str, *, reasoning_field: str | None = "reasoning"):
    metadata = {"provider": provider}
    if reasoning_field:
        metadata["reasoning_field"] = reasoning_field
    return Message(role="assistant", content=[ThinkingBlock(thinking="cot"), TextBlock("ans")], usage_metadata=metadata)


def test_to_openai_auto_replay_uses_sniffed_field():
    sync_custom_endpoint_specs({"chat": CHAT_ENDPOINT})
    out = MessageHistory([_assistant("custom:chat", reasoning_field="reasoning")]).to_openai(
        provider="custom:chat", model="m"
    )[0]
    assert out["reasoning"] == "cot"
    assert out["content"] == [{"type": "text", "text": "ans"}]

    out = MessageHistory([_assistant("custom:chat", reasoning_field="reasoning_content")]).to_openai(
        provider="custom:chat", model="m"
    )[0]
    assert out["reasoning_content"] == "cot"
    assert "reasoning" not in out


def test_to_openai_auto_replay_falls_back_to_reasoning():
    sync_custom_endpoint_specs({"chat": CHAT_ENDPOINT})
    out = MessageHistory([_assistant("custom:chat", reasoning_field=None)]).to_openai(
        provider="custom:chat", model="m"
    )[0]
    assert out["reasoning"] == "cot"
    assert out["content"] == [{"type": "text", "text": "ans"}]


def test_to_openai_replay_override_and_off():
    sync_custom_endpoint_specs(
        {
            "content": {**CHAT_ENDPOINT, "reasoning_replay": "reasoning_content"},
            "off": {**CHAT_ENDPOINT, "reasoning_replay": "off"},
        }
    )
    out = MessageHistory([_assistant("custom:content", reasoning_field="reasoning")]).to_openai(
        provider="custom:content", model="m"
    )[0]
    assert out["reasoning_content"] == "cot"

    out = MessageHistory([_assistant("custom:off")]).to_openai(provider="custom:off", model="m")[0]
    assert "reasoning" not in out
    assert out["content"] == [{"type": "text", "text": "*Thinking:*\ncot"}, {"type": "text", "text": "ans"}]


def test_to_openai_replay_never_crosses_endpoints():
    sync_custom_endpoint_specs({"chat": CHAT_ENDPOINT, "other": CHAT_ENDPOINT})
    out = MessageHistory([_assistant("custom:chat")]).to_openai(provider="custom:other", model="m")[0]
    assert "reasoning" not in out
    assert {"type": "text", "text": "*Thinking:*\ncot"} in out["content"]


# --- client routing -------------------------------------------------------


def test_llm_client_routes_custom_endpoints():
    llm = LLMClient(provider="custom:chat", api_key="", base_url="http://localhost:1234/v1", api_style="openai_chat")
    from kolega_code.llm.providers.openai import OpenAIProvider

    assert isinstance(llm.provider, OpenAIProvider)
    assert llm.provider.base_url == "http://localhost:1234/v1"
    assert llm.provider_name == "custom:chat"

    llm = LLMClient(provider="custom:local", api_key="", base_url="http://localhost:8080", api_style="anthropic")
    from kolega_code.llm.providers.anthropic import AnthropicProvider

    assert isinstance(llm.provider, AnthropicProvider)
    assert llm.provider.use_local_token_counting

    llm = LLMClient(
        provider="custom:vllm", api_key="", base_url="http://localhost:8000/v1", api_style="openai_responses"
    )
    from kolega_code.llm.providers.openai_responses import OpenAIResponsesProvider

    assert isinstance(llm.provider, OpenAIResponsesProvider)
    assert llm.provider.base_url == "http://localhost:8000/v1"


def test_llm_client_custom_endpoint_requires_base_url_and_style():
    from kolega_code.llm.exceptions import LLMError

    with pytest.raises(LLMError, match="base_url and api_style"):
        LLMClient(provider="custom:chat", api_key="", api_style="openai_chat")
    with pytest.raises(LLMError, match="base_url and api_style"):
        LLMClient(provider="custom:chat", api_key="", base_url="http://localhost:1/v1")
    with pytest.raises(LLMError, match="api_style"):
        LLMClient(provider="custom:chat", api_key="", base_url="http://localhost:1/v1", api_style="grpc")


def test_responses_include_gated_to_builtin_backends():
    sync_custom_endpoint_specs(
        {
            "vllm": {
                "api_style": "openai_responses",
                "base_url": "http://localhost:8000/v1",
                "thinking": {"mode": "openai_responses_reasoning"},
            }
        }
    )
    custom = LLMClient(
        provider="custom:vllm", api_key="", base_url="http://localhost:8000/v1", api_style="openai_responses"
    )
    req = cast(Any, custom.provider)._build_request(
        MessageHistory([Message(role="user", content=[TextBlock(text="hi")])]),
        None,
        GenerationParams(thinking="low", temperature=1.0),
        {"model": "gpt-oss"},
    )
    assert req.get("reasoning") == {"effort": "low", "summary": "auto"}
    assert "include" not in req

    builtin = LLMClient(provider="openai", api_key="sk-x")
    req = cast(Any, builtin.provider)._build_request(
        MessageHistory([Message(role="user", content=[TextBlock(text="hi")])]),
        None,
        GenerationParams(thinking="medium", temperature=1.0),
        {"model": "gpt-5.6-sol"},
    )
    assert req.get("include") == ["reasoning.encrypted_content"]


# --- AgentConfig ----------------------------------------------------------


def test_agent_config_custom_endpoint_plumbing():
    config = _config({"lmstudio": CHAT_ENDPOINT}, provider="custom:lmstudio", model="m")
    endpoint = config.custom_endpoint_for(config.long_context_config)
    assert endpoint is not None
    assert endpoint.api_style == "openai_chat"
    assert endpoint.base_url == "http://localhost:1234/v1"
    assert config.get_api_key(ModelProvider("custom:lmstudio")) is None
    assert ("custom:lmstudio", "*") in WILDCARD_MODEL_SPECS


def test_agent_config_endpoint_api_key_resolves():
    config = _config(
        {"lmstudio": {**CHAT_ENDPOINT, "api_key": "secret"}},
        provider="custom:lmstudio",
        model="m",
    )
    assert config.get_api_key(ModelProvider("custom:lmstudio")) == "secret"
    assert config.get_api_key(ModelProvider.OPENAI) is None


def test_agent_config_keyless_custom_provider_passes_validation():
    _config({"lmstudio": CHAT_ENDPOINT}, provider="custom:lmstudio", model="m")


def test_custom_endpoint_config_validation():
    config = CustomEndpointConfig(api_style="openai_chat", base_url="http://localhost:1234/v1/")
    assert config.base_url == "http://localhost:1234/v1"
    with pytest.raises(ValueError):
        CustomEndpointConfig(api_style="openai_chat", base_url="localhost:1234/v1")
    with pytest.raises(ValueError):
        CustomEndpointConfig(api_style="openai_chat", base_url="http://x/v1", reasoning_replay="sideways")
    with pytest.raises(ValueError):
        CustomEndpointConfig.model_validate({"api_style": "grpc", "base_url": "http://x/v1"})


# --- usage normalization --------------------------------------------------


def test_normalize_usage_for_custom_endpoints_by_shape():
    from kolega_code.llm.usage import normalize_usage, usage_token_fields

    openai_shaped = {"prompt_tokens": 11, "completion_tokens": 22, "total_tokens": 33}
    norm = normalize_usage(openai_shaped, "custom:chat", "m")
    assert (norm.input_tokens, norm.output_tokens, norm.total_tokens) == (11, 22, 33)

    anthropic_shaped = {"input_tokens": 11, "output_tokens": 22}
    norm = normalize_usage(anthropic_shaped, "custom:anthropic", "m")
    assert (norm.input_tokens, norm.output_tokens) == (11, 22)

    unreported = normalize_usage({}, "custom:chat", "m")
    assert unreported.reported is False

    assert usage_token_fields("custom:chat", metadata=openai_shaped) == ("prompt_tokens", "completion_tokens")
    assert usage_token_fields("custom:anthropic", metadata=anthropic_shaped) == ("input_tokens", "output_tokens")
    assert usage_token_fields("custom:chat") == ("prompt_tokens", "completion_tokens")


# --- temperature ----------------------------------------------------------


def test_sync_maps_endpoint_temperature_into_spec():
    sync_custom_endpoint_specs(
        {
            "hot": {**CHAT_ENDPOINT, "temperature": 0.3},
            "default": CHAT_ENDPOINT,
            "bad": {**CHAT_ENDPOINT, "temperature": 7},
            "override": {**CHAT_ENDPOINT, "temperature": 0.4, "models": {"m": {"temperature": 0.9}}},
        }
    )
    assert get_model_specs("custom:hot", "m")["default_temperature"] == 0.3
    assert get_model_specs("custom:default", "m")["default_temperature"] == 1.0
    assert get_model_specs("custom:bad", "m")["default_temperature"] == 1.0
    assert get_model_specs("custom:override", "m")["default_temperature"] == 0.9


def test_custom_endpoint_config_temperature_validation():
    CustomEndpointConfig(api_style="openai_chat", base_url="http://x/v1", temperature=0.5)
    CustomEndpointConfig(api_style="openai_chat", base_url="http://x/v1", temperature=2)
    with pytest.raises(ValueError):
        CustomEndpointConfig(api_style="openai_chat", base_url="http://x/v1", temperature=0)
    with pytest.raises(ValueError):
        CustomEndpointConfig(api_style="openai_chat", base_url="http://x/v1", temperature=2.5)
