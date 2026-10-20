"""First-party MiniMax catalog/registration invariants, without account access.

The numbers asserted here come from live probes against
``https://api.minimax.io/anthropic`` (recorded in ``specs/catalog/minimax.py``);
this module keeps them honest without spending tokens.
"""

from numbers import Real

import pytest

from kolega_code.cli.provider_registry import (
    MODEL_LABELS,
    PROVIDER_DEFAULT_MODEL,
    PROVIDER_LABELS,
    ui_model_options,
)
from kolega_code.config import ModelProvider
from kolega_code.llm.client import LLMClient
from kolega_code.llm.providers.anthropic import AnthropicProvider
from kolega_code.llm.specs import MODEL_SPECS, get_model_specs, supports_vision
from kolega_code.llm.specs.accessors import resolve_max_input_tokens
from kolega_code.llm.specs.thinking import build_thinking_request_params
from kolega_code.llm.usage import ANTHROPIC_USAGE_PROVIDERS, usage_token_fields

PROVIDER = "minimax"
DEFAULT = "MiniMax-M3.1-Flash"
FRONTIER = {DEFAULT, "MiniMax-M3"}
M2_FAMILY = {
    "MiniMax-M2.7",
    "MiniMax-M2.7-highspeed",
    "MiniMax-M2.5",
    "MiniMax-M2.5-highspeed",
    "MiniMax-M2.1",
    "MiniMax-M2.1-highspeed",
    "MiniMax-M2",
}


def _catalog() -> dict[str, dict]:
    return {model: specs for (provider, model), specs in MODEL_SPECS.items() if provider == PROVIDER}


def test_provider_routes_to_the_anthropic_compatible_endpoint() -> None:
    client = LLMClient(provider=PROVIDER, api_key="sk-test")
    assert isinstance(client.provider, AnthropicProvider)
    assert client.provider.base_url == "https://api.minimax.io/anthropic"
    assert client.provider.provider_name == PROVIDER
    # MiniMax serves POST /anthropic/v1/messages/count_tokens, so billing and
    # preflight counting must use the provider's own numbers, not tiktoken.
    assert client.provider.use_local_token_counting is False


def test_catalog_registers_every_live_model_with_flash_first() -> None:
    provider = ModelProvider(PROVIDER)
    catalog = _catalog()
    assert set(catalog) == FRONTIER | M2_FAMILY
    assert next(iter(catalog)) == DEFAULT
    assert PROVIDER_DEFAULT_MODEL[provider] == DEFAULT
    assert PROVIDER_LABELS[provider] == "MiniMax"
    assert [model for _, model in ui_model_options(PROVIDER)] == list(catalog)
    for model in catalog:
        assert model in MODEL_LABELS


def test_flash_and_m3_have_a_separate_output_allowance() -> None:
    # Live: 600206 input + max_tokens=524288 is accepted (1.12M combined) while
    # 1.1M input alone is rejected, so the 1M window is input-only.
    for model in FRONTIER:
        specs = get_model_specs(PROVIDER, model)
        assert specs["context_length"] == 1048576, model
        assert specs["max_completion_tokens"] == 524288, model
        assert specs["input_budget"] == "separate_output_limit", model
        assert resolve_max_input_tokens(specs) == 1048576, model
        assert specs["supports_vision"] is True, model


def test_m2_family_budgets_the_shared_combined_window() -> None:
    # Live: 131236 input + 196608 output accepted; 140136 + 196608 rejected.
    # The agent always requests the full output cap, so the combined ceiling is
    # modelled as context_length and the input budget falls out of it.
    for model in M2_FAMILY:
        specs = get_model_specs(PROVIDER, model)
        assert specs["context_length"] == 327680, model
        assert specs["max_completion_tokens"] == 196608, model
        assert specs["input_budget"] == "window_minus_output", model
        assert resolve_max_input_tokens(specs) == 131072, model
        assert specs["supports_vision"] is False, model


def test_every_spec_declares_physical_limits() -> None:
    catalog = _catalog()
    assert catalog
    for model, specs in catalog.items():
        context = specs["context_length"]
        output = specs["max_completion_tokens"]
        assert type(context) is int and context > 0, model
        assert type(output) is int and 0 < output < context, model
        assert isinstance(specs["default_temperature"], Real), model
        assert not isinstance(specs["default_temperature"], bool), model
        assert isinstance(specs["supports_vision"], bool), model
        assert supports_vision(PROVIDER, model) is specs["supports_vision"], model
        assert get_model_specs(PROVIDER, model) == specs, model


def test_only_flash_exposes_a_thinking_ladder() -> None:
    from kolega_code.llm.specs import thinking_effort_options

    assert thinking_effort_options(PROVIDER, DEFAULT) == ("low", "medium", "high", "xhigh", "max")
    # M3 never surfaces thinking blocks; M2.x always emits them and ignores the
    # toggle. Exposing a control that does nothing would mislead the picker.
    for model in FRONTIER - {DEFAULT} | M2_FAMILY:
        assert thinking_effort_options(PROVIDER, model) == (), model


def test_flash_thinking_serializes_to_adaptive_effort() -> None:
    params = build_thinking_request_params(PROVIDER, DEFAULT, "low")
    assert params == {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}}
    default_params = build_thinking_request_params(PROVIDER, DEFAULT, "max")
    assert default_params == {"thinking": {"type": "adaptive"}, "output_config": {"effort": "max"}}


def test_flash_rejects_disabling_thinking() -> None:
    # The API answers 400 for both spellings; catching it here keeps the
    # picker from offering an option the endpoint refuses.
    with pytest.raises(ValueError):
        build_thinking_request_params(PROVIDER, DEFAULT, "none")
    with pytest.raises(ValueError):
        build_thinking_request_params(PROVIDER, DEFAULT, "auto")


def test_vision_picker_only_lists_the_multimodal_models() -> None:
    assert dict(ui_model_options(PROVIDER, vision_only=True)) == {
        "MiniMax M3.1 Flash": "MiniMax-M3.1-Flash",
        "MiniMax M3": "MiniMax-M3",
    }


def test_usage_is_parsed_from_the_anthropic_shape() -> None:
    assert PROVIDER in ANTHROPIC_USAGE_PROVIDERS
    assert usage_token_fields(PROVIDER) == ("input_tokens", "output_tokens")
