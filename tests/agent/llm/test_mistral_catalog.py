"""First-party Mistral catalog/registration invariants, without account access."""

from numbers import Real

import pytest

from kolega_code.cli.provider_registry import PROVIDER_DEFAULT_MODEL, get_ui_model, ui_model_options
from kolega_code.config import ModelProvider
from kolega_code.llm.client import LLMClient
from kolega_code.llm.providers.mistral import MistralProvider
from kolega_code.llm.specs import MODEL_SPECS, get_model_specs, supports_vision
from kolega_code.llm.specs.accessors import resolve_max_input_tokens
from kolega_code.llm.specs.thinking import build_thinking_request_params, validate_thinking_effort
from kolega_code.llm.usage import OPENAI_USAGE_PROVIDERS, usage_token_fields

PROVIDER = "mistral"
DEFAULT = "mistral-medium-3-5"
REQUIRED_MODELS = {
    DEFAULT,
    "mistral-small-2603",
    "codestral-2508",
    "ministral-3b-2512",
    "ministral-8b-2512",
    "ministral-14b-2512",
}


def _catalog() -> dict[str, dict]:
    return {model: specs for (provider, model), specs in MODEL_SPECS.items() if provider == PROVIDER}


def test_native_provider_is_registered_and_medium_is_first_default() -> None:
    provider = ModelProvider(PROVIDER)
    catalog = _catalog()
    assert REQUIRED_MODELS <= catalog.keys()
    assert {"mistral-large-4", "mistral-large-4-0"} & catalog.keys(), "Include the native Large 4 preview"
    assert next(iter(catalog)) == DEFAULT
    assert PROVIDER_DEFAULT_MODEL[provider] == DEFAULT
    assert LLMClient._provider_class(PROVIDER) is MistralProvider
    assert PROVIDER in OPENAI_USAGE_PROVIDERS
    assert usage_token_fields(PROVIDER) == ("prompt_tokens", "completion_tokens")


def test_native_entries_have_positive_explicit_context_and_output_policy() -> None:
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
        assert specs["input_budget"] == "window_minus_output", model
        assert resolve_max_input_tokens(specs) == context - output, model
        assert get_model_specs(PROVIDER, model) == specs, model
        assert not specs.get("supports_hosted_web_search", False), model


@pytest.mark.parametrize("model", [DEFAULT, "mistral-small-2603"])
def test_medium_and_small_have_vision_and_native_none_high_effort(model: str) -> None:
    assert supports_vision(PROVIDER, model)
    specs = get_model_specs(PROVIDER, model)
    effort = specs["thinking_effort"]
    assert effort.options == ("none", "high")
    assert effort.default in effort.options
    assert build_thinking_request_params(PROVIDER, model, "none") == {"reasoning_effort": "none"}
    assert build_thinking_request_params(PROVIDER, model, "high") == {"reasoning_effort": "high"}
    for unsupported in ("minimal", "low", "medium", "xhigh", "max"):
        with pytest.raises(ValueError):
            validate_thinking_effort(PROVIDER, model, unsupported)


def test_only_eligible_chat_families_are_seeded_not_retired_or_non_chat_models() -> None:
    catalog = _catalog()
    assert catalog
    for model in catalog:
        assert model.startswith(
            ("mistral-medium", "mistral-small", "mistral-large-4", "mistral-large-latest", "ministral-", "codestral-")
        ), model
        assert not any(
            excluded in model for excluded in ("devstral", "magistral", "embed", "ocr", "voxtral", "moderation")
        ), model


def test_text_only_codestral_is_not_advertised_as_vision_capable() -> None:
    assert not supports_vision(PROVIDER, "codestral-2508")


def test_picker_resolves_native_catalog_without_gateway_prefixes() -> None:
    catalog = _catalog()
    listed = [model for _label, model in ui_model_options(PROVIDER)]
    assert listed == list(catalog)
    for model in catalog:
        option = get_ui_model(PROVIDER, model)
        assert option is not None
        assert option.model == model
        assert "/" not in model
    assert DEFAULT in listed
