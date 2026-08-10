"""Invariants of the bundled Ollama Cloud model catalog."""

from numbers import Real

from kolega_code.cli.provider_registry import PROVIDER_DEFAULT_MODEL, get_ui_model, ui_model_options
from kolega_code.config import ModelProvider
from kolega_code.llm.specs import MODEL_SPECS
from kolega_code.llm.specs.catalog.ollama_cloud import OLLAMA_CLOUD_SPECS

PROVIDER = ModelProvider.OLLAMA_CLOUD.value
OLLAMA_CLOUD_KEYS = list(OLLAMA_CLOUD_SPECS)
OLLAMA_CLOUD_MODELS = [model for _provider, model in OLLAMA_CLOUD_KEYS]


def test_catalog_is_populated_and_namespaced() -> None:
    assert OLLAMA_CLOUD_KEYS
    assert all(provider == PROVIDER for provider, _model in OLLAMA_CLOUD_KEYS)


def test_every_entry_has_explicit_valid_metadata() -> None:
    for key, specs in OLLAMA_CLOUD_SPECS.items():
        context_length = specs["context_length"]
        max_completion_tokens = specs["max_completion_tokens"]
        default_temperature = specs["default_temperature"]

        assert isinstance(context_length, int) and not isinstance(context_length, bool) and context_length > 0, key
        assert (
            isinstance(max_completion_tokens, int)
            and not isinstance(max_completion_tokens, bool)
            and max_completion_tokens > 0
        ), key
        assert isinstance(default_temperature, Real) and not isinstance(default_temperature, bool), key
        assert isinstance(specs.get("supports_vision"), bool), key


def test_thinking_efforts_use_the_reviewed_common_subset() -> None:
    for key, specs in OLLAMA_CLOUD_SPECS.items():
        effort = specs.get("thinking_effort")
        if effort is None:
            continue
        assert effort.mode == "openai_reasoning_effort", key
        assert effort.options == ("low", "medium", "high"), key
        assert effort.default == "medium", key


def test_model_ids_are_in_alphabetical_insertion_order() -> None:
    assert OLLAMA_CLOUD_MODELS == sorted(OLLAMA_CLOUD_MODELS)


def test_provider_default_is_catalogued_resolvable_and_listed() -> None:
    default = PROVIDER_DEFAULT_MODEL[ModelProvider.OLLAMA_CLOUD]
    assert (PROVIDER, default) in MODEL_SPECS

    option = get_ui_model(PROVIDER, default)
    assert option is not None
    assert option.model == default

    listed = [model for _label, model in ui_model_options(PROVIDER)]
    assert listed == OLLAMA_CLOUD_MODELS
    assert default in listed
