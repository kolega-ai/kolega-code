"""Invariants of the generated OpenRouter catalog.

The catalog is machine-generated from OpenRouter's API, so these guard the
properties the rest of the codebase relies on rather than individual models
(which change with every refresh).
"""

from kolega_code.cli.provider_registry import PROVIDER_DEFAULT_MODEL, get_ui_model, ui_model_options
from kolega_code.config import ModelProvider
from kolega_code.llm.specs import MODEL_SPECS, is_featured_model
from kolega_code.llm.specs.openrouter_catalog import CANONICAL_EFFORT_ORDER, FEATURED_COUNT, PROVIDER

OPENROUTER_KEYS = [key for key in MODEL_SPECS if key[0] == PROVIDER]
OPENROUTER_MODELS = [model for _provider, model in OPENROUTER_KEYS]
FEATURED_MODELS = [model for model in OPENROUTER_MODELS if is_featured_model(PROVIDER, model)]


def test_catalog_is_populated_and_namespaced() -> None:
    assert len(OPENROUTER_MODELS) > 100, "the generated catalog looks truncated"
    # Every id is a `vendor/model` slug, optionally with a routing variant.
    assert all("/" in model for model in OPENROUTER_MODELS)


def test_every_entry_has_usable_budget_and_vision_metadata() -> None:
    for key in OPENROUTER_KEYS:
        specs = MODEL_SPECS[key]
        assert specs["context_length"] > 0, key
        # Consumed by agent context budgeting even though the wire request omits it.
        assert specs["max_completion_tokens"] > 0, key
        # tests/agent/llm/test_supports_vision.py enforces this catalog-wide; assert
        # it here too so a generator regression names OpenRouter directly.
        assert "supports_vision" in specs, key


def test_thinking_efforts_are_canonical_and_self_consistent() -> None:
    for provider, model in OPENROUTER_KEYS:
        spec = MODEL_SPECS[(provider, model)].get("thinking_effort")
        if spec is None:
            continue
        assert spec.mode == "openrouter_reasoning", model
        assert spec.options, model
        assert spec.default in spec.options, model
        # Options are ordered least- to most-reasoning, without duplicates.
        positions = [CANONICAL_EFFORT_ORDER.index(option) for option in spec.options]
        assert positions == sorted(set(positions)), (model, spec.options)


def test_featured_set_is_the_head_of_the_catalog() -> None:
    # Catalog order is OpenRouter's usage ranking, and featuring is a pure
    # prefix of it (plus, at most, the provider default pinned in).
    assert len(FEATURED_MODELS) in (FEATURED_COUNT, FEATURED_COUNT + 1)
    head = OPENROUTER_MODELS[:FEATURED_COUNT]
    assert FEATURED_MODELS[: len(head)] == head
    extra = FEATURED_MODELS[len(head) :]
    assert extra in ([], [PROVIDER_DEFAULT_MODEL[ModelProvider.OPENROUTER]])


def test_provider_default_is_usable_and_offered() -> None:
    default = PROVIDER_DEFAULT_MODEL[ModelProvider.OPENROUTER]
    assert (PROVIDER, default) in MODEL_SPECS
    # The Settings picker only lists featured models, so a default outside that
    # set would silently switch the user's model when the panel repopulates.
    assert is_featured_model(PROVIDER, default)
    option = get_ui_model(PROVIDER, default)
    assert option is not None
    # Sub-agents include a browser agent that requires image input.
    assert option.supports_vision


def test_picker_lists_featured_models_while_the_rest_stay_resolvable() -> None:
    listed = [model for _label, model in ui_model_options(PROVIDER)]
    assert listed == FEATURED_MODELS

    unlisted = [model for model in OPENROUTER_MODELS if model not in set(FEATURED_MODELS)]
    assert unlisted, "expected the catalog to be larger than the featured set"
    assert get_ui_model(PROVIDER, unlisted[0]) is not None


def test_edit_protocol_defaults_to_claude_code_except_for_openai_models() -> None:
    protocols = {model: MODEL_SPECS[(PROVIDER, model)].get("preferred_edit_protocol") for model in OPENROUTER_MODELS}
    assert all(protocol is not None for protocol in protocols.values())

    openai_models = {model for model in OPENROUTER_MODELS if model.startswith("openai/")}
    assert openai_models
    assert {protocols[model] for model in openai_models} == {"codex_apply_patch"}
    assert {protocols[model] for model in OPENROUTER_MODELS if model not in openai_models} == {"claude_code"}


def test_anthropic_models_opt_out_of_reasoning_replay() -> None:
    anthropic_models = [model for model in OPENROUTER_MODELS if model.startswith("anthropic/")]
    assert anthropic_models
    for model in anthropic_models:
        assert MODEL_SPECS[(PROVIDER, model)].get("drop_prior_reasoning") is True, model


def test_openrouter_entries_do_not_shadow_direct_provider_models() -> None:
    # Direct providers used to be keyed by bare model names while the gateway
    # always carried a vendor prefix, so the strings could never collide.
    # Tinker is the first direct provider with vendor-prefixed ids (e.g.
    # "Qwen/Qwen3-8B" is both a Tinker id and an OpenRouter id), so the string
    # sets now overlap — but the merged catalog is keyed by (provider, model)
    # tuples and the gateway is merged last, so the real invariant is that a
    # direct provider's spec is never replaced by the gateway's. Assert that
    # directly: shared ids keep their direct spec, not the gateway's markers.
    shared = {model for provider, model in MODEL_SPECS if provider != PROVIDER and model in OPENROUTER_MODELS}
    assert shared  # the overlap exists today (Qwen ids served by both)

    for provider, model in MODEL_SPECS:
        if provider == PROVIDER or model not in shared:
            continue
        spec = MODEL_SPECS[(provider, model)]
        assert "featured" not in spec
        assert "preferred_edit_protocol" not in spec
