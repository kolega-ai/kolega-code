"""Settings "Other…" model picker affordance (TUI-only module)."""

from kolega_code.cli.provider_registry import ui_model_options
from kolega_code.cli.tui import custom_model
from kolega_code.llm.specs import MODEL_SPECS, is_featured_model


def test_gateway_picker_appends_other_last() -> None:
    options = custom_model.settings_model_options("openrouter")

    assert options[-1] == (custom_model.CUSTOM_MODEL_LABEL, custom_model.CUSTOM_MODEL_SENTINEL)
    assert [model for _label, model in options[:-1]] == [model for _label, model in ui_model_options("openrouter")]


def test_gateway_vision_only_picker_appends_other() -> None:
    options = custom_model.settings_model_options("openrouter", vision_only=True)

    assert options[-1] == (custom_model.CUSTOM_MODEL_LABEL, custom_model.CUSTOM_MODEL_SENTINEL)
    assert [model for _label, model in options[:-1]] == [
        model for _label, model in ui_model_options("openrouter", vision_only=True)
    ]


def test_non_gateway_pickers_are_unchanged() -> None:
    for provider in ("anthropic", "moonshot", "deepseek"):
        assert custom_model.settings_model_options(provider) == ui_model_options(provider)


def test_sentinel_never_collides_with_a_catalogued_id() -> None:
    catalogued = {model for provider, model in MODEL_SPECS}
    assert custom_model.CUSTOM_MODEL_SENTINEL not in catalogued


def test_resolve_custom_model_accepts_catalogued_ids() -> None:
    featured = next(model for _label, model in ui_model_options("openrouter"))
    non_featured = next(
        model
        for provider, model in MODEL_SPECS
        if provider == "openrouter" and not is_featured_model("openrouter", model)
    )

    assert custom_model.resolve_custom_model("openrouter", featured) == featured
    assert custom_model.resolve_custom_model("openrouter", non_featured) == non_featured
    assert custom_model.resolve_custom_model("openrouter", f"  {non_featured}  ") == non_featured


def test_resolve_custom_model_rejects_unknown_and_empty() -> None:
    assert custom_model.resolve_custom_model("openrouter", "vendor/not-real") is None
    assert custom_model.resolve_custom_model("openrouter", "  vendor/not-real  ") is None
    assert custom_model.resolve_custom_model("openrouter", "") is None
    assert custom_model.resolve_custom_model("openrouter", "   ") is None
