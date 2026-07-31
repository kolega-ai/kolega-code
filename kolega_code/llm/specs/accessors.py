from typing import Any, Dict, Optional

from .catalog import MODEL_SPECS
from .types import ThinkingEffortSpec


def _provider_value(provider: Any) -> str:
    return provider.value if hasattr(provider, "value") else provider


def get_model_specs(provider: str, model_name: str) -> Dict[str, Any]:
    """
    Get the specifications for a given model.

    Args:
        provider: The LLM provider (e.g., 'anthropic', 'openai') - can be string or enum
        model_name: The name of the model

    Returns:
        Dictionary containing context_length, max_completion_tokens, and default_temperature
    """
    # Handle both string and enum provider types
    provider_str = _provider_value(provider)
    key = (provider_str, model_name)

    if key not in MODEL_SPECS:
        raise ValueError(f"Model {model_name} from provider {provider_str} is not supported.")

    return MODEL_SPECS[key]


def supports_vision(provider: str, model_name: str) -> bool:
    """Whether a model accepts image input.

    Defaults to ``False`` for any entry that omits the flag, so a missing key
    is safely treated as non-vision (a clear guard message beats a mid-flight
    provider error).
    """
    return get_model_specs(provider, model_name).get("supports_vision", False)


def get_thinking_effort_spec(provider: str, model_name: str) -> Optional[ThinkingEffortSpec]:
    """Return the thinking effort spec for a model, if it supports a public control."""
    return get_model_specs(provider, model_name).get("thinking_effort")


def thinking_effort_options(provider: str, model_name: str) -> tuple[str, ...]:
    """Return supported effort values for a model."""
    spec = get_thinking_effort_spec(provider, model_name)
    return spec.options if spec else ()


def default_thinking_effort(provider: str, model_name: str) -> Optional[str]:
    """Return Kolega's default thinking effort for a model."""
    spec = get_thinking_effort_spec(provider, model_name)
    return spec.default if spec else None


def is_featured_model(provider: str, model_name: str) -> bool:
    """Whether a model is one of a gateway provider's listed (featured) models.

    Only gateway catalogs large enough to overwhelm a picker mark entries
    featured; for every other provider this is uniformly ``False`` and callers
    fall back to listing the whole provider.
    """
    specs = MODEL_SPECS.get((_provider_value(provider), model_name))
    return bool(specs and specs.get("featured", False))


def provider_has_featured_models(provider: str) -> bool:
    """Whether ``provider`` marks any model featured, i.e. wants a short list."""
    provider_str = _provider_value(provider)
    return any(key[0] == provider_str and specs.get("featured", False) for key, specs in MODEL_SPECS.items())


def prior_reasoning_is_replayable(provider: str, model_name: str) -> bool:
    """Whether prior reasoning may be sent back to this model.

    Defaults to ``True`` — including for unknown models — so only a catalog
    entry that explicitly opts out changes behavior. Anthropic models reached
    through a gateway opt out: their reasoning is carried by signed thinking
    blocks that no plain-text replay field can reconstruct.
    """
    specs = MODEL_SPECS.get((_provider_value(provider), model_name))
    if specs is None:
        return True
    return not specs.get("drop_prior_reasoning", False)


def preferred_edit_protocol(provider: str, model_name: str) -> Optional[str]:
    """Return the catalogue-preferred edit protocol, when one is configured.

    Unknown models and entries without a preference deliberately return ``None``
    so callers can retain the stable search/replace fallback.
    """

    provider_str = _provider_value(provider)
    specs = MODEL_SPECS.get((provider_str, model_name))
    if specs is None:
        return None
    value = specs.get("preferred_edit_protocol")
    return str(value) if value is not None else None
