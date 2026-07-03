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
