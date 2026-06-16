from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class ThinkingEffortSpec:
    """Model-specific thinking/effort controls and provider serialization mode."""

    options: tuple[str, ...]
    default: str
    mode: str
    budgets: dict[str, int] = field(default_factory=dict)

# Dictionary mapping (provider, model_name) to model specifications
# Each entry contains context_length (maximum input tokens), max_completion_tokens, default_temperature,
# and optional model capability flags.
MODEL_SPECS: Dict[Tuple[str, str], Dict[str, Any]] = {
    # Anthropic models
    ("anthropic", "claude-opus-4-8"): {
        "context_length": 1000000,
        "max_completion_tokens": 128000,
        "default_temperature": 1.0,
        "supports_temperature": False,
        "thinking_effort": ThinkingEffortSpec(
            options=("low", "medium", "high", "xhigh", "max"),
            default="medium",
            mode="anthropic_adaptive_effort",
        ),
    },
    ("anthropic", "claude-opus-4-7"): {
        "context_length": 1000000,
        "max_completion_tokens": 128000,
        "default_temperature": 1.0,
        "supports_temperature": False,
        "thinking_effort": ThinkingEffortSpec(
            options=("low", "medium", "high", "xhigh", "max"),
            default="medium",
            mode="anthropic_adaptive_effort",
        ),
    },
    ("anthropic", "claude-opus-4-6"): {
        "context_length": 1000000,
        "max_completion_tokens": 128000,
        "default_temperature": 1.0,
        "thinking_effort": ThinkingEffortSpec(
            options=("low", "medium", "high", "max"),
            default="medium",
            mode="anthropic_adaptive_effort",
        ),
    },
    ("anthropic", "claude-sonnet-4-6"): {
        "context_length": 1000000,
        "max_completion_tokens": 64000,
        "default_temperature": 1.0,
        "thinking_effort": ThinkingEffortSpec(
            options=("low", "medium", "high", "max"),
            default="medium",
            mode="anthropic_adaptive_effort",
        ),
    },
    ("anthropic", "claude-sonnet-4-5-20250929"): {"context_length": 200000, "max_completion_tokens": 16384, "default_temperature": 1.0},
    ("anthropic", "claude-opus-4-5-20251101"): {"context_length": 200000, "max_completion_tokens": 16384, "default_temperature": 1.0},
    ("anthropic", "claude-haiku-4-5-20251001"): {"context_length": 200000, "max_completion_tokens": 16384, "default_temperature": 1.0},
    # Moonshot models (recommended default first)
    ("moonshot", "kimi-k2.7-code"): {
        "context_length": 262144,
        "max_completion_tokens": 32768,
        "default_temperature": 1.0,
        "thinking_effort": ThinkingEffortSpec(
            options=("auto",),
            default="auto",
            mode="moonshot_toggle",
        ),
    },
    ("moonshot", "kimi-k2.6"): {
        "context_length": 262144,
        "max_completion_tokens": 32768,
        "default_temperature": 1.0,
        "thinking_effort": ThinkingEffortSpec(
            options=("auto", "none"),
            default="auto",
            mode="moonshot_toggle",
        ),
    },
    ("moonshot", "kimi-k2.7-code-highspeed"): {
        "context_length": 262144,
        "max_completion_tokens": 32768,
        "default_temperature": 1.0,
        "thinking_effort": ThinkingEffortSpec(
            options=("auto",),
            default="auto",
            mode="moonshot_toggle",
        ),
    },
    # DeepSeek models
    ("deepseek", "deepseek-v4-pro"): {
        "context_length": 1000000,
        "max_completion_tokens": 384000,
        "default_temperature": 1.0,
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "high", "max"),
            default="high",
            mode="deepseek_effort",
        ),
    },
    ("deepseek", "deepseek-v4-flash"): {
        "context_length": 1000000,
        "max_completion_tokens": 384000,
        "default_temperature": 1.0,
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "high", "max"),
            default="high",
            mode="deepseek_effort",
        ),
    },
    # OpenAI models
    ("openai", "gpt-5.5"): {
        "context_length": 1050000,
        "max_completion_tokens": 128000,
        "default_temperature": 1.0,
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "medium", "high", "xhigh"),
            default="medium",
            mode="openai_reasoning_effort",
        ),
    },
    ("openai", "gpt-5.4-mini"): {
        "context_length": 400000,
        "max_completion_tokens": 128000,
        "default_temperature": 1.0,
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "medium", "high", "xhigh"),
            default="medium",
            mode="openai_reasoning_effort",
        ),
    },
    # Together.ai models
    ("together", "moonshotai/Kimi-K2.7-Code"): {"context_length": 262144, "max_completion_tokens": 32768, "default_temperature": 1.0},
    ("together", "zai-org/GLM-5.1"): {"context_length": 202752, "max_completion_tokens": 16384, "default_temperature": 0.6},
    # Google models
    ("google", "gemini-3.1-pro-preview"): {
        "context_length": 1048576,
        "max_completion_tokens": 65536,
        "default_temperature": 1.0,
        "thinking_effort": ThinkingEffortSpec(
            options=("low", "medium", "high"),
            default="high",
            mode="google_thinking_level",
        ),
    },
    ("google", "gemini-3.5-flash"): {
        "context_length": 1048576,
        "max_completion_tokens": 65536,
        "default_temperature": 1.0,
        "thinking_effort": ThinkingEffortSpec(
            options=("minimal", "low", "medium", "high"),
            default="medium",
            mode="google_thinking_level",
        ),
    },
    # X.ai models
    ("xai", "grok-4.3"): {
        "context_length": 1000000,
        "max_completion_tokens": 16384,
        "default_temperature": 1.0,
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "medium", "high"),
            default="low",
            mode="openai_reasoning_effort",
        ),
    },
    ("xai", "grok-build-0.1"): {"context_length": 256000, "max_completion_tokens": 16384, "default_temperature": 1.0},
    # Fireworks models
    ("fireworks", "accounts/fireworks/models/glm-5p1"): {"context_length": 202752, "max_completion_tokens": 16384, "default_temperature": 0.6},
    ("fireworks", "accounts/fireworks/models/kimi-k2p7-code"): {"context_length": 262144, "max_completion_tokens": 16384, "default_temperature": 1.0},
    # DashScope / Qwen models
    ("dashscope", "qwen3-coder-plus"): {"context_length": 1000000, "max_completion_tokens": 65536, "default_temperature": 0.7},
    ("dashscope", "qwen3-coder-flash"): {"context_length": 1000000, "max_completion_tokens": 65536, "default_temperature": 0.7},
}


def _provider_value(provider: str) -> str:
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

    return MODEL_SPECS.get(key)


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


def validate_thinking_effort(provider: str, model_name: str, effort: Optional[Any]) -> Optional[str]:
    """Validate and normalize a model-specific thinking effort value."""
    if effort is None:
        return None

    effort_value = effort.value if hasattr(effort, "value") else effort
    normalized = str(effort_value).strip().lower()
    if not normalized:
        return None

    options = thinking_effort_options(provider, model_name)
    provider_str = _provider_value(provider)
    if not options:
        raise ValueError(f"Model {model_name} from provider {provider_str} does not support thinking effort.")
    if normalized not in options:
        valid = ", ".join(options)
        raise ValueError(
            f"Unsupported thinking effort '{effort}' for {provider_str}/{model_name}. Valid values: {valid}"
        )
    return normalized


def normalize_thinking_effort(provider: str, model_name: str, effort: Optional[Any]) -> Optional[str]:
    """Validate an explicit effort or return the model default when no effort is set."""
    if effort is None or not str(effort).strip():
        return default_thinking_effort(provider, model_name)
    return validate_thinking_effort(provider, model_name, effort)


def build_thinking_request_params(provider: str, model_name: str, effort: Optional[Any]) -> Dict[str, Any]:
    """Convert a model-specific effort value into provider request parameters."""
    normalized = validate_thinking_effort(provider, model_name, effort)
    if normalized is None:
        return {}

    spec = get_thinking_effort_spec(provider, model_name)
    if spec is None:
        return {}

    if spec.mode == "anthropic_adaptive_effort":
        return {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": normalized},
        }

    if spec.mode == "deepseek_effort":
        if normalized == "none":
            return {"thinking": {"type": "disabled"}}
        return {
            "thinking": {"type": "enabled"},
            "output_config": {"effort": normalized},
        }

    if spec.mode == "moonshot_toggle":
        if normalized == "none":
            return {"thinking": {"type": "disabled"}}
        return {"thinking": {"type": "enabled"}}

    if spec.mode == "google_thinking_budget":
        return {"thinking_config": {"thinking_budget": spec.budgets[normalized]}}

    if spec.mode == "google_thinking_level":
        return {"thinking_config": {"thinking_level": normalized}}

    if spec.mode == "openai_reasoning_effort":
        return {"reasoning_effort": normalized}

    raise ValueError(f"Unknown thinking effort mode '{spec.mode}' for {_provider_value(provider)}/{model_name}.")
