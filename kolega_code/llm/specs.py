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
#
# ``supports_vision`` indicates whether a model accepts image input. When
# uncertain, default to False (the safe failure mode is a clear "model doesn't
# support images" message rather than a mid-conversation API error). The flag
# is the single tunable knob and is consumed by ``supports_vision()`` below,
# ``BaseAgent._unsupported_attachment_message`` (replacing the old hardcoded
# DeepSeek guard), and the ``read_image`` tool gate.
MODEL_SPECS: Dict[Tuple[str, str], Dict[str, Any]] = {
    # Anthropic models
    ("anthropic", "claude-opus-4-8"): {
        "context_length": 1000000,
        "max_completion_tokens": 128000,
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
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
        "supports_vision": True,
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
        "supports_vision": True,
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
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("low", "medium", "high", "max"),
            default="medium",
            mode="anthropic_adaptive_effort",
        ),
    },
    ("anthropic", "claude-sonnet-4-5-20250929"): {
        "context_length": 200000,
        "max_completion_tokens": 16384,
        "default_temperature": 1.0,
        "supports_vision": True,
    },
    ("anthropic", "claude-opus-4-5-20251101"): {
        "context_length": 200000,
        "max_completion_tokens": 16384,
        "default_temperature": 1.0,
        "supports_vision": True,
    },
    ("anthropic", "claude-haiku-4-5-20251001"): {
        "context_length": 200000,
        "max_completion_tokens": 16384,
        "default_temperature": 1.0,
        "supports_vision": True,
    },
    # Moonshot models (recommended default first)
    ("moonshot", "kimi-k2.7-code"): {
        "context_length": 262144,
        "max_completion_tokens": 32768,
        "default_temperature": 1.0,
        "supports_vision": True,
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
        "supports_vision": True,
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
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("auto",),
            default="auto",
            mode="moonshot_toggle",
        ),
    },
    # Kimi Coding Plan — separate Anthropic-compatible endpoint, single stable model ID.
    # thinking "auto" (enabled) -> K2.7 Code; "none" (disabled) -> K2.6.
    ("kimi_coding", "kimi-for-coding"): {
        "context_length": 262144,
        "max_completion_tokens": 32768,
        "default_temperature": 1.0,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("auto", "none"),
            default="auto",
            mode="moonshot_toggle",
        ),
    },
    # DeepSeek models
    ("deepseek", "deepseek-v4-pro"): {
        "context_length": 1000000,
        "max_completion_tokens": 384000,
        "default_temperature": 1.0,
        "supports_vision": False,
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
        "supports_vision": False,
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "high", "max"),
            default="high",
            mode="deepseek_effort",
        ),
    },
    # OpenAI models
    # The api-key `openai` provider now uses the Responses API (gpt-5.x reject
    # function tools + reasoning_effort on Chat Completions). These mirror the
    # ("openai_chatgpt", …) specs below: nested reasoning + no temperature.
    ("openai", "gpt-5.5"): {
        "context_length": 1050000,
        "max_completion_tokens": 128000,
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("minimal", "low", "medium", "high", "xhigh"),
            default="medium",
            mode="openai_responses_reasoning",
        ),
    },
    ("openai", "gpt-5.4"): {
        "context_length": 400000,
        "max_completion_tokens": 128000,
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("minimal", "low", "medium", "high"),
            default="medium",
            mode="openai_responses_reasoning",
        ),
    },
    ("openai", "gpt-5.4-mini"): {
        "context_length": 400000,
        "max_completion_tokens": 128000,
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("minimal", "low", "medium", "high"),
            default="medium",
            mode="openai_responses_reasoning",
        ),
    },
    # Note: gpt-5.3-codex-spark is intentionally NOT on the API-key `openai`
    # provider — it's a Codex model that 404s on Chat Completions and is only
    # reachable through the ChatGPT-subscription backend (openai_chatgpt) below.
    # OpenAI via ChatGPT subscription (Responses API, OAuth). Model slugs mirror
    # the Codex model picker; context/output limits mirror the API gpt-5.x specs
    # and are server-enforced (we never send max_output_tokens).
    ("openai_chatgpt", "gpt-5.5"): {
        "context_length": 1050000,
        "max_completion_tokens": 128000,
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("minimal", "low", "medium", "high", "xhigh"),
            default="medium",
            mode="openai_responses_reasoning",
        ),
    },
    ("openai_chatgpt", "gpt-5.4"): {
        "context_length": 400000,
        "max_completion_tokens": 128000,
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("minimal", "low", "medium", "high"),
            default="medium",
            mode="openai_responses_reasoning",
        ),
    },
    ("openai_chatgpt", "gpt-5.4-mini"): {
        "context_length": 400000,
        "max_completion_tokens": 128000,
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("minimal", "low", "medium", "high"),
            default="medium",
            mode="openai_responses_reasoning",
        ),
    },
    ("openai_chatgpt", "gpt-5.3-codex-spark"): {
        "context_length": 256000,
        "max_completion_tokens": 128000,
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("minimal", "low", "medium"),
            default="low",
            mode="openai_responses_reasoning",
        ),
    },
    # Together.ai models
    ("together", "moonshotai/Kimi-K2.7-Code"): {
        "context_length": 262144,
        "max_completion_tokens": 32768,
        "default_temperature": 1.0,
        "supports_vision": True,
    },
    ("together", "zai-org/GLM-5.1"): {
        "context_length": 202752,
        "max_completion_tokens": 16384,
        "default_temperature": 1.0,
        "supports_vision": False,
    },
    # Google models
    ("google", "gemini-3.1-pro-preview"): {
        "context_length": 1048576,
        "max_completion_tokens": 65536,
        "default_temperature": 1.0,
        "supports_vision": True,
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
        "supports_vision": True,
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
        "default_temperature": 0.6,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "medium", "high"),
            default="low",
            mode="openai_reasoning_effort",
        ),
    },
    ("xai", "grok-build-0.1"): {
        "context_length": 256000,
        "max_completion_tokens": 16384,
        "default_temperature": 0.6,
        "supports_vision": False,
    },
    # Fireworks models (OpenAI-compatible endpoint). Fireworks reasoning models
    # expose reasoning_content in responses and accept flat reasoning_effort
    # values on chat completions. "none" disables reasoning.
    ("fireworks", "accounts/fireworks/models/glm-5p2"): {
        "context_length": 1048576,
        "max_completion_tokens": 131072,
        "default_temperature": 1.0,
        "supports_vision": False,
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "medium", "high", "max"),
            default="medium",
            mode="openai_reasoning_effort",
        ),
    },
    ("fireworks", "accounts/fireworks/models/glm-5p1"): {
        "context_length": 202800,
        "max_completion_tokens": 131072,
        "default_temperature": 1.0,
        "supports_vision": False,
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "medium", "high", "max"),
            default="medium",
            mode="openai_reasoning_effort",
        ),
    },
    ("fireworks", "accounts/fireworks/models/kimi-k2p7-code"): {
        "context_length": 262144,
        "max_completion_tokens": 262144,
        "default_temperature": 1.0,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "medium", "high", "max"),
            default="medium",
            mode="openai_reasoning_effort",
        ),
    },
    ("fireworks", "accounts/fireworks/models/deepseek-v4-pro"): {
        "context_length": 1048576,
        "max_completion_tokens": 384000,
        "default_temperature": 1.0,
        "supports_vision": False,
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "medium", "high", "max"),
            default="medium",
            mode="openai_reasoning_effort",
        ),
    },
    ("fireworks", "accounts/fireworks/models/deepseek-v4-flash"): {
        "context_length": 1048576,
        "max_completion_tokens": 384000,
        "default_temperature": 1.0,
        "supports_vision": False,
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "medium", "high", "max"),
            default="medium",
            mode="openai_reasoning_effort",
        ),
    },
    ("fireworks", "accounts/fireworks/models/minimax-m3"): {
        "context_length": 512000,
        "max_completion_tokens": 512000,
        "default_temperature": 1.0,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "medium", "high", "max"),
            default="medium",
            mode="openai_reasoning_effort",
        ),
    },
    ("fireworks", "accounts/fireworks/models/qwen3p7-plus"): {
        "context_length": 262144,
        "max_completion_tokens": 65536,
        "default_temperature": 1.0,
        "supports_vision": False,
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "medium", "high", "max"),
            default="medium",
            mode="openai_reasoning_effort",
        ),
    },
    # DashScope / Qwen models
    ("dashscope", "qwen3-coder-plus"): {
        "context_length": 1000000,
        "max_completion_tokens": 65536,
        "default_temperature": 0.7,
        "supports_vision": False,
    },
    ("dashscope", "qwen3-coder-flash"): {
        "context_length": 1000000,
        "max_completion_tokens": 65536,
        "default_temperature": 0.7,
        "supports_vision": False,
    },
    # Z.AI (GLM Coding Plan) models — Anthropic-compatible endpoint (recommended default first)
    ("zai", "glm-5.2"): {
        "context_length": 1000000,
        "max_completion_tokens": 131072,
        "default_temperature": 0.6,
        "supports_vision": False,
        "thinking_effort": ThinkingEffortSpec(
            options=("high", "max"),
            default="max",
            mode="zai_effort",
        ),
    },
    ("zai", "glm-5.1"): {
        "context_length": 202752,
        "max_completion_tokens": 16384,
        "default_temperature": 0.6,
        "supports_vision": False,
        # GLM-5.1 predates GLM-5.2's named effort levels, so it's a plain
        # enable/disable toggle (no output_config.effort).
        "thinking_effort": ThinkingEffortSpec(
            options=("auto", "none"),
            default="auto",
            mode="zai_effort",
        ),
    },
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

    if spec.mode == "zai_effort":
        # Z.AI GLM toggles thinking via {"thinking": {"type": "enabled"|"disabled"}}.
        # GLM-5.2 adds two named effort levels (High/Max) carried in output_config.effort.
        if normalized == "none":
            return {"thinking": {"type": "disabled"}}
        params: Dict[str, Any] = {"thinking": {"type": "enabled"}}
        if normalized in ("high", "max"):
            params["output_config"] = {"effort": normalized}
        return params

    if spec.mode == "moonshot_toggle":
        if normalized == "none":
            return {"thinking": {"type": "disabled"}}
        return {"thinking": {"type": "enabled"}}

    if spec.mode == "google_thinking_budget":
        return {"thinking_config": {"thinking_budget": spec.budgets[normalized]}}

    if spec.mode == "google_thinking_level":
        return {"thinking_config": {"thinking_level": normalized}}

    if spec.mode == "openai_reasoning_effort":
        # OpenAI-compatible reasoning APIs use a flat reasoning_effort field.
        # Fireworks additionally accepts "none" to disable reasoning.
        return {"reasoning_effort": normalized}

    if spec.mode == "openai_responses_reasoning":
        # The Responses API nests reasoning effort under a "reasoning" object,
        # unlike Chat Completions' flat "reasoning_effort". We request
        # summary="auto" so the backend streams a human-readable reasoning summary
        # for the TUI thinking display. (Codex defaults these models to summary
        # "none" and shows no reasoning text; kolega surfaces it, so we keep it
        # on.) This is independent of reasoning continuity, which is carried by
        # reasoning.encrypted_content (see the ChatGPT provider), not the summary.
        return {"reasoning": {"effort": normalized, "summary": "auto"}}

    raise ValueError(f"Unknown thinking effort mode '{spec.mode}' for {_provider_value(provider)}/{model_name}.")
