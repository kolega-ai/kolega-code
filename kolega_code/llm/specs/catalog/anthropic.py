from kolega_code.llm.specs.types import ThinkingEffortSpec

# Anthropic models
ANTHROPIC_SPECS = {
    ("anthropic", "claude-fable-5"): {
        "context_length": 1000000,
        "max_completion_tokens": 128000,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("low", "medium", "high", "xhigh", "max"),
            default="medium",
            mode="anthropic_adaptive_effort",
        ),
    },
    ("anthropic", "claude-opus-5"): {
        "context_length": 1000000,
        "max_completion_tokens": 128000,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("low", "medium", "high", "xhigh", "max"),
            default="medium",
            mode="anthropic_adaptive_effort",
        ),
    },
    ("anthropic", "claude-opus-4-8"): {
        "context_length": 1000000,
        "max_completion_tokens": 128000,
        "input_budget": "window_minus_output",
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
        "input_budget": "window_minus_output",
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
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("low", "medium", "high", "max"),
            default="medium",
            mode="anthropic_adaptive_effort",
        ),
    },
    ("anthropic", "claude-sonnet-5"): {
        "context_length": 1000000,
        "max_completion_tokens": 128000,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("low", "medium", "high", "xhigh", "max"),
            default="medium",
            mode="anthropic_adaptive_effort",
        ),
    },
    ("anthropic", "claude-sonnet-4-6"): {
        "context_length": 1000000,
        "max_completion_tokens": 64000,
        "input_budget": "window_minus_output",
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
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": True,
    },
    ("anthropic", "claude-opus-4-5-20251101"): {
        "context_length": 200000,
        "max_completion_tokens": 16384,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": True,
    },
    ("anthropic", "claude-haiku-4-5-20251001"): {
        "context_length": 200000,
        "max_completion_tokens": 16384,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": True,
    },
    # Claude 5.1/5.5 generation (Sept 2026). All three keep the 1M context and
    # 128K output ceiling, ship adaptive thinking that cannot be turned off, and
    # support the full low/medium/high/xhigh/max effort ladder. `high` is
    # Anthropic's documented default except on Opus 5.5, which defaults to
    # `medium`. Appended after the older entries on purpose: catalog insertion
    # order drives the Settings provider-switch fallback.
    ("anthropic", "claude-fable-5-1"): {
        "context_length": 1000000,
        "max_completion_tokens": 128000,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("low", "medium", "high", "xhigh", "max"),
            default="high",
            mode="anthropic_adaptive_effort",
        ),
    },
    ("anthropic", "claude-opus-5-5"): {
        "context_length": 1000000,
        "max_completion_tokens": 128000,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("low", "medium", "high", "xhigh", "max"),
            default="medium",
            mode="anthropic_adaptive_effort",
        ),
    },
    ("anthropic", "claude-sonnet-5-5"): {
        "context_length": 1000000,
        "max_completion_tokens": 128000,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("low", "medium", "high", "xhigh", "max"),
            default="high",
            mode="anthropic_adaptive_effort",
        ),
    },
}
