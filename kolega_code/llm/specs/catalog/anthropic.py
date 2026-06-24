from kolega_code.llm.specs.types import ThinkingEffortSpec

# Anthropic models
ANTHROPIC_SPECS = {
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
}
