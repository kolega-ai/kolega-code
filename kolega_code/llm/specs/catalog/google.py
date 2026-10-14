from kolega_code.llm.specs.types import ThinkingEffortSpec

# Google models
GOOGLE_SPECS = {
    ("google", "gemini-3.8-flash"): {
        "context_length": 1048576,
        "max_completion_tokens": 65536,
        "input_budget": "separate_output_limit",
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("low", "medium", "high"),
            default="medium",
            mode="google_thinking_level",
        ),
    },
    ("google", "gemini-3.7-flash"): {
        "context_length": 1048576,
        "max_completion_tokens": 65536,
        "input_budget": "separate_output_limit",
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("low", "medium", "high"),
            default="medium",
            mode="google_thinking_level",
        ),
    },
    ("google", "gemini-3.6-flash"): {
        "context_length": 1048576,
        "max_completion_tokens": 65536,
        "input_budget": "separate_output_limit",
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("minimal", "low", "medium", "high"),
            default="medium",
            mode="google_thinking_level",
        ),
    },
    ("google", "gemini-3.5-flash-lite"): {
        "context_length": 1048576,
        "max_completion_tokens": 65536,
        "input_budget": "separate_output_limit",
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("minimal", "low", "medium", "high"),
            default="minimal",
            mode="google_thinking_level",
        ),
    },
    ("google", "gemini-3.1-pro-preview"): {
        "context_length": 1048576,
        "max_completion_tokens": 65536,
        "input_budget": "separate_output_limit",
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
        "input_budget": "separate_output_limit",
        "default_temperature": 1.0,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("minimal", "low", "medium", "high"),
            default="medium",
            mode="google_thinking_level",
        ),
    },
    ("google", "gemini-4-argon"): {
        # Input, thinking, and output share one window (unlike Gemini 3).
        # Keep the synchronous limit; long decode is not used by this provider.
        "context_length": 1048576,
        "max_completion_tokens": 65536,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("low", "medium", "high"),
            default="medium",
            mode="google_thinking_level",
        ),
    },
}
