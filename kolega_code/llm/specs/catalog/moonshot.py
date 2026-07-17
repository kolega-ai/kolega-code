from kolega_code.llm.specs.types import ThinkingEffortSpec

# Moonshot models (recommended default first)
MOONSHOT_SPECS = {
    ("moonshot", "kimi-k3"): {
        "context_length": 1048576,
        "max_completion_tokens": 131072,
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("max",),
            default="max",
            mode="moonshot_reasoning_effort",
        ),
    },
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
}
