from kolega_code.llm.specs.types import ThinkingEffortSpec

# Kimi Coding Plan — separate Anthropic-compatible endpoint. K3 availability
# depends on membership tier; kimi-for-coding remains available on every tier.
KIMI_CODING_SPECS = {
    ("kimi_coding", "k3"): {
        "context_length": 262144,
        "max_completion_tokens": 131072,
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("max",),
            default="max",
            mode="kimi_coding_effort",
        ),
    },
    ("kimi_coding", "k3[1m]"): {
        "context_length": 1048576,
        "max_completion_tokens": 131072,
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("max",),
            default="max",
            mode="kimi_coding_effort",
        ),
    },
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
}
