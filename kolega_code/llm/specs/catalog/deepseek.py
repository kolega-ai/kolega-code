from kolega_code.llm.specs.types import ThinkingEffortSpec

# DeepSeek models
DEEPSEEK_SPECS = {
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
}
