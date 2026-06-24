from kolega_code.llm.specs.types import ThinkingEffortSpec

# X.ai models
XAI_SPECS = {
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
}
