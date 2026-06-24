from kolega_code.llm.specs.types import ThinkingEffortSpec

# Kimi Coding Plan — separate Anthropic-compatible endpoint, single stable model ID.
# thinking "auto" (enabled) -> K2.7 Code; "none" (disabled) -> K2.6.
KIMI_CODING_SPECS = {
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
