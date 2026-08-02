from kolega_code.llm.specs.types import ThinkingEffortSpec

# DeepSeek models
DEEPSEEK_SPECS = {
    ("deepseek", "deepseek-v4-pro"): {
        "context_length": 1000000,
        "max_completion_tokens": 384000,
        "default_temperature": 1.0,
        "supports_vision": False,
        "preferred_edit_protocol": "claude_code",
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "high", "max"),
            default="high",
            mode="deepseek_effort",
        ),
    },
    # deepseek-v4-flash speaks the Responses API (see DeepSeekResponsesProvider), so its
    # reasoning effort must be emitted as the Responses `reasoning` block rather than the
    # Chat-Completions `reasoning_effort` field the shared Responses request builder ignores.
    # DeepSeek's Responses API is stateless (no reasoning.encrypted_content), so this mode
    # also correctly excludes flash from the flat reasoning_content replay path.
    ("deepseek", "deepseek-v4-flash"): {
        "context_length": 1000000,
        "max_completion_tokens": 384000,
        "default_temperature": 1.0,
        "supports_vision": False,
        "preferred_edit_protocol": "claude_code",
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "high", "max"),
            default="high",
            mode="openai_responses_reasoning",
        ),
    },
}
