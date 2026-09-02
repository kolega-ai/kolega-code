from kolega_code.llm.specs.types import ThinkingEffortSpec

# Z.AI (GLM Coding Plan) models — Anthropic-compatible endpoint (recommended default first)
ZAI_SPECS = {
    ("zai", "glm-5.3"): {
        "context_length": 1000000,
        "max_completion_tokens": 131072,
        "input_budget": "window_minus_output",
        "default_temperature": 0.6,
        "supports_vision": False,
        "thinking_effort": ThinkingEffortSpec(
            options=("high", "max"),
            default="max",
            mode="zai_effort",
        ),
    },
    ("zai", "glm-5.3-flash"): {
        "context_length": 1000000,
        "max_completion_tokens": 131072,
        "input_budget": "window_minus_output",
        "default_temperature": 0.6,
        # Native multimodal input (image/video/file); shares GLM-5.3's
        # forced-reasoning effort vocabulary (high/max).
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("high", "max"),
            default="max",
            mode="zai_effort",
        ),
    },
    ("zai", "glm-5.2"): {
        "context_length": 1000000,
        "max_completion_tokens": 131072,
        "input_budget": "window_minus_output",
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
        "input_budget": "window_minus_output",
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
