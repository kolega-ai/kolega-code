from kolega_code.llm.specs.types import ThinkingEffortSpec

# Moonshot models (recommended default first)
#
# Budget convention: separate_output_limit — live probe 2026-08-11:
# moonshot/kimi-k2.7-code accepted 250K input with max_tokens=32768 (283K
# combined > 256K window), so the window is input-only and output is an
# independent allowance (not window_minus_output).
MOONSHOT_SPECS = {
    ("moonshot", "kimi-k3"): {
        "context_length": 1048576,
        "max_completion_tokens": 131072,
        "input_budget": "separate_output_limit",
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
        "input_budget": "separate_output_limit",
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
        "input_budget": "separate_output_limit",
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
        "input_budget": "separate_output_limit",
        "default_temperature": 1.0,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("auto",),
            default="auto",
            mode="moonshot_toggle",
        ),
    },
    # K3.1 Pro (2026-10 release): post-trained K3 with a 1M window. Thinking is
    # always on and exposes the low/high/max ladder (default max); temperature is
    # fixed at 1.0, and sampling params must not be sent.
    ("moonshot", "kimi-k3.1-pro"): {
        "context_length": 1048576,
        "max_completion_tokens": 131072,
        "input_budget": "separate_output_limit",
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("low", "high", "max"),
            default="max",
            mode="moonshot_reasoning_effort",
        ),
    },
    # Same model at ~2-3x output speed, with identical limits and controls.
    ("moonshot", "kimi-k3.1-pro-highspeed"): {
        "context_length": 1048576,
        "max_completion_tokens": 131072,
        "input_budget": "separate_output_limit",
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("low", "high", "max"),
            default="max",
            mode="moonshot_reasoning_effort",
        ),
    },
}
