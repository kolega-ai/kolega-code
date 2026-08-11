from kolega_code.llm.specs.types import ThinkingEffortSpec

# Fireworks models (OpenAI-compatible endpoint). Fireworks reasoning models
# expose reasoning_content in responses and accept flat reasoning_effort
# values on chat completions. "none" disables reasoning where supported; Kimi
# K3 always reasons and exposes only its maximum effort.
FIREWORKS_SPECS = {
    ("fireworks", "accounts/fireworks/models/kimi-k3"): {
        "context_length": 1048576,
        "max_completion_tokens": 131072,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("max",),
            default="max",
            mode="openai_reasoning_effort",
        ),
    },
    ("fireworks", "accounts/fireworks/models/glm-5p2"): {
        "context_length": 1048576,
        "max_completion_tokens": 131072,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": False,
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "medium", "high", "max"),
            default="medium",
            mode="openai_reasoning_effort",
        ),
    },
    ("fireworks", "accounts/fireworks/models/glm-5p1"): {
        "context_length": 202800,
        "max_completion_tokens": 131072,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": False,
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "medium", "high", "max"),
            default="medium",
            mode="openai_reasoning_effort",
        ),
    },
    ("fireworks", "accounts/fireworks/models/kimi-k2p7-code"): {
        "context_length": 262144,
        "max_completion_tokens": 262144,
        "input_budget": "output_shares_window",
        "default_temperature": 1.0,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "medium", "high", "max"),
            default="medium",
            mode="openai_reasoning_effort",
        ),
    },
    # Measured 2026-08-03 (tests/agent/llm/test_deepseek_output_cap_live.py):
    # uncapped, the Fireworks deployment cuts at its 32768 DEFAULT max_tokens
    # (honest finish_reason="length"); an explicit 64000 cap is honored and
    # reported honestly. 65536 mirrors the first-party measured family ceiling
    # (see catalog/deepseek.py); the wire clamp sends 64000 regardless.
    ("fireworks", "accounts/fireworks/models/deepseek-v4-pro"): {
        "context_length": 1048576,
        "max_completion_tokens": 65536,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": False,
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "medium", "high", "max"),
            default="medium",
            mode="openai_reasoning_effort",
        ),
    },
    ("fireworks", "accounts/fireworks/models/deepseek-v4-flash"): {
        "context_length": 1048576,
        "max_completion_tokens": 65536,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": False,
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "medium", "high", "max"),
            default="medium",
            mode="openai_reasoning_effort",
        ),
    },
    ("fireworks", "accounts/fireworks/models/minimax-m3"): {
        "context_length": 512000,
        "max_completion_tokens": 512000,
        "input_budget": "output_shares_window",
        "default_temperature": 1.0,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "medium", "high", "max"),
            default="medium",
            mode="openai_reasoning_effort",
        ),
    },
    ("fireworks", "accounts/fireworks/models/qwen3p7-plus"): {
        "context_length": 262144,
        "max_completion_tokens": 65536,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": False,
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "medium", "high", "max"),
            default="medium",
            mode="openai_reasoning_effort",
        ),
    },
}
