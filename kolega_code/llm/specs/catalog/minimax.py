from kolega_code.llm.specs.types import ThinkingEffortSpec

# MiniMax (https://platform.minimax.io) — first-party Anthropic-compatible endpoint
# (https://api.minimax.io/anthropic, the vendor's recommended integration path).
#
# Everything below was probed live on 2026-10-20 against the real endpoint; the
# comments record the evidence because none of these numbers are published as a
# machine-readable catalog (GET /v1/models returns only ids and dates).
#
# Wire model IDs are case-sensitive in the docs but the server canonicalizes:
# "minimax-m3.1-flash" and "MiniMax-M3.1-Flash" both answer as MiniMax-M3.1-Flash.
# Catalog keys use the documented spelling so pickers and errors match the vendor.
#
# Budget conventions:
#   * M3 / M3.1-Flash: 1M *input* window with a separate 524288 output allowance.
#     Probe: 600206 input + max_tokens=524288 accepted (1.12M combined), while
#     1.1M input alone was rejected with "context window exceeds limit". So the
#     window is input-only and output is independent (separate_output_limit).
#   * M2.x: the 262144 input window and the output allowance are *not*
#     independent. Probe: 131236 input + 196608 output accepted (327844
#     combined); 140136 + 196608 rejected (336744 combined). The agent always
#     requests the full output cap, so the safe model is the combined ceiling:
#     context_length 327680 with max_completion_tokens 196608 leaves the agent
#     131072 input tokens and a combined 327680, which probes clean.
#
# Thinking:
#   * M3.1-Flash requires adaptive thinking ("thinking.type=disabled" and
#     output_config.effort="none" both 400). Effort rides output_config.effort,
#     low..max, and the documented default is max.
#   * M3 never emits thinking blocks (probed on a reasoning puzzle: content was
#     text-only at 342 output tokens), so it exposes no thinking control.
#   * M2.x always streams thinking blocks and ignores the enable/disable toggle
#     (both forms returned thinking blocks), so it exposes no control either.
MINIMAX_SPECS = {
    # Flagship Flash tier, GA 2026-10-19 (previewed as "Space Bunny" / M3.1-Flash-Preview).
    ("minimax", "MiniMax-M3.1-Flash"): {
        "context_length": 1048576,
        "max_completion_tokens": 524288,
        "input_budget": "separate_output_limit",
        "default_temperature": 1.0,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("low", "medium", "high", "xhigh", "max"),
            default="max",
            mode="anthropic_adaptive_effort",
        ),
    },
    # Previous frontier model. Same window and output ceiling as M3.1-Flash, but
    # reasoning is not surfaced as thinking blocks.
    ("minimax", "MiniMax-M3"): {
        "context_length": 1048576,
        "max_completion_tokens": 524288,
        "input_budget": "separate_output_limit",
        "default_temperature": 1.0,
        "supports_vision": True,
    },
    # M2 generation: text-only, 196608 output ceiling shared with the input window.
    ("minimax", "MiniMax-M2.7"): {
        "context_length": 327680,
        "max_completion_tokens": 196608,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": False,
    },
    ("minimax", "MiniMax-M2.7-highspeed"): {
        "context_length": 327680,
        "max_completion_tokens": 196608,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": False,
    },
    ("minimax", "MiniMax-M2.5"): {
        "context_length": 327680,
        "max_completion_tokens": 196608,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": False,
    },
    ("minimax", "MiniMax-M2.5-highspeed"): {
        "context_length": 327680,
        "max_completion_tokens": 196608,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": False,
    },
    ("minimax", "MiniMax-M2.1"): {
        "context_length": 327680,
        "max_completion_tokens": 196608,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": False,
    },
    ("minimax", "MiniMax-M2.1-highspeed"): {
        "context_length": 327680,
        "max_completion_tokens": 196608,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": False,
    },
    ("minimax", "MiniMax-M2"): {
        "context_length": 327680,
        "max_completion_tokens": 196608,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": False,
    },
}
