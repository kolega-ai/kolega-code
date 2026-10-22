"""MiniMax: first-party Anthropic-compatible endpoint (``api.minimax.io/anthropic``).

``GET /v1/models`` publishes ids and dates only, so the numbers here were probed
live against the endpoint rather than read from a catalog.

- M3 / M3.1-Flash: 1M *input* window with an independent 524288 output allowance.
- M2.x: the 262144 window and the output allowance share one budget, so the entry
  is the combined ceiling the agent can actually reach.
- Thinking: M3.1-Flash accepts adaptive effort only (disabling it 400s); M3 and
  M2.x emit no control, so they expose none.
"""

from kolega_code.llm.specs.types import ThinkingEffortSpec

MINIMAX_SPECS = {
    # Flagship Flash tier, GA 2026-10-19.
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
    # Same budget as Flash, but reasoning is not surfaced.
    ("minimax", "MiniMax-M3"): {
        "context_length": 1048576,
        "max_completion_tokens": 524288,
        "input_budget": "separate_output_limit",
        "default_temperature": 1.0,
        "supports_vision": True,
    },
    # M2 generation: text-only, output ceiling shared with the input window.
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
