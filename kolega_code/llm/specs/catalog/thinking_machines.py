from kolega_code.llm.specs.types import ThinkingEffortSpec

# Thinking Machines Lab models — reached through Tinker's Anthropic-compatible
# endpoint (base https://tinker.thinkingmachines.dev/services/tinker-prod/anthropic/api,
# the Anthropic SDK appends /v1/messages). Model IDs are case-sensitive.
#
# Effort mirrors Tinker's documented set for the Messages endpoint: "low",
# "medium", "high", "xhigh", "max" ("max" behaves the same as "xhigh"); "none"
# turns reasoning off via thinking={"type": "disabled"}. The vendor default
# when effort is omitted is "high".
THINKING_MACHINES_SPECS = {
    ("thinking_machines", "thinkingmachines/Inkling"): {
        "context_length": 1000000,
        "max_completion_tokens": 32768,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "medium", "high", "xhigh", "max"),
            default="high",
            mode="tinker_effort",
        ),
    },
    ("thinking_machines", "thinkingmachines/Inkling-Small"): {
        "context_length": 1000000,
        "max_completion_tokens": 32768,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "medium", "high", "xhigh", "max"),
            default="high",
            mode="tinker_effort",
        ),
    },
}
