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
        "preferred_edit_protocol": "claude_code",
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "medium", "high", "xhigh", "max"),
            default="high",
            mode="tinker_effort",
        ),
        "pricing": {
            "schema_version": 1,
            "pricing_id": "thinking_machines:thinkingmachines/Inkling:2026-11-01",
            "provider": "thinking_machines",
            "model": "thinkingmachines/Inkling",
            "status": "priced",
            "billing_mode": "serverless_inference",
            "currency": "USD",
            "unit": "per_million_tokens",
            "rates": {"input": "1.00", "cache_read": "0.17", "output": "4.05"},
            "checked_at": "2026-11-01",
            "sources": [
                {
                    "url": "https://tinker-docs.thinkingmachines.ai/tinker/serverless.json",
                    "model_id": "thinkingmachines/Inkling:peft:262144:sampling-nvfp4",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {},
            "schedules": [],
            "exclusions": ["Taxes, account credits/free allowances, and negotiated discounts."],
            "scope": "serverless_published_list_rates",
            "cache_policy": {"cache_read": "separate", "cache_write": "included"},
            "assumptions": [
                "Published serverless rates for thinkingmachines/Inkling:peft:262144:sampling-nvfp4 are "
                "applied to the supported thinkingmachines/Inkling alias as a catalogue estimate, not a "
                "provider-reported charge.",
                "Cache creation uses the published input rate; only cache reads have a separately published rate.",
            ],
        },
    },
    ("thinking_machines", "thinkingmachines/Inkling-Small"): {
        "context_length": 1000000,
        "max_completion_tokens": 32768,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": True,
        "preferred_edit_protocol": "claude_code",
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "medium", "high", "xhigh", "max"),
            default="high",
            mode="tinker_effort",
        ),
        "pricing": {
            "schema_version": 1,
            "pricing_id": "thinking_machines:thinkingmachines/Inkling-Small:2026-11-01",
            "provider": "thinking_machines",
            "model": "thinkingmachines/Inkling-Small",
            "status": "priced",
            "billing_mode": "serverless_inference",
            "currency": "USD",
            "unit": "per_million_tokens",
            "rates": {"input": "0.30", "cache_read": "0.06", "output": "1.20"},
            "checked_at": "2026-11-01",
            "sources": [
                {
                    "url": "https://tinker-docs.thinkingmachines.ai/tinker/serverless.json",
                    "model_id": "thinkingmachines/Inkling-Small:peft:262144:sampling-nvfp4",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {},
            "schedules": [],
            "exclusions": ["Taxes, account credits/free allowances, and negotiated discounts."],
            "scope": "serverless_published_list_rates",
            "cache_policy": {"cache_read": "separate", "cache_write": "included"},
            "assumptions": [
                "Published serverless rates for thinkingmachines/Inkling-Small:peft:262144:sampling-nvfp4 "
                "are applied to the supported thinkingmachines/Inkling-Small alias as a catalogue "
                "estimate, not a provider-reported charge.",
                "Cache creation uses the published input rate; only cache reads have a separately published rate.",
            ],
        },
    },
}
