from kolega_code.llm.specs.types import ThinkingEffortSpec

# Kimi Coding Plan — separate Anthropic-compatible endpoint. K3 availability
# depends on membership tier; kimi-for-coding remains available on every tier.
#
# Wire model IDs (verified against https://www.kimi.com/code/docs/en/kimi-code/models
# and live API responses): k3, k3-256k, kimi-for-coding, kimi-for-coding-highspeed.
# There is NO separate 1M id: `k3` itself carries up to 1M context on Allegretto
# and higher tiers (256K on Moderato), so the context budget here is the 256K
# floor that holds on every K3-eligible tier.
#
# Budget convention: separate_output_limit — live probe 2026-08-11: kimi_coding/k3
# accepted 200K input with max_tokens=131072 (331K combined > 256K window), so the
# window is input-only and output is an independent allowance (not window_minus_output).
KIMI_CODING_SPECS = {
    ("kimi_coding", "k3"): {
        # Tier-dependent window (256K on Moderato, up to 1M on Allegretto+).
        # Catalogued at the safe 256K floor: the key does not reveal the tier,
        # and over-budgeting would make the API reject input on Moderato plans.
        "context_length": 262144,
        "max_completion_tokens": 131072,
        "input_budget": "separate_output_limit",
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("max",),
            default="max",
            mode="kimi_coding_effort",
        ),
    },
    ("kimi_coding", "k3-256k"): {
        # Fixed 256K K3 route (~half the quota burn of k3 on higher tiers;
        # image input only, no video). Same reasoning controls as k3.
        "context_length": 262144,
        "max_completion_tokens": 131072,
        "input_budget": "separate_output_limit",
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("max",),
            default="max",
            mode="kimi_coding_effort",
        ),
    },
    ("kimi_coding", "kimi-for-coding"): {
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
    ("kimi_coding", "kimi-for-coding-highspeed"): {
        # K2.7 Code HighSpeed route (~5-6x faster output, ~3x quota usage,
        # Allegretto and higher tiers). Same K2.7 Code thinking controls.
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
}
