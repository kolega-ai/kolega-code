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
        "pricing": {
            "schema_version": 1,
            "pricing_id": "kimi_coding:k3:2026-11-01",
            "provider": "kimi_coding",
            "model": "k3",
            "status": "subscription",
            "billing_mode": "membership_with_optional_extra_usage",
            "currency": None,
            "unit": "per_million_tokens",
            "rates": {},
            "checked_at": "2026-11-01",
            "sources": [
                {
                    "url": "https://www.kimi.com/code/docs/kimi-code/membership.html",
                    "model_id": "k3",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {},
            "schedules": [],
            "exclusions": ["Taxes, account credits/free allowances, and negotiated discounts."],
            "reason": "k3: K3 membership model; 1M access is plan gated. Included usage has no published exact "
            "per-token conversion; Extra Usage is account-priced in CNY and only described as approximately "
            "API-priced.",
            "denomination": "kimi_code_membership_quota",
        },
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
        "pricing": {
            "schema_version": 1,
            "pricing_id": "kimi_coding:k3-256k:2026-11-01",
            "provider": "kimi_coding",
            "model": "k3-256k",
            "status": "subscription",
            "billing_mode": "membership_with_optional_extra_usage",
            "currency": None,
            "unit": "per_million_tokens",
            "rates": {},
            "checked_at": "2026-11-01",
            "sources": [
                {
                    "url": "https://www.kimi.com/code/docs/kimi-code/membership.html",
                    "model_id": "k3-256k",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {},
            "schedules": [],
            "exclusions": ["Taxes, account credits/free allowances, and negotiated discounts."],
            "reason": "k3-256k: K3 256K membership route; different documented quota consumption from k3. Included "
            "usage has no published exact per-token conversion; Extra Usage is account-priced in CNY and "
            "only described as approximately API-priced.",
            "denomination": "kimi_code_membership_quota",
        },
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
        "pricing": {
            "schema_version": 1,
            "pricing_id": "kimi_coding:kimi-for-coding:2026-11-01",
            "provider": "kimi_coding",
            "model": "kimi-for-coding",
            "status": "subscription",
            "billing_mode": "membership_with_optional_extra_usage",
            "currency": None,
            "unit": "per_million_tokens",
            "rates": {},
            "checked_at": "2026-11-01",
            "sources": [
                {
                    "url": "https://www.kimi.com/code/docs/kimi-code/membership.html",
                    "model_id": "kimi-for-coding",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {},
            "schedules": [],
            "exclusions": ["Taxes, account credits/free allowances, and negotiated discounts."],
            "reason": "kimi-for-coding: Current K2.8 Preview membership alias; do not inherit a Moonshot API rate. "
            "Included usage has no published exact per-token conversion; Extra Usage is account-priced in "
            "CNY and only described as approximately API-priced.",
            "denomination": "kimi_code_membership_quota",
        },
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
        "pricing": {
            "schema_version": 1,
            "pricing_id": "kimi_coding:kimi-for-coding-highspeed:2026-11-01",
            "provider": "kimi_coding",
            "model": "kimi-for-coding-highspeed",
            "status": "subscription",
            "billing_mode": "membership_with_optional_extra_usage",
            "currency": None,
            "unit": "per_million_tokens",
            "rates": {},
            "checked_at": "2026-11-01",
            "sources": [
                {
                    "url": "https://www.kimi.com/code/docs/kimi-code/membership.html",
                    "model_id": "kimi-for-coding-highspeed",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {},
            "schedules": [],
            "exclusions": ["Taxes, account credits/free allowances, and negotiated discounts."],
            "reason": "kimi-for-coding-highspeed: K2.7 Code Highspeed membership route with Pro-tier gating. Included "
            "usage has no published exact per-token conversion; Extra Usage is account-priced in CNY and "
            "only described as approximately API-priced.",
            "denomination": "kimi_code_membership_quota",
        },
    },
    # K3.1 Pro (2026-10 release). Available on Plus and above, but the 1M window
    # is gated to Pro and above, so the catalog keeps the 256K floor that holds
    # on every K3.1-eligible tier. Thinking cannot be disabled; the effort ladder
    # is low/high/max and sampling parameters are fixed at 1.0.
    ("kimi_coding", "k3.1-pro"): {
        "context_length": 262144,
        "max_completion_tokens": 131072,
        "input_budget": "separate_output_limit",
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("low", "high", "max"),
            default="max",
            mode="kimi_coding_effort",
        ),
        "pricing": {
            "schema_version": 1,
            "pricing_id": "kimi_coding:k3.1-pro:2026-11-01",
            "provider": "kimi_coding",
            "model": "k3.1-pro",
            "status": "subscription",
            "billing_mode": "membership_with_optional_extra_usage",
            "currency": None,
            "unit": "per_million_tokens",
            "rates": {},
            "checked_at": "2026-11-01",
            "sources": [
                {
                    "url": "https://www.kimi.com/code/docs/kimi-code/membership.html",
                    "model_id": "k3.1-pro",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {},
            "schedules": [],
            "exclusions": ["Taxes, account credits/free allowances, and negotiated discounts."],
            "reason": "k3.1-pro: K3.1 Pro membership route; 1M context has separate plan gating. Included usage has "
            "no published exact per-token conversion; Extra Usage is account-priced in CNY and only "
            "described as approximately API-priced.",
            "denomination": "kimi_code_membership_quota",
        },
    },
    # Highspeed is Pro-and-above only, so every plan that can reach it has the
    # full 1M window. Same controls, ~2-3x the speed at ~2x quota usage.
    ("kimi_coding", "k3.1-pro-highspeed"): {
        "context_length": 1048576,
        "max_completion_tokens": 131072,
        "input_budget": "separate_output_limit",
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("low", "high", "max"),
            default="max",
            mode="kimi_coding_effort",
        ),
        "pricing": {
            "schema_version": 1,
            "pricing_id": "kimi_coding:k3.1-pro-highspeed:2026-11-01",
            "provider": "kimi_coding",
            "model": "k3.1-pro-highspeed",
            "status": "subscription",
            "billing_mode": "membership_with_optional_extra_usage",
            "currency": None,
            "unit": "per_million_tokens",
            "rates": {},
            "checked_at": "2026-11-01",
            "sources": [
                {
                    "url": "https://www.kimi.com/code/docs/kimi-code/membership.html",
                    "model_id": "k3.1-pro-highspeed",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {},
            "schedules": [],
            "exclusions": ["Taxes, account credits/free allowances, and negotiated discounts."],
            "reason": "k3.1-pro-highspeed: K3.1 Pro Highspeed membership route; approximately 2x quota is not an "
            "exact token rate. Included usage has no published exact per-token conversion; Extra Usage is "
            "account-priced in CNY and only described as approximately API-priced.",
            "denomination": "kimi_code_membership_quota",
        },
    },
}
