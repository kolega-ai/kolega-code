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
        "pricing": {
            "schema_version": 1,
            "pricing_id": "minimax:MiniMax-M3.1-Flash:2026-11-01",
            "provider": "minimax",
            "model": "MiniMax-M3.1-Flash",
            "status": "priced",
            "billing_mode": "pay_as_you_go",
            "currency": "USD",
            "unit": "per_million_tokens",
            "rates": {"input": "0.15", "output": "0.60", "cache_read": "0.015"},
            "checked_at": "2026-11-01",
            "sources": [
                {
                    "url": "https://platform.minimax.io/docs/guides/pricing-paygo",
                    "model_id": "MiniMax-M3.1-Flash",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {},
            "schedules": [],
            "exclusions": ["Taxes, account credits/free allowances, and negotiated discounts."],
        },
    },
    # Same budget as Flash, but reasoning is not surfaced.
    ("minimax", "MiniMax-M3"): {
        "context_length": 1048576,
        "max_completion_tokens": 524288,
        "input_budget": "separate_output_limit",
        "default_temperature": 1.0,
        "supports_vision": True,
        "pricing": {
            "schema_version": 1,
            "pricing_id": "minimax:MiniMax-M3:2026-11-01",
            "provider": "minimax",
            "model": "MiniMax-M3",
            "status": "priced",
            "billing_mode": "pay_as_you_go",
            "currency": "USD",
            "unit": "per_million_tokens",
            "rates": {"input": "0.30", "output": "1.20", "cache_read": "0.06"},
            "checked_at": "2026-11-01",
            "sources": [
                {
                    "url": "https://platform.minimax.io/docs/guides/pricing-paygo",
                    "model_id": "MiniMax-M3",
                    "checked_at": "2026-11-01",
                },
                {
                    "url": "https://platform.minimax.io/docs/api-reference/text-anthropic-api",
                    "model_id": "MiniMax-M3",
                    "checked_at": "2026-11-01",
                },
            ],
            "conditions": {},
            "schedules": [
                {
                    "id": "standard_up_to_512k",
                    "when": {"service_tier": "standard", "max_input_tokens": 512000},
                    "rates": {"input": "0.30", "output": "1.20", "cache_read": "0.06"},
                },
                {
                    "id": "standard_above_512k",
                    "when": {"service_tier": "standard", "min_input_tokens": 512001},
                    "rates": {"input": "0.60", "output": "2.40", "cache_read": "0.12"},
                },
                {
                    "id": "priority_up_to_512k",
                    "when": {"service_tier": "priority", "max_input_tokens": 512000},
                    "rates": {"input": "0.45", "output": "1.80", "cache_read": "0.09"},
                },
                {
                    "id": "priority_above_512k",
                    "when": {"service_tier": "priority", "min_input_tokens": 512001},
                    "rates": {"input": "0.90", "output": "3.60", "cache_read": "0.18"},
                },
            ],
            "exclusions": ["Taxes, account credits/free allowances, and negotiated discounts."],
            "promotion": {"description": "Permanent 50% reduction, as stated in the rate card."},
            "requires_schedule": True,
            "default_context": {"service_tier": "standard"},
            "assumptions": ["512k billing boundary is interpreted as 512,000 input tokens; default tier is Standard."],
        },
    },
    # M2 generation: text-only, output ceiling shared with the input window.
    ("minimax", "MiniMax-M2.7"): {
        "context_length": 327680,
        "max_completion_tokens": 196608,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": False,
        "pricing": {
            "schema_version": 1,
            "pricing_id": "minimax:MiniMax-M2.7:2026-11-01",
            "provider": "minimax",
            "model": "MiniMax-M2.7",
            "status": "unverified",
            "billing_mode": "pay_as_you_go",
            "currency": "USD",
            "unit": "per_million_tokens",
            "rates": {},
            "checked_at": "2026-11-01",
            "sources": [
                {
                    "url": "https://platform.minimax.io/docs/guides/pricing-paygo",
                    "model_id": "MiniMax-M2.7",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {},
            "schedules": [],
            "exclusions": ["Taxes, account credits/free allowances, and negotiated discounts."],
            "reason": "MiniMax-M2.7: current official LLM rate card and Anthropic-compatible supported-model table "
            "list only M3.1-Flash and M3. No exact current legacy-route/cache/tier price verified; no "
            "M3-family substitution.",
        },
    },
    ("minimax", "MiniMax-M2.7-highspeed"): {
        "context_length": 327680,
        "max_completion_tokens": 196608,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": False,
        "pricing": {
            "schema_version": 1,
            "pricing_id": "minimax:MiniMax-M2.7-highspeed:2026-11-01",
            "provider": "minimax",
            "model": "MiniMax-M2.7-highspeed",
            "status": "unverified",
            "billing_mode": "pay_as_you_go",
            "currency": "USD",
            "unit": "per_million_tokens",
            "rates": {},
            "checked_at": "2026-11-01",
            "sources": [
                {
                    "url": "https://platform.minimax.io/docs/guides/pricing-paygo",
                    "model_id": "MiniMax-M2.7-highspeed",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {},
            "schedules": [],
            "exclusions": ["Taxes, account credits/free allowances, and negotiated discounts."],
            "reason": "MiniMax-M2.7-highspeed: current official LLM rate card and Anthropic-compatible "
            "supported-model table list only M3.1-Flash and M3. No exact current legacy-route/cache/tier "
            "price verified; no M3-family substitution.",
        },
    },
    ("minimax", "MiniMax-M2.5"): {
        "context_length": 327680,
        "max_completion_tokens": 196608,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": False,
        "pricing": {
            "schema_version": 1,
            "pricing_id": "minimax:MiniMax-M2.5:2026-11-01",
            "provider": "minimax",
            "model": "MiniMax-M2.5",
            "status": "unverified",
            "billing_mode": "pay_as_you_go",
            "currency": "USD",
            "unit": "per_million_tokens",
            "rates": {},
            "checked_at": "2026-11-01",
            "sources": [
                {
                    "url": "https://platform.minimax.io/docs/guides/pricing-paygo",
                    "model_id": "MiniMax-M2.5",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {},
            "schedules": [],
            "exclusions": ["Taxes, account credits/free allowances, and negotiated discounts."],
            "reason": "MiniMax-M2.5: current official LLM rate card and Anthropic-compatible supported-model table "
            "list only M3.1-Flash and M3. No exact current legacy-route/cache/tier price verified; no "
            "M3-family substitution.",
        },
    },
    ("minimax", "MiniMax-M2.5-highspeed"): {
        "context_length": 327680,
        "max_completion_tokens": 196608,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": False,
        "pricing": {
            "schema_version": 1,
            "pricing_id": "minimax:MiniMax-M2.5-highspeed:2026-11-01",
            "provider": "minimax",
            "model": "MiniMax-M2.5-highspeed",
            "status": "unverified",
            "billing_mode": "pay_as_you_go",
            "currency": "USD",
            "unit": "per_million_tokens",
            "rates": {},
            "checked_at": "2026-11-01",
            "sources": [
                {
                    "url": "https://platform.minimax.io/docs/guides/pricing-paygo",
                    "model_id": "MiniMax-M2.5-highspeed",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {},
            "schedules": [],
            "exclusions": ["Taxes, account credits/free allowances, and negotiated discounts."],
            "reason": "MiniMax-M2.5-highspeed: current official LLM rate card and Anthropic-compatible "
            "supported-model table list only M3.1-Flash and M3. No exact current legacy-route/cache/tier "
            "price verified; no M3-family substitution.",
        },
    },
    ("minimax", "MiniMax-M2.1"): {
        "context_length": 327680,
        "max_completion_tokens": 196608,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": False,
        "pricing": {
            "schema_version": 1,
            "pricing_id": "minimax:MiniMax-M2.1:2026-11-01",
            "provider": "minimax",
            "model": "MiniMax-M2.1",
            "status": "unverified",
            "billing_mode": "pay_as_you_go",
            "currency": "USD",
            "unit": "per_million_tokens",
            "rates": {},
            "checked_at": "2026-11-01",
            "sources": [
                {
                    "url": "https://platform.minimax.io/docs/guides/pricing-paygo",
                    "model_id": "MiniMax-M2.1",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {},
            "schedules": [],
            "exclusions": ["Taxes, account credits/free allowances, and negotiated discounts."],
            "reason": "MiniMax-M2.1: current official LLM rate card and Anthropic-compatible supported-model table "
            "list only M3.1-Flash and M3. No exact current legacy-route/cache/tier price verified; no "
            "M3-family substitution.",
        },
    },
    ("minimax", "MiniMax-M2.1-highspeed"): {
        "context_length": 327680,
        "max_completion_tokens": 196608,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": False,
        "pricing": {
            "schema_version": 1,
            "pricing_id": "minimax:MiniMax-M2.1-highspeed:2026-11-01",
            "provider": "minimax",
            "model": "MiniMax-M2.1-highspeed",
            "status": "unverified",
            "billing_mode": "pay_as_you_go",
            "currency": "USD",
            "unit": "per_million_tokens",
            "rates": {},
            "checked_at": "2026-11-01",
            "sources": [
                {
                    "url": "https://platform.minimax.io/docs/guides/pricing-paygo",
                    "model_id": "MiniMax-M2.1-highspeed",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {},
            "schedules": [],
            "exclusions": ["Taxes, account credits/free allowances, and negotiated discounts."],
            "reason": "MiniMax-M2.1-highspeed: current official LLM rate card and Anthropic-compatible "
            "supported-model table list only M3.1-Flash and M3. No exact current legacy-route/cache/tier "
            "price verified; no M3-family substitution.",
        },
    },
    ("minimax", "MiniMax-M2"): {
        "context_length": 327680,
        "max_completion_tokens": 196608,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": False,
        "pricing": {
            "schema_version": 1,
            "pricing_id": "minimax:MiniMax-M2:2026-11-01",
            "provider": "minimax",
            "model": "MiniMax-M2",
            "status": "unverified",
            "billing_mode": "pay_as_you_go",
            "currency": "USD",
            "unit": "per_million_tokens",
            "rates": {},
            "checked_at": "2026-11-01",
            "sources": [
                {
                    "url": "https://platform.minimax.io/docs/guides/pricing-paygo",
                    "model_id": "MiniMax-M2",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {},
            "schedules": [],
            "exclusions": ["Taxes, account credits/free allowances, and negotiated discounts."],
            "reason": "MiniMax-M2: current official LLM rate card and Anthropic-compatible supported-model table list "
            "only M3.1-Flash and M3. No exact current legacy-route/cache/tier price verified; no M3-family "
            "substitution.",
        },
    },
}
