from kolega_code.llm.specs.types import ThinkingEffortSpec

# OpenAI via ChatGPT subscription (Responses API, OAuth). Model slugs mirror
# the Codex model picker; context/output limits mirror the API gpt-5.x specs
# and are server-enforced (we never send max_output_tokens).
# supports_hosted_web_search: the Codex backend executes {"type": "web_search"}
# server-side and restores searched content on replay of the web_search_call
# items — verified live through this OAuth transport 2026-08-04 (open_page
# executed, replay accepted, ~4.4k tokens restored on a clean follow-up,
# matching api.openai.com). Codex itself ships hosted search against this
# backend, so every slug here gets the flag.
# Note: The Codex backend advertises a 272K context window for GPT-5.6 and
# GPT-5.5, despite the API models exposing a larger window. Keep the
# subscription-specific values here so compression runs before backend limits.
OPENAI_CHATGPT_SPECS = {
    ("openai_chatgpt", "gpt-6-astra"): {
        # Retain the conservative subscription budget used by GPT-5.6 until
        # Astra's larger input window is verified on the Codex backend.
        "context_length": 400000,
        "max_completion_tokens": 128000,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "supports_hosted_web_search": True,
        "preferred_edit_protocol": "codex_apply_patch",
        "thinking_effort": ThinkingEffortSpec(
            options=("low", "medium", "high", "xhigh", "max"),
            default="medium",
            mode="openai_responses_reasoning",
        ),
        "pricing": {
            "schema_version": 1,
            "pricing_id": "openai_chatgpt:gpt-6-astra:2026-11-01",
            "provider": "openai_chatgpt",
            "model": "gpt-6-astra",
            "billing_model": "gpt-6-astra",
            "status": "priced",
            "billing_mode": "subscription_or_purchased_credits",
            "unit": "per_million_tokens",
            "checked_at": "2026-11-01",
            "rates": {"input": "250", "output": "1250", "cache_read": "25"},
            "default_context": {
                "service_tier": "standard",
                "region": "global",
                "modality": "text",
                "billing_surface": "purchased_credits",
            },
            "sources": [
                {
                    "url": "https://developers.openai.com/codex/pricing",
                    "model_id": "gpt-6-astra",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {"billing_surface": ["purchased_credits", "enterprise_pay_as_you_go_credits"]},
            "schedules": [
                {
                    "id": "published-0",
                    "mode": "whole_request",
                    "when": {"service_tier": "standard"},
                    "rates": {"input": "250", "output": "1250", "cache_read": "25"},
                },
                {
                    "id": "published-1",
                    "mode": "whole_request",
                    "when": {"service_tier": "fast"},
                    "rates": {"input": "500", "output": "2500", "cache_read": "50"},
                },
                {
                    "id": "published-2",
                    "mode": "whole_request",
                    "when": {"service_tier": "ultrafast"},
                    "rates": {"input": "1500", "output": "7500", "cache_read": "150"},
                },
            ],
            "cache_policy": {"cache_read": "separate", "cache_write": "included"},
            "exclusions": ["Taxes, negotiated discounts and account-level allowances."],
            "denomination": "openai_codex_credits",
            "assumptions": [
                "Non-monetary Standard credit list value, not marginal subscription cash cost.",
                "Included subscription limits have different speed multipliers and cannot be priced from this card.",
            ],
            "requires_schedule": True,
        },
    },
    ("openai_chatgpt", "gpt-5.6-sol"): {
        "context_length": 400000,
        "max_completion_tokens": 128000,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "supports_hosted_web_search": True,
        "preferred_edit_protocol": "codex_apply_patch",
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "medium", "high", "xhigh", "max"),
            default="medium",
            mode="openai_responses_reasoning",
        ),
        "pricing": {
            "schema_version": 1,
            "pricing_id": "openai_chatgpt:gpt-5.6-sol:2026-11-01",
            "provider": "openai_chatgpt",
            "model": "gpt-5.6-sol",
            "billing_model": "gpt-5.6-sol",
            "status": "priced",
            "billing_mode": "subscription_or_purchased_credits",
            "unit": "per_million_tokens",
            "checked_at": "2026-11-01",
            "rates": {"input": "100", "output": "500", "cache_read": "10"},
            "default_context": {
                "service_tier": "standard",
                "region": "global",
                "modality": "text",
                "billing_surface": "purchased_credits",
            },
            "sources": [
                {
                    "url": "https://developers.openai.com/codex/pricing",
                    "model_id": "gpt-5.6-sol",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {"billing_surface": ["purchased_credits", "enterprise_pay_as_you_go_credits"]},
            "schedules": [
                {
                    "id": "published-0",
                    "mode": "whole_request",
                    "when": {"service_tier": "standard"},
                    "rates": {"input": "100", "output": "500", "cache_read": "10"},
                },
                {
                    "id": "published-1",
                    "mode": "whole_request",
                    "when": {"service_tier": "fast"},
                    "rates": {"input": "200", "output": "1000", "cache_read": "20"},
                },
            ],
            "cache_policy": {"cache_read": "separate", "cache_write": "included"},
            "exclusions": ["Taxes, negotiated discounts and account-level allowances."],
            "denomination": "openai_codex_credits",
            "assumptions": [
                "Non-monetary Standard credit list value, not marginal subscription cash cost.",
                "Included subscription limits have different speed multipliers and cannot be priced from this card.",
            ],
            "requires_schedule": True,
            "effective_to": "2026-11-22",
        },
    },
    ("openai_chatgpt", "gpt-5.6-terra"): {
        "context_length": 400000,
        "max_completion_tokens": 128000,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "supports_hosted_web_search": True,
        "preferred_edit_protocol": "codex_apply_patch",
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "medium", "high", "xhigh", "max"),
            default="medium",
            mode="openai_responses_reasoning",
        ),
        "pricing": {
            "schema_version": 1,
            "pricing_id": "openai_chatgpt:gpt-5.6-terra:2026-11-01",
            "provider": "openai_chatgpt",
            "model": "gpt-5.6-terra",
            "billing_model": "gpt-5.6-terra",
            "status": "priced",
            "billing_mode": "subscription_or_purchased_credits",
            "unit": "per_million_tokens",
            "checked_at": "2026-11-01",
            "rates": {"input": "50", "output": "300", "cache_read": "5"},
            "default_context": {
                "service_tier": "standard",
                "region": "global",
                "modality": "text",
                "billing_surface": "purchased_credits",
            },
            "sources": [
                {
                    "url": "https://developers.openai.com/codex/pricing",
                    "model_id": "gpt-5.6-terra",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {"billing_surface": ["purchased_credits", "enterprise_pay_as_you_go_credits"]},
            "schedules": [
                {
                    "id": "published-0",
                    "mode": "whole_request",
                    "when": {"service_tier": "standard"},
                    "rates": {"input": "50", "output": "300", "cache_read": "5"},
                },
                {
                    "id": "published-1",
                    "mode": "whole_request",
                    "when": {"service_tier": "fast"},
                    "rates": {"input": "100", "output": "600", "cache_read": "10"},
                },
            ],
            "cache_policy": {"cache_read": "separate", "cache_write": "included"},
            "exclusions": ["Taxes, negotiated discounts and account-level allowances."],
            "denomination": "openai_codex_credits",
            "assumptions": [
                "Non-monetary Standard credit list value, not marginal subscription cash cost.",
                "Included subscription limits have different speed multipliers and cannot be priced from this card.",
            ],
            "requires_schedule": True,
        },
    },
    ("openai_chatgpt", "gpt-5.6-luna"): {
        "context_length": 400000,
        "max_completion_tokens": 128000,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "supports_hosted_web_search": True,
        "preferred_edit_protocol": "codex_apply_patch",
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "medium", "high", "xhigh", "max"),
            default="medium",
            mode="openai_responses_reasoning",
        ),
        "pricing": {
            "schema_version": 1,
            "pricing_id": "openai_chatgpt:gpt-5.6-luna:2026-11-01",
            "provider": "openai_chatgpt",
            "model": "gpt-5.6-luna",
            "billing_model": "gpt-5.6-luna",
            "status": "priced",
            "billing_mode": "subscription_or_purchased_credits",
            "unit": "per_million_tokens",
            "checked_at": "2026-11-01",
            "rates": {"input": "5", "output": "30", "cache_read": "0.5"},
            "default_context": {
                "service_tier": "standard",
                "region": "global",
                "modality": "text",
                "billing_surface": "purchased_credits",
            },
            "sources": [
                {
                    "url": "https://developers.openai.com/codex/pricing",
                    "model_id": "gpt-5.6-luna",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {"billing_surface": ["purchased_credits", "enterprise_pay_as_you_go_credits"]},
            "schedules": [
                {
                    "id": "published-0",
                    "mode": "whole_request",
                    "when": {"service_tier": "standard"},
                    "rates": {"input": "5", "output": "30", "cache_read": "0.5"},
                },
                {
                    "id": "published-1",
                    "mode": "whole_request",
                    "when": {"service_tier": "fast"},
                    "rates": {"input": "10", "output": "60", "cache_read": "1.0"},
                },
            ],
            "cache_policy": {"cache_read": "separate", "cache_write": "included"},
            "exclusions": ["Taxes, negotiated discounts and account-level allowances."],
            "denomination": "openai_codex_credits",
            "assumptions": [
                "Non-monetary Standard credit list value, not marginal subscription cash cost.",
                "Included subscription limits have different speed multipliers and cannot be priced from this card.",
            ],
            "requires_schedule": True,
        },
    },
    ("openai_chatgpt", "gpt-5.5"): {
        "context_length": 400000,
        "max_completion_tokens": 128000,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "supports_hosted_web_search": True,
        "preferred_edit_protocol": "codex_apply_patch",
        "thinking_effort": ThinkingEffortSpec(
            options=("minimal", "low", "medium", "high", "xhigh"),
            default="medium",
            mode="openai_responses_reasoning",
        ),
        "pricing": {
            "schema_version": 1,
            "pricing_id": "openai_chatgpt:gpt-5.5:2026-11-01",
            "provider": "openai_chatgpt",
            "model": "gpt-5.5",
            "billing_model": "gpt-5.5",
            "status": "subscription",
            "billing_mode": "subscription",
            "unit": "per_million_tokens",
            "checked_at": "2026-11-01",
            "rates": {},
            "default_context": {
                "service_tier": "standard",
                "region": "global",
                "modality": "text",
                "billing_surface": "api",
            },
            "sources": [
                {
                    "url": "https://developers.openai.com/codex/pricing",
                    "model_id": "gpt-5.5",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {},
            "schedules": [],
            "exclusions": ["Taxes, negotiated discounts and account-level allowances."],
            "reason": "gpt-5.5 is not listed in the current Codex token-credit card. OAuth subscription usage cannot "
            "be converted using API-key USD rates; included limits and purchased-credit eligibility differ.",
        },
    },
    ("openai_chatgpt", "gpt-5.4"): {
        "context_length": 400000,
        "max_completion_tokens": 128000,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "supports_hosted_web_search": True,
        "preferred_edit_protocol": "codex_apply_patch",
        "thinking_effort": ThinkingEffortSpec(
            options=("minimal", "low", "medium", "high"),
            default="medium",
            mode="openai_responses_reasoning",
        ),
        "pricing": {
            "schema_version": 1,
            "pricing_id": "openai_chatgpt:gpt-5.4:2026-11-01",
            "provider": "openai_chatgpt",
            "model": "gpt-5.4",
            "billing_model": "gpt-5.4",
            "status": "subscription",
            "billing_mode": "subscription",
            "unit": "per_million_tokens",
            "checked_at": "2026-11-01",
            "rates": {},
            "default_context": {
                "service_tier": "standard",
                "region": "global",
                "modality": "text",
                "billing_surface": "api",
            },
            "sources": [
                {
                    "url": "https://developers.openai.com/codex/pricing",
                    "model_id": "gpt-5.4",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {},
            "schedules": [],
            "exclusions": ["Taxes, negotiated discounts and account-level allowances."],
            "reason": "gpt-5.4 is not listed in the current Codex token-credit card. OAuth subscription usage cannot "
            "be converted using API-key USD rates; included limits and purchased-credit eligibility differ.",
        },
    },
    ("openai_chatgpt", "gpt-5.4-mini"): {
        "context_length": 400000,
        "max_completion_tokens": 128000,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "supports_hosted_web_search": True,
        "preferred_edit_protocol": "codex_apply_patch",
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "medium", "high", "xhigh"),
            default="medium",
            mode="openai_responses_reasoning",
        ),
        "pricing": {
            "schema_version": 1,
            "pricing_id": "openai_chatgpt:gpt-5.4-mini:2026-11-01",
            "provider": "openai_chatgpt",
            "model": "gpt-5.4-mini",
            "billing_model": "gpt-5.4-mini",
            "status": "subscription",
            "billing_mode": "subscription",
            "unit": "per_million_tokens",
            "checked_at": "2026-11-01",
            "rates": {},
            "default_context": {
                "service_tier": "standard",
                "region": "global",
                "modality": "text",
                "billing_surface": "api",
            },
            "sources": [
                {
                    "url": "https://developers.openai.com/codex/pricing",
                    "model_id": "gpt-5.4-mini",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {},
            "schedules": [],
            "exclusions": ["Taxes, negotiated discounts and account-level allowances."],
            "reason": "gpt-5.4-mini is not listed in the current Codex token-credit card. OAuth subscription usage "
            "cannot be converted using API-key USD rates; included limits and purchased-credit eligibility "
            "differ.",
        },
    },
    ("openai_chatgpt", "gpt-5.3-codex-spark"): {
        # Input limit probed live 2026-08-11: accepted 122,473, rejected ~129.6k.
        "context_length": 128000,
        "max_completion_tokens": 128000,
        "input_budget": "separate_output_limit",
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "supports_hosted_web_search": True,
        "preferred_edit_protocol": "codex_apply_patch",
        "thinking_effort": ThinkingEffortSpec(
            options=("minimal", "low", "medium"),
            default="low",
            mode="openai_responses_reasoning",
        ),
        "pricing": {
            "schema_version": 1,
            "pricing_id": "openai_chatgpt:gpt-5.3-codex-spark:2026-11-01",
            "provider": "openai_chatgpt",
            "model": "gpt-5.3-codex-spark",
            "billing_model": "gpt-5.3-codex-spark",
            "status": "subscription",
            "billing_mode": "subscription",
            "unit": "per_million_tokens",
            "checked_at": "2026-11-01",
            "rates": {},
            "default_context": {
                "service_tier": "standard",
                "region": "global",
                "modality": "text",
                "billing_surface": "api",
            },
            "sources": [
                {
                    "url": "https://developers.openai.com/codex/pricing",
                    "model_id": "gpt-5.3-codex-spark",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {},
            "schedules": [],
            "exclusions": ["Taxes, negotiated discounts and account-level allowances."],
            "reason": "gpt-5.3-codex-spark is not listed in the current Codex token-credit card. OAuth subscription "
            "usage cannot be converted using API-key USD rates; included limits and purchased-credit "
            "eligibility differ.",
        },
    },
    # GPT-6 Sol / Luna / 6.1 Sol reach Codex and ChatGPT Work on the
    # subscription backend (mirrored here so the /model picker matches the
    # Codex model picker). Budgets stay at the conservative subscription value
    # used by the rest of the GPT-6 family until the Codex backend's larger
    # window is verified, as with gpt-6-astra.
    ("openai_chatgpt", "gpt-6-sol"): {
        "context_length": 400000,
        "max_completion_tokens": 128000,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "supports_hosted_web_search": True,
        "preferred_edit_protocol": "codex_apply_patch",
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "medium", "high", "xhigh", "max"),
            default="medium",
            mode="openai_responses_reasoning",
        ),
        "pricing": {
            "schema_version": 1,
            "pricing_id": "openai_chatgpt:gpt-6-sol:2026-11-01",
            "provider": "openai_chatgpt",
            "model": "gpt-6-sol",
            "billing_model": "gpt-6-sol",
            "status": "priced",
            "billing_mode": "subscription_or_purchased_credits",
            "unit": "per_million_tokens",
            "checked_at": "2026-11-01",
            "rates": {"input": "50", "output": "250", "cache_read": "5"},
            "default_context": {
                "service_tier": "standard",
                "region": "global",
                "modality": "text",
                "billing_surface": "purchased_credits",
            },
            "sources": [
                {
                    "url": "https://developers.openai.com/codex/pricing",
                    "model_id": "gpt-6-sol",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {"billing_surface": ["purchased_credits", "enterprise_pay_as_you_go_credits"]},
            "schedules": [
                {
                    "id": "published-0",
                    "mode": "whole_request",
                    "when": {"service_tier": "standard"},
                    "rates": {"input": "50", "output": "250", "cache_read": "5"},
                },
                {
                    "id": "published-1",
                    "mode": "whole_request",
                    "when": {"service_tier": "fast"},
                    "rates": {"input": "100", "output": "500", "cache_read": "10"},
                },
            ],
            "cache_policy": {"cache_read": "separate", "cache_write": "included"},
            "exclusions": ["Taxes, negotiated discounts and account-level allowances."],
            "denomination": "openai_codex_credits",
            "assumptions": [
                "Non-monetary Standard credit list value, not marginal subscription cash cost.",
                "Included subscription limits have different speed multipliers and cannot be priced from this card.",
            ],
            "requires_schedule": True,
        },
    },
    ("openai_chatgpt", "gpt-6-luna"): {
        "context_length": 400000,
        "max_completion_tokens": 128000,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "supports_hosted_web_search": True,
        "preferred_edit_protocol": "codex_apply_patch",
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "medium", "high", "xhigh", "max"),
            default="medium",
            mode="openai_responses_reasoning",
        ),
        "pricing": {
            "schema_version": 1,
            "pricing_id": "openai_chatgpt:gpt-6-luna:2026-11-01",
            "provider": "openai_chatgpt",
            "model": "gpt-6-luna",
            "billing_model": "gpt-6-luna",
            "status": "priced",
            "billing_mode": "subscription_or_purchased_credits",
            "unit": "per_million_tokens",
            "checked_at": "2026-11-01",
            "rates": {"input": "2.5", "output": "12.5", "cache_read": "0.25"},
            "default_context": {
                "service_tier": "standard",
                "region": "global",
                "modality": "text",
                "billing_surface": "purchased_credits",
            },
            "sources": [
                {
                    "url": "https://developers.openai.com/codex/pricing",
                    "model_id": "gpt-6-luna",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {"billing_surface": ["purchased_credits", "enterprise_pay_as_you_go_credits"]},
            "schedules": [
                {
                    "id": "published-0",
                    "mode": "whole_request",
                    "when": {"service_tier": "standard"},
                    "rates": {"input": "2.5", "output": "12.5", "cache_read": "0.25"},
                },
                {
                    "id": "published-1",
                    "mode": "whole_request",
                    "when": {"service_tier": "fast"},
                    "rates": {"input": "5.0", "output": "25.0", "cache_read": "0.50"},
                },
            ],
            "cache_policy": {"cache_read": "separate", "cache_write": "included"},
            "exclusions": ["Taxes, negotiated discounts and account-level allowances."],
            "denomination": "openai_codex_credits",
            "assumptions": [
                "Non-monetary Standard credit list value, not marginal subscription cash cost.",
                "Included subscription limits have different speed multipliers and cannot be priced from this card.",
            ],
            "requires_schedule": True,
        },
    },
    ("openai_chatgpt", "gpt-6.1-sol"): {
        "context_length": 400000,
        "max_completion_tokens": 128000,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "supports_hosted_web_search": True,
        "preferred_edit_protocol": "codex_apply_patch",
        "thinking_effort": ThinkingEffortSpec(
            options=("low", "medium", "high", "xhigh", "max"),
            default="medium",
            mode="openai_responses_reasoning",
        ),
        "pricing": {
            "schema_version": 1,
            "pricing_id": "openai_chatgpt:gpt-6.1-sol:2026-11-01",
            "provider": "openai_chatgpt",
            "model": "gpt-6.1-sol",
            "billing_model": "gpt-6.1-sol",
            "status": "priced",
            "billing_mode": "subscription_or_purchased_credits",
            "unit": "per_million_tokens",
            "checked_at": "2026-11-01",
            "rates": {"input": "50", "output": "250", "cache_read": "2.5"},
            "default_context": {
                "service_tier": "standard",
                "region": "global",
                "modality": "text",
                "billing_surface": "purchased_credits",
            },
            "sources": [
                {
                    "url": "https://developers.openai.com/codex/pricing",
                    "model_id": "gpt-6.1-sol",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {"billing_surface": ["purchased_credits", "enterprise_pay_as_you_go_credits"]},
            "schedules": [
                {
                    "id": "published-0",
                    "mode": "whole_request",
                    "when": {"service_tier": "standard"},
                    "rates": {"input": "50", "output": "250", "cache_read": "2.5"},
                },
                {
                    "id": "published-1",
                    "mode": "whole_request",
                    "when": {"service_tier": "fast"},
                    "rates": {"input": "100", "output": "500", "cache_read": "5.0"},
                },
                {
                    "id": "published-2",
                    "mode": "whole_request",
                    "when": {"service_tier": "ultrafast"},
                    "rates": {"input": "300", "output": "1500", "cache_read": "15.0"},
                },
            ],
            "cache_policy": {"cache_read": "separate", "cache_write": "included"},
            "exclusions": ["Taxes, negotiated discounts and account-level allowances."],
            "denomination": "openai_codex_credits",
            "assumptions": [
                "Non-monetary Standard credit list value, not marginal subscription cash cost.",
                "Included subscription limits have different speed multipliers and cannot be priced from this card.",
            ],
            "requires_schedule": True,
        },
    },
    # GPT-6.1 Terra mirrors the API spec, including the `none` rung that GPT-6.1
    # Sol lacks. The Codex backend advertises a 500,000-token window for it and
    # accepted a 520,009-token input (probed 2026-10-27), so this entry uses the
    # real window instead of the family's conservative 400,000.
    ("openai_chatgpt", "gpt-6.1-terra"): {
        "context_length": 500000,
        "max_completion_tokens": 128000,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "supports_hosted_web_search": True,
        "preferred_edit_protocol": "codex_apply_patch",
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "medium", "high", "xhigh", "max"),
            default="medium",
            mode="openai_responses_reasoning",
        ),
        "pricing": {
            "schema_version": 1,
            "pricing_id": "openai_chatgpt:gpt-6.1-terra:2026-11-01",
            "provider": "openai_chatgpt",
            "model": "gpt-6.1-terra",
            "billing_model": "gpt-6.1-terra",
            "status": "priced",
            "billing_mode": "subscription_or_purchased_credits",
            "unit": "per_million_tokens",
            "checked_at": "2026-11-01",
            "rates": {"input": "25", "output": "125", "cache_read": "1.25"},
            "default_context": {
                "service_tier": "standard",
                "region": "global",
                "modality": "text",
                "billing_surface": "purchased_credits",
            },
            "sources": [
                {
                    "url": "https://developers.openai.com/codex/pricing",
                    "model_id": "gpt-6.1-terra",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {"billing_surface": ["purchased_credits", "enterprise_pay_as_you_go_credits"]},
            "schedules": [
                {
                    "id": "published-0",
                    "mode": "whole_request",
                    "when": {"service_tier": "standard"},
                    "rates": {"input": "25", "output": "125", "cache_read": "1.25"},
                },
                {
                    "id": "published-1",
                    "mode": "whole_request",
                    "when": {"service_tier": "fast"},
                    "rates": {"input": "50", "output": "250", "cache_read": "2.50"},
                },
            ],
            "cache_policy": {"cache_read": "separate", "cache_write": "included"},
            "exclusions": ["Taxes, negotiated discounts and account-level allowances."],
            "denomination": "openai_codex_credits",
            "assumptions": [
                "Non-monetary Standard credit list value, not marginal subscription cash cost.",
                "Included subscription limits have different speed multipliers and cannot be priced from this card.",
            ],
            "requires_schedule": True,
        },
    },
}
