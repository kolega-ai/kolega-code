"""Groq catalog specs: fast OpenAI-compatible open-weight inference.

Groq serves open-weight models (gpt-oss, Qwen, Llama 4) through an
OpenAI-compatible chat-completions endpoint at 131K context. The client
already routes ``groq`` to https://api.groq.com/openai/v1; these specs make
the models catalogued, validatable, and visible in the provider picker.
The deprecated llama-3.1-8b-instant / llama-3.3-70b-versatile entries are
deliberately absent (Groq retired them 2026-08-16).
"""

GROQ_SPECS = {
    ("groq", "openai/gpt-oss-120b"): {
        "context_length": 131072,
        "max_completion_tokens": 16384,
        "input_budget": "window_minus_output",
        "default_temperature": 0.6,
        "supports_vision": False,
        "pricing": {
            "schema_version": 1,
            "pricing_id": "groq:openai/gpt-oss-120b:2026-11-01",
            "provider": "groq",
            "model": "openai/gpt-oss-120b",
            "billing_model": "openai/gpt-oss-120b",
            "status": "priced",
            "billing_mode": "pay_as_you_go",
            "currency": "USD",
            "unit": "per_million_tokens",
            "checked_at": "2026-11-01",
            "rates": {"input": "0.15", "output": "0.6", "cache_read": "0.075"},
            "default_context": {
                "service_tier": "standard",
                "region": "global",
                "modality": "text",
                "billing_surface": "api",
            },
            "sources": [
                {
                    "url": "https://console.groq.com/docs/models",
                    "model_id": "openai/gpt-oss-120b",
                    "checked_at": "2026-11-01",
                },
                {
                    "url": "https://console.groq.com/docs/model/openai/gpt-oss-120b",
                    "model_id": "openai/gpt-oss-120b",
                    "checked_at": "2026-11-01",
                },
                {
                    "url": "https://console.groq.com/docs/batch",
                    "model_id": "openai/gpt-oss-120b",
                    "checked_at": "2026-11-01",
                },
                {
                    "url": "https://console.groq.com/docs/flex-processing",
                    "model_id": "openai/gpt-oss-120b",
                    "checked_at": "2026-11-01",
                },
                {
                    "url": "https://console.groq.com/docs/prompt-caching",
                    "model_id": "openai/gpt-oss-120b",
                    "checked_at": "2026-11-01",
                },
            ],
            "conditions": {},
            "schedules": [
                {
                    "id": "published-0",
                    "mode": "whole_request",
                    "when": {"service_tier": ["standard", "on_demand"]},
                    "rates": {"input": "0.15", "output": "0.6", "cache_read": "0.075"},
                },
                {
                    "id": "published-1",
                    "mode": "whole_request",
                    "when": {"service_tier": "batch"},
                    "rates": {"input": "0.075", "output": "0.30", "cache_read": "0.075"},
                },
                {
                    "id": "published-2",
                    "mode": "whole_request",
                    "when": {"service_tier": "flex"},
                    "rates": {"input": "0.15", "output": "0.6", "cache_read": "0.075"},
                },
            ],
            "cache_policy": {"cache_read": "separate", "cache_write": "included"},
            "exclusions": [
                "Taxes, negotiated discounts and account-level allowances.",
                "Performance tier has separate availability/pricing; not on-demand pricing.",
            ],
            "requires_schedule": True,
        },
    },
    ("groq", "openai/gpt-oss-20b"): {
        "context_length": 131072,
        "max_completion_tokens": 16384,
        "input_budget": "window_minus_output",
        "default_temperature": 0.6,
        "supports_vision": False,
        "pricing": {
            "schema_version": 1,
            "pricing_id": "groq:openai/gpt-oss-20b:2026-11-01",
            "provider": "groq",
            "model": "openai/gpt-oss-20b",
            "billing_model": "openai/gpt-oss-20b",
            "status": "priced",
            "billing_mode": "pay_as_you_go",
            "currency": "USD",
            "unit": "per_million_tokens",
            "checked_at": "2026-11-01",
            "rates": {"input": "0.075", "output": "0.3", "cache_read": "0.037"},
            "default_context": {
                "service_tier": "standard",
                "region": "global",
                "modality": "text",
                "billing_surface": "api",
            },
            "sources": [
                {
                    "url": "https://console.groq.com/docs/models",
                    "model_id": "openai/gpt-oss-20b",
                    "checked_at": "2026-11-01",
                },
                {
                    "url": "https://console.groq.com/docs/model/openai/gpt-oss-20b",
                    "model_id": "openai/gpt-oss-20b",
                    "checked_at": "2026-11-01",
                },
                {
                    "url": "https://console.groq.com/docs/batch",
                    "model_id": "openai/gpt-oss-20b",
                    "checked_at": "2026-11-01",
                },
                {
                    "url": "https://console.groq.com/docs/flex-processing",
                    "model_id": "openai/gpt-oss-20b",
                    "checked_at": "2026-11-01",
                },
                {
                    "url": "https://console.groq.com/docs/prompt-caching",
                    "model_id": "openai/gpt-oss-20b",
                    "checked_at": "2026-11-01",
                },
            ],
            "conditions": {},
            "schedules": [
                {
                    "id": "published-0",
                    "mode": "whole_request",
                    "when": {"service_tier": ["standard", "on_demand"]},
                    "rates": {"input": "0.075", "output": "0.3", "cache_read": "0.037"},
                },
                {
                    "id": "published-1",
                    "mode": "whole_request",
                    "when": {"service_tier": "batch"},
                    "rates": {"input": "0.0375", "output": "0.15", "cache_read": "0.0375"},
                },
                {
                    "id": "published-2",
                    "mode": "whole_request",
                    "when": {"service_tier": "flex"},
                    "rates": {"input": "0.075", "output": "0.3", "cache_read": "0.037"},
                },
            ],
            "cache_policy": {"cache_read": "separate", "cache_write": "included"},
            "exclusions": [
                "Taxes, negotiated discounts and account-level allowances.",
                "Performance tier has separate availability/pricing; not on-demand pricing.",
            ],
            "requires_schedule": True,
        },
    },
    ("groq", "qwen/qwen3.6-27b"): {
        "context_length": 131072,
        "max_completion_tokens": 16384,
        "input_budget": "window_minus_output",
        "default_temperature": 0.6,
        "supports_vision": True,
        "pricing": {
            "schema_version": 1,
            "pricing_id": "groq:qwen/qwen3.6-27b:2026-11-01",
            "provider": "groq",
            "model": "qwen/qwen3.6-27b",
            "billing_model": "qwen/qwen3.6-27b",
            "status": "retired",
            "billing_mode": "pay_as_you_go",
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
                    "url": "https://console.groq.com/docs/deprecations",
                    "model_id": "qwen/qwen3.6-27b",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {},
            "schedules": [],
            "exclusions": ["Taxes, negotiated discounts and account-level allowances."],
            "reason": "qwen/qwen3.6-27b was shut down for public tiers on 2026-09-14; replacement-model prices do not "
            "apply. Committed-spend enterprise exceptions, where documented, require the enterprise "
            "agreement.",
        },
    },
    ("groq", "qwen/qwen3-32b"): {
        "context_length": 131072,
        "max_completion_tokens": 16384,
        "input_budget": "window_minus_output",
        "default_temperature": 0.6,
        "supports_vision": False,
        "pricing": {
            "schema_version": 1,
            "pricing_id": "groq:qwen/qwen3-32b:2026-11-01",
            "provider": "groq",
            "model": "qwen/qwen3-32b",
            "billing_model": "qwen/qwen3-32b",
            "status": "retired",
            "billing_mode": "pay_as_you_go",
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
                    "url": "https://console.groq.com/docs/deprecations",
                    "model_id": "qwen/qwen3-32b",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {},
            "schedules": [],
            "exclusions": ["Taxes, negotiated discounts and account-level allowances."],
            "reason": "qwen/qwen3-32b was shut down for public tiers on 2026-07-17; replacement-model prices do not "
            "apply. Committed-spend enterprise exceptions, where documented, require the enterprise "
            "agreement.",
        },
    },
    ("groq", "meta-llama/llama-4-scout-17b-16e-instruct"): {
        "context_length": 131072,
        "max_completion_tokens": 16384,
        "input_budget": "window_minus_output",
        "default_temperature": 0.6,
        "supports_vision": True,
        "pricing": {
            "schema_version": 1,
            "pricing_id": "groq:meta-llama/llama-4-scout-17b-16e-instruct:2026-11-01",
            "provider": "groq",
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "billing_model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "status": "retired",
            "billing_mode": "pay_as_you_go",
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
                    "url": "https://console.groq.com/docs/deprecations",
                    "model_id": "meta-llama/llama-4-scout-17b-16e-instruct",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {},
            "schedules": [],
            "exclusions": ["Taxes, negotiated discounts and account-level allowances."],
            "reason": "meta-llama/llama-4-scout-17b-16e-instruct was shut down for public tiers on 2026-07-17; "
            "replacement-model prices do not apply. Committed-spend enterprise exceptions, where "
            "documented, require the enterprise agreement.",
        },
    },
    ("groq", "meta-llama/llama-4-maverick-17b-128e-instruct"): {
        "context_length": 131072,
        "max_completion_tokens": 16384,
        "input_budget": "window_minus_output",
        "default_temperature": 0.6,
        "supports_vision": True,
        "pricing": {
            "schema_version": 1,
            "pricing_id": "groq:meta-llama/llama-4-maverick-17b-128e-instruct:2026-11-01",
            "provider": "groq",
            "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
            "billing_model": "meta-llama/llama-4-maverick-17b-128e-instruct",
            "status": "retired",
            "billing_mode": "pay_as_you_go",
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
                    "url": "https://console.groq.com/docs/deprecations",
                    "model_id": "meta-llama/llama-4-maverick-17b-128e-instruct",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {},
            "schedules": [],
            "exclusions": ["Taxes, negotiated discounts and account-level allowances."],
            "reason": "meta-llama/llama-4-maverick-17b-128e-instruct was shut down for public tiers on 2026-03-09; "
            "replacement-model prices do not apply. Committed-spend enterprise exceptions, where "
            "documented, require the enterprise agreement.",
        },
    },
}
