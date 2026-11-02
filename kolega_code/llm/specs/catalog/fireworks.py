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
        "pricing": {
            "schema_version": 1,
            "pricing_id": "fireworks:accounts/fireworks/models/kimi-k3:2026-11-01",
            "provider": "fireworks",
            "model": "accounts/fireworks/models/kimi-k3",
            "billing_model": "accounts/fireworks/models/kimi-k3",
            "status": "priced",
            "billing_mode": "pay_as_you_go",
            "currency": "USD",
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
                    "url": "https://docs.fireworks.ai/serverless/pricing",
                    "model_id": "accounts/fireworks/models/kimi-k3",
                    "checked_at": "2026-11-01",
                },
                {
                    "url": "https://docs.fireworks.ai/serverless/serverless-modes",
                    "model_id": "accounts/fireworks/models/kimi-k3",
                    "checked_at": "2026-11-01",
                },
                {
                    "url": "https://docs.fireworks.ai/serverless/us-only-serverless",
                    "model_id": "accounts/fireworks/models/kimi-k3",
                    "checked_at": "2026-11-01",
                },
            ],
            "conditions": {},
            "schedules": [
                {
                    "id": "published-0",
                    "mode": "whole_request",
                    "when": {"service_tier": "standard", "region": "global"},
                    "rates": {"input": "3", "output": "15", "cache_read": "0.3"},
                },
                {
                    "id": "published-1",
                    "mode": "whole_request",
                    "when": {"service_tier": "priority", "region": "global"},
                    "rates": {"input": "3.75", "output": "18.75", "cache_read": "0.375"},
                },
                {
                    "id": "published-2",
                    "mode": "whole_request",
                    "when": {"service_tier": "batch", "region": "global"},
                    "rates": {"input": "1.5", "output": "7.5"},
                },
            ],
            "cache_policy": {"cache_read": "separate", "cache_write": "included"},
            "exclusions": [
                "Taxes, negotiated discounts and account-level allowances.",
                "Batch cached-input discount is not specified in the Batch rule; no inferred cached Batch rate.",
                "Fast and US-only require accounts/fireworks/routers/kimi-k3-fast and "
                "accounts/fireworks/routers/kimi-k3-us; they are not service tiers of this base model ID.",
            ],
            "requires_schedule": True,
        },
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
        "pricing": {
            "schema_version": 1,
            "pricing_id": "fireworks:accounts/fireworks/models/glm-5p2:2026-11-01",
            "provider": "fireworks",
            "model": "accounts/fireworks/models/glm-5p2",
            "billing_model": "accounts/fireworks/models/glm-5p2",
            "status": "retired",
            "billing_mode": "serverless_inference",
            "retired_at": "2026-09-25",
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
                    "url": "https://docs.fireworks.ai/updates/changelog",
                    "model_id": "accounts/fireworks/models/glm-5p2",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {},
            "schedules": [],
            "exclusions": ["Taxes, negotiated discounts and account-level allowances."],
            "reason": "Fireworks retired glm-5p2 from public serverless on September 25, 2026, including Fast "
            "and US-only variants. Dedicated deployments are unaffected but have different billing; "
            "neither their GPU costs nor a successor's token rates price this serverless route.",
        },
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
        "pricing": {
            "schema_version": 1,
            "pricing_id": "fireworks:accounts/fireworks/models/glm-5p1:2026-11-01",
            "provider": "fireworks",
            "model": "accounts/fireworks/models/glm-5p1",
            "billing_model": "accounts/fireworks/models/glm-5p1",
            "status": "requires_configuration",
            "billing_mode": "dedicated_deployment",
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
                    "url": "https://app.fireworks.ai/models/fireworks/glm-5p1",
                    "model_id": "accounts/fireworks/models/glm-5p1",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {},
            "schedules": [],
            "exclusions": ["Taxes, negotiated discounts and account-level allowances."],
            "reason": "The glm-5p1 model page explicitly says Serverless: Not supported. Dedicated GPU deployment "
            "pricing requires hardware, replica and uptime configuration; a successor's serverless or "
            "size-band rate is not applicable.",
        },
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
        "pricing": {
            "schema_version": 1,
            "pricing_id": "fireworks:accounts/fireworks/models/kimi-k2p7-code:2026-11-01",
            "provider": "fireworks",
            "model": "accounts/fireworks/models/kimi-k2p7-code",
            "billing_model": "accounts/fireworks/models/kimi-k2p7-code",
            "status": "retired",
            "billing_mode": "serverless_inference",
            "retired_at": "2026-09-25",
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
                    "url": "https://docs.fireworks.ai/updates/changelog",
                    "model_id": "accounts/fireworks/models/kimi-k2p7-code",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {},
            "schedules": [],
            "exclusions": ["Taxes, negotiated discounts and account-level allowances."],
            "reason": "Fireworks retired kimi-k2p7-code from public serverless on September 25, 2026. "
            "Dedicated deployments are unaffected but have different billing; a successor's token rates "
            "or dedicated GPU costs do not price this serverless route.",
        },
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
        "pricing": {
            "schema_version": 1,
            "pricing_id": "fireworks:accounts/fireworks/models/deepseek-v4-pro:2026-11-01",
            "provider": "fireworks",
            "model": "accounts/fireworks/models/deepseek-v4-pro",
            "billing_model": "accounts/fireworks/models/deepseek-v4-pro",
            "status": "retired",
            "billing_mode": "serverless_inference",
            "retired_at": "2026-08-27",
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
                    "url": "https://docs.fireworks.ai/updates/changelog",
                    "model_id": "accounts/fireworks/models/deepseek-v4-pro",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {},
            "schedules": [],
            "exclusions": ["Taxes, negotiated discounts and account-level allowances."],
            "reason": "Fireworks retired the unsuffixed deepseek-v4-pro serverless ID on August 27, 2026. "
            "The separately named 0813 replacement was later retired on September 25, 2026. Neither "
            "replacement rates nor dedicated GPU costs price this exact serverless route.",
        },
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
        "pricing": {
            "schema_version": 1,
            "pricing_id": "fireworks:accounts/fireworks/models/deepseek-v4-flash:2026-11-01",
            "provider": "fireworks",
            "model": "accounts/fireworks/models/deepseek-v4-flash",
            "billing_model": "accounts/fireworks/models/deepseek-v4-flash",
            "status": "retired",
            "billing_mode": "serverless_inference",
            "retired_at": "2026-08-14",
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
                    "url": "https://docs.fireworks.ai/updates/changelog",
                    "model_id": "accounts/fireworks/models/deepseek-v4-flash",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {},
            "schedules": [],
            "exclusions": ["Taxes, negotiated discounts and account-level allowances."],
            "reason": "Fireworks retired the unsuffixed deepseek-v4-flash serverless ID on August 14, 2026. "
            "The separately named 0731 replacement was later retired on September 25, 2026. Neither "
            "replacement rates nor dedicated GPU costs price this exact serverless route.",
        },
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
        "pricing": {
            "schema_version": 1,
            "pricing_id": "fireworks:accounts/fireworks/models/minimax-m3:2026-11-01",
            "provider": "fireworks",
            "model": "accounts/fireworks/models/minimax-m3",
            "billing_model": "accounts/fireworks/models/minimax-m3",
            "status": "priced",
            "billing_mode": "pay_as_you_go",
            "currency": "USD",
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
                    "url": "https://docs.fireworks.ai/serverless/pricing",
                    "model_id": "accounts/fireworks/models/minimax-m3",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {},
            "schedules": [
                {
                    "id": "published-0",
                    "mode": "whole_request",
                    "when": {"service_tier": "standard", "region": "global"},
                    "rates": {"input": "0.3", "output": "1.2", "cache_read": "0.06"},
                },
                {
                    "id": "published-1",
                    "mode": "whole_request",
                    "when": {"service_tier": "priority", "region": "global"},
                    "rates": {"input": "0.45", "output": "1.8", "cache_read": "0.09"},
                },
                {
                    "id": "published-2",
                    "mode": "whole_request",
                    "when": {"service_tier": "batch", "region": "global"},
                    "rates": {"input": "0.15", "output": "0.60"},
                },
            ],
            "cache_policy": {"cache_read": "separate", "cache_write": "included"},
            "exclusions": [
                "Taxes, negotiated discounts and account-level allowances.",
                "Batch cached-input discount is not specified in the Batch rule; no inferred cached Batch rate.",
            ],
            "requires_schedule": True,
        },
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
        "pricing": {
            "schema_version": 1,
            "pricing_id": "fireworks:accounts/fireworks/models/qwen3p7-plus:2026-11-01",
            "provider": "fireworks",
            "model": "accounts/fireworks/models/qwen3p7-plus",
            "billing_model": "accounts/fireworks/models/qwen3p7-plus",
            "status": "requires_configuration",
            "billing_mode": "contact_required",
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
                    "url": "https://app.fireworks.ai/models/fireworks/qwen3p7-plus",
                    "model_id": "accounts/fireworks/models/qwen3p7-plus",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {},
            "schedules": [],
            "exclusions": ["Taxes, negotiated discounts and account-level allowances."],
            "reason": "The exact qwen3p7-plus model page marks Serverless and On-demand as not supported "
            "and directs users to contact Fireworks. No applicable public token rate is established; "
            "do not infer dedicated availability, a size-band rate, or another model's price.",
        },
    },
    # Ember-1 (Fireworks Research research preview, 2026-09-22): a post-trained
    # Kimi K3 that emits ~40% shorter reasoning traces at K3 quality. 1M window /
    # 131072 output ceiling, matching the K3 entry on the same serverless stack.
    # Unlike K3's max-only entry, Ember-1 honors the full effort ladder: probed
    # live 2026-09-30, all five values are accepted and `none` measurably
    # disables reasoning (0 reasoning tokens vs >0 at every other level).
    # Appended last on purpose: catalog insertion order drives the Settings
    # provider-switch fallback.
    ("fireworks", "accounts/fireworks/models/ember-1"): {
        "context_length": 1048576,
        "max_completion_tokens": 131072,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "medium", "high", "max"),
            default="medium",
            mode="openai_reasoning_effort",
        ),
        "pricing": {
            "schema_version": 1,
            "pricing_id": "fireworks:accounts/fireworks/models/ember-1:2026-11-01",
            "provider": "fireworks",
            "model": "accounts/fireworks/models/ember-1",
            "billing_model": "accounts/fireworks/models/ember-1",
            "status": "priced",
            "billing_mode": "pay_as_you_go",
            "currency": "USD",
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
                    "url": "https://docs.fireworks.ai/serverless/pricing",
                    "model_id": "accounts/fireworks/models/ember-1",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {},
            "schedules": [
                {
                    "id": "published-0",
                    "mode": "whole_request",
                    "when": {"service_tier": "standard", "region": "global"},
                    "rates": {"input": "3", "output": "15", "cache_read": "0.3"},
                },
                {
                    "id": "published-1",
                    "mode": "whole_request",
                    "when": {"service_tier": "priority", "region": "global"},
                    "rates": {"input": "3.75", "output": "18.75", "cache_read": "0.375"},
                },
                {
                    "id": "published-2",
                    "mode": "whole_request",
                    "when": {"service_tier": "batch", "region": "global"},
                    "rates": {"input": "1.5", "output": "7.5"},
                },
            ],
            "cache_policy": {"cache_read": "separate", "cache_write": "included"},
            "exclusions": [
                "Taxes, negotiated discounts and account-level allowances.",
                "Batch cached-input discount is not specified in the Batch rule; no inferred cached Batch rate.",
            ],
            "requires_schedule": True,
        },
    },
}
