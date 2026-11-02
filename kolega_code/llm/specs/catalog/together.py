# Together.ai models
TOGETHER_SPECS = {
    ("together", "moonshotai/Kimi-K2.7-Code"): {
        "context_length": 262144,
        "max_completion_tokens": 32768,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": True,
        "pricing": {
            "schema_version": 1,
            "pricing_id": "together:moonshotai/Kimi-K2.7-Code:2026-11-01",
            "provider": "together",
            "model": "moonshotai/Kimi-K2.7-Code",
            "billing_model": "moonshotai/Kimi-K2.7-Code",
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
                    "url": "https://docs.together.ai/docs/deprecations",
                    "model_id": "moonshotai/Kimi-K2.7-Code",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {},
            "schedules": [],
            "exclusions": ["Taxes, negotiated discounts and account-level allowances."],
            "reason": "moonshotai/Kimi-K2.7-Code was removed from Together serverless on 2026-08-27. Dedicated "
            "deployments are billed by deployment hardware and cannot use an invented serverless token "
            "rate.",
        },
    },
    ("together", "zai-org/GLM-5.1"): {
        "context_length": 202752,
        "max_completion_tokens": 16384,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": False,
        "pricing": {
            "schema_version": 1,
            "pricing_id": "together:zai-org/GLM-5.1:2026-11-01",
            "provider": "together",
            "model": "zai-org/GLM-5.1",
            "billing_model": "zai-org/GLM-5.1",
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
                    "url": "https://docs.together.ai/docs/deprecations",
                    "model_id": "zai-org/GLM-5.1",
                    "checked_at": "2026-11-01",
                }
            ],
            "conditions": {},
            "schedules": [],
            "exclusions": ["Taxes, negotiated discounts and account-level allowances."],
            "reason": "zai-org/GLM-5.1 was removed from Together serverless on 2026-07-10. Dedicated deployments are "
            "billed by deployment hardware and cannot use an invented serverless token rate.",
        },
    },
}
