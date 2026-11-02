from kolega_code.llm.specs.types import ThinkingEffortSpec

# Z.AI (GLM Coding Plan) models — Anthropic-compatible endpoint (recommended default first)
ZAI_SPECS = {
    ("zai", "glm-5.3"): {
        "context_length": 1000000,
        "max_completion_tokens": 131072,
        "input_budget": "window_minus_output",
        "default_temperature": 0.6,
        "supports_vision": False,
        "thinking_effort": ThinkingEffortSpec(
            options=("high", "max"),
            default="max",
            mode="zai_effort",
        ),
        "pricing": {
            "schema_version": 1,
            "pricing_id": "zai:glm-5.3:2026-11-01",
            "provider": "zai",
            "model": "glm-5.3",
            "status": "priced",
            "billing_mode": "coding_plan",
            "currency": None,
            "unit": "per_million_tokens",
            "rates": {"input": "345.0", "cache_read": "85.0", "output": "1200.0"},
            "checked_at": "2026-11-01",
            "sources": [
                {"url": "https://docs.z.ai/devpack/overview", "model_id": "GLM-5.3", "checked_at": "2026-11-01"},
                {"url": "https://docs.z.ai/guides/overview/pricing", "model_id": "GLM-5.3", "checked_at": "2026-11-01"},
            ],
            "conditions": {},
            "schedules": [
                {
                    "id": "coding_plan_peak",
                    "when": {
                        "timezone": "Asia/Singapore",
                        "weekdays": [0, 1, 2, 3, 4],
                        "start_time": "14:00",
                        "end_time": "18:00",
                    },
                    "rates": {"input": "690", "cache_read": "170", "output": "2400"},
                }
            ],
            "exclusions": ["Taxes, account credits/free allowances, and negotiated discounts."],
            "denomination": "zai_coding_plan_credit",
            "cache_policy": {"cache_read": "separate", "cache_write": "included"},
            "assumptions": [
                "Credit formula is weighted tokens divided by 10,000; these rates are credits per million tokens.",
                "Coding Plan credits have no universal monetary conversion; API USD list prices do not apply.",
            ],
            "published_tool_rates": {"web_search": "1.2", "web_reader": "1.2", "zread": "1.2"},
            "alternative_billing_surface": {
                "product": "Z.AI API, not the configured Coding Plan route",
                "model_id": "GLM-5.3",
                "currency": "USD",
                "unit": "per_million_tokens",
                "rates": {"input": "1.40", "output": "4.40", "cache_read": "0.26"},
                "url": "https://docs.z.ai/guides/overview/pricing",
            },
        },
    },
    ("zai", "glm-5.3-flash"): {
        "context_length": 1000000,
        "max_completion_tokens": 131072,
        "input_budget": "window_minus_output",
        "default_temperature": 0.6,
        # Native multimodal input (image/video/file); shares GLM-5.3's
        # forced-reasoning effort vocabulary (high/max).
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("high", "max"),
            default="max",
            mode="zai_effort",
        ),
        "pricing": {
            "schema_version": 1,
            "pricing_id": "zai:glm-5.3-flash:2026-11-01",
            "provider": "zai",
            "model": "glm-5.3-flash",
            "status": "priced",
            "billing_mode": "coding_plan",
            "currency": None,
            "unit": "per_million_tokens",
            "rates": {"input": "115.0", "cache_read": "28.0", "output": "400.0"},
            "checked_at": "2026-11-01",
            "sources": [
                {"url": "https://docs.z.ai/devpack/overview", "model_id": "GLM-5.3-Flash", "checked_at": "2026-11-01"},
                {
                    "url": "https://docs.z.ai/guides/overview/pricing",
                    "model_id": "GLM-5.3-Flash",
                    "checked_at": "2026-11-01",
                },
            ],
            "conditions": {},
            "schedules": [
                {
                    "id": "coding_plan_peak",
                    "when": {
                        "timezone": "Asia/Singapore",
                        "weekdays": [0, 1, 2, 3, 4],
                        "start_time": "14:00",
                        "end_time": "18:00",
                    },
                    "rates": {"input": "230", "cache_read": "56", "output": "800"},
                }
            ],
            "exclusions": ["Taxes, account credits/free allowances, and negotiated discounts."],
            "denomination": "zai_coding_plan_credit",
            "cache_policy": {"cache_read": "separate", "cache_write": "included"},
            "assumptions": [
                "Credit formula is weighted tokens divided by 10,000; these rates are credits per million tokens.",
                "Coding Plan credits have no universal monetary conversion; API USD list prices do not apply.",
            ],
            "published_tool_rates": {"web_search": "1.2", "web_reader": "1.2", "zread": "1.2"},
            "alternative_billing_surface": {
                "product": "Z.AI API, not the configured Coding Plan route",
                "model_id": "GLM-5.3-Flash",
                "currency": "USD",
                "unit": "per_million_tokens",
                "rates": {"input": "0.15", "output": "0.50", "cache_read": "0.03"},
                "url": "https://docs.z.ai/guides/overview/pricing",
            },
        },
    },
    ("zai", "glm-5.2"): {
        "context_length": 1000000,
        "max_completion_tokens": 131072,
        "input_budget": "window_minus_output",
        "default_temperature": 0.6,
        "supports_vision": False,
        "thinking_effort": ThinkingEffortSpec(
            options=("high", "max"),
            default="max",
            mode="zai_effort",
        ),
        "pricing": {
            "schema_version": 1,
            "pricing_id": "zai:glm-5.2:2026-11-01",
            "provider": "zai",
            "model": "glm-5.2",
            "status": "priced",
            "billing_mode": "coding_plan",
            "currency": None,
            "unit": "per_million_tokens",
            "rates": {"input": "345.0", "cache_read": "85.0", "output": "1200.0"},
            "checked_at": "2026-11-01",
            "sources": [
                {"url": "https://docs.z.ai/devpack/overview", "model_id": "GLM-5.3", "checked_at": "2026-11-01"},
                {"url": "https://docs.z.ai/guides/overview/pricing", "model_id": "GLM-5.2", "checked_at": "2026-11-01"},
            ],
            "conditions": {},
            "schedules": [
                {
                    "id": "coding_plan_peak",
                    "when": {
                        "timezone": "Asia/Singapore",
                        "weekdays": [0, 1, 2, 3, 4],
                        "start_time": "14:00",
                        "end_time": "18:00",
                    },
                    "rates": {"input": "690", "cache_read": "170", "output": "2400"},
                }
            ],
            "exclusions": ["Taxes, account credits/free allowances, and negotiated discounts."],
            "denomination": "zai_coding_plan_credit",
            "cache_policy": {"cache_read": "separate", "cache_write": "included"},
            "assumptions": [
                "Credit formula is weighted tokens divided by 10,000; these rates are credits per million tokens.",
                "Coding Plan credits have no universal monetary conversion; API USD list prices do not apply.",
            ],
            "published_tool_rates": {"web_search": "1.2", "web_reader": "1.2", "zread": "1.2"},
            "documented_alias": {"requested": "glm-5.2", "billing_model": "GLM-5.3"},
            "alternative_billing_surface": {
                "product": "Z.AI API, not the configured Coding Plan route",
                "model_id": "GLM-5.2",
                "currency": "USD",
                "unit": "per_million_tokens",
                "rates": {"input": "1.40", "output": "4.40", "cache_read": "0.26"},
                "url": "https://docs.z.ai/guides/overview/pricing",
            },
        },
    },
    ("zai", "glm-5.1"): {
        "context_length": 202752,
        "max_completion_tokens": 16384,
        "input_budget": "window_minus_output",
        "default_temperature": 0.6,
        "supports_vision": False,
        # GLM-5.1 predates GLM-5.2's named effort levels, so it's a plain
        # enable/disable toggle (no output_config.effort).
        "thinking_effort": ThinkingEffortSpec(
            options=("auto", "none"),
            default="auto",
            mode="zai_effort",
        ),
        "pricing": {
            "schema_version": 1,
            "pricing_id": "zai:glm-5.1:2026-11-01",
            "provider": "zai",
            "model": "glm-5.1",
            "status": "priced",
            "billing_mode": "coding_plan",
            "currency": None,
            "unit": "per_million_tokens",
            "rates": {"input": "345.0", "cache_read": "85.0", "output": "1200.0"},
            "checked_at": "2026-11-01",
            "sources": [
                {"url": "https://docs.z.ai/devpack/overview", "model_id": "GLM-5.3", "checked_at": "2026-11-01"},
                {"url": "https://docs.z.ai/guides/overview/pricing", "model_id": "GLM-5.1", "checked_at": "2026-11-01"},
            ],
            "conditions": {},
            "schedules": [
                {
                    "id": "coding_plan_peak",
                    "when": {
                        "timezone": "Asia/Singapore",
                        "weekdays": [0, 1, 2, 3, 4],
                        "start_time": "14:00",
                        "end_time": "18:00",
                    },
                    "rates": {"input": "690", "cache_read": "170", "output": "2400"},
                }
            ],
            "exclusions": ["Taxes, account credits/free allowances, and negotiated discounts."],
            "denomination": "zai_coding_plan_credit",
            "cache_policy": {"cache_read": "separate", "cache_write": "included"},
            "assumptions": [
                "Credit formula is weighted tokens divided by 10,000; these rates are credits per million tokens.",
                "Coding Plan credits have no universal monetary conversion; API USD list prices do not apply.",
            ],
            "published_tool_rates": {"web_search": "1.2", "web_reader": "1.2", "zread": "1.2"},
            "documented_alias": {"requested": "glm-5.1", "billing_model": "GLM-5.3"},
            "alternative_billing_surface": {
                "product": "Z.AI API, not the configured Coding Plan route",
                "model_id": "GLM-5.1",
                "currency": "USD",
                "unit": "per_million_tokens",
                "rates": {"input": "1.40", "output": "4.40", "cache_read": "0.26"},
                "url": "https://docs.z.ai/guides/overview/pricing",
            },
        },
    },
}
