from kolega_code.llm.specs.types import ThinkingEffortSpec


REFUSELESS_SPECS = {
    ("refuseless", "glm-5.3-abliterated"): {
        "context_length": 1048576,
        "max_completion_tokens": 131072,
        "input_budget": "output_shares_window",
        "default_temperature": 1.0,
        "supports_temperature": True,
        "supports_vision": False,
        "thinking_effort": ThinkingEffortSpec(
            options=("low", "high", "max"), default="low", mode="openai_reasoning_effort"
        ),
        "pricing": {
            "schema_version": 1,
            "pricing_id": "refuseless:glm-5.3-abliterated:2026-11-05",
            "provider": "refuseless",
            "model": "glm-5.3-abliterated",
            "status": "priced",
            "billing_mode": "prepaid_credits",
            "currency": "USD",
            "unit": "per_million_tokens",
            "checked_at": "2026-11-05",
            "rates": {"input": "1.00", "cache_read": "0.15", "output": "2.00"},
            "conditions": {},
            "schedules": [],
            "exclusions": [],
            "sources": [
                {
                    "url": "https://refuseless.com/models/glm-5.3-abliterated",
                    "model_id": "glm-5.3-abliterated",
                    "checked_at": "2026-11-05",
                }
            ],
        },
    },
    ("refuseless", "glm-5.3-flash-abliterated"): {
        "context_length": 1048576,
        "max_completion_tokens": 131072,
        "input_budget": "output_shares_window",
        "default_temperature": 1.0,
        "supports_temperature": True,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("low", "high", "max"), default="low", mode="openai_reasoning_effort"
        ),
        "pricing": {
            "schema_version": 1,
            "pricing_id": "refuseless:glm-5.3-flash-abliterated:2026-11-05",
            "provider": "refuseless",
            "model": "glm-5.3-flash-abliterated",
            "status": "priced",
            "billing_mode": "prepaid_credits",
            "currency": "USD",
            "unit": "per_million_tokens",
            "checked_at": "2026-11-05",
            "rates": {"input": "0.12", "cache_read": "0.03", "output": "0.50"},
            "conditions": {},
            "schedules": [],
            "exclusions": [],
            "sources": [
                {
                    "url": "https://refuseless.com/models/glm-5.3-flash-abliterated",
                    "model_id": "glm-5.3-flash-abliterated",
                    "checked_at": "2026-11-05",
                }
            ],
        },
    },
    ("refuseless", "deepseek-v4.1-flash-abliterated"): {
        "context_length": 1048576,
        # This exact route finished with length at 65,536 despite its higher published limit.
        "max_completion_tokens": 65536,
        "input_budget": "output_shares_window",
        "default_temperature": 1.0,
        "supports_temperature": True,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("off", "low", "high", "max"), default="low", mode="openai_reasoning_effort"
        ),
        "pricing": {
            "schema_version": 1,
            "pricing_id": "refuseless:deepseek-v4.1-flash-abliterated:2026-11-05",
            "provider": "refuseless",
            "model": "deepseek-v4.1-flash-abliterated",
            "status": "priced",
            "billing_mode": "prepaid_credits",
            "currency": "USD",
            "unit": "per_million_tokens",
            "checked_at": "2026-11-05",
            "rates": {"input": "0.12", "cache_read": "0.025", "output": "0.45"},
            "conditions": {},
            "schedules": [],
            "exclusions": [],
            "sources": [
                {
                    "url": "https://refuseless.com/models/deepseek-v4.1-flash-abliterated",
                    "model_id": "deepseek-v4.1-flash-abliterated",
                    "checked_at": "2026-11-05",
                }
            ],
        },
    },
    ("refuseless", "deepseek-v4.1-flash-fast"): {
        "context_length": 1048576,
        "max_completion_tokens": 262144,
        "input_budget": "output_shares_window",
        "default_temperature": 1.0,
        "supports_temperature": True,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("off", "low", "high", "max"), default="low", mode="openai_reasoning_effort"
        ),
        "pricing": {
            "schema_version": 1,
            "pricing_id": "refuseless:deepseek-v4.1-flash-fast:2026-11-05",
            "provider": "refuseless",
            "model": "deepseek-v4.1-flash-fast",
            "status": "priced",
            "billing_mode": "prepaid_credits",
            "currency": "USD",
            "unit": "per_million_tokens",
            "checked_at": "2026-11-05",
            "rates": {"input": "0.01", "cache_read": "0.002", "output": "0.08"},
            "conditions": {},
            "schedules": [],
            "exclusions": [],
            "sources": [
                {
                    "url": "https://refuseless.com/models/deepseek-v4.1-flash-fast",
                    "model_id": "deepseek-v4.1-flash-fast",
                    "checked_at": "2026-11-05",
                }
            ],
        },
    },
    ("refuseless", "minimax-m3.1-flash"): {
        "context_length": 1048576,
        "max_completion_tokens": 512000,
        "input_budget": "output_shares_window",
        "default_temperature": 1.0,
        "supports_temperature": True,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("low", "high", "max"), default="low", mode="openai_reasoning_effort"
        ),
        "pricing": {
            "schema_version": 1,
            "pricing_id": "refuseless:minimax-m3.1-flash:2026-11-05",
            "provider": "refuseless",
            "model": "minimax-m3.1-flash",
            "status": "priced",
            "billing_mode": "prepaid_credits",
            "currency": "USD",
            "unit": "per_million_tokens",
            "checked_at": "2026-11-05",
            "rates": {"input": "0.02", "cache_read": "0.002", "output": "0.08"},
            "conditions": {},
            "schedules": [],
            "exclusions": [],
            "sources": [
                {
                    "url": "https://refuseless.com/models/minimax-m3.1-flash",
                    "model_id": "minimax-m3.1-flash",
                    "checked_at": "2026-11-05",
                }
            ],
        },
    },
}
