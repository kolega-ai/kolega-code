from kolega_code.llm.specs.types import ThinkingEffortSpec

# Cloudflare Workers AI (account-scoped OpenAI-compatible Chat Completions).
# Windows, vision flags, and effort ladders/defaults come from Cloudflare's model
# metadata (``/ai/models/search``); Workers AI publishes no output ceilings, so
# caps are Kolega's own reservation inside a shared window. Only tool-calling
# models with at least a 128K window are curated; ``models refresh`` discovers
# the rest.
CLOUDFLARE_WORKERS_AI_SPECS = {
    ("cloudflare_workers_ai", "@cf/zai-org/glm-5.3"): {
        "context_length": 1048576,
        "max_completion_tokens": 131072,
        "input_budget": "window_minus_output",
        "default_temperature": 0.6,
        "supports_vision": False,
        "thinking_effort": ThinkingEffortSpec(
            options=("low", "high", "max"),
            default="max",
            mode="cloudflare_reasoning_effort",
        ),
    },
    ("cloudflare_workers_ai", "@cf/zai-org/glm-5.3-flash"): {
        "context_length": 1048576,
        "max_completion_tokens": 131072,
        "input_budget": "window_minus_output",
        "default_temperature": 0.6,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("low", "high", "max"),
            default="max",
            mode="cloudflare_reasoning_effort",
        ),
    },
    ("cloudflare_workers_ai", "@cf/moonshotai/kimi-k2.7-code"): {
        "context_length": 262144,
        "max_completion_tokens": 32768,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": True,
    },
    ("cloudflare_workers_ai", "@cf/deepseek-ai/deepseek-v4.1-flash-1017"): {
        "context_length": 1048576,
        "max_completion_tokens": 65536,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "high", "max"),
            default="high",
            mode="cloudflare_reasoning_effort",
        ),
    },
    ("cloudflare_workers_ai", "@cf/deepseek-ai/deepseek-v4-pro-0813"): {
        "context_length": 1048576,
        "max_completion_tokens": 65536,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": False,
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "high", "max"),
            default="high",
            mode="cloudflare_reasoning_effort",
        ),
    },
    ("cloudflare_workers_ai", "@cf/zai-org/glm-5.2"): {
        "context_length": 262144,
        "max_completion_tokens": 32768,
        "input_budget": "window_minus_output",
        "default_temperature": 0.6,
        "supports_vision": False,
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "high", "max"),
            default="max",
            mode="cloudflare_reasoning_effort",
        ),
    },
    ("cloudflare_workers_ai", "@cf/moonshotai/kimi-k2.6"): {
        "context_length": 262144,
        "max_completion_tokens": 32768,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "high"),
            default="high",
            mode="cloudflare_reasoning_effort",
        ),
    },
    ("cloudflare_workers_ai", "@cf/qwen/qwen3.8-27b"): {
        "context_length": 262144,
        "max_completion_tokens": 32768,
        "input_budget": "window_minus_output",
        "default_temperature": 0.6,
        "supports_vision": True,
        "thinking_effort": ThinkingEffortSpec(
            options=("low", "medium", "xhigh"),
            default="xhigh",
            mode="cloudflare_reasoning_effort",
        ),
    },
    ("cloudflare_workers_ai", "@cf/google/gemma-4-26b-a4b-it"): {
        "context_length": 256000,
        "max_completion_tokens": 32768,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": True,
        # Replayed reasoning makes it answer inside reasoning_content with empty content
        # after a tool result (probed 2026-10-26: 3/5 replayed vs 5/5 dropped).
        "drop_prior_reasoning": True,
    },
    ("cloudflare_workers_ai", "@cf/nvidia/nemotron-3-120b-a12b"): {
        "context_length": 256000,
        "max_completion_tokens": 32768,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": False,
    },
    ("cloudflare_workers_ai", "@cf/openai/gpt-oss-120b"): {
        "context_length": 128000,
        "max_completion_tokens": 32768,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": False,
        "preferred_edit_protocol": "codex_apply_patch",
        "thinking_effort": ThinkingEffortSpec(
            options=("low", "medium", "high"),
            default="medium",
            mode="cloudflare_reasoning_effort",
        ),
    },
    ("cloudflare_workers_ai", "@cf/openai/gpt-oss-20b"): {
        "context_length": 128000,
        "max_completion_tokens": 32768,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": False,
        "preferred_edit_protocol": "codex_apply_patch",
        "thinking_effort": ThinkingEffortSpec(
            options=("low", "medium", "high"),
            default="medium",
            mode="cloudflare_reasoning_effort",
        ),
    },
    ("cloudflare_workers_ai", "@cf/zai-org/glm-4.7-flash"): {
        "context_length": 131072,
        "max_completion_tokens": 16384,
        "input_budget": "window_minus_output",
        "default_temperature": 0.6,
        "supports_vision": False,
    },
}
