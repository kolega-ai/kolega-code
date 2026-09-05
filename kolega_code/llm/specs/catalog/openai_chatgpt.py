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
    },
}
