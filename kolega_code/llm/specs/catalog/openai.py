from kolega_code.llm.specs.types import ThinkingEffortSpec

# OpenAI models
# The api-key `openai` provider now uses the Responses API (gpt-5.x reject
# function tools + reasoning_effort on Chat Completions). These mirror the
# ("openai_chatgpt", …) specs below: nested reasoning + no temperature.
# supports_hosted_web_search: the Responses surface executes
# {"type": "web_search"} server-side and restores searched content on replay
# of the web_search_call items (verified live on gpt-5.6-sol 2026-08-04,
# findings/probes/hosted_web_search_probe.py; documented for all gpt-5.x).
# The ChatGPT-subscription backend (openai_chatgpt) carries the flag too,
# verified through its own OAuth transport the same day.
OPENAI_SPECS = {
    ("openai", "gpt-5.6-sol"): {
        "context_length": 1050000,
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
    ("openai", "gpt-5.6-terra"): {
        "context_length": 1050000,
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
    ("openai", "gpt-5.6-luna"): {
        "context_length": 1050000,
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
    ("openai", "gpt-5.5"): {
        "context_length": 1050000,
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
    ("openai", "gpt-5.4"): {
        "context_length": 1050000,
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
    ("openai", "gpt-5.4-mini"): {
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
    # Note: gpt-5.3-codex-spark is intentionally NOT on the API-key `openai`
    # provider — it's a Codex model that 404s on Chat Completions and is only
    # reachable through the ChatGPT-subscription backend (openai_chatgpt) below.
}
