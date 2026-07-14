from kolega_code.llm.specs.types import ThinkingEffortSpec

# OpenAI via ChatGPT subscription (Responses API, OAuth). Model slugs mirror
# the Codex model picker; context/output limits mirror the API gpt-5.x specs
# and are server-enforced (we never send max_output_tokens).
# Note: The Codex backend advertises a 272K context window for GPT-5.6 and
# GPT-5.5, despite the API models exposing a larger window. Keep the
# subscription-specific values here so compression runs before backend limits.
OPENAI_CHATGPT_SPECS = {
    ("openai_chatgpt", "gpt-5.6-sol"): {
        "context_length": 272000,
        "max_completion_tokens": 128000,
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "preferred_edit_protocol": "codex_apply_patch",
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "medium", "high", "xhigh", "max"),
            default="medium",
            mode="openai_responses_reasoning",
        ),
    },
    ("openai_chatgpt", "gpt-5.6-terra"): {
        "context_length": 272000,
        "max_completion_tokens": 128000,
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "preferred_edit_protocol": "codex_apply_patch",
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "medium", "high", "xhigh", "max"),
            default="medium",
            mode="openai_responses_reasoning",
        ),
    },
    ("openai_chatgpt", "gpt-5.6-luna"): {
        "context_length": 272000,
        "max_completion_tokens": 128000,
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "preferred_edit_protocol": "codex_apply_patch",
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "medium", "high", "xhigh", "max"),
            default="medium",
            mode="openai_responses_reasoning",
        ),
    },
    ("openai_chatgpt", "gpt-5.5"): {
        "context_length": 272000,
        "max_completion_tokens": 128000,
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
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
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
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
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "preferred_edit_protocol": "codex_apply_patch",
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "medium", "high", "xhigh"),
            default="medium",
            mode="openai_responses_reasoning",
        ),
    },
    ("openai_chatgpt", "gpt-5.3-codex-spark"): {
        "context_length": 256000,
        "max_completion_tokens": 128000,
        "default_temperature": 1.0,
        "supports_temperature": False,
        "supports_vision": True,
        "preferred_edit_protocol": "codex_apply_patch",
        "thinking_effort": ThinkingEffortSpec(
            options=("minimal", "low", "medium"),
            default="low",
            mode="openai_responses_reasoning",
        ),
    },
}
