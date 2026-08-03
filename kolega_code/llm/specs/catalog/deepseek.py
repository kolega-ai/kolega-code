from kolega_code.llm.specs.types import ThinkingEffortSpec

# max_completion_tokens is the MEASURED per-response output ceiling, not
# DeepSeek's published 384000. Probed 2026-08-03 by
# tests/agent/llm/test_deepseek_output_cap_live.py:
#   - deepseek-v4-pro (chat), uncapped: the server cuts the stream at exactly
#     65536 output tokens, reported honestly as finish_reason="length".
#   - deepseek-v4-flash (Responses), uncapped: no hard cut observed — the model
#     SELF-CENSORS ~7.4k tokens in ("response would exceed the maximum output
#     length allowed for this chat interface") with a clean status="completed".
#     With an explicit max_output_tokens=64000 it streams the full 64000 and
#     reports the cap hit honestly (status="incomplete"/max_output_tokens).
#     Family ceiling assumed equal to pro's measured 65536.
#   - Over-ceiling caps (384000) are accepted silently on both paths.
# Requests never send this value raw: both request paths clamp to
# DEEPSEEK_WIRE_OUTPUT_CAP=64000 (specs/accessors.py), so the client cap always
# fires before the server ceiling and truncation is reported honestly.
DEEPSEEK_SPECS = {
    ("deepseek", "deepseek-v4-pro"): {
        "context_length": 1000000,
        "max_completion_tokens": 65536,
        "default_temperature": 1.0,
        "supports_vision": False,
        "preferred_edit_protocol": "claude_code",
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "high", "max"),
            default="high",
            mode="deepseek_effort",
        ),
    },
    # deepseek-v4-flash speaks the Responses API (see DeepSeekResponsesProvider), so its
    # reasoning effort must be emitted as the Responses `reasoning` block rather than the
    # Chat-Completions `reasoning_effort` field the shared Responses request builder ignores.
    # DeepSeek's Responses API is stateless (no reasoning.encrypted_content), so this mode
    # also correctly excludes flash from the flat reasoning_content replay path.
    ("deepseek", "deepseek-v4-flash"): {
        "context_length": 1000000,
        "max_completion_tokens": 65536,
        "default_temperature": 1.0,
        "supports_vision": False,
        "preferred_edit_protocol": "claude_code",
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "high", "max"),
            default="high",
            mode="openai_responses_reasoning",
        ),
    },
}
