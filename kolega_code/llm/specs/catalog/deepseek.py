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
#   - Over-ceiling caps (384000) are accepted silently on both paths.
# flash is 384000 (corrected from 65536, which was pro's measured chat ceiling
# assumed to apply here): those probes only measured visible output, while a
# reasoning-heavy call was later measured running to 112990 tokens in a single
# Responses call.
# pro never sends this value raw: the Responses path clamps to
# DEEPSEEK_WIRE_OUTPUT_CAP=64000 (specs/accessors.py) so the client cap fires
# before the server ceiling and truncation is reported honestly. flash passes
# through.
DEEPSEEK_SPECS = {
    ("deepseek", "deepseek-v4-pro"): {
        "context_length": 1000000,
        "max_completion_tokens": 65536,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": False,
        "supports_hosted_web_search": True,
        "preferred_edit_protocol": "claude_code",
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "high", "max"),
            default="high",
            mode="openai_responses_reasoning",
        ),
    },
    # deepseek-v4-flash speaks the Responses API (see DeepSeekResponsesProvider), so its
    # reasoning effort must be emitted as the Responses `reasoning` block rather than the
    # Chat-Completions `reasoning_effort` field the shared Responses request builder ignores.
    # This mode also correctly excludes flash from the flat reasoning_content replay path:
    # flash's reasoning rides in Responses reasoning ITEMS instead. The endpoint returns no
    # reasoning.encrypted_content (the include param is silently ignored) but exposes the
    # raw chain-of-thought as plain reasoning_text content, which the stream wrapper
    # retains and resends next turn (mirroring Codex, the client this endpoint was adapted
    # for) — so the context gauge counts it from the history like any other content. The
    # backend ALSO restores prior CoT server-side keyed on its own function_call call_ids,
    # billing it as input and deduping explicit copies (verified live 2026-08-04); deep
    # reasoning-bearing replays with a foreign call_id 400 with "reasoning_text must be
    # passed back", minimal single-round histories with foreign ids are tolerated.
    ("deepseek", "deepseek-v4-flash"): {
        "context_length": 1000000,
        "max_completion_tokens": 384000,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": False,
        # /responses executes {"type": "web_search"} server-side (search +
        # open_page) and restores the searched content on replay of the
        # web_search_call items, billed as input (verified live 2026-08-04,
        # findings/probes/hosted_web_search_probe.py).
        "supports_hosted_web_search": True,
        "preferred_edit_protocol": "claude_code",
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "high", "max"),
            default="high",
            mode="openai_responses_reasoning",
        ),
    },
    ("deepseek", "deepseek-v4-flash-vision-exp"): {
        "context_length": 1000000,
        "max_completion_tokens": 384000,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": True,
        "supports_hosted_web_search": True,
        "preferred_edit_protocol": "claude_code",
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "high", "max"),
            default="high",
            mode="openai_responses_reasoning",
        ),
    },
}
