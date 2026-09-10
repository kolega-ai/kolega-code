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
#
# INSERTION ORDER IS LOAD-BEARING: the first entry is what the Settings model
# picker shows first and what a provider switch falls back to
# (ui_model_options(provider)[0]); it must stay in sync with
# PROVIDER_DEFAULT_MODEL[DEEPSEEK] in cli/provider_registry.py.
DEEPSEEK_SPECS = {
    # Released DeepSeek Flash (V4.1), served under the canonical `deepseek-flash`, and
    # the provider default. Verified live on /responses 2026-09-10: the retired preview
    # ID (`deepseek-v4.1-flash-expires-on-0910`) as well as the older `deepseek-v4-flash`
    # and `deepseek-v4-flash-vision-exp` aliases all now answer as
    # `model="deepseek-flash"`, and `/models` lists exactly `deepseek-flash` and
    # `deepseek-v4-pro`; a versioned `deepseek-v4.1-flash` is rejected ("The supported
    # API model names are ..."). Vision (image input), function calls, and
    # none/low/high/max reasoning effort all work on this route.
    # NO hosted web search: DeepSeek removed it silently with the V4.1 release. Their
    # Responses guide said "web_search / web_search_2025_08_26: Supported, executed on
    # the server side" and listed {"type": "web_search"} under tool_choice on
    # 2026-09-09; by 2026-09-10 the same rows read "web_search / file_search /
    # code_interpreter / computer_use / mcp / other built-in tools: Ignored", with
    # web_search dropped from tool_choice and its streaming events deleted. Nothing on
    # /updates, /news, the docs home, or the FAQ mentions the removal. Live probes on
    # 2026-09-10 returned zero web_search_call items while the model answered that it
    # has no such tool (at effort=high it emitted bogus tool-call XML as text instead).
    # Requests with the tool declared are accepted and silently do nothing, so Kolega
    # must not advertise the /web toggle here. Replayed web_search_call items from
    # older requests are still restored into context, per the same guide.
    # Context measured at 1048576 tokens (oversized request -> "This model's maximum
    # context length is 1048576 tokens"); keep the Flash-family 1000000 budget so the
    # window_minus_output input budget retains margin under the hard ceiling.
    # max_output_tokens above 393216 is rejected ("the valid range of max_tokens is
    # [1, 393216]"), so the conservative 384000 output budget stands.
    # Reasoning mechanics (inherited from the retired V4 Flash entries this id replaces,
    # and still the shape this backend serves): effort must ride the Responses
    # `reasoning` block, not the Chat-Completions `reasoning_effort` field the shared
    # builder ignores, and this mode excludes flash from the flat reasoning_content
    # replay path — reasoning comes back as Responses reasoning ITEMS instead. The
    # endpoint returns no reasoning.encrypted_content (the include param is silently
    # ignored) but exposes raw chain-of-thought as plain reasoning_text content, which
    # the stream wrapper retains and resends next turn (mirroring Codex) so the context
    # gauge counts it like any other content. The backend ALSO re-attaches prior CoT
    # server-side keyed on its own function_call call_ids, billing it as input and
    # deduping explicit copies (verified live 2026-08-04 on V4 Flash); reasoning-bearing
    # replays with a foreign call_id 400 with "reasoning_text must be passed back",
    # while minimal single-round histories with foreign ids are tolerated.
    ("deepseek", "deepseek-flash"): {
        "context_length": 1000000,
        "max_completion_tokens": 384000,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": True,
        "supports_hosted_web_search": False,
        "preferred_edit_protocol": "claude_code",
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "high", "max"),
            default="high",
            mode="openai_responses_reasoning",
        ),
    },
    ("deepseek", "deepseek-v4-pro"): {
        "context_length": 1000000,
        "max_completion_tokens": 65536,
        "input_budget": "window_minus_output",
        "default_temperature": 1.0,
        "supports_vision": False,
        # Hosted search still works on the V4 Pro model — re-verified live 2026-09-10
        # (probe A body -> ws_calls=1, real answer), and it is the last first-party
        # route that executes it. NOTE: from 12:00 Beijing Time 2026-09-14 DeepSeek
        # routes deepseek-v4-pro requests to V4.1 Flash, whose architecture has no
        # server-side web search; re-check this flag and the /web toggle after that.
        "supports_hosted_web_search": True,
        "preferred_edit_protocol": "claude_code",
        "thinking_effort": ThinkingEffortSpec(
            options=("none", "low", "high", "max"),
            default="high",
            mode="openai_responses_reasoning",
        ),
    },
}
