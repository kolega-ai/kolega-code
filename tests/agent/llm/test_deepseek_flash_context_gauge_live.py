"""Live check: retained flash reasoning keeps the context estimate ≈ billed input.

The stream wrapper retains flash's plain-text reasoning items and resends them
(mirroring Codex, the client DeepSeek adapted this endpoint for); the server
dedupes the resent copies against its own call_id-keyed restore, so billing is
unchanged while the client-side history now carries — and counts — the
reasoning. This probe verifies the whole loop against the API: over a short
multi-round tool-call session,

    tiktoken(history incl. retained reasoning)  ≈  usage.input_tokens

and the resent reasoning items are accepted (no "reasoning_text must be passed
back" 400, no double-billing).

Kept deliberately small (<= 4 sequential requests) — the DeepSeek endpoint is
shared with live benchmark runs. Run with:

    ./.venv/bin/python -m pytest tests/agent/llm/test_deepseek_flash_context_gauge_live.py -s -m integration
"""

import os

import pytest

from kolega_code.cli.config import API_KEY_ENV
from kolega_code.config import ModelProvider
from kolega_code.llm.models import (
    Message,
    MessageHistory,
    ResponsesReasoningBlock,
    TextBlock,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)
from kolega_code.llm.providers.deepseek_responses import DeepSeekResponsesProvider
from kolega_code.llm.providers.models import GenerationParams

pytestmark = [pytest.mark.slow, pytest.mark.integration]

SKIP_IN_CI = bool(os.getenv("CI")) or bool(os.getenv("GITLAB_CI"))

MODEL = "deepseek-v4-flash"
MAX_ROUNDS = 4
# Fixed offset band between tiktoken and DeepSeek's tokenizer/scaffolding
# (measured ~300 on a bare request) plus per-round item wrapping. With retention
# the estimate must stay inside this band even as reasoning accumulates; without
# it the gauge fell behind by the full reasoning total (measured 2026-08-04).
TOLERANCE = 700

FACTS = {
    "alpha": "The alpha subsystem processes 1200 requests per second at peak.",
    "beta": "The beta subsystem has a p99 latency of 45 milliseconds.",
}

PROMPT = (
    "First, solve this carefully in your head: in how many ways can you tile a 2x10 grid "
    "with 1x2 dominoes and 2x2 squares? Double-check the recurrence for several sizes. "
    "THEN call lookup_fact for topic 'alpha' (one call). After the result arrives, call "
    "lookup_fact for 'beta', then give a one-sentence final answer with the tiling count "
    "and both facts."
)


def _require_key() -> str:
    if SKIP_IN_CI:
        pytest.skip("Skipping live provider call in CI")
    env_name = API_KEY_ENV[ModelProvider.DEEPSEEK]
    api_key = os.getenv(env_name)
    if not api_key:
        pytest.skip(f"{env_name} not set")
    return api_key


@pytest.mark.asyncio
async def test_retained_reasoning_keeps_estimate_on_billed_input():
    provider = DeepSeekResponsesProvider(api_key=_require_key())
    tool = ToolDefinition(
        name="lookup_fact",
        description="Look up one internal fact by topic name. Topics: alpha, beta.",
        parameters=[ToolParameter(name="topic", type="string", description="Topic to look up", required=True)],
    )
    params = GenerationParams(tools=[tool], thinking="high")
    system = Message(role="system", content=[TextBlock(text="You are a meticulous mathematician.")])
    history = MessageHistory([Message(role="user", content=[TextBlock(text=PROMPT)])])

    retained_reasoning_chars = 0
    checked_rounds = 0
    for _round in range(MAX_ROUNDS):
        counted = await provider.count_tokens(messages=history, system=system, model=MODEL, tools=[tool])
        estimate = counted.input_tokens

        # The resent reasoning items must be accepted — a 400 here would mean the
        # replay shape regressed.
        message = await provider.generate(history, system=system, params=params, model=MODEL)
        usage = message.usage_metadata or {}
        billed_input = usage.get("prompt_tokens", 0)
        assert billed_input > 0, "flash stopped reporting usage; probe cannot conclude"

        drift = billed_input - estimate
        print(
            f"\nFLASH-GAUGE-PROBE round={_round} estimate={estimate} billed_input={billed_input} "
            f"drift={drift} retained_reasoning_chars={retained_reasoning_chars}"
        )
        assert abs(drift) <= TOLERANCE, (
            f"estimate off billed input by {drift} tokens with "
            f"{retained_reasoning_chars} chars of retained reasoning in history"
        )
        checked_rounds += 1

        history.append(message)
        reasoning_blocks = [b for b in message.content or [] if isinstance(b, ResponsesReasoningBlock)]
        retained_reasoning_chars += sum(len(t) for b in reasoning_blocks for t in b.content)
        if message.tool_calls:
            # Reasoning-bearing tool rounds must retain their reasoning, or the
            # gauge silently degrades to the old undercounting behavior.
            reasoning_tokens = usage.get("reasoning_output_tokens", 0) or 0
            if reasoning_tokens > 100:
                assert reasoning_blocks, "flash tool round with reasoning retained no reasoning block"
        else:
            break
        results = []
        for tool_call in message.tool_calls:
            topic = tool_call.input.get("topic", "") if isinstance(tool_call.input, dict) else ""
            fact = FACTS.get(str(topic).lower(), f"No fact recorded for topic '{topic}'.")
            results.append(ToolResult(tool_use_id=tool_call.id, content=fact, name=tool_call.name, is_error=False))
        history.append(Message(role="user", content=results))

    assert checked_rounds >= 2, "session ended before a multi-round comparison was possible"
    assert retained_reasoning_chars > 0, "no reasoning was retained across the session"
