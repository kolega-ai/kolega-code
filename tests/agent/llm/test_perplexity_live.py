"""Live Perplexity Agent API probes (server-side tools + replay acceptance).

The parametrized live suites already exercise perplexity_agent text/usage;
these probes cover the Perplexity-specific behaviors they do not:

1. server-side web_search — ``GenerationParams(server_tools=["web_search"])``
   must produce ``search_results`` output items captured as WebSearchCallBlocks
   (item_type/payload) with a grounded answer.
2. replay acceptance — resending the assistant turn on a follow-up must be
   ACCEPTED by /v1/responses. The captured search_results items are output-only
   (replaying one 400s with 'unknown item type', live-probed 2026-08-14), so
   acceptance here proves the transport copy drops them correctly.

Costs real money when it runs (sonar tokens + $0.005 per web_search call);
skips without PERPLEXITY_API_KEY and in CI. The Gateway is not probed live:
its Router API is preview-gated (403 for keys without access, verified
2026-08-14) and rides the shared Chat Completions provider anyway.

Hermetic counterpart: tests/agent/llm/test_perplexity_provider.py.
"""

import os

import pytest

from kolega_code.cli.config import API_KEY_ENV
from kolega_code.config import ModelProvider
from kolega_code.llm.models import Message, MessageHistory, TextBlock, WebSearchCallBlock
from kolega_code.llm.providers.models import GenerationParams
from kolega_code.llm.providers.perplexity_responses import PerplexityResponsesProvider

pytestmark = [pytest.mark.slow, pytest.mark.integration]

SKIP_IN_CI = bool(os.getenv("CI")) or bool(os.getenv("GITLAB_CI"))

MODEL = "perplexity/sonar"
SEARCH_PROMPT = "Search the web for the latest stable Python release and reply with just the version number."
FOLLOWUP_PROMPT = (
    "Without searching again, in one short sentence: what is Python's release cadence for stable versions?"
)


def _provider() -> PerplexityResponsesProvider:
    api_key = os.environ.get(API_KEY_ENV[ModelProvider.PERPLEXITY_AGENT])
    if not api_key:
        pytest.skip(f"{API_KEY_ENV[ModelProvider.PERPLEXITY_AGENT]} not set")
    return PerplexityResponsesProvider(api_key=api_key)


@pytest.mark.asyncio
async def test_live_server_tool_search_results_captured_and_replayable() -> None:
    if SKIP_IN_CI:
        pytest.skip("Skipping live provider call in CI")
    provider = _provider()
    system = Message(role="system", content=[TextBlock(text="You are a terse assistant.")])

    stream = await provider.stream(
        MessageHistory([Message(role="user", content=[TextBlock(text=SEARCH_PROMPT)])]),
        system=system,
        params=GenerationParams(server_tools=["web_search"]),
        model=MODEL,
    )
    hosted_deltas = []
    async with stream:
        async for chunk in stream:
            if chunk.type == "hosted_tool_call":
                hosted_deltas.append(chunk.tool_call_delta)
    message = await stream.get_final_message()

    assert hosted_deltas, "no hosted_tool_call chunk for the server-side search"
    assert all(delta.get("item_type") == "search_results" for delta in hosted_deltas)
    blocks = [block for block in message.content if isinstance(block, WebSearchCallBlock)]
    assert blocks and blocks[0].item_type == "search_results"
    assert blocks[0].payload and blocks[0].payload.get("results"), "no client-visible results captured"
    assert message.get_text_content(), "empty answer despite search results"
    assert message.usage_metadata.get("cost") is not None, "Perplexity cost breakdown missing"

    # Replaying the assistant turn must be accepted (output-only items dropped).
    followup = MessageHistory(
        [
            Message(role="user", content=[TextBlock(text=SEARCH_PROMPT)]),
            message,
            Message(role="user", content=[TextBlock(text=FOLLOWUP_PROMPT)]),
        ]
    )
    reply = await provider.generate(
        followup, system=system, params=GenerationParams(server_tools=["web_search"]), model=MODEL
    )
    assert reply.get_text_content(), "empty follow-up reply after replaying search results"
