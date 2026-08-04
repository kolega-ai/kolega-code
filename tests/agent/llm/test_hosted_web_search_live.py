"""Live hosted web_search on the Responses providers (DeepSeek flash + OpenAI).

Two requests per provider, mirroring the Phase-0 probe
(findings/probes/hosted_web_search_probe.py, results 2026-08-04):

1. force a search — `tools` gains `{"type": "web_search"}` via
   ``GenerationParams(hosted_web_search=True)``; the stream wrapper must
   capture the server-side ``web_search_call`` items as WebSearchCallBlocks.
2. replay + follow-up — resending those blocks must be ACCEPTED (a 400 here
   means the replay shape regressed) and must restore the searched content
   server-side, observable as billed input exceeding the client-side count
   (that surplus is exactly what BaseAgent's hosted-search residual feeds the
   context gauge).

Hermetic counterpart: tests/agent/llm/test_responses_hosted_web_search.py and
tests/agent/test_hosted_search_accounting.py.
"""

import os

import pytest

from kolega_code.cli.config import API_KEY_ENV
from kolega_code.config import ModelProvider
from kolega_code.llm.models import (
    Message,
    MessageHistory,
    TextBlock,
    WebSearchCallBlock,
)
from kolega_code.llm.providers.deepseek_responses import DeepSeekResponsesProvider
from kolega_code.llm.providers.models import GenerationParams
from kolega_code.llm.providers.openai_responses import OpenAIResponsesProvider

pytestmark = [pytest.mark.slow, pytest.mark.integration]

SKIP_IN_CI = bool(os.getenv("CI")) or bool(os.getenv("GITLAB_CI"))

SEARCH_PROMPT = (
    "Use the web search tool to open https://blog.rust-lang.org/releases/ and "
    "answer: what is the newest release version listed there? Reply with just "
    "the version number."
)
FOLLOWUP_PROMPT = (
    "Without searching or opening any page again, name two OTHER version "
    "numbers (not the one you already gave) that appeared on the page you "
    "opened. Reply with just the two version numbers."
)

# The replay turn's billed input must exceed the client count by at least the
# restored search corpus. Probe floor: ~2.7k (deepseek) / ~3.5k (openai); a
# conservative >1k asserts restoration without pinning provider internals. The
# ceiling guards against runaway telescoped billing on a clean turn.
MIN_RESTORED_TOKENS = 1_000
MAX_RESTORED_TOKENS = 100_000

PROVIDERS = {
    "deepseek": (DeepSeekResponsesProvider, ModelProvider.DEEPSEEK, "deepseek-v4-flash", "high"),
    "openai": (OpenAIResponsesProvider, ModelProvider.OPENAI, "gpt-5.6-sol", "low"),
}


def _provider_for(name):
    if SKIP_IN_CI:
        pytest.skip("Skipping live provider call in CI")
    provider_cls, provider_enum, model, thinking = PROVIDERS[name]
    env_name = API_KEY_ENV[provider_enum]
    api_key = os.getenv(env_name)
    if not api_key:
        pytest.skip(f"{env_name} not set")
    return provider_cls(api_key=api_key), model, thinking


@pytest.mark.parametrize("provider_name", ["deepseek", "openai"])
@pytest.mark.asyncio
async def test_hosted_search_capture_replay_and_restore(provider_name):
    provider, model, thinking = _provider_for(provider_name)
    params = GenerationParams(hosted_web_search=True, thinking=thinking, max_completion_tokens=4000)
    history = MessageHistory([Message(role="user", content=[TextBlock(text=SEARCH_PROMPT)])])

    # --- turn 1: the model must actually search, and we must capture it ---
    message = await provider.generate(history, system=None, params=params, model=model)
    search_blocks = [b for b in message.content or [] if isinstance(b, WebSearchCallBlock)]
    assert search_blocks, f"{provider_name} answered without a web_search_call — hosted tool not exposed?"
    for block in search_blocks:
        assert block.action_type in ("search", "open_page", "find_in_page"), block.action
    answer_1 = message.get_text_content()
    assert answer_1.strip(), "search turn produced no visible answer"
    usage_1 = message.usage_metadata or {}
    assert usage_1.get("prompt_tokens", 0) > 0

    # --- turn 2: replay must be accepted and the searched content restored ---
    history.append(message)
    history.append(Message(role="user", content=[TextBlock(text=FOLLOWUP_PROMPT)]))
    counted = await provider.count_tokens(messages=history, system=None, model=model, tools=[])
    followup = await provider.generate(history, system=None, params=params, model=model)

    answer_2 = followup.get_text_content()
    assert answer_2.strip(), "follow-up produced no answer — replay may have been rejected"
    new_searches = [b for b in followup.content or [] if isinstance(b, WebSearchCallBlock)]
    billed = (followup.usage_metadata or {}).get("prompt_tokens", 0)
    restored = billed - counted.input_tokens
    print(
        f"\nHOSTED-SEARCH-LIVE provider={provider_name} turn1_searches={len(search_blocks)} "
        f"turn2_searches={len(new_searches)} billed={billed} counted={counted.input_tokens} "
        f"restored={restored} answer2={answer_2[:80]!r}"
    )
    if not new_searches:
        # Clean follow-up: the residual measures the restored corpus directly.
        assert MIN_RESTORED_TOKENS < restored < MAX_RESTORED_TOKENS, (
            f"billed−counted={restored}: outside the restore band — either the server stopped "
            "restoring searched content on replay (gauge residual now overcounts nothing) or "
            "billing semantics changed (residual would poison the gauge)"
        )
