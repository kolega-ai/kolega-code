"""Live billed-vs-counted drift on DeepSeek flash (Responses).

The generalized context residual assumes billed input ≥ the client-side count
on clean single-turn requests (a small constant request-wrapper overhead;
probed live: +6 at every size, no tokenizer divergence on prose or code).
This pins the sign of that drift so the positive-only residual keeps reading
billed-true. Multi-turn tool traffic adds server-side re-attachment overhead
on top; text-only chats can bill BELOW the count (the server drops resent
reasoning with no function_call anchor), which the clamp reads as zero.
Hermetic counterpart: tests/agent/test_context_residual_accounting.py.
"""

import os

import pytest

from kolega_code.cli.config import API_KEY_ENV
from kolega_code.config import ModelProvider
from kolega_code.llm.models import Message, MessageHistory, TextBlock
from kolega_code.llm.providers.deepseek_responses import DeepSeekResponsesProvider
from kolega_code.llm.providers.models import GenerationParams

pytestmark = [pytest.mark.slow, pytest.mark.integration]

SKIP_IN_CI = bool(os.getenv("CI")) or bool(os.getenv("GITLAB_CI"))


@pytest.mark.asyncio
async def test_flash_bills_at_least_the_client_count():
    if SKIP_IN_CI:
        pytest.skip("Skipping live provider call in CI")
    api_key = os.getenv(API_KEY_ENV[ModelProvider.DEEPSEEK])
    if not api_key:
        pytest.skip(f"{API_KEY_ENV[ModelProvider.DEEPSEEK]} not set")
    provider = DeepSeekResponsesProvider(api_key=api_key)
    model = "deepseek-v4-flash"
    history = MessageHistory(
        [Message(role="user", content=[TextBlock(text="Reply with the single word: ok")])]
    )
    counted = await provider.count_tokens(messages=history, system=None, model=model, tools=[])
    message = await provider.generate(
        history, system=None, params=GenerationParams(thinking="high", max_completion_tokens=2000), model=model
    )
    billed = (message.usage_metadata or {}).get("prompt_tokens", 0)
    drift = billed - counted.input_tokens
    print(f"\nRESIDUAL-LIVE billed={billed} counted={counted.input_tokens} drift={drift}")
    assert billed > 0
    assert drift >= 0, "billed below client count — positive-only residual would misread"
