"""Local token counting must not crash on tiktoken special-token literals.

A message that merely contains a string like ``<|endoftext|>`` (a GPT-2 marker,
common in tokenizer / ML repos and chat-format prompts) used to abort the whole
session: tiktoken's default ``disallowed_special="all"`` raised a ``ValueError``,
``count_tokens`` wrapped it as ``LLMInvalidRequestError``, and ``kolega-code ask``
exited non-zero. Counting now treats special-token literals as ordinary text
(``disallowed_special=()``), which is correct for a context-size estimate.
"""

import pytest

from kolega_code.llm.models import Message, MessageHistory, TextBlock
from kolega_code.llm.providers._token_encoding import get_counting_encoding
from kolega_code.llm.providers.anthropic import AnthropicProvider
from kolega_code.llm.providers.deepseek_responses import DeepSeekResponsesProvider
from kolega_code.llm.providers.openai import OpenAIProvider

# Real tiktoken special tokens that appear verbatim in ML/tokenizer content. Each
# of these makes a default ``Encoding.encode`` raise.
_SPECIAL_TOKEN_TEXT = "GPT-2 uses <|endoftext|>; chat formats use <|im_start|> and <|im_end|>."


def _history() -> MessageHistory:
    return MessageHistory([Message(role="user", content=[TextBlock(text=_SPECIAL_TOKEN_TEXT)])])


@pytest.mark.parametrize("encoding_name", ["cl100k_base", "p50k_base"])
def test_counting_encoding_treats_special_tokens_as_text(encoding_name):
    enc = get_counting_encoding(encoding_name)
    # Would raise ValueError under tiktoken's default disallowed_special="all".
    assert len(enc.encode(_SPECIAL_TOKEN_TEXT)) > 0


@pytest.mark.asyncio
async def test_openai_count_tokens_survives_special_token_literals():
    provider = OpenAIProvider.__new__(OpenAIProvider)  # __new__ skips __init__ (no API key)
    result = await provider.count_tokens(messages=_history(), model="gpt-4")
    assert result.input_tokens > 0


@pytest.mark.asyncio
async def test_deepseek_responses_count_tokens_survives_special_token_literals():
    # deepseek-v4-flash runs on the Responses API; DeepSeekResponsesProvider inherits
    # count_tokens from OpenAIProvider, which is exactly where the original crash was.
    provider = DeepSeekResponsesProvider.__new__(DeepSeekResponsesProvider)
    result = await provider.count_tokens(messages=_history(), model="deepseek-v4-flash")
    assert result.input_tokens > 0


@pytest.mark.asyncio
async def test_anthropic_count_tokens_survives_special_token_literals():
    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider.use_local_token_counting = True  # force the local tiktoken branch
    result = await provider.count_tokens(messages=_history(), model="claude-3-5-sonnet")
    assert result.input_tokens > 0
