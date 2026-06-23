"""Live test for the ChatGPT-subscription (OAuth) provider.

Unlike the key-based providers, this one needs OAuth tokens from an interactive
``/login``. It is skipped unless real tokens are available, so it never runs in
CI. To run it locally after signing in via the TUI::

    pytest -m integration tests/agent/llm/test_chatgpt_live.py -v

Token source (first match wins):
  - KOLEGA_CODE_CHATGPT_TOKENS env var holding the JSON of an OAuthTokens dump, or
  - the stored token in the local settings file (the one /login writes).
"""

import json
import os

import pytest

from kolega_code.auth.tokens import ChatGPTTokenManager, OAuthTokens
from kolega_code.auth import constants as chatgpt_constants
from kolega_code.cli.settings import SettingsStore
from kolega_code.llm.client import LLMClient
from kolega_code.llm.models import Message, MessageHistory, TextBlock

pytestmark = pytest.mark.integration

SKIP_IN_CI = bool(os.getenv("CI")) or bool(os.getenv("GITLAB_CI"))


def _load_tokens() -> OAuthTokens | None:
    raw = os.getenv("KOLEGA_CODE_CHATGPT_TOKENS")
    if raw:
        try:
            return OAuthTokens.model_validate(json.loads(raw))
        except (ValueError, TypeError):
            return None
    stored = SettingsStore().load().get_oauth_token(chatgpt_constants.PROVIDER_KEY)
    if stored:
        try:
            return OAuthTokens.model_validate(stored)
        except (ValueError, TypeError):
            return None
    return None


def _require_tokens() -> OAuthTokens:
    if SKIP_IN_CI:
        pytest.skip("Skipping live ChatGPT call in CI")
    tokens = _load_tokens()
    if tokens is None:
        pytest.skip("No ChatGPT tokens available; sign in with /login or set KOLEGA_CODE_CHATGPT_TOKENS")
    return tokens


@pytest.mark.asyncio
async def test_live_chatgpt_generate() -> None:
    tokens = _require_tokens()
    client = LLMClient(
        provider=chatgpt_constants.PROVIDER_KEY,
        api_key="unused",
        token_manager=ChatGPTTokenManager(tokens),
    )

    response = await client.generate(
        messages=MessageHistory(
            [Message(role="user", content=[TextBlock(text="What is 2 + 2? Reply with just the number.")])]
        ),
        system=Message(role="system", content=[TextBlock(text="You are concise.")]),
        model=chatgpt_constants.DEFAULT_MODEL,
        max_completion_tokens=4096,
        thinking="low",
    )

    assert response.role == "assistant"
    assert response.get_text_content(), "empty response from ChatGPT backend"


@pytest.mark.asyncio
async def test_live_chatgpt_stream() -> None:
    tokens = _require_tokens()
    client = LLMClient(
        provider=chatgpt_constants.PROVIDER_KEY,
        api_key="unused",
        token_manager=ChatGPTTokenManager(tokens),
    )

    text = ""
    async with await client.stream(
        messages=MessageHistory(
            [Message(role="user", content=[TextBlock(text="Say the word 'ready' and nothing else.")])]
        ),
        model=chatgpt_constants.DEFAULT_MODEL,
        max_completion_tokens=4096,
        thinking="low",
    ) as stream:
        async for chunk in stream:
            if chunk.type == "text" and chunk.text:
                text += chunk.text
        message = await stream.get_final_message()

    assert text or message.get_text_content(), "empty stream from ChatGPT backend"
