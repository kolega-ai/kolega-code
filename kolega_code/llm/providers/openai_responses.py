"""api-key OpenAI provider that speaks the **Responses API** (``/v1/responses``).

gpt-5.x reasoning models reject ``function tools + reasoning_effort`` on Chat
Completions ("...not supported... Please use /v1/responses instead"), so the
api-key ``openai`` provider routes here instead of :class:`OpenAIProvider`. The
OpenAI-*compatible* providers (xai, together, groq, fireworks, llama, dashscope)
keep using the Chat Completions :class:`OpenAIProvider`.

Request building, streaming, reasoning continuity, and token counting are shared
with the ChatGPT-subscription provider via
:class:`~kolega_code.llm.providers.responses_common.ResponsesProviderBase`; only
the transport differs — a plain api key against ``api.openai.com``.
"""

from __future__ import annotations

import uuid
from typing import Optional

from openai import AsyncOpenAI

from .base import BaseLLMProvider
from .responses_common import ResponsesProviderBase

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-5.5"


class OpenAIResponsesProvider(ResponsesProviderBase):
    """OpenAI Responses-API provider authenticated with a standard api key."""

    def __init__(
        self,
        api_key: str,
        max_retries: int = 3,
        requests_per_minute: Optional[int] = None,
        tokens_per_minute: Optional[int] = None,
        base_url: Optional[str] = None,
        provider_name: str = "openai",
    ) -> None:
        # Skip OpenAIProvider.__init__ (it builds Chat Completions sync+async
        # clients); wire a Responses-only async client instead.
        BaseLLMProvider.__init__(
            self,
            api_key=api_key,
            max_retries=max_retries,
            requests_per_minute=requests_per_minute,
            tokens_per_minute=tokens_per_minute,
            base_url=base_url or DEFAULT_BASE_URL,
        )
        self.provider_name = provider_name
        # A stable per-session prompt_cache_key lets the backend cache the prompt
        # prefix (incl. resent reasoning) across turns.
        self._session_id = str(uuid.uuid4())
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=self.base_url, max_retries=max_retries)
        # The Responses path is async-only; the sync client is unused.
        self.sync_client = None

    def _default_model(self) -> str:
        return DEFAULT_MODEL
