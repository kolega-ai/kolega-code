"""DeepSeek provider that speaks the **Responses API** (``/responses``).

TEMPORARY: ``deepseek-v4-flash`` is the one DeepSeek model on the Responses API
today; the rest of the deepseek catalog stays on Chat Completions
(:class:`~kolega_code.llm.providers.openai.OpenAIProvider`). DeepSeek plans to move
its catalog over (``deepseek-v4-pro`` in early Aug 2026), at which point this class
and its routing branch in ``client.py`` can be deleted and the whole ``deepseek``
provider routed to Responses wholesale.

Request building, streaming, and token counting are shared with the OpenAI
Responses providers via
:class:`~kolega_code.llm.providers.responses_common.ResponsesProviderBase`; only the
transport differs — a plain api key against ``api.deepseek.com``. DeepSeek's Responses
API is stateless (no ``previous_response_id`` / ``reasoning.encrypted_content``), so
there is no cross-turn reasoning continuity, but unsupported params (including the
``include=["reasoning.encrypted_content"]`` the shared builder sends) are silently
ignored, so the shared request builder works unmodified. The base URL is the bare
host — DeepSeek's Responses guide uses ``https://api.deepseek.com`` (no ``/v1``).
"""

from __future__ import annotations

import uuid
from typing import Optional

from openai import AsyncOpenAI

from .base import BaseLLMProvider
from .responses_common import ResponsesProviderBase

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"


class DeepSeekResponsesProvider(ResponsesProviderBase):
    """DeepSeek Responses-API provider authenticated with a standard api key."""

    def __init__(
        self,
        api_key: str,
        max_retries: int = 3,
        requests_per_minute: Optional[int] = None,
        tokens_per_minute: Optional[int] = None,
        base_url: Optional[str] = None,
        provider_name: str = "deepseek",
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
        # prefix across turns (DeepSeek ignores it if unsupported, which is harmless).
        self._session_id = str(uuid.uuid4())
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=self.base_url, max_retries=max_retries)
        # The Responses path is async-only; the sync client is unused.
        self.sync_client = None

    def _default_model(self) -> str:
        return DEFAULT_MODEL
