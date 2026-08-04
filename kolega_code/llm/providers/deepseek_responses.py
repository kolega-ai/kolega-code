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
transport differs — a plain api key against ``api.deepseek.com``.

Reasoning continuity: DeepSeek returns no ``reasoning.encrypted_content`` (the
``include`` param the shared builder sends is silently ignored) but exposes the raw
chain-of-thought as plain ``reasoning_text`` content on each reasoning item. The
stream wrapper retains those items (``ResponsesReasoningBlock`` with ``content``)
and resends them next turn — the exact shape Codex records and replays against
this endpoint. The backend is NOT stateless despite its docs saying so: it ALSO
re-attaches every prior round's chain-of-thought server-side, keyed on its own
``function_call`` call_ids, bills it as input, and dedupes explicit resent copies
(verified live 2026-08-04: with-vs-without replay of the same state billed
identically; with 788 reasoning tokens in round N, round N+1's billed input grew by
806 while a reasoning-free client history grew by 51). Retaining client-side keeps
the context gauge honest (reasoning is counted from the history like any other
content) and keeps continuity independent of the server store's unspecified
retention. Passback is part of the documented thinking-mode contract, enforced
only on reasoning-bearing histories: replaying a deep multi-round state with a
foreign call_id 400s with "The reasoning_text in the thinking mode must be passed
back to the API" (findings/flash-output-token-efficiency-levers.md), while a
minimal single-round history with a foreign call_id — fabricated or a genuine
flash id from another conversation — is simply tolerated (probed 2026-08-04).

The base URL is the bare host — DeepSeek's Responses guide uses
``https://api.deepseek.com`` (no ``/v1``).
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from openai import AsyncOpenAI

from ..models import Message, MessageHistory
from ..specs import deepseek_output_token_cap
from .base import BaseLLMProvider
from .models import GenerationParams
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

    def _build_request(
        self,
        messages: MessageHistory,
        system: Optional[Message],
        params: Optional[GenerationParams],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Add the clamped output cap the shared Responses builder omits.

        DeepSeek truncates silently at its server ceiling when no cap is sent,
        but enforces AND honestly reports an explicit client cap
        (status="incomplete", incomplete_details.reason="max_output_tokens") —
        see DEEPSEEK_WIRE_OUTPUT_CAP in specs/accessors.py for the probe record.
        """
        request = super()._build_request(messages, system, params, kwargs)
        request["max_output_tokens"] = deepseek_output_token_cap(
            self.provider_name,
            str(request["model"]),
            params.max_completion_tokens if params else None,
        )
        return request
