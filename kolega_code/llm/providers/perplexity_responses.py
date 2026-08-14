"""Perplexity provider that speaks the **Agent API** (``/responses``).

Perplexity's Responses-compatible multi-provider surface: one
``PERPLEXITY_API_KEY`` reaches ``openai/...``, ``anthropic/...``,
``google/...``, ``xai/...`` and ``perplexity/...`` models through
``https://api.perplexity.ai/v1``. (Perplexity's other surface, the inference
Gateway at ``/router/v1``, is preview-gated and is not shipped as a provider.)

The OpenAI SDK's ``responses.create`` posts to ``/v1/responses``, which
Perplexity officially accepts as an alias of its canonical ``/v1/agent``
endpoint (docs.perplexity.ai, OpenAI Compatibility Guide, verified 2026-08-14),
so request building, streaming, and token counting are shared with the OpenAI
Responses providers via
:class:`~kolega_code.llm.providers.responses_common.ResponsesProviderBase`.

Request rules specific to this backend (verified against the docs 2026-08-14):

- ``anthropic/*`` models return HTTP 400 unless ``max_output_tokens`` is sent,
  so this provider always sends one for them (the caller's cap, else the
  catalog spec's ``max_completion_tokens``).
- Perplexity's server-side tools (``web_search``, ``fetch_url``,
  ``finance_search``, ``people_search``) ride the shared ``tools`` array as bare
  ``{"type": ...}`` entries when enabled via
  ``GenerationParams.server_tools``; they execute on Perplexity's
  infrastructure and are never routed through the local tool executor. They are
  a model-spec property (see the generated catalog), gated by the shared web
  tool mode exactly like the hosted web_search on the OpenAI/DeepSeek
  Responses providers.
- The ``include`` param (encrypted reasoning) is NOT sent: support is
  unverified, and plain ``reasoning_text`` content items are retained/replayed
  by the shared stream wrapper when the backend emits them.

The response extends the OpenAI shape with ``search_results`` /
``fetch_url_results`` output items and a ``usage.cost`` breakdown; the shared
duck-typed parsing handles both.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from openai import AsyncOpenAI

from ..models import Message, MessageHistory
from ..specs import get_model_specs
from .base import BaseLLMProvider
from .models import GenerationParams
from .responses_common import ResponsesProviderBase

DEFAULT_BASE_URL = "https://api.perplexity.ai/v1"
DEFAULT_MODEL = "openai/gpt-5.6-sol"

# Anthropic models on the Agent API reject requests without an explicit output
# cap ("validation failed: max_output_tokens ...", HTTP 400) — always send one.
_ANTHROPIC_PREFIX = "anthropic/"


class PerplexityResponsesProvider(ResponsesProviderBase):
    """Perplexity Agent API provider authenticated with a standard api key."""

    def __init__(
        self,
        api_key: str,
        max_retries: int = 3,
        requests_per_minute: Optional[int] = None,
        tokens_per_minute: Optional[int] = None,
        base_url: Optional[str] = None,
        provider_name: str = "perplexity_agent",
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
        """Apply the Agent API request rules on top of the shared builder."""
        request = super()._build_request(messages, system, params, kwargs)
        model = str(request["model"])
        if model.startswith(_ANTHROPIC_PREFIX) and "max_output_tokens" not in request:
            # anthropic/* models 400 without an explicit cap.
            cap = params.max_completion_tokens if params else None
            if cap is None:
                specs = get_model_specs(self.provider_name, model)
                cap = int(specs["max_completion_tokens"])
            request["max_output_tokens"] = cap
        return request
