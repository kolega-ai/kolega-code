"""OpenAI Responses provider backed by a ChatGPT subscription (OAuth).

Authenticates with a refreshing ChatGPT OAuth bearer token and calls the
**Responses API** at ``chatgpt.com/backend-api/codex``. The Responses request
building, streaming, and parsing are shared with the api-key
:class:`~kolega_code.llm.providers.openai_responses.OpenAIResponsesProvider` via
:mod:`kolega_code.llm.providers.responses_common`; only the transport (OAuth auth,
backend URL, Codex client headers, default model) is wired here.
"""

from __future__ import annotations

import uuid
from typing import AsyncGenerator, Optional

import httpx
from openai import AsyncOpenAI

from kolega_code.auth import constants as chatgpt_constants
from kolega_code.auth.tokens import ChatGPTTokenManager

from .base import BaseLLMProvider
from .responses_common import (  # noqa: F401  (re-exported for callers/tests)
    ResponsesProviderBase,
    ResponsesStreamWrapper,
    instructions_from,
    responses_tools,
    to_responses_input,
)


class ChatGPTAuth(httpx.Auth):
    """httpx auth flow that injects a fresh bearer token + account id per request.

    On a 401 it forces a token refresh and retries the request once, so an
    expired access token self-heals mid-session.
    """

    def __init__(self, token_manager: ChatGPTTokenManager) -> None:
        self._manager = token_manager

    async def async_auth_flow(self, request: httpx.Request) -> AsyncGenerator[httpx.Request, httpx.Response]:
        access_token, account_id = await self._manager.authorization()
        self._apply(request, access_token, account_id)
        response = yield request
        if response.status_code == 401:
            await self._manager.refresh()
            access_token, account_id = await self._manager.authorization()
            self._apply(request, access_token, account_id)
            yield request

    @staticmethod
    def _apply(request: httpx.Request, access_token: str, account_id: str) -> None:
        request.headers["Authorization"] = f"Bearer {access_token}"
        if account_id:
            request.headers[chatgpt_constants.ACCOUNT_ID_HEADER] = account_id


class ChatGPTOAuthProvider(ResponsesProviderBase):
    """OpenAI Responses-API provider backed by a ChatGPT subscription token."""

    def __init__(
        self,
        token_manager: ChatGPTTokenManager,
        max_retries: int = 3,
        requests_per_minute: Optional[int] = None,
        tokens_per_minute: Optional[int] = None,
        base_url: Optional[str] = None,
        provider_name: str = chatgpt_constants.PROVIDER_KEY,
    ) -> None:
        # Skip OpenAIProvider.__init__ (it builds Chat Completions clients with an
        # api_key); wire our own Responses client with the refreshing auth flow.
        BaseLLMProvider.__init__(
            self,
            api_key="chatgpt-oauth",
            max_retries=max_retries,
            requests_per_minute=requests_per_minute,
            tokens_per_minute=tokens_per_minute,
            base_url=base_url or chatgpt_constants.INFERENCE_BASE_URL,
        )
        self.provider_name = provider_name
        self._token_manager = token_manager
        self._session_id = str(uuid.uuid4())
        # Match Codex's client identity. The HTTP /responses backend does NOT take
        # an OpenAI-Beta header (that's the WebSocket path only); it expects the
        # codex_cli_rs originator + User-Agent.
        default_headers = {
            "originator": chatgpt_constants.ORIGINATOR,
            "User-Agent": chatgpt_constants.USER_AGENT,
            "session-id": self._session_id,
        }
        self.async_client = AsyncOpenAI(
            api_key="chatgpt-oauth",
            base_url=self.base_url,
            max_retries=max_retries,
            default_headers=default_headers,
            # Bound connect (the flat 600.0 also meant a 600s *connect* timeout); the
            # per-request streaming timeout in responses_common.stream() caps the read.
            http_client=httpx.AsyncClient(
                auth=ChatGPTAuth(token_manager),
                timeout=httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=10.0),
            ),
        )
        # The Responses path is async-only; the sync client is unused.
        self.sync_client = None

    def _default_model(self) -> str:
        return chatgpt_constants.DEFAULT_MODEL
