"""Tavily cloud backend (requires an API key), via the ``tavily-python`` SDK."""

from __future__ import annotations

import asyncio
from typing import Any

from ..network_status import looks_like_connect_failure
from .base import DEFAULT_RESULTS, SearchBackend, clamp_results
from .errors import (
    SearchBackendError,
    SearchBackendNotConfigured,
    SearchBackendRateLimited,
    SearchBackendUnavailable,
    SearchBackendUnreachable,
)
from .models import SearchResponse, SearchResult


def _new_client(*, api_key: str) -> Any:
    """Construct an SDK client, importing tavily-python on first use.

    The SDK imports ``requests`` and its whole client tree at import time — ~48 ms of
    every CLI start, paid by every command and TUI launch for a backend most sessions
    never select. Only a keyed search builds a client, so the import waits until then;
    the interpreter caches the module for later searches.
    """
    from tavily import TavilyClient

    return TavilyClient(api_key=api_key)


def _map_sdk_error(exc: Exception) -> SearchBackendError:
    """Map a ``tavily-python`` exception to this package's typed errors."""
    from tavily import errors

    message = str(exc) or exc.__class__.__name__
    if isinstance(exc, (errors.InvalidAPIKeyError, errors.MissingAPIKeyError, errors.ForbiddenError)):
        return SearchBackendNotConfigured(str(exc) or "Tavily rejected the API key.")
    if isinstance(exc, errors.UsageLimitExceededError):
        return SearchBackendRateLimited(str(exc) or "Tavily usage limit exceeded.")
    if looks_like_connect_failure(message):
        return SearchBackendUnreachable(message, host="api.tavily.com")
    return SearchBackendUnavailable(message)


class TavilyBackend(SearchBackend):
    """Search via Tavily. The synchronous SDK call runs in a worker thread."""

    name = "tavily"
    label = "Tavily"
    requires_api_key = True
    accepts_api_key = True
    env_var = "TAVILY_API_KEY"

    async def search(self, query: str, max_results: int = DEFAULT_RESULTS) -> SearchResponse:
        if not self.api_key:
            raise SearchBackendNotConfigured("The 'tavily' search backend requires an API key.")
        count = clamp_results(max_results)
        try:
            payload = await asyncio.wait_for(asyncio.to_thread(self._search, query, count), timeout=self.timeout)
        except asyncio.TimeoutError as exc:
            raise SearchBackendUnavailable(f"timed out after {self.timeout:.0f}s") from exc
        except Exception as exc:
            raise _map_sdk_error(exc) from exc

        rows = payload.get("results") or []
        results = [
            SearchResult(
                title=(row.get("title") or "").strip(),
                url=(row.get("url") or "").strip(),
                snippet=(row.get("content") or "").strip(),
            )
            for row in rows
            if row.get("url")
        ]
        answer = payload.get("answer") or None
        return SearchResponse(query=query, results=results, backend=self.name, answer=answer)

    def _search(self, query: str, count: int) -> dict:
        client = _new_client(api_key=self.api_key or "")
        return client.search(
            query,
            max_results=count,
            include_answer=True,
            timeout=self.timeout,
        )
