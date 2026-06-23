"""Tavily cloud backend (requires an API key), via the ``tavily-python`` SDK."""

from __future__ import annotations

import asyncio

from tavily import TavilyClient
from tavily.errors import (
    ForbiddenError,
    InvalidAPIKeyError,
    MissingAPIKeyError,
    UsageLimitExceededError,
)

from .base import DEFAULT_RESULTS, SearchBackend, clamp_results
from .errors import (
    SearchBackendNotConfigured,
    SearchBackendRateLimited,
    SearchBackendUnavailable,
)
from .models import SearchResponse, SearchResult


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
        except (InvalidAPIKeyError, MissingAPIKeyError, ForbiddenError) as exc:
            raise SearchBackendNotConfigured(str(exc) or "Tavily rejected the API key.") from exc
        except UsageLimitExceededError as exc:
            raise SearchBackendRateLimited(str(exc) or "Tavily usage limit exceeded.") from exc
        except Exception as exc:
            raise SearchBackendUnavailable(str(exc) or exc.__class__.__name__) from exc

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
        client = TavilyClient(api_key=self.api_key)
        return client.search(
            query,
            max_results=count,
            include_answer=True,
            timeout=self.timeout,
        )
