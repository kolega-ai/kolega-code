"""Firecrawl backend.

Firecrawl's search works keyless (its free tier — 1,000 credits/month, no
``Authorization`` header) and also with an API key for higher rate limits. We honor both:
keyless requests go straight to the REST endpoint via httpx; when a key is configured we
use the official ``firecrawl-py`` SDK (which requires a key) for the keyed path.
"""

from __future__ import annotations

import asyncio

import httpx
from firecrawl import Firecrawl

from .base import DEFAULT_RESULTS, SearchBackend, clamp_results
from .errors import (
    SearchBackendError,
    SearchBackendNotConfigured,
    SearchBackendRateLimited,
    SearchBackendUnavailable,
)
from .models import SearchResponse, SearchResult

# Same REST endpoint the SDK posts to; keyless requests simply omit the Authorization header.
FIRECRAWL_SEARCH_URL = "https://api.firecrawl.dev/v2/search"


def _field(item, key: str):
    """Read a field from a Firecrawl result that may be a pydantic model (SDK) or dict (REST)."""
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


class FirecrawlBackend(SearchBackend):
    """Search via Firecrawl. Keyless by default (direct REST); uses the official SDK for
    the keyed path when an API key is configured (higher rate limits)."""

    name = "firecrawl"
    label = "Firecrawl"
    requires_api_key = False
    accepts_api_key = True
    env_var = "FIRECRAWL_API_KEY"

    async def search(self, query: str, max_results: int = DEFAULT_RESULTS) -> SearchResponse:
        count = clamp_results(max_results)
        web = await (self._search_keyed(query, count) if self.api_key else self._search_keyless(query, count))

        results = []
        for item in web:
            url = _field(item, "url")
            if not url:
                continue
            results.append(
                SearchResult(
                    title=(_field(item, "title") or "").strip(),
                    url=str(url).strip(),
                    snippet=(_field(item, "description") or "").strip(),
                    content=(_field(item, "markdown") or "").strip(),
                )
            )
        return SearchResponse(query=query, results=results, backend=self.name)

    async def _search_keyed(self, query: str, count: int) -> list:
        """Keyed path: official SDK (sync) in a worker thread. Firecrawl timeout is ms."""
        try:
            data = await asyncio.wait_for(
                asyncio.to_thread(self._sdk_search, query, count), timeout=self.timeout
            )
        except asyncio.TimeoutError as exc:
            raise SearchBackendUnavailable(f"timed out after {self.timeout:.0f}s") from exc
        except Exception as exc:
            raise self._map_sdk_error(exc)
        return _field(data, "web") or []

    def _sdk_search(self, query: str, count: int):
        client = Firecrawl(api_key=self.api_key, timeout=self.timeout)
        return client.search(query, limit=count, timeout=int(self.timeout * 1000))

    async def _search_keyless(self, query: str, count: int) -> list:
        """Keyless path: Firecrawl's free tier — POST /v2/search with no Authorization header."""
        body = {"query": query, "limit": count}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    FIRECRAWL_SEARCH_URL, json=body, headers={"Content-Type": "application/json"}
                )
        except httpx.TimeoutException as exc:
            raise SearchBackendUnavailable(f"timed out after {self.timeout:.0f}s") from exc
        except httpx.HTTPError as exc:
            raise SearchBackendUnavailable(str(exc) or exc.__class__.__name__) from exc

        if response.status_code == 429:
            raise SearchBackendRateLimited(
                "Firecrawl keyless rate limit reached; add a Firecrawl API key in "
                "Settings > Web Search for higher limits."
            )
        if response.status_code >= 400:
            raise SearchBackendUnavailable(f"Firecrawl returned HTTP {response.status_code}.")
        try:
            payload = response.json()
        except ValueError as exc:
            raise SearchBackendUnavailable("Firecrawl returned a non-JSON response.") from exc
        if payload.get("success") is False:
            raise SearchBackendUnavailable("Firecrawl search was unsuccessful.")
        return (payload.get("data") or {}).get("web") or []

    @staticmethod
    def _map_sdk_error(exc: Exception) -> SearchBackendError:
        message = str(exc) or exc.__class__.__name__
        low = message.lower()
        if "429" in message or "rate limit" in low or "ratelimit" in exc.__class__.__name__.lower():
            return SearchBackendRateLimited(message)
        if "401" in message or "403" in message or "unauthorized" in low or "forbidden" in low or "api key" in low:
            return SearchBackendNotConfigured(message)
        return SearchBackendUnavailable(message)
