"""Keyless self-hosted SearXNG backend, via its JSON search API over httpx."""

from __future__ import annotations

import httpx

from .base import DEFAULT_RESULTS, SearchBackend, clamp_results
from .errors import (
    SearchBackendNotConfigured,
    SearchBackendRateLimited,
    SearchBackendUnavailable,
)
from .models import SearchResponse, SearchResult


class SearxngBackend(SearchBackend):
    """Query a self-hosted SearXNG instance's ``/search?format=json`` endpoint.

    Keyless, but requires the instance's base URL. ``format=json`` must be enabled in
    the instance's ``settings.yml`` (``search.formats``)."""

    name = "searxng"
    label = "SearXNG (self-hosted)"
    requires_api_key = False
    requires_base_url = True

    async def search(self, query: str, max_results: int = DEFAULT_RESULTS) -> SearchResponse:
        if not self.base_url:
            raise SearchBackendNotConfigured("The 'searxng' search backend requires a base URL.")
        count = clamp_results(max_results)
        url = self.base_url.rstrip("/") + "/search"
        params = {"q": query, "format": "json"}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=params)
        except httpx.TimeoutException as exc:
            raise SearchBackendUnavailable(f"timed out after {self.timeout:.0f}s") from exc
        except httpx.HTTPError as exc:
            raise SearchBackendUnavailable(str(exc) or exc.__class__.__name__) from exc

        if response.status_code == 429:
            raise SearchBackendRateLimited("SearXNG rate-limited the request (HTTP 429).")
        if response.status_code >= 400:
            raise SearchBackendUnavailable(f"SearXNG returned HTTP {response.status_code}.")
        try:
            payload = response.json()
        except ValueError as exc:
            raise SearchBackendUnavailable(
                "SearXNG returned a non-JSON response (is the JSON format enabled?)."
            ) from exc

        rows = payload.get("results") or []
        results = [
            SearchResult(
                title=(row.get("title") or "").strip(),
                url=(row.get("url") or "").strip(),
                snippet=(row.get("content") or "").strip(),
            )
            for row in rows[:count]
            if row.get("url")
        ]
        return SearchResponse(query=query, results=results, backend=self.name)
