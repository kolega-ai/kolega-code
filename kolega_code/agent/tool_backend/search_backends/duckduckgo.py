"""Keyless DuckDuckGo backend (default), via the ``ddgs`` library."""

from __future__ import annotations

import asyncio
from typing import List

from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

from .base import DEFAULT_RESULTS, SearchBackend, clamp_results
from .errors import SearchBackendRateLimited, SearchBackendUnavailable
from .models import SearchResponse, SearchResult


class DuckDuckGoBackend(SearchBackend):
    """Search DuckDuckGo without an API key. ``ddgs`` is synchronous, so the call runs
    in a worker thread (mirroring how WebFetchTool wraps trafilatura)."""

    name = "duckduckgo"
    label = "DuckDuckGo (no key)"
    requires_api_key = False

    async def search(self, query: str, max_results: int = DEFAULT_RESULTS) -> SearchResponse:
        count = clamp_results(max_results)
        try:
            rows = await asyncio.wait_for(asyncio.to_thread(self._fetch, query, count), timeout=self.timeout)
        except (asyncio.TimeoutError, TimeoutException) as exc:
            raise SearchBackendUnavailable(f"timed out after {self.timeout:.0f}s") from exc
        except RatelimitException as exc:
            raise SearchBackendRateLimited(str(exc) or "rate-limited by DuckDuckGo") from exc
        except DDGSException as exc:
            raise SearchBackendUnavailable(str(exc) or exc.__class__.__name__) from exc
        except Exception as exc:  # defensive: ddgs has changed exception types across versions
            message = str(exc) or exc.__class__.__name__
            if "ratelimit" in exc.__class__.__name__.lower() or "202" in message or "429" in message:
                raise SearchBackendRateLimited(message) from exc
            raise SearchBackendUnavailable(message) from exc

        results = [
            SearchResult(
                title=(row.get("title") or "").strip(),
                url=(row.get("href") or row.get("url") or "").strip(),
                snippet=(row.get("body") or "").strip(),
            )
            for row in rows
            if row.get("href") or row.get("url")
        ]
        return SearchResponse(query=query, results=results, backend=self.name)

    @staticmethod
    def _fetch(query: str, count: int) -> List[dict]:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=count))
