"""Pluggable web-search backends for the ``web_search`` tool.

To add a backend: create a module with a ``SearchBackend`` subclass, import it below,
and add it to ``_BACKENDS``. The TUI Select options and config validation derive from
this registry, so no other wiring is needed.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type

from .base import DEFAULT_RESULTS, DEFAULT_TIMEOUT_SECONDS, SearchBackend, clamp_results
from .duckduckgo import DuckDuckGoBackend
from .errors import (
    SearchBackendError,
    SearchBackendNotConfigured,
    SearchBackendRateLimited,
    SearchBackendUnavailable,
)
from .firecrawl import FirecrawlBackend
from .models import SearchResponse, SearchResult
from .searxng import SearxngBackend
from .tavily import TavilyBackend

DEFAULT_BACKEND = "duckduckgo"

_BACKENDS: Dict[str, Type[SearchBackend]] = {
    backend.name: backend for backend in (DuckDuckGoBackend, FirecrawlBackend, TavilyBackend, SearxngBackend)
}


def backend_names() -> List[str]:
    """All registered backend names."""
    return list(_BACKENDS)


def available_backends() -> List[Tuple[str, str]]:
    """``(label, name)`` pairs for the TUI Select, with the default backend first."""
    ordered = [_BACKENDS[DEFAULT_BACKEND]] + [backend for name, backend in _BACKENDS.items() if name != DEFAULT_BACKEND]
    return [(backend.label, backend.name) for backend in ordered]


def get_backend_class(name: str) -> Type[SearchBackend]:
    """Look up a backend class by name, or raise SearchBackendError for an unknown name."""
    try:
        return _BACKENDS[name]
    except KeyError as exc:
        valid = ", ".join(_BACKENDS)
        raise SearchBackendError(f"Unknown web_search backend '{name}'. Valid: {valid}.") from exc


def build_search_backend(
    name: Optional[str],
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> SearchBackend:
    """Construct the configured backend instance."""
    backend_cls = get_backend_class(name or DEFAULT_BACKEND)
    return backend_cls(api_key=api_key, base_url=base_url, timeout=timeout)


__all__ = [
    "DEFAULT_BACKEND",
    "DEFAULT_RESULTS",
    "DEFAULT_TIMEOUT_SECONDS",
    "SearchBackend",
    "SearchBackendError",
    "SearchBackendNotConfigured",
    "SearchBackendRateLimited",
    "SearchBackendUnavailable",
    "SearchResponse",
    "SearchResult",
    "available_backends",
    "backend_names",
    "build_search_backend",
    "clamp_results",
    "get_backend_class",
]
