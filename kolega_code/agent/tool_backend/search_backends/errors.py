"""Typed errors raised by web-search backends.

The WebSearchTool catches these and turns them into friendly ``Error: ...`` tool
results, so backends never leak raw SDK exceptions across the tool boundary.
"""

from __future__ import annotations


class SearchBackendError(Exception):
    """Base class for any recoverable web-search backend failure."""


class SearchBackendNotConfigured(SearchBackendError):
    """A required API key or base URL is missing (or the backend name is unknown)."""


class SearchBackendUnavailable(SearchBackendError):
    """Network failure, timeout, or a backend server error."""


class SearchBackendUnreachable(SearchBackendUnavailable):
    """The backend endpoint could not be reached at all (DNS, TCP, or TLS connect failure)."""

    def __init__(self, message: str, *, host: str | None = None) -> None:
        super().__init__(message)
        self.host = host


class SearchBackendRateLimited(SearchBackendError):
    """The backend rate-limited or blocked the request."""
