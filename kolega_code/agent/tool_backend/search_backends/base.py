"""The SearchBackend abstraction and shared helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Optional

from .models import SearchResponse

DEFAULT_TIMEOUT_SECONDS = 15.0
MIN_RESULTS = 1
MAX_RESULTS = 10
DEFAULT_RESULTS = 5


def clamp_results(max_results: int) -> int:
    """Clamp a requested result count into a sane range to bound latency/cost."""
    try:
        value = int(max_results)
    except (TypeError, ValueError):
        return DEFAULT_RESULTS
    return max(MIN_RESULTS, min(MAX_RESULTS, value))


class SearchBackend(ABC):
    """A pluggable web-search backend.

    Subclasses set the class-level metadata (``name``/``label``/``requires_api_key``/
    ``requires_base_url``/``env_var``) which drives both the TUI Select options and the
    Settings reveal logic, and implement the async ``search`` method.
    """

    name: ClassVar[str]
    label: ClassVar[str]
    # requires_api_key: a key is mandatory (the backend errors without one).
    # accepts_api_key: a key is used when present (optionally, for higher rate limits),
    #   but the backend also works keyless. A required key implies it is also accepted.
    requires_api_key: ClassVar[bool] = False
    accepts_api_key: ClassVar[bool] = False
    requires_base_url: ClassVar[bool] = False
    # Environment variable that supplies this backend's key, e.g. "TAVILY_API_KEY".
    env_var: ClassVar[Optional[str]] = None

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout

    @abstractmethod
    async def search(self, query: str, max_results: int = DEFAULT_RESULTS) -> SearchResponse:
        """Run a search and return a SearchResponse. Must not block the event loop."""
        raise NotImplementedError
