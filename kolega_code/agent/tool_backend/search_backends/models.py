"""Result models shared by all web-search backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class SearchResult:
    """A single web-search hit."""

    title: str
    url: str
    snippet: str = ""
    # Fuller page text when a backend returns it (e.g. a scraped result). Empty otherwise.
    content: str = ""


@dataclass(frozen=True)
class SearchResponse:
    """The result of a single web search."""

    query: str
    results: List[SearchResult] = field(default_factory=list)
    backend: str = ""
    # Some backends (e.g. Tavily) return a direct synthesized answer.
    answer: Optional[str] = None
