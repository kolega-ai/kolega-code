"""Quality-gated local HTML extractors."""

from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass, field
from typing import Protocol

import trafilatura
from bs4 import BeautifulSoup
from markdownify import markdownify
from readability import Document

DEFAULT_EXTRACTOR = "auto"
HIGH_QUALITY_SCORE = 42.0

_JS_MARKERS = (
    "enable javascript",
    "javascript is required",
    "javascript required",
    "please turn javascript on",
    "you need to enable javascript",
    "checking your browser",
    "verify you are human",
    "challenge-platform",
)
_APP_ROOT_PATTERNS = (
    r'<(?:div|main)[^>]+id=["\'](?:app|root|__next|__nuxt)["\'][^>]*>\s*</(?:div|main)>',
    r"<script[^>]+(?:webpack|vite|next|nuxt|react|angular)",
)


class HtmlExtractor(Protocol):
    name: str
    label: str

    def extract(self, html: str, url: str) -> str:
        """Return extracted Markdown/text, or an empty string."""
        ...


@dataclass(frozen=True)
class ExtractorAttempt:
    name: str
    score: float
    content_length: int
    usable: bool
    error: str | None = None


@dataclass(frozen=True)
class ExtractedPage:
    content: str
    method: str
    score: float
    spa_detected: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)
    attempts: tuple[ExtractorAttempt, ...] = field(default_factory=tuple)


def _clean_markdown(value: str) -> str:
    lines = [line.rstrip() for line in (value or "").replace("\x00", "").splitlines()]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{4,}", "\n\n\n", cleaned)
    return cleaned.strip()


def _normalized_lines(content: str) -> list[str]:
    return [re.sub(r"\s+", " ", line).strip() for line in content.splitlines() if line.strip()]


def quality_score(content: str, html: str) -> tuple[float, bool]:
    """Score usefulness without making short-but-legitimate pages impossible."""
    cleaned = _clean_markdown(content)
    if not cleaned:
        return -100.0, False
    lower = cleaned.lower()
    lines = _normalized_lines(cleaned)
    chars = len(cleaned)

    score = min(45.0, math.log2(max(chars, 2)) * 4.0)
    score += min(12.0, sum(1 for line in lines if line.startswith("#")) * 1.5)
    score += min(8.0, sum(1 for line in lines if len(line) >= 80) * 0.5)

    if any(marker in lower for marker in _JS_MARKERS) and chars < 2_000:
        score -= 45.0
    if len(lines) > 10:
        short_ratio = sum(1 for line in lines if len(line) < 40) / len(lines)
        if short_ratio > 0.72:
            score -= 18.0
    if lines:
        duplicate_ratio = 1.0 - (len(set(lines)) / len(lines))
        score -= min(18.0, duplicate_ratio * 30.0)
    link_count = len(re.findall(r"\[[^\]]+\]\([^)]+\)", cleaned))
    word_count = max(1, len(cleaned.split()))
    if link_count / word_count > 0.18:
        score -= 12.0

    raw_visible = BeautifulSoup(html, "lxml").get_text(" ", strip=True)
    if chars < 80 and len(raw_visible) > 500:
        score -= 20.0
    elif chars < 80 and len(raw_visible) <= 500:
        score += 8.0

    usable = chars >= 40 and score >= 8.0
    return round(score, 2), usable


class TrafilaturaExtractor:
    name = "trafilatura"
    label = "Trafilatura"

    def extract(self, html: str, url: str) -> str:
        balanced = (
            trafilatura.extract(
                html,
                url=url,
                include_comments=False,
                include_tables=True,
                include_links=True,
                include_formatting=True,
                output_format="markdown",
            )
            or ""
        )
        if len(balanced.strip()) >= 1_000:
            return _clean_markdown(balanced)
        recall = (
            trafilatura.extract(
                html,
                url=url,
                favor_recall=True,
                include_comments=False,
                include_tables=True,
                include_links=True,
                include_formatting=True,
                output_format="markdown",
            )
            or ""
        )
        if not balanced.strip() or len(recall) >= len(balanced) * 1.35:
            return _clean_markdown(recall)
        return _clean_markdown(balanced)


class ReadabilityExtractor:
    name = "readability"
    label = "Readability"

    def extract(self, html: str, url: str) -> str:
        article_html = Document(html, url=url).summary(html_partial=True)
        return _clean_markdown(markdownify(article_html, heading_style="ATX", bullets="-"))


class SemanticDomExtractor:
    name = "semantic_dom"
    label = "Semantic DOM"

    def extract(self, html: str, url: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup.select("script, style, noscript, template, svg, nav, header, footer, aside"):
            tag.decompose()
        node = None
        for selector in ("[data-pagefind-body]", "main article", "article", "main", "[role='main']", "body"):
            candidate = soup.select_one(selector)
            if candidate and candidate.get_text(" ", strip=True):
                node = candidate
                break
        if node is None:
            return ""
        return _clean_markdown(markdownify(str(node), heading_style="ATX", bullets="-"))


class FullTextExtractor:
    name = "full_text"
    label = "Full visible text"

    def extract(self, html: str, url: str) -> str:
        del url
        soup = BeautifulSoup(html, "lxml")
        for tag in soup.select("script, style, noscript, template, svg"):
            tag.decompose()
        return _clean_markdown(soup.get_text("\n", strip=True))


_EXTRACTORS: dict[str, HtmlExtractor] = {}
for _extractor in (
    TrafilaturaExtractor(),
    ReadabilityExtractor(),
    SemanticDomExtractor(),
    FullTextExtractor(),
):
    _EXTRACTORS[_extractor.name] = _extractor


def extractor_names() -> list[str]:
    """Return internal extractor names for diagnostics and benchmarks."""
    return [DEFAULT_EXTRACTOR, *_EXTRACTORS]


def _looks_like_spa(html: str, attempts: list[ExtractorAttempt]) -> bool:
    lower = html.lower()
    marker = any(marker in lower for marker in _JS_MARKERS)
    app_shell = any(re.search(pattern, html, re.IGNORECASE) for pattern in _APP_ROOT_PATTERNS)
    scripts = len(re.findall(r"<script\b", html, re.IGNORECASE))
    visible = BeautifulSoup(html, "lxml").get_text(" ", strip=True)
    unusable = not attempts or all(not attempt.usable for attempt in attempts)
    return marker or (unusable and app_shell and (scripts >= 3 or len(visible) < 300))


async def extract_html(html: str, url: str, preference: str = DEFAULT_EXTRACTOR) -> ExtractedPage:
    if preference not in extractor_names():
        preference = DEFAULT_EXTRACTOR
    default_order = list(_EXTRACTORS)
    order = (
        default_order
        if preference == DEFAULT_EXTRACTOR
        else [preference, *[x for x in default_order if x != preference]]
    )

    candidates: list[tuple[str, str, float, bool]] = []
    attempts: list[ExtractorAttempt] = []
    for name in order:
        extractor = _EXTRACTORS[name]
        try:
            content = await asyncio.wait_for(asyncio.to_thread(extractor.extract, html, url), timeout=10.0)
            score, usable = quality_score(content, html)
            attempts.append(ExtractorAttempt(name, score, len(content), usable))
            candidates.append((name, content, score, usable))
            if usable and score >= HIGH_QUALITY_SCORE:
                break
        except Exception as exc:
            attempts.append(ExtractorAttempt(name, -100.0, 0, False, f"{type(exc).__name__}: {exc}"))

    usable_candidates = [candidate for candidate in candidates if candidate[3]]
    selected = max(usable_candidates or candidates, key=lambda item: item[2], default=None)
    spa_detected = _looks_like_spa(html, attempts)
    warnings: list[str] = []
    if spa_detected:
        warnings.append(
            "This page appears to require JavaScript; web_fetch does not run a browser, so content may be incomplete."
        )
    if selected is None or not selected[1].strip():
        return ExtractedPage("", "none", -100.0, spa_detected, tuple(warnings), tuple(attempts))
    return ExtractedPage(
        content=selected[1],
        method=selected[0],
        score=selected[2],
        spa_detected=spa_detected,
        warnings=tuple(warnings),
        attempts=tuple(attempts),
    )
