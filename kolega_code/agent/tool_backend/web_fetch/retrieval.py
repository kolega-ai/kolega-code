"""Bounded, retrying HTTP retrieval for ``web_fetch``."""

from __future__ import annotations

import asyncio
import random
import re
import time
from dataclasses import dataclass, field
from email.message import Message as EmailMessage
from typing import Callable, Optional
from urllib.parse import urlsplit, urlunsplit

import httpx

TEXT_MAX_BYTES = 10 * 1024 * 1024
DOCUMENT_MAX_BYTES = 50 * 1024 * 1024
OVERALL_TIMEOUT_SECONDS = 30.0
MAX_ATTEMPTS = 3
MAX_REDIRECTS = 10
MAX_RETRY_AFTER_SECONDS = 5.0

_RETRYABLE_STATUSES = {408, 425, 429, 500, 502, 503, 504}
_DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".xls"}
_DOCUMENT_MIMES = {
    "application/pdf",
    "application/msword",
    "application/vnd.ms-excel",
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}
_USER_AGENTS = (
    "Mozilla/5.0 (compatible; KolegaCode/1.0; +https://github.com/kolega-ai/kolega-code)",
    "curl/8.0",
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"),
)
_BOT_MARKERS = (
    "cloudflare",
    "captcha",
    "challenge-platform",
    "access denied",
    "bot detection",
    "verify you are human",
)


class RetrievalError(RuntimeError):
    """A normal, user-visible failure to retrieve a URL."""


@dataclass(frozen=True)
class FetchAttempt:
    attempt: int
    user_agent: str
    elapsed_seconds: float
    status_code: Optional[int] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class FetchedResource:
    requested_url: str
    final_url: str
    status_code: int
    content_type: str
    charset: Optional[str]
    content_disposition: Optional[str]
    body: bytes
    attempts: tuple[FetchAttempt, ...] = field(default_factory=tuple)

    @property
    def byte_count(self) -> int:
        return len(self.body)


def normalize_url(value: str) -> str:
    """Normalize common model-produced URL variants and require HTTP(S)."""
    candidate = (value or "").strip()
    if candidate.startswith("<") and candidate.endswith(">"):
        candidate = candidate[1:-1].strip()
    candidate = re.sub(r"^(https?):/(?!/)", r"\1://", candidate, flags=re.IGNORECASE)
    if "://" not in candidate:
        candidate = f"https://{candidate}"

    parsed = urlsplit(candidate)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise RetrievalError("Provide a valid http(s) URL with a hostname.")
    if parsed.username or parsed.password:
        raise RetrievalError("URLs containing embedded credentials are not supported.")
    try:
        port = parsed.port
    except ValueError as exc:
        raise RetrievalError(f"Invalid URL port: {exc}") from exc

    hostname = parsed.hostname.encode("idna").decode("ascii")
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    if port is not None:
        netloc += f":{port}"
    return urlunsplit((parsed.scheme.lower(), netloc, parsed.path or "/", parsed.query, ""))


def _content_type_parts(value: str) -> tuple[str, Optional[str]]:
    message = EmailMessage()
    message["content-type"] = value or "application/octet-stream"
    mime = message.get_content_type().lower()
    charset = message.get_param("charset", header="content-type")
    return mime, str(charset) if charset else None


def _filename_from_disposition(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    message = EmailMessage()
    message["content-disposition"] = value
    filename = message.get_filename()
    return str(filename) if filename else None


def _looks_like_document(url: str, mime: str, content_disposition: Optional[str]) -> bool:
    if mime in _DOCUMENT_MIMES:
        return True
    path = urlsplit(url).path.lower()
    filename = (_filename_from_disposition(content_disposition) or "").lower()
    return any(path.endswith(ext) or filename.endswith(ext) for ext in _DOCUMENT_EXTENSIONS)


def _max_bytes(url: str, mime: str, content_disposition: Optional[str]) -> int:
    return DOCUMENT_MAX_BYTES if _looks_like_document(url, mime, content_disposition) else TEXT_MAX_BYTES


def _retry_after_seconds(value: Optional[str]) -> float:
    if not value:
        return 0.5
    try:
        seconds = float(value)
    except ValueError:
        from email.utils import parsedate_to_datetime
        from datetime import datetime, timezone

        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            seconds = (retry_at - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            seconds = 0.5
    return min(MAX_RETRY_AFTER_SECONDS, max(0.0, seconds))


def _is_bot_block(status: int, body: bytes) -> bool:
    if status not in {403, 503}:
        return False
    sample = body[:256_000].decode("utf-8", errors="ignore").lower()
    return any(marker in sample for marker in _BOT_MARKERS)


ClientFactory = Callable[..., httpx.AsyncClient]


class WebRetriever:
    """Fetch one URL while bounding retries, elapsed time, redirects, and bytes."""

    def __init__(self, client_factory: ClientFactory = httpx.AsyncClient) -> None:
        self._client_factory = client_factory

    async def fetch(self, url: str) -> FetchedResource:
        normalized = normalize_url(url)
        attempts: list[FetchAttempt] = []
        last_error = "unknown retrieval failure"
        timeout = httpx.Timeout(connect=5.0, read=15.0, write=5.0, pool=5.0)

        try:
            async with asyncio.timeout(OVERALL_TIMEOUT_SECONDS):
                async with self._client_factory(
                    timeout=timeout,
                    follow_redirects=True,
                    max_redirects=MAX_REDIRECTS,
                ) as client:
                    for attempt_number, user_agent in enumerate(_USER_AGENTS, start=1):
                        started = time.monotonic()
                        status: Optional[int] = None
                        try:
                            async with client.stream(
                                "GET",
                                normalized,
                                headers={
                                    "User-Agent": user_agent,
                                    "Accept": (
                                        "text/html,application/xhtml+xml,application/json,application/xml,"
                                        "text/plain,application/pdf,*/*;q=0.8"
                                    ),
                                    "Accept-Language": "en-US,en;q=0.7",
                                },
                            ) as response:
                                status = response.status_code
                                final_url = str(response.url)
                                parsed_final = urlsplit(final_url)
                                if parsed_final.scheme.lower() not in {"http", "https"}:
                                    raise RetrievalError("The URL redirected to an unsupported scheme.")

                                content_type_header = response.headers.get("content-type", "")
                                mime, charset = _content_type_parts(content_type_header)
                                disposition = response.headers.get("content-disposition")
                                limit = _max_bytes(final_url, mime, disposition)
                                content_length = response.headers.get("content-length")
                                if content_length:
                                    try:
                                        declared_size = int(content_length)
                                    except ValueError:
                                        declared_size = 0
                                    if declared_size > limit:
                                        raise RetrievalError(
                                            f"Response is too large ({declared_size:,} bytes; limit {limit:,} bytes)."
                                        )

                                chunks: list[bytes] = []
                                total = 0
                                async for chunk in response.aiter_bytes():
                                    total += len(chunk)
                                    if total > limit:
                                        raise RetrievalError(
                                            f"Response exceeded the {limit:,}-byte download limit while streaming."
                                        )
                                    chunks.append(chunk)
                                body = b"".join(chunks)

                                elapsed = time.monotonic() - started
                                attempts.append(FetchAttempt(attempt_number, user_agent, elapsed, status_code=status))

                                retryable = status in _RETRYABLE_STATUSES or _is_bot_block(status, body)
                                if retryable and attempt_number < MAX_ATTEMPTS:
                                    delay = _retry_after_seconds(response.headers.get("retry-after"))
                                    if status != 429:
                                        delay = min(MAX_RETRY_AFTER_SECONDS, delay * (2 ** (attempt_number - 1)))
                                    await asyncio.sleep(delay + random.uniform(0.0, 0.15))
                                    continue
                                if not response.is_success:
                                    raise RetrievalError(f"HTTP {status} while fetching {final_url}.")
                                if not body:
                                    raise RetrievalError(f"No content was returned by {final_url}.")

                                return FetchedResource(
                                    requested_url=normalized,
                                    final_url=final_url,
                                    status_code=status,
                                    content_type=mime,
                                    charset=charset,
                                    content_disposition=disposition,
                                    body=body,
                                    attempts=tuple(attempts),
                                )
                        except RetrievalError:
                            raise
                        except (httpx.TimeoutException, httpx.TransportError, httpx.TooManyRedirects) as exc:
                            elapsed = time.monotonic() - started
                            last_error = str(exc) or type(exc).__name__
                            attempts.append(
                                FetchAttempt(
                                    attempt_number,
                                    user_agent,
                                    elapsed,
                                    status_code=status,
                                    error=last_error,
                                )
                            )
                            if attempt_number >= MAX_ATTEMPTS:
                                break
                            await asyncio.sleep(min(2.0, 0.25 * (2 ** (attempt_number - 1))))
        except TimeoutError as exc:
            raise RetrievalError(f"Timed out fetching {normalized} after {OVERALL_TIMEOUT_SECONDS:g} seconds.") from exc

        raise RetrievalError(f"Failed to fetch {normalized}: {last_error}")
