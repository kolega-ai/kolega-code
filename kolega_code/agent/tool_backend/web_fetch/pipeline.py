"""Content-type-aware local retrieval and extraction orchestration."""

from __future__ import annotations

import asyncio
import json
import re
import time
import zipfile
from dataclasses import dataclass, field
from email.message import Message as EmailMessage
from io import BytesIO
from pathlib import PurePosixPath
from typing import Optional
from urllib.parse import urlsplit

from bs4 import UnicodeDammit

from .documents import DocumentConversionError, DocumentConverter, SUPPORTED_DOCUMENT_EXTENSIONS
from .extractors import DEFAULT_EXTRACTOR, ExtractorAttempt, extract_html
from .retrieval import FetchAttempt, FetchedResource, RetrievalError, WebRetriever


class WebContentError(RuntimeError):
    """A user-visible failure after retrieval/content dispatch."""


@dataclass(frozen=True)
class WebContent:
    requested_url: str
    final_url: str
    content: str
    content_type: str
    method: str
    warnings: tuple[str, ...] = field(default_factory=tuple)
    attempts: tuple[ExtractorAttempt, ...] = field(default_factory=tuple)
    byte_count: int = 0
    fetch_attempts: tuple[FetchAttempt, ...] = field(default_factory=tuple)
    retrieval_seconds: float = 0.0
    processing_seconds: float = 0.0


_DOCUMENT_MIME_TO_EXTENSION = {
    "application/pdf": ".pdf",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
}
_LEGACY_DOCUMENT_MIMES = {
    "application/msword": "DOC",
    "application/vnd.ms-powerpoint": "PPT",
}


def _disposition_filename(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    message = EmailMessage()
    message["content-disposition"] = value
    filename = message.get_filename()
    return str(filename) if filename else None


def _document_extension(resource: FetchedResource) -> str:
    if resource.body.startswith(b"%PDF-"):
        return ".pdf"
    mime_extension = _DOCUMENT_MIME_TO_EXTENSION.get(resource.content_type)
    if mime_extension:
        return mime_extension
    filename = _disposition_filename(resource.content_disposition)
    path = filename or urlsplit(resource.final_url).path
    extension = PurePosixPath(path).suffix.lower()
    if extension in SUPPORTED_DOCUMENT_EXTENSIONS:
        return extension
    if resource.body.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(BytesIO(resource.body)) as archive:
                names = set(archive.namelist())
            if "word/document.xml" in names:
                return ".docx"
            if "ppt/presentation.xml" in names:
                return ".pptx"
            if "xl/workbook.xml" in names:
                return ".xlsx"
        except (OSError, zipfile.BadZipFile):
            pass
    return ""


def _decode(resource: FetchedResource) -> str:
    encodings = [resource.charset] if resource.charset else None
    decoded = UnicodeDammit(resource.body, known_definite_encodings=encodings).unicode_markup
    return (decoded or resource.body.decode("utf-8", errors="replace")).replace("\x00", "")


def _looks_like_html(text: str) -> bool:
    prefix = text[:4096].lower()
    return bool(re.search(r"<(?:!doctype\s+html|html|head|body|main|article|div)\b", prefix))


def _looks_binary(body: bytes) -> bool:
    sample = body[:8192]
    if not sample:
        return False
    if b"\x00" in sample:
        return True
    control_bytes = sum(1 for value in sample if value < 9 or 13 < value < 32)
    return control_bytes / len(sample) > 0.08


class LocalWebContentPipeline:
    def __init__(
        self,
        retriever: Optional[WebRetriever] = None,
        document_converter: Optional[DocumentConverter] = None,
    ) -> None:
        self.retriever = retriever or WebRetriever()
        self.document_converter = document_converter or DocumentConverter()

    async def load(self, url: str, extractor_preference: str = DEFAULT_EXTRACTOR) -> WebContent:
        try:
            resource = await self.retriever.fetch(url)
        except RetrievalError as exc:
            raise WebContentError(str(exc)) from exc
        processing_started = time.monotonic()
        retrieval_seconds = sum(attempt.elapsed_seconds for attempt in resource.attempts)

        extension = _document_extension(resource)
        if extension:
            try:
                converted = await self.document_converter.convert(resource.body, extension, resource.final_url)
            except DocumentConversionError as exc:
                raise WebContentError(str(exc)) from exc
            warnings = (
                ("Document conversion output was truncated to 1,000,000 characters.",) if converted.truncated else ()
            )
            return WebContent(
                resource.requested_url,
                resource.final_url,
                converted.content,
                resource.content_type,
                converted.method,
                warnings,
                byte_count=resource.byte_count,
                fetch_attempts=resource.attempts,
                retrieval_seconds=retrieval_seconds,
                processing_seconds=time.monotonic() - processing_started,
            )

        path_suffix = PurePosixPath(urlsplit(resource.final_url).path).suffix.lower()
        legacy_name = _LEGACY_DOCUMENT_MIMES.get(resource.content_type)
        if path_suffix in {".doc", ".ppt"} or legacy_name:
            legacy_name = legacy_name or path_suffix[1:].upper()
            raise WebContentError(
                f"Legacy {legacy_name} documents are not supported; convert the file to a modern Office format first."
            )
        if _looks_binary(resource.body) and resource.content_type in {"", "application/octet-stream"}:
            raise WebContentError("The response is binary but its document format is unsupported or unrecognized.")

        text = await asyncio.to_thread(_decode, resource)
        mime = resource.content_type
        is_html = "html" in mime or "xhtml" in mime or _looks_like_html(text)
        if is_html:
            extracted = await extract_html(text, resource.final_url, extractor_preference)
            if not extracted.content.strip():
                if extracted.spa_detected:
                    warning = (
                        extracted.warnings[0] if extracted.warnings else "This page appears to require JavaScript."
                    )
                    raise WebContentError(warning)
                errors = "; ".join(
                    f"{attempt.name}: {attempt.error}" for attempt in extracted.attempts if attempt.error
                )
                suffix = f" ({errors})" if errors else ""
                raise WebContentError(f"No local HTML extractor produced usable content{suffix}.")
            return WebContent(
                resource.requested_url,
                resource.final_url,
                extracted.content,
                mime,
                extracted.method,
                extracted.warnings,
                extracted.attempts,
                resource.byte_count,
                resource.attempts,
                retrieval_seconds,
                time.monotonic() - processing_started,
            )

        if "json" in mime or text.lstrip().startswith(("{", "[")):
            try:
                text = json.dumps(json.loads(text), indent=2, ensure_ascii=False)
                method = "json"
            except json.JSONDecodeError:
                method = "text"
        elif any(token in mime for token in ("xml", "rss", "atom")):
            method = "xml"
        elif mime.startswith("text/") or not mime or mime == "application/octet-stream":
            method = "text"
        else:
            raise WebContentError(f"Unsupported content type: {mime or 'unknown'}.")

        cleaned = text.strip()
        if not cleaned:
            raise WebContentError("The fetched resource contained no readable text.")
        return WebContent(
            resource.requested_url,
            resource.final_url,
            cleaned,
            mime,
            method,
            byte_count=resource.byte_count,
            fetch_attempts=resource.attempts,
            retrieval_seconds=retrieval_seconds,
            processing_seconds=time.monotonic() - processing_started,
        )
