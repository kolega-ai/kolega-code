"""Bounded local document-to-Markdown conversion."""

from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass
from typing import Any, Optional

DOCUMENT_OUTPUT_MAX_CHARS = 1_000_000
SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".xls"}


class DocumentConversionError(RuntimeError):
    """A normal, user-visible document conversion failure."""


@dataclass(frozen=True)
class ConvertedDocument:
    content: str
    method: str
    truncated: bool = False


class DocumentConverter:
    def __init__(self) -> None:
        self._converter: Optional[Any] = None

    def _get_converter(self):
        if self._converter is None:
            from markitdown import MarkItDown

            self._converter = MarkItDown(enable_builtins=True, enable_plugins=False)
        return self._converter

    async def convert(self, body: bytes, extension: str, url: str) -> ConvertedDocument:
        if extension not in SUPPORTED_DOCUMENT_EXTENSIONS:
            raise DocumentConversionError(f"Unsupported document format: {extension or 'unknown'}.")

        def _convert() -> str:
            result = self._get_converter().convert_stream(io.BytesIO(body), file_extension=extension, url=url)
            return (result.text_content or "").strip()

        try:
            content = await asyncio.wait_for(asyncio.to_thread(_convert), timeout=30.0)
        except asyncio.TimeoutError as exc:
            raise DocumentConversionError("Document conversion timed out after 30 seconds.") from exc
        except Exception as exc:
            raise DocumentConversionError(f"Document conversion failed: {exc}") from exc

        if not content:
            if extension == ".pdf":
                raise DocumentConversionError(
                    "The PDF contains no extractable text and may be scanned or image-only; OCR is not supported."
                )
            raise DocumentConversionError("The document contained no extractable text.")
        truncated = len(content) > DOCUMENT_OUTPUT_MAX_CHARS
        return ConvertedDocument(content[:DOCUMENT_OUTPUT_MAX_CHARS], f"markitdown:{extension[1:]}", truncated)
