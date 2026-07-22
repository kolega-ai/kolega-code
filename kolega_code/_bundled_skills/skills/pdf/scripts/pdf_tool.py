#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bounded, non-interactive PDF operations with JSON I/O."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html as html_lib
import importlib.metadata
import io
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Generator, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any, NoReturn

import pdfplumber
from PIL import Image, ImageChops, ImageDraw
from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.errors import PdfReadError
from pypdf.generic import (
    ArrayObject,
    ByteStringObject,
    ContentStream,
    FloatObject,
    IndirectObject,
    NameObject,
    TextStringObject,
)
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A3, A4, A5, LEGAL, LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    Image as FlowImage,
)
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

SCHEMA_VERSION = 1
TOOL_VERSION = "1.1.0"
MAX_SOURCE_BYTES = 512 * 1024 * 1024
MAX_PAGES = 500
MAX_EXTRACTED_CHARS = 20_000_000
MAX_IMAGES = 2_000
MAX_TABLES = 2_000
MAX_PIXELS_PER_RENDER = 40_000_000
MAX_TOTAL_RENDER_PIXELS = 200_000_000
MAX_IMAGE_PIXELS = 40_000_000
MAX_IMAGE_BYTES = 100 * 1024 * 1024
MAX_JSON_BYTES = 20 * 1024 * 1024
MAX_STORY_ITEMS = 10_000
MAX_OUTPUTS = 500
DEFAULT_RENDER_DPI = 150
DEFAULT_VISUAL_CHECK_DPI = 96
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_FONT_ENTRIES = 2_000
MAX_FONT_OBJECT_VISITS = 10_000
MAX_FONT_XOBJECT_DEPTH = 8
MAX_CONTENT_OPERATIONS = 500_000
MAX_REDACTION_TARGETS = 1_000
BASE14_FONT_NAMES = frozenset(
    {
        "Courier",
        "Courier-Bold",
        "Courier-Oblique",
        "Courier-BoldOblique",
        "Helvetica",
        "Helvetica-Bold",
        "Helvetica-Oblique",
        "Helvetica-BoldOblique",
        "Times-Roman",
        "Times-Bold",
        "Times-Italic",
        "Times-BoldItalic",
        "Symbol",
        "ZapfDingbats",
    }
)
BASE14_ALIAS_FAMILIES = ("arial", "timesnewroman", "couriernew")
BASE14_ALIAS_SUFFIXES = frozenset(
    {
        "",
        "mt",
        "ps",
        "psmt",
        "bold",
        "boldmt",
        "boldps",
        "boldpsmt",
        "italic",
        "italicmt",
        "oblique",
        "bolditalic",
        "bolditalicmt",
        "bolditalicps",
        "bolditalicpsmt",
        "boldoblique",
    }
)
SUBSET_PREFIX_PATTERN = re.compile(r"^[A-Z]{6}\+")
REVIEW_REQUIRED = "REVIEW-REQUIRED"
# Provenance below matches reviewed artifacts recorded in references/ocr.md; a manifest
# derive fills these fields only on an exact SHA-256 match against a local file.
KNOWN_ARTIFACTS = {
    "1f18abe17b1ed8b4e47ee9b1ad0e274c93daf5efbb6b29a04ff1712e37051e05": {
        "identifier": "datalab-to/surya-ocr-2-gguf/surya-2.gguf",
        "revision": "6a3a4c30e5e74446d4f8b6afd05b2f2da970f470",
        "source": "https://huggingface.co/datalab-to/surya-ocr-2-gguf",
        "license": "modified AI Pubs Open Rail-M",
    },
    "98c0563673b1657ff6d021d1e5f04af06cbf61bb40c63ac613e8bb71b42fb2c0": {
        "identifier": "datalab-to/surya-ocr-2-gguf/surya-2-mmproj.gguf",
        "revision": "6a3a4c30e5e74446d4f8b6afd05b2f2da970f470",
        "source": "https://huggingface.co/datalab-to/surya-ocr-2-gguf",
        "license": "modified AI Pubs Open Rail-M",
    },
    "85218d2e3d98f5a21c58b4220627be923a97aee5db3cc71f39536ab31ac53960": {
        "identifier": "PaddlePaddle/PP-OCRv6_medium_det",
        "revision": "8e0f56fb2ef86b461d99cfc7ac5c137738985f61",
        "source": "https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det",
        "license": "Apache-2.0",
    },
    "1b01c79a914587933f615569e75de54f2e638ebb5d3f3b3c1b38c24ede8c7319": {
        "identifier": "PaddlePaddle/PP-OCRv6_medium_rec",
        "revision": "e5a92bcbc5cc1b494628e458d267778f0704fd7c",
        "source": "https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec",
        "license": "Apache-2.0",
    },
}
TESSDATA_SOURCES = {
    "tessdata_fast": {
        "identifier_prefix": "tesseract-ocr/tessdata_fast",
        "revision": "87416418657359cb625c412a48b6e1d6d41c29bd",
        "source": "https://github.com/tesseract-ocr/tessdata_fast",
        "license": "Apache-2.0",
    },
    "tessdata_best": {
        "identifier_prefix": "tesseract-ocr/tessdata_best",
        "revision": "e12c65a915945e4c28e237a9b52bc4a8f39a0cec",
        "source": "https://github.com/tesseract-ocr/tessdata_best",
        "license": "Apache-2.0",
    },
}
WELL_KNOWN_TESSDATA_DIRS = (
    "/opt/homebrew/share/tessdata",
    "/usr/local/share/tessdata",
    "/usr/share/tessdata",
    "/usr/share/tesseract-ocr/5/tessdata",
    "/usr/share/tesseract-ocr/4.00/tessdata",
)
PAGE_SIZES = {
    "a3": A3,
    "a4": A4,
    "a5": A5,
    "letter": LETTER,
    "legal": LEGAL,
}
EXIT_CODES = {
    "bad_input": 2,
    "unsupported_operation": 3,
    "missing_dependency": 4,
    "ambiguous_edit": 5,
    "resource_limit": 6,
    "license_precondition": 7,
    "validation_failed": 8,
    "engine_failed": 9,
    "internal_error": 10,
}
SUPPORTED_SURYA_MAJOR = 0
SUPPORTED_PADDLE_MAJOR = 3
SUPPORTED_TESSERACT_MAJOR = 5
PADDLE_STRUCTURE_ROLES = {
    "layout_detection_model_dir",
    "table_classification_model_dir",
    "wired_table_structure_recognition_model_dir",
    "wireless_table_structure_recognition_model_dir",
    "wired_table_cells_detection_model_dir",
    "wireless_table_cells_detection_model_dir",
}
PADDLE_BASE_ROLES = {
    "text_detection_model_dir",
    "text_recognition_model_dir",
}
SECRET_VALUES: set[str] = set()


class ToolError(Exception):
    """A categorized, user-actionable failure."""

    def __init__(self, category: str, message: str, *, details: Any = None):
        super().__init__(message)
        self.category = category
        self.message = message
        self.details = details


class JsonArgumentParser(argparse.ArgumentParser):
    """Make argparse failures conform to the stderr JSON contract."""

    def error(self, message: str) -> NoReturn:
        raise ToolError("bad_input", message)


@dataclass(frozen=True)
class SourceInfo:
    path: Path
    reader: PdfReader
    password_used: bool


def clean_text(value: Any) -> str:
    text = str(value)
    for secret in SECRET_VALUES:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if isinstance(value, dict):
        return {clean_text(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    try:
        return float(value)
    except (TypeError, ValueError):
        return clean_text(value)


def emit_stdout(payload: dict[str, Any]) -> None:
    print(json.dumps(json_safe(payload), ensure_ascii=False, sort_keys=True, allow_nan=False))


def emit_stderr(payload: dict[str, Any]) -> None:
    print(
        json.dumps(json_safe(payload), ensure_ascii=False, sort_keys=True, allow_nan=False),
        file=sys.stderr,
    )


def success(operation: str, **fields: Any) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "operation": operation,
        "tool_version": TOOL_VERSION,
        "libraries": library_versions(),
        **fields,
    }


def library_versions() -> dict[str, str]:
    versions = {}
    for package in ("pypdf", "pdfplumber", "pypdfium2", "reportlab", "Pillow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def read_json(path: Path) -> dict[str, Any]:
    require_regular_file(path)
    if path.suffix.lower() != ".json":
        raise ToolError("bad_input", f"JSON input must use a .json extension: {path}")
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ToolError("resource_limit", f"JSON input exceeds {MAX_JSON_BYTES} bytes")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"Non-finite JSON number: {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ToolError("bad_input", f"Cannot read JSON job: {clean_text(exc)}") from exc
    if not isinstance(value, dict):
        raise ToolError("bad_input", "JSON job root must be an object")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ToolError("bad_input", "JSON job requires schema_version 1")
    return value


def require_regular_file(path: Path) -> None:
    if not path.exists():
        raise ToolError("bad_input", f"Input does not exist: {path}")
    if path.is_symlink() or not path.is_file():
        raise ToolError("bad_input", f"Input must be a regular file: {path}")


def require_extension(path: Path, suffix: str, purpose: str) -> None:
    if path.suffix.lower() != suffix:
        raise ToolError("bad_input", f"{purpose} must use a {suffix} extension")


def finite_number(
    value: Any,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
    category: str = "bad_input",
) -> float:
    if isinstance(value, bool):
        raise ToolError(category, f"{field} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ToolError(category, f"{field} must be a finite number") from exc
    if not math.isfinite(number):
        raise ToolError(category, f"{field} must be a finite number")
    if minimum is not None and (number < minimum or (number == minimum and not minimum_inclusive)):
        comparison = "greater than or equal to" if minimum_inclusive else "greater than"
        raise ToolError(category, f"{field} must be {comparison} {minimum}")
    if maximum is not None and (number > maximum or (number == maximum and not maximum_inclusive)):
        comparison = "less than or equal to" if maximum_inclusive else "less than"
        raise ToolError(category, f"{field} must be {comparison} {maximum}")
    return number


def finite_integer(
    value: Any,
    field: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
    category: str = "bad_input",
) -> int:
    number = finite_number(
        value,
        field,
        minimum=minimum,
        maximum=maximum,
        category=category,
    )
    if not number.is_integer():
        raise ToolError(category, f"{field} must be an integer")
    return int(number)


def validate_rectangle(
    *,
    x: Any,
    y: Any,
    width: Any,
    height: Any,
    page_width: float,
    page_height: float,
    field: str,
) -> tuple[float, float, float, float]:
    left = finite_number(x, f"{field}.x", minimum=0)
    top = finite_number(y, f"{field}.y", minimum=0)
    box_width = finite_number(width, f"{field}.width", minimum=0, minimum_inclusive=False)
    box_height = finite_number(height, f"{field}.height", minimum=0, minimum_inclusive=False)
    if left + box_width > page_width or top + box_height > page_height:
        raise ToolError("bad_input", f"{field} rectangle must fit within the page geometry")
    return left, top, box_width, box_height


def parse_languages(value: str, *, field: str = "languages") -> list[str]:
    raw_values = value.split(",")
    languages = [item.strip() for item in raw_values]
    if not languages or any(not item for item in languages):
        raise ToolError("bad_input", f"{field} requires nonempty comma-separated values")
    invalid = [item for item in languages if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", item)]
    if invalid:
        raise ToolError(
            "bad_input",
            f"{field} contains invalid language identifiers: {', '.join(invalid)}",
        )
    if len(set(languages)) != len(languages):
        raise ToolError("bad_input", f"{field} must not contain duplicate language identifiers")
    return languages


def require_pdf_signature(path: Path) -> None:
    require_regular_file(path)
    if path.suffix.lower() != ".pdf":
        raise ToolError("bad_input", f"PDF input must use a .pdf extension: {path}")
    size = path.stat().st_size
    if size > MAX_SOURCE_BYTES:
        raise ToolError(
            "resource_limit",
            f"PDF exceeds the {MAX_SOURCE_BYTES}-byte source limit: {path}",
        )
    with path.open("rb") as handle:
        if b"%PDF-" not in handle.read(1024):
            raise ToolError("bad_input", f"Input does not have a PDF signature: {path}")


def resolve_password(direct: str | None, env_name: str | None) -> str | None:
    if direct and env_name:
        raise ToolError("bad_input", "Use either --password or --password-env, not both")
    value = direct
    if env_name:
        if env_name not in os.environ:
            raise ToolError("bad_input", f"Password environment variable is not set: {env_name}")
        value = os.environ[env_name]
    if value:
        SECRET_VALUES.add(value)
    return value


def open_pdf(path: Path, password: str | None = None) -> SourceInfo:
    require_pdf_signature(path)
    try:
        reader = PdfReader(str(path), strict=True)
    except (OSError, PdfReadError, ValueError) as exc:
        raise ToolError("bad_input", f"Cannot parse PDF {path}: {clean_text(exc)}") from exc
    password_used = False
    if reader.is_encrypted:
        if password is None:
            raise ToolError(
                "unsupported_operation",
                f"PDF is encrypted; provide an approved password for {path}",
            )
        try:
            result = reader.decrypt(password)
        except Exception as exc:
            raise ToolError("bad_input", f"Cannot decrypt PDF {path}") from exc
        if not result:
            raise ToolError("bad_input", f"Password did not decrypt PDF {path}")
        password_used = True
    if len(reader.pages) > MAX_PAGES:
        raise ToolError(
            "resource_limit",
            f"PDF has {len(reader.pages)} pages; limit is {MAX_PAGES}",
        )
    return SourceInfo(path=path, reader=reader, password_used=password_used)


def parse_page_selection(
    expression: str | None,
    page_count: int,
    *,
    allow_repeats: bool = False,
) -> list[int]:
    if page_count < 0:
        raise ToolError("bad_input", "Invalid page count")
    if expression is None or expression.strip().lower() in {"", "all"}:
        return list(range(page_count))
    selected: list[int] = []
    for raw_part in expression.split(","):
        part = raw_part.strip()
        if not part:
            raise ToolError("bad_input", f"Invalid page expression: {expression}")
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not match:
            raise ToolError("bad_input", f"Invalid page token: {part}")
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start < 1 or end < 1 or start > page_count or end > page_count:
            raise ToolError("bad_input", f"Page token is outside 1-{page_count}: {part}")
        step = 1 if end >= start else -1
        selected.extend(index - 1 for index in range(start, end + step, step))
    if not allow_repeats:
        seen: set[int] = set()
        selected = [page for page in selected if not (page in seen or seen.add(page))]
    if len(selected) > MAX_PAGES:
        raise ToolError("resource_limit", f"Page selection exceeds {MAX_PAGES} pages")
    return selected


def parse_rotation(value: Any) -> int:
    rotation = finite_integer(value, "Rotation")
    if rotation % 90:
        raise ToolError("bad_input", "Rotation must be a multiple of 90 degrees")
    return rotation % 360


def page_geometry(page: Any) -> dict[str, float | int]:
    box = page.mediabox
    return {
        "width": round(float(box.width), 3),
        "height": round(float(box.height), 3),
        "rotation": int(page.rotation or 0) % 360,
    }


def classify_page(
    text: str,
    words: Sequence[dict[str, Any]],
    images: Sequence[dict[str, Any]],
    page_width: float,
    page_height: float,
) -> tuple[str, list[str]]:
    stripped = re.sub(r"\s+", "", text or "")
    warnings: list[str] = []
    page_area = max(page_width * page_height, 1)
    image_area = 0.0
    for image in images:
        try:
            image_area += max(0.0, float(image["x1"]) - float(image["x0"])) * max(
                0.0, float(image["bottom"]) - float(image["top"])
            )
        except (KeyError, TypeError, ValueError):
            continue
    image_coverage = min(image_area / page_area, 1.0)
    if not stripped and not images:
        return "blank", warnings
    if images and image_coverage >= 0.8 and len(stripped) < 100:
        if stripped:
            warnings.append(
                "A dominant page image has only sparse digital text; routing as likely scanned."
            )
        return "likely-scanned", warnings
    if len(stripped) >= 20 and images:
        if image_coverage >= 0.5:
            warnings.append(
                "Digital text and large image content coexist; duplicate text is possible."
            )
        return "hybrid", warnings
    if len(stripped) >= 20:
        return "digital-text", warnings
    if images:
        if stripped:
            warnings.append("Sparse digital text accompanies image content.")
            return "hybrid", warnings
        return "likely-scanned", warnings
    if words or stripped:
        warnings.append("Page has sparse extractable text.")
        return "digital-text", warnings
    return "blank", warnings


def normalize_metadata(reader: PdfReader) -> dict[str, Any]:
    metadata = reader.metadata
    if not metadata:
        return {}
    return {str(key): json_safe(value) for key, value in metadata.items()}


def form_inventory(reader: PdfReader) -> dict[str, Any]:
    fields = reader.get_fields() or {}
    items = []
    for name, field in sorted(fields.items()):
        items.append(
            {
                "name": name,
                "field_type": json_safe(field.get("/FT")),
                "value": json_safe(field.get("/V")),
                "default_value": json_safe(field.get("/DV")),
                "flags": json_safe(field.get("/Ff")),
                "alternate_name": json_safe(field.get("/TU")),
            }
        )
    root = reader.trailer.get("/Root", {})
    acroform = root.get("/AcroForm") if hasattr(root, "get") else None
    has_xfa = bool(acroform and acroform.get_object().get("/XFA"))
    return {"count": len(items), "fields": items, "has_xfa": has_xfa}


_NORMALIZED_BASE14 = frozenset(
    re.sub(r"[^a-z0-9]", "", name.casefold()) for name in BASE14_FONT_NAMES
)


def is_base14_font_name(name: str) -> bool:
    if name in BASE14_FONT_NAMES:
        return True
    normalized = re.sub(r"[^a-z0-9]", "", name.casefold())
    if normalized in _NORMALIZED_BASE14:
        return True
    for family in BASE14_ALIAS_FAMILIES:
        if normalized.startswith(family) and normalized[len(family) :] in BASE14_ALIAS_SUFFIXES:
            return True
    return False


def font_record(font_object: Any) -> dict[str, Any]:
    font = font_object.get_object()
    subtype = str(font.get("/Subtype", "")).lstrip("/") or None
    base_font = font.get("/BaseFont")
    descendant_subtype = None
    descriptor = font.get("/FontDescriptor")
    if subtype == "Type0":
        descendants = font.get("/DescendantFonts")
        if descendants is not None:
            descendant_array = descendants.get_object()
            if descendant_array:
                descendant = descendant_array[0].get_object()
                descendant_value = descendant.get("/Subtype")
                if descendant_value is not None:
                    descendant_subtype = str(descendant_value).lstrip("/") or None
                descriptor = descendant.get("/FontDescriptor")
                if base_font is None:
                    base_font = descendant.get("/BaseFont")
    raw_name = str(base_font).lstrip("/") if base_font is not None else "(unnamed)"
    name = SUBSET_PREFIX_PATTERN.sub("", raw_name)
    subset = bool(SUBSET_PREFIX_PATTERN.match(raw_name))
    if subtype == "Type3":
        embedded = True
    else:
        descriptor_object = descriptor.get_object() if descriptor is not None else {}
        embedded = any(
            key in descriptor_object for key in ("/FontFile", "/FontFile2", "/FontFile3")
        )
    encoding = font.get("/Encoding")
    if encoding is None:
        encoding_name = None
    else:
        encoding_object = encoding.get_object() if hasattr(encoding, "get_object") else encoding
        if isinstance(encoding_object, dict):
            base_encoding = encoding_object.get("/BaseEncoding")
            encoding_name = str(base_encoding).lstrip("/") if base_encoding else "Custom"
        else:
            encoding_name = str(encoding_object).lstrip("/")
    return {
        "base_font": raw_name,
        "name": name,
        "subtype": subtype,
        "descendant_subtype": descendant_subtype,
        "embedded": embedded,
        "subset": subset,
        "encoding": encoding_name,
        "base14": is_base14_font_name(name),
    }


def collect_fonts(reader: PdfReader) -> tuple[dict[str, Any], list[str]]:
    """Inventory every font reachable from page, form, and annotation resources."""
    entries: dict[tuple[Any, ...], dict[str, Any]] = {}
    warnings: list[str] = []
    truncated = False
    visits = 0
    visited_references: set[tuple[int, int]] = set()

    def record(font_object: Any, page_number: int | None) -> None:
        nonlocal truncated
        try:
            item = font_record(font_object)
        except Exception as exc:
            warnings.append(f"Font inventory entry failed: {clean_text(exc)}")
            return
        key = (item["base_font"], item["subtype"], item["embedded"], item["encoding"])
        existing = entries.get(key)
        if existing is None:
            if len(entries) >= MAX_FONT_ENTRIES:
                truncated = True
                return
            item["pages"] = set()
            entries[key] = item
            existing = item
        if page_number is not None:
            existing["pages"].add(page_number)

    def charge_visit() -> bool:
        nonlocal visits, truncated
        visits += 1
        if visits > MAX_FONT_OBJECT_VISITS:
            truncated = True
            return False
        return True

    def walk_resources(resources: Any, page_number: int | None, depth: int) -> None:
        nonlocal truncated
        if resources is None:
            return
        if depth > MAX_FONT_XOBJECT_DEPTH:
            truncated = True
            return
        try:
            resources = resources.get_object()
        except Exception as exc:
            warnings.append(f"Font inventory resource lookup failed: {clean_text(exc)}")
            return
        if not isinstance(resources, dict):
            return
        fonts = resources.get("/Font")
        if fonts is not None:
            try:
                font_map = fonts.get_object()
                for font_reference in font_map.values():
                    if not charge_visit():
                        return
                    record(font_reference, page_number)
            except Exception as exc:
                warnings.append(f"Font inventory font map failed: {clean_text(exc)}")
        xobjects = resources.get("/XObject")
        if xobjects is None:
            return
        try:
            xobject_map = xobjects.get_object()
        except Exception as exc:
            warnings.append(f"Font inventory XObject lookup failed: {clean_text(exc)}")
            return
        if not isinstance(xobject_map, dict):
            return
        for xobject_reference in xobject_map.values():
            if not charge_visit():
                return
            if isinstance(xobject_reference, IndirectObject):
                reference_key = (xobject_reference.idnum, xobject_reference.generation)
                if reference_key in visited_references:
                    continue
                visited_references.add(reference_key)
            try:
                xobject = xobject_reference.get_object()
                if str(xobject.get("/Subtype", "")) == "/Form":
                    walk_resources(xobject.get("/Resources"), page_number, depth + 1)
            except Exception as exc:
                warnings.append(f"Font inventory XObject walk failed: {clean_text(exc)}")

    def walk_annotations(page: Any, page_number: int) -> None:
        annotations = page.get("/Annots")
        if annotations is None:
            return
        try:
            annotation_array = annotations.get_object()
        except Exception:
            return
        for annotation_reference in annotation_array:
            if not charge_visit():
                return
            try:
                annotation = annotation_reference.get_object()
                appearances = annotation.get("/AP")
                if appearances is None:
                    continue
                appearance_map = appearances.get_object()
                if not isinstance(appearance_map, dict):
                    continue
                for state in appearance_map.values():
                    state_object = state.get_object()
                    if isinstance(state_object, dict) and "/Resources" not in state_object:
                        for nested in state_object.values():
                            nested_object = nested.get_object()
                            if isinstance(nested_object, dict):
                                walk_resources(nested_object.get("/Resources"), page_number, 1)
                    elif isinstance(state_object, dict):
                        walk_resources(state_object.get("/Resources"), page_number, 1)
            except Exception as exc:
                warnings.append(f"Font inventory annotation walk failed: {clean_text(exc)}")

    try:
        for page_index, page in enumerate(reader.pages):
            walk_resources(page.get("/Resources"), page_index + 1, 0)
            walk_annotations(page, page_index + 1)
        root = reader.trailer.get("/Root", {})
        acroform = root.get("/AcroForm") if hasattr(root, "get") else None
        if acroform is not None:
            acroform_object = acroform.get_object()
            walk_resources(acroform_object.get("/DR"), None, 0)
    except Exception as exc:
        warnings.append(f"Font inventory walk failed: {clean_text(exc)}")
        truncated = True

    normalized_entries = []
    for item in sorted(entries.values(), key=lambda entry: (entry["name"], entry["base_font"])):
        pages = sorted(item.pop("pages"))
        item["pages"] = pages[:200]
        normalized_entries.append(item)
    unembedded = sorted({item["name"] for item in normalized_entries if not item["embedded"]})
    unembedded_non_base14 = sorted(
        {item["name"] for item in normalized_entries if not item["embedded"] and not item["base14"]}
    )
    inventory = {
        "scope": "document",
        "count": len(normalized_entries),
        "entries": normalized_entries,
        "unembedded": unembedded,
        "unembedded_non_base14": unembedded_non_base14,
        "truncated": truncated,
    }
    return inventory, warnings


def font_release_warnings(fonts: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if fonts["unembedded_non_base14"]:
        names = ", ".join(fonts["unembedded_non_base14"])
        warnings.append(
            "RELEASE BLOCKER: non-embedded fonts outside the standard base-14 set: "
            f"{names}. Other viewers will substitute different glyphs and metrics, which "
            "can change line breaks, spacing, and symbol coverage. Re-export with these "
            "fonts embedded or replace them before release."
        )
    unembedded_base14 = [
        name for name in fonts["unembedded"] if name not in fonts["unembedded_non_base14"]
    ]
    if unembedded_base14:
        warnings.append(
            f"Non-embedded standard fonts are in use ({', '.join(unembedded_base14)}); "
            "conforming viewers substitute metrics-compatible fonts, but appearance can "
            "vary slightly."
        )
    if fonts["truncated"]:
        warnings.append(
            "Font inventory truncated at resource-walk limits; the inventory may be incomplete."
        )
    return warnings


def plumber_page_data(
    page: Any,
    *,
    mode: str,
    include_words: bool,
    include_tables: bool,
) -> dict[str, Any]:
    kwargs = {"layout": mode == "layout"}
    try:
        text = page.extract_text(**kwargs) or ""
    except Exception as exc:
        text = ""
        text_error = clean_text(exc)
    else:
        text_error = None
    try:
        words = page.extract_words(
            keep_blank_chars=False,
            use_text_flow=mode == "layout",
        )
    except Exception as exc:
        words = []
        words_error = clean_text(exc)
    else:
        words_error = None
    images = [json_safe(image) for image in page.images[:MAX_IMAGES]]
    classification, warnings = classify_page(
        text,
        words,
        images,
        float(page.width),
        float(page.height),
    )
    result: dict[str, Any] = {
        "page": page.page_number,
        "width": round(float(page.width), 3),
        "height": round(float(page.height), 3),
        "classification": classification,
        "text": text,
        "text_character_count": len(text),
        "embedded_image_count": len(page.images),
        "images": images,
        "warnings": warnings,
    }
    if include_words:
        result["words"] = [json_safe(word) for word in words]
    if include_tables:
        try:
            tables = page.extract_tables()
        except Exception as exc:
            tables = []
            result["warnings"].append(f"Table extraction failed: {clean_text(exc)}")
        if len(tables) > MAX_TABLES:
            raise ToolError("resource_limit", f"Table count exceeds {MAX_TABLES}")
        result["tables"] = json_safe(tables)
        if tables:
            result["warnings"].append(
                "Table boundaries and cell order are heuristic; verify against the rendered page."
            )
    if text_error:
        result["warnings"].append(f"Text extraction failed: {text_error}")
    if words_error:
        result["warnings"].append(f"Word extraction failed: {words_error}")
    if mode == "layout":
        result["warnings"].append("Layout reading order is heuristic.")
    return result


def inspect_document(
    path: Path,
    password: str | None,
    pages_expression: str | None,
    *,
    mode: str,
    include_words: bool,
    include_tables: bool,
) -> dict[str, Any]:
    source = open_pdf(path, password)
    selected = parse_page_selection(pages_expression, len(source.reader.pages))
    page_results = []
    total_chars = 0
    total_images = 0
    total_tables = 0
    with pdfplumber.open(str(path), password=password) as plumber_pdf:
        for page_index in selected:
            result = plumber_page_data(
                plumber_pdf.pages[page_index],
                mode=mode,
                include_words=include_words,
                include_tables=include_tables,
            )
            total_chars += result["text_character_count"]
            total_images += result["embedded_image_count"]
            total_tables += len(result.get("tables", []))
            if total_chars > MAX_EXTRACTED_CHARS:
                raise ToolError(
                    "resource_limit",
                    f"Extracted text exceeds {MAX_EXTRACTED_CHARS} characters",
                )
            if total_images > MAX_IMAGES:
                raise ToolError(
                    "resource_limit",
                    f"Embedded image count exceeds {MAX_IMAGES}",
                )
            if total_tables > MAX_TABLES:
                raise ToolError("resource_limit", f"Table count exceeds {MAX_TABLES}")
            result["geometry"] = page_geometry(source.reader.pages[page_index])
            page_results.append(result)
    forms = form_inventory(source.reader)
    fonts, font_walk_warnings = collect_fonts(source.reader)
    warnings = []
    if forms["has_xfa"]:
        warnings.append("XFA is present and is not supported for editing.")
    if source.reader.is_encrypted:
        warnings.append("Encrypted source was opened; output encryption depends on the operation.")
    warnings.extend(font_walk_warnings)
    warnings.extend(font_release_warnings(fonts))
    return {
        "source": str(path.resolve()),
        "encrypted": bool(source.reader.is_encrypted),
        "password_used": source.password_used,
        "page_count": len(source.reader.pages),
        "selected_pages": [page + 1 for page in selected],
        "metadata": normalize_metadata(source.reader),
        "forms": forms,
        "fonts": fonts,
        "pages": page_results,
        "warnings": warnings,
    }


def output_payload(
    operation: str,
    data: dict[str, Any],
    output: Path | None,
    *,
    overwrite: bool = False,
    protected_sources: Iterable[Path] = (),
) -> dict[str, Any]:
    if output is None:
        return success(operation, result=data, output_path=None, verification={"valid": True})
    if output.suffix.lower() != ".json":
        raise ToolError("bad_input", "Inspect/extract sidecar must use a .json extension")
    write_json_atomic(
        output,
        data,
        overwrite=overwrite,
        protected_sources=protected_sources,
    )
    return success(
        operation,
        result_summary={
            "page_count": data.get("page_count"),
            "selected_pages": data.get("selected_pages"),
            "warning_count": len(data.get("warnings", [])),
        },
        output_path=str(output.resolve()),
        verification={"valid": True, "json_reopened": True},
    )


@contextmanager
def temporary_sibling(destination: Path, suffix: str | None = None) -> Generator[Path, None, None]:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=suffix or destination.suffix or ".tmp",
        dir=destination.parent,
    )
    os.close(fd)
    temp_path = Path(name)
    try:
        yield temp_path
    finally:
        temp_path.unlink(missing_ok=True)


def paths_alias(first: Path, second: Path) -> bool:
    if first.resolve() == second.resolve():
        return True
    try:
        return first.exists() and second.exists() and first.samefile(second)
    except OSError:
        return False


def check_destination(
    destination: Path,
    *,
    sources: Iterable[Path] = (),
    overwrite: bool,
) -> None:
    for source in sources:
        if paths_alias(destination, source):
            raise ToolError(
                "bad_input",
                "Destination must be distinct from every source, even with --overwrite",
            )
    if destination.exists() and not overwrite:
        raise ToolError("bad_input", f"Destination exists; pass --overwrite: {destination}")
    if destination.exists() and (destination.is_symlink() or not destination.is_file()):
        raise ToolError("bad_input", f"Destination must be a regular file: {destination}")


def atomic_publish(temp_path: Path, destination: Path) -> None:
    try:
        os.replace(temp_path, destination.resolve())
    except OSError as exc:
        raise ToolError("validation_failed", f"Atomic publish failed: {clean_text(exc)}") from exc


def check_directory_destination(
    output_dir: Path,
    *,
    overwrite: bool,
    exists_message: str,
    not_directory_message: str,
    nonempty_message: str,
) -> None:
    if output_dir.exists():
        if not overwrite:
            raise ToolError("bad_input", exists_message)
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise ToolError("bad_input", not_directory_message)
        if any(output_dir.iterdir()):
            raise ToolError("bad_input", nonempty_message)


def stage_directory(output_dir: Path) -> Path:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent.resolve()))


def atomic_publish_directory(staged_dir: Path, output_dir: Path, *, label: str) -> None:
    try:
        os.replace(staged_dir, output_dir.resolve())
    except OSError as exc:
        raise ToolError(
            "validation_failed",
            f"Atomic {label} publish failed: {clean_text(exc)}",
        ) from exc


def write_json_atomic(
    path: Path,
    payload: Any,
    *,
    overwrite: bool,
    protected_sources: Iterable[Path] = (),
) -> None:
    check_destination(path, sources=protected_sources, overwrite=overwrite)
    with temporary_sibling(path, ".json") as temp_path:
        temp_path.write_text(
            json.dumps(
                json_safe(payload),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        try:
            json.loads(temp_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ToolError("validation_failed", "Cannot reopen JSON output") from exc
        atomic_publish(temp_path, path)


def write_text_atomic(
    path: Path,
    text: str,
    *,
    overwrite: bool,
    protected_sources: Iterable[Path] = (),
) -> None:
    check_destination(path, sources=protected_sources, overwrite=overwrite)
    with temporary_sibling(path, path.suffix or ".txt") as temp_path:
        temp_path.write_text(text, encoding="utf-8")
        temp_path.read_text(encoding="utf-8")
        atomic_publish(temp_path, path)


def validate_pdf_output(
    path: Path,
    *,
    expected_pages: int | None = None,
    expect_encrypted: bool = False,
    password: str | None = None,
) -> dict[str, Any]:
    require_pdf_signature(path)
    try:
        reader = PdfReader(str(path), strict=True)
        encrypted = reader.is_encrypted
        if encrypted and password:
            if not reader.decrypt(password):
                raise ToolError("validation_failed", "Output password did not decrypt output")
        if expected_pages is not None and len(reader.pages) != expected_pages:
            raise ToolError(
                "validation_failed",
                f"Output has {len(reader.pages)} pages; expected {expected_pages}",
            )
        if encrypted != expect_encrypted:
            raise ToolError(
                "validation_failed",
                f"Output encryption state is {encrypted}; expected {expect_encrypted}",
            )
        return {
            "valid": True,
            "signature": "%PDF-",
            "page_count": len(reader.pages),
            "encrypted": encrypted,
            "reopened": True,
        }
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(
            "validation_failed",
            f"Cannot reopen generated PDF: {clean_text(exc)}",
        ) from exc


def write_pdf_atomic(
    destination: Path,
    builder: Callable[[Path], None],
    *,
    sources: Iterable[Path] = (),
    overwrite: bool,
    expected_pages: int | None = None,
    expect_encrypted: bool = False,
    password: str | None = None,
) -> dict[str, Any]:
    check_destination(destination, sources=sources, overwrite=overwrite)
    with temporary_sibling(destination, ".pdf") as temp_path:
        builder(temp_path)
        verification = validate_pdf_output(
            temp_path,
            expected_pages=expected_pages,
            expect_encrypted=expect_encrypted,
            password=password,
        )
        atomic_publish(temp_path, destination)
    return verification


def page_size_from_spec(value: Any) -> tuple[float, float]:
    if value is None:
        return LETTER
    if isinstance(value, str):
        try:
            return PAGE_SIZES[value.lower()]
        except KeyError as exc:
            raise ToolError("bad_input", f"Unsupported page size: {value}") from exc
    if isinstance(value, list) and len(value) == 2:
        return (
            finite_number(
                value[0],
                "page_size[0]",
                minimum=0,
                minimum_inclusive=False,
                maximum=14_400,
            ),
            finite_number(
                value[1],
                "page_size[1]",
                minimum=0,
                minimum_inclusive=False,
                maximum=14_400,
            ),
        )
    raise ToolError("bad_input", "Page size must be a known name or [width, height]")


def validate_image(path: Path) -> tuple[int, int]:
    require_regular_file(path)
    if path.stat().st_size > MAX_IMAGE_BYTES:
        raise ToolError(
            "resource_limit",
            f"Image exceeds the {MAX_IMAGE_BYTES}-byte input limit: {path}",
        )
    try:
        with Image.open(path) as image:
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise ToolError(
                    "resource_limit",
                    f"Image exceeds the {MAX_IMAGE_PIXELS}-pixel limit: {path}",
                )
            image.verify()
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError("bad_input", f"Cannot decode image {path}") from exc
    return width, height


def color_value(value: Any, default: colors.Color = colors.black) -> colors.Color:
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return colors.HexColor(value)
        except ValueError as exc:
            named = getattr(colors, value.lower(), None)
            if named is None:
                raise ToolError("bad_input", f"Invalid color: {value}") from exc
            return named
    if isinstance(value, list) and len(value) in {3, 4}:
        parts = [
            finite_number(item, f"color[{index}]", minimum=0, maximum=1)
            for index, item in enumerate(value)
        ]
        return colors.Color(*parts[:3], alpha=parts[3] if len(parts) == 4 else 1)
    raise ToolError("bad_input", f"Invalid color: {value}")


def draw_wrapped_text(
    target: canvas.Canvas,
    text: str,
    x: float,
    top: float,
    width: float,
    page_height: float,
    *,
    page_width: float,
    font: str,
    size: float,
    leading: float,
    color: colors.Color,
    align: str,
) -> float:
    x, top, width, _ = validate_rectangle(
        x=x,
        y=top,
        width=width,
        height=finite_number(size, "text.font_size", minimum=0, minimum_inclusive=False),
        page_width=page_width,
        page_height=page_height,
        field="text",
    )
    size = finite_number(size, "text.font_size", minimum=0, minimum_inclusive=False)
    leading = finite_number(leading, "text.leading", minimum=0, minimum_inclusive=False)
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and stringWidth(candidate, font, size) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current or not words:
        lines.append(current)
    rendered_height = size + max(0, len(lines) - 1) * leading
    if top + rendered_height > page_height:
        raise ToolError("bad_input", "text content does not fit within the page geometry")
    target.setFont(font, size)
    target.setFillColor(color)
    y = page_height - top - size
    for line in lines:
        if align == "right":
            target.drawRightString(x + width, y, line)
        elif align == "center":
            target.drawCentredString(x + width / 2, y, line)
        else:
            target.drawString(x, y, line)
        y -= leading
    return page_height - y


def draw_table(
    target: canvas.Canvas,
    element: dict[str, Any],
    page_width: float,
    page_height: float,
) -> None:
    data = element.get("data")
    if not isinstance(data, list) or not data or not all(isinstance(row, list) for row in data):
        raise ToolError("bad_input", "Table data must be a nonempty array of row arrays")
    row_height = finite_number(
        element.get("row_height", 22),
        "table.row_height",
        minimum=0,
        minimum_inclusive=False,
    )
    x, top, width, _ = validate_rectangle(
        x=element.get("x", 72),
        y=element.get("y", 72),
        width=element.get("width", 468),
        height=row_height * len(data),
        page_width=page_width,
        page_height=page_height,
        field="table",
    )
    columns = max(len(row) for row in data)
    if columns < 1:
        raise ToolError("bad_input", "Table must contain at least one column")
    col_widths = element.get("column_widths") or [width / columns] * columns
    if len(col_widths) != columns:
        raise ToolError("bad_input", "column_widths must match the maximum table column count")
    normalized_widths = [
        finite_number(
            item,
            f"table.column_widths[{index}]",
            minimum=0,
            minimum_inclusive=False,
        )
        for index, item in enumerate(col_widths)
    ]
    if sum(normalized_widths) > width:
        raise ToolError("bad_input", "table.column_widths must fit within table.width")
    normalized = [row + [""] * (columns - len(row)) for row in data]
    table = Table(
        normalized,
        colWidths=normalized_widths,
        rowHeights=[row_height] * len(normalized),
    )
    commands: list[tuple[Any, ...]] = [
        (
            "GRID",
            (0, 0),
            (-1, -1),
            finite_number(element.get("line_width", 0.5), "table.line_width", minimum=0),
            colors.black,
        ),
        ("FONT", (0, 0), (-1, -1), element.get("font", "Helvetica")),
        (
            "FONTSIZE",
            (0, 0),
            (-1, -1),
            finite_number(
                element.get("font_size", 9),
                "table.font_size",
                minimum=0,
                minimum_inclusive=False,
            ),
        ),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    if element.get("header", True):
        commands.extend(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    color_value(element.get("header_color"), colors.lightgrey),
                ),
                ("FONT", (0, 0), (-1, 0), element.get("header_font", "Helvetica-Bold")),
            ]
        )
    table.setStyle(TableStyle(commands))
    table.wrapOn(target, width, row_height * len(normalized))
    y = page_height - top - row_height * len(normalized)
    table.drawOn(target, x, y)


def draw_image(
    target: canvas.Canvas,
    element: dict[str, Any],
    page_width: float,
    page_height: float,
    sources: list[Path],
) -> None:
    image_path = Path(str(element.get("path", ""))).expanduser()
    validate_image(image_path)
    sources.append(image_path)
    x, top, width, height = validate_rectangle(
        x=element.get("x", 72),
        y=element.get("y", 72),
        width=element.get("width", 200),
        height=element.get("height", 150),
        page_width=page_width,
        page_height=page_height,
        field="image",
    )
    target.drawImage(
        ImageReader(str(image_path)),
        x,
        page_height - top - height,
        width=width,
        height=height,
        preserveAspectRatio=bool(element.get("preserve_aspect_ratio", True)),
        mask="auto",
    )


def draw_form_field(
    target: canvas.Canvas,
    element: dict[str, Any],
    page_width: float,
    page_height: float,
) -> None:
    field_type = element.get("field_type", "text")
    name = element.get("name")
    if not isinstance(name, str) or not name:
        raise ToolError("bad_input", "Form field requires a nonempty name")
    x, top, width, height = validate_rectangle(
        x=element.get("x", 72),
        y=element.get("y", 72),
        width=element.get("width", 180),
        height=element.get("height", 20),
        page_width=page_width,
        page_height=page_height,
        field=f"form[{name}]",
    )
    y = page_height - top - height
    if field_type == "text":
        target.acroForm.textfield(
            name=name,
            value=str(element.get("value", "")),
            x=x,
            y=y,
            width=width,
            height=height,
            borderStyle="solid",
            borderWidth=1,
            forceBorder=True,
        )
    elif field_type == "checkbox":
        target.acroForm.checkbox(
            name=name,
            checked=bool(element.get("value", False)),
            x=x,
            y=y,
            size=min(width, height),
            borderWidth=1,
            forceBorder=True,
        )
    else:
        raise ToolError("unsupported_operation", f"Unsupported form field type: {field_type}")


def draw_page_spec(
    target: canvas.Canvas,
    page: dict[str, Any],
    page_number: int,
    default_size: tuple[float, float],
    document_header: dict[str, Any] | None,
    document_footer: dict[str, Any] | None,
    sources: list[Path],
) -> None:
    page_size = page_size_from_spec(page["size"]) if page.get("size") is not None else default_size
    target.setPageSize(page_size)
    page_width, page_height = page_size
    header_footer_positions = (
        (document_header, 24),
        (document_footer, page_height - 36),
    )
    for header_or_footer, default_top in header_footer_positions:
        if header_or_footer:
            value = str(header_or_footer.get("text", "")).replace("{page}", str(page_number))
            draw_wrapped_text(
                target,
                value,
                header_or_footer.get("x", 54),
                header_or_footer.get("y", default_top),
                header_or_footer.get("width", page_width - 108),
                page_height,
                page_width=page_width,
                font=str(header_or_footer.get("font", "Helvetica")),
                size=header_or_footer.get("font_size", 9),
                leading=header_or_footer.get("leading", 11),
                color=color_value(header_or_footer.get("color")),
                align=str(header_or_footer.get("align", "left")),
            )
    elements = page.get("elements", [])
    if not isinstance(elements, list):
        raise ToolError("bad_input", "Page elements must be an array")
    for element in elements:
        if not isinstance(element, dict):
            raise ToolError("bad_input", "Every page element must be an object")
        element_type = element.get("type")
        if element_type in {"text", "paragraph"}:
            draw_wrapped_text(
                target,
                str(element.get("text", "")),
                element.get("x", 72),
                element.get("y", 72),
                element.get("width", page_width - 144),
                page_height,
                page_width=page_width,
                font=str(element.get("font", "Helvetica")),
                size=element.get("font_size", 11),
                leading=element.get(
                    "leading",
                    finite_number(
                        element.get("font_size", 11),
                        "text.font_size",
                        minimum=0,
                        minimum_inclusive=False,
                    )
                    * 1.25,
                ),
                color=color_value(element.get("color")),
                align=str(element.get("align", "left")),
            )
        elif element_type == "table":
            draw_table(target, element, page_width, page_height)
        elif element_type == "image":
            draw_image(target, element, page_width, page_height, sources)
        elif element_type == "form":
            draw_form_field(target, element, page_width, page_height)
        elif element_type == "line":
            x1 = finite_number(element.get("x1", 72), "line.x1", minimum=0)
            y1 = finite_number(element.get("y1", 72), "line.y1", minimum=0)
            x2 = finite_number(element.get("x2", page_width - 72), "line.x2", minimum=0)
            y2 = finite_number(element.get("y2", 72), "line.y2", minimum=0)
            if x1 > page_width or x2 > page_width or y1 > page_height or y2 > page_height:
                raise ToolError("bad_input", "Line coordinates must fit within the page geometry")
            target.setStrokeColor(color_value(element.get("color")))
            target.setLineWidth(
                finite_number(element.get("line_width", 1), "line.line_width", minimum=0)
            )
            target.line(
                x1,
                page_height - y1,
                x2,
                page_height - y2,
            )
        else:
            raise ToolError("unsupported_operation", f"Unsupported create element: {element_type}")
    target.showPage()


def metadata_for_reportlab(target: canvas.Canvas, metadata: dict[str, Any]) -> None:
    setters = {
        "title": target.setTitle,
        "author": target.setAuthor,
        "subject": target.setSubject,
        "creator": target.setCreator,
        "keywords": target.setKeywords,
    }
    for key, setter in setters.items():
        if key in metadata:
            setter(str(metadata[key]))


def create_layout_pdf(spec: dict[str, Any], output: Path) -> tuple[int, list[Path]]:
    pages = spec.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ToolError("bad_input", "Layout create job requires a nonempty pages array")
    if len(pages) > MAX_PAGES:
        raise ToolError("resource_limit", f"Create job exceeds {MAX_PAGES} pages")
    default_size = page_size_from_spec(spec.get("page_size"))
    for field in ("header", "footer", "metadata"):
        if spec.get(field) is not None and not isinstance(spec[field], dict):
            raise ToolError("bad_input", f"{field} must be an object")
    target = canvas.Canvas(str(output), pagesize=default_size, pageCompression=1)
    metadata_for_reportlab(target, spec.get("metadata") or {})
    sources: list[Path] = []
    for index, page in enumerate(pages, start=1):
        if not isinstance(page, dict):
            raise ToolError("bad_input", "Every page specification must be an object")
        draw_page_spec(
            target,
            page,
            index,
            default_size,
            spec.get("header"),
            spec.get("footer"),
            sources,
        )
    target.save()
    return len(pages), sources


def paragraph_alignment(value: str) -> int:
    return {
        "left": TA_LEFT,
        "center": TA_CENTER,
        "right": TA_RIGHT,
        "justify": TA_JUSTIFY,
    }.get(value, TA_LEFT)


def create_story_pdf(spec: dict[str, Any], output: Path) -> tuple[int, list[Path]]:
    story_spec = spec.get("story")
    if not isinstance(story_spec, list) or not story_spec:
        raise ToolError("bad_input", "Story create job requires a nonempty story array")
    if len(story_spec) > MAX_STORY_ITEMS:
        raise ToolError("resource_limit", f"Story exceeds {MAX_STORY_ITEMS} items")
    page_size = page_size_from_spec(spec.get("page_size"))
    margins = spec.get("margins")
    if margins is None:
        margins = {}
    if not isinstance(margins, dict):
        raise ToolError("bad_input", "margins must be an object")
    normalized_margins = {
        side: finite_number(margins.get(side, default), f"margins.{side}", minimum=0)
        for side, default in (
            ("left", 54),
            ("right", 54),
            ("top", 64),
            ("bottom", 64),
        )
    }
    if normalized_margins["left"] + normalized_margins["right"] >= page_size[0]:
        raise ToolError("bad_input", "Horizontal margins must leave positive page width")
    if normalized_margins["top"] + normalized_margins["bottom"] >= page_size[1]:
        raise ToolError("bad_input", "Vertical margins must leave positive page height")
    for field in ("header", "footer", "metadata"):
        if spec.get(field) is not None and not isinstance(spec[field], dict):
            raise ToolError("bad_input", f"{field} must be an object")
    doc = SimpleDocTemplate(
        str(output),
        pagesize=page_size,
        leftMargin=normalized_margins["left"],
        rightMargin=normalized_margins["right"],
        topMargin=normalized_margins["top"],
        bottomMargin=normalized_margins["bottom"],
        title=str((spec.get("metadata") or {}).get("title", "")),
        author=str((spec.get("metadata") or {}).get("author", "")),
    )
    sample_styles = getSampleStyleSheet()
    flowables = []
    sources: list[Path] = []
    for item in story_spec:
        if not isinstance(item, dict):
            raise ToolError("bad_input", "Every story item must be an object")
        item_type = item.get("type")
        if item_type in {"paragraph", "heading"}:
            base_name = "Heading1" if item_type == "heading" else "BodyText"
            base = sample_styles[base_name]
            style = ParagraphStyle(
                f"Generated{len(flowables)}",
                parent=base,
                fontName=str(item.get("font", base.fontName)),
                fontSize=finite_number(
                    item.get("font_size", base.fontSize),
                    "story.font_size",
                    minimum=0,
                    minimum_inclusive=False,
                ),
                leading=finite_number(
                    item.get("leading", base.leading),
                    "story.leading",
                    minimum=0,
                    minimum_inclusive=False,
                ),
                textColor=color_value(item.get("color"), base.textColor),
                alignment=paragraph_alignment(str(item.get("align", "left"))),
                spaceAfter=finite_number(
                    item.get("space_after", 8),
                    "story.space_after",
                    minimum=0,
                ),
            )
            flowables.append(Paragraph(str(item.get("text", "")), style))
        elif item_type == "spacer":
            flowables.append(
                Spacer(
                    1,
                    finite_number(
                        item.get("height", 12),
                        "story.spacer.height",
                        minimum=0,
                    ),
                )
            )
        elif item_type == "page_break":
            flowables.append(PageBreak())
        elif item_type == "table":
            data = item.get("data")
            if not isinstance(data, list) or not data:
                raise ToolError("bad_input", "Story table data must be a nonempty array")
            column_widths = item.get("column_widths")
            if column_widths is not None:
                if not isinstance(column_widths, list) or not column_widths:
                    raise ToolError("bad_input", "Story table column_widths must be an array")
                column_widths = [
                    finite_number(
                        value,
                        f"story.table.column_widths[{index}]",
                        minimum=0,
                        minimum_inclusive=False,
                    )
                    for index, value in enumerate(column_widths)
                ]
                available_width = (
                    page_size[0] - normalized_margins["left"] - normalized_margins["right"]
                )
                if sum(column_widths) > available_width:
                    raise ToolError(
                        "bad_input",
                        "Story table column widths must fit within the page margins",
                    )
            table = Table(
                data,
                colWidths=column_widths,
                repeatRows=finite_integer(
                    item.get("repeat_rows", 1),
                    "story.table.repeat_rows",
                    minimum=0,
                    maximum=len(data),
                ),
            )
            table.setStyle(
                TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            flowables.append(table)
        elif item_type == "image":
            image_path = Path(str(item.get("path", ""))).expanduser()
            validate_image(image_path)
            sources.append(image_path)
            flowables.append(
                FlowImage(
                    str(image_path),
                    width=finite_number(
                        item.get("width", 240),
                        "story.image.width",
                        minimum=0,
                        minimum_inclusive=False,
                        maximum=page_size[0]
                        - normalized_margins["left"]
                        - normalized_margins["right"],
                    ),
                    height=finite_number(
                        item.get("height", 180),
                        "story.image.height",
                        minimum=0,
                        minimum_inclusive=False,
                        maximum=page_size[1]
                        - normalized_margins["top"]
                        - normalized_margins["bottom"],
                    ),
                )
            )
        else:
            raise ToolError("unsupported_operation", f"Unsupported story item: {item_type}")

    header = spec.get("header")
    footer = spec.get("footer")
    metadata = spec.get("metadata") or {}

    def decorate_page(target: canvas.Canvas, document: Any) -> None:
        metadata_for_reportlab(target, metadata)
        width, height = page_size
        for value, default_y in ((header, height - 30), (footer, 24)):
            if value:
                text = str(value.get("text", "")).replace("{page}", str(document.page))
                target.setFont(
                    str(value.get("font", "Helvetica")),
                    finite_number(
                        value.get("font_size", 9),
                        "header_footer.font_size",
                        minimum=0,
                        minimum_inclusive=False,
                    ),
                )
                x = finite_number(value.get("x", 54), "header_footer.x", minimum=0)
                baseline = finite_number(
                    value.get("baseline", default_y),
                    "header_footer.baseline",
                    minimum=0,
                )
                if x > width or baseline > height:
                    raise ToolError(
                        "bad_input",
                        "Header/footer position must fit within the page geometry",
                    )
                target.drawString(
                    x,
                    baseline,
                    text,
                )

    doc.build(flowables, onFirstPage=decorate_page, onLaterPages=decorate_page)
    reader = PdfReader(str(output))
    return len(reader.pages), sources


def handle_create(args: argparse.Namespace) -> dict[str, Any]:
    job_path = Path(args.job)
    spec = read_json(job_path)
    output = Path(args.output)
    if output.suffix.lower() != ".pdf":
        raise ToolError("bad_input", "Create output must use a .pdf extension")
    if "pages" in spec and "story" in spec:
        raise ToolError("ambiguous_edit", "Create job cannot contain both pages and story")
    builder_result: dict[str, Any] = {}

    def builder(temp_path: Path) -> None:
        if "story" in spec:
            count, sources = create_story_pdf(spec, temp_path)
        else:
            count, sources = create_layout_pdf(spec, temp_path)
        builder_result["page_count"] = count
        builder_result["sources"] = sources

    check_destination(output, sources=[job_path], overwrite=args.overwrite)
    with temporary_sibling(output, ".pdf") as temp_path:
        builder(temp_path)
        verification = validate_pdf_output(
            temp_path,
            expected_pages=builder_result["page_count"],
        )
        for source in builder_result["sources"]:
            if paths_alias(output, source):
                raise ToolError("bad_input", "Output cannot replace an input image")
        atomic_publish(temp_path, output)
    return success(
        "create",
        output_path=str(output.resolve()),
        counts={"pages": builder_result["page_count"]},
        warnings=[],
        verification=verification,
    )


def extract_embedded_images(
    source: SourceInfo,
    selected: Sequence[int],
    output_dir: Path,
    *,
    overwrite: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    check_directory_destination(
        output_dir,
        overwrite=overwrite,
        exists_message=f"Image output directory exists: {output_dir}",
        not_directory_message=f"Image output must be a directory: {output_dir}",
        nonempty_message="Refusing to overwrite a nonempty image output directory",
    )
    staged_dir = stage_directory(output_dir)
    extracted = []
    warnings = []
    count = 0
    try:
        for page_index in selected:
            try:
                page_images = source.reader.pages[page_index].images
            except Exception as exc:
                warnings.append(f"Page {page_index + 1} image inventory failed: {clean_text(exc)}")
                continue
            for image_index, image in enumerate(page_images, start=1):
                count += 1
                if count > MAX_IMAGES:
                    raise ToolError("resource_limit", f"Embedded image count exceeds {MAX_IMAGES}")
                extension = Path(image.name).suffix.lower() or ".bin"
                name = f"page-{page_index + 1:04d}-image-{image_index:04d}{extension}"
                staged_destination = staged_dir / name
                published_destination = output_dir / name
                try:
                    staged_destination.write_bytes(image.data)
                    extracted.append(
                        {
                            "page": page_index + 1,
                            "name": image.name,
                            "output_path": str(published_destination.resolve()),
                            "bytes": len(image.data),
                        }
                    )
                except Exception as exc:
                    warnings.append(
                        f"Page {page_index + 1} image {image_index} extraction failed: "
                        f"{clean_text(exc)}"
                    )
        atomic_publish_directory(staged_dir, output_dir, label="image-directory")
    finally:
        shutil.rmtree(staged_dir, ignore_errors=True)
    return extracted, warnings


def handle_inspect_or_extract(args: argparse.Namespace) -> dict[str, Any]:
    password = resolve_password(args.password, args.password_env)
    path = Path(args.input)
    data = inspect_document(
        path,
        password,
        args.pages,
        mode=args.mode,
        include_words=args.words,
        include_tables=args.tables,
    )
    if getattr(args, "images_dir", None):
        source = open_pdf(path, password)
        selected = parse_page_selection(args.pages, len(source.reader.pages))
        images, warnings = extract_embedded_images(
            source,
            selected,
            Path(args.images_dir),
            overwrite=args.overwrite,
        )
        data["extracted_images"] = images
        data["warnings"].extend(warnings)
    return output_payload(
        args.command,
        data,
        Path(args.output) if args.output else None,
        overwrite=args.overwrite,
        protected_sources=[path],
    )


def handle_render(args: argparse.Namespace) -> dict[str, Any]:
    password = resolve_password(args.password, args.password_env)
    source_path = Path(args.input)
    source = open_pdf(source_path, password)
    selected = parse_page_selection(args.pages, len(source.reader.pages))
    if not selected:
        raise ToolError("bad_input", "Source PDF has no pages to render")
    output_dir = Path(args.output)
    if paths_alias(output_dir, source_path):
        raise ToolError("bad_input", "Render output directory must differ from the source")
    check_directory_destination(
        output_dir,
        overwrite=args.overwrite,
        exists_message=f"Render output directory exists: {output_dir}",
        not_directory_message=f"Render output must be a directory: {output_dir}",
        nonempty_message="Refusing to overwrite a nonempty render output directory",
    )
    dpi = args.dpi
    warnings: list[str] = []
    fonts, font_walk_warnings = collect_fonts(source.reader)
    warnings.extend(font_walk_warnings)
    if fonts["unembedded_non_base14"]:
        warnings.append(
            "Rendered rasters substitute this machine's local fonts for non-embedded "
            "fonts; other viewers may substitute differently."
        )
    if source.reader.is_encrypted:
        warnings.append(
            "Encrypted source was rendered; the published PNG files are not password protected."
        )
    budget = RenderBudget(
        total_message=(
            f"Render total exceeds {MAX_TOTAL_RENDER_PIXELS} pixels; "
            "select fewer pages or lower --dpi"
        )
    )
    renders: list[dict[str, Any]] = []
    with pdfplumber.open(str(source_path), password=password) as plumber_pdf:
        for page_index in selected:
            page = plumber_pdf.pages[page_index]
            budget.charge(
                float(page.width),
                float(page.height),
                dpi,
                page_label=f"Page {page_index + 1}",
            )
        staged_dir = stage_directory(output_dir)
        try:
            for page_index in selected:
                page = plumber_pdf.pages[page_index]
                name = f"page-{page_index + 1:04d}.png"
                staged_path = staged_dir / name
                rasterize_region(
                    page,
                    None,
                    dpi,
                    staged_path,
                    page_label=f"page {page_index + 1}",
                )
                with staged_path.open("rb") as handle:
                    if handle.read(8) != PNG_SIGNATURE:
                        raise ToolError(
                            "validation_failed",
                            f"Rendered file is not a PNG: {name}",
                        )
                try:
                    with Image.open(staged_path) as image:
                        image.load()
                        actual_width, actual_height = image.size
                except Exception as exc:
                    raise ToolError(
                        "validation_failed",
                        f"Cannot reopen rendered PNG {name}: {clean_text(exc)}",
                    ) from exc
                if actual_width * actual_height > MAX_PIXELS_PER_RENDER:
                    raise ToolError(
                        "resource_limit",
                        f"Page {page_index + 1} render exceeds {MAX_PIXELS_PER_RENDER} pixels",
                    )
                renders.append(
                    {
                        "page": page_index + 1,
                        "file": name,
                        "output_path": str((output_dir / name).resolve()),
                        "width_px": actual_width,
                        "height_px": actual_height,
                        "bytes": staged_path.stat().st_size,
                        "sha256": sha256_file(staged_path),
                    }
                )
            atomic_publish_directory(staged_dir, output_dir, label="render-directory")
        finally:
            shutil.rmtree(staged_dir, ignore_errors=True)
    reopened = 0
    for item in renders:
        published = Path(item["output_path"])
        try:
            published_hash = sha256_file(published)
        except OSError as exc:
            raise ToolError(
                "validation_failed",
                f"Cannot reopen published PNG {item['file']}",
            ) from exc
        if published_hash != item["sha256"]:
            raise ToolError(
                "validation_failed",
                f"Published PNG hash changed: {item['file']}",
            )
        reopened += 1
    if len({(item["width_px"], item["height_px"]) for item in renders}) > 1:
        warnings.append(
            "Rendered pages have differing pixel dimensions because source page sizes differ."
        )
    return success(
        "render",
        source=str(source_path.resolve()),
        output_dir=str(output_dir.resolve()),
        encrypted=bool(source.reader.is_encrypted),
        password_used=source.password_used,
        page_count=len(source.reader.pages),
        selected_pages=[index + 1 for index in selected],
        dpi=dpi,
        renders=renders,
        counts={"pages_rendered": len(renders), "total_pixels": budget.total_pixels},
        fonts=fonts,
        warnings=warnings + font_release_warnings(fonts),
        verification={
            "valid": True,
            "atomic_publish": True,
            "pngs_reopened": reopened,
            "renderer": "pdfplumber/pypdfium2",
        },
    )


def page_refs_from_output(
    output_spec: dict[str, Any],
    inputs: dict[str, SourceInfo],
) -> list[tuple[SourceInfo, int, int]]:
    refs = output_spec.get("pages")
    if not isinstance(refs, list) or not refs:
        raise ToolError("bad_input", "Each pages output requires a nonempty pages array")
    resolved: list[tuple[SourceInfo, int, int]] = []
    for ref in refs:
        if not isinstance(ref, dict):
            raise ToolError("bad_input", "Every page reference must be an object")
        input_id = ref.get("input")
        if input_id not in inputs:
            raise ToolError("bad_input", f"Unknown page input id: {input_id}")
        source = inputs[input_id]
        page_number = finite_integer(
            ref.get("page"),
            f"Page reference for {input_id}",
            minimum=1,
            maximum=len(source.reader.pages),
        )
        if not 1 <= page_number <= len(source.reader.pages):
            raise ToolError(
                "bad_input",
                f"Page reference for {input_id} must be within 1-{len(source.reader.pages)}",
            )
        repeat = finite_integer(
            ref.get("repeat", 1),
            "Page repeat",
            minimum=1,
            maximum=MAX_PAGES,
        )
        rotation = parse_rotation(ref.get("rotate", 0))
        resolved.extend((source, page_number - 1, rotation) for _ in range(repeat))
        if len(resolved) > MAX_PAGES:
            raise ToolError("resource_limit", f"Output exceeds {MAX_PAGES} pages")
    return resolved


def build_pages_output(
    temp_path: Path,
    refs: Sequence[tuple[SourceInfo, int, int]],
    metadata: dict[str, Any] | None,
) -> None:
    writer = PdfWriter()
    for source, page_index, rotation in refs:
        writer.add_page(source.reader.pages[page_index])
        if rotation:
            writer.pages[-1].rotate(rotation)
    if metadata:
        writer.add_metadata(
            {
                key if str(key).startswith("/") else f"/{key}": str(value)
                for key, value in metadata.items()
            }
        )
    with temp_path.open("wb") as handle:
        writer.write(handle)


def handle_pages(args: argparse.Namespace) -> dict[str, Any]:
    job_path = Path(args.job)
    job = read_json(job_path)
    input_specs = job.get("inputs")
    output_specs = job.get("outputs")
    if not isinstance(input_specs, list) or not input_specs:
        raise ToolError("bad_input", "Pages job requires a nonempty inputs array")
    if not isinstance(output_specs, list) or not output_specs:
        raise ToolError("bad_input", "Pages job requires a nonempty outputs array")
    if len(output_specs) > MAX_OUTPUTS:
        raise ToolError("resource_limit", f"Pages job exceeds {MAX_OUTPUTS} outputs")
    inputs: dict[str, SourceInfo] = {}
    source_paths: list[Path] = []
    warnings: list[str] = []
    for item in input_specs:
        if not isinstance(item, dict):
            raise ToolError("bad_input", "Every pages input must be an object")
        input_id = item.get("id")
        if not isinstance(input_id, str) or not input_id or input_id in inputs:
            raise ToolError("bad_input", f"Input id must be nonempty and unique: {input_id}")
        password = item.get("password")
        if password:
            SECRET_VALUES.add(str(password))
        path = Path(str(item.get("path", "")))
        inputs[input_id] = open_pdf(path, str(password) if password is not None else None)
        if inputs[input_id].reader.is_encrypted:
            if item.get("allow_decrypted_output") is not True:
                raise ToolError(
                    "unsupported_operation",
                    f"Encrypted pages input {input_id} requires allow_decrypted_output: true",
                )
            warnings.append(
                f"Encrypted input {input_id} is explicitly authorized to produce "
                "unencrypted page output."
            )
        source_paths.append(path)
    prepared: list[tuple[Path, Path, int]] = []
    output_paths: list[Path] = []
    try:
        for item in output_specs:
            if not isinstance(item, dict):
                raise ToolError("bad_input", "Every pages output must be an object")
            destination = Path(str(item.get("path", "")))
            if destination.suffix.lower() != ".pdf":
                raise ToolError("bad_input", "Every pages output must use a .pdf extension")
            if any(paths_alias(destination, prior) for prior in output_paths):
                raise ToolError("ambiguous_edit", f"Duplicate output destination: {destination}")
            output_paths.append(destination)
            check_destination(
                destination,
                sources=[job_path, *source_paths],
                overwrite=args.overwrite,
            )
            destination.resolve().parent.mkdir(parents=True, exist_ok=True)
            refs = page_refs_from_output(item, inputs)
            metadata = item.get("metadata")
            if metadata is not None and not isinstance(metadata, dict):
                raise ToolError("bad_input", "Pages output metadata must be an object")
            fd, temp_name = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                suffix=".pdf",
                dir=destination.resolve().parent,
            )
            os.close(fd)
            temp_path = Path(temp_name)
            build_pages_output(temp_path, refs, metadata)
            validate_pdf_output(temp_path, expected_pages=len(refs))
            prepared.append((temp_path, destination, len(refs)))
        for temp_path, destination, _ in prepared:
            atomic_publish(temp_path, destination)
    finally:
        for temp_path, _, _ in prepared:
            temp_path.unlink(missing_ok=True)
    if len(prepared) > 1:
        warnings.append(
            "Each split output was atomically published; publication of the output set "
            "is not transactional."
        )
    return success(
        "pages",
        outputs=[
            {
                "path": str(destination.resolve()),
                "page_count": count,
                "verification": {"valid": True, "reopened": True},
            }
            for _, destination, count in prepared
        ],
        counts={
            "inputs": len(inputs),
            "outputs": len(prepared),
            "pages": sum(count for _, _, count in prepared),
        },
        warnings=warnings,
    )


def page_indexes_for_operation(operation: dict[str, Any], page_count: int) -> list[int]:
    return parse_page_selection(operation.get("pages"), page_count)


def make_text_overlay(
    page_width: float,
    page_height: float,
    operation: dict[str, Any],
) -> PdfReader:
    buffer = io.BytesIO()
    target = canvas.Canvas(buffer, pagesize=(page_width, page_height), pageCompression=1)
    opacity = finite_number(operation.get("opacity", 1), "stamp.opacity", minimum=0, maximum=1)
    if hasattr(target, "setFillAlpha"):
        target.setFillAlpha(opacity)
    target.setFillColor(color_value(operation.get("color")))
    target.setFont(
        str(operation.get("font", "Helvetica")),
        finite_number(
            operation.get("font_size", 24),
            "stamp.font_size",
            minimum=0,
            minimum_inclusive=False,
        ),
    )
    text = str(operation.get("text", ""))
    x = finite_number(
        operation.get("x", page_width / 2),
        "stamp.x",
        minimum=0,
        maximum=page_width,
    )
    y = finite_number(
        operation.get("y", page_height / 2),
        "stamp.y",
        minimum=0,
        maximum=page_height,
    )
    angle = finite_number(operation.get("angle", 0), "stamp.angle")
    target.saveState()
    target.translate(x, y)
    target.rotate(angle)
    target.drawCentredString(0, 0, text)
    target.restoreState()
    target.save()
    buffer.seek(0)
    return PdfReader(buffer)


def make_image_overlay(
    page_width: float,
    page_height: float,
    operation: dict[str, Any],
    sources: list[Path],
) -> PdfReader:
    image_path = Path(str(operation.get("path", "")))
    validate_image(image_path)
    sources.append(image_path)
    buffer = io.BytesIO()
    target = canvas.Canvas(buffer, pagesize=(page_width, page_height), pageCompression=1)
    x, y_from_top, width, height = validate_rectangle(
        x=operation.get("x", 36),
        y=page_height
        - finite_number(operation.get("y", 36), "stamp_image.y", minimum=0)
        - finite_number(
            operation.get("height", 80),
            "stamp_image.height",
            minimum=0,
            minimum_inclusive=False,
        ),
        width=operation.get("width", 120),
        height=operation.get("height", 80),
        page_width=page_width,
        page_height=page_height,
        field="stamp_image",
    )
    y = page_height - y_from_top - height
    target.drawImage(
        ImageReader(str(image_path)),
        x,
        y,
        width=width,
        height=height,
        preserveAspectRatio=True,
        mask="auto",
    )
    target.save()
    buffer.seek(0)
    return PdfReader(buffer)


def apply_stamp_pdf(
    target_page: Any,
    operation: dict[str, Any],
    sources: list[Path],
) -> None:
    stamp_path = Path(str(operation.get("path", "")))
    password = operation.get("password")
    if password:
        SECRET_VALUES.add(str(password))
    stamp_source = open_pdf(stamp_path, str(password) if password is not None else None)
    sources.append(stamp_path)
    stamp_page_number = finite_integer(
        operation.get("stamp_page", 1),
        "stamp_pdf.stamp_page",
        minimum=1,
    )
    if not 1 <= stamp_page_number <= len(stamp_source.reader.pages):
        raise ToolError("bad_input", "stamp_page is outside the stamp PDF")
    stamp_page = stamp_source.reader.pages[stamp_page_number - 1]
    scale = finite_number(
        operation.get("scale", 1),
        "stamp_pdf.scale",
        minimum=0,
        minimum_inclusive=False,
    )
    x = finite_number(operation.get("x", 0), "stamp_pdf.x", minimum=0)
    y = finite_number(operation.get("y", 0), "stamp_pdf.y", minimum=0)
    target_width = finite_number(target_page.mediabox.width, "target_page.width", minimum=0)
    target_height = finite_number(target_page.mediabox.height, "target_page.height", minimum=0)
    stamp_width = finite_number(stamp_page.mediabox.width, "stamp_page.width", minimum=0)
    stamp_height = finite_number(stamp_page.mediabox.height, "stamp_page.height", minimum=0)
    if x + stamp_width * scale > target_width or y + stamp_height * scale > target_height:
        raise ToolError("bad_input", "Scaled PDF stamp must fit within the target page geometry")
    transformation = (
        Transformation()
        .scale(scale)
        .translate(
            tx=x,
            ty=y,
        )
    )
    target_page.merge_transformed_page(
        stamp_page,
        transformation,
        over=bool(operation.get("over", True)),
    )


def normalize_metadata_job(metadata: dict[str, Any]) -> dict[str, str]:
    return {
        key if str(key).startswith("/") else f"/{key}": str(value)
        for key, value in metadata.items()
    }


def handle_edit(args: argparse.Namespace) -> dict[str, Any]:
    password = resolve_password(args.password, args.password_env)
    source_path = Path(args.input)
    source = open_pdf(source_path, password)
    if args.read_fields:
        return success(
            "edit",
            source=str(source_path.resolve()),
            mode="read_fields",
            forms=form_inventory(source.reader),
            output_path=None,
            verification={"valid": True, "source_unchanged": True},
        )
    if not args.job or not args.output:
        raise ToolError("bad_input", "Mutation edit requires --job and --output")
    job_path = Path(args.job)
    job = read_json(job_path)
    operations = job.get("operations")
    if not isinstance(operations, list) or not operations:
        raise ToolError("bad_input", "Edit job requires a nonempty operations array")
    writer = PdfWriter()
    writer.clone_document_from_reader(source.reader)
    warnings: list[str] = []
    extra_sources: list[Path] = []
    encryption: dict[str, Any] | None = None
    explicit_decrypt = False
    for operation in operations:
        if not isinstance(operation, dict):
            raise ToolError("bad_input", "Every edit operation must be an object")
        kind = operation.get("op")
        if kind in {"stamp_text", "stamp_image", "stamp_pdf"}:
            for page_index in page_indexes_for_operation(operation, len(writer.pages)):
                target_page = writer.pages[page_index]
                width = float(target_page.mediabox.width)
                height = float(target_page.mediabox.height)
                if kind == "stamp_text":
                    overlay = make_text_overlay(width, height, operation)
                    target_page.merge_page(overlay.pages[0], over=bool(operation.get("over", True)))
                elif kind == "stamp_image":
                    overlay = make_image_overlay(width, height, operation, extra_sources)
                    target_page.merge_page(overlay.pages[0], over=bool(operation.get("over", True)))
                else:
                    apply_stamp_pdf(target_page, operation, extra_sources)
        elif kind == "fill_form":
            values = operation.get("values")
            if not isinstance(values, dict) or not values:
                raise ToolError("bad_input", "fill_form requires a nonempty values object")
            available = source.reader.get_fields() or {}
            missing = sorted(set(values) - set(available))
            if missing:
                raise ToolError("ambiguous_edit", f"Unknown form fields: {', '.join(missing)}")
            page_indexes = page_indexes_for_operation(operation, len(writer.pages))
            for page_index in page_indexes:
                writer.update_page_form_field_values(
                    writer.pages[page_index],
                    {str(key): value for key, value in values.items()},
                    auto_regenerate=False,
                )
            warnings.append(
                "Form appearance streams vary by viewer; visually verify filled values."
            )
        elif kind == "set_metadata":
            metadata = operation.get("metadata")
            if not isinstance(metadata, dict):
                raise ToolError("bad_input", "set_metadata requires a metadata object")
            writer.add_metadata(normalize_metadata_job(metadata))
        elif kind == "remove_metadata":
            keys = operation.get("keys")
            if keys in (None, "all"):
                writer.metadata = None
            elif isinstance(keys, list):
                current = dict(writer.metadata or {})
                for key in keys:
                    current.pop(key if str(key).startswith("/") else f"/{key}", None)
                writer.metadata = current
            else:
                raise ToolError("bad_input", "remove_metadata keys must be an array or 'all'")
        elif kind == "encrypt":
            if encryption is not None:
                raise ToolError("ambiguous_edit", "Only one encrypt operation is allowed")
            algorithm = operation.get("algorithm")
            if algorithm not in {"AES-128", "AES-256-R5", "AES-256"}:
                raise ToolError(
                    "bad_input",
                    "encrypt requires explicit algorithm AES-128, AES-256-R5, or AES-256",
                )
            user_password = operation.get("user_password")
            owner_password = operation.get("owner_password")
            if not isinstance(user_password, str) or not user_password:
                raise ToolError("bad_input", "encrypt requires a nonempty user_password")
            SECRET_VALUES.add(user_password)
            if owner_password:
                SECRET_VALUES.add(str(owner_password))
            encryption = {
                "user_password": user_password,
                "owner_password": str(owner_password or user_password),
                "algorithm": algorithm,
            }
        elif kind == "decrypt":
            explicit_decrypt = True
            if not source.reader.is_encrypted:
                warnings.append("decrypt requested for an unencrypted source.")
        else:
            raise ToolError("unsupported_operation", f"Unsupported edit operation: {kind}")
    if source.reader.is_encrypted and not explicit_decrypt and encryption is None:
        raise ToolError(
            "unsupported_operation",
            "Editing an encrypted PDF requires an explicit decrypt operation or a new "
            "explicit encrypt operation",
        )
    if source.reader.is_encrypted and encryption is not None:
        warnings.append("Encrypted source is explicitly rewritten with the selected new AES mode.")
    if encryption:
        writer.encrypt(
            encryption["user_password"],
            encryption["owner_password"],
            algorithm=encryption["algorithm"],
        )
    destination = Path(args.output)
    require_extension(destination, ".pdf", "Edit output")

    def builder(temp_path: Path) -> None:
        with temp_path.open("wb") as handle:
            writer.write(handle)

    verification = write_pdf_atomic(
        destination,
        builder,
        sources=[source_path, job_path, *extra_sources],
        overwrite=args.overwrite,
        expected_pages=len(writer.pages),
        expect_encrypted=bool(encryption),
        password=encryption["user_password"] if encryption else None,
    )
    return success(
        "edit",
        output_path=str(destination.resolve()),
        counts={"pages": len(writer.pages), "operations": len(operations)},
        warnings=warnings,
        verification=verification,
    )


# ---------------------------------------------------------------------------
# Secure redaction: true content removal with layered verification.
# ---------------------------------------------------------------------------

REDACT_JOB_KEYS = {
    "schema_version",
    "targets",
    "image_policy",
    "annotation_policy",
    "fill_color",
    "strip_docinfo_keys",
    "strip_xmp",
    "strip_thumbnails",
    "strip_structure_tree",
    "acknowledge",
}
REDACT_TARGET_KEYS = {
    "type",
    "page",
    "rect",
    "coordinate_origin",
    "text",
    "pages",
    "match",
    "grow_points",
}
REDACT_ACKNOWLEDGE_KEYS = {"signatures_invalidated", "attachments_present"}
REDACT_VISUAL_DIFF_LIMIT = 0.0005
REDACT_MASK_DILATION_PIXELS = 3
REDACT_CHECK_DPI = 96
PATH_PAINT_OPERATORS = {b"S", b"s", b"f", b"F", b"f*", b"B", b"B*", b"b", b"b*"}


def identity_matrix() -> tuple[float, float, float, float, float, float]:
    return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def mat_mult(
    a: Sequence[float], b: Sequence[float]
) -> tuple[float, float, float, float, float, float]:
    return (
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
        a[4] * b[0] + a[5] * b[2] + b[4],
        a[4] * b[1] + a[5] * b[3] + b[5],
    )


def apply_matrix(m: Sequence[float], x: float, y: float) -> tuple[float, float]:
    return (m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5])


def transform_rect(m: Sequence[float], rect: Sequence[float]) -> tuple[float, float, float, float]:
    corners = [
        apply_matrix(m, rect[0], rect[1]),
        apply_matrix(m, rect[2], rect[1]),
        apply_matrix(m, rect[0], rect[3]),
        apply_matrix(m, rect[2], rect[3]),
    ]
    xs = [corner[0] for corner in corners]
    ys = [corner[1] for corner in corners]
    return (min(xs), min(ys), max(xs), max(ys))


def rects_overlap(a: Sequence[float], b: Sequence[float]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def quad_corners(m: Sequence[float], rect: Sequence[float]) -> list[tuple[float, float]]:
    return [
        apply_matrix(m, rect[0], rect[1]),
        apply_matrix(m, rect[2], rect[1]),
        apply_matrix(m, rect[2], rect[3]),
        apply_matrix(m, rect[0], rect[3]),
    ]


def convex_polygons_overlap(
    poly_a: Sequence[tuple[float, float]], poly_b: Sequence[tuple[float, float]]
) -> bool:
    """Separating-axis test for two convex polygons (touching edges do not count)."""
    for polygon in (poly_a, poly_b):
        count = len(polygon)
        for index in range(count):
            x1, y1 = polygon[index]
            x2, y2 = polygon[(index + 1) % count]
            axis_x, axis_y = -(y2 - y1), x2 - x1
            if axis_x == 0 and axis_y == 0:
                continue
            a_min = min(axis_x * px + axis_y * py for px, py in poly_a)
            a_max = max(axis_x * px + axis_y * py for px, py in poly_a)
            b_min = min(axis_x * px + axis_y * py for px, py in poly_b)
            b_max = max(axis_x * px + axis_y * py for px, py in poly_b)
            if a_max <= b_min or b_max <= a_min:
                return False
    return True


def quad_intersects_rect(corners: Sequence[tuple[float, float]], rect: Sequence[float]) -> bool:
    rect_polygon = [
        (rect[0], rect[1]),
        (rect[2], rect[1]),
        (rect[2], rect[3]),
        (rect[0], rect[3]),
    ]
    return convex_polygons_overlap(corners, rect_polygon)


def displayed_rect_to_user(
    rect: Sequence[float], reader_page: Any
) -> tuple[float, float, float, float]:
    _, displayed_height = displayed_page_size(reader_page)
    rotation = int(reader_page.rotation or 0) % 360
    box = reader_page.mediabox
    origin_x = float(box.left)
    origin_y = float(box.bottom)
    width = float(box.width)
    height = float(box.height)
    corners = []
    for x, y_top in (
        (rect[0], rect[1]),
        (rect[2], rect[1]),
        (rect[0], rect[3]),
        (rect[2], rect[3]),
    ):
        y_up = displayed_height - y_top
        if rotation == 90:
            a, b = width - y_up, x
        elif rotation == 180:
            a, b = width - x, height - y_up
        elif rotation == 270:
            a, b = y_up, height - x
        else:
            a, b = x, y_up
        corners.append((origin_x + a, origin_y + b))
    xs = [corner[0] for corner in corners]
    ys = [corner[1] for corner in corners]
    return (min(xs), min(ys), max(xs), max(ys))


def user_rect_to_displayed(
    rect: Sequence[float], reader_page: Any
) -> tuple[float, float, float, float]:
    rotation = int(reader_page.rotation or 0) % 360
    box = reader_page.mediabox
    origin_x = float(box.left)
    origin_y = float(box.bottom)
    width = float(box.width)
    height = float(box.height)
    corners = []
    for u, v in (
        (rect[0], rect[1]),
        (rect[2], rect[1]),
        (rect[0], rect[3]),
        (rect[2], rect[3]),
    ):
        a, b = u - origin_x, v - origin_y
        if rotation == 90:
            x, y_top = b, a
        elif rotation == 180:
            x, y_top = width - a, b
        elif rotation == 270:
            x, y_top = height - b, width - a
        else:
            x, y_top = a, height - b
        corners.append((x, y_top))
    xs = [corner[0] for corner in corners]
    ys = [corner[1] for corner in corners]
    return (min(xs), min(ys), max(xs), max(ys))


BASE14_STYLE_MAP = {
    ("helvetica", ""): "Helvetica",
    ("helvetica", "bold"): "Helvetica-Bold",
    ("helvetica", "italic"): "Helvetica-Oblique",
    ("helvetica", "bolditalic"): "Helvetica-BoldOblique",
    ("times", ""): "Times-Roman",
    ("times", "bold"): "Times-Bold",
    ("times", "italic"): "Times-Italic",
    ("times", "bolditalic"): "Times-BoldItalic",
    ("courier", ""): "Courier",
    ("courier", "bold"): "Courier-Bold",
    ("courier", "italic"): "Courier-Oblique",
    ("courier", "bolditalic"): "Courier-BoldOblique",
    ("symbol", ""): "Symbol",
    ("zapfdingbats", ""): "ZapfDingbats",
}


def canonical_base14_name(name: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9]", "", name.casefold())
    family = None
    remainder = ""
    for prefix, canonical in (
        ("helvetica", "helvetica"),
        ("arial", "helvetica"),
        ("timesnewroman", "times"),
        ("timesroman", "times"),
        ("times", "times"),
        ("couriernew", "courier"),
        ("courier", "courier"),
        ("zapfdingbats", "zapfdingbats"),
        ("symbol", "symbol"),
    ):
        if normalized.startswith(prefix):
            family = canonical
            remainder = normalized[len(prefix) :]
            break
    if family is None:
        return None
    if "bold" in remainder and ("italic" in remainder or "oblique" in remainder):
        style = "bolditalic"
    elif "bold" in remainder:
        style = "bold"
    elif "italic" in remainder or "oblique" in remainder:
        style = "italic"
    else:
        style = ""
    return BASE14_STYLE_MAP.get((family, style))


@dataclass
class InterpreterFont:
    resource_name: str
    subtype: str
    bytes_per_code: int
    widths: dict[int, float]
    default_width: float | None
    bbox_width: float
    ascent: float
    descent: float
    reliable: bool
    type3: bool


FALLBACK_INTERPRETER_FONT = InterpreterFont(
    resource_name="",
    subtype="",
    bytes_per_code=1,
    widths={},
    default_width=None,
    bbox_width=1000.0,
    ascent=900.0,
    descent=-300.0,
    reliable=False,
    type3=False,
)


def parse_cid_widths(w_array: Any) -> dict[int, float]:
    widths: dict[int, float] = {}
    items = [item.get_object() if hasattr(item, "get_object") else item for item in w_array]
    index = 0
    while index < len(items):
        try:
            first = int(items[index])
        except (TypeError, ValueError):
            break
        if index + 1 < len(items) and isinstance(items[index + 1], (list, ArrayObject)):
            for offset, value in enumerate(items[index + 1]):
                try:
                    widths[first + offset] = float(value)
                except (TypeError, ValueError):
                    continue
            index += 2
        elif index + 2 < len(items):
            try:
                last = int(items[index + 1])
                value = float(items[index + 2])
            except (TypeError, ValueError):
                break
            for code in range(first, min(last, first + 65_535) + 1):
                widths[code] = value
            index += 3
        else:
            break
    return widths


def build_interpreter_font(resource_name: str, font_reference: Any) -> InterpreterFont:
    try:
        font = font_reference.get_object()
    except Exception:
        return replace(FALLBACK_INTERPRETER_FONT, resource_name=resource_name)
    subtype = str(font.get("/Subtype", "")).lstrip("/")
    if subtype == "Type3":
        return replace(
            FALLBACK_INTERPRETER_FONT,
            resource_name=resource_name,
            subtype=subtype,
            type3=True,
        )
    widths: dict[int, float] = {}
    default_width: float | None = None
    bytes_per_code = 1
    reliable = True
    descriptor = None
    try:
        if subtype == "Type0":
            bytes_per_code = 2
            encoding = font.get("/Encoding")
            encoding_name = ""
            if encoding is not None:
                encoding_object = (
                    encoding.get_object() if hasattr(encoding, "get_object") else encoding
                )
                if not isinstance(encoding_object, dict):
                    encoding_name = str(encoding_object)
            if "Identity" not in encoding_name:
                reliable = False
            descendants = font.get("/DescendantFonts")
            if descendants is not None:
                descendant = descendants.get_object()[0].get_object()
                w_array = descendant.get("/W")
                if w_array is not None:
                    widths = parse_cid_widths(w_array.get_object())
                default_value = descendant.get("/DW")
                default_width = float(default_value) if default_value is not None else 1000.0
                descriptor = descendant.get("/FontDescriptor")
        else:
            first_char = font.get("/FirstChar")
            widths_array = font.get("/Widths")
            if widths_array is not None and first_char is not None:
                for offset, value in enumerate(widths_array.get_object()):
                    try:
                        widths[int(first_char) + offset] = float(
                            value.get_object() if hasattr(value, "get_object") else value
                        )
                    except (TypeError, ValueError):
                        continue
            else:
                base_name = SUBSET_PREFIX_PATTERN.sub(
                    "", str(font.get("/BaseFont", "")).lstrip("/")
                )
                canonical = canonical_base14_name(base_name)
                if canonical is not None:
                    try:
                        standard_font = pdfmetrics.getFont(canonical)
                        widths = {
                            code: float(width) for code, width in enumerate(standard_font.widths)
                        }
                    except Exception:
                        reliable = False
                else:
                    reliable = False
                encoding = font.get("/Encoding")
                if encoding is not None:
                    encoding_object = (
                        encoding.get_object() if hasattr(encoding, "get_object") else encoding
                    )
                    if isinstance(encoding_object, dict) and ("/Differences" in encoding_object):
                        reliable = False
            descriptor = font.get("/FontDescriptor")
            if descriptor is not None:
                missing = descriptor.get_object().get("/MissingWidth")
                if missing is not None:
                    default_width = float(missing)
    except Exception:
        reliable = False
    ascent, descent, bbox_width = 900.0, -300.0, 1000.0
    try:
        if descriptor is not None:
            descriptor_object = descriptor.get_object()
            ascent_value = descriptor_object.get("/Ascent")
            if ascent_value is not None and float(ascent_value):
                ascent = float(ascent_value)
            descent_value = descriptor_object.get("/Descent")
            if descent_value is not None and float(descent_value):
                descent = min(float(descent_value), 0.0)
            bbox = descriptor_object.get("/FontBBox")
            if bbox is not None:
                values = [float(item) for item in bbox.get_object()]
                bbox_width = max(abs(values[2] - values[0]), 1.0)
                ascent = max(ascent, values[3])
                descent = min(descent, values[1])
    except Exception:
        pass
    if not reliable:
        widths = {}
        default_width = None
    return InterpreterFont(
        resource_name=resource_name,
        subtype=subtype,
        bytes_per_code=bytes_per_code,
        widths=widths,
        default_width=default_width,
        bbox_width=bbox_width,
        ascent=ascent,
        descent=descent,
        reliable=reliable,
        type3=False,
    )


@dataclass
class InterpreterTextState:
    font: InterpreterFont | None = None
    font_size: float = 0.0
    char_spacing: float = 0.0
    word_spacing: float = 0.0
    horizontal_scale: float = 100.0
    leading: float = 0.0
    rise: float = 0.0
    render_mode: int = 0


@dataclass
class PageRedactionResult:
    text_operators_removed: int = 0
    characters_removed: int = 0
    inline_images_removed: int = 0
    image_xobjects_removed: int = 0
    vector_paints_removed: int = 0
    annotations_removed: int = 0
    removed_extents_user: list[tuple[float, float, float, float]] = dataclass_field(
        default_factory=list
    )
    removed_operand_bytes: list[bytes] = dataclass_field(default_factory=list)
    removed_decoded_text: list[str] = dataclass_field(default_factory=list)
    removed_image_hashes: list[str] = dataclass_field(default_factory=list)


def string_segment_bytes(value: Any) -> bytes:
    resolved = value.get_object() if hasattr(value, "get_object") else value
    if isinstance(resolved, ByteStringObject):
        return bytes(resolved)
    if isinstance(resolved, TextStringObject):
        original = getattr(resolved, "original_bytes", None)
        if original is not None:
            return bytes(original)
        return str(resolved).encode("latin-1", "ignore")
    if isinstance(resolved, bytes):
        return resolved
    return str(resolved).encode("latin-1", "ignore")


def iterate_codes(data: bytes, bytes_per_code: int) -> Iterable[int]:
    if bytes_per_code == 1:
        yield from data
        return
    for index in range(0, len(data) - 1, 2):
        yield (data[index] << 8) | data[index + 1]
    if len(data) % 2:
        yield data[-1] << 8


def show_operation_metrics(
    font: InterpreterFont,
    state: InterpreterTextState,
    segments: Sequence[Any],
) -> tuple[float, float, float, list[bytes], str, int]:
    horizontal = state.horizontal_scale / 100.0
    x = 0.0
    min_x = 0.0
    max_x = 0.0
    raw_segments: list[bytes] = []
    decoded_parts: list[str] = []
    glyphs = 0
    for segment in segments:
        resolved = segment.get_object() if hasattr(segment, "get_object") else segment
        if isinstance(resolved, (int, float)) and not isinstance(resolved, bool):
            x -= float(resolved) / 1000.0 * state.font_size * horizontal
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            continue
        raw = string_segment_bytes(resolved)
        raw_segments.append(raw)
        if font.bytes_per_code == 1:
            decoded_parts.append(raw.decode("cp1252", "ignore"))
        for code in iterate_codes(raw, font.bytes_per_code):
            glyphs += 1
            width_units = font.widths.get(code)
            if width_units is None:
                width_units = (
                    font.default_width if font.default_width is not None else font.bbox_width
                )
            advance = (
                width_units / 1000.0 * state.font_size
                + state.char_spacing
                + (state.word_spacing if code == 32 and font.bytes_per_code == 1 else 0.0)
            ) * horizontal
            x += advance
            min_x = min(min_x, x)
            max_x = max(max_x, x)
    return min_x, max_x, x, raw_segments, "".join(decoded_parts), glyphs


def page_font_map(writer_page: Any) -> dict[str, InterpreterFont]:
    fonts: dict[str, InterpreterFont] = {}
    try:
        resources = writer_page.get("/Resources")
        if resources is None:
            return fonts
        font_dict = resources.get_object().get("/Font")
        if font_dict is None:
            return fonts
        for name, reference in font_dict.get_object().items():
            fonts[str(name)] = build_interpreter_font(str(name), reference)
    except Exception:
        pass
    return fonts


def page_xobject_map(writer_page: Any) -> dict[str, dict[str, Any]]:
    xobjects: dict[str, dict[str, Any]] = {}
    try:
        resources = writer_page.get("/Resources")
        if resources is None:
            return xobjects
        xobject_dict = resources.get_object().get("/XObject")
        if xobject_dict is None:
            return xobjects
        for name, reference in xobject_dict.get_object().items():
            try:
                xobject = reference.get_object()
                subtype = str(xobject.get("/Subtype", ""))
                info: dict[str, Any] = {"subtype": subtype, "object": xobject}
                if subtype == "/Form":
                    bbox = xobject.get("/BBox")
                    matrix = xobject.get("/Matrix")
                    info["bbox"] = (
                        [float(item) for item in bbox.get_object()]
                        if bbox is not None
                        else [0.0, 0.0, 1.0, 1.0]
                    )
                    info["matrix"] = (
                        tuple(float(item) for item in matrix.get_object())
                        if matrix is not None
                        else identity_matrix()
                    )
                xobjects[str(name)] = info
            except Exception:
                continue
    except Exception:
        pass
    return xobjects


def redact_page_content(
    writer_page: Any,
    writer: PdfWriter,
    user_rects: list[tuple[float, float, float, float]],
    *,
    image_policy: str,
    page_number: int,
    fill_rects_user: list[tuple[float, float, float, float]],
    fill_rgb: tuple[float, float, float],
) -> PageRedactionResult:
    result = PageRedactionResult()
    contents = writer_page.get_contents()
    if contents is None:
        raise ToolError(
            "unsupported_operation",
            f"Page {page_number} has no content stream to redact",
        )
    fonts = page_font_map(writer_page)
    xobjects = page_xobject_map(writer_page)
    stream = ContentStream(contents, writer)
    operations = stream.operations
    if len(operations) > MAX_CONTENT_OPERATIONS:
        raise ToolError(
            "resource_limit",
            f"Page {page_number} content exceeds {MAX_CONTENT_OPERATIONS} operations",
        )
    new_operations: list[tuple[Any, bytes]] = []
    ctm = identity_matrix()
    stack: list[tuple[tuple[float, ...], InterpreterTextState]] = []
    text = InterpreterTextState()
    tm = identity_matrix()
    tlm = identity_matrix()
    clip_pending = False
    path_points: list[tuple[float, float]] = []
    removed_image_names: set[str] = set()

    def record_removed_text(
        extent: tuple[float, float, float, float],
        raw_segments: list[bytes],
        decoded: str,
        glyphs: int,
    ) -> None:
        result.text_operators_removed += 1
        result.characters_removed += glyphs
        result.removed_extents_user.append(extent)
        result.removed_operand_bytes.extend(
            segment for segment in raw_segments if len(segment) >= 3
        )
        cleaned = decoded.strip()
        if len(cleaned) >= 4:
            result.removed_decoded_text.append(cleaned)

    for operands, operator in operations:
        if operator == b"q":
            stack.append((ctm, replace(text)))
            new_operations.append((operands, operator))
            continue
        if operator == b"Q":
            if stack:
                ctm, text = stack.pop()
            new_operations.append((operands, operator))
            continue
        if operator == b"cm":
            try:
                matrix = tuple(float(value) for value in operands)
                ctm = mat_mult(matrix, ctm)
            except (TypeError, ValueError):
                pass
            new_operations.append((operands, operator))
            continue
        if operator == b"BT":
            tm = identity_matrix()
            tlm = identity_matrix()
            new_operations.append((operands, operator))
            continue
        if operator == b"ET":
            new_operations.append((operands, operator))
            continue
        if operator == b"Tf":
            text.font = fonts.get(str(operands[0]), FALLBACK_INTERPRETER_FONT)
            try:
                text.font_size = float(operands[1])
            except (TypeError, ValueError):
                text.font_size = 0.0
            new_operations.append((operands, operator))
            continue
        if operator == b"Tc":
            text.char_spacing = float(operands[0])
            new_operations.append((operands, operator))
            continue
        if operator == b"Tw":
            text.word_spacing = float(operands[0])
            new_operations.append((operands, operator))
            continue
        if operator == b"Tz":
            text.horizontal_scale = float(operands[0])
            new_operations.append((operands, operator))
            continue
        if operator == b"TL":
            text.leading = float(operands[0])
            new_operations.append((operands, operator))
            continue
        if operator == b"Ts":
            text.rise = float(operands[0])
            new_operations.append((operands, operator))
            continue
        if operator == b"Tr":
            try:
                text.render_mode = int(operands[0])
            except (TypeError, ValueError):
                text.render_mode = 0
            new_operations.append((operands, operator))
            continue
        if operator == b"Td":
            tlm = mat_mult((1, 0, 0, 1, float(operands[0]), float(operands[1])), tlm)
            tm = tlm
            new_operations.append((operands, operator))
            continue
        if operator == b"TD":
            text.leading = -float(operands[1])
            tlm = mat_mult((1, 0, 0, 1, float(operands[0]), float(operands[1])), tlm)
            tm = tlm
            new_operations.append((operands, operator))
            continue
        if operator == b"Tm":
            tlm = tuple(float(value) for value in operands)
            tm = tlm
            new_operations.append((operands, operator))
            continue
        if operator == b"T*":
            tlm = mat_mult((1, 0, 0, 1, 0, -text.leading), tlm)
            tm = tlm
            new_operations.append((operands, operator))
            continue
        if operator in (b"Tj", b"'", b'"', b"TJ"):
            if operator == b'"':
                text.word_spacing = float(operands[0])
                text.char_spacing = float(operands[1])
                string_operand = operands[2]
            elif operator == b"TJ":
                string_operand = None
            else:
                string_operand = operands[0]
            if operator in (b"'", b'"'):
                tlm = mat_mult((1, 0, 0, 1, 0, -text.leading), tlm)
                tm = tlm
            if operator == b"TJ":
                array = operands[0]
                segments = list(array.get_object() if hasattr(array, "get_object") else array)
            else:
                segments = [string_operand]
            font = text.font or FALLBACK_INTERPRETER_FONT
            min_x, max_x, net_advance, raw_segments, decoded, glyphs = show_operation_metrics(
                font, text, segments
            )
            y_low = font.descent / 1000.0 * text.font_size + text.rise
            y_high = font.ascent / 1000.0 * text.font_size + text.rise
            text_matrix = mat_mult(tm, ctm)
            text_box = (min_x, min(y_low, y_high), max_x, max(y_low, y_high))
            # Test the true oriented text quad, not its axis-aligned bounding box: a
            # rotated stamp's AABB is far larger than the glyphs and would trigger
            # spurious removal of unrelated content it merely spans. The quad is still the
            # conservative whole-operator bound, just not inflated by rotation.
            corners = quad_corners(text_matrix, text_box)
            extent = transform_rect(text_matrix, text_box)
            intersects = any(quad_intersects_rect(corners, rect) for rect in user_rects)
            if intersects:
                if font.type3:
                    raise ToolError(
                        "unsupported_operation",
                        f"Page {page_number} shows Type3 font text inside a redaction "
                        "target; Type3 glyph extents cannot be measured safely",
                    )
                if text.render_mode >= 4:
                    raise ToolError(
                        "unsupported_operation",
                        f"Page {page_number} uses text-clipping render mode inside a "
                        "redaction target; redact cannot rewrite text clips safely",
                    )
                if not font.reliable:
                    raise ToolError(
                        "unsupported_operation",
                        f"Page {page_number} text inside a redaction target uses a font "
                        "without measurable glyph widths; redact refuses rather than "
                        "guess extents",
                    )
                record_removed_text(extent, raw_segments, decoded, glyphs)
                if operator == b'"':
                    new_operations.append(([FloatObject(text.word_spacing)], b"Tw"))
                    new_operations.append(([FloatObject(text.char_spacing)], b"Tc"))
                if operator in (b"'", b'"'):
                    new_operations.append(([], b"T*"))
                horizontal = text.horizontal_scale / 100.0
                if abs(net_advance) > 1e-9 and text.font_size > 0 and horizontal > 0:
                    offset = -net_advance / horizontal / text.font_size * 1000.0
                    new_operations.append(([ArrayObject([FloatObject(offset)])], b"TJ"))
            else:
                new_operations.append((operands, operator))
            tm = mat_mult((1, 0, 0, 1, net_advance, 0), tm)
            continue
        if operator == b"INLINE IMAGE":
            extent = transform_rect(ctm, (0.0, 0.0, 1.0, 1.0))
            if any(rects_overlap(extent, rect) for rect in user_rects):
                if image_policy == "refuse":
                    raise ToolError(
                        "unsupported_operation",
                        f"Page {page_number} inline image intersects a redaction "
                        "target and image_policy is refuse",
                    )
                result.inline_images_removed += 1
                result.removed_extents_user.append(extent)
                continue
            new_operations.append((operands, operator))
            continue
        if operator == b"Do":
            name = str(operands[0])
            info = xobjects.get(name)
            if info is None:
                new_operations.append((operands, operator))
                continue
            if info["subtype"] == "/Image":
                extent = transform_rect(ctm, (0.0, 0.0, 1.0, 1.0))
                if any(rects_overlap(extent, rect) for rect in user_rects):
                    if image_policy == "refuse":
                        raise ToolError(
                            "unsupported_operation",
                            f"Page {page_number} image XObject {name} intersects a "
                            "redaction target and image_policy is refuse",
                        )
                    result.image_xobjects_removed += 1
                    result.removed_extents_user.append(extent)
                    removed_image_names.add(name)
                    try:
                        digest = hashlib.sha256(info["object"].get_data()).hexdigest()
                        result.removed_image_hashes.append(digest)
                    except Exception:
                        pass
                    continue
                new_operations.append((operands, operator))
                continue
            if info["subtype"] == "/Form":
                extent = transform_rect(mat_mult(info["matrix"], ctm), info["bbox"])
                if any(rects_overlap(extent, rect) for rect in user_rects):
                    raise ToolError(
                        "unsupported_operation",
                        f"Page {page_number} Form XObject {name} intersects a "
                        "redaction target; redaction inside Form XObjects is not "
                        "supported — flatten or rasterize the page first",
                    )
                new_operations.append((operands, operator))
                continue
            new_operations.append((operands, operator))
            continue
        if operator in (b"m", b"l"):
            try:
                path_points.append(apply_matrix(ctm, float(operands[0]), float(operands[1])))
            except (TypeError, ValueError, IndexError):
                pass
            new_operations.append((operands, operator))
            continue
        if operator == b"c":
            try:
                for index in range(0, 6, 2):
                    path_points.append(
                        apply_matrix(ctm, float(operands[index]), float(operands[index + 1]))
                    )
            except (TypeError, ValueError, IndexError):
                pass
            new_operations.append((operands, operator))
            continue
        if operator in (b"v", b"y"):
            try:
                for index in range(0, 4, 2):
                    path_points.append(
                        apply_matrix(ctm, float(operands[index]), float(operands[index + 1]))
                    )
            except (TypeError, ValueError, IndexError):
                pass
            new_operations.append((operands, operator))
            continue
        if operator == b"re":
            try:
                x0 = float(operands[0])
                y0 = float(operands[1])
                w = float(operands[2])
                h = float(operands[3])
                for corner_x, corner_y in (
                    (x0, y0),
                    (x0 + w, y0),
                    (x0, y0 + h),
                    (x0 + w, y0 + h),
                ):
                    path_points.append(apply_matrix(ctm, corner_x, corner_y))
            except (TypeError, ValueError, IndexError):
                pass
            new_operations.append((operands, operator))
            continue
        if operator in (b"W", b"W*"):
            clip_pending = True
            new_operations.append((operands, operator))
            continue
        if operator in PATH_PAINT_OPERATORS:
            if path_points and not clip_pending:
                xs = [point[0] for point in path_points]
                ys = [point[1] for point in path_points]
                extent = (min(xs), min(ys), max(xs), max(ys))
                if any(rects_overlap(extent, rect) for rect in user_rects):
                    result.vector_paints_removed += 1
                    result.removed_extents_user.append(extent)
                    path_points = []
                    new_operations.append(([], b"n"))
                    continue
            path_points = []
            clip_pending = False
            new_operations.append((operands, operator))
            continue
        if operator == b"n":
            path_points = []
            clip_pending = False
            new_operations.append((operands, operator))
            continue
        new_operations.append((operands, operator))

    for name in sorted(removed_image_names):
        still_used = any(
            operator == b"Do" and str(operands[0]) == name for operands, operator in new_operations
        )
        if not still_used:
            try:
                xobject_dict = writer_page["/Resources"].get_object()["/XObject"].get_object()
                if name in xobject_dict:
                    del xobject_dict[name]
            except Exception:
                pass
    for rect in fill_rects_user:
        new_operations.append(([], b"q"))
        new_operations.append(
            (
                [
                    FloatObject(fill_rgb[0]),
                    FloatObject(fill_rgb[1]),
                    FloatObject(fill_rgb[2]),
                ],
                b"rg",
            )
        )
        new_operations.append(
            (
                [
                    FloatObject(rect[0]),
                    FloatObject(rect[1]),
                    FloatObject(rect[2] - rect[0]),
                    FloatObject(rect[3] - rect[1]),
                ],
                b"re",
            )
        )
        new_operations.append(([], b"f"))
        new_operations.append(([], b"Q"))
    stream.operations = new_operations
    writer_page.replace_contents(stream)
    return result


def normalize_redaction_term(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def parse_redact_job(job: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(job) - REDACT_JOB_KEYS)
    if unknown:
        raise ToolError("bad_input", f"Unknown redact job keys: {', '.join(unknown)}")
    targets = job.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ToolError("bad_input", "Redact job requires a nonempty targets array")
    if len(targets) > MAX_REDACTION_TARGETS:
        raise ToolError("resource_limit", f"Redact job exceeds {MAX_REDACTION_TARGETS} targets")
    image_policy = job.get("image_policy", "remove")
    if image_policy not in {"remove", "refuse"}:
        raise ToolError("bad_input", "image_policy must be remove or refuse")
    annotation_policy = job.get("annotation_policy", "remove-intersecting")
    if annotation_policy != "remove-intersecting":
        raise ToolError("bad_input", "annotation_policy must be remove-intersecting")
    strip_docinfo = job.get("strip_docinfo_keys", "matched")
    if strip_docinfo not in {"matched", "all", "none"}:
        raise ToolError("bad_input", "strip_docinfo_keys must be matched, all, or none")
    strip_xmp = job.get("strip_xmp", "if-matched")
    if strip_xmp not in {"if-matched", "always", "never"}:
        raise ToolError("bad_input", "strip_xmp must be if-matched, always, or never")
    for flag_name in ("strip_thumbnails", "strip_structure_tree"):
        if flag_name in job and not isinstance(job[flag_name], bool):
            raise ToolError("bad_input", f"{flag_name} must be a boolean")
    acknowledge = job.get("acknowledge", {})
    if not isinstance(acknowledge, dict):
        raise ToolError("bad_input", "acknowledge must be an object")
    unknown_acknowledge = sorted(set(acknowledge) - REDACT_ACKNOWLEDGE_KEYS)
    if unknown_acknowledge:
        raise ToolError(
            "bad_input",
            f"Unknown acknowledge keys: {', '.join(unknown_acknowledge)}",
        )
    for key, value in acknowledge.items():
        if not isinstance(value, bool):
            raise ToolError("bad_input", f"acknowledge.{key} must be a boolean")
    fill = color_value(job.get("fill_color", "#000000"))
    return {
        "targets": targets,
        "image_policy": image_policy,
        "strip_docinfo_keys": strip_docinfo,
        "strip_xmp": strip_xmp,
        "strip_thumbnails": bool(job.get("strip_thumbnails", True)),
        "strip_structure_tree": bool(job.get("strip_structure_tree", False)),
        "acknowledge": {
            "signatures_invalidated": bool(acknowledge.get("signatures_invalidated")),
            "attachments_present": bool(acknowledge.get("attachments_present")),
        },
        "fill_rgb": (float(fill.red), float(fill.green), float(fill.blue)),
    }


def validate_redact_targets(targets: list[Any], page_count: int) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, target in enumerate(targets):
        if not isinstance(target, dict):
            raise ToolError("bad_input", "Every redact target must be an object")
        unknown = sorted(set(target) - REDACT_TARGET_KEYS)
        if unknown:
            raise ToolError("bad_input", f"Unknown redact target keys: {', '.join(unknown)}")
        kind = target.get("type")
        if kind == "rect":
            page = finite_integer(
                target.get("page"),
                f"targets[{index}].page",
                minimum=1,
                maximum=page_count,
            )
            rect = target.get("rect")
            if not isinstance(rect, (list, tuple)) or len(rect) != 4:
                raise ToolError(
                    "bad_input",
                    f"targets[{index}].rect must be [x, y, width, height]",
                )
            values = [
                finite_number(item, f"targets[{index}].rect[{position}]", minimum=0)
                for position, item in enumerate(rect)
            ]
            if values[2] <= 0 or values[3] <= 0:
                raise ToolError(
                    "bad_input", f"targets[{index}].rect requires positive width/height"
                )
            origin = target.get("coordinate_origin", "top-left")
            if origin not in {"top-left", "bottom-left"}:
                raise ToolError(
                    "bad_input",
                    f"targets[{index}].coordinate_origin must be top-left or bottom-left",
                )
            normalized.append(
                {
                    "kind": "rect",
                    "index": index,
                    "page": page,
                    "rect": tuple(values),
                    "origin": origin,
                }
            )
        elif kind == "text":
            text = target.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ToolError("bad_input", f"targets[{index}].text must be a nonempty string")
            match = target.get("match", "exact")
            if match not in {"exact", "case-insensitive"}:
                raise ToolError(
                    "bad_input",
                    f"targets[{index}].match must be exact or case-insensitive",
                )
            grow = finite_number(
                target.get("grow_points", 2.0),
                f"targets[{index}].grow_points",
                minimum=0,
                maximum=72,
            )
            pages_expression = target.get("pages", "all")
            if pages_expression is not None and not isinstance(pages_expression, str):
                raise ToolError("bad_input", f"targets[{index}].pages must be a string expression")
            normalized.append(
                {
                    "kind": "text",
                    "index": index,
                    "text": text,
                    "pages": pages_expression,
                    "match": match,
                    "grow": grow,
                }
            )
        else:
            raise ToolError("unsupported_operation", f"Unsupported redact target type: {kind}")
    return normalized


def resolve_redaction_rects(
    source: SourceInfo,
    plumber_pdf: Any,
    targets: list[dict[str, Any]],
    page_count: int,
) -> dict[int, list[dict[str, Any]]]:
    rects_by_page: dict[int, list[dict[str, Any]]] = {}

    def add_rect(page_number: int, rect: tuple[float, float, float, float], index: int) -> None:
        if rect[2] <= rect[0] or rect[3] <= rect[1]:
            raise ToolError(
                "bad_input",
                f"targets[{index}] resolves to an empty rectangle on page {page_number}",
            )
        rects_by_page.setdefault(page_number, []).append({"rect": rect, "target_index": index})

    for target in targets:
        if target["kind"] == "rect":
            page_number = target["page"]
            reader_page = source.reader.pages[page_number - 1]
            displayed_width, displayed_height = displayed_page_size(reader_page)
            x, y, width, height = target["rect"]
            top = displayed_height - y - height if target["origin"] == "bottom-left" else y
            rect = (
                min(max(x, 0.0), displayed_width),
                min(max(top, 0.0), displayed_height),
                min(max(x + width, 0.0), displayed_width),
                min(max(top + height, 0.0), displayed_height),
            )
            add_rect(page_number, rect, target["index"])
            continue
        term = normalize_redaction_term(target["text"])
        tokens = term.split(" ")
        case_insensitive = target["match"] == "case-insensitive"
        page_indexes = parse_page_selection(target["pages"], page_count)
        matches = 0
        for page_index in page_indexes:
            page = plumber_pdf.pages[page_index]
            displayed_width = float(page.width)
            displayed_height = float(page.height)
            try:
                words = page.extract_words()
            except Exception as exc:
                raise ToolError(
                    "unsupported_operation",
                    f"Cannot resolve text targets on page {page_index + 1}: {clean_text(exc)}",
                ) from exc
            for start in range(len(words)):
                window = words[start : start + len(tokens)]
                if len(window) < len(tokens):
                    continue
                joined = " ".join(str(word["text"]) for word in window)
                if case_insensitive:
                    matched = joined.casefold() == term.casefold()
                else:
                    matched = joined == term
                if not matched and len(tokens) == 1:
                    haystack = str(words[start]["text"])
                    if case_insensitive:
                        matched = term.casefold() in haystack.casefold()
                    else:
                        matched = term in haystack
                if not matched:
                    continue
                grow = target["grow"]
                rect = (
                    max(0.0, min(float(word["x0"]) for word in window) - grow),
                    max(0.0, min(float(word["top"]) for word in window) - grow),
                    min(
                        displayed_width,
                        max(float(word["x1"]) for word in window) + grow,
                    ),
                    min(
                        displayed_height,
                        max(float(word["bottom"]) for word in window) + grow,
                    ),
                )
                add_rect(page_index + 1, rect, target["index"])
                matches += 1
        if not matches:
            raise ToolError(
                "bad_input",
                f"Redaction text target {target['index']} matched nothing on its "
                "selected pages; nothing was redacted",
            )
    return rects_by_page


def collect_object_strings(value: Any, depth: int = 0, budget: int = 500) -> list[str]:
    if depth > 4 or budget <= 0:
        return []
    strings: list[str] = []
    try:
        resolved = value.get_object() if hasattr(value, "get_object") else value
    except Exception:
        return []
    if isinstance(resolved, str):
        return [str(resolved)]
    if isinstance(resolved, dict):
        for item in resolved.values():
            found = collect_object_strings(item, depth + 1, budget - len(strings))
            strings.extend(found)
            if len(strings) >= budget:
                break
    elif isinstance(resolved, (list, tuple)):
        for item in resolved:
            found = collect_object_strings(item, depth + 1, budget - len(strings))
            strings.extend(found)
            if len(strings) >= budget:
                break
    return strings


def term_in_text(term: str, text: str) -> bool:
    normalized_term = re.sub(r"\s+", "", term).casefold()
    normalized_text = re.sub(r"\s+", "", text).casefold()
    return bool(normalized_term) and normalized_term in normalized_text


def prune_acroform_fields(writer: PdfWriter, removed_references: list[Any]) -> None:
    """Detach removed widget annotations from the AcroForm field tree.

    Deleting a widget from a page's /Annots without pruning the AcroForm /Fields entry
    leaves an orphaned field that some viewers still render and whose value stays
    extractable. Remove the top-level field entries that point at removed widgets and
    clear their value.
    """
    removed_ids = {
        (reference.idnum, reference.generation)
        for reference in removed_references
        if isinstance(reference, IndirectObject)
    }
    if not removed_ids:
        return
    try:
        acroform = writer._root_object.get("/AcroForm")
        if acroform is None:
            return
        fields = acroform.get_object().get("/Fields")
        if fields is None:
            return
        field_array = fields.get_object()
    except Exception:
        return
    retained_fields = ArrayObject()
    for field_reference in field_array:
        drop = (
            isinstance(field_reference, IndirectObject)
            and (field_reference.idnum, field_reference.generation) in removed_ids
        )
        if not drop:
            try:
                field = field_reference.get_object()
                kids = field.get("/Kids")
                if kids is not None:
                    kid_ids = {
                        (kid.idnum, kid.generation)
                        for kid in kids.get_object()
                        if isinstance(kid, IndirectObject)
                    }
                    if kid_ids and kid_ids <= removed_ids:
                        drop = True
            except Exception:
                drop = False
        if drop:
            try:
                field_reference.get_object()[NameObject("/V")] = TextStringObject("")
            except Exception:
                pass
        else:
            retained_fields.append(field_reference)
    acroform.get_object()[NameObject("/Fields")] = retained_fields


def sweep_annotations(
    writer: PdfWriter,
    user_rects_by_page: dict[int, list[tuple[float, float, float, float]]],
    terms: list[str],
) -> dict[int, int]:
    removed_by_page: dict[int, int] = {}
    removed_references: list[Any] = []
    for page_index, page in enumerate(writer.pages):
        annotations = page.get("/Annots")
        if annotations is None:
            continue
        try:
            annotation_array = annotations.get_object()
        except Exception:
            continue
        retained = ArrayObject()
        removed = 0
        for annotation_reference in annotation_array:
            remove = False
            try:
                annotation = annotation_reference.get_object()
                rect_value = annotation.get("/Rect")
                if rect_value is not None:
                    raw = [float(item) for item in rect_value.get_object()]
                    annotation_rect = (
                        min(raw[0], raw[2]),
                        min(raw[1], raw[3]),
                        max(raw[0], raw[2]),
                        max(raw[1], raw[3]),
                    )
                    for user_rect in user_rects_by_page.get(page_index + 1, []):
                        if rects_overlap(annotation_rect, user_rect):
                            remove = True
                            break
                if not remove and terms:
                    for value in collect_object_strings(annotation):
                        if any(term_in_text(term, value) for term in terms):
                            remove = True
                            break
            except Exception:
                remove = False
            if remove:
                removed += 1
                removed_references.append(annotation_reference)
            else:
                retained.append(annotation_reference)
        if removed:
            removed_by_page[page_index + 1] = removed
            if retained:
                page[NameObject("/Annots")] = retained
            elif "/Annots" in page:
                del page[NameObject("/Annots")]
    prune_acroform_fields(writer, removed_references)
    return removed_by_page


def redact_document_level(
    writer: PdfWriter,
    config: dict[str, Any],
    terms: list[str],
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "docinfo_keys_stripped": [],
        "xmp_stripped": False,
        "structure_tree": "absent",
        "thumbnails_stripped": 0,
    }
    metadata = dict(writer.metadata or {})
    if config["strip_docinfo_keys"] == "all":
        if metadata:
            evidence["docinfo_keys_stripped"] = sorted(str(key) for key in metadata)
            writer.metadata = None
    elif metadata:
        matched_keys = [
            key
            for key, value in metadata.items()
            if any(term_in_text(term, str(value)) for term in terms)
        ]
        if matched_keys:
            if config["strip_docinfo_keys"] == "none":
                raise ToolError(
                    "unsupported_operation",
                    "Document information values contain redaction terms and "
                    "strip_docinfo_keys is none",
                )
            for key in matched_keys:
                metadata.pop(key, None)
            evidence["docinfo_keys_stripped"] = sorted(str(key) for key in matched_keys)
            writer.metadata = metadata
    root = writer._root_object
    metadata_stream = root.get("/Metadata")
    if metadata_stream is not None:
        xmp_text = ""
        try:
            xmp_text = metadata_stream.get_object().get_data().decode("utf-8", "ignore")
        except Exception:
            xmp_text = ""
        xmp_matched = any(term_in_text(term, xmp_text) for term in terms)
        if config["strip_xmp"] == "always" or (config["strip_xmp"] == "if-matched" and xmp_matched):
            del root[NameObject("/Metadata")]
            evidence["xmp_stripped"] = True
        elif xmp_matched:
            raise ToolError(
                "unsupported_operation",
                "XMP metadata contains redaction terms and strip_xmp is never",
            )
    if config["strip_thumbnails"]:
        for page in writer.pages:
            if "/Thumb" in page:
                del page[NameObject("/Thumb")]
                evidence["thumbnails_stripped"] += 1
    structure_root = root.get("/StructTreeRoot")
    if structure_root is not None:
        evidence["structure_tree"] = "present"
        structure_matched = False
        if terms:
            for value in collect_object_strings(structure_root, budget=2000):
                if any(term_in_text(term, value) for term in terms):
                    structure_matched = True
                    break
        if structure_matched:
            if not config["strip_structure_tree"]:
                raise ToolError(
                    "unsupported_operation",
                    "The structure tree contains redaction terms; set "
                    "strip_structure_tree true to remove the whole structure tree",
                )
            del root[NameObject("/StructTreeRoot")]
            if "/MarkInfo" in root:
                del root[NameObject("/MarkInfo")]
            evidence["structure_tree"] = "stripped"
    outlines = root.get("/Outlines")
    if outlines is not None and terms:
        for value in collect_object_strings(outlines, budget=2000):
            if any(term_in_text(term, value) for term in terms):
                raise ToolError(
                    "unsupported_operation",
                    "Document outlines contain redaction terms; remove or rewrite "
                    "the outlines before redacting",
                )
    names = root.get("/Names")
    if names is not None and terms:
        names_object = names.get_object()
        destinations = names_object.get("/Dests")
        if destinations is not None:
            for value in collect_object_strings(destinations, budget=2000):
                if any(term_in_text(term, value) for term in terms):
                    raise ToolError(
                        "unsupported_operation",
                        "Named destinations contain redaction terms; remove them before redacting",
                    )
    return evidence


def redact_preconditions(
    source: SourceInfo,
    config: dict[str, Any],
    terms: list[str],
) -> dict[str, Any]:
    forms = form_inventory(source.reader)
    if forms["has_xfa"]:
        raise ToolError(
            "unsupported_operation",
            "XFA forms can duplicate page content in XML; redaction of XFA "
            "documents is not supported",
        )
    root = source.reader.trailer.get("/Root", {})
    acroform = root.get("/AcroForm") if hasattr(root, "get") else None
    signatures_present = False
    if acroform is not None:
        acroform_object = acroform.get_object()
        sig_flags = acroform_object.get("/SigFlags")
        if sig_flags:
            signatures_present = True
    fields = source.reader.get_fields() or {}
    for name, field_value in fields.items():
        if str(field_value.get("/FT")) == "/Sig" and field_value.get("/V") is not None:
            signatures_present = True
        value = field_value.get("/V")
        if value is not None and terms:
            if any(term_in_text(term, str(value)) for term in terms):
                raise ToolError(
                    "unsupported_operation",
                    f"Form field {name} contains a redaction term; clear it with an "
                    "edit fill_form operation before redacting",
                )
    if signatures_present and not config["acknowledge"]["signatures_invalidated"]:
        raise ToolError(
            "unsupported_operation",
            "The document carries digital signatures that redaction invalidates; "
            "set acknowledge.signatures_invalidated true to proceed",
        )
    attachments_present = False
    names = root.get("/Names") if hasattr(root, "get") else None
    if names is not None:
        try:
            if names.get_object().get("/EmbeddedFiles") is not None:
                attachments_present = True
        except Exception:
            attachments_present = False
    if attachments_present and not config["acknowledge"]["attachments_present"]:
        raise ToolError(
            "unsupported_operation",
            "The document carries embedded files that redaction does not scan; "
            "set acknowledge.attachments_present true to proceed",
        )
    return {
        "signatures_present": signatures_present,
        "attachments_present": attachments_present,
    }


def redact_residual_scan(
    output_bytes: bytes,
    output_reader: PdfReader,
    full_scope_terms: list[str],
    partial_scope_terms: list[str],
    removed_operand_bytes: list[bytes],
) -> dict[str, Any]:
    def term_encodings(term: str) -> list[bytes]:
        compact = re.sub(r"\s+", " ", term.strip())
        variants = {compact, compact.lower(), compact.upper()}
        encoded: list[bytes] = []
        for variant in variants:
            encoded.append(variant.encode("utf-8"))
            encoded.append(variant.encode("utf-16-be"))
            encoded.append(variant.encode("latin-1", "ignore"))
        return [item for item in encoded if len(item) >= 3]

    stream_datas: list[bytes] = []
    size = int(output_reader.trailer.get("/Size", 0))
    for object_number in range(1, min(size, 50_000)):
        try:
            candidate = IndirectObject(object_number, 0, output_reader).get_object()
        except Exception:
            continue
        if hasattr(candidate, "get_data"):
            try:
                stream_datas.append(candidate.get_data())
            except Exception:
                continue
    raw_hits = 0
    stream_hits = 0
    hard_failures: list[str] = []
    for term in full_scope_terms + partial_scope_terms:
        is_full_scope = term in full_scope_terms
        for encoded in term_encodings(term):
            in_raw = encoded in output_bytes
            in_stream = any(encoded in data for data in stream_datas)
            if in_raw:
                raw_hits += 1
            if in_stream:
                stream_hits += 1
            if is_full_scope and (in_raw or in_stream):
                hard_failures.append(term)
                break
    operand_hits = 0
    for operand in removed_operand_bytes:
        if len(operand) < 4:
            continue
        if operand in output_bytes or any(operand in data for data in stream_datas):
            operand_hits += 1
    if hard_failures:
        raise ToolError(
            "validation_failed",
            "Redaction residual scan found target terms in the output",
            details={
                "terms_sha256": [
                    hashlib.sha256(term.encode("utf-8")).hexdigest() for term in hard_failures
                ]
            },
        )
    return {
        "streams_scanned": len(stream_datas),
        "raw_byte_hits": raw_hits,
        "decompressed_stream_hits": stream_hits,
        "operand_byte_matches_elsewhere": operand_hits,
    }


def handle_redact(args: argparse.Namespace) -> dict[str, Any]:
    source_path = Path(args.input)
    require_extension(source_path, ".pdf", "Redact input")
    job_path = Path(args.job)
    job = read_json(job_path)
    output = Path(args.output)
    require_extension(output, ".pdf", "Redact output")
    report_path = Path(args.report) if args.report else None
    if report_path is not None:
        require_extension(report_path, ".json", "Redact report")
    render_check_dir = Path(args.render_check_dir) if args.render_check_dir else None
    try:
        source = open_pdf(source_path, None)
    except ToolError as exc:
        if "encrypted" in exc.message.lower():
            raise ToolError(
                "unsupported_operation",
                "Redaction of encrypted PDFs is not supported; decrypt explicitly "
                "with an edit decrypt operation first",
            ) from exc
        raise
    config = parse_redact_job(job)
    page_count = len(source.reader.pages)
    targets = validate_redact_targets(config["targets"], page_count)
    text_targets = [target for target in targets if target["kind"] == "text"]
    terms = [normalize_redaction_term(target["text"]) for target in text_targets]
    full_scope_terms = [
        normalize_redaction_term(target["text"])
        for target in text_targets
        if target["pages"] in (None, "", "all")
    ]
    partial_scope_terms = [term for term in terms if term not in full_scope_terms]
    precondition_evidence = redact_preconditions(source, config, terms)
    with pdfplumber.open(str(source_path)) as plumber_pdf:
        rects_by_page = resolve_redaction_rects(source, plumber_pdf, targets, page_count)
    writer = PdfWriter()
    writer.clone_document_from_reader(source.reader)
    warnings: list[str] = [
        "Redaction is not forensic file sanitization; prior copies, backups, "
        "version control history, and filesystem artifacts are outside this tool.",
    ]
    page_evidence: dict[int, PageRedactionResult] = {}
    user_rects_by_page: dict[int, list[tuple[float, float, float, float]]] = {}
    displayed_rects_by_page: dict[int, list[tuple[float, float, float, float]]] = {}
    classifications: dict[int, str] = {}
    for page_number, page_rects in sorted(rects_by_page.items()):
        reader_page = source.reader.pages[page_number - 1]
        writer_page = writer.pages[page_number - 1]
        displayed_rects = [item["rect"] for item in page_rects]
        user_rects = [displayed_rect_to_user(rect, reader_page) for rect in displayed_rects]
        user_rects_by_page[page_number] = user_rects
        displayed_rects_by_page[page_number] = displayed_rects
        result = redact_page_content(
            writer_page,
            writer,
            user_rects,
            image_policy=config["image_policy"],
            page_number=page_number,
            fill_rects_user=user_rects,
            fill_rgb=config["fill_rgb"],
        )
        page_evidence[page_number] = result
    with pdfplumber.open(str(source_path)) as plumber_pdf:
        for page_number in rects_by_page:
            page = plumber_pdf.pages[page_number - 1]
            try:
                page_data = plumber_page_data(
                    page, mode="plain", include_words=False, include_tables=False
                )
                classifications[page_number] = page_data["classification"]
            except Exception:
                classifications[page_number] = "unknown"
    for page_number, classification in classifications.items():
        if classification in {"likely-scanned", "hybrid"} and terms:
            warnings.append(
                f"Page {page_number} is {classification}: text targets match only "
                "extractable digital text, not pixels. Redact scanned content with "
                "rect targets (build them from OCR sidecar geometry if needed)."
            )
    # Sweep annotations for the operator's actual targets only, mirroring the
    # document-level sweep. Collateral text removed by the whole-operator policy is not a
    # redaction target and must not drive further removal (a stamp reading "APPROVED"
    # collaterally removed near a target must never delete a field named "approved_by").
    annotations_removed = sweep_annotations(writer, user_rects_by_page, terms)
    for page_number, count in annotations_removed.items():
        if page_number in page_evidence:
            page_evidence[page_number].annotations_removed = count
    document_evidence = redact_document_level(writer, config, terms)
    check_destination(output, sources=[source_path, job_path], overwrite=args.overwrite)
    removed_operand_bytes = [
        operand for result in page_evidence.values() for operand in result.removed_operand_bytes
    ]
    removed_image_hashes = {
        digest for result in page_evidence.values() for digest in result.removed_image_hashes
    }
    residual_scan: dict[str, Any] = {}
    visual_check: dict[str, Any] = {"checked": False}
    render_outputs: list[str] = []
    with temporary_sibling(output, ".pdf") as temp_path:
        intermediate = io.BytesIO()
        writer.write(intermediate)
        intermediate.seek(0)
        try:
            intermediate_reader = PdfReader(intermediate, strict=True)
        except Exception as exc:
            raise ToolError(
                "validation_failed",
                f"Redacted document failed to reopen: {clean_text(exc)}",
            ) from exc
        final_writer = PdfWriter()
        final_writer.clone_document_from_reader(intermediate_reader)
        with temp_path.open("wb") as handle:
            final_writer.write(handle)
        verification = validate_pdf_output(temp_path, expected_pages=page_count)
        output_bytes = temp_path.read_bytes()
        if output_bytes.count(b"%%EOF") != 1:
            raise ToolError(
                "validation_failed",
                "Redacted output does not contain exactly one revision",
            )
        output_reader = PdfReader(str(temp_path), strict=True)
        for index in range(page_count):
            source_box = source.reader.pages[index].mediabox
            output_box = output_reader.pages[index].mediabox
            if (
                abs(float(output_box.width) - float(source_box.width)) > 0.01
                or abs(float(output_box.height) - float(source_box.height)) > 0.01
            ):
                raise ToolError(
                    "validation_failed",
                    f"Page {index + 1} geometry changed during redaction",
                )
        for digest in removed_image_hashes:
            size = int(output_reader.trailer.get("/Size", 0))
            for object_number in range(1, min(size, 50_000)):
                try:
                    candidate = IndirectObject(object_number, 0, output_reader).get_object()
                except Exception:
                    continue
                if hasattr(candidate, "get_data"):
                    try:
                        if hashlib.sha256(candidate.get_data()).hexdigest() == digest:
                            raise ToolError(
                                "validation_failed",
                                "A removed image stream is still reachable in the redacted output",
                            )
                    except ToolError:
                        raise
                    except Exception:
                        continue
        with pdfplumber.open(str(temp_path)) as plumber_output:
            for page_number, displayed_rects in displayed_rects_by_page.items():
                page = plumber_output.pages[page_number - 1]
                try:
                    characters = page.chars
                except Exception:
                    characters = []
                for rect in displayed_rects:
                    inset = 0.5
                    x0, top, x1, bottom = (
                        rect[0] + inset,
                        rect[1] + inset,
                        rect[2] - inset,
                        rect[3] - inset,
                    )
                    residual_characters = [
                        character
                        for character in characters
                        if x0 < (float(character["x0"]) + float(character["x1"])) / 2 < x1
                        and top
                        < (float(character["top"]) + float(character["bottom"])) / 2
                        < bottom
                    ]
                    if residual_characters:
                        raise ToolError(
                            "validation_failed",
                            f"Page {page_number} still extracts "
                            f"{len(residual_characters)} characters inside a "
                            "redaction target",
                        )
            for target in text_targets:
                term = normalize_redaction_term(target["text"])
                page_indexes = parse_page_selection(target["pages"], page_count)
                for page_index in page_indexes:
                    page = plumber_output.pages[page_index]
                    try:
                        page_text = page.extract_text() or ""
                    except Exception:
                        page_text = ""
                    if term_in_text(term, page_text):
                        raise ToolError(
                            "validation_failed",
                            f"Redaction term for target {target['index']} is still "
                            f"extractable on page {page_index + 1}",
                        )
        residual_scan = redact_residual_scan(
            output_bytes,
            output_reader,
            full_scope_terms,
            partial_scope_terms,
            removed_operand_bytes,
        )
        if partial_scope_terms and (
            residual_scan["raw_byte_hits"] or residual_scan["decompressed_stream_hits"]
        ):
            warnings.append(
                "Byte-level scan found matches for page-scoped terms; instances "
                "outside the selected pages were intentionally retained."
            )
        visual_budget = RenderBudget(
            total_message=(
                "Redaction visual check exceeds the render pixel budget; redact "
                "fewer pages per invocation"
            )
        )
        max_outside_fraction = 0.0
        with tempfile.TemporaryDirectory(prefix="pdf-redact-check-") as check_name:
            check_dir = Path(check_name)
            with (
                pdfplumber.open(str(source_path)) as before_pdf,
                pdfplumber.open(str(temp_path)) as after_pdf,
            ):
                for page_number in sorted(displayed_rects_by_page):
                    reader_page = source.reader.pages[page_number - 1]
                    before_page = before_pdf.pages[page_number - 1]
                    after_page = after_pdf.pages[page_number - 1]
                    for _ in range(2):
                        visual_budget.charge(
                            float(before_page.width),
                            float(before_page.height),
                            REDACT_CHECK_DPI,
                            page_label=f"Page {page_number}",
                        )
                    before_png = check_dir / f"before-{page_number:04d}.png"
                    after_png = check_dir / f"after-{page_number:04d}.png"
                    rasterize_region(
                        before_page,
                        None,
                        REDACT_CHECK_DPI,
                        before_png,
                        page_label=f"page {page_number}",
                    )
                    rasterize_region(
                        after_page,
                        None,
                        REDACT_CHECK_DPI,
                        after_png,
                        page_label=f"page {page_number}",
                    )
                    scale = REDACT_CHECK_DPI / 72.0
                    with (
                        Image.open(before_png) as before_image,
                        Image.open(after_png) as after_image,
                    ):
                        before_rgb = before_image.convert("RGB")
                        after_rgb = after_image.convert("RGB")
                        if before_rgb.size != after_rgb.size:
                            raise ToolError(
                                "validation_failed",
                                f"Page {page_number} raster size changed during redaction",
                            )
                        mask = Image.new("1", before_rgb.size, 0)
                        mask_draw = ImageDraw.Draw(mask)
                        expected_rects = list(displayed_rects_by_page[page_number])
                        evidence = page_evidence.get(page_number)
                        if evidence is not None:
                            expected_rects.extend(
                                user_rect_to_displayed(extent, reader_page)
                                for extent in evidence.removed_extents_user
                            )
                        for rect in expected_rects:
                            mask_draw.rectangle(
                                (
                                    int(rect[0] * scale) - REDACT_MASK_DILATION_PIXELS,
                                    int(rect[1] * scale) - REDACT_MASK_DILATION_PIXELS,
                                    int(rect[2] * scale) + REDACT_MASK_DILATION_PIXELS,
                                    int(rect[3] * scale) + REDACT_MASK_DILATION_PIXELS,
                                ),
                                fill=1,
                            )
                        difference = ImageChops.difference(before_rgb, after_rgb).convert("L")
                        threshold = difference.point(lambda v: 255 if v > 8 else 0)
                        outside = ImageChops.subtract(
                            threshold.convert("L"),
                            mask.convert("L").point(lambda v: 255 if v else 0),
                        )
                        outside_pixels = sum(outside.histogram()[1:])
                        total_pixels = before_rgb.size[0] * before_rgb.size[1]
                        fraction = outside_pixels / total_pixels if total_pixels else 0.0
                        max_outside_fraction = max(max_outside_fraction, fraction)
                        if fraction > REDACT_VISUAL_DIFF_LIMIT:
                            raise ToolError(
                                "validation_failed",
                                f"Page {page_number} changed visually outside the "
                                f"expected redaction regions (fraction {fraction:.6f})",
                            )
                if render_check_dir is not None:
                    check_directory_destination(
                        render_check_dir,
                        overwrite=args.overwrite,
                        exists_message=(f"Render check directory exists: {render_check_dir}"),
                        not_directory_message=(
                            f"Render check output must be a directory: {render_check_dir}"
                        ),
                        nonempty_message=(
                            "Refusing to overwrite a nonempty render check directory"
                        ),
                    )
                    staged_dir = stage_directory(render_check_dir)
                    try:
                        for item in sorted(check_dir.iterdir()):
                            shutil.copy2(item, staged_dir / item.name)
                            render_outputs.append(str((render_check_dir / item.name).resolve()))
                        atomic_publish_directory(staged_dir, render_check_dir, label="render-check")
                    finally:
                        shutil.rmtree(staged_dir, ignore_errors=True)
        visual_check = {
            "checked": True,
            "dpi": REDACT_CHECK_DPI,
            "max_fraction_changed_outside_expected": round(max_outside_fraction, 6),
            "threshold": REDACT_VISUAL_DIFF_LIMIT,
        }
        atomic_publish(temp_path, output)
    target_reports = []
    for target in targets:
        resolved = [
            {"page": page_number, "rect": [round(value, 3) for value in item["rect"]]}
            for page_number, items in sorted(rects_by_page.items())
            for item in items
            if item["target_index"] == target["index"]
        ]
        entry: dict[str, Any] = {
            "index": target["index"],
            "type": target["kind"],
            "resolved_rects": resolved,
        }
        if target["kind"] == "text":
            entry["text_sha256"] = hashlib.sha256(
                normalize_redaction_term(target["text"]).encode("utf-8")
            ).hexdigest()
            entry["match"] = target["match"]
        target_reports.append(entry)
    removal_evidence = {
        "pages": [
            {
                "page": page_number,
                "classification": classifications.get(page_number),
                "text_operators_removed": evidence.text_operators_removed,
                "characters_removed": evidence.characters_removed,
                "inline_images_removed": evidence.inline_images_removed,
                "image_xobjects_removed": evidence.image_xobjects_removed,
                "vector_paints_removed": evidence.vector_paints_removed,
                "annotations_removed": evidence.annotations_removed,
            }
            for page_number, evidence in sorted(page_evidence.items())
        ],
        **document_evidence,
        "signatures_present": precondition_evidence["signatures_present"],
        "attachments_present": precondition_evidence["attachments_present"],
        "incremental_history_dropped": True,
    }
    payload = success(
        "redact",
        output_path=str(output.resolve()),
        source=str(source_path.resolve()),
        targets=target_reports,
        removal_evidence=removal_evidence,
        residual_scan=residual_scan,
        render_check_outputs=render_outputs,
        warnings=warnings,
        verification={
            **verification,
            "page_count_unchanged": True,
            "dimensions_unchanged": True,
            "single_revision": True,
            "visual_diff": visual_check,
        },
    )
    if report_path is not None:
        write_json_atomic(
            report_path,
            payload,
            overwrite=args.overwrite,
            protected_sources=[source_path, job_path],
        )
        payload["report_path"] = str(report_path.resolve())
    return payload


def pdf_to_text_or_json(args: argparse.Namespace, to_format: str) -> dict[str, Any]:
    password = resolve_password(args.password, args.password_env)
    data = inspect_document(
        Path(args.input),
        password,
        args.pages,
        mode=args.mode,
        include_words=to_format == "json" and args.words,
        include_tables=to_format == "json" and args.tables,
    )
    destination = Path(args.output)
    if to_format == "json":
        if destination.suffix.lower() != ".json":
            raise ToolError("bad_input", "PDF-to-JSON output must use a .json extension")
        write_json_atomic(
            destination,
            data,
            overwrite=args.overwrite,
            protected_sources=[Path(args.input)],
        )
    else:
        if destination.suffix.lower() != ".txt":
            raise ToolError("bad_input", "PDF-to-text output must use a .txt extension")
        text = "\n\n".join(f"--- Page {page['page']} ---\n{page['text']}" for page in data["pages"])
        write_text_atomic(
            destination,
            text + "\n",
            overwrite=args.overwrite,
            protected_sources=[Path(args.input)],
        )
    return success(
        "convert",
        mapping=f"pdf-to-{to_format}",
        output_path=str(destination.resolve()),
        counts={"pages": len(data["pages"])},
        warnings=["PDF text conversion is lossy; reading order is heuristic."],
        verification={"valid": True, "reopened": True},
    )


def pdf_tables_to_csv(args: argparse.Namespace) -> dict[str, Any]:
    password = resolve_password(args.password, args.password_env)
    data = inspect_document(
        Path(args.input),
        password,
        args.pages,
        mode="plain",
        include_words=False,
        include_tables=True,
    )
    output_dir = Path(args.output)
    output_existed = output_dir.exists()
    if output_existed:
        if not args.overwrite:
            raise ToolError("bad_input", f"Output directory exists: {output_dir}")
        if output_dir.is_symlink() or not output_dir.is_dir() or any(output_dir.iterdir()):
            raise ToolError("bad_input", "CSV output directory must be empty when overwritten")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent.resolve())
    )
    outputs = []
    try:
        for page in data["pages"]:
            for table_index, table_data in enumerate(page.get("tables", []), start=1):
                name = f"page-{page['page']:04d}-table-{table_index:04d}.csv"
                path = staged_dir / name
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle, lineterminator="\n")
                    writer.writerows(
                        [["" if cell is None else cell for cell in row] for row in table_data]
                    )
                path.read_text(encoding="utf-8")
                outputs.append(str((output_dir / name).resolve()))
        try:
            os.replace(staged_dir, output_dir.resolve())
        except OSError as exc:
            raise ToolError(
                "validation_failed",
                f"Atomic CSV-directory publish failed: {clean_text(exc)}",
            ) from exc
    finally:
        shutil.rmtree(staged_dir, ignore_errors=True)
    return success(
        "convert",
        mapping="pdf-tables-to-csv",
        output_path=str(output_dir.resolve()),
        outputs=outputs,
        counts={"tables": len(outputs)},
        warnings=["Table boundaries and cell order are heuristic; verify every CSV."],
        verification={"valid": True, "files_reopened": len(outputs)},
    )


def text_to_pdf(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input)
    require_regular_file(input_path)
    try:
        text = input_path.read_text(encoding=args.encoding)
    except (OSError, UnicodeError) as exc:
        raise ToolError("bad_input", f"Cannot decode text input: {clean_text(exc)}") from exc
    if len(text) > MAX_EXTRACTED_CHARS:
        raise ToolError("resource_limit", "Text input is too large")
    destination = Path(args.output)
    if destination.suffix.lower() != ".pdf":
        raise ToolError("bad_input", "Text-to-PDF output must use a .pdf extension")
    page_size = page_size_from_spec(args.page_size)

    def builder(temp_path: Path) -> None:
        doc = SimpleDocTemplate(
            str(temp_path),
            pagesize=page_size,
            leftMargin=54,
            rightMargin=54,
            topMargin=54,
            bottomMargin=54,
        )
        style = getSampleStyleSheet()["BodyText"]
        flowables = []
        for paragraph in re.split(r"\n\s*\n", text):
            escaped = (
                paragraph.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br/>")
            )
            flowables.append(Paragraph(escaped, style))
            flowables.append(Spacer(1, 8))
        doc.build(flowables)

    verification = write_pdf_atomic(
        destination,
        builder,
        sources=[input_path],
        overwrite=args.overwrite,
    )
    return success(
        "convert",
        mapping="text-to-pdf",
        output_path=str(destination.resolve()),
        warnings=["Text-to-PDF conversion uses a simple flowing layout."],
        verification=verification,
    )


def images_to_pdf(args: argparse.Namespace) -> dict[str, Any]:
    paths = [Path(path) for path in args.images]
    if not paths:
        raise ToolError("bad_input", "images-to-pdf requires at least one --images path")
    if len(paths) > MAX_PAGES:
        raise ToolError("resource_limit", f"Image count exceeds {MAX_PAGES}")
    for path in paths:
        validate_image(path)
    destination = Path(args.output)
    if destination.suffix.lower() != ".pdf":
        raise ToolError("bad_input", "Images-to-PDF output must use a .pdf extension")
    page_size = page_size_from_spec(args.page_size)
    margin = finite_number(args.margin, "margin", minimum=0)
    if 2 * margin >= min(page_size):
        raise ToolError("bad_input", "Margin must leave positive drawable page geometry")

    def builder(temp_path: Path) -> None:
        target = canvas.Canvas(str(temp_path), pagesize=page_size, pageCompression=1)
        page_width, page_height = page_size
        for path in paths:
            width, height = validate_image(path)
            scale = min((page_width - 2 * margin) / width, (page_height - 2 * margin) / height)
            draw_width, draw_height = width * scale, height * scale
            target.drawImage(
                ImageReader(str(path)),
                (page_width - draw_width) / 2,
                (page_height - draw_height) / 2,
                width=draw_width,
                height=draw_height,
                preserveAspectRatio=True,
                mask="auto",
            )
            target.showPage()
        target.save()

    verification = write_pdf_atomic(
        destination,
        builder,
        sources=paths,
        overwrite=args.overwrite,
        expected_pages=len(paths),
    )
    return success(
        "convert",
        mapping="images-to-pdf",
        output_path=str(destination.resolve()),
        counts={"pages": len(paths), "images": len(paths)},
        warnings=["Each raster image becomes one fitted PDF page."],
        verification=verification,
    )


def handle_convert(args: argparse.Namespace) -> dict[str, Any]:
    if args.to in {"txt", "json", "tables-csv"}:
        if not args.input or Path(args.input).suffix.lower() != ".pdf":
            raise ToolError("bad_input", f"{args.to} conversion requires one PDF --input")
        if args.to == "tables-csv":
            return pdf_tables_to_csv(args)
        return pdf_to_text_or_json(args, args.to)
    if args.to == "pdf":
        if args.images:
            if args.input:
                raise ToolError("ambiguous_edit", "Use --input or --images, not both")
            return images_to_pdf(args)
        if not args.input:
            raise ToolError("bad_input", "Text-to-PDF requires --input")
        return text_to_pdf(args)
    raise ToolError("unsupported_operation", f"Unsupported conversion: {args.to}")


def executable_version(command: str, engine: str) -> tuple[str, Path]:
    resolved = shutil.which(command)
    if not resolved:
        path = Path(command)
        if not path.exists():
            raise ToolError("missing_dependency", f"{engine} executable is unavailable: {command}")
        resolved = str(path.resolve())
    probes = {
        "tesseract": [[resolved, "--version"]],
        "surya": [[resolved, "--help"]],
        "paddle": [
            [
                resolved,
                "-c",
                "import importlib.metadata; print(importlib.metadata.version('paddleocr'))",
            ]
        ],
    }
    if engine == "surya":
        interpreter = interpreter_for_script(Path(resolved))
        if interpreter:
            probes["surya"].insert(
                0,
                [
                    interpreter,
                    "-c",
                    "import importlib.metadata; print(importlib.metadata.version('surya-ocr'))",
                ],
            )
    for probe in probes[engine]:
        try:
            completed = bounded_subprocess(
                probe,
                timeout=20,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        output = (completed.stdout + "\n" + completed.stderr).strip()
        match = re.search(r"\d+(?:\.\d+){1,3}", output)
        if completed.returncode == 0 and match:
            return match.group(0), Path(resolved)
    raise ToolError("missing_dependency", f"Cannot determine {engine} runtime version")


def bounded_subprocess(
    command: list[str],
    *,
    timeout: float,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    tail_bytes: int = 16_000,
) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryFile(mode="w+b") as stdout_file:
        with tempfile.TemporaryFile(mode="w+b") as stderr_file:
            completed = subprocess.run(
                command,
                stdout=stdout_file,
                stderr=stderr_file,
                timeout=timeout,
                env=env,
                cwd=cwd,
                check=False,
            )
            stdout_file.seek(0, os.SEEK_END)
            stdout_size = stdout_file.tell()
            stdout_file.seek(max(0, stdout_size - tail_bytes))
            stdout_tail = stdout_file.read().decode("utf-8", errors="replace")
            stderr_file.seek(0, os.SEEK_END)
            stderr_size = stderr_file.tell()
            stderr_file.seek(max(0, stderr_size - tail_bytes))
            stderr_tail = stderr_file.read().decode("utf-8", errors="replace")
    return subprocess.CompletedProcess(
        command,
        completed.returncode,
        stdout_tail,
        stderr_tail,
    )


def interpreter_for_script(path: Path) -> str | None:
    try:
        first_line = path.open("rb").readline(200).decode("utf-8", "replace").strip()
    except OSError:
        return None
    if not first_line.startswith("#!"):
        return None
    parts = first_line[2:].split()
    if not parts:
        return None
    if Path(parts[0]).name == "env" and len(parts) > 1:
        return shutil.which(parts[1])
    return parts[0]


def available_engines() -> dict[str, Any]:
    return {
        "surya": {
            "cli": shutil.which("surya_ocr"),
            "backend": shutil.which("llama-server") or shutil.which("docker"),
        },
        "paddle": {
            "cli": shutil.which("paddleocr"),
            "python": shutil.which("python"),
        },
        "tesseract": {"cli": shutil.which("tesseract")},
    }


def hardware_inventory(requested: str) -> dict[str, Any]:
    if requested != "auto":
        return {"requested": requested, "selected": requested, "detected": False}
    selected = "cpu"
    evidence = []
    if shutil.which("nvidia-smi"):
        selected = "cuda"
        evidence.append("nvidia-smi")
    elif sys.platform == "darwin":
        try:
            machine = os.uname().machine
        except AttributeError:
            machine = ""
        if machine == "arm64":
            selected = "mps"
            evidence.append("Apple Silicon host")
    return {"requested": "auto", "selected": selected, "detected": True, "evidence": evidence}


def resolve_ocr_device(engine: str, requested: str) -> tuple[dict[str, str], list[str]]:
    inventory = hardware_inventory(requested)
    selected = inventory["selected"]
    warnings: list[str] = []
    if engine == "tesseract":
        if selected != "cpu":
            warnings.append(
                f"Tesseract is CPU-only; resolved requested device {requested!r} to CPU."
            )
        return {
            "requested_device": requested,
            "resolved_device": "cpu",
            "runtime_backend": "tesseract-cpu",
            "adapter_device": "cpu",
        }, warnings
    if engine == "paddle":
        if selected == "mps":
            warnings.append("PaddleOCR has no MPS adapter path in this skill; resolved MPS to CPU.")
            selected = "cpu"
        if selected == "cuda":
            return {
                "requested_device": requested,
                "resolved_device": "cuda:0",
                "runtime_backend": "paddlepaddle",
                "adapter_device": "gpu:0",
            }, warnings
        return {
            "requested_device": requested,
            "resolved_device": "cpu",
            "runtime_backend": "paddlepaddle",
            "adapter_device": "cpu",
        }, warnings
    warnings.append(
        "Surya uses the reviewed llama.cpp backend; compute-device offload is backend-managed "
        "and is not independently proven by the adapter."
    )
    return {
        "requested_device": requested,
        "resolved_device": "backend-managed",
        "runtime_backend": "llamacpp",
        "adapter_device": selected,
    }, warnings


def surya_license_summary() -> dict[str, Any]:
    return {
        "code_license": "Apache-2.0",
        "model_license": "modified AI Pubs Open Rail-M",
        "current_weight_terms_summary": (
            "Free for research, personal use, and startups under USD 5 million in "
            "funding/revenue; broader commercial use requires a commercial license "
            "from the model provider."
        ),
        "operator_confirmation_required": True,
    }


def recommend_ocr_engine(
    page_results: Sequence[dict[str, Any]],
    hardware: dict[str, Any],
    layout: str,
    volume_hint: int,
    manifest: dict[str, Any] | None,
) -> tuple[str | None, list[str]]:
    candidates = [
        page for page in page_results if page["classification"] in {"likely-scanned", "hybrid"}
    ]
    if not candidates:
        return None, [
            "Selected pages have usable digital text or are blank; do not OCR by default."
        ]
    surya_permitted = True
    if manifest and manifest.get("engine") == "surya":
        model = manifest.get("model") or {}
        surya_permitted = bool(
            model.get("license_terms_accepted") and model.get("use_case_eligibility_confirmed")
        )
    effective_layout = layout
    if layout == "auto":
        effective_layout = (
            "complex"
            if any(page["classification"] == "hybrid" for page in candidates)
            else "simple"
        )
    if (
        effective_layout == "complex"
        and hardware["selected"] in {"cuda", "mps"}
        and surya_permitted
    ):
        return "surya", [
            "Complex layout benefits from layout-aware blocks and reading order.",
            f"Selected hardware backend is {hardware['selected']}.",
            "Execution still requires accepted Surya model terms and provisioned local weights.",
        ]
    if effective_layout == "complex":
        return "paddle", [
            "Complex layout needs structure-aware output.",
            "PaddleOCR is the CPU-oriented fallback when acceleration or Surya model "
            "eligibility is unsuitable.",
        ]
    if volume_hint >= 25:
        return "tesseract", [
            "The workload is declared simple, clean printed text at batch scale.",
            "Flat text/TSV is sufficient and CPU throughput is prioritized over layout fidelity.",
        ]
    return "paddle", [
        "Selected pages need OCR but are not a clean high-volume batch.",
        "PaddleOCR is the conservative no-GPU local recommendation for general printed OCR.",
    ]


def handle_ocr_plan(args: argparse.Namespace) -> dict[str, Any]:
    password = resolve_password(args.password, args.password_env)
    languages = parse_languages(args.languages, field="ocr-plan languages")
    data = inspect_document(
        Path(args.input),
        password,
        args.pages,
        mode="plain",
        include_words=True,
        include_tables=False,
    )
    selected_hardware = hardware_inventory(args.hardware)
    manifest = read_json(Path(args.model_manifest)) if args.model_manifest else None
    recommendation, rationale = recommend_ocr_engine(
        data["pages"],
        selected_hardware,
        args.layout,
        args.volume_hint or len(data["pages"]),
        manifest,
    )
    dpi = args.dpi
    estimates = []
    total_pixels = 0
    for page in data["pages"]:
        pixels = math.ceil(page["width"] * dpi / 72) * math.ceil(page["height"] * dpi / 72)
        total_pixels += pixels
        estimates.append({"page": page["page"], "pixels": pixels, "dpi": dpi})
    limits_ok = all(item["pixels"] <= MAX_PIXELS_PER_RENDER for item in estimates) and (
        total_pixels <= MAX_TOTAL_RENDER_PIXELS
    )
    return success(
        "ocr-plan",
        source=data["source"],
        selected_pages=data["selected_pages"],
        page_classification=[
            {
                "page": page["page"],
                "classification": page["classification"],
                "text_character_count": page["text_character_count"],
                "embedded_image_count": page["embedded_image_count"],
            }
            for page in data["pages"]
        ],
        render_estimate={
            "pages": estimates,
            "total_pixels": total_pixels,
            "within_limits": limits_ok,
        },
        layout_complexity=args.layout,
        volume_hint=args.volume_hint or len(data["pages"]),
        hardware=selected_hardware,
        engine_availability=available_engines(),
        languages=languages,
        recommendation={
            "engine": recommendation,
            "paddle_pipeline": "structure"
            if recommendation == "paddle"
            and (
                args.layout == "complex"
                or (
                    args.layout == "auto"
                    and any(page["classification"] == "hybrid" for page in data["pages"])
                )
            )
            else ("ocr" if recommendation == "paddle" else None),
            "run_ocr": recommendation is not None,
            "rationale": rationale,
        },
        prerequisites={
            "all_engines": [
                "Explicit engine selection",
                "Explicit language selection",
                "Local model/language artifacts",
                "Artifact source, revision, checksum when available, and license review",
                "No hosted API use",
            ],
            "surya": surya_license_summary(),
            "paddle": {
                "code_license": "Apache-2.0",
                "model_license_must_be_recorded": True,
                "local_detection_and_recognition_model_directories_required": True,
                "local_layout_detection_model_directory_required_for_structure": True,
                "adapter_language_limit": "exactly one language identifier per run",
            },
            "tesseract": {
                "code_license": "Apache-2.0",
                "language_data_license_must_be_recorded": True,
            },
        },
        warnings=data["warnings"]
        + ([] if limits_ok else ["Requested render would exceed configured pixel limits."]),
        verification={"source_unchanged": True, "engine_executed": False},
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_inventory(path: Path) -> tuple[int, str, int]:
    """Return total bytes and a deterministic path/size/content inventory digest."""
    digest = hashlib.sha256()
    total_size = 0
    file_count = 0
    for item in sorted(
        path.rglob("*"), key=lambda candidate: candidate.relative_to(path).as_posix()
    ):
        if item.is_symlink():
            raise ToolError(
                "license_precondition",
                f"Artifact directory inventory contains a symlink: {item}",
            )
        if item.is_dir():
            continue
        if not item.is_file():
            raise ToolError(
                "license_precondition",
                f"Artifact directory inventory contains a non-file entry: {item}",
            )
        relative = item.relative_to(path).as_posix().encode("utf-8")
        size = item.stat().st_size
        digest.update(relative)
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
        total_size += size
        file_count += 1
    if file_count == 0:
        raise ToolError("license_precondition", f"Artifact directory is empty: {path}")
    return total_size, digest.hexdigest(), file_count


def placeholder_value(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    lowered = value.strip().lower()
    return lowered == REVIEW_REQUIRED.lower() or lowered.startswith("replace-with")


def preflight_manifest(
    manifest_path: Path,
    engine: str,
    languages: list[str],
) -> tuple[dict[str, Any], dict[str, Path]]:
    manifest = read_json(manifest_path)
    if manifest.get("engine") != engine:
        raise ToolError(
            "license_precondition",
            f"Manifest engine {manifest.get('engine')} does not match selected engine {engine}",
        )
    model = manifest.get("model")
    if not isinstance(model, dict):
        raise ToolError("license_precondition", "Manifest requires a model object")
    required_model_fields = ("identifier", "revision", "source", "license")
    missing = [field for field in required_model_fields if not model.get(field)]
    if missing:
        raise ToolError(
            "license_precondition",
            f"Manifest model is missing: {', '.join(missing)}",
        )
    placeholder_fields = [
        field for field in required_model_fields if placeholder_value(model.get(field))
    ]
    if placeholder_fields:
        raise ToolError(
            "license_precondition",
            "Manifest model contains unreviewed placeholder values: "
            + ", ".join(placeholder_fields),
        )
    if not model.get("license_terms_accepted"):
        raise ToolError("license_precondition", "Model/license-data terms are not accepted")
    if not model.get("use_case_eligibility_confirmed"):
        raise ToolError(
            "license_precondition",
            "Operator has not confirmed this use case is eligible under the selected "
            "artifact license",
        )
    if engine == "surya":
        stated_license = str(model.get("license", "")).lower()
        if "open rail" not in stated_license and "openrail" not in stated_license:
            raise ToolError(
                "license_precondition",
                "Current Surya weights must record their modified AI Pubs Open Rail-M license",
            )
        runtime = manifest.get("runtime")
        if not isinstance(runtime, dict):
            raise ToolError(
                "license_precondition",
                "Surya manifest requires a reviewed llama.cpp runtime object",
            )
        required_runtime_fields = ("identifier", "revision", "source", "license")
        missing_runtime = [field for field in required_runtime_fields if not runtime.get(field)]
        if missing_runtime:
            raise ToolError(
                "license_precondition",
                f"Surya runtime is missing: {', '.join(missing_runtime)}",
            )
        runtime_placeholders = [
            field for field in required_runtime_fields if placeholder_value(runtime.get(field))
        ]
        if runtime_placeholders:
            raise ToolError(
                "license_precondition",
                "Surya runtime contains unreviewed placeholder values: "
                + ", ".join(runtime_placeholders),
            )
        if not runtime.get("license_terms_accepted"):
            raise ToolError(
                "license_precondition",
                "Surya llama.cpp runtime terms are not accepted",
            )
    manifest_languages = model.get("languages")
    if manifest_languages is not None:
        if (
            not isinstance(manifest_languages, list)
            or not manifest_languages
            or any(not isinstance(item, str) or not item for item in manifest_languages)
        ):
            raise ToolError(
                "license_precondition",
                "Manifest model languages must be a nonempty string array",
            )
        if not set(languages).issubset(set(manifest_languages)):
            raise ToolError(
                "license_precondition",
                "Requested languages are not all recorded in the model manifest",
            )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ToolError("license_precondition", "Manifest requires provisioned artifacts")
    roles: dict[str, Path] = {}
    checked = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ToolError("license_precondition", "Every artifact must be an object")
        role = artifact.get("role")
        path_value = artifact.get("path")
        if not isinstance(role, str) or not role or not isinstance(path_value, str):
            raise ToolError("license_precondition", "Every artifact needs role and path")
        path = Path(path_value).expanduser()
        if not path.exists() or path.is_symlink():
            raise ToolError("license_precondition", f"Artifact is not provisioned: {path}")
        expected_size = artifact.get("size_bytes")
        expected_hash = artifact.get("sha256")
        if role in roles:
            raise ToolError("license_precondition", f"Duplicate artifact role: {role}")
        provenance_placeholders = [
            field
            for field in ("identifier", "revision", "source", "license")
            if placeholder_value(artifact.get(field))
        ]
        if provenance_placeholders:
            raise ToolError(
                "license_precondition",
                f"Artifact {role} contains unreviewed placeholder values "
                f"({', '.join(provenance_placeholders)}); explicit artifact-level "
                "preflight is required",
            )
        if engine == "paddle" and role in PADDLE_BASE_ROLES | PADDLE_STRUCTURE_ROLES:
            required_artifact_fields = ("identifier", "revision", "source", "license")
            missing_provenance = [
                field for field in required_artifact_fields if not artifact.get(field)
            ]
            if expected_size is None:
                missing_provenance.append("size_bytes")
            if not expected_hash:
                missing_provenance.append("sha256")
            if missing_provenance:
                raise ToolError(
                    "license_precondition",
                    f"Paddle artifact {role} has unknown provenance fields "
                    f"({', '.join(missing_provenance)}); explicit artifact-level preflight "
                    "is required",
                )
        if engine == "surya" and role == "backend" and (expected_size is None or not expected_hash):
            raise ToolError(
                "license_precondition",
                "Surya backend artifact requires size_bytes and sha256",
            )
        normalized_size = (
            finite_integer(
                expected_size,
                f"artifact[{role}].size_bytes",
                minimum=0,
                category="license_precondition",
            )
            if expected_size is not None
            else None
        )
        normalized_hash = str(expected_hash).lower() if expected_hash else None
        if normalized_hash and not re.fullmatch(r"[0-9a-f]{64}", normalized_hash):
            raise ToolError(
                "license_precondition",
                f"Artifact {role} sha256 must be 64 hexadecimal characters",
            )
        file_count: int | None = None
        if path.is_file():
            actual_size = path.stat().st_size
            actual_hash = sha256_file(path) if normalized_hash else None
            if normalized_size is not None and actual_size != normalized_size:
                raise ToolError(
                    "license_precondition",
                    f"Artifact size mismatch: {path}",
                    details={
                        "role": role,
                        "path": str(path),
                        "expected_size_bytes": normalized_size,
                        "actual_size_bytes": actual_size,
                    },
                )
            if normalized_hash and actual_hash != normalized_hash:
                raise ToolError(
                    "license_precondition",
                    f"Artifact checksum mismatch: {path}",
                    details={
                        "role": role,
                        "path": str(path),
                        "expected_sha256": normalized_hash,
                        "actual_sha256": actual_hash,
                    },
                )
        elif path.is_dir():
            actual_size = None
            actual_hash = None
            if normalized_size is not None or normalized_hash:
                measured_size, measured_hash, file_count = directory_inventory(path)
                actual_size = measured_size
                actual_hash = measured_hash
            if normalized_size is not None and actual_size != normalized_size:
                raise ToolError(
                    "license_precondition",
                    f"Artifact directory inventory size mismatch: {path}",
                    details={
                        "role": role,
                        "path": str(path),
                        "expected_size_bytes": normalized_size,
                        "actual_size_bytes": actual_size,
                    },
                )
            if normalized_hash and actual_hash != normalized_hash:
                raise ToolError(
                    "license_precondition",
                    f"Artifact directory inventory checksum mismatch: {path}",
                    details={
                        "role": role,
                        "path": str(path),
                        "expected_sha256": normalized_hash,
                        "actual_sha256": actual_hash,
                    },
                )
        else:
            raise ToolError("license_precondition", f"Artifact must be a file or directory: {path}")
        roles[role] = path.resolve()
        checked.append(
            {
                "role": role,
                "path": str(path.resolve()),
                "identifier": artifact.get("identifier"),
                "revision": artifact.get("revision"),
                "source": artifact.get("source"),
                "license": artifact.get("license"),
                "sha256_verified": bool(normalized_hash),
                "size_verified": normalized_size is not None,
                "inventory_file_count": file_count,
                "provenance_status": (
                    "verified"
                    if normalized_hash and normalized_size is not None
                    else "unverified-explicit-preflight-required"
                ),
            }
        )
    required_roles = {
        "surya": {"backend", "model", "mmproj"},
        "paddle": PADDLE_BASE_ROLES,
        "tesseract": set(),
    }[engine]
    if not required_roles.issubset(roles):
        raise ToolError(
            "license_precondition",
            f"{engine} manifest requires artifact roles: {', '.join(sorted(required_roles))}",
        )
    if engine == "surya" and (
        not roles["backend"].is_file() or not os.access(roles["backend"], os.X_OK)
    ):
        raise ToolError(
            "license_precondition",
            "Surya backend artifact must be an executable regular file",
        )
    if engine == "paddle":
        invalid_model_roles = sorted(
            role
            for role in (PADDLE_BASE_ROLES | PADDLE_STRUCTURE_ROLES).intersection(roles)
            if not roles[role].is_dir()
        )
        if invalid_model_roles:
            raise ToolError(
                "license_precondition",
                "Paddle model artifacts must be directories: " + ", ".join(invalid_model_roles),
            )
    if engine == "tesseract":
        language_roles = {"language_data", *(f"language_data:{lang}" for lang in languages)}
        if not language_roles.intersection(roles):
            raise ToolError(
                "license_precondition",
                "Tesseract manifest requires language_data or language_data:<language> artifacts",
            )
    return {
        "manifest_path": str(manifest_path.resolve()),
        "model": model,
        "artifacts": checked,
        "license_preflight": "passed",
    }, roles


def tesseract_languages(executable: Path, tessdata_dir: Path) -> list[str]:
    try:
        completed = bounded_subprocess(
            [str(executable), "--tessdata-dir", str(tessdata_dir), "--list-langs"],
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ToolError("missing_dependency", "Cannot query Tesseract languages") from exc
    if completed.returncode:
        raise ToolError("missing_dependency", "Tesseract language query failed")
    return [
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip() and not line.lower().startswith("list of available")
    ]


MANIFEST_EXCLUDED_CHECKS = ("engine_version", "tesseract_language_query", "backend_probe")


def parse_artifact_arguments(values: Sequence[str]) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    for item in values:
        role, separator, path_value = item.partition("=")
        if not separator or not role or not path_value:
            raise ToolError("bad_input", f"--artifact requires ROLE=PATH: {item}")
        if role in artifacts:
            raise ToolError("bad_input", f"Duplicate --artifact role: {role}")
        artifacts[role] = Path(path_value).expanduser()
    return artifacts


def measure_artifact(role: str, path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.exists():
        raise ToolError("bad_input", f"Artifact is not provisioned: {path}")
    if path.is_file():
        return {
            "role": role,
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    if path.is_dir():
        size, digest, file_count = directory_inventory(path)
        return {
            "role": role,
            "path": str(path.resolve()),
            "size_bytes": size,
            "sha256": digest,
            "inventory_file_count": file_count,
        }
    raise ToolError("bad_input", f"Artifact must be a file or directory: {path}")


def manifest_placeholder_provenance() -> dict[str, str]:
    return {
        "identifier": REVIEW_REQUIRED,
        "revision": REVIEW_REQUIRED,
        "source": REVIEW_REQUIRED,
        "license": REVIEW_REQUIRED,
    }


def locate_tessdata_directory(languages: list[str], explicit: str | None) -> Path:
    searched: list[str] = []
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    else:
        env_value = os.environ.get("TESSDATA_PREFIX")
        if env_value:
            candidates.append(Path(env_value).expanduser())
            candidates.append(Path(env_value).expanduser() / "tessdata")
        candidates.extend(Path(item) for item in WELL_KNOWN_TESSDATA_DIRS)
    qualifying: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        searched.append(key)
        # A symlinked directory is normal here (Homebrew symlinks its share tree); is_dir
        # follows the link. The no-symlink rule applies to the traineddata files, which the
        # manifest preflight enforces on the resolved artifact paths.
        if not candidate.is_dir():
            continue
        if all(
            (candidate / f"{language}.traineddata").is_file()
            and not (candidate / f"{language}.traineddata").is_symlink()
            for language in languages
        ):
            qualifying.append(candidate)
    if not qualifying:
        raise ToolError(
            "bad_input",
            "No tessdata directory contains every requested language "
            f"({', '.join(languages)}); searched: {', '.join(searched)}. "
            "Pass --tessdata-dir explicitly.",
        )
    if len(qualifying) > 1:
        raise ToolError(
            "bad_input",
            "Multiple tessdata directories contain the requested languages: "
            + ", ".join(str(path) for path in qualifying)
            + ". Pass --tessdata-dir explicitly.",
        )
    return qualifying[0]


def derive_manifest(
    engine: str,
    languages: list[str] | None,
    artifacts_by_role: dict[str, Path],
    *,
    tessdata_dir: str | None,
    assume_source: str | None,
    accept_license: bool,
    confirm_eligibility: bool,
) -> tuple[dict[str, Any], list[str], list[str]]:
    warnings: list[str] = []
    reasons: list[str] = []
    declared_fields: list[str] = []
    approvals = {
        "license_terms_accepted": accept_license,
        "use_case_eligibility_confirmed": confirm_eligibility,
    }
    runtime: dict[str, Any] | None = None
    if engine == "tesseract":
        if not languages:
            raise ToolError("bad_input", "Tesseract manifest derivation requires --languages")
        if artifacts_by_role:
            raise ToolError(
                "bad_input",
                "Tesseract derivation locates language data itself; use --tessdata-dir, "
                "not --artifact",
            )
        directory = locate_tessdata_directory(languages, tessdata_dir)
        declared = TESSDATA_SOURCES[assume_source] if assume_source else None
        artifact_entries = []
        for language in languages:
            entry = measure_artifact(
                f"language_data:{language}", directory / f"{language}.traineddata"
            )
            if declared:
                entry.update(
                    {
                        "identifier": (f"{declared['identifier_prefix']}/{language}.traineddata"),
                        "revision": declared["revision"],
                        "source": declared["source"],
                        "license": declared["license"],
                        "provenance": "declared-by-operator",
                    }
                )
            else:
                entry.update(manifest_placeholder_provenance())
                entry["provenance"] = REVIEW_REQUIRED
            artifact_entries.append(entry)
        if declared:
            model_provenance = {
                "identifier": f"{declared['identifier_prefix']}:{'+'.join(languages)}",
                "revision": declared["revision"],
                "source": declared["source"],
                "license": declared["license"],
            }
            declared_fields.extend(["identifier", "revision", "source", "license"])
            warnings.append(
                "Operator declared artifact provenance via --assume-source; the tool "
                "verified only local paths, sizes, and SHA-256 values, not the files' "
                "origin."
            )
        else:
            model_provenance = manifest_placeholder_provenance()
            reasons.append(
                "Model provenance is REVIEW-REQUIRED; pass --assume-source after "
                "confirming the files' origin, or record reviewed facts in the manifest."
            )
        model: dict[str, Any] = {**model_provenance, "languages": languages, **approvals}
    elif engine == "surya":
        required_roles = {"model", "mmproj", "backend"}
        missing = sorted(required_roles - set(artifacts_by_role))
        if missing:
            raise ToolError(
                "bad_input",
                f"Surya derivation requires --artifact roles: {', '.join(missing)}",
            )
        unknown = sorted(set(artifacts_by_role) - required_roles)
        if unknown:
            raise ToolError("bad_input", f"Unknown Surya artifact roles: {', '.join(unknown)}")
        artifact_entries = []
        model_match: dict[str, str] | None = None
        for role in ("model", "mmproj", "backend"):
            entry = measure_artifact(role, artifacts_by_role[role])
            if role in {"model", "mmproj"}:
                known = KNOWN_ARTIFACTS.get(entry["sha256"])
                if known:
                    entry.update(
                        {
                            field: known[field]
                            for field in ("identifier", "revision", "source", "license")
                        }
                    )
                    entry["provenance"] = "matched-reviewed-artifact"
                    if role == "model":
                        model_match = known
                else:
                    entry.update(manifest_placeholder_provenance())
                    entry["provenance"] = REVIEW_REQUIRED
                    reasons.append(
                        f"Surya {role} artifact SHA-256 does not match a reviewed "
                        "artifact; review and record its provenance."
                    )
            else:
                entry["provenance"] = "measured-local-binary"
            artifact_entries.append(entry)
        if model_match:
            model_provenance = {
                "identifier": model_match["identifier"].rsplit("/", 1)[0],
                "revision": model_match["revision"],
                "source": model_match["source"],
                "license": model_match["license"],
            }
        else:
            model_provenance = manifest_placeholder_provenance()
        model = {**model_provenance, **approvals}
        if languages:
            model["languages"] = languages
        runtime = {
            "identifier": "ggerganov/llama.cpp",
            "revision": REVIEW_REQUIRED,
            "source": "https://github.com/ggml-org/llama.cpp",
            "license": "MIT",
            "license_terms_accepted": False,
        }
        reasons.append(
            "Surya runtime revision is REVIEW-REQUIRED; record the reviewed llama.cpp "
            "build revision and accept its terms in the manifest."
        )
        warnings.append(
            "Review the Surya model license before running OCR: "
            + surya_license_summary()["current_weight_terms_summary"]
        )
    else:
        valid_roles = PADDLE_BASE_ROLES | PADDLE_STRUCTURE_ROLES
        unknown = sorted(set(artifacts_by_role) - valid_roles)
        if unknown:
            raise ToolError("bad_input", f"Unknown Paddle artifact roles: {', '.join(unknown)}")
        missing_base = sorted(PADDLE_BASE_ROLES - set(artifacts_by_role))
        if missing_base:
            raise ToolError(
                "bad_input",
                f"Paddle derivation requires --artifact roles: {', '.join(missing_base)}",
            )
        artifact_entries = []
        for role in sorted(artifacts_by_role):
            path = artifacts_by_role[role]
            if path.is_symlink() or not path.is_dir():
                raise ToolError(
                    "bad_input", f"Paddle artifact {role} must be a model directory: {path}"
                )
            entry = measure_artifact(role, path)
            known = None
            if role in PADDLE_BASE_ROLES:
                main_parameters = path / "inference.pdiparams"
                if main_parameters.is_file():
                    known = KNOWN_ARTIFACTS.get(sha256_file(main_parameters))
            if known:
                entry.update(
                    {
                        field: known[field]
                        for field in ("identifier", "revision", "source", "license")
                    }
                )
                entry["provenance"] = "matched-reviewed-main-parameter-file"
                warnings.append(
                    f"Paddle role {role} matched the reviewed main parameter file for "
                    f"{known['identifier']}; the recorded size/sha256 cover the complete "
                    "directory inventory, while the review match covers only the main "
                    "parameter file."
                )
            else:
                entry.update(manifest_placeholder_provenance())
                entry["provenance"] = REVIEW_REQUIRED
                if role in PADDLE_STRUCTURE_ROLES:
                    reasons.append(
                        f"Structure role {role} is never auto-filled; review and record "
                        "its provenance explicitly."
                    )
                else:
                    reasons.append(
                        f"Paddle role {role} did not match a reviewed main parameter "
                        "file; review and record its provenance."
                    )
            artifact_entries.append(entry)
        model = {**manifest_placeholder_provenance(), **approvals}
        if languages:
            model["languages"] = languages
        reasons.append(
            "Paddle model provenance is REVIEW-REQUIRED; record the reviewed model "
            "identifier and revision in the manifest."
        )
    if accept_license:
        declared_fields.append("license_terms_accepted")
    else:
        reasons.append(
            "license_terms_accepted is false; pass --accept-license after reviewing "
            "the artifact license terms."
        )
    if confirm_eligibility:
        declared_fields.append("use_case_eligibility_confirmed")
    else:
        reasons.append(
            "use_case_eligibility_confirmed is false; pass --confirm-eligibility after "
            "confirming this use case is eligible."
        )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "engine": engine,
        "model": model,
        "artifacts": artifact_entries,
        "derivation": {
            "mode": "derive",
            "tool_version": TOOL_VERSION,
            "measured": ["path", "size_bytes", "sha256"],
            "declared_by_operator": declared_fields,
        },
    }
    if runtime is not None:
        manifest["runtime"] = runtime
    return manifest, warnings, reasons


def manifest_check(
    manifest_path: Path,
    engine: str,
    languages: list[str] | None,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: Any, **details: Any) -> None:
        checks.append({"check": name, "ok": bool(ok), **details})

    add(
        "engine_matches",
        manifest.get("engine") == engine,
        expected=engine,
        actual=manifest.get("engine"),
    )
    model = manifest.get("model")
    model_ok = isinstance(model, dict)
    add("model_object", model_ok)
    if model_ok:
        for field in ("identifier", "revision", "source", "license"):
            value = model.get(field)
            add(
                f"model_{field}",
                bool(value) and not placeholder_value(value),
                actual=value if isinstance(value, str) else None,
            )
        add("model_license_terms_accepted", bool(model.get("license_terms_accepted")))
        add(
            "model_use_case_eligibility_confirmed",
            bool(model.get("use_case_eligibility_confirmed")),
        )
        declared_languages = model.get("languages")
        if languages and isinstance(declared_languages, list):
            add(
                "languages_covered",
                set(languages) <= {str(item) for item in declared_languages},
                requested=languages,
                declared=declared_languages,
            )
    if engine == "surya":
        if model_ok:
            stated = str(model.get("license", "")).lower()
            add(
                "surya_model_license_records_open_rail",
                "open rail" in stated or "openrail" in stated,
            )
        runtime = manifest.get("runtime")
        runtime_ok = isinstance(runtime, dict)
        add("runtime_object", runtime_ok)
        if runtime_ok:
            for field in ("identifier", "revision", "source", "license"):
                value = runtime.get(field)
                add(
                    f"runtime_{field}",
                    bool(value) and not placeholder_value(value),
                    actual=value if isinstance(value, str) else None,
                )
            add("runtime_license_terms_accepted", bool(runtime.get("license_terms_accepted")))
    artifacts = manifest.get("artifacts")
    artifacts_ok = isinstance(artifacts, list) and bool(artifacts)
    add("artifacts_array", artifacts_ok)
    roles: set[str] = set()
    if artifacts_ok:
        for index, artifact in enumerate(artifacts):
            if not isinstance(artifact, dict):
                add(f"artifact_{index}_object", False)
                continue
            role_value = artifact.get("role")
            role = role_value if isinstance(role_value, str) and role_value else f"#{index}"
            path_value = artifact.get("path")
            record = {"role": role, "path": path_value}
            if role in roles:
                add("artifact_role_unique", False, **record)
            roles.add(role)
            if not isinstance(path_value, str) or not path_value:
                add("artifact_path", False, **record)
                continue
            path = Path(path_value).expanduser()
            provisioned = path.exists() and not path.is_symlink()
            add("artifact_provisioned", provisioned, **record)
            if not provisioned:
                continue
            expected_size = artifact.get("size_bytes")
            expected_hash_value = artifact.get("sha256")
            expected_hash = str(expected_hash_value).lower() if expected_hash_value else None
            if expected_hash and not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
                add("artifact_sha256_format", False, **record, expected_sha256=expected_hash)
                expected_hash = None
            try:
                if path.is_file():
                    actual_size = path.stat().st_size
                    actual_hash = sha256_file(path)
                else:
                    actual_size, actual_hash, _ = directory_inventory(path)
            except ToolError as exc:
                add("artifact_inventory", False, **record, error=exc.message)
                continue
            size_hash_required = (
                engine == "paddle" and role in PADDLE_BASE_ROLES | PADDLE_STRUCTURE_ROLES
            ) or (engine == "surya" and role == "backend")
            if expected_size is not None:
                try:
                    normalized_size = finite_integer(expected_size, "size_bytes", minimum=0)
                except ToolError:
                    add(
                        "artifact_size_format",
                        False,
                        **record,
                        expected_size_bytes=expected_size,
                    )
                else:
                    add(
                        "artifact_size",
                        normalized_size == actual_size,
                        **record,
                        expected_size_bytes=normalized_size,
                        actual_size_bytes=actual_size,
                    )
            else:
                add(
                    "artifact_size_recorded",
                    not size_hash_required,
                    **record,
                    actual_size_bytes=actual_size,
                    note="size_bytes is not recorded in the manifest",
                )
            if expected_hash:
                add(
                    "artifact_sha256",
                    expected_hash == actual_hash,
                    **record,
                    expected_sha256=expected_hash,
                    actual_sha256=actual_hash,
                )
            else:
                add(
                    "artifact_sha256_recorded",
                    not size_hash_required,
                    **record,
                    actual_sha256=actual_hash,
                    note="sha256 is not recorded in the manifest",
                )
            placeholders = [
                field
                for field in ("identifier", "revision", "source", "license")
                if placeholder_value(artifact.get(field))
            ]
            add(
                "artifact_provenance_reviewed",
                not placeholders,
                **record,
                placeholder_fields=placeholders,
            )
            if engine == "paddle" and role in PADDLE_BASE_ROLES | PADDLE_STRUCTURE_ROLES:
                missing_fields = [
                    field
                    for field in ("identifier", "revision", "source", "license")
                    if not artifact.get(field)
                ]
                add(
                    "paddle_artifact_provenance_complete",
                    not missing_fields,
                    **record,
                    missing_fields=missing_fields,
                )
    required_roles = {
        "surya": {"backend", "model", "mmproj"},
        "paddle": set(PADDLE_BASE_ROLES),
        "tesseract": set(),
    }[engine]
    add(
        "required_roles_present",
        required_roles <= roles,
        required=sorted(required_roles),
        present=sorted(roles),
    )
    failures = [item for item in checks if not item["ok"]]
    summary = {"checks": len(checks), "failed": len(failures)}
    if failures:
        raise ToolError(
            "license_precondition",
            f"Manifest check found {len(failures)} problem(s)",
            details={
                "summary": summary,
                "checks": checks,
                "excluded_checks": list(MANIFEST_EXCLUDED_CHECKS),
            },
        )
    return success(
        "manifest",
        mode="check",
        engine=engine,
        manifest_path=str(manifest_path.resolve()),
        summary=summary,
        checks=checks,
        preflight_would_pass=True,
        excluded_checks=list(MANIFEST_EXCLUDED_CHECKS),
        warnings=[],
        verification={"engine_executed": False},
    )


def handle_manifest(args: argparse.Namespace) -> dict[str, Any]:
    languages = (
        parse_languages(args.languages, field="manifest languages") if args.languages else None
    )
    if args.mode == "check":
        if not args.model_manifest:
            raise ToolError("bad_input", "manifest --mode check requires --model-manifest")
        return manifest_check(Path(args.model_manifest), args.engine, languages)
    if not args.output:
        raise ToolError("bad_input", "manifest --mode derive requires --output")
    output = Path(args.output)
    require_extension(output, ".json", "Derived manifest output")
    artifacts_by_role = parse_artifact_arguments(args.artifact)
    manifest, warnings, reasons = derive_manifest(
        args.engine,
        languages,
        artifacts_by_role,
        tessdata_dir=args.tessdata_dir,
        assume_source=args.assume_source,
        accept_license=args.accept_license,
        confirm_eligibility=args.confirm_eligibility,
    )
    write_json_atomic(output, manifest, overwrite=args.overwrite)
    return success(
        "manifest",
        mode="derive",
        engine=args.engine,
        output_path=str(output.resolve()),
        artifacts=[
            {
                "role": entry["role"],
                "path": entry["path"],
                "size_bytes": entry["size_bytes"],
                "sha256": entry["sha256"],
                "provenance": entry.get("provenance"),
            }
            for entry in manifest["artifacts"]
        ],
        ready_for_preflight=not reasons,
        reasons_not_ready=reasons,
        license_summary=surya_license_summary() if args.engine == "surya" else None,
        warnings=warnings,
        verification={"engine_executed": False, "manifest_reopened": True},
    )


def check_engine_preflight(
    engine: str,
    executable_arg: str | None,
    manifest_path: Path,
    languages: list[str],
) -> tuple[dict[str, Any], dict[str, Path], Path]:
    default_commands = {"surya": "surya_ocr", "paddle": sys.executable, "tesseract": "tesseract"}
    command = executable_arg or default_commands[engine]
    version, executable = executable_version(command, engine)
    try:
        major = int(version.split(".")[0])
    except ValueError as exc:
        raise ToolError("missing_dependency", f"Unparseable {engine} version: {version}") from exc
    expected_major = {
        "surya": SUPPORTED_SURYA_MAJOR,
        "paddle": SUPPORTED_PADDLE_MAJOR,
        "tesseract": SUPPORTED_TESSERACT_MAJOR,
    }[engine]
    if major != expected_major:
        raise ToolError(
            "unsupported_operation",
            f"Untested {engine} major version {major}; tested major is {expected_major}",
        )
    manifest, roles = preflight_manifest(manifest_path, engine, languages)
    if engine == "tesseract":
        language_paths = []
        generic = roles.get("language_data")
        if generic:
            if generic.is_dir():
                language_paths = [generic / f"{language}.traineddata" for language in languages]
            elif len(languages) == 1:
                language_paths = [generic]
            else:
                raise ToolError(
                    "license_precondition",
                    "A single language_data file cannot satisfy multiple languages",
                )
        else:
            language_paths = [roles.get(f"language_data:{language}") for language in languages]
        if any(path is None for path in language_paths):
            raise ToolError(
                "license_precondition",
                "Manifest does not identify every requested Tesseract language file",
            )
        checked_language_paths = [path for path in language_paths if path is not None]
        for language, path in zip(languages, checked_language_paths, strict=True):
            if path.name != f"{language}.traineddata" or not path.is_file():
                raise ToolError(
                    "license_precondition",
                    f"Tesseract artifact for {language} must be {language}.traineddata",
                )
        tessdata_dirs = {path.parent.resolve() for path in checked_language_paths}
        if len(tessdata_dirs) != 1:
            raise ToolError(
                "license_precondition",
                "All Tesseract language files must share one tessdata directory",
            )
        roles["tessdata_dir"] = tessdata_dirs.pop()
        available = tesseract_languages(executable, roles["tessdata_dir"])
        missing = sorted(set(languages) - set(available))
        if missing:
            raise ToolError(
                "missing_dependency",
                f"Tesseract language data unavailable: {', '.join(missing)}",
            )
    manifest["engine_version"] = version
    manifest["executable"] = str(executable)
    if engine == "surya":
        manifest["surya_weight_terms"] = surya_license_summary()
        try:
            backend_version = bounded_subprocess(
                [str(roles["backend"]), "--version"],
                timeout=20,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ToolError(
                "missing_dependency",
                "Cannot execute the approved llama.cpp backend",
            ) from exc
        if backend_version.returncode:
            raise ToolError(
                "missing_dependency",
                "Approved llama.cpp backend failed its version probe",
            )
        backend_output = (backend_version.stdout + "\n" + backend_version.stderr).strip()
        if "llama" not in roles["backend"].name.lower() or not re.search(
            r"(?:version[^0-9\n]{0,20}\d+|llama(?:\.cpp|-server)[^0-9\n]{0,40}\d+)",
            backend_output,
            re.IGNORECASE,
        ):
            raise ToolError(
                "missing_dependency",
                "Approved backend did not report a recognized llama.cpp version signature",
            )
        manifest["backend"] = {
            "executable": str(roles["backend"]),
            "version_output": clean_text(backend_output[:500]),
            "forced_backend": "llamacpp",
        }
    return manifest, roles, executable


@dataclass
class RenderBudget:
    """Shared pixel accounting for every rasterization path."""

    total_message: str = f"Render total exceeds {MAX_TOTAL_RENDER_PIXELS} pixels"
    total_pixels: int = 0

    def charge(
        self,
        width_points: float,
        height_points: float,
        dpi: int,
        *,
        page_label: str,
    ) -> tuple[int, int]:
        pixel_width = math.ceil(width_points * dpi / 72)
        pixel_height = math.ceil(height_points * dpi / 72)
        pixels = pixel_width * pixel_height
        if pixels > MAX_PIXELS_PER_RENDER:
            raise ToolError(
                "resource_limit",
                f"{page_label} render exceeds {MAX_PIXELS_PER_RENDER} pixels",
            )
        self.total_pixels += pixels
        if self.total_pixels > MAX_TOTAL_RENDER_PIXELS:
            raise ToolError("resource_limit", self.total_message)
        return pixel_width, pixel_height


def rasterize_region(
    plumber_page: Any,
    bbox: tuple[float, float, float, float] | None,
    dpi: int,
    destination: Path,
    *,
    page_label: str,
) -> None:
    crop = plumber_page.crop(bbox) if bbox else plumber_page
    try:
        rendered = crop.to_image(resolution=dpi, antialias=True)
        rendered.save(str(destination), format="PNG")
    except Exception as exc:
        raise ToolError(
            "validation_failed",
            f"Cannot render {page_label}: {clean_text(exc)}",
        ) from exc


def render_regions_for_ocr(
    source_path: Path,
    password: str | None,
    pages_expression: str | None,
    dpi: int,
    work_dir: Path,
    *,
    force: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    data = inspect_document(
        source_path,
        password,
        pages_expression,
        mode="plain",
        include_words=True,
        include_tables=False,
    )
    render_jobs = []
    warnings = list(data["warnings"])
    budget = RenderBudget(
        total_message=f"OCR render total exceeds {MAX_TOTAL_RENDER_PIXELS} pixels"
    )
    with pdfplumber.open(str(source_path), password=password) as plumber_pdf:
        for page_data in data["pages"]:
            page_number = page_data["page"]
            classification = page_data["classification"]
            if classification in {"digital-text", "blank"} and not force:
                warnings.append(f"Page {page_number} skipped: classification is {classification}.")
                continue
            page = plumber_pdf.pages[page_number - 1]
            regions: list[tuple[float, float, float, float] | None]
            if classification == "hybrid" and page.images and not force:
                regions = []
                for image in page.images:
                    try:
                        regions.append(
                            (
                                max(0.0, float(image["x0"])),
                                max(0.0, float(image["top"])),
                                min(float(page.width), float(image["x1"])),
                                min(float(page.height), float(image["bottom"])),
                            )
                        )
                    except (KeyError, TypeError, ValueError):
                        continue
                if not regions:
                    warnings.append(
                        f"Page {page_number} hybrid image regions were ambiguous; page skipped."
                    )
                    continue
            else:
                regions = [None]
            for region_index, bbox in enumerate(regions, start=1):
                crop = page.crop(bbox) if bbox else page
                width_points = float(crop.width)
                height_points = float(crop.height)
                pixel_width, pixel_height = budget.charge(
                    width_points,
                    height_points,
                    dpi,
                    page_label=f"Page {page_number}",
                )
                image_path = work_dir / f"page-{page_number:04d}-region-{region_index:04d}.png"
                rasterize_region(
                    page,
                    bbox,
                    dpi,
                    image_path,
                    page_label=f"page {page_number}",
                )
                render_jobs.append(
                    {
                        "source_page": page_number,
                        "region": list(bbox) if bbox else None,
                        "image_path": image_path,
                        "dpi": dpi,
                        "pixel_width": pixel_width,
                        "pixel_height": pixel_height,
                        "classification": classification,
                        "page_geometry": page_data["geometry"],
                    }
                )
    if not render_jobs:
        raise ToolError("unsupported_operation", "No selected page or image region needs OCR")
    return render_jobs, warnings


def sidecar_page_geometry(job: dict[str, Any]) -> dict[str, Any]:
    geometry = job["page_geometry"]
    return {
        "width_points": geometry["width"],
        "height_points": geometry["height"],
        "rotation": geometry["rotation"],
    }


def run_process(
    command: list[str],
    *,
    timeout: float,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = bounded_subprocess(
            command,
            timeout=timeout,
            env=env,
            cwd=cwd,
            tail_bytes=4_000,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolError("engine_failed", f"OCR engine exceeded {timeout} seconds") from exc
    except OSError as exc:
        raise ToolError(
            "missing_dependency",
            f"Cannot execute OCR engine: {clean_text(exc)}",
        ) from exc
    if completed.returncode:
        diagnostic = clean_text((completed.stderr or completed.stdout)[-4000:])
        raise ToolError(
            "engine_failed",
            f"OCR engine exited with status {completed.returncode}",
            details={"diagnostic": diagnostic},
        )
    return completed


def normalize_box(box: Any, field: str) -> list[float] | None:
    if box is None:
        return None
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ToolError("engine_failed", f"{field} must contain four coordinates")
    normalized = [
        finite_number(value, f"{field}[{index}]", minimum=0, category="engine_failed")
        for index, value in enumerate(box)
    ]
    if normalized[2] <= normalized[0] or normalized[3] <= normalized[1]:
        raise ToolError("engine_failed", f"{field} has impossible coordinate ordering")
    return [round(value, 3) for value in normalized]


def normalize_polygon(polygon: Any, field: str) -> list[list[float]] | None:
    if polygon is None:
        return None
    if not isinstance(polygon, (list, tuple)) or len(polygon) < 3:
        raise ToolError("engine_failed", f"{field} must contain at least three points")
    normalized = []
    for point_index, point in enumerate(polygon):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ToolError("engine_failed", f"{field}[{point_index}] must be a coordinate pair")
        normalized_point = [
            finite_number(
                value,
                f"{field}[{point_index}][{coordinate_index}]",
                minimum=0,
                category="engine_failed",
            )
            for coordinate_index, value in enumerate(point)
        ]
        normalized.append([round(value, 3) for value in normalized_point])
    return normalized


def normalize_confidence(value: Any, field: str) -> float | None:
    if value is None:
        return None
    return finite_number(value, field, minimum=0, maximum=1, category="engine_failed")


def normalize_order(value: Any, field: str, fallback: int) -> tuple[int, bool]:
    if value is None:
        return fallback, True
    return (
        finite_integer(value, field, minimum=0, category="engine_failed"),
        False,
    )


def validate_unique_block_order(blocks: list[dict[str, Any]], engine: str) -> None:
    orders = [block["order"] for block in blocks]
    if len(set(orders)) != len(orders):
        raise ToolError("engine_failed", f"{engine} supplied duplicate block order values")
    blocks.sort(key=lambda block: block["order"])


def validate_block_coordinate_bounds(
    blocks: Sequence[dict[str, Any]],
    *,
    width: int,
    height: int,
    engine: str,
) -> None:
    for index, block in enumerate(blocks):
        box = block.get("bbox")
        if box is not None and (box[2] > width or box[3] > height):
            raise ToolError(
                "engine_failed",
                f"{engine} block {index} bbox exceeds the rendered coordinate space",
            )
        polygon = block.get("polygon")
        if polygon is not None and any(point[0] > width or point[1] > height for point in polygon):
            raise ToolError(
                "engine_failed",
                f"{engine} block {index} polygon exceeds the rendered coordinate space",
            )


def read_engine_json(path: Path, engine: str) -> Any:
    try:
        if path.stat().st_size > MAX_JSON_BYTES:
            raise ToolError(
                "engine_failed",
                f"{engine} result JSON exceeds {MAX_JSON_BYTES} bytes",
            )
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"Non-finite JSON number: {value}")
            ),
        )
    except ToolError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ToolError("engine_failed", f"Cannot parse {engine} result JSON") from exc


def bbox_to_polygon(box: list[float] | None) -> list[list[float]] | None:
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def coerce_engine_error(value: Any) -> str | None:
    """Normalize an engine-supplied block/page error flag.

    Surya reports a healthy block as ``"error": false`` and an error as a message string;
    older schema versions used null. Coerce the boolean form (false -> no error, true ->
    an unlabeled error marker) and stringify any other unexpected type rather than failing
    a whole run over a benign flag type.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return "error" if value else None
    if isinstance(value, str):
        return value or None
    return clean_text(value)


def tesseract_blocks(tsv_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    required_fields = {
        "page_num",
        "block_num",
        "par_num",
        "line_num",
        "left",
        "top",
        "width",
        "height",
        "conf",
        "text",
    }
    parsed_rows: list[dict[str, Any]] = []
    try:
        with tsv_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            missing_fields = sorted(required_fields - set(reader.fieldnames or []))
            if missing_fields:
                raise ToolError(
                    "engine_failed",
                    "Tesseract TSV is missing required fields: " + ", ".join(missing_fields),
                )
            for row_number, row in enumerate(reader, start=2):
                text_value = row.get("text")
                if not isinstance(text_value, str):
                    raise ToolError(
                        "engine_failed",
                        f"Tesseract TSV row {row_number} text must be a string",
                    )
                parsed = {
                    "page_num": finite_integer(
                        row.get("page_num"),
                        f"Tesseract TSV row {row_number} page_num",
                        minimum=1,
                        category="engine_failed",
                    ),
                    "block_num": finite_integer(
                        row.get("block_num"),
                        f"Tesseract TSV row {row_number} block_num",
                        minimum=0,
                        category="engine_failed",
                    ),
                    "par_num": finite_integer(
                        row.get("par_num"),
                        f"Tesseract TSV row {row_number} par_num",
                        minimum=0,
                        category="engine_failed",
                    ),
                    "line_num": finite_integer(
                        row.get("line_num"),
                        f"Tesseract TSV row {row_number} line_num",
                        minimum=0,
                        category="engine_failed",
                    ),
                    "left": finite_integer(
                        row.get("left"),
                        f"Tesseract TSV row {row_number} left",
                        minimum=0,
                        category="engine_failed",
                    ),
                    "top": finite_integer(
                        row.get("top"),
                        f"Tesseract TSV row {row_number} top",
                        minimum=0,
                        category="engine_failed",
                    ),
                    "width": finite_integer(
                        row.get("width"),
                        f"Tesseract TSV row {row_number} width",
                        minimum=1,
                        category="engine_failed",
                    ),
                    "height": finite_integer(
                        row.get("height"),
                        f"Tesseract TSV row {row_number} height",
                        minimum=1,
                        category="engine_failed",
                    ),
                    "confidence": finite_number(
                        row.get("conf"),
                        f"Tesseract TSV row {row_number} conf",
                        minimum=-1,
                        maximum=100,
                        category="engine_failed",
                    ),
                }
                if not text_value.strip():
                    continue
                parsed["text"] = text_value.strip()
                parsed_rows.append(parsed)
    except ToolError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ToolError("engine_failed", "Cannot parse Tesseract TSV output") from exc
    grouped: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in parsed_rows:
        key = (row["page_num"], row["block_num"], row["par_num"], row["line_num"])
        grouped[key].append(row)
    blocks = []
    for order, key in enumerate(sorted(grouped)):
        line_rows = grouped[key]
        text = " ".join(row["text"].strip() for row in line_rows)
        x0 = min(row["left"] for row in line_rows)
        y0 = min(row["top"] for row in line_rows)
        x1 = max(row["left"] + row["width"] for row in line_rows)
        y1 = max(row["top"] + row["height"] for row in line_rows)
        confidences = [row["confidence"] for row in line_rows if row["confidence"] >= 0]
        confidence = sum(confidences) / len(confidences) / 100 if confidences else None
        box = [float(x0), float(y0), float(x1), float(y1)]
        words = [
            {
                "text": row["text"],
                "bbox": [
                    float(row["left"]),
                    float(row["top"]),
                    float(row["left"] + row["width"]),
                    float(row["top"] + row["height"]),
                ],
                "confidence": (
                    round(row["confidence"] / 100, 6) if row["confidence"] >= 0 else None
                ),
            }
            for row in line_rows
        ]
        blocks.append(
            {
                "order": order,
                "text": text,
                "block_type": "line",
                "bbox": box,
                "polygon": bbox_to_polygon(box),
                "confidence": round(confidence, 6) if confidence is not None else None,
                "words": words,
            }
        )
    return blocks, {"tsv_rows": len(parsed_rows)}


def run_tesseract(
    executable: Path,
    render_jobs: Sequence[dict[str, Any]],
    raw_dir: Path,
    roles: dict[str, Path],
    languages: list[str],
    timeout: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pages = []
    deadline = time.monotonic() + timeout
    for job in render_jobs:
        base = raw_dir / job["image_path"].stem
        command = [
            str(executable),
            str(job["image_path"]),
            str(base),
            "-l",
            "+".join(languages),
            "--tessdata-dir",
            str(roles["tessdata_dir"]),
            "--dpi",
            str(job["dpi"]),
            "tsv",
        ]
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ToolError(
                "engine_failed",
                f"Tesseract exceeded the invocation-wide {timeout}-second deadline",
            )
        try:
            run_process(command, timeout=remaining)
        except ToolError as exc:
            if exc.category == "engine_failed" and "exceeded" in exc.message:
                raise ToolError(
                    "engine_failed",
                    f"Tesseract exceeded the invocation-wide {timeout}-second deadline",
                ) from exc
            raise
        tsv_path = base.with_suffix(".tsv")
        if not tsv_path.exists():
            raise ToolError("engine_failed", "Tesseract did not produce TSV output")
        blocks, engine_specific = tesseract_blocks(tsv_path)
        validate_block_coordinate_bounds(
            blocks,
            width=job["pixel_width"],
            height=job["pixel_height"],
            engine="Tesseract",
        )
        pages.append(
            {
                "source_page": job["source_page"],
                "source_region": job["region"],
                "classification": job["classification"],
                "page_geometry": sidecar_page_geometry(job),
                "coordinate_space": {
                    "unit": "rendered_pixel",
                    "dpi": job["dpi"],
                    "width": job["pixel_width"],
                    "height": job["pixel_height"],
                },
                "blocks": blocks,
                "warnings": [],
                "engine_specific": engine_specific,
            }
        )
    return pages, {"format": "tsv"}


def find_one_results_json(raw_dir: Path) -> Path:
    paths = sorted(raw_dir.rglob("results.json"))
    if len(paths) != 1:
        raise ToolError(
            "engine_failed",
            f"Expected one Surya results.json, found {len(paths)}",
        )
    return paths[0]


def run_surya(
    executable: Path,
    render_jobs: Sequence[dict[str, Any]],
    raw_dir: Path,
    roles: dict[str, Path],
    timeout: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_dir = raw_dir / "inputs"
    input_dir.mkdir()
    jobs_by_stem = {}
    for job in render_jobs:
        destination = input_dir / job["image_path"].name
        shutil.copy2(job["image_path"], destination)
        jobs_by_stem[destination.stem] = job
    output_dir = raw_dir / "surya-output"
    env = os.environ.copy()
    env.update(
        {
            "SURYA_INFERENCE_BACKEND": "llamacpp",
            "SURYA_INFERENCE_AUTOSTART": "1",
            "SURYA_GGUF_LOCAL_MODEL_PATH": str(roles["model"]),
            "SURYA_GGUF_LOCAL_MMPROJ_PATH": str(roles["mmproj"]),
            "LLAMA_CPP_BINARY": str(roles["backend"]),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "SURYA_INFERENCE_KEEP_ALIVE": "0",
        }
    )
    run_process(
        [str(executable), str(input_dir), "--output_dir", str(output_dir)],
        timeout=timeout,
        env=env,
    )
    result_path = find_one_results_json(output_dir)
    raw = read_engine_json(result_path, "Surya")
    if not isinstance(raw, dict):
        raise ToolError("engine_failed", "Surya results.json root must be an object")
    pages = []
    mapped_stems: set[str] = set()
    for stem, result_pages in raw.items():
        job = jobs_by_stem.get(Path(stem).stem) or jobs_by_stem.get(stem)
        if not job:
            continue
        mapped_stem = Path(stem).stem
        if mapped_stem in mapped_stems:
            raise ToolError("engine_failed", f"Duplicate Surya result key: {stem}")
        mapped_stems.add(mapped_stem)
        if not isinstance(result_pages, list) or len(result_pages) != 1:
            raise ToolError(
                "engine_failed",
                f"Surya result {stem} must contain exactly one image page",
            )
        for raw_page in result_pages:
            if not isinstance(raw_page, dict):
                raise ToolError("engine_failed", f"Surya result page for {stem} must be an object")
            raw_blocks = raw_page.get("blocks", [])
            if not isinstance(raw_blocks, list):
                raise ToolError("engine_failed", f"Surya blocks for {stem} must be an array")
            blocks = []
            page_warnings: list[str] = []
            for order, block in enumerate(raw_blocks):
                if not isinstance(block, dict):
                    raise ToolError(
                        "engine_failed",
                        f"Every Surya block for {stem} must be an object",
                    )
                raw_html = block.get("html")
                if raw_html is not None and not isinstance(raw_html, str):
                    raise ToolError(
                        "engine_failed",
                        f"Surya block {order} html for {stem} must be a string or null",
                    )
                html_value = raw_html if isinstance(raw_html, str) else None
                text = html_lib.unescape(re.sub(r"<[^>]+>", "", html_value or ""))
                reading_order, order_was_missing = normalize_order(
                    block.get("reading_order"),
                    f"Surya block {order} reading_order",
                    order,
                )
                if order_was_missing:
                    page_warnings.append(
                        f"Surya block {order} omitted reading_order; array order was retained."
                    )
                label = block.get("label")
                if label is not None and (not isinstance(label, str) or not label):
                    raise ToolError(
                        "engine_failed",
                        f"Surya block {order} label for {stem} must be a nonempty string or null",
                    )
                skipped = block.get("skipped")
                if skipped is not None and not isinstance(skipped, bool):
                    skipped = bool(skipped)
                block_error = coerce_engine_error(block.get("error"))
                blocks.append(
                    {
                        "order": reading_order,
                        "text": text,
                        "block_type": label,
                        "bbox": normalize_box(block.get("bbox"), f"Surya block {order} bbox"),
                        "polygon": normalize_polygon(
                            block.get("polygon"),
                            f"Surya block {order} polygon",
                        ),
                        "confidence": normalize_confidence(
                            block.get("confidence"),
                            f"Surya block {order} confidence",
                        ),
                        "html": html_value,
                        "markdown": None,
                        "table": (
                            html_value
                            if isinstance(html_value, str) and "<table" in html_value.lower()
                            else None
                        ),
                        "engine_specific": {
                            "raw_label": block.get("raw_label"),
                            "skipped": skipped,
                            "error": block_error,
                        },
                    }
                )
            validate_unique_block_order(blocks, "Surya")
            validate_block_coordinate_bounds(
                blocks,
                width=job["pixel_width"],
                height=job["pixel_height"],
                engine="Surya",
            )
            page_error = coerce_engine_error(raw_page.get("error"))
            pages.append(
                {
                    "source_page": job["source_page"],
                    "source_region": job["region"],
                    "classification": job["classification"],
                    "page_geometry": sidecar_page_geometry(job),
                    "coordinate_space": {
                        "unit": "rendered_pixel",
                        "dpi": job["dpi"],
                        "width": job["pixel_width"],
                        "height": job["pixel_height"],
                    },
                    "blocks": blocks,
                    "warnings": page_warnings
                    + ([f"Surya page error: {page_error}"] if page_error else []),
                    "engine_specific": {
                        "image_bbox": normalize_box(
                            raw_page.get("image_bbox"),
                            f"Surya page image_bbox for {stem}",
                        )
                    },
                }
            )
    missing = sorted(set(jobs_by_stem) - mapped_stems)
    unmapped = sorted(Path(key).stem for key in set(raw) if Path(key).stem not in jobs_by_stem)
    if missing or unmapped or len(pages) != len(render_jobs):
        raise ToolError(
            "engine_failed",
            "Surya output did not map one-to-one to rendered inputs",
            details={"missing": missing, "unmapped": unmapped},
        )
    return pages, {
        "results_path": str(result_path.relative_to(raw_dir)),
        "unmapped_result_keys": [],
    }


PADDLE_HELPER = r"""
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
common = {
    "text_detection_model_dir": config["text_detection_model_dir"],
    "text_recognition_model_dir": config["text_recognition_model_dir"],
    "use_doc_orientation_classify": False,
    "use_doc_unwarping": False,
    "use_textline_orientation": False,
    "device": config["device"],
}
if config["pipeline"] == "structure":
    from paddleocr import PPStructureV3
    ocr = PPStructureV3(
        layout_detection_model_dir=config["layout_detection_model_dir"],
        table_classification_model_dir=config["table_classification_model_dir"],
        wired_table_structure_recognition_model_dir=config[
            "wired_table_structure_recognition_model_dir"
        ],
        wireless_table_structure_recognition_model_dir=config[
            "wireless_table_structure_recognition_model_dir"
        ],
        wired_table_cells_detection_model_dir=config[
            "wired_table_cells_detection_model_dir"
        ],
        wireless_table_cells_detection_model_dir=config[
            "wireless_table_cells_detection_model_dir"
        ],
        use_seal_recognition=False,
        use_table_recognition=True,
        use_formula_recognition=False,
        use_chart_recognition=False,
        use_region_detection=False,
        **common,
    )
else:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang=config["language"], **common)
for item in config["items"]:
    results = list(ocr.predict(item["input"]))
    if len(results) != 1:
        raise RuntimeError(f"Expected one result for {item['input']}, got {len(results)}")
    results[0].save_to_json(item["output"])
    if config["pipeline"] == "structure":
        results[0].save_to_markdown(item["markdown_output"])
"""


def run_paddle(
    executable: Path,
    render_jobs: Sequence[dict[str, Any]],
    raw_dir: Path,
    roles: dict[str, Path],
    languages: list[str],
    device: str,
    pipeline: str,
    timeout: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(languages) != 1:
        raise ToolError(
            "unsupported_operation",
            "Paddle adapter requires exactly one compatible --languages value per run",
        )
    helper_path = raw_dir / "paddle_adapter.py"
    helper_path.write_text(PADDLE_HELPER, encoding="utf-8")
    items = [
        {
            "input": str(job["image_path"]),
            "output": str(raw_dir / f"{job['image_path'].stem}.json"),
            "markdown_output": str(raw_dir / f"{job['image_path'].stem}.md"),
        }
        for job in render_jobs
    ]
    config = {
        "text_detection_model_dir": str(roles["text_detection_model_dir"]),
        "text_recognition_model_dir": str(roles["text_recognition_model_dir"]),
        "language": languages[0],
        "device": device,
        "pipeline": pipeline,
        "items": items,
    }
    if pipeline == "structure":
        for role in PADDLE_STRUCTURE_ROLES:
            config[role] = str(roles[role])
    config_path = raw_dir / "paddle-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    env = os.environ.copy()
    env["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    run_process(
        [str(executable), str(helper_path), str(config_path)],
        timeout=timeout,
        env=env,
    )
    pages = []
    for job, item in zip(render_jobs, items, strict=True):
        result_path = Path(item["output"])
        root = read_engine_json(result_path, "PaddleOCR")
        if not isinstance(root, dict):
            raise ToolError("engine_failed", "PaddleOCR result root must be an object")
        result = root.get("res", root)
        if not isinstance(result, dict):
            raise ToolError("engine_failed", "PaddleOCR result payload must be an object")
        blocks = []
        page_warnings: list[str] = []
        if pipeline == "structure":
            if "parsing_res_list" not in result:
                raise ToolError(
                    "engine_failed",
                    "PaddleOCR structure result is missing parsing_res_list",
                )
            parsing_results = result["parsing_res_list"]
            if not isinstance(parsing_results, list):
                raise ToolError(
                    "engine_failed",
                    "PaddleOCR parsing_res_list must be an array",
                )
            for index, block in enumerate(parsing_results):
                if not isinstance(block, dict):
                    raise ToolError(
                        "engine_failed",
                        "Every PaddleOCR parsing result must be an object",
                    )
                box = normalize_box(
                    block.get("block_bbox"),
                    f"PaddleOCR structure block {index} bbox",
                )
                content = block.get("block_content")
                if content is not None and not isinstance(content, str):
                    raise ToolError(
                        "engine_failed",
                        f"PaddleOCR structure block {index} content must be a string or null",
                    )
                content_text = content or ""
                block_order, order_was_missing = normalize_order(
                    block.get("block_order"),
                    f"PaddleOCR structure block {index} order",
                    index,
                )
                if order_was_missing:
                    page_warnings.append(
                        f"PaddleOCR structure block {index} omitted block_order; "
                        "array order was retained."
                    )
                block_label = block.get("block_label")
                if block_label is not None and (
                    not isinstance(block_label, str) or not block_label
                ):
                    raise ToolError(
                        "engine_failed",
                        f"PaddleOCR structure block {index} label must be a "
                        "nonempty string or null",
                    )
                block_id = block.get("block_id")
                if block_id is not None:
                    block_id = finite_integer(
                        block_id,
                        f"PaddleOCR structure block {index} block_id",
                        minimum=0,
                        category="engine_failed",
                    )
                blocks.append(
                    {
                        "order": block_order,
                        "text": content_text,
                        "block_type": block_label,
                        "bbox": box,
                        "polygon": bbox_to_polygon(box),
                        "confidence": None,
                        "html": content_text
                        if block_label == "table" and "<table" in content_text.lower()
                        else None,
                        "markdown": content_text or None,
                        "table": content_text if block_label == "table" else None,
                        "engine_specific": {
                            "block_id": block_id,
                        },
                    }
                )
            validate_unique_block_order(blocks, "PaddleOCR")
        else:
            if "rec_texts" not in result or not isinstance(result["rec_texts"], list):
                raise ToolError(
                    "engine_failed",
                    "PaddleOCR recognition result requires a rec_texts array",
                )
            texts = result["rec_texts"]
            optional_arrays: dict[str, list[Any] | None] = {}
            for field in ("rec_scores", "rec_polys", "rec_boxes"):
                value = result.get(field)
                if value is not None and not isinstance(value, list):
                    raise ToolError(
                        "engine_failed",
                        f"PaddleOCR recognition field {field} must be an array or null",
                    )
                if value is not None and len(value) != len(texts):
                    raise ToolError(
                        "engine_failed",
                        f"PaddleOCR recognition field {field} length does not match rec_texts",
                    )
                optional_arrays[field] = value
            for index, text in enumerate(texts):
                if not isinstance(text, str):
                    raise ToolError(
                        "engine_failed",
                        f"PaddleOCR recognition text {index} must be a string",
                    )
                scores = optional_arrays["rec_scores"]
                polygons = optional_arrays["rec_polys"]
                boxes = optional_arrays["rec_boxes"]
                confidence = scores[index] if scores is not None else None
                polygon = (
                    normalize_polygon(
                        polygons[index],
                        f"PaddleOCR recognition polygon {index}",
                    )
                    if polygons is not None
                    else None
                )
                box = (
                    normalize_box(boxes[index], f"PaddleOCR recognition box {index}")
                    if boxes is not None
                    else None
                )
                blocks.append(
                    {
                        "order": index,
                        "text": text,
                        "block_type": "line",
                        "bbox": box,
                        "polygon": polygon or bbox_to_polygon(box),
                        "confidence": normalize_confidence(
                            confidence,
                            f"PaddleOCR recognition confidence {index}",
                        ),
                    }
                )
        validate_block_coordinate_bounds(
            blocks,
            width=job["pixel_width"],
            height=job["pixel_height"],
            engine="PaddleOCR",
        )
        pages.append(
            {
                "source_page": job["source_page"],
                "source_region": job["region"],
                "classification": job["classification"],
                "page_geometry": sidecar_page_geometry(job),
                "coordinate_space": {
                    "unit": "rendered_pixel",
                    "dpi": job["dpi"],
                    "width": job["pixel_width"],
                    "height": job["pixel_height"],
                },
                "blocks": blocks,
                "warnings": page_warnings,
                "engine_specific": {
                    "pipeline": pipeline,
                    "model_settings": result.get("model_settings"),
                    "text_det_params": result.get("text_det_params"),
                    "text_type": result.get("text_type"),
                    "layout_det_res": result.get("layout_det_res")
                    if pipeline == "structure"
                    else None,
                    "table_res_list": result.get("table_res_list")
                    if pipeline == "structure"
                    else None,
                },
            }
        )
    result_files = sorted(
        str(path.relative_to(raw_dir))
        for path in raw_dir.rglob("*.json")
        if path.name != "paddle-config.json"
    )
    markdown_files = sorted(str(path.relative_to(raw_dir)) for path in raw_dir.rglob("*.md"))
    return pages, {
        "pipeline": pipeline,
        "result_files": result_files,
        "markdown_files": markdown_files,
    }


def publish_ocr_outputs(
    normalized_output: Path,
    payload: dict[str, Any],
    raw_dir: Path,
    raw_destination: Path | None,
    *,
    overwrite: bool,
    protected_source: Path,
) -> str | None:
    require_extension(normalized_output, ".json", "Normalized OCR output")
    check_destination(
        normalized_output,
        sources=[protected_source],
        overwrite=overwrite,
    )
    if raw_destination is None:
        write_json_atomic(
            normalized_output,
            payload,
            overwrite=overwrite,
            protected_sources=[protected_source],
        )
        return None
    normalized_resolved = normalized_output.resolve()
    raw_resolved = raw_destination.resolve()
    if (
        normalized_resolved == raw_resolved
        or raw_resolved in normalized_resolved.parents
        or normalized_resolved in raw_resolved.parents
    ):
        raise ToolError(
            "bad_input",
            "Normalized and raw outputs must not contain or alias one another",
        )
    if paths_alias(raw_destination, protected_source):
        raise ToolError("bad_input", "Raw output destination must differ from the source")
    raw_existed = raw_destination.exists()
    if raw_existed:
        if not overwrite:
            raise ToolError("bad_input", f"Raw output destination exists: {raw_destination}")
        if (
            raw_destination.is_symlink()
            or not raw_destination.is_dir()
            or any(raw_destination.iterdir())
        ):
            raise ToolError("bad_input", "Raw output destination must be an empty directory")
    raw_destination.parent.mkdir(parents=True, exist_ok=True)
    staged_raw = Path(
        tempfile.mkdtemp(prefix=f".{raw_destination.name}.", dir=raw_destination.parent.resolve())
    )
    try:
        for source in raw_dir.rglob("*"):
            relative = source.relative_to(raw_dir)
            target = staged_raw / relative
            if source.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        with temporary_sibling(normalized_output, ".json") as staged_json:
            staged_json.write_text(
                json.dumps(
                    json_safe(payload),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            json.loads(staged_json.read_text(encoding="utf-8"))
            try:
                os.replace(staged_raw, raw_resolved)
                atomic_publish(staged_json, normalized_output)
            except Exception:
                shutil.rmtree(raw_resolved, ignore_errors=True)
                if raw_existed:
                    raw_destination.mkdir()
                raise
    finally:
        shutil.rmtree(staged_raw, ignore_errors=True)
    return str(raw_resolved)


def handle_ocr(args: argparse.Namespace) -> dict[str, Any]:
    password = resolve_password(args.password, args.password_env)
    languages = parse_languages(args.languages, field="OCR languages")
    source = Path(args.input)
    require_extension(source, ".pdf", "OCR input")
    output = Path(args.output)
    if not args.preflight_only:
        require_extension(output, ".json", "Normalized OCR output")
    device_metadata, device_warnings = resolve_ocr_device(args.engine, args.device)
    manifest, roles, executable = check_engine_preflight(
        args.engine,
        args.engine_executable,
        Path(args.model_manifest),
        languages,
    )
    if (
        args.engine == "paddle"
        and args.paddle_pipeline == "structure"
        and not PADDLE_STRUCTURE_ROLES.issubset(roles)
    ):
        missing_roles = sorted(PADDLE_STRUCTURE_ROLES - set(roles))
        raise ToolError(
            "license_precondition",
            "Paddle structure pipeline requires artifact roles: " + ", ".join(missing_roles),
        )
    if args.preflight_only:
        return success(
            "ocr",
            mode="preflight-only",
            engine=args.engine,
            languages=languages,
            device=device_metadata,
            warnings=device_warnings,
            preflight=manifest,
            output_path=None,
            verification={"engine_executed": False, "source_unchanged": True},
        )
    require_pdf_signature(source)
    try:
        source_sha256 = sha256_file(source)
    except OSError as exc:
        raise ToolError("bad_input", "Cannot hash OCR source before engine execution") from exc
    with tempfile.TemporaryDirectory(prefix="pdf-ocr-") as temp_name:
        work_dir = Path(temp_name)
        render_dir = work_dir / "rendered"
        raw_dir = work_dir / "raw"
        render_dir.mkdir()
        raw_dir.mkdir()
        render_jobs, warnings = render_regions_for_ocr(
            source,
            password,
            args.pages,
            args.dpi,
            render_dir,
            force=args.force,
        )
        start = time.monotonic()
        if args.engine == "tesseract":
            pages, engine_specific = run_tesseract(
                executable,
                render_jobs,
                raw_dir,
                roles,
                languages,
                args.timeout,
            )
        elif args.engine == "surya":
            pages, engine_specific = run_surya(
                executable,
                render_jobs,
                raw_dir,
                roles,
                args.timeout,
            )
        else:
            pages, engine_specific = run_paddle(
                executable,
                render_jobs,
                raw_dir,
                roles,
                languages,
                device_metadata["adapter_device"],
                args.paddle_pipeline,
                args.timeout,
            )
        elapsed = round(time.monotonic() - start, 3)
        raw_output_path = str(Path(args.raw_output_dir).resolve()) if args.raw_output_dir else None
        low_confidence_pages = []
        for page in pages:
            supplied = [
                block["confidence"]
                for block in page["blocks"]
                if block.get("confidence") is not None
            ]
            if supplied and sum(supplied) / len(supplied) < args.low_confidence_threshold:
                page["warnings"].append("Engine-supplied mean confidence is below threshold.")
                low_confidence_pages.append(page["source_page"])
        result = {
            "schema_version": SCHEMA_VERSION,
            "operation": "ocr",
            "engine": {
                "name": args.engine,
                "version": manifest["engine_version"],
                "model_identifier": manifest["model"]["identifier"],
                "model_revision": manifest["model"]["revision"],
                "requested_device": device_metadata["requested_device"],
                "resolved_device": device_metadata["resolved_device"],
                "runtime_backend": device_metadata["runtime_backend"],
                "device_backend": device_metadata["resolved_device"],
                "languages": languages,
            },
            "source": {
                "path": str(source.resolve()),
                "sha256": source_sha256,
                "selected_pages": sorted({job["source_page"] for job in render_jobs}),
                "immutable": True,
            },
            "pages": pages,
            "raw_output_path": raw_output_path,
            "warnings": warnings
            + device_warnings
            + (
                [f"Low engine-supplied confidence on pages: {low_confidence_pages}"]
                if low_confidence_pages
                else []
            ),
            "engine_specific": engine_specific,
            "preflight": manifest,
            "timing_seconds": elapsed,
        }
        try:
            current_source_sha256 = sha256_file(source)
        except OSError as exc:
            raise ToolError(
                "bad_input", "Cannot re-read OCR source after engine execution"
            ) from exc
        if current_source_sha256 != source_sha256:
            raise ToolError("bad_input", "OCR source changed during the invocation")
        raw_output_path = publish_ocr_outputs(
            output,
            result,
            raw_dir,
            Path(args.raw_output_dir) if args.raw_output_dir else None,
            overwrite=args.overwrite,
            protected_source=source,
        )
    return success(
        "ocr",
        engine=result["engine"],
        output_path=str(output.resolve()),
        raw_output_path=raw_output_path,
        counts={
            "pages_or_regions": len(result["pages"]),
            "blocks": sum(len(page["blocks"]) for page in result["pages"]),
        },
        warnings=result["warnings"],
        verification={
            "valid": True,
            "normalized_schema_version": SCHEMA_VERSION,
            "source_unchanged": True,
            "output_reopened": True,
        },
    )


def displayed_page_size(reader_page: Any) -> tuple[float, float]:
    box = reader_page.mediabox
    width = float(box.width)
    height = float(box.height)
    rotation = int(reader_page.rotation or 0) % 360
    if rotation in {90, 270}:
        return height, width
    return width, height


def displayed_to_user_transformation(rotation: int, mediabox: Any) -> Transformation:
    """Map displayed-orientation coordinates back into raw page user space."""
    origin_x = float(mediabox.left)
    origin_y = float(mediabox.bottom)
    width = float(mediabox.width)
    height = float(mediabox.height)
    if rotation == 90:
        return Transformation().rotate(90).translate(tx=origin_x + width, ty=origin_y)
    if rotation == 180:
        return Transformation().rotate(180).translate(tx=origin_x + width, ty=origin_y + height)
    if rotation == 270:
        return Transformation().rotate(270).translate(tx=origin_x, ty=origin_y + height)
    return Transformation().translate(tx=origin_x, ty=origin_y)


def pixel_bbox_to_displayed(
    bbox: Sequence[float],
    dpi: float,
    region: Sequence[float] | None,
) -> tuple[float, float, float, float]:
    scale = 72.0 / dpi
    offset_x = float(region[0]) if region else 0.0
    offset_top = float(region[1]) if region else 0.0
    return (
        float(bbox[0]) * scale + offset_x,
        float(bbox[1]) * scale + offset_top,
        float(bbox[2]) * scale + offset_x,
        float(bbox[3]) * scale + offset_top,
    )


def image_diff_fraction(first: Path, second: Path, *, channel_tolerance: int = 8) -> float:
    with Image.open(first) as image_a, Image.open(second) as image_b:
        composite_a = image_a.convert("RGB")
        composite_b = image_b.convert("RGB")
        if composite_a.size != composite_b.size:
            return 1.0
        difference = ImageChops.difference(composite_a, composite_b).convert("L")
        histogram = difference.histogram()
        total = composite_a.size[0] * composite_a.size[1]
        if not total:
            return 0.0
        return sum(histogram[channel_tolerance + 1 :]) / total


def load_compose_sidecar(sidecar_path: Path, source_path: Path) -> dict[str, Any]:
    sidecar = read_json(sidecar_path)
    if sidecar.get("operation") != "ocr":
        raise ToolError("bad_input", "Compose sidecar operation must be 'ocr'")
    source_info = sidecar.get("source")
    if not isinstance(source_info, dict):
        raise ToolError("bad_input", "Compose sidecar requires a source object")
    recorded_hash = str(source_info.get("sha256", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", recorded_hash):
        raise ToolError("bad_input", "Compose sidecar source sha256 is malformed")
    actual_hash = sha256_file(source_path)
    if actual_hash != recorded_hash:
        raise ToolError(
            "bad_input",
            "Compose sidecar was produced from a different PDF (source sha256 mismatch)",
            details={"expected_sha256": recorded_hash, "actual_sha256": actual_hash},
        )
    pages = sidecar.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ToolError("bad_input", "Compose sidecar requires a nonempty pages array")
    return sidecar


def validate_sidecar_page(entry: Any, page_count: int) -> tuple[int, float, list[float] | None]:
    if not isinstance(entry, dict):
        raise ToolError("bad_input", "Every compose sidecar page must be an object")
    source_page = finite_integer(
        entry.get("source_page"), "Sidecar source_page", minimum=1, maximum=page_count
    )
    coordinate_space = entry.get("coordinate_space")
    if not isinstance(coordinate_space, dict) or coordinate_space.get("unit") != "rendered_pixel":
        raise ToolError(
            "bad_input",
            f"Sidecar page {source_page} requires a rendered_pixel coordinate space",
        )
    dpi = finite_number(
        coordinate_space.get("dpi"),
        f"Sidecar page {source_page} dpi",
        minimum=72,
        maximum=600,
    )
    region_value = entry.get("source_region")
    region: list[float] | None = None
    if region_value is not None:
        if not isinstance(region_value, (list, tuple)) or len(region_value) != 4:
            raise ToolError(
                "bad_input",
                f"Sidecar page {source_page} source_region must have four coordinates",
            )
        region = [
            finite_number(
                value,
                f"Sidecar page {source_page} source_region[{index}]",
                minimum=0,
            )
            for index, value in enumerate(region_value)
        ]
        if region[2] <= region[0] or region[3] <= region[1]:
            raise ToolError(
                "bad_input",
                f"Sidecar page {source_page} source_region has impossible ordering",
            )
    return source_page, dpi, region


def prepare_compose_font(font_path_value: str | None) -> dict[str, Any]:
    if not font_path_value:
        return {
            "name": "Helvetica",
            "embedded": False,
            "path": None,
            "sha256": None,
            "coverage": None,
        }
    path = Path(font_path_value).expanduser()
    require_regular_file(path)
    if path.suffix.lower() not in {".ttf", ".otf"}:
        raise ToolError("bad_input", "Compose font must be a .ttf or .otf file")
    font_name = f"ComposeFont-{path.stem}"
    try:
        font = TTFont(font_name, str(path))
        pdfmetrics.registerFont(font)
    except Exception as exc:
        raise ToolError("bad_input", f"Cannot register compose font: {clean_text(exc)}") from exc
    coverage = None
    char_map = getattr(getattr(font, "face", None), "charToGlyph", None)
    if isinstance(char_map, dict) and char_map:
        coverage = set(char_map)
    return {
        "name": font_name,
        "embedded": True,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "coverage": coverage,
    }


def unmappable_compose_characters(text: str, font: dict[str, Any]) -> list[str]:
    unmappable: set[str] = set()
    if font["coverage"] is not None:
        for character in text:
            if not character.isspace() and ord(character) not in font["coverage"]:
                unmappable.add(character)
    elif not font["embedded"]:
        for character in text:
            try:
                character.encode("cp1252")
            except UnicodeEncodeError:
                unmappable.add(character)
    return sorted(unmappable)


def handle_ocr_compose(args: argparse.Namespace) -> dict[str, Any]:
    password = resolve_password(args.password, args.password_env)
    source_path = Path(args.input)
    require_extension(source_path, ".pdf", "Compose input")
    output = Path(args.output)
    require_extension(output, ".pdf", "Compose output")
    sidecar_path = Path(args.sidecar)
    sidecar = load_compose_sidecar(sidecar_path, source_path)
    source = open_pdf(source_path, password)
    if source.reader.is_encrypted and not args.allow_decrypted_output:
        raise ToolError(
            "unsupported_operation",
            "Composing an encrypted source produces a decrypted output; pass "
            "--allow-decrypted-output to authorize it",
        )
    page_count = len(source.reader.pages)
    selected = set(parse_page_selection(args.pages, page_count))
    font = prepare_compose_font(args.font)
    warnings: list[str] = []
    if font["embedded"]:
        warnings.append(
            "The declared compose font is embedded and subset into the output; the "
            "operator is responsible for its embedding license."
        )
    if source.reader.is_encrypted:
        warnings.append(
            "Encrypted source was composed into an explicitly authorized decrypted "
            "output; re-encrypt with edit if needed."
        )
    warnings.append(
        "OCR text is probabilistic; the invisible text layer inherits recognition errors."
    )
    counts = {
        "pages_total": page_count,
        "pages_composed": 0,
        "pages_skipped": 0,
        "blocks_drawn": 0,
        "words_drawn": 0,
        "characters_dropped": 0,
        "blocks_without_geometry": 0,
    }
    clamp_counts = {"font_size": 0, "horizontal_scale": 0, "page_bounds": 0}
    page_reports: list[dict[str, Any]] = []
    composed_records: dict[int, dict[str, Any]] = {}
    unmappable_report: list[dict[str, Any]] = []
    writer = PdfWriter()
    writer.clone_document_from_reader(source.reader)
    seen_entries: set[tuple[int, tuple[float, ...] | None]] = set()
    for entry in sidecar["pages"]:
        source_page, sidecar_dpi, region = validate_sidecar_page(entry, page_count)
        page_index = source_page - 1
        if page_index not in selected:
            continue
        entry_key = (source_page, tuple(region) if region else None)
        if entry_key in seen_entries:
            warnings.append(f"Duplicate sidecar entry for page {source_page} was skipped.")
            continue
        seen_entries.add(entry_key)
        classification = entry.get("classification")
        if classification == "digital-text" or (classification == "hybrid" and region is None):
            if not args.allow_duplicate_text:
                warnings.append(
                    f"Page {source_page} skipped: composing {classification} content "
                    "would duplicate extractable text; pass --allow-duplicate-text to "
                    "include it."
                )
                counts["pages_skipped"] += 1
                page_reports.append(
                    {
                        "page": source_page,
                        "blocks_drawn": 0,
                        "mean_confidence": None,
                        "skipped_reason": "duplicate-digital-text",
                    }
                )
                continue
        blocks = entry.get("blocks")
        if not isinstance(blocks, list):
            raise ToolError("bad_input", f"Sidecar page {source_page} blocks must be an array")
        reader_page = source.reader.pages[page_index]
        displayed_width, displayed_height = displayed_page_size(reader_page)
        rotation = int(reader_page.rotation or 0) % 360
        drawable_units: list[dict[str, Any]] = []
        confidences: list[float] = []
        for block in blocks:
            if not isinstance(block, dict):
                raise ToolError("bad_input", f"Sidecar page {source_page} blocks must be objects")
            confidence = block.get("confidence")
            if confidence is not None:
                confidence = finite_number(
                    confidence,
                    f"Sidecar page {source_page} block confidence",
                    minimum=0,
                    maximum=1,
                )
                confidences.append(confidence)
                if confidence < args.min_confidence:
                    continue
            block_text = block.get("text")
            if not isinstance(block_text, str) or not block_text.strip():
                continue
            words = block.get("words")
            units: list[dict[str, Any]]
            if isinstance(words, list) and words:
                units = []
                for word in words:
                    if not isinstance(word, dict):
                        raise ToolError(
                            "bad_input",
                            f"Sidecar page {source_page} words must be objects",
                        )
                    units.append(
                        {
                            "text": word.get("text"),
                            "bbox": word.get("bbox"),
                            "kind": "word",
                        }
                    )
            else:
                units = [{"text": block_text, "bbox": block.get("bbox"), "kind": "block"}]
            block_units = []
            for unit in units:
                unit_text = unit["text"]
                bbox = unit["bbox"]
                if not isinstance(unit_text, str) or not unit_text.strip():
                    continue
                if bbox is None:
                    counts["blocks_without_geometry"] += 1
                    continue
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    raise ToolError(
                        "bad_input",
                        f"Sidecar page {source_page} unit bbox must have four coordinates",
                    )
                normalized_bbox = [
                    finite_number(
                        value,
                        f"Sidecar page {source_page} unit bbox[{index}]",
                        minimum=0,
                    )
                    for index, value in enumerate(bbox)
                ]
                if normalized_bbox[2] <= normalized_bbox[0]:
                    continue
                if normalized_bbox[3] <= normalized_bbox[1]:
                    continue
                unmappable = unmappable_compose_characters(unit_text, font)
                if unmappable:
                    if args.on_unmappable == "fail":
                        unmappable_report.append(
                            {
                                "page": source_page,
                                "sample_characters": unmappable[:8],
                            }
                        )
                        continue
                    counts["characters_dropped"] += len(unit_text)
                    continue
                block_units.append(
                    {
                        "text": unit_text.strip(),
                        "bbox": normalized_bbox,
                        "kind": unit["kind"],
                        "confidence": confidence,
                    }
                )
            if block_units:
                counts["blocks_drawn"] += 1
                drawable_units.extend(
                    {**unit, "dpi": sidecar_dpi, "region": region} for unit in block_units
                )
        if args.on_unmappable == "fail" and unmappable_report:
            continue
        if not drawable_units:
            warnings.append(f"Page {source_page} skipped: no OCR blocks to compose.")
            counts["pages_skipped"] += 1
            page_reports.append(
                {
                    "page": source_page,
                    "blocks_drawn": 0,
                    "mean_confidence": (
                        round(sum(confidences) / len(confidences), 6) if confidences else None
                    ),
                    "skipped_reason": "no-blocks",
                }
            )
            continue
        buffer = io.BytesIO()
        overlay = canvas.Canvas(
            buffer, pagesize=(displayed_width, displayed_height), pageCompression=1
        )
        record = composed_records.setdefault(
            source_page,
            {
                "page": source_page,
                "anchors": [],
                "drawn_units": [],
                "spot_anchor": None,
                "units_drawn": 0,
                "blocks_drawn": 0,
            },
        )
        for unit in drawable_units:
            x0, top, x1, bottom = pixel_bbox_to_displayed(unit["bbox"], unit["dpi"], unit["region"])
            clamped_x0 = min(max(x0, 0.0), displayed_width)
            clamped_x1 = min(max(x1, 0.0), displayed_width)
            clamped_top = min(max(top, 0.0), displayed_height)
            clamped_bottom = min(max(bottom, 0.0), displayed_height)
            if (
                clamped_x0 != x0
                or clamped_x1 != x1
                or clamped_top != top
                or clamped_bottom != bottom
            ):
                clamp_counts["page_bounds"] += 1
            if clamped_x1 <= clamped_x0 or clamped_bottom <= clamped_top:
                continue
            box_height = clamped_bottom - clamped_top
            box_width = clamped_x1 - clamped_x0
            font_size = min(max(box_height, 1.0), 144.0)
            if font_size != box_height:
                clamp_counts["font_size"] += 1
            descent = pdfmetrics.getDescent(font["name"]) / 1000.0 * font_size
            baseline = displayed_height - clamped_bottom - descent
            try:
                natural_width = stringWidth(unit["text"], font["name"], font_size)
            except Exception as exc:
                raise ToolError(
                    "bad_input",
                    f"Cannot measure compose text width: {clean_text(exc)}",
                ) from exc
            horizontal_scale = 100.0
            if natural_width > 0 and box_width > 0:
                horizontal_scale = 100.0 * box_width / natural_width
                clamped_scale = min(max(horizontal_scale, 10.0), 1000.0)
                if clamped_scale != horizontal_scale:
                    clamp_counts["horizontal_scale"] += 1
                horizontal_scale = clamped_scale
            text_object = overlay.beginText()
            text_object.setTextRenderMode(3)
            text_object.setFont(font["name"], font_size)
            text_object.setTextOrigin(clamped_x0, baseline)
            text_object.setHorizScale(horizontal_scale)
            text_object.textOut(unit["text"])
            overlay.drawText(text_object)
            record["units_drawn"] += 1
            record["drawn_units"].append(unit)
            if unit["kind"] == "word":
                counts["words_drawn"] += 1
            alphanumeric = re.sub(r"[^0-9A-Za-z]", "", unit["text"])
            if (
                record["spot_anchor"] is None
                and unit["kind"] == "word"
                and " " not in unit["text"]
                and len(alphanumeric) >= 4
            ):
                record["spot_anchor"] = {
                    "text": unit["text"],
                    "x0": clamped_x0,
                    "top": clamped_top,
                }
        overlay.showPage()
        overlay.save()
        buffer.seek(0)
        overlay_reader = PdfReader(buffer)
        transformation = displayed_to_user_transformation(rotation, reader_page.mediabox)
        writer.pages[page_index].merge_transformed_page(
            overlay_reader.pages[0], transformation, over=True
        )
        drawn_candidates = sorted(
            (
                unit
                for unit in record["drawn_units"]
                if len(re.sub(r"[^0-9A-Za-z]", "", unit["text"])) >= 4
            ),
            key=lambda unit: -(unit["confidence"] or 0.0),
        )
        record["anchors"] = [{"text": unit["text"]} for unit in drawn_candidates[:5]]
        record["mean_confidence"] = (
            round(sum(confidences) / len(confidences), 6) if confidences else None
        )
    if unmappable_report:
        raise ToolError(
            "unsupported_operation",
            "Sidecar text contains characters the compose font cannot map; supply "
            "--font with coverage or use --on-unmappable skip-block",
            details={"unmappable": unmappable_report[:20]},
        )
    if not composed_records:
        raise ToolError(
            "unsupported_operation",
            "No sidecar page produced drawable OCR text for composition",
        )
    for record in composed_records.values():
        counts["pages_composed"] += 1
        page_reports.append(
            {
                "page": record["page"],
                "blocks_drawn": record["units_drawn"],
                "mean_confidence": record.get("mean_confidence"),
                "skipped_reason": None,
            }
        )
    page_reports.sort(key=lambda item: item["page"])
    if clamp_counts["font_size"]:
        warnings.append(
            f"{clamp_counts['font_size']} composed units clamped font size into 1-144 pt."
        )
    if clamp_counts["horizontal_scale"]:
        warnings.append(
            f"{clamp_counts['horizontal_scale']} composed units clamped horizontal "
            "scale into 10-1000%."
        )
    if clamp_counts["page_bounds"]:
        warnings.append(
            f"{clamp_counts['page_bounds']} composed units were clamped to the page bounds."
        )
    if counts["blocks_without_geometry"]:
        warnings.append(
            f"{counts['blocks_without_geometry']} OCR units had no geometry and were not composed."
        )
    check_destination(output, sources=[source_path, sidecar_path], overwrite=args.overwrite)
    anchors_checked = 0
    anchors_found = 0
    max_spot_delta = 0.0
    visual_diff_summary: dict[str, Any] = {"checked": False}
    with temporary_sibling(output, ".pdf") as temp_path:
        with temp_path.open("wb") as handle:
            writer.write(handle)
        verification = validate_pdf_output(temp_path, expected_pages=page_count)
        output_reader = PdfReader(str(temp_path), strict=True)
        for index in range(page_count):
            source_box = source.reader.pages[index].mediabox
            output_box = output_reader.pages[index].mediabox
            if (
                abs(float(output_box.width) - float(source_box.width)) > 0.01
                or abs(float(output_box.height) - float(source_box.height)) > 0.01
                or (int(output_reader.pages[index].rotation or 0) % 360)
                != (int(source.reader.pages[index].rotation or 0) % 360)
            ):
                raise ToolError(
                    "validation_failed",
                    f"Page {index + 1} geometry changed during composition",
                )
        missing_anchors: list[dict[str, Any]] = []
        with pdfplumber.open(str(temp_path)) as plumber_output:
            for record in composed_records.values():
                page = plumber_output.pages[record["page"] - 1]
                try:
                    page_text = page.extract_text() or ""
                except Exception:
                    page_text = ""
                normalized_page = re.sub(r"\s+", "", page_text).casefold()
                for anchor in record["anchors"]:
                    anchors_checked += 1
                    normalized_anchor = re.sub(r"\s+", "", anchor["text"]).casefold()
                    if normalized_anchor and normalized_anchor in normalized_page:
                        anchors_found += 1
                    else:
                        missing_anchors.append(
                            {"page": record["page"], "anchor": anchor["text"][:80]}
                        )
                spot = record.get("spot_anchor")
                if spot:
                    try:
                        words = page.extract_words()
                    except Exception:
                        words = []
                    matches = [
                        word
                        for word in words
                        if str(word.get("text", "")).strip().casefold()
                        == spot["text"].strip().casefold()
                    ]
                    if matches:
                        best = min(
                            matches,
                            key=lambda word: (
                                abs(float(word["x0"]) - spot["x0"])
                                + abs(float(word["top"]) - spot["top"])
                            ),
                        )
                        delta = max(
                            abs(float(best["x0"]) - spot["x0"]),
                            abs(float(best["top"]) - spot["top"]),
                        )
                        max_spot_delta = max(max_spot_delta, delta)
                        if delta > 3.0:
                            raise ToolError(
                                "validation_failed",
                                f"Composed text geometry deviates by {delta:.2f} points "
                                f"on page {record['page']}",
                            )
                    else:
                        warnings.append(
                            f"Page {record['page']} geometry spot-check anchor was not "
                            "extractable as a standalone word; the containment check "
                            "still applies."
                        )
        if missing_anchors:
            raise ToolError(
                "validation_failed",
                "Composed OCR text is not extractable from the output",
                details={"missing_anchors": missing_anchors[:10]},
            )
        if not args.skip_visual_check:
            max_fraction = 0.0
            visual_budget = RenderBudget(
                total_message=(
                    "Compose visual check exceeds the render pixel budget; lower "
                    "--visual-check-dpi, select fewer pages, or pass --skip-visual-check"
                )
            )
            with tempfile.TemporaryDirectory(prefix="pdf-compose-check-") as check_name:
                check_dir = Path(check_name)
                with (
                    pdfplumber.open(str(source_path), password=password) as before_pdf,
                    pdfplumber.open(str(temp_path)) as after_pdf,
                ):
                    for record in sorted(composed_records):
                        page_number = record
                        before_page = before_pdf.pages[page_number - 1]
                        after_page = after_pdf.pages[page_number - 1]
                        for _ in range(2):
                            visual_budget.charge(
                                float(before_page.width),
                                float(before_page.height),
                                args.visual_check_dpi,
                                page_label=f"Page {page_number}",
                            )
                        before_png = check_dir / f"before-{page_number:04d}.png"
                        after_png = check_dir / f"after-{page_number:04d}.png"
                        rasterize_region(
                            before_page,
                            None,
                            args.visual_check_dpi,
                            before_png,
                            page_label=f"page {page_number}",
                        )
                        rasterize_region(
                            after_page,
                            None,
                            args.visual_check_dpi,
                            after_png,
                            page_label=f"page {page_number}",
                        )
                        fraction = image_diff_fraction(before_png, after_png)
                        max_fraction = max(max_fraction, fraction)
                        if fraction > args.max_visual_diff:
                            raise ToolError(
                                "validation_failed",
                                f"Invisible text layer visibly changed page "
                                f"{page_number} (diff fraction {fraction:.6f} exceeds "
                                f"{args.max_visual_diff})",
                            )
            visual_diff_summary = {
                "checked": True,
                "dpi": args.visual_check_dpi,
                "max_fraction_changed": round(max_fraction, 6),
                "threshold": args.max_visual_diff,
            }
        atomic_publish(temp_path, output)
    low_confidence_pages = [
        report["page"]
        for report in page_reports
        if report["skipped_reason"] is None
        and report["mean_confidence"] is not None
        and report["mean_confidence"] < 0.5
    ]
    if low_confidence_pages:
        warnings.append(f"Low engine-supplied confidence on composed pages: {low_confidence_pages}")
    return success(
        "ocr-compose",
        output_path=str(output.resolve()),
        source=str(source_path.resolve()),
        sidecar_path=str(sidecar_path.resolve()),
        sidecar_source_sha256_matched=True,
        font={
            "name": font["name"],
            "embedded": font["embedded"],
            "path": font["path"],
            "sha256": font["sha256"],
        },
        counts=counts,
        pages=page_reports,
        warnings=warnings,
        verification={
            **verification,
            "page_count_unchanged": True,
            "dimensions_unchanged": True,
            "anchors": {"checked": anchors_checked, "found": anchors_found},
            "geometry_spot_check_max_delta_points": round(max_spot_delta, 3),
            "visual_diff": visual_diff_summary,
        },
    )


def add_password_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--password", help="PDF password; never emitted")
    parser.add_argument(
        "--password-env",
        help="Read the PDF password from this environment variable",
    )


def add_overwrite_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing distinct destination",
    )


def add_extract_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True)
    parser.add_argument("--pages", default="all", help="1-based list/ranges or all")
    parser.add_argument("--mode", choices=("plain", "layout"), default="plain")
    parser.add_argument("--words", action="store_true")
    parser.add_argument("--tables", action="store_true")
    parser.add_argument("--output", help="Optional JSON sidecar")
    add_password_options(parser)
    add_overwrite_option(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = JsonArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=TOOL_VERSION)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inventory a PDF")
    add_extract_options(inspect_parser)
    inspect_parser.set_defaults(handler=handle_inspect_or_extract)

    extract_parser = subparsers.add_parser("extract", help="Extract PDF content")
    add_extract_options(extract_parser)
    extract_parser.add_argument("--images-dir", help="Extract embedded images")
    extract_parser.set_defaults(handler=handle_inspect_or_extract)

    render_parser = subparsers.add_parser(
        "render", help="Rasterize pages to PNG files for visual review"
    )
    render_parser.add_argument("--input", required=True)
    render_parser.add_argument(
        "--output", required=True, help="New or empty directory for one PNG per page"
    )
    render_parser.add_argument("--pages", default="all", help="1-based list/ranges or all")
    render_parser.add_argument("--dpi", type=int, default=DEFAULT_RENDER_DPI)
    add_password_options(render_parser)
    add_overwrite_option(render_parser)
    render_parser.set_defaults(handler=handle_render)

    create_parser = subparsers.add_parser("create", help="Create a PDF from JSON")
    create_parser.add_argument("--job", required=True)
    create_parser.add_argument("--output", required=True)
    add_overwrite_option(create_parser)
    create_parser.set_defaults(handler=handle_create)

    pages_parser = subparsers.add_parser("pages", help="Merge/split/reorder pages from JSON")
    pages_parser.add_argument("--job", required=True)
    add_overwrite_option(pages_parser)
    pages_parser.set_defaults(handler=handle_pages)

    edit_parser = subparsers.add_parser("edit", help="Stamp, fill, secure, or edit metadata")
    edit_parser.add_argument("--input", required=True)
    edit_parser.add_argument("--job")
    edit_parser.add_argument("--output")
    edit_parser.add_argument("--read-fields", action="store_true")
    add_password_options(edit_parser)
    add_overwrite_option(edit_parser)
    edit_parser.set_defaults(handler=handle_edit)

    convert_parser = subparsers.add_parser("convert", help="Perform an explicit lossy mapping")
    convert_parser.add_argument("--input")
    convert_parser.add_argument("--images", nargs="+")
    convert_parser.add_argument(
        "--to",
        required=True,
        choices=("txt", "json", "tables-csv", "pdf"),
    )
    convert_parser.add_argument("--output", required=True)
    convert_parser.add_argument("--pages", default="all")
    convert_parser.add_argument("--mode", choices=("plain", "layout"), default="plain")
    convert_parser.add_argument("--words", action="store_true")
    convert_parser.add_argument("--tables", action="store_true")
    convert_parser.add_argument("--encoding", default="utf-8")
    convert_parser.add_argument("--page-size", default="letter")
    convert_parser.add_argument("--margin", type=float, default=36)
    add_password_options(convert_parser)
    add_overwrite_option(convert_parser)
    convert_parser.set_defaults(handler=handle_convert)

    plan_parser = subparsers.add_parser("ocr-plan", help="Recommend OCR without running it")
    plan_parser.add_argument("--input", required=True)
    plan_parser.add_argument("--pages", default="all")
    plan_parser.add_argument("--languages", required=True)
    plan_parser.add_argument("--hardware", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    plan_parser.add_argument("--layout", choices=("auto", "simple", "complex"), default="auto")
    plan_parser.add_argument("--volume-hint", type=int)
    plan_parser.add_argument("--dpi", type=int, default=300)
    plan_parser.add_argument("--model-manifest")
    add_password_options(plan_parser)
    plan_parser.set_defaults(handler=handle_ocr_plan)

    ocr_parser = subparsers.add_parser("ocr", help="Run one explicitly selected local OCR engine")
    ocr_parser.add_argument("--input", required=True)
    ocr_parser.add_argument("--output", required=True)
    ocr_parser.add_argument("--engine", required=True, choices=("surya", "paddle", "tesseract"))
    ocr_parser.add_argument("--engine-executable")
    ocr_parser.add_argument("--model-manifest", required=True)
    ocr_parser.add_argument("--languages", required=True)
    ocr_parser.add_argument("--pages", default="all")
    ocr_parser.add_argument("--dpi", type=int, default=300)
    ocr_parser.add_argument("--timeout", type=int, default=600)
    ocr_parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    ocr_parser.add_argument("--paddle-pipeline", choices=("ocr", "structure"), default="ocr")
    ocr_parser.add_argument("--force", action="store_true")
    ocr_parser.add_argument("--preflight-only", action="store_true")
    ocr_parser.add_argument("--raw-output-dir")
    ocr_parser.add_argument("--low-confidence-threshold", type=float, default=0.5)
    add_password_options(ocr_parser)
    add_overwrite_option(ocr_parser)
    ocr_parser.set_defaults(handler=handle_ocr)

    compose_parser = subparsers.add_parser(
        "ocr-compose",
        help="Compose a searchable PDF from a completed OCR sidecar without running an engine",
    )
    compose_parser.add_argument("--input", required=True)
    compose_parser.add_argument("--sidecar", required=True, help="Completed OCR sidecar JSON")
    compose_parser.add_argument("--output", required=True)
    compose_parser.add_argument("--pages", default="all")
    compose_parser.add_argument("--min-confidence", type=float, default=0.0)
    compose_parser.add_argument(
        "--font", help="TTF/OTF font for text the base-14 default cannot map"
    )
    compose_parser.add_argument("--on-unmappable", choices=("fail", "skip-block"), default="fail")
    compose_parser.add_argument("--allow-duplicate-text", action="store_true")
    compose_parser.add_argument("--max-visual-diff", type=float, default=0.002)
    compose_parser.add_argument("--visual-check-dpi", type=int, default=DEFAULT_VISUAL_CHECK_DPI)
    compose_parser.add_argument("--skip-visual-check", action="store_true")
    compose_parser.add_argument("--allow-decrypted-output", action="store_true")
    add_password_options(compose_parser)
    add_overwrite_option(compose_parser)
    compose_parser.set_defaults(handler=handle_ocr_compose)

    manifest_parser = subparsers.add_parser(
        "manifest", help="Derive or check an OCR model manifest without running any engine"
    )
    manifest_parser.add_argument("--mode", required=True, choices=("derive", "check"))
    manifest_parser.add_argument(
        "--engine", required=True, choices=("surya", "paddle", "tesseract")
    )
    manifest_parser.add_argument("--languages")
    manifest_parser.add_argument("--tessdata-dir")
    manifest_parser.add_argument("--assume-source", choices=("tessdata_fast", "tessdata_best"))
    manifest_parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="ROLE=PATH",
        help="Provisioned local artifact for derive",
    )
    manifest_parser.add_argument("--accept-license", action="store_true")
    manifest_parser.add_argument("--confirm-eligibility", action="store_true")
    manifest_parser.add_argument("--model-manifest", help="Existing manifest for --mode check")
    manifest_parser.add_argument("--output", help="Derived manifest destination (.json)")
    add_overwrite_option(manifest_parser)
    manifest_parser.set_defaults(handler=handle_manifest)

    redact_parser = subparsers.add_parser(
        "redact", help="Remove content under target regions; not a cosmetic overlay"
    )
    redact_parser.add_argument("--input", required=True)
    redact_parser.add_argument("--job", required=True, help="Redaction job JSON")
    redact_parser.add_argument("--output", required=True)
    redact_parser.add_argument("--report", help="Optional full report destination (.json)")
    redact_parser.add_argument(
        "--render-check-dir",
        help="Publish before/after verification rasters to this new directory",
    )
    add_overwrite_option(redact_parser)
    redact_parser.set_defaults(handler=handle_redact)

    return parser


def validate_cli_limits(args: argparse.Namespace) -> None:
    if hasattr(args, "dpi") and not 72 <= args.dpi <= 600:
        raise ToolError("bad_input", "DPI must be between 72 and 600")
    if hasattr(args, "visual_check_dpi") and not 72 <= args.visual_check_dpi <= 600:
        raise ToolError("bad_input", "Visual-check DPI must be between 72 and 600")
    if hasattr(args, "min_confidence"):
        finite_number(args.min_confidence, "Minimum confidence", minimum=0, maximum=1)
    if hasattr(args, "max_visual_diff"):
        finite_number(args.max_visual_diff, "Maximum visual diff", minimum=0, maximum=1)
    if hasattr(args, "timeout") and not 1 <= args.timeout <= 86_400:
        raise ToolError("bad_input", "Timeout must be between 1 and 86400 seconds")
    if hasattr(args, "low_confidence_threshold"):
        finite_number(
            args.low_confidence_threshold,
            "Low-confidence threshold",
            minimum=0,
            maximum=1,
        )
    if hasattr(args, "volume_hint") and args.volume_hint is not None and args.volume_hint < 1:
        raise ToolError("bad_input", "Volume hint must be positive")
    if hasattr(args, "margin"):
        finite_number(args.margin, "Margin", minimum=0)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        for logger_name in ("pypdf", "pdfminer", "pdfplumber", "PIL"):
            logging.getLogger(logger_name).setLevel(logging.CRITICAL)
        parser = build_parser()
        args = parser.parse_args(argv)
        validate_cli_limits(args)
        payload = args.handler(args)
        emit_stdout(payload)
        return 0
    except ToolError as exc:
        emit_stderr(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "error",
                "category": exc.category,
                "message": clean_text(exc.message),
                "details": json_safe(exc.details),
            }
        )
        return EXIT_CODES.get(exc.category, EXIT_CODES["internal_error"])
    except KeyboardInterrupt:
        emit_stderr(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "error",
                "category": "engine_failed",
                "message": "Interrupted",
            }
        )
        return EXIT_CODES["engine_failed"]
    except Exception as exc:
        emit_stderr(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "error",
                "category": "internal_error",
                "message": clean_text(exc),
            }
        )
        return EXIT_CODES["internal_error"]


if __name__ == "__main__":
    raise SystemExit(main())
