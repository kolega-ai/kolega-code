#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Safely inspect, create, edit, and convert DOCX documents."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

try:
    from docx import Document
    from docx.document import Document as DocumentObject
    from docx.enum.section import WD_ORIENT, WD_SECTION
    from docx.enum.style import WD_STYLE_TYPE
    from docx.enum.text import (
        WD_ALIGN_PARAGRAPH,
        WD_BREAK,
        WD_LINE_SPACING,
        WD_TAB_ALIGNMENT,
        WD_TAB_LEADER,
    )
    from docx.image.image import Image
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.section import _Footer, _Header
    from docx.shared import Inches, Pt, RGBColor
    from docx.table import Table, _Cell
    from docx.text.paragraph import Paragraph
    from docx.text.run import Run
    from lxml import etree
    from PIL import Image as PILImage
    from pypdf import PdfReader
    from pypdf import filters as pdf_filters
    from pypdf.errors import LimitReachedError
except ModuleNotFoundError as exc:
    print(
        json.dumps(
            {
                "schema_version": 1,
                "status": "error",
                "error": {
                    "category": "missing_dependency",
                    "message": "A required Python distribution is not installed.",
                    "details": {"module": exc.name},
                },
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    raise SystemExit(4) from exc

SCHEMA_VERSION = 1
MAX_PACKAGE_BYTES = 128 * 1024 * 1024
MAX_EXPANDED_BYTES = 256 * 1024 * 1024
MAX_MEMBERS = 2_048
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_PDF_BYTES = 64 * 1024 * 1024
MAX_PDF_PAGES = 1_000
MAX_PDF_TEXT_PAGES = 50
MAX_PDF_TEXT_CHARACTERS = 1_000_000
MAX_PDF_DECOMPRESSED_STREAM_BYTES = 8 * 1024 * 1024
DEFAULT_CONVERSION_TIMEOUT = 90
PROCESS_TERMINATION_GRACE_SECONDS = 2.0
MAX_SUBPROCESS_DIAGNOSTIC_BYTES = 4_000
MIN_RENDER_DPI = 36
MAX_RENDER_DPI = 300
DEFAULT_RENDER_DPI = 150
MAX_RENDER_PAGES = 200
MAX_RENDER_PIXELS_PER_PAGE = 25_000_000
MAX_RENDER_TOTAL_PIXELS = 250_000_000

# Portable: shipped with both Windows and macOS for decades; no substitution warning.
PORTABLE_FONTS = {
    "arial",
    "arial black",
    "comic sans ms",
    "courier new",
    "georgia",
    "impact",
    "tahoma",
    "times new roman",
    "trebuchet ms",
    "verdana",
}
# Common: bundled with Microsoft Office or one major OS; metric-compatible LibreOffice
# substitutes exist for the Office defaults (Carlito for Calibri, Caladea for Cambria).
# Courier, Symbol, and Wingdings appear in default Word/python-docx templates (bullet and
# mono glyphs) and are substituted essentially everywhere; blocking them would flag every
# default-created document.
COMMON_FONTS = {
    "aptos",
    "aptos display",
    "calibri",
    "calibri light",
    "cambria",
    "cambria math",
    "candara",
    "consolas",
    "constantia",
    "corbel",
    "courier",
    "franklin gothic",
    "helvetica",
    "segoe ui",
    "segoe ui light",
    "segoe ui semibold",
    "symbol",
    "wingdings",
}

EXIT_CODES = {
    "bad_input": 2,
    "unsupported_operation": 3,
    "missing_dependency": 4,
    "ambiguous_edit": 5,
    "resource_limit": 6,
    "licensing_precondition": 7,
    "validation_failed": 8,
    "output_conflict": 9,
    "external_tool_failed": 10,
}

WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
A14_NS = "http://schemas.microsoft.com/office/drawing/2010/main"
W14_NS = "http://schemas.microsoft.com/office/word/2010/wordml"
DECORATIVE_EXTENSION_URI = "{C183D7F6-B498-43B3-948B-1728B52AA6E4}"

PARAGRAPH_LAYOUT_KEYS = {
    "space_before_pt",
    "space_after_pt",
    "line_spacing",
    "left_indent_inches",
    "right_indent_inches",
    "first_line_indent_inches",
    "hanging_indent_inches",
    "keep_with_next",
    "keep_together",
    "page_break_before",
    "widow_control",
    "tab_stops",
}
TABLE_KEYS = {
    "type",
    "rows",
    "style",
    "header_rows",
    "layout",
    "column_widths_inches",
    "allow_row_split",
    "row_allow_split",
}
SECTION_KEYS = {
    "paper_size",
    "page_width_inches",
    "page_height_inches",
    "orientation",
    "top_margin_inches",
    "right_margin_inches",
    "bottom_margin_inches",
    "left_margin_inches",
    "header_distance_inches",
    "footer_distance_inches",
    "different_first_page",
    "page_number_start",
    "page_number_format",
}

REVISION_TAGS = {
    qn("w:ins"),
    qn("w:del"),
    qn("w:moveFrom"),
    qn("w:moveTo"),
    qn("w:moveFromRangeStart"),
    qn("w:moveFromRangeEnd"),
    qn("w:moveToRangeStart"),
    qn("w:moveToRangeEnd"),
}
REVISION_WRAPPER_TAGS = {
    qn("w:ins"),
    qn("w:del"),
    qn("w:moveFrom"),
    qn("w:moveTo"),
}
REVISION_RANGE_START_TAGS = {
    qn("w:moveFromRangeStart"),
    qn("w:moveToRangeStart"),
}
REVISION_RANGE_END_TAGS = {
    qn("w:moveFromRangeEnd"),
    qn("w:moveToRangeEnd"),
}

UNSUPPORTED_PART_PREFIXES = (
    "word/activeX/",
    "word/charts/",
    "word/diagrams/",
    "word/embeddings/",
    "word/glossary/",
    "word/ink/",
    "word/webExtensions/",
    "_xmlsignatures/",
)
STANDARD_CUSTOM_XML_PARTS = frozenset(
    {
        "customXml/_rels/item1.xml.rels",
        "customXml/item1.xml",
        "customXml/itemProps1.xml",
    }
)


class ToolError(Exception):
    """Expected failure with a stable machine-readable category."""

    def __init__(
        self,
        category: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.message = message
        self.details = details or {}


class JsonArgumentParser(argparse.ArgumentParser):
    """Route argument errors through the JSON failure contract."""

    def error(self, message: str) -> NoReturn:
        raise ToolError("bad_input", "Invalid command-line arguments.", details={"reason": message})


@dataclass
class PreflightReport:
    """Validated package facts used by later operations."""

    path: Path
    member_count: int
    expanded_bytes: int
    warnings: list[dict[str, Any]]
    members: list[str]


@dataclass
class TextPiece:
    """A text interval and its run-level edit eligibility."""

    start: int
    end: int
    text: str
    run_element: Any | None
    unsafe_reason: str | None


@dataclass
class ParagraphTextMap:
    """Visible paragraph text with run and boundary information."""

    paragraph: Paragraph
    text: str
    pieces: list[TextPiece]
    boundaries: list[tuple[int, str]]


def diagnostic(
    level: str,
    code: str,
    message: str,
    **details: Any,
) -> None:
    """Write one redacted JSON diagnostic to stderr."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "level": level,
        "code": code,
        "message": message,
    }
    if details:
        payload["details"] = redact(details)
    print(json.dumps(payload, sort_keys=True), file=sys.stderr)


def redact(value: Any) -> Any:
    """Redact credential-shaped values before diagnostics are emitted."""

    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if re.search(r"password|passwd|secret|token|credential", str(key), re.I):
                result[str(key)] = "[REDACTED]"
            else:
                result[str(key)] = redact(item)
        return result
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return re.sub(
            r"(?i)(password|passwd|secret|token)=([^\s&]+)",
            r"\1=[REDACTED]",
            value,
        )
    return value


def package_versions(include_libreoffice: str | None = None) -> dict[str, str | None]:
    """Return versions relevant to reproducibility."""

    versions: dict[str, str | None] = {
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "python-docx": distribution_version("python-docx"),
        "lxml": distribution_version("lxml"),
        "pypdf": distribution_version("pypdf"),
        "pillow": distribution_version("Pillow"),
        "pypdfium2": optional_distribution_version("pypdfium2"),
        "libreoffice": include_libreoffice,
    }
    return versions


def distribution_version(name: str) -> str:
    """Read a distribution version without importing private package state."""

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise ToolError(
            "missing_dependency",
            f"Required Python distribution is not installed: {name}",
            details={"distribution": name},
        ) from exc


def optional_distribution_version(name: str) -> str | None:
    """Read an optional distribution version, or None when it is not installed."""

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def secure_xml_parser() -> etree.XMLParser:
    """Build a parser that cannot load DTDs, entities, or network resources."""

    return etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        load_dtd=False,
        dtd_validation=False,
        recover=False,
        huge_tree=False,
        remove_comments=False,
    )


def parse_xml_safely(data: bytes, member: str) -> etree._Element:
    """Parse package XML after an explicit DTD prohibition."""

    if re.search(rb"<!DOCTYPE|<!ENTITY", data, re.I):
        raise ToolError(
            "bad_input",
            "DTD and entity declarations are not allowed in OOXML.",
            details={"member": member},
        )
    try:
        return etree.fromstring(data, parser=secure_xml_parser())
    except etree.XMLSyntaxError as exc:
        raise ToolError(
            "bad_input",
            "Malformed XML in OOXML package.",
            details={"member": member, "reason": str(exc)},
        ) from exc


def preflight_docx(
    path_value: str | Path,
    *,
    allow_external_relationships: bool = False,
) -> PreflightReport:
    """Validate a DOCX signature, ZIP bounds, content type, and package XML."""

    path = Path(path_value).expanduser()
    if path.suffix.lower() != ".docx":
        raise ToolError(
            "bad_input",
            "Expected a .docx input; macro-enabled and other Office formats are rejected.",
            details={"path": str(path)},
        )
    if not path.is_file():
        raise ToolError("bad_input", "DOCX input does not exist.", details={"path": str(path)})
    size = path.stat().st_size
    if size > MAX_PACKAGE_BYTES:
        raise ToolError(
            "resource_limit",
            "Compressed DOCX exceeds the package size limit.",
            details={"path": str(path), "bytes": size, "limit": MAX_PACKAGE_BYTES},
        )
    try:
        with path.open("rb") as handle:
            signature = handle.read(4)
    except OSError as exc:
        raise ToolError(
            "bad_input",
            "DOCX input could not be read.",
            details={"path": str(path), "reason": str(exc)},
        ) from exc
    if signature not in {b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"}:
        raise ToolError(
            "bad_input",
            "Input does not have a ZIP/OOXML signature.",
            details={"path": str(path)},
        )

    warnings: list[dict[str, Any]] = []
    required = {"[Content_Types].xml", "_rels/.rels", "word/document.xml"}
    expanded = 0
    try:
        with zipfile.ZipFile(path) as package:
            infos = package.infolist()
            if len(infos) > MAX_MEMBERS:
                raise ToolError(
                    "resource_limit",
                    "OOXML package has too many ZIP members.",
                    details={"members": len(infos), "limit": MAX_MEMBERS},
                )
            names = [info.filename for info in infos]
            duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
            if duplicates:
                raise ToolError(
                    "bad_input",
                    "OOXML package contains duplicate ZIP member names.",
                    details={"members": duplicates[:20]},
                )
            missing = sorted(required.difference(names))
            if missing:
                raise ToolError(
                    "bad_input",
                    "OOXML package is missing required members.",
                    details={"members": missing},
                )
            for info in infos:
                member_path = PurePosixPath(info.filename)
                if (
                    info.filename.startswith(("/", "\\"))
                    or "\\" in info.filename
                    or ".." in member_path.parts
                ):
                    raise ToolError(
                        "bad_input",
                        "OOXML package contains an unsafe member path.",
                        details={"member": info.filename},
                    )
                if info.flag_bits & 0x1:
                    raise ToolError(
                        "unsupported_operation",
                        "Encrypted OOXML packages are not supported.",
                        details={"member": info.filename},
                    )
                expanded += info.file_size
                if expanded > MAX_EXPANDED_BYTES:
                    raise ToolError(
                        "resource_limit",
                        "OOXML expanded size exceeds the safety limit.",
                        details={"bytes": expanded, "limit": MAX_EXPANDED_BYTES},
                    )
                if info.filename.lower().endswith((".xml", ".rels")):
                    parse_xml_safely(package.read(info), info.filename)

            content_types = package.read("[Content_Types].xml")
            lowered_types = content_types.lower()
            if b"macroenabled" in lowered_types or any(
                name.lower().endswith("vbaproject.bin") for name in names
            ):
                raise ToolError(
                    "unsupported_operation",
                    "Macro-enabled OOXML packages are not supported.",
                    details={"path": str(path)},
                )
            for rel_name in (name for name in names if name.endswith(".rels")):
                root = parse_xml_safely(package.read(rel_name), rel_name)
                external = root.xpath(
                    "count(//*[local-name()='Relationship' and @TargetMode='External'])"
                )
                if external:
                    details = {
                        "part": rel_name,
                        "count": int(external),
                    }
                    if not allow_external_relationships:
                        raise ToolError(
                            "bad_input",
                            "DOCX external relationships are rejected by default.",
                            details=details,
                        )
                    warnings.append(
                        {
                            "code": "external_relationships_allowed",
                            "message": (
                                "Package contains external relationships and processing was "
                                "explicitly allowed; consuming applications may access external "
                                "targets."
                            ),
                            **details,
                        }
                    )
    except zipfile.BadZipFile as exc:
        raise ToolError("bad_input", "Input is not a well-formed ZIP package.") from exc
    except OSError as exc:
        raise ToolError(
            "bad_input",
            "OOXML package could not be read.",
            details={"path": str(path), "reason": str(exc)},
        ) from exc

    return PreflightReport(path, len(infos), expanded, warnings, names)


def load_document(path: Path) -> DocumentObject:
    """Open a preflighted document and normalize library errors."""

    try:
        return Document(str(path))
    except Exception as exc:
        raise ToolError(
            "bad_input",
            "DOCX could not be opened by python-docx.",
            details={"path": str(path), "reason": str(exc)},
        ) from exc


def validate_schema_version(payload: Any, source: str) -> dict[str, Any]:
    """Require the stable schema marker on every JSON input."""

    if not isinstance(payload, dict):
        raise ToolError("bad_input", "JSON input must be an object.", details={"source": source})
    received = payload.get("schema_version")
    if isinstance(received, bool) or received != SCHEMA_VERSION:
        raise ToolError(
            "bad_input",
            "JSON input must contain schema_version: 1.",
            details={"source": source, "received": received},
        )
    return payload


def load_json(path_value: str | Path) -> tuple[dict[str, Any], Path]:
    """Load a versioned JSON object and return its path-relative base directory."""

    path = Path(path_value).expanduser()
    if not path.is_file():
        raise ToolError("bad_input", "JSON input does not exist.", details={"path": str(path)})
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ToolError(
            "bad_input",
            "JSON input could not be decoded.",
            details={"path": str(path), "reason": str(exc)},
        ) from exc
    return validate_schema_version(payload, str(path)), path.resolve().parent


def resolve_path(value: Any, base_dir: Path, field: str) -> Path:
    """Resolve one required path field relative to its JSON file."""

    if not isinstance(value, str) or not value.strip():
        raise ToolError("bad_input", f"{field} must be a non-empty path string.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path


def optional_path(value: Any, base_dir: Path, field: str) -> Path | None:
    """Resolve an optional path."""

    if value is None:
        return None
    return resolve_path(value, base_dir, field)


def warning(code: str, message: str, **details: Any) -> dict[str, Any]:
    """Create a warning object."""

    item: dict[str, Any] = {"code": code, "message": message}
    item.update(details)
    return item


def run_format(run: Run) -> dict[str, Any]:
    """Serialize formatting that is stable through public python-docx APIs."""

    color = run.font.color.rgb
    size = run.font.size
    return {
        "bold": run.bold,
        "italic": run.italic,
        "underline": run.underline,
        "font_name": run.font.name,
        "font_size_pt": round(size.pt, 3) if size is not None else None,
        "color": str(color) if color is not None else None,
        "style": run.style.name if run.style is not None else None,
    }


def paragraph_record(paragraph: Paragraph, index: int) -> dict[str, Any]:
    """Serialize a paragraph, including list and run structure."""

    style = paragraph.style.name if paragraph.style is not None else None
    p_pr = paragraph._p.pPr
    num_pr = p_pr.numPr if p_pr is not None else None
    list_info = None
    if num_pr is not None:
        list_info = {
            "num_id": int(num_pr.numId.val) if num_pr.numId is not None else None,
            "level": int(num_pr.ilvl.val) if num_pr.ilvl is not None else None,
            "source": "paragraph",
        }
    elif (style or "").lower().startswith(("list bullet", "list number")):
        list_info = {"num_id": None, "level": None, "source": "style"}
    heading = None
    match = re.fullmatch(r"Heading\s+([1-9])", style or "", re.I)
    if match:
        heading = int(match.group(1))
    paragraph_format = paragraph.paragraph_format
    tabs = []
    for tab in paragraph_format.tab_stops:
        tabs.append(
            {
                "position_inches": length_inches(tab.position),
                "alignment": tab.alignment.name.lower() if tab.alignment is not None else None,
                "leader": tab.leader.name.lower() if tab.leader is not None else None,
            }
        )
    line_spacing = paragraph_format.line_spacing
    layout = {
        "space_before_pt": (
            round(paragraph_format.space_before.pt, 3)
            if paragraph_format.space_before is not None
            else None
        ),
        "space_after_pt": (
            round(paragraph_format.space_after.pt, 3)
            if paragraph_format.space_after is not None
            else None
        ),
        "line_spacing": (
            round(line_spacing.pt, 3)
            if hasattr(line_spacing, "pt")
            else round(float(line_spacing), 3)
            if line_spacing is not None
            else None
        ),
        "line_spacing_rule": (
            paragraph_format.line_spacing_rule.name.lower()
            if paragraph_format.line_spacing_rule is not None
            else None
        ),
        "left_indent_inches": length_inches(paragraph_format.left_indent),
        "right_indent_inches": length_inches(paragraph_format.right_indent),
        "first_line_indent_inches": length_inches(paragraph_format.first_line_indent),
        "keep_with_next": paragraph_format.keep_with_next,
        "keep_together": paragraph_format.keep_together,
        "page_break_before": paragraph_format.page_break_before,
        "widow_control": paragraph_format.widow_control,
        "tab_stops": tabs,
    }
    return {
        "index": index,
        "text": paragraph.text,
        "style": style,
        "heading_level": heading,
        "list": list_info,
        "alignment": paragraph.alignment.name if paragraph.alignment is not None else None,
        "layout": layout,
        "runs": [
            {"index": run_index, "text": run.text, "format": run_format(run)}
            for run_index, run in enumerate(paragraph.runs)
        ],
    }


def table_record(table: Table, index: int, section_index: int | None = None) -> dict[str, Any]:
    """Serialize a table and its cell paragraphs."""

    rows: list[list[dict[str, Any]]] = []
    row_allows_split: list[bool] = []
    for row in table.rows:
        cells: list[dict[str, Any]] = []
        for cell in row.cells:
            cells.append(
                {
                    "text": cell.text,
                    "paragraphs": [
                        paragraph_record(paragraph, p_index)
                        for p_index, paragraph in enumerate(cell.paragraphs)
                    ],
                }
            )
        rows.append(cells)
        tr_pr = row._tr.trPr
        cant_split = tr_pr.find(qn("w:cantSplit")) if tr_pr is not None else None
        cant_split_value = cant_split.get(qn("w:val")) if cant_split is not None else None
        row_allows_split.append(cant_split is None or cant_split_value in {"0", "false", "off"})
    autofit = table.autofit
    header_rows = 0
    for row in table.rows:
        repeat = row._tr.get_or_add_trPr().find(qn("w:tblHeader"))
        if repeat is None or repeat.get(qn("w:val")) in {"0", "false", "off"}:
            break
        header_rows += 1
    return {
        "index": index,
        "section_index": section_index,
        "style": table.style.name if table.style is not None else None,
        "layout": ("autofit" if autofit is True else "fixed" if autofit is False else "default"),
        "column_widths_inches": [length_inches(column.width) for column in table.columns],
        "row_allows_split": row_allows_split,
        "header_rows": header_rows,
        "rows": rows,
        "row_count": len(table.rows),
        "column_count": max((len(row.cells) for row in table.rows), default=0),
    }


def iter_body_blocks(document: DocumentObject) -> Iterator[tuple[str, Paragraph | Table]]:
    """Yield body paragraphs and tables in document order."""

    body = document.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield "paragraph", Paragraph(child, document._body)
        elif child.tag == qn("w:tbl"):
            yield "table", Table(child, document._body)


def length_inches(value: Any) -> float | None:
    """Convert a python-docx length to a concise inch value."""

    return round(value.inches, 4) if value is not None else None


def section_record(section: Any, index: int) -> dict[str, Any]:
    """Serialize page and margin settings."""

    pg_num = section._sectPr.find(qn("w:pgNumType"))
    width = length_inches(section.page_width)
    height = length_inches(section.page_height)
    usable = (
        round(
            section.page_width.inches - section.left_margin.inches - section.right_margin.inches, 4
        )
        if section.page_width and section.left_margin and section.right_margin
        else None
    )
    return {
        "index": index,
        "start_type": section.start_type.name if section.start_type is not None else None,
        "orientation": section.orientation.name if section.orientation is not None else None,
        "page_width_inches": length_inches(section.page_width),
        "page_height_inches": length_inches(section.page_height),
        "paper_size": (
            "letter"
            if width is not None
            and height is not None
            and {round(width, 2), round(height, 2)} == {8.5, 11.0}
            else "a4"
            if width is not None
            and height is not None
            and {round(width, 2), round(height, 2)} == {8.27, 11.69}
            else None
        ),
        "usable_width_inches": usable,
        "top_margin_inches": length_inches(section.top_margin),
        "right_margin_inches": length_inches(section.right_margin),
        "bottom_margin_inches": length_inches(section.bottom_margin),
        "left_margin_inches": length_inches(section.left_margin),
        "header_distance_inches": length_inches(section.header_distance),
        "footer_distance_inches": length_inches(section.footer_distance),
        "different_first_page": bool(section.different_first_page_header_footer),
        "page_number_start": (
            int(pg_num.get(qn("w:start")))
            if pg_num is not None and pg_num.get(qn("w:start"))
            else None
        ),
        "page_number_format": pg_num.get(qn("w:fmt")) if pg_num is not None else None,
    }


def story_record(story: _Header | _Footer, section_index: int, kind: str) -> dict[str, Any]:
    """Serialize one header/footer story."""

    return {
        "section_index": section_index,
        "kind": kind,
        "part": str(story.part.partname),
        "linked_to_previous": bool(story.is_linked_to_previous),
        "paragraphs": [
            paragraph_record(paragraph, index) for index, paragraph in enumerate(story.paragraphs)
        ],
        "tables": [table_record(table, index) for index, table in enumerate(story.tables)],
    }


def core_properties_record(document: DocumentObject) -> dict[str, Any]:
    """Serialize supported core metadata."""

    props = document.core_properties
    fields = (
        "title",
        "subject",
        "author",
        "keywords",
        "comments",
        "category",
        "last_modified_by",
        "revision",
        "identifier",
        "language",
        "version",
    )
    result = {field: getattr(props, field) for field in fields}
    for field in ("created", "modified", "last_printed"):
        value = getattr(props, field)
        result[field] = value.isoformat() if value is not None else None
    return result


def inspect_styles(document: DocumentObject) -> list[dict[str, Any]]:
    """Serialize paragraph/character style definitions relevant to authored content."""

    records = []
    for style in document.styles:
        if style.type not in {WD_STYLE_TYPE.PARAGRAPH, WD_STYLE_TYPE.CHARACTER}:
            continue
        font = style.font
        record = {
            "name": style.name,
            "type": "paragraph" if style.type == WD_STYLE_TYPE.PARAGRAPH else "character",
            "based_on": style.base_style.name if style.base_style is not None else None,
            "font": {
                "name": font.name,
                "size_pt": round(font.size.pt, 3) if font.size is not None else None,
                "bold": font.bold,
                "italic": font.italic,
                "underline": font.underline,
                "color": str(font.color.rgb) if font.color.rgb is not None else None,
            },
        }
        if style.type == WD_STYLE_TYPE.PARAGRAPH:
            p_pr = style.element.pPr
            outline = p_pr.find(qn("w:outlineLvl")) if p_pr is not None else None
            record["outline_level"] = int(outline.get(qn("w:val"))) if outline is not None else None
            record["paragraph"] = {
                "alignment": (
                    style.paragraph_format.alignment.name.lower()
                    if style.paragraph_format.alignment is not None
                    else None
                ),
                "space_before_pt": (
                    round(style.paragraph_format.space_before.pt, 3)
                    if style.paragraph_format.space_before is not None
                    else None
                ),
                "space_after_pt": (
                    round(style.paragraph_format.space_after.pt, 3)
                    if style.paragraph_format.space_after is not None
                    else None
                ),
                "keep_with_next": style.paragraph_format.keep_with_next,
                "keep_together": style.paragraph_format.keep_together,
            }
        records.append(record)
    return records


def inspect_fields(path: Path) -> list[dict[str, Any]]:
    """Inspect simple and complex fields with instruction, state, and visible result."""

    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as package:
        for part in sorted(
            name
            for name in package.namelist()
            if name == "word/document.xml" or re.fullmatch(r"word/(header|footer)\d+\.xml", name)
        ):
            root = parse_xml_safely(package.read(part), part)
            for index, simple in enumerate(root.findall(f".//{{{WORD_NS}}}fldSimple")):
                instruction = simple.get(qn("w:instr"), "")
                records.append(
                    {
                        "story_address": f"{part}#simple-{index}",
                        "type": instruction.strip().split(maxsplit=1)[0].upper()
                        if instruction.strip()
                        else None,
                        "instruction": instruction,
                        "dirty": simple.get(qn("w:dirty")) in {"1", "true", "on"},
                        "locked": simple.get(qn("w:fldLock")) in {"1", "true", "on"},
                        "result": "".join(simple.itertext()),
                    }
                )
            stack: list[dict[str, Any]] = []
            complex_index = 0
            for element in root.iter():
                if element.tag == qn("w:fldChar"):
                    field_type = element.get(qn("w:fldCharType"))
                    if field_type == "begin":
                        stack.append(
                            {
                                "instruction": [],
                                "result": [],
                                "separate": False,
                                "dirty": element.get(qn("w:dirty")) in {"1", "true", "on"},
                                "locked": element.get(qn("w:fldLock")) in {"1", "true", "on"},
                            }
                        )
                    elif field_type == "separate" and stack:
                        stack[-1]["separate"] = True
                    elif field_type == "end" and stack:
                        active = stack.pop()
                        instruction = "".join(active["instruction"])
                        records.append(
                            {
                                "story_address": f"{part}#complex-{complex_index}",
                                "type": instruction.strip().split(maxsplit=1)[0].upper()
                                if instruction.strip()
                                else None,
                                "instruction": instruction,
                                "dirty": active["dirty"],
                                "locked": active["locked"],
                                "result": "".join(active["result"]),
                            }
                        )
                        complex_index += 1
                elif stack and element.tag == qn("w:instrText") and element.text:
                    stack[-1]["instruction"].append(element.text)
                elif stack and element.tag == qn("w:t") and element.text:
                    for active in stack:
                        if active["separate"]:
                            active["result"].append(element.text)
    return records


def inspect_media(path: Path, document: DocumentObject) -> dict[str, Any]:
    """Inventory media members and inline occurrences across XML parts."""

    media: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as package:
        for info in package.infolist():
            if info.filename.startswith("word/media/"):
                blob = package.read(info)
                media.append(
                    {
                        "part": info.filename,
                        "bytes": len(blob),
                        "sha256": hashlib.sha256(blob).hexdigest(),
                    }
                )

    body_sections: list[int] = []
    section_index = 0
    for kind, block in iter_body_blocks(document):
        body_sections.extend(section_index for _ in block._element.iter(f"{{{WP_NS}}}inline"))
        if kind == "paragraph":
            p_pr = block._p.pPr
            if p_pr is not None and p_pr.sectPr is not None:
                section_index = min(section_index + 1, len(document.sections) - 1)

    occurrences: list[dict[str, Any]] = []
    for part in document.part.package.parts:
        element = getattr(part, "element", None)
        if element is None:
            continue
        for local_index, inline in enumerate(element.iter(f"{{{WP_NS}}}inline")):
            blip = next(inline.iter(f"{{{A_NS}}}blip"), None)
            if blip is None or blip.get(f"{{{REL_NS}}}embed") is None:
                continue
            extent = next(inline.iter(f"{{{WP_NS}}}extent"), None)
            relationship_id = blip.get(f"{{{REL_NS}}}embed")
            image_part = part.related_parts.get(relationship_id)
            pixel_width = pixel_height = None
            if image_part is not None:
                try:
                    source = Image.from_blob(image_part.blob)
                    pixel_width, pixel_height = source.px_width, source.px_height
                except Exception:
                    pass
            width_emu = int(extent.get("cx")) if extent is not None else None
            height_emu = int(extent.get("cy")) if extent is not None else None
            doc_pr = next(inline.iter(f"{{{WP_NS}}}docPr"), None)
            display_width = round(width_emu / 914400, 4) if width_emu else None
            display_height = round(height_emu / 914400, 4) if height_emu else None
            occurrences.append(
                {
                    "container_part": str(part.partname).lstrip("/"),
                    "story_address": f"{str(part.partname).lstrip('/')}#inline-{local_index}",
                    "relationship_id": relationship_id,
                    "section_index": (
                        body_sections[local_index]
                        if str(part.partname).lstrip("/") == "word/document.xml"
                        and local_index < len(body_sections)
                        else None
                    ),
                    "width_emu": width_emu,
                    "height_emu": height_emu,
                    "display_width_inches": display_width,
                    "display_height_inches": display_height,
                    "pixel_width": pixel_width,
                    "pixel_height": pixel_height,
                    "effective_ppi_x": (
                        round(pixel_width / display_width, 1)
                        if pixel_width and display_width
                        else None
                    ),
                    "effective_ppi_y": (
                        round(pixel_height / display_height, 1)
                        if pixel_height and display_height
                        else None
                    ),
                    "upscaled": bool(
                        pixel_width
                        and pixel_height
                        and display_width
                        and display_height
                        and (display_width * 96 > pixel_width or display_height * 96 > pixel_height)
                    ),
                    "alt_text": doc_pr.get("descr") if doc_pr is not None else None,
                    "title": doc_pr.get("title") if doc_pr is not None else None,
                    "decorative": (
                        doc_pr.find(
                            f"{{{A_NS}}}extLst/{{{A_NS}}}ext"
                            f"[@uri='{DECORATIVE_EXTENSION_URI}']/"
                            f"{{{A14_NS}}}decorative[@val='1']"
                        )
                        is not None
                        if doc_pr is not None
                        else False
                    ),
                }
            )
    return {"media_parts": media, "inline_occurrences": occurrences}


def inspect_fonts(path: Path) -> dict[str, Any]:
    """Inventory referenced fonts and fonts actually embedded in the DOCX package."""

    referenced_by_casefold: dict[str, str] = {}
    embedded_by_casefold: dict[str, str] = {}
    references: list[dict[str, str]] = []
    dangling_embeddings: list[dict[str, str]] = []
    run_fonts_tag = f"{{{WORD_NS}}}rFonts"
    font_attributes = {
        f"{{{WORD_NS}}}ascii",
        f"{{{WORD_NS}}}hAnsi",
        f"{{{WORD_NS}}}eastAsia",
        f"{{{WORD_NS}}}cs",
    }
    theme_attributes = {
        f"{{{WORD_NS}}}asciiTheme",
        f"{{{WORD_NS}}}hAnsiTheme",
        f"{{{WORD_NS}}}eastAsiaTheme",
        f"{{{WORD_NS}}}csTheme",
    }
    embedding_tags = {
        f"{{{WORD_NS}}}embedRegular",
        f"{{{WORD_NS}}}embedBold",
        f"{{{WORD_NS}}}embedItalic",
        f"{{{WORD_NS}}}embedBoldItalic",
    }

    with zipfile.ZipFile(path) as package:
        names = set(package.namelist())
        theme_tokens: set[str] = set()
        theme_fonts: dict[str, str] = {}
        for name in names:
            if name.startswith("word/theme/") and name.endswith(".xml"):
                root = parse_xml_safely(package.read(name), name)
                major = root.find(f".//{{{A_NS}}}majorFont")
                minor = root.find(f".//{{{A_NS}}}minorFont")
                for prefix, group in (("major", major), ("minor", minor)):
                    if group is None:
                        continue
                    for suffix, tag in (("Ascii", "latin"), ("EastAsia", "ea"), ("Bidi", "cs")):
                        element = group.find(f"{{{A_NS}}}{tag}")
                        value = (
                            (element.get("typeface") or "").strip() if element is not None else ""
                        )
                        if value:
                            theme_fonts[prefix + suffix] = value
        for name in names:
            if not name.startswith("word/") or not name.lower().endswith(".xml"):
                continue
            root = parse_xml_safely(package.read(name), name)
            if name != "word/fontTable.xml":
                for element in root.iter(run_fonts_tag):
                    for attribute in font_attributes:
                        value = (element.get(attribute) or "").strip()
                        if value and not value.startswith("+"):
                            referenced_by_casefold.setdefault(value.casefold(), value)
                            references.append(
                                {
                                    "part": name,
                                    "slot": attribute.rsplit("}", 1)[-1],
                                    "font": value,
                                    "source": "explicit",
                                }
                            )
                    for attribute in theme_attributes:
                        token = (element.get(attribute) or "").strip()
                        if token:
                            theme_tokens.add(token)
                            aliases = {
                                "majorHAnsi": "majorAscii",
                                "minorHAnsi": "minorAscii",
                            }
                            value = theme_fonts.get(aliases.get(token, token))
                            if value:
                                referenced_by_casefold.setdefault(value.casefold(), value)
                                references.append(
                                    {
                                        "part": name,
                                        "slot": attribute.rsplit("}", 1)[-1],
                                        "font": value,
                                        "source": token,
                                    }
                                )

        font_table = "word/fontTable.xml"
        if font_table in names:
            root = parse_xml_safely(package.read(font_table), font_table)
            relationships: dict[str, str] = {}
            rel_name = "word/_rels/fontTable.xml.rels"
            if rel_name in names:
                rel_root = parse_xml_safely(package.read(rel_name), rel_name)
                relationships = {
                    rel.get("Id", ""): str(PurePosixPath("word") / rel.get("Target", ""))
                    for rel in rel_root
                    if rel.get("TargetMode") != "External"
                }
            for font in root.findall(f".//{{{WORD_NS}}}font"):
                font_name = (font.get(f"{{{WORD_NS}}}name") or "").strip()
                for child in font:
                    if child.tag not in embedding_tags or not font_name:
                        continue
                    rid = child.get(f"{{{REL_NS}}}id", "")
                    target = relationships.get(rid)
                    if target in names:
                        embedded_by_casefold.setdefault(font_name.casefold(), font_name)
                    else:
                        dangling_embeddings.append(
                            {"font": font_name, "relationship_id": rid, "target": target or ""}
                        )

    referenced = list(referenced_by_casefold.values())
    embedded = list(embedded_by_casefold.values())
    return {
        "referenced": sorted(referenced, key=str.casefold),
        "embedded": sorted(embedded, key=str.casefold),
        "unembedded": sorted(
            (
                value
                for key, value in referenced_by_casefold.items()
                if key not in embedded_by_casefold
            ),
            key=str.casefold,
        ),
        "references": references,
        "theme_tokens_referenced": sorted(theme_tokens),
        "dangling_embedding_relationships": dangling_embeddings,
    }


def scan_features(path: Path) -> tuple[dict[str, int], list[str]]:
    """Count fields and fidelity-sensitive constructs in package XML."""

    counts = Counter(
        {
            "fields": 0,
            "revisions": 0,
            "comments": 0,
            "floating_drawings": 0,
            "unsupported_drawings": 0,
        }
    )
    unsupported_parts: list[str] = []
    with zipfile.ZipFile(path) as package:
        names = package.namelist()
        if "word/comments.xml" in names:
            counts["comments"] += 1
        unsupported_parts.extend(
            name
            for name in names
            if any(name.startswith(prefix) for prefix in UNSUPPORTED_PART_PREFIXES)
        )
        custom_xml_parts = {name for name in names if name.startswith("customXml/")}
        if STANDARD_CUSTOM_XML_PARTS.issubset(custom_xml_parts):
            unsupported_parts.extend(sorted(custom_xml_parts - STANDARD_CUSTOM_XML_PARTS))
        elif custom_xml_parts:
            unsupported_parts.extend(sorted(custom_xml_parts))
        for name in names:
            if not name.lower().endswith((".xml", ".rels")):
                continue
            root = parse_xml_safely(package.read(name), name)
            counts["fields"] += len(root.findall(f".//{{{WORD_NS}}}fldSimple"))
            counts["fields"] += len(root.findall(f".//{{{WORD_NS}}}instrText"))
            counts["revisions"] += sum(len(root.findall(f".//{tag}")) for tag in REVISION_TAGS)
            counts["comments"] += len(root.findall(f".//{{{WORD_NS}}}commentRangeStart"))
            counts["floating_drawings"] += len(root.findall(f".//{{{WP_NS}}}anchor"))
            counts["unsupported_drawings"] += len(root.findall(f".//{{{WORD_NS}}}pict"))
            counts["unsupported_drawings"] += len(root.findall(f".//{{{WORD_NS}}}object"))
            for inline in root.findall(f".//{{{WP_NS}}}inline"):
                if inline.find(f".//{{{A_NS}}}blip") is None:
                    counts["unsupported_drawings"] += 1
    return dict(counts), sorted(set(unsupported_parts))


def inspect_document(
    path_value: str | Path,
    *,
    allow_external_relationships: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return a complete structured inspection and warnings."""

    preflight = preflight_docx(
        path_value,
        allow_external_relationships=allow_external_relationships,
    )
    document = load_document(preflight.path)
    ordered: list[dict[str, Any]] = []
    paragraphs: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    section_index = 0
    for block_index, (kind, block) in enumerate(iter_body_blocks(document)):
        if kind == "paragraph":
            record = paragraph_record(block, len(paragraphs))
            paragraphs.append(record)
            ordered.append({"block_index": block_index, "type": kind, "value": record})
            p_pr = block._p.pPr
            if p_pr is not None and p_pr.sectPr is not None:
                section_index = min(section_index + 1, len(document.sections) - 1)
        else:
            record = table_record(block, len(tables), section_index)
            tables.append(record)
            ordered.append({"block_index": block_index, "type": kind, "value": record})

    headers: list[dict[str, Any]] = []
    footers: list[dict[str, Any]] = []
    for index, section in enumerate(document.sections):
        headers.extend(
            (
                story_record(section.header, index, "default"),
                story_record(section.first_page_header, index, "first_page"),
                story_record(section.even_page_header, index, "even_page"),
            )
        )
        footers.extend(
            (
                story_record(section.footer, index, "default"),
                story_record(section.first_page_footer, index, "first_page"),
                story_record(section.even_page_footer, index, "even_page"),
            )
        )

    feature_counts, unsupported_parts = scan_features(preflight.path)
    fonts = inspect_fonts(preflight.path)
    images = inspect_media(preflight.path, document)
    fields = inspect_fields(preflight.path)
    warnings = list(preflight.warnings)
    warning_specs = (
        ("fields", "fields_present", "Fields require a layout application to update."),
        (
            "revisions",
            "revisions_present",
            "Tracked revisions are reported but not interpreted or safely edited.",
        ),
        (
            "comments",
            "comments_present",
            "Comments are reported but complete comment workflows are unsupported.",
        ),
        (
            "floating_drawings",
            "floating_drawings_present",
            "Floating drawings are preserved best-effort but cannot be replaced.",
        ),
        (
            "unsupported_drawings",
            "unsupported_drawings_present",
            "Unsupported drawing or embedded-object markup is present.",
        ),
    )
    for key, code, message in warning_specs:
        if feature_counts[key]:
            warnings.append(warning(code, message, count=feature_counts[key]))
    if unsupported_parts:
        warnings.append(
            warning(
                "unsupported_parts",
                "Package contains parts outside the supported editing model.",
                parts=unsupported_parts,
            )
        )
    if fonts["unembedded"]:
        warnings.append(
            warning(
                "unembedded_fonts",
                (
                    "DOCX references fonts that are not embedded in the package. PDF font "
                    "embedding does not make the DOCX portable; another Word renderer may "
                    "substitute fonts and change wrapping, page breaks, object flow, and TOC "
                    "page references. Use conservative cross-renderer fonts or embed licensed "
                    "fonts, and do not claim DOCX/PDF pagination fidelity from a PDF-only check."
                ),
                fonts=fonts["unembedded"],
            )
        )
        common_fonts = [font for font in fonts["unembedded"] if font.casefold() in COMMON_FONTS]
        nonportable_fonts = [
            font
            for font in fonts["unembedded"]
            if font.casefold() not in PORTABLE_FONTS and font.casefold() not in COMMON_FONTS
        ]
        if common_fonts:
            warnings.append(
                warning(
                    "common_unembedded_fonts",
                    (
                        "DOCX references unembedded fonts that ship with Microsoft Office or "
                        "one major operating system. Systems without them substitute fonts "
                        "(LibreOffice uses metric-compatible Carlito and Caladea for Calibri "
                        "and Cambria), so wrapping and pagination may shift slightly. This is "
                        "acceptable for most business documents; embed the fonts or switch to "
                        "the portable core set for print-critical deliverables."
                    ),
                    fonts=common_fonts,
                )
            )
        if nonportable_fonts:
            warnings.append(
                warning(
                    "nonportable_unembedded_fonts",
                    (
                        "RELEASE BLOCKER: DOCX references unembedded fonts outside the "
                        "portable core set and the common Office-bundled set. Renderer "
                        "substitution can change wrapping and pagination; replace or embed "
                        "these fonts before releasing matching DOCX and PDF deliverables."
                    ),
                    fonts=nonportable_fonts,
                )
            )
    if fonts["dangling_embedding_relationships"]:
        warnings.append(
            warning(
                "dangling_font_embedding_relationships",
                "Font embedding markup has missing or invalid relationship targets.",
                relationships=fonts["dangling_embedding_relationships"],
            )
        )
    metadata = core_properties_record(document)
    if not metadata.get("title"):
        warnings.append(warning("missing_document_title", "Document metadata has no title."))
    if not metadata.get("language"):
        warnings.append(warning("missing_document_language", "Document metadata has no language."))
    previous_heading = 0
    for paragraph in paragraphs:
        level = paragraph["heading_level"]
        if level and previous_heading and level > previous_heading + 1:
            warnings.append(
                warning(
                    "heading_level_skip",
                    "Heading hierarchy skips a level.",
                    paragraph_index=paragraph["index"],
                    previous_level=previous_heading,
                    heading_level=level,
                )
            )
        if level:
            previous_heading = level
    for table in tables:
        if table["row_count"] > 1 and table["header_rows"] == 0:
            warnings.append(
                warning(
                    "data_table_without_header_row",
                    "Table has no repeated header row.",
                    table_index=table["index"],
                )
            )
        widths = [value or 0 for value in table["column_widths_inches"]]
        table_section = table["section_index"] if table["section_index"] is not None else 0
        usable = section_record(document.sections[table_section], table_section)[
            "usable_width_inches"
        ]
        if usable and sum(widths) > usable + 0.01:
            warnings.append(
                warning(
                    "table_exceeds_printable_width",
                    "Table width exceeds the section usable width.",
                    table_index=table["index"],
                    table_width_inches=round(sum(widths), 4),
                    usable_width_inches=usable,
                )
            )
    section_usable = [
        section_record(section, index)["usable_width_inches"]
        for index, section in enumerate(document.sections)
    ]
    for image in images["inline_occurrences"]:
        if not image["decorative"] and not image["alt_text"]:
            warnings.append(
                warning(
                    "informative_image_missing_alt_text",
                    "Informative image has no alternative text.",
                    story_address=image["story_address"],
                )
            )
        ppi_values = [
            value for value in (image["effective_ppi_x"], image["effective_ppi_y"]) if value
        ]
        if ppi_values and min(ppi_values) < 150:
            warnings.append(
                warning(
                    "low_resolution_image",
                    "Effective image resolution is below 150 PPI.",
                    story_address=image["story_address"],
                    effective_ppi=min(ppi_values),
                )
            )
        if image["container_part"] == "word/document.xml" and image["display_width_inches"]:
            image_section = image["section_index"]
            if image_section is None or image_section >= len(section_usable):
                warnings.append(
                    warning(
                        "image_printable_width_uncertain",
                        (
                            "Image could not be mapped to a body section; "
                            "printable-width fit is unknown."
                        ),
                        story_address=image["story_address"],
                    )
                )
            elif (
                section_usable[image_section]
                and image["display_width_inches"] > section_usable[image_section]
            ):
                warnings.append(
                    warning(
                        "image_exceeds_printable_width",
                        "Image display width exceeds section usable width.",
                        story_address=image["story_address"],
                        section_index=image_section,
                        usable_width_inches=section_usable[image_section],
                    )
                )
    for field in fields:
        if field["type"] == "TOC" and (
            not field["result"].strip() or "right-click and update" in field["result"].casefold()
        ):
            warnings.append(
                warning(
                    "toc_placeholder_only",
                    (
                        "TOC visible result is only a placeholder and needs a "
                        "layout-application update."
                    ),
                    story_address=field["story_address"],
                )
            )
    update_fields = document.settings.element.find(qn("w:updateFields"))

    result = {
        "ordered_blocks": ordered,
        "paragraphs": paragraphs,
        "tables": tables,
        "sections": [
            section_record(section, index) for index, section in enumerate(document.sections)
        ],
        "headers": headers,
        "footers": footers,
        "metadata": metadata,
        "styles": inspect_styles(document),
        "images": images,
        "fonts": fonts,
        "fields": fields,
        "settings": {
            "odd_and_even_pages": bool(document.settings.odd_and_even_pages_header_footer),
            "update_fields_on_open": (
                update_fields is not None
                and update_fields.get(qn("w:val"), "true") not in {"0", "false", "off"}
            ),
        },
        "features": feature_counts,
        "unsupported_parts": unsupported_parts,
        "package": {
            "member_count": preflight.member_count,
            "expanded_bytes": preflight.expanded_bytes,
        },
    }
    return result, warnings


def ensure_output_target(
    output: Path,
    *,
    expected_suffix: str,
    overwrite: bool,
    source: Path | None = None,
) -> None:
    """Enforce explicit overwrite and a usable destination directory."""

    if output.suffix.lower() != expected_suffix:
        raise ToolError(
            "bad_input",
            f"Output must use the {expected_suffix} extension.",
            details={"path": str(output)},
        )
    if not output.parent.is_dir():
        raise ToolError(
            "bad_input",
            "Output parent directory does not exist.",
            details={"path": str(output.parent)},
        )
    same_as_source = source is not None and paths_same(output, source)
    if same_as_source and not overwrite:
        raise ToolError(
            "output_conflict",
            "Source and destination must differ unless --overwrite is explicit.",
            details={"path": str(output)},
        )
    if output.exists() and not overwrite:
        raise ToolError(
            "output_conflict",
            "Destination exists; pass --overwrite only with explicit authorization.",
            details={"path": str(output)},
        )


def paths_same(left: Path, right: Path) -> bool:
    """Compare paths without requiring both to exist."""

    try:
        return left.resolve() == right.resolve()
    except OSError:
        return os.path.abspath(left) == os.path.abspath(right)


def publish_temp_file(temp_path: Path, output: Path, *, overwrite: bool) -> None:
    """Publish atomically and preserve no-clobber semantics under races."""

    if overwrite:
        os.replace(temp_path, output)
        return
    try:
        os.link(temp_path, output)
    except FileExistsError as exc:
        raise ToolError(
            "output_conflict",
            "Destination appeared before publication; output was not replaced.",
            details={"path": str(output)},
        ) from exc
    temp_path.unlink()


def atomic_docx_save(
    document: DocumentObject,
    output: Path,
    *,
    overwrite: bool,
    source: Path | None,
    allow_external_relationships: bool,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Save, preflight, reopen, and atomically publish a DOCX."""

    ensure_output_target(output, expected_suffix=".docx", overwrite=overwrite, source=source)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output.name}.",
            suffix=".docx",
            dir=output.parent,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
        document.save(str(temp_path))
        with temp_path.open("rb") as handle:
            os.fsync(handle.fileno())
        inspection, warnings = inspect_document(
            temp_path,
            allow_external_relationships=allow_external_relationships,
        )
        if len(inspection["sections"]) < 1:
            raise ToolError(
                "validation_failed",
                "Post-write reopen found no document section.",
            )
        publish_temp_file(temp_path, output, overwrite=overwrite)
        temp_path = None
        return (
            {
                "preflight": True,
                "reopened": True,
                "structural_inspection": True,
                "atomic_publish": True,
                **inspection["package"],
            },
            inspection,
            warnings,
        )
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(
            "validation_failed",
            "DOCX save or post-write validation failed.",
            details={"path": str(output), "reason": str(exc)},
        ) from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def apply_run_format(run: Run, spec: dict[str, Any]) -> None:
    """Apply supported run formatting."""

    if "bold" in spec:
        run.bold = optional_boolean(spec["bold"], "run.bold")
    if "italic" in spec:
        run.italic = optional_boolean(spec["italic"], "run.italic")
    if "underline" in spec:
        run.underline = optional_boolean(spec["underline"], "run.underline")
    if "font" in spec:
        run.font.name = required_string(spec["font"], "run.font")
    if "size_pt" in spec:
        run.font.size = Pt(required_number(spec["size_pt"], "run.size_pt", minimum=1))
    if "color" in spec:
        color = required_string(spec["color"], "run.color").lstrip("#")
        if not re.fullmatch(r"[0-9A-Fa-f]{6}", color):
            raise ToolError("bad_input", "run.color must be a six-digit RGB hex value.")
        run.font.color.rgb = RGBColor.from_string(color.upper())
    if "style" in spec:
        assign_named_style(run, required_string(spec["style"], "run.style"), "run.style")


def assign_named_style(target: Any, style: str, field: str) -> None:
    """Assign a named style and classify missing styles as bad input."""

    try:
        target.style = style
    except (KeyError, ValueError) as exc:
        raise ToolError(
            "bad_input",
            f"{field} does not exist in the document or template.",
            details={"style": style},
        ) from exc


def add_runs(paragraph: Paragraph, block: dict[str, Any]) -> None:
    """Add either plain text or explicitly formatted runs to a paragraph."""

    if "text" in block and "runs" in block:
        raise ToolError("bad_input", "Use text or runs, not both.")
    if "runs" in block:
        runs = block["runs"]
        if not isinstance(runs, list):
            raise ToolError("bad_input", "runs must be an array.")
        for run_spec in runs:
            if not isinstance(run_spec, dict):
                raise ToolError("bad_input", "Each run must be an object.")
            run = paragraph.add_run(string_value(run_spec.get("text"), "run.text"))
            apply_run_format(run, run_spec)
    elif "text" in block:
        paragraph.add_run(string_value(block["text"], "text"))


def apply_paragraph_format(paragraph: Paragraph, block: dict[str, Any]) -> None:
    """Apply supported paragraph-level formatting."""

    alignment = block.get("alignment")
    if alignment is not None:
        alignments = {
            "left": WD_ALIGN_PARAGRAPH.LEFT,
            "center": WD_ALIGN_PARAGRAPH.CENTER,
            "right": WD_ALIGN_PARAGRAPH.RIGHT,
            "justify": WD_ALIGN_PARAGRAPH.JUSTIFY,
        }
        if alignment not in alignments:
            raise ToolError("bad_input", "Unsupported paragraph alignment.")
        paragraph.alignment = alignments[alignment]
    paragraph_format = paragraph.paragraph_format
    for key, attr in (
        ("space_before_pt", "space_before"),
        ("space_after_pt", "space_after"),
    ):
        if key in block:
            setattr(paragraph_format, attr, Pt(required_number(block[key], key, minimum=0)))
    for key, attr in (
        ("left_indent_inches", "left_indent"),
        ("right_indent_inches", "right_indent"),
    ):
        if key in block:
            setattr(paragraph_format, attr, Inches(required_number(block[key], key, minimum=0)))
    if "first_line_indent_inches" in block and "hanging_indent_inches" in block:
        raise ToolError("bad_input", "first_line_indent_inches and hanging_indent_inches conflict.")
    if "first_line_indent_inches" in block:
        paragraph_format.first_line_indent = Inches(
            required_number(block["first_line_indent_inches"], "first_line_indent_inches")
        )
    if "hanging_indent_inches" in block:
        paragraph_format.first_line_indent = Inches(
            -required_number(block["hanging_indent_inches"], "hanging_indent_inches", minimum=0)
        )
    if "line_spacing" in block:
        value = block["line_spacing"]
        if isinstance(value, str):
            rules = {
                "single": WD_LINE_SPACING.SINGLE,
                "one_and_half": WD_LINE_SPACING.ONE_POINT_FIVE,
                "double": WD_LINE_SPACING.DOUBLE,
            }
            if value not in rules:
                raise ToolError("bad_input", "line_spacing string is unsupported.")
            paragraph_format.line_spacing_rule = rules[value]
        else:
            paragraph_format.line_spacing = required_number(value, "line_spacing", minimum=0.1)
    for key in ("keep_with_next", "keep_together", "page_break_before", "widow_control"):
        if key in block:
            setattr(paragraph_format, key, optional_boolean(block[key], key))
    if "tab_stops" in block:
        paragraph_format.tab_stops.clear_all()
        alignments = {
            "left": WD_TAB_ALIGNMENT.LEFT,
            "center": WD_TAB_ALIGNMENT.CENTER,
            "right": WD_TAB_ALIGNMENT.RIGHT,
            "decimal": WD_TAB_ALIGNMENT.DECIMAL,
            "bar": WD_TAB_ALIGNMENT.BAR,
        }
        leaders = {
            "spaces": WD_TAB_LEADER.SPACES,
            "dots": WD_TAB_LEADER.DOTS,
            "dashes": WD_TAB_LEADER.DASHES,
            "lines": WD_TAB_LEADER.LINES,
            "heavy": WD_TAB_LEADER.HEAVY,
            "middle_dot": WD_TAB_LEADER.MIDDLE_DOT,
        }
        for tab in block["tab_stops"]:
            paragraph_format.tab_stops.add_tab_stop(
                Inches(tab["position_inches"]),
                alignments[tab.get("alignment", "left")],
                leaders[tab.get("leader", "spaces")],
            )


def required_string(value: Any, field: str) -> str:
    """Validate a non-empty string."""

    if not isinstance(value, str) or not value:
        raise ToolError("bad_input", f"{field} must be a non-empty string.")
    return value


def required_integer(
    value: Any,
    field: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Validate an integer but not a boolean."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolError("bad_input", f"{field} must be an integer.")
    if minimum is not None and value < minimum:
        raise ToolError("bad_input", f"{field} must be at least {minimum}.")
    if maximum is not None and value > maximum:
        raise ToolError("bad_input", f"{field} must be at most {maximum}.")
    return value


def required_number(value: Any, field: str, *, minimum: float | None = None) -> float:
    """Validate a finite numeric value."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolError("bad_input", f"{field} must be numeric.")
    number = float(value)
    if not number < float("inf") or not number > float("-inf"):
        raise ToolError("bad_input", f"{field} must be finite.")
    if minimum is not None and number < minimum:
        raise ToolError("bad_input", f"{field} must be at least {minimum}.")
    return number


def optional_boolean(value: Any, field: str, *, default: bool = False) -> bool:
    """Accept only a JSON Boolean for an optional Boolean field."""

    if value is None:
        return default
    if not isinstance(value, bool):
        raise ToolError("bad_input", f"{field} must be a Boolean.")
    return value


def string_value(value: Any, field: str) -> str:
    """Validate a JSON string, allowing an intentional empty value."""

    if not isinstance(value, str):
        raise ToolError("bad_input", f"{field} must be a string.")
    return value


def require_keys(spec: dict[str, Any], required: set[str], field: str) -> None:
    """Reject missing required object properties."""

    missing = sorted(required.difference(spec))
    if missing:
        raise ToolError(
            "bad_input",
            f"{field} is missing required properties.",
            details={"properties": missing},
        )


def allow_keys(spec: dict[str, Any], allowed: set[str], field: str) -> None:
    """Reject unknown operational object properties."""

    unknown = sorted(set(spec).difference(allowed))
    if unknown:
        raise ToolError(
            "bad_input",
            f"{field} contains unknown properties.",
            details={"properties": unknown},
        )


def validate_runs(runs: Any, field: str) -> None:
    """Validate a formatted-runs array without coercing JSON values."""

    if not isinstance(runs, list):
        raise ToolError("bad_input", f"{field} must be an array.")
    for index, run in enumerate(runs):
        item_field = f"{field}[{index}]"
        if not isinstance(run, dict):
            raise ToolError("bad_input", f"{item_field} must be an object.")
        allow_keys(
            run,
            {"text", "bold", "italic", "underline", "font", "size_pt", "color", "style"},
            item_field,
        )
        require_keys(run, {"text"}, item_field)
        string_value(run["text"], f"{item_field}.text")
        for name in ("bold", "italic", "underline"):
            if name in run:
                optional_boolean(run[name], f"{item_field}.{name}")
        for name in ("font", "color", "style"):
            if name in run:
                required_string(run[name], f"{item_field}.{name}")
        if "size_pt" in run:
            required_number(run["size_pt"], f"{item_field}.size_pt", minimum=1)


def validate_text_and_runs(spec: dict[str, Any], field: str) -> None:
    """Validate paragraph text representation and common formatting."""

    if "text" in spec and "runs" in spec:
        raise ToolError(
            "bad_input",
            f"{field} must use text or runs, not both.",
        )
    if "text" in spec:
        string_value(spec["text"], f"{field}.text")
    if "runs" in spec:
        validate_runs(spec["runs"], f"{field}.runs")
    if "style" in spec:
        required_string(spec["style"], f"{field}.style")
    if "alignment" in spec:
        alignment = string_value(spec["alignment"], f"{field}.alignment")
        if alignment not in {"left", "center", "right", "justify"}:
            raise ToolError("bad_input", f"{field}.alignment is unsupported.")
    for name in (
        "space_before_pt",
        "space_after_pt",
        "left_indent_inches",
        "right_indent_inches",
        "hanging_indent_inches",
    ):
        if name in spec:
            required_number(spec[name], f"{field}.{name}", minimum=0)
    if "first_line_indent_inches" in spec:
        required_number(spec["first_line_indent_inches"], f"{field}.first_line_indent_inches")
    if "first_line_indent_inches" in spec and "hanging_indent_inches" in spec:
        raise ToolError("bad_input", f"{field} cannot combine first-line and hanging indents.")
    if "line_spacing" in spec:
        value = spec["line_spacing"]
        if isinstance(value, str):
            if value not in {"single", "one_and_half", "double"}:
                raise ToolError("bad_input", f"{field}.line_spacing is unsupported.")
        else:
            required_number(value, f"{field}.line_spacing", minimum=0.1)
    for name in ("keep_with_next", "keep_together", "page_break_before", "widow_control"):
        if name in spec:
            optional_boolean(spec[name], f"{field}.{name}")
    if "tab_stops" in spec:
        tabs = spec["tab_stops"]
        if not isinstance(tabs, list):
            raise ToolError("bad_input", f"{field}.tab_stops must be an array.")
        for index, tab in enumerate(tabs):
            tab_field = f"{field}.tab_stops[{index}]"
            if not isinstance(tab, dict):
                raise ToolError("bad_input", f"{tab_field} must be an object.")
            allow_keys(tab, {"position_inches", "alignment", "leader"}, tab_field)
            require_keys(tab, {"position_inches"}, tab_field)
            required_number(tab["position_inches"], f"{tab_field}.position_inches", minimum=0)
            if tab.get("alignment", "left") not in {"left", "center", "right", "decimal", "bar"}:
                raise ToolError("bad_input", f"{tab_field}.alignment is unsupported.")
            if tab.get("leader", "spaces") not in {
                "spaces",
                "dots",
                "dashes",
                "lines",
                "heavy",
                "middle_dot",
            }:
                raise ToolError("bad_input", f"{tab_field}.leader is unsupported.")


def validate_cell_value(value: Any, field: str) -> None:
    """Validate a table cell as plain text or a paragraph-format object."""

    if isinstance(value, str):
        return
    if not isinstance(value, dict):
        raise ToolError(
            "bad_input",
            f"{field} must be a string or paragraph object.",
        )
    allow_keys(value, {"text", "runs", "style", "alignment"} | PARAGRAPH_LAYOUT_KEYS, field)
    validate_text_and_runs(value, field)


def validate_table_block(block: dict[str, Any], field: str) -> None:
    """Require a rectangular table with non-empty rows and typed cells."""

    require_keys(block, {"rows"}, field)
    rows = block["rows"]
    if not isinstance(rows, list) or not rows:
        raise ToolError("bad_input", f"{field}.rows must be a non-empty array.")
    width: int | None = None
    for row_index, row in enumerate(rows):
        row_field = f"{field}.rows[{row_index}]"
        if not isinstance(row, list) or not row:
            raise ToolError("bad_input", f"{row_field} must be a non-empty array.")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ToolError(
                "bad_input",
                f"{field}.rows must be rectangular.",
                details={
                    "row": row_index,
                    "expected_columns": width,
                    "actual_columns": len(row),
                },
            )
        for column_index, value in enumerate(row):
            validate_cell_value(value, f"{row_field}[{column_index}]")
    if "style" in block:
        required_string(block["style"], f"{field}.style")
    if "header_rows" in block:
        required_integer(
            block["header_rows"],
            f"{field}.header_rows",
            minimum=0,
            maximum=len(rows),
        )
    if "layout" in block:
        layout = string_value(block["layout"], f"{field}.layout")
        if layout not in {"autofit", "fixed"}:
            raise ToolError(
                "bad_input",
                f"{field}.layout must be 'autofit' or 'fixed'.",
            )
    if "column_widths_inches" in block:
        widths = block["column_widths_inches"]
        if not isinstance(widths, list):
            raise ToolError(
                "bad_input",
                f"{field}.column_widths_inches must be an array.",
            )
        if width is None or len(widths) != width:
            raise ToolError(
                "bad_input",
                f"{field}.column_widths_inches must contain one width per column.",
                details={
                    "expected_columns": width,
                    "actual_widths": len(widths),
                },
            )
        for column_index, value in enumerate(widths):
            required_number(
                value,
                f"{field}.column_widths_inches[{column_index}]",
                minimum=0.1,
            )
    if "allow_row_split" in block:
        optional_boolean(
            block["allow_row_split"],
            f"{field}.allow_row_split",
        )
    if "row_allow_split" in block:
        values = block["row_allow_split"]
        if not isinstance(values, list) or len(values) != len(rows):
            raise ToolError(
                "bad_input", f"{field}.row_allow_split must contain one Boolean per row."
            )
        for index, value in enumerate(values):
            optional_boolean(value, f"{field}.row_allow_split[{index}]")


def validate_block(block: Any, field: str, *, allow_sections: bool) -> None:
    """Validate one content block against its type-specific schema."""

    if not isinstance(block, dict):
        raise ToolError("bad_input", f"{field} must be an object.")
    block_type = block.get("type", "paragraph")
    if not isinstance(block_type, str):
        raise ToolError("bad_input", f"{field}.type must be a string.")
    paragraph_keys = {"type", "text", "runs", "style", "alignment"} | PARAGRAPH_LAYOUT_KEYS
    if block_type == "paragraph":
        allow_keys(block, paragraph_keys, field)
        validate_text_and_runs(block, field)
        return
    if block_type == "heading":
        allow_keys(block, paragraph_keys | {"level"}, field)
        validate_text_and_runs(block, field)
        if "level" in block:
            required_integer(block["level"], f"{field}.level", minimum=0, maximum=9)
        return
    if block_type == "page_break":
        allow_keys(block, {"type"}, field)
        return
    if block_type == "image":
        allow_keys(
            block,
            {
                "type",
                "path",
                "width_inches",
                "height_inches",
                "style",
                "alignment",
                "prefix_runs",
                "suffix_runs",
                "alt_text",
                "decorative",
                "title",
                "caption",
                "caption_style",
                "attribution",
            },
            field,
        )
        require_keys(block, {"path"}, field)
        string_value(block["path"], f"{field}.path")
        for name in ("width_inches", "height_inches"):
            if name in block:
                required_number(block[name], f"{field}.{name}", minimum=0.01)
        if "style" in block:
            required_string(block["style"], f"{field}.style")
        if "alignment" in block:
            validate_text_and_runs({"alignment": block["alignment"]}, field)
        for name in ("prefix_runs", "suffix_runs"):
            if name in block:
                validate_runs(block[name], f"{field}.{name}")
        for name in ("alt_text", "title", "caption", "caption_style", "attribution"):
            if name in block:
                string_value(block[name], f"{field}.{name}")
        decorative = optional_boolean(block.get("decorative"), f"{field}.decorative")
        if decorative and block.get("alt_text"):
            raise ToolError("bad_input", f"{field} cannot combine decorative and alt_text.")
        return
    if block_type == "field":
        require_keys(block, {"field"}, field)
        field_name = required_string(block["field"], f"{field}.field")
        if field_name not in {
            "page_number",
            "toc",
            "num_pages",
            "section_pages",
            "seq",
            "ref",
            "date",
        }:
            raise ToolError("bad_input", f"{field}.field is unsupported.")
        allowed_field_keys = {
            "type",
            "field",
            "prefix",
            "suffix",
            "placeholder",
            "style",
            "alignment",
        }
        if field_name in {"toc", "seq", "ref", "date"}:
            allowed_field_keys.add("instruction")
        allowed_field_keys |= {"locked"}
        allow_keys(
            block,
            allowed_field_keys,
            field,
        )
        for name in ("prefix", "suffix", "placeholder", "instruction"):
            if name in block:
                string_value(block[name], f"{field}.{name}")
        if "locked" in block:
            optional_boolean(block["locked"], f"{field}.locked")
        if "style" in block:
            required_string(block["style"], f"{field}.style")
        if "alignment" in block:
            validate_text_and_runs({"alignment": block["alignment"]}, field)
        return
    if block_type == "list":
        allow_keys(block, {"type", "items", "ordered"}, field)
        require_keys(block, {"items"}, field)
        items = block["items"]
        if not isinstance(items, list):
            raise ToolError("bad_input", f"{field}.items must be an array.")
        if "ordered" in block:
            optional_boolean(block["ordered"], f"{field}.ordered")
        for index, item in enumerate(items):
            item_field = f"{field}.items[{index}]"
            if isinstance(item, str):
                continue
            if not isinstance(item, dict):
                raise ToolError(
                    "bad_input",
                    f"{item_field} must be a string or paragraph object.",
                )
            allow_keys(item, {"text", "runs", "alignment"}, item_field)
            validate_text_and_runs(item, item_field)
        return
    if block_type == "table":
        allow_keys(block, TABLE_KEYS, field)
        validate_table_block(block, field)
        return
    if block_type == "section_break":
        if not allow_sections:
            raise ToolError(
                "bad_input",
                f"{field} cannot contain a section break.",
            )
        section_keys = {"type", "start"} | SECTION_KEYS
        allow_keys(block, section_keys, field)
        if "start" in block:
            start = string_value(block["start"], f"{field}.start")
            if start not in {
                "new_page",
                "continuous",
                "even_page",
                "odd_page",
                "new_column",
            }:
                raise ToolError("bad_input", f"{field}.start is unsupported.")
        if "orientation" in block:
            orientation = string_value(block["orientation"], f"{field}.orientation")
            if orientation not in {"portrait", "landscape"}:
                raise ToolError("bad_input", f"{field}.orientation is unsupported.")
        validate_section_spec(block, field, extra_keys={"type", "start"})
        return
    raise ToolError("bad_input", f"{field}.type is unsupported.", details={"type": block_type})


def validate_blocks(blocks: Any, field: str, *, allow_sections: bool) -> None:
    """Validate a content-block array."""

    if not isinstance(blocks, list):
        raise ToolError("bad_input", f"{field} must be an array.")
    for index, block in enumerate(blocks):
        validate_block(block, f"{field}[{index}]", allow_sections=allow_sections)


def validate_story_configuration(raw: Any, field: str) -> None:
    """Validate compact or section-explicit header/footer configuration."""

    if isinstance(raw, dict):
        allow_keys(raw, {"default", "first_page", "even_page"}, field)
        for kind, blocks in raw.items():
            validate_blocks(blocks, f"{field}.{kind}", allow_sections=False)
        return
    if isinstance(raw, list):
        for index, entry in enumerate(raw):
            item_field = f"{field}[{index}]"
            if not isinstance(entry, dict):
                raise ToolError("bad_input", f"{item_field} must be an object.")
            allow_keys(entry, {"section_index", "kind", "blocks"}, item_field)
            require_keys(entry, {"section_index", "kind", "blocks"}, item_field)
            required_integer(entry["section_index"], f"{item_field}.section_index", minimum=0)
            kind = required_string(entry["kind"], f"{item_field}.kind")
            if kind not in {"default", "first_page", "even_page"}:
                raise ToolError("bad_input", f"{item_field}.kind is unsupported.")
            validate_blocks(entry["blocks"], f"{item_field}.blocks", allow_sections=False)
        return
    raise ToolError("bad_input", f"{field} must be an object or array.")


def validate_metadata(metadata: Any, field: str) -> None:
    """Validate supported metadata names and value types."""

    if not isinstance(metadata, dict):
        raise ToolError("bad_input", f"{field} must be an object.")
    supported = {
        "title",
        "subject",
        "author",
        "keywords",
        "comments",
        "category",
        "last_modified_by",
        "revision",
        "identifier",
        "language",
        "version",
    }
    allow_keys(metadata, supported, field)
    for key, value in metadata.items():
        if key == "revision":
            required_integer(value, f"{field}.revision", minimum=1)
        else:
            string_value(value, f"{field}.{key}")


def validate_section_spec(spec: Any, field: str, *, extra_keys: set[str] | None = None) -> None:
    """Validate page geometry and numbering controls."""

    if not isinstance(spec, dict):
        raise ToolError("bad_input", f"{field} must be an object.")
    allow_keys(spec, SECTION_KEYS | (extra_keys or set()), field)
    if "paper_size" in spec and spec["paper_size"] not in {"letter", "a4"}:
        raise ToolError("bad_input", f"{field}.paper_size must be letter or a4.")
    for name in SECTION_KEYS - {
        "paper_size",
        "orientation",
        "different_first_page",
        "page_number_start",
        "page_number_format",
    }:
        if name in spec:
            minimum = 0.1 if name in {"page_width_inches", "page_height_inches"} else 0
            required_number(spec[name], f"{field}.{name}", minimum=minimum)
    if spec.get("orientation") not in {None, "portrait", "landscape"}:
        raise ToolError("bad_input", f"{field}.orientation is unsupported.")
    if "different_first_page" in spec:
        optional_boolean(spec["different_first_page"], f"{field}.different_first_page")
    if "page_number_start" in spec:
        required_integer(spec["page_number_start"], f"{field}.page_number_start", minimum=0)
    formats = {
        "decimal",
        "upperRoman",
        "lowerRoman",
        "upperLetter",
        "lowerLetter",
    }
    if "page_number_format" in spec and spec["page_number_format"] not in formats:
        raise ToolError("bad_input", f"{field}.page_number_format is unsupported.")


def validate_style_definitions(raw: Any, field: str) -> None:
    """Validate paragraph and character style definitions."""

    if not isinstance(raw, list):
        raise ToolError("bad_input", f"{field} must be an array.")
    for index, spec in enumerate(raw):
        item = f"{field}[{index}]"
        if not isinstance(spec, dict):
            raise ToolError("bad_input", f"{item} must be an object.")
        allow_keys(
            spec,
            {
                "name",
                "type",
                "based_on",
                "font",
                "size_pt",
                "bold",
                "italic",
                "underline",
                "color",
                "paragraph",
                "outline_level",
            },
            item,
        )
        require_keys(spec, {"name", "type"}, item)
        required_string(spec["name"], f"{item}.name")
        if spec["type"] not in {"paragraph", "character"}:
            raise ToolError("bad_input", f"{item}.type is unsupported.")
        if "based_on" in spec:
            required_string(spec["based_on"], f"{item}.based_on")
        validate_runs(
            [
                {
                    "text": "",
                    **{
                        key: spec[key]
                        for key in ("font", "size_pt", "bold", "italic", "underline", "color")
                        if key in spec
                    },
                }
            ],
            f"{item}.font",
        )
        if "paragraph" in spec:
            if spec["type"] != "paragraph" or not isinstance(spec["paragraph"], dict):
                raise ToolError("bad_input", f"{item}.paragraph requires a paragraph style.")
            allow_keys(
                spec["paragraph"], PARAGRAPH_LAYOUT_KEYS | {"alignment"}, f"{item}.paragraph"
            )
            validate_text_and_runs(spec["paragraph"], f"{item}.paragraph")
        if "outline_level" in spec:
            if spec["type"] != "paragraph":
                raise ToolError("bad_input", f"{item}.outline_level requires a paragraph style.")
            required_integer(spec["outline_level"], f"{item}.outline_level", minimum=0, maximum=9)


def validate_create_content(content: Any, field: str = "content") -> None:
    """Validate the complete create-content object."""

    if not isinstance(content, dict):
        raise ToolError("bad_input", f"{field} must be an object.")
    allow_keys(
        content,
        {
            "template",
            "letterhead",
            "metadata",
            "section",
            "styles",
            "update_fields_on_open",
            "blocks",
            "headers",
            "footers",
        },
        field,
    )
    if "template" in content and "letterhead" in content:
        raise ToolError("bad_input", f"{field} must specify only one of template or letterhead.")
    for name in ("template", "letterhead"):
        if name in content:
            required_string(content[name], f"{field}.{name}")
    if "metadata" in content:
        validate_metadata(content["metadata"], f"{field}.metadata")
    if "section" in content:
        validate_section_spec(content["section"], f"{field}.section")
    if "styles" in content:
        validate_style_definitions(content["styles"], f"{field}.styles")
    if "update_fields_on_open" in content:
        optional_boolean(content["update_fields_on_open"], f"{field}.update_fields_on_open")
    if "blocks" in content:
        validate_blocks(content["blocks"], f"{field}.blocks", allow_sections=True)
    for name in ("headers", "footers"):
        if name in content:
            validate_story_configuration(content[name], f"{field}.{name}")


def validate_edit_operation(operation: Any, field: str) -> None:
    """Validate one edit operation and all nested operational keys."""

    if not isinstance(operation, dict):
        raise ToolError("bad_input", f"{field} must be an object.")
    require_keys(operation, {"type"}, field)
    operation_type = required_string(operation["type"], f"{field}.type")
    if operation_type == "replace_text":
        allow_keys(
            operation,
            {
                "type",
                "find",
                "replace",
                "expected_count",
                "replace_all",
                "scope",
                "cross_run_policy",
            },
            field,
        )
        require_keys(operation, {"find", "replace", "expected_count"}, field)
        required_string(operation["find"], f"{field}.find")
        string_value(operation["replace"], f"{field}.replace")
        required_integer(operation["expected_count"], f"{field}.expected_count", minimum=0)
        if "replace_all" in operation:
            optional_boolean(operation["replace_all"], f"{field}.replace_all")
        if "scope" in operation:
            scope = string_value(operation["scope"], f"{field}.scope")
            if scope not in {"body", "headers", "footers", "all"}:
                raise ToolError("bad_input", f"{field}.scope is unsupported.")
        if "cross_run_policy" in operation:
            policy = string_value(operation["cross_run_policy"], f"{field}.cross_run_policy")
            if policy != "first_run":
                raise ToolError("bad_input", f"{field}.cross_run_policy is unsupported.")
        return
    if operation_type == "insert_paragraph":
        allow_keys(operation, {"type", "block", "paragraph_index", "position"}, field)
        require_keys(operation, {"block", "position"}, field)
        validate_block(operation["block"], f"{field}.block", allow_sections=False)
        if operation["block"].get("type", "paragraph") not in {
            "paragraph",
            "heading",
            "page_break",
            "image",
            "field",
        }:
            raise ToolError("bad_input", f"{field}.block must be paragraph-like.")
        position = string_value(operation["position"], f"{field}.position")
        if position == "append":
            if "paragraph_index" in operation:
                raise ToolError(
                    "bad_input",
                    f"{field}.paragraph_index must be omitted when position is append.",
                )
        elif position in {"before", "after"}:
            require_keys(operation, {"paragraph_index"}, field)
            required_integer(operation["paragraph_index"], f"{field}.paragraph_index", minimum=0)
        else:
            raise ToolError("bad_input", f"{field}.position is unsupported.")
        return
    if operation_type == "delete_paragraph":
        allow_keys(operation, {"type", "paragraph_index"}, field)
        require_keys(operation, {"paragraph_index"}, field)
        required_integer(operation["paragraph_index"], f"{field}.paragraph_index", minimum=0)
        return
    if operation_type == "insert_table":
        allow_keys(operation, {"type", "table", "paragraph_index", "position"}, field)
        require_keys(operation, {"table"}, field)
        table = operation["table"]
        if not isinstance(table, dict):
            raise ToolError("bad_input", f"{field}.table must be an object.")
        allow_keys(table, TABLE_KEYS, f"{field}.table")
        if table.get("type", "table") != "table":
            raise ToolError("bad_input", f"{field}.table.type must be table.")
        validate_table_block(table, f"{field}.table")
        if "paragraph_index" in operation:
            required_integer(operation["paragraph_index"], f"{field}.paragraph_index", minimum=0)
            if "position" in operation:
                position = string_value(operation["position"], f"{field}.position")
                if position not in {"before", "after"}:
                    raise ToolError("bad_input", f"{field}.position is unsupported.")
        elif "position" in operation:
            raise ToolError("bad_input", f"{field}.position requires paragraph_index.")
        return
    if operation_type == "update_table":
        allow_keys(
            operation,
            {
                "type",
                "table_index",
                "row",
                "column",
                "value",
                "style",
                "header_rows",
                "layout",
                "column_widths_inches",
                "allow_row_split",
                "row_allow_split",
            },
            field,
        )
        require_keys(operation, {"table_index"}, field)
        cell_keys = {"row", "column", "value"}
        if cell_keys.intersection(operation) and not cell_keys.issubset(operation):
            raise ToolError("bad_input", f"{field} must supply row, column, and value together.")
        for name in ("table_index", "row", "column"):
            if name not in operation:
                continue
            required_integer(operation[name], f"{field}.{name}", minimum=0)
        if "value" in operation:
            validate_cell_value(operation["value"], f"{field}.value")
        if "style" in operation:
            required_string(operation["style"], f"{field}.style")
        if "header_rows" in operation:
            required_integer(operation["header_rows"], f"{field}.header_rows", minimum=0)
        if operation.get("layout") not in {None, "autofit", "fixed"}:
            raise ToolError("bad_input", f"{field}.layout is unsupported.")
        if "column_widths_inches" in operation:
            widths = operation["column_widths_inches"]
            if not isinstance(widths, list) or not widths:
                raise ToolError("bad_input", f"{field}.column_widths_inches must be an array.")
            for index, width in enumerate(widths):
                required_number(width, f"{field}.column_widths_inches[{index}]", minimum=0.1)
        if "allow_row_split" in operation:
            optional_boolean(operation["allow_row_split"], f"{field}.allow_row_split")
        if "row_allow_split" in operation:
            values = operation["row_allow_split"]
            if not isinstance(values, list):
                raise ToolError("bad_input", f"{field}.row_allow_split must be an array.")
            for index, value in enumerate(values):
                optional_boolean(value, f"{field}.row_allow_split[{index}]")
        return
    if operation_type in {"insert_image", "replace_image"}:
        index_name = "paragraph_index" if operation_type == "insert_image" else "inline_image_index"
        allowed = {
            "type",
            "path",
            "width_inches",
            "height_inches",
            index_name,
            "story_address",
            "alt_text",
            "decorative",
            "title",
        }
        allow_keys(operation, allowed, field)
        required = {"path"} if operation_type == "replace_image" else {"path"}
        require_keys(operation, required, field)
        if operation_type == "replace_image" and not {
            "inline_image_index",
            "story_address",
        }.intersection(operation):
            raise ToolError("bad_input", f"{field} requires inline_image_index or story_address.")
        required_string(operation["path"], f"{field}.path")
        if index_name in operation:
            required_integer(operation[index_name], f"{field}.{index_name}", minimum=0)
        for name in ("width_inches", "height_inches"):
            if name in operation:
                required_number(operation[name], f"{field}.{name}", minimum=0.01)
        for name in ("alt_text", "title", "story_address"):
            if name in operation:
                string_value(operation[name], f"{field}.{name}")
        if "decorative" in operation:
            optional_boolean(operation["decorative"], f"{field}.decorative")
        if operation.get("decorative") and operation.get("alt_text"):
            raise ToolError("bad_input", f"{field} cannot combine decorative and alt_text.")
        return
    if operation_type == "set_metadata":
        allow_keys(operation, {"type", "metadata"}, field)
        require_keys(operation, {"metadata"}, field)
        validate_metadata(operation["metadata"], f"{field}.metadata")
        return
    if operation_type in {"set_header", "set_footer"}:
        allow_keys(
            operation,
            {"type", "section_index", "kind", "mode", "blocks", "link_to_previous"},
            field,
        )
        if "link_to_previous" not in operation:
            require_keys(operation, {"blocks"}, field)
        if "section_index" in operation:
            required_integer(operation["section_index"], f"{field}.section_index", minimum=0)
        if "kind" in operation:
            kind = required_string(operation["kind"], f"{field}.kind")
            if kind not in {"default", "first_page", "even_page"}:
                raise ToolError("bad_input", f"{field}.kind is unsupported.")
        if "mode" in operation:
            mode = required_string(operation["mode"], f"{field}.mode")
            if mode not in {"replace", "append"}:
                raise ToolError("bad_input", f"{field}.mode is unsupported.")
        if "link_to_previous" in operation:
            optional_boolean(operation["link_to_previous"], f"{field}.link_to_previous")
            if operation["link_to_previous"] and "blocks" in operation:
                raise ToolError(
                    "bad_input",
                    f"{field} cannot write blocks while link_to_previous is true.",
                )
        if "blocks" in operation:
            validate_blocks(operation["blocks"], f"{field}.blocks", allow_sections=False)
        return
    if operation_type == "configure_section":
        allow_keys(operation, {"type", "section_index"} | SECTION_KEYS, field)
        if "section_index" in operation:
            required_integer(operation["section_index"], f"{field}.section_index", minimum=0)
        validate_section_spec(operation, field, extra_keys={"type", "section_index"})
        return
    if operation_type == "upsert_style":
        allow_keys(operation, {"type", "style"}, field)
        require_keys(operation, {"style"}, field)
        validate_style_definitions([operation["style"]], f"{field}.style")
        return
    if operation_type == "set_update_fields_on_open":
        allow_keys(operation, {"type", "enabled"}, field)
        require_keys(operation, {"enabled"}, field)
        optional_boolean(operation["enabled"], f"{field}.enabled")
        return
    raise ToolError(
        "bad_input",
        f"{field}.type is unsupported.",
        details={"type": operation_type},
    )


def validate_edit_operations(operations: Any, field: str = "operations") -> None:
    """Validate a non-empty edit-operation array."""

    if not isinstance(operations, list) or not operations:
        raise ToolError("bad_input", f"{field} must be a non-empty array.")
    for index, operation in enumerate(operations):
        validate_edit_operation(operation, f"{field}[{index}]")


def validate_direct_create_spec(spec: dict[str, Any]) -> None:
    """Validate the direct create JSON envelope."""

    allow_keys(spec, {"schema_version", "content"}, "create specification")
    require_keys(spec, {"content"}, "create specification")
    validate_create_content(spec["content"])


def validate_direct_edit_spec(spec: dict[str, Any]) -> None:
    """Validate the direct edit JSON envelope."""

    allow_keys(spec, {"schema_version", "operations"}, "edit specification")
    require_keys(spec, {"operations"}, "edit specification")
    validate_edit_operations(spec["operations"])


def validate_job(job: dict[str, Any]) -> str:
    """Validate a complete job envelope and return its operation."""

    operation = required_string(job.get("operation"), "operation")
    common = {"schema_version", "operation", "allow_external_relationships"}
    if "allow_external_relationships" in job:
        optional_boolean(job["allow_external_relationships"], "allow_external_relationships")
    if operation == "inspect":
        allow_keys(job, common | {"input"}, "inspect job")
        require_keys(job, {"input"}, "inspect job")
    elif operation == "create":
        allow_keys(job, common | {"output", "overwrite", "content"}, "create job")
        require_keys(job, {"output", "content"}, "create job")
        if "overwrite" in job:
            optional_boolean(job["overwrite"], "overwrite")
        validate_create_content(job["content"])
    elif operation == "edit":
        allow_keys(
            job,
            common | {"input", "output", "overwrite", "operations"},
            "edit job",
        )
        require_keys(job, {"input", "output", "operations"}, "edit job")
        if "overwrite" in job:
            optional_boolean(job["overwrite"], "overwrite")
        validate_edit_operations(job["operations"])
    elif operation == "convert":
        allow_keys(
            job,
            common | {"input", "output", "format", "timeout", "overwrite"},
            "convert job",
        )
        require_keys(job, {"input", "output", "format"}, "convert job")
        format_name = required_string(job["format"], "format")
        if format_name not in {"text", "pdf"}:
            raise ToolError("bad_input", "format must be text or pdf.")
        if "timeout" in job:
            required_integer(job["timeout"], "timeout", minimum=1, maximum=3_600)
        if "overwrite" in job:
            optional_boolean(job["overwrite"], "overwrite")
    elif operation == "render":
        allow_keys(
            job,
            common | {"input", "output", "dpi", "pages", "timeout"},
            "render job",
        )
        require_keys(job, {"input", "output"}, "render job")
        if "dpi" in job:
            required_integer(job["dpi"], "dpi", minimum=MIN_RENDER_DPI, maximum=MAX_RENDER_DPI)
        if "pages" in job:
            required_string(job["pages"], "pages")
        if "timeout" in job:
            required_integer(job["timeout"], "timeout", minimum=1, maximum=3_600)
    else:
        raise ToolError(
            "bad_input",
            "operation is unsupported.",
            details={"operation": operation},
        )
    for name in ("input", "output"):
        if name in job:
            required_string(job[name], name)
    return operation


def validate_image(path: Path) -> None:
    """Bound image input and reject unknown signatures."""

    if not path.is_file():
        raise ToolError("bad_input", "Image input does not exist.", details={"path": str(path)})
    size = path.stat().st_size
    if size > MAX_IMAGE_BYTES:
        raise ToolError(
            "resource_limit",
            "Image exceeds the size limit.",
            details={"path": str(path), "bytes": size, "limit": MAX_IMAGE_BYTES},
        )
    signature = path.read_bytes()[:16]
    supported = (
        signature.startswith(b"\x89PNG\r\n\x1a\n")
        or signature.startswith(b"\xff\xd8\xff")
        or signature.startswith((b"GIF87a", b"GIF89a"))
        or signature.startswith((b"II*\x00", b"MM\x00*"))
        or signature[:2] == b"BM"
    )
    if not supported:
        raise ToolError(
            "bad_input",
            "Image signature is not a supported PNG, JPEG, GIF, TIFF, or BMP.",
            details={"path": str(path)},
        )


def image_dimensions(spec: dict[str, Any]) -> tuple[Any | None, Any | None]:
    """Read optional dimensions in inches."""

    width = (
        Inches(required_number(spec["width_inches"], "width_inches", minimum=0.01))
        if "width_inches" in spec
        else None
    )
    height = (
        Inches(required_number(spec["height_inches"], "height_inches", minimum=0.01))
        if "height_inches" in spec
        else None
    )
    return width, height


def append_field(paragraph: Paragraph, field_name: str, spec: dict[str, Any]) -> None:
    """Append bounded common complex-field markup."""

    if field_name == "page_number":
        instruction = " PAGE "
        placeholder = string_value(spec.get("placeholder", "1"), "field.placeholder")
    elif field_name == "toc":
        instruction = string_value(
            spec.get("instruction", ' TOC \\o "1-3" \\h \\z \\u '),
            "field.instruction",
        )
        if not instruction.strip().upper().startswith("TOC"):
            raise ToolError("bad_input", "TOC instruction must begin with TOC.")
        placeholder = string_value(
            spec.get("placeholder", "Right-click and update field in a layout application."),
            "field.placeholder",
        )
    else:
        defaults = {
            "num_pages": (" NUMPAGES ", "1"),
            "section_pages": (" SECTIONPAGES ", "1"),
            "seq": (" SEQ Figure \\* ARABIC ", "1"),
            "ref": (" REF Bookmark ", "Error! Reference source not found."),
            "date": (' DATE \\@ "MMMM d, yyyy" ', "Date"),
        }
        if field_name not in defaults:
            raise ToolError("unsupported_operation", f"Unsupported field type: {field_name}")
        instruction, default_placeholder = defaults[field_name]
        if "instruction" in spec:
            instruction = string_value(spec["instruction"], "field.instruction")
        expected = {
            "seq": "SEQ",
            "ref": "REF",
            "date": "DATE",
        }.get(field_name)
        if expected and not instruction.strip().upper().startswith(expected):
            raise ToolError("bad_input", f"{field_name} instruction must begin with {expected}.")
        if len(instruction) > 512 or any(character in instruction for character in "\r\n\x00"):
            raise ToolError("bad_input", "field.instruction is unsafe or too long.")
        placeholder = string_value(
            spec.get("placeholder", default_placeholder), "field.placeholder"
        )

    if len(instruction) > 512 or any(character in instruction for character in "\r\n\x00"):
        raise ToolError("bad_input", "field.instruction is unsafe or too long.")
    begin_run = OxmlElement("w:r")
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    begin.set(qn("w:dirty"), "true")
    if optional_boolean(spec.get("locked"), "field.locked"):
        begin.set(qn("w:fldLock"), "true")
    begin_run.append(begin)

    instruction_run = OxmlElement("w:r")
    instruction_text = OxmlElement("w:instrText")
    instruction_text.set(qn("xml:space"), "preserve")
    instruction_text.text = instruction
    instruction_run.append(instruction_text)

    separate_run = OxmlElement("w:r")
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    separate_run.append(separate)

    result_run = OxmlElement("w:r")
    result_text = OxmlElement("w:t")
    result_text.text = placeholder
    result_run.append(result_text)

    end_run = OxmlElement("w:r")
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    end_run.append(end)

    for element in (begin_run, instruction_run, separate_run, result_run, end_run):
        paragraph._p.append(element)


def add_paragraph_block(
    container: Any,
    block: dict[str, Any],
    base_dir: Path,
) -> Paragraph:
    """Add one paragraph-like block and return it."""

    block_type = block.get("type", "paragraph")
    style = block.get("style")
    if style is not None:
        style = required_string(style, "block.style")
    if block_type == "heading":
        level = required_integer(block.get("level", 1), "heading.level", minimum=0, maximum=9)
        default_style = f"Heading {level}" if level else "Title"
        paragraph = container.add_paragraph()
        assign_named_style(paragraph, style or default_style, "block.style")
        add_runs(paragraph, block)
    elif block_type == "paragraph":
        paragraph = container.add_paragraph()
        if style is not None:
            assign_named_style(paragraph, style, "block.style")
        add_runs(paragraph, block)
    elif block_type == "page_break":
        paragraph = container.add_paragraph()
        paragraph.add_run().add_break(WD_BREAK.PAGE)
    elif block_type == "image":
        paragraph = container.add_paragraph()
        if style is not None:
            assign_named_style(paragraph, style, "block.style")
        add_runs(paragraph, {"runs": block.get("prefix_runs", [])})
        image = resolve_path(block.get("path"), base_dir, "image.path")
        validate_image(image)
        width, height = image_dimensions(block)
        shape = paragraph.add_run().add_picture(str(image), width=width, height=height)
        set_drawing_accessibility(shape._inline, block)
        add_runs(paragraph, {"runs": block.get("suffix_runs", [])})
    elif block_type == "field":
        paragraph = container.add_paragraph()
        if style is not None:
            assign_named_style(paragraph, style, "block.style")
        if "prefix" in block:
            paragraph.add_run(string_value(block["prefix"], "field.prefix"))
        append_field(paragraph, required_string(block.get("field"), "field.field"), block)
        if "suffix" in block:
            paragraph.add_run(string_value(block["suffix"], "field.suffix"))
    else:
        raise ToolError(
            "unsupported_operation",
            f"Block type is not paragraph-like: {block_type}",
        )
    apply_paragraph_format(paragraph, block)
    if block_type == "image" and ("caption" in block or "attribution" in block):
        caption_text = string_value(block.get("caption", ""), "image.caption")
        attribution = string_value(block.get("attribution", ""), "image.attribution")
        text = caption_text + (f" — {attribution}" if attribution else "")
        caption = container.add_paragraph(text)
        assign_named_style(caption, block.get("caption_style", "Caption"), "image.caption_style")
        paragraph.paragraph_format.keep_with_next = True
        paragraph.paragraph_format.keep_together = True
        caption.paragraph_format.keep_together = True
    return paragraph


def set_drawing_accessibility(inline: Any, spec: dict[str, Any]) -> None:
    """Write title, description, and decorative drawing metadata."""

    doc_pr = inline.find(qn("wp:docPr"))
    if doc_pr is None:
        return
    for name in ("descr", "title"):
        doc_pr.attrib.pop(name, None)
    if spec.get("alt_text"):
        doc_pr.set("descr", string_value(spec["alt_text"], "image.alt_text"))
    if spec.get("title"):
        doc_pr.set("title", string_value(spec["title"], "image.title"))
    ext_list = doc_pr.find(f"{{{A_NS}}}extLst")
    if ext_list is not None:
        for extension in list(ext_list):
            if extension.get("uri") == DECORATIVE_EXTENSION_URI:
                ext_list.remove(extension)
        if not len(ext_list):
            doc_pr.remove(ext_list)
    if optional_boolean(spec.get("decorative"), "image.decorative"):
        ext_list = doc_pr.find(f"{{{A_NS}}}extLst")
        if ext_list is None:
            ext_list = etree.Element(f"{{{A_NS}}}extLst")
            doc_pr.append(ext_list)
        extension = etree.SubElement(ext_list, f"{{{A_NS}}}ext")
        extension.set("uri", DECORATIVE_EXTENSION_URI)
        decorative = etree.Element(f"{{{A14_NS}}}decorative", nsmap={"a14": A14_NS})
        decorative.set("val", "1")
        extension.append(decorative)


def apply_cell_widths(table: Table, widths: list[Any]) -> None:
    """Set preferred widths, summing covered grid columns for merged cells."""

    for column, width in zip(table.columns, widths, strict=True):
        column.width = width
    for row in table.rows:
        grid_column = 0
        for tc in row._tr.tc_lst:
            grid_span = tc.tcPr.find(qn("w:gridSpan"))
            span = int(grid_span.get(qn("w:val"), "1")) if grid_span is not None else 1
            _Cell(tc, table).width = sum(widths[grid_column : grid_column + span])
            grid_column += span


def fill_table(table: Table, block: dict[str, Any]) -> None:
    """Fill a newly-created table from row/cell values."""

    rows = block.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ToolError("bad_input", "table.rows must be a non-empty array.")
    for row_index, row_spec in enumerate(rows):
        if not isinstance(row_spec, list):
            raise ToolError("bad_input", "Each table row must be an array.")
        for column_index, cell_spec in enumerate(row_spec):
            cell = table.cell(row_index, column_index)
            set_cell_value(cell, cell_spec)
    if "style" in block:
        assign_named_style(
            table,
            required_string(block["style"], "table.style"),
            "table.style",
        )
    layout = block.get(
        "layout",
        "fixed" if "column_widths_inches" in block else "autofit",
    )
    table.autofit = layout == "autofit"
    if "column_widths_inches" in block:
        widths = [
            Inches(required_number(value, "column_width", minimum=0.1))
            for value in block["column_widths_inches"]
        ]
        apply_cell_widths(table, widths)
    header_rows = required_integer(block.get("header_rows", 0), "header_rows", minimum=0)
    for row_index, row in enumerate(table.rows):
        tr_pr = row._tr.get_or_add_trPr()
        for existing in list(tr_pr.findall(qn("w:tblHeader"))):
            tr_pr.remove(existing)
        if row_index < header_rows:
            repeat = OxmlElement("w:tblHeader")
            repeat.set(qn("w:val"), "true")
            tr_pr.append(repeat)
        for existing in list(tr_pr.findall(qn("w:cantSplit"))):
            tr_pr.remove(existing)
        row_split = (
            block["row_allow_split"][row_index]
            if "row_allow_split" in block
            else block.get("allow_row_split", True)
        )
        if not optional_boolean(row_split, "allow_row_split", default=True):
            cant_split = OxmlElement("w:cantSplit")
            cant_split.set(qn("w:val"), "true")
            tr_pr.append(cant_split)


def set_cell_value(cell: _Cell, value: Any) -> None:
    """Set a cell to plain text or a paragraph/run object."""

    if isinstance(value, dict):
        paragraph = cell.paragraphs[0]
        paragraph.clear()
        if "style" in value:
            assign_named_style(
                paragraph,
                required_string(value["style"], "cell.style"),
                "cell.style",
            )
        add_runs(paragraph, value)
        apply_paragraph_format(paragraph, value)
    else:
        cell.text = string_value(value, "cell")


def add_table_block(container: Any, block: dict[str, Any]) -> Table:
    """Add and fill one table."""

    rows = block.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ToolError("bad_input", "table.rows must be a non-empty array.")
    columns = max((len(row) if isinstance(row, list) else 0 for row in rows), default=0)
    if columns < 1:
        raise ToolError("bad_input", "A table must have at least one column.")
    if isinstance(container, (_Header, _Footer)):
        document = container._document_part.document
        usable = next(
            (
                section.page_width - section.left_margin - section.right_margin
                for section in document.sections
                if (
                    section.header.part is container.part
                    or section.first_page_header.part is container.part
                    or section.even_page_header.part is container.part
                    or section.footer.part is container.part
                    or section.first_page_footer.part is container.part
                    or section.even_page_footer.part is container.part
                )
            ),
            document.sections[0].page_width
            - document.sections[0].left_margin
            - document.sections[0].right_margin,
        )
        table = container.add_table(rows=len(rows), cols=columns, width=usable)
    else:
        table = container.add_table(rows=len(rows), cols=columns)
    fill_table(table, block)
    return table


def configure_section(section: Any, spec: dict[str, Any]) -> None:
    """Apply supported section page settings."""

    requested_orientation = spec.get("orientation")
    if requested_orientation not in {None, "portrait", "landscape"}:
        raise ToolError("bad_input", "section.orientation must be portrait or landscape.")
    current_landscape = section.orientation == WD_ORIENT.LANDSCAPE or (
        section.page_width is not None
        and section.page_height is not None
        and section.page_width > section.page_height
    )
    final_landscape = (
        requested_orientation == "landscape"
        if requested_orientation is not None
        else current_landscape
    )
    if "paper_size" in spec:
        sizes = {"letter": (8.5, 11.0), "a4": (8.2677, 11.6929)}
        width, height = sizes[required_string(spec["paper_size"], "section.paper_size")]
        if final_landscape:
            width, height = height, width
        section.page_width, section.page_height = Inches(width), Inches(height)
    if "page_width_inches" in spec:
        section.page_width = Inches(required_number(spec["page_width_inches"], "page_width_inches"))
    if "page_height_inches" in spec:
        section.page_height = Inches(
            required_number(spec["page_height_inches"], "page_height_inches")
        )
    if requested_orientation is not None:
        desired = WD_ORIENT.LANDSCAPE if final_landscape else WD_ORIENT.PORTRAIT
        section.orientation = desired
        dimensions_are_landscape = (
            section.page_width is not None
            and section.page_height is not None
            and section.page_width > section.page_height
        )
        if dimensions_are_landscape != final_landscape:
            section.page_width, section.page_height = section.page_height, section.page_width
    margin_fields = {
        "top_margin_inches": "top_margin",
        "right_margin_inches": "right_margin",
        "bottom_margin_inches": "bottom_margin",
        "left_margin_inches": "left_margin",
        "header_distance_inches": "header_distance",
        "footer_distance_inches": "footer_distance",
    }
    for input_name, property_name in margin_fields.items():
        if input_name in spec:
            setattr(
                section,
                property_name,
                Inches(required_number(spec[input_name], input_name, minimum=0)),
            )
    if (
        section.page_width is not None
        and section.left_margin is not None
        and section.right_margin is not None
        and section.page_width <= section.left_margin + section.right_margin
    ):
        raise ToolError(
            "bad_input",
            "Section left and right margins must leave positive printable width.",
        )
    if (
        section.page_height is not None
        and section.top_margin is not None
        and section.bottom_margin is not None
        and section.page_height <= section.top_margin + section.bottom_margin
    ):
        raise ToolError(
            "bad_input",
            "Section top and bottom margins must leave positive printable height.",
        )
    if "different_first_page" in spec:
        section.different_first_page_header_footer = optional_boolean(
            spec["different_first_page"], "different_first_page"
        )
    if "page_number_start" in spec or "page_number_format" in spec:
        pg_num = section._sectPr.find(qn("w:pgNumType"))
        if pg_num is None:
            pg_num = OxmlElement("w:pgNumType")
            later_children = {
                qn(f"w:{name}")
                for name in (
                    "cols",
                    "formProt",
                    "vAlign",
                    "noEndnote",
                    "titlePg",
                    "textDirection",
                    "bidi",
                    "rtlGutter",
                    "docGrid",
                    "printerSettings",
                    "sectPrChange",
                )
            }
            insertion_index = next(
                (
                    index
                    for index, child in enumerate(section._sectPr)
                    if child.tag in later_children
                ),
                len(section._sectPr),
            )
            section._sectPr.insert(insertion_index, pg_num)
        if "page_number_start" in spec:
            pg_num.set(qn("w:start"), str(spec["page_number_start"]))
        if "page_number_format" in spec:
            pg_num.set(qn("w:fmt"), spec["page_number_format"])


def apply_blocks(
    container: Any,
    blocks: Any,
    base_dir: Path,
    *,
    allow_sections: bool,
) -> Counter[str]:
    """Apply supported content blocks to a document or story."""

    if not isinstance(blocks, list):
        raise ToolError("bad_input", "blocks must be an array.")
    counts: Counter[str] = Counter()
    for block in blocks:
        if not isinstance(block, dict):
            raise ToolError("bad_input", "Each block must be an object.")
        block_type = block.get("type", "paragraph")
        if block_type in {"paragraph", "heading", "page_break", "image", "field"}:
            add_paragraph_block(container, block, base_dir)
        elif block_type == "list":
            items = block.get("items")
            if not isinstance(items, list):
                raise ToolError("bad_input", "list.items must be an array.")
            ordered = optional_boolean(block.get("ordered"), "list.ordered")
            list_style = "List Number" if ordered else "List Bullet"
            for item in items:
                if not isinstance(item, (dict, str)):
                    raise ToolError(
                        "bad_input",
                        "Each list item must be a string or paragraph object.",
                    )
                item_spec = item if isinstance(item, dict) else {"text": item}
                item_spec = {**item_spec, "type": "paragraph", "style": list_style}
                add_paragraph_block(container, item_spec, base_dir)
            counts["list_items"] += len(items)
        elif block_type == "table":
            add_table_block(container, block)
        elif block_type == "section_break":
            if not allow_sections or not isinstance(container, DocumentObject):
                raise ToolError(
                    "unsupported_operation",
                    "Section breaks are supported only in the main document body.",
                )
            starts = {
                "new_page": WD_SECTION.NEW_PAGE,
                "continuous": WD_SECTION.CONTINUOUS,
                "even_page": WD_SECTION.EVEN_PAGE,
                "odd_page": WD_SECTION.ODD_PAGE,
                "new_column": WD_SECTION.NEW_COLUMN,
            }
            start_name = block.get("start", "new_page")
            if start_name not in starts:
                raise ToolError("bad_input", "Unsupported section break start type.")
            section = container.add_section(starts[start_name])
            configure_section(section, block)
        else:
            raise ToolError("unsupported_operation", f"Unsupported block type: {block_type}")
        counts[str(block_type)] += 1
    return counts


def clear_story(story: _Header | _Footer) -> None:
    """Remove story paragraphs and tables before adding replacement content."""

    element = story._element
    for child in list(element):
        element.remove(child)


def story_for(section: Any, kind: str, header: bool) -> _Header | _Footer:
    """Select one of the three header/footer story types."""

    if kind == "default":
        return section.header if header else section.footer
    if kind == "first_page":
        section.different_first_page_header_footer = True
        return section.first_page_header if header else section.first_page_footer
    if kind == "even_page":
        section._document_part.document.settings.odd_and_even_pages_header_footer = True
        return section.even_page_header if header else section.even_page_footer
    raise ToolError("bad_input", f"Unsupported story kind: {kind}")


def apply_story_configuration(
    document: DocumentObject,
    raw: Any,
    base_dir: Path,
    *,
    header: bool,
) -> int:
    """Apply compact or section-explicit header/footer configuration."""

    if raw is None:
        return 0
    entries: list[dict[str, Any]]
    if isinstance(raw, dict):
        entries = [
            {"section_index": 0, "kind": kind, "blocks": blocks} for kind, blocks in raw.items()
        ]
    elif isinstance(raw, list):
        entries = raw
    else:
        raise ToolError("bad_input", "headers/footers must be an object or array.")
    applied = 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise ToolError("bad_input", "Each header/footer entry must be an object.")
        section_index = required_integer(entry.get("section_index", 0), "section_index", minimum=0)
        if section_index >= len(document.sections):
            raise ToolError("bad_input", "Header/footer section index is out of range.")
        kind = required_string(entry.get("kind", "default"), "story.kind")
        story = story_for(document.sections[section_index], kind, header)
        story.is_linked_to_previous = False
        clear_story(story)
        apply_blocks(story, entry.get("blocks", []), base_dir, allow_sections=False)
        if not story.paragraphs:
            story.add_paragraph()
        applied += 1
    return applied


def set_metadata(document: DocumentObject, metadata: Any) -> int:
    """Set supported core properties."""

    if metadata is None:
        return 0
    if not isinstance(metadata, dict):
        raise ToolError("bad_input", "metadata must be an object.")
    supported = {
        "title",
        "subject",
        "author",
        "keywords",
        "comments",
        "category",
        "last_modified_by",
        "revision",
        "identifier",
        "language",
        "version",
    }
    unknown = sorted(set(metadata).difference(supported))
    if unknown:
        raise ToolError(
            "bad_input",
            "Unknown metadata properties.",
            details={"properties": unknown},
        )
    props = document.core_properties
    for key, value in metadata.items():
        if key == "revision":
            value = required_integer(value, "metadata.revision", minimum=1)
        else:
            value = string_value(value, f"metadata.{key}")
        setattr(props, key, value)
    return len(metadata)


def set_update_fields_on_open(document: DocumentObject, enabled: bool) -> None:
    """Set Word's update-fields-on-open document setting without duplicates."""

    settings = document.settings.element
    for existing in list(settings.findall(qn("w:updateFields"))):
        settings.remove(existing)
    element = OxmlElement("w:updateFields")
    element.set(qn("w:val"), "true" if enabled else "false")
    settings.append(element)


def upsert_style(document: DocumentObject, spec: dict[str, Any]) -> None:
    """Create or update a paragraph/character named style."""

    name = required_string(spec["name"], "style.name")
    style_type = WD_STYLE_TYPE.PARAGRAPH if spec["type"] == "paragraph" else WD_STYLE_TYPE.CHARACTER
    try:
        style = document.styles[name]
        if style.type != style_type:
            raise ToolError(
                "bad_input", "Existing style has a different type.", details={"style": name}
            )
    except KeyError:
        style = document.styles.add_style(name, style_type)
    if "based_on" in spec:
        try:
            style.base_style = document.styles[spec["based_on"]]
        except KeyError as exc:
            raise ToolError("bad_input", "style.based_on does not exist.") from exc
    font_spec = {
        key: spec[key]
        for key in ("font", "size_pt", "bold", "italic", "underline", "color")
        if key in spec
    }
    apply_run_format(type("_StyleRun", (), {"font": style.font, "style": None})(), font_spec)
    if spec["type"] == "paragraph" and "paragraph" in spec:
        paragraph_spec = spec["paragraph"]
        pseudo = type(
            "_StyleParagraph",
            (),
            {"paragraph_format": style.paragraph_format, "alignment": None},
        )()
        apply_paragraph_format(pseudo, paragraph_spec)
        if "alignment" in paragraph_spec:
            style.paragraph_format.alignment = {
                "left": WD_ALIGN_PARAGRAPH.LEFT,
                "center": WD_ALIGN_PARAGRAPH.CENTER,
                "right": WD_ALIGN_PARAGRAPH.RIGHT,
                "justify": WD_ALIGN_PARAGRAPH.JUSTIFY,
            }[paragraph_spec["alignment"]]
    if "outline_level" in spec:
        p_pr = style.element.get_or_add_pPr()
        for existing in list(p_pr.findall(qn("w:outlineLvl"))):
            p_pr.remove(existing)
        outline = OxmlElement("w:outlineLvl")
        outline.set(qn("w:val"), str(spec["outline_level"]))
        p_pr.append(outline)


def create_document(
    spec: dict[str, Any],
    base_dir: Path,
    output: Path,
    *,
    overwrite: bool,
    allow_external_relationships: bool,
    template_override: Path | None = None,
) -> dict[str, Any]:
    """Create a DOCX from a versioned specification."""

    content = spec.get("content")
    if not isinstance(content, dict):
        raise ToolError("bad_input", "create.content must be an object.")
    template_value = template_override
    if template_value is None:
        template_raw = content.get("template")
        letterhead_raw = content.get("letterhead")
        if template_raw is not None and letterhead_raw is not None:
            raise ToolError("bad_input", "Specify only one of template or letterhead.")
        template_value = optional_path(
            template_raw if template_raw is not None else letterhead_raw,
            base_dir,
            "template",
        )
    source = None
    if template_value is not None:
        preflight_docx(
            template_value,
            allow_external_relationships=allow_external_relationships,
        )
        document = load_document(template_value)
        source = template_value
    else:
        document = Document()

    if "section" in content:
        configure_section(document.sections[0], content["section"])
    for style in content.get("styles", []):
        upsert_style(document, style)
    if "update_fields_on_open" in content:
        set_update_fields_on_open(document, content["update_fields_on_open"])
    counts = apply_blocks(document, content.get("blocks", []), base_dir, allow_sections=True)
    counts["styles"] += len(content.get("styles", []))
    counts["metadata"] += set_metadata(document, content.get("metadata"))
    counts["headers"] += apply_story_configuration(
        document, content.get("headers"), base_dir, header=True
    )
    counts["footers"] += apply_story_configuration(
        document, content.get("footers"), base_dir, header=False
    )
    verification, inspection, warnings = atomic_docx_save(
        document,
        output,
        overwrite=overwrite,
        source=source,
        allow_external_relationships=allow_external_relationships,
    )
    return success_summary(
        "create",
        paths={"template": str(source) if source else None, "output": str(output)},
        counts=dict(counts),
        warnings=warnings,
        verification=verification,
        result={
            "paragraphs": len(inspection["paragraphs"]),
            "tables": len(inspection["tables"]),
            "inline_images": len(inspection["images"]["inline_occurrences"]),
            "sections": len(inspection["sections"]),
        },
    )


def body_paragraphs_including_tables(document: DocumentObject) -> list[Paragraph]:
    """Return body and table-cell paragraphs in reading order."""

    paragraphs: list[Paragraph] = []
    for kind, block in iter_body_blocks(document):
        if kind == "paragraph":
            paragraphs.append(block)
        else:
            for row in block.rows:
                for cell in row.cells:
                    paragraphs.extend(cell.paragraphs)
    return paragraphs


def story_reference_exists(
    document: DocumentObject,
    section_index: int,
    *,
    header: bool,
    kind: str,
) -> bool:
    """Check for an explicit or inherited story without creating a part."""

    reference_tag = qn("w:headerReference" if header else "w:footerReference")
    type_name = {"default": "default", "first_page": "first", "even_page": "even"}[kind]
    for index in range(section_index, -1, -1):
        for reference in document.sections[index]._sectPr.findall(reference_tag):
            if reference.get(qn("w:type")) == type_name:
                return True
    return False


def unique_stories(document: DocumentObject, header: bool) -> list[_Header | _Footer]:
    """Return existing unique inherited story parts without creating empty ones."""

    stories: list[_Header | _Footer] = []
    seen: set[str] = set()
    for section_index, section in enumerate(document.sections):
        candidates = (
            (
                ("default", section.header),
                ("first_page", section.first_page_header),
                ("even_page", section.even_page_header),
            )
            if header
            else (
                ("default", section.footer),
                ("first_page", section.first_page_footer),
                ("even_page", section.even_page_footer),
            )
        )
        for kind, story in candidates:
            if not story_reference_exists(
                document,
                section_index,
                header=header,
                kind=kind,
            ):
                continue
            key = str(story.part.partname)
            if key not in seen:
                seen.add(key)
                stories.append(story)
    return stories


def story_paragraphs(story: _Header | _Footer) -> list[Paragraph]:
    """Return story and table-cell paragraphs in reading order."""

    paragraphs = list(story.paragraphs)
    for table in story.tables:
        for row in table.rows:
            for cell in row.cells:
                paragraphs.extend(cell.paragraphs)
    return paragraphs


def paragraph_groups_for_scope(document: DocumentObject, scope: str) -> list[list[Paragraph]]:
    """Select independent paragraph sequences for a replacement operation."""

    groups: list[list[Paragraph]] = []
    if scope in {"body", "all"}:
        groups.append(body_paragraphs_including_tables(document))
    if scope in {"headers", "all"}:
        for story in unique_stories(document, True):
            groups.append(story_paragraphs(story))
    if scope in {"footers", "all"}:
        for story in unique_stories(document, False):
            groups.append(story_paragraphs(story))
    if scope not in {"body", "headers", "footers", "all"}:
        raise ToolError("bad_input", "replace_text.scope is unsupported.")
    return groups


def element_text(element: Any) -> str:
    """Collect visible text tokens from an OOXML element."""

    pieces: list[str] = []
    for descendant in element.iter():
        if descendant.tag in {qn("w:t"), qn("w:delText")} and descendant.text:
            pieces.append(descendant.text)
        elif descendant.tag == qn("w:tab"):
            pieces.append("\t")
        elif descendant.tag in {qn("w:br"), qn("w:cr")}:
            pieces.append("\n")
    return "".join(pieces)


def map_paragraph_text(
    paragraph: Paragraph,
    active_comments: set[str] | None = None,
    active_revisions: set[str] | None = None,
) -> ParagraphTextMap:
    """Map visible text to direct runs while marking protected boundaries."""

    if active_comments is None:
        active_comments = set()
    if active_revisions is None:
        active_revisions = set()
    text_parts: list[str] = []
    pieces: list[TextPiece] = []
    boundaries: list[tuple[int, str]] = []
    position = 0
    field_depth = 0
    for child in paragraph._p.iterchildren():
        if child.tag == qn("w:commentRangeStart"):
            active_comments.add(child.get(qn("w:id"), "unknown"))
            boundaries.append((position, "comment"))
            continue
        if child.tag == qn("w:commentRangeEnd"):
            boundaries.append((position, "comment"))
            active_comments.discard(child.get(qn("w:id"), "unknown"))
            continue
        if child.tag in REVISION_RANGE_START_TAGS:
            active_revisions.add(child.get(qn("w:id"), "unknown"))
            boundaries.append((position, "revision"))
            continue
        if child.tag in REVISION_RANGE_END_TAGS:
            boundaries.append((position, "revision"))
            active_revisions.discard(child.get(qn("w:id"), "unknown"))
            continue
        if child.tag in REVISION_WRAPPER_TAGS or child.tag == qn("w:hyperlink"):
            reason = "revision" if child.tag in REVISION_WRAPPER_TAGS else "hyperlink"
            value = element_text(child)
            boundaries.extend(((position, reason), (position + len(value), reason)))
            pieces.append(TextPiece(position, position + len(value), value, None, reason))
            text_parts.append(value)
            position += len(value)
            continue
        if child.tag != qn("w:r"):
            value = element_text(child)
            if value:
                pieces.append(
                    TextPiece(position, position + len(value), value, None, "unsupported_markup")
                )
                text_parts.append(value)
                position += len(value)
            boundaries.append((position, "unsupported_markup"))
            continue

        field_chars = list(child.iter(qn("w:fldChar")))
        field_types = [item.get(qn("w:fldCharType")) for item in field_chars]
        starts_field = "begin" in field_types
        ends_field = "end" in field_types
        contains_field = bool(field_chars) or child.find(f".//{{{WORD_NS}}}instrText") is not None
        contains_drawing = any(
            child.find(f".//{tag}") is not None
            for tag in (qn("w:drawing"), qn("w:pict"), qn("w:object"))
        )
        contains_comment_ref = child.find(f".//{{{WORD_NS}}}commentReference") is not None
        unsafe_reason = None
        if field_depth or starts_field or contains_field:
            unsafe_reason = "field"
        elif active_comments or contains_comment_ref:
            unsafe_reason = "comment"
        elif active_revisions:
            unsafe_reason = "revision"
        elif contains_drawing:
            unsafe_reason = "drawing"
        value = Run(child, paragraph).text
        if unsafe_reason:
            boundaries.extend(((position, unsafe_reason), (position + len(value), unsafe_reason)))
        pieces.append(TextPiece(position, position + len(value), value, child, unsafe_reason))
        text_parts.append(value)
        position += len(value)
        if starts_field:
            field_depth += field_types.count("begin")
        if ends_field:
            field_depth = max(0, field_depth - field_types.count("end"))
    return ParagraphTextMap(paragraph, "".join(text_parts), pieces, boundaries)


def map_paragraph_groups(groups: list[list[Paragraph]]) -> list[ParagraphTextMap]:
    """Map paragraph groups while carrying protected range state within each story."""

    mapped: list[ParagraphTextMap] = []
    for paragraphs in groups:
        active_comments: set[str] = set()
        active_revisions: set[str] = set()
        for paragraph in paragraphs:
            mapped.append(map_paragraph_text(paragraph, active_comments, active_revisions))
    return mapped


def replace_in_paragraph(
    mapped: ParagraphTextMap,
    find: str,
    replacement: str,
    cross_run_policy: str | None,
) -> int:
    """Validate and perform all non-overlapping matches in one paragraph."""

    matches = list(re.finditer(re.escape(find), mapped.text))
    plans: list[tuple[int, int, list[TextPiece]]] = []
    for match in matches:
        start, end = match.span()
        overlapped = [
            piece
            for piece in mapped.pieces
            if piece.end > start and piece.start < end and piece.text
        ]
        unsafe = next((piece.unsafe_reason for piece in overlapped if piece.unsafe_reason), None)
        boundary = next(
            (reason for position, reason in mapped.boundaries if start < position < end),
            None,
        )
        if unsafe or boundary or not overlapped:
            raise ToolError(
                "ambiguous_edit",
                "Replacement match crosses or occupies protected markup.",
                details={
                    "find": find,
                    "reason": unsafe or boundary or "unmapped_text",
                    "paragraph": mapped.text,
                },
            )
        runs = []
        for piece in overlapped:
            if piece.run_element not in runs:
                runs.append(piece.run_element)
        if len(runs) > 1 and cross_run_policy != "first_run":
            raise ToolError(
                "ambiguous_edit",
                "Cross-run replacement requires cross_run_policy: first_run.",
                details={"find": find, "paragraph": mapped.text},
            )
        if len(runs) == 1 and cross_run_policy not in {None, "first_run"}:
            raise ToolError("bad_input", "Unsupported cross_run_policy.")
        plans.append((start, end, overlapped))

    for start, end, overlapped in reversed(plans):
        first_piece = overlapped[0]
        last_piece = overlapped[-1]
        first_run = Run(first_piece.run_element, mapped.paragraph)
        last_run = Run(last_piece.run_element, mapped.paragraph)
        start_in_first = start - first_piece.start
        end_in_last = end - last_piece.start
        prefix = first_run.text[:start_in_first]
        suffix = last_run.text[end_in_last:]
        unique_elements: list[Any] = []
        for piece in overlapped:
            if piece.run_element not in unique_elements:
                unique_elements.append(piece.run_element)
        if len(unique_elements) == 1:
            first_run.text = prefix + replacement + suffix
        else:
            first_run.text = prefix + replacement
            for element in unique_elements[1:-1]:
                Run(element, mapped.paragraph).text = ""
            last_run.text = suffix
    return len(matches)


def replace_text(document: DocumentObject, operation: dict[str, Any]) -> int:
    """Perform a count-bounded, run-aware replacement."""

    find = required_string(operation.get("find"), "replace_text.find")
    replacement = string_value(operation.get("replace"), "replace_text.replace")
    expected = required_integer(
        operation.get("expected_count"), "replace_text.expected_count", minimum=0
    )
    replace_all = optional_boolean(operation.get("replace_all"), "replace_text.replace_all")
    scope = string_value(operation.get("scope", "body"), "replace_text.scope")
    policy = operation.get("cross_run_policy")
    if policy is not None and policy != "first_run":
        raise ToolError("bad_input", "Only cross_run_policy: first_run is supported.")
    mapped_paragraphs = map_paragraph_groups(paragraph_groups_for_scope(document, scope))
    total = sum(
        len(list(re.finditer(re.escape(find), mapped.text))) for mapped in mapped_paragraphs
    )
    if total != expected:
        raise ToolError(
            "ambiguous_edit",
            "Replacement count did not match expected_count.",
            details={"find": find, "expected_count": expected, "actual_count": total},
        )
    if total > 1 and not replace_all:
        raise ToolError(
            "ambiguous_edit",
            "Multiple matches require replace_all: true.",
            details={"find": find, "actual_count": total},
        )
    changed = 0
    for mapped in mapped_paragraphs:
        if find in mapped.text:
            changed += replace_in_paragraph(mapped, find, replacement, policy)
    return changed


def paragraph_at(document: DocumentObject, index: Any) -> Paragraph:
    """Return one top-level body paragraph by index."""

    value = required_integer(index, "paragraph_index", minimum=0)
    paragraphs = document.paragraphs
    if value >= len(paragraphs):
        raise ToolError("bad_input", "paragraph_index is out of range.")
    return paragraphs[value]


def table_at(document: DocumentObject, index: Any) -> Table:
    """Return one top-level body table by index."""

    value = required_integer(index, "table_index", minimum=0)
    tables = document.tables
    if value >= len(tables):
        raise ToolError("bad_input", "table_index is out of range.")
    return tables[value]


def insert_paragraph(document: DocumentObject, operation: dict[str, Any], base_dir: Path) -> None:
    """Append explicitly, or insert relative to a pre-mutation paragraph index."""

    block = operation.get("block")
    if not isinstance(block, dict):
        raise ToolError("bad_input", "insert_paragraph.block must be an object.")
    position = operation.get("position")
    if position == "append":
        if "paragraph_index" in operation:
            raise ToolError(
                "bad_input",
                "paragraph_index must be omitted when insert_paragraph.position is append.",
            )
        add_paragraph_block(document, block, base_dir)
        return
    if position not in {"before", "after"}:
        raise ToolError(
            "bad_input",
            "insert_paragraph.position must be append, before, or after.",
        )
    if "paragraph_index" not in operation:
        raise ToolError(
            "bad_input",
            "paragraph_index is required for relative paragraph insertion.",
        )
    anchor = paragraph_at(document, operation["paragraph_index"])
    paragraph = add_paragraph_block(document, block, base_dir)
    if position == "before":
        anchor._p.addprevious(paragraph._p)
    else:
        anchor._p.addnext(paragraph._p)


def delete_paragraph(document: DocumentObject, operation: dict[str, Any]) -> None:
    """Delete one top-level body paragraph."""

    paragraph = paragraph_at(document, operation.get("paragraph_index"))
    parent = paragraph._p.getparent()
    parent.remove(paragraph._p)


def insert_table(document: DocumentObject, operation: dict[str, Any]) -> None:
    """Insert a table, optionally relative to a paragraph."""

    block = operation.get("table")
    if not isinstance(block, dict):
        raise ToolError("bad_input", "insert_table.table must be an object.")
    table = add_table_block(document, block)
    if "paragraph_index" not in operation:
        return
    anchor = paragraph_at(document, operation["paragraph_index"])
    position = operation.get("position", "after")
    if position == "before":
        anchor._p.addprevious(table._tbl)
    elif position == "after":
        anchor._p.addnext(table._tbl)
    else:
        raise ToolError("bad_input", "insert_table.position must be before or after.")


def update_table(document: DocumentObject, operation: dict[str, Any]) -> None:
    """Update table properties and optionally one cell."""

    table = table_at(document, operation.get("table_index"))
    if "style" in operation:
        assign_named_style(
            table,
            required_string(operation["style"], "update_table.style"),
            "update_table.style",
        )
    property_spec = {
        key: operation[key]
        for key in (
            "header_rows",
            "layout",
            "column_widths_inches",
            "allow_row_split",
            "row_allow_split",
        )
        if key in operation
    }
    if "column_widths_inches" in property_spec and len(
        property_spec["column_widths_inches"]
    ) != len(table.columns):
        raise ToolError("bad_input", "column_widths_inches must contain one width per column.")
    if "header_rows" in property_spec and property_spec["header_rows"] > len(table.rows):
        raise ToolError("bad_input", "header_rows exceeds the table row count.")
    if "row_allow_split" in property_spec and len(property_spec["row_allow_split"]) != len(
        table.rows
    ):
        raise ToolError("bad_input", "row_allow_split must contain one Boolean per row.")
    if property_spec:
        fill_table_properties(table, property_spec)
    if "row" in operation:
        row_index = required_integer(operation["row"], "update_table.row", minimum=0)
        column_index = required_integer(operation["column"], "update_table.column", minimum=0)
        if row_index >= len(table.rows) or column_index >= len(table.rows[row_index].cells):
            raise ToolError("bad_input", "Table cell index is out of range.")
        set_cell_value(table.cell(row_index, column_index), operation["value"])


def fill_table_properties(table: Table, spec: dict[str, Any]) -> None:
    """Apply table layout properties without touching cell content."""

    table.autofit = spec.get("layout", "autofit" if table.autofit else "fixed") == "autofit"
    if "column_widths_inches" in spec:
        widths = [Inches(value) for value in spec["column_widths_inches"]]
        apply_cell_widths(table, widths)
    header_rows = spec.get("header_rows")
    for row_index, row in enumerate(table.rows):
        tr_pr = row._tr.get_or_add_trPr()
        if header_rows is not None:
            for node in list(tr_pr.findall(qn("w:tblHeader"))):
                tr_pr.remove(node)
            if row_index < header_rows:
                node = OxmlElement("w:tblHeader")
                node.set(qn("w:val"), "true")
                tr_pr.append(node)
        if "allow_row_split" in spec or "row_allow_split" in spec:
            for node in list(tr_pr.findall(qn("w:cantSplit"))):
                tr_pr.remove(node)
            allows = (
                spec["row_allow_split"][row_index]
                if "row_allow_split" in spec
                else spec["allow_row_split"]
            )
            if not allows:
                node = OxmlElement("w:cantSplit")
                node.set(qn("w:val"), "true")
                tr_pr.append(node)


def insert_image(
    document: DocumentObject,
    operation: dict[str, Any],
    base_dir: Path,
) -> None:
    """Insert an inline image into an existing or new paragraph."""

    image = resolve_path(operation.get("path"), base_dir, "insert_image.path")
    validate_image(image)
    width, height = image_dimensions(operation)
    if "paragraph_index" in operation:
        paragraph = paragraph_at(document, operation["paragraph_index"])
    else:
        paragraph = document.add_paragraph()
    shape = paragraph.add_run().add_picture(str(image), width=width, height=height)
    set_drawing_accessibility(shape._inline, operation)


def replace_image(
    document: DocumentObject,
    operation: dict[str, Any],
    base_dir: Path,
) -> None:
    """Replace one body inline image relationship without flattening its drawing."""

    if "story_address" in operation:
        match = re.fullmatch(r"([^#]+)#inline-(\d+)", operation["story_address"])
        if not match or match.group(1) != "word/document.xml":
            raise ToolError("bad_input", "Only body story_address image replacement is supported.")
        index = int(match.group(2))
    else:
        index = required_integer(
            operation.get("inline_image_index"), "inline_image_index", minimum=0
        )
    shapes = document.inline_shapes
    if index >= len(shapes):
        raise ToolError("bad_input", "inline_image_index is out of range.")
    image = resolve_path(operation.get("path"), base_dir, "replace_image.path")
    validate_image(image)
    width, height = image_dimensions(operation)
    try:
        relationship_id, _ = document.part.get_or_add_image(str(image))
        inline = shapes[index]._inline
        blip = next(inline.iter(f"{{{A_NS}}}blip"))
        blip.set(f"{{{REL_NS}}}embed", relationship_id)
        if width is not None:
            shapes[index].width = width
        if height is not None:
            shapes[index].height = height
        if {"alt_text", "decorative", "title"}.intersection(operation):
            decorative = (
                inline.docPr.find(
                    f"{{{A_NS}}}extLst/{{{A_NS}}}ext"
                    f"[@uri='{DECORATIVE_EXTENSION_URI}']/"
                    f"{{{A14_NS}}}decorative[@val='1']"
                )
                is not None
            )
            existing = {
                "alt_text": inline.docPr.get("descr"),
                "title": inline.docPr.get("title"),
                "decorative": decorative,
            }
            set_drawing_accessibility(inline, {**existing, **operation})
    except Exception as exc:
        raise ToolError(
            "validation_failed",
            "Inline image relationship replacement failed.",
            details={"reason": str(exc)},
        ) from exc


def set_story_operation(
    document: DocumentObject,
    operation: dict[str, Any],
    base_dir: Path,
    *,
    header: bool,
) -> None:
    """Replace or append one header/footer story."""

    section_index = required_integer(operation.get("section_index", 0), "section_index", minimum=0)
    if section_index >= len(document.sections):
        raise ToolError("bad_input", "Header/footer section index is out of range.")
    story = story_for(
        document.sections[section_index],
        required_string(operation.get("kind", "default"), "story.kind"),
        header,
    )
    if operation.get("link_to_previous") and "blocks" in operation:
        raise ToolError("bad_input", "Cannot write blocks while link_to_previous is true.")
    if "link_to_previous" in operation:
        story.is_linked_to_previous = operation["link_to_previous"]
        if operation["link_to_previous"] and "blocks" not in operation:
            return
    else:
        story.is_linked_to_previous = False
    mode = operation.get("mode", "replace")
    if mode == "replace":
        clear_story(story)
    elif mode != "append":
        raise ToolError("bad_input", "Header/footer mode must be replace or append.")
    apply_blocks(story, operation.get("blocks", []), base_dir, allow_sections=False)
    if not story.paragraphs:
        story.add_paragraph()


def apply_edit_operations(
    document: DocumentObject,
    operations: Any,
    base_dir: Path,
) -> Counter[str]:
    """Apply sequential edit operations in memory."""

    if not isinstance(operations, list) or not operations:
        raise ToolError("bad_input", "edit.operations must be a non-empty array.")
    counts: Counter[str] = Counter()
    for operation in operations:
        if not isinstance(operation, dict):
            raise ToolError("bad_input", "Each edit operation must be an object.")
        operation_type = required_string(operation.get("type"), "operation.type")
        if operation_type == "replace_text":
            counts["text_matches"] += replace_text(document, operation)
        elif operation_type == "insert_paragraph":
            insert_paragraph(document, operation, base_dir)
        elif operation_type == "delete_paragraph":
            delete_paragraph(document, operation)
        elif operation_type == "insert_table":
            insert_table(document, operation)
        elif operation_type == "update_table":
            update_table(document, operation)
        elif operation_type == "insert_image":
            insert_image(document, operation, base_dir)
        elif operation_type == "replace_image":
            replace_image(document, operation, base_dir)
        elif operation_type == "set_metadata":
            set_metadata(document, operation.get("metadata"))
        elif operation_type == "set_header":
            set_story_operation(document, operation, base_dir, header=True)
        elif operation_type == "set_footer":
            set_story_operation(document, operation, base_dir, header=False)
        elif operation_type == "configure_section":
            index = required_integer(operation.get("section_index", 0), "section_index", minimum=0)
            if index >= len(document.sections):
                raise ToolError("bad_input", "section_index is out of range.")
            configure_section(document.sections[index], operation)
        elif operation_type == "upsert_style":
            upsert_style(document, operation["style"])
        elif operation_type == "set_update_fields_on_open":
            set_update_fields_on_open(document, operation["enabled"])
        else:
            raise ToolError(
                "unsupported_operation",
                f"Unsupported edit operation: {operation_type}",
            )
        counts[operation_type] += 1
    return counts


def edit_document(
    source: Path,
    spec: dict[str, Any],
    base_dir: Path,
    output: Path,
    *,
    overwrite: bool,
    allow_external_relationships: bool,
) -> dict[str, Any]:
    """Apply an atomic edit job."""

    preflight_docx(
        source,
        allow_external_relationships=allow_external_relationships,
    )
    document = load_document(source)
    counts = apply_edit_operations(document, spec.get("operations"), base_dir)
    verification, inspection, warnings = atomic_docx_save(
        document,
        output,
        overwrite=overwrite,
        source=source,
        allow_external_relationships=allow_external_relationships,
    )
    return success_summary(
        "edit",
        paths={"input": str(source), "output": str(output)},
        counts=dict(counts),
        warnings=warnings,
        verification=verification,
        result={
            "paragraphs": len(inspection["paragraphs"]),
            "tables": len(inspection["tables"]),
            "inline_images": len(inspection["images"]["inline_occurrences"]),
        },
    )


def extract_text(inspection: dict[str, Any]) -> str:
    """Extract body text and identity-distinct header/footer text and tables."""

    lines: list[str] = []
    for block in inspection["ordered_blocks"]:
        if block["type"] == "paragraph":
            lines.append(block["value"]["text"])
        else:
            for row in block["value"]["rows"]:
                lines.append("\t".join(cell["text"] for cell in row))
    seen_stories: set[tuple[str, str]] = set()
    for label, stories in (("Header", inspection["headers"]), ("Footer", inspection["footers"])):
        for story in stories:
            values = [item["text"] for item in story["paragraphs"] if item["text"]]
            for table in story["tables"]:
                values.extend("\t".join(cell["text"] for cell in row) for row in table["rows"])
            identity = (label, story["part"])
            if values and identity not in seen_stories:
                seen_stories.add(identity)
                lines.extend(("", f"[{label}: {story['kind']}]", *values))
    return "\n".join(lines).rstrip() + "\n"


def atomic_bytes_write(
    output: Path,
    data: bytes,
    *,
    suffix: str,
    overwrite: bool,
    source: Path,
    validator: Callable[[Path], dict[str, Any]],
) -> dict[str, Any]:
    """Write bytes to a sibling temporary file, validate, and publish."""

    ensure_output_target(output, expected_suffix=suffix, overwrite=overwrite, source=source)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output.name}.",
            suffix=suffix,
            dir=output.parent,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        verification = validator(temp_path)
        publish_temp_file(temp_path, output, overwrite=overwrite)
        temp_path = None
        return {"atomic_publish": True, **verification}
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(
            "validation_failed",
            "Output write or validation failed.",
            details={"path": str(output), "reason": str(exc)},
        ) from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def validate_text(path: Path) -> dict[str, Any]:
    """Reopen a UTF-8 text output."""

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ToolError("validation_failed", "Text output did not reopen as UTF-8.") from exc
    return {"reopened": True, "utf8": True, "characters": len(text)}


def validate_pdf(path: Path) -> dict[str, Any]:
    """Check and reopen a PDF within fixed byte, page, and extraction bounds."""

    try:
        byte_count = path.stat().st_size
        if byte_count > MAX_PDF_BYTES:
            raise ToolError(
                "resource_limit",
                "Converted PDF exceeds the byte limit.",
                details={"bytes": byte_count, "limit": MAX_PDF_BYTES},
            )
        with path.open("rb") as handle:
            signature = handle.read(5)
        if signature != b"%PDF-":
            raise ToolError("validation_failed", "Converted output lacks a PDF signature.")
        previous_zlib_limit = pdf_filters.ZLIB_MAX_OUTPUT_LENGTH
        pdf_filters.ZLIB_MAX_OUTPUT_LENGTH = MAX_PDF_DECOMPRESSED_STREAM_BYTES
        try:
            reader = PdfReader(str(path))
            page_count = len(reader.pages)
            if page_count < 1:
                raise ToolError("validation_failed", "Converted PDF has no pages.")
            if page_count > MAX_PDF_PAGES:
                raise ToolError(
                    "resource_limit",
                    "Converted PDF exceeds the page limit.",
                    details={"pages": page_count, "limit": MAX_PDF_PAGES},
                )
            text_characters = 0
            extraction_failures = 0
            page_quality: list[dict[str, Any]] = []
            text_pages_checked = min(page_count, MAX_PDF_TEXT_PAGES)
            for index in range(text_pages_checked):
                try:
                    page = reader.pages[index]
                    contents = page.get_contents()
                    if contents is not None:
                        contents.get_data()
                    extracted = page.extract_text() or ""
                except LimitReachedError as exc:
                    raise ToolError(
                        "resource_limit",
                        "Converted PDF exceeds the decompressed-stream validation limit.",
                        details={
                            "limit": MAX_PDF_DECOMPRESSED_STREAM_BYTES,
                            "page": index,
                        },
                    ) from exc
                except Exception:
                    extraction_failures += 1
                    continue
                text_characters += len(extracted)
                media_box = page.mediabox
                normalized = " ".join(extracted.split())
                resources = page.get("/Resources")
                if resources is not None and hasattr(resources, "get_object"):
                    resources = resources.get_object()
                xobjects = resources.get("/XObject") if resources is not None else None
                if xobjects is not None and hasattr(xobjects, "get_object"):
                    xobjects = xobjects.get_object()
                has_nontext_content = bool(xobjects)
                low_text = len(normalized) < 20
                page_quality.append(
                    {
                        "page_index": index,
                        "width_points": round(float(media_box.width), 3),
                        "height_points": round(float(media_box.height), 3),
                        "text_characters": len(extracted),
                        "low_text": low_text,
                        "has_nontext_content": has_nontext_content,
                        "nearly_blank": low_text and not has_nontext_content,
                        "content_anchors": {
                            "start": normalized[:80],
                            "end": normalized[-80:] if normalized else "",
                        },
                    }
                )
                if text_characters > MAX_PDF_TEXT_CHARACTERS:
                    raise ToolError(
                        "resource_limit",
                        "Converted PDF exceeds the text-extraction character limit.",
                        details={
                            "characters": text_characters,
                            "limit": MAX_PDF_TEXT_CHARACTERS,
                            "pages_checked": index + 1,
                        },
                    )
        finally:
            pdf_filters.ZLIB_MAX_OUTPUT_LENGTH = previous_zlib_limit
        return {
            "signature": True,
            "reopened": True,
            "pdf_bytes": byte_count,
            "pdf_byte_limit": MAX_PDF_BYTES,
            "page_count": page_count,
            "page_limit": MAX_PDF_PAGES,
            "text_pages_checked": text_pages_checked,
            "text_page_limit": MAX_PDF_TEXT_PAGES,
            "text_characters": text_characters,
            "text_character_limit": MAX_PDF_TEXT_CHARACTERS,
            "decompressed_stream_byte_limit": MAX_PDF_DECOMPRESSED_STREAM_BYTES,
            "text_extraction_failures": extraction_failures,
            "content_quality_report": {
                "deterministic": True,
                "method": "pypdf text extraction and page-resource XObject detection",
                "renderer_version": distribution_version("pypdf"),
                "limitation": (
                    "This is not visual rendering. Non-text detection is limited to page "
                    "XObject resources and may not identify all visible vector content."
                ),
                "pages": page_quality,
                "nearly_blank_pages": [
                    page["page_index"] for page in page_quality if page["nearly_blank"]
                ],
                "raster_review_artifacts": [],
            },
        }
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(
            "validation_failed",
            "Converted PDF could not be reopened.",
            details={"reason": str(exc)},
        ) from exc


def process_group_exists(process_group: int) -> bool:
    """Return whether a POSIX process group still has members."""

    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_process_group(process: subprocess.Popen[bytes]) -> bool:
    """Terminate an isolated process group and verify that no group members remain."""

    process_group = process.pid
    deadline = time.monotonic() + PROCESS_TERMINATION_GRACE_SECONDS
    if process_group_exists(process_group):
        try:
            os.killpg(process_group, signal.SIGTERM)
        except (PermissionError, ProcessLookupError):
            pass
    try:
        process.wait(timeout=max(0.01, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        pass
    while process_group_exists(process_group) and time.monotonic() < deadline:
        time.sleep(0.05)
    if process_group_exists(process_group):
        try:
            os.killpg(process_group, signal.SIGKILL)
        except (PermissionError, ProcessLookupError):
            pass
    if process.poll() is None:
        process.kill()
    process.wait()
    kill_deadline = time.monotonic() + PROCESS_TERMINATION_GRACE_SECONDS
    while process_group_exists(process_group) and time.monotonic() < kill_deadline:
        time.sleep(0.05)
    return not process_group_exists(process_group)


def bounded_process_run(
    command: list[str],
    *,
    timeout: int,
    environment: dict[str, str] | None = None,
) -> tuple[int, str, str, bool, bool]:
    """Run in a new session, bound diagnostics, and clean up the process group."""

    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(
            command,
            stdout=stdout_file,
            stderr=stderr_file,
            env=environment,
            start_new_session=True,
        )
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            process_group_cleaned = terminate_process_group(process)
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read(MAX_SUBPROCESS_DIAGNOSTIC_BYTES).decode(
            "utf-8",
            errors="replace",
        )
        stderr = stderr_file.read(MAX_SUBPROCESS_DIAGNOSTIC_BYTES).decode(
            "utf-8",
            errors="replace",
        )
    return process.returncode, stdout, stderr, timed_out, process_group_cleaned


def find_soffice() -> str:
    """Resolve LibreOffice: explicit override, PATH, then well-known app locations."""

    override = os.environ.get("DOCX_SOFFICE")
    if override:
        candidate = Path(override)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
        raise ToolError(
            "missing_dependency",
            "DOCX_SOFFICE does not point to an executable LibreOffice binary.",
            details={"path": override},
        )
    for name in ("soffice", "libreoffice"):
        executable = shutil.which(name)
        if executable is not None:
            return executable
    if sys.platform == "darwin":
        for candidate in (
            Path("/Applications/LibreOffice.app/Contents/MacOS/soffice"),
            Path.home() / "Applications/LibreOffice.app/Contents/MacOS/soffice",
        ):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    raise ToolError(
        "missing_dependency",
        "LibreOffice soffice is required for DOCX-to-PDF conversion.",
        details={
            "searched": [
                "DOCX_SOFFICE",
                "PATH:soffice",
                "PATH:libreoffice",
                "darwin:/Applications/LibreOffice.app/Contents/MacOS/soffice",
                "darwin:~/Applications/LibreOffice.app/Contents/MacOS/soffice",
            ],
        },
    )


def libreoffice_version(executable: str) -> str:
    """Read a bounded LibreOffice version string."""

    try:
        returncode, stdout, stderr, timed_out, process_group_cleaned = bounded_process_run(
            [executable, "--version"],
            timeout=15,
        )
    except OSError as exc:
        raise ToolError(
            "missing_dependency",
            "LibreOffice executable could not be started.",
            details={"reason": str(exc)},
        ) from exc
    if timed_out:
        raise ToolError(
            "missing_dependency",
            "LibreOffice version check timed out.",
            details={"timeout_seconds": 15},
        )
    if not process_group_cleaned:
        raise ToolError(
            "missing_dependency",
            "LibreOffice version-check process group could not be cleaned up.",
        )
    text = (stdout or stderr).strip()
    if returncode != 0 and not text:
        return "unknown"
    return text[:300] or "unknown"


def require_pypdfium2() -> Any:
    """Import the optional pypdfium2 rasterizer or fail with an install hint."""

    try:
        import pypdfium2
    except ModuleNotFoundError as exc:
        raise ToolError(
            "missing_dependency",
            "Page rendering requires the optional pypdfium2 package.",
            details={"install": f'"{sys.executable}" -m pip install pypdfium2'},
        ) from exc
    return pypdfium2


def file_sha256(path: Path) -> str:
    """Hash a file in bounded chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_render_pages(value: str | None, page_count: int) -> list[int]:
    """Select 1-based PDF page numbers, validated against the converted page count."""

    if value is None:
        selected = list(range(1, page_count + 1))
    else:
        chosen: set[int] = set()
        for part in value.split(","):
            text = part.strip()
            if not text.isdigit():
                raise ToolError(
                    "bad_input",
                    "pages must be a comma-separated list of 1-based page numbers.",
                    details={"pages": value},
                )
            page = int(text)
            if not 1 <= page <= page_count:
                raise ToolError(
                    "bad_input",
                    "Requested page does not exist in the converted PDF.",
                    details={"page": page, "pdf_pages": page_count},
                )
            chosen.add(page)
        if not chosen:
            raise ToolError("bad_input", "pages must select at least one page.")
        selected = sorted(chosen)
    if len(selected) > MAX_RENDER_PAGES:
        raise ToolError(
            "resource_limit",
            "Rendering would produce too many pages; select a subset with pages.",
            details={"pages": len(selected), "limit": MAX_RENDER_PAGES},
        )
    return selected


def publish_directory_atomic(staged: Path, output: Path) -> None:
    """Publish a staged directory to a destination that must not already exist."""

    if output.exists():
        raise ToolError(
            "output_conflict",
            "Atomic render publication requires a destination directory that does not "
            "already exist.",
            details={"path": str(output)},
        )
    try:
        os.replace(staged, output)
    except OSError as exc:
        shutil.rmtree(staged, ignore_errors=True)
        raise ToolError(
            "validation_failed",
            "Could not atomically publish the output directory.",
            details={"path": str(output), "reason": str(exc)},
        ) from exc


def convert_pdf_bytes(source: Path, timeout: int) -> tuple[bytes, dict[str, Any], str]:
    """Run headless LibreOffice with an isolated temporary profile."""

    executable = find_soffice()
    version = libreoffice_version(executable)
    with tempfile.TemporaryDirectory(prefix="docx-tool-lo-") as temp_name:
        root = Path(temp_name)
        profile = root / "profile"
        output_dir = root / "output"
        output_dir.mkdir()
        command = [
            executable,
            "--headless",
            f"-env:UserInstallation={profile.resolve().as_uri()}",
            "--convert-to",
            "pdf",
            "--outdir",
            str(output_dir),
            str(source.resolve()),
        ]
        environment = {
            **os.environ,
            "HOME": str(root / "home"),
            "TMPDIR": str(root / "tmp"),
            "SAL_USE_VCLPLUGIN": "gen",
        }
        Path(environment["HOME"]).mkdir()
        Path(environment["TMPDIR"]).mkdir()
        try:
            returncode, stdout, stderr, timed_out, process_group_cleaned = bounded_process_run(
                command,
                timeout=timeout,
                environment=environment,
            )
        except OSError as exc:
            raise ToolError(
                "external_tool_failed",
                "LibreOffice conversion could not start.",
                details={"reason": str(exc)},
            ) from exc
        diagnostics = {
            "returncode": returncode,
            "stdout": redact(stdout.strip()),
            "stderr": redact(stderr.strip()),
            "timeout_seconds": timeout,
            "isolated_profile": True,
            "isolated_process_group": True,
            "process_group_cleaned": process_group_cleaned,
        }
        if timed_out:
            raise ToolError(
                "external_tool_failed",
                (
                    "LibreOffice conversion timed out."
                    if not process_group_cleaned
                    else "LibreOffice conversion timed out and its process group was terminated."
                ),
                details=diagnostics,
            )
        if not process_group_cleaned:
            raise ToolError(
                "external_tool_failed",
                "LibreOffice conversion process group could not be cleaned up.",
                details=diagnostics,
            )
        if returncode != 0:
            raise ToolError(
                "external_tool_failed",
                "LibreOffice conversion failed.",
                details=diagnostics,
            )
        pdfs = list(output_dir.glob("*.pdf"))
        if len(pdfs) != 1:
            raise ToolError(
                "external_tool_failed",
                "LibreOffice did not produce exactly one PDF.",
                details={**diagnostics, "outputs": [path.name for path in pdfs]},
            )
        try:
            validate_pdf(pdfs[0])
        except ToolError as exc:
            if exc.category == "resource_limit":
                raise
            raise ToolError(
                "external_tool_failed",
                "LibreOffice produced an invalid PDF.",
                details={
                    **diagnostics,
                    "validation": {
                        "category": exc.category,
                        "message": exc.message,
                        "details": exc.details,
                    },
                },
            ) from exc
        return pdfs[0].read_bytes(), diagnostics, version


def convert_document(
    source: Path,
    output: Path,
    format_name: str,
    *,
    overwrite: bool,
    timeout: int,
    allow_external_relationships: bool,
) -> dict[str, Any]:
    """Convert DOCX to UTF-8 text or best-effort PDF."""

    if timeout < 1 or timeout > 3_600:
        raise ToolError("bad_input", "timeout must be between 1 and 3600 seconds.")
    preflight_docx(
        source,
        allow_external_relationships=allow_external_relationships,
    )
    if format_name == "text":
        inspection, warnings = inspect_document(
            source,
            allow_external_relationships=allow_external_relationships,
        )
        text = extract_text(inspection)
        verification = atomic_bytes_write(
            output,
            text.encode("utf-8"),
            suffix=".txt",
            overwrite=overwrite,
            source=source,
            validator=validate_text,
        )
        return success_summary(
            "convert",
            paths={"input": str(source), "output": str(output)},
            counts={
                "paragraphs": len(inspection["paragraphs"]),
                "tables": len(inspection["tables"]),
                "characters": len(text),
            },
            warnings=warnings,
            verification=verification,
            result={"format": "text"},
        )
    if format_name == "pdf":
        source_inspection, source_warnings = inspect_document(
            source,
            allow_external_relationships=allow_external_relationships,
        )
        data, conversion_diagnostics, version = convert_pdf_bytes(source, timeout)
        verification = atomic_bytes_write(
            output,
            data,
            suffix=".pdf",
            overwrite=overwrite,
            source=source,
            validator=validate_pdf,
        )
        diagnostic(
            "info",
            "libreoffice",
            "LibreOffice conversion completed.",
            **conversion_diagnostics,
        )
        return success_summary(
            "convert",
            paths={"input": str(source), "output": str(output)},
            counts={
                "pages": verification["page_count"],
                "source_fields": source_inspection["features"]["fields"],
                "source_revisions": source_inspection["features"]["revisions"],
                "source_comments": source_inspection["features"]["comments"],
            },
            warnings=[
                *source_warnings,
                warning(
                    "layout_engine_variance",
                    "DOCX-to-PDF is best-effort and may differ from Microsoft Word layout.",
                ),
            ],
            verification=verification,
            result={"format": "pdf", "conversion": conversion_diagnostics},
            libreoffice=version,
        )
    raise ToolError(
        "unsupported_operation",
        "convert format must be text or pdf.",
        details={"format": format_name},
    )


def render_document(
    source: Path,
    output: Path,
    *,
    dpi: int,
    pages: str | None,
    timeout: int,
    allow_external_relationships: bool,
) -> dict[str, Any]:
    """Render document pages to one PNG each through LibreOffice and PDFium."""

    if (
        isinstance(dpi, bool)
        or not isinstance(dpi, int)
        or not MIN_RENDER_DPI <= dpi <= MAX_RENDER_DPI
    ):
        raise ToolError(
            "bad_input",
            f"dpi must be an integer between {MIN_RENDER_DPI} and {MAX_RENDER_DPI}.",
            details={"dpi": dpi},
        )
    if timeout < 1 or timeout > 3_600:
        raise ToolError("bad_input", "timeout must be between 1 and 3600 seconds.")
    if output.suffix:
        raise ToolError(
            "bad_input",
            "Render output must be a directory path without a suffix.",
            details={"path": str(output)},
        )
    if output.exists():
        raise ToolError(
            "output_conflict",
            "Atomic render publication requires a destination directory that does not "
            "already exist.",
            details={"path": str(output)},
        )
    if not output.parent.is_dir():
        raise ToolError(
            "bad_input",
            "Destination parent directory does not exist.",
            details={"path": str(output.parent)},
        )
    pdfium = require_pypdfium2()
    preflight_docx(source, allow_external_relationships=allow_external_relationships)
    source_inspection, source_warnings = inspect_document(
        source,
        allow_external_relationships=allow_external_relationships,
    )
    source_hash = file_sha256(source)
    data, conversion_diagnostics, version = convert_pdf_bytes(source, timeout)
    images: list[dict[str, Any]] = []
    staged = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        with tempfile.TemporaryDirectory(prefix="docx-tool-render-") as temp_name:
            temp_pdf = Path(temp_name) / "converted.pdf"
            temp_pdf.write_bytes(data)
            pdf_verification = validate_pdf(temp_pdf)
            page_count = pdf_verification["page_count"]
            try:
                document = pdfium.PdfDocument(str(temp_pdf))
            except Exception as exc:
                raise ToolError(
                    "external_tool_failed",
                    "PDFium could not open the converted PDF.",
                    details={"reason": str(exc)},
                ) from exc
            try:
                if len(document) != page_count:
                    raise ToolError(
                        "validation_failed",
                        "PDFium and pypdf disagree about the converted page count.",
                        details={"pypdf_pages": page_count, "pdfium_pages": len(document)},
                    )
                selected = parse_render_pages(pages, page_count)
                total_pixels = 0
                for page_number in selected:
                    page = document[page_number - 1]
                    width_pt, height_pt = page.get_size()
                    pixels = math.ceil(width_pt / 72 * dpi) * math.ceil(height_pt / 72 * dpi)
                    if pixels > MAX_RENDER_PIXELS_PER_PAGE:
                        raise ToolError(
                            "resource_limit",
                            "Rendered pixel count for one page exceeds the limit.",
                            details={
                                "page": page_number,
                                "pixels": pixels,
                                "limit": MAX_RENDER_PIXELS_PER_PAGE,
                            },
                        )
                    total_pixels += pixels
                    if total_pixels > MAX_RENDER_TOTAL_PIXELS:
                        raise ToolError(
                            "resource_limit",
                            "Total rendered pixel count exceeds the limit.",
                            details={"pixels": total_pixels, "limit": MAX_RENDER_TOTAL_PIXELS},
                        )
                    file_name = f"page-{page_number:03d}.png"
                    staged_file = staged / file_name
                    try:
                        page.render(scale=dpi / 72).to_pil().save(staged_file, format="PNG")
                    except Exception as exc:
                        raise ToolError(
                            "external_tool_failed",
                            "PDFium could not rasterize a PDF page.",
                            details={"page": page_number, "reason": str(exc)},
                        ) from exc
                    with staged_file.open("rb") as rendered:
                        if rendered.read(8) != b"\x89PNG\r\n\x1a\n":
                            raise ToolError(
                                "validation_failed",
                                "Rendered file does not have a PNG signature.",
                                details={"file": file_name},
                            )
                    with PILImage.open(staged_file) as verified:
                        verified.load()
                        size = verified.size
                    if size[0] * size[1] > MAX_RENDER_PIXELS_PER_PAGE:
                        raise ToolError(
                            "validation_failed",
                            "Rendered image exceeds the per-page pixel limit.",
                            details={"file": file_name},
                        )
                    png_bytes = staged_file.read_bytes()
                    images.append(
                        {
                            "page": page_number,
                            "file": file_name,
                            "width_px": size[0],
                            "height_px": size[1],
                            "bytes": len(png_bytes),
                            "sha256": hashlib.sha256(png_bytes).hexdigest(),
                        }
                    )
            finally:
                document.close()
        if file_sha256(source) != source_hash:
            raise ToolError(
                "validation_failed",
                "Source changed before the destination publication gate.",
                details={"path": str(source)},
            )
        publish_directory_atomic(staged, output)
    except Exception:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    reopened = 0
    for record in images:
        published = output / record["file"]
        png_bytes = published.read_bytes()
        if hashlib.sha256(png_bytes).hexdigest() != record["sha256"]:
            raise ToolError(
                "validation_failed",
                "Published rendering does not match its staged hash.",
                details={"file": record["file"]},
            )
        with PILImage.open(published) as verified:
            verified.load()
        reopened += 1
    pdf_verification["content_quality_report"]["raster_review_artifacts"] = [
        record["file"] for record in images
    ]
    warnings = [
        *source_warnings,
        warning(
            "layout_engine_variance",
            "DOCX-to-PDF is best-effort and may differ from Microsoft Word layout.",
        ),
        warning(
            "render_review_required",
            "Rendering exists to be looked at: open the published PNG pages and review "
            "layout before claiming visual correctness.",
        ),
    ]
    if conversion_diagnostics.get("stderr"):
        warnings.append(
            warning(
                "libreoffice_diagnostics",
                "LibreOffice emitted diagnostics; inspect result.conversion.",
            )
        )
    diagnostic(
        "info",
        "libreoffice",
        "LibreOffice conversion completed.",
        **conversion_diagnostics,
    )
    return success_summary(
        "render",
        paths={"input": str(source), "output": str(output)},
        counts={
            "pdf_pages": page_count,
            "pages_rendered": len(images),
            "source_fields": source_inspection["features"]["fields"],
            "source_revisions": source_inspection["features"]["revisions"],
            "source_comments": source_inspection["features"]["comments"],
        },
        warnings=warnings,
        verification={
            "atomic_publish": True,
            "png_reopened": reopened,
            "source_unchanged_at_publish_gate": True,
            **pdf_verification,
        },
        result={
            "format": "png",
            "dpi": dpi,
            "images": images,
            "conversion": conversion_diagnostics,
        },
        libreoffice=version,
    )


def preservation_summary(warnings: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Report preservation uncertainty without claiming universal round trips."""

    uncertain_codes = {
        "revisions_present",
        "comments_present",
        "floating_drawings_present",
        "unsupported_drawings_present",
        "unsupported_parts",
        "external_relationships_allowed",
    }
    return {
        "unknown_untouched_content": "preserved_best_effort",
        "perfect_round_trip_guaranteed": False,
        "uncertainty": [item["code"] for item in warnings if item.get("code") in uncertain_codes],
    }


def success_summary(
    operation: str,
    *,
    paths: dict[str, Any],
    counts: dict[str, Any],
    warnings: list[dict[str, Any]],
    verification: dict[str, Any],
    result: dict[str, Any],
    libreoffice: str | None = None,
) -> dict[str, Any]:
    """Build the stable success envelope."""

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "operation": operation,
        "counts": counts,
        "warnings": warnings,
        "versions": package_versions(libreoffice),
        "paths": paths,
        "verification": verification,
        "preservation": preservation_summary(warnings),
        "result": result,
    }


def inspect_command(
    source: Path,
    *,
    allow_external_relationships: bool,
) -> dict[str, Any]:
    """Run inspect and wrap its result."""

    result, warnings = inspect_document(
        source,
        allow_external_relationships=allow_external_relationships,
    )
    return success_summary(
        "inspect",
        paths={"input": str(source)},
        counts={
            "paragraphs": len(result["paragraphs"]),
            "tables": len(result["tables"]),
            "sections": len(result["sections"]),
            "inline_images": len(result["images"]["inline_occurrences"]),
            **result["features"],
        },
        warnings=warnings,
        verification={"preflight": True, "reopened": True},
        result=result,
    )


def add_common_output_arguments(parser: argparse.ArgumentParser) -> None:
    """Add output and overwrite flags shared by mutating commands."""

    parser.add_argument("--output", required=True, help="Destination path.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly authorize replacement of an existing destination.",
    )


def add_external_relationship_argument(parser: argparse.ArgumentParser) -> None:
    """Add the explicit unsafe-input opt-in shared by DOCX-reading commands."""

    parser.add_argument(
        "--allow-external-relationships",
        action="store_true",
        help=(
            "Allow DOCX external relationships. This does not isolate network access by "
            "LibreOffice or other consumers."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the non-interactive command-line interface."""

    parser = JsonArgumentParser(description="Safely inspect, create, edit, and convert DOCX files.")
    parser.add_argument(
        "--job",
        help="Run a schema_version 1 JSON job instead of a direct subcommand.",
    )
    subparsers = parser.add_subparsers(dest="command")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a DOCX.")
    inspect_parser.add_argument("--input", required=True, help="Source DOCX path.")
    add_external_relationship_argument(inspect_parser)

    create_parser = subparsers.add_parser("create", help="Create a DOCX from JSON.")
    create_parser.add_argument("--spec", required=True, help="Create specification JSON.")
    create_parser.add_argument("--template", help="Optional template or letterhead DOCX.")
    add_external_relationship_argument(create_parser)
    add_common_output_arguments(create_parser)

    edit_parser = subparsers.add_parser("edit", help="Apply an atomic JSON edit job.")
    edit_parser.add_argument("--input", required=True, help="Source DOCX path.")
    edit_parser.add_argument("--spec", required=True, help="Edit specification JSON.")
    add_external_relationship_argument(edit_parser)
    add_common_output_arguments(edit_parser)

    convert_parser = subparsers.add_parser("convert", help="Convert DOCX to text or PDF.")
    convert_parser.add_argument("--input", required=True, help="Source DOCX path.")
    convert_parser.add_argument("--format", required=True, choices=("text", "pdf"))
    convert_parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_CONVERSION_TIMEOUT,
        help="LibreOffice timeout in seconds.",
    )
    add_external_relationship_argument(convert_parser)
    add_common_output_arguments(convert_parser)

    render_parser = subparsers.add_parser("render", help="Render DOCX pages to PNG images.")
    render_parser.add_argument("--input", required=True, help="Source DOCX path.")
    render_parser.add_argument(
        "--output",
        required=True,
        help="Destination directory; must not already exist.",
    )
    render_parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_RENDER_DPI,
        help=f"Raster resolution between {MIN_RENDER_DPI} and {MAX_RENDER_DPI}.",
    )
    render_parser.add_argument(
        "--pages",
        help="Comma-separated 1-based PDF page numbers; default renders every page.",
    )
    render_parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_CONVERSION_TIMEOUT,
        help="LibreOffice timeout in seconds.",
    )
    add_external_relationship_argument(render_parser)
    return parser


def execute_direct(args: argparse.Namespace) -> dict[str, Any]:
    """Execute direct subcommand arguments."""

    cwd = Path.cwd()
    if args.command == "inspect":
        return inspect_command(
            resolve_path(args.input, cwd, "input"),
            allow_external_relationships=args.allow_external_relationships,
        )
    if args.command == "create":
        spec, base_dir = load_json(args.spec)
        validate_direct_create_spec(spec)
        template = optional_path(args.template, cwd, "template")
        return create_document(
            spec,
            base_dir,
            resolve_path(args.output, cwd, "output"),
            overwrite=args.overwrite,
            allow_external_relationships=args.allow_external_relationships,
            template_override=template,
        )
    if args.command == "edit":
        spec, base_dir = load_json(args.spec)
        validate_direct_edit_spec(spec)
        return edit_document(
            resolve_path(args.input, cwd, "input"),
            spec,
            base_dir,
            resolve_path(args.output, cwd, "output"),
            overwrite=args.overwrite,
            allow_external_relationships=args.allow_external_relationships,
        )
    if args.command == "convert":
        return convert_document(
            resolve_path(args.input, cwd, "input"),
            resolve_path(args.output, cwd, "output"),
            args.format,
            overwrite=args.overwrite,
            timeout=args.timeout,
            allow_external_relationships=args.allow_external_relationships,
        )
    if args.command == "render":
        return render_document(
            resolve_path(args.input, cwd, "input"),
            resolve_path(args.output, cwd, "output"),
            dpi=args.dpi,
            pages=args.pages,
            timeout=args.timeout,
            allow_external_relationships=args.allow_external_relationships,
        )
    raise ToolError("bad_input", "Choose a subcommand or provide --job.")


def execute_job(job_path: str) -> dict[str, Any]:
    """Execute a complete versioned JSON job."""

    job, base_dir = load_json(job_path)
    operation = validate_job(job)
    overwrite = optional_boolean(job.get("overwrite"), "overwrite")
    allow_external_relationships = optional_boolean(
        job.get("allow_external_relationships"),
        "allow_external_relationships",
    )
    if operation == "inspect":
        return inspect_command(
            resolve_path(job.get("input"), base_dir, "input"),
            allow_external_relationships=allow_external_relationships,
        )
    if operation == "create":
        return create_document(
            job,
            base_dir,
            resolve_path(job.get("output"), base_dir, "output"),
            overwrite=overwrite,
            allow_external_relationships=allow_external_relationships,
            template_override=None,
        )
    if operation == "edit":
        return edit_document(
            resolve_path(job.get("input"), base_dir, "input"),
            job,
            base_dir,
            resolve_path(job.get("output"), base_dir, "output"),
            overwrite=overwrite,
            allow_external_relationships=allow_external_relationships,
        )
    if operation == "convert":
        return convert_document(
            resolve_path(job.get("input"), base_dir, "input"),
            resolve_path(job.get("output"), base_dir, "output"),
            required_string(job.get("format"), "format"),
            overwrite=overwrite,
            timeout=required_integer(
                job.get("timeout", DEFAULT_CONVERSION_TIMEOUT),
                "timeout",
                minimum=1,
                maximum=3_600,
            ),
            allow_external_relationships=allow_external_relationships,
        )
    if operation == "render":
        return render_document(
            resolve_path(job.get("input"), base_dir, "input"),
            resolve_path(job.get("output"), base_dir, "output"),
            dpi=required_integer(
                job.get("dpi", DEFAULT_RENDER_DPI),
                "dpi",
                minimum=MIN_RENDER_DPI,
                maximum=MAX_RENDER_DPI,
            ),
            pages=(required_string(job["pages"], "pages") if "pages" in job else None),
            timeout=required_integer(
                job.get("timeout", DEFAULT_CONVERSION_TIMEOUT),
                "timeout",
                minimum=1,
                maximum=3_600,
            ),
            allow_external_relationships=allow_external_relationships,
        )
    raise ToolError("bad_input", f"Unsupported job operation: {operation}")


def error_payload(error: ToolError) -> dict[str, Any]:
    """Build the stable failure envelope."""

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "error",
        "error": {
            "category": error.category,
            "message": error.message,
            "details": redact(error.details),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    try:
        parser = build_parser()
        args = parser.parse_args(argv)
        if args.job and args.command:
            raise ToolError("bad_input", "Use either --job or a subcommand, not both.")
        summary = execute_job(args.job) if args.job else execute_direct(args)
        for item in summary["warnings"]:
            details = {key: value for key, value in item.items() if key not in {"code", "message"}}
            diagnostic("warning", item["code"], item["message"], **details)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except ToolError as error:
        print(json.dumps(error_payload(error), sort_keys=True), file=sys.stderr)
        return EXIT_CODES.get(error.category, 1)
    except Exception as exc:
        error = ToolError(
            "validation_failed",
            "Unexpected internal failure.",
            details={"type": type(exc).__name__, "reason": str(exc)},
        )
        print(json.dumps(error_payload(error), sort_keys=True), file=sys.stderr)
        return EXIT_CODES[error.category]


if __name__ == "__main__":
    raise SystemExit(main())
