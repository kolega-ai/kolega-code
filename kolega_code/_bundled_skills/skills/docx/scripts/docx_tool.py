#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Safely inspect, create, edit, and convert DOCX documents."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
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
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.section import _Footer, _Header
    from docx.shared import Inches, Pt, RGBColor
    from docx.table import Table, _Cell
    from docx.text.paragraph import Paragraph
    from docx.text.run import Run
    from lxml import etree
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
    return {
        "index": index,
        "text": paragraph.text,
        "style": style,
        "heading_level": heading,
        "list": list_info,
        "alignment": paragraph.alignment.name if paragraph.alignment is not None else None,
        "runs": [
            {"index": run_index, "text": run.text, "format": run_format(run)}
            for run_index, run in enumerate(paragraph.runs)
        ],
    }


def table_record(table: Table, index: int) -> dict[str, Any]:
    """Serialize a table and its cell paragraphs."""

    rows: list[list[dict[str, Any]]] = []
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
    return {
        "index": index,
        "style": table.style.name if table.style is not None else None,
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

    return {
        "index": index,
        "start_type": section.start_type.name if section.start_type is not None else None,
        "orientation": section.orientation.name if section.orientation is not None else None,
        "page_width_inches": length_inches(section.page_width),
        "page_height_inches": length_inches(section.page_height),
        "top_margin_inches": length_inches(section.top_margin),
        "right_margin_inches": length_inches(section.right_margin),
        "bottom_margin_inches": length_inches(section.bottom_margin),
        "left_margin_inches": length_inches(section.left_margin),
        "header_distance_inches": length_inches(section.header_distance),
        "footer_distance_inches": length_inches(section.footer_distance),
        "different_first_page": bool(section.different_first_page_header_footer),
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

    occurrences: list[dict[str, Any]] = []
    for part in document.part.package.parts:
        element = getattr(part, "element", None)
        if element is None:
            continue
        for inline in element.iter(f"{{{WP_NS}}}inline"):
            blip = next(inline.iter(f"{{{A_NS}}}blip"), None)
            if blip is None or blip.get(f"{{{REL_NS}}}embed") is None:
                continue
            extent = next(inline.iter(f"{{{WP_NS}}}extent"), None)
            occurrences.append(
                {
                    "container_part": str(part.partname).lstrip("/"),
                    "relationship_id": (
                        blip.get(f"{{{REL_NS}}}embed") if blip is not None else None
                    ),
                    "width_emu": int(extent.get("cx")) if extent is not None else None,
                    "height_emu": int(extent.get("cy")) if extent is not None else None,
                }
            )
    return {"media_parts": media, "inline_occurrences": occurrences}


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
    for block_index, (kind, block) in enumerate(iter_body_blocks(document)):
        if kind == "paragraph":
            record = paragraph_record(block, len(paragraphs))
            paragraphs.append(record)
            ordered.append({"block_index": block_index, "type": kind, "value": record})
        else:
            record = table_record(block, len(tables))
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

    result = {
        "ordered_blocks": ordered,
        "paragraphs": paragraphs,
        "tables": tables,
        "sections": [
            section_record(section, index) for index, section in enumerate(document.sections)
        ],
        "headers": headers,
        "footers": footers,
        "metadata": core_properties_record(document),
        "images": inspect_media(preflight.path, document),
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


def validate_cell_value(value: Any, field: str) -> None:
    """Validate a table cell as plain text or a paragraph-format object."""

    if isinstance(value, str):
        return
    if not isinstance(value, dict):
        raise ToolError(
            "bad_input",
            f"{field} must be a string or paragraph object.",
        )
    allow_keys(value, {"text", "runs", "style", "alignment"}, field)
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


def validate_block(block: Any, field: str, *, allow_sections: bool) -> None:
    """Validate one content block against its type-specific schema."""

    if not isinstance(block, dict):
        raise ToolError("bad_input", f"{field} must be an object.")
    block_type = block.get("type", "paragraph")
    if not isinstance(block_type, str):
        raise ToolError("bad_input", f"{field}.type must be a string.")
    paragraph_keys = {"type", "text", "runs", "style", "alignment"}
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
        return
    if block_type == "field":
        require_keys(block, {"field"}, field)
        field_name = required_string(block["field"], f"{field}.field")
        if field_name not in {"page_number", "toc"}:
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
        if field_name == "toc":
            allowed_field_keys.add("instruction")
        allow_keys(
            block,
            allowed_field_keys,
            field,
        )
        for name in ("prefix", "suffix", "placeholder", "instruction"):
            if name in block:
                string_value(block[name], f"{field}.{name}")
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
        allow_keys(block, {"type", "rows", "style", "header_rows"}, field)
        validate_table_block(block, field)
        return
    if block_type == "section_break":
        if not allow_sections:
            raise ToolError(
                "bad_input",
                f"{field} cannot contain a section break.",
            )
        section_keys = {
            "type",
            "start",
            "orientation",
            "top_margin_inches",
            "right_margin_inches",
            "bottom_margin_inches",
            "left_margin_inches",
            "header_distance_inches",
            "footer_distance_inches",
        }
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
        for name in section_keys.difference({"type", "start", "orientation"}):
            if name in block:
                required_number(block[name], f"{field}.{name}", minimum=0)
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


def validate_create_content(content: Any, field: str = "content") -> None:
    """Validate the complete create-content object."""

    if not isinstance(content, dict):
        raise ToolError("bad_input", f"{field} must be an object.")
    allow_keys(
        content, {"template", "letterhead", "metadata", "blocks", "headers", "footers"}, field
    )
    if "template" in content and "letterhead" in content:
        raise ToolError("bad_input", f"{field} must specify only one of template or letterhead.")
    for name in ("template", "letterhead"):
        if name in content:
            required_string(content[name], f"{field}.{name}")
    if "metadata" in content:
        validate_metadata(content["metadata"], f"{field}.metadata")
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
        allow_keys(table, {"type", "rows", "style", "header_rows"}, f"{field}.table")
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
            {"type", "table_index", "row", "column", "value", "style"},
            field,
        )
        require_keys(operation, {"table_index", "row", "column", "value"}, field)
        for name in ("table_index", "row", "column"):
            required_integer(operation[name], f"{field}.{name}", minimum=0)
        validate_cell_value(operation["value"], f"{field}.value")
        if "style" in operation:
            required_string(operation["style"], f"{field}.style")
        return
    if operation_type in {"insert_image", "replace_image"}:
        index_name = "paragraph_index" if operation_type == "insert_image" else "inline_image_index"
        allowed = {"type", "path", "width_inches", "height_inches", index_name}
        allow_keys(operation, allowed, field)
        required = {"path", "inline_image_index"} if operation_type == "replace_image" else {"path"}
        require_keys(operation, required, field)
        required_string(operation["path"], f"{field}.path")
        if index_name in operation:
            required_integer(operation[index_name], f"{field}.{index_name}", minimum=0)
        for name in ("width_inches", "height_inches"):
            if name in operation:
                required_number(operation[name], f"{field}.{name}", minimum=0.01)
        return
    if operation_type == "set_metadata":
        allow_keys(operation, {"type", "metadata"}, field)
        require_keys(operation, {"metadata"}, field)
        validate_metadata(operation["metadata"], f"{field}.metadata")
        return
    if operation_type in {"set_header", "set_footer"}:
        allow_keys(operation, {"type", "section_index", "kind", "mode", "blocks"}, field)
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
        validate_blocks(operation["blocks"], f"{field}.blocks", allow_sections=False)
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
    """Append PAGE or TOC complex-field markup."""

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
        raise ToolError("unsupported_operation", f"Unsupported field type: {field_name}")

    begin_run = OxmlElement("w:r")
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    begin.set(qn("w:dirty"), "true")
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
        paragraph.add_run().add_picture(str(image), width=width, height=height)
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
    return paragraph


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
    header_rows = required_integer(block.get("header_rows", 0), "header_rows", minimum=0)
    for row in table.rows[:header_rows]:
        tr_pr = row._tr.get_or_add_trPr()
        repeat = OxmlElement("w:tblHeader")
        repeat.set(qn("w:val"), "true")
        tr_pr.append(repeat)


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
        table = container.add_table(rows=len(rows), cols=columns, width=Inches(6))
    else:
        table = container.add_table(rows=len(rows), cols=columns)
    fill_table(table, block)
    return table


def configure_section(section: Any, spec: dict[str, Any]) -> None:
    """Apply supported section page settings."""

    if "orientation" in spec:
        orientation = spec["orientation"]
        if orientation not in {"portrait", "landscape"}:
            raise ToolError("bad_input", "section.orientation must be portrait or landscape.")
        desired = WD_ORIENT.LANDSCAPE if orientation == "landscape" else WD_ORIENT.PORTRAIT
        if section.orientation != desired:
            section.orientation = desired
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

    counts = apply_blocks(document, content.get("blocks", []), base_dir, allow_sections=True)
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
    """Update one table cell and optionally its table style."""

    table = table_at(document, operation.get("table_index"))
    if "style" in operation:
        assign_named_style(
            table,
            required_string(operation["style"], "update_table.style"),
            "update_table.style",
        )
    row_index = required_integer(operation.get("row"), "update_table.row", minimum=0)
    column_index = required_integer(operation.get("column"), "update_table.column", minimum=0)
    if row_index >= len(table.rows) or column_index >= len(table.rows[row_index].cells):
        raise ToolError("bad_input", "Table cell index is out of range.")
    set_cell_value(table.cell(row_index, column_index), operation.get("value", ""))


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
    paragraph.add_run().add_picture(str(image), width=width, height=height)


def replace_image(
    document: DocumentObject,
    operation: dict[str, Any],
    base_dir: Path,
) -> None:
    """Replace one body inline image relationship without flattening its drawing."""

    index = required_integer(operation.get("inline_image_index"), "inline_image_index", minimum=0)
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


def convert_pdf_bytes(source: Path, timeout: int) -> tuple[bytes, dict[str, Any], str]:
    """Run headless LibreOffice with an isolated temporary profile."""

    executable = shutil.which("soffice")
    if executable is None:
        raise ToolError(
            "missing_dependency",
            "LibreOffice soffice is required for DOCX-to-PDF conversion.",
        )
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
