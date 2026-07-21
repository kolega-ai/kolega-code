#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Safely inspect, create, edit, and convert PPTX presentations."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
import warnings as python_warnings
import zipfile
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Any, Never
from urllib.parse import unquote, unquote_plus, urlsplit, urlunsplit

try:
    import lxml.etree as etree
    import pptx
    import pypdf
    from PIL import Image
    from pptx import Presentation
    from pptx.chart.data import CategoryChartData, XyChartData
    from pptx.enum.chart import XL_CHART_TYPE
    from pptx.enum.shapes import MSO_SHAPE_TYPE, PP_PLACEHOLDER
    from pptx.enum.text import PP_ALIGN
    from pptx.util import Inches, Pt
except ModuleNotFoundError as exc:
    print(
        json.dumps(
            {
                "schema_version": 1,
                "success": False,
                "error": {
                    "category": "missing_dependency",
                    "message": f"missing Python dependency: {exc.name}",
                    "details": {
                        "install": "python -m pip install -r requirements.txt",
                    },
                },
            },
            separators=(",", ":"),
        ),
        file=sys.stderr,
    )
    raise SystemExit(4) from None


SCHEMA_VERSION = 1
SUPPORTED_PPTX_MAJOR = 1
MAX_SOURCE_BYTES = 100 * 1024 * 1024
MAX_MEMBERS = 5_000
MAX_EXPANDED_BYTES = 512 * 1024 * 1024
MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_COMPRESSION_RATIO = 200
MAX_IMAGE_BYTES = 50 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000
MAX_IMAGE_DIMENSION = 20_000
MAX_DECODED_IMAGE_BYTES = 200 * 1024 * 1024
MAX_SLIDES = 500
MAX_SHAPES = 5_000
MAX_TEXT_CHARS = 2_000_000
MAX_TABLE_ROWS = 200
MAX_TABLE_COLUMNS = 200
MAX_CHART_POINTS = 10_000
MAX_GEOMETRY_INCHES = 100.0

CONTENT_TYPES = "[Content_Types].xml"
PRESENTATION_PART = "ppt/presentation.xml"
PRESENTATION_RELS = "ppt/_rels/presentation.xml.rels"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CONTENT_TYPE_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NOTES_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide"
SLIDE_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide"
OFFICE_DOCUMENT_REL = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
)
PRESENTATION_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"
)
SLIDE_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.presentationml.slide+xml"
IMAGE_CONTENT_TYPES = {
    "BMP": "image/bmp",
    "GIF": "image/gif",
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "TIFF": "image/tiff",
    "WMF": "image/x-wmf",
}
SENSITIVE_QUERY_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "auth",
    "authorization",
    "client_secret",
    "code",
    "credential",
    "credentials",
    "id_token",
    "key",
    "password",
    "passwd",
    "refresh_token",
    "secret",
    "session",
    "session_id",
    "sessionid",
    "sig",
    "signature",
    "token",
}
RELATIONSHIP_ID_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9._-]*$")
SENSITIVE_QUERY_KEYS_COMPACT = {key.replace("_", "") for key in SENSITIVE_QUERY_KEYS}
CHART_TYPES = {
    "column": XL_CHART_TYPE.COLUMN_CLUSTERED,
    "bar": XL_CHART_TYPE.BAR_CLUSTERED,
    "line": XL_CHART_TYPE.LINE,
    "line_markers": XL_CHART_TYPE.LINE_MARKERS,
    "pie": XL_CHART_TYPE.PIE,
    "doughnut": XL_CHART_TYPE.DOUGHNUT,
    "area": XL_CHART_TYPE.AREA,
    "scatter": XL_CHART_TYPE.XY_SCATTER,
    "scatter_lines": XL_CHART_TYPE.XY_SCATTER_LINES,
}
SCATTER_TYPES = {"scatter", "scatter_lines"}
ALIGNMENTS = {
    "left": PP_ALIGN.LEFT,
    "center": PP_ALIGN.CENTER,
    "right": PP_ALIGN.RIGHT,
    "justify": PP_ALIGN.JUSTIFY,
}
EXIT_CODES = {
    "bad_input": 2,
    "unsupported_operation": 3,
    "missing_dependency": 4,
    "ambiguous_edit": 5,
    "resource_limit": 6,
    "external_precondition": 7,
    "post_write_validation": 8,
    "external_tool_failure": 9,
    "internal_error": 10,
}


class ToolError(Exception):
    """A stable, user-facing CLI failure."""

    def __init__(
        self,
        category: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.message = message
        self.details = details or {}
        self.exit_code = EXIT_CODES[category]


class JsonArgumentParser(argparse.ArgumentParser):
    """Raise structured argument errors instead of printing usage."""

    def error(self, message: str) -> Never:
        raise ToolError("bad_input", message)


@dataclass(frozen=True)
class PackageReport:
    path: Path
    file_size: int
    member_count: int
    expanded_size: int
    external_relationships: int
    slide_ids: tuple[int, ...]
    slide_parts: tuple[str, ...]
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path.resolve()),
            "file_size": self.file_size,
            "member_count": self.member_count,
            "expanded_size": self.expanded_size,
            "external_relationships": self.external_relationships,
            "slide_ids": list(self.slide_ids),
            "slide_parts": list(self.slide_parts),
            "internal_relationship_targets_exist": True,
            "owner_relationship_references_resolved": True,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class PptxSourceSnapshot:
    original_path: Path
    snapshot_path: Path
    sha256: str
    report: PackageReport


@dataclass(frozen=True)
class ImageSnapshot:
    data: bytes
    content_type: str
    width: int
    height: int
    sha256: str


@dataclass
class RunMatch:
    paragraph: Any
    first_run: int
    first_offset: int
    last_run: int
    last_offset: int
    cross_run: bool
    hyperlink_boundary: bool


@dataclass
class FrameMatch:
    text_frame: Any
    start: int
    end: int


def _json_dump(value: dict[str, Any], stream: Any = sys.stdout) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")), file=stream)


def _versions() -> dict[str, str]:
    versions = {
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "python-pptx": pptx.__version__,
        "lxml": ".".join(str(part) for part in etree.LXML_VERSION),
        "pypdf": pypdf.__version__,
    }
    for distribution, key in (
        ("Pillow", "Pillow"),
        ("XlsxWriter", "XlsxWriter"),
        ("typing_extensions", "typing-extensions"),
    ):
        try:
            versions[key] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[key] = "not-installed"
    return versions


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ToolError("bad_input", f"{label} must be a JSON object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ToolError("bad_input", f"{label} must be a JSON array")
    return value


def _require_string(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise ToolError("bad_input", f"{label} must be {qualifier}")
    return value


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ToolError(
            "bad_input",
            f"{label} contains unsupported key(s)",
            {"keys": unknown},
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hash_matches(path: Path, expected: str | None) -> bool:
    if expected is None:
        return False
    try:
        return _sha256_file(path) == expected
    except OSError:
        return False


def _safe_xml(data: bytes, member: str) -> etree._Element:
    upper = data.upper()
    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
        raise ToolError(
            "bad_input",
            "OOXML package contains a forbidden DTD or entity declaration",
            {"member": member},
        )
    parser = etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        load_dtd=False,
        dtd_validation=False,
        huge_tree=False,
        recover=False,
    )
    try:
        return etree.fromstring(data, parser=parser)
    except etree.XMLSyntaxError as exc:
        raise ToolError(
            "bad_input",
            "OOXML package contains malformed XML",
            {"member": member, "reason": str(exc)},
        ) from exc


def _rels_owner(rels_name: str) -> str:
    if rels_name == "_rels/.rels":
        return ""
    path = PurePosixPath(rels_name)
    if path.parent.name != "_rels" or not path.name.endswith(".rels"):
        raise ToolError("bad_input", "invalid relationship part name", {"member": rels_name})
    owner = str(path.parent.parent / path.name.removesuffix(".rels"))
    expected = str(PurePosixPath(owner).parent / "_rels" / f"{PurePosixPath(owner).name}.rels")
    if expected != rels_name:
        raise ToolError("bad_input", "noncanonical relationship part name", {"member": rels_name})
    return owner


def _resolve_target(owner: str, target: str) -> str:
    target_path = unquote(urlsplit(target).path)
    if not target_path:
        raise ToolError("bad_input", "internal relationship has an empty target")
    if target_path.startswith("/"):
        resolved = posixpath.normpath(target_path.lstrip("/"))
    else:
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(owner), target_path))
    if resolved == ".." or resolved.startswith("../") or resolved.startswith("/"):
        raise ToolError(
            "bad_input",
            "internal relationship escapes the OOXML package",
            {"owner": owner, "target": target},
        )
    return resolved


def _feature_warnings(names: set[str], xml_roots: dict[str, etree._Element]) -> list[str]:
    warnings: list[str] = []
    tests = (
        ("_xmlsignatures/", "digital signatures may be invalidated by rewriting"),
        ("ppt/comments/", "comments are not editable and may not round-trip"),
        ("ppt/threadedcomments/", "threaded comments are not supported"),
        ("ppt/diagrams/", "SmartArt/diagram semantics may not round-trip"),
        ("ppt/embeddings/", "embedded objects are not editable"),
        ("ppt/activex/", "ActiveX content is unsupported"),
        ("customxml/", "custom XML preservation is not guaranteed"),
    )
    lower_names = {name.lower() for name in names}
    for prefix, warning in tests:
        if any(name.startswith(prefix) for name in lower_names):
            warnings.append(warning)
    media = [
        name
        for name in lower_names
        if name.startswith("ppt/media/")
        and not name.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".wmf"))
    ]
    if media:
        warnings.append("audio, video, or unsupported media may not round-trip")
    has_transition = False
    has_timing = False
    for name, root in xml_roots.items():
        if not name.startswith("ppt/slides/"):
            continue
        has_transition = has_transition or bool(
            root.xpath(".//p:transition", namespaces={"p": P_NS})
        )
        has_timing = has_timing or bool(root.xpath(".//p:timing", namespaces={"p": P_NS}))
    if has_transition:
        warnings.append("slide transitions may not round-trip")
    if has_timing:
        warnings.append("animations/timing may not round-trip")
    return warnings


def preflight_pptx(path: Path) -> PackageReport:
    path = path.expanduser()
    if path.suffix.lower() != ".pptx":
        raise ToolError(
            "bad_input",
            "source must use the .pptx extension; macro-enabled presentations are unsupported",
            {"path": str(path)},
        )
    if not path.exists():
        raise ToolError("bad_input", "source does not exist", {"path": str(path)})
    if not path.is_file():
        raise ToolError("bad_input", "source is not a regular file", {"path": str(path)})
    file_size = path.stat().st_size
    if file_size > MAX_SOURCE_BYTES:
        raise ToolError(
            "resource_limit",
            "PPTX source exceeds the compressed size limit",
            {"bytes": file_size, "limit": MAX_SOURCE_BYTES},
        )
    with path.open("rb") as source:
        if source.read(4) not in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"):
            raise ToolError("bad_input", "source does not have a ZIP/PPTX signature")

    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ToolError("bad_input", "source is not a readable PPTX ZIP package") from exc

    with archive:
        infos = archive.infolist()
        if len(infos) > MAX_MEMBERS:
            raise ToolError(
                "resource_limit",
                "PPTX package has too many members",
                {"members": len(infos), "limit": MAX_MEMBERS},
            )
        names: set[str] = set()
        normalized_names: set[str] = set()
        expanded = 0
        xml_roots: dict[str, etree._Element] = {}
        for info in infos:
            name = info.filename
            pure = PurePosixPath(name)
            normalized = posixpath.normpath(name)
            if (
                not name
                or "\\" in name
                or name.startswith("/")
                or normalized == ".."
                or normalized.startswith("../")
                or any(part in ("", ".", "..") for part in pure.parts)
            ):
                raise ToolError("bad_input", "unsafe ZIP member path", {"member": name})
            folded = normalized.casefold()
            if name in names or folded in normalized_names:
                raise ToolError("bad_input", "duplicate ZIP member path", {"member": name})
            names.add(name)
            normalized_names.add(folded)
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise ToolError("bad_input", "symbolic-link ZIP members are unsupported")
            if info.flag_bits & 0x1:
                raise ToolError("bad_input", "encrypted PPTX ZIP members are unsupported")
            if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise ToolError(
                    "bad_input",
                    "unsupported ZIP compression method",
                    {"member": name, "method": info.compress_type},
                )
            if info.file_size > MAX_MEMBER_BYTES:
                raise ToolError(
                    "resource_limit",
                    "PPTX package member exceeds the expanded size limit",
                    {"member": name, "bytes": info.file_size, "limit": MAX_MEMBER_BYTES},
                )
            expanded += info.file_size
            if expanded > MAX_EXPANDED_BYTES:
                raise ToolError(
                    "resource_limit",
                    "PPTX package exceeds the expanded size limit",
                    {"bytes": expanded, "limit": MAX_EXPANDED_BYTES},
                )
            if (
                info.file_size > 1024 * 1024
                and info.compress_size > 0
                and info.file_size / info.compress_size > MAX_COMPRESSION_RATIO
            ):
                raise ToolError(
                    "resource_limit",
                    "PPTX package member has a suspicious compression ratio",
                    {"member": name},
                )
            if name.lower().endswith((".xml", ".rels", ".vml")):
                try:
                    xml_roots[name] = _safe_xml(archive.read(info), name)
                except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                    raise ToolError(
                        "bad_input",
                        "could not read an OOXML package member",
                        {"member": name},
                    ) from exc

        required = {CONTENT_TYPES, "_rels/.rels", PRESENTATION_PART, PRESENTATION_RELS}
        missing = sorted(required - names)
        if missing:
            raise ToolError(
                "bad_input",
                "PPTX package is missing required parts",
                {"members": missing},
            )
        lower_names = {name.lower() for name in names}
        if any(
            name.endswith("vbaproject.bin")
            or name.startswith("ppt/activex/")
            and name.endswith(".bin")
            for name in lower_names
        ):
            raise ToolError("unsupported_operation", "macro or ActiveX-bearing PPTX is unsupported")
        content_type_root = xml_roots[CONTENT_TYPES]
        overrides: dict[str, str] = {}
        for override in content_type_root.findall(f"{{{CONTENT_TYPE_NS}}}Override"):
            part_name = (override.get("PartName") or "").lstrip("/")
            content_type = override.get("ContentType") or ""
            if not part_name or not content_type or part_name in overrides:
                raise ToolError("bad_input", "malformed or duplicate content-type override")
            overrides[part_name] = content_type
        all_content_types = [
            node.get("ContentType") or ""
            for node in content_type_root.findall(f"{{{CONTENT_TYPE_NS}}}*")
        ]
        if any(
            token in content_type.lower()
            for content_type in all_content_types
            for token in ("macroenabled", "vbaproject", "activex")
        ):
            raise ToolError(
                "unsupported_operation",
                "macro or ActiveX content types are unsupported",
            )
        if overrides.get(PRESENTATION_PART) != PRESENTATION_CONTENT_TYPE:
            raise ToolError(
                "unsupported_operation",
                "package is not a standard non-macro PPTX presentation",
                {"content_type": overrides.get(PRESENTATION_PART)},
            )

        external_relationships = 0
        rel_maps: dict[str, dict[str, tuple[str, str]]] = {}
        for name, root in xml_roots.items():
            if not name.endswith(".rels"):
                continue
            owner = _rels_owner(name)
            if owner and owner not in names:
                raise ToolError(
                    "bad_input",
                    "relationship part owner is missing",
                    {"member": name, "owner": owner},
                )
            if owner in rel_maps:
                raise ToolError(
                    "bad_input",
                    "multiple relationship parts map to one owner",
                    {"member": name, "owner": owner},
                )
            if root.tag != f"{{{REL_NS}}}Relationships":
                raise ToolError(
                    "bad_input",
                    "relationship part has an invalid root element",
                    {"member": name},
                )
            owner_rels: dict[str, tuple[str, str]] = {}
            for rel in root:
                if rel.tag != f"{{{REL_NS}}}Relationship":
                    raise ToolError(
                        "bad_input",
                        "relationship part contains an invalid child element",
                        {"member": name},
                    )
                rel_id = rel.get("Id")
                target = rel.get("Target")
                rel_type = rel.get("Type") or ""
                target_mode = rel.get("TargetMode")
                if "vbaproject" in rel_type.lower() or "activex" in rel_type.lower():
                    raise ToolError(
                        "unsupported_operation",
                        "macro or ActiveX relationships are unsupported",
                    )
                if (
                    not rel_id
                    or RELATIONSHIP_ID_PATTERN.fullmatch(rel_id) is None
                    or not rel_type
                    or not target
                    or target_mode not in {None, "Internal", "External"}
                    or rel_id in owner_rels
                ):
                    raise ToolError(
                        "bad_input",
                        "malformed or duplicate OOXML relationship",
                        {"member": name, "relationship": rel_id},
                    )
                if target_mode == "External":
                    external_relationships += 1
                    owner_rels[rel_id] = (target, rel_type)
                    continue
                target_parts = urlsplit(target)
                if target_parts.scheme or target_parts.netloc:
                    raise ToolError(
                        "bad_input",
                        "internal relationship target is not a package-relative URI",
                        {"owner": owner, "relationship": rel_id},
                    )
                resolved = _resolve_target(owner, target)
                if resolved not in names:
                    raise ToolError(
                        "bad_input",
                        "OOXML relationship target is missing",
                        {"owner": owner, "relationship": rel_id, "target": resolved},
                    )
                owner_rels[rel_id] = (resolved, rel_type)
            rel_maps[owner] = owner_rels

        for owner, root in xml_roots.items():
            if owner.endswith(".rels") or owner == CONTENT_TYPES:
                continue
            owner_rels = rel_maps.get(owner, {})
            for element in root.iter():
                for attribute_name, rel_id in element.attrib.items():
                    if not attribute_name.startswith(f"{{{R_NS}}}"):
                        continue
                    if rel_id not in owner_rels:
                        raise ToolError(
                            "bad_input",
                            "owner XML references a missing relationship ID",
                            {
                                "owner": owner,
                                "attribute": etree.QName(attribute_name).localname,
                                "relationship": rel_id,
                            },
                        )

        root_office_documents = [
            target
            for target, rel_type in rel_maps.get("", {}).values()
            if rel_type == OFFICE_DOCUMENT_REL
        ]
        if root_office_documents != [PRESENTATION_PART]:
            raise ToolError(
                "bad_input",
                "root relationships must identify one PresentationML office document",
                {"targets": root_office_documents},
            )

        presentation = xml_roots[PRESENTATION_PART]
        slide_ids: list[int] = []
        slide_parts: list[str] = []
        presentation_rels = rel_maps.get(PRESENTATION_PART, {})
        for slide_id_node in presentation.xpath(
            "./p:sldIdLst/p:sldId",
            namespaces={"p": P_NS, "r": R_NS},
        ):
            raw_id = slide_id_node.get("id")
            rel_id = slide_id_node.get(f"{{{R_NS}}}id")
            try:
                slide_id = int(raw_id or "")
            except ValueError as exc:
                raise ToolError("bad_input", "presentation has an invalid slide ID") from exc
            if slide_id in slide_ids:
                raise ToolError("bad_input", "presentation has duplicate slide IDs")
            relationship = presentation_rels.get(rel_id or "")
            if relationship is None or relationship[1] != SLIDE_REL:
                raise ToolError(
                    "bad_input",
                    "slide ID does not resolve through a slide relationship",
                    {"slide_id": slide_id, "relationship": rel_id},
                )
            slide_ids.append(slide_id)
            slide_parts.append(relationship[0])
        if len(slide_ids) > MAX_SLIDES:
            raise ToolError(
                "resource_limit",
                "presentation exceeds the slide limit",
                {"slides": len(slide_ids), "limit": MAX_SLIDES},
            )

        package_slide_parts = {
            name
            for name in names
            if name.startswith("ppt/slides/") and name.endswith(".xml") and "/_rels/" not in name
        }
        if len(slide_parts) != len(set(slide_parts)) or set(slide_parts) != package_slide_parts:
            raise ToolError(
                "bad_input",
                "presentation contains unreachable or multiply referenced slide parts",
                {
                    "reachable": sorted(slide_parts),
                    "packaged": sorted(package_slide_parts),
                },
            )
        invalid_slide_types = {
            part: overrides.get(part)
            for part in slide_parts
            if overrides.get(part) != SLIDE_CONTENT_TYPE
        }
        if invalid_slide_types:
            raise ToolError(
                "bad_input",
                "slide parts do not declare the standard PresentationML content type",
                {"parts": invalid_slide_types},
            )
        warnings = _feature_warnings(names, xml_roots)
        return PackageReport(
            path=path,
            file_size=file_size,
            member_count=len(infos),
            expanded_size=expanded,
            external_relationships=external_relationships,
            slide_ids=tuple(slide_ids),
            slide_parts=tuple(slide_parts),
            warnings=tuple(warnings),
        )


@contextmanager
def _pptx_source_snapshot(path: Path) -> Generator[PptxSourceSnapshot, None, None]:
    original = path.expanduser()
    if original.suffix.lower() != ".pptx":
        raise ToolError(
            "bad_input",
            "source must use the .pptx extension; macro-enabled presentations are unsupported",
            {"path": str(original)},
        )
    try:
        original = original.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ToolError("bad_input", "source does not exist", {"path": str(original)}) from exc
    if not original.is_file():
        raise ToolError("bad_input", "source is not a regular file", {"path": str(original)})
    try:
        with original.open("rb") as source:
            data = source.read(MAX_SOURCE_BYTES + 1)
    except OSError as exc:
        raise ToolError("bad_input", "source is not readable", {"path": str(original)}) from exc
    if len(data) > MAX_SOURCE_BYTES:
        raise ToolError(
            "resource_limit",
            "PPTX source exceeds the compressed size limit",
            {"bytes": len(data), "limit": MAX_SOURCE_BYTES},
        )
    digest = hashlib.sha256(data).hexdigest()
    with tempfile.TemporaryDirectory(prefix="pptx-source-snapshot-") as temp_dir:
        snapshot_path = Path(temp_dir) / original.name
        try:
            snapshot_path.write_bytes(data)
            snapshot_path.chmod(0o400)
        except OSError as exc:
            raise ToolError(
                "internal_error", "could not create an immutable source snapshot"
            ) from exc
        report = replace(preflight_pptx(snapshot_path), path=original)
        yield PptxSourceSnapshot(original, snapshot_path, digest, report)


class OoxmlAdapter:
    """Contain private python-pptx/OOXML access required by unsupported public operations."""

    @staticmethod
    def require_supported_version() -> None:
        try:
            major = int(pptx.__version__.partition(".")[0])
        except ValueError:
            major = -1
        if major != SUPPORTED_PPTX_MAJOR:
            raise ToolError(
                "missing_dependency",
                "private OOXML adapter requires a supported python-pptx version",
                {"supported": ">=1,<2", "actual": pptx.__version__},
            )

    @classmethod
    def remove_slide(cls, presentation: Any, index: int) -> None:
        cls.require_supported_version()
        slide_id_list = presentation.slides._sldIdLst
        slide_id = slide_id_list[index]
        presentation.part.drop_rel(slide_id.rId)
        slide_id_list.remove(slide_id)

    @classmethod
    def reorder_slides(cls, presentation: Any, ordered_ids: list[int]) -> None:
        cls.require_supported_version()
        slide_id_list = presentation.slides._sldIdLst
        by_id = {int(element.id): element for element in list(slide_id_list)}
        for element in list(slide_id_list):
            slide_id_list.remove(element)
        for slide_id in ordered_ids:
            slide_id_list.append(by_id[slide_id])

    @classmethod
    def paragraph_has_field(cls, paragraph: Any) -> bool:
        cls.require_supported_version()
        return bool(paragraph._p.xpath("./a:fld"))

    @classmethod
    def replace_picture_relationship(cls, picture: Any, blob: bytes) -> None:
        cls.require_supported_version()
        old_relationship_id = picture._element.blip_rId
        _, new_relationship_id = picture.part.get_or_add_image_part(io.BytesIO(blob))
        picture._element.blipFill.blip.set(f"{{{R_NS}}}embed", new_relationship_id)
        used_relationship_ids = {
            blip.get(f"{{{R_NS}}}embed")
            for blip in picture.part._element.xpath(".//a:blip")
            if blip.get(f"{{{R_NS}}}embed") is not None
        }
        if (
            old_relationship_id != new_relationship_id
            and old_relationship_id not in used_relationship_ids
        ):
            picture.part.drop_rel(old_relationship_id)


def _presentation_slide_ids(presentation: Any) -> list[int]:
    return [int(slide.slide_id) for slide in presentation.slides]


def _picture_hashes(presentation: Any) -> dict[tuple[int, int], str]:
    return {
        (int(slide.slide_id), int(shape.shape_id)): hashlib.sha256(shape.image.blob).hexdigest()
        for slide in presentation.slides
        for shape in slide.shapes
        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE
    }


def _enforce_presentation_limits(presentation: Any) -> None:
    shape_count = 0
    text_chars = 0
    for slide in presentation.slides:
        shape_count += len(slide.shapes)
        for shape in slide.shapes:
            if shape.has_text_frame:
                text_chars += len(shape.text_frame.text)
    if shape_count > MAX_SHAPES:
        raise ToolError(
            "resource_limit",
            "presentation exceeds the shape limit",
            {"shapes": shape_count, "limit": MAX_SHAPES},
        )
    if text_chars > MAX_TEXT_CHARS:
        raise ToolError(
            "resource_limit",
            "presentation exceeds the text limit",
            {"characters": text_chars, "limit": MAX_TEXT_CHARS},
        )


def _open_presentation(path: Path, category: str = "bad_input") -> Any:
    try:
        return Presentation(str(path))
    except Exception as exc:
        raise ToolError(category, "python-pptx could not reopen the presentation") from exc


def _validate_pptx_output(path: Path, expected_ids: list[int]) -> PackageReport:
    try:
        report = preflight_pptx(path)
        reopened = _open_presentation(path, "post_write_validation")
    except ToolError as exc:
        if exc.category in {"bad_input", "resource_limit", "unsupported_operation"}:
            raise ToolError(
                "post_write_validation",
                "written PPTX failed safe package validation",
                {"reason": exc.message, **exc.details},
            ) from exc
        raise
    actual_ids = _presentation_slide_ids(reopened)
    if actual_ids != expected_ids or list(report.slide_ids) != expected_ids:
        raise ToolError(
            "post_write_validation",
            "written PPTX slide ID/order verification failed",
            {"expected": expected_ids, "actual": actual_ids},
        )
    return report


def _checkpoint_graph_mutation(
    presentation: Any,
    checkpoint_dir: Path,
    ordinal: int,
    before_ids: list[int],
    expected_ids: list[int],
) -> tuple[Any, dict[str, Any]]:
    if len(before_ids) != len(set(before_ids)) or len(expected_ids) != len(set(expected_ids)):
        raise ToolError("post_write_validation", "slide graph contains duplicate slide IDs")
    retained = [slide_id for slide_id in before_ids if slide_id in expected_ids]
    if set(retained) != set(before_ids) & set(expected_ids):
        raise ToolError("post_write_validation", "retained slide ID calculation failed")
    path = checkpoint_dir / f"graph-{ordinal}.pptx"
    try:
        presentation.save(str(path))
    except Exception as exc:
        raise ToolError(
            "post_write_validation",
            "could not save a slide-graph checkpoint",
        ) from exc
    report = _validate_pptx_output(path, expected_ids)
    reopened = _open_presentation(path, "post_write_validation")
    reopened_ids = _presentation_slide_ids(reopened)
    if [slide_id for slide_id in reopened_ids if slide_id in retained] != [
        slide_id for slide_id in expected_ids if slide_id in retained
    ]:
        raise ToolError(
            "post_write_validation",
            "retained slide IDs changed during graph mutation",
        )
    return reopened, {
        "ordinal": ordinal,
        "before_ids": before_ids,
        "after_ids": expected_ids,
        "package_reopen": True,
        "internal_relationship_targets_exist": True,
        "owner_relationship_references_resolved": True,
        "slide_relationship_graph_verified": True,
        "member_count": report.member_count,
    }


def _geometry(shape: Any) -> dict[str, Any]:
    values = {
        "x": int(shape.left),
        "y": int(shape.top),
        "width": int(shape.width),
        "height": int(shape.height),
    }
    return {
        "emu": values,
        "inches": {key: round(value / 914400, 4) for key, value in values.items()},
    }


def _enum_name(value: Any) -> str | None:
    if value is None:
        return None
    return getattr(value, "name", str(value))


def _font_info(font: Any) -> dict[str, Any]:
    color: str | None = None
    try:
        rgb = font.color.rgb
        color = str(rgb) if rgb is not None else None
    except (AttributeError, TypeError, ValueError):
        color = None
    return {
        "name": font.name,
        "size_points": round(font.size.pt, 2) if font.size is not None else None,
        "bold": font.bold,
        "italic": font.italic,
        "color": color,
    }


def _is_sensitive_query_key(raw_key: str) -> bool:
    decoded = unquote_plus(raw_key).strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", decoded).strip("_")
    compact = normalized.replace("_", "")
    return normalized in SENSITIVE_QUERY_KEYS or compact in SENSITIVE_QUERY_KEYS_COMPACT


def _sanitize_query(query: str) -> str:
    sanitized_fields: list[str] = []
    for field in query.split("&"):
        if not field:
            sanitized_fields.append(field)
            continue
        key, _, _ = field.partition("=")
        if _is_sensitive_query_key(key):
            sanitized_fields.append(f"{key}=<redacted>")
        else:
            sanitized_fields.append(field)
    return "&".join(sanitized_fields)


def _sanitize_url(value: str) -> str:
    try:
        parts = urlsplit(value)
    except ValueError:
        prefix, query_separator, remainder = value.partition("?")
        query, fragment_separator, fragment = remainder.partition("#")
        prefix = re.sub(
            r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s@]+@",
            r"\1<redacted-userinfo>@",
            prefix,
        )
        if not query_separator:
            return prefix
        return (
            f"{prefix}?{_sanitize_query(query)}"
            f"{fragment_separator}{fragment if fragment_separator else ''}"
        )
    netloc = parts.netloc.rsplit("@", 1)[-1]
    return urlunsplit(
        (
            parts.scheme,
            netloc,
            parts.path,
            _sanitize_query(parts.query),
            parts.fragment,
        )
    )


def _text_frame_info(text_frame: Any) -> dict[str, Any]:
    paragraphs: list[dict[str, Any]] = []
    for paragraph_index, paragraph in enumerate(text_frame.paragraphs):
        runs = [
            {
                "index": run_index,
                "text": run.text,
                "font": _font_info(run.font),
                "hyperlink": (
                    _sanitize_url(run.hyperlink.address)
                    if run.hyperlink.address is not None
                    else None
                ),
            }
            for run_index, run in enumerate(paragraph.runs)
        ]
        paragraphs.append(
            {
                "index": paragraph_index,
                "text": paragraph.text,
                "level": paragraph.level,
                "alignment": _enum_name(paragraph.alignment),
                "runs": runs,
                "contains_field": OoxmlAdapter.paragraph_has_field(paragraph),
            }
        )
    return {
        "text": text_frame.text,
        "paragraphs": paragraphs,
        "word_wrap": text_frame.word_wrap,
    }


def _table_info(table: Any) -> dict[str, Any]:
    return {
        "rows": len(table.rows),
        "columns": len(table.columns),
        "data": [[cell.text for cell in row.cells] for row in table.rows],
    }


def _json_scalar(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _chart_info(chart: Any) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    result: dict[str, Any] = {
        "chart_type": _enum_name(chart.chart_type),
        "has_legend": chart.has_legend,
        "has_title": chart.has_title,
        "title": chart.chart_title.text_frame.text if chart.has_title else None,
        "series": [],
        "categories": [],
    }
    try:
        for plot in chart.plots:
            categories = getattr(plot, "categories", None)
            if categories is not None and not result["categories"]:
                result["categories"] = [
                    _json_scalar(getattr(category, "label", category)) for category in categories
                ]
    except (AttributeError, TypeError, ValueError) as exc:
        warnings.append(f"chart categories unavailable: {type(exc).__name__}")
    try:
        for series in chart.series:
            item: dict[str, Any] = {
                "name": series.name,
                "values": [_json_scalar(value) for value in series.values],
            }
            x_values = getattr(series, "x_values", None)
            if x_values is not None:
                item["x_values"] = [_json_scalar(value) for value in x_values]
            result["series"].append(item)
    except (AttributeError, TypeError, ValueError) as exc:
        warnings.append(f"chart series data unavailable: {type(exc).__name__}")
    return result, warnings


def _notes_by_slide_part(path: Path, report: PackageReport) -> dict[str, str]:
    notes: dict[str, str] = {}
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        for slide_part in report.slide_parts:
            slide_path = PurePosixPath(slide_part)
            rels_name = str(slide_path.parent / "_rels" / f"{slide_path.name}.rels")
            if rels_name not in names:
                notes[slide_part] = ""
                continue
            rel_root = _safe_xml(archive.read(rels_name), rels_name)
            note_part: str | None = None
            for rel in rel_root.findall(f"{{{REL_NS}}}Relationship"):
                if (
                    rel.get("Type") == NOTES_REL
                    and (rel.get("TargetMode") or "").lower() != "external"
                ):
                    note_part = _resolve_target(slide_part, rel.get("Target") or "")
                    break
            if note_part is None:
                notes[slide_part] = ""
                continue
            note_root = _safe_xml(archive.read(note_part), note_part)
            paragraphs: list[str] = []
            for shape in note_root.xpath("./p:cSld/p:spTree/p:sp", namespaces={"p": P_NS}):
                placeholders = shape.xpath("./p:nvSpPr/p:nvPr/p:ph", namespaces={"p": P_NS})
                placeholder_type = placeholders[0].get("type") if placeholders else None
                if placeholder_type in {"sldImg", "hdr", "ftr", "dt", "sldNum"}:
                    continue
                for paragraph in shape.xpath(".//a:p", namespaces={"a": A_NS}):
                    text = "".join(paragraph.xpath(".//a:t/text()", namespaces={"a": A_NS}))
                    if text:
                        paragraphs.append(text)
            notes[slide_part] = "\n".join(paragraphs)
    return notes


def _inspect_pptx_snapshot(
    path: Path,
    source: Path,
    report: PackageReport,
) -> dict[str, Any]:
    presentation = _open_presentation(path)
    notes = _notes_by_slide_part(path, report)
    warnings = list(report.warnings)
    slides: list[dict[str, Any]] = []
    image_count = table_count = chart_count = shape_count = text_chars = 0
    for slide_index, slide in enumerate(presentation.slides):
        shapes: list[dict[str, Any]] = []
        for shape_index, shape in enumerate(slide.shapes):
            shape_count += 1
            if shape_count > MAX_SHAPES:
                raise ToolError("resource_limit", "presentation has too many shapes")
            item: dict[str, Any] = {
                "index": shape_index,
                "shape_id": int(shape.shape_id),
                "name": shape.name,
                "shape_type": _enum_name(shape.shape_type),
                "geometry": _geometry(shape),
                "is_placeholder": shape.is_placeholder,
            }
            if shape.is_placeholder:
                item["placeholder"] = {
                    "type": _enum_name(shape.placeholder_format.type),
                    "index": shape.placeholder_format.idx,
                }
            if shape.has_text_frame:
                item["text"] = _text_frame_info(shape.text_frame)
                text_chars += len(shape.text_frame.text)
            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                image_count += 1
                image = shape.image
                blob = image.blob
                item["image"] = {
                    "filename": image.filename,
                    "content_type": image.content_type,
                    "bytes": len(blob),
                    "size_pixels": {
                        "width": image.size[0],
                        "height": image.size[1],
                    },
                    "sha256": hashlib.sha256(blob).hexdigest(),
                }
            if shape.has_table:
                table_count += 1
                item["table"] = _table_info(shape.table)
            if shape.has_chart:
                chart_count += 1
                chart_data, chart_warnings = _chart_info(shape.chart)
                item["chart"] = chart_data
                warnings.extend(
                    f"slide {slide_index}, shape {shape_index}: {warning}"
                    for warning in chart_warnings
                )
            shapes.append(item)
        slides.append(
            {
                "index": slide_index,
                "slide_id": int(slide.slide_id),
                "layout": {
                    "name": slide.slide_layout.name,
                    "index": next(
                        (
                            index
                            for index, layout in enumerate(presentation.slide_layouts)
                            if layout == slide.slide_layout
                        ),
                        None,
                    ),
                },
                "part": report.slide_parts[slide_index],
                "notes": notes.get(report.slide_parts[slide_index], ""),
                "shapes": shapes,
            }
        )
    if text_chars > MAX_TEXT_CHARS:
        raise ToolError("resource_limit", "presentation text exceeds the inspection limit")
    properties = presentation.core_properties
    core_properties = {
        key: _json_scalar(getattr(properties, key))
        for key in (
            "title",
            "subject",
            "author",
            "keywords",
            "comments",
            "last_modified_by",
            "created",
            "modified",
            "category",
            "content_status",
            "identifier",
            "language",
            "version",
        )
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "success": True,
        "operation": "inspect",
        "source": str(source.resolve()),
        "versions": _versions(),
        "package": report.as_dict(),
        "presentation": {
            "slide_size": {
                "width_emu": int(presentation.slide_width),
                "height_emu": int(presentation.slide_height),
                "width_inches": round(presentation.slide_width / 914400, 4),
                "height_inches": round(presentation.slide_height / 914400, 4),
            },
            "core_properties": core_properties,
            "masters": [
                {"index": index, "name": master.name}
                for index, master in enumerate(presentation.slide_masters)
            ],
            "layouts": [
                {"index": index, "name": layout.name}
                for index, layout in enumerate(presentation.slide_layouts)
            ],
            "slides": slides,
        },
        "counts": {
            "slides": len(slides),
            "shapes": shape_count,
            "images": image_count,
            "tables": table_count,
            "charts": chart_count,
            "notes_with_text": sum(bool(slide["notes"]) for slide in slides),
        },
        "warnings": sorted(set(warnings)),
        "verification": {
            "immutable_source_snapshot": True,
            "package_preflight": True,
            "package_reopen": True,
            "internal_relationship_targets_exist": True,
            "owner_relationship_references_resolved": True,
            "slide_count": len(slides),
            "slide_ids": list(report.slide_ids),
        },
    }


def inspect_pptx(path: Path) -> dict[str, Any]:
    with _pptx_source_snapshot(path) as snapshot:
        result = _inspect_pptx_snapshot(
            snapshot.snapshot_path,
            snapshot.original_path,
            snapshot.report,
        )
        path_matches_snapshot = _source_hash_matches(
            snapshot.original_path,
            snapshot.sha256,
        )
    result["verification"]["source_path_matched_snapshot_at_completion"] = path_matches_snapshot
    if not path_matches_snapshot:
        result["warnings"].append(
            "source path changed during inspection; results describe the validated immutable "
            "snapshot"
        )
        result["warnings"] = sorted(set(result["warnings"]))
    return result


def _load_job(path_value: str, expected_operation: str) -> tuple[dict[str, Any], Path]:
    if path_value == "-":
        try:
            text = sys.stdin.read()
        except OSError as exc:
            raise ToolError("bad_input", "could not read job JSON from stdin") from exc
        base_dir = Path.cwd()
        label = "stdin"
    else:
        path = Path(path_value).expanduser()
        if not path.is_file():
            raise ToolError("bad_input", "job file does not exist", {"path": str(path)})
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ToolError("bad_input", "job file must be readable UTF-8 JSON") from exc
        base_dir = path.resolve().parent
        label = str(path)

    def reject_constant(value: str) -> Never:
        raise ValueError(f"non-standard JSON constant {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key}")
            result[key] = item
        return result

    try:
        value = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        details: dict[str, Any] = {"path": label}
        if isinstance(exc, json.JSONDecodeError):
            details.update({"line": exc.lineno, "column": exc.colno})
        raise ToolError(
            "bad_input",
            "job is not strict valid JSON",
            details,
        ) from exc
    job = _require_mapping(value, "job")
    if job.get("schema_version") != SCHEMA_VERSION:
        raise ToolError(
            "bad_input",
            "job schema_version must be 1",
            {"actual": job.get("schema_version")},
        )
    if job.get("operation") != expected_operation:
        raise ToolError(
            "bad_input",
            "job operation does not match the CLI command",
            {"expected": expected_operation, "actual": job.get("operation")},
        )
    return job, base_dir


def _resolve_resource(base_dir: Path, value: Any, label: str) -> Path:
    text = _require_string(value, label)
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ToolError("bad_input", f"{label} does not exist", {"path": str(candidate)}) from exc
    if not resolved.is_file():
        raise ToolError("bad_input", f"{label} must be a regular file", {"path": str(resolved)})
    return resolved


def _prepare_destination(
    destination: Path,
    suffix: str,
    overwrite: bool,
    source: Path | None = None,
) -> Path:
    destination = destination.expanduser()
    if destination.is_symlink():
        raise ToolError("bad_input", "destination must not be a symbolic link")
    destination = destination.resolve()
    if destination.suffix.lower() != suffix:
        raise ToolError(
            "bad_input",
            f"destination must use the {suffix} extension",
            {"path": str(destination)},
        )
    if destination.exists() and not destination.is_file():
        raise ToolError("bad_input", "destination exists and is not a regular file")
    if destination.exists() and not overwrite:
        raise ToolError(
            "bad_input",
            "destination already exists; pass --overwrite to replace it",
            {"path": str(destination)},
        )
    if source is not None and destination == source.resolve():
        raise ToolError(
            "bad_input",
            "source and destination must always be distinct",
        )
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ToolError("bad_input", "could not create destination directory") from exc
    return destination


def _temporary_sibling(destination: Path, suffix: str) -> Path:
    return destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp{suffix}"


def _fsync(path: Path) -> None:
    with path.open("rb") as output:
        os.fsync(output.fileno())


def _atomic_publish(temporary: Path, destination: Path, overwrite: bool) -> None:
    try:
        _fsync(temporary)
        if overwrite:
            os.replace(temporary, destination)
        else:
            try:
                os.link(temporary, destination)
            except FileExistsError as exc:
                raise ToolError(
                    "bad_input",
                    "destination was created while the output was being prepared",
                    {"path": str(destination)},
                ) from exc
            temporary.unlink()
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise ToolError(
            "post_write_validation",
            "could not atomically publish the verified destination",
            {"path": str(destination)},
        ) from exc


def _layout_for(presentation: Any, value: Any) -> Any:
    if value is None:
        index = 1
        if len(presentation.slide_layouts) <= index:
            raise ToolError("bad_input", "presentation does not contain default layout index 1")
        return presentation.slide_layouts[index]
    selector = _require_mapping(value, "layout")
    _reject_unknown(selector, {"name", "index"}, "layout")
    if ("name" in selector) == ("index" in selector):
        raise ToolError("bad_input", "layout must contain exactly one of name or index")
    if "index" in selector:
        index = selector["index"]
        if not isinstance(index, int) or isinstance(index, bool):
            raise ToolError("bad_input", "layout.index must be an integer")
        if not 0 <= index < len(presentation.slide_layouts):
            raise ToolError(
                "bad_input",
                "layout index is out of range",
                {"index": index, "layouts": len(presentation.slide_layouts)},
            )
        return presentation.slide_layouts[index]
    name = _require_string(selector["name"], "layout.name")
    matches = [layout for layout in presentation.slide_layouts if layout.name == name]
    if len(matches) != 1:
        raise ToolError(
            "ambiguous_edit",
            "layout name must match exactly one layout",
            {"name": name, "matches": len(matches)},
        )
    return matches[0]


def _box(value: Any, label: str, *, image: bool = False) -> dict[str, Any]:
    box = _require_mapping(value, label)
    _reject_unknown(box, {"x", "y", "width", "height"}, label)
    required = {"x", "y"} if image else {"x", "y", "width", "height"}
    missing = sorted(required - set(box))
    if missing:
        raise ToolError("bad_input", f"{label} is missing geometry", {"keys": missing})
    if image and "width" not in box and "height" not in box:
        raise ToolError("bad_input", f"{label} requires width or height")
    converted: dict[str, Any] = {}
    for key in ("x", "y", "width", "height"):
        if key not in box:
            converted[key] = None
            continue
        raw = box[key]
        if (
            not isinstance(raw, (int, float))
            or isinstance(raw, bool)
            or not math.isfinite(raw)
            or raw < 0
            or (key in {"width", "height"} and raw == 0)
            or raw > MAX_GEOMETRY_INCHES
        ):
            raise ToolError(
                "bad_input",
                f"{label}.{key} must be a finite valid inch value",
            )
        converted[key] = Inches(float(raw))
    return converted


def _apply_run_spec(run: Any, spec: dict[str, Any], label: str) -> None:
    _reject_unknown(
        spec,
        {"text", "bold", "italic", "font_name", "font_size", "color"},
        label,
    )
    run.text = _require_string(spec.get("text"), f"{label}.text", allow_empty=True)
    for key in ("bold", "italic"):
        if key in spec:
            if not isinstance(spec[key], bool):
                raise ToolError("bad_input", f"{label}.{key} must be boolean")
            setattr(run.font, key, spec[key])
    if "font_name" in spec:
        run.font.name = _require_string(spec["font_name"], f"{label}.font_name")
    if "font_size" in spec:
        size = spec["font_size"]
        if not isinstance(size, (int, float)) or isinstance(size, bool) or not 1 <= size <= 400:
            raise ToolError("bad_input", f"{label}.font_size must be between 1 and 400")
        run.font.size = Pt(float(size))
    if "color" in spec:
        color = _require_string(spec["color"], f"{label}.color").upper()
        if len(color) != 6 or any(char not in "0123456789ABCDEF" for char in color):
            raise ToolError("bad_input", f"{label}.color must be six hexadecimal digits")
        from pptx.dml.color import RGBColor

        run.font.color.rgb = RGBColor.from_string(color)


def _normalize_paragraphs(value: Any, label: str) -> list[dict[str, Any]]:
    if isinstance(value, str):
        return [{"text": value}]
    if isinstance(value, dict):
        return [value]
    paragraphs = _require_list(value, label)
    return [_require_mapping(item, f"{label}[{index}]") for index, item in enumerate(paragraphs)]


def _set_text_frame(text_frame: Any, value: Any, label: str) -> None:
    paragraphs = _normalize_paragraphs(value, label)
    text_frame.clear()
    for index, spec in enumerate(paragraphs):
        _reject_unknown(spec, {"text", "runs", "level", "alignment"}, f"{label}[{index}]")
        if ("text" in spec) == ("runs" in spec):
            raise ToolError(
                "bad_input",
                f"{label}[{index}] must contain exactly one of text or runs",
            )
        paragraph = text_frame.paragraphs[0] if index == 0 else text_frame.add_paragraph()
        if "level" in spec:
            level = spec["level"]
            if not isinstance(level, int) or isinstance(level, bool) or not 0 <= level <= 8:
                raise ToolError("bad_input", f"{label}[{index}].level must be 0 through 8")
            paragraph.level = level
        if "alignment" in spec:
            alignment = _require_string(spec["alignment"], f"{label}[{index}].alignment").lower()
            if alignment not in ALIGNMENTS:
                raise ToolError("bad_input", "unsupported paragraph alignment")
            paragraph.alignment = ALIGNMENTS[alignment]
        if "text" in spec:
            paragraph.text = _require_string(
                spec["text"], f"{label}[{index}].text", allow_empty=True
            )
        else:
            runs = _require_list(spec["runs"], f"{label}[{index}].runs")
            for run_index, run_value in enumerate(runs):
                run_spec = _require_mapping(run_value, f"{label}[{index}].runs[{run_index}]")
                run = paragraph.add_run()
                _apply_run_spec(run, run_spec, f"{label}[{index}].runs[{run_index}]")


def _body_placeholder(slide: Any) -> Any | None:
    preferred = {
        PP_PLACEHOLDER.BODY,
        PP_PLACEHOLDER.OBJECT,
        PP_PLACEHOLDER.SUBTITLE,
        PP_PLACEHOLDER.VERTICAL_BODY,
        PP_PLACEHOLDER.VERTICAL_OBJECT,
        PP_PLACEHOLDER.VERTICAL_TITLE,
    }
    matches = [
        shape
        for shape in slide.placeholders
        if shape.placeholder_format.type in preferred and shape.has_text_frame
    ]
    return matches[0] if matches else None


def _validate_image(path: Path) -> ImageSnapshot:
    try:
        with path.open("rb") as source:
            data = source.read(MAX_IMAGE_BYTES + 1)
    except OSError as exc:
        raise ToolError("bad_input", "image resource could not be read") from exc
    if len(data) > MAX_IMAGE_BYTES:
        raise ToolError(
            "resource_limit",
            "image exceeds the size limit",
            {"path": str(path), "bytes": len(data), "limit": MAX_IMAGE_BYTES},
        )
    try:
        with python_warnings.catch_warnings():
            python_warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                image_format = image.format or ""
                width, height = image.size
                frames = getattr(image, "n_frames", 1)
                if (
                    width > MAX_IMAGE_DIMENSION
                    or height > MAX_IMAGE_DIMENSION
                    or width * height > MAX_IMAGE_PIXELS
                ):
                    raise ToolError(
                        "resource_limit",
                        "image exceeds decoded dimension limits",
                        {
                            "width": width,
                            "height": height,
                            "pixel_limit": MAX_IMAGE_PIXELS,
                        },
                    )
                if frames != 1:
                    raise ToolError(
                        "unsupported_operation",
                        "animated or multi-frame images are unsupported",
                        {"frames": frames},
                    )
                decoded_bytes = width * height * max(1, len(image.getbands()))
                if decoded_bytes > MAX_DECODED_IMAGE_BYTES:
                    raise ToolError(
                        "resource_limit",
                        "image exceeds the decoded size limit",
                        {"bytes": decoded_bytes, "limit": MAX_DECODED_IMAGE_BYTES},
                    )
                image.verify()
            with Image.open(io.BytesIO(data)) as decoded_image:
                if decoded_image.size != (width, height):
                    raise ToolError(
                        "bad_input",
                        "image dimensions differ between validation passes",
                    )
                decoded_image.load()
    except ToolError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ToolError("resource_limit", "image exceeds Pillow decompression limits") from exc
    except MemoryError as exc:
        raise ToolError(
            "resource_limit",
            "image could not be decoded within memory limits",
        ) from exc
    except (OSError, ValueError) as exc:
        raise ToolError("bad_input", "resource is not a supported valid image") from exc
    content_type = IMAGE_CONTENT_TYPES.get(image_format.upper())
    if content_type is None:
        raise ToolError(
            "unsupported_operation",
            "image format is unsupported",
            {"format": image_format},
        )
    return ImageSnapshot(
        data=data,
        content_type=content_type,
        width=width,
        height=height,
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _validate_numeric(value: Any, label: str) -> int | float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ToolError("bad_input", f"{label} must be a finite number")
    return value


def _chart_data(spec: dict[str, Any], label: str) -> tuple[str, Any, int]:
    chart_type = _require_string(spec.get("chart_type"), f"{label}.chart_type").lower()
    if chart_type not in CHART_TYPES:
        raise ToolError(
            "unsupported_operation",
            "unsupported chart type",
            {"chart_type": chart_type},
        )
    series_specs = _require_list(spec.get("series"), f"{label}.series")
    if not series_specs:
        raise ToolError("bad_input", f"{label}.series must not be empty")
    points = 0
    if chart_type in SCATTER_TYPES:
        data = XyChartData()
        for index, value in enumerate(series_specs):
            series_spec = _require_mapping(value, f"{label}.series[{index}]")
            _reject_unknown(series_spec, {"name", "points"}, f"{label}.series[{index}]")
            series = data.add_series(
                _require_string(series_spec.get("name"), f"{label}.series[{index}].name")
            )
            rows = _require_list(series_spec.get("points"), f"{label}.series[{index}].points")
            for point_index, point_value in enumerate(rows):
                point = _require_list(point_value, f"{label}.series[{index}].points[{point_index}]")
                if len(point) != 2:
                    raise ToolError("bad_input", "XY chart points must contain [x, y]")
                series.add_data_point(
                    _validate_numeric(point[0], "XY x value"),
                    _validate_numeric(point[1], "XY y value"),
                )
                points += 1
    else:
        categories = _require_list(spec.get("categories"), f"{label}.categories")
        if not categories:
            raise ToolError("bad_input", f"{label}.categories must not be empty")
        if not all(isinstance(item, (str, int, float)) for item in categories):
            raise ToolError("bad_input", "chart categories must be strings or numbers")
        data = CategoryChartData()
        data.categories = categories
        for index, value in enumerate(series_specs):
            series_spec = _require_mapping(value, f"{label}.series[{index}]")
            _reject_unknown(series_spec, {"name", "values"}, f"{label}.series[{index}]")
            values = _require_list(series_spec.get("values"), f"{label}.series[{index}].values")
            if len(values) != len(categories):
                raise ToolError(
                    "bad_input",
                    "chart series length must equal category length",
                    {"series": index, "values": len(values), "categories": len(categories)},
                )
            normalized = [
                None if item is None else _validate_numeric(item, "chart value") for item in values
            ]
            data.add_series(
                _require_string(series_spec.get("name"), f"{label}.series[{index}].name"),
                normalized,
            )
            points += len(values)
    if points > MAX_CHART_POINTS:
        raise ToolError(
            "resource_limit",
            "chart exceeds the point limit",
            {"points": points, "limit": MAX_CHART_POINTS},
        )
    return chart_type, data, points


def _add_element(slide: Any, value: Any, base_dir: Path, label: str) -> dict[str, int]:
    spec = _require_mapping(value, label)
    element_type = _require_string(spec.get("type"), f"{label}.type").lower()
    counts = {"shapes": 1, "images": 0, "tables": 0, "charts": 0}
    if element_type == "text":
        _reject_unknown(spec, {"type", "name", "box", "paragraphs"}, label)
        geometry = _box(spec.get("box"), f"{label}.box")
        shape = slide.shapes.add_textbox(
            geometry["x"], geometry["y"], geometry["width"], geometry["height"]
        )
        _set_text_frame(shape.text_frame, spec.get("paragraphs"), f"{label}.paragraphs")
    elif element_type == "image":
        _reject_unknown(spec, {"type", "name", "path", "box"}, label)
        path = _resolve_resource(base_dir, spec.get("path"), f"{label}.path")
        image = _validate_image(path)
        geometry = _box(spec.get("box"), f"{label}.box", image=True)
        shape = slide.shapes.add_picture(
            io.BytesIO(image.data),
            geometry["x"],
            geometry["y"],
            width=geometry["width"],
            height=geometry["height"],
        )
        counts["images"] = 1
    elif element_type == "table":
        _reject_unknown(spec, {"type", "name", "box", "data"}, label)
        data = _require_list(spec.get("data"), f"{label}.data")
        if not data:
            raise ToolError("bad_input", "table data must not be empty")
        rows = [_require_list(row, f"{label}.data[{index}]") for index, row in enumerate(data)]
        columns = len(rows[0])
        if not columns or any(len(row) != columns for row in rows):
            raise ToolError("bad_input", "table data must be a non-empty rectangular matrix")
        if len(rows) > MAX_TABLE_ROWS or columns > MAX_TABLE_COLUMNS:
            raise ToolError("resource_limit", "table exceeds row/column limits")
        geometry = _box(spec.get("box"), f"{label}.box")
        shape = slide.shapes.add_table(
            len(rows),
            columns,
            geometry["x"],
            geometry["y"],
            geometry["width"],
            geometry["height"],
        )
        for row_index, row in enumerate(rows):
            for column_index, cell_value in enumerate(row):
                if not isinstance(cell_value, (str, int, float, bool)) and cell_value is not None:
                    raise ToolError("bad_input", "table cells must contain JSON scalar values")
                shape.table.cell(row_index, column_index).text = (
                    "" if cell_value is None else str(cell_value)
                )
        counts["tables"] = 1
    elif element_type == "chart":
        _reject_unknown(
            spec,
            {
                "type",
                "name",
                "box",
                "chart_type",
                "categories",
                "series",
                "title",
                "has_legend",
            },
            label,
        )
        geometry = _box(spec.get("box"), f"{label}.box")
        chart_type, data, _ = _chart_data(spec, label)
        shape = slide.shapes.add_chart(
            CHART_TYPES[chart_type],
            geometry["x"],
            geometry["y"],
            geometry["width"],
            geometry["height"],
            data,
        )
        chart = shape.chart
        if "title" in spec:
            chart.has_title = True
            chart.chart_title.text_frame.text = _require_string(
                spec["title"], f"{label}.title", allow_empty=True
            )
        if "has_legend" in spec:
            if not isinstance(spec["has_legend"], bool):
                raise ToolError("bad_input", f"{label}.has_legend must be boolean")
            chart.has_legend = spec["has_legend"]
        counts["charts"] = 1
    else:
        raise ToolError(
            "unsupported_operation",
            "unsupported slide element type",
            {"type": element_type},
        )
    if "name" in spec:
        shape.name = _require_string(spec["name"], f"{label}.name")
    return counts


def _set_notes(slide: Any, text: Any, label: str) -> None:
    notes = _require_string(text, label, allow_empty=True)
    slide.notes_slide.notes_text_frame.text = notes


def _add_slide(presentation: Any, spec_value: Any, base_dir: Path) -> tuple[Any, dict[str, int]]:
    spec = _require_mapping(spec_value, "slide")
    _reject_unknown(spec, {"layout", "title", "body", "elements", "notes"}, "slide")
    if len(presentation.slides) >= MAX_SLIDES:
        raise ToolError("resource_limit", "presentation exceeds the slide limit")
    layout = _layout_for(presentation, spec.get("layout"))
    slide = presentation.slides.add_slide(layout)
    counts = {"slides": 1, "shapes": 0, "images": 0, "tables": 0, "charts": 0}
    if "title" in spec:
        title = slide.shapes.title
        if title is None or not title.has_text_frame:
            raise ToolError("bad_input", "selected layout has no title text placeholder")
        _set_text_frame(title.text_frame, spec["title"], "slide.title")
    if "body" in spec:
        body = _body_placeholder(slide)
        if body is None:
            raise ToolError("bad_input", "selected layout has no body/subtitle text placeholder")
        _set_text_frame(body.text_frame, spec["body"], "slide.body")
    elements = _require_list(spec.get("elements", []), "slide.elements")
    for index, element in enumerate(elements):
        delta = _add_element(slide, element, base_dir, f"slide.elements[{index}]")
        for key, value in delta.items():
            counts[key] += value
    if "notes" in spec:
        _set_notes(slide, spec["notes"], "slide.notes")
    counts["shapes"] += len(slide.shapes)
    return slide, counts


def _apply_metadata(presentation: Any, value: Any) -> None:
    metadata_spec = _require_mapping(value, "metadata")
    allowed = {
        "title",
        "subject",
        "author",
        "keywords",
        "comments",
        "last_modified_by",
        "category",
        "content_status",
        "identifier",
        "language",
        "version",
    }
    _reject_unknown(metadata_spec, allowed, "metadata")
    for key, raw in metadata_spec.items():
        setattr(
            presentation.core_properties,
            key,
            _require_string(raw, f"metadata.{key}", allow_empty=True),
        )


def _save_atomic_presentation(
    presentation: Any,
    destination: Path,
    expected_ids: list[int],
    overwrite: bool,
    source: Path | None = None,
    source_hash: str | None = None,
) -> tuple[PackageReport, bool]:
    temporary = _temporary_sibling(destination, ".pptx")
    try:
        _enforce_presentation_limits(presentation)
        try:
            presentation.save(str(temporary))
        except Exception as exc:
            raise ToolError("post_write_validation", "could not save the staged PPTX") from exc
        report = _validate_pptx_output(temporary, expected_ids)
        reopened = _open_presentation(temporary, "post_write_validation")
        if _picture_hashes(reopened) != _picture_hashes(presentation):
            raise ToolError(
                "post_write_validation",
                "written PPTX picture relationship verification failed",
            )
        if source is not None and not _source_hash_matches(source, source_hash):
            raise ToolError(
                "post_write_validation",
                "source changed before the destination publication gate",
            )
        _atomic_publish(temporary, destination, overwrite)
        post_publish_source_changed = source is not None and not _source_hash_matches(
            source, source_hash
        )
        return report, post_publish_source_changed
    finally:
        temporary.unlink(missing_ok=True)


def create_pptx(
    job: dict[str, Any],
    base_dir: Path,
    destination: Path,
    overwrite: bool,
) -> dict[str, Any]:
    _reject_unknown(
        job,
        {
            "schema_version",
            "operation",
            "template",
            "keep_template_slides",
            "metadata",
            "slides",
        },
        "create job",
    )
    template: Path | None = None
    source_hash: str | None = None
    warnings: list[str] = []
    if "template" in job:
        template = _resolve_resource(base_dir, job["template"], "template")
        with _pptx_source_snapshot(template) as snapshot:
            warnings.extend(snapshot.report.warnings)
            source_hash = snapshot.sha256
            presentation = _open_presentation(snapshot.snapshot_path)
        destination = _prepare_destination(destination, ".pptx", overwrite, template)
    else:
        destination = _prepare_destination(destination, ".pptx", overwrite)
        presentation = Presentation()
    keep_template_slides = job.get("keep_template_slides", False)
    if not isinstance(keep_template_slides, bool):
        raise ToolError("bad_input", "keep_template_slides must be boolean")
    checkpoints: list[dict[str, Any]] = []
    graph_ordinal = 0
    with tempfile.TemporaryDirectory(prefix="pptx-create-checkpoints-") as temp_dir:
        checkpoint_dir = Path(temp_dir)
        if template is not None and not keep_template_slides:
            while len(presentation.slides):
                before = _presentation_slide_ids(presentation)
                OoxmlAdapter.remove_slide(presentation, len(presentation.slides) - 1)
                expected = before[:-1]
                graph_ordinal += 1
                presentation, checkpoint = _checkpoint_graph_mutation(
                    presentation,
                    checkpoint_dir,
                    graph_ordinal,
                    before,
                    expected,
                )
                checkpoints.append(checkpoint)
        if "metadata" in job:
            _apply_metadata(presentation, job["metadata"])
        slides = _require_list(job.get("slides"), "slides")
        if not slides:
            raise ToolError("bad_input", "create job must contain at least one slide")
        counts = {"slides_added": 0, "shapes": 0, "images": 0, "tables": 0, "charts": 0}
        for slide_spec in slides:
            before = _presentation_slide_ids(presentation)
            _, added = _add_slide(presentation, slide_spec, base_dir)
            expected = _presentation_slide_ids(presentation)
            graph_ordinal += 1
            presentation, checkpoint = _checkpoint_graph_mutation(
                presentation,
                checkpoint_dir,
                graph_ordinal,
                before,
                expected,
            )
            checkpoints.append(checkpoint)
            counts["slides_added"] += 1
            counts["shapes"] += added["shapes"]
            counts["images"] += added["images"]
            counts["tables"] += added["tables"]
            counts["charts"] += added["charts"]
        expected_ids = _presentation_slide_ids(presentation)
        output_report, post_publish_source_changed = _save_atomic_presentation(
            presentation,
            destination,
            expected_ids,
            overwrite,
            template,
            source_hash,
        )
    counts["slides"] = len(presentation.slides)
    counts["shapes"] = sum(len(slide.shapes) for slide in presentation.slides)
    warnings.extend(output_report.warnings)
    if post_publish_source_changed:
        warnings.append(
            "template path changed after destination publication; the published deck was built "
            "from the validated immutable snapshot"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "success": True,
        "operation": "create",
        "template": str(template) if template else None,
        "output": str(destination),
        "versions": _versions(),
        "counts": counts,
        "warnings": sorted(set(warnings)),
        "verification": {
            "atomic_publish": True,
            "package_preflight": True,
            "package_reopen": True,
            "internal_relationship_targets_exist": True,
            "owner_relationship_references_resolved": True,
            "source_unchanged_at_publish_gate": True,
            "post_publish_source_changed": post_publish_source_changed,
            "slide_count": len(expected_ids),
            "slide_ids": expected_ids,
            "graph_checkpoints": checkpoints,
        },
    }


def _slide_selector(value: Any, label: str = "slide") -> dict[str, Any]:
    selector = _require_mapping(value, label)
    _reject_unknown(selector, {"slide_id", "slide_index"}, label)
    if ("slide_id" in selector) == ("slide_index" in selector):
        raise ToolError("bad_input", f"{label} must contain exactly one slide selector")
    return selector


def _select_one_slide(presentation: Any, value: Any) -> tuple[int, Any]:
    selector = _slide_selector(value)
    if "slide_id" in selector:
        slide_id = selector["slide_id"]
        if not isinstance(slide_id, int) or isinstance(slide_id, bool):
            raise ToolError("bad_input", "slide.slide_id must be an integer")
        matches = [
            (index, slide)
            for index, slide in enumerate(presentation.slides)
            if int(slide.slide_id) == slide_id
        ]
    else:
        index = selector["slide_index"]
        if not isinstance(index, int) or isinstance(index, bool):
            raise ToolError("bad_input", "slide.slide_index must be an integer")
        matches = (
            [(index, presentation.slides[index])] if 0 <= index < len(presentation.slides) else []
        )
    if len(matches) != 1:
        raise ToolError(
            "ambiguous_edit",
            "slide selector must identify exactly one current slide",
            {"selector": selector, "matches": len(matches)},
        )
    return matches[0]


def _slides_in_scope(presentation: Any, value: Any | None) -> list[tuple[int, Any]]:
    if value is None:
        return list(enumerate(presentation.slides))
    return [_select_one_slide(presentation, value)]


def _shape_selector(value: Any) -> dict[str, Any]:
    selector = _require_mapping(value, "shape")
    allowed = {"shape_name", "shape_index", "table_index", "chart_index", "image_index"}
    _reject_unknown(selector, allowed, "shape")
    if len(selector) != 1:
        raise ToolError("bad_input", "shape must contain exactly one selector")
    return selector


def _shapes_across_scope(
    presentation: Any,
    slide_selector: Any | None,
    shape_selector: Any | None,
) -> list[tuple[int, Any]]:
    slides = _slides_in_scope(presentation, slide_selector)
    pool = [(slide_index, shape) for slide_index, slide in slides for shape in slide.shapes]
    if shape_selector is None:
        return pool
    selector = _shape_selector(shape_selector)
    key, raw = next(iter(selector.items()))
    if key == "shape_name":
        name = _require_string(raw, "shape.shape_name")
        matches = [(slide_index, shape) for slide_index, shape in pool if shape.name == name]
    else:
        if not isinstance(raw, int) or isinstance(raw, bool) or raw < 0:
            raise ToolError("bad_input", f"shape.{key} must be a non-negative integer")
        if key == "shape_index":
            typed_pool = pool
        elif key == "table_index":
            typed_pool = [(index, shape) for index, shape in pool if shape.has_table]
        elif key == "chart_index":
            typed_pool = [(index, shape) for index, shape in pool if shape.has_chart]
        else:
            typed_pool = [
                (index, shape)
                for index, shape in pool
                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE
            ]
        matches = [typed_pool[raw]] if raw < len(typed_pool) else []
    if len(matches) != 1:
        raise ToolError(
            "ambiguous_edit",
            "shape selector must identify exactly one shape in the selected slide scope",
            {"selector": selector, "matches": len(matches)},
        )
    return matches


def _select_one_shape_across_scope(
    presentation: Any,
    operation: dict[str, Any],
) -> tuple[int, Any]:
    matches = _shapes_across_scope(
        presentation,
        operation.get("slide"),
        operation.get("shape"),
    )
    if len(matches) != 1:
        raise ToolError(
            "ambiguous_edit",
            "shape selection must identify exactly one shape in the selected slide scope",
            {
                "slide": operation.get("slide"),
                "shape": operation.get("shape"),
                "matches": len(matches),
            },
        )
    return matches[0]


def _all_occurrences(text: str, needle: str) -> list[tuple[int, int]]:
    occurrences: list[tuple[int, int]] = []
    start = 0
    while True:
        found = text.find(needle, start)
        if found < 0:
            return occurrences
        occurrences.append((found, found + len(needle)))
        start = found + len(needle)


def _run_position(runs: list[Any], offset: int, *, end: bool = False) -> tuple[int, int]:
    target = offset - 1 if end else offset
    cumulative = 0
    for index, run in enumerate(runs):
        next_offset = cumulative + len(run.text)
        if target < next_offset:
            inner = target - cumulative + (1 if end else 0)
            return index, inner
        cumulative = next_offset
    raise ToolError("internal_error", "could not map replacement span to a text run")


def _collect_run_matches(
    presentation: Any,
    operation: dict[str, Any],
    needle: str,
) -> tuple[list[RunMatch], list[str]]:
    matches: list[RunMatch] = []
    unsafe: list[str] = []
    for slide_index, shape in _shapes_across_scope(
        presentation,
        operation.get("slide"),
        operation.get("shape"),
    ):
        if not shape.has_text_frame:
            continue
        text_frame = shape.text_frame
        if "\n" in needle and needle in text_frame.text:
            unsafe.append(f"slide {slide_index} shape {shape.name!r}: match crosses paragraphs")
        for paragraph_index, paragraph in enumerate(text_frame.paragraphs):
            runs = list(paragraph.runs)
            run_by_element = {id(run._r): (index, run) for index, run in enumerate(runs)}
            groups: list[list[tuple[int, Any]]] = []
            current: list[tuple[int, Any]] = []
            visible_parts: list[tuple[str, str, int | None]] = []
            group_number = 0
            for child in paragraph._p:
                local_name = etree.QName(child).localname
                if local_name == "r":
                    indexed_run = run_by_element.get(id(child))
                    if indexed_run is not None:
                        current.append(indexed_run)
                        visible_parts.append(("run", indexed_run[1].text, group_number))
                elif local_name in {"fld", "br"}:
                    if current:
                        groups.append(current)
                        current = []
                        group_number += 1
                    field_text = "".join(child.itertext()) if local_name == "fld" else "\n"
                    visible_parts.append((local_name, field_text, None))
            if current:
                groups.append(current)

            visible_text = "".join(part for _, part, _ in visible_parts)
            visible_kinds: list[tuple[str, int | None]] = [
                (kind, group) for kind, part, group in visible_parts for _ in part
            ]
            for start, end in _all_occurrences(visible_text, needle):
                span = visible_kinds[start:end]
                if not span or any(kind != "run" for kind, _ in span):
                    unsafe.append(
                        f"slide {slide_index} shape {shape.name!r} paragraph "
                        f"{paragraph_index}: match intersects a field or break"
                    )
                elif len({group for _, group in span}) != 1:
                    unsafe.append(
                        f"slide {slide_index} shape {shape.name!r} paragraph "
                        f"{paragraph_index}: match crosses a field or break"
                    )

            for group in groups:
                group_runs = [run for _, run in group]
                run_text = "".join(run.text for run in group_runs)
                for start, end in _all_occurrences(run_text, needle):
                    local_first, first_offset = _run_position(group_runs, start)
                    local_last, last_offset = _run_position(group_runs, end, end=True)
                    first_run = group[local_first][0]
                    last_run = group[local_last][0]
                    addresses = {
                        runs[index].hyperlink.address for index in range(first_run, last_run + 1)
                    }
                    matches.append(
                        RunMatch(
                            paragraph=paragraph,
                            first_run=first_run,
                            first_offset=first_offset,
                            last_run=last_run,
                            last_offset=last_offset,
                            cross_run=first_run != last_run,
                            hyperlink_boundary=len(addresses) > 1,
                        )
                    )
    return matches, unsafe


def _collect_frame_matches(
    presentation: Any,
    operation: dict[str, Any],
    needle: str,
) -> list[FrameMatch]:
    matches: list[FrameMatch] = []
    for _, shape in _shapes_across_scope(
        presentation,
        operation.get("slide"),
        operation.get("shape"),
    ):
        if not shape.has_text_frame:
            continue
        for start, end in _all_occurrences(shape.text_frame.text, needle):
            matches.append(FrameMatch(shape.text_frame, start, end))
    return matches


def _choose_matches(matches: list[Any], operation: dict[str, Any]) -> list[Any]:
    replace_all = operation.get("replace_all", False)
    if not isinstance(replace_all, bool):
        raise ToolError("bad_input", "replace_all must be boolean")
    occurrence = operation.get("occurrence")
    if replace_all and occurrence is not None:
        raise ToolError("bad_input", "replace_all and occurrence are mutually exclusive")
    if occurrence is not None:
        if not isinstance(occurrence, int) or isinstance(occurrence, bool) or occurrence < 1:
            raise ToolError("bad_input", "occurrence must be a positive one-based integer")
        if occurrence > len(matches):
            raise ToolError(
                "ambiguous_edit",
                "requested replacement occurrence does not exist",
                {"occurrence": occurrence, "matches": len(matches)},
            )
        return [matches[occurrence - 1]]
    if replace_all:
        return matches
    if len(matches) != 1:
        raise ToolError(
            "ambiguous_edit",
            "replacement must match exactly once unless replace_all or occurrence is supplied",
            {"matches": len(matches)},
        )
    return matches


def _replace_text(presentation: Any, operation: dict[str, Any]) -> tuple[int, list[str]]:
    _reject_unknown(
        operation,
        {
            "action",
            "slide",
            "shape",
            "find",
            "replace",
            "replace_all",
            "occurrence",
            "formatting_policy",
            "destructive_reconstruction",
        },
        "replace_text operation",
    )
    needle = _require_string(operation.get("find"), "replace_text.find")
    replacement = _require_string(
        operation.get("replace"), "replace_text.replace", allow_empty=True
    )
    destructive = operation.get("destructive_reconstruction", False)
    if not isinstance(destructive, bool):
        raise ToolError("bad_input", "destructive_reconstruction must be boolean")
    warnings: list[str] = []
    if destructive:
        matches = _choose_matches(
            _collect_frame_matches(presentation, operation, needle),
            operation,
        )
        grouped: dict[int, tuple[Any, list[FrameMatch]]] = {}
        for match in matches:
            key = id(match.text_frame)
            grouped.setdefault(key, (match.text_frame, []))[1].append(match)
        for text_frame, frame_matches in grouped.values():
            text = text_frame.text
            for match in sorted(frame_matches, key=lambda item: item.start, reverse=True):
                text = text[: match.start] + replacement + text[match.end :]
            text_frame.text = text
        warnings.append(
            "destructive text reconstruction flattened formatting, fields, and hyperlinks"
        )
        return len(matches), warnings

    matches, unsafe = _collect_run_matches(presentation, operation, needle)
    if unsafe:
        raise ToolError(
            "ambiguous_edit",
            "text replacement crosses a paragraph or field boundary",
            {"matches": unsafe},
        )
    selected = _choose_matches(matches, operation)
    formatting_policy = operation.get("formatting_policy")
    if formatting_policy is not None and formatting_policy != "first_run":
        raise ToolError(
            "bad_input",
            "formatting_policy must be first_run when supplied",
        )
    if any(match.hyperlink_boundary for match in selected):
        raise ToolError(
            "ambiguous_edit",
            "text replacement crosses incompatible hyperlink boundaries",
        )
    if any(match.cross_run for match in selected) and formatting_policy != "first_run":
        raise ToolError(
            "ambiguous_edit",
            "cross-run replacement requires formatting_policy first_run",
        )
    by_paragraph: dict[int, tuple[Any, list[RunMatch]]] = {}
    for match in selected:
        key = id(match.paragraph)
        by_paragraph.setdefault(key, (match.paragraph, []))[1].append(match)
    for paragraph, paragraph_matches in by_paragraph.values():
        runs = list(paragraph.runs)
        for match in sorted(
            paragraph_matches,
            key=lambda item: (item.first_run, item.first_offset),
            reverse=True,
        ):
            first = runs[match.first_run]
            if match.first_run == match.last_run:
                first.text = (
                    first.text[: match.first_offset] + replacement + first.text[match.last_offset :]
                )
                continue
            last = runs[match.last_run]
            first.text = first.text[: match.first_offset] + replacement
            last.text = last.text[match.last_offset :]
            for run_index in range(match.first_run + 1, match.last_run):
                runs[run_index].text = ""
    return len(selected), warnings


def _update_table(presentation: Any, operation: dict[str, Any]) -> int:
    _reject_unknown(
        operation,
        {"action", "slide", "shape", "data", "cells"},
        "update_table operation",
    )
    _, shape = _select_one_shape_across_scope(presentation, operation)
    if not shape.has_table:
        raise ToolError("ambiguous_edit", "selected shape is not a table")
    table = shape.table
    if ("data" in operation) == ("cells" in operation):
        raise ToolError("bad_input", "update_table requires exactly one of data or cells")
    updates = 0
    if "data" in operation:
        data = _require_list(operation["data"], "update_table.data")
        rows = [_require_list(row, f"update_table.data[{index}]") for index, row in enumerate(data)]
        expected = (len(table.rows), len(table.columns))
        actual = (len(rows), len(rows[0]) if rows else 0)
        if not rows or any(len(row) != actual[1] for row in rows) or actual != expected:
            raise ToolError(
                "ambiguous_edit",
                "replacement table data must match the existing dimensions exactly",
                {"expected": expected, "actual": actual},
            )
        for row_index, row in enumerate(rows):
            for column_index, value in enumerate(row):
                if not isinstance(value, (str, int, float, bool)) and value is not None:
                    raise ToolError("bad_input", "table cells must contain JSON scalar values")
                table.cell(row_index, column_index).text = "" if value is None else str(value)
                updates += 1
    else:
        cells = _require_list(operation["cells"], "update_table.cells")
        seen: set[tuple[int, int]] = set()
        for index, value in enumerate(cells):
            cell_spec = _require_mapping(value, f"update_table.cells[{index}]")
            _reject_unknown(
                cell_spec,
                {"row", "column", "text"},
                f"update_table.cells[{index}]",
            )
            row = cell_spec.get("row")
            column = cell_spec.get("column")
            if (
                not isinstance(row, int)
                or isinstance(row, bool)
                or not isinstance(column, int)
                or isinstance(column, bool)
                or not 0 <= row < len(table.rows)
                or not 0 <= column < len(table.columns)
            ):
                raise ToolError("bad_input", "table cell coordinates are out of range")
            coordinate = (row, column)
            if coordinate in seen:
                raise ToolError("bad_input", "table cell is updated more than once")
            seen.add(coordinate)
            table.cell(row, column).text = _require_string(
                cell_spec.get("text"),
                f"update_table.cells[{index}].text",
                allow_empty=True,
            )
            updates += 1
    return updates


def _chart_type_name(value: Any) -> str | None:
    for name, enum_value in CHART_TYPES.items():
        if value == enum_value:
            return name
    return None


def _update_chart(presentation: Any, operation: dict[str, Any]) -> int:
    _reject_unknown(
        operation,
        {
            "action",
            "slide",
            "shape",
            "chart_type",
            "categories",
            "series",
            "title",
            "has_legend",
        },
        "update_chart operation",
    )
    _, shape = _select_one_shape_across_scope(presentation, operation)
    if not shape.has_chart:
        raise ToolError("ambiguous_edit", "selected shape is not a chart")
    chart_type, data, points = _chart_data(operation, "update_chart")
    actual_type = _chart_type_name(shape.chart.chart_type)
    if actual_type != chart_type:
        raise ToolError(
            "unsupported_operation",
            "update_chart cannot change an existing chart type",
            {
                "existing": actual_type or _enum_name(shape.chart.chart_type),
                "requested": chart_type,
            },
        )
    shape.chart.replace_data(data)
    if "title" in operation:
        shape.chart.has_title = True
        shape.chart.chart_title.text_frame.text = _require_string(
            operation["title"], "update_chart.title", allow_empty=True
        )
    if "has_legend" in operation:
        if not isinstance(operation["has_legend"], bool):
            raise ToolError("bad_input", "update_chart.has_legend must be boolean")
        shape.chart.has_legend = operation["has_legend"]
    return points


def _replace_image(
    presentation: Any,
    operation: dict[str, Any],
    base_dir: Path,
) -> int:
    _reject_unknown(
        operation,
        {"action", "slide", "shape", "path"},
        "replace_image operation",
    )
    slide_index, shape = _select_one_shape_across_scope(presentation, operation)
    if shape.shape_type != MSO_SHAPE_TYPE.PICTURE:
        raise ToolError("ambiguous_edit", "selected shape is not a picture")
    slide = presentation.slides[slide_index]
    image_path = _resolve_resource(base_dir, operation.get("path"), "replace_image.path")
    image = _validate_image(image_path)
    if image.content_type != shape.image.content_type:
        raise ToolError(
            "unsupported_operation",
            "replacement image must use the existing image content type",
            {"existing": shape.image.content_type, "replacement": image.content_type},
        )
    before_hashes = _picture_hashes(presentation)
    selected_key = (int(slide.slide_id), int(shape.shape_id))
    if selected_key not in before_hashes:
        raise ToolError("internal_error", "selected picture was not present in the picture map")
    OoxmlAdapter.replace_picture_relationship(shape, image.data)
    after_hashes = _picture_hashes(presentation)
    if set(after_hashes) != set(before_hashes) or after_hashes.get(selected_key) != image.sha256:
        raise ToolError(
            "post_write_validation",
            "selected picture relationship replacement could not be verified",
        )
    changed_unselected = [
        key
        for key, digest in before_hashes.items()
        if key != selected_key and after_hashes.get(key) != digest
    ]
    if changed_unselected:
        raise ToolError(
            "post_write_validation",
            "picture replacement changed an unselected picture",
            {"count": len(changed_unselected)},
        )
    return 1


def _operation_slide_spec(operation: dict[str, Any]) -> dict[str, Any]:
    return _require_mapping(operation.get("slide"), "add_slide.slide")


def edit_pptx(
    source: Path,
    job: dict[str, Any],
    base_dir: Path,
    destination: Path,
    overwrite: bool,
) -> dict[str, Any]:
    _reject_unknown(job, {"schema_version", "operation", "operations"}, "edit job")
    with _pptx_source_snapshot(source) as snapshot:
        source_report = snapshot.report
        source_hash = snapshot.sha256
        presentation = _open_presentation(snapshot.snapshot_path)
    source = snapshot.original_path
    destination = _prepare_destination(destination, ".pptx", overwrite, source)
    operations = _require_list(job.get("operations"), "operations")
    if not operations:
        raise ToolError("bad_input", "edit job must contain at least one operation")
    warnings = list(source_report.warnings)
    counts = {
        "operations": 0,
        "slides_added": 0,
        "slides_removed": 0,
        "slides_reordered": 0,
        "text_replacements": 0,
        "table_cells_updated": 0,
        "chart_points_updated": 0,
        "images_replaced": 0,
        "notes_updated": 0,
    }
    checkpoints: list[dict[str, Any]] = []
    graph_ordinal = 0
    with tempfile.TemporaryDirectory(prefix="pptx-edit-checkpoints-") as temp_dir:
        checkpoint_dir = Path(temp_dir)
        for operation_index, value in enumerate(operations):
            operation = _require_mapping(value, f"operations[{operation_index}]")
            action = _require_string(
                operation.get("action"), f"operations[{operation_index}].action"
            )
            if action == "add_slide":
                _reject_unknown(operation, {"action", "slide", "index"}, "add_slide operation")
                before = _presentation_slide_ids(presentation)
                slide, _ = _add_slide(
                    presentation,
                    _operation_slide_spec(operation),
                    base_dir,
                )
                counts["slides_added"] += 1
                expected = _presentation_slide_ids(presentation)
                if "index" in operation:
                    index = operation["index"]
                    if (
                        not isinstance(index, int)
                        or isinstance(index, bool)
                        or not 0 <= index < len(presentation.slides)
                    ):
                        raise ToolError("bad_input", "add_slide.index is out of range")
                    if index != len(presentation.slides) - 1:
                        expected = expected[:]
                        expected.remove(int(slide.slide_id))
                        expected.insert(index, int(slide.slide_id))
                        OoxmlAdapter.reorder_slides(presentation, expected)
                        counts["slides_reordered"] += 1
                graph_ordinal += 1
                presentation, checkpoint = _checkpoint_graph_mutation(
                    presentation,
                    checkpoint_dir,
                    graph_ordinal,
                    before,
                    expected,
                )
                checkpoints.append(checkpoint)
            elif action == "replace_text":
                replacements, replacement_warnings = _replace_text(presentation, operation)
                counts["text_replacements"] += replacements
                warnings.extend(replacement_warnings)
            elif action == "update_table":
                counts["table_cells_updated"] += _update_table(presentation, operation)
            elif action == "update_chart":
                counts["chart_points_updated"] += _update_chart(presentation, operation)
            elif action == "replace_image":
                counts["images_replaced"] += _replace_image(presentation, operation, base_dir)
            elif action == "set_notes":
                _reject_unknown(operation, {"action", "slide", "text"}, "set_notes operation")
                _, slide = _select_one_slide(presentation, operation.get("slide"))
                _set_notes(slide, operation.get("text"), "set_notes.text")
                counts["notes_updated"] += 1
            elif action == "remove_slide":
                _reject_unknown(operation, {"action", "slide"}, "remove_slide operation")
                if len(presentation.slides) <= 1:
                    raise ToolError(
                        "unsupported_operation",
                        "removing the final slide is unsupported",
                    )
                index, _ = _select_one_slide(presentation, operation.get("slide"))
                before = _presentation_slide_ids(presentation)
                expected = before[:index] + before[index + 1 :]
                OoxmlAdapter.remove_slide(presentation, index)
                graph_ordinal += 1
                presentation, checkpoint = _checkpoint_graph_mutation(
                    presentation,
                    checkpoint_dir,
                    graph_ordinal,
                    before,
                    expected,
                )
                checkpoints.append(checkpoint)
                counts["slides_removed"] += 1
            elif action == "reorder_slides":
                _reject_unknown(
                    operation,
                    {"action", "slide_ids"},
                    "reorder_slides operation",
                )
                ordered = _require_list(operation.get("slide_ids"), "reorder_slides.slide_ids")
                if not all(
                    isinstance(item, int) and not isinstance(item, bool) for item in ordered
                ):
                    raise ToolError("bad_input", "reorder slide_ids must be integers")
                before = _presentation_slide_ids(presentation)
                if len(set(ordered)) != len(ordered) or set(ordered) != set(before):
                    raise ToolError(
                        "ambiguous_edit",
                        "reorder slide_ids must contain every current slide ID exactly once",
                        {"current": before, "requested": ordered},
                    )
                OoxmlAdapter.reorder_slides(presentation, ordered)
                graph_ordinal += 1
                presentation, checkpoint = _checkpoint_graph_mutation(
                    presentation,
                    checkpoint_dir,
                    graph_ordinal,
                    before,
                    ordered,
                )
                checkpoints.append(checkpoint)
                counts["slides_reordered"] += 1
            else:
                raise ToolError(
                    "unsupported_operation",
                    "unsupported edit action",
                    {"action": action},
                )
            counts["operations"] += 1
        expected_ids = _presentation_slide_ids(presentation)
        output_report, post_publish_source_changed = _save_atomic_presentation(
            presentation,
            destination,
            expected_ids,
            overwrite,
            source,
            source_hash,
        )
    warnings.extend(output_report.warnings)
    if post_publish_source_changed:
        warnings.append(
            "source path changed after destination publication; the published deck was built "
            "from the validated immutable snapshot"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "success": True,
        "operation": "edit",
        "source": str(source.resolve()),
        "output": str(destination),
        "versions": _versions(),
        "counts": counts,
        "warnings": sorted(set(warnings)),
        "verification": {
            "atomic_publish": True,
            "package_preflight": True,
            "package_reopen": True,
            "internal_relationship_targets_exist": True,
            "owner_relationship_references_resolved": True,
            "source_unchanged_at_publish_gate": True,
            "post_publish_source_changed": post_publish_source_changed,
            "retained_slide_ids": [
                slide_id for slide_id in source_report.slide_ids if slide_id in expected_ids
            ],
            "slide_count": len(expected_ids),
            "slide_ids": expected_ids,
            "graph_checkpoints": checkpoints,
        },
    }


def _find_libreoffice() -> str:
    executable = shutil.which("soffice") or shutil.which("libreoffice")
    if executable is None:
        raise ToolError(
            "external_precondition",
            "LibreOffice executable was not found on PATH",
            {"executables": ["soffice", "libreoffice"]},
        )
    return executable


def _validate_pdf(path: Path, expected_pages: int) -> tuple[int, list[str]]:
    try:
        with path.open("rb") as source:
            if source.read(5) != b"%PDF-":
                raise ToolError(
                    "external_tool_failure",
                    "LibreOffice output does not have a PDF signature",
                )
        reader = pypdf.PdfReader(str(path), strict=True)
        pages = len(reader.pages)
        warnings = list(reader.metadata.keys()) if reader.metadata else []
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(
            "external_tool_failure",
            "LibreOffice PDF output could not be opened",
        ) from exc
    if pages != expected_pages:
        raise ToolError(
            "post_write_validation",
            "converted PDF page count does not equal the PPTX slide count",
            {"slides": expected_pages, "pages": pages},
        )
    return pages, warnings


def _libreoffice_environment(work: Path) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "SYSTEMROOT", "WINDIR"}
    }
    environment.update(
        {
            "HOME": str(work),
            "TMPDIR": str(work),
            "TMP": str(work),
            "TEMP": str(work),
            "XDG_CACHE_HOME": str(work / "cache"),
            "XDG_CONFIG_HOME": str(work / "config"),
        }
    )
    return environment


def _redact_diagnostic(text: str, paths: list[Path]) -> str:
    redacted = text[-4000:]
    replacements: set[str] = set()
    for path in paths:
        try:
            resolved = path.resolve()
            replacements.update(
                {
                    str(path),
                    str(resolved),
                    resolved.as_uri(),
                    path.name,
                }
            )
        except (OSError, ValueError):
            replacements.add(str(path))
    for value in sorted(replacements, key=len, reverse=True):
        if value:
            redacted = redacted.replace(value, "<redacted-path>")
    redacted = re.sub(
        r"(?i)\b[a-z][a-z0-9+.-]*://[^\s<>'\"]+",
        lambda match: _sanitize_url(match.group(0)),
        redacted,
    )
    redacted = re.sub(
        r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s@]+@",
        r"\1<redacted-userinfo>@",
        redacted,
    )
    redacted = re.sub(
        r"(?i)\b(api[_-]?key|token|password|passwd|secret|authorization|session[_-]?id)"
        r"\s*[:=]\s*[^\s&;,]+",
        r"\1=<redacted>",
        redacted,
    )
    return redacted


def _convert_pptx_snapshot(
    snapshot: PptxSourceSnapshot,
    destination: Path,
    overwrite: bool,
    timeout: float,
) -> dict[str, Any]:
    if not math.isfinite(timeout) or not 1 <= timeout <= 1800:
        raise ToolError("bad_input", "timeout must be between 1 and 1800 seconds")
    source = snapshot.original_path
    source_report = snapshot.report
    if source_report.external_relationships:
        raise ToolError(
            "unsupported_operation",
            "conversion rejects presentations containing external relationships",
            {"count": source_report.external_relationships},
        )
    source_hash = snapshot.sha256
    destination = _prepare_destination(destination, ".pdf", overwrite, source)
    executable = _find_libreoffice()
    temporary = _temporary_sibling(destination, ".pdf")
    stdout = ""
    stderr = ""
    office_version = "unknown"
    try:
        with tempfile.TemporaryDirectory(prefix="pptx-libreoffice-") as temp_dir:
            work = Path(temp_dir)
            output_dir = work / "output"
            profile = work / "profile"
            output_dir.mkdir()
            profile.mkdir()
            environment = _libreoffice_environment(work)
            command = [
                executable,
                "--headless",
                "--nologo",
                "--nodefault",
                "--nolockcheck",
                "--nofirststartwizard",
                f"-env:UserInstallation={profile.resolve().as_uri()}",
                "--convert-to",
                "pdf",
                "--outdir",
                str(output_dir),
                str(snapshot.snapshot_path),
            ]
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=environment,
                )
            except subprocess.TimeoutExpired as exc:
                raise ToolError(
                    "external_tool_failure",
                    "LibreOffice conversion timed out",
                    {"timeout_seconds": timeout},
                ) from exc
            except OSError as exc:
                raise ToolError(
                    "external_tool_failure",
                    "LibreOffice could not be launched",
                ) from exc
            diagnostic_paths = [
                source,
                snapshot.snapshot_path,
                destination,
                temporary,
                work,
                output_dir,
                profile,
            ]
            stdout = _redact_diagnostic(completed.stdout, diagnostic_paths)
            stderr = _redact_diagnostic(completed.stderr, diagnostic_paths)
            if completed.returncode != 0:
                raise ToolError(
                    "external_tool_failure",
                    "LibreOffice conversion failed",
                    {
                        "returncode": completed.returncode,
                        "stdout": stdout,
                        "stderr": stderr,
                    },
                )
            generated = output_dir / f"{source.stem}.pdf"
            if not generated.is_file():
                candidates = sorted(output_dir.glob("*.pdf"))
                if len(candidates) != 1:
                    raise ToolError(
                        "external_tool_failure",
                        "LibreOffice did not produce exactly one PDF",
                        {"outputs": [path.name for path in candidates]},
                    )
                generated = candidates[0]
            shutil.copyfile(generated, temporary)
            try:
                version_result = subprocess.run(
                    [executable, "--version"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    env=environment,
                )
                office_version = _redact_diagnostic(
                    version_result.stdout.strip(),
                    diagnostic_paths,
                )
            except (OSError, subprocess.TimeoutExpired):
                office_version = "unknown"
        pages, _ = _validate_pdf(temporary, len(source_report.slide_ids))
        if not _source_hash_matches(source, source_hash):
            raise ToolError(
                "post_write_validation",
                "source changed before the destination publication gate",
            )
        _atomic_publish(temporary, destination, overwrite)
        post_publish_source_changed = not _source_hash_matches(source, source_hash)
    finally:
        temporary.unlink(missing_ok=True)
    warnings = list(source_report.warnings)
    if stderr.strip():
        warnings.append("LibreOffice emitted diagnostics; inspect conversion.diagnostics")
    if post_publish_source_changed:
        warnings.append(
            "source path changed after destination publication; the published PDF was built "
            "from the validated immutable snapshot"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "success": True,
        "operation": "convert",
        "source": str(source.resolve()),
        "output": str(destination),
        "versions": {**_versions(), "libreoffice": office_version or "unknown"},
        "counts": {
            "source_slides": len(source_report.slide_ids),
            "pdf_pages": pages,
        },
        "warnings": sorted(set(warnings)),
        "conversion": {
            "engine": Path(executable).name,
            "timeout_seconds": timeout,
            "diagnostics": {"stdout": stdout, "stderr": stderr},
        },
        "verification": {
            "atomic_publish": True,
            "source_unchanged_at_publish_gate": True,
            "post_publish_source_changed": post_publish_source_changed,
            "package_preflight": True,
            "immutable_source_snapshot": True,
            "internal_relationship_targets_exist": True,
            "owner_relationship_references_resolved": True,
            "pdf_signature": True,
            "pdf_openable": True,
            "source_slide_count": len(source_report.slide_ids),
            "pdf_page_count": pages,
        },
    }


def convert_pptx(
    source: Path,
    destination: Path,
    overwrite: bool,
    timeout: float,
) -> dict[str, Any]:
    with _pptx_source_snapshot(source) as snapshot:
        return _convert_pptx_snapshot(snapshot, destination, overwrite, timeout)


def _build_parser() -> JsonArgumentParser:
    parser = JsonArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="inspect a PPTX")
    inspect_parser.add_argument("source", type=Path)

    create_parser = subparsers.add_parser("create", help="create a PPTX from a JSON job")
    create_parser.add_argument("--job", required=True, help="schema-version-1 JSON file or -")
    create_parser.add_argument("--output", required=True, type=Path)
    create_parser.add_argument("--overwrite", action="store_true")

    edit_parser = subparsers.add_parser("edit", help="edit a PPTX from a JSON job")
    edit_parser.add_argument("source", type=Path)
    edit_parser.add_argument("--job", required=True, help="schema-version-1 JSON file or -")
    edit_parser.add_argument("--output", required=True, type=Path)
    edit_parser.add_argument("--overwrite", action="store_true")

    convert_parser = subparsers.add_parser("convert", help="convert PPTX to PDF")
    convert_parser.add_argument("source", type=Path)
    convert_parser.add_argument("--output", required=True, type=Path)
    convert_parser.add_argument("--overwrite", action="store_true")
    convert_parser.add_argument("--timeout", type=float, default=120.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = _build_parser().parse_args(argv)
        if args.command == "inspect":
            result = inspect_pptx(args.source)
        elif args.command == "create":
            job, base_dir = _load_job(args.job, "create")
            result = create_pptx(job, base_dir, args.output, args.overwrite)
        elif args.command == "edit":
            job, base_dir = _load_job(args.job, "edit")
            result = edit_pptx(
                args.source,
                job,
                base_dir,
                args.output,
                args.overwrite,
            )
        elif args.command == "convert":
            result = convert_pptx(
                args.source,
                args.output,
                args.overwrite,
                args.timeout,
            )
        else:
            raise ToolError("unsupported_operation", "unsupported command")
        _json_dump(result)
        return 0
    except ToolError as exc:
        _json_dump(
            {
                "schema_version": SCHEMA_VERSION,
                "success": False,
                "error": {
                    "category": exc.category,
                    "message": exc.message,
                    "details": exc.details,
                },
            },
            sys.stderr,
        )
        return exc.exit_code
    except BrokenPipeError:
        return 0
    except KeyboardInterrupt:
        _json_dump(
            {
                "schema_version": SCHEMA_VERSION,
                "success": False,
                "error": {
                    "category": "internal_error",
                    "message": "operation interrupted",
                    "details": {},
                },
            },
            sys.stderr,
        )
        return EXIT_CODES["internal_error"]
    except Exception as exc:
        _json_dump(
            {
                "schema_version": SCHEMA_VERSION,
                "success": False,
                "error": {
                    "category": "internal_error",
                    "message": "unexpected internal failure",
                    "details": {"type": type(exc).__name__},
                },
            },
            sys.stderr,
        )
        return EXIT_CODES["internal_error"]


if __name__ == "__main__":
    raise SystemExit(main())
