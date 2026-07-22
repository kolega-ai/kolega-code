#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise the public DOCX CLI entirely inside a temporary directory."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import zipfile
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject, NumberObject

CLI_TIMEOUT_SECONDS = 30
MAX_PDF_BYTES = 64 * 1024 * 1024
MAX_PDF_PAGES = 1_000
MAX_PDF_DECOMPRESSED_STREAM_BYTES = 8 * 1024 * 1024


def png_chunk(kind: bytes, data: bytes) -> bytes:
    """Encode one PNG chunk."""

    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def make_png(path: Path, rgb: tuple[int, int, int]) -> None:
    """Generate a dependency-free 12x12 RGB PNG fixture."""

    width = height = 12
    row = b"\x00" + bytes(rgb) * width
    raw = row * height
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(raw, 9))
        + png_chunk(b"IEND", b"")
    )
    path.write_bytes(payload)


def run_cli(
    tool: Path,
    *arguments: str,
    expected_status: int = 0,
    timeout: int = CLI_TIMEOUT_SECONDS,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Invoke the public CLI and parse its machine-readable output."""

    try:
        completed = subprocess.run(
            [sys.executable, str(tool), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, **(environment or {})},
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            "CLI exceeded the smoke-test timeout.\n"
            f"command: {arguments}\n"
            f"timeout: {timeout}\n"
            f"stdout: {exc.stdout}\n"
            f"stderr: {exc.stderr}"
        ) from exc
    if completed.returncode != expected_status:
        raise AssertionError(
            "CLI returned an unexpected status.\n"
            f"command: {arguments}\n"
            f"expected: {expected_status}\n"
            f"actual: {completed.returncode}\n"
            f"stdout: {completed.stdout}\n"
            f"stderr: {completed.stderr}"
        )
    channel = completed.stdout if expected_status == 0 else completed.stderr
    lines = [line for line in channel.splitlines() if line.strip()]
    if expected_status == 0:
        payload = json.loads(completed.stdout)
    else:
        payload = json.loads(lines[-1])
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a deterministic fixture job."""

    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def add_external_relationship(source: Path, destination: Path) -> None:
    """Copy a DOCX while adding one external hyperlink relationship."""

    relationship_part = "word/_rels/document.xml.rels"
    namespace = "http://schemas.openxmlformats.org/package/2006/relationships"
    with (
        zipfile.ZipFile(source) as source_zip,
        zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as target_zip,
    ):
        for info in source_zip.infolist():
            data = source_zip.read(info)
            if info.filename == relationship_part:
                root = ElementTree.fromstring(data)
                ElementTree.SubElement(
                    root,
                    f"{{{namespace}}}Relationship",
                    {
                        "Id": "rIdSmokeExternal",
                        "Type": (
                            "http://schemas.openxmlformats.org/officeDocument/2006/"
                            "relationships/hyperlink"
                        ),
                        "Target": "https://example.invalid/docx-smoke",
                        "TargetMode": "External",
                    },
                )
                data = ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)
            target_zip.writestr(info, data)


def add_custom_xml_part(source: Path, destination: Path) -> None:
    """Copy a DOCX while adding a nonstandard custom XML part."""

    with (
        zipfile.ZipFile(source) as source_zip,
        zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as target_zip,
    ):
        for info in source_zip.infolist():
            target_zip.writestr(info, source_zip.read(info))
        target_zip.writestr("customXml/smoke-extra.xml", b"<smoke>extra</smoke>")


def rewrite_font_references(
    source: Path, destination: Path, font_name: str, *, add_decoys: bool = False
) -> None:
    """Rewrite real font references and optionally add misleading non-font attributes."""

    word_namespace = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    font_attributes = tuple(
        f"{{{word_namespace}}}{name}" for name in ("ascii", "hAnsi", "eastAsia", "cs")
    )
    with (
        zipfile.ZipFile(source) as source_zip,
        zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as target_zip,
    ):
        for info in source_zip.infolist():
            data = source_zip.read(info)
            if info.filename.startswith("word/") and info.filename.endswith(".xml"):
                root = ElementTree.fromstring(data)
                for element in root.iter(f"{{{word_namespace}}}rFonts"):
                    for attribute in font_attributes:
                        if element.get(attribute) is not None:
                            element.set(attribute, font_name)
                if info.filename.startswith("word/theme/"):
                    for element in root.iter():
                        if element.get("typeface") is not None:
                            element.set("typeface", font_name)
                if add_decoys and info.filename == "word/settings.xml":
                    ElementTree.SubElement(
                        root,
                        f"{{{word_namespace}}}lang",
                        {f"{{{word_namespace}}}eastAsia": "ja-JP"},
                    )
                    ElementTree.SubElement(
                        root,
                        f"{{{word_namespace}}}decimalSymbol",
                        {
                            f"{{{word_namespace}}}val": ".",
                            f"{{{word_namespace}}}ascii": ".",
                            f"{{{word_namespace}}}hAnsi": ",",
                        },
                    )
                data = ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)
            target_zip.writestr(info, data)


def write_pdf(path: Path, pages: int) -> None:
    """Write a deterministic blank PDF fixture."""

    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=72, height=72)
    with path.open("wb") as handle:
        writer.write(handle)


def write_compressed_stream_pdf(path: Path) -> None:
    """Write a PDF whose content stream exceeds the decompression validation limit."""

    writer = PdfWriter()
    page = writer.add_blank_page(width=72, height=72)
    stream = DecodedStreamObject()
    stream.set_data(b"BT (" + (b"A" * (MAX_PDF_DECOMPRESSED_STREAM_BYTES + 1)) + b") Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream.flate_encode())
    with path.open("wb") as handle:
        writer.write(handle)


def write_image_only_pdf(path: Path) -> None:
    """Write a one-page PDF whose only visible content is an image XObject."""

    writer = PdfWriter()
    page = writer.add_blank_page(width=72, height=72)
    image = DecodedStreamObject()
    image.set_data(b"\x80")
    image.update(
        {
            NameObject("/Type"): NameObject("/XObject"),
            NameObject("/Subtype"): NameObject("/Image"),
            NameObject("/Width"): NumberObject(1),
            NameObject("/Height"): NumberObject(1),
            NameObject("/ColorSpace"): NameObject("/DeviceGray"),
            NameObject("/BitsPerComponent"): NumberObject(8),
        }
    )
    image_reference = writer._add_object(image)
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/XObject"): DictionaryObject({NameObject("/ImSmoke"): image_reference})}
    )
    contents = DecodedStreamObject()
    contents.set_data(b"q 36 0 0 36 18 18 cm /ImSmoke Do Q")
    page[NameObject("/Contents")] = writer._add_object(contents)
    with path.open("wb") as handle:
        writer.write(handle)


def rewrite_theme_tokens(source: Path, destination: Path) -> None:
    """Reference HAnsi theme tokens and give major/minor Latin fonts distinct names."""

    word_namespace = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    drawing_namespace = "http://schemas.openxmlformats.org/drawingml/2006/main"
    with (
        zipfile.ZipFile(source) as source_zip,
        zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as target_zip,
    ):
        for info in source_zip.infolist():
            data = source_zip.read(info)
            if info.filename.startswith("word/") and info.filename.endswith(".xml"):
                root = ElementTree.fromstring(data)
                for fonts in root.iter(f"{{{word_namespace}}}rFonts"):
                    fonts.attrib.clear()
                    fonts.set(f"{{{word_namespace}}}hAnsiTheme", "majorHAnsi")
                    fonts.set(f"{{{word_namespace}}}asciiTheme", "minorHAnsi")
                if info.filename.startswith("word/theme/"):
                    major = root.find(f".//{{{drawing_namespace}}}majorFont")
                    minor = root.find(f".//{{{drawing_namespace}}}minorFont")
                    if major is not None:
                        major.find(f"{{{drawing_namespace}}}latin").set("typeface", "Major Smoke")
                    if minor is not None:
                        minor.find(f"{{{drawing_namespace}}}latin").set("typeface", "Minor Smoke")
                data = ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)
            target_zip.writestr(info, data)


def make_fake_soffice(path: Path) -> None:
    """Create a controllable `soffice` executable for bounded conversion tests."""

    path.write_text(
        """#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

if "--version" in sys.argv:
    print("LibreOffice smoke fake 1.0")
    raise SystemExit(0)

mode = os.environ.get("DOCX_SMOKE_SOFFICE_MODE", "copy")
outdir = Path(sys.argv[sys.argv.index("--outdir") + 1])
source = Path(sys.argv[-1])
output = outdir / f"{source.stem}.pdf"
if mode == "timeout":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    Path(os.environ["DOCX_SMOKE_CHILD_PID"]).write_text(str(child.pid), encoding="utf-8")
    time.sleep(300)
elif mode == "oversized":
    with output.open("wb") as handle:
        handle.write(b"%PDF-1.7\\n")
        handle.truncate(int(os.environ["DOCX_SMOKE_OVERSIZED_BYTES"]))
elif mode == "malformed":
    output.write_bytes(b"not a pdf")
else:
    shutil.copyfile(os.environ["DOCX_SMOKE_PDF_FIXTURE"], output)
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def assert_process_gone(pid: int) -> None:
    """Wait briefly for an expected process-group descendant to disappear."""

    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    raise AssertionError(f"Timed-out LibreOffice descendant survived: pid={pid}")


def inline_image_sha256(document: Any, index: int) -> str:
    """Resolve an inline image occurrence and hash its related image part."""

    shape = document.inline_shapes[index]
    blip = next(shape._inline.iter(qn("a:blip")))
    relationship_id = blip.get(qn("r:embed"))
    return hashlib.sha256(document.part.related_parts[relationship_id].blob).hexdigest()


def assert_docx_structure(path: Path, replacement_image: Path) -> None:
    """Reopen and verify structure plus the declared first-run policy."""

    document = Document(str(path))
    assert document.paragraphs[0].text == "DOCX Smoke Test"
    styled = next(p for p in document.paragraphs if "AlpXta" in p.text)
    assert styled.style.name == "Quote"
    assert styled.runs[0].text == "AlpX"
    assert styled.runs[0].bold is True
    assert styled.runs[1].text == "ta"
    assert styled.runs[1].italic is True
    assert any(paragraph.text == "Done Done" for paragraph in document.paragraphs)
    table = document.tables[0]
    assert table.cell(1, 1).text == "Updated"
    assert table.autofit is False
    assert [round(column.width.inches, 2) for column in table.columns] == [1.4, 4.6]
    assert all(row._tr.get_or_add_trPr().find(qn("w:cantSplit")) is not None for row in table.rows)
    assert len(document.inline_shapes) == 1
    expected_hash = hashlib.sha256(replacement_image.read_bytes()).hexdigest()
    assert inline_image_sha256(document, 0) == expected_hash
    assert document.sections[0].header.paragraphs[0].text == "Smoke header"
    assert "Page " in document.sections[0].footer.paragraphs[0].text
    instructions = [
        element.text or ""
        for element in document.sections[0].footer._element.iter(qn("w:instrText"))
    ]
    assert any(value.strip() == "PAGE" for value in instructions)


def assert_inspection(payload: dict[str, Any]) -> None:
    """Assert required structures are visible through inspect."""

    assert payload["status"] == "ok"
    assert payload["operation"] == "inspect"
    assert payload["counts"]["paragraphs"] >= 3
    assert payload["counts"]["tables"] == 1
    assert payload["counts"]["inline_images"] == 1
    assert payload["counts"]["fields"] >= 1
    fonts = payload["result"]["fonts"]
    assert fonts["referenced"]
    assert set(fonts["unembedded"]) == set(fonts["referenced"])
    assert fonts["embedded"] == []
    font_warning = next(item for item in payload["warnings"] if item["code"] == "unembedded_fonts")
    assert set(font_warning["fonts"]) == set(fonts["unembedded"])
    assert "PDF font embedding does not make the DOCX portable" in font_warning["message"]
    assert any(item["heading_level"] == 1 for item in payload["result"]["paragraphs"])
    table = payload["result"]["tables"][0]
    assert table["layout"] == "fixed"
    assert table["column_widths_inches"] == [1.4, 4.6]
    assert table["row_allows_split"] == [False, False]
    assert any(
        story["paragraphs"][0]["text"] == "Smoke header"
        for story in payload["result"]["headers"]
        if story["paragraphs"]
    )


def run_font_portability_regressions(tool: Path, root: Path, source: Path) -> None:
    """Exclude non-font attributes and distinguish allowed from custom fonts."""

    for index, font_name in enumerate(("aRiAl", "Times New Roman", "Courier New")):
        allowed = root / f"allowed-font-{index}.docx"
        rewrite_font_references(source, allowed, font_name, add_decoys=index == 0)
        payload = run_cli(tool, "inspect", "--input", str(allowed))
        fonts = payload["result"]["fonts"]
        assert fonts["referenced"] == [font_name]
        assert "ja-JP" not in fonts["referenced"]
        assert "." not in fonts["referenced"]
        assert "," not in fonts["referenced"]
        assert not any(
            item["code"] == "nonportable_unembedded_fonts" for item in payload["warnings"]
        )

    common = root / "common-font.docx"
    rewrite_font_references(source, common, "Calibri")
    payload = run_cli(tool, "inspect", "--input", str(common))
    assert payload["result"]["fonts"]["referenced"] == ["Calibri"]
    common_warning = next(
        item for item in payload["warnings"] if item["code"] == "common_unembedded_fonts"
    )
    assert common_warning["fonts"] == ["Calibri"]
    assert "RELEASE BLOCKER" not in common_warning["message"]
    assert not any(item["code"] == "nonportable_unembedded_fonts" for item in payload["warnings"])

    custom = root / "custom-font.docx"
    rewrite_font_references(source, custom, "Avenir")
    payload = run_cli(tool, "inspect", "--input", str(custom))
    assert payload["result"]["fonts"]["referenced"] == ["Avenir"]
    blocker = next(
        item for item in payload["warnings"] if item["code"] == "nonportable_unembedded_fonts"
    )
    assert blocker["fonts"] == ["Avenir"]
    assert blocker["message"].startswith("RELEASE BLOCKER:")
    assert not any(item["code"] == "common_unembedded_fonts" for item in payload["warnings"])
    assert any(item["code"] == "unembedded_fonts" for item in payload["warnings"])


def run_pdf_check(tool: Path, source: Path, output: Path) -> dict[str, Any]:
    """Convert through LibreOffice and verify page count and content anchors."""

    payload = run_cli(
        tool,
        "convert",
        "--input",
        str(source),
        "--format",
        "pdf",
        "--output",
        str(output),
        timeout=120,
    )
    reader = PdfReader(str(output))
    assert len(reader.pages) == 1
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    for anchor in ("DOCX Smoke Test", "Updated"):
        assert anchor in text, f"Missing PDF text anchor: {anchor!r}"
    assert payload["verification"]["page_count"] == 1
    return {
        "status": "passed",
        "page_count": len(reader.pages),
        "anchors": ["DOCX Smoke Test", "Updated"],
    }


def run_quality_upgrade_regressions(
    tool: Path, root: Path, image: Path, *, libreoffice: bool
) -> dict[str, Any]:
    """Exercise section, layout, style, table, field, and accessibility upgrades."""

    spec = root / "quality-create.json"
    created = root / "quality-created.docx"
    write_json(
        spec,
        {
            "schema_version": 1,
            "content": {
                "metadata": {"title": "Quality fixture", "language": "en-US"},
                "section": {
                    "paper_size": "letter",
                    "margin_left_inches": 1,
                    "different_first_page": True,
                    "page_number_start": 3,
                    "page_number_format": "lowerRoman",
                },
                "styles": [
                    {
                        "name": "Smoke Body",
                        "type": "paragraph",
                        "based_on": "Normal",
                        "font": "Arial",
                        "size_pt": 10,
                        "paragraph": {
                            "alignment": "justify",
                            "space_after_pt": 6,
                            "widow_control": True,
                        },
                        "outline_level": 0,
                    },
                    {
                        "name": "Smoke Emphasis",
                        "type": "character",
                        "based_on": "Default Paragraph Font",
                        "italic": True,
                    },
                ],
                "update_fields_on_open": True,
                "blocks": [
                    {"type": "heading", "level": 1, "text": "Quality"},
                    {
                        "type": "paragraph",
                        "style": "Smoke Body",
                        "text": "Pagination controls",
                        "space_before_pt": 4,
                        "space_after_pt": 8,
                        "line_spacing": "one_and_half",
                        "left_indent_inches": 0.2,
                        "first_line_indent_inches": 0.1,
                        "keep_with_next": True,
                        "keep_together": True,
                        "page_break_before": False,
                        "widow_control": True,
                        "tab_stops": [
                            {
                                "position_inches": 2,
                                "alignment": "right",
                                "leader": "dots",
                            }
                        ],
                    },
                    {
                        "type": "table",
                        "style": "Table Grid",
                        "layout": "fixed",
                        "column_widths_inches": [2, 3],
                        "header_rows": 1,
                        "row_allow_split": [False, True],
                        "rows": [["Name", "Value"], ["A", "B"]],
                    },
                    {
                        "type": "image",
                        "path": image.name,
                        "width_inches": 2,
                        "alt_text": "A red smoke-test square",
                        "title": "Smoke image",
                        "caption": "Figure smoke",
                        "attribution": "Generated fixture",
                    },
                    {
                        "type": "image",
                        "path": image.name,
                        "width_inches": 0.5,
                        "decorative": True,
                    },
                    {"type": "heading", "level": 3, "text": "Skipped heading"},
                    {
                        "type": "table",
                        "rows": [["No", "Header"], ["Data", "Row"]],
                    },
                    {"type": "field", "field": "num_pages"},
                    {"type": "field", "field": "seq", "instruction": " SEQ Figure "},
                    {"type": "field", "field": "toc"},
                    {"type": "page_break"},
                    {"type": "paragraph", "text": "Second-page anchor"},
                ],
                "headers": {"first_page": [{"type": "paragraph", "text": "First"}]},
                "footers": {
                    "default": [{"type": "field", "field": "page_number", "prefix": "Page "}]
                },
            },
        },
    )
    # Catch accidental schema drift in the shared section vocabulary.
    content = json.loads(spec.read_text(encoding="utf-8"))
    content["content"]["section"]["left_margin_inches"] = content["content"]["section"].pop(
        "margin_left_inches"
    )
    write_json(spec, content)
    run_cli(tool, "create", "--spec", str(spec), "--output", str(created))
    inspected = run_cli(tool, "inspect", "--input", str(created))
    result = inspected["result"]
    section = result["sections"][0]
    assert section["paper_size"] == "letter"
    assert section["page_number_start"] == 3
    assert section["page_number_format"] == "lowerRoman"
    assert result["settings"]["update_fields_on_open"] is True
    paragraph = next(item for item in result["paragraphs"] if item["text"] == "Pagination controls")
    assert paragraph["layout"]["keep_with_next"] is True
    assert paragraph["layout"]["tab_stops"][0]["position_inches"] == 2
    table = result["tables"][0]
    assert table["layout"] == "fixed" and table["header_rows"] == 1
    assert table["row_allows_split"] == [False, True]
    occurrence = result["images"]["inline_occurrences"][0]
    assert occurrence["alt_text"] == "A red smoke-test square"
    assert occurrence["decorative"] is False
    assert occurrence["effective_ppi_x"] is not None
    assert result["images"]["inline_occurrences"][1]["decorative"] is True
    assert {field["type"] for field in result["fields"]} >= {"NUMPAGES", "SEQ", "TOC"}
    assert any(item["code"] == "toc_placeholder_only" for item in inspected["warnings"])
    assert not any(
        item["code"] == "informative_image_missing_alt_text" for item in inspected["warnings"]
    )
    assert any(item["code"] == "heading_level_skip" for item in inspected["warnings"])
    assert any(item["code"] == "data_table_without_header_row" for item in inspected["warnings"])
    smoke_style = next(style for style in result["styles"] if style["name"] == "Smoke Body")
    assert smoke_style["paragraph"]["alignment"] == "justify"

    edit_spec = root / "quality-edit.json"
    edited = root / "quality-edited.docx"
    write_json(
        edit_spec,
        {
            "schema_version": 1,
            "operations": [
                {
                    "type": "insert_table",
                    "table": {
                        "rows": [["H"], ["V"]],
                        "style": "Table Grid",
                        "layout": "fixed",
                        "column_widths_inches": [2],
                        "header_rows": 1,
                        "allow_row_split": False,
                    },
                },
                {
                    "type": "update_table",
                    "table_index": 0,
                    "style": "Table Grid",
                    "layout": "fixed",
                    "column_widths_inches": [2.5, 2.5],
                    "header_rows": 1,
                    "row_allow_split": [True, False],
                },
                {
                    "type": "configure_section",
                    "section_index": 0,
                    "paper_size": "a4",
                    "orientation": "portrait",
                    "page_number_start": 1,
                    "page_number_format": "decimal",
                },
                {
                    "type": "upsert_style",
                    "style": {
                        "name": "Smoke Body",
                        "type": "paragraph",
                        "based_on": "Normal",
                        "font": "Arial",
                        "paragraph": {"keep_together": True},
                    },
                },
                {
                    "type": "set_header",
                    "section_index": 0,
                    "kind": "default",
                    "link_to_previous": False,
                    "blocks": [{"type": "paragraph", "text": "Default header"}],
                },
            ],
        },
    )
    run_cli(
        tool, "edit", "--input", str(created), "--spec", str(edit_spec), "--output", str(edited)
    )
    edited_result = run_cli(tool, "inspect", "--input", str(edited))["result"]
    assert edited_result["sections"][0]["paper_size"] == "a4"
    assert edited_result["tables"][0]["row_allows_split"] == [True, False]
    assert edited_result["tables"][2]["header_rows"] == 1
    assert edited_result["headers"][0]["linked_to_previous"] is False

    pdf = {"status": "skipped"}
    if libreoffice:
        pdf_path = root / "quality.pdf"
        payload = run_cli(
            tool,
            "convert",
            "--input",
            str(edited),
            "--format",
            "pdf",
            "--output",
            str(pdf_path),
            timeout=120,
        )
        report = payload["verification"]["content_quality_report"]
        assert payload["verification"]["page_count"] >= 2
        assert len(report["pages"]) >= 2
        assert all("content_anchors" in page for page in report["pages"])
        pdf = {"status": "passed", "pages": payload["verification"]["page_count"]}
    return {"status": "passed", "pdf": pdf}


def run_confirmed_defect_regressions(tool: Path, root: Path, image: Path) -> None:
    """Cover ordering, shared stories, metadata, fields, widths, and section geometry."""

    ordered_spec = root / "ordered-section.json"
    ordered_docx = root / "ordered-section.docx"
    write_json(
        ordered_spec,
        {
            "schema_version": 1,
            "content": {
                "section": {
                    "different_first_page": True,
                    "page_number_start": 2,
                },
                "blocks": [],
            },
        },
    )
    run_cli(tool, "create", "--spec", str(ordered_spec), "--output", str(ordered_docx))
    with zipfile.ZipFile(ordered_docx) as package:
        root_xml = ElementTree.fromstring(package.read("word/document.xml"))
    sect_pr = root_xml.find(
        ".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}sectPr"
    )
    assert sect_pr is not None
    child_names = [child.tag.rsplit("}", 1)[-1] for child in sect_pr]
    assert child_names.index("pgNumType") < child_names.index("titlePg")

    decorative_spec = root / "decorative.json"
    decorative_docx = root / "decorative.docx"
    write_json(
        decorative_spec,
        {
            "schema_version": 1,
            "content": {"blocks": [{"type": "image", "path": image.name, "decorative": True}]},
        },
    )
    run_cli(tool, "create", "--spec", str(decorative_spec), "--output", str(decorative_docx))
    with zipfile.ZipFile(decorative_docx) as package:
        drawing_root = ElementTree.fromstring(package.read("word/document.xml"))
    wp = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
    a = "http://schemas.openxmlformats.org/drawingml/2006/main"
    a14 = "http://schemas.microsoft.com/office/drawing/2010/main"
    doc_pr = drawing_root.find(f".//{{{wp}}}docPr")
    assert doc_pr is not None
    ext_list = doc_pr.find(f"{{{a}}}extLst")
    assert ext_list is not None and len(ext_list) == 1
    extension = ext_list[0]
    assert extension.tag == f"{{{a}}}ext"
    assert extension.get("uri") == "{C183D7F6-B498-43B3-948B-1728B52AA6E4}"
    assert len(extension) == 1
    assert extension[0].tag == f"{{{a14}}}decorative"
    assert extension[0].get("val") == "1"
    assert (
        run_cli(tool, "inspect", "--input", str(decorative_docx))["result"]["images"][
            "inline_occurrences"
        ][0]["decorative"]
        is True
    )

    insert_spec = root / "insert-image-metadata.json"
    inserted_docx = root / "insert-image-metadata.docx"
    write_json(
        insert_spec,
        {
            "schema_version": 1,
            "operations": [
                {
                    "type": "insert_image",
                    "path": image.name,
                    "alt_text": "Inserted smoke image",
                    "title": "Inserted title",
                }
            ],
        },
    )
    run_cli(
        tool,
        "edit",
        "--input",
        str(ordered_docx),
        "--spec",
        str(insert_spec),
        "--output",
        str(inserted_docx),
    )
    inserted = run_cli(tool, "inspect", "--input", str(inserted_docx))["result"]["images"][
        "inline_occurrences"
    ][0]
    assert inserted["alt_text"] == "Inserted smoke image"
    assert inserted["title"] == "Inserted title"
    assert inserted["decorative"] is False

    landscape_source = root / "landscape-source.docx"
    landscape = Document()
    section = landscape.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width, section.page_height = Inches(11), Inches(8.5)
    landscape.save(landscape_source)
    landscape_spec = root / "landscape-paper.json"
    landscape_output = root / "landscape-paper.docx"
    write_json(
        landscape_spec,
        {
            "schema_version": 1,
            "operations": [{"type": "configure_section", "section_index": 0, "paper_size": "a4"}],
        },
    )
    run_cli(
        tool,
        "edit",
        "--input",
        str(landscape_source),
        "--spec",
        str(landscape_spec),
        "--output",
        str(landscape_output),
    )
    landscape_record = run_cli(tool, "inspect", "--input", str(landscape_output))["result"][
        "sections"
    ][0]
    assert landscape_record["orientation"] == "LANDSCAPE"
    assert landscape_record["page_width_inches"] > landscape_record["page_height_inches"]

    for orientation, expect_landscape in (("landscape", True), ("portrait", False)):
        explicit_spec = root / f"landscape-paper-{orientation}.json"
        explicit_output = root / f"landscape-paper-{orientation}.docx"
        write_json(
            explicit_spec,
            {
                "schema_version": 1,
                "operations": [
                    {
                        "type": "configure_section",
                        "section_index": 0,
                        "paper_size": "letter",
                        "orientation": orientation,
                    }
                ],
            },
        )
        run_cli(
            tool,
            "edit",
            "--input",
            str(landscape_source),
            "--spec",
            str(explicit_spec),
            "--output",
            str(explicit_output),
        )
        explicit_record = run_cli(tool, "inspect", "--input", str(explicit_output))["result"][
            "sections"
        ][0]
        assert (explicit_record["page_width_inches"] > explicit_record["page_height_inches"]) is (
            expect_landscape
        )
        assert explicit_record["orientation"] == orientation.upper()

    shared_source = root / "shared-header.docx"
    shared = Document()
    shared.sections[0].header.is_linked_to_previous = False
    shared.sections[0].header.paragraphs[0].text = "Prior story survives"
    shared.add_section(WD_SECTION.NEW_PAGE).header.is_linked_to_previous = True
    shared.save(shared_source)
    shared_spec = root / "shared-header-edit.json"
    shared_output = root / "shared-header-output.docx"
    write_json(
        shared_spec,
        {
            "schema_version": 1,
            "operations": [
                {
                    "type": "set_header",
                    "section_index": 1,
                    "link_to_previous": True,
                    "blocks": [{"type": "paragraph", "text": "Must not overwrite"}],
                }
            ],
        },
    )
    payload = run_cli(
        tool,
        "edit",
        "--input",
        str(shared_source),
        "--spec",
        str(shared_spec),
        "--output",
        str(shared_output),
        expected_status=2,
    )
    assert payload["error"]["category"] == "bad_input"
    assert not shared_output.exists()
    reopened_shared = Document(str(shared_source))
    assert reopened_shared.sections[0].header.paragraphs[0].text == "Prior story survives"
    assert reopened_shared.sections[1].header.is_linked_to_previous is True

    nested_docx = root / "nested-fields.docx"
    nested = Document()
    paragraph = nested.add_paragraph()

    def field_char(kind: str) -> Any:
        run = OxmlElement("w:r")
        char = OxmlElement("w:fldChar")
        char.set(qn("w:fldCharType"), kind)
        run.append(char)
        return run

    def field_text(tag: str, text: str) -> Any:
        run = OxmlElement("w:r")
        node = OxmlElement(tag)
        node.text = text
        run.append(node)
        return run

    for node in (
        field_char("begin"),
        field_text("w:instrText", ' TOC \\o "1-3" '),
        field_char("separate"),
        field_text("w:t", "Entry "),
        field_char("begin"),
        field_text("w:instrText", " PAGEREF _Toc1 "),
        field_char("separate"),
        field_text("w:t", "7"),
        field_char("end"),
        field_text("w:t", " tail"),
        field_char("end"),
    ):
        paragraph._p.append(node)
    nested.save(nested_docx)
    fields = run_cli(tool, "inspect", "--input", str(nested_docx))["result"]["fields"]
    pageref = next(field for field in fields if field["type"] == "PAGEREF")
    toc = next(field for field in fields if field["type"] == "TOC")
    assert pageref["result"] == "7"
    assert toc["result"] == "Entry 7 tail"

    themed_docx = root / "theme-tokens.docx"
    rewrite_theme_tokens(ordered_docx, themed_docx)
    themed_fonts = run_cli(tool, "inspect", "--input", str(themed_docx))["result"]["fonts"]
    assert {"majorHAnsi", "minorHAnsi"} <= set(themed_fonts["theme_tokens_referenced"])
    assert {"Major Smoke", "Minor Smoke"} <= set(themed_fonts["referenced"])

    merged_source = root / "merged-source.docx"
    merged = Document()
    table = merged.add_table(rows=1, cols=3)
    table.cell(0, 0).merge(table.cell(0, 1))
    merged.save(merged_source)
    merged_spec = root / "merged-widths.json"
    merged_output = root / "merged-widths.docx"
    write_json(
        merged_spec,
        {
            "schema_version": 1,
            "operations": [
                {
                    "type": "update_table",
                    "table_index": 0,
                    "layout": "fixed",
                    "column_widths_inches": [1, 2, 3],
                }
            ],
        },
    )
    run_cli(
        tool,
        "edit",
        "--input",
        str(merged_source),
        "--spec",
        str(merged_spec),
        "--output",
        str(merged_output),
    )
    merged_reopened = Document(str(merged_output))
    physical_cells = merged_reopened.tables[0].rows[0]._tr.tc_lst
    assert int(physical_cells[0].tcPr.tcW.w) == 4320
    assert int(physical_cells[1].tcPr.tcW.w) == 4320

    section_image_spec = root / "section-image.json"
    section_image_docx = root / "section-image.docx"
    write_json(
        section_image_spec,
        {
            "schema_version": 1,
            "content": {
                "blocks": [
                    {
                        "type": "image",
                        "path": image.name,
                        "width_inches": 7,
                        "alt_text": "Too wide in first section",
                    },
                    {
                        "type": "section_break",
                        "orientation": "landscape",
                        "left_margin_inches": 0.5,
                        "right_margin_inches": 0.5,
                    },
                    {"type": "paragraph", "text": "Wide second section"},
                ]
            },
        },
    )
    run_cli(tool, "create", "--spec", str(section_image_spec), "--output", str(section_image_docx))
    section_image = run_cli(tool, "inspect", "--input", str(section_image_docx))
    occurrence = section_image["result"]["images"]["inline_occurrences"][0]
    assert occurrence["section_index"] == 0
    warning_item = next(
        item
        for item in section_image["warnings"]
        if item["code"] == "image_exceeds_printable_width"
    )
    assert warning_item["section_index"] == 0


def run_schema_regressions(tool: Path, root: Path, source: Path) -> None:
    """Prove malformed and ambiguous JSON fails closed as bad_input."""

    invalid_create_contents = (
        {"blocks": [], "unknown_key": True},
        {
            "blocks": [
                {
                    "type": "paragraph",
                    "text": "ambiguous",
                    "runs": [{"text": "ambiguous"}],
                }
            ]
        },
        {"blocks": [{"type": "paragraph", "text": 17}]},
        {"blocks": [{"type": "table", "rows": [["A", "B"], ["C"]]}]},
        {"blocks": [{"type": "table", "rows": [["A"], []]}]},
        {
            "blocks": [
                {
                    "type": "table",
                    "rows": [["A", "B"]],
                    "column_widths_inches": [2.0],
                }
            ]
        },
        {
            "blocks": [
                {
                    "type": "table",
                    "rows": [["A"]],
                    "layout": "fluid",
                }
            ]
        },
        {
            "blocks": [
                {
                    "type": "table",
                    "rows": [["A"]],
                    "allow_row_split": "false",
                }
            ]
        },
        {"metadata": {"author": False}, "blocks": []},
        {
            "blocks": [
                {
                    "type": "field",
                    "field": "page_number",
                    "instruction": " PAGE ",
                }
            ]
        },
        {
            "blocks": [
                {
                    "type": "field",
                    "field": "toc",
                    "instruction": "TOC\nINJECTED",
                }
            ]
        },
        {
            "blocks": [
                {
                    "type": "field",
                    "field": "page_number",
                    "style": "Smoke Missing Style",
                }
            ]
        },
        {
            "blocks": [
                {
                    "type": "image",
                    "path": "original.png",
                    "style": "Smoke Missing Style",
                }
            ]
        },
        {"section": {"page_width_inches": 0}, "blocks": []},
        {
            "section": {
                "page_width_inches": 1,
                "left_margin_inches": 0.5,
                "right_margin_inches": 0.5,
            },
            "blocks": [],
        },
        {
            "section": {
                "page_height_inches": 1,
                "top_margin_inches": 0.5,
                "bottom_margin_inches": 0.5,
            },
            "blocks": [],
        },
    )
    for index, content in enumerate(invalid_create_contents):
        spec = root / f"invalid-create-{index}.json"
        output = root / f"invalid-create-{index}.docx"
        write_json(spec, {"schema_version": 1, "content": content})
        payload = run_cli(
            tool,
            "create",
            "--spec",
            str(spec),
            "--output",
            str(output),
            expected_status=2,
        )
        assert payload["error"]["category"] == "bad_input"
        assert not output.exists()

    invalid_edit = root / "invalid-edit-key.json"
    invalid_edit_output = root / "invalid-edit-key.docx"
    write_json(
        invalid_edit,
        {
            "schema_version": 1,
            "operations": [
                {
                    "type": "delete_paragraph",
                    "paragraph_index": 0,
                    "unexpected": "rejected",
                }
            ],
        },
    )
    payload = run_cli(
        tool,
        "edit",
        "--input",
        str(source),
        "--spec",
        str(invalid_edit),
        "--output",
        str(invalid_edit_output),
        expected_status=2,
    )
    assert payload["error"]["category"] == "bad_input"
    assert not invalid_edit_output.exists()

    missing_image_index = root / "missing-image-index.json"
    missing_image_index_output = root / "missing-image-index.docx"
    write_json(
        missing_image_index,
        {
            "schema_version": 1,
            "operations": [{"type": "replace_image", "path": "replacement.png"}],
        },
    )
    payload = run_cli(
        tool,
        "edit",
        "--input",
        str(source),
        "--spec",
        str(missing_image_index),
        "--output",
        str(missing_image_index_output),
        expected_status=2,
    )
    assert payload["error"]["category"] == "bad_input"
    assert not missing_image_index_output.exists()

    ignored_timeout_output = root / "ignored-timeout.txt"
    payload = run_cli(
        tool,
        "convert",
        "--input",
        str(source),
        "--format",
        "text",
        "--output",
        str(ignored_timeout_output),
        "--timeout",
        "-1",
        expected_status=2,
    )
    assert payload["error"]["category"] == "bad_input"
    assert not ignored_timeout_output.exists()


def run_insert_regressions(tool: Path, root: Path, source: Path) -> None:
    """Validate pre-insertion indexes and explicit append semantics."""

    paragraph_count = len(Document(str(source)).paragraphs)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    out_of_range_spec = root / "insert-out-of-range.json"
    out_of_range_output = root / "insert-out-of-range.docx"
    write_json(
        out_of_range_spec,
        {
            "schema_version": 1,
            "operations": [
                {
                    "type": "insert_paragraph",
                    "position": "before",
                    "paragraph_index": paragraph_count,
                    "block": {"type": "paragraph", "text": "Must not appear"},
                }
            ],
        },
    )
    payload = run_cli(
        tool,
        "edit",
        "--input",
        str(source),
        "--spec",
        str(out_of_range_spec),
        "--output",
        str(out_of_range_output),
        expected_status=2,
    )
    assert payload["error"]["category"] == "bad_input"
    assert not out_of_range_output.exists()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash

    implicit_append_spec = root / "implicit-append.json"
    implicit_append_output = root / "implicit-append.docx"
    write_json(
        implicit_append_spec,
        {
            "schema_version": 1,
            "operations": [
                {
                    "type": "insert_paragraph",
                    "block": {"type": "paragraph", "text": "Implicit append rejected"},
                }
            ],
        },
    )
    payload = run_cli(
        tool,
        "edit",
        "--input",
        str(source),
        "--spec",
        str(implicit_append_spec),
        "--output",
        str(implicit_append_output),
        expected_status=2,
    )
    assert payload["error"]["category"] == "bad_input"
    assert not implicit_append_output.exists()

    append_spec = root / "explicit-append.json"
    append_output = root / "explicit-append.docx"
    write_json(
        append_spec,
        {
            "schema_version": 1,
            "operations": [
                {
                    "type": "insert_paragraph",
                    "position": "append",
                    "block": {"type": "paragraph", "text": "Explicit append"},
                }
            ],
        },
    )
    run_cli(
        tool,
        "edit",
        "--input",
        str(source),
        "--spec",
        str(append_spec),
        "--output",
        str(append_output),
    )
    appended = Document(str(append_output))
    assert len(appended.paragraphs) == paragraph_count + 1
    assert appended.paragraphs[-1].text == "Explicit append"


def run_protected_boundary_regression(tool: Path, root: Path, source: Path) -> None:
    """Prove replacement cannot flatten the PAGE field boundary."""

    spec = root / "protected-field.json"
    output = root / "protected-field.docx"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    write_json(
        spec,
        {
            "schema_version": 1,
            "operations": [
                {
                    "type": "replace_text",
                    "find": "1",
                    "replace": "2",
                    "expected_count": 1,
                    "scope": "footers",
                }
            ],
        },
    )
    payload = run_cli(
        tool,
        "edit",
        "--input",
        str(source),
        "--spec",
        str(spec),
        "--output",
        str(output),
        expected_status=5,
    )
    assert payload["error"]["category"] == "ambiguous_edit"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    assert not output.exists()


def run_story_text_regression(tool: Path, root: Path) -> None:
    """Keep same-text stories distinct and include story table rows."""

    spec = root / "story-text.json"
    document_path = root / "story-text.docx"
    text_path = root / "story-text.txt"
    story_blocks = [
        {"type": "paragraph", "text": "Same story"},
        {"type": "table", "rows": [["Story", "Table"]]},
    ]
    write_json(
        spec,
        {
            "schema_version": 1,
            "content": {
                "blocks": [{"type": "paragraph", "text": "Body"}],
                "headers": {
                    "default": story_blocks,
                    "first_page": [{"type": "paragraph", "text": "Same story"}],
                },
                "footers": {
                    "default": [
                        {"type": "paragraph", "text": "Same story"},
                        {"type": "table", "rows": [["Footer", "Table"]]},
                    ]
                },
            },
        },
    )
    run_cli(tool, "create", "--spec", str(spec), "--output", str(document_path))
    run_cli(
        tool,
        "convert",
        "--input",
        str(document_path),
        "--format",
        "text",
        "--output",
        str(text_path),
    )
    text = text_path.read_text(encoding="utf-8")
    assert text.count("Same story") == 3
    assert "[Header: default]" in text
    assert "[Header: first_page]" in text
    assert "[Footer: default]" in text
    assert "Story\tTable" in text
    assert "Footer\tTable" in text


def run_external_relationship_regression(tool: Path, root: Path, source: Path) -> None:
    """Reject external relationships unless a strictly typed opt-in is explicit."""

    external = root / "external.docx"
    add_external_relationship(source, external)
    payload = run_cli(
        tool,
        "inspect",
        "--input",
        str(external),
        expected_status=2,
    )
    assert payload["error"]["category"] == "bad_input"

    rejected_text = root / "external-rejected.txt"
    payload = run_cli(
        tool,
        "convert",
        "--input",
        str(external),
        "--format",
        "text",
        "--output",
        str(rejected_text),
        expected_status=2,
    )
    assert payload["error"]["category"] == "bad_input"
    assert not rejected_text.exists()

    allowed = run_cli(
        tool,
        "inspect",
        "--input",
        str(external),
        "--allow-external-relationships",
    )
    assert any(item["code"] == "external_relationships_allowed" for item in allowed["warnings"])

    bad_job = root / "external-opt-in-wrong-type.json"
    write_json(
        bad_job,
        {
            "schema_version": 1,
            "operation": "inspect",
            "input": external.name,
            "allow_external_relationships": "true",
        },
    )
    payload = run_cli(tool, "--job", str(bad_job), expected_status=2)
    assert payload["error"]["category"] == "bad_input"


def run_custom_xml_regression(tool: Path, root: Path, source: Path) -> None:
    """Ignore only the standard fresh-template customXml set."""

    custom = root / "custom-extra.docx"
    add_custom_xml_part(source, custom)
    payload = run_cli(tool, "inspect", "--input", str(custom))
    unsupported = next(item for item in payload["warnings"] if item["code"] == "unsupported_parts")
    assert "customXml/smoke-extra.xml" in unsupported["parts"]
    assert "customXml/_rels/item1.xml.rels" not in unsupported["parts"]
    assert "customXml/item1.xml" not in unsupported["parts"]
    assert "customXml/itemProps1.xml" not in unsupported["parts"]


def run_fake_pdf_regressions(tool: Path, root: Path, source: Path) -> None:
    """Exercise PDF bounds and process-group timeout cleanup without real LibreOffice."""

    fake_bin = root / "fake-bin"
    fake_bin.mkdir()
    fake_soffice = fake_bin / "soffice"
    make_fake_soffice(fake_soffice)
    environment = {"PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}"}

    valid_pdf = root / "one-page-fixture.pdf"
    write_pdf(valid_pdf, 1)
    valid_output = root / "fake-valid.pdf"
    payload = run_cli(
        tool,
        "convert",
        "--input",
        str(source),
        "--format",
        "pdf",
        "--output",
        str(valid_output),
        environment={
            **environment,
            "DOCX_SMOKE_SOFFICE_MODE": "copy",
            "DOCX_SMOKE_PDF_FIXTURE": str(valid_pdf),
        },
    )
    assert payload["verification"]["page_count"] == 1
    assert payload["verification"]["pdf_byte_limit"] == MAX_PDF_BYTES
    assert payload["verification"]["page_limit"] == MAX_PDF_PAGES
    assert payload["verification"]["text_pages_checked"] == 1
    blank_page = payload["verification"]["content_quality_report"]["pages"][0]
    assert blank_page["low_text"] is True
    assert blank_page["has_nontext_content"] is False
    assert blank_page["nearly_blank"] is True
    assert payload["verification"]["content_quality_report"]["limitation"]

    image_pdf = root / "image-only-fixture.pdf"
    write_image_only_pdf(image_pdf)
    image_output = root / "fake-image-only.pdf"
    payload = run_cli(
        tool,
        "convert",
        "--input",
        str(source),
        "--format",
        "pdf",
        "--output",
        str(image_output),
        environment={
            **environment,
            "DOCX_SMOKE_SOFFICE_MODE": "copy",
            "DOCX_SMOKE_PDF_FIXTURE": str(image_pdf),
        },
    )
    image_page = payload["verification"]["content_quality_report"]["pages"][0]
    assert image_page["low_text"] is True
    assert image_page["has_nontext_content"] is True
    assert image_page["nearly_blank"] is False

    oversized_output = root / "fake-oversized.pdf"
    payload = run_cli(
        tool,
        "convert",
        "--input",
        str(source),
        "--format",
        "pdf",
        "--output",
        str(oversized_output),
        expected_status=6,
        environment={
            **environment,
            "DOCX_SMOKE_SOFFICE_MODE": "oversized",
            "DOCX_SMOKE_OVERSIZED_BYTES": str(MAX_PDF_BYTES + 1),
        },
    )
    assert payload["error"]["category"] == "resource_limit"
    assert not oversized_output.exists()

    too_many_pages = root / "too-many-pages-fixture.pdf"
    write_pdf(too_many_pages, MAX_PDF_PAGES + 1)
    page_output = root / "fake-too-many-pages.pdf"
    payload = run_cli(
        tool,
        "convert",
        "--input",
        str(source),
        "--format",
        "pdf",
        "--output",
        str(page_output),
        expected_status=6,
        environment={
            **environment,
            "DOCX_SMOKE_SOFFICE_MODE": "copy",
            "DOCX_SMOKE_PDF_FIXTURE": str(too_many_pages),
        },
    )
    assert payload["error"]["category"] == "resource_limit"
    assert not page_output.exists()

    decompressed_stream_pdf = root / "decompressed-stream-fixture.pdf"
    write_compressed_stream_pdf(decompressed_stream_pdf)
    stream_output = root / "fake-decompressed-stream.pdf"
    payload = run_cli(
        tool,
        "convert",
        "--input",
        str(source),
        "--format",
        "pdf",
        "--output",
        str(stream_output),
        expected_status=6,
        environment={
            **environment,
            "DOCX_SMOKE_SOFFICE_MODE": "copy",
            "DOCX_SMOKE_PDF_FIXTURE": str(decompressed_stream_pdf),
        },
    )
    assert payload["error"]["category"] == "resource_limit"
    assert not stream_output.exists()

    malformed_output = root / "fake-malformed.pdf"
    payload = run_cli(
        tool,
        "convert",
        "--input",
        str(source),
        "--format",
        "pdf",
        "--output",
        str(malformed_output),
        expected_status=10,
        environment={
            **environment,
            "DOCX_SMOKE_SOFFICE_MODE": "malformed",
        },
    )
    assert payload["error"]["category"] == "external_tool_failed"
    assert not malformed_output.exists()

    child_pid_path = root / "soffice-child.pid"
    timeout_output = root / "fake-timeout.pdf"
    payload = run_cli(
        tool,
        "convert",
        "--input",
        str(source),
        "--format",
        "pdf",
        "--output",
        str(timeout_output),
        "--timeout",
        "1",
        expected_status=10,
        timeout=10,
        environment={
            **environment,
            "DOCX_SMOKE_SOFFICE_MODE": "timeout",
            "DOCX_SMOKE_CHILD_PID": str(child_pid_path),
        },
    )
    assert payload["error"]["category"] == "external_tool_failed"
    assert payload["error"]["details"].get("process_group_cleaned") is True, payload
    assert not timeout_output.exists()
    assert child_pid_path.is_file()
    assert_process_gone(int(child_pid_path.read_text(encoding="utf-8")))


def run_fake_render_regressions(tool: Path, root: Path, source: Path) -> dict[str, Any]:
    """Exercise render mechanics and failure paths without real LibreOffice."""

    if importlib.util.find_spec("pypdfium2") is None:
        result = {
            "status": "skipped",
            "reason": "pypdfium2 is not installed; fake render regressions skipped.",
        }
        print(json.dumps({"diagnostic": "fake_render", **result}), file=sys.stderr)
        return result

    fake_bin = root / "fake-render-bin"
    fake_bin.mkdir()
    make_fake_soffice(fake_bin / "soffice")
    fixture = root / "render-fixture.pdf"
    write_pdf(fixture, 3)
    environment = {
        "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
        "DOCX_SMOKE_SOFFICE_MODE": "copy",
        "DOCX_SMOKE_PDF_FIXTURE": str(fixture),
    }

    output = root / "rendered-fake"
    payload = run_cli(
        tool,
        "render",
        "--input",
        str(source),
        "--output",
        str(output),
        environment=environment,
    )
    assert payload["status"] == "ok"
    assert payload["operation"] == "render"
    assert payload["counts"]["pdf_pages"] == 3
    assert payload["counts"]["pages_rendered"] == 3
    assert payload["verification"]["atomic_publish"] is True
    assert payload["verification"]["png_reopened"] == 3
    assert payload["verification"]["source_unchanged_at_publish_gate"] is True
    artifacts = payload["verification"]["content_quality_report"]["raster_review_artifacts"]
    assert artifacts == ["page-001.png", "page-002.png", "page-003.png"]
    assert any(item["code"] == "render_review_required" for item in payload["warnings"])
    for record in payload["result"]["images"]:
        data = (output / record["file"]).read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        assert hashlib.sha256(data).hexdigest() == record["sha256"]
        assert record["width_px"] > 0 and record["height_px"] > 0
    assert sorted(item.name for item in output.iterdir()) == artifacts

    subset_output = root / "rendered-subset"
    payload = run_cli(
        tool,
        "render",
        "--input",
        str(source),
        "--output",
        str(subset_output),
        "--pages",
        "2",
        environment=environment,
    )
    assert payload["counts"]["pages_rendered"] == 1
    assert [item.name for item in subset_output.iterdir()] == ["page-002.png"]

    bad_dpi_output = root / "rendered-bad-dpi"
    payload = run_cli(
        tool,
        "render",
        "--input",
        str(source),
        "--output",
        str(bad_dpi_output),
        "--dpi",
        "20",
        expected_status=2,
        environment=environment,
    )
    assert payload["error"]["category"] == "bad_input"
    assert not bad_dpi_output.exists()

    payload = run_cli(
        tool,
        "render",
        "--input",
        str(source),
        "--output",
        str(output),
        expected_status=9,
        environment=environment,
    )
    assert payload["error"]["category"] == "output_conflict"

    missing_page_output = root / "rendered-missing-page"
    payload = run_cli(
        tool,
        "render",
        "--input",
        str(source),
        "--output",
        str(missing_page_output),
        "--pages",
        "9",
        expected_status=2,
        environment=environment,
    )
    assert payload["error"]["category"] == "bad_input"
    assert payload["error"]["details"]["pdf_pages"] == 3
    assert not missing_page_output.exists()

    malformed_output = root / "rendered-malformed"
    payload = run_cli(
        tool,
        "render",
        "--input",
        str(source),
        "--output",
        str(malformed_output),
        expected_status=10,
        environment={**environment, "DOCX_SMOKE_SOFFICE_MODE": "malformed"},
    )
    assert payload["error"]["category"] == "external_tool_failed"
    assert not malformed_output.exists()
    assert not any(item.name.startswith(".rendered-") for item in root.iterdir())

    return {"status": "passed", "pages": 3}


def run_soffice_discovery_regressions(tool: Path, root: Path, source: Path) -> None:
    """Exercise the DOCX_SOFFICE override, including its no-fallthrough failure."""

    fake_bin = root / "fake-discovery-bin"
    fake_bin.mkdir()
    fake_soffice = fake_bin / "soffice"
    make_fake_soffice(fake_soffice)
    fixture = root / "discovery-fixture.pdf"
    write_pdf(fixture, 1)

    override_output = root / "discovery-valid.pdf"
    payload = run_cli(
        tool,
        "convert",
        "--input",
        str(source),
        "--format",
        "pdf",
        "--output",
        str(override_output),
        environment={
            "DOCX_SOFFICE": str(fake_soffice),
            "DOCX_SMOKE_SOFFICE_MODE": "copy",
            "DOCX_SMOKE_PDF_FIXTURE": str(fixture),
        },
    )
    assert payload["verification"]["page_count"] == 1
    # The blank fixture proves the override binary ran, not any soffice from PATH.
    assert payload["verification"]["content_quality_report"]["pages"][0]["nearly_blank"] is True
    assert override_output.is_file()

    missing_binary = root / "missing-soffice"
    missing_output = root / "discovery-missing.pdf"
    payload = run_cli(
        tool,
        "convert",
        "--input",
        str(source),
        "--format",
        "pdf",
        "--output",
        str(missing_output),
        expected_status=4,
        environment={"DOCX_SOFFICE": str(missing_binary)},
    )
    assert payload["error"]["category"] == "missing_dependency"
    assert payload["error"]["details"]["path"] == str(missing_binary)
    assert not missing_output.exists()


def find_soffice_for_smoke() -> str | None:
    """Mirror the tool's LibreOffice discovery order for gating optional checks."""

    override = os.environ.get("DOCX_SOFFICE")
    if override:
        candidate = Path(override)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
        return None
    for name in ("soffice", "libreoffice"):
        executable = shutil.which(name)
        if executable:
            return executable
    if sys.platform == "darwin":
        for candidate in (
            Path("/Applications/LibreOffice.app/Contents/MacOS/soffice"),
            Path.home() / "Applications/LibreOffice.app/Contents/MacOS/soffice",
        ):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    return None


def run_render_check(tool: Path, source: Path, output: Path) -> dict[str, Any]:
    """Render through real LibreOffice and verify the published PNG pages."""

    if importlib.util.find_spec("pypdfium2") is None:
        result = {
            "status": "skipped",
            "reason": "pypdfium2 is not installed; optional render check skipped.",
        }
        print(json.dumps({"diagnostic": "render_check", **result}), file=sys.stderr)
        return result
    payload = run_cli(
        tool,
        "render",
        "--input",
        str(source),
        "--output",
        str(output),
        timeout=120,
    )
    assert payload["counts"]["pages_rendered"] == payload["counts"]["pdf_pages"]
    assert payload["counts"]["pages_rendered"] >= 1
    assert payload["verification"]["png_reopened"] == payload["counts"]["pages_rendered"]
    for record in payload["result"]["images"]:
        data = (output / record["file"]).read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        assert hashlib.sha256(data).hexdigest() == record["sha256"]
        assert record["width_px"] > 0 and record["height_px"] > 0
    return {"status": "passed", "pages_rendered": payload["counts"]["pages_rendered"]}


def build_parser() -> argparse.ArgumentParser:
    """Build smoke-test arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-libreoffice",
        action="store_true",
        help="Fail unless soffice is available and DOCX-to-PDF validation passes.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the isolated round-trip scenario."""

    args = build_parser().parse_args(argv)
    tool = Path(__file__).with_name("docx_tool.py").resolve()
    with tempfile.TemporaryDirectory(prefix="docx-skill-smoke-") as temp_name:
        root = Path(temp_name)
        original_image = root / "original.png"
        replacement_image = root / "replacement.png"
        make_png(original_image, (210, 45, 45))
        make_png(replacement_image, (30, 110, 210))

        create_spec = root / "create.json"
        created = root / "created.docx"
        write_json(
            create_spec,
            {
                "schema_version": 1,
                "content": {
                    "metadata": {
                        "title": "DOCX smoke fixture",
                        "author": "Kolega smoke test",
                    },
                    "blocks": [
                        {"type": "heading", "level": 1, "text": "DOCX Smoke Test"},
                        {
                            "type": "paragraph",
                            "style": "Quote",
                            "runs": [
                                {"text": "Alpha ", "bold": True},
                                {"text": "Beta", "italic": True},
                            ],
                        },
                        {"type": "paragraph", "text": "Repeat Repeat"},
                        {
                            "type": "table",
                            "style": "Table Grid",
                            "header_rows": 1,
                            "layout": "fixed",
                            "column_widths_inches": [1.4, 4.6],
                            "allow_row_split": False,
                            "rows": [["Key", "Value"], ["State", "Original"]],
                        },
                        {
                            "type": "image",
                            "path": original_image.name,
                            "width_inches": 0.35,
                        },
                    ],
                    "headers": {"default": [{"type": "paragraph", "text": "Smoke header"}]},
                    "footers": {
                        "default": [
                            {
                                "type": "field",
                                "field": "page_number",
                                "prefix": "Page ",
                            }
                        ]
                    },
                },
            },
        )
        create_payload = run_cli(
            tool,
            "create",
            "--spec",
            str(create_spec),
            "--output",
            str(created),
        )
        assert create_payload["status"] == "ok"
        assert not any(item["code"] == "unsupported_parts" for item in create_payload["warnings"])
        assert created.is_file()
        created_hash = hashlib.sha256(created.read_bytes()).hexdigest()

        invalid_overwrite_job = root / "invalid-overwrite.json"
        write_json(
            invalid_overwrite_job,
            {
                "schema_version": 1,
                "operation": "create",
                "output": created.name,
                "overwrite": "false",
                "content": {"blocks": []},
            },
        )
        invalid_overwrite = run_cli(
            tool,
            "--job",
            str(invalid_overwrite_job),
            expected_status=2,
        )
        assert invalid_overwrite["error"]["category"] == "bad_input"
        assert hashlib.sha256(created.read_bytes()).hexdigest() == created_hash

        inspect_payload = run_cli(tool, "inspect", "--input", str(created))
        assert_inspection(inspect_payload)
        run_font_portability_regressions(tool, root, created)
        run_confirmed_defect_regressions(tool, root, original_image)

        edit_spec = root / "edit.json"
        edited = root / "edited.docx"
        write_json(
            edit_spec,
            {
                "schema_version": 1,
                "operations": [
                    {
                        "type": "replace_text",
                        "find": "ha Be",
                        "replace": "X",
                        "expected_count": 1,
                        "cross_run_policy": "first_run",
                    },
                    {
                        "type": "replace_text",
                        "find": "Repeat",
                        "replace": "Done",
                        "expected_count": 2,
                        "replace_all": True,
                    },
                    {
                        "type": "update_table",
                        "table_index": 0,
                        "row": 1,
                        "column": 1,
                        "value": "Updated",
                    },
                    {
                        "type": "replace_image",
                        "inline_image_index": 0,
                        "path": replacement_image.name,
                    },
                ],
            },
        )
        edit_payload = run_cli(
            tool,
            "edit",
            "--input",
            str(created),
            "--spec",
            str(edit_spec),
            "--output",
            str(edited),
        )
        assert edit_payload["counts"]["text_matches"] == 3
        assert hashlib.sha256(created.read_bytes()).hexdigest() == created_hash
        assert created.read_bytes() != edited.read_bytes()
        assert_docx_structure(edited, replacement_image)
        edited_inspection = run_cli(tool, "inspect", "--input", str(edited))
        assert_inspection(edited_inspection)

        text_output = root / "edited.txt"
        text_payload = run_cli(
            tool,
            "convert",
            "--input",
            str(edited),
            "--format",
            "text",
            "--output",
            str(text_output),
        )
        text = text_output.read_text(encoding="utf-8")
        assert "DOCX Smoke Test" in text
        assert "Updated" in text
        assert text_payload["verification"]["utf8"] is True

        run_schema_regressions(tool, root, created)
        run_insert_regressions(tool, root, created)
        run_protected_boundary_regression(tool, root, created)
        run_story_text_regression(tool, root)
        run_external_relationship_regression(tool, root, created)
        run_custom_xml_regression(tool, root, created)
        run_fake_pdf_regressions(tool, root, created)
        fake_render_result = run_fake_render_regressions(tool, root, created)
        run_soffice_discovery_regressions(tool, root, created)

        corrupt = root / "corrupt.docx"
        corrupt.write_bytes(b"not a zip")
        failure = run_cli(
            tool,
            "inspect",
            "--input",
            str(corrupt),
            expected_status=2,
        )
        assert failure["error"]["category"] == "bad_input"

        soffice = find_soffice_for_smoke()
        quality_result = run_quality_upgrade_regressions(
            tool, root, original_image, libreoffice=bool(soffice)
        )
        if soffice:
            pdf_result = run_pdf_check(tool, edited, root / "edited.pdf")
            render_result = run_render_check(tool, edited, root / "rendered-pages")
        elif args.require_libreoffice:
            print(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "error",
                        "error": {
                            "category": "missing_dependency",
                            "message": "LibreOffice soffice is required by the smoke-test flag.",
                        },
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return 1
        else:
            pdf_result = {
                "status": "skipped",
                "reason": "LibreOffice soffice is not installed; optional PDF check skipped.",
            }
            print(json.dumps({"diagnostic": "pdf_check", **pdf_result}), file=sys.stderr)
            render_result = {
                "status": "skipped",
                "reason": "LibreOffice soffice is not installed; optional render check skipped.",
            }
            print(json.dumps({"diagnostic": "render_check", **render_result}), file=sys.stderr)

        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "ok",
                    "scenario": "docx_round_trip",
                    "checks": {
                        "create": "passed",
                        "inspect": "passed",
                        "font_portability": "passed",
                        "confirmed_defect_regressions": "passed",
                        "edit": "passed",
                        "reopen": "passed",
                        "formatting_policy": "first_run passed",
                        "text_conversion": "passed",
                        "strict_schemas": "passed",
                        "paragraph_insertion": "passed",
                        "protected_boundary": "passed",
                        "story_text": "passed",
                        "external_relationships": "passed",
                        "custom_xml": "passed",
                        "bounded_pdf_and_process_cleanup": "passed",
                        "fake_render": fake_render_result,
                        "soffice_discovery": "passed",
                        "quality_upgrade": quality_result,
                        "corrupt_input": "passed",
                        "pdf_conversion": pdf_result,
                        "render": render_result,
                    },
                    "fixtures": "temporary_directory_removed",
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
