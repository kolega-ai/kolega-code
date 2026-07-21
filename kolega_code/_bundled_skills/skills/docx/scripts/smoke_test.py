#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise the public DOCX CLI entirely inside a temporary directory."""

from __future__ import annotations

import argparse
import hashlib
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
from docx.oxml.ns import qn
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject

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
    assert document.tables[0].cell(1, 1).text == "Updated"
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
    assert any(item["heading_level"] == 1 for item in payload["result"]["paragraphs"])
    assert any(
        story["paragraphs"][0]["text"] == "Smoke header"
        for story in payload["result"]["headers"]
        if story["paragraphs"]
    )


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

        soffice = shutil.which("soffice")
        if soffice:
            pdf_result = run_pdf_check(tool, edited, root / "edited.pdf")
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

        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "ok",
                    "scenario": "docx_round_trip",
                    "checks": {
                        "create": "passed",
                        "inspect": "passed",
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
                        "corrupt_input": "passed",
                        "pdf_conversion": pdf_result,
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
