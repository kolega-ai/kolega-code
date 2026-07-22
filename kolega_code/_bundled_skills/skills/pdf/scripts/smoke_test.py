#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generated-fixture smoke tests for the public PDF CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    ArrayObject,
    ContentStream,
    DictionaryObject,
    NameObject,
    NumberObject,
    StreamObject,
)

SCHEMA_VERSION = 1
TOOL = Path(__file__).with_name("pdf_tool.py")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class SmokeFailure(Exception):
    """A focused smoke-test assertion failure."""


def run_tool(
    arguments: list[str],
    *,
    env: dict[str, str] | None = None,
    expected_status: int = 0,
) -> dict[str, Any]:
    with tempfile.TemporaryFile(mode="w+b") as stdout_file:
        with tempfile.TemporaryFile(mode="w+b") as stderr_file:
            completed = subprocess.run(
                [sys.executable, str(TOOL), *arguments],
                stdout=stdout_file,
                stderr=stderr_file,
                env=env,
                timeout=1_800,
                check=False,
            )
            stdout_file.seek(0, os.SEEK_END)
            stdout_size = stdout_file.tell()
            stdout_file.seek(max(0, stdout_size - 1_000_000))
            stdout = stdout_file.read().decode("utf-8", errors="replace")
            stderr_file.seek(0, os.SEEK_END)
            stderr_size = stderr_file.tell()
            stderr_file.seek(max(0, stderr_size - 1_000_000))
            stderr = stderr_file.read().decode("utf-8", errors="replace")
    if completed.returncode != expected_status:
        raise SmokeFailure(
            f"Command returned {completed.returncode}, expected {expected_status}: "
            f"{arguments}; stdout={stdout!r}; stderr={stderr!r}"
        )
    stream = stdout if expected_status == 0 else stderr
    lines = [line for line in stream.splitlines() if line.strip()]
    if len(lines) != 1:
        raise SmokeFailure(
            f"Expected exactly one JSON line on {'stdout' if expected_status == 0 else 'stderr'}; "
            f"got {stream!r}"
        )
    try:
        payload = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise SmokeFailure(f"CLI emitted invalid JSON: {stream!r}") from exc
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise SmokeFailure(f"Unexpected schema version: {payload}")
    if expected_status == 0 and payload.get("status") != "ok":
        raise SmokeFailure(f"Expected successful payload: {payload}")
    if expected_status != 0 and payload.get("status") != "error":
        raise SmokeFailure(f"Expected error payload: {payload}")
    return payload


def write_test_image(path: Path, *, complex_layout: bool = False) -> None:
    image = Image.new("RGB", (1_600, 1_000), "white")
    draw = ImageDraw.Draw(image)
    if complex_layout:
        draw.text((80, 50), "OCR COMPLEX LAYOUT", fill="black")
        draw.text((80, 130), "LEFT COLUMN\nAlpha\nBeta\nGamma", fill="black", spacing=18)
        draw.text((900, 130), "RIGHT COLUMN\nOne\nTwo\nThree", fill="black", spacing=18)
        x0, y0, cell_width, cell_height = 300, 500, 330, 90
        for row in range(4):
            draw.line(
                (x0, y0 + row * cell_height, x0 + 3 * cell_width, y0 + row * cell_height),
                fill="black",
                width=4,
            )
        for column in range(4):
            draw.line(
                (x0 + column * cell_width, y0, x0 + column * cell_width, y0 + 3 * cell_height),
                fill="black",
                width=4,
            )
        draw.text((x0 + 20, y0 + 25), "NAME", fill="black")
        draw.text((x0 + cell_width + 20, y0 + 25), "QTY", fill="black")
        draw.text((x0 + 2 * cell_width + 20, y0 + 25), "TOTAL", fill="black")
    else:
        draw.text((180, 300), "OCR SMOKE CLEAN PRINTED TEXT 12345", fill="black")
        draw.text((180, 390), "Single column English fixture", fill="black")
    image.save(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def assert_bad_input(arguments: list[str], label: str) -> dict[str, Any]:
    payload = run_tool(arguments, expected_status=2)
    if payload.get("category") != "bad_input":
        raise SmokeFailure(f"{label} did not map to bad_input: {payload}")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_pdf(path: Path, page_count: int, *, password: str | None = None) -> PdfReader:
    reader = PdfReader(str(path), strict=True)
    if reader.is_encrypted:
        if not password or not reader.decrypt(password):
            raise SmokeFailure(f"Could not decrypt generated fixture: {path}")
    if len(reader.pages) != page_count:
        raise SmokeFailure(f"{path} has {len(reader.pages)} pages, expected {page_count}")
    return reader


def run_core(root: Path) -> dict[str, Any]:
    clean_image = root / "clean.png"
    write_test_image(clean_image)
    source_pdf = root / "source.pdf"
    create_job = root / "create.json"
    write_json(
        create_job,
        {
            "schema_version": 1,
            "page_size": "letter",
            "metadata": {"title": "PDF smoke fixture", "author": "Smoke test"},
            "header": {"text": "Generated fixture — page {page}"},
            "footer": {"text": "CORE-SMOKE-{page}"},
            "pages": [
                {
                    "elements": [
                        {
                            "type": "text",
                            "text": "DIGITAL PAGE ONE",
                            "x": 72,
                            "y": 90,
                            "font": "Helvetica-Bold",
                            "font_size": 18,
                        },
                        {
                            "type": "table",
                            "x": 72,
                            "y": 150,
                            "width": 420,
                            "data": [
                                ["Item", "Count", "Price"],
                                ["Widget", "2", "10.00"],
                                ["Total", "", "20.00"],
                            ],
                        },
                        {
                            "type": "form",
                            "field_type": "text",
                            "name": "customer_name",
                            "x": 72,
                            "y": 300,
                            "width": 220,
                            "value": "",
                        },
                    ]
                },
                {
                    "elements": [
                        {
                            "type": "text",
                            "text": "DIGITAL PAGE TWO",
                            "x": 72,
                            "y": 90,
                            "font_size": 18,
                        }
                    ]
                },
                {
                    "elements": [
                        {
                            "type": "image",
                            "path": str(clean_image),
                            "x": 0,
                            "y": 0,
                            "width": 612,
                            "height": 792,
                            "preserve_aspect_ratio": False,
                        }
                    ]
                },
            ],
        },
    )
    create_result = run_tool(["create", "--job", str(create_job), "--output", str(source_pdf)])
    if create_result["verification"]["page_count"] != 3:
        raise SmokeFailure("Create did not verify three pages")
    assert_pdf(source_pdf, 3)
    create_hardlink = root / "create-hardlink.pdf"
    os.link(clean_image, create_hardlink)
    create_hardlink_result = run_tool(
        [
            "create",
            "--job",
            str(create_job),
            "--output",
            str(create_hardlink),
            "--overwrite",
        ],
        expected_status=2,
    )
    if create_hardlink_result["category"] != "bad_input":
        raise SmokeFailure(f"Create hard-link alias protection failed: {create_hardlink_result}")
    if create_hardlink.read_bytes() != clean_image.read_bytes():
        raise SmokeFailure("Rejected create hard-link alias modified the source image")
    assert_bad_input(
        ["create", "--job", str(create_job), "--output", str(root / "wrong-create.json")],
        "Create destination extension",
    )
    malformed_create_job = root / "malformed-create.json"
    write_json(
        malformed_create_job,
        {
            "schema_version": 1,
            "page_size": [True, 792],
            "pages": [{"elements": []}],
        },
    )
    assert_bad_input(
        [
            "create",
            "--job",
            str(malformed_create_job),
            "--output",
            str(root / "malformed-create.pdf"),
        ],
        "Malformed page size",
    )
    write_json(
        malformed_create_job,
        {
            "schema_version": 1,
            "pages": [
                {
                    "elements": [
                        {
                            "type": "text",
                            "text": "invalid color",
                            "color": [0, "not-a-number", 0],
                        }
                    ]
                }
            ],
        },
    )
    assert_bad_input(
        [
            "create",
            "--job",
            str(malformed_create_job),
            "--output",
            str(root / "malformed-color.pdf"),
        ],
        "Malformed color number",
    )
    write_json(
        malformed_create_job,
        {
            "schema_version": 1,
            "page_size": [200, 200],
            "margins": {"left": 120, "right": 120},
            "story": [{"type": "paragraph", "text": "impossible margins"}],
        },
    )
    assert_bad_input(
        [
            "create",
            "--job",
            str(malformed_create_job),
            "--output",
            str(root / "impossible-margins.pdf"),
        ],
        "Impossible page margins",
    )

    inspect_result = run_tool(
        [
            "inspect",
            "--input",
            str(source_pdf),
            "--words",
            "--tables",
        ]
    )
    document = inspect_result["result"]
    classifications = [page["classification"] for page in document["pages"]]
    if classifications[0] not in {"digital-text", "hybrid"}:
        raise SmokeFailure(f"Page one was not digital: {classifications}")
    if classifications[2] != "likely-scanned":
        raise SmokeFailure(f"Image-only page was not routed as scanned: {classifications}")
    if not document["forms"]["fields"]:
        raise SmokeFailure("Created form field was not inventoried")
    if not document["pages"][0].get("tables"):
        raise SmokeFailure("Created table was not extracted")

    extract_sidecar = root / "extract.json"
    run_tool(
        [
            "extract",
            "--input",
            str(source_pdf),
            "--pages",
            "1",
            "--mode",
            "plain",
            "--words",
            "--tables",
            "--output",
            str(extract_sidecar),
        ]
    )
    extracted = json.loads(extract_sidecar.read_text(encoding="utf-8"))
    if "DIGITAL PAGE ONE" not in extracted["pages"][0]["text"]:
        raise SmokeFailure("Text anchor was not extracted")

    pages_pdf = root / "pages.pdf"
    split_pdf = root / "split.pdf"
    pages_job = root / "pages.json"
    write_json(
        pages_job,
        {
            "schema_version": 1,
            "inputs": [{"id": "source", "path": str(source_pdf)}],
            "outputs": [
                {
                    "path": str(pages_pdf),
                    "pages": [
                        {"input": "source", "page": 2, "rotate": 90},
                        {"input": "source", "page": 1},
                        {"input": "source", "page": 1, "repeat": 2},
                    ],
                },
                {
                    "path": str(split_pdf),
                    "pages": [{"input": "source", "page": 3}],
                },
            ],
        },
    )
    pages_result = run_tool(["pages", "--job", str(pages_job)])
    if pages_result["counts"] != {"inputs": 1, "outputs": 2, "pages": 5}:
        raise SmokeFailure(f"Unexpected pages counts: {pages_result['counts']}")
    reordered = assert_pdf(pages_pdf, 4)
    if reordered.pages[0].rotation != 90:
        raise SmokeFailure("Rotation was not applied")
    if "DIGITAL PAGE TWO" not in (reordered.pages[0].extract_text() or ""):
        raise SmokeFailure("Reordered first page did not preserve its text")
    assert_pdf(split_pdf, 1)
    alias_job = root / "pages-alias.json"
    write_json(
        alias_job,
        {
            "schema_version": 1,
            "inputs": [{"id": "source", "path": str(source_pdf)}],
            "outputs": [
                {
                    "path": str(source_pdf),
                    "pages": [{"input": "source", "page": 1}],
                }
            ],
        },
    )
    alias_result = run_tool(
        ["pages", "--job", str(alias_job), "--overwrite"],
        expected_status=2,
    )
    if alias_result["category"] != "bad_input" or assert_pdf(source_pdf, 3) is None:
        raise SmokeFailure(f"Source/output alias protection failed: {alias_result}")
    hardlink_pdf = root / "source-hardlink.pdf"
    os.link(source_pdf, hardlink_pdf)
    hardlink_job = root / "pages-hardlink.json"
    write_json(
        hardlink_job,
        {
            "schema_version": 1,
            "inputs": [{"id": "source", "path": str(source_pdf)}],
            "outputs": [
                {
                    "path": str(hardlink_pdf),
                    "pages": [{"input": "source", "page": 1}],
                }
            ],
        },
    )
    hardlink_result = run_tool(
        ["pages", "--job", str(hardlink_job), "--overwrite"],
        expected_status=2,
    )
    if hardlink_result["category"] != "bad_input":
        raise SmokeFailure(f"Hard-link alias protection failed: {hardlink_result}")
    assert_pdf(source_pdf, 3)
    wrong_pages_job = root / "wrong-pages-extension.json"
    write_json(
        wrong_pages_job,
        {
            "schema_version": 1,
            "inputs": [{"id": "source", "path": str(source_pdf)}],
            "outputs": [
                {
                    "path": str(root / "wrong-pages.json"),
                    "pages": [{"input": "source", "page": 1}],
                }
            ],
        },
    )
    assert_bad_input(
        ["pages", "--job", str(wrong_pages_job)],
        "Pages destination extension",
    )

    edited_pdf = root / "edited.pdf"
    edit_job = root / "edit.json"
    write_json(
        edit_job,
        {
            "schema_version": 1,
            "operations": [
                {
                    "op": "stamp_text",
                    "pages": "1-2",
                    "text": "REVIEWED",
                    "x": 306,
                    "y": 396,
                    "angle": 35,
                    "opacity": 0.4,
                },
                {
                    "op": "fill_form",
                    "pages": "1",
                    "values": {"customer_name": "Ada Example"},
                },
                {
                    "op": "set_metadata",
                    "metadata": {"Title": "Edited smoke fixture", "Subject": "Verified"},
                },
            ],
        },
    )
    run_tool(
        [
            "edit",
            "--input",
            str(source_pdf),
            "--job",
            str(edit_job),
            "--output",
            str(edited_pdf),
        ]
    )
    edited = assert_pdf(edited_pdf, 3)
    if edited.metadata.title != "Edited smoke fixture":
        raise SmokeFailure("Edited metadata did not persist")
    fields = edited.get_fields() or {}
    if fields.get("customer_name", {}).get("/V") != "Ada Example":
        raise SmokeFailure(f"Form value did not persist: {fields}")
    if "REVIEWED" not in (edited.pages[0].extract_text() or ""):
        raise SmokeFailure("Text stamp was not extractable")
    assert_bad_input(
        [
            "edit",
            "--input",
            str(source_pdf),
            "--job",
            str(edit_job),
            "--output",
            str(root / "wrong-edit.json"),
        ],
        "Edit destination extension",
    )
    malformed_edit_job = root / "malformed-edit.json"
    write_json(
        malformed_edit_job,
        {
            "schema_version": 1,
            "operations": [{"op": "stamp_text", "text": "bad", "opacity": "NaN"}],
        },
    )
    assert_bad_input(
        [
            "edit",
            "--input",
            str(source_pdf),
            "--job",
            str(malformed_edit_job),
            "--output",
            str(root / "malformed-edit.pdf"),
        ],
        "Malformed edit numeric field",
    )

    encrypted_pdf = root / "encrypted.pdf"
    encrypt_job = root / "encrypt.json"
    write_json(
        encrypt_job,
        {
            "schema_version": 1,
            "operations": [
                {
                    "op": "encrypt",
                    "algorithm": "AES-256",
                    "user_password": "smoke-user-password",
                    "owner_password": "smoke-owner-password",
                }
            ],
        },
    )
    run_tool(
        [
            "edit",
            "--input",
            str(edited_pdf),
            "--job",
            str(encrypt_job),
            "--output",
            str(encrypted_pdf),
        ]
    )
    encrypted_reader = PdfReader(str(encrypted_pdf))
    if not encrypted_reader.is_encrypted:
        raise SmokeFailure("Encryption state did not persist")
    assert_pdf(encrypted_pdf, 3, password="smoke-user-password")
    encrypted_pages_job = root / "encrypted-pages.json"
    encrypted_page_output = root / "encrypted-page-output.pdf"
    encrypted_input: dict[str, Any] = {
        "id": "encrypted",
        "path": str(encrypted_pdf),
        "password": "smoke-user-password",
    }
    write_json(
        encrypted_pages_job,
        {
            "schema_version": 1,
            "inputs": [encrypted_input],
            "outputs": [
                {
                    "path": str(encrypted_page_output),
                    "pages": [{"input": "encrypted", "page": 1}],
                }
            ],
        },
    )
    authorization_result = run_tool(
        ["pages", "--job", str(encrypted_pages_job)],
        expected_status=3,
    )
    if authorization_result["category"] != "unsupported_operation":
        raise SmokeFailure(f"Encrypted pages authorization failure changed: {authorization_result}")
    encrypted_input["allow_decrypted_output"] = True
    write_json(
        encrypted_pages_job,
        {
            "schema_version": 1,
            "inputs": [encrypted_input],
            "outputs": [
                {
                    "path": str(encrypted_page_output),
                    "pages": [{"input": "encrypted", "page": 1}],
                }
            ],
        },
    )
    run_tool(["pages", "--job", str(encrypted_pages_job)])
    authorized_output = assert_pdf(encrypted_page_output, 1)
    if authorized_output.is_encrypted:
        raise SmokeFailure("Explicitly authorized pages output remained encrypted")

    decrypted_pdf = root / "decrypted.pdf"
    decrypt_job = root / "decrypt.json"
    write_json(
        decrypt_job,
        {"schema_version": 1, "operations": [{"op": "decrypt"}]},
    )
    password_env = os.environ.copy()
    password_env["PDF_SMOKE_PASSWORD"] = "smoke-user-password"
    run_tool(
        [
            "edit",
            "--input",
            str(encrypted_pdf),
            "--password-env",
            "PDF_SMOKE_PASSWORD",
            "--job",
            str(decrypt_job),
            "--output",
            str(decrypted_pdf),
        ],
        env=password_env,
    )
    decrypted = assert_pdf(decrypted_pdf, 3)
    if decrypted.is_encrypted:
        raise SmokeFailure("Decrypt operation remained encrypted")

    text_output = root / "converted.txt"
    run_tool(
        [
            "convert",
            "--input",
            str(decrypted_pdf),
            "--to",
            "txt",
            "--output",
            str(text_output),
            "--pages",
            "1-2",
        ]
    )
    if "DIGITAL PAGE ONE" not in text_output.read_text(encoding="utf-8"):
        raise SmokeFailure("PDF-to-text conversion lost the expected anchor")

    table_dir = root / "tables"
    table_result = run_tool(
        [
            "convert",
            "--input",
            str(source_pdf),
            "--to",
            "tables-csv",
            "--output",
            str(table_dir),
            "--pages",
            "1",
        ]
    )
    if table_result["counts"]["tables"] < 1:
        raise SmokeFailure("Table-to-CSV conversion produced no files")

    source_text = root / "source.txt"
    source_text.write_text("Text conversion fixture\n\nSecond paragraph\n", encoding="utf-8")
    text_pdf = root / "text.pdf"
    run_tool(
        [
            "convert",
            "--input",
            str(source_text),
            "--to",
            "pdf",
            "--output",
            str(text_pdf),
        ]
    )
    assert_pdf(text_pdf, 1)

    image_pdf = root / "images.pdf"
    run_tool(
        [
            "convert",
            "--images",
            str(clean_image),
            "--to",
            "pdf",
            "--output",
            str(image_pdf),
        ]
    )
    assert_pdf(image_pdf, 1)
    assert_bad_input(
        [
            "convert",
            "--images",
            str(clean_image),
            "--to",
            "pdf",
            "--output",
            str(root / "invalid-margin.pdf"),
            "--margin",
            "nan",
        ],
        "Non-finite image margin",
    )

    plan_result = run_tool(
        [
            "ocr-plan",
            "--input",
            str(source_pdf),
            "--pages",
            "3",
            "--languages",
            "eng",
            "--hardware",
            "cpu",
            "--layout",
            "simple",
        ]
    )
    recommendation = plan_result["recommendation"]
    if not recommendation["run_ocr"] or recommendation["engine"] not in {
        "surya",
        "paddle",
        "tesseract",
    }:
        raise SmokeFailure(f"Image-only OCR routing failed: {recommendation}")
    if plan_result["verification"]["engine_executed"]:
        raise SmokeFailure("Core OCR plan unexpectedly ran an engine")
    assert_bad_input(
        [
            "ocr-plan",
            "--input",
            str(source_pdf),
            "--pages",
            "3",
            "--languages",
            " , ",
        ],
        "Empty OCR-plan languages",
    )
    assert_bad_input(
        [
            "ocr",
            "--input",
            str(source_pdf),
            "--output",
            str(root / "wrong-ocr.txt"),
            "--engine",
            "tesseract",
            "--model-manifest",
            str(root / "missing-manifest.json"),
            "--languages",
            "eng",
        ],
        "Normalized OCR destination extension",
    )
    assert_bad_input(
        [
            "ocr",
            "--input",
            str(root / "wrong-ocr-input.txt"),
            "--output",
            str(root / "unused-preflight"),
            "--engine",
            "tesseract",
            "--model-manifest",
            str(root / "missing-manifest.json"),
            "--languages",
            "eng",
            "--preflight-only",
        ],
        "OCR preflight input extension",
    )

    missing_result = run_tool(
        ["inspect", "--input", str(root / "missing.pdf")],
        expected_status=2,
    )
    if missing_result["category"] != "bad_input":
        raise SmokeFailure(f"Missing-input failure category changed: {missing_result}")

    return {
        "create": True,
        "inspect_extract": True,
        "pages": True,
        "source_output_alias_rejected": True,
        "edit_forms_metadata_stamps": True,
        "encrypt_decrypt": True,
        "encrypted_pages_require_authorization": True,
        "convert": True,
        "image_only_ocr_routing_without_engine": True,
        "malformed_numeric_rejected": True,
        "output_extensions_enforced": True,
        "ocr_plan_languages_validated": True,
        "failure_path": True,
    }


def write_font_fixture(source_pdf: Path, out_pdf: Path, *, embedded: bool) -> None:
    """Inject a synthetic /Font entry so the inventory has a deterministic subject."""
    reader = PdfReader(str(source_pdf))
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    page = writer.pages[0]
    base_name = "/ABCDEF+FakeSans" if embedded else "/FakeSans"
    descriptor = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/FontDescriptor"),
            NameObject("/FontName"): NameObject(base_name),
            NameObject("/Flags"): NumberObject(32),
            NameObject("/ItalicAngle"): NumberObject(0),
            NameObject("/Ascent"): NumberObject(800),
            NameObject("/Descent"): NumberObject(-200),
            NameObject("/CapHeight"): NumberObject(700),
            NameObject("/StemV"): NumberObject(80),
            NameObject("/FontBBox"): ArrayObject(
                [NumberObject(0), NumberObject(-200), NumberObject(1000), NumberObject(800)]
            ),
        }
    )
    if embedded:
        font_file = StreamObject()
        font_file.set_data(b"\x00\x01\x00\x00fake-truetype-data")
        descriptor[NameObject("/FontFile2")] = writer._add_object(font_file)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/TrueType"),
            NameObject("/BaseFont"): NameObject(base_name),
            NameObject("/FirstChar"): NumberObject(32),
            NameObject("/LastChar"): NumberObject(32),
            NameObject("/Widths"): ArrayObject([NumberObject(500)]),
            NameObject("/FontDescriptor"): writer._add_object(descriptor),
            NameObject("/Encoding"): NameObject("/WinAnsiEncoding"),
        }
    )
    resources = page["/Resources"].get_object()
    fonts = resources.get("/Font")
    if fonts is None:
        fonts = DictionaryObject()
        resources[NameObject("/Font")] = fonts
    else:
        fonts = fonts.get_object()
    fonts[NameObject("/FZZ1")] = writer._add_object(font)
    with out_pdf.open("wb") as handle:
        writer.write(handle)


def run_render_and_fonts(root: Path) -> dict[str, Any]:
    base = root / "render-fonts"
    base.mkdir()
    fixture = base / "fixture.pdf"
    job = base / "job.json"
    write_json(
        job,
        {
            "schema_version": 1,
            "page_size": "letter",
            "pages": [
                {
                    "elements": [
                        {
                            "type": "text",
                            "text": "RENDER FIXTURE PAGE ONE",
                            "x": 72,
                            "y": 90,
                            "font_size": 16,
                        }
                    ]
                },
                {
                    "elements": [
                        {
                            "type": "text",
                            "text": "RENDER FIXTURE PAGE TWO",
                            "x": 72,
                            "y": 90,
                            "font_size": 16,
                        }
                    ]
                },
            ],
        },
    )
    run_tool(["create", "--job", str(job), "--output", str(fixture)])

    inspect_result = run_tool(["inspect", "--input", str(fixture)])
    document = inspect_result["result"]
    fonts = document.get("fonts")
    if not fonts or fonts.get("scope") != "document":
        raise SmokeFailure(f"Inspect omitted the document font inventory: {fonts}")
    helvetica = [entry for entry in fonts["entries"] if entry["name"] == "Helvetica"]
    if not helvetica or helvetica[0]["embedded"] or not helvetica[0]["base14"]:
        raise SmokeFailure(f"Helvetica inventory entry is wrong: {fonts['entries']}")
    if helvetica[0]["subtype"] != "Type1":
        raise SmokeFailure(f"Helvetica subtype changed: {helvetica}")
    if any(w.startswith("RELEASE BLOCKER") for w in document["warnings"]):
        raise SmokeFailure("Base-14-only fixture raised a release blocker")
    if not any(w.startswith("Non-embedded standard fonts") for w in document["warnings"]):
        raise SmokeFailure("Base-14 informational font warning is missing")
    extract_result = run_tool(["extract", "--input", str(fixture), "--pages", "1"])
    if "fonts" not in extract_result["result"]:
        raise SmokeFailure("Extract result omitted the font inventory")

    unembedded_pdf = base / "unembedded.pdf"
    write_font_fixture(fixture, unembedded_pdf, embedded=False)
    unembedded = run_tool(["inspect", "--input", str(unembedded_pdf)])["result"]
    if unembedded["fonts"]["unembedded_non_base14"] != ["FakeSans"]:
        raise SmokeFailure(f"FakeSans was not flagged: {unembedded['fonts']}")
    blockers = [
        w
        for w in unembedded["warnings"]
        if w.startswith("RELEASE BLOCKER: non-embedded fonts outside the standard base-14 set")
    ]
    if len(blockers) != 1 or "FakeSans" not in blockers[0]:
        raise SmokeFailure(f"Release blocker warning is wrong: {unembedded['warnings']}")

    embedded_pdf = base / "embedded.pdf"
    write_font_fixture(fixture, embedded_pdf, embedded=True)
    embedded = run_tool(["inspect", "--input", str(embedded_pdf)])["result"]
    fake_entries = [entry for entry in embedded["fonts"]["entries"] if entry["name"] == "FakeSans"]
    if not fake_entries or not fake_entries[0]["embedded"] or not fake_entries[0]["subset"]:
        raise SmokeFailure(f"Embedded subset fixture inventory is wrong: {fake_entries}")
    if any(w.startswith("RELEASE BLOCKER") for w in embedded["warnings"]):
        raise SmokeFailure("Embedded fixture raised a release blocker")

    render_dir = base / "pngs"
    render_result = run_tool(
        ["render", "--input", str(fixture), "--output", str(render_dir), "--dpi", "96"]
    )
    if render_result["counts"]["pages_rendered"] != 2:
        raise SmokeFailure(f"Render page count wrong: {render_result['counts']}")
    if render_result["verification"]["pngs_reopened"] != 2:
        raise SmokeFailure("Render did not reopen every published PNG")
    for item in render_result["renders"]:
        png = Path(item["output_path"])
        if not png.is_file():
            raise SmokeFailure(f"Rendered PNG missing: {png}")
        with png.open("rb") as handle:
            if handle.read(8) != PNG_SIGNATURE:
                raise SmokeFailure(f"Rendered file is not a PNG: {png}")
        with Image.open(png) as image:
            width, height = image.size
        expected_width = math.ceil(612 * 96 / 72)
        expected_height = math.ceil(792 * 96 / 72)
        if (width, height) != (expected_width, expected_height):
            raise SmokeFailure(f"Rendered dimensions wrong: {(width, height)}")
        if (item["width_px"], item["height_px"]) != (width, height):
            raise SmokeFailure("Render summary dimensions disagree with the PNG")
        if sha256(png) != item["sha256"]:
            raise SmokeFailure(f"Rendered PNG hash mismatch: {png}")

    selection_dir = base / "pngs-page2"
    selection = run_tool(
        [
            "render",
            "--input",
            str(fixture),
            "--output",
            str(selection_dir),
            "--pages",
            "2",
            "--dpi",
            "96",
        ]
    )
    files = sorted(path.name for path in selection_dir.iterdir())
    if files != ["page-0002.png"] or selection["counts"]["pages_rendered"] != 1:
        raise SmokeFailure(f"Render page selection wrong: {files}")

    rotated_pdf = base / "rotated.pdf"
    rotate_job = base / "rotate.json"
    write_json(
        rotate_job,
        {
            "schema_version": 1,
            "inputs": [{"id": "fixture", "path": str(fixture)}],
            "outputs": [
                {
                    "path": str(rotated_pdf),
                    "pages": [{"input": "fixture", "page": 1, "rotate": 90}],
                }
            ],
        },
    )
    run_tool(["pages", "--job", str(rotate_job)])
    rotated_dir = base / "pngs-rotated"
    rotated = run_tool(
        ["render", "--input", str(rotated_pdf), "--output", str(rotated_dir), "--dpi", "96"]
    )
    rotated_render = rotated["renders"][0]
    if (rotated_render["width_px"], rotated_render["height_px"]) != (
        math.ceil(792 * 96 / 72),
        math.ceil(612 * 96 / 72),
    ):
        raise SmokeFailure(f"Rotated render did not swap dimensions: {rotated_render}")

    nonempty_dir = base / "nonempty"
    nonempty_dir.mkdir()
    (nonempty_dir / "existing.txt").write_text("occupied", encoding="utf-8")
    assert_bad_input(
        [
            "render",
            "--input",
            str(fixture),
            "--output",
            str(nonempty_dir),
            "--overwrite",
        ],
        "Nonempty render directory",
    )
    assert_bad_input(
        [
            "render",
            "--input",
            str(fixture),
            "--output",
            str(base / "too-high-dpi"),
            "--dpi",
            "1200",
        ],
        "Render DPI ceiling",
    )
    big_job = base / "big.json"
    big_pdf = base / "big.pdf"
    write_json(
        big_job,
        {
            "schema_version": 1,
            "page_size": [5000, 5000],
            "pages": [{"elements": []}],
        },
    )
    run_tool(["create", "--job", str(big_job), "--output", str(big_pdf)])
    over_limit = run_tool(
        [
            "render",
            "--input",
            str(big_pdf),
            "--output",
            str(base / "over-limit"),
            "--dpi",
            "600",
        ],
        expected_status=6,
    )
    if over_limit["category"] != "resource_limit":
        raise SmokeFailure(f"Oversized render failure category changed: {over_limit}")
    return {
        "render": True,
        "render_rotation": True,
        "render_failure_paths": True,
        "font_inventory": True,
        "font_release_blocker": True,
    }


def run_manifest_compose_redact(root: Path) -> dict[str, Any]:
    base = root / "advanced"
    base.mkdir()

    # Manifest derive and check.
    tessdata = base / "tessdata"
    tessdata.mkdir()
    (tessdata / "eng.traineddata").write_bytes(os.urandom(2_048))
    manifest_path = base / "manifest.json"
    derive = run_tool(
        [
            "manifest",
            "--mode",
            "derive",
            "--engine",
            "tesseract",
            "--languages",
            "eng",
            "--tessdata-dir",
            str(tessdata),
            "--assume-source",
            "tessdata_fast",
            "--output",
            str(manifest_path),
        ]
    )
    if derive["ready_for_preflight"]:
        raise SmokeFailure("Derive without approvals claimed preflight readiness")
    derive = run_tool(
        [
            "manifest",
            "--mode",
            "derive",
            "--engine",
            "tesseract",
            "--languages",
            "eng",
            "--tessdata-dir",
            str(tessdata),
            "--assume-source",
            "tessdata_fast",
            "--accept-license",
            "--confirm-eligibility",
            "--output",
            str(manifest_path),
            "--overwrite",
        ]
    )
    if not derive["ready_for_preflight"]:
        raise SmokeFailure(f"Approved derive is not preflight-ready: {derive}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest["model"]["license_terms_accepted"]:
        raise SmokeFailure("Derived manifest lost the license approval flag")
    if manifest["artifacts"][0]["provenance"] != "declared-by-operator":
        raise SmokeFailure(f"Derived provenance wrong: {manifest['artifacts']}")
    check = run_tool(
        [
            "manifest",
            "--mode",
            "check",
            "--engine",
            "tesseract",
            "--languages",
            "eng",
            "--model-manifest",
            str(manifest_path),
        ]
    )
    if not check["preflight_would_pass"] or check["summary"]["failed"]:
        raise SmokeFailure(f"Clean manifest check failed: {check['summary']}")
    with (tessdata / "eng.traineddata").open("ab") as handle:
        handle.write(b"tampered")
    tampered = run_tool(
        [
            "manifest",
            "--mode",
            "check",
            "--engine",
            "tesseract",
            "--languages",
            "eng",
            "--model-manifest",
            str(manifest_path),
        ],
        expected_status=7,
    )
    failed_checks = [item for item in tampered["details"]["checks"] if not item["ok"]]
    if not any(
        item["check"] == "artifact_sha256"
        and item.get("expected_sha256")
        and item.get("actual_sha256")
        for item in failed_checks
    ):
        raise SmokeFailure(f"Tampered check lacks expected/actual hashes: {failed_checks}")
    missing_language = run_tool(
        [
            "manifest",
            "--mode",
            "derive",
            "--engine",
            "tesseract",
            "--languages",
            "deu",
            "--tessdata-dir",
            str(tessdata),
            "--output",
            str(base / "missing.json"),
        ],
        expected_status=2,
    )
    if missing_language["category"] != "bad_input":
        raise SmokeFailure(f"Missing language derive category changed: {missing_language}")
    paddle_root = base / "paddle"
    for role_dir in ("det", "rec", "layout"):
        (paddle_root / role_dir).mkdir(parents=True)
        (paddle_root / role_dir / "inference.pdiparams").write_bytes(os.urandom(1_024))
    paddle_manifest = base / "paddle-manifest.json"
    paddle_derive = run_tool(
        [
            "manifest",
            "--mode",
            "derive",
            "--engine",
            "paddle",
            "--languages",
            "en",
            "--artifact",
            f"text_detection_model_dir={paddle_root / 'det'}",
            "--artifact",
            f"text_recognition_model_dir={paddle_root / 'rec'}",
            "--artifact",
            f"layout_detection_model_dir={paddle_root / 'layout'}",
            "--output",
            str(paddle_manifest),
        ]
    )
    if paddle_derive["ready_for_preflight"]:
        raise SmokeFailure("Unreviewed Paddle derive claimed readiness")
    paddle_data = json.loads(paddle_manifest.read_text(encoding="utf-8"))
    structure_entries = [
        entry for entry in paddle_data["artifacts"] if entry["role"] == "layout_detection_model_dir"
    ]
    if structure_entries[0]["identifier"] != "REVIEW-REQUIRED":
        raise SmokeFailure("Paddle structure role provenance was auto-filled")

    # Searchable OCR composition from a fabricated sidecar.
    scan_image = base / "scan.png"
    image = Image.new("RGB", (1_275, 1_650), "white")
    draw = ImageDraw.Draw(image)
    draw.text((240, 240), "COMPOSE FIXTURE WORDS", fill="black")
    image.save(scan_image)
    scan_pdf = base / "scan.pdf"
    run_tool(
        [
            "convert",
            "--images",
            str(scan_image),
            "--to",
            "pdf",
            "--output",
            str(scan_pdf),
            "--page-size",
            "letter",
            "--margin",
            "0",
        ]
    )
    sidecar_path = base / "sidecar.json"
    sidecar = {
        "schema_version": 1,
        "operation": "ocr",
        "source": {
            "path": str(scan_pdf.resolve()),
            "sha256": sha256(scan_pdf),
            "selected_pages": [1],
            "immutable": True,
        },
        "pages": [
            {
                "source_page": 1,
                "source_region": None,
                "classification": "likely-scanned",
                "page_geometry": {
                    "width_points": 612,
                    "height_points": 792,
                    "rotation": 0,
                },
                "coordinate_space": {
                    "unit": "rendered_pixel",
                    "dpi": 150,
                    "width": 1275,
                    "height": 1650,
                },
                "blocks": [
                    {
                        "order": 0,
                        "text": "COMPOSE FIXTURE WORDS",
                        "block_type": "line",
                        "bbox": [240, 240, 760, 280],
                        "confidence": 0.9,
                        "words": [
                            {
                                "text": "COMPOSE",
                                "bbox": [240, 240, 420, 280],
                                "confidence": 0.9,
                            },
                            {
                                "text": "FIXTURE",
                                "bbox": [435, 240, 600, 280],
                                "confidence": 0.9,
                            },
                            {
                                "text": "WORDS",
                                "bbox": [615, 240, 760, 280],
                                "confidence": 0.9,
                            },
                        ],
                    }
                ],
                "warnings": [],
                "engine_specific": {},
            }
        ],
    }
    write_json(sidecar_path, sidecar)
    searchable_pdf = base / "searchable.pdf"
    compose = run_tool(
        [
            "ocr-compose",
            "--input",
            str(scan_pdf),
            "--sidecar",
            str(sidecar_path),
            "--output",
            str(searchable_pdf),
        ]
    )
    anchors = compose["verification"]["anchors"]
    if anchors["checked"] == 0 or anchors["checked"] != anchors["found"]:
        raise SmokeFailure(f"Compose anchors failed: {anchors}")
    if compose["verification"]["geometry_spot_check_max_delta_points"] > 3:
        raise SmokeFailure("Compose geometry spot check deviated")
    if not compose["verification"]["visual_diff"]["checked"]:
        raise SmokeFailure("Compose skipped its visual check")
    composed_text = run_tool(["extract", "--input", str(searchable_pdf)])["result"]["pages"][0][
        "text"
    ]
    if "COMPOSE" not in composed_text or "WORDS" not in composed_text:
        raise SmokeFailure(f"Composed text not extractable: {composed_text!r}")
    mismatched = dict(sidecar)
    mismatched["source"] = dict(sidecar["source"], sha256="0" * 64)
    mismatch_path = base / "sidecar-mismatch.json"
    write_json(mismatch_path, mismatched)
    assert_bad_input(
        [
            "ocr-compose",
            "--input",
            str(scan_pdf),
            "--sidecar",
            str(mismatch_path),
            "--output",
            str(base / "never.pdf"),
        ],
        "Compose sidecar hash mismatch",
    )
    digital = dict(sidecar)
    digital_pages = [dict(sidecar["pages"][0], classification="digital-text")]
    digital["pages"] = digital_pages
    digital_path = base / "sidecar-digital.json"
    write_json(digital_path, digital)
    skipped = run_tool(
        [
            "ocr-compose",
            "--input",
            str(scan_pdf),
            "--sidecar",
            str(digital_path),
            "--output",
            str(base / "never2.pdf"),
        ],
        expected_status=3,
    )
    if skipped["category"] != "unsupported_operation":
        raise SmokeFailure(f"Digital-text compose skip category changed: {skipped}")

    # Redaction.
    doc_job = base / "redact-doc.json"
    doc_pdf = base / "redact-doc.pdf"
    write_json(
        doc_job,
        {
            "schema_version": 1,
            "page_size": "letter",
            "pages": [
                {
                    "elements": [
                        {
                            "type": "text",
                            "text": "PUBLIC HEADER LINE",
                            "x": 72,
                            "y": 90,
                            "font_size": 14,
                        },
                        {
                            "type": "text",
                            "text": "CODEWORD REDACT-ME-42 END",
                            "x": 72,
                            "y": 140,
                            "font_size": 14,
                        },
                        {
                            "type": "text",
                            "text": "PUBLIC FOOTER LINE",
                            "x": 72,
                            "y": 190,
                            "font_size": 14,
                        },
                    ]
                }
            ],
        },
    )
    run_tool(["create", "--job", str(doc_job), "--output", str(doc_pdf)])
    # Simulate an incremental-update tail (stale bytes plus a second trailer block
    # that re-points at the original xref) so the rewrite must drop it.
    original_bytes = doc_pdf.read_bytes()
    startxref_offset = original_bytes.rfind(b"startxref")
    if startxref_offset < 0:
        raise SmokeFailure("Fixture PDF has no startxref to duplicate")
    xref_pointer = original_bytes[startxref_offset:].split()[1]
    doc_pdf.write_bytes(
        original_bytes + b"\n% JUNK-REVISION-MARKER-99\nstartxref\n" + xref_pointer + b"\n%%EOF\n"
    )
    redact_job = base / "redact-job.json"
    write_json(
        redact_job,
        {
            "schema_version": 1,
            "targets": [
                {
                    "type": "text",
                    "text": "REDACT-ME-42",
                    "pages": "all",
                    "match": "exact",
                    "grow_points": 2.0,
                }
            ],
        },
    )
    redacted_pdf = base / "redacted.pdf"
    report_path = base / "redact-report.json"
    redact = run_tool(
        [
            "redact",
            "--input",
            str(doc_pdf),
            "--job",
            str(redact_job),
            "--output",
            str(redacted_pdf),
            "--report",
            str(report_path),
        ]
    )
    evidence = redact["removal_evidence"]["pages"][0]
    if evidence["text_operators_removed"] < 1:
        raise SmokeFailure(f"Redaction removed no text operators: {evidence}")
    if evidence["characters_removed"] <= len("REDACT-ME-42"):
        raise SmokeFailure(f"Whole-operator collateral removal was not measured: {evidence}")
    if redact["residual_scan"]["raw_byte_hits"]:
        raise SmokeFailure(f"Residual scan found raw hits: {redact['residual_scan']}")
    if not redact["verification"]["visual_diff"]["checked"]:
        raise SmokeFailure("Redaction skipped its visual diff check")
    redacted_bytes = redacted_pdf.read_bytes()
    if b"REDACT-ME-42" in redacted_bytes:
        raise SmokeFailure("Redacted output still contains the term bytes")
    if b"JUNK-REVISION-MARKER-99" in redacted_bytes:
        raise SmokeFailure("Redacted output retained stale revision bytes")
    if redacted_bytes.count(b"%%EOF") != 1:
        raise SmokeFailure("Redacted output is not a single revision")
    redacted_text = run_tool(["extract", "--input", str(redacted_pdf)])["result"]["pages"][0][
        "text"
    ]
    if "REDACT-ME-42" in redacted_text:
        raise SmokeFailure("Redacted term is still extractable")
    if "PUBLIC HEADER LINE" not in redacted_text or "PUBLIC FOOTER LINE" not in redacted_text:
        raise SmokeFailure(f"Redaction damaged retained text: {redacted_text!r}")
    if not report_path.is_file():
        raise SmokeFailure("Redaction report file was not published")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if any("REDACT-ME-42" in json.dumps(target) for target in report["targets"]):
        raise SmokeFailure("Redaction report leaked the target text")
    no_match_job = base / "redact-nomatch.json"
    write_json(
        no_match_job,
        {
            "schema_version": 1,
            "targets": [{"type": "text", "text": "NO-SUCH-TERM-EXISTS", "pages": "all"}],
        },
    )
    assert_bad_input(
        [
            "redact",
            "--input",
            str(doc_pdf),
            "--job",
            str(no_match_job),
            "--output",
            str(base / "never3.pdf"),
        ],
        "Redaction no-match",
    )
    encrypt_job = base / "redact-encrypt.json"
    encrypted_pdf = base / "redact-encrypted.pdf"
    write_json(
        encrypt_job,
        {
            "schema_version": 1,
            "operations": [
                {
                    "op": "encrypt",
                    "algorithm": "AES-256",
                    "user_password": "redact-smoke-password",
                }
            ],
        },
    )
    run_tool(
        [
            "edit",
            "--input",
            str(doc_pdf),
            "--job",
            str(encrypt_job),
            "--output",
            str(encrypted_pdf),
        ]
    )
    refused = run_tool(
        [
            "redact",
            "--input",
            str(encrypted_pdf),
            "--job",
            str(redact_job),
            "--output",
            str(base / "never4.pdf"),
        ],
        expected_status=3,
    )
    if refused["category"] != "unsupported_operation":
        raise SmokeFailure(f"Encrypted redaction category changed: {refused}")
    form_pdf = base / "form-xobject.pdf"
    reader = PdfReader(str(doc_pdf))
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    page = writer.pages[0]
    form = StreamObject()
    form[NameObject("/Type")] = NameObject("/XObject")
    form[NameObject("/Subtype")] = NameObject("/Form")
    form[NameObject("/BBox")] = ArrayObject(
        [NumberObject(0), NumberObject(0), NumberObject(612), NumberObject(792)]
    )
    form.set_data(b"q Q")
    form_reference = writer._add_object(form)
    resources = page["/Resources"].get_object()
    xobjects = resources.get("/XObject")
    if xobjects is None:
        xobjects = DictionaryObject()
        resources[NameObject("/XObject")] = xobjects
    else:
        xobjects = xobjects.get_object()
    xobjects[NameObject("/FXZ1")] = form_reference
    content = ContentStream(page.get_contents(), writer)
    content.operations.append(([NameObject("/FXZ1")], b"Do"))
    page.replace_contents(content)
    with form_pdf.open("wb") as handle:
        writer.write(handle)
    form_refused = run_tool(
        [
            "redact",
            "--input",
            str(form_pdf),
            "--job",
            str(redact_job),
            "--output",
            str(base / "never5.pdf"),
        ],
        expected_status=3,
    )
    if form_refused["category"] != "unsupported_operation" or "FXZ1" not in form_refused["message"]:
        raise SmokeFailure(f"Form XObject refusal changed: {form_refused}")

    # A large rotated stamp near a target must not be removed: its axis-aligned bounding
    # box overlaps the target rect, but its true oriented glyph quad does not. Redaction
    # tests the oriented quad, so the stamp survives while the target is removed.
    stamp_doc_job = base / "redact-stamp.json"
    stamp_pdf = base / "redact-stamp.pdf"
    write_json(
        stamp_doc_job,
        {
            "schema_version": 1,
            "page_size": "letter",
            "pages": [
                {
                    "elements": [
                        {
                            "type": "text",
                            "text": "SECRET GAMMA-88 CODE",
                            "x": 72,
                            "y": 700,
                            "width": 300,
                            "font_size": 12,
                        }
                    ]
                }
            ],
        },
    )
    run_tool(["create", "--job", str(stamp_doc_job), "--output", str(stamp_pdf)])
    stamp_edit_job = base / "redact-stamp-edit.json"
    stamped_pdf = base / "redact-stamped.pdf"
    write_json(
        stamp_edit_job,
        {
            "schema_version": 1,
            "operations": [
                {
                    "op": "stamp_text",
                    "pages": "1",
                    "text": "CONFIDENTIAL",
                    "x": 306,
                    "y": 300,
                    "angle": 45,
                    "font_size": 72,
                    "opacity": 0.3,
                }
            ],
        },
    )
    run_tool(
        [
            "edit",
            "--input",
            str(stamp_pdf),
            "--job",
            str(stamp_edit_job),
            "--output",
            str(stamped_pdf),
        ]
    )
    stamp_redact_job = base / "redact-stamp-target.json"
    write_json(
        stamp_redact_job,
        {
            "schema_version": 1,
            "targets": [{"type": "text", "text": "GAMMA-88", "pages": "all", "match": "exact"}],
        },
    )
    stamp_redacted = base / "redact-stamp-out.pdf"
    stamp_redaction = run_tool(
        [
            "redact",
            "--input",
            str(stamped_pdf),
            "--job",
            str(stamp_redact_job),
            "--output",
            str(stamp_redacted),
        ]
    )
    # Assert on removal evidence, not extracted text: pdfplumber renders the 45-degree
    # stamp as one letter per line, so the contiguous word never appears in extraction
    # even when every stamp glyph survives. Exactly one operator (the target line, 20
    # chars) must be removed; a regression to axis-aligned extents would also remove the
    # stamp operator, doubling both counts.
    stamp_evidence = stamp_redaction["removal_evidence"]["pages"][0]
    if stamp_evidence["text_operators_removed"] != 1:
        raise SmokeFailure(
            "Rotated stamp removed by a distant target — oriented-quad extent regressed "
            f"to an axis-aligned bounding box: {stamp_evidence}"
        )
    if stamp_evidence["characters_removed"] != len("SECRET GAMMA-88 CODE"):
        raise SmokeFailure(f"Unexpected redaction collateral near stamp: {stamp_evidence}")
    stamp_text = run_tool(["extract", "--input", str(stamp_redacted)])["result"]["pages"][0]["text"]
    if "GAMMA-88" in stamp_text:
        raise SmokeFailure("Redaction target survived near a rotated stamp")
    # The stamp glyphs (CONFIDENTIAL, rendered reversed one-per-line) must still be present.
    if "".join(stamp_text.split()).count("C") < 1 or "AITNEDIFNOC" not in "".join(
        stamp_text.split()
    ):
        raise SmokeFailure(f"Rotated stamp glyphs did not survive redaction: {stamp_text!r}")

    return {
        "manifest_derive_and_check": True,
        "manifest_structure_roles_never_autofilled": True,
        "ocr_compose": True,
        "ocr_compose_failure_paths": True,
        "redact_text_target": True,
        "redact_single_revision": True,
        "redact_failure_paths": True,
        "redact_form_xobject_refused": True,
        "redact_oriented_quad_spares_rotated_stamp": True,
    }


def run_optional_ocr(
    root: Path,
    *,
    engine: str,
    manifest: Path,
    executable: str | None,
    languages: str,
    device: str,
) -> dict[str, Any]:
    clean_image = root / "ocr-clean.png"
    complex_image = root / "ocr-complex.png"
    write_test_image(clean_image)
    write_test_image(complex_image, complex_layout=True)
    fixture_pdf = root / "ocr-fixture.pdf"
    job = root / "ocr-fixture.json"
    write_json(
        job,
        {
            "schema_version": 1,
            "pages": [
                {
                    "elements": [
                        {
                            "type": "image",
                            "path": str(clean_image),
                            "x": 0,
                            "y": 0,
                            "width": 612,
                            "height": 792,
                            "preserve_aspect_ratio": False,
                        }
                    ]
                },
                {
                    "elements": [
                        {
                            "type": "image",
                            "path": str(complex_image),
                            "x": 0,
                            "y": 0,
                            "width": 612,
                            "height": 792,
                            "preserve_aspect_ratio": False,
                        }
                    ]
                },
            ],
        },
    )
    run_tool(["create", "--job", str(job), "--output", str(fixture_pdf)])

    preflight_args = [
        "ocr",
        "--input",
        str(fixture_pdf),
        "--output",
        str(root / "unused-preflight.json"),
        "--engine",
        engine,
        "--model-manifest",
        str(manifest),
        "--languages",
        languages,
        "--preflight-only",
    ]
    if executable:
        preflight_args.extend(["--engine-executable", executable])
    preflight = run_tool(preflight_args)
    if preflight["preflight"]["license_preflight"] != "passed":
        raise SmokeFailure("Optional engine license/model preflight did not pass")
    preflight_model = preflight["preflight"].get("model", {})
    if not preflight.get("device", {}).get("resolved_device"):
        raise SmokeFailure("Optional preflight omitted resolved device metadata")
    expected_languages = [item.strip() for item in languages.split(",")]
    if preflight.get("languages") != expected_languages:
        raise SmokeFailure("Optional preflight omitted parsed OCR language metadata")
    if preflight.get("warnings") is None or not isinstance(preflight["warnings"], list):
        raise SmokeFailure("Optional preflight warnings are not an array")

    selected_pages = "1" if engine == "tesseract" else "1-2"
    result_path = root / f"ocr-{engine}.json"
    raw_dir = root / f"ocr-{engine}-raw"
    ocr_args = [
        "ocr",
        "--input",
        str(fixture_pdf),
        "--output",
        str(result_path),
        "--raw-output-dir",
        str(raw_dir),
        "--engine",
        engine,
        "--model-manifest",
        str(manifest),
        "--languages",
        languages,
        "--pages",
        selected_pages,
        "--device",
        device,
        "--timeout",
        "1200",
    ]
    if executable:
        ocr_args.extend(["--engine-executable", executable])
    if engine == "paddle":
        ocr_args.extend(["--paddle-pipeline", "structure"])
    source_hash_before = sha256(fixture_pdf)
    ocr_summary = run_tool(ocr_args)
    normalized = json.loads(result_path.read_text(encoding="utf-8"))
    if normalized.get("schema_version") != 1 or normalized.get("engine", {}).get("name") != engine:
        raise SmokeFailure("Normalized OCR envelope is invalid")
    if normalized.get("source", {}).get("sha256") != source_hash_before:
        raise SmokeFailure("Normalized OCR source hash does not match the input PDF")
    if not normalized.get("pages"):
        raise SmokeFailure("Normalized OCR contains no pages")
    engine_metadata = normalized["engine"]
    if not isinstance(engine_metadata.get("version"), str) or not engine_metadata["version"]:
        raise SmokeFailure("Normalized OCR omitted engine version")
    if engine_metadata.get("model_identifier") != preflight_model.get("identifier"):
        raise SmokeFailure("Normalized OCR model identifier differs from preflight")
    if engine_metadata.get("model_revision") != preflight_model.get("revision"):
        raise SmokeFailure("Normalized OCR model revision differs from preflight")
    if engine_metadata.get("languages") != expected_languages:
        raise SmokeFailure("Normalized OCR omitted parsed OCR language metadata")
    if engine_metadata.get("requested_device") != device:
        raise SmokeFailure("Normalized OCR omitted the requested-device audit field")
    if engine_metadata.get("resolved_device") in {None, "", "auto", "mps"}:
        raise SmokeFailure(f"OCR device was not actually resolved: {engine_metadata}")
    if engine == "tesseract" and (
        engine_metadata.get("resolved_device") != "cpu"
        or engine_metadata.get("runtime_backend") != "tesseract-cpu"
    ):
        raise SmokeFailure(f"Tesseract did not report its CPU backend: {engine_metadata}")
    if engine == "paddle" and device == "mps" and engine_metadata.get("resolved_device") != "cpu":
        raise SmokeFailure(f"Paddle MPS request was not resolved to CPU: {engine_metadata}")
    if engine == "surya" and (
        engine_metadata.get("resolved_device") != "backend-managed"
        or engine_metadata.get("runtime_backend") != "llamacpp"
    ):
        raise SmokeFailure(f"Surya did not report its actual runtime backend: {engine_metadata}")
    if normalized.get("warnings") is None or not isinstance(normalized["warnings"], list):
        raise SmokeFailure("Normalized OCR warnings are not an array")
    blocks = [block for page in normalized["pages"] for block in page.get("blocks", [])]
    if not blocks or not any(block.get("text", "").strip() for block in blocks):
        raise SmokeFailure("Normalized OCR contains no recognized text")
    if not any(block.get("bbox") or block.get("polygon") for block in blocks):
        raise SmokeFailure("Normalized OCR contains no supplied coordinates")
    for page in normalized["pages"]:
        if not isinstance(page.get("source_page"), int) or page["source_page"] < 1:
            raise SmokeFailure(f"Invalid normalized source page: {page}")
        if not isinstance(page.get("warnings"), list):
            raise SmokeFailure(f"Page warnings are not an array: {page}")
        width = page["coordinate_space"]["width"]
        height = page["coordinate_space"]["height"]
        orders = [block["order"] for block in page.get("blocks", [])]
        if orders != sorted(orders) or len(orders) != len(set(orders)):
            raise SmokeFailure(f"Block order is not deterministic and unique: {orders}")
        for block in page.get("blocks", []):
            confidence = block.get("confidence")
            if confidence is not None and (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence)
                or not 0 <= confidence <= 1
            ):
                raise SmokeFailure(f"Invalid engine-supplied confidence: {confidence}")
            box = block.get("bbox")
            if box is not None and (
                len(box) != 4
                or not all(math.isfinite(value) and value >= 0 for value in box)
                or not (box[0] < box[2] <= width and box[1] < box[3] <= height)
            ):
                raise SmokeFailure(f"Invalid normalized bounding box: {box}")
            polygon = block.get("polygon")
            if polygon is not None and (
                len(polygon) < 3
                or any(
                    len(point) != 2
                    or not all(math.isfinite(value) for value in point)
                    or not (0 <= point[0] <= width and 0 <= point[1] <= height)
                    for point in polygon
                )
            ):
                raise SmokeFailure(f"Invalid normalized polygon: {polygon}")
    if not raw_dir.is_dir() or not any(raw_dir.rglob("*")):
        raise SmokeFailure("Raw OCR output was not retained")
    if normalized.get("raw_output_path") != str(raw_dir.resolve()):
        raise SmokeFailure("Normalized OCR raw-output path does not match the retained directory")
    if ocr_summary.get("raw_output_path") != str(raw_dir.resolve()):
        raise SmokeFailure("OCR summary raw-output path does not match the retained directory")
    if sha256(fixture_pdf) != source_hash_before:
        raise SmokeFailure("Optional OCR modified the source PDF")

    complex_plan = run_tool(
        [
            "ocr-plan",
            "--input",
            str(fixture_pdf),
            "--pages",
            "2",
            "--languages",
            languages,
            "--hardware",
            device,
            "--layout",
            "complex",
        ]
    )
    if complex_plan["recommendation"]["engine"] == "tesseract":
        raise SmokeFailure("Tesseract was selected for the complex fixture")
    if not complex_plan["recommendation"]["rationale"]:
        raise SmokeFailure("Complex OCR recommendation has no rationale")
    return {
        "engine": engine,
        "preflight": True,
        "engine_version": engine_metadata["version"],
        "model_revision": engine_metadata["model_revision"],
        "resolved_device": engine_metadata["resolved_device"],
        "runtime_backend": engine_metadata["runtime_backend"],
        "confidence_and_warnings_validated": True,
        "normalized_blocks": len(blocks),
        "raw_output_retained": True,
        "complex_plan_excludes_tesseract": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocr-engine", choices=("surya", "paddle", "tesseract"))
    parser.add_argument("--model-manifest")
    parser.add_argument("--engine-executable")
    parser.add_argument("--languages", default="eng")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    return parser


def main() -> int:
    try:
        args = build_parser().parse_args()
        if bool(args.ocr_engine) != bool(args.model_manifest):
            raise SmokeFailure("--ocr-engine and --model-manifest must be supplied together")
        with tempfile.TemporaryDirectory(prefix="pdf-skill-smoke-") as temp_name:
            root = Path(temp_name)
            core = run_core(root)
            core.update(run_render_and_fonts(root))
            core.update(run_manifest_compose_redact(root))
            optional = None
            if args.ocr_engine:
                optional = run_optional_ocr(
                    root,
                    engine=args.ocr_engine,
                    manifest=Path(args.model_manifest),
                    executable=args.engine_executable,
                    languages=args.languages,
                    device=args.device,
                )
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "ok",
                    "core": core,
                    "optional_ocr": optional,
                    "optional_profiles_untested": []
                    if optional
                    else ["surya", "paddle", "tesseract"],
                },
                sort_keys=True,
            )
        )
        return 0
    except (SmokeFailure, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "error",
                    "message": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
