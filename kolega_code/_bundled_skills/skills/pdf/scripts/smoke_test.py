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
from pypdf import PdfReader

SCHEMA_VERSION = 1
TOOL = Path(__file__).with_name("pdf_tool.py")


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
