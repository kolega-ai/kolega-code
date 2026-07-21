#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate temporary PPTX fixtures and exercise the public CLI end to end."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
import zlib
from pathlib import Path
from typing import Any

DEPENDENCY_ERROR: ModuleNotFoundError | None = None
Image: Any = None
ImageDraw: Any = None
Presentation: Any = None
OxmlElement: Any = None
PdfReader: Any = None
PptxTool: Any = None
try:
    import pptx_tool as PptxToolModule
    from PIL import Image as PillowImage
    from PIL import ImageDraw as PillowImageDraw
    from pptx import Presentation as PptxPresentation
    from pptx.oxml.xmlchemy import OxmlElement as PptxOxmlElement
    from pypdf import PdfReader as PypdfReader

    Image = PillowImage
    ImageDraw = PillowImageDraw
    Presentation = PptxPresentation
    OxmlElement = PptxOxmlElement
    PdfReader = PypdfReader
    PptxTool = PptxToolModule
except ModuleNotFoundError as exc:
    DEPENDENCY_ERROR = exc

TOOL = Path(__file__).with_name("pptx_tool.py")


class SmokeFailure(Exception):
    """Raised when a smoke-test assertion fails."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _run(
    arguments: list[str],
    *,
    expected_status: int = 0,
    timeout_seconds: float = 120,
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [sys.executable, str(TOOL), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        operation = arguments[0] if arguments else "<unknown>"
        raise SmokeFailure(
            f"{operation} subprocess timed out after {timeout_seconds:g} seconds"
        ) from exc
    if completed.returncode != expected_status:
        raise SmokeFailure(
            f"command status {completed.returncode}, expected {expected_status}: "
            f"{' '.join(arguments)}\nstdout={completed.stdout}\nstderr={completed.stderr}"
        )
    payload_text = completed.stdout if expected_status == 0 else completed.stderr
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise SmokeFailure(f"command did not emit JSON: {payload_text!r}") from exc
    if expected_status == 0 and payload.get("success") is not True:
        raise SmokeFailure(f"successful command reported failure: {payload}")
    if expected_status != 0 and payload.get("success") is not False:
        raise SmokeFailure(f"failed command reported success: {payload}")
    return payload


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def _make_image(path: Path, colors: tuple[str, str], label: str) -> None:
    image = Image.new("RGB", (320, 180), colors[0])
    draw = ImageDraw.Draw(image)
    draw.rectangle((18, 18, 302, 162), outline=colors[1], width=8)
    draw.text((46, 74), label, fill=colors[1])
    image.save(path, format="PNG")


def _make_oversized_dimension_png(path: Path) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)

    header = struct.pack(">IIBBBBB", 20_001, 1, 8, 2, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IEND", b""))


def _rewrite_zip(
    source: Path,
    destination: Path,
    *,
    replacements: dict[str, bytes] | None = None,
    additions: dict[str, bytes] | None = None,
) -> None:
    replacements = replacements or {}
    additions = additions or {}
    with zipfile.ZipFile(source) as source_archive:
        members = {
            info.filename: replacements.get(info.filename, source_archive.read(info))
            for info in source_archive.infolist()
        }
    members.update(additions)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for name, data in members.items():
            output.writestr(name, data)


def _shape_with_name(slide: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [shape for shape in slide["shapes"] if shape["name"] == name]
    if len(matches) != 1:
        raise SmokeFailure(f"expected one shape named {name!r}, found {len(matches)}")
    return matches[0]


def _shape_containing(slide: dict[str, Any], text: str) -> dict[str, Any]:
    matches = [shape for shape in slide["shapes"] if text in shape.get("text", {}).get("text", "")]
    if len(matches) != 1:
        raise SmokeFailure(f"expected one shape containing {text!r}, found {len(matches)}")
    return matches[0]


def _exercise_publication_races(work: Path) -> None:
    source = work / "race-source.pptx"
    changed = work / "race-changed.pptx"
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[0])
    if slide.shapes.title is None:
        raise SmokeFailure("race fixture layout unexpectedly lacks a title")
    slide.shapes.title.text = "Baseline"
    presentation.save(str(source))
    baseline_bytes = source.read_bytes()
    baseline_hash = hashlib.sha256(baseline_bytes).hexdigest()

    changed_presentation = Presentation(str(source))
    changed_presentation.slides[0].shapes.title.text = "Changed"
    changed_presentation.save(str(changed))
    changed_bytes = changed.read_bytes()

    before_gate_output = work / "race-before-gate.pptx"
    source.write_bytes(changed_bytes)
    try:
        PptxTool._save_atomic_presentation(
            Presentation(io.BytesIO(baseline_bytes)),
            before_gate_output,
            [int(slide.slide_id) for slide in Presentation(io.BytesIO(baseline_bytes)).slides],
            False,
            source,
            baseline_hash,
        )
    except PptxTool.ToolError as exc:
        _assert(
            exc.category == "post_write_validation",
            "pre-publication source race failure category mismatch",
        )
    else:
        raise SmokeFailure("pre-publication source race was not rejected")
    _assert(
        not before_gate_output.exists(),
        "pre-publication source race published a destination",
    )

    source.write_bytes(baseline_bytes)
    after_gate_output = work / "race-after-gate.pptx"
    original_publish = PptxTool._atomic_publish

    def publish_then_change(temporary: Path, destination: Path, overwrite: bool) -> None:
        original_publish(temporary, destination, overwrite)
        source.write_bytes(changed_bytes)

    PptxTool._atomic_publish = publish_then_change
    try:
        output_presentation = Presentation(io.BytesIO(baseline_bytes))
        expected_ids = [int(item.slide_id) for item in output_presentation.slides]
        _, post_publish_source_changed = PptxTool._save_atomic_presentation(
            output_presentation,
            after_gate_output,
            expected_ids,
            False,
            source,
            baseline_hash,
        )
    finally:
        PptxTool._atomic_publish = original_publish
    _assert(
        post_publish_source_changed,
        "post-publication source race was not reported",
    )
    _assert(
        after_gate_output.is_file(),
        "post-publication source race lost the committed destination",
    )
    Presentation(str(after_gate_output))


def _run_smoke(require_libreoffice: bool) -> dict[str, Any]:
    checks: list[str] = []
    conversion: dict[str, Any]
    with tempfile.TemporaryDirectory(prefix="pptx-skill-smoke-") as temp_dir:
        work = Path(temp_dir)
        first_image = work / "first.png"
        second_image = work / "second.png"
        _make_image(first_image, ("#E9F3F0", "#0D5545"), "FIRST")
        _make_image(second_image, ("#FFF1D6", "#A63B21"), "SECOND")

        template = work / "template.pptx"
        template_presentation = Presentation()
        seed = template_presentation.slides.add_slide(template_presentation.slide_layouts[5])
        seed_title = seed.shapes.title
        if seed_title is None:
            raise SmokeFailure("fixture layout unexpectedly lacks a title placeholder")
        seed_title.text = "Template seed slide"
        template_presentation.save(str(template))
        template_hash = _sha256(template)

        create_job = work / "create.json"
        created = work / "created.pptx"
        _write_json(
            create_job,
            {
                "schema_version": 1,
                "operation": "create",
                "template": "template.pptx",
                "keep_template_slides": False,
                "metadata": {
                    "title": "Smoke test deck",
                    "author": "PPTX skill",
                },
                "slides": [
                    {
                        "layout": {"index": 0},
                        "title": "Operations brief",
                        "body": "Generated fixture",
                        "elements": [
                            {
                                "type": "image",
                                "name": "Logo",
                                "path": "first.png",
                                "box": {
                                    "x": 4.8,
                                    "y": 4.8,
                                    "width": 2.4,
                                },
                            },
                            {
                                "type": "image",
                                "name": "Shared Image",
                                "path": "first.png",
                                "box": {
                                    "x": 7.5,
                                    "y": 4.8,
                                    "width": 1.2,
                                },
                            },
                        ],
                        "notes": "Original opening note",
                    },
                    {
                        "layout": {"index": 1},
                        "title": "Editable evidence",
                        "body": [
                            {
                                "runs": [
                                    {"text": "Run", "bold": True, "color": "0D5545"},
                                    {"text": " aware", "italic": True},
                                    {"text": " replacement"},
                                ]
                            }
                        ],
                        "elements": [
                            {
                                "type": "table",
                                "name": "Metrics Table",
                                "box": {
                                    "x": 0.8,
                                    "y": 3.2,
                                    "width": 4.0,
                                    "height": 2.0,
                                },
                                "data": [["Metric", "Value"], ["Orders", "12"]],
                            },
                            {
                                "type": "chart",
                                "chart_type": "column",
                                "name": "Sales Chart",
                                "box": {
                                    "x": 5.0,
                                    "y": 2.5,
                                    "width": 4.2,
                                    "height": 3.2,
                                },
                                "categories": ["Q1", "Q2"],
                                "series": [{"name": "Sales", "values": [10, 15]}],
                                "title": "Sales",
                                "has_legend": False,
                            },
                        ],
                        "notes": "Evidence note",
                    },
                    {
                        "layout": {"index": 5},
                        "title": "Appendix anchor",
                        "elements": [
                            {
                                "type": "text",
                                "name": "Appendix Text",
                                "box": {
                                    "x": 1.0,
                                    "y": 2.0,
                                    "width": 7.0,
                                    "height": 1.5,
                                },
                                "paragraphs": [
                                    {
                                        "text": "Retained slide identity",
                                        "alignment": "center",
                                    }
                                ],
                            }
                        ],
                    },
                ],
            },
        )
        create_result = _run(["create", "--job", str(create_job), "--output", str(created)])
        _assert(create_result["counts"]["slides"] == 3, "create slide count mismatch")
        _assert(_sha256(template) == template_hash, "create modified its template")
        _assert(
            len(create_result["verification"]["graph_checkpoints"]) == 4,
            "template removal and slide additions were not checkpointed",
        )
        _assert(
            create_result["verification"]["internal_relationship_targets_exist"]
            and create_result["verification"]["owner_relationship_references_resolved"],
            "create relationship target/reference verification missing",
        )
        Presentation(str(created))
        checks.append("create/reopen")

        created_hash = _sha256(created)
        first_inspection = _run(["inspect", str(created)])
        _assert(_sha256(created) == created_hash, "inspect modified its source")
        slides = first_inspection["presentation"]["slides"]
        original_ids = [slide["slide_id"] for slide in slides]
        _assert(len(set(original_ids)) == 3, "slide IDs are not unique")
        _assert(first_inspection["counts"]["images"] == 2, "image inventory mismatch")
        _assert(first_inspection["counts"]["tables"] == 1, "table inventory mismatch")
        _assert(first_inspection["counts"]["charts"] == 1, "chart inventory mismatch")
        _assert(slides[0]["notes"] == "Original opening note", "notes inspection mismatch")
        body_shape = _shape_containing(slides[1], "Run aware replacement")
        body_name = body_shape["name"]
        checks.append("inspect/content-inventory")

        first_edit_job = work / "edit-content.json"
        first_edit = work / "edited-content.pptx"
        _write_json(
            first_edit_job,
            {
                "schema_version": 1,
                "operation": "edit",
                "operations": [
                    {
                        "action": "replace_text",
                        "slide": {"slide_id": original_ids[1]},
                        "shape": {"shape_name": body_name},
                        "find": "Run aware",
                        "replace": "Style-safe",
                        "formatting_policy": "first_run",
                    },
                    {
                        "action": "update_table",
                        "shape": {"table_index": 0},
                        "cells": [{"row": 1, "column": 1, "text": "14"}],
                    },
                    {
                        "action": "update_chart",
                        "shape": {"chart_index": 0},
                        "chart_type": "column",
                        "categories": ["Q1", "Q2"],
                        "series": [{"name": "Sales", "values": [11, 18]}],
                        "title": "Updated sales",
                        "has_legend": False,
                    },
                    {
                        "action": "replace_image",
                        "shape": {"image_index": 0},
                        "path": "second.png",
                    },
                    {
                        "action": "set_notes",
                        "slide": {"slide_id": original_ids[0]},
                        "text": "Updated opening note",
                    },
                    {
                        "action": "add_slide",
                        "index": 1,
                        "slide": {
                            "layout": {"index": 5},
                            "title": "Inserted decision",
                            "elements": [
                                {
                                    "type": "text",
                                    "name": "Inserted Text",
                                    "box": {
                                        "x": 1.0,
                                        "y": 2.0,
                                        "width": 7.0,
                                        "height": 1.0,
                                    },
                                    "paragraphs": [{"text": "New slide content"}],
                                }
                            ],
                            "notes": "Inserted note",
                        },
                    },
                ],
            },
        )
        first_edit_result = _run(
            [
                "edit",
                str(created),
                "--job",
                str(first_edit_job),
                "--output",
                str(first_edit),
            ]
        )
        _assert(_sha256(created) == created_hash, "edit modified distinct source")
        _assert(
            first_edit_result["counts"]["text_replacements"] == 1,
            "cross-run replacement count mismatch",
        )
        _assert(
            len(first_edit_result["verification"]["graph_checkpoints"]) == 1,
            "insert reorder was not checkpointed",
        )
        second_inspection = _run(["inspect", str(first_edit)])
        second_slides = second_inspection["presentation"]["slides"]
        second_ids = [slide["slide_id"] for slide in second_slides]
        _assert(all(slide_id in second_ids for slide_id in original_ids), "retained ID changed")
        new_ids = [slide_id for slide_id in second_ids if slide_id not in original_ids]
        _assert(len(new_ids) == 1 and second_ids[1] == new_ids[0], "inserted slide order mismatch")
        evidence = next(slide for slide in second_slides if slide["slide_id"] == original_ids[1])
        _assert(
            "Style-safe replacement" in _shape_containing(evidence, "Style-safe")["text"]["text"],
            "cross-run replacement content mismatch",
        )
        _assert(
            _shape_with_name(evidence, "Metrics Table")["table"]["data"][1][1] == "14",
            "table update mismatch",
        )
        chart = _shape_with_name(evidence, "Sales Chart")["chart"]
        _assert(chart["title"] == "Updated sales", "chart title update mismatch")
        _assert(chart["series"][0]["values"] == [11.0, 18.0], "chart data update mismatch")
        opening = next(slide for slide in second_slides if slide["slide_id"] == original_ids[0])
        _assert(opening["notes"] == "Updated opening note", "note update mismatch")
        logo = _shape_with_name(opening, "Logo")["image"]
        _assert(
            logo["sha256"] == hashlib.sha256(second_image.read_bytes()).hexdigest(),
            "image replacement mismatch",
        )
        shared_image = _shape_with_name(opening, "Shared Image")["image"]
        _assert(
            shared_image["sha256"] == hashlib.sha256(first_image.read_bytes()).hexdigest(),
            "replacement changed a picture that shared the original image part",
        )
        checks.append("edit/text-global-table-chart-image-notes-add")

        graph_job = work / "edit-graph.json"
        final_deck = work / "final.pptx"
        requested_order = [new_ids[0], original_ids[2], original_ids[0], original_ids[1]]
        _write_json(
            graph_job,
            {
                "schema_version": 1,
                "operation": "edit",
                "operations": [
                    {"action": "reorder_slides", "slide_ids": requested_order},
                    {
                        "action": "remove_slide",
                        "slide": {"slide_id": original_ids[1]},
                    },
                ],
            },
        )
        first_edit_hash = _sha256(first_edit)
        graph_result = _run(
            [
                "edit",
                str(first_edit),
                "--job",
                str(graph_job),
                "--output",
                str(final_deck),
            ]
        )
        _assert(_sha256(first_edit) == first_edit_hash, "graph edit modified source")
        expected_final_ids = [new_ids[0], original_ids[2], original_ids[0]]
        _assert(
            graph_result["verification"]["slide_ids"] == expected_final_ids,
            "graph result order mismatch",
        )
        _assert(
            len(graph_result["verification"]["graph_checkpoints"]) == 2,
            "remove/reorder checkpoints missing",
        )
        final_inspection = _run(["inspect", str(final_deck)])
        _assert(
            final_inspection["verification"]["slide_ids"] == expected_final_ids,
            "final retained ID/order mismatch",
        )
        Presentation(str(final_deck))
        checks.append("edit/reorder-remove-id-relationships")

        append_job = work / "edit-append.json"
        appended = work / "appended.pptx"
        _write_json(
            append_job,
            {
                "schema_version": 1,
                "operation": "edit",
                "operations": [
                    {
                        "action": "replace_text",
                        "shape": {"shape_name": "Appendix Text"},
                        "find": "Retained slide identity",
                        "replace": "Globally selected identity",
                    },
                    {
                        "action": "add_slide",
                        "slide": {
                            "layout": {"index": 5},
                            "title": "Appended checkpoint",
                        },
                    },
                ],
            },
        )
        append_result = _run(
            ["edit", str(final_deck), "--job", str(append_job), "--output", str(appended)]
        )
        _assert(
            len(append_result["verification"]["graph_checkpoints"]) == 1,
            "plain append was not checkpointed",
        )
        appended_inspection = _run(["inspect", str(appended)])
        _assert(
            any(
                "Globally selected identity" in shape.get("text", {}).get("text", "")
                for slide in appended_inspection["presentation"]["slides"]
                for shape in slide["shapes"]
            ),
            "deck-wide shape selector did not resolve globally",
        )
        checks.append("edit/append-checkpoint")

        same_path_hash = _sha256(final_deck)
        same_path_failure = _run(
            [
                "edit",
                str(final_deck),
                "--job",
                str(append_job),
                "--output",
                str(final_deck),
                "--overwrite",
            ],
            expected_status=2,
        )
        _assert(
            same_path_failure["error"]["category"] == "bad_input",
            "same-path edit failure category mismatch",
        )
        _assert(_sha256(final_deck) == same_path_hash, "same-path rejection changed source")
        checks.append("failure/immutable-source")

        occupied = work / "occupied.pptx"
        occupied.write_bytes(b"sentinel")
        occupied_hash = _sha256(occupied)
        no_overwrite_failure = _run(
            ["edit", str(final_deck), "--job", str(append_job), "--output", str(occupied)],
            expected_status=2,
        )
        _assert(
            no_overwrite_failure["error"]["category"] == "bad_input",
            "no-overwrite failure category mismatch",
        )
        _assert(_sha256(occupied) == occupied_hash, "no-overwrite changed destination")
        checks.append("failure/no-overwrite")
        overwrite_result = _run(
            [
                "edit",
                str(final_deck),
                "--job",
                str(append_job),
                "--output",
                str(occupied),
                "--overwrite",
            ]
        )
        _assert(
            overwrite_result["verification"]["atomic_publish"],
            "overwrite publication verification missing",
        )
        Presentation(str(occupied))
        checks.append("edit/overwrite-distinct-destination")

        _exercise_publication_races(work)
        checks.append("publication/source-race-boundary")

        corrupt = work / "corrupt.pptx"
        corrupt.write_bytes(b"not a zip package")
        failure = _run(["inspect", str(corrupt)], expected_status=2)
        _assert(
            failure["error"]["category"] == "bad_input",
            "corrupt input failure category mismatch",
        )
        checks.append("failure/corrupt-package")

        with zipfile.ZipFile(created) as archive:
            slide_xml = archive.read("ppt/slides/slide1.xml")
        dangling_relationship = work / "dangling-relationship.pptx"
        _rewrite_zip(
            created,
            dangling_relationship,
            replacements={
                "ppt/slides/slide1.xml": slide_xml.replace(
                    b"<p:cSld",
                    b'<p:cSld r:id="rIdMissing"',
                    1,
                )
            },
        )
        dangling_failure = _run(
            ["inspect", str(dangling_relationship)],
            expected_status=2,
        )
        _assert(
            dangling_failure["error"]["category"] == "bad_input"
            and "relationship ID" in dangling_failure["error"]["message"],
            "dangling owner relationship ID was not rejected",
        )

        orphan_relationship = work / "orphan-relationship.pptx"
        _rewrite_zip(
            created,
            orphan_relationship,
            additions={
                "ppt/orphan/_rels/ghost.xml.rels": (
                    b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                    b'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/'
                    b'relationships"/>'
                )
            },
        )
        orphan_failure = _run(
            ["inspect", str(orphan_relationship)],
            expected_status=2,
        )
        _assert(
            orphan_failure["error"]["category"] == "bad_input"
            and "owner is missing" in orphan_failure["error"]["message"],
            "orphan relationship part was not rejected",
        )
        checks.append("failure/malformed-relationships")

        oversized_image = work / "oversized-dimension.png"
        _make_oversized_dimension_png(oversized_image)
        oversized_job = work / "oversized-image.json"
        _write_json(
            oversized_job,
            {
                "schema_version": 1,
                "operation": "create",
                "slides": [
                    {
                        "layout": {"index": 5},
                        "title": "Oversized image",
                        "elements": [
                            {
                                "type": "image",
                                "path": "oversized-dimension.png",
                                "box": {"x": 1, "y": 1, "width": 2},
                            }
                        ],
                    }
                ],
            },
        )
        oversized_failure = _run(
            [
                "create",
                "--job",
                str(oversized_job),
                "--output",
                str(work / "oversized-image.pptx"),
            ],
            expected_status=6,
        )
        _assert(
            oversized_failure["error"]["category"] == "resource_limit",
            "decoded image dimension failure category mismatch",
        )
        checks.append("failure/image-dimension-limit")

        oversized_bytes = work / "oversized-bytes.png"
        with oversized_bytes.open("wb") as oversized_stream:
            oversized_stream.seek(50 * 1024 * 1024)
            oversized_stream.write(b"x")
        oversized_bytes_job = work / "oversized-image-bytes.json"
        _write_json(
            oversized_bytes_job,
            {
                "schema_version": 1,
                "operation": "create",
                "slides": [
                    {
                        "layout": {"index": 5},
                        "title": "Oversized image bytes",
                        "elements": [
                            {
                                "type": "image",
                                "path": "oversized-bytes.png",
                                "box": {"x": 1, "y": 1, "width": 2},
                            }
                        ],
                    }
                ],
            },
        )
        oversized_bytes_failure = _run(
            [
                "create",
                "--job",
                str(oversized_bytes_job),
                "--output",
                str(work / "oversized-image-bytes.pptx"),
            ],
            expected_status=6,
        )
        _assert(
            oversized_bytes_failure["error"]["category"] == "resource_limit",
            "compressed image byte limit failure category mismatch",
        )
        checks.append("failure/image-byte-limit")

        field_deck = work / "field-boundaries.pptx"
        field_presentation = Presentation()
        field_slide = field_presentation.slides.add_slide(field_presentation.slide_layouts[0])
        field_title = field_slide.shapes.title
        if field_title is None:
            raise SmokeFailure("field fixture layout unexpectedly lacks a title")
        field_paragraph = field_title.text_frame.paragraphs[0]
        field_paragraph.text = "Safe before "
        field = OxmlElement("a:fld")
        field.set("id", "{00000000-0000-0000-0000-000000000001}")
        field.set("type", "slidenum")
        field_text = OxmlElement("a:t")
        field_text.text = "1"
        field.append(field_text)
        field_paragraph._p.append(field)
        field_paragraph.add_run().text = " safe after"
        field_presentation.save(str(field_deck))

        field_safe_job = work / "field-safe.json"
        field_safe_output = work / "field-safe.pptx"
        _write_json(
            field_safe_job,
            {
                "schema_version": 1,
                "operation": "edit",
                "operations": [
                    {
                        "action": "replace_text",
                        "find": "safe after",
                        "replace": "updated after",
                    }
                ],
            },
        )
        _run(
            [
                "edit",
                str(field_deck),
                "--job",
                str(field_safe_job),
                "--output",
                str(field_safe_output),
            ]
        )
        field_safe_inspection = _run(["inspect", str(field_safe_output)])
        _assert(
            any(
                "updated after" in shape.get("text", {}).get("text", "")
                for slide in field_safe_inspection["presentation"]["slides"]
                for shape in slide["shapes"]
            ),
            "safe replacement after a field was rejected or lost",
        )

        field_unsafe_job = work / "field-unsafe.json"
        _write_json(
            field_unsafe_job,
            {
                "schema_version": 1,
                "operation": "edit",
                "operations": [
                    {
                        "action": "replace_text",
                        "find": "1 safe after",
                        "replace": "unsafe",
                    }
                ],
            },
        )
        field_failure = _run(
            [
                "edit",
                str(field_deck),
                "--job",
                str(field_unsafe_job),
                "--output",
                str(work / "field-unsafe.pptx"),
            ],
            expected_status=5,
        )
        _assert(
            field_failure["error"]["category"] == "ambiguous_edit",
            "field-boundary failure category mismatch",
        )
        checks.append("edit/field-boundaries")

        external_deck = work / "external.pptx"
        external_presentation = Presentation()
        external_slide = external_presentation.slides.add_slide(
            external_presentation.slide_layouts[0]
        )
        external_title = external_slide.shapes.title
        if external_title is None:
            raise SmokeFailure("fixture title layout unexpectedly lacks a title")
        external_title.text = ""
        hyperlink_paragraph = external_title.text_frame.paragraphs[0]
        password_link = hyperlink_paragraph.add_run()
        password_link.text = "Password link"
        password_link.hyperlink.address = (
            "https://alice:hunter2@example.invalid/resource?"
            "safe=visible&token=secret-one&API%5FKEY=hidden&token=secret-two#anchor"
        )
        username_link = hyperlink_paragraph.add_run()
        username_link.text = " Username link"
        username_link.hyperlink.address = "https://bob@example.invalid/other?mode=preview"
        external_presentation.save(str(external_deck))
        hyperlink_inspection = _run(["inspect", str(external_deck)])
        inspected_hyperlinks = [
            run["hyperlink"]
            for slide in hyperlink_inspection["presentation"]["slides"]
            for shape in slide["shapes"]
            for paragraph in shape.get("text", {}).get("paragraphs", [])
            for run in paragraph["runs"]
            if run["hyperlink"] is not None
        ]
        serialized_hyperlinks = json.dumps(inspected_hyperlinks)
        _assert(len(inspected_hyperlinks) == 2, "hyperlink inventory mismatch")
        _assert(
            all(
                secret not in serialized_hyperlinks
                for secret in (
                    "alice",
                    "hunter2",
                    "bob@",
                    "secret-one",
                    "secret-two",
                    "hidden",
                )
            ),
            "inspected hyperlinks exposed userinfo or sensitive query values",
        )
        _assert(
            "safe=visible" in serialized_hyperlinks
            and "mode=preview" in serialized_hyperlinks
            and serialized_hyperlinks.count("<redacted>") == 3,
            "hyperlink sanitization did not preserve useful URL structure",
        )
        checks.append("inspect/hyperlink-redaction")
        external_failure = _run(
            ["convert", str(external_deck), "--output", str(work / "external.pdf")],
            expected_status=3,
        )
        _assert(
            external_failure["error"]["category"] == "unsupported_operation",
            "external-relationship conversion failure category mismatch",
        )
        checks.append("failure/convert-external-relationship")

        libreoffice = shutil.which("soffice") or shutil.which("libreoffice")
        if libreoffice is None:
            if require_libreoffice:
                raise SmokeFailure("LibreOffice was required but is not available on PATH")
            conversion = {
                "status": "skipped",
                "reason": "LibreOffice is not available on PATH",
            }
        else:
            pdf = work / "final.pdf"
            conversion_result = _run(
                [
                    "convert",
                    str(final_deck),
                    "--output",
                    str(pdf),
                    "--timeout",
                    "120",
                ],
                timeout_seconds=150,
            )
            reader = PdfReader(str(pdf), strict=True)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            _assert(len(reader.pages) == 3, "converted PDF page count mismatch")
            _assert("Inserted decision" in text, "converted PDF text anchor missing")
            _assert(
                conversion_result["verification"]["pdf_openable"],
                "PDF verification missing",
            )
            checks.append("convert/libreoffice-pdf")
            conversion = {
                "status": "passed",
                "executable": libreoffice,
                "pages": len(reader.pages),
                "text_anchor": "Inserted decision",
            }

    return {
        "schema_version": 1,
        "success": True,
        "operation": "smoke_test",
        "checks": checks,
        "conversion": conversion,
        "fixtures": "generated and removed in a temporary directory",
    }


def main(argv: list[str] | None = None) -> int:
    if DEPENDENCY_ERROR is not None:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "success": False,
                    "operation": "smoke_test",
                    "error": {
                        "category": "missing_dependency",
                        "message": f"missing Python dependency: {DEPENDENCY_ERROR.name}",
                        "details": {
                            "install": "python -m pip install -r requirements.txt",
                        },
                    },
                },
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 4
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-libreoffice",
        action="store_true",
        help="fail unless LibreOffice conversion can be exercised",
    )
    args = parser.parse_args(argv)
    try:
        result = _run_smoke(args.require_libreoffice)
    except (SmokeFailure, OSError, subprocess.SubprocessError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "success": False,
                    "operation": "smoke_test",
                    "error": {
                        "category": "smoke_failure",
                        "message": str(exc),
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
