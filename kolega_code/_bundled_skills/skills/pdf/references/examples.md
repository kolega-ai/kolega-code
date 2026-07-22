# PDF examples

## Contents

- [Create, inspect, edit, and verify](#create-inspect-edit-and-verify)
- [Render and visual review](#render-and-visual-review)
- [Page merge and split](#page-merge-and-split)
- [Redaction](#redaction)
- [Conversions](#conversions)
- [OCR plan](#ocr-plan)
- [Surya preflight and OCR](#surya-preflight-and-ocr)
- [PaddleOCR preflight and OCR](#paddleocr-preflight-and-ocr)
- [Tesseract preflight and OCR](#tesseract-preflight-and-ocr)
- [Searchable OCR composition](#searchable-ocr-composition)
- [Expected normalized output](#expected-normalized-output)
- [Failure examples](#failure-examples)

Follow the runtime guidance in
[Operations](references/operations.md#environment-and-json-contract) first. It defines
`SKILL_ROOT`, `PDF_PYTHON`, and the required user notice before installation:

```bash
SKILL_ROOT="/absolute/path/to/skills/pdf"
```

## Create, inspect, edit, and verify

Create `invoice-create.json`:

```json
{
  "schema_version": 1,
  "page_size": "letter",
  "metadata": {"title": "Invoice 1042", "author": "Example Company"},
  "header": {"text": "Invoice 1042 — page {page}"},
  "pages": [
    {
      "elements": [
        {
          "type": "text",
          "text": "INVOICE 1042",
          "x": 72,
          "y": 80,
          "font": "Helvetica-Bold",
          "font_size": 20
        },
        {
          "type": "table",
          "x": 72,
          "y": 135,
          "width": 420,
          "data": [
            ["Description", "Qty", "Amount"],
            ["Widget", "2", "20.00"],
            ["Total", "", "20.00"]
          ]
        },
        {
          "type": "form",
          "field_type": "text",
          "name": "approved_by",
          "x": 72,
          "y": 300,
          "width": 220
        }
      ]
    }
  ]
}
```

Run:

```bash
"$PDF_PYTHON" "$SKILL_ROOT/scripts/pdf_tool.py" create \
  --job invoice-create.json \
  --output invoice.pdf

"$PDF_PYTHON" "$SKILL_ROOT/scripts/pdf_tool.py" inspect \
  --input invoice.pdf \
  --words \
  --tables \
  --output invoice-inspect.json
```

Representative stdout:

```json
{"counts":{"pages":1},"libraries":{"Pillow":"12.3.0","pdfplumber":"0.11.10","pypdf":"6.14.2","reportlab":"5.0.0"},"operation":"create","output_path":"/work/invoice.pdf","schema_version":1,"status":"ok","tool_version":"1.0.0","verification":{"encrypted":false,"page_count":1,"reopened":true,"signature":"%PDF-","valid":true},"warnings":[]}
```

Create `invoice-edit.json`:

```json
{
  "schema_version": 1,
  "operations": [
    {
      "op": "fill_form",
      "pages": "1",
      "values": {"approved_by": "Ada Example"}
    },
    {
      "op": "stamp_text",
      "pages": "1",
      "text": "APPROVED",
      "x": 306,
      "y": 396,
      "angle": 30,
      "opacity": 0.35
    },
    {
      "op": "set_metadata",
      "metadata": {"Subject": "Approved invoice"}
    }
  ]
}
```

```bash
"$PDF_PYTHON" "$SKILL_ROOT/scripts/pdf_tool.py" edit \
  --input invoice.pdf \
  --job invoice-edit.json \
  --output invoice-approved.pdf

"$PDF_PYTHON" "$SKILL_ROOT/scripts/pdf_tool.py" edit \
  --input invoice-approved.pdf \
  --read-fields

"$PDF_PYTHON" "$SKILL_ROOT/scripts/pdf_tool.py" inspect \
  --input invoice-approved.pdf \
  --pages 1
```

Expected assertions:

- both PDFs begin with `%PDF-`;
- the source remains byte-for-byte untouched;
- output page count is one;
- `approved_by` is `Ada Example`;
- metadata subject is `Approved invoice`;
- `APPROVED` is visible and extractable;
- table cells are checked against a rendered page because extraction is heuristic.

## Render and visual review

Rasterize the published output and look at it:

```bash
"$PDF_PYTHON" "$SKILL_ROOT/scripts/pdf_tool.py" render \
  --input invoice.pdf \
  --output invoice-render \
  --dpi 150
```

Expected summary shape:

```json
{
  "operation": "render",
  "dpi": 150,
  "renders": [
    {"page": 1, "file": "page-0001.png", "width_px": 1275, "height_px": 1650,
     "sha256": "…"}
  ],
  "counts": {"pages_rendered": 1, "total_pixels": 2103750},
  "verification": {"valid": true, "atomic_publish": true, "pngs_reopened": 1,
                   "renderer": "pdfplumber/pypdfium2"}
}
```

Then open `invoice-render/page-0001.png` (and every other changed page) with the Read tool
and examine it. The summary above is structural evidence that PNG files were produced — it
says nothing about how the pages look.

Review the font inventory from the same `inspect` you already ran:

```bash
"$PDF_PYTHON" "$SKILL_ROOT/scripts/pdf_tool.py" inspect --input invoice.pdf \
  --output invoice-inspect.json
```

In the sidecar, `fonts.unembedded_non_base14` must be empty and no warning may begin with
`RELEASE BLOCKER:` before release. A base-14-only document instead carries an informational
substitution warning, which is acceptable when disclosed.

## Page merge and split

Create `pages.json`:

```json
{
  "schema_version": 1,
  "inputs": [
    {"id": "invoice", "path": "invoice-approved.pdf"},
    {"id": "appendix", "path": "appendix.pdf"}
  ],
  "outputs": [
    {
      "path": "packet.pdf",
      "pages": [
        {"input": "invoice", "page": 1},
        {"input": "appendix", "page": 2, "rotate": 90},
        {"input": "appendix", "page": 1, "repeat": 2}
      ]
    },
    {
      "path": "appendix-page-3.pdf",
      "pages": [{"input": "appendix", "page": 3}]
    }
  ]
}
```

```bash
"$PDF_PYTHON" "$SKILL_ROOT/scripts/pdf_tool.py" pages --job pages.json
```

Assert `packet.pdf` has four pages in the listed order, page two is rotated 90 degrees, and
`appendix-page-3.pdf` has one page. Each output is atomic, but the two-output publication is
not a transaction.

## Redaction

Create `redact.json` (full schema and policy in
[Operations](references/operations.md#redact) and [Redaction](references/redaction.md)):

```json
{
  "schema_version": 1,
  "targets": [
    {"type": "text", "text": "ACME-SECRET-9931", "pages": "all",
     "match": "exact", "grow_points": 2.0},
    {"type": "rect", "page": 2, "rect": [72, 640, 468, 60],
     "coordinate_origin": "top-left"}
  ]
}
```

```bash
"$PDF_PYTHON" "$SKILL_ROOT/scripts/pdf_tool.py" redact \
  --input contract.pdf \
  --job redact.json \
  --output contract-redacted.pdf \
  --report contract-redacted.report.json \
  --render-check-dir contract-redacted-qa
```

Expected report excerpt — note the target text appears only as a digest:

```json
{
  "targets": [
    {"index": 0, "type": "text", "text_sha256": "…",
     "resolved_rects": [{"page": 1, "rect": [175.0, 140.9, 255.3, 158.9]}]}
  ],
  "removal_evidence": {
    "pages": [{"page": 1, "text_operators_removed": 1, "characters_removed": 26,
               "image_xobjects_removed": 0, "annotations_removed": 0}],
    "incremental_history_dropped": true
  },
  "residual_scan": {"raw_byte_hits": 0, "decompressed_stream_hits": 0},
  "verification": {"visual_diff": {"checked": true,
                   "max_fraction_changed_outside_expected": 0.0}}
}
```

Open the published `contract-redacted-qa/after-*.png` rasters and confirm the black boxes
sit exactly where intended. `characters_removed` larger than the term length is the
documented whole-operator collateral — text sharing a show operator with the target is
removed with it.

## Conversions

```bash
"$PDF_PYTHON" "$SKILL_ROOT/scripts/pdf_tool.py" convert \
  --input packet.pdf --to txt --pages 1-2 --output packet.txt

"$PDF_PYTHON" "$SKILL_ROOT/scripts/pdf_tool.py" convert \
  --input invoice.pdf --to tables-csv --pages 1 --output invoice-tables

printf 'Generated report\n\nSecond paragraph\n' > report.txt
"$PDF_PYTHON" "$SKILL_ROOT/scripts/pdf_tool.py" convert \
  --input report.txt --to pdf --output report.pdf

"$PDF_PYTHON" "$SKILL_ROOT/scripts/pdf_tool.py" convert \
  --images scan-001.png scan-002.png --to pdf --output scans.pdf
```

Expect deterministic CSV names such as `page-0001-table-0001.csv`, one image per page in
`scans.pdf`, and explicit warnings that these mappings are lossy.

## OCR plan

Planning does not require any OCR profile:

```bash
"$PDF_PYTHON" "$SKILL_ROOT/scripts/pdf_tool.py" ocr-plan \
  --input scans.pdf \
  --pages all \
  --languages eng \
  --hardware cpu \
  --layout simple \
  --volume-hint 40
```

Representative selection:

```json
{
  "recommendation": {
    "engine": "tesseract",
    "paddle_pipeline": null,
    "run_ocr": true,
    "rationale": [
      "The workload is declared simple, clean printed text at batch scale.",
      "Flat text/TSV is sufficient and CPU throughput is prioritized over layout fidelity."
    ]
  },
  "verification": {
    "engine_executed": false,
    "source_unchanged": true
  }
}
```

For a complex table on a compatible accelerator, use `--layout complex --hardware cuda`;
the planner recommends Surya subject to model eligibility. With CPU constraints or
unsuitable Surya terms, it recommends PaddleOCR. It does not infer license acceptance from
an installed executable.

## Surya preflight and OCR

After the operator explicitly selects Surya, follow the OCR installation procedure in
[Operations](references/operations.md#environment-and-json-contract). Announce the missing
dependencies, declared profile, scope, and mechanism, then install only
`requirements-ocr-surya.txt` through the selected `ENGINE_PYTHON`. Do not merge it with the
Paddle profile. Resolve the installed CLI to an absolute `ENGINE_EXECUTABLE`.

Provision the exact approved GGUF revision and a reviewed local `llama-server` executable
outside this command. Record the executable under the manifest `backend` role. Do not run
the engine until the operator has reviewed the current official model terms. Current
weights use a modified AI Pubs Open Rail-M license and are free for research, personal use,
and startups under USD 5 million in funding/revenue; broader commercial use requires the
provider's commercial license.

Provision the external backend through the normal platform package manager as described in
Operations, preferring Homebrew when available on macOS. Installation is not approval.
Before use, review and record the exact package version, upstream source and immutable
revision, license, installed version output, binary size, and SHA-256 in the existing Surya
`runtime` and `backend` manifest entries. The installed binary must pass
manifest/version/checksum preflight.

Use the complete Surya manifest schema from [OCR](references/ocr.md), then:

```bash
"$PDF_PYTHON" "$SKILL_ROOT/scripts/pdf_tool.py" ocr \
  --input complex-scan.pdf \
  --output unused.json \
  --engine surya \
  --engine-executable "$ENGINE_EXECUTABLE" \
  --model-manifest surya-model.json \
  --languages en \
  --preflight-only

"$PDF_PYTHON" "$SKILL_ROOT/scripts/pdf_tool.py" ocr \
  --input complex-scan.pdf \
  --output complex-scan.surya.json \
  --raw-output-dir complex-scan.surya.raw \
  --engine surya \
  --engine-executable "$ENGINE_EXECUTABLE" \
  --model-manifest surya-model.json \
  --languages en \
  --pages 1-4 \
  --device auto \
  --dpi 300 \
  --timeout 1200
```

Preflight must say `license_preflight: passed`, show the exact model revision and checked
artifacts, and show the current model-weight restriction. The OCR sidecar should preserve
ordered labels, coordinates, confidence when supplied, HTML/table payloads, errors, and raw
output path.

## PaddleOCR preflight and OCR

After the operator explicitly selects PaddleOCR, follow the OCR installation procedure in
[Operations](references/operations.md#environment-and-json-contract). Announce the missing
dependencies, declared profile, scope, and mechanism, then install only
`requirements-ocr-paddle.txt` through the selected `ENGINE_PYTHON`.

Paddle remains pip-profile based: do not install PaddleOCR with Homebrew or apt, do not
change the profile constraints, and do not merge this environment with core or Surya. Installing
the Python profile does not provision or approve models. Models must remain reviewed local
artifacts; the OCR command must never download them.

Provision complete local detection and recognition directories and record them as
`text_detection_model_dir` and `text_recognition_model_dir`. For
`--paddle-pipeline structure`, independently review and provision the PPStructureV3 layout
model under `layout_detection_model_dir` and the five table model directories shown below.
Those six structure families are **not independently reviewed by this skill**; their exact
revision, complete directory size/checksum, source, and license are unknown until the
operator records them. The following is an intentionally incomplete inventory template,
not an approved structure manifest:

```json
{
  "schema_version": 1,
  "engine": "paddle",
  "model": {
    "identifier": "PaddlePaddle/PP-OCRv6_medium_det+PP-OCRv6_medium_rec",
    "revision": "det:8e0f56fb2ef86b461d99cfc7ac5c137738985f61;rec:e5a92bcbc5cc1b494628e458d267778f0704fd7c",
    "source": "https://huggingface.co/PaddlePaddle",
    "license": "Apache-2.0",
    "languages": ["en"],
    "license_terms_accepted": true,
    "use_case_eligibility_confirmed": true
  },
  "artifacts": [
    {
      "role": "text_detection_model_dir",
      "path": "/models/paddle/PP-OCRv6_medium_det",
      "identifier": "PaddlePaddle/PP-OCRv6_medium_det",
      "revision": "8e0f56fb2ef86b461d99cfc7ac5c137738985f61",
      "source": "https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det",
      "license": "Apache-2.0",
      "size_bytes": "REPLACE-WITH-CANONICAL-DIRECTORY-BYTE-COUNT",
      "sha256": "REPLACE-WITH-CANONICAL-DIRECTORY-SHA256"
    },
    {
      "role": "text_recognition_model_dir",
      "path": "/models/paddle/PP-OCRv6_medium_rec",
      "identifier": "PaddlePaddle/PP-OCRv6_medium_rec",
      "revision": "e5a92bcbc5cc1b494628e458d267778f0704fd7c",
      "source": "https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec",
      "license": "Apache-2.0",
      "size_bytes": "REPLACE-WITH-CANONICAL-DIRECTORY-BYTE-COUNT",
      "sha256": "REPLACE-WITH-CANONICAL-DIRECTORY-SHA256"
    },
    {
      "role": "layout_detection_model_dir",
      "path": "/models/paddle/PP-DocLayout_plus-L",
      "identifier": "REVIEW-REQUIRED",
      "revision": "REVIEW-REQUIRED",
      "source": "REVIEW-REQUIRED",
      "license": "REVIEW-REQUIRED",
      "size_bytes": "REVIEW-REQUIRED",
      "sha256": "REVIEW-REQUIRED"
    },
    {
      "role": "table_classification_model_dir",
      "path": "/models/paddle/PP-LCNet_x1_0_table_cls",
      "identifier": "REVIEW-REQUIRED",
      "revision": "REVIEW-REQUIRED",
      "source": "REVIEW-REQUIRED",
      "license": "REVIEW-REQUIRED",
      "size_bytes": "REVIEW-REQUIRED",
      "sha256": "REVIEW-REQUIRED"
    },
    {
      "role": "wired_table_structure_recognition_model_dir",
      "path": "/models/paddle/SLANeXt_wired",
      "identifier": "REVIEW-REQUIRED",
      "revision": "REVIEW-REQUIRED",
      "source": "REVIEW-REQUIRED",
      "license": "REVIEW-REQUIRED",
      "size_bytes": "REVIEW-REQUIRED",
      "sha256": "REVIEW-REQUIRED"
    },
    {
      "role": "wireless_table_structure_recognition_model_dir",
      "path": "/models/paddle/SLANet_plus",
      "identifier": "REVIEW-REQUIRED",
      "revision": "REVIEW-REQUIRED",
      "source": "REVIEW-REQUIRED",
      "license": "REVIEW-REQUIRED",
      "size_bytes": "REVIEW-REQUIRED",
      "sha256": "REVIEW-REQUIRED"
    },
    {
      "role": "wired_table_cells_detection_model_dir",
      "path": "/models/paddle/RT-DETR-L_wired_table_cell_det",
      "identifier": "REVIEW-REQUIRED",
      "revision": "REVIEW-REQUIRED",
      "source": "REVIEW-REQUIRED",
      "license": "REVIEW-REQUIRED",
      "size_bytes": "REVIEW-REQUIRED",
      "sha256": "REVIEW-REQUIRED"
    },
    {
      "role": "wireless_table_cells_detection_model_dir",
      "path": "/models/paddle/RT-DETR-L_wireless_table_cell_det",
      "identifier": "REVIEW-REQUIRED",
      "revision": "REVIEW-REQUIRED",
      "source": "REVIEW-REQUIRED",
      "license": "REVIEW-REQUIRED",
      "size_bytes": "REVIEW-REQUIRED",
      "sha256": "REVIEW-REQUIRED"
    }
  ]
}
```

```bash
"$PDF_PYTHON" "$SKILL_ROOT/scripts/pdf_tool.py" ocr \
  --input cpu-scan.pdf \
  --output unused.json \
  --engine paddle \
  --engine-executable "$ENGINE_PYTHON" \
  --model-manifest paddle-model.json \
  --languages en \
  --preflight-only

"$PDF_PYTHON" "$SKILL_ROOT/scripts/pdf_tool.py" ocr \
  --input cpu-scan.pdf \
  --output cpu-scan.paddle.json \
  --raw-output-dir cpu-scan.paddle.raw \
  --engine paddle \
  --engine-executable "$ENGINE_PYTHON" \
  --model-manifest paddle-model.json \
  --languages en \
  --pages 1-10 \
  --paddle-pipeline structure \
  --device cpu \
  --timeout 1200
```

The subprocess uses the official Python result API because the official CLI does not
document strict machine-readable JSON output. Use the default `--paddle-pipeline ocr` for
plain recognition. Structure mode invokes `PPStructureV3`, retains layout/table parsing and
Markdown, and explicitly disables unprovisioned formula, chart, seal, and region models.
Local required model directories and disabled optional modules prevent the adapter from
requesting default models. Preflight rejects every `REVIEW-REQUIRED`/`REPLACE` placeholder:
replace each with independently reviewed artifact facts and canonical directory inventory
values described in [OCR](references/ocr.md).

## Tesseract preflight and OCR

After the operator explicitly selects Tesseract, follow the external-runtime installation
procedure in [Operations](references/operations.md#environment-and-json-contract). Use the
normal platform package manager, preferring Homebrew when available on macOS, and install
only the CLI and language packages required for the declared languages. Do not install a
bulk language package unless that scope is intentional. Package installation alone is not
manifest approval. Resolve the executable without assuming an install path, and record its
actual version:

```bash
ENGINE_EXECUTABLE="$(command -v tesseract)"
test -n "$ENGINE_EXECUTABLE"
"$ENGINE_EXECUTABLE" --version
```

The fastest correct path is manifest derivation — the tool measures paths, sizes, and
hashes; you declare the origin and approvals explicitly after reviewing them:

```bash
"$PDF_PYTHON" "$SKILL_ROOT/scripts/pdf_tool.py" manifest \
  --mode derive \
  --engine tesseract \
  --languages eng \
  --assume-source tessdata_fast \
  --accept-license \
  --confirm-eligibility \
  --output tesseract-model.json

"$PDF_PYTHON" "$SKILL_ROOT/scripts/pdf_tool.py" manifest \
  --mode check \
  --engine tesseract \
  --languages eng \
  --model-manifest tesseract-model.json
```

`--assume-source` records that you assert the local files came from the pinned official
repository revision in [OCR](references/ocr.md); omit it to get `REVIEW-REQUIRED`
placeholders that preflight rejects until you fill them. The check mode reports every
problem at once with expected and actual hashes. The equivalent hand-written manifest —
still the authoritative schema reference — is:

```json
{
  "schema_version": 1,
  "engine": "tesseract",
  "model": {
    "identifier": "tesseract-ocr/tessdata_fast:eng",
    "revision": "87416418657359cb625c412a48b6e1d6d41c29bd",
    "source": "https://github.com/tesseract-ocr/tessdata_fast",
    "license": "Apache-2.0",
    "languages": ["eng"],
    "license_terms_accepted": true,
    "use_case_eligibility_confirmed": true
  },
  "artifacts": [
    {
      "role": "language_data:eng",
      "path": "/opt/tessdata/eng.traineddata",
      "size_bytes": 0,
      "sha256": "replace-with-the-actual-file-sha256"
    }
  ]
}
```

Replace the placeholder size and checksum; a zero size will fail unless the file is actually
empty.

```bash
"$PDF_PYTHON" "$SKILL_ROOT/scripts/pdf_tool.py" ocr \
  --input clean-batch.pdf \
  --output unused.json \
  --engine tesseract \
  --engine-executable "$ENGINE_EXECUTABLE" \
  --model-manifest tesseract-model.json \
  --languages eng \
  --preflight-only

"$PDF_PYTHON" "$SKILL_ROOT/scripts/pdf_tool.py" ocr \
  --input clean-batch.pdf \
  --output clean-batch.tesseract.json \
  --raw-output-dir clean-batch.tesseract.raw \
  --engine tesseract \
  --engine-executable "$ENGINE_EXECUTABLE" \
  --model-manifest tesseract-model.json \
  --languages eng \
  --pages all \
  --device cpu \
  --dpi 300
```

Expect line blocks derived from TSV, pixel coordinates, and confidence only where Tesseract
supplied nonnegative word values. The Tesseract adapter also records word-level geometry
under `blocks[].words`, which `ocr-compose` uses for word-accurate placement. The
normalized engine metadata reports `cpu` and `tesseract-cpu` even if another device label
was requested. The one `--timeout` budget is shared by every selected page. Do not expect
headings, reading order, table HTML, math, or layout semantics.

## Searchable OCR composition

After any successful `ocr` run, compose the sidecar into a searchable PDF — no engine, no
manifest, fully re-runnable:

```bash
"$PDF_PYTHON" "$SKILL_ROOT/scripts/pdf_tool.py" ocr-compose \
  --input scanned.pdf \
  --sidecar scanned.tesseract.json \
  --output scanned-searchable.pdf
```

Expected verification shape:

```json
{
  "operation": "ocr-compose",
  "counts": {"pages_composed": 4, "pages_skipped": 0, "words_drawn": 812,
             "characters_dropped": 0},
  "font": {"name": "Helvetica", "embedded": false},
  "verification": {
    "page_count_unchanged": true,
    "anchors": {"checked": 20, "found": 20},
    "geometry_spot_check_max_delta_points": 0.4,
    "visual_diff": {"checked": true, "max_fraction_changed": 0.0002,
                    "threshold": 0.002}
  }
}
```

The run fails rather than publishing when composed text does not extract, when its
geometry deviates, or when the supposedly invisible layer visibly changes a page. The
sidecar's recorded source hash must match `--input`; re-run OCR if the scan changed.
Non-Latin sidecars need `--font /path/to/covering.ttf`. Confirm selectability the same way
a user would: `extract` the output and check the recognized anchors appear.

## Expected normalized output

Every engine output must assert:

- `schema_version` equals `1`;
- engine name/version and model identifier/revision match preflight;
- strictly parsed requested languages and the requested device are retained for audit;
- resolved device/backend metadata follows the adapter semantics documented in
  [OCR](references/ocr.md);
- source path, source SHA-256, and one-based page numbers are correct;
- every non-null polygon/bounding box is in the declared rendered-pixel coordinate space;
- confidence is engine-supplied or `null`;
- unsupported fields remain absent/null or under `engine_specific`;
- `raw_output_path` exists when requested;
- warnings identify skipped pages, low confidence, ambiguity, or engine errors;
- the source PDF is unchanged.

See the representative sidecar in [OCR](references/ocr.md).

## Failure examples

Missing input:

```bash
"$PDF_PYTHON" "$SKILL_ROOT/scripts/pdf_tool.py" inspect --input missing.pdf
```

```json
{"category":"bad_input","details":null,"message":"Input does not exist: missing.pdf","schema_version":1,"status":"error"}
```

Corrupt or unsupported signature:

```bash
printf 'not a pdf\n' > corrupt.pdf
"$PDF_PYTHON" "$SKILL_ROOT/scripts/pdf_tool.py" inspect --input corrupt.pdf
```

The command exits `2` with category `bad_input` and does not create an output.

Wrong destination extensions and malformed numerics are also `bad_input`:

```bash
"$PDF_PYTHON" "$SKILL_ROOT/scripts/pdf_tool.py" edit \
  --input source.pdf --job edit.json --output edited.json
"$PDF_PYTHON" "$SKILL_ROOT/scripts/pdf_tool.py" convert \
  --images scan.png --to pdf --output scan.pdf --margin nan
```

The first command is rejected because PDF-producing edits require `.pdf`; the second is
rejected because geometry and margins must be finite. Normalized OCR output similarly
requires `.json`, and `ocr-plan --languages ' , '` is rejected before planning.

Render refuses unsafe destinations and oversized rasters:

```bash
"$PDF_PYTHON" "$SKILL_ROOT/scripts/pdf_tool.py" render \
  --input invoice.pdf --output occupied-directory --overwrite   # bad_input (2)
"$PDF_PYTHON" "$SKILL_ROOT/scripts/pdf_tool.py" render \
  --input invoice.pdf --output out --dpi 1200                   # bad_input (2)
"$PDF_PYTHON" "$SKILL_ROOT/scripts/pdf_tool.py" render \
  --input poster-5000pt.pdf --output out --dpi 600              # resource_limit (6)
```

A redaction text target that matches nothing exits `2` (`bad_input`) without writing
anything — silent no-op redaction does not exist. A redaction target intersecting a Form
XObject exits `3` (`unsupported_operation`) naming the page and XObject; flatten or
rasterize first. Redacting an encrypted input exits `3`; decrypt explicitly with `edit`
first. `ocr-compose` with a sidecar whose recorded source hash mismatches `--input` exits
`2`. `manifest --mode check` against a tampered artifact exits `7` with every failed check,
including expected and actual hashes, in `details`.

Encrypted input without an approved password exits `3` with `unsupported_operation`.
An OCR request without a model manifest is rejected by argument parsing. A manifest without
operator approval — or with `REVIEW-REQUIRED`/`replace-with` placeholder provenance —
exits `7` with `license_precondition`. An untested engine major exits `3`;
the tool never downgrades, installs, downloads, or switches engines within an invocation.
If Surya and PaddleOCR are unavailable, tell the user before starting a separate explicit
Tesseract fallback invocation.
