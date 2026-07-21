# PDF tool operations

## Contents

- [Environment and JSON contract](#environment-and-json-contract)
- [Input safety and limits](#input-safety-and-limits)
- [Inspect and extract](#inspect-and-extract)
- [Create](#create)
- [Pages](#pages)
- [Edit](#edit)
- [Convert](#convert)
- [OCR planning and execution](#ocr-planning-and-execution)
- [Exit statuses](#exit-statuses)

## Environment and JSON contract

Resolve the skill root and select any available Python 3.11+ interpreter. Do not assume a
launcher name. The examples use `PDF_PYTHON` for the selected interpreter:

```bash
SKILL_ROOT="/absolute/path/to/skills/pdf"
PDF_PYTHON="/path/to/selected/python"
"$PDF_PYTHON" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'
```

Check the required imports. Before installing anything, tell the user what is missing, what
will be installed, the target scope, and the installer. Use the selected interpreter's
`-m pip` with the core requirements file:

```bash
"$PDF_PYTHON" -m pip install -r "$SKILL_ROOT/requirements.txt"
```

Prefer the active or user interpreter. If platform policy blocks those scopes, explain the
fallback before using a local environment. Use the normal platform package manager for a
missing interpreter or external runtime; prefer Homebrew on macOS when available. Never use
privileged pip or `--break-system-packages`, and do not edit host-project dependency
metadata. Verify the CLI with the same selected interpreter:

```bash
"$PDF_PYTHON" -m pip check
"$PDF_PYTHON" "$SKILL_ROOT/scripts/pdf_tool.py" --help
```

For OCR, select exactly one engine and tell the user about its dependencies before installing
them. Install only that engine's declared profile:

```bash
"$ENGINE_PYTHON" -m pip install -r "$ENGINE_REQUIREMENTS"
```

Do not combine profiles. If Surya or PaddleOCR is unavailable, tell the user before selecting
or installing Tesseract as the fallback, then use a new explicit Tesseract invocation.
Python-profile installation never provisions models. Models and language data remain
reviewed local artifacts; manifest, checksum, and license checks remain mandatory.

All structured jobs and results use `schema_version: 1`. Successful commands emit exactly
one JSON object on stdout. Failures emit a JSON object on stderr. The CLI does not prompt.
Passwords may be supplied directly, but `--password-env NAME` avoids putting one in shell
history. Password values are redacted from owned diagnostics.

Every mutation requires a destination distinct from every source. Existing destinations are
rejected unless `--overwrite` is present. The tool writes a temporary sibling, validates its
signature and reopen state, and uses an atomic filesystem replacement. PDF-producing
`create`, `pages`, `edit`, and `convert` destinations must end in `.pdf`; normalized OCR and
other JSON sidecars must end in `.json`.

## Input safety and limits

The initial release enforces these hard limits:

| Resource | Limit |
| --- | ---: |
| Source PDF | 512 MiB |
| Pages read or written per PDF | 500 |
| Extracted characters | 20,000,000 |
| Embedded images | 2,000 |
| Extracted tables | 2,000 |
| JSON job | 20 MiB |
| Story items | 10,000 |
| Split outputs | 500 |
| OCR render per page/region | 40,000,000 pixels |
| OCR renders per invocation | 200,000,000 pixels |
| OCR DPI | 72–600 |
| OCR timeout | 1–86,400 seconds |

The parser requires a PDF signature in the first 1,024 bytes and opens PDFs in strict mode.
It rejects missing passwords and page references outside the document. Page expressions are
one-based: `all`, `1`, `1,3,5`, or `2-6`. A descending range such as `6-2` is allowed where
order is meaningful. JSON and CLI dimensions, margins, colors, sizes, opacity, scale, and
coordinates must be finite and in their documented ranges. Negative geometry, zero
dimensions, non-finite values, and rectangles or margins that cannot fit the selected page
fail with `bad_input`.

## Inspect and extract

```text
pdf_tool.py inspect --input INPUT [--pages EXPR] [--mode plain|layout]
                    [--words] [--tables] [--output RESULT.json]
                    [--password-env NAME] [--overwrite]

pdf_tool.py extract --input INPUT [inspect options]
                    [--images-dir EMPTY_DIRECTORY]
```

Both commands return metadata, encryption state, page geometry and rotation, AcroForm
inventory, XFA detection, text, embedded-image inventory, and one of these page
classifications:

- `blank`: no extractable text and no detected images;
- `digital-text`: usable text without a dominant image;
- `likely-scanned`: image content without usable digital text;
- `hybrid`: digital text and image content coexist.

`--mode layout` asks pdfplumber to preserve approximate spacing. `--words` adds word-level
coordinates. `--tables` adds heuristic table arrays. `extract --images-dir` writes
deterministically named embedded streams and reports per-image failures without discarding
successful extractions. An `--output` JSON sidecar receives the full result; stdout then
contains a bounded summary.

## Create

```text
pdf_tool.py create --job CREATE.json --output OUTPUT.pdf [--overwrite]
```

Use exactly one of `pages` or `story`.

### Positioned layout

Coordinates are PDF points measured from the top-left for layout elements. One point is
1/72 inch. Known page names are `a3`, `a4`, `a5`, `letter`, and `legal`; `[width, height]`
is also accepted.

```json
{
  "schema_version": 1,
  "page_size": "letter",
  "metadata": {
    "title": "Invoice",
    "author": "Example"
  },
  "header": {
    "text": "Invoice — page {page}",
    "x": 54,
    "y": 24,
    "width": 504
  },
  "footer": {
    "text": "Confidential",
    "x": 54,
    "y": 756,
    "width": 504
  },
  "pages": [
    {
      "elements": [
        {
          "type": "text",
          "text": "Invoice 1042",
          "x": 72,
          "y": 72,
          "width": 300,
          "font": "Helvetica-Bold",
          "font_size": 20
        },
        {
          "type": "table",
          "x": 72,
          "y": 135,
          "width": 420,
          "data": [
            ["Item", "Quantity", "Amount"],
            ["Widget", "2", "20.00"]
          ]
        },
        {
          "type": "image",
          "path": "/absolute/path/logo.png",
          "x": 420,
          "y": 55,
          "width": 100,
          "height": 50
        },
        {
          "type": "form",
          "field_type": "text",
          "name": "approved_by",
          "x": 72,
          "y": 320,
          "width": 200,
          "height": 22
        }
      ]
    }
  ]
}
```

Supported layout elements are `text`, `paragraph`, `table`, `image`, `form` (`text` and
`checkbox`), and `line`. Built-in ReportLab fonts avoid hidden font dependencies.

### Flowing story

`story` supports `heading`, `paragraph`, `spacer`, `page_break`, `table`, and `image`.
Paragraph text is ReportLab paragraph markup; do not pass untrusted markup without escaping
it. Story tables repeat one header row by default. Header/footer text supports `{page}`.

## Pages

```text
pdf_tool.py pages --job PAGES.json [--overwrite]
```

One page-reference schema implements merge, split, select, reorder, repeat, delete-by-
omission, and rotate:

```json
{
  "schema_version": 1,
  "inputs": [
    {"id": "front", "path": "front.pdf"},
    {
      "id": "body",
      "path": "body.pdf",
      "password": "supplied-secret",
      "allow_decrypted_output": true
    }
  ],
  "outputs": [
    {
      "path": "combined.pdf",
      "metadata": {"Title": "Combined"},
      "pages": [
        {"input": "front", "page": 1},
        {"input": "body", "page": 3, "rotate": 90},
        {"input": "body", "page": 1, "repeat": 2}
      ]
    },
    {
      "path": "split-part.pdf",
      "pages": [
        {"input": "body", "page": 2}
      ]
    }
  ]
}
```

Omit unwanted pages. List references in the desired order. Add another input to merge.
Create multiple outputs to split. Each output publication is atomic; publication of a
multi-output set is not transactional. Because page assembly does not preserve encryption,
an encrypted input must set `allow_decrypted_output: true`; authentication alone is not
authorization to publish decrypted pages.

## Edit

Read fields without mutation:

```text
pdf_tool.py edit --input INPUT.pdf --read-fields [--password-env NAME]
```

Apply a versioned mutation job:

```text
pdf_tool.py edit --input INPUT.pdf --job EDIT.json --output OUTPUT.pdf
                 [--password-env NAME] [--overwrite]
```

Supported operations:

- `stamp_text`: `text`, optional `pages`, `x`, `y`, `angle`, `font`, `font_size`, `color`,
  `opacity`, and `over`;
- `stamp_image`: `path`, optional `pages`, `x`, `y`, `width`, `height`, and `over`;
- `stamp_pdf`: `path`, optional password, `stamp_page`, target `pages`, `x`, `y`, `scale`,
  and `over`;
- `fill_form`: exact `values` object and optional target `pages`; unknown fields fail as an
  ambiguous edit;
- `set_metadata`: a metadata object;
- `remove_metadata`: `keys` array or `"all"`;
- `encrypt`: an explicit `algorithm` of `AES-128`, `AES-256-R5`, or `AES-256`, plus
  `user_password` and optional `owner_password`;
- `decrypt`: write the already authenticated source without encryption.

Example:

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
      "pages": "all",
      "text": "DRAFT",
      "x": 306,
      "y": 396,
      "angle": 35,
      "opacity": 0.25
    },
    {
      "op": "set_metadata",
      "metadata": {"Title": "Reviewed invoice"}
    }
  ]
}
```

Encryption passwords in a JSON job are sensitive. Generate that job in a protected temporary
location and delete it after use. The CLI never reports password values.

## Convert

```text
pdf_tool.py convert --input INPUT.pdf --to txt|json --output FILE
                    [--pages EXPR] [--mode plain|layout]
pdf_tool.py convert --input INPUT.pdf --to tables-csv --output EMPTY_DIRECTORY
                    [--pages EXPR]
pdf_tool.py convert --input INPUT.txt --to pdf --output OUTPUT.pdf
                    [--encoding utf-8] [--page-size letter]
pdf_tool.py convert --images A.png B.jpg --to pdf --output OUTPUT.pdf
                    [--page-size letter] [--margin 36]
```

PDF→text/JSON preserves extracted content, not appearance. PDF tables produce one file named
`page-NNNN-table-NNNN.csv` per detected table. Text→PDF uses a basic flowing document.
Images→PDF fits one raster image per page. These mappings do not emulate a browser or office
renderer.

## OCR planning and execution

`ocr-plan` never starts an engine:

```text
pdf_tool.py ocr-plan --input INPUT.pdf --pages EXPR --languages eng
                     [--hardware auto|cpu|cuda|mps]
                     [--layout auto|simple|complex]
                     [--volume-hint COUNT] [--dpi 300]
```

`ocr` rejects `auto` as an engine:

```text
pdf_tool.py ocr --input INPUT.pdf --output RESULT.json
                --engine surya|paddle|tesseract
                --model-manifest MANIFEST.json --languages eng
                [--pages EXPR] [--dpi 300] [--timeout 600]
                [--engine-executable PATH] [--device auto|cpu|cuda|mps]
                [--paddle-pipeline ocr|structure]
                [--preflight-only] [--raw-output-dir DIRECTORY]
```

Read [OCR](references/ocr.md) for the decision policy, manifest schema, runtime isolation,
and normalized result. Language lists are trimmed, must contain nonempty unique identifiers,
and are validated by both planning and execution. Result metadata retains the requested
device separately from the resolved compute device and runtime backend; `auto` is never
reported as the resolved device. `--paddle-pipeline structure` explicitly selects local
`PPStructureV3` layout/table parsing and requires the complete local layout, table
classification, wired/wireless table-structure, and wired/wireless cell-detection artifact
roles listed in [OCR](references/ocr.md).
`--raw-output-dir` and the normalized JSON are staged before publication and rolled back
together if publication raises an error. A normal OCR run writes only sidecars; it never
changes the source or creates a searchable replacement PDF.

## Exit statuses

| Status | Category | Meaning |
| ---: | --- | --- |
| 2 | `bad_input` | Missing, malformed, mismatched, or unsafe input |
| 3 | `unsupported_operation` | Deliberately unsupported content or operation |
| 4 | `missing_dependency` | Required runtime, executable, or language data absent |
| 5 | `ambiguous_edit` | Mutation target is not exact or operations conflict |
| 6 | `resource_limit` | A configured size, count, render, or time bound was exceeded |
| 7 | `license_precondition` | Artifact provenance, license, eligibility, or approval failed |
| 8 | `validation_failed` | Output could not be reopened or atomically published |
| 9 | `engine_failed` | Selected OCR subprocess timed out or failed |
| 10 | `internal_error` | Unexpected implementation failure |

Error envelope:

```json
{
  "schema_version": 1,
  "status": "error",
  "category": "bad_input",
  "message": "Input does not exist: missing.pdf",
  "details": null
}
```
