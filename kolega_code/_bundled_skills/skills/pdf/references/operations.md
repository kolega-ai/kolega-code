# PDF tool operations

## Contents

- [Environment and JSON contract](#environment-and-json-contract)
- [Input safety and limits](#input-safety-and-limits)
- [Inspect and extract](#inspect-and-extract)
- [Font inventory](#font-inventory)
- [Render](#render)
- [Create](#create)
- [Pages](#pages)
- [Edit](#edit)
- [Redact](#redact)
- [Convert](#convert)
- [OCR planning and execution](#ocr-planning-and-execution)
- [Manifest derive and check](#manifest-derive-and-check)
- [Searchable OCR composition](#searchable-ocr-composition)
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
| Rasterized render per page/region (`render`, `ocr`, verification rasters) | 40,000,000 pixels |
| Rasterized pixels per invocation (`render`, `ocr`, verification rasters) | 200,000,000 pixels |
| Render/OCR DPI (`render` default 150, `ocr` default 300, verification 96) | 72–600 |
| OCR timeout | 1–86,400 seconds |
| Font inventory entries | 2,000 |
| Font resource-walk visits / XObject depth | 10,000 / 8 |
| Redaction targets per job | 1,000 |
| Content operations per redacted page | 500,000 |

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

## Font inventory

Every `inspect` and `extract` result carries a whole-document `fonts` block regardless of
`--pages`, built from page, Form-XObject, AcroForm, and annotation appearance resources:

```json
"fonts": {
  "scope": "document",
  "count": 2,
  "entries": [
    {
      "base_font": "ABCDEF+Lato-Regular",
      "name": "Lato-Regular",
      "subtype": "TrueType",
      "descendant_subtype": null,
      "embedded": true,
      "subset": true,
      "encoding": "WinAnsiEncoding",
      "base14": false,
      "pages": [1, 2]
    }
  ],
  "unembedded": ["Helvetica"],
  "unembedded_non_base14": [],
  "truncated": false
}
```

`embedded` reflects a `FontFile`/`FontFile2`/`FontFile3` in the descriptor; Type3 fonts are
always embedded because their glyph procedures live in the document. `base14` covers the
standard fourteen Type1 fonts plus the universally substituted Arial, Times New Roman, and
Courier New alias families. The release gate:

- Unembedded fonts **outside** the base-14 set add a warning that begins
  `RELEASE BLOCKER: non-embedded fonts outside the standard base-14 set: …`. Resolve it
  before releasing the document — other viewers substitute different glyphs and metrics,
  which can change line breaks, spacing, and symbol coverage.
- Unembedded base-14 fonts add only an informational substitution warning.
- A `truncated: true` inventory hit the bounded resource walk; treat it as incomplete, not
  as clean.

## Render

```text
pdf_tool.py render --input INPUT.pdf --output DIRECTORY
                   [--pages EXPR] [--dpi 150]
                   [--password-env NAME] [--overwrite]
```

Rasterizes each selected page to `page-NNNN.png` (1-based source page numbers) in a new or
empty directory with the same non-clobbering, staged, atomic semantics as
`extract --images-dir`. The full pixel budget is charged from page geometry before any
rasterization, every PNG is signature-checked, reopened, and hashed, and the published
files are reopened once more after the atomic publish. There is no `--timeout`: rendering is
in-process PDFium (via pdfplumber), not a subprocess.

Rendering exists to be looked at. Open the produced PNG files and examine them; a
successful render summary is structural evidence only, never visual proof. The rasters are
produced from the delivered PDF itself by PDFium, so for static page appearance they are
authoritative — with two carve-outs: non-embedded fonts render with this machine's
substitutes (other viewers may differ; see the font inventory gate), and viewer-interactive
behavior (form appearance regeneration, annotations, JavaScript, overprint) is not proven
by any raster. Encrypted sources render after authentication with a warning that the PNG
files themselves are not password protected.

At 150 DPI a letter page is about 2.1 million pixels, so roughly 95 letter pages fit one
invocation's 200,000,000-pixel budget; select pages or lower `--dpi` for longer documents.

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

## Redact

```text
pdf_tool.py redact --input INPUT.pdf --job REDACT.json --output OUTPUT.pdf
                   [--report REPORT.json] [--render-check-dir DIRECTORY]
                   [--overwrite]
```

`redact` removes content — text-showing operators, images, vector paints, and annotations
whose measured extents intersect the target regions — then rewrites the document as a
single revision and draws an opaque confirmation rectangle over each target. It is the only
supported redaction path; `edit` stamps and white rectangles are cosmetic overlays, never
redaction. Read [Redaction](references/redaction.md) for the full leakage-policy table, the
verification contract, and everything v1 refuses.

Job schema (closed; unknown keys fail):

```json
{
  "schema_version": 1,
  "targets": [
    {
      "type": "rect",
      "page": 3,
      "rect": [72, 144, 200, 40],
      "coordinate_origin": "top-left"
    },
    {
      "type": "text",
      "text": "ACME-SECRET-9931",
      "pages": "all",
      "match": "exact",
      "grow_points": 2.0
    }
  ],
  "image_policy": "remove",
  "annotation_policy": "remove-intersecting",
  "fill_color": "#000000",
  "strip_docinfo_keys": "matched",
  "strip_xmp": "if-matched",
  "strip_thumbnails": true,
  "strip_structure_tree": false,
  "acknowledge": {
    "signatures_invalidated": false,
    "attachments_present": false
  }
}
```

- `rect` targets use `[x, y, width, height]` in displayed-page points; `coordinate_origin`
  is `top-left` (default, matching inspect/extract word coordinates) or `bottom-left`.
- `text` targets resolve to rectangles through extracted word geometry on their selected
  pages, grown by `grow_points` (default 2). A target that matches nothing fails as
  `bad_input` — nothing is silently skipped. Text targets match only extractable digital
  text; redact scanned pixels with `rect` targets.
- Whole intersecting operators are removed and their advance is preserved so retained text
  does not shift; adjacent text inside the same operator is removed with it, counted, and
  visible in the verification rasters.
- `image_policy: remove` (default) drops any intersecting image entirely; `refuse` fails
  instead. There is no partial image masking in v1.
- Verification before publication: zero extractable characters inside every target region,
  target terms unextractable on their selected pages, a byte-level residual scan over the
  raw file and every decompressed stream, removed image streams unreachable, page geometry
  unchanged, exactly one `%%EOF`, and a before/after raster diff that must be clean outside
  the expected change regions. Any failure aborts with `validation_failed` and publishes
  nothing.
- `--report` writes the full JSON report; it carries SHA-256 digests of text targets, never
  the target text. `--render-check-dir` publishes the before/after verification rasters for
  review.

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
together if publication raises an error. An `ocr` run writes only sidecars and never
changes the source; producing a searchable PDF is the separate, explicit
[`ocr-compose`](#searchable-ocr-composition) step.

## Manifest derive and check

```text
pdf_tool.py manifest --mode derive --engine tesseract --languages eng,deu
                     [--tessdata-dir DIR] [--assume-source tessdata_fast|tessdata_best]
                     [--accept-license] [--confirm-eligibility]
                     --output MANIFEST.json [--overwrite]

pdf_tool.py manifest --mode derive --engine surya
                     --artifact model=/models/surya/surya-2.gguf
                     --artifact mmproj=/models/surya/surya-2-mmproj.gguf
                     --artifact backend=/opt/llama.cpp/llama-server
                     --output MANIFEST.json

pdf_tool.py manifest --mode derive --engine paddle
                     --artifact text_detection_model_dir=/models/paddle/det
                     --artifact text_recognition_model_dir=/models/paddle/rec
                     --output MANIFEST.json

pdf_tool.py manifest --mode check --engine ENGINE --languages LANGS
                     --model-manifest MANIFEST.json
```

`manifest` never executes any binary — it stats and hashes local files only. The derive
principle: **the tool measures; the operator declares; approval is never inferred.**
Measured facts (paths, byte sizes, SHA-256, canonical directory inventories) are filled
automatically. Provenance (identifier, revision, source, license) is filled only from an
explicit operator declaration (`--assume-source` for the pinned tessdata repositories) or
an exact SHA-256 match against the artifacts already reviewed in [OCR](references/ocr.md); everything
else is emitted as `REVIEW-REQUIRED`, which `ocr` preflight rejects. The six PP-StructureV3
structure roles are never auto-filled by any flag. `--accept-license` and
`--confirm-eligibility` are explicit operator acts; without them the emitted approvals are
`false` and the stdout summary lists every reason the manifest is not preflight-ready.

`--mode check` verifies an existing manifest without fail-fast: it reports every problem at
once with expected and actual sizes/hashes, then exits 0 when preflight would pass or 7
with the full check array in `details`. It excludes (and names) the checks only `ocr` can
perform: the engine version probe, the Tesseract language query, and the backend probe.

## Searchable OCR composition

```text
pdf_tool.py ocr-compose --input SCANNED.pdf --sidecar RESULT.json --output OUT.pdf
                        [--pages EXPR] [--min-confidence 0.0]
                        [--font FONT.ttf] [--on-unmappable fail|skip-block]
                        [--allow-duplicate-text]
                        [--max-visual-diff 0.002] [--visual-check-dpi 96]
                        [--skip-visual-check]
                        [--password-env NAME] [--allow-decrypted-output] [--overwrite]
```

`ocr-compose` overlays the recognized text from a completed OCR sidecar as an invisible
text layer (text render mode 3), producing a searchable, selectable PDF. It is
deterministic: it never runs an engine, needs no model manifest, and can be re-run with
different options against the same sidecar. The sidecar's recorded `source.sha256` must
match `--input` exactly; there is no override.

- Word geometry (`blocks[].words`, recorded by the Tesseract adapter) is used when present;
  otherwise whole blocks are placed. Blocks below `--min-confidence` or without geometry
  are skipped and counted.
- The default Helvetica layer maps cp1252 text only. Unmappable characters fail the run
  (default) listing the affected pages, or are dropped with counts under
  `--on-unmappable skip-block`; characters are never silently substituted. Pass `--font`
  with a covering TTF/OTF for other scripts — it is embedded and subset, and the operator
  owns its embedding license.
- Pages classified `digital-text` (or `hybrid` composed without a region) are skipped by
  default because composing them duplicates extractable text; `--allow-duplicate-text`
  overrides. Hybrid region results overlay only their region. Pages absent from the
  sidecar pass through unchanged.
- Encrypted sources require the password and `--allow-decrypted-output`, mirroring `pages`.
- Verification before publication: page count, dimensions, and rotation unchanged; the
  highest-confidence composed texts must extract from the output; one anchor per page must
  extract within 3 points of its composed position; and a before/after raster diff must
  stay under `--max-visual-diff` — an invisible layer must be invisible. The composed text
  inherits OCR recognition errors; the output records confidence statistics per page.

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
