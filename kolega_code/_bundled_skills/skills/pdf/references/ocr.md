# Local OCR policy and adapter contract

## Contents

- [Non-negotiable rules](#non-negotiable-rules)
- [Decision matrix](#decision-matrix)
- [Engine boundaries](#engine-boundaries)
- [Artifact and license preflight](#artifact-and-license-preflight)
- [Model manifest schema](#model-manifest-schema)
- [Manifest derivation and checking](#manifest-derivation-and-checking)
- [Rendering and execution](#rendering-and-execution)
- [Normalized sidecar](#normalized-sidecar)
- [Engine normalization](#engine-normalization)
- [Optional profile smoke tests](#optional-profile-smoke-tests)

## Non-negotiable rules

1. Extract and classify first. Do not OCR usable digital text.
2. Require the user to select `surya`, `paddle`, or `tesseract`. There is no `auto` engine.
3. Require explicit languages. Report unsupported or uncertain languages instead of retrying
   indefinitely.
4. Provision every explicitly selected runtime, model, and language-data file outside the
   OCR command. The command neither installs nor downloads, and provisioning never changes
   the selected engine.
5. Record the exact artifact source, revision, license, size, and checksum required by the
   manifest preflight. Require the operator to accept the terms and confirm the use case is
   eligible.
6. Never switch engines silently or inside a failed invocation. If Surya and PaddleOCR are
   unavailable, tell the user and explicitly select Tesseract for a new fallback invocation.
7. Keep processing local. A hosted endpoint requires a separate privacy and security
   decision.
8. The `ocr` command produces sidecar JSON only and never modifies the source. The
   normalized destination must use a `.json` extension. Producing a searchable PDF is the
   separate, explicit `ocr-compose` step, which consumes a completed sidecar
   deterministically and never runs an engine
   (see [Operations](references/operations.md#searchable-ocr-composition)).

On hybrid pages, the adapter renders detected embedded-image regions when practical. It
skips digital and blank pages unless `--force` is explicit. `--force` means OCR the selected
full pages; it does not mean overwrite the source or bypass model/license preflight.

## Decision matrix

| Situation | Recommendation | Why | Avoid |
| --- | --- | --- | --- |
| Usable digital text | No OCR | Preserves original text and avoids duplicate, probabilistic output | Every OCR engine |
| Complex layout, reading order, tables, math, multilingual structure; suitable accelerator and eligible model use | Surya | Rich ordered blocks, labels, HTML/table payloads | Tesseract |
| CPU-oriented local OCR, no suitable accelerator, or Surya terms unsuitable | PaddleOCR | General OCR and structure-aware pipeline family with local model directories | Silent Surya fallback |
| Surya and PaddleOCR unavailable, or at least 25 clean, high-resolution, mostly single-column printed pages; flat output acceptable | Tesseract | Available last-resort local CPU path with mature batch text/TSV output | Complex tables, handwriting, math, mixed layouts without a fidelity warning |
| Hybrid page | OCR image regions only where detected | Avoids duplicating digital text | Unconditional full-page OCR |

`ocr-plan` reports page classes, estimated pixels, declared layout and volume, detected
hardware, executable availability, language request, model/license prerequisites,
recommendation, and rationale. Availability does not authorize installation. When its
preferred recommendation cannot be provisioned, the agent may propose Tesseract to the user
and then run it only through an explicit `--engine tesseract` invocation.

## Engine boundaries

### Surya

Use Surya for modern layout-aware work. The version-aware adapter supports the declared
`surya-ocr` range and the official `surya_ocr` CLI result schema. It accepts only major
version 0.

Surya code is Apache-2.0. Its current model weights are different: the official project
states that they use a **modified AI Pubs Open Rail-M license** and are free for research,
personal use, and startups under **USD 5 million in funding/revenue**. Broader commercial
use requires a commercial license from the provider. Do not infer eligibility from the code
license. Require both `license_terms_accepted` and `use_case_eligibility_confirmed`.

The tested local llama.cpp artifact identifiers are:

| Item | Revision checked 2026-07-16 | Size | SHA-256 |
| --- | --- | ---: | --- |
| `datalab-to/surya-ocr-2-gguf/surya-2.gguf` | `6a3a4c30e5e74446d4f8b6afd05b2f2da970f470` | 1,266,400,864 bytes | `1f18abe17b1ed8b4e47ee9b1ad0e274c93daf5efbb6b29a04ff1712e37051e05` |
| `datalab-to/surya-ocr-2-gguf/surya-2-mmproj.gguf` | same | 204,986,688 bytes | `98c0563673b1657ff6d021d1e5f04af06cbf61bb40c63ac613e8bb71b42fb2c0` |

These values identify a reviewed revision; they do not bundle or download it. Recheck the
official model card and terms before provisioning. The adapter requires a manifest
`backend` artifact for the reviewed local llama.cpp executable, checks that it reports a
llama.cpp version, forces `llamacpp`, sets local GGUF paths and offline flags, and enables
only local autostart. Remote inference and alternate backends are outside this adapter
because they require separate controls.

Provision llama.cpp through the normal platform package manager, preferring Homebrew when
available on macOS. Follow the installation notice and scope procedure in
[Operations](references/operations.md#environment-and-json-contract). Installation is not
approval: review the exact package version, upstream source and immutable revision, license,
installed `llama-server --version` output, binary size, and SHA-256. Record those facts in
the existing manifest `runtime` object and `backend` artifact, and proceed only when
version/checksum/license preflight passes.

### PaddleOCR

Use PaddleOCR as an operator-selected or planner-recommended no-GPU alternative, never an
automatic runtime fallback. Install it only from the declared pip profile
`requirements-ocr-paddle.txt`; do not use Homebrew or apt Paddle packages. The profile
supports PaddleOCR major version 3 and PaddlePaddle major version 3; pip resolves their
transitives. The adapter accepts only PaddleOCR major version 3.

The official CLI prints a human-oriented representation and does not document strict JSON
stdout. Therefore the tool launches the selected profile's Python interpreter as a
subprocess and uses the official result API and `save_to_json()`. The explicit default
`--paddle-pipeline ocr` uses `PaddleOCR`. For complex CPU-oriented layouts,
`--paddle-pipeline structure` uses `PPStructureV3`, retains parsing order, labels, layout
boxes, Markdown, and table payloads. It requires local layout detection, table
classification, wired/wireless table structure, and wired/wireless table cell-detection
directories.
Both modes pass local `text_detection_model_dir` and `text_recognition_model_dir`, disable
orientation, unwarping, and text-line-orientation modules, and disable the model-source
check. Structure mode enables the fully provisioned table path while disabling seal,
formula, chart, and region models not provisioned by this bounded adapter. Omitting a
required directory would permit an official model download, so preflight rejects the run.

The profile was designed against the official PP-OCRv6 medium identifiers:

| Item | Revision checked 2026-07-16 | Main parameter size | Main parameter SHA-256 | License recorded by official model card |
| --- | --- | ---: | --- | --- |
| `PaddlePaddle/PP-OCRv6_medium_det` | `8e0f56fb2ef86b461d99cfc7ac5c137738985f61` | 61,960,476 bytes | `85218d2e3d98f5a21c58b4220627be923a97aee5db3cc71f39536ab31ac53960` | Apache-2.0 |
| `PaddlePaddle/PP-OCRv6_medium_rec` | `e5a92bcbc5cc1b494628e458d267778f0704fd7c` | 76,465,087 bytes | `1b01c79a914587933f615569e75de54f2e638ebb5d3f3b3c1b38c24ede8c7319` | Apache-2.0 |

Directories also contain configuration files. Record and verify all provisioned files in
your artifact process, not only the main parameter file. The reviewed sizes and hashes above
cover only the named main parameter files; they are **not** complete directory-inventory
values. Model/language compatibility must match the explicit `--languages` value. This
adapter accepts one Paddle language code per run.

The complete PP-StructureV3 role review status is:

| Required role | Candidate family shown by current examples | Artifact review status | Revision | Complete directory size/checksum | Artifact license |
| --- | --- | --- | --- | --- | --- |
| `layout_detection_model_dir` | `PP-DocLayout_plus-L` | Not independently reviewed | Unknown—explicit preflight required | Unknown—explicit preflight required | Unknown—explicit preflight required |
| `table_classification_model_dir` | `PP-LCNet_x1_0_table_cls` | Not independently reviewed | Unknown—explicit preflight required | Unknown—explicit preflight required | Unknown—explicit preflight required |
| `wired_table_structure_recognition_model_dir` | `SLANeXt_wired` | Not independently reviewed | Unknown—explicit preflight required | Unknown—explicit preflight required | Unknown—explicit preflight required |
| `wireless_table_structure_recognition_model_dir` | `SLANet_plus` | Not independently reviewed | Unknown—explicit preflight required | Unknown—explicit preflight required | Unknown—explicit preflight required |
| `wired_table_cells_detection_model_dir` | `RT-DETR-L_wired_table_cell_det` | Not independently reviewed | Unknown—explicit preflight required | Unknown—explicit preflight required | Unknown—explicit preflight required |
| `wireless_table_cells_detection_model_dir` | `RT-DETR-L_wireless_table_cell_det` | Not independently reviewed | Unknown—explicit preflight required | Unknown—explicit preflight required | Unknown—explicit preflight required |

These family names are routing hints, not approved artifact identifiers. Before structure
mode can pass, review each selected official artifact and put its exact `identifier`,
immutable `revision`, official `source`, `license`, canonical directory `size_bytes`, and
canonical directory `sha256` in that role's manifest entry. Do not infer the structure-model
license from PaddleOCR's code license or from the base OCR models.

### Tesseract

Use the external `tesseract` CLI; no Python wrapper is installed. The adapter accepts only
major version 5, verifies requested language IDs with `--list-langs`, invokes TSV output,
and creates line-level blocks. It does not invent layout labels or confidence. Tesseract
supplies word confidence values on a 0–100 scale; normalization averages available values
within a line and maps them to 0–1.

Choose and pin official `tessdata_fast` or `tessdata_best` language files deliberately.
Official repository revisions observed on 2026-07-16 were
`87416418657359cb625c412a48b6e1d6d41c29bd` for `tessdata_fast` and
`e12c65a915945e4c28e237a9b52bc4a8f39a0cec` for `tessdata_best`. Record the exact
`.traineddata` path and checksum used. Tesseract code is Apache-2.0; verify the selected
language-data repository and file terms independently in the manifest.

After Tesseract is explicitly selected, use the normal platform package manager, preferring
Homebrew when available on macOS, and install only the CLI and explicitly needed language
packages. Follow the installation notice and scope procedure in
[Operations](references/operations.md#environment-and-json-contract). Resolve the
executable without assuming its path. Package installation does not satisfy manifest
review: record the installed engine version and exact language-data path, source/package
version, license, size, and checksum, then run preflight.

## Artifact and license preflight

Run:

```bash
"$PDF_PYTHON" "$SKILL_ROOT/scripts/pdf_tool.py" ocr \
  --input scanned.pdf \
  --output unused.json \
  --engine surya \
  --engine-executable "$ENGINE_EXECUTABLE" \
  --model-manifest surya-model.json \
  --languages en \
  --preflight-only
```

Preflight verifies:

- engine and manifest match;
- runtime major version is tested;
- model identifier, immutable revision, official source, and license are present;
- the operator accepted the terms and confirmed use-case eligibility;
- requested languages are covered when the manifest declares a language list;
- required artifact roles exist and are not symlinks;
- supplied file sizes and SHA-256 values match;
- every Paddle model role has artifact-level identifier, revision, source, license, size,
  and checksum provenance;
- Paddle directory size and checksum match the canonical inventory algorithm below;
- Tesseract reports every requested language.

Directory hashes are not guessed. For a Paddle model directory, sort every regular file by
its UTF-8 relative POSIX path and hash, for each file, `path + NUL + decimal byte size + NUL
+ file bytes + NUL`; `size_bytes` is the sum of all file sizes. Symlinks, special files, and
empty directories are rejected. Compute those values during artifact review and record
them in the manifest. Missing or malformed Paddle artifact revision, size, checksum, source,
or license fails with `license_precondition` and an explicit-preflight message; the tool
does not report that provenance as passed.

## Model manifest schema

All engines use the model/artifact portions of this schema; the shown `runtime` object is
additionally required for Surya:

```json
{
  "schema_version": 1,
  "engine": "surya",
  "model": {
    "identifier": "datalab-to/surya-ocr-2-gguf",
    "revision": "6a3a4c30e5e74446d4f8b6afd05b2f2da970f470",
    "source": "https://huggingface.co/datalab-to/surya-ocr-2-gguf",
    "license": "modified AI Pubs Open Rail-M",
    "languages": ["en", "fr"],
    "license_terms_accepted": true,
    "use_case_eligibility_confirmed": true
  },
  "runtime": {
    "identifier": "ggerganov/llama.cpp",
    "revision": "replace-with-reviewed-immutable-revision",
    "source": "https://github.com/ggml-org/llama.cpp",
    "license": "MIT",
    "license_terms_accepted": true
  },
  "artifacts": [
    {
      "role": "model",
      "path": "/models/surya/surya-2.gguf",
      "size_bytes": 1266400864,
      "sha256": "1f18abe17b1ed8b4e47ee9b1ad0e274c93daf5efbb6b29a04ff1712e37051e05"
    },
    {
      "role": "mmproj",
      "path": "/models/surya/surya-2-mmproj.gguf",
      "size_bytes": 204986688,
      "sha256": "98c0563673b1657ff6d021d1e5f04af06cbf61bb40c63ac613e8bb71b42fb2c0"
    },
    {
      "role": "backend",
      "path": "/opt/llama.cpp/llama-server",
      "size_bytes": 12345678,
      "sha256": "replace-with-the-reviewed-binary-sha256"
    }
  ]
}
```

Required roles:

| Engine | Roles |
| --- | --- |
| Surya | `model`, `mmproj`, `backend` |
| PaddleOCR `ocr` | `text_detection_model_dir`, `text_recognition_model_dir` |
| PaddleOCR `structure` | preceding roles plus `layout_detection_model_dir`, `table_classification_model_dir`, `wired_table_structure_recognition_model_dir`, `wireless_table_structure_recognition_model_dir`, `wired_table_cells_detection_model_dir`, and `wireless_table_cells_detection_model_dir` |
| Tesseract | `language_data` directory/file, or one `language_data:<language>` file per requested language |

For Tesseract, a generic `language_data` directory must contain the exact requested
`<language>.traineddata` names; a generic file is valid only for one language. All
language-specific files must share one directory, which is passed exactly through
`--tessdata-dir`. For PaddleOCR, every role points to a complete local inference-model
directory. Every Paddle model artifact entry additionally requires `identifier`, `revision`,
`source`, `license`, canonical directory `size_bytes`, and canonical directory `sha256`.

The `runtime` object is required for Surya and records the exact llama.cpp build represented
by the `backend` artifact. Its required fields are `identifier`, immutable `revision`,
official `source`, `license`, and `license_terms_accepted`. The backend must be a regular
executable file with exact `size_bytes` and `sha256`, must pass its bounded `--version`
probe, and must report a recognized llama.cpp/`llama-server` signature.

Preflight rejects every textual placeholder (`REVIEW-REQUIRED`, `replace-with-…`) in model,
runtime, or artifact provenance fields, in addition to rejecting absent required fields.

## Manifest derivation and checking

The `manifest` command reduces hand-written-manifest ceremony without weakening any gate.
It never executes a binary — filesystem stat and hashing only. Full CLI forms are in
[Operations](references/operations.md#manifest-derive-and-check).

Derivation fills two kinds of fields differently:

- **Measured facts** — paths, byte sizes, SHA-256 values, and canonical Paddle directory
  inventories — are computed by the tool from the local files.
- **Provenance facts** — identifier, revision, source, license — are filled only from an
  explicit operator declaration or an exact SHA-256 match against the reviewed artifacts
  recorded in this document (the two Surya GGUF files and the two PP-OCRv6 main parameter
  files). Everything else is emitted as `REVIEW-REQUIRED`, which preflight rejects, and the
  derive summary lists every reason the manifest is not ready.

For Tesseract, `--assume-source tessdata_fast|tessdata_best` is an operator declaration
that the local `.traineddata` files came from the pinned official repository revisions
above; the manifest records which fields were declared versus measured, and the tool warns
that it verified only path, size, and hash — not origin. Language files are located via
`--tessdata-dir`, `$TESSDATA_PREFIX`, or well-known directories; ambiguity or a missing
language fails rather than guessing. The six PP-StructureV3 structure roles are **never**
auto-filled by any flag. `license_terms_accepted` and `use_case_eligibility_confirmed`
default to `false` and flip only via the explicit `--accept-license` and
`--confirm-eligibility` flags.

`manifest --mode check` sweeps an existing manifest without fail-fast and reports every
problem with expected and actual values — the diagnosis tool when `ocr` preflight fails
with a single mismatch. It excludes, and names, the engine-execution checks that only `ocr`
performs.

## Rendering and execution

The core environment renders only selected pages or hybrid image regions with pdfplumber/
pypdfium2. Before rendering, it estimates pixels from page points and DPI. It rejects more
than 40 million pixels per region, 200 million total pixels, over 500 pages, DPI outside
72–600, and timeouts outside 1–86,400 seconds.

The Tesseract timeout is one monotonic deadline shared by every page/region subprocess in
the invocation; it is not reset for each page. Surya and Paddle each use one bounded engine
subprocess.

Device reporting separates the requested label from adapter-resolved device/backend
metadata. Tesseract
always reports CPU. Paddle reports `cpu` for CPU and MPS requests (with a mapping warning)
and `cuda:0` when the adapter is configured as `gpu:0`; this is adapter configuration, not
an independent hardware-availability probe. `auto` is resolved before execution. Surya
reports the reviewed runtime backend as `llamacpp` and the compute device as
`backend-managed`, because this adapter does not independently prove llama.cpp offload.
Preflight output retains the strictly parsed requested language list and device mapping but
does not claim that an engine ran.

`engine.languages` is the strictly parsed requested-language audit field. Tesseract receives
the complete list, and Paddle's `ocr` pipeline receives its required single identifier.
Surya performs multilingual recognition without a per-run language selector, and Paddle's
`structure` constructor does not consume the language value; for those two modes the field
records operator intent rather than claiming that the engine was constrained to those
languages.

The original PDF remains untouched. Rendered inputs and unretained raw output live in a
temporary directory. `--raw-output-dir` and normalized JSON are completely staged before
publication; an in-process publication failure removes the new raw directory and leaves the
normalized destination unpublished. A nonempty raw destination is never overwritten.
Engine stdout/stderr is redirected to temporary files and only the final 4,000 characters
of each stream can appear in a failure diagnostic.

## Normalized sidecar

The output JSON contains:

```json
{
  "schema_version": 1,
  "operation": "ocr",
  "engine": {
    "name": "paddle",
    "version": "3.7.0",
    "model_identifier": "PaddlePaddle/PP-OCRv6_medium",
    "model_revision": "operator-pinned-revision",
    "requested_device": "mps",
    "resolved_device": "cpu",
    "runtime_backend": "paddlepaddle",
    "device_backend": "cpu",
    "languages": ["en"]
  },
  "source": {
    "path": "/work/scanned.pdf",
    "sha256": "64 lowercase hexadecimal characters",
    "selected_pages": [2],
    "immutable": true
  },
  "pages": [
    {
      "source_page": 2,
      "source_region": null,
      "classification": "likely-scanned",
      "page_geometry": {
        "width_points": 612,
        "height_points": 792,
        "rotation": 0
      },
      "coordinate_space": {
        "unit": "rendered_pixel",
        "dpi": 300,
        "width": 2550,
        "height": 3300
      },
      "blocks": [
        {
          "order": 0,
          "text": "Example heading",
          "block_type": "line",
          "bbox": [100.0, 80.0, 900.0, 160.0],
          "polygon": [[100.0, 80.0], [900.0, 80.0], [900.0, 160.0], [100.0, 160.0]],
          "confidence": 0.94,
          "words": [
            {"text": "Example", "bbox": [100.0, 80.0, 480.0, 160.0], "confidence": 0.95},
            {"text": "heading", "bbox": [500.0, 80.0, 900.0, 160.0], "confidence": 0.93}
          ]
        }
      ],
      "warnings": [],
      "engine_specific": {}
    }
  ],
  "raw_output_path": null,
  "warnings": [],
  "engine_specific": {},
  "preflight": {},
  "timing_seconds": 4.2
}
```

`confidence` is `null` when the engine does not supply one. No value is estimated merely to
complete the schema. `bbox`, `polygon`, and block type remain `null` or conservative when
the source engine is flat. Table HTML/Markdown is retained only when supplied. Fields that
cannot be normalized without loss remain under `engine_specific`.

`page_geometry` records the displayed page size and rotation of the source page so
consumers can cross-check coordinate mappings. `blocks[].words` carries word-level geometry
**only when the engine supplied it** (the Tesseract adapter records its TSV word rows);
words are never synthesized for engines that emit line or block geometry only. Both fields
are additive within `schema_version` 1, and `ocr-compose` uses them for word-accurate
invisible text placement.

## Engine normalization

- **Surya:** requires every rendered input to map to exactly one result image page, then
  preserves ordered labels, raw labels, polygons, boxes, engine confidence, HTML, table
  HTML, skipped state, and errors. Malformed, duplicate, or negative reading order and
  invalid confidence/geometry fail as `engine_failed`; blocks are sorted by supplied order.
- **PaddleOCR `ocr`:** preserves recognition text/scores, recognition polygons/boxes,
  model settings, and detection parameters from official saved JSON.
- **PaddleOCR `structure`:** preserves parsing order, block labels and boxes, Markdown,
  table content, layout details, and raw table results from official `PPStructureV3`
  saved JSON. Content stays string-typed, and invalid order or geometry fails rather than
  being coerced or silently discarded. Structure confidence remains `null` because the
  consumed parsing blocks do not supply a confidence field.
- **Tesseract:** groups nonempty TSV words into line blocks, computes a bounding rectangle,
  and averages only nonnegative engine-supplied word confidences. Required TSV columns,
  row text type, finite integer geometry, documented confidence range, and deterministic
  numeric line order are validated even for rows with empty text; malformed TSV fails as
  `engine_failed`.

Low-confidence warnings are based only on supplied values. Reading order, table structure,
language, and transcription remain probabilistic and require human review.

## Optional profile smoke tests

Use the canonical interpreter selection, user notice, scope, and installation procedure in
[Operations](references/operations.md#environment-and-json-contract). Install only the
explicitly selected profile and do not combine Surya and Paddle requirements or change
their pins. Set `ENGINE_PYTHON` to the selected profile interpreter and, for CLI engines,
resolve `ENGINE_EXECUTABLE` to an absolute path. Setup never authorizes model downloads,
bypasses manifest/license preflight, or permits automatic engine fallback.

```bash
# Run only after explicitly selecting and provisioning Surya.
"$PDF_PYTHON" "$SKILL_ROOT/scripts/smoke_test.py" \
  --ocr-engine surya \
  --engine-executable "$ENGINE_EXECUTABLE" \
  --model-manifest "/models/surya/manifest.json" \
  --languages en \
  --device auto

# Run only after explicitly selecting and provisioning PaddleOCR.
"$PDF_PYTHON" "$SKILL_ROOT/scripts/smoke_test.py" \
  --ocr-engine paddle \
  --engine-executable "$ENGINE_PYTHON" \
  --model-manifest "/models/paddle/manifest.json" \
  --languages en \
  --device cpu

# Run only after explicitly selecting and separately provisioning Tesseract.
ENGINE_EXECUTABLE="$(command -v tesseract)"
test -n "$ENGINE_EXECUTABLE"
"$PDF_PYTHON" "$SKILL_ROOT/scripts/smoke_test.py" \
  --ocr-engine tesseract \
  --engine-executable "$ENGINE_EXECUTABLE" \
  --model-manifest "/models/tesseract/manifest.json" \
  --languages eng \
  --device cpu
```

The optional mode creates clean printed and complex table/layout fixtures at runtime. It
checks engine version, model identifier/revision, resolved device/backend, deterministic
order, finite in-bounds coordinates, supplied confidence range, warning arrays, source
immutability, raw-output retention, and that the planner does not select Tesseract for the
complex fixture.
