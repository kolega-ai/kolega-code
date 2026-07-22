---
name: pdf
description: Create, inspect, render, extract, edit, secure, redact, reorganize, and convert PDF-native documents; diagnose scanned or hybrid PDFs, run explicitly selected local OCR, and compose searchable PDFs from OCR results. Use for PDF page operations, forms, metadata, encryption, spatial text or table extraction, image/text-to-PDF generation, page rasterization for visual review, true content redaction, scanned-PDF OCR planning, and making scanned PDFs searchable. Do not use for authoring DOCX, XLSX, or PPTX source files before PDF export.
license: Apache-2.0
compatibility: Requires Python 3.11+ and the declared core environment. OCR engines, model weights, language data, and compatible acceleration are optional external capabilities installed separately.
metadata:
  author: Kolega
  version: "1.1"
  schema_version: "1"
---

# PDF operations

Use `scripts/pdf_tool.py` for deterministic PDF-native work. Keep every source immutable,
write to a distinct destination, and verify the published output.

## Workspace discipline

Work directly in visible workspace paths. Never create or use `.build` or another hidden
build/work directory. Write requested deliverables directly to their declared destinations;
if intermediate files are necessary, keep them in visible, narrowly scoped paths rather than
staging the task in a hidden subtree.

## Route the task

- Use `inspect` before mutation to inventory pages, encryption, forms, fonts, images,
  text, and likely scanned content.
- Use `extract` for text, coordinates, tables, and embedded images.
- Use `render` to rasterize pages to PNG files for visual review.
- Use `create` for a new PDF from a versioned JSON layout or story specification.
- Use `pages` to merge, split, select, reorder, repeat, delete, or rotate pages.
- Use `edit` for stamps, form values, metadata, encryption, or decryption.
- Use `redact` for true content removal with a verification report. Never present `edit`
  stamps, white rectangles, or crops as redaction.
- Use `convert` only for the explicit lossy mappings it supports.
- Use `manifest` to derive or check an OCR model manifest before OCR; it hashes local
  files and never runs an engine.
- Use `ocr-plan` before OCR. Use `ocr` only after the user selects an engine and approves
  its separately provisioned models or language data. If Surya and PaddleOCR are unavailable,
  tell the user and offer Tesseract as the explicit fallback.
- Use `ocr-compose` after `ocr` to turn the sidecar into a searchable PDF with an
  invisible text layer; it never runs an engine.
- Keep DOCX/XLSX/PPTX authoring and their PDF export in the source-format workflow.

Read [operations](references/operations.md) for command and job schemas. Read
[examples](references/examples.md) before the first end-to-end invocation. Read
[quality](references/quality.md) before promising a polished deliverable, and
[limitations](references/limitations.md) before promising fidelity or signatures. Read
[redaction](references/redaction.md) before any redaction request.

## Core workflow

1. Resolve the skill root and choose any available Python 3.11+ interpreter; do not assume a
   launcher name. Check the required imports before installing anything. If something is
   missing, first tell the user what you intend to install, where, and with which installer.
Use the selected interpreter's `-m pip` for the declared core or explicitly selected OCR
   profile. Use the platform's package manager for Python and external OCR runtimes,
   preferring Homebrew on macOS when available. A local environment is a fallback, not a
   prerequisite. See
   [Environment and JSON contract](references/operations.md#environment-and-json-contract).
2. Run `inspect` on every source. Stop on encryption without an approved password,
   malformed input, unexpected scale, or unsupported content that affects the request.
3. Choose the narrowest operation. Select pages explicitly for large documents.
4. Write to a new path. Use `--overwrite` only when replacing an existing destination is
   intentional; it never permits a source path to be its own destination.
5. Reopen the output and inspect the JSON verification result. Treat warnings as unresolved
   preservation or interpretation questions, not as success noise. Review the `fonts`
   inventory and resolve every `RELEASE BLOCKER` font warning before release.
6. **Render and look.** When layout, formatting, or visual fidelity matters, run `render`
   to produce one PNG per page, then open the rendered PNG files with the Read tool and
   examine them — at minimum every page you changed, plus the first page. Check for text
   overflow or truncation, overlapping or clipped content, missing images, and broken
   table layout. Do not claim visual correctness from JSON verification alone. The PNG
   files are PDFium rasters of the delivered PDF itself, so they are authoritative for
   static page appearance; non-embedded fonts still render with local substitutes, and
   interactive form or annotation appearance can differ per viewer.

The CLI emits a schema-versioned JSON object on stdout. Errors and diagnostics are JSON on
stderr with stable categories and nonzero exit statuses. It bounds source size, page count,
extracted content, rendered pixels, image count, and subprocess time. Outputs are built as
temporary siblings, reopened, and atomically published.

## OCR decision

Never OCR usable digital text by default. Preserve digital text on hybrid pages and OCR only
identified image regions when practical.

1. Prefer **Surya** for complex layouts, reading order, tables, multilingual pages, math, and
   richer structure when its runtime is suitable and the operator confirms the selected
   model terms permit the use. Surya code is Apache-2.0; current Surya model weights use a
   modified AI Pubs Open Rail-M license and are free for research, personal use, and startups
   under USD 5 million in funding/revenue. Broader commercial use requires a commercial
   license from the model provider.
2. Prefer **PaddleOCR** when no suitable accelerator is available, Surya model terms are
   unsuitable, or a local CPU-oriented pipeline is required. Use explicit
   `--paddle-pipeline structure` with a provisioned local layout model when complex-layout
   parsing is the reason for that recommendation; otherwise use `ocr`.
3. Use **Tesseract** as the last-resort local fallback when Surya and PaddleOCR are
   unavailable. It is best for high-volume, clean, high-resolution, mostly single-column
   printed text on CPU when flat text/TSV is sufficient; warn about reduced fidelity for
   complex layouts, tables, handwriting, or math.

The `pdf_tool.py ocr` command never accepts `auto` as an OCR engine, never installs an
engine, never downloads a model, and never silently falls back within an invocation. After
informing the user that a preferred engine is unavailable, the agent may explicitly select
Tesseract and start a new invocation. Operator-run provisioning may install only the
explicitly selected OCR profile as documented in
[OCR](references/ocr.md); provisioning does not select or approve an engine. Each run
requires `--engine
surya|paddle|tesseract`, explicit nonempty languages, and a model manifest recording
artifact-level source, revision, license, size/inventory checksum, and operator approval.
The PP-StructureV3 artifact families have not all been independently reviewed; unknown
facts require explicit preflight and must never be filled with assumptions. Read the
complete [OCR policy and manifest schema](references/ocr.md) before planning or running
OCR.

## Resources

- [Operations](references/operations.md): CLI options, JSON jobs, limits, and exit statuses.
- [Examples](references/examples.md): copy-pasteable core, redaction, and OCR workflows.
- [Quality](references/quality.md): design intake, delivery profiles, acceptance criteria,
  and the structural-versus-visual review contract.
- [Redaction](references/redaction.md): leakage policy table, verification contract, and
  explicit non-guarantees.
- [OCR](references/ocr.md): decision matrix, artifact preflight, manifest derivation,
  adapters, and normalized output.
- [Limitations](references/limitations.md): fidelity and security boundaries.
- `scripts/smoke_test.py`: generated-fixture core smoke test; add `--ocr-engine` only in a
  separately prepared optional engine environment.

## Final verification

Run:

```bash
"$PDF_PYTHON" "$SKILL_ROOT/scripts/smoke_test.py"
```

Use the absolute `SKILL_ROOT` and `PDF_PYTHON` selected by the canonical environment
procedure. For a requested artifact, also confirm against the
[acceptance criteria](references/quality.md#acceptance-criteria): expected page
count/order, text anchors, a font inventory with zero `RELEASE BLOCKER` warnings, form and
encryption state, metadata, output signature — and rendered-page review of every changed
page plus page 1 via `render` and the Read tool. For a redaction, additionally confirm the
report's removal evidence, clean residual scan, and the published verification rasters.
Report every optional OCR profile not exercised.
