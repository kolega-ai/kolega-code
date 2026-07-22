# PPTX tool operations

## Contents

- [Runtime prerequisite and installation](#runtime-prerequisite-and-installation)
- [Invocation and result contract](#invocation-and-result-contract)
- [Safe package handling](#safe-package-handling)
- [Inspect](#inspect)
- [Extract](#extract)
- [Create jobs](#create-jobs)
- [Edit jobs](#edit-jobs)
- [Selectors and replacement rules](#selectors-and-replacement-rules)
- [Content specifications](#content-specifications)
- [LibreOffice prerequisite](#libreoffice-prerequisite)
- [Convert](#convert)
- [Render](#render)
- [Font portability](#font-portability)
- [PPTX and PDF release gate](#pptx-and-pdf-release-gate)
- [Exit statuses](#exit-statuses)

## Runtime prerequisite and installation

Resolve the skill root and select any available Python 3.11+ interpreter. Do not assume a
launcher name. The examples use `PPTX_PYTHON` for the selected interpreter:

```bash
SKILL_ROOT="/absolute/path/to/skills/pptx"
PPTX_PYTHON="/path/to/selected/python"
PPTX_TOOL="$SKILL_ROOT/scripts/pptx_tool.py"
PPTX_SMOKE_TEST="$SKILL_ROOT/scripts/smoke_test.py"
"$PPTX_PYTHON" -c 'import sys; print(sys.version.split()[0]); raise SystemExit(0 if sys.version_info >= (3, 11) else "Python 3.11+ is required")'
```

Check the required imports. Before installing anything, tell the user what is missing, what
will be installed, the target scope, and the installer. Use the selected interpreter's
`-m pip` with the requirements file:

```bash
"$PPTX_PYTHON" -m pip install -r "$SKILL_ROOT/requirements.txt"
```

Prefer the active or user interpreter. If platform policy blocks those scopes, explain the
fallback before using a local environment. Use the normal platform package manager for a
missing interpreter or optional LibreOffice; prefer Homebrew on macOS when available. Never
use privileged pip or `--break-system-packages`, and do not edit host-project dependency
metadata. Use the same `PPTX_PYTHON` for installation, tools, and smoke tests.

## Invocation and result contract

Run the tool with the same active interpreter used to install requirements. Quote every resolved
path so skill locations and documents containing spaces remain safe:

```bash
"$PPTX_PYTHON" "$PPTX_TOOL" inspect "/path/to/input deck.pptx"
"$PPTX_PYTHON" "$PPTX_TOOL" extract "/path/to/input deck.pptx" --output "/path/to/image dir"
"$PPTX_PYTHON" "$PPTX_TOOL" create --job "/path/to/create job.json" --output "/path/to/output deck.pptx"
"$PPTX_PYTHON" "$PPTX_TOOL" edit "/path/to/input deck.pptx" --job "/path/to/edit job.json" --output "/path/to/output deck.pptx"
"$PPTX_PYTHON" "$PPTX_TOOL" convert "/path/to/input deck.pptx" --output "/path/to/output deck.pdf"
"$PPTX_PYTHON" "$PPTX_TOOL" render "/path/to/input deck.pptx" --output "/path/to/slide images"
"$PPTX_PYTHON" "$PPTX_SMOKE_TEST"
```

Use `--overwrite` to replace an existing destination. A template/source and destination must
always be distinct paths; `--overwrite` applies only to a separate destination and never enables
in-place mutation. `--job -` reads a job from stdin and resolves relative resource paths from the
current directory. A job file resolves them from its own directory.

Every successful stdout document is JSON with:

- `schema_version`, `success`, and `operation`;
- operation-specific counts and warnings;
- exact Python/library versions;
- source/output paths;
- `verification`, including package preflight, reopen, slide order/count, existence of internal
  relationship targets, resolution of owner XML relationship references, and atomic publication
  where applicable.

Failures are one JSON object on stderr and use a nonzero stable status. Do not parse human usage
text to determine failure type.

## Safe package handling

Preflight accepts only local `.pptx` ZIP packages with required PresentationML parts. It rejects:

- wrong extensions or signatures, encrypted ZIP members, macros, or `vbaProject.bin`;
- absolute, parent-traversing, backslash, duplicate, or symbolic-link ZIP entries;
- more than 5,000 members, more than 512 MiB expanded data, a member over 64 MiB, suspicious
  compression ratios, unsupported ZIP compression, or a source over 100 MiB;
- OOXML `.xml`, `.rels`, or `.vml` members containing a DTD or entities; malformed or orphan
  relationship parts; duplicate/malformed relationship IDs; broken internal targets; unresolved
  owner `r:id`/`r:embed`/`r:link` references in those XML-based members; missing slide targets;
  duplicate slide IDs; or unreachable slide parts.

XML is parsed with DTD loading, external entities, network access, and huge-tree mode disabled.
External hyperlinks are counted but never followed. Macro-enabled `.pptm` files are unsupported.
Inspection removes URL userinfo and replaces values of credential-, token-, session-, signature-,
and secret-like query parameters with `<redacted>` while preserving useful scheme/host/path,
ordinary query fields, duplicate keys, and fragments. Reported hyperlink URLs are therefore safe
representations rather than byte-identical source values.

## Inspect

`inspect INPUT.pptx` returns:

- slide size, core properties, masters, layouts, slide IDs/order/layouts;
- every shape's stable shape ID, name, type, placeholder metadata, and geometry in EMU/inches;
- text frames with ordered paragraphs, levels, alignment, runs, font properties, and field
  warnings;
- image filename/type/byte count/dimensions/SHA-256;
- table cell text and chart type/title/categories/series values available through public APIs;
- plain speaker-note text read from package XML without creating a notes part;
- a package font inventory under `fonts`: `referenced`, `embedded`, `unembedded`, per-part
  `references`, `theme_tokens_referenced`, `symbol_and_bullet`, and
  `dangling_embedding_relationships`, with an informational warning for unembedded fonts and a
  `RELEASE BLOCKER:` warning for unembedded fonts outside Arial, Times New Roman, and Courier
  New;
- package feature warnings for signatures, comments, SmartArt, embedded objects, media,
  transitions, animations, and other uncertain content.

Inspection is read-only. Unsupported chart metadata is reported as a warning rather than invented.

## Extract

`extract INPUT.pptx --output DIR [--overwrite]` exports every picture shape's image bytes to a
new directory, verbatim and without re-encoding. File names follow
`slide-<slide_id>-shape-<shape_id>-<sha256 prefix>.<ext>`, so the same image placed on two
shapes produces two files whose shared hash makes the duplication visible. The summary lists
each image's slide/shape identity, content type, byte count, SHA-256, and file name.

Scope is picture shapes only: shape-fill images, slide backgrounds, audio, video, and embedded
objects are not exported; unexported `ppt/media/` members are counted and named in a warning.
An image above the per-image byte limit is skipped with a warning rather than failing the whole
extraction. The destination must not already exist; with `--overwrite` it must be an empty
directory — the tool never deletes existing files. Output is staged and atomically published,
then every published file is re-read and its hash verified.

## Create jobs

A create job has this top-level shape:

```json
{
  "schema_version": 1,
  "operation": "create",
  "template": "brand-template.pptx",
  "keep_template_slides": false,
  "metadata": {"title": "Quarterly review", "author": "Analyst"},
  "slides": []
}
```

`template` is optional. Without one, `python-pptx`'s default presentation is used. With one, the
template's masters and layouts are retained. Existing template slides are removed by default;
set `keep_template_slides` to `true` to retain them. Template removal uses the same
version-aware, reopen-verified relationship adapter as edit operations.

A slide may select a `layout` with either `{"name": "Exact Layout Name"}` or `{"index": 1}`.
If `layout` is omitted, layout index 1 is used and must exist. Duplicate layout names are
ambiguous. Supported slide keys are `title`, `body`, `elements`, and `notes`.

## Edit jobs

An edit job contains ordered operations:

```json
{
  "schema_version": 1,
  "operation": "edit",
  "operations": [
    {"action": "replace_text", "find": "Old", "replace": "New"},
    {"action": "set_notes", "slide": {"slide_id": 256}, "text": "Plain notes"}
  ]
}
```

Supported actions:

- `add_slide`: use a create-style slide object under `slide`; append by default or provide an
  insertion `index`.
- `replace_text`: provide `find`, `replace`, optional slide/shape selectors, and exactly one of
  `replace_all: true` or `occurrence` when the selection is not unique.
- `set_hyperlink`: with the default `scope: "text"`, provide `find` (plus `occurrence` when not
  unique) and a validated `url`; every covered run receives the address. With `scope: "shape"`,
  select exactly one shape and set its click action instead; `find`/`occurrence` are rejected.
- `remove_hyperlink`: same selection rules without `url`. Every covered run (or the shape click
  action) must currently carry an address; partial or absent coverage is rejected.
- `update_table`: select one table and provide an exact-size `data` matrix or `cells` entries
  such as `{"row": 1, "column": 2, "text": "42"}`.
- `update_table_structure`: select one table and provide ordered `changes`, each
  `{"op": "insert_row"|"remove_row"|"insert_column"|"remove_column", "index": N}` with optional
  scalar `cells` content for insertions (sized to the current cross-dimension). Each change is
  validated against the table as already changed. Insert indexes run `0..n` (`n` appends);
  removal indexes run `0..n-1`, and the final row or column cannot be removed. Tables containing
  merged cells are rejected. New rows and columns clone neighbor formatting with cleared text; an
  inserted column widens the table rather than redistributing widths. A before/after text-matrix
  verification guards every structural edit.
- `update_chart`: select one chart and provide a complete category or XY chart data specification.
- `replace_image`: select one picture and provide `path`. The replacement must use the same media
  content type. Geometry/crop and unselected picture relationships remain unchanged; the selected
  picture receives a relationship to the validated replacement image.
- `set_notes`: select a slide and provide plain `text`.
- `remove_slide`: select exactly one slide.
- `reorder_slides`: provide `slide_ids` containing every current slide ID exactly once.

After every add, remove, or reorder, the adapter saves a checkpoint, runs safe preflight, reopens
it, and verifies retained IDs, exact order/count, internal target existence, and owner
relationship-reference resolution before continuing. The adapter is enabled only for the
supported `python-pptx` range and fails closed outside it.

## Selectors and replacement rules

Slide indexes are zero-based and refer to the current operation state. Prefer
`{"slide_id": 256}` over `{"slide_index": 0}`. Shape selectors support exact `shape_name`,
zero-based `shape_index`, or a type-local zero-based `table_index`, `chart_index`, or
`image_index`. Indexes apply to the complete selected slide scope; without a slide selector that
scope is the whole deck. This deck-wide behavior applies consistently to text, table, chart, and
image operations. A selector must identify exactly one object. Omitting `shape` is allowed only
when the selected slide/deck scope itself contains exactly one shape; the tool never silently
chooses the first shape.

Replacement searches each paragraph's ordered runs. A match inside one run preserves that run's
formatting. A cross-run match is rejected unless `formatting_policy` is `"first_run"`; replacement
text then uses the first matched run's formatting while unmatched prefixes/suffixes retain their
runs. Matches crossing paragraphs, fields, or incompatible hyperlink boundaries are rejected.
Set `destructive_reconstruction: true` only to replace within a flattened text frame and accept
loss of run/paragraph formatting, fields, and hyperlink structure. Empty search strings are
invalid.

Hyperlink actions never split runs: the matched text must align exactly with whole-run
boundaries, or the action fails as ambiguous with guidance to isolate the target text with
`replace_text` first. URLs must be `http`, `https`, or `mailto`, at most 2048 characters, and
free of whitespace and control characters; other schemes (`file:`, `javascript:`, `data:`,
`ppaction:`) are rejected as unsupported. Inspection continues to report sanitized URL
representations regardless of how a hyperlink was authored.

## Content specifications

All geometry uses inches:

```json
{"x": 1.0, "y": 1.5, "width": 6.0, "height": 2.0}
```

`title` may be a string or a paragraph object. `body` is a string or paragraph list. A paragraph
supports `text` or `runs`, `level` (0-8), and `alignment` (`left`, `center`, `right`, `justify`).
A run supports `text`, `bold`, `italic`, `font_name`, `font_size`, and six-digit `color`.

Elements:

- Text: `{"type":"text","name":"Callout","box":{...},"paragraphs":[...]}`
- Image: `{"type":"image","name":"Photo","path":"photo.png","box":{...}}`; omit height or width
  to preserve aspect ratio.
- Table: `{"type":"table","name":"Data","box":{...},"data":[["A","B"],["1","2"]]}`.
- Category chart:
  `{"type":"chart","chart_type":"column","name":"Chart","box":{...},"categories":["Q1"],"series":[{"name":"Sales","values":[10]}]}`.
- XY chart:
  `{"type":"chart","chart_type":"scatter","name":"Chart","box":{...},"series":[{"name":"Fit","points":[[1,2],[2,4]]}]}`.

Chart types are `column`, `bar`, `line`, `line_markers`, `pie`, `doughnut`, `area`, `scatter`,
and `scatter_lines`. Add `title` and `has_legend` as needed. Tables and charts remain editable.
Notes are plain text only. Resource files are bounded and validated before use. Images are limited
to 50 MiB compressed, 20,000 pixels per dimension, 50 million pixels, one frame, and an estimated
200 MiB decoded raster. Each image path is read once into a bounded immutable byte snapshot;
Pillow validation, full decode, and `python-pptx` ingestion all consume that same snapshot.

## LibreOffice prerequisite

LibreOffice is optional and needed only for PPTX-to-PDF conversion. PPTX-to-PDF remains a
source-format operation owned and validated by this skill; do not hand it off as PDF-native
editing. Before conversion, verify an executable is on `PATH`:

```bash
command -v soffice || command -v libreoffice
```

If conversion is requested and the executable is missing, obtain user approval before installing
the optional system package. First state that LibreOffice is missing, that the optional
LibreOffice system package will be installed, its system scope, and the platform package-manager
mechanism. Use the platform's normal package manager, preferring Homebrew on macOS when available,
then repeat the PATH check.

Stop with an external-precondition failure if neither command resolves after installation. Do not
claim conversion support merely because LibreOffice's application files exist.

## Convert

`convert` supports PPTX to PDF only and requires `soffice` or `libreoffice` on `PATH`. It:

1. preflights the source and rejects any external relationship;
2. launches headless LibreOffice with a fresh isolated user profile and configurable
   `--timeout` (default 120 seconds) in a minimal environment;
3. bounds and sanitizes diagnostics, including paths, URL credentials, and secret-like values;
4. checks the PDF signature, opens it with `pypdf`, and requires one PDF page per slide;
5. atomically publishes the requested destination.

Inspection and every template/source-backed create/edit/convert operation consume a validated
immutable source snapshot. Immediately before atomic publication, the live source path must still
match that snapshot. An immediate post-publication comparison reports an observed source race by
setting `verification.post_publish_source_changed: true` and emitting a warning, without turning
the committed operation into a contradictory failure. Thus `source_unchanged_at_publish_gate`
proves the decisive pre-commit check, not indefinite immutability of the path after commit.

Conversion is best effort. Always visually review typography, line wrapping, charts, and media in
the target environment.

## Render

`render INPUT.pptx --output DIR [--overwrite] [--dpi 96] [--slides ID,ID] [--timeout 120]`
produces one PNG per slide so the result can be reviewed visually. Rendering exists to be
looked at: open the produced PNGs and examine them; do not treat a successful render summary as
visual proof by itself.

Prerequisites are LibreOffice (as for `convert`) plus the optional `pypdfium2` Python package.
Neither is installed automatically: if `pypdfium2` is missing, the command fails with
`missing_dependency` and an install hint; tell the user what will be installed and where before
running `"$PPTX_PYTHON" -m pip install pypdfium2`. Users who decline the package can instead
convert to PDF and rasterize with an external tool such as poppler's `pdftoppm`.

Behavior:

1. The source is preflighted and any external relationship is rejected, exactly as `convert`.
2. LibreOffice converts the whole deck to a PDF in an isolated workspace; the PDF must contain
   one page per slide, which also proves the page-to-slide mapping.
3. Requested pages (all slides, or the `--slides` comma-separated slide-ID subset) are
   rasterized with PDFium at `--dpi` (36-300). Predicted pixel sizes are bounded per slide and
   in total before LibreOffice launches.
4. Files are named `slide-<ordinal>-id-<slide_id>.png`, verified (PNG signature, Pillow reopen,
   identical dimensions, pixel caps), staged, and atomically published to a new directory with
   the same empty-destination rules as `extract`.

The PNGs show LibreOffice's interpretation of the deck, not renderer-identical PowerPoint
output; treat them as a layout review aid, with the target presentation application as the
final authority.

## Font portability

A font counts as embedded only when `ppt/presentation.xml` declares it in `p:embeddedFontLst`
and the embedding relationship resolves to a font part inside the package. Theme tokens such as
`+mj-lt` and `+mn-lt` are resolved through the theme font scheme; slide masters always reference
them, so the theme's major and minor fonts always count as referenced even before any slide sets
an explicit font. Fonts embedded in a generated PDF are not thereby embedded in the source PPTX,
and `pdffonts` describes only the PDF. Renderer substitution changes wrapping, autofit
shrink-to-fit, overflow, and slide layout even when the LibreOffice PDF looks correct.

## PPTX and PDF release gate

For matching PPTX and PDF deliverables:

1. Use only Arial for headings and sans body, Times New Roman for serif body, and Courier New
   for code or logs, including table and chart text.
2. Do not release while an unembedded font outside that set remains; replace every
   `RELEASE BLOCKER:` font warning or genuinely embed the licensed font in the PPTX.
3. Inspect the final PPTX and review its `fonts` inventory; render the final PDF and review it
   slide by slide.
4. PDF font embedding does not satisfy PPTX portability, and a clean LibreOffice PDF alone does
   not prove that PowerPoint preserves wrapping, autofit, or slide layout.

## Exit statuses

| Status | Category | Meaning |
|---:|---|---|
| 2 | `bad_input` | Invalid arguments, JSON, schema, path, extension, or malformed package |
| 3 | `unsupported_operation` | Unsupported content, format, or job action |
| 4 | `missing_dependency` | Required Python package is missing or version is incompatible |
| 5 | `ambiguous_edit` | A selector or replacement is unsafe or non-unique |
| 6 | `resource_limit` | A package, image, table, chart, or text limit was exceeded |
| 7 | `external_precondition` | Optional external capability is unavailable |
| 8 | `post_write_validation` | Reopen, ID/order, relationship, or PDF verification failed |
| 9 | `external_tool_failure` | LibreOffice failed, timed out, or produced invalid output |
| 10 | `internal_error` | Unexpected failure; no destination is published |
