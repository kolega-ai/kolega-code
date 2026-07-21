# Operations contract

## Contents

- [Environment](#environment)
- [Invocation forms](#invocation-forms)
- [JSON and output envelopes](#json-and-output-envelopes)
- [Inspect](#inspect)
- [Create](#create)
- [Edit](#edit)
- [Convert](#convert)
- [Content blocks](#content-blocks)
- [Headers and footers](#headers-and-footers)
- [Replacement policy](#replacement-policy)
- [Safety and atomicity](#safety-and-atomicity)
- [Exit statuses](#exit-statuses)

## Environment

Resolve the skill root and select any available Python 3.11+ interpreter. Do not assume a
launcher name. The examples use `DOCX_PYTHON` for the selected interpreter:

```bash
SKILL_ROOT="/absolute/path/to/skills/docx"
DOCX_PYTHON="/path/to/selected/python"
test -f "$SKILL_ROOT/requirements.txt"
"$DOCX_PYTHON" -c 'import sys; print(sys.version); raise SystemExit(0 if sys.version_info >= (3, 11) else "Python 3.11+ is required")'
```

Check the required imports. Before installing anything, tell the user what is missing, what
will be installed, the target scope, and the installer. Use the selected interpreter's
`-m pip` with the requirements file:

```bash
"$DOCX_PYTHON" -m pip install -r "$SKILL_ROOT/requirements.txt"
```

Prefer the active or user interpreter. If platform policy blocks those scopes, explain the
fallback before using a local environment. Use the normal platform package manager for a
missing interpreter or system application; prefer Homebrew on macOS when available. Never
use privileged pip or `--break-system-packages`, and do not edit host-project dependency
metadata.

Run `"$SKILL_ROOT/scripts/docx_tool.py"` with the same verified `DOCX_PYTHON`. LibreOffice is
optional and needed only for PDF conversion. Install it only when required, after the same
user notice, using the platform's normal package manager.

## Invocation forms

Use one direct subcommand:

```text
docx_tool.py inspect --input SOURCE.docx [--allow-external-relationships]
docx_tool.py create --spec CREATE.json --output DEST.docx [--template BASE.docx] [--allow-external-relationships] [--overwrite]
docx_tool.py edit --input SOURCE.docx --spec EDIT.json --output DEST.docx [--allow-external-relationships] [--overwrite]
docx_tool.py convert --input SOURCE.docx --format text|pdf --output DEST [--timeout 90] [--allow-external-relationships] [--overwrite]
```

Or dispatch one complete job:

```text
docx_tool.py --job JOB.json
```

Do not combine `--job` with a subcommand. Paths inside JSON are resolved relative to the JSON
file. Flag paths are resolved relative to the current directory.

## JSON and output envelopes

Every JSON input is an object containing exactly this supported version marker:

```json
{"schema_version": 1}
```

Every object uses a strict operational schema. Unknown top-level, content, block, run,
header/footer, metadata, job, and edit-operation properties fail as `bad_input`. Unknown
operation and block types also fail as `bad_input`; values are not stringified or otherwise
coerced to make invalid JSON fit.

Boolean properties such as `overwrite`, `replace_all`, `ordered`, and run-format flags must
be JSON `true` or `false`; strings and numbers are rejected. Text-bearing properties must be
JSON strings. A paragraph-like object may contain `text` or `runs`, but not both.

Success writes one JSON object to stdout:

```json
{
  "schema_version": 1,
  "status": "ok",
  "operation": "inspect",
  "counts": {},
  "warnings": [],
  "versions": {},
  "paths": {},
  "verification": {},
  "preservation": {},
  "result": {}
}
```

Warnings are repeated as one-line JSON diagnostics on stderr. Failures write a JSON error to
stderr and leave stdout empty:

```json
{
  "schema_version": 1,
  "status": "error",
  "error": {
    "category": "bad_input",
    "message": "DOCX input does not exist.",
    "details": {}
  }
}
```

Do not parse human prose from either channel. Treat the named keys and exit statuses below as
the stable schema-version 1 contract. Additional result fields may be added compatibly.

## Inspect

Direct form:

```bash
"$DOCX_PYTHON" "$SKILL_ROOT/scripts/docx_tool.py" inspect --input "report.docx"
```

Job form:

```json
{
  "schema_version": 1,
  "operation": "inspect",
  "input": "report.docx",
  "allow_external_relationships": false
}
```

Inspection reports:

- Main-body paragraphs and tables in order.
- Named paragraph, run, and table styles.
- Run text and supported formatting.
- Heading levels and OOXML list identifiers.
- Section starts, orientation, dimensions, margins, and header/footer distance.
- Default, first-page, and even-page header/footer stories.
- Core metadata.
- Media hashes and inline-image occurrences.
- Fields, revisions, comments, floating drawings, unsupported drawings, unsupported parts,
  and explicitly allowed external-relationship warnings.
- ZIP member and expanded-byte counts.

Inspect before editing. A successful inspection proves package preflight and library reopen;
it does not prove visual fidelity.

## Create

A direct create specification contains `schema_version` and a `content` object:

```json
{
  "schema_version": 1,
  "content": {
    "template": "letterhead.docx",
    "metadata": {"title": "Quarterly brief", "author": "Example Team"},
    "blocks": [],
    "headers": {},
    "footers": {}
  }
}
```

For complete `--job` dispatch, add:

```json
{
  "schema_version": 1,
  "operation": "create",
  "output": "brief.docx",
  "overwrite": false,
  "allow_external_relationships": false,
  "content": {"blocks": []}
}
```

`content.template` and `content.letterhead` are equivalent DOCX base-document inputs; specify
at most one. `--template` overrides either JSON property. The template is preflighted and
never modified. Without a base document, the operation starts from the standard
`python-docx` document.

Supported metadata properties are `title`, `subject`, `author`, `keywords`, `comments`,
`category`, `last_modified_by`, `revision`, `identifier`, `language`, and `version`.

## Edit

A direct edit specification is:

```json
{
  "schema_version": 1,
  "operations": [
    {
      "type": "replace_text",
      "find": "draft",
      "replace": "final",
      "expected_count": 1,
      "scope": "body"
    }
  ]
}
```

For complete `--job` dispatch, add `operation`, `input`, `output`, and optional `overwrite`:

```json
{
  "schema_version": 1,
  "operation": "edit",
  "input": "source.docx",
  "output": "edited.docx",
  "overwrite": false,
  "allow_external_relationships": false,
  "operations": [{"type": "delete_paragraph", "paragraph_index": 0}]
}
```

Operations execute sequentially against one in-memory document. Indices therefore refer to
the document state at that point in the array. Any failure discards the whole mutation.

Supported edit operations:

- `insert_paragraph`: `block` and an explicit `position`. Use `"position": "append"` with no
  `paragraph_index`, or use `"position": "before"`/`"after"` with a required top-level
  `paragraph_index`. Relative indices are validated against the paragraph count before the
  new paragraph is created.
- `delete_paragraph`: top-level `paragraph_index`.
- `replace_text`: `find`, `replace`, `expected_count`, optional `replace_all`, `scope`, and
  `cross_run_policy`.
- `insert_table`: `table`, optional `paragraph_index`, and `position`.
- `update_table`: `table_index`, `row`, `column`, `value`, and optional `style`.
- `insert_image`: `path`, optional `paragraph_index`, `width_inches`, and `height_inches`.
- `replace_image`: body `inline_image_index`, `path`, optional dimensions.
- `set_metadata`: `metadata`.
- `set_header` and `set_footer`: `section_index`, `kind`, `mode`, and `blocks`.

Top-level paragraph and table indices exclude nested cell/story content. `replace_text` can
include nested table cells and stories through its scope.

## Convert

DOCX to UTF-8 plain text:

```bash
"$DOCX_PYTHON" "$SKILL_ROOT/scripts/docx_tool.py" convert \
  --input "source.docx" --format text --output "source.txt"
```

The converter uses the inspector's ordered body model. Body and header/footer table cells are
tab-delimited. Non-empty headers and footers are appended with story labels and deduplicated
only when the same story part is inherited; distinct stories are never collapsed by equal
displayed text, and header and footer identities remain separate.

DOCX to PDF:

```bash
"$DOCX_PYTHON" "$SKILL_ROOT/scripts/docx_tool.py" convert \
  --input "source.docx" --format pdf --output "source.pdf" --timeout 90
```

PDF conversion:

1. Preflights the DOCX.
2. Locates `soffice`.
3. Runs headless LibreOffice with a unique temporary user profile, home, temp directory, and
   POSIX process group.
4. Applies the bounded timeout, captures at most 4,000 bytes from each diagnostic stream, and
   terminates the process group (including ordinary descendants) before removing temporary
   directories.
5. Requires exactly one output with a `%PDF-` signature and at most 64 MiB.
6. Reopens the output with `pypdf`, requires 1–1,000 pages, limits each decompressed stream
   encountered during validation to 8 MiB, and extracts text from at most the first 50 pages
   with a 1,000,000-character validation limit.
7. Atomically publishes the validated PDF.

Conversion is best-effort. LibreOffice layout can differ from Word layout.
The isolated profile and process group are lifecycle/configuration controls, not a network
sandbox.

A complete conversion job is:

```json
{
  "schema_version": 1,
  "operation": "convert",
  "input": "source.docx",
  "output": "source.pdf",
  "format": "pdf",
  "timeout": 90,
  "overwrite": false,
  "allow_external_relationships": false
}
```

## Content blocks

Use these objects in `content.blocks`, list items, table cells where noted, and header/footer
stories.

### Paragraph

```json
{
  "type": "paragraph",
  "style": "Quote",
  "alignment": "left",
  "runs": [
    {
      "text": "Important",
      "bold": true,
      "italic": false,
      "underline": false,
      "font": "Aptos",
      "size_pt": 11,
      "color": "B42318",
      "style": "Emphasis"
    }
  ]
}
```

Use `text` or `runs`, never both. Text and each run's required `text` property must be JSON
strings. Supported alignments are `left`, `center`, `right`, and `justify`. Prefer named
styles; direct formatting is for intentional exceptions.

### Heading

```json
{"type": "heading", "level": 1, "text": "Summary"}
```

Levels 1–9 use `Heading N`; level 0 uses `Title`.

### List

```json
{
  "type": "list",
  "ordered": false,
  "items": ["First", {"runs": [{"text": "Second", "bold": true}]}]
}
```

Lists use the template's `List Bullet` or `List Number` named style.

### Page break and section break

```json
{"type": "page_break"}
```

```json
{
  "type": "section_break",
  "start": "new_page",
  "orientation": "landscape",
  "top_margin_inches": 0.75,
  "right_margin_inches": 0.75,
  "bottom_margin_inches": 0.75,
  "left_margin_inches": 0.75
}
```

Section starts are `new_page`, `continuous`, `even_page`, `odd_page`, and `new_column`.
Section breaks are main-body only.

### Table

```json
{
  "type": "table",
  "style": "Table Grid",
  "header_rows": 1,
  "rows": [
    ["Item", "Status"],
    [{"runs": [{"text": "A", "bold": true}]}, "Ready"]
  ]
}
```

`rows` must be a non-empty rectangular array; every row must contain the same positive number
of cells. A cell accepts a JSON string or a paragraph-style object with `text`/`runs`, `style`,
and `alignment`. Empty rows, ragged rows, numbers, Booleans, objects outside that schema, and
other value types fail as `bad_input`.

### Inline image

```json
{
  "type": "image",
  "path": "chart.png",
  "width_inches": 5.5,
  "prefix_runs": [{"text": "Figure 1: "}],
  "suffix_runs": [{"text": " Source: internal."}]
}
```

Supported signatures are PNG, JPEG, GIF, TIFF, and BMP. Images are bounded to 32 MiB. Omit
one dimension to retain aspect ratio. Floating images are not created or replaced.

### Field

```json
{"type": "field", "field": "page_number", "prefix": "Page "}
```

```json
{
  "type": "field",
  "field": "toc",
  "instruction": " TOC \\o \"1-3\" \\h \\z \\u "
}
```

Fields are OOXML markup with placeholder results. Saving does not paginate the document or
refresh the TOC. Open and update fields in a layout application.

## Headers and footers

Compact create form applies stories to section 0:

```json
{
  "headers": {
    "default": [{"type": "paragraph", "text": "Confidential"}],
    "first_page": [{"type": "paragraph", "text": "Cover"}]
  },
  "footers": {
    "default": [{"type": "field", "field": "page_number", "prefix": "Page "}]
  }
}
```

Explicit create form accepts an array:

```json
{
  "headers": [
    {
      "section_index": 1,
      "kind": "even_page",
      "blocks": [{"type": "paragraph", "text": "Even page"}]
    }
  ]
}
```

Kinds are `default`, `first_page`, and `even_page`. Setting a story unlinks it from its
previous section. Edit operations default to `mode: "replace"`; use `mode: "append"` to retain
existing story content.

## Replacement policy

`replace_text` searches each selected paragraph independently:

- `scope` is `body`, `headers`, `footers`, or `all`.
- `expected_count` is mandatory and must equal the number of visible, non-overlapping
  matches. A mismatch is `ambiguous_edit`.
- More than one match additionally requires `"replace_all": true`.
- A match wholly within one run keeps that run's formatting.
- A match spanning plain adjacent runs requires `"cross_run_policy": "first_run"`. Replacement
  text uses the first run's formatting; unaffected suffix text remains in its original last
  run.
- A match crossing or occupying a hyperlink, field, drawing, comment range, revision, or
  unsupported markup is rejected. The operation never flattens those boundaries.
- Matches do not cross paragraph boundaries.

Inspect runs first and set a narrow expected count. Do not use broad replacement as a document
normalizer.

## Safety and atomicity

Preflight requires:

- A `.docx` path and ZIP signature.
- Required OOXML package members.
- At most 2,048 ZIP members.
- At most 128 MiB compressed input and 256 MiB expanded content.
- Safe, unique member paths.
- No encrypted ZIP members.
- No macro-enabled content types or VBA project.
- Well-formed XML without DTD/entity declarations.

Direct XML parsing disables DTD loading, entity resolution, huge-tree mode, recovery, and
network access for those parser calls. DOCX external relationships are rejected by default.
Use `--allow-external-relationships` or the strictly Boolean job property
`"allow_external_relationships": true` only after reviewing the package and accepting the
risk. The opt-in permits processing; it does not disable, proxy, or sandbox network access by
LibreOffice or another consuming application, which may access external targets.

Mutations require a distinct source and destination unless `--overwrite` is explicit.
Existing destinations also require `--overwrite`. The tool writes a sibling temporary file,
flushes it, preflights/reopens it, and uses an atomic filesystem replacement only after
validation. A failed mutation does not publish partial output.

Unknown untouched package content is preserved best-effort where `python-docx` retains it.
The summary always states that perfect round-trip fidelity is not guaranteed.

## Exit statuses

| Status | Category | Meaning |
|---:|---|---|
| `0` | success | The summary was written to stdout. |
| `2` | `bad_input` | Missing, malformed, wrong-type, unsafe, or schema-invalid input. |
| `3` | `unsupported_operation` | Known format/feature or requested operation is unsupported. |
| `4` | `missing_dependency` | A required Python package or `soffice` is unavailable. |
| `5` | `ambiguous_edit` | Match count, run policy, or protected-boundary rules failed. |
| `6` | `resource_limit` | Package, expansion, member, image, or generated-PDF bound failed. |
| `7` | `licensing_precondition` | A required redistribution/license condition was unmet. |
| `8` | `validation_failed` | Save, reopen, signature, or post-write verification failed. |
| `9` | `output_conflict` | Destination exists or aliases the source without overwrite. |
| `10` | `external_tool_failed` | LibreOffice timed out or returned invalid conversion output. |

`licensing_precondition` is reserved in schema version 1 for workflows that introduce
redistributed assets. Current create/edit jobs reference user-supplied files and do not
redistribute bundled third-party assets.

See [examples](references/examples.md) for complete jobs and
[limitations](references/limitations.md) before using advanced Word features.
