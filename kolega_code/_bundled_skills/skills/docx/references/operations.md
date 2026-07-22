# Operations contract

## Contents

- [Environment](#environment)
- [Invocation and envelopes](#invocation-and-envelopes)
- [Inspect schema](#inspect-schema)
- [Create schema](#create-schema)
- [Edit operations](#edit-operations)
- [Conversion contract](#conversion-contract)
- [Render](#render)
- [Content blocks](#content-blocks)
- [Sections and styles](#sections-and-styles)
- [Headers and footers](#headers-and-footers)
- [Warnings](#warnings)
- [Safety, atomicity, and exits](#safety-atomicity-and-exits)

## Environment

Resolve the skill root and select any Python 3.11+ interpreter; examples use:

```bash
SKILL_ROOT="/absolute/path/to/skills/docx"
DOCX_PYTHON="/path/to/python"
"$DOCX_PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,11) else "Python 3.11+ required")'
"$DOCX_PYTHON" -c 'import docx,lxml,pypdf'
```

Before installing a missing dependency, tell the user what will be installed, where, and by
which installer. Then use `"$DOCX_PYTHON" -m pip install -r
"$SKILL_ROOT/requirements.txt"`. Never use privileged pip, `--break-system-packages`, or host
project dependency metadata. LibreOffice/`soffice` is optional and required only for PDF
conversion and page rendering; install it through the normal platform package manager only
after notice. The optional `pypdfium2` package is required only for `render`.

LibreOffice discovery order: the `DOCX_SOFFICE` environment variable (must point to an
executable; a bad value fails rather than falling through), then `soffice`/`libreoffice` on
PATH, then the standard macOS app-bundle locations under `/Applications` and
`~/Applications`.

## Invocation and envelopes

```text
docx_tool.py inspect --input SOURCE.docx [--allow-external-relationships]
docx_tool.py create --spec CREATE.json --output DEST.docx [--template BASE.docx] [--allow-external-relationships] [--overwrite]
docx_tool.py edit --input SOURCE.docx --spec EDIT.json --output DEST.docx [--allow-external-relationships] [--overwrite]
docx_tool.py convert --input SOURCE.docx --format text|pdf --output DEST [--timeout 90] [--allow-external-relationships] [--overwrite]
docx_tool.py render --input SOURCE.docx --output DEST_DIR [--dpi 150] [--pages 1,3] [--timeout 90] [--allow-external-relationships]
docx_tool.py --job JOB.json
```

Do not combine `--job` with a subcommand. JSON-relative paths resolve from the JSON file;
flag paths resolve from the current directory. Every input has `"schema_version": 1`.
Operational schemas are strict: unknown keys, wrong JSON types, and unknown operation/block
types are `bad_input`. Booleans are JSON Booleans; text is never coerced.

Direct create accepts only `schema_version` and `content`. Direct edit accepts only
`schema_version` and `operations`. A job adds `operation` plus operation-specific paths and
options. Common job options are `overwrite` and `allow_external_relationships`, both Boolean.

Success emits one JSON object on stdout with `schema_version`, `status`, `operation`, `counts`,
`warnings`, `versions`, `paths`, `verification`, `preservation`, and `result`. Warnings are
also emitted as JSON lines on stderr. Failure leaves stdout empty and emits
`{"schema_version":1,"status":"error","error":{"category":...,"message":...,"details":...}}`.

## Inspect schema

```bash
"$DOCX_PYTHON" "$SKILL_ROOT/scripts/docx_tool.py" inspect --input report.docx
```

`result` contains:

- `ordered_blocks`: body paragraphs/tables with `block_index`, `type`, and nested `value`;
  plus top-level `paragraphs` and `tables`.
- Paragraph records: `index`, `text`, `style`, `heading_level`, list data, `alignment`, runs,
  and `layout`. Layout reports spacing, line spacing/rule, left/right/first-line indents,
  pagination flags, and tab stops. A hanging indent is reported as a negative
  `first_line_indent_inches`.
- Table records: `index`, `section_index`, style, effective `layout`,
  `column_widths_inches`, `row_allows_split`, `header_rows`, dimensions, and nested cell
  paragraphs.
- `sections`: start type, orientation, dimensions, detected `paper_size`, usable width,
  margins/distances, first-page setting, and page-number start/format.
- `headers`/`footers`: every section and `default`, `first_page`, `even_page` story with part,
  `linked_to_previous`, paragraphs, and tables.
- `metadata`: supported core properties; `styles`: paragraph/character style definitions;
  `settings`: `odd_and_even_pages` and `update_fields_on_open`.
- `fields`: simple and nested complex fields across body/header/footer XML, each with
  `story_address`, inferred `type`, full `instruction`, `dirty`, `locked`, and visible
  `result`. Nested complex-field result text is associated with each active containing field.
- `images.media_parts` and `images.inline_occurrences`: package identity, story/section,
  dimensions, pixels, effective X/Y PPI, upscaling, alt text, title, and decorative state.
- `fonts`: `referenced`, `embedded`, `unembedded`, per-part `references`,
  `theme_tokens_referenced`, and `dangling_embedding_relationships`.
- feature counts, unsupported parts, and package member/expanded-byte counts.

Inspection is direct, not computed layout. Style records expose direct definition values,
`based_on`, font values, outline level, and limited paragraph values; they do not resolve the
fully inherited/effective cascade. A successful inspect proves preflight and reopen, not
visual fidelity.

## Create schema

`content` accepts only:

| Key | Contract |
|---|---|
| `template`, `letterhead` | Optional DOCX base path; mutually exclusive. `--template` overrides both. The base is preflighted and unchanged. |
| `metadata` | Core properties: `title`, `subject`, `author`, `keywords`, `comments`, `category`, `last_modified_by`, `revision` (integer >= 1), `identifier`, `language`, `version`. Other values are strings. |
| `section` | Initial-section schema in [Sections](#section-schema). |
| `styles` | Style-definition array in [Styles](#style-definitions). Applied before blocks. |
| `update_fields_on_open` | Boolean; writes the Word setting. It does not refresh fields itself. |
| `blocks` | Ordered content blocks. Default `[]`. |
| `headers`, `footers` | Compact or section-explicit story configuration. |

Without a template, creation starts from `python-docx`'s standard document. For polished
unbranded work, explicitly define `section` and `styles`. With a placeholder template, inspect
and use `edit` for bounded placeholder replacement/insertion instead of creating by append.

Order of application is initial section, styles, field-update setting, body blocks, metadata,
headers, and footers. Section-break blocks may configure each added section.

## Edit operations

Operations execute sequentially in memory; indices address the current state. Any failure
discards the mutation.

- **`replace_text`** — requires `find` (non-empty), `replace` (string), and
  `expected_count >= 0`; optional `replace_all`, `scope` (`body`, `headers`, `footers`, `all`),
  and `cross_run_policy` (only `first_run`). Scope defaults to `body`; `replace_all` defaults
  to `false`. Matches are paragraph-local. Multiple matches require `replace_all: true`; count
  mismatch is `ambiguous_edit`. Protected hyperlink, field, drawing, comment, revision, and
  unsupported boundaries are not flattened.
- **`insert_paragraph`** — requires paragraph-like `block` and `position`. `append` forbids an
  index; `before`/`after` require top-level `paragraph_index`. Allowed block types are
  paragraph, heading, page break, image, and field.
- **`delete_paragraph`** — requires top-level body `paragraph_index`.
- **`insert_table`** — requires `table`; without an index it appends. With
  `paragraph_index`, optional `position` is `before`/`after` and defaults to `after`.
  Accepts full table parity: style, headers, layout, widths, and split controls.
- **`update_table`** — requires `table_index`. Optionally update one cell by supplying all of
  `row`, `column`, `value`; and/or update `style`, `header_rows`, `layout`,
  `column_widths_inches`, `allow_row_split`, and `row_allow_split`. Width count must equal
  columns; row-split count must equal rows. Property updates preserve cell content.
- **`insert_image`** — requires `path`; optional body `paragraph_index`, dimensions,
  `alt_text`, `decorative`, and `title`. Without an index it appends a paragraph. The strict
  schema also accepts `story_address`, but the current operation does not use it; omit it.
  Images are inline.
- **`replace_image`** — requires `path` and either body `inline_image_index` or an inspected
  `story_address` of form `word/document.xml#inline-N`; optional dimensions and accessibility
  metadata. If accessibility keys are omitted, existing values remain. Header/footer
  story-address replacement is unsupported.
- **`set_metadata`** — requires `metadata` using the create metadata schema.
- **`set_header` / `set_footer`** — optional `section_index` default 0; `kind` default
  `default`; `mode` default `replace`; optional `blocks` and `link_to_previous`. If linking
  `true`, do not supply blocks. Details are in [Headers and footers](#headers-and-footers).
- **`configure_section`** — optional `section_index` default 0 plus any section-schema keys.
- **`upsert_style`** — requires one `style` object from [Styles](#style-definitions). Existing
  type must match; `based_on` must already exist.
- **`set_update_fields_on_open`** — requires Boolean `enabled`.

Top-level paragraph/table indices exclude nested table and story content. Inspect again after
edits; specifically compare `row_allows_split`, widths, style records, story linkage, field
settings, and image accessibility/PPI.

## Conversion contract

```bash
"$DOCX_PYTHON" "$SKILL_ROOT/scripts/docx_tool.py" convert \
  --input source.docx --format text --output source.txt
"$DOCX_PYTHON" "$SKILL_ROOT/scripts/docx_tool.py" convert \
  --input source.docx --format pdf --output source.pdf --timeout 90
```

Text conversion uses ordered body content, tab-delimits table cells, and appends labeled,
part-aware header/footer stories.

PDF conversion preflights, runs headless LibreOffice with an isolated temporary profile/home
and process group, bounds diagnostics/timeout, requires one PDF signature, and atomically
publishes only after validation. Validation limits are 64 MiB, 1–1,000 pages, 8 MiB per
decompressed stream, text extraction from at most 50 pages and 1,000,000 characters.

`verification.content_quality_report` contains `deterministic`, method, pypdf version,
limitation, per-page dimensions/text count/`low_text`/`has_nontext_content`/`nearly_blank` and
start/end anchors, `nearly_blank_pages`, and `raster_review_artifacts` (empty for `convert`;
`render` fills it with the published PNG file names). XObject detection is incomplete. This
report is structural and does not render pixels or replace the
[visual quality review](references/quality.md#visual-review-versus-structural-checks).

## Render

```bash
"$DOCX_PYTHON" "$SKILL_ROOT/scripts/docx_tool.py" render \
  --input source.docx --output pages-dir --dpi 150 --pages 1,2
```

Render converts the document to PDF through the same LibreOffice contract and limits as
`convert`, then rasterizes the selected PDF pages to one PNG each with the optional
`pypdfium2` package. Output naming is `page-001.png` upward, using 1-based PDF page numbers.
Because DOCX has no fixed pagination, the page count is known only after conversion; `--pages`
is validated against the converted PDF and a miss reports the real `pdf_pages`. Pages may
differ in dimensions when sections mix orientation or paper size.

The destination is a directory that must not already exist; pages are staged in a sibling
temporary directory and published atomically only after every PNG passes signature, reopen,
and pixel-budget verification, and the published files are reopened and hash-checked again.
There is no `--overwrite` for render.

Limits: `--dpi` 36–300 (default 150), at most 200 pages per invocation, 25,000,000 pixels per
page, and 250,000,000 pixels per run, pre-charged from page geometry before rasterizing.
`--timeout` bounds the LibreOffice step (default 90 seconds). LibreOffice or PDFium failures
exit 10; pixel budgets exit 6; an existing destination exits 9.

The result carries the source inspection warnings (including the font-portability tiers),
`layout_engine_variance`, `libreoffice_diagnostics` when LibreOffice wrote to stderr, and
`render_review_required`. Rendering exists to be looked at: open the published PNG pages and
review layout before claiming visual correctness. The PNGs show LibreOffice's interpretation
of the document, not Microsoft Word's, and fields are not refreshed before rendering.

## Content blocks

### Paragraph-like blocks

Paragraph and heading (`level` 0–9; default 1) accept `text` or `runs`, never both; optional
`style`, `alignment`, and all paragraph-layout keys. A run requires string `text` and may set
Boolean `bold`/`italic`/`underline`, `font`, `size_pt >= 1`, six-digit RGB `color`, and
character `style`. Heading default style is `Title` for level 0, otherwise `Heading N`.

Paragraph-layout keys:

- `space_before_pt`, `space_after_pt` >= 0
- `line_spacing`: number >= 0.1 or `single`, `one_and_half`, `double`
- `left_indent_inches`, `right_indent_inches`, `hanging_indent_inches` >= 0
- `first_line_indent_inches` (may be negative); mutually exclusive with hanging indent
- Boolean `keep_with_next`, `keep_together`, `page_break_before`, `widow_control`
- `tab_stops`: array of `{position_inches, alignment?, leader?}`; position >= 0;
  alignment defaults `left` and permits `left`, `center`, `right`, `decimal`, `bar`; leader
  defaults `spaces` and permits `spaces`, `dots`, `dashes`, `lines`, `heavy`, `middle_dot`.
  Supplying the array clears existing direct tab stops.

`page_break` accepts only `type`. `list` requires `items` and optional Boolean `ordered`;
items are strings or objects with `text`/`runs` and alignment. Lists use `List Bullet` or
`List Number`.

### Tables

A table requires non-empty rectangular `rows`. Each cell is a string or a paragraph object
with `text`/`runs`, `style`, `alignment`, and paragraph-layout keys. Optional keys:

- `style`; `header_rows` from 0 through row count (default 0).
- `layout`: `autofit` or `fixed`; default is fixed when widths are supplied, autofit otherwise.
- `column_widths_inches`: one value >= 0.1 per column.
- `allow_row_split`: Boolean global default, default `true`.
- `row_allow_split`: one Boolean per row; takes precedence over `allow_row_split`.

Inspection reports widths and row behavior. For printable-width arithmetic and release rules,
use [the quality contract](references/quality.md#acceptance-criteria).

### Images and figures

Image block requires `path`; optional dimensions >= 0.01 inch, paragraph style/alignment,
formatted `prefix_runs`/`suffix_runs`, `alt_text`, Boolean `decorative`, `title`, `caption`,
`caption_style`, and `attribution`. Decorative and non-empty alt text conflict. Default
caption style is `Caption`; caption/attribution creates a following paragraph and applies
keep controls. Supported signatures are PNG, JPEG, GIF, TIFF, and BMP; maximum 32 MiB. Omit
one dimension to preserve aspect ratio. Only inline images are created.

### Fields

Field block requires `field`: `page_number`, `toc`, `num_pages`, `section_pages`, `seq`,
`ref`, or `date`. All accept optional `prefix`, `suffix`, `placeholder`, style/alignment, and
Boolean `locked`. `instruction` is accepted only for `toc`, `seq`, `ref`, and `date`; it must
start with its matching field token. Every custom instruction rejects newline/NUL and lengths
above 512 characters.

Default instructions/placeholders are PAGE/`1`; TOC `TOC \o "1-3" \h \z \u`/update prompt;
NUMPAGES/`1`; SECTIONPAGES/`1`; `SEQ Figure \* ARABIC`/`1`; `REF Bookmark`/missing-reference
message; and `DATE \@ "MMMM d, yyyy"`/`Date`. Fields are emitted dirty. They are not refreshed.

### Section breaks

Main-body-only `section_break` accepts `start` (`new_page` default, `continuous`,
`even_page`, `odd_page`, `new_column`) plus the full section schema.

## Sections and styles

### Section schema

Accepted keys are:

- `paper_size`: `letter` (8.5 × 11) or `a4` (approximately 8.2677 × 11.6929);
  `page_width_inches` and `page_height_inches` allow custom dimensions of at least 0.1 inch.
- `orientation`: `portrait` or `landscape`. Applying orientation swaps current width/height
  when needed. Paper size without explicit orientation retains an existing landscape state.
- Nonnegative `top_margin_inches`, `right_margin_inches`, `bottom_margin_inches`,
  `left_margin_inches`, `header_distance_inches`, `footer_distance_inches`.
  Horizontal and vertical margin pairs must each leave positive printable space.
- Boolean `different_first_page`.
- `page_number_start`: integer >= 0; `page_number_format`: `decimal`, `upperRoman`,
  `lowerRoman`, `upperLetter`, or `lowerLetter`.

The create-level `section` configures section 0. A section-break configures the new section.
`configure_section` edits one existing section (default index 0).

### Style definitions

Each `styles` entry or `upsert_style.style` requires `name` and `type` (`paragraph` or
`character`). Optional `based_on` must name an existing style. Font properties are `font`,
`size_pt >= 1`, Boolean `bold`/`italic`/`underline`, and six-digit RGB `color`.

Paragraph styles additionally accept `paragraph` containing alignment and any
paragraph-layout keys, plus `outline_level` 0–9. Character styles reject those paragraph-only
properties. Upsert changes only supplied direct properties; inspection does not calculate
fully inherited/effective values.

## Headers and footers

Create compact form is an object keyed by `default`, `first_page`, and `even_page`, each a
block array, and applies to section 0. Explicit form is an array of
`{section_index, kind, blocks}`. Story blocks allow all blocks except section breaks. Writing
a create story unlinks it from the previous section; first/even kinds enable their document
settings.

Edit defaults are section 0, default story, and replace mode. `mode: "append"` retains story
content. `link_to_previous: true` links and returns without writing; `false` explicitly
unlinks and may be combined with blocks. Inspect the resulting `linked_to_previous` values.

## Warnings

Inspection may emit preflight/external-relationship warnings and these public codes:

- Fidelity/features: `fields_present`, `revisions_present`, `comments_present`,
  `floating_drawings_present`, `unsupported_drawings_present`, `unsupported_parts`.
- Fonts: `unembedded_fonts`, `common_unembedded_fonts` (Office-bundled families such as
  Calibri, Cambria, Aptos, or Segoe UI; substitution possible but usually metric-compatible),
  `nonportable_unembedded_fonts` (`RELEASE BLOCKER`; fonts outside the portable core and
  common Office sets), `dangling_font_embedding_relationships`. Treat font-name allowlists in
  warning wording as a heuristic only: portability still depends on embedding/license or
  availability on every intended renderer.
- Accessibility/structure: `missing_document_title`, `missing_document_language`,
  `heading_level_skip`, `data_table_without_header_row`,
  `informative_image_missing_alt_text`.
- Fit/media: `table_exceeds_printable_width`, `low_resolution_image` (below 150 PPI),
  `image_printable_width_uncertain`, `image_exceeds_printable_width`.
- Fields: `toc_placeholder_only`.

Warnings are diagnostics, not a complete accessibility, typography, or visual audit. Apply
the requested profile in [quality](references/quality.md#delivery-profiles).

## Safety, atomicity, and exits

Preflight requires `.docx` ZIP/OOXML signatures and required members; at most 2,048 members,
128 MiB compressed and 256 MiB expanded; safe unique paths; no encryption, macros/VBA,
DTD/entities, or malformed XML. External relationships are rejected unless explicitly
allowed. The opt-in is not network isolation.

Mutations require a distinct destination unless `--overwrite`; existing destinations also
require it. A sibling temporary file is flushed, preflighted, reopened, and atomically
replaced. Unknown untouched content is preserved best-effort; signatures are not preserved as
valid and signed document mutation is not supported.

| Exit | Category |
|---:|---|
| 0 | success |
| 2 | `bad_input` |
| 3 | `unsupported_operation` |
| 4 | `missing_dependency` |
| 5 | `ambiguous_edit` |
| 6 | `resource_limit` |
| 7 | `licensing_precondition` (reserved) |
| 8 | `validation_failed` |
| 9 | `output_conflict` |
| 10 | `external_tool_failed` |

See [examples](references/examples.md#examples) for executable jobs and
[remaining limits](references/limitations.md#remaining-limits) before fidelity-sensitive work.
