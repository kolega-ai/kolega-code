# XLSX operations

## Table of contents

- [Runtime and JSON contract](#runtime-and-json-contract)
- [Safety and publication](#safety-and-publication)
- [CLI overview](#cli-overview)
- [Inspect](#inspect)
- [Extract](#extract)
- [Create job](#create-job)
- [Edit job](#edit-job)
- [Clean job](#clean-job)
- [Summarize job](#summarize-job)
- [Convert](#convert)
- [PDF export](#pdf-export)
- [Render](#render)
- [Styles, tables, validation, and charts](#styles-tables-validation-and-charts)
- [Formula policy](#formula-policy)
- [Exit statuses](#exit-statuses)

## Runtime and JSON contract

Resolve the installed skill directory and select any available Python 3.11+ interpreter.
Do not assume a launcher name. The examples use `XLSX_PYTHON` for the selected interpreter:

```bash
SKILL_ROOT="/absolute/path/to/skills/xlsx"
XLSX_PYTHON="/path/to/selected/python"
"$XLSX_PYTHON" -c 'import sys; print(sys.version.split()[0]); raise SystemExit(sys.version_info < (3, 11))'
```

Check the required imports. Before installing anything, tell the user what is missing, what
will be installed, the target scope, and the installer. Use the selected interpreter's
`-m pip` with the requirements file:

```bash
"$XLSX_PYTHON" -m pip install -r "$SKILL_ROOT/requirements.txt"
```

Prefer the active or user interpreter. If platform policy blocks those scopes, explain the
fallback before using a local environment. Use the normal platform package manager for a
missing interpreter, preferring Homebrew on macOS when available. Never use privileged pip
or `--break-system-packages`, and do not edit host-project dependency metadata.

Invoke the public CLI with `"$XLSX_PYTHON" "$SKILL_ROOT/scripts/xlsx_tool.py"`.

Every successful command prints one JSON object to stdout and exits 0:

```json
{
  "schema_version": 1,
  "ok": true,
  "operation": "inspect",
  "result": {},
  "warnings": [],
  "versions": {
    "python": "3.11.4",
    "openpyxl": "3.1.5",
    "pandas": "3.0.3",
    "numpy": "2.4.6",
    "pypdf": "6.1.3"
  }
}
```

PDF export and page rendering additionally use LibreOffice (`soffice` on PATH) and, for
rendering only, the optional `pypdfium2` package. Both are reported as `missing_dependency`
with an install hint when absent; no other operation needs them.

Expected failures print one JSON object to stderr, leave stdout empty, and use a stable nonzero
status:

```json
{
  "schema_version": 1,
  "ok": false,
  "error": {
    "category": "bad_input",
    "message": "Source workbook does not exist.",
    "details": {"path": "missing.xlsx"}
  }
}
```

Create, edit, clean, and summarize jobs are UTF-8 JSON objects with
`"schema_version": 1`. Other versions are rejected. Schema version 1 is closed: unknown
top-level, operation, sheet, source, table, chart, validation, conditional-format, cleanup,
summary, and nested style keys are rejected as `bad_input` rather than ignored.

## Safety and publication

Workbook input must:

- use the `.xlsx` extension and begin with an OOXML ZIP signature;
- contain `[Content_Types].xml` and `xl/workbook.xml`;
- contain no macro content types/parts, encrypted ZIP members, unsafe member paths, DTDs, or
  entity declarations;
- stay within 100 MiB compressed, 10,000 members, 512 MiB total expanded, 128 MiB per member,
  and the high-volume 1,000:1 compression-ratio limit.

Every XML/relationship member is strictly decoded from its BOM/XML encoding, checked for
DTD/entity declarations in decoded text, and parsed before `openpyxl` runs. Invalid, unknown,
or unsupported XML encodings and malformed XML are `bad_input`; DTD/entity declarations are
`unsupported_operation`.

Rectangular operations are limited to 2,000,000 cells. Create/edit JSON row matrices, cell
maps, and range operations share one cumulative per-job budget, including ragged rows.
`create` and `edit` accept `--cell-limit N` to lower (never raise) that cumulative limit for
more restrictive callers and controlled validation.
`inspect --max-cells` defaults to 10,000 and is capped at 100,000 emitted cell records per
selected sheet; a truncated inventory is flagged by `cell_inventory_truncated` and a warning.

Mutation writes a temporary sibling, saves, applies the same package preflight, reopens the
workbook, confirms sheet order, fsyncs, and atomically replaces the destination. The source is
never opened for write. Use `--overwrite` both for an existing destination and for an
explicitly requested same-path replacement.

One-file-per-sheet conversion atomically publishes a new output directory. It refuses an
existing output directory because replacing a populated directory as one atomic unit is not
portable.

## CLI overview

```text
xlsx_tool.py inspect SOURCE [--sheet NAME] [--range A1:D20]
                     [--no-cells] [--max-cells N]
xlsx_tool.py extract SOURCE OUTPUT --format json|csv|tsv
                     [--sheet NAME] [--range A1:D20]
                     [--values formulas|cached] [delimited options]
xlsx_tool.py create JOB OUTPUT [--overwrite]
xlsx_tool.py edit SOURCE JOB OUTPUT [--overwrite]
xlsx_tool.py clean SOURCE JOB OUTPUT [--overwrite]
xlsx_tool.py summarize SOURCE JOB OUTPUT [--overwrite]
xlsx_tool.py convert SOURCE OUTPUT [conversion options] [--overwrite]
xlsx_tool.py convert SOURCE OUTPUT.pdf [--timeout SECONDS] [--overwrite]
xlsx_tool.py render SOURCE OUTPUT_DIR [--dpi N] [--pages 1,3] [--timeout SECONDS]
```

`--debug` is a global option placed before the subcommand. It adds a traceback only to
unexpected internal-error JSON.

## Inspect

`inspect` performs package preflight, opens formula and cached-value workbook views, and emits:

- compressed/expanded byte counts and ZIP member count;
- sheet order, active sheet, visibility, selected/used ranges, merges, dimensions, freeze
  panes, filters, tab color, gridline visibility, zoom, and read-only page setup
  (orientation, paper size, scale, fit-to settings, print area, print titles);
- non-empty, styled, and formula cell counts;
- cell coordinates, values/formulas, cached values, data types, style IDs, and number formats;
- a per-sheet `errors` block counting formula error values (`#REF!`, `#DIV/0!`, `#VALUE!`,
  `#NAME?`, `#N/A`, `#NULL!`, `#NUM!`, `#SPILL!`, `#CALC!`) as literal error cells or cached
  formula results, with capped coordinate samples, plus a workbook `counts.error_cells`
  total for the selected sheets/range;
- a workbook `fonts` inventory (`referenced`, `by_source` for cells/named styles/charts,
  `theme`, always-empty `embedded`, and `unembedded`) built from fonts effectively used by
  non-empty cells and chart text — see [PDF export](#pdf-export) for the portability gate;
- named ranges, workbook properties, calculation settings, and external-link inventory;
- tables, chart type/title/anchor/series count, pivots, conditional formats, and validations;
- warnings for formulas, error values, font portability, external links, pivots, and
  truncated cell inventories.

Without `--sheet`, all sheets are inventoried. `--range` applies to each selected sheet.
`--no-cells` omits cell records while retaining structural counts; the error scan and font
inventory still run. `--max-cells` defaults to 10,000 records per sheet (cap 100,000), so
large sheets truncate the `cells` array — raise it explicitly when full cell inventories
matter. A cached formula error can only be detected when a cached value exists; a workbook
that was never calculated reports formulas, not errors.

## Extract

`extract` writes one selected sheet or A1 range to JSON, CSV, or TSV:

```bash
"$XLSX_PYTHON" "$SKILL_ROOT/scripts/xlsx_tool.py" extract report.xlsx sales.tsv \
  --sheet Sales --range A1:F200 --format tsv --values cached \
  --encoding utf-8 --quoting minimal --na-value ""
```

- `--values formulas` reads formula expressions; `cached` reads stored results without
  calculating them.
- `--delimiter` overrides comma or tab with one character.
- `--quoting` is `minimal`, `all`, `nonnumeric`, or `none`.
- JSON extraction writes a versioned payload containing source, sheet, range, value mode, and
  an exact row matrix. Formula expressions and formula-like strings are never apostrophe
  prefixed in JSON.
- For CSV/TSV only, `--allow-formulas` disables text formula-injection protection.

CSV/TSV output converts `None` to `--na-value`. Dates and times use ISO-8601 JSON
serialization; CSV uses the Python/pandas scalar representation selected by the source value.

## Create job

Create a workbook:

```bash
"$XLSX_PYTHON" "$SKILL_ROOT/scripts/xlsx_tool.py" create create.json report.xlsx
```

Top-level fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | Required integer `1`. |
| `properties` | Title, subject, creator, keywords, description, category, last modifier, and ISO date-times. |
| `named_styles` | Reusable named style objects. |
| `sheets` | Required non-empty ordered sheet array. |
| `defined_names` | Workbook or sheet-local names. |
| `calculation` | `mode`, `full_calc_on_load`, and `force_full_calc` request flags. |

A sheet object requires `name` and accepts:

- `rows`: row arrays containing scalars or cell specification objects;
- `source`: a CSV/TSV source object instead of `rows`;
- `cells`: an A1-coordinate map applied after rows/source;
- `state`: `visible`, `hidden`, or `veryHidden`;
- `merges`, `freeze_panes`, `dimensions`, `auto_filter`, `tab_color`, `show_gridlines`, and
  `zoom_scale`;
- `auto_width`: `true` for every used column, or an object with `columns` (letters),
  `min_width`, `max_width`, and `sample_rows`; widths use a character-count heuristic over
  stored cell text (formula text, not results), run before `dimensions` so explicit widths
  win, and default to bounds 8/60 over 200 sampled rows;
- `tables`, `conditional_formats`, `data_validations`, and `charts`.

At least one sheet must remain visible. Merging a range that covers non-top-left values
keeps only the top-left value; during create this is reported as a warning.

A delimited `source` object accepts:

```json
{
  "path": "input.csv",
  "format": "csv",
  "encoding": "utf-8",
  "delimiter": ",",
  "header": true,
  "leading_zeros": "preserve",
  "recognize_default_na": false,
  "na_values": ["N/A"],
  "malformed_rows": "error",
  "quoting": "minimal",
  "schema": {
    "Account": "string",
    "Amount": {"type": "float", "errors": "raise"}
  }
}
```

Relative paths are resolved from the job file's directory. Schema types are `string`,
`integer`, `float`, `boolean`, `date`, and `datetime`; errors are `raise` or `coerce`.
`header: true` writes the source's first row as worksheet headers. `header: false` writes only
source data and never invents a numeric header row. With `recognize_default_na: false` (the
default), empty/default pandas NA spellings remain literal unless listed in `na_values`;
setting it to `true` enables pandas' default NA spellings. Use `na_values` for explicit custom
markers.

A cell specification accepts one of `value` or `formula`, plus `style`/`named_style`, `font`,
`fill`, `border`, `alignment`, `protection`, `number_format`, and an `http`, `https`, or
`mailto` hyperlink. `formula` must begin with `=`. A `value` beginning with `=` is stored as a
literal string unless `allow_formula` is explicitly true.

Defined names use:

```json
{"name": "Revenue", "refers_to": "'Sales'!$C$2:$C$100", "local_sheet": "Sales"}
```

## Edit job

Apply ordered operations:

```bash
"$XLSX_PYTHON" "$SKILL_ROOT/scripts/xlsx_tool.py" edit source.xlsx edit.json result.xlsx
```

```json
{
  "schema_version": 1,
  "operations": [
    {"op": "set_cell", "sheet": "Sales", "cell": "C2", "value": 125.5},
    {"op": "set_cell", "sheet": "Sales", "cell": "D2", "formula": "=C2*0.2"},
    {
      "op": "format_range",
      "sheet": "Sales",
      "range": "C2:D100",
      "style": {"number_format": "$#,##0.00"}
    }
  ]
}
```

Supported operations:

| Operation | Required/important fields |
| --- | --- |
| `set_cell` | `sheet`, `cell`, and `spec`, `formula`, or `value`; optional `allow_formula`. |
| `set_range` | `sheet`, exact `range`, and same-sized row matrix `values`. |
| `format_range` | `sheet`, `range`, `style`. |
| `add_sheet` | `name`, optional `index`, and create-sheet fields. |
| `remove_sheet` | `sheet`; optional `allow_unupdated_dependencies`. |
| `rename_sheet` | `sheet`, `new_name`; optional `allow_unupdated_dependencies`. |
| `set_sheet` | `sheet`; optional state, freeze panes, filter, dimensions, `auto_width`, `merges` (with `allow_merge_data_loss`), `unmerge`, `tab_color`, `show_gridlines`, `zoom_scale`. |
| `add_table` | `sheet`, `table`. |
| `update_table` | `table` containing current `name` and optional new range/name/style. |
| `add_chart` | `sheet`, `chart`. |
| `update_chart` | `sheet`, zero-based `index`, replacement `chart`. |
| `add_conditional_format` | `sheet`, `rule`. |
| `add_data_validation` | `sheet`, `validation`. |
| `set_properties` | `properties`. |
| `add_defined_name` | `defined_name`. |

Insert/delete/move row, column, and range operation names are explicitly refused. Sheet
rename/removal inspects formula, defined-name, and chart dependencies and rejects uncertainty
unless `allow_unupdated_dependencies` accepts unchanged/dangling references. Sheet-reference
matching is case-insensitive, like Excel sheet names.

`set_sheet` merges are guarded: merging a range whose non-top-left cells hold values is
`ambiguous_edit` unless the operation sets `"allow_merge_data_loss": true`, because the merge
discards those values. Each `unmerge` entry must exactly match one currently merged range
(the failure lists the current merges). Merge data-loss scanning is bounded at 100,000 cells
per range.

Every edit warns that rewriting through `openpyxl` can alter or drop unsupported OOXML
extension content. If external-link parts are present, edit also warns that targets, cached
values, and preservation are not repaired or guaranteed.

## Clean job

`clean` reads a rectangular range through the cached-value workbook view, applies only declared
pandas policies, and writes a separate destination sheet in a semantic copy of the source
workbook:

```json
{
  "schema_version": 1,
  "source": {"sheet": "Raw", "range": "A1:F500", "header_row": 1},
  "destination_sheet": "Cleaned",
  "replace_destination": false,
  "policies": {
    "whitespace": {
      "columns": ["Customer"],
      "strip": true,
      "collapse": true
    },
    "missing": {
      "markers": ["N/A", "-"],
      "fill": {"Country": "Unknown"},
      "required_columns": ["Customer"]
    },
    "malformed_rows": {
      "required_columns": ["Customer", "Amount"],
      "max_missing_fraction": 0.5,
      "action": "drop"
    },
    "coercion": {
      "Amount": {"type": "float", "errors": "coerce"},
      "Date": {"type": "date", "errors": "raise"}
    },
    "rename": {"Customer": "Customer Name"},
    "duplicates": {
      "subset": ["Customer Name", "Date"],
      "keep": "first"
    },
    "sort": [{"column": "Date", "ascending": true}],
    "na_position": "last"
  },
  "table": {"name": "CleanedData"}
}
```

`missing.drop_rows` can be `any` or `all`. `duplicates.keep` can be `first`, `last`, or
`none`. The source sheet cannot be the destination. Replacing another destination sheet
requires `replace_destination: true`.

## Summarize job

`summarize` writes a static grouped/pivot-style pandas result, not a native pivot:

```json
{
  "schema_version": 1,
  "source": {"sheet": "Sales", "range": "A1:D1000", "header_row": 1},
  "destination_sheet": "Static Summary",
  "group_by": ["Region"],
  "columns": ["Quarter"],
  "values": [
    {"column": "Revenue", "aggregation": "sum", "name": "Revenue total"},
    {"column": "Units", "aggregation": "mean", "name": "Average units"}
  ],
  "fill_value": 0,
  "dropna": true,
  "sort": true,
  "observed": true,
  "chart": {"type": "bar", "title": "Regional summary"}
}
```

Aggregations are `sum`, `mean`, `median`, `min`, `max`, `count`, `nunique`, `first`, and
`last`. The output label in row 1 states that the summary is static. Data starts at row 3.
An optional native chart references the static result.

## Convert

Supported directions are CSV/TSV to XLSX, XLSX to CSV/TSV, and XLSX to PDF (see
[PDF export](#pdf-export)):

```bash
# Preserve account codes as strings and coerce declared amounts.
"$XLSX_PYTHON" "$SKILL_ROOT/scripts/xlsx_tool.py" convert input.csv output.xlsx \
  --leading-zeros preserve --schema schema.json

# Export one cached-value sheet.
"$XLSX_PYTHON" "$SKILL_ROOT/scripts/xlsx_tool.py" convert report.xlsx sales.tsv \
  --output-format tsv --sheet Sales --values cached \
  --encoding utf-8 --quoting minimal --na-value ""

# Export each raw sheet to a new directory as 001-Sheet.csv, 002-Other.csv, ...
"$XLSX_PYTHON" "$SKILL_ROOT/scripts/xlsx_tool.py" convert report.xlsx exported-sheets \
  --all-sheets --output-format csv
```

Options:

- `--input-format`/`--output-format`: `auto`, `xlsx`, `csv`, or `tsv`;
- `--sheet`, `--range`, `--header-row`, or `--all-sheets` for XLSX input/export;
- `--sheet-policy raw|header` for XLSX input/export;
- `--values formulas|cached`;
- `--schema` containing a JSON column-to-type map;
- `--leading-zeros preserve|numeric`;
- `--encoding`, `--delimiter`, `--quoting`, `--na-value`, and `--date-format`;
- `--index` for delimited output;
- `--no-header` and `--malformed-rows error|warn|skip` for delimited input;
- `--input-na-policy literal|default|custom` and repeatable `--input-na-value` for delimited
  input; `--na-value` is the CSV/TSV output representation only;
- `--allow-formulas` to opt into executable formulas.

Direction-inapplicable non-default options are rejected as `bad_input`, rather than silently
ignored. In particular, schema, leading-zero, input-header/NA, and malformed-row controls are
for CSV/TSV input; range, header-row, sheet-policy, all-sheet, cached/formula value, index, and
output NA/date controls are for XLSX input/export.

The default XLSX export policy is `raw`: emit each selected sheet's rectangular used range
exactly as rows, with no header interpretation, pandas column construction, or invented
headers. A truly blank sheet produces an empty file. This policy supports static-summary
title/padding rows and sheets whose first row contains duplicate or empty values. It includes
visible, hidden, and very-hidden sheets in workbook order. All-sheet filenames are
deterministic: a one-based, zero-padded sheet index plus an ASCII-safe sheet-name stem, such as
`001-Sales.csv`; the JSON result maps every original sheet name to its file, so sanitized-name
collisions remain unambiguous.

Select `--sheet-policy header` only for a conventional table with one non-empty, unique header
row. That mode enables `--header-row` and `--index`; these options are rejected in raw mode.
Header-aware output includes the selected header row. `--schema` applies only when importing
CSV/TSV into XLSX.

For CSV/TSV input, `--no-header` writes the first source row to XLSX row 1 and does not create,
style, freeze, or filter a synthetic header. `--input-na-policy literal` preserves default NA
spellings, `default` enables pandas' defaults, and `custom` requires one or more
`--input-na-value` markers.

When importing delimited data, potentially executable strings are stored as literal XLSX text
by default. With `--allow-formulas`, values beginning with `=` become formulas. Formula
injection apostrophe-prefixing is applied only to spreadsheet-executable CSV/TSV output,
including raw rows and header-aware column labels. It is never applied to JSON extraction.
Leading spaces, tabs, CR/LF, BOM, and non-breaking spaces are considered before `=`, `+`, `-`,
or `@`; leading tab/CR/LF/BOM is itself guarded.

## PDF export

`convert SOURCE OUTPUT.pdf` exports the whole workbook to PDF through headless LibreOffice
with an isolated profile, converting an immutable snapshot copy of the preflighted source:

```bash
"$XLSX_PYTHON" "$SKILL_ROOT/scripts/xlsx_tool.py" convert report.xlsx report.pdf --timeout 120
```

- Only `--timeout` (seconds, 1–1800, default 120) and `--overwrite` apply; every delimited or
  sheet-selection option is rejected as `bad_input`. Sheet-level control uses the workbook
  itself: hidden sheets are excluded from the PDF, and print areas/page setup bound what each
  sheet prints.
- Workbooks with external links are refused (`unsupported_operation`) because LibreOffice may
  try to resolve them.
- The output is verified (`%PDF-` signature, pypdf reopen, at least one page) and atomically
  published behind a source-hash gate. The spreadsheet page count is not predictable from the
  sheet count; the result reports `counts.pdf_pages`.
- The result embeds the `fonts` inventory and its portability warnings. Fonts outside the
  conservative cross-renderer set (Arial, Times New Roman, Courier New) raise a
  `RELEASE BLOCKER` warning: XLSX cannot embed fonts, so any renderer without them
  substitutes metrics and can change column fit, wrapping, and pagination.
- Values in the PDF come from LibreOffice's load-time calculation; pagination follows
  LibreOffice print semantics (print areas, fit settings, printed gridlines are off by
  default) and is not proven identical to Excel's. These caveats are emitted as warnings.

## Render

`render SOURCE OUTPUT_DIR` converts the workbook to PDF the same way, then rasterizes pages
to PNG with PDFium (`pypdfium2`):

```bash
"$XLSX_PYTHON" "$SKILL_ROOT/scripts/xlsx_tool.py" render report.xlsx review-pages --dpi 96
```

- `--dpi` accepts 36–300 (default 96). `--pages` selects 1-based PDF page numbers
  (`--pages 1,3`); page numbers are validated against the actual converted page count.
  `--timeout` bounds the LibreOffice step.
- The destination directory must not exist; pages are staged and published atomically as
  `page-001.png`, `page-002.png`, … with an `images` array reporting per-page dimensions,
  byte counts, and SHA-256 hashes. Pages may legitimately differ in size when sheets mix
  orientation or paper settings.
- Limits: at most 200 pages per invocation (select `--pages`, set print areas, or hide
  sheets), 25,000,000 pixels per page, and 250,000,000 pixels per run.
- Rendering exists to be looked at. Open the published PNG files and review layout — column
  overflow (`#####`), clipped or wrapped text, merge layout, chart appearance, conditional
  formatting — before claiming visual correctness; structural JSON cannot show any of that.
- The same external-link refusal, snapshot, hash-gate, formula-calculation, and
  print-semantics behavior as PDF export applies.

## Styles, tables, validation, and charts

Style objects support:

- `font`: name, size, bold, italic, underline, strike, vertical alignment, color, and related
  public font flags;
- `fill`: type, foreground/color, and background color;
- `border`: left/right/top/bottom/diagonal/vertical/horizontal sides with style and color;
- `alignment`: horizontal, vertical, rotation, wrapping, shrink-to-fit, indent, and reading
  order;
- `protection`: locked and hidden;
- `number_format`: Excel number-format code.

Colors are six- or eight-digit hexadecimal RGB/ARGB strings.

Tables require a workbook-unique valid `name`, bounded rectangular `range`, non-empty unique
headers, and at least one data row. Add and update use the same name, range, header, and style
validation before mutating a table. Table style fields are `name`, `show_first_column`,
`show_last_column`, `show_row_stripes`, and `show_column_stripes`. Adding a table clears a
sheet-level auto filter whose range overlaps the table: Excel treats that combination as
invalid content, and the table provides its own filter.

Conditional-format types:

- `cell_is`: operator, formula list, optional differential font/fill/border;
- `formula`: formula list and optional differential style;
- `color_scale`: two or three colors plus optional value types/values;
- `data_bar`: bounds, color, and value display;
- `icon_set`: style, threshold type/values, reverse, and value display.

Data validation exposes type, operator, one or two formulas, blank/dropdown/message controls,
and input/error titles/messages.

Native chart types are `bar`, `line`, `pie`, `area`, and `scatter`. Standard charts use `data`
and optional `categories` A1 ranges, including `Sheet!A1:B4` references. Scatter charts use one
`x_values` range and an array of same-sized one-column `y_values` ranges. Common fields include
title, style, anchor, dimensions, axis titles, and grouping.

## Formula policy

Formula syntax is preserved as text beginning with `=`. It is never evaluated by this tool.
Use a cell `formula` property for intended formulas. A scalar/`value` string beginning with `=`
is forced to XLSX string type.

Formula caches reported by `inspect`, cached `extract`, `clean`, `summarize`, and cached
`convert` may be stale or absent. Warnings are part of the successful JSON result.

The production CLI does not impose an overall wall-clock timeout; only the LibreOffice step
of PDF export and rendering honors `--timeout`. Callers should still enforce a subprocess
deadline appropriate to their workload and treat a timeout as an aborted operation. Graceful
Python failures clean temporary outputs; an operating-system hard kill cannot guarantee
cleanup.

## Exit statuses

| Status | Category | Meaning |
| ---: | --- | --- |
| 0 | success | Verified operation completed. |
| 2 | `bad_input` | Missing, malformed, invalid encoding/name/style/value, unknown schema key, wrong extension/range, or conflicting flags. |
| 3 | `unsupported_operation` | Macro/encryption, unsupported format/operation, or atomic-directory constraint. |
| 4 | `missing_dependency` | Required Python packages, LibreOffice (PDF/render), or pypdfium2 (render) are not installed. |
| 5 | `ambiguous_edit` | Unsafe dependency, duplicate name, destructive merge, or unresolved replacement choice. |
| 6 | `resource_limit` | Package/member/cell/inspection/merge-scan/render bound exceeded. |
| 7 | `licensing_precondition` | Reserved common-contract status; no XLSX core operation currently uses it. |
| 8 | `post_write_validation` | Save, reopen, verification, fsync, or atomic publication failed. |
| 9 | `internal_error` | Unexpected implementation/runtime failure or interruption. |
| 10 | `external_tool_failure` | LibreOffice or PDFium launch, conversion, timeout, or output validation failed. |
