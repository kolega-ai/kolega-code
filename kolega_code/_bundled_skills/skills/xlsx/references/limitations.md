# XLSX limitations

## Table of contents

- [Formula calculation](#formula-calculation)
- [Macros, encryption, and signatures](#macros-encryption-and-signatures)
- [Structural edits and references](#structural-edits-and-references)
- [Pivot tables and advanced workbook content](#pivot-tables-and-advanced-workbook-content)
- [Round-trip fidelity](#round-trip-fidelity)
- [Delimited and pandas transformations](#delimited-and-pandas-transformations)
- [Rendering and conversion](#rendering-and-conversion)
- [Resource limits](#resource-limits)
- [Future work](#future-work)

## Formula calculation

`openpyxl` reads and writes formula expressions but does not evaluate them. Cached values can
be stale or absent. The tool can set `fullCalcOnLoad` and `forceFullCalc`; these flags request
recalculation in a compatible spreadsheet application and do not prove recalculation occurred.
The tool does not implement Excel's calculation engine, dynamic-array semantics, data tables,
or application-specific functions.

Loading cached values for `clean`, `summarize`, or cached extraction can therefore produce
missing or outdated input. Recalculate the source in a trusted calculation application first
when current values are required.

## Macros, encryption, and signatures

The tool refuses `.xlsm`, `.xltm`, `.xla`, `.xlam`, macro-bearing OOXML packages, and encrypted
ZIP members. It does not preserve, inspect, remove, or execute VBA; handle ActiveX controls;
validate digital signatures; or make security claims about macro-bearing workbooks.

Workbook- and worksheet-level Excel protection is not cryptographic file encryption. Creating
cell protection styles does not secure a workbook.

## Structural edits and references

Row insertion/deletion, column insertion/deletion, range moves, and row/column moves are
refused. Updating every dependent formula, structured table reference, chart formula, defined
name, validation, conditional format, external reference, and unsupported extension cannot be
guaranteed.

Sheet rename/removal is rejected when detectable dependencies exist unless
`allow_unupdated_dependencies` is explicitly true. That acknowledgement can leave references
unchanged or dangling. Formula, defined-name, and chart sheet-token matching is
case-insensitive, but dependency detection remains conservative and cannot detect every
reference embedded in unsupported parts.

Formula text is not rewritten when ranges, table boundaries, or sheet names change. Guaranteed
structural-reference updates and external-link repair are not supported.

## Pivot tables and advanced workbook content

Existing native pivot tables are inventory-only. The tool does not create, edit, refresh, or
validate pivot caches. Preservation is best effort and should be verified in Excel.

Slicers, timelines, Power Query, the data model, cube formulas, scenarios, solver models,
custom XML, rich data types, threaded comments, form controls, ActiveX, sparklines, and many
vendor extensions are outside the supported schema.

`summarize` creates a static pandas grouped/pivot-style report, not a native Excel pivot table.
It is visibly labeled and does not refresh when source cells change.

## Round-trip fidelity

Supported workbook content uses public `openpyxl` semantics where practical. Unsupported OOXML
extensions may be dropped or altered during save. Existing pivots and external links are
inventoried and warned about, not proven lossless. Every edit reports this rewrite risk;
workbooks with detected external-link parts receive an additional warning because targets,
cached values, and preservation are not repaired or guaranteed.

Charts created by this tool are editable native charts with a bounded type/property set. The
tool does not expose every chart axis, series, marker, trendline, secondary-axis, combo-chart,
theme, or drawing option. Chart titles and anchors can be inventoried, but arbitrary chart XML
is not preserved under every edit.

Conditional formatting and data validation support a practical subset. Application-specific
rendering and icon/color behavior can differ.

## Delimited and pandas transformations

CSV and TSV have no standard representation for workbook styles, formulas, merged cells,
multiple sheets, charts, dimensions, validations, or types. Conversion is intentionally lossy.

Schema inference is not proof of business meaning. Declare string columns to preserve leading
zeros. Declare numeric/date types only when coercion is intended. CSV dialects outside the
exposed delimiter, quoting, encoding, NA, date, header, and malformed-row policies may need
preprocessing.

Formula-injection protection prefixes risky text in CSV/TSV output. The apostrophe is data and
may be visible in consumers that do not interpret spreadsheet conventions. Enabling formulas
for untrusted input can execute spreadsheet expressions when a user opens the output. JSON
extraction is not spreadsheet-executable and is never modified by this guard.

XLSX-to-CSV/TSV defaults to a raw-sheet policy: the rectangular used range is emitted without
interpreting headers, a truly blank sheet becomes an empty file, and static-summary title or
padding rows remain present. This allows duplicate and empty first-row values but does not
provide DataFrame schema semantics. Header-aware export is explicit and requires non-empty,
unique headers. Both mappings remain lossy with respect to styles and workbook structure.

Delimited input defaults to literal NA handling. Default pandas NA spellings or custom markers
must be selected explicitly. Headerless input writes no generated header; schema declarations
that rely on names therefore require a header-aware input.

For one-file-per-sheet conversion, atomic directory publication requires a destination
directory that does not yet exist. Existing directories are refused even with `--overwrite`
to avoid partially replacing a set of files.

## Rendering and conversion

The skill does not render workbooks, compare visual layout, export PDF, or guarantee
renderer-identical output. Native charts and dimensions are structurally verified, not
pixel-rendered. PDF-native work belongs to a PDF workflow; renderer-identical XLSX-to-PDF
export remains unsupported.

## Resource limits

Default package limits are 100 MiB compressed, 10,000 ZIP members, 512 MiB total expanded,
128 MiB per member, and a 1,000:1 high-volume compression ratio. Create/edit JSON row and cell
operations share a cumulative 2,000,000-cell budget (which `create`/`edit --cell-limit` may
lower); individual rectangular operations use the
same ceiling. Inspection allows at most 100,000 emitted cell records per selected sheet.

The CLI has no built-in wall-clock deadline. Callers should impose subprocess timeouts. A hard
operating-system kill can interrupt cleanup even though normal failures remove temporary
artifacts.

These limits reduce accidental resource exhaustion; they are not a malware scanner or a
complete sandbox. Use an isolated environment for untrusted documents.

## Future work

- Formula evaluation with a separately validated calculation engine.
- Native pivot creation/editing and refresh validation.
- Macro, ActiveX, encryption, and digital-signature workflows.
- External-link and guaranteed structured-reference repair.
- Renderer-backed visual review and validated PDF export.
- Broader chart, validation, conditional-format, comment, and drawing semantics.
