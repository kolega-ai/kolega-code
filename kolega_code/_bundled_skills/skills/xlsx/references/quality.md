# Quality contract

## Contents

- [Design intake](#design-intake)
- [Delivery profiles](#delivery-profiles)
- [Acceptance criteria](#acceptance-criteria)
- [Visual review versus structural checks](#visual-review-versus-structural-checks)

## Design intake

Record this profile before polished workbook creation or release:

| Decision | Required intake |
|---|---|
| Consumer | People reading the workbook, downstream tools ingesting it, or both |
| Deliverable type | Data handoff, working model, formatted report, print/PDF, or interchange files |
| Target renderer | Excel version/platform, LibreOffice, Google Sheets, or other |
| Recalculation | Who recalculates and where; whether cached values are acceptable |
| Fonts | Approved fonts; whether the conservative set (Arial, Times New Roman, Courier New) is required |
| Print geometry | Paper size, orientation, print areas, title rows/columns, fit settings |
| Interchange | Encoding, header, NA, date, quoting, schema, and formula-injection policies |
| Protection | Sheet/workbook protection expectations (styling only — never security) |

XLSX cannot embed fonts, so a named font is portable only when installed on every intended
renderer. Page setup is read-only in this tool: when print geometry must change, set it in a
spreadsheet application before conversion, or bound the output with print areas, hidden
sheets, and `--pages`.

## Delivery profiles

Profiles describe checks performed, not certifications:

- **Data fidelity:** structure and content verified through `inspect`/`extract` — sheet
  order, values, formulas, types, tables, names, validations match the job;
  `counts.error_cells` is zero or every error is explained; recalculation status is stated.
  No visual claim. Font warnings are informational.
- **Visual review:** data fidelity plus `render`, then page-by-page examination of the PNG
  output for column overflow (`#####`), clipped or wrapped text, merge layout, chart
  appearance, and conditional-format rendering, remembering the renderer is LibreOffice.
- **Print/PDF:** visual review plus print geometry matching the intake, intended hidden-sheet
  exclusion, the font `RELEASE BLOCKER` resolved, and page-by-page review of the final PDF.
  State that values reflect LibreOffice's calculation and that pagination is not proven
  identical to Excel's.
- **Interchange:** declared encoding/header/NA/date/schema policies asserted on the delimited
  or JSON output, leading zeros preserved where declared, and the formula-injection policy
  confirmed for the trust level of the data.

Combine profiles when requested. State the renderer, unresolved warnings, and any check not
performed.

## Acceptance criteria

Apply the rows relevant to the delivery profile. A failed required row blocks release unless
the user explicitly accepts the named exception.

| Area | Measurable acceptance |
|---|---|
| Package integrity | Source preflight passes; result reports atomic publication and reopen verification; source hash unchanged unless same-path `--overwrite` was requested. |
| Structure | Sheet names/order, tables, named ranges, validations, conditional formats, and chart counts match the job; merges are intentional and destructive merges were explicitly acknowledged. |
| Formula integrity | `counts.error_cells` is zero or each stored error is explained; recalculation ownership is stated and cached-value caveats are surfaced, never silently dropped. |
| Column fit | No truncated numbers (`#####`) or clipped text on rendered pages; `auto_width` or explicit dimensions cover columns whose content matters. |
| Print geometry | Inspected `page_setup` (orientation, paper, fit, print areas, titles) matches the intake for print/PDF deliverables; hidden sheets excluded from output are intended. |
| Font portability | `fonts.referenced` reviewed; for print/PDF deliverables no `RELEASE BLOCKER` warning remains, or the user accepts the named substitution risk. Never infer portability from a clean local PDF. |
| PDF gate | `verification.pdf_signature`/`pdf_openable` true, `counts.pdf_pages` at least 1, LibreOffice diagnostics reviewed when warned. |
| Visual QA | Every changed page of the rendered output examined with the Read tool; layout, merges, charts, and conditional formats look correct on the rendering renderer. |
| Interchange | Encoding, header policy, NA policy, schema typing, leading zeros, and injection guarding asserted on the produced files. |

## Visual review versus structural checks

`inspect` verifies package structure and exposes measurable workbook properties. The PDF
gate verifies signature, reopenability, and page count. These are **structural checks**, not
pixels: they cannot detect `#####` overflow, clipped text, font substitution, chart
appearance, or whether a page looks correct.

Real visual review means rendering and looking: run `render`, open the PNG pages with the
Read tool, and examine every changed sheet's pages. The renderer is LibreOffice — an honest
renderer, but not proof of Excel-identical appearance. If rendering cannot be performed
(LibreOffice or pypdfium2 unavailable), report "structural checks only; visual QA not
performed" and make no layout or print claim.
