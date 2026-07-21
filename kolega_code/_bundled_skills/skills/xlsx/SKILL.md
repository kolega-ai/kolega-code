---
name: xlsx
description: Create, inspect, extract, edit, clean, summarize, and convert Microsoft Excel .xlsx workbooks, including formulas, styles, tables, conditional formatting, validation, native charts, and CSV/TSV interchange. Use for spreadsheet artifact work where workbook semantics or safe tabular conversion matter; do not use for legacy .xls, macro-enabled .xlsm, PDF-native work, or general analysis that does not require a spreadsheet artifact.
license: Apache-2.0
compatibility: Requires Python 3.11+ and the packages declared in requirements.txt.
metadata:
  author: Kolega
  version: "1.0"
  schema_version: "1"
---

# XLSX workbooks

Operate on `.xlsx` files with a deterministic CLI. Keep source workbooks immutable, refuse
macro-enabled content, and verify every workbook after saving.

## Workflow

1. **Prepare the runtime proactively.** Resolve the skill root and choose any available Python
   3.11+ interpreter; do not assume a launcher name. Check the required imports before
   installing anything. If something is missing, first tell the user what you intend to
   install, where, and with which installer. Use the selected interpreter's `-m pip` for the
declared requirements. Use the platform's package manager for Python itself, preferring
   Homebrew on macOS when available. A local environment is a fallback, not a prerequisite.
   Read the runtime section in [operations](references/operations.md).
2. **Inventory before mutation.** Run `inspect` and review sheet order and visibility, used
   ranges, merges, formulas and cached values, names, external links, tables, styles,
   validations, conditional formatting, charts, pivots, and warnings.
3. **Choose the semantic path.**
   - Use `openpyxl`-backed `create` and `edit` for rich workbook semantics.
   - Use `clean` or `summarize` only for explicit rectangular pandas transformations. These
     write a separate result sheet and preserve the original source sheet.
   - Use `extract` or `convert` for JSON/CSV/TSV interchange. State encoding, schema,
     leading-zero, NA, date, quoting, index, and formula policies when they matter.
     XLSX-to-delimited conversion defaults to raw used-range rows; select header-aware mode
     only for conventional unique, non-empty headers.
   - Never round-trip a rich workbook through a DataFrame merely to edit cells.
4. **Use a versioned job.** Set `"schema_version": 1` in every create, edit, clean, or
   summarize JSON job. Prefer exact sheet names, A1 ranges, and named operations.
5. **Write to a new destination.** A destination must differ from the source unless
   `--overwrite` is explicit. Existing destinations also require `--overwrite`.
6. **Review the JSON result.** Require `"ok": true`, a reopened verification result, expected
sheet order/counts, warnings, output paths, and library versions.
7. **Inspect the output again.** Confirm workbook structure and content. If formulas exist,
   open in a calculation application when fresh results are required; calculation flags
   request recalculation but do not prove it happened.

## Safety rules

- Accept only `.xlsx` OOXML ZIP packages. Refuse `.xlsm`, `.xltm`, `.xls`, `.xlsb`, macro
  parts, encrypted ZIP members, DTD/entity declarations, malformed signatures, path
  traversal, and packages beyond bounded member/count/expanded-size limits.
- Treat source files as immutable. Mutation saves to a temporary sibling, reopens and
  validates it, then atomically publishes the destination.
- Reject row, column, and range moves because formulas, tables, charts, and names cannot all
  be updated reliably. Reject sheet rename/removal with detected dependencies unless the job
  explicitly accepts unresolved references.
- Represent executable formulas only with a `formula` field or explicit `--allow-formulas`.
  Literal values beginning with `=` remain strings. CSV/TSV output prefixes potentially
  executable untrusted text unless formulas are explicitly allowed. JSON extraction is never
  formula-injection sanitized.
- Treat schema version 1 as closed. Unknown job, operation, sheet, source, and nested schema
  keys are errors rather than forward-compatible ignored values.
- Never assume formula caches are current. `openpyxl` writes formulas but does not calculate
  them.
- Refuse macro-bearing edits rather than attempting to preserve VBA, ActiveX, signatures, or
  macro security state.
- Treat every edit as an OOXML rewrite: review warnings for unsupported extension content,
  pivots, and external-link preservation before accepting the result.

## Commands and resources

After runtime setup, run `"$XLSX_PYTHON" "$SKILL_ROOT/scripts/xlsx_tool.py" --help` for command
flags.

- Read [operations](references/operations.md) for CLI behavior, job schemas, exit statuses,
  safety limits, and operation details.
- Read [examples](references/examples.md) for copy-pasteable workflows, representative JSON,
  assertions, and failure paths.
- Read [limitations](references/limitations.md) before working with formulas, pivots,
  external links, unsupported OOXML, or renderer-specific output.
- Run `"$XLSX_PYTHON" "$SKILL_ROOT/scripts/smoke_test.py"` after installation. It creates every
  fixture in a temporary directory and exercises create, inspect, extract, edit, clean,
  summarize,
  raw all-sheet/static-summary export, JSON formula preservation, case-insensitive
  dependencies, cumulative budgets, CSV/TSV policies, XML security, timeout handling, and
  categorized failure paths.

## Completion checks

- The source hash is unchanged unless same-path `--overwrite` was explicitly requested.
- The result reports atomic publication and successful reopen verification.
- Sheet names/order, formulas, table names/ranges, styles, conditional formats, validations,
  and chart counts match the requested job.
- Static summaries are visibly labeled static.
- Raw all-sheet exports include blank and static-summary sheets under deterministic numbered
  names; delimited outputs preserve the declared encoding/header/NA/schema behavior and apply
  the expected formula-injection policy.
- Warnings about stale formula caches, external links, pivots, or accepted dependency
  uncertainty are reported to the caller.
