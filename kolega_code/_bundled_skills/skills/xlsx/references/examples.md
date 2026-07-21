# XLSX examples

## Table of contents

- [Install and smoke test](#install-and-smoke-test)
- [End-to-end create, inspect, edit, and extract](#end-to-end-create-inspect-edit-and-extract)
- [Static summary](#static-summary)
- [UTF-8 CSV/TSV interchange](#utf-8-csvtsv-interchange)
- [Expected artifact assertions](#expected-artifact-assertions)
- [Failure examples](#failure-examples)

## Install and smoke test

Follow [Runtime and JSON contract](references/operations.md#runtime-and-json-contract) to select
`XLSX_PYTHON`, notify the user before any installation, and install the declared requirements.
Then resolve the skill root to an absolute path and run:

```bash
SKILL_ROOT="/absolute/path/to/skills/xlsx"
"$XLSX_PYTHON" "$SKILL_ROOT/scripts/smoke_test.py"
```

Expected final stdout:

```json
{"fixtures":"temporary","ok":true,"operations":["create","inspect","extract","edit","clean","summarize","convert_csv_xlsx_tsv_xlsx","raw_all_sheet_static_summary_export","header_aware_export","conversion_option_direction_validation","json_formula_preservation","case_insensitive_formula_dependencies","bad_input_categories","xml_encoding_security","subprocess_timeout","cumulative_json_cell_budget","external_link_preservation_warning","macro_refusal"],"schema_version":1,"test":"xlsx_smoke"}
```

The test uses `TemporaryDirectory`; no workbook or delimited fixture is left in the skill.

## End-to-end create, inspect, edit, and extract

Use a temporary working directory:

```bash
work="$(mktemp -d)"
cat >"$work/create.json" <<'JSON'
{
  "schema_version": 1,
  "properties": {"title": "Regional café sales", "creator": "Spreadsheet workflow"},
  "named_styles": [
    {
      "name": "Header",
      "font": {"bold": true, "color": "FFFFFF"},
      "fill": {"color": "1F4E78"},
      "alignment": {"horizontal": "center"}
    }
  ],
  "sheets": [
    {
      "name": "Sales",
      "rows": [
        ["Region", "Product", "Revenue", "Units"],
        ["North", "Café", 120.5, 3],
        ["South", "Thé", 80, 2],
        ["North", "Crème", 150, 5]
      ],
      "cells": {
        "A1": {"value": "Region", "style": "Header"},
        "B1": {"value": "Product", "style": "Header"},
        "C1": {"value": "Revenue", "style": "Header"},
        "D1": {"value": "Units", "style": "Header"},
        "F1": {"value": "Average", "style": "Header"},
        "F2": {"formula": "=AVERAGE(C2:C4)", "number_format": "$0.00"}
      },
      "freeze_panes": "A2",
      "tables": [
        {"name": "SalesTable", "range": "A1:D4", "style": {"name": "TableStyleMedium2"}}
      ],
      "conditional_formats": [
        {
          "range": "C2:C4",
          "type": "color_scale",
          "colors": ["F8696B", "FFEB84", "63BE7B"]
        }
      ],
      "charts": [
        {
          "type": "bar",
          "title": "Revenue by product",
          "data": "C1:C4",
          "categories": "B2:B4",
          "anchor": "H2"
        }
      ]
    },
    {"name": "Notes", "rows": [["Status"], ["Draft"]]}
  ],
  "defined_names": [
    {"name": "RevenueData", "refers_to": "'Sales'!$C$2:$C$4"}
  ]
}
JSON

"$XLSX_PYTHON" "$SKILL_ROOT/scripts/xlsx_tool.py" create "$work/create.json" "$work/report.xlsx"
"$XLSX_PYTHON" "$SKILL_ROOT/scripts/xlsx_tool.py" inspect "$work/report.xlsx" --max-cells 100 \
  >"$work/inspection.json"
```

Representative create stdout (paths and package byte counts vary):

```json
{
  "ok": true,
  "operation": "create",
  "result": {
    "counts": {
    "cells": 24,
      "charts": 1,
      "conditional_formats": 1,
      "defined_names": 1,
      "named_styles": 1,
      "tables": 1
    },
    "output": "/tmp/.../report.xlsx",
    "sheets": ["Sales", "Notes"],
    "verification": {
      "reopened": true,
      "sheet_names": ["Sales", "Notes"]
    }
  },
  "schema_version": 1,
  "warnings": [
    "Formulas were written but not calculated; calculation flags request, not prove, refresh."
  ]
}
```

Edit with exact operations:

```bash
cat >"$work/edit.json" <<'JSON'
{
  "schema_version": 1,
  "operations": [
    {"op": "set_cell", "sheet": "Sales", "cell": "B2", "value": "Café au lait"},
    {
      "op": "format_range",
      "sheet": "Sales",
      "range": "C2:C4",
      "style": {"number_format": "$#,##0.00"}
    },
    {
      "op": "add_sheet",
      "name": "Audit",
      "rows": [["Action", "Count"], ["Updated", 1]]
    }
  ]
}
JSON

"$XLSX_PYTHON" "$SKILL_ROOT/scripts/xlsx_tool.py" edit \
  "$work/report.xlsx" "$work/edit.json" "$work/edited.xlsx"
"$XLSX_PYTHON" "$SKILL_ROOT/scripts/xlsx_tool.py" extract \
  "$work/edited.xlsx" "$work/sales.json" \
  --sheet Sales --range A1:F2 --format json
```

The source remains unchanged. The edited workbook has sheet order
`["Sales", "Notes", "Audit"]`; `Sales!B2` is `Café au lait`; and cells `C2:C4` use the
requested number format.

JSON extraction is data-preserving rather than spreadsheet-executable: formula expressions
and text such as `=2+2` remain exactly `=2+2` in the JSON row matrix. Apostrophe formula
injection guarding applies only to CSV/TSV output.

## Static summary

```bash
cat >"$work/summary.json" <<'JSON'
{
  "schema_version": 1,
  "source": {"sheet": "Sales", "range": "A1:D4", "header_row": 1},
  "destination_sheet": "Static Summary",
  "group_by": ["Region"],
  "values": [
    {"column": "Revenue", "aggregation": "sum", "name": "Revenue total"},
    {"column": "Units", "aggregation": "sum", "name": "Unit total"}
  ],
  "chart": {"type": "bar", "title": "Regional totals"}
}
JSON

"$XLSX_PYTHON" "$SKILL_ROOT/scripts/xlsx_tool.py" summarize \
  "$work/edited.xlsx" "$work/summary.json" "$work/summarized.xlsx"
```

Representative result fields:

```json
{
  "ok": true,
  "operation": "summarize",
  "result": {
    "destination_sheet": "Static Summary",
    "rows": 2,
    "columns": 3,
    "charts": 1,
    "verification": {"reopened": true}
  },
  "warnings": [
    "The generated grouped/pivot-style report is static, not a native Excel pivot table."
  ]
}
```

Cell `A1` on the result sheet begins with
`STATIC SUMMARY — values do not refresh automatically`.

## UTF-8 CSV/TSV interchange

Preserve leading zeros and neutralize formula-like text:

```bash
cat >"$work/input.csv" <<'CSV'
Code,City,Text,Amount
00123,Montréal,"café, crème",10.5
00007,Zürich,"=2+2",20
CSV

cat >"$work/schema.json" <<'JSON'
{"Code": "string", "Amount": "float"}
JSON

"$XLSX_PYTHON" "$SKILL_ROOT/scripts/xlsx_tool.py" convert \
  "$work/input.csv" "$work/from-csv.xlsx" \
  --schema "$work/schema.json" --leading-zeros preserve

"$XLSX_PYTHON" "$SKILL_ROOT/scripts/xlsx_tool.py" convert \
  "$work/from-csv.xlsx" "$work/output.tsv" \
  --output-format tsv --sheet Data --encoding utf-8
```

Expected behavior:

- XLSX `Data!A2` is the string `00123`;
- XLSX `Data!C3` is the string `=2+2`, not a formula;
- TSV remains UTF-8 and contains `Montréal`, `Zürich`, and `00123`;
- TSV formula-like text is exported as `'=2+2`;
- JSON results report `formula_injection_guarded`.

Use `--allow-formulas` only when the data is trusted and execution is intended.

Headerless input and explicit NA recognition are separate policies:

```bash
cat >"$work/no-header.csv" <<'CSV'
00123,NA,
00007,,value
CSV

"$XLSX_PYTHON" "$SKILL_ROOT/scripts/xlsx_tool.py" convert \
  "$work/no-header.csv" "$work/no-header.xlsx" \
  --no-header --input-na-policy literal

"$XLSX_PYTHON" "$SKILL_ROOT/scripts/xlsx_tool.py" convert \
  "$work/input.csv" "$work/custom-na.xlsx" \
  --input-na-policy custom --input-na-value 20
```

The first command writes `00123` to `A1` and literal `NA` to `B1`, with no synthetic numeric
header row. The second treats only the explicit marker `20` as missing. Use
`--input-na-policy default` only when pandas' standard NA spellings are intended.

Export every workbook sheet to a new directory:

```bash
"$XLSX_PYTHON" "$SKILL_ROOT/scripts/xlsx_tool.py" convert \
  "$work/summarized.xlsx" "$work/exported" \
  --all-sheets --output-format csv
```

The default `raw` policy exports every sheet's used-range rows without interpreting a header.
It therefore preserves the static-summary title and padding rows, permits duplicate/empty
first-row values, and writes a truly blank sheet as an empty file. Files are deterministic,
for example `001-Sales.csv` and `004-Static_Summary.csv`; the JSON result retains the exact
sheet-to-file mapping. Add `--sheet-policy header` only for conventional unique, non-empty
headers when `--schema`, `--header-row`, or `--index` is needed. The directory is published
after all numbered sheet files are written and reopened.

## Expected artifact assertions

The following independent check uses only the declared environment:

```bash
"$XLSX_PYTHON" - "$work/summarized.xlsx" <<'PY'
import sys
from openpyxl import load_workbook

book = load_workbook(sys.argv[1], data_only=False)
assert book.sheetnames == ["Sales", "Notes", "Audit", "Static Summary"]
assert book["Sales"]["F2"].data_type == "f"
assert book["Sales"]["B2"].value == "Café au lait"
assert len(book["Sales"].tables) == 1
assert len(book["Sales"].conditional_formatting) == 1
assert len(book["Sales"]._charts) == 1
assert book["Static Summary"]["A1"].value.startswith("STATIC SUMMARY")
assert len(book["Static Summary"]._charts) == 1
book.close()
print("artifact assertions passed")
PY
```

For a user-visible formula result, open the workbook in a trusted spreadsheet application,
recalculate it, save it, and inspect cached values again. A successful save by this tool alone
does not establish formula freshness.

## Failure examples

Missing input produces status 2 and JSON on stderr:

```bash
"$XLSX_PYTHON" "$SKILL_ROOT/scripts/xlsx_tool.py" inspect "$work/missing.xlsx"
echo "$?"
```

```json
{"error":{"category":"bad_input","details":{"path":"/tmp/.../missing.xlsx"},"message":"Source workbook does not exist."},"ok":false,"schema_version":1}
```

A macro-enabled extension is refused before package parsing with status 3:

```bash
: >"$work/refused.xlsm"
"$XLSX_PYTHON" "$SKILL_ROOT/scripts/xlsx_tool.py" inspect "$work/refused.xlsm"
echo "$?"
```

```json
{"error":{"category":"unsupported_operation","details":{"extension":".xlsm","path":"/tmp/.../refused.xlsm"},"message":"Macro-enabled workbooks are refused."},"ok":false,"schema_version":1}
```

A corrupt `.xlsx` signature also fails without creating an output:

```bash
printf 'not an OOXML package' >"$work/corrupt.xlsx"
"$XLSX_PYTHON" "$SKILL_ROOT/scripts/xlsx_tool.py" inspect "$work/corrupt.xlsx"
```

```json
{"error":{"category":"bad_input","details":{"path":"/tmp/.../corrupt.xlsx"},"message":"Workbook does not have an OOXML ZIP signature."},"ok":false,"schema_version":1}
```

An attempted structural move returns `ambiguous_edit` with status 5:

```json
{
  "schema_version": 1,
  "operations": [
    {"op": "move_range", "sheet": "Sales", "range": "A1:D4", "target": "B2"}
  ]
}
```

The tool refuses it rather than claiming that every dependent formula, table, chart, and name
was safely rewritten.
