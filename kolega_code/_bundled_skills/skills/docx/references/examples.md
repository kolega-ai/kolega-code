# Examples

All jobs use schema version 1. Paths inside JSON resolve relative to that JSON file.

## Example index

- [Setup](#setup)
- [Polished end-to-end baseline](#polished-end-to-end-baseline)
- [Approved template creation](#approved-template-creation)
- [Paragraph pagination and tabs](#paragraph-pagination-and-tabs)
- [Accessible captioned figure](#accessible-captioned-figure)
- [Fixed printable table](#fixed-printable-table)
- [Sections and page numbering](#sections-and-page-numbering)
- [Update fields on open](#update-fields-on-open)
- [Edit-time table, image, and style](#edit-time-table-image-and-style)
- [Inspection and release flow](#inspection-and-release-flow)
- [Render for visual review](#render-for-visual-review)
- [Complete job dispatch](#complete-job-dispatch)

## Setup

```bash
SKILL_ROOT="/absolute/path/to/skills/docx"
DOCX_PYTHON="/path/to/python"
DOCX_TOOL="$SKILL_ROOT/scripts/docx_tool.py"
WORK="$PWD/docx-work"
mkdir -p "$WORK"
```

These examples assume an agreed profile such as: general audience; screen and print brief;
Letter; target Microsoft Word desktop plus the actual PDF converter; editable-DOCX fidelity
medium; no brand; accessibility-oriented; TOC omitted because the brief is short; ordinary
PDF for viewing/print, with no tagged-PDF or PDF/A claim. Font choices below are design
examples, not portability claims: use a licensed embedded font or ensure every unembedded font
is installed on every intended renderer.

## Polished end-to-end baseline

This baseline deliberately defines geometry, styles, metadata, table layout, stories, and
field-update behavior. It does not rely on default formatting.

```bash
cat >"$WORK/polished-create.json" <<'JSON'
{
  "schema_version": 1,
  "content": {
    "metadata": {
      "title": "Service readiness brief",
      "subject": "Launch readiness",
      "author": "Operations Team",
      "language": "en-US",
      "keywords": "readiness, launch"
    },
    "section": {
      "paper_size": "letter",
      "orientation": "portrait",
      "top_margin_inches": 0.75,
      "right_margin_inches": 0.8,
      "bottom_margin_inches": 0.75,
      "left_margin_inches": 0.8,
      "header_distance_inches": 0.3,
      "footer_distance_inches": 0.3,
      "different_first_page": true,
      "page_number_start": 1,
      "page_number_format": "decimal"
    },
    "styles": [
      {
        "name": "Normal",
        "type": "paragraph",
        "font": "Arial",
        "size_pt": 10.5,
        "color": "202124",
        "paragraph": {
          "space_after_pt": 6,
          "line_spacing": 1.15,
          "widow_control": true
        }
      },
      {
        "name": "Brief Title",
        "type": "paragraph",
        "based_on": "Title",
        "font": "Arial",
        "size_pt": 24,
        "bold": true,
        "color": "17365D",
        "outline_level": 0,
        "paragraph": {
          "space_after_pt": 12,
          "keep_with_next": true
        }
      },
      {
        "name": "Brief Heading",
        "type": "paragraph",
        "based_on": "Heading 1",
        "font": "Arial",
        "size_pt": 15,
        "bold": true,
        "color": "17365D",
        "outline_level": 0,
        "paragraph": {
          "space_before_pt": 12,
          "space_after_pt": 5,
          "keep_with_next": true,
          "keep_together": true
        }
      }
    ],
    "update_fields_on_open": true,
    "blocks": [
      {"type": "paragraph", "style": "Brief Title", "text": "Service readiness"},
      {
        "type": "paragraph",
        "text": "A concise decision brief for launch owners.",
        "space_after_pt": 12,
        "keep_together": true
      },
      {"type": "paragraph", "style": "Brief Heading", "text": "Decision"},
      {
        "type": "paragraph",
        "text": "Proceed after the two open controls are verified.",
        "widow_control": true
      },
      {"type": "paragraph", "style": "Brief Heading", "text": "Readiness"},
      {
        "type": "table",
        "style": "Table Grid",
        "header_rows": 1,
        "layout": "fixed",
        "column_widths_inches": [2.0, 2.2, 2.7],
        "allow_row_split": false,
        "row_allow_split": [false, false, false],
        "rows": [
          ["Control", "Owner", "Status"],
          ["Monitoring", "Platform", "Ready"],
          ["Rollback", "Release", "Verify"]
        ]
      }
    ],
    "headers": {
      "first_page": [],
      "default": [
        {"type": "paragraph", "text": "Service readiness", "alignment": "right"}
      ]
    },
    "footers": {
      "default": [
        {
          "type": "field",
          "field": "page_number",
          "prefix": "Page ",
          "alignment": "center"
        }
      ]
    }
  }
}
JSON

"$DOCX_PYTHON" "$DOCX_TOOL" create \
  --spec "$WORK/polished-create.json" --output "$WORK/readiness.docx"
"$DOCX_PYTHON" "$DOCX_TOOL" inspect \
  --input "$WORK/readiness.docx" >"$WORK/readiness-inspect.json"
```

For A4, change only the agreed geometry and recheck every table/image against the resulting
`usable_width_inches`. Add a TOC only if requested or justified by complexity.

## Approved template creation

Use this only when `approved-template.docx` is a true base document intended to receive
appended content. Preflight it first:

```bash
"$DOCX_PYTHON" "$DOCX_TOOL" inspect \
  --input "$WORK/approved-template.docx" >"$WORK/template-inspect.json"

cat >"$WORK/template-create.json" <<'JSON'
{
  "schema_version": 1,
  "content": {
    "template": "approved-template.docx",
    "metadata": {
      "title": "Approved-format brief",
      "author": "Editorial Team",
      "language": "en-US"
    },
    "blocks": [
      {"type": "heading", "level": 1, "text": "Summary"},
      {"type": "paragraph", "text": "Approved content."}
    ]
  }
}
JSON

"$DOCX_PYTHON" "$DOCX_TOOL" create \
  --spec "$WORK/template-create.json" --output "$WORK/approved-output.docx"
```

If the template contains `{{TITLE}}`, `{{BODY}}`, or other placeholders, do not use append:
inspect and run bounded `replace_text` operations with exact `expected_count`, then insert
larger structures relative to inspected anchors.

## Paragraph pagination and tabs

```json
{
  "type": "paragraph",
  "style": "Heading 2",
  "text": "Operational detail",
  "space_before_pt": 10,
  "space_after_pt": 4,
  "line_spacing": "single",
  "keep_with_next": true,
  "keep_together": true,
  "page_break_before": false,
  "widow_control": true,
  "tab_stops": [
    {"position_inches": 5.5, "alignment": "right", "leader": "dots"}
  ]
}
```

For hanging labels, use `hanging_indent_inches`; do not combine it with
`first_line_indent_inches`.

## Accessible captioned figure

Assuming `trend.png` exists next to the JSON:

```json
{
  "type": "image",
  "path": "trend.png",
  "width_inches": 5.5,
  "alignment": "center",
  "alt_text": "Monthly completion rose from 62 percent in January to 91 percent in June.",
  "title": "Monthly completion trend",
  "caption": "Figure 1. Monthly completion rate, January–June.",
  "caption_style": "Caption",
  "attribution": "Source: Operations dashboard"
}
```

Use `"decorative": true` instead of `alt_text` only when the image conveys no information.
Inspect effective PPI and displayed width after creation.

## Fixed printable table

For a 7.0-inch usable width, these columns sum to exactly 7.0:

```json
{
  "type": "table",
  "style": "Table Grid",
  "header_rows": 1,
  "layout": "fixed",
  "column_widths_inches": [1.6, 3.5, 1.9],
  "allow_row_split": true,
  "row_allow_split": [false, true, true],
  "rows": [
    ["ID", "Finding", "Status"],
    ["A-01", "Long finding that may need to split if it grows beyond one page.", "Open"],
    ["A-02", "Short finding.", "Closed"]
  ]
}
```

Use fixed widths when predictable print layout matters. Reinspect because renderer layout may
still differ from preferred-width markup.

## Sections and page numbering

```json
[
  {
    "type": "section_break",
    "start": "new_page",
    "paper_size": "letter",
    "orientation": "landscape",
    "top_margin_inches": 0.6,
    "right_margin_inches": 0.6,
    "bottom_margin_inches": 0.6,
    "left_margin_inches": 0.6,
    "page_number_start": 1,
    "page_number_format": "upperRoman"
  },
  {"type": "heading", "level": 1, "text": "Landscape appendix"}
]
```

Equivalent edit-time section configuration:

```json
{
  "type": "configure_section",
  "section_index": 1,
  "orientation": "landscape",
  "page_number_start": 1,
  "page_number_format": "decimal",
  "different_first_page": false
}
```

Pair numbering markup with a footer field and inspect/render the intended restart.

## Update fields on open

Create-level:

```json
{
  "update_fields_on_open": true,
  "blocks": [
    {"type": "field", "field": "toc"},
    {"type": "field", "field": "num_pages", "prefix": "Total pages: "}
  ]
}
```

Edit-time:

```json
{"type": "set_update_fields_on_open", "enabled": true}
```

This setting is not a refresh guarantee. Open in the target layout application, update fields,
save, then inspect the visible results. Omit the TOC for short/simple documents unless asked.

## Edit-time table, image, and style

Assuming `trend.png` exists and inspection confirms paragraph 2:

```bash
cat >"$WORK/edit-structures.json" <<'JSON'
{
  "schema_version": 1,
  "operations": [
    {
      "type": "upsert_style",
      "style": {
        "name": "Callout",
        "type": "paragraph",
        "based_on": "Normal",
        "font": "Arial",
        "size_pt": 10,
        "bold": true,
        "color": "17365D",
        "paragraph": {
          "space_before_pt": 6,
          "space_after_pt": 6,
          "keep_together": true
        }
      }
    },
    {
      "type": "insert_paragraph",
      "paragraph_index": 2,
      "position": "after",
      "block": {"type": "paragraph", "style": "Callout", "text": "Review required."}
    },
    {
      "type": "insert_table",
      "paragraph_index": 2,
      "position": "after",
      "table": {
        "type": "table",
        "style": "Table Grid",
        "header_rows": 1,
        "layout": "fixed",
        "column_widths_inches": [2.0, 4.9],
        "row_allow_split": [false, false],
        "rows": [["Owner", "Action"], ["Platform", "Verify monitoring"]]
      }
    },
    {
      "type": "insert_image",
      "path": "trend.png",
      "width_inches": 4.5,
      "alt_text": "Completion trend rising from January through June.",
      "title": "Completion trend"
    },
    {
      "type": "update_table",
      "table_index": 0,
      "row": 1,
      "column": 1,
      "value": {
        "text": "Verify monitoring and rollback",
        "keep_together": true
      },
      "header_rows": 1,
      "layout": "fixed",
      "column_widths_inches": [2.0, 4.9],
      "row_allow_split": [false, false]
    }
  ]
}
JSON

"$DOCX_PYTHON" "$DOCX_TOOL" edit \
  --input "$WORK/readiness.docx" --spec "$WORK/edit-structures.json" \
  --output "$WORK/readiness-edited.docx"
```

Operations are sequential: inserting a paragraph changes later paragraph indices, but does not
change table index 0 in this example. Inspect the source and plan indices before dispatch.

## Inspection and release flow

```bash
"$DOCX_PYTHON" "$DOCX_TOOL" inspect \
  --input "$WORK/readiness-edited.docx" >"$WORK/final-inspect.json"

"$DOCX_PYTHON" - "$WORK/final-inspect.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
r = d["result"]
print("warnings:", [w["code"] for w in d["warnings"]])
print("sections:", r["sections"])
print("tables:", [
    (t["index"], t["section_index"], t["column_widths_inches"],
     t["row_allows_split"], t["header_rows"]) for t in r["tables"]
])
print("images:", r["images"]["inline_occurrences"])
print("fields:", r["fields"])
print("fonts:", r["fonts"])
print("settings:", r["settings"])
PY

"$DOCX_PYTHON" "$DOCX_TOOL" convert \
  --input "$WORK/readiness-edited.docx" --format pdf \
  --output "$WORK/readiness-edited.pdf" --timeout 90
```

Then apply [the requested delivery profile](references/quality.md#delivery-profiles). Review all DOCX
warnings, refresh fields in the target application, and render and review every page as shown
in [render for visual review](#render-for-visual-review). The PDF's `content_quality_report`
can flag structural anomalies; it is not visual QA. If visual review or a target renderer is
unavailable, disclose that instead of claiming print, accessibility, archive, or
cross-renderer fidelity.

## Render for visual review

```bash
"$DOCX_PYTHON" "$DOCX_TOOL" render \
  --input "$WORK/readiness-edited.docx" \
  --output "$WORK/readiness-pages" --dpi 150 --timeout 90
ls "$WORK/readiness-pages"
```

The destination directory must not already exist; rerunning the command against the same
path exits 9 (`output_conflict`), so render into a fresh directory per revision. Open each
published `page-NNN.png` with the Read tool and review it against the
[Visual QA acceptance row](references/quality.md#acceptance-criteria). Render a subset while
iterating with `--pages 1,3`; page numbers are 1-based positions in the converted PDF, and a
number past the last page fails with the real `pdf_pages` count in the error details.

## Complete job dispatch

```bash
cat >"$WORK/inspect-job.json" <<'JSON'
{
  "schema_version": 1,
  "operation": "inspect",
  "input": "readiness-edited.docx",
  "allow_external_relationships": false
}
JSON

"$DOCX_PYTHON" "$DOCX_TOOL" --job "$WORK/inspect-job.json"
```

Complete create/edit/convert jobs add the same fields shown in
[invocation and envelopes](references/operations.md#invocation-and-envelopes), including `output`,
`overwrite`, and operation-specific `content`, `operations`, `format`, or `timeout`.
