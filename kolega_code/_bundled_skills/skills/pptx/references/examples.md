# PPTX examples

## Contents

- [Create and inspect](#create-and-inspect)
- [Edit safely](#edit-safely)
- [Convert and render](#convert-and-render)
- [Font portability check](#font-portability-check)
- [Failure examples](#failure-examples)
- [Expected assertions](#expected-assertions)

Before using these examples, select `PPTX_PYTHON`, resolve `SKILL_ROOT` and `PPTX_TOOL` to
absolute paths, and install the declared runtime as described in
[operations](references/operations.md#runtime-prerequisite-and-installation).

## Create and inspect

Create `deck.json`:

```json
{
  "schema_version": 1,
  "operation": "create",
  "metadata": {
    "title": "Field report",
    "author": "Operations"
  },
  "slides": [
    {
      "layout": {"index": 0},
      "title": "Field report",
      "body": "16 July 2026",
      "notes": "Open with the decision, not the chronology."
    },
    {
      "layout": {"index": 1},
      "title": "Weekly throughput",
      "body": [
        {"runs": [{"text": "Output ", "bold": true}, {"text": "rose 12%."}]}
      ],
      "elements": [
        {
          "type": "chart",
          "chart_type": "column",
          "name": "Throughput Chart",
          "box": {"x": 1.0, "y": 2.4, "width": 5.5, "height": 3.8},
          "categories": ["W1", "W2", "W3"],
          "series": [{"name": "Units", "values": [80, 87, 96]}],
          "title": "Units completed",
          "has_legend": false
        }
      ]
    }
  ]
}
```

Run:

```bash
"$PPTX_PYTHON" "$PPTX_TOOL" create \
  --job "$PWD/deck.json" --output "/tmp/field-report.pptx"
"$PPTX_PYTHON" "$PPTX_TOOL" inspect "/tmp/field-report.pptx" \
  > /tmp/field-report.inspect.json
```

Representative stdout fields:

```json
{
  "schema_version": 1,
  "success": true,
  "operation": "create",
  "counts": {"slides": 2, "shapes": 5},
  "verification": {
    "atomic_publish": true,
    "package_reopen": true,
    "internal_relationship_targets_exist": true,
    "owner_relationship_references_resolved": true,
    "source_unchanged_at_publish_gate": true,
    "post_publish_source_changed": false,
    "slide_count": 2
  }
}
```

Absolute paths and version fields are omitted above; callers must not assume this abbreviated
object is the complete output.

To export the deck's images (picture shapes only, written verbatim to a new directory):

```bash
"$PPTX_PYTHON" "$PPTX_TOOL" extract "/tmp/field-report.pptx" --output "/tmp/field-report-images"
```

Each summary record pairs a `slide-<id>-shape-<id>-<hash>.<ext>` file with its slide/shape
identity and SHA-256; verify the hash against the matching `inspect` image record.

## Edit safely

Use slide IDs from `inspect`, not guessed indexes. Save this as `edit.json`:

```json
{
  "schema_version": 1,
  "operation": "edit",
  "operations": [
    {
      "action": "replace_text",
      "slide": {"slide_id": 257},
      "find": "Output rose 12%.",
      "replace": "Output rose 14%.",
      "formatting_policy": "first_run"
    },
    {
      "action": "update_chart",
      "slide": {"slide_id": 257},
      "shape": {"shape_name": "Throughput Chart"},
      "chart_type": "column",
      "categories": ["W1", "W2", "W3"],
      "series": [{"name": "Units", "values": [80, 87, 98]}]
    },
    {
      "action": "set_notes",
      "slide": {"slide_id": 257},
      "text": "The final week includes the recovered shift."
    },
    {
      "action": "set_hyperlink",
      "slide": {"slide_id": 257},
      "find": "recovered shift",
      "url": "https://wiki.example.com/recovered-shift"
    },
    {
      "action": "remove_hyperlink",
      "slide": {"slide_id": 256},
      "find": "Field report"
    }
  ]
}
```

`set_hyperlink`/`remove_hyperlink` require the matched text to align with whole runs and accept
only `http`, `https`, or `mailto` URLs. Use `scope: "shape"` with a shape selector (and no
`find`) to set or remove a shape's click action instead.

To restructure a table, provide ordered changes; each is validated against the table as already
changed by the preceding entries:

```json
{
  "schema_version": 1,
  "operation": "edit",
  "operations": [
    {
      "action": "update_table_structure",
      "slide": {"slide_id": 257},
      "shape": {"shape_name": "Metrics Table"},
      "changes": [
        {"op": "insert_row", "index": 2, "cells": ["Refunds", "3"]},
        {"op": "insert_column", "index": 0, "cells": ["Kind", "A", "B"]},
        {"op": "remove_row", "index": 1}
      ]
    }
  ]
}
```

Run:

```bash
"$PPTX_PYTHON" "$PPTX_TOOL" edit "/tmp/field-report.pptx" \
  --job "$PWD/edit.json" --output "/tmp/field-report-v2.pptx"
"$PPTX_PYTHON" "$PPTX_TOOL" inspect "/tmp/field-report-v2.pptx"
```

For reordering, provide every current ID once:

```json
{
  "schema_version": 1,
  "operation": "edit",
  "operations": [
    {"action": "reorder_slides", "slide_ids": [257, 256]},
    {"action": "remove_slide", "slide": {"slide_id": 256}}
  ]
}
```

## Convert and render

PPTX-to-PDF remains owned by this skill. Verify LibreOffice is on `PATH`:

```bash
command -v soffice || command -v libreoffice
```

If conversion was requested and LibreOffice is missing, follow
[operations](references/operations.md#libreoffice-prerequisite) for the optional, user-notified
system installation, then repeat the PATH check.
Convert only after the check succeeds:

```bash
"$PPTX_PYTHON" "$PPTX_TOOL" convert \
  "/tmp/field-report-v2.pptx" --output "/tmp/field-report-v2.pdf" --timeout 120
```

Require `verification.pdf_openable: true` and equal `source_slide_count`/`pdf_page_count`.
Conversion success does not replace visual review.

To review slides visually, render one PNG per slide (requires LibreOffice plus the optional
`pypdfium2` package, installed only with user approval):

```bash
"$PPTX_PYTHON" "$PPTX_TOOL" render \
  "/tmp/field-report-v2.pptx" --output "/tmp/field-report-v2-slides" --dpi 96
```

Then open each produced `slide-<ordinal>-id-<slide_id>.png` and look at it — check text
overflow, truncation, overlapping or clipped shapes, missing images, and table/chart layout.
The render summary alone proves file integrity, not visual correctness.

## Font portability check

Before releasing matching PPTX and PDF deliverables, inspect the final deck and review its font
inventory:

```bash
"$PPTX_PYTHON" "$PPTX_TOOL" inspect "/tmp/field-report-v2.pptx"
```

Review `fonts.referenced`, `fonts.embedded`, and `fonts.unembedded`, and resolve every
unembedded-font warning. Do not leave a `RELEASE BLOCKER:` font warning unreviewed for a
PPTX+PDF fidelity task. A correct PDF rendering and clean `pdffonts` output do not prove PPTX
font portability: they describe the generated PDF, not what another PowerPoint renderer will
substitute when opening the deck.

## Failure examples

Missing input:

```bash
"$PPTX_PYTHON" "$PPTX_TOOL" inspect "/tmp/missing.pptx"
```

Representative stderr and status 2:

```json
{"schema_version":1,"success":false,"error":{"category":"bad_input","message":"source does not exist","details":{"path":"/tmp/missing.pptx"}}}
```

Unsupported macro-enabled input is rejected before parsing:

```bash
"$PPTX_PYTHON" "$PPTX_TOOL" inspect "/tmp/deck.pptm"
```

A corrupt `.pptx`, a cross-run replacement without `formatting_policy`, a duplicate layout name,
and a reorder list missing a current slide all fail without publishing the destination.
Passing the source path as `--output`, even with `--overwrite`, also fails: use a distinct output
path and verify that the source hash remains unchanged.

## Expected assertions

After create/edit:

- source SHA-256 is unchanged; source and destination are always distinct;
- the output begins with a ZIP signature and reopens with `python-pptx`;
- retained slide IDs are unchanged and reported order equals the requested order;
- all internal relationship targets exist, and relationship-namespace references in owner XML
  resolve through that owner's relationship part;
- table/chart/image/note inspection reflects the requested values;
- for PPTX+PDF deliverables, no `RELEASE BLOCKER:` font warning remains in the final inspection;
- no sibling temporary output remains after success or failure.
