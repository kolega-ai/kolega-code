# PPTX examples

## Contents

- [Create and inspect](#create-and-inspect)
- [Edit safely](#edit-safely)
- [Convert to PDF](#convert-to-pdf)
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

## Convert to PDF

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
- no sibling temporary output remains after success or failure.
