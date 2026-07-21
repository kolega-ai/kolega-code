# Examples

## Contents

- [End-to-end create, inspect, edit, and extract](#end-to-end-create-inspect-edit-and-extract)
- [Representative stdout](#representative-stdout)
- [Complete job dispatch](#complete-job-dispatch)
- [Explicit paragraph append](#explicit-paragraph-append)
- [External-relationship opt-in](#external-relationship-opt-in)
- [Corrupt input](#corrupt-input)
- [Optional PDF conversion](#optional-pdf-conversion)

All paths in JSON are relative to the JSON file. Before using these examples, follow
[environment setup](references/operations.md#environment) to set absolute `SKILL_ROOT` and
verified `DOCX_PYTHON` values and install any missing requirements. Installation
guidance is intentionally kept in operations rather than repeated here.

## End-to-end create, inspect, edit, and extract

Create `/tmp/docx-demo/create.json`:

```json
{
  "schema_version": 1,
  "content": {
    "metadata": {
      "title": "Release brief",
      "author": "Documentation Team"
    },
    "blocks": [
      {
        "type": "heading",
        "level": 1,
        "text": "Release brief"
      },
      {
        "type": "paragraph",
        "style": "Quote",
        "runs": [
          {"text": "Status: ", "bold": true},
          {"text": "Draft", "italic": true}
        ]
      },
      {
        "type": "table",
        "style": "Table Grid",
        "header_rows": 1,
        "rows": [
          ["Owner", "State"],
          ["Documentation", "Draft"]
        ]
      }
    ],
    "headers": {
      "default": [
        {"type": "paragraph", "text": "Internal release brief"}
      ]
    },
    "footers": {
      "default": [
        {"type": "field", "field": "page_number", "prefix": "Page "}
      ]
    }
  }
}
```

Create `/tmp/docx-demo/edit.json`:

```json
{
  "schema_version": 1,
  "operations": [
    {
      "type": "replace_text",
      "find": "Draft",
      "replace": "Approved",
      "expected_count": 2,
      "replace_all": true,
      "scope": "body"
    },
    {
      "type": "update_table",
      "table_index": 0,
      "row": 1,
      "column": 1,
      "value": "Approved"
    },
    {
      "type": "set_metadata",
      "metadata": {
        "title": "Approved release brief"
      }
    }
  ]
}
```

The replacement above intentionally updates both occurrences before the table operation writes
the same final cell value. This demonstrates count-bounded multi-match replacement. The next
sequence uses a narrower cross-run match.

A copy-pasteable end-to-end shell sequence with a corrected single text match is:

```bash
set -eu
mkdir -p "/tmp/docx-demo"
cat >"/tmp/docx-demo/create.json" <<'JSON'
{
  "schema_version": 1,
  "content": {
    "metadata": {"title": "Release brief", "author": "Documentation Team"},
    "blocks": [
      {"type": "heading", "level": 1, "text": "Release brief"},
      {
        "type": "paragraph",
        "style": "Quote",
        "runs": [
          {"text": "Status: ", "bold": true},
          {"text": "Draft", "italic": true}
        ]
      },
      {
        "type": "table",
        "style": "Table Grid",
        "header_rows": 1,
        "rows": [["Owner", "State"], ["Documentation", "Draft"]]
      }
    ],
    "headers": {
      "default": [{"type": "paragraph", "text": "Internal release brief"}]
    },
    "footers": {
      "default": [{"type": "field", "field": "page_number", "prefix": "Page "}]
    }
  }
}
JSON
cat >"/tmp/docx-demo/edit.json" <<'JSON'
{
  "schema_version": 1,
  "operations": [
    {
      "type": "replace_text",
      "find": "Status: Draft",
      "replace": "Status: Approved",
      "expected_count": 1,
      "cross_run_policy": "first_run"
    },
    {
      "type": "update_table",
      "table_index": 0,
      "row": 1,
      "column": 1,
      "value": "Approved"
    }
  ]
}
JSON
"$DOCX_PYTHON" "$SKILL_ROOT/scripts/docx_tool.py" create \
  --spec "/tmp/docx-demo/create.json" \
  --output "/tmp/docx-demo/created.docx"
"$DOCX_PYTHON" "$SKILL_ROOT/scripts/docx_tool.py" inspect \
  --input "/tmp/docx-demo/created.docx" \
  >"/tmp/docx-demo/created.inspect.json"
"$DOCX_PYTHON" "$SKILL_ROOT/scripts/docx_tool.py" edit \
  --input "/tmp/docx-demo/created.docx" \
  --spec "/tmp/docx-demo/edit.json" \
  --output "/tmp/docx-demo/approved.docx"
"$DOCX_PYTHON" "$SKILL_ROOT/scripts/docx_tool.py" convert \
  --input "/tmp/docx-demo/approved.docx" \
  --format text \
  --output "/tmp/docx-demo/approved.txt"
"$DOCX_PYTHON" "$SKILL_ROOT/scripts/docx_tool.py" inspect \
  --input "/tmp/docx-demo/approved.docx" \
  >"/tmp/docx-demo/approved.inspect.json"
grep -q '"status": "ok"' "/tmp/docx-demo/approved.inspect.json"
grep -q 'Status: Approved' "/tmp/docx-demo/approved.txt"
test -s "/tmp/docx-demo/approved.docx"
```

Expected artifact assertions:

- `created.docx` and `approved.docx` have ZIP signatures and reopen.
- `created.docx` remains byte-for-byte unchanged by the edit command.
- `approved.docx` contains one heading, the `Quote` paragraph, one table, a header, a footer,
  and a PAGE field.
- `Status: Approved` uses the first matched run's formatting because the match crossed runs
  with an explicit `first_run` policy.
- `approved.txt` contains `Release brief`, `Status: Approved`, and the tab-delimited table.
- Both inspection summaries report `verification.preflight` and `verification.reopened` as
  true.

## Representative stdout

Exact counts vary with templates. A successful create emits this shape:

```json
{
  "counts": {
    "footers": 1,
    "headers": 1,
    "heading": 1,
    "metadata": 2,
    "paragraph": 1,
    "table": 1
  },
  "operation": "create",
  "paths": {
    "output": "/tmp/docx-demo/created.docx",
    "template": null
  },
  "preservation": {
    "perfect_round_trip_guaranteed": false,
    "uncertainty": [],
    "unknown_untouched_content": "preserved_best_effort"
  },
  "result": {
    "inline_images": 0,
    "paragraphs": 2,
    "sections": 1,
    "tables": 1
  },
  "schema_version": 1,
  "status": "ok",
  "verification": {
    "atomic_publish": true,
    "preflight": true,
    "reopened": true
  },
  "versions": {
    "libreoffice": null,
    "lxml": "6.1.1",
    "pypdf": "6.14.2",
    "python": "3.11.x",
    "python-docx": "1.2.0"
  },
  "warnings": [
    {
      "code": "fields_present",
      "count": 1,
      "message": "Fields require a layout application to update."
    }
  ]
}
```

The actual envelope also includes package byte/member verification. Warning diagnostics are
repeated as JSON on stderr.

## Complete job dispatch

Create `/tmp/docx-demo/inspect-job.json`:

```json
{
  "schema_version": 1,
  "operation": "inspect",
  "input": "approved.docx"
}
```

Run:

```bash
"$DOCX_PYTHON" "$SKILL_ROOT/scripts/docx_tool.py" \
  --job "/tmp/docx-demo/inspect-job.json"
```

Create a document with a TOC field:

```json
{
  "schema_version": 1,
  "operation": "create",
  "output": "toc-document.docx",
  "content": {
    "blocks": [
      {"type": "heading", "level": 1, "text": "Contents"},
      {"type": "field", "field": "toc"},
      {"type": "heading", "level": 1, "text": "Introduction"},
      {"type": "paragraph", "text": "Body text."}
    ]
  }
}
```

The TOC is field markup with a placeholder. Open the file in a layout application and update
the field before judging entries or pagination.

## Explicit paragraph append

Paragraph insertion never infers append from a missing index. Use `position: "append"`
explicitly and omit `paragraph_index`:

```json
{
  "schema_version": 1,
  "operations": [
    {
      "type": "insert_paragraph",
      "position": "append",
      "block": {"type": "paragraph", "text": "Appended intentionally."}
    }
  ]
}
```

For relative insertion, use `position: "before"` or `"after"` and supply
`paragraph_index`. The index must exist in the top-level paragraph list before insertion; an
index equal to the old paragraph count is not an append alias.

## External-relationship opt-in

External relationships are rejected by default. After reviewing and accepting the package,
an inspect command can opt in explicitly:

```bash
"$DOCX_PYTHON" "$SKILL_ROOT/scripts/docx_tool.py" inspect \
  --input "/tmp/docx-demo/reviewed-external.docx" \
  --allow-external-relationships
```

The equivalent complete-job property is a JSON Boolean:

```json
{
  "schema_version": 1,
  "operation": "inspect",
  "input": "reviewed-external.docx",
  "allow_external_relationships": true
}
```

The result includes an `external_relationships_allowed` warning. The opt-in permits
processing but does not network-isolate LibreOffice or another consumer; those applications
may access external targets. A string such as `"true"` is rejected as `bad_input`.

## Corrupt input

```bash
printf 'not OOXML' >"/tmp/docx-demo/corrupt.docx"
set +e
"$DOCX_PYTHON" "$SKILL_ROOT/scripts/docx_tool.py" inspect \
  --input "/tmp/docx-demo/corrupt.docx"
status=$?
set -e
test "$status" -eq 2
```

Stderr:

```json
{"error":{"category":"bad_input","details":{"path":"/tmp/docx-demo/corrupt.docx"},"message":"Input does not have a ZIP/OOXML signature."},"schema_version":1,"status":"error"}
```

No result is written to stdout. Encrypted, macro-enabled, malformed, path-unsafe, and
oversized packages likewise fail before `python-docx` opens them; category/status can differ
according to [the exit table](references/operations.md#exit-statuses).

## Optional PDF conversion

LibreOffice is optional and needed only for DOCX-to-PDF conversion. Check whether `soffice`
is already available:

```bash
command -v soffice
soffice --version
```

If it is absent and PDF conversion is required, follow the consent-first, platform-neutral
installation guidance in [operations](references/operations.md#environment). After
`soffice --version` succeeds:

```bash
"$DOCX_PYTHON" "$SKILL_ROOT/scripts/docx_tool.py" convert \
  --input "/tmp/docx-demo/approved.docx" \
  --format pdf \
  --output "/tmp/docx-demo/approved.pdf" \
  --timeout 90
```

Assert the signature and reopen through the smoke test:

```bash
"$DOCX_PYTHON" "$SKILL_ROOT/scripts/smoke_test.py" --require-libreoffice
```

PDF conversion is layout-engine dependent. The tool rejects generated PDFs over 64 MiB or
1,000 pages and limits text validation to the first 50 pages and 1,000,000 characters.
Validate content anchors within those bounds and visually render the actual document; bounded
structural validation does not prove layout correctness.
