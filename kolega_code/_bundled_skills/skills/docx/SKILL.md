---
name: docx
description: Create, inspect, edit, extract text from, convert, and render Microsoft Word DOCX documents while preserving formatting and validating OOXML safely. Use for .docx authoring, reading, structured edits, templates or letterheads, tables, images, headers, footers, fields, DOCX-to-text, DOCX-to-PDF, and page-rendering-for-review requests. Do not use for PDF-native, spreadsheet, or presentation work.
license: Apache-2.0
compatibility: Requires Python 3.11+ and the packages declared in requirements.txt. LibreOffice and the optional pypdfium2 package are needed only for PDF conversion and page rendering.
metadata:
  author: Kolega
  version: "1.1"
---

# DOCX

Use the bundled deterministic CLI. Treat sources as untrusted and immutable unless overwrite
is explicitly authorized.

## Workspace discipline

Work directly in visible workspace paths. Never create or use `.build` or another hidden
build/work directory. Write requested deliverables directly to their declared destinations;
if intermediate files are necessary, keep them in visible, narrowly scoped paths rather than
staging the task in a hidden subtree.

## Route the task

- Use this skill for DOCX creation, inspection, bounded editing, text extraction,
  DOCX-to-PDF conversion, and page rendering for visual review. Route PDF-native work to
  `pdf`, spreadsheets to `xlsx`, and presentations to `pptx`.
- For creation, read [design intake and templates](references/quality.md#design-intake-and-templates),
  then the [create contract](references/operations.md#create-schema). Use a supplied or approved
  template for branded or publication-ready output. Otherwise build a deliberate baseline with
  `section` and `styles`; do not accept the library defaults as polished design.
- For existing documents, inspect first using [inspection](references/operations.md#inspect-schema),
  then use [edit operations](references/operations.md#edit-operations). A placeholder template is content
  to fill with bounded replacement/insertion, not a blank base to append after.
- For fields, headers, footers, sections, tables, or images, jump to
  [content blocks](references/operations.md#content-blocks),
  [headers and footers](references/operations.md#headers-and-footers), or
  [edit operations](references/operations.md#edit-operations).
- For conversion and release, use [conversion](references/operations.md#conversion-contract) and apply the
  one authoritative [quality contract](references/quality.md#acceptance-criteria) according to the
  requested [delivery profile](references/quality.md#delivery-profiles).
- Before advanced or fidelity-sensitive work, read
  [limitations](references/limitations.md#remaining-limits). Activation preserves formatting
  best-effort; route signed documents, tracked-change resolution, revision-heavy editing, and
  fidelity-critical mutation to a native Microsoft Word workflow.
- Use [examples by task](references/examples.md#example-index) for executable jobs.

## Core workflow

1. Resolve the skill root and select Python 3.11+. Follow
   [environment setup](references/operations.md#environment); never install without first
   stating what, where, and with which installer.
2. Capture the design intake and delivery profile. Preflight any template before authoring.
3. Inspect before editing. Review warnings, ordered blocks, styles, sections, stories, fields,
   tables, images, fonts, revisions, comments, and unsupported parts.
4. Prefer template inheritance and named styles over direct formatting. Use only the public
   CLI schemas in [operations](references/operations.md#invocation-and-envelopes).
5. Keep edits explicit and bounded. Write to a distinct destination unless overwrite was
   explicitly authorized.
6. Reinspect the output and run the profile-dependent checks in
   [quality](references/quality.md#acceptance-criteria). Resolve every font warning: a
   `RELEASE BLOCKER` blocks release until the font is replaced, embedded, or the user
   explicitly accepts the named substitution risk.
7. **Render and look.** Whenever layout, pagination, print, or PDF matters, run `render`
   (see [render](references/operations.md#render)), open the produced PNG pages with the
   Read tool, and examine every changed page plus page 1 for text overflow or truncation,
   clipped or overlapping content, missing images, broken tables, and header/footer and
   page-number rendering. Inspect-only PDF metrics are not visual review; never claim visual
   correctness from JSON output alone. If rendering is unavailable, report "structural
   checks only; visual QA not performed." Announce before installing the optional
   `pypdfium2` package.
8. Report warnings, renderer assumptions, field-refresh status, and preservation uncertainty.
   Never promise perfect round-trip fidelity.

## Safety rules

- Do not bypass OOXML preflight, ZIP/resource limits, signature checks, external-relationship
  policy, or post-write reopen validation.
- Never use `sudo pip`, an interpreter under `sudo` for pip, or `--break-system-packages`;
  do not add skill dependencies to the host project's dependency metadata.
- Reject macro-enabled, encrypted, malformed, or oversized packages. Reject external
  relationships by default; explicit opt-in permits processing but is not a network sandbox.
- Supply expected replacement counts. Do not flatten text across hyperlinks, fields, drawings,
  comments, revisions, or unsupported markup.
- Page, TOC, reference, sequence, and date fields are markup only. A layout application must
  refresh them; `update_fields_on_open` is a request, not a guarantee.
- LibreOffice's isolated profile and process-group cleanup are lifecycle controls, not network
  isolation.

## Resources

- [Operations contract](references/operations.md#operations-contract)
- [Executable examples](references/examples.md#examples)
- [Quality contract](references/quality.md#quality-contract)
- [Remaining limits](references/limitations.md#remaining-limits)
- Production CLI: [`scripts/docx_tool.py`](scripts/docx_tool.py)
- Environment smoke test: [`scripts/smoke_test.py`](scripts/smoke_test.py)

Finish by confirming the source was not changed without authorization, the destination
reopens, requested structures are present, the applicable
[acceptance criteria](references/quality.md#acceptance-criteria) passed, no unaccepted
`RELEASE BLOCKER` font warning remains, rendered pages were opened and reviewed (or the
structural-only disclaimer was reported), and no generated fixtures remain in the skill
directory.
