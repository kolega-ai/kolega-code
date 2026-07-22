---
name: pptx
description: Create, inspect, edit, and convert PowerPoint presentations (`.pptx`) with template-first layouts, structured text, images, image extraction, editable tables and charts, speaker notes, hyperlinks, and guarded slide operations. Use for presentation-native work such as building a deck from an outline or template, inventorying slide content, replacing text or media, updating tables/charts/notes, adding or removing table rows and columns, setting or removing hyperlinks, adding/removing/reordering slides, or converting PPTX to PDF. Do not use for PDF-native editing or for spreadsheets and word-processing documents.
license: Apache-2.0
compatibility: Requires Python 3.11+ and dependencies from requirements.txt. LibreOffice and the pypdfium2 Python package are optional, needed only for PPTX-to-PDF conversion and slide rendering respectively.
metadata:
  author: Kolega
  version: "1.0"
---

# PPTX presentations

Use `scripts/pptx_tool.py` for deterministic presentation work. Keep source files unchanged,
write to a distinct destination, and use `--overwrite` only when replacement is intentional.

## Workspace discipline

Work directly in visible workspace paths. Never create or use `.build` or another hidden
build/work directory. Write requested deliverables directly to their declared destinations;
if intermediate files are necessary, keep them in visible, narrowly scoped paths rather than
staging the task in a hidden subtree.

## Workflow

1. **Prepare the runtime.** Resolve the skill root and choose any available Python 3.11+
   interpreter; do not assume a launcher name. Check the required imports before installing
   anything. If something is missing, first tell the user what you intend to install, where,
and with which installer. Use the selected interpreter's `-m pip` for the declared
   requirements. Use the platform's package manager for Python or optional LibreOffice,
   preferring Homebrew on macOS when available. A local environment is a fallback, not a
   prerequisite. See
   [runtime installation](references/operations.md#runtime-prerequisite-and-installation).
2. **Inspect first.** Run `inspect` before editing. Review slide IDs and order, layout names,
   placeholders, geometry, runs, images, tables, charts, notes, and preservation warnings. Use
   `extract` when the actual image bytes are needed outside the deck; it writes picture-shape
   images to a new directory and refuses to overwrite non-empty destinations.
3. **Choose a template.** For branded decks, start `create` from a supplied `.pptx` and select
   its layouts by exact name or index. Prefer existing masters, layouts, and placeholders over
   direct formatting. If `layout` is omitted, index 1 is used. Do not attempt arbitrary theme or
   master rewrites.
4. **Prepare a schema-version-1 job.** Use `create` for a complete deck and `edit` for explicit
   operations. Read [operations](references/operations.md) for the job contract and safety
   limits. Read [examples](references/examples.md) for copy-pasteable jobs.
5. **Make edits narrowly.** Select slides by stable slide ID when possible and shapes by exact
   name. Treat multiple matches as ambiguous. Supply `formatting_policy: "first_run"` for an
   intentional cross-run replacement, or `destructive_reconstruction: true` only when flattening
   the affected text frame is acceptable.
6. **Verify the result.** Require the successful JSON summary, package preflight, reopen check,
   expected slide IDs/order/count, existing internal relationship targets, and resolved owner
   relationship references. Inspect the output again; when visual fidelity matters, continue
   with step 7 and treat the target presentation application as the final authority. Review the
   font inventory (`fonts.referenced`,
   `fonts.embedded`, `fonts.unembedded`). For PPTX+PDF deliverables apply this release gate: use
   only Arial for headings and sans body, Times New Roman for serif body, and Courier New for
   code or logs, including table and chart text. Do not use Avenir, Calibri, Calibri Light,
   Cambria, Georgia, Trebuchet, Palatino, or any other unembedded custom/system font; replace
   every `RELEASE BLOCKER` font warning before release. Decks built on the default Office theme
   report Calibri through `+mj-lt`/`+mn-lt` until the template's theme fonts are changed or
   genuinely embedded. PDF font embedding is not PPTX font embedding. A clean LibreOffice PDF
   alone does not prove that PowerPoint will preserve wrapping, autofit, or slide layout.
7. **Render and look.** When layout or visual fidelity matters and LibreOffice plus the optional
   `pypdfium2` package are available, run `render` to produce one PNG per slide, then open the
   rendered PNGs with the Read tool and examine them — at minimum every slide you changed, plus
   the first slide. Check for text overflow or truncation, overlapping or clipped shapes,
   missing images, and table or chart layout problems. Do not claim visual correctness from
   JSON verification alone. Rendering shows LibreOffice's interpretation, not
   renderer-identical PowerPoint output; the target presentation application remains the final
   authority. If `pypdfium2` is missing, tell the user before installing it with the selected
   interpreter's `-m pip`.
8. **Convert optionally.** PPTX-to-PDF remains owned by this skill. When conversion is requested,
   verify `soffice` or `libreoffice` on `PATH`. If it is missing, first tell the user the intended
   optional system install and mechanism, then use the platform's normal package manager,
   preferring Homebrew on macOS when available. Use `convert` only when LibreOffice is available
   and inspection reports no external relationships. The command uses a minimal environment, an
   isolated profile, and sanitized diagnostics; it validates PDF openability and page count but
   cannot prove renderer-identical layout. Read
   [operations](references/operations.md#libreoffice-prerequisite) for installation and PATH
   verification.

## Safety rules

- Accept `.pptx` only. Reject macro-enabled, encrypted, malformed, oversized, or suspicious ZIP
  packages.
- Never modify a source path in place. Outputs are staged beside the destination, reopened,
  structurally verified, and atomically published.
- Keep all inputs local. The tool does not fetch network resources or resolve XML entities.
- Do not log secrets. CLI results and failures are JSON; nonzero statuses identify stable failure
  categories.
- Treat digital signatures, unsupported effects, SmartArt, media, animations, transitions,
  comments, and renderer-specific fonts as preservation risks.

## Resources

- Read [operations](references/operations.md) for commands, schemas, selectors, exit statuses,
  and verification behavior.
- Read [examples](references/examples.md) before authoring a new job or diagnosing a failure.
- Read [limitations](references/limitations.md) before promising fidelity or editing unsupported
  presentation features.
- Run `"$PPTX_PYTHON" "$SKILL_ROOT/scripts/smoke_test.py"` after installation. Add
  `--require-libreoffice` only when PDF conversion must also be exercised.

## Final verification

Confirm that the source remains unchanged unless overwrite was authorized, the destination
reopens, requested structures are present, every unembedded-font warning was resolved or
explicitly reviewed, and no temporary files or generated fixtures remain in the skill directory.
For matching PPTX/PDF deliverables, treat the `RELEASE BLOCKER` font warning as a release
blocker; confirm typography, table and chart text, and slide layout in both files before
trusting the export. `pdffonts` can inspect only the PDF and must never be cited as evidence
that the PPTX embeds or preserves those fonts.
