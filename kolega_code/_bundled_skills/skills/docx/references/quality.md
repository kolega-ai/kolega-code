# Quality contract

## Contents

- [Design intake and templates](#design-intake-and-templates)
- [Delivery profiles](#delivery-profiles)
- [Acceptance criteria](#acceptance-criteria)
- [Visual review versus structural checks](#visual-review-versus-structural-checks)

## Design intake and templates

Record this profile before polished creation:

| Decision | Required intake |
|---|---|
| Audience | Primary readers, reading context, and accessibility needs |
| Document type | Report, brief, letter, manual, form, publication, or other |
| Page | Letter or A4; note any landscape sections |
| Renderer | Microsoft Word version/platform, LibreOffice, Word Online, or other target |
| Editable fidelity | Low, medium, or high importance for reflow and editability |
| Brand/template | Supplied template, approved template, brand rules, or explicitly unbranded |
| Accessibility | Requested standard/accommodations, language, alt-text ownership |
| Medium | Print, screen, or both; monochrome and duplex requirements |
| TOC | Requested or complexity-triggered; proposed depth |
| PDF | None, basic viewing, print, accessibility-oriented, or archive-oriented |

Use a supplied or user-approved template for branded, client-facing publication, regulated
layout, or publication-ready output. Preflight it before use and preserve its theme, styles,
sections, and stories unless the brief calls for bounded changes. If a template contains
placeholder text, inspect and replace/insert at those anchors with expected counts; appending
new content to the end does not fill a template.

If no approved template is required, create a deliberate baseline through `content.section`
and `content.styles`: establish page geometry, Normal/body typography, heading hierarchy,
caption/table treatment, spacing, and pagination behavior. Record the renderer and font
assumptions. A named font is portable only when licensed and embedded in the DOCX or installed
on every intended renderer.

## Delivery profiles

Profiles describe checks performed, not certifications:

- **Basic viewing:** editable DOCX opens, intended content and hierarchy are present, and
  representative rendering is legible on the named renderer. No print, accessibility, or
  archive claim.
- **Print:** basic viewing plus correct paper size, margins, printable widths, page breaks,
  headers/footers, numbering, image resolution, color/monochrome behavior, and page-by-page
  review of the final print PDF.
- **Accessibility-oriented:** basic viewing plus metadata language/title, logical heading
  hierarchy, meaningful link text where applicable, informative-image alt text, decorative
  marking, captions, table header rows, readable order, contrast, and keyboard/native-reader
  checks in the target Word application. The CLI does **not** validate tagged PDF.
- **Archive-oriented:** preserve the editable DOCX and provenance, use stable documented fonts
  and fields, and produce an ordinary PDF only if requested. The CLI does **not** create or
  validate PDF/A; call output “PDF/A” only after a separate conforming tool validates the
  required level.

Combine profiles when requested. State renderer, template, unresolved warnings, and any check
not performed. Never claim tagged PDF, PDF/UA, PDF/A, WCAG, or identical cross-renderer
pagination from this CLI alone.

## Acceptance criteria

Apply the rows relevant to the delivery profile. A failed required row blocks release unless
the user explicitly accepts the named exception.

| Area | Measurable acceptance |
|---|---|
| Template preflight | Template passes CLI preflight/reopen; external relationships, signatures, revisions/comments, unsupported parts, fonts, sections, stories, and placeholder anchors are reviewed before mutation. Branded/publication-ready output uses a supplied or approved template. |
| Style hierarchy and typography | Body, title, headings, captions, lists, and tables use named styles; heading levels do not skip; direct formatting is limited to intentional exceptions. Font inventory has no unexplained references. Unembedded fonts are confirmed installed on every intended renderer or replaced; embedded fonts are license-approved. |
| Section and page geometry | Every section's inspected paper size/orientation/dimensions/margins/header/footer distances match the intake. Usable width equals page width minus left and right margins. Landscape and numbering changes start in the intended section. |
| Paragraph pagination | Headings use `keep_with_next`; captions stay with figures; short atomic blocks use `keep_together`; body text has widow control when required; page-break-before and explicit breaks are intentional. Rendered pages have no orphan headings, stranded captions, avoidable widows/orphans, or accidental blank pages. |
| Paragraph rhythm | Named styles define consistent before/after spacing and line spacing; manual blank paragraphs are not used as layout. Tabs use declared positions/alignment/leaders rather than spaces. |
| Tables | Data tables have at least one repeated header row. For each table, `sum(column_widths_inches) <= section.page_width_inches - left_margin_inches - right_margin_inches` (allow at most 0.01 inch inspection tolerance). Fixed widths are used when predictable print layout is required. Row splitting is disabled only for rows expected to fit on one page; inspect per-row parity after updates. Render confirms no clipping, overflow, unreadably narrow columns, or detached continuation headers. |
| Images and figures | Every informative image has non-empty alt text; every purely decorative image is marked decorative and has no alt text. Figures have captions when needed for identification/reference and attribution when required. Display width fits the section usable width. Effective PPI is checked in both axes: below 150 is a release warning; target at least 300 PPI for high-quality print unless source constraints are accepted. No unintended stretching or cropping appears in render. |
| Accessibility | Metadata includes title and BCP-47-style language appropriate to the document; headings form a logical outline; table headers, reading order, contrast, link meaning, image semantics, and native accessibility checker results meet the requested profile. Automated warnings are triage, not certification. |
| TOC and fields | Include a TOC only when requested or when navigation complexity warrants it (normally multi-section documents with several heading levels or roughly 10+ pages). Use live fields for editable pagination. Inspect field type/instruction/result/dirty/locked state; refresh in the target layout application and confirm the TOC is populated and references are correct. |
| Headers, footers, numbering | Story kind, section index, first/even-page settings, and `linked_to_previous` match the design. Page-number fields and `page_number_start`/format render correctly, including intentional restarts; no empty or duplicated inherited stories appear. |
| Font portability | Do not infer portability from font names or PDF embedding. Licensed DOCX-embedded fonts are allowed. The font inventory carries zero unaccepted `RELEASE BLOCKER` warnings; `common_unembedded_fonts` (Office-bundled families) is acceptable when the audience is Office-equipped or the user accepts the substitution risk. Every other unembedded font must exist on every intended renderer; otherwise replace it or disclose reflow risk. Compare wrapping and pagination on each fidelity-critical renderer. |
| Visual QA | Run `render` and open every produced page with the Read tool, reviewing each final deliverable at normal view and fit-page/print preview. Confirm correct paper/orientation, margins, hierarchy, line/table wrapping, image placement, captions, headers/footers, numbering, fields/TOC, no clipping/overlap, no accidental blank or nearly blank pages, and no unexplained reflow from the approved reference. |

## Visual review versus structural checks

`inspect` verifies package structure and exposes measurable DOCX properties. PDF
`verification.content_quality_report` verifies signature/reopen bounds and reports extracted
text anchors, page dimensions, low-text/nearly-blank heuristics, and limited XObject presence.
These are **structural content checks**, not pixels: they cannot detect overlap, clipping,
font substitution, bad wrapping, contrast, visual hierarchy, or whether a page looks correct.

Real visual review means producing pixels and looking at them. The bundled `render` command
(see [render](references/operations.md#render)) is the built-in path: it publishes one PNG per
page for review with the Read tool. The PNGs show LibreOffice's interpretation; the target
application (normally Microsoft Word) remains the final authority for fidelity-critical
deliverables. If neither is possible, report “structural checks only; visual QA not
performed” and do not make layout, print, accessibility, or archive-conformance claims.
