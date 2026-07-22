# Quality contract

## Contents

- [Design intake](#design-intake)
- [Delivery profiles](#delivery-profiles)
- [Acceptance criteria](#acceptance-criteria)
- [Visual review versus structural checks](#visual-review-versus-structural-checks)

## Design intake

Record this profile before creating or mutating a deliverable:

| Decision | Required intake |
|---|---|
| Audience | Primary readers, reading context, and assistive-technology needs |
| Deliverable type | New PDF, page assembly (merge/split/reorder), stamp/fill/metadata edit, redaction, OCR sidecar, or searchable composition |
| Source of truth | Digital-native, scanned, or hybrid — this drives the extraction, OCR, and redaction posture |
| Target viewers | Acrobat, browser built-ins, mobile viewers, print RIP, or unspecified |
| Medium | Print, screen, or both; monochrome and duplex requirements |
| Page | Letter, A4, or custom; orientation; mixed sizes |
| Fonts | Base-14 only, or embedded fonts with license approval for embedding |
| Forms | Fillable AcroForm preserved, values flattened, or none; XFA disclosed as unsupported |
| Encryption | None, user password, owner password; algorithm; who holds the passwords |
| Metadata | Title/author/language to set; removal requests (removal is not sanitization) |
| PDF profile expectation | None, basic viewing, print, accessibility-oriented, or archive-oriented |

## Delivery profiles

Profiles describe checks performed, not certifications:

- **Basic viewing:** the output reopens in strict mode, has the expected page count and
  order, named text anchors extract, and representative rendered pages are legible.
- **Print:** basic viewing plus paper size, orientation, and margins verified from inspected
  geometry; image effective PPI checked; page-by-page render review at 150 DPI or higher;
  monochrome legibility when requested.
- **Accessibility-oriented:** basic viewing plus metadata title and language, logical
  content order in extraction, and meaningful link/anchor text. The CLI does **not** author
  or validate tagged PDF; never claim tagged PDF or PDF/UA from this CLI alone.
- **Archive-oriented:** provenance recorded, every font embedded or base-14, no encryption
  unless requested. The CLI does **not** create or validate PDF/A; call output "PDF/A" only
  after a separate conforming validator passes it.

Combine profiles when requested. State unresolved warnings and any check not performed.

## Acceptance criteria

Apply the rows relevant to the delivery profile. A failed required row blocks release unless
the user explicitly accepts the named exception.

| Area | Measurable acceptance |
|---|---|
| Source discipline | Every source was inspected before mutation; sources are byte-identical afterward; the destination is distinct from every source; the JSON `verification` object was reopened and reviewed. |
| Page inventory | Output page count, order, rotation, and dimensions match the plan exactly, verified from `inspect` geometry — not assumed from the request. |
| Text anchors | Named anchor strings extract on the expected pages of the published output. |
| Font portability | The `fonts` inventory was reviewed and carries zero `RELEASE BLOCKER:` warnings. Embedded fonts are license-approved for embedding. Base-14-only output is acceptable when the metric-substitution caveat is disclosed. A `truncated` inventory is incomplete, not clean. |
| Images | Effective PPI per axis is `source_pixels / (placed_points / 72)` from inspected image records; below 150 is a release warning; target at least 300 PPI for print. No unintended stretching or cropping in render. |
| Forms | Field inventory, values, and flags match intent after edits (`edit --read-fields`); no unintended flattening; XFA presence is disclosed as unsupported. |
| Metadata | Title/author/language are as requested; removals are verified by reopening; metadata removal is never described as sanitization or redaction. |
| Encryption | Output encryption state and algorithm match the request; publishing decrypted content from an encrypted source was explicitly authorized (`allow_decrypted_output` / `--allow-decrypted-output`). |
| Stamps and watermarks | Position, opacity, and page coverage confirmed visually. Stamps, white rectangles, and crops are never presented as redaction — true removal is the `redact` command with its verification report. |
| Redaction | The `redact` report shows the expected removal evidence, a clean residual scan, and a clean out-of-region visual diff; the black boxes were confirmed in rendered pages. |
| Visual QA | `render` every changed page plus page 1 at 150 DPI or higher and open each PNG with the Read tool: no clipping, overlap, truncation, missing images, broken tables, or accidental blank pages. |

## Visual review versus structural checks

The CLI's `verification` JSON (signature, reopen, page count, encryption state) and the
inspect/extract results — including classifications, warnings, and the font inventory — are
**structural checks only**. They cannot detect overlap, clipping, glyph substitution or
tofu, z-order mistakes, or whether a page looks right. Do not cite them as visual proof.

Real visual review is `render` plus opening every relevant PNG with the Read tool. Unlike
DOCX or PPTX review through an intermediate converter, these rasters are produced from the
delivered PDF itself by PDFium, so the pixels reflect the actual artifact — a strictly
stronger guarantee, with two carve-outs: rasters are fully deterministic only when every
font is embedded (non-embedded fonts show this machine's substitutes), and no raster proves
viewer-interactive behavior (form appearance regeneration, annotation states, JavaScript,
layers, overprint, or color management).

If rendering cannot be performed, report "structural checks only; visual QA not performed"
and make no layout, print, accessibility, or archive-conformance claims.
