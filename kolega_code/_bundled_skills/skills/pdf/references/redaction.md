# Redaction policy and verification contract

## Contents

- [What redact does](#what-redact-does)
- [Leakage policy table](#leakage-policy-table)
- [Verification contract](#verification-contract)
- [Redacting scanned pages](#redacting-scanned-pages)
- [What redact does not guarantee](#what-redact-does-not-guarantee)

## What redact does

`redact` interprets each targeted page's content stream with a bounded graphics-state
machine: it tracks the transformation and text matrices, measures every text-showing
operator's extent from the font's width tables (conservatively over-approximating when a
width is missing), and removes any operator whose extent intersects a target rectangle.
Removed text operators are replaced by advance-preserving no-ops so retained text does not
shift; intersecting images, inline images, vector paints, and annotations are removed
whole. The document is then rewritten as a single revision (dropping prior incremental
revisions and orphaned objects) and an opaque rectangle is drawn over each target region as
visual confirmation — the rectangle is confirmation, not the mechanism.

The whole-operator policy is deliberate: any overlap removes the entire show operator, so
text adjacent to the target inside the same operator is removed with it. The report counts
this collateral and the verification rasters make it visible. Prefer tight targets;
`grow_points` (default 2) trades collateral for margin against sub-glyph leakage.

## Leakage policy table

Every known channel is handled, or detected and refused — never silently ignored.

| Channel | v1 policy |
| --- | --- |
| Text operators (`Tj`/`TJ`/`'`/`"`) | **Handled**: whole-operator removal on any overlap, conservative extents from width tables |
| Unmeasurable glyph widths (broken or exotic font dictionaries) | Extent detection over-approximates via the font bounding box; an intersecting operator whose widths cannot be measured is **refused** rather than guessed |
| Type3 fonts shown inside a target | **Refused** — glyph procedures are content streams with unmeasurable extents |
| Text-clipping render modes (Tr 4–7) inside a target | **Refused** — rewriting text clips changes later content |
| Inline images (`BI`/`ID`/`EI`) | **Handled**: removed whole on any overlap |
| Image XObjects | **Handled**: removed whole (default) or the run is refused under `image_policy: refuse`; removed streams are verified unreachable in the output |
| Form XObjects whose transformed `/BBox` intersects | **Refused** — their inner streams can carry text and may be shared across pages; flatten or rasterize first |
| Vector paths painted inside a target | **Handled**: the painting operator becomes a no-op (`n`) so path data cannot be painted later; clipping paths are preserved |
| Optional-content (OCG) hidden text | **Handled implicitly**: the interpreter ignores visibility, so hidden operators in the region are removed; `BDC`/`EMC` stay balanced |
| Annotations intersecting a target | **Handled**: the whole annotation (including appearance streams) is removed |
| Annotations elsewhere containing a target term | **Handled**: the term sweep removes them and reports it |
| Document information (DocInfo) | **Handled** per `strip_docinfo_keys` (`matched` default); a term hit under `none` is refused |
| XMP metadata | **Handled** per `strip_xmp` (`if-matched` default); a term hit under `never` is refused |
| Page thumbnails (`/Thumb`) | **Handled**: stripped on every page by default |
| Structure tree (`/ActualText`, `/Alt`) | Term hits are **refused** unless `strip_structure_tree: true`, which removes the whole structure tree (and its accessibility tagging) |
| Outlines and named destinations containing a term | **Refused** — rewrite them before redacting |
| Form field values containing a term | **Refused** — clear them with an `edit` `fill_form` operation first |
| Embedded files / attachments | **Detected**; requires `acknowledge.attachments_present` — attachment contents are not scanned |
| Digital signatures (`/SigFlags`, signed `/Sig` fields) | **Detected**; requires `acknowledge.signatures_invalidated` because rewriting invalidates signatures |
| XFA forms | **Refused** — the XFA XML can duplicate page content |
| Prior incremental revisions, orphan objects, object streams | **Handled** by the single-revision rewrite; verified (exactly one `%%EOF`, stale bytes absent) |
| Encrypted input | **Refused** in v1 — decrypt explicitly with `edit` first |
| JavaScript / launch actions | **Detected and warned** elsewhere in the skill; their code is not term-scanned |

## Verification contract

All checks run on the staged output before publication; any failure exits
`validation_failed` and publishes nothing.

1. **Region extraction:** zero extractable characters inside every target rectangle
   (0.5-point inset tolerance).
2. **Term extraction:** every text target is unextractable on its selected pages.
3. **Residual byte scan:** the raw file and every decompressed stream are scanned for
   UTF-8, UTF-16BE, and Latin-1 encodings of each term (case variants included). A hit for
   a term whose scope was every page is a hard failure; hits for page-scoped terms are
   reported because instances outside the selected pages were intentionally retained. The
   scan cannot enumerate every conceivable glyph-code encoding of text it never removed —
   the extraction checks above are the primary gate.
4. **Image reachability:** the decoded bytes of every removed image are verified absent
   from all streams in the output.
5. **Structure:** page count and page dimensions unchanged; exactly one `%%EOF`.
6. **Visual diff:** before/after rasters of every affected page; pixel changes outside the
   union of target regions and removed-content extents fail the run. `--render-check-dir`
   publishes these rasters — open them and confirm the black boxes sit exactly where
   intended.

The report (`--report`) records resolved rectangles, per-page removal evidence, sweep
results, and the residual scan. Text targets appear as SHA-256 digests only; the report is
designed not to become a leak channel itself.

## Redacting scanned pages

Text targets match extractable digital text only — on `likely-scanned` pages the text is
pixels and cannot be matched. Redact scanned content with `rect` targets. To locate the
rectangles, run OCR and convert the sidecar's word geometry to displayed points: each
bbox coordinate times `72 / coordinate_space.dpi`, plus the `source_region` offset when the
page was region-rendered. The intersecting page image is then removed whole (v1 has no
partial image masking), so re-add retained imagery separately if needed.

## What redact does not guarantee

- No protection against prior copies, backups, version-control history, filesystem
  journaling or slack space, swap, or anything outside this one output file.
- Attachment contents are not scanned; JavaScript is not term-scanned.
- Whole-operator removal deletes adjacent text inside the same show operator; the report
  quantifies it, but content integrity immediately around the target is not promised.
- Removing an intersecting page image removes the whole image, which can blank a full
  scanned page; the visual check makes this obvious rather than preventing it.
- Rewriting invalidates digital signatures and drops incremental history by design.
- `strip_structure_tree` removes accessibility tagging for the whole document, not just the
  matched entries.
