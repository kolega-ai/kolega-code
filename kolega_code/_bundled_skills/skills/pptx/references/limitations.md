# PPTX limitations

## Contents

- [Round-trip fidelity](#round-trip-fidelity)
- [Editing boundaries](#editing-boundaries)
- [Conversion boundaries](#conversion-boundaries)
- [Future work](#future-work)

## Round-trip fidelity

PresentationML permits features that `python-pptx` does not model completely. Saving a deck can
change or discard unsupported details even when the corresponding shape was not edited. Treat the
following as preservation warnings rather than supported content:

- SmartArt and diagram semantics;
- audio, video, embedded packages, OLE objects, and linked media;
- animations, transitions, timing trees, and action behavior;
- comments, threaded comments, ink, custom XML, and extension lists;
- unsupported shape effects, three-dimensional formatting, and advanced chart extensions;
- notes-page layout beyond plain notes text;
- digital signatures and signature assurances.

The tool inventories recognizable package parts but does not prove lossless preservation. Rewriting
a signed file invalidates signature assurances. It rejects macros rather than attempting to retain
or execute them.

## Editing boundaries

- Creation is template-first. It uses existing masters/layouts/placeholders but does not rewrite
  arbitrary themes, masters, or layout XML.
- Layout names must match exactly and uniquely. Indexes are stable only for the specific template.
- Text replacement does not infer intent. Cross-run formatting needs an explicit policy. Destructive
  reconstruction flattens formatting, fields, and hyperlinks in the affected text frame.
- Hyperlink editing sets or removes one external `http`, `https`, or `mailto` address on whole
  runs or on a shape click action. It never splits runs, and internal slide jumps, hover
  actions, tooltips, and `ppaction` behaviors are not authored.
- Image replacement requires the same media content type. It does not change crop, geometry, or
  aspect settings.
- Structural table edits insert or remove whole rows and columns only. Tables containing merged
  cells are rejected for structural edits, and merged-cell authoring and advanced table styles
  remain unsupported. New rows and columns clone neighbor formatting with cleared text, and the
  table's overall width or height grows or shrinks rather than redistributing existing sizes.
- Chart updates replace the complete cached/native data payload supported by `python-pptx`. Complex
  combination charts, secondary axes, trendlines, error bars, data labels, external workbook links,
  and chart extensions are not preserved by a data replacement guarantee.
- Slide cloning across unrelated masters is not supported. `add_slide` creates content against a
  layout already present in the destination.
- Shape z-order editing, grouping/ungrouping, connector repair, and arbitrary relationship editing
  are outside the interface.
- Extraction exports picture shapes only. Shape-fill images, slide backgrounds, and audio/video
  media are not exported, and extracted bytes are copied verbatim without decode validation.

Slide removal/reordering and table row/column mutation use a small adapter constrained to the
supported `python-pptx` range because the library has no public mutation API for those
operations. Every slide-graph mutation is checkpointed and reopened, and every table structure
change is verified against an expected text matrix, but structural verification cannot prove
rendering equivalence.

## Conversion boundaries

LibreOffice conversion is optional and best effort. Presentations with external relationships are
rejected rather than passed to LibreOffice. Results otherwise vary with:

- installed fonts and font substitution;
- LibreOffice version and platform rendering;
- unavailable codecs, media, linked resources, or embedded objects;
- chart/effect support and line-breaking differences;
- notes, comments, hidden slides, and print/export configuration.

The tool proves PDF signature, openability, and page count only. Text extraction anchors used by the
opt-in smoke test are not a general visual-fidelity test. Keep the source presentation and visually
review important exports.

`render` shares every LibreOffice variance above because its PNGs are rasterized from the same
LibreOffice PDF. PNG verification proves signature, openability, page count, and dimensions —
not renderer-identical pixels. Rendered slides support layout review, not a PowerPoint-fidelity
claim.

LibreOffice can visually wrap text even when the frame's serialized wrapping setting is false,
while PowerPoint may honor that setting and let the line overflow horizontally. The tool
explicitly enables wrapping for text frames it authors, but imported frames can retain either
setting. Use `inspect` and check `text.word_wrap` rather than treating a LibreOffice render as
proof of PowerPoint wrapping. Wrapping can increase vertical text usage, so box height and
vertical overflow still require visual review.

The font inventory over-approximates: fonts referenced only by unused layouts, masters, notes or
handout masters, or table styles still count as referenced because they ship with the deck and
re-engage as soon as a layout is used. Script-specific theme fonts (`a:font script="..."`), VML
text, chart-style (`cs:`) parts, embedded chart workbooks, and renderer fallback chains are not
scanned. Symbol and bullet fonts (`a:sym`, `a:buFont`) are inventoried under
`fonts.symbol_and_bullet` but excluded from the release blocker because substitution changes a
glyph, not text layout. Embedding detection proves a resolvable in-package font part exists, not
that its payload is a valid, complete, or licensed font; obfuscated embedded fonts are not parsed.

## Future work

The initial release does not provide:

- arbitrary theme/master/layout editing;
- lossless cloning between unrelated presentations;
- complete unsupported-content preservation;
- animation, transition, media, SmartArt, comment, or notes-page-layout editing;
- digital signing, signature preservation, macro/ActiveX handling, or password decryption;
- renderer-identical PDF output or pixel-diff validation;
- exhaustive chart-type and table-style editing.
