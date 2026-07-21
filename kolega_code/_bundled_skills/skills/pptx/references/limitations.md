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
- Image replacement requires the same media content type. It does not change crop, geometry, or
  aspect settings.
- Table updates do not add/delete rows or columns. Merged-cell semantics and advanced table styles
  are not authored.
- Chart updates replace the complete cached/native data payload supported by `python-pptx`. Complex
  combination charts, secondary axes, trendlines, error bars, data labels, external workbook links,
  and chart extensions are not preserved by a data replacement guarantee.
- Slide cloning across unrelated masters is not supported. `add_slide` creates content against a
  layout already present in the destination.
- Shape z-order editing, grouping/ungrouping, connector repair, and arbitrary relationship editing
  are outside the interface.

Slide removal/reordering uses a small adapter constrained to the supported `python-pptx` range because
the library has no public mutation API for those operations. Every slide-graph mutation is
checkpointed and reopened, but structural verification cannot prove rendering equivalence.

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

## Future work

The initial release does not provide:

- arbitrary theme/master/layout editing;
- lossless cloning between unrelated presentations;
- complete unsupported-content preservation;
- animation, transition, media, SmartArt, comment, or notes-page-layout editing;
- digital signing, signature preservation, macro/ActiveX handling, or password decryption;
- renderer-identical PDF output or pixel-diff validation;
- exhaustive chart-type and table-style editing.
