# Limitations

## Contents

- [Round-trip fidelity](#round-trip-fidelity)
- [Revisions and comments](#revisions-and-comments)
- [Drawings and media](#drawings-and-media)
- [Fields and layout](#fields-and-layout)
- [Conversion](#conversion)
- [Security boundaries](#security-boundaries)

## Round-trip fidelity

The tool preserves unknown untouched content best-effort where `python-docx` retains package
parts and relationships. It does not promise lossless arbitrary OOXML editing or perfect
round trips. Unsupported parts are inventoried when recognizable, but an empty warning list
is not proof that every producer-specific extension is understood.

Direct edits can change serialization, relationship ordering, generated identifiers, or
application metadata. Reinspect the output and compare rendered pages when fidelity matters.

Digital signatures are not preserved. Any package mutation invalidates a signature even if
signature parts remain. Do not use this tool where signature preservation is required.

## Revisions and comments

Tracked changes are detected and reported, not interpreted as accepted/rejected text. The
tool does not create, accept, reject, or safely edit revisions. Replacement refuses matches
that occupy or cross revision markup.

Comment parts and ranges are detected, but complete comment workflows are unsupported. The
tool does not create, reply to, resolve, or reliably renumber comments. Replacement refuses
matches that occupy or cross comment ranges.

## Drawings and media

Creation and insertion support inline raster images. Body inline-image replacement changes
the selected image relationship and can adjust its extent.

The following are not fully supported:

- Floating-image insertion or replacement.
- Text wrapping, anchors, positioning, crop, rotation, artistic effects, or grouped shapes.
- Charts, diagrams, SmartArt, ink, OLE objects, ActiveX, and embedded packages.
- Image replacement inside headers, footers, text boxes, comments, footnotes, or endnotes.
- Universal preservation of producer-specific drawing extensions.

Image replacement can leave an unused old media part in the package. This favors preserving
relationships over package compaction.

## Fields and layout

The tool inserts PAGE and TOC field markup with placeholder result text. It does not run a
pagination engine.

- Saving a PAGE field does not prove its page number.
- Saving a TOC field does not refresh entries, page numbers, links, or formatting.
- Automatic TOC refresh is unsupported.
- Update fields in a layout application, then inspect rendered output.

The inspector reports field markup but does not evaluate arbitrary field instructions.
Section, header, footer, widow/orphan, keep-with-next, font substitution, line-breaking, and
printer-metric interactions can only be judged after layout.

## Conversion

Plain-text conversion represents main-body paragraphs and tables, then non-empty
header/footer stories and their tables. Inherited stories are deduplicated by package-part
identity, not displayed text; distinct same-text stories and header/footer roles remain
distinct. It does not reproduce page layout, columns, text boxes, floating drawings, footnote
placement, revision semantics, or every field result.

DOCX-to-PDF is best-effort and requires an external LibreOffice `soffice` executable.
LibreOffice output can differ from Microsoft Word because of fonts, layout algorithms,
platform metrics, fields, external links, and unsupported features. Validation rejects PDFs
over 64 MiB or 1,000 pages, checks text on at most the first 50 pages, and rejects more than
1,000,000 extracted validation characters. Signature, reopen, bounded extraction, and
nonzero-page checks do not prove visual correctness or inspect pages beyond the extraction
limit.

If LibreOffice is unavailable, PDF conversion returns `missing_dependency`. The default smoke
test still validates all core Python workflows and reports the PDF check as skipped.

## Security boundaries

Preflight rejects macros, encrypted packages, unsafe ZIP paths, malformed XML, DTD/entity
declarations, configured size/member excesses, and DOCX external relationships by default.
The explicit `--allow-external-relationships` flag or Boolean job opt-in permits processing
after review.

These controls reduce common OOXML parser and archive risks but are not a malware scanner,
data-loss-prevention system, network sandbox, or content trust decision. Secure direct XML
parsing does not imply that LibreOffice is network-isolated. When external relationships are
allowed, LibreOffice or another consuming application may access external targets.
User-supplied templates and images remain the user's responsibility.

The standard `customXml` part set emitted by a fresh bundled `python-docx` template is not
warned about. Additional or differently shaped `customXml` parts remain fidelity-sensitive
and are reported as unsupported.

The CLI redacts credential-shaped diagnostics but cannot determine whether ordinary document
text is confidential. Store outputs and temporary environments according to the document's
data classification.

Read [the operation contract](references/operations.md) for exact safety bounds and
[examples](references/examples.md) for validation patterns.
