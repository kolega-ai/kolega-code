# Remaining limits

## Contents

- [Fidelity and styles](#fidelity-and-styles)
- [Stories and replacement](#stories-and-replacement)
- [Images and drawings](#images-and-drawings)
- [Fields and layout](#fields-and-layout)
- [Revisions, signatures, and protected workflows](#revisions-signatures-and-protected-workflows)
- [PDF and conformance](#pdf-and-conformance)
- [Security boundary](#security-boundary)

## Fidelity and styles

`python-docx` preserves unknown untouched package content only best-effort. Opening and saving
can normalize OOXML, relationships, compatibility markup, numbering, drawings, fields, themes,
or application-specific extensions. Inspect warnings before and after mutation and never
promise byte-identical or perfect round-trip fidelity.

Style inspection reports direct paragraph/character style definitions and `based_on`; it does
not compute the fully inherited/effective property cascade from document defaults, themes,
linked styles, table styles, or direct formatting. A renderer is required to confirm actual
typography and pagination. For fidelity-critical edits, use native Microsoft Word automation
or a reviewed manual Word workflow.

## Stories and replacement

Body, table-cell, header, and footer paragraphs can be inspected. `replace_text` can search
body/header/footer scopes, but story-address image replacement is body-only:
`word/document.xml#inline-N`. Header/footer inline images cannot be replaced by story address.

Replacement is paragraph-local and rejects protected boundaries. It does not cross paragraphs
or safely normalize hyperlinks, fields, drawings, comments, revisions, or unsupported markup.
Insert/delete paragraph and table indices address top-level body content only.

## Images and drawings

The tool creates and replaces inline images only. It does not create, reposition, or replace
floating/anchored drawings, text boxes, SmartArt, charts, equations, OLE objects, or other
embedded objects. Existing unsupported drawings are preservation-sensitive.

Image alt text, title, decorative state, display dimensions, pixels, and effective PPI are
reported for supported inline occurrences. This does not prove reading order, crop behavior,
contrast, semantic quality, or renderer presentation.

## Fields and layout

PAGE, NUMPAGES, SECTIONPAGES, TOC, SEQ, REF, and DATE operations write field markup and
placeholder results. `update_fields_on_open` asks a consumer to update fields, but Word,
LibreOffice, policy settings, locked fields, protected view, or automation mode may ignore it.
There is no guaranteed field refresh, pagination engine, TOC generation, or reference
resolution in the CLI.

Inspect field instructions/results after opening, refreshing, and saving in the target layout
application. Do not derive editable-DOCX pagination claims from a PDF rendered elsewhere.

## Revisions, signatures, and protected workflows

Tracked revisions and comments are detected but not interpreted. The tool does not accept,
reject, resolve, author, or safely rewrite tracked changes. Route revision-heavy documents and
tracked-change resolution to native Word.

Do not mutate digitally signed documents. Any package mutation can invalidate signatures; the
tool does not preserve, verify, or re-sign them. It also does not provide full workflows for
content controls, forms, document protection, rights management, mail merge, or macros.

## PDF and conformance

LibreOffice conversion is best-effort and may differ from Microsoft Word in font substitution,
line/table wrapping, page breaks, fields, drawing placement, and TOC references.

`render` output inherits every conversion caveat: the PNG pages show LibreOffice's
interpretation of the document with unrefreshed fields, not Microsoft Word's layout. Rendered
pages are a review aid, never a deliverable, and are bounded by the pixel and page budgets in
[render](references/operations.md#render). Rendering additionally requires the optional
`pypdfium2` package; without it, only structural checks are available and that must be
disclosed.

`verification.content_quality_report` uses pypdf extraction, page boxes, and limited XObject
detection. Its text anchors, low-text/nearly-blank flags, and page dimensions are structural
metrics—not raster rendering or visual QA. They cannot detect clipping, overlap, hierarchy,
contrast, bad wrapping, or whether a page is visually correct.

The CLI neither creates nor validates tagged PDF, PDF/UA, or PDF/A. Accessibility-oriented and
archive-oriented delivery profiles are honest check sets, not conformance claims. Use separate
specialist tooling and the relevant standard's validator before making such claims.

## Security boundary

Preflight rejects macros, encrypted or malformed packages, unsafe ZIP members, DTD/entities,
oversized resources, and external relationships by default. Allowing external relationships
permits processing; it is not a network sandbox. XML parser network settings do not prevent
LibreOffice or another consumer from following external targets.

LibreOffice's temporary profile/home and process group bound configuration and cleanup. They
do not provide OS-level network, filesystem, or process isolation. Treat all source documents
as untrusted and use stronger sandboxing when the threat model requires it.

For supported syntax see [operations](references/operations.md#operations-contract); for release
acceptance use the single [quality contract](references/quality.md#acceptance-criteria).
