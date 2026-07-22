# PDF limitations and security boundaries

## Contents

- [Extraction and rendering](#extraction-and-rendering)
- [Creation and page mutation](#creation-and-page-mutation)
- [Forms, metadata, and encryption](#forms-metadata-and-encryption)
- [OCR](#ocr)
- [Security](#security)
- [OCR composition](#ocr-composition)
- [Future work](#future-work)

## Extraction and rendering

- PDF content streams do not encode a universal reading order. Plain and layout text,
  words, columns, and tables are heuristic and must be checked against rendered pages.
- Repeated headers, footers, ligatures, rotated glyphs, clipping, transparency, unusual
  encodings, and nested forms can produce incomplete or surprising extraction.
- Table detection is not a structural guarantee. Missing rulings, merged cells, nested
  tables, and visual alignment can change rows or columns.
- Embedded-image inventory describes image objects, not necessarily every visible raster.
  Masks, inline images, tiling patterns, and reused XObjects can be represented differently.
- Page classification is routing evidence, not proof. Sparse digital pages may look scanned;
  scanned pages with an existing text layer may look hybrid.
- `render` rasterizes the delivered PDF itself with PDFium (via pdfplumber). Unlike office
  formats reviewed through an intermediate converter, the pixels reflect the actual
  artifact — a strictly stronger review guarantee. Two bounds on that claim: rasters are
  fully deterministic only when every font is embedded (non-embedded fonts render with this
  machine's substitutes, which is exactly what the font `RELEASE BLOCKER` warning guards),
  and rasters do not prove viewer-interactive behavior — form-field appearance
  regeneration, annotation states, JavaScript, optional-content layers, overprint, or ICC
  color handling.
- The font inventory walks page, form, and annotation resource dictionaries with bounded
  depth. Fonts referenced only from constructs beyond that walk are reported as
  `truncated`, never silently ignored.
- This tool does not render HTML and does not claim browser-equivalent conversion.

## Creation and page mutation

- ReportLab creation supports basic built-in fonts, positioned text/paragraphs, simple
  tables and images, headers/footers, lines, and basic AcroForm fields. It is not a
  typesetting replacement for arbitrary office, TeX, or design documents.
- Page merges and stamps may alter resource dictionaries, transparency, clipping, or
  appearance in unusual PDFs. Visually inspect representative pages.
- A watermark or overlay can often be removed and does not erase underlying content.
- There is no arbitrary word replacement. PDF text may be split, transformed, encoded, or
  drawn as outlines; covering and retyping text is not equivalent to editing it.
- Multi-output split publication is atomic per file, not transactional across the entire
  set.
- Unknown content is copied through pypdf where possible, but lossless round-trip fidelity
  for arbitrary PDFs is not promised.

## Forms, metadata, and encryption

- AcroForm filling requires exact field names. XFA editing is unsupported.
- Some viewers require regenerated appearance streams; the tool deliberately avoids
  claiming that every viewer will render every filled value identically.
- Buttons, signatures, JavaScript actions, calculations, dynamic forms, and uncommon
  appearance states may not behave as expected after rewriting.
- Metadata removal does not prove that equivalent information is absent from content,
  attachments, annotations, XMP packets, incremental revisions, or filesystem metadata.
- Encryption controls access but does not remove content already disclosed elsewhere.
  Choose passwords and key handling outside this tool.
- Every rewrite invalidates existing digital-signature assurances even if a signature object
  remains visible. The tool does not preserve or validate signature trust.

## OCR

- OCR is probabilistic. Names, numbers, punctuation, low-resolution text, handwriting,
  uncommon scripts, math, and dense tables require review.
- Engine confidence values are not calibrated across engines. Missing confidence remains
  `null`; it is never invented.
- Hybrid-region routing depends on detectable image boxes and may skip or duplicate content.
  Use `--force` only after accepting full-page duplicate-text risk.
- Surya and Paddle structure remain model predictions. Tesseract TSV is intentionally flat
  and weak for reading order or table reconstruction.
- Language selection does not guarantee script coverage. Unsupported and uncertain language
  should be reported, not hidden by repeated retries.
- Models and language data are separate artifacts with separate licenses and versions. Code
  licensing does not establish model-use eligibility.
- The adapters cover local execution only. No document is sent to a hosted service.

## Security

- Crop, overlay, white rectangles, stamps, and watermarks are **not secure redaction**.
  Underlying text, images, revisions, attachments, metadata, or object streams remain. The
  only supported removal path is the `redact` command, which deletes intersecting content,
  rewrites the document as a single revision, and verifies the result — read
  [Redaction](references/redaction.md) for its policy table, verification contract, and the explicit
  non-guarantees (prior copies, backups, attachments, and collateral whole-operator
  removal, among others).
- Redaction v1 refuses rather than guesses: encrypted inputs, Type3 or width-unmeasurable
  fonts inside targets, intersecting Form XObjects, text-clipping modes, XFA, and term hits
  in outlines, named destinations, or form-field values all fail loudly with instructions.
- Deleting visible pages does not constitute forensic sanitization.
- Embedded files, JavaScript, launch actions, malicious links, and other active content are
  not comprehensively analyzed or removed.
- Passwords passed directly on a command line may be visible to local process inspection.
  Prefer `--password-env`; protect and delete JSON jobs containing encryption passwords.
- The parser is bounded and strict, but opening untrusted documents still exercises third-
  party parsers. Use an isolated, least-privilege environment for hostile input.
- Atomic replacement protects against partial publication of one output, not disk failure,
  malicious concurrent replacement, or rollback across multiple destinations.

## OCR composition

- `ocr-compose` writes an invisible text layer from a completed sidecar; the layer inherits
  every OCR recognition error, and its verification proves extractability and geometry, not
  transcription correctness.
- The default base-14 layer maps cp1252 text only; other scripts need an operator-supplied
  font, which is embedded and subset under the operator's license responsibility.
- Composition preserves the source pixels (verified by a raster diff) but does not create
  tagged structure, reading-order guarantees, or accessibility conformance.

## Future work

This release intentionally does not implement:

- forensic sanitization beyond the `redact` contract — no partial image masking, no
  redaction inside Form XObjects, no encrypted-input redaction, and no claims about copies
  outside the output file (see [Redaction](references/redaction.md) for the refusal list);
- digital signing, signature validation, or long-term validation;
- XFA creation or editing;
- arbitrary word-level PDF editing beyond removal — covering and retyping text is still not
  editing it;
- complete annotation, attachment, JavaScript, portfolio, 3D, or multimedia workflows;
- accessibility tree repair, tagged-PDF authoring, or PDF/UA certification;
- HTML rendering;
- remote OCR endpoints.

Use a specialized reviewed workflow for any of these outcomes.
