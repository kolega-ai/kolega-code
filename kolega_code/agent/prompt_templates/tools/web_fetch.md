Fetch URL content locally, follow an instruction, and return a grounded response.

This tool handles HTML through a quality-gated local extractor chain, reads
textual formats directly, converts PDF and modern Office documents locally,
and asks the fast model to apply the instruction with source evidence. It does
not run JavaScript or send content to a third-party reader service. For a page
reported as JavaScript-rendered, use the browser tools instead.

Returns:
    A source-attributed answer with evidence, or bounded extracted content if
    the internal answering stage cannot complete.
