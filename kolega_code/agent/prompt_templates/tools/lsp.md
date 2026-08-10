Query language server intelligence: diagnostics, definition, references, hover, symbols, status.

This versatile read-only tool interacts with the project's language servers;
operations need different arguments — see the list below.

Operations and required arguments:
- ``diagnostics`` — errors/warnings/hints for a file (``path``)
- ``definition`` (go-to-definition), ``type_definition`` (go-to-type-definition), ``implementation`` (find implementations), ``references`` (find all references), ``hover`` (hover/type info), ``call_hierarchy`` (incoming/outgoing calls), ``code_actions`` (list fixes/refactors without applying them) — all take (``path``, ``line``, ``symbol``)
- ``document_symbols`` — symbols in a file (``path``)
- ``workspace_symbols`` — project-wide symbol search (``query``)
- ``status`` — LSP server status (no args)
- ``capabilities`` — server capabilities (optional ``path``)
- ``reload`` — restart servers and re-detect (no args)

For position operations, ``line`` is 1-based and ``symbol`` is the name to
find on that line. Use ``name#N`` for the Nth occurrence.

Returns:
    Markdown-formatted results for the requested operation.
