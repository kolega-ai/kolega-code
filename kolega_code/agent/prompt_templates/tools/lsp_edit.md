Apply trusted LSP edits such as rename, file rename, formatting, and code actions.

This is the mutating companion to the read-only ``lsp`` tool. Use
``apply=False`` to preview the server-provided WorkspaceEdit without
writing files. ``path`` and ``new_path`` may be project-relative, use
``../`` traversal, or be absolute; local file URIs returned by the
server may also target files outside the project.
