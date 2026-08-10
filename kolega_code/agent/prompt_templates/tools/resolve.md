Apply or discard a pending preview action.

Pending actions are created by preview-only tools such as lsp_edit(apply=false).
Applying a pending action checks that the source files still match the preview
inputs before writing, unless force=true is explicitly provided.

Returns:
    Markdown summary of the resolve operation.
