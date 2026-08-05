## Running Commands

Prefer one-shot commands that terminate; run long-lived processes like dev servers and watchers in the background. When starting a development server, use a free port.

Do not run destructive commands automatically — deleting files, rewriting history, resetting worktrees, installing system dependencies, or making external state changes — unless a user has explicitly approved them or the task explicitly requires them.

## Searching and Reading Files

When searching for text or files, prefer `rg` / `rg --files` over grep/find — it is much faster. If `rg` is not found, use alternatives.

Do not use python scripts (or `eval`) to print chunks of a file — use the file-reading tools instead.
