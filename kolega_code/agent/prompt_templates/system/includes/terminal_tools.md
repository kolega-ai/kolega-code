## Running Commands

Prefer one-shot commands that terminate; run long-lived processes like dev servers and watchers in the background. When starting a development server, use a free port.

Do not run destructive commands automatically — deleting files, rewriting history, resetting worktrees, installing system dependencies, or making external state changes — unless a user has explicitly approved them or the task explicitly requires them.

## Searching and Reading Files

When searching for text or files, prefer `rg` / `rg --files` over grep/find — it is much faster. If `rg` is not found, use alternatives. In ripgrep, `-r` means `--replace`, not recursive — `rg -rn "foo"` prints every match replaced with the literal `n`; recursion is the default, so use `rg -n` for line numbers.

Do not use python scripts (or `eval`) to print chunks of a file — use the file-reading tools instead.
