---
title: Language servers
description: Configure LSP diagnostics, code intelligence, and trusted LSP edits.
---

Kolega Code can start local Language Server Protocol (LSP) servers for detected
project languages. LSP is enabled by default and is used for:

- diagnostics after `edit`, `multi_edit`, and `write`;
- the generic read-only `lsp` tool;
- the permission-gated `lsp_edit` tool;
- `/lsp` and the Settings screen's Tools status display.

The `lsp` tool remains read-only. Mutating LSP operations are exposed through
`lsp_edit`, which is treated like `edit`, `multi_edit`, and `write` for
permission prompts. Kolega Code does not expose raw arbitrary LSP requests.

## Tool operations

The generic `lsp` tool supports:

- `diagnostics` for file errors, warnings, and hints;
- `definition`, `type_definition`, `implementation`, `references`, and `hover`;
- `document_symbols` and `workspace_symbols`;
- `call_hierarchy` for incoming and outgoing calls;
- `code_actions` to list available quick fixes and refactors without applying them;
- `status`, `capabilities`, and `reload`.

Position operations use `path`, a 1-based `line`, and a `symbol` on that line.
Use `symbol#2` to select the second occurrence of a symbol on the same line.
`code_actions` includes an `action_id` for each action so a later `lsp_edit`
call can apply the exact server-provided action.

The mutating `lsp_edit` tool supports:

- `rename` for symbol rename;
- `rename_file` for file moves with `workspace/willRenameFiles` and
  `workspace/didRenameFiles` notifications;
- `format_document` and `format_range`;
- `apply_code_action` by `action_id`, title/query, or index.

`lsp_edit` applies server-provided `WorkspaceEdit` payloads only. If the server
does not advertise or return an edit for an operation, Kolega Code reports that
the operation is unsupported instead of constructing a manual fallback edit.
Pass `apply: false` to preview the LSP edit without writing files.

### External paths, permissions, and undo

Local `file:` URIs returned by a language server may target files outside the
project. Initiating paths can likewise be project-relative, use `../` traversal,
or be absolute. The existing edit permission gate still applies to the request,
and the Vibe edit policy checks every touched path before mutation.

A mutation that touches any external path—including a mix of project and
external files—is not snapshotted and cannot be undone through snapshot restore.
With `apply: false`, such an external or mixed edit can be previewed but cannot
create a snapshot-backed, resolvable pending action. To perform it, rerun the
operation with `apply: true`.

## Project configuration

Create `<project>/.kolega/lsp.json` to override LSP behavior for a repository.
Project LSP config is trusted input because custom servers can execute local
binaries. Kolega Code ignores this file unless the session is started with
`--trust-lsp` or the project is listed in trusted LSP projects in user settings.

```json
{
  "enabled": true,
  "auto_diagnostics_on_edit": true,
  "max_diagnostics": 20,
  "auto_fallback": true,
  "prompt_on_missing": true,
  "disabled_languages": [],
  "preferences": {
    "python": "pyright"
  },
  "diagnostic_servers": ["ruff-lsp"],
  "initialization_options": {
    "pyright": {
      "python": {
        "analysis": {
          "typeCheckingMode": "basic"
        }
      }
    }
  },
  "workspace_configuration": {
    "pyright": {
      "python": {
        "analysis": {
          "typeCheckingMode": "basic"
        }
      }
    }
  },
  "servers": {
    "custom-python-lsp": {
      "bin": "custom-lsp",
      "args": ["--stdio"],
      "languages": ["python"],
      "env": {
        "CUSTOM_LSP_MODE": "project"
      }
    }
  }
}
```

The project file cannot change the user's global LSP kill switch: `enabled` is
accepted in user settings, but ignored from `.kolega/lsp.json`.

`preferences` chooses the preferred server for a language. If
`auto_fallback` is `false`, Kolega Code checks only the preferred server before
reporting the language as missing. If it is `true`, installed alternatives can be
used automatically.

`diagnostic_servers` starts additional servers for diagnostics alongside the
primary server. This is useful for setups such as Pyright plus Ruff.

`initialization_options` is sent in the LSP `initialize` request. Some servers
also ask for `workspace/configuration`; use `workspace_configuration` for those
responses, keyed first by server name and then by requested section.

## Status and troubleshooting

Run `/lsp` in the TUI to see detected languages, missing servers, active
sessions, and recent diagnostic counts. The full Settings screen's **Tools**
category has a global LSP toggle; applying a change rebuilds the agent.

If a server is missing, `/lsp` shows the server name and install guidance from
Kolega Code's bundled registry. You can suppress a language with
`disabled_languages` or point it at a custom server with `servers` and
`preferences`.
