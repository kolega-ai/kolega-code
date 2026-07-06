---
title: Language servers
description: Configure LSP diagnostics and read-only code intelligence.
---

Kolega Code can start local Language Server Protocol (LSP) servers for detected
project languages. LSP is enabled by default and is used for:

- diagnostics after `edit`, `multi_edit`, and `write`;
- the `lsp_diagnostics` tool;
- the generic read-only `lsp` tool;
- `/lsp` and the Settings tab status display.

The LSP integration is read-only from the agent's perspective. It lists code
actions but does not apply them, and it does not expose rename, formatting, or
raw arbitrary LSP requests.

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

## Project configuration

Create `<project>/.kolega/lsp.json` to override LSP behavior for a repository:

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
sessions, and recent diagnostic counts. The Settings tab has a global LSP toggle;
changing it requires restarting the agent session.

If a server is missing, `/lsp` shows the server name and install guidance from
Kolega Code's bundled registry. You can suppress a language with
`disabled_languages` or point it at a custom server with `servers` and
`preferences`.
