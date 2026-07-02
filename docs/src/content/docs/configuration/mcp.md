---
title: MCP servers
description: Configure Model Context Protocol servers for Kolega Code tools.
---

Kolega Code can expose verified [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server tools as first-class agent tools.

MCP support is local-first and opt-in:

- Global servers are read from `<state_dir>/mcp_servers.json`.
- Project servers are read from `<project>/.kolega/mcp_servers.json` only after the project is trusted for MCP.
- Agent startup never opens a browser and never starts an OAuth flow. OAuth is only started by an explicit `kolega-code mcp verify` command or the TUI **Verify** button.
- MCP tools are not propagated to sub-agents.
- In `ask` permission mode, every MCP tool call prompts unless you save an exact-tool or whole-server allow rule.

## Config locations

| Scope | Path | Enabled when |
| --- | --- | --- |
| Global | `<state_dir>/mcp_servers.json` | Always loaded |
| Project | `<project>/.kolega/mcp_servers.json` | Project is trusted with `--trust-mcp` or from the TUI |

`<state_dir>` is the Kolega Code state directory. You can override it with `--state-dir` for CLI management commands.

Project MCP config is intentionally gated because it can point at local executables or remote services controlled by the repository. Trust a project only after reviewing `.kolega/mcp_servers.json`.

## Config format

```json
{
  "schema_version": 1,
  "servers": [
    {
      "id": "docs",
      "name": "Docs search",
      "transport": "streamable_http",
      "url": "https://mcp.example.com/mcp",
      "headers": {
        "Authorization": "Bearer ..."
      },
      "enabled": true
    }
  ]
}
```

`servers` may be an array, or an object keyed by server ID. Server IDs may contain letters, numbers, `_`, and `-`.

### `streamable_http`

```json
{
  "id": "remote-http",
  "transport": "streamable_http",
  "url": "https://mcp.example.com/mcp",
  "headers": {
    "Authorization": "Bearer ..."
  }
}
```

### `sse`

```json
{
  "id": "remote-sse",
  "transport": "sse",
  "url": "https://mcp.example.com/sse",
  "headers": {
    "Authorization": "Bearer ..."
  },
  "sse_read_timeout_seconds": 300
}
```

### `stdio`

```json
{
  "id": "local-server",
  "transport": "stdio",
  "command": "npx",
  "args": ["-y", "@vendor/mcp-server"],
  "env": {
    "TOKEN": "..."
  },
  "cwd": "."
}
```

`stdio` servers execute a local command. `kolega-code mcp verify` refuses to start `stdio` commands in non-interactive mode unless you pass `--yes`.

## OAuth

Enable OAuth for HTTP transports with the `oauth` object:

```json
{
  "id": "oauth-server",
  "transport": "streamable_http",
  "url": "https://mcp.example.com/mcp",
  "oauth": {
    "enabled": true,
    "scope": "read write"
  }
}
```

Run verification to start the OAuth flow:

```bash
kolega-code mcp verify oauth-server --project .
```

Use `--no-browser` to print the authorization URL instead of opening a browser. Tokens and dynamic client registration data are stored locally in `<state_dir>/mcp_oauth_tokens.json`. Kolega Code writes this file with owner-only permissions where the platform supports POSIX modes, redacts those values from diagnostics, and does not encrypt the file beyond the protection provided by the OS and filesystem permissions.

## CLI management

List servers and status:

```bash
kolega-code mcp --project . list
```

Add a remote server to global config:

```bash
kolega-code mcp --project . add docs \
  --transport streamable_http \
  --url https://mcp.example.com/mcp \
  --header 'Authorization=Bearer ...'
```

Add a project config server:

```bash
kolega-code mcp --project . add repo-local \
  --project-config \
  --transport stdio \
  --command npx \
  --arg -y \
  --arg @vendor/mcp-server
```

Trust project MCP config:

```bash
kolega-code . --trust-mcp
# or for one-shot use:
kolega-code ask "use repo tools" --project . --trust-mcp
```

Verify one server or all enabled servers:

```bash
kolega-code mcp --project . verify docs
kolega-code mcp --project . verify --all --yes
```

Enable, disable, or remove servers:

```bash
kolega-code mcp --project . disable docs
kolega-code mcp --project . enable docs
kolega-code mcp --project . remove docs
```

## TUI management

Open the **Settings** tab and use the **MCP Servers** section to:

- Refresh status.
- Trust project MCP config.
- Create, update, delete, enable, and disable global MCP servers.
- Verify a selected server.
- Clear stored OAuth tokens for a selected server.

Project servers are shown in the TUI but are read-only there. Edit `.kolega/mcp_servers.json` directly or use `kolega-code mcp ... --project-config`.

## Verification and tool exposure

Verification opens a real MCP session, calls `initialize`, lists all tools (including paginated results), and stores a fingerprinted status in `<state_dir>/mcp_server_status.json`.

A server's tools are exposed only when:

1. the server is enabled,
2. verification succeeded, and
3. the saved fingerprint still matches the current server config.

Changing connection details, command args, headers, env, OAuth settings, or URL makes the saved status stale and hides tools until you verify again. Changing only `enabled` does not invalidate verification.

Exposed tools are named:

```text
mcp__{server_id}__{tool_id}
```

The server-provided `inputSchema` is preserved verbatim.

## Permissions

MCP tools participate in the same project permission flow as shell and edit tools.

In `ask` mode, the TUI can save:

- an exact-tool allow rule for `mcp__server__tool`, or
- a whole-server allow rule for all verified tools from one server.

Saved rules live in `<project>/.kolega/permissions.json`.
