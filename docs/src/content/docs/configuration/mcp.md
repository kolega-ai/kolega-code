---
title: MCP servers
description: Configure Model Context Protocol servers for Kolega Code tools.
---

Kolega Code can expose verified [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server tools as first-class agent tools.

MCP support is local-first and opt-in:

- A server's tools are exposed only after you connect to it once, and again whenever its definition changes.
- Servers that a repository defines load only after you trust the project.
- Agent startup never connects to a server, starts a local command or opens OAuth. Connecting and signing in are explicit.
- MCP tools are not propagated to sub-agents.
- In `ask` permission mode, every MCP tool call prompts unless you save an exact-tool or whole-server allow rule.

## Adding a server

The `kolega-code mcp` commands manage servers from a shell. `/mcp` does the same inside the TUI.

### From a URL

```bash
kolega-code mcp add --transport http notion https://mcp.notion.com/mcp
kolega-code mcp add --transport http secure-api https://api.example.com/mcp \
  --header 'Authorization: Bearer ${SECURE_API_TOKEN}'
```

### From a launch command

Put the server's command after `--`, so its own flags such as `-y` or `--port` are passed through untouched:

```bash
kolega-code mcp add airtable -- npx -y airtable-mcp-server
kolega-code mcp add --env AIRTABLE_API_KEY=YOUR_KEY --transport stdio airtable \
  -- npx -y airtable-mcp-server
```

When `--transport` is omitted, an `http://` or `https://` target is an HTTP server and anything else is a stdio command.

### From setup instructions written for another client

A server's documentation usually shows an `mcpServers` JSON block. Pass the entry inside it, unchanged:

```bash
kolega-code mcp add-json weather-api '{"type":"http","url":"https://api.weather.com/mcp","headers":{"Authorization":"Bearer ${WEATHER_TOKEN}"}}'
```

The argument is JSON text, a path to a JSON file, or `-` for stdin. A whole `{"mcpServers": {...}}` block works too: with one entry its name is taken from the block, and with several every server is added.

### Connecting

```bash
kolega-code mcp verify notion
```

Verification opens a real MCP session and lists the server's tools. `add --verify` does it immediately after saving.

## Scopes

| Scope | `--scope` | Stored in | Shared |
| --- | --- | --- | --- |
| User | `user` (default) | `<state_dir>/mcp_servers.json` | No: all your projects |
| Local | `local` | `settings.json`, keyed by project path | No: this project only |
| Project | `project` | `<project>/.mcp.json` | Yes: commit it |

- A git worktree has its own `local` definitions and project enable/disable choices.
- The CLI uses trust for the specified project path. In the TUI, linked workspaces use the launch project's trust decision.
- A name defined in more than one scope is loaded once, from the highest-precedence scope: editor-provided → `local` → `.kolega/mcp_servers.json` → `.mcp.json` → `user`. The whole definition wins and fields are never merged. `kolega-code mcp list` reports what is shadowed.
- `--scope project` writes `.kolega/mcp_servers.json` only for a project that already has that file and no `.mcp.json`; every other project gets `.mcp.json`.
- Writing a literal header or env value to the shared file prints a warning: prefer `${NAME}` references there.

`.mcp.json` uses the portable shape other tools read as well:

```json
{
  "mcpServers": {
    "repo-tools": {
      "type": "stdio",
      "command": "uvx",
      "args": ["repo-mcp-tools"],
      "env": { "GITHUB_TOKEN": "${GITHUB_TOKEN}" }
    },
    "docs": { "type": "http", "url": "https://docs.example.com/mcp" }
  }
}
```

- `type` accepts `stdio`, `http` (also `streamable-http`) and `sse`. An entry with a `url` and no `type` is inferred from the URL. WebSocket and `sdk` entries are skipped with a diagnostic.
- `timeout` is in milliseconds. `disabled: true` disables an entry. Non-string `env` and `headers` values are stored as strings.
- Kolega-only settings (a display name, `cwd`, OAuth client details) go under an `x-kolega` object, which other clients ignore.
- Edits through Kolega touch only the named entry. Other entries, fields Kolega does not know and key order are preserved.

### Trusting a project's servers

`.mcp.json` can point at local executables or remote services chosen by the repository, so its servers wait until you trust the project. Until then they are listed as **Pending approval** and never loaded or connected.

```bash
kolega-code mcp list               # shows "Pending approval"
kolega-code mcp trust              # load them
kolega-code . --trust-mcp          # or at launch
kolega-code mcp untrust            # stop loading them
```

When a project defines servers you have not trusted, the TUI shows a notice at startup. Open `/mcp` and press `t` to review and trust them.

### Enabling and disabling

`enable` and `disable` edit a `user` or `local` definition in place. A `project` definition is never edited for this: the choice is recorded privately for the project in `settings.json`, so the shared file does not change.

```bash
kolega-code mcp disable ci
kolega-code mcp enable ci
kolega-code mcp reset-project-choices   # forget the enable/disable choices and the trust decision
```

## Environment variable references

`command`, each `arg`, `env` values, `url` and `headers` values may reference the environment with `${NAME}` or `${NAME:-default}`:

```json
"headers": { "Authorization": "Bearer ${NOTION_TOKEN:-}" }
```

- An empty variable counts as set. An unset variable with no default fails verification with a message that names it, and `kolega-code mcp list` shows `missing env: NAME`.
- References are resolved only when Kolega connects to the server. The stored configuration keeps the literal reference, so a resolved secret never reaches a fingerprint, status file or listing.
- Anything else, such as VS Code's `${input:name}`, is left as written.

## Config format

Kolega's own files (`<state_dir>/mcp_servers.json` and the legacy `<project>/.kolega/mcp_servers.json`) use this shape. `servers` may be an array or an object keyed by server ID:

```json
{
  "schema_version": 1,
  "servers": [
    {
      "id": "docs",
      "name": "Docs search",
      "transport": "streamable_http",
      "url": "https://mcp.example.com/mcp",
      "headers": { "Authorization": "Bearer ..." },
      "enabled": true
    }
  ]
}
```

Server IDs may contain letters, numbers, `_` and `-`. The legacy project file is still read, behind the same trust gate, and takes precedence over `.mcp.json`.

### Transports

| `transport` | Needs | Notes |
| --- | --- | --- |
| `streamable_http` | `url`, optional `headers` | Recommended for remote servers |
| `sse` | `url`, optional `headers` | `sse_read_timeout_seconds` defaults to 300 |
| `stdio` | `command`, optional `args`, `env`, `cwd` | Runs a local command; `cwd` resolves relative to the project |

## OAuth

Enable OAuth for HTTP transports with the `oauth` object:

```json
{
  "id": "oauth-server",
  "transport": "streamable_http",
  "url": "https://mcp.example.com/mcp",
  "oauth": { "enabled": true, "scope": "read write" }
}
```

Sign in from a shell or from `/mcp`:

```bash
kolega-code mcp login oauth-server
kolega-code mcp logout oauth-server
```

`login` runs the OAuth flow and connects; `logout` clears the stored sign-in and cached tools, so the server reads as `Needs auth` until you sign in again. Use `--no-browser` to print the authorization URL instead of opening a browser. A sign-in you do not finish within five minutes is reported the same way, so an abandoned sign-in is never mistaken for an unreachable server.

`kolega-code mcp add` takes the common OAuth options directly: `--oauth`, `--client-id`, `--client-secret` (prompts with masked input when no value is given), `--callback-port` and `--oauth-scope`.

Tokens and dynamic client registration data are stored locally in `<state_dir>/mcp_oauth_tokens.json`. Kolega Code writes this file with owner-only permissions where the platform supports POSIX modes, redacts those values from diagnostics, and does not encrypt the file beyond the protection provided by the OS and filesystem permissions.

### Pre-registered OAuth servers (HubSpot, Slack, Enterprise IdPs)

Some hosted MCP servers require a pre-registered OAuth application instead of Dynamic Client Registration (RFC 7591). For example, [HubSpot's setup guide](https://developers.hubspot.com/docs/apps/developer-platform/build-apps/integrate-with-the-remote-hubspot-mcp-server) describes creating an MCP auth app, and [Slack's MCP documentation](https://docs.slack.dev/ai/slack-mcp-server/) describes its client registration requirements. Follow your provider's instructions to obtain a `client_id` and, for confidential clients, a `client_secret`. Public clients use a client ID with PKCE and no secret.

When registering the app in the provider's dashboard:

1. Set the redirect URI to a fixed localhost port, for example `http://127.0.0.1:33418/callback`.
2. Configure the exact same `redirect_uri` in your server entry so the local callback server listens on that port.

```json
{
  "id": "hubspot",
  "transport": "streamable_http",
  "url": "https://mcp.hubspot.com",
  "oauth": {
    "enabled": true,
    "client_id": "your-hubspot-client-id",
    "client_secret_env": "HUBSPOT_MCP_CLIENT_SECRET",
    "redirect_uri": "http://127.0.0.1:33418/callback",
    "token_endpoint_auth_method": "client_secret_post"
  }
}
```

```bash
kolega-code mcp add hubspot --transport streamable_http --url https://mcp.hubspot.com \
  --oauth-client-id "your-client-id" \
  --oauth-client-secret-env "HUBSPOT_MCP_CLIENT_SECRET" \
  --redirect-uri "http://127.0.0.1:33418/callback" \
  --oauth-auth-method "client_secret_post"
```

OAuth configuration options:

| Field | Description |
| --- | --- |
| `client_id` | Pre-registered OAuth client ID. |
| `client_secret` | Direct client secret (suitable for user config). |
| `client_secret_env` | Environment variable name holding the client secret (recommended for version control). |
| `redirect_uri` | Callback URI. Must be an `http` URL on `127.0.0.1` or `localhost`. Omitting the port uses an ephemeral port. |
| `scope` | Space-separated OAuth scopes to request. When set, these override the server-advertised defaults. |
| `token_endpoint_auth_method` | `client_secret_post`, `client_secret_basic`, or `none` (defaults to `client_secret_post` when a secret is configured). |

Providing a client ID enables OAuth unless `enabled` is explicitly `false`. A secret or secret environment variable requires a client ID; `client_secret_post` and `client_secret_basic` also require a secret. When both secret fields are set, the direct secret takes precedence. Callback URIs cannot contain embedded credentials or fragments.

In `.mcp.json`, the camelCase spellings other clients use are understood too: `oauth.clientId`, `oauth.callbackPort` and `oauth.scopes`.

Cached tokens are tied to the server URL and OAuth configuration using a salted PBKDF2 fingerprint. Changing credentials, scopes, or the configured redirect URI requires verification again. Pre-registered tokens without this binding, and tokens using the earlier SHA-256 fingerprint format, require one new authorization. Expiry and authorization-server metadata are saved with new tokens so token refresh works across connections.

## CLI reference

Global options go before the subcommand: `kolega-code mcp --project . <command>`. `--state-dir` overrides the state directory and `--trust-mcp` trusts the project first.

Where a name is defined in more than one scope, `get`, `remove`, `enable` and `disable` need `--scope` to choose one.
If both project files define the same name, `get` and `remove` select the legacy
`.kolega/mcp_servers.json` definition first. Removing it leaves the `.mcp.json` definition in place.

| Command | What it does |
| --- | --- |
| `list [--scope S] [--json]` | Lists servers with their status, scope and tool count. `--json` prints machine-readable output without secret values. |
| `list --check [--yes]` | Connects to each enabled server to refresh its status; exits 1 if any fail. Stdio servers need a confirmation, or `--yes` without a terminal. |
| `get <name> [--scope S] [--json]` | Shows the definition's scope and file, status, endpoint, the names of header and env keys, OAuth mode, timeout and tools. |
| `add …` | Adds or updates a server. Prints `Added` or `Updated`. |
| `add-json [<name>] <json>` | Adds a server from a portable entry or block. |
| `remove <name> [--scope S]` | Removes the definition and its saved status and sign-in. |
| `enable`, `disable <name> [--scope S]` | Toggles a server without removing it. |
| `verify [<name> \| --all] [--yes] [--no-browser] [--json]` | Connects and lists tools. |
| `login <name> [--no-browser]` | Signs in with OAuth and connects. |
| `logout <name>` | Clears the sign-in and cached tools. |
| `trust`, `untrust` | Starts or stops loading this project's servers. |
| `reset-project-choices` | Forgets the project's enable/disable choices and its trust decision. |
| `import …` | Copies servers from another tool. See [Importing](#importing-from-other-tools). |

`add` options:

| Option | Description |
| --- | --- |
| `-s, --scope user\|project\|local` | Where to store the server. Default: `user`. |
| `-t, --transport stdio\|http\|sse` | Inferred when omitted. |
| `-H, --header 'Name: value'` | HTTP header (repeatable). `Name=value` is accepted too. |
| `-e, --env KEY=VALUE` | Environment variable (repeatable). |
| `--name`, `--cwd`, `--disabled` | Display name, working directory, add disabled. |
| `--oauth`, `--client-id`, `--client-secret[=VALUE]`, `--callback-port` | OAuth. A bare `--client-secret` prompts with masked input; place it last or before `--`. |
| `--oauth-client-secret-env`, `--oauth-auth-method`, `--oauth-scope`, `--redirect-uri` | Further OAuth settings. |
| `--verify` | Connect after saving. |

`kolega-code doctor` reports each server's status offline.

### Status

| Status | Meaning |
| --- | --- |
| `● Connected` | Connected and its tools are exposed. |
| `! Needs auth` | An OAuth server whose sign-in is missing or expired: run `kolega-code mcp login <name>`. |
| `! Config changed` | The definition changed since the last connection; connect again. |
| `✗ Failed` | The last connection attempt failed. |
| `○ Not connected` | Never connected. |
| `⊘ Disabled` | Not loaded; `disabled for this project` marks a private choice. |
| `⏸ Pending approval` | A project server waiting for trust. |

## The `/mcp` panel

Type `/mcp` in the TUI to list every server with its status and act on the selected one:

| Key | Action |
| --- | --- |
| `enter` | List the selected server's exposed tools |
| `v` | Reconnect (connect again and refresh tools) |
| `a` | Sign in with OAuth |
| `space` | Enable or disable |
| `c` | Clear the stored sign-in |
| `t` | Trust this project's servers |
| `i` | Import servers from another tool |
| `s` | Edit the server in Settings |
| `r`, `esc` | Refresh, close |

The text forms do the same without opening the panel:

```text
/mcp reconnect all          # every enabled server that is not connected
/mcp reconnect notion
/mcp disable ci
/mcp enable ci
/mcp import
```

Changes wait for the running turn to end, so a session is never rebuilt underneath an agent.

## TUI Settings

Open the full Settings screen with `/settings` and choose **MCP Servers** to:

- Create, update, delete, enable and disable `user` and `local` servers. New servers get a **Scope** selector; an existing server keeps the scope it was defined in.
- Verify a selected server or save and verify in one step.
- Trust the project's servers and clear a server's stored OAuth tokens.

Project servers are shown but read-only there; edit `.mcp.json` or use `kolega-code mcp add --scope project`.

MCP actions save immediately rather than waiting for the Settings **Apply Changes** button. Trusting project configuration, deleting a server, and clearing sign-in ask for confirmation first.

## Importing from other tools

```bash
kolega-code mcp import
```

The picker lists only tools on this machine that define at least one server, then shows their servers with a secret-free summary. Unsupported entries and servers you already have are listed with the reason and cannot be chosen.

Supported sources:

| Source id | Tool | Files read |
| --- | --- | --- |
| `claude-code` | Claude Code | `~/.claude.json`: your servers, and this project's private ones |
| `claude-desktop` | Claude Desktop | `claude_desktop_config.json` |
| `cursor` | Cursor | `~/.cursor/mcp.json`, `<project>/.cursor/mcp.json` |
| `vscode` | VS Code / Copilot | `.vscode/mcp.json`, the user-profile `mcp.json`, `~/.copilot/mcp-config.json` |
| `windsurf` | Windsurf | `~/.codeium/windsurf/mcp_config.json` |
| `gemini` | Gemini CLI | `~/.gemini/settings.json`, `<project>/.gemini/settings.json` |
| `codex` | Codex | `~/.codex/config.toml` |

Without a terminal, name the sources and the scope and confirm with `--yes` or preview with `--dry-run`:

```bash
kolega-code mcp import --from codex --from cursor --scope user --dry-run
kolega-code mcp import --from codex --scope user --yes
kolega-code mcp import --file ./team-mcp.json --scope project --yes
```

- `--server NAME` narrows the import and `--overwrite` replaces servers already defined in the target scope.
- Names are adjusted to Kolega's rules (`My Tool.v2` becomes `My-Tool-v2`); the picker shows the original.
- Codex's `env_vars` become `${NAME}` references and `bearer_token_env_var` becomes an `Authorization: Bearer ${NAME}` header, so no secret value is copied. Fields Kolega cannot use, such as Codex per-tool settings, are reported as dropped.
- Nothing is read from this project's own `.mcp.json`: Kolega already loads it. In the TUI, press `i` in the `/mcp` panel.

When importing into the shared project scope, prefer `${NAME}` references over
literal header or env values. Use `user` or `local` scope to keep those values
out of the shared file.

Imported servers start unconnected. Project servers need approval before they
can connect. Once they are trusted, run `kolega-code mcp verify --all` or
reconnect them from `/mcp`.

## Verification and tool exposure

Verification opens a real MCP session, calls `initialize`, lists all tools (including paginated results), and stores a fingerprinted status in `<state_dir>/mcp_server_status.json`.

A server's tools are exposed only when:

1. the server is enabled,
2. verification succeeded, and
3. the saved fingerprint still matches the current server definition.

Changing connection details, command args, headers, env, OAuth settings, or URL makes the saved status stale and hides tools until you connect again. Changing only `enabled` does not invalidate verification.

Exposed tools are named:

```text
mcp__{server_id}__{tool_id}
```

The server-provided `inputSchema` is preserved verbatim. A tool name another provider would reject gets a compatible name, and the adjustment is reported.

## Permissions

MCP tools participate in the same project permission flow as shell and edit tools.

In `ask` mode, the TUI can save:

- an exact-tool allow rule for `mcp__server__tool`, or
- a whole-server allow rule for all verified tools from one server.

Saved rules live in `<project>/.kolega/permissions.json`.
