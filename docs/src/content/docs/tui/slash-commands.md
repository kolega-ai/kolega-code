---
title: Slash Commands
description: Every slash command available in the Kolega Code composer.
---

Type `/` in the [composer](../composer/) to run a slash command. Commands come from
three sources, all surfaced in the same autocomplete dropdown:

- **Agent built-ins** — handled inside the agent loop.
- **TUI commands** — handled by the app.
- **Skills** — bundled, user, and project [Agent Skills](../../skills/),
  invoked as `/skill-name`.

If a name collides, agent and TUI commands take precedence over a skill of the same
name.

## Agent built-ins

These operate on the conversation itself.

| Command | Description |
| --- | --- |
| `/help` | Show the list of available agent commands |
| `/compress` | Compress the message history to reclaim context |
| `/clear` | Clear message history; in the TUI, also clears Terminal output and Logs |
| `/reset` | Clear message history, Terminal output, and Logs (alias of `/clear` in the TUI) |
| `/context` | Show the current context token count |

## TUI commands

These control the app and your session.

| Command | Description |
| --- | --- |
| `/skills` | List available Agent Skills |
| `/agents` | List or validate [custom agents](../../custom-agents/) (`/agents validate`) |
| `/init` | Create or update `AGENTS.md` for this repository |
| `/attach` | Attach an image: clipboard if no path, or `/attach <path>` for a file |
| `/detach` | Remove pending image attachments |
| `/plan` | Switch to [Plan mode](../modes/) |
| `/build` | Switch to [Build mode](../modes/) |
| `/sidebar` | Show or hide the side panel |
| `/settings` | Open the full-screen Settings editor |
| `/memory` | Open [Project Memory](../project-memory/) or run a memory subcommand |
| `/permissions` | Show or switch the shell/edit permission mode |
| `/model` | Choose the active model |
| `/effort` | Choose the active model's thinking effort |
| `/login` | Sign in to a provider, e.g. `/login chatgpt` |
| `/logout` | Sign out of a provider, e.g. `/logout chatgpt` |
| `/gigacode` | Toggle [gigacode](../../gigacode/) workflow orchestration on or off |
| `/web-search` | Show or set the web tool mode: `auto`, `hosted`, `client`, or `off` |
| `/goal` | Set, show, or clear an autonomous completion goal |
| `/loop` | Run a prompt on a repeating schedule |
| `/tasks` | Show the shared task list |
| `/queue-clear` | Clear queued follow-up messages |
| `/share` | Share a live link to this session (`/share lan`, `/share <port>`, `/share stop`) |
| `/copy` | Copy the last response to the clipboard |
| `/diagnostics` | Show version, model/endpoint, and the local diagnostics log path |
| `/bug` | Package local diagnostics into a shareable zip for a bug report |
| `/version` | Show the Kolega Code version |
| `/update` | Update Kolega Code to the latest version |
| `/quit` | Save the session and exit |
| `/exit` | Save the session and exit |

## Sharing a live session

`/share` starts a server for the session you are in and copies the link to your
clipboard. Whoever opens it watches the session as it happens — reasoning, tool
calls, terminal output — with a **LIVE** badge and the option to scrub back through
what already happened.

```
/share             # reachable from this machine
/share lan         # reachable from your local network
/share 9000        # ...on a port you choose
/share lan 9000
/share stop        # stop sharing
```

Sharing listens on port `8765` unless you name another one. If something already
has that port, the share quietly takes a different one and says so; a port you
asked for explicitly is treated as a requirement instead, because a tunnel is
probably pointed at it.

The link carries a token that is generated per share and is the only thing gating
access, so treat it as the secret it is. Sharing stops when you run `/share stop`
or leave the session; nothing keeps listening afterwards.

Viewers are **read-only**: they cannot type, approve a permission prompt, or
interrupt a turn.

:::caution
`/share lan` binds your machine's address on the local network, so anyone who can
reach it there and has the link can read that whole session, including file
contents and command output. A live link is also **not redacted** — unlike
`kolega-code share export`, whatever the agent printed is visible to whoever
holds it. Run `/share stop` when you are done.
:::

To let someone watch from outside your network, leave the share on loopback and
forward the port through a tunnel you control. There is no built-in relay. See
[Letting someone off your machine watch](/cli/share/#letting-someone-off-your-machine-watch) for verified
Tailscale, ngrok, and SSH recipes, and for how to turn the link `/share` gave you
into the one you send.

To send someone a recording rather than live access,
[`kolega-code share export`](/cli/share/) writes a static bundle with secrets
scrubbed.

Run `/goal <condition>` to set an autonomous completion goal the agent works
toward, verifying its own progress after each turn until the goal is met, the turn
cap is hit, or you pause it. `/goal` (no args) shows the goal status; `/goal clear`
(aliases: `stop`, `off`, `reset`, `none`, `cancel`) removes it. See
[Goal-Conditioned Work](../../goal/) for the full loop behavior, safety model, and
examples.

Run `/loop <interval> <prompt>` to re-run a prompt on a schedule — for example
`/loop 5m check whether CI went green`. Cron works too, via
`/loop --cron "0 9 * * 1-5" <prompt>`. Iterations only start when the session is
idle, missed windows never pile up, and your typed messages always take priority.
`/loop status` shows the schedule and next fire; `/loop stop` (aliases: `clear`,
`off`, `cancel`, `none`, `reset`) or `Esc` ends it. With no prompt, `/loop` reads
`.kolega/loop.md`. See [Scheduled Loops](../../loop/) for the interval and cron
syntax, `--fresh`, caps and expiry, and the unattended-run caveats.

Run `/model` to open a selectable list of supported models for the current
provider. You can also switch directly with `/model <name>`.

Run `/settings` to open the categorized Settings editor without navigating to the
sidebar tab. Unsaved changes remain a draft until you select **Apply Changes**.

Run `/memory` with no arguments (alias: `/memory browse`) to browse and edit the
project's private memory bank. Lightweight subcommands are also available:

| Command | Description |
| --- | --- |
| `/memory status` | Show enabled/backend/identity state, sizes, and the exact bounded startup context the agent receives |
| `/memory on` / `/memory off` | Enable agent access or disable it without deleting data |
| `/memory files` | List entries and sizes |
| `/memory show [path]` | Show bounded content; defaults to `MEMORY.md` |
| `/memory path` | Show the private local backend directory |

See [Project Memory](../project-memory/) for project identity, limits,
concurrency, privacy, and model-exposure details.

Run `/effort` to open a selectable list of supported effort values for the
active model. You can also switch directly with `/effort <level>`.

Run `/login chatgpt` to sign in with a ChatGPT subscription and use OpenAI models
without an API key; `/logout chatgpt` removes the stored credentials. See
[Sign in with ChatGPT](../../configuration/sign-in-with-chatgpt/).

Run `/queue-clear` to discard follow-up prompts that you queued while the current
turn is running. It removes their `Queued` transcript entries, but it does not
cancel or otherwise stop the active agent turn.

Run `/tasks` to print the shared task list into the transcript. It reads session
state directly and never starts an agent turn, so it is available in both
interaction modes. [Build mode](../modes/) owns the list and is the only mode that
can change it; in [Plan mode](../modes/) the agent can read it but not modify it.

Run `/agents` or `/agents list` to inspect all effective user and project
[custom-agent definitions](../../custom-agents/), including agents configured for
the other interaction mode. Run `/agents validate` to rescan the files and report
invalid definitions. File changes become dispatchable after the active agent is
rebuilt (for example, by switching modes) or the TUI is restarted.

Run `/diagnostics` to print a snapshot of this session — version, platform and
terminal, active model and endpoint, which providers have keys, and how many
event-loop stalls or LLM errors have been recorded — followed by the path to the
local diagnostics log. Run `/bug` to package that log, any captured stack dumps,
and the current session into a single shareable zip for a bug report (API keys are
scrubbed; the conversation and file contents are kept). See
[Diagnostics & Bug Reports](../../troubleshooting/diagnostics/) for what gets
captured, where it lives, and the privacy model.

Run `/init` to have the agent inspect the repository and create or update a
concise root `AGENTS.md`. Extra text after the command is passed as focus or
constraints:

```text
/init focus on Python packaging and test commands
```

## Skills

Any bundled or locally discovered [skill](../../skills/) is available as
`/skill-name`. Local skills are loaded from user or project `.agents/skills/`
directories, with project and user versions able to override a bundled skill of the
same name. Running a skill loads its instructions (and a manifest of its bundled
resources) into the conversation. Pass extra text after the command to run the skill
against a specific request:

```text
/release-notes summarize changes since the last tag
```

Use `/skills` at any time to see what's available in the current project.

When [Agent Skills are disabled](../../skills/#disabling-agent-skills), `/skills`
reports that Agent Skills are disabled and no `/skill-name` commands are
available or listed in the completion dropdown.
