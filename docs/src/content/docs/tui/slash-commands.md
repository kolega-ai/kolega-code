---
title: Slash Commands
description: Every slash command available in the Kolega Code composer.
---

Type `/` in the [composer](../composer/) to run a slash command. Commands come from
three sources, all surfaced in the same autocomplete dropdown:

- **Agent built-ins** — handled inside the agent loop.
- **TUI commands** — handled by the app.
- **Skills** — dynamically discovered project/user [Agent Skills](../../skills/),
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
| `/goal` | Set, show, or clear an autonomous completion goal |
| `/queue-clear` | Clear queued follow-up messages |
| `/copy` | Copy the last response to the clipboard |
| `/diagnostics` | Show version, model/endpoint, and the local diagnostics log path |
| `/bug` | Package local diagnostics into a shareable zip for a bug report |
| `/version` | Show the Kolega Code version |
| `/update` | Update Kolega Code to the latest version |
| `/quit` | Save the session and exit |
| `/exit` | Save the session and exit |

Run `/goal <condition>` to set an autonomous completion goal the agent works
toward, verifying its own progress after each turn until the goal is met, the turn
cap is hit, or you pause it. `/goal` (no args) shows the goal status; `/goal clear`
(aliases: `stop`, `off`, `reset`, `none`, `cancel`) removes it. See
[Goal-Conditioned Work](../../goal/) for the full loop behavior, safety model, and
examples.

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
| `/memory clear` | Confirm, then clear only the active backend |

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

Any [skill](../../skills/) discovered in `.agents/skills/` is available as
`/skill-name`. Running it loads the skill's instructions (and a manifest of its
bundled resources) into the conversation. Pass extra text after the command to run
the skill against a specific request:

```text
/release-notes summarize changes since the last tag
```

Use `/skills` at any time to see what's available in the current project.
