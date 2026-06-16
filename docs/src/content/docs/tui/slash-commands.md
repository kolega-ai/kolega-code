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
| `/clear` | Clear the message history |
| `/reset` | Clear the message history (alias of `/clear`) |
| `/context` | Show the current context token count |

## TUI commands

These control the app and your session.

| Command | Description |
| --- | --- |
| `/skills` | List available Agent Skills |
| `/plan` | Switch to [Plan mode](../modes/) |
| `/build` | Switch to [Build mode](../modes/) |
| `/model` | Choose the active model |
| `/effort` | Choose the active model's thinking effort |
| `/copy` | Copy the last response to the clipboard |
| `/version` | Show the Kolega Code version |
| `/update` | Update Kolega Code to the latest version |
| `/quit` | Save the session and exit |

Run `/model` to open a selectable list of supported models for the current
provider. You can also switch directly with `/model <name>`.

Run `/effort` to open a selectable list of supported effort values for the
active model. You can also switch directly with `/effort <level>`.

## Skills

Any [skill](../../skills/) discovered in `.agents/skills/` is available as
`/skill-name`. Running it loads the skill's instructions (and a manifest of its
bundled resources) into the conversation. Pass extra text after the command to run
the skill against a specific request:

```text
/release-notes summarize changes since the last tag
```

Use `/skills` at any time to see what's available in the current project.
