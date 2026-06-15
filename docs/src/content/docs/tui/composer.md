---
title: Chat Composer
description: File mentions, slash commands, completions, and keyboard shortcuts.
---

The **composer** is the text box at the bottom of the TUI where you type prompts.
It's more than a plain input — it has autocomplete for files and commands, supports
multi-line input, and drives the option lists the agent shows you.

## Sending and editing text

| Keys | Action |
| --- | --- |
| `Enter` | Send the prompt |
| `Shift+Enter` | Insert a newline (keep typing) |
| `Ctrl+J` | Insert a newline (alternative) |

## File mentions with `@`

Type `@` to reference a file. A fuzzy-searchable dropdown of project files appears
as you type:

```text
Explain @src/main.py and how it wires up the CLI
```

- The file index is **gitignore-aware**, so ignored files don't clutter the
  results.
- Mentioned files are attached to your prompt so the agent has their contents.
- Mentions also work in [`kolega-code ask`](../../cli/ask/).

## Slash commands with `/`

Type `/` at the start of the composer to run a command. A dropdown shows matching
commands with descriptions:

```text
/model        Show or switch the active model
/effort       Show or set the active thinking effort
/plan         Switch to plan mode
/context      Show current context token count
```

See the [Slash Commands](../slash-commands/) reference for the full list. Commands
come from three sources — the agent's built-ins, the TUI, and any project
[Skills](../../skills/).

## Working with the completion dropdown

Both `@` and `/` open the same kind of dropdown. Navigate it with:

| Keys | Action |
| --- | --- |
| `Up` / `Down` | Move between matches |
| `Tab` | Accept the highlighted match |
| `Enter` | Accept (or send, if no dropdown is open) |
| `Escape` | Dismiss the dropdown |

## Option lists

When the agent asks you to make a choice — answering a question, or deciding what
to do with a finished plan — the options appear as a **vertical list** above the
composer:

| Keys | Action |
| --- | --- |
| `Up` / `Down` | Move the highlight |
| `1`–`9` | Select an option by number |
| `Enter` | Confirm the highlighted option |

When the list closes, focus returns to the composer so you can keep typing.

## Cancelling and quitting

| Keys | Action |
| --- | --- |
| `Ctrl+C` / `Escape` | Cancel the current generation |
| `Ctrl+Q` | Save the session and quit |
