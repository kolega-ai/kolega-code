---
title: ask
description: Run a single prompt non-interactively and print the answer.
---

`kolega-code ask` runs one prompt against the agent and prints the response. It's
the scriptable counterpart to the TUI — useful for automation, quick questions, and
piping output into other tools.

```bash
kolega-code ask "<prompt>" [options]
```

## Arguments & options

| Argument / option | Description |
| --- | --- |
| `prompt` | The prompt to send (required, positional) |
| `--project <PATH>` | Project directory to work in (default `.`) |
| `--save` | Persist the session after the prompt completes |
| `--json` | Emit response chunks and events as JSON |
| `--browser-visible` | Launch visible Playwright browser windows |
| `--session <ID>` | Resume or create a specific session |
| `--state-dir <PATH>` | Directory for CLI session state |

All the [global model options](../overview/#global-model-options)
(`--provider`, `--model`, `--fast-model`, …) are also accepted.
`ask` requires a provider/model from those options, environment variables, or
saved Settings. API key variables alone are not enough.

## Examples

Ask a question about the current project:

```bash
kolega-code ask "summarize this repository" --project .
```

Pick a provider and model just for this run:

```bash
kolega-code ask "summarize this repository" --project . \
  --provider deepseek --model deepseek-v4-pro
```

Save the result as a resumable session:

```bash
kolega-code ask "add unit tests for the parser" --project . --save
```

## File mentions

Just like the TUI composer, `ask` understands `@` file mentions. Referenced files
are attached to the prompt:

```bash
kolega-code ask "explain @src/main.py and suggest improvements" --project .
```

If a mention can't be resolved, the CLI notes it on stderr and sends the text
as-is:

```text
Note: @missing/file.py not found, sent as plain text
```

## Skills

If your prompt is a skill command (e.g. `/skills` or `/my-skill`), `ask` resolves
it against the project's [Agent Skills](../../skills/):

- `kolega-code ask "/skills"` prints the available-skills catalog.
- `kolega-code ask "/my-skill"` (with no extra text and no `--save`/`--session`)
  prints the skill's activation content.
- `kolega-code ask "/my-skill do the thing"` activates the skill and runs the
  remaining prompt.

## JSON output

With `--json`, the command streams newline-delimited JSON objects, each tagged with
a `kind`, so you can parse them programmatically:

```bash
kolega-code ask "count the Python files" --project . --json
```

The stream includes:

| `kind` | Meaning |
| --- | --- |
| `chunk` | A streamed piece of the response |
| `event` | An agent event (sub-agent activity, tool calls, terminal output) |
| `skill` | Skill activation metadata |
| `summary` | A final object with the chunk count and `session_id` |

In plain (non-JSON) mode, the answer is written to **stdout** while sub-agent and
tool activity is reported on **stderr** — so piping stdout gives you just the
answer.
