---
title: Hooks
description: Run shell commands, Python callables, or LLM checks on agent lifecycle events to observe, block, or modify what the agent does.
---

**Hooks** let you attach handlers to the agent's **lifecycle events** — points like
"before a tool runs", "a prompt was submitted", or "the agent is about to stop". A hook
can **observe** (log, notify), **block** (deny a tool, end a turn, keep the agent
working), or **modify** (rewrite a tool's input or output, inject context).

This is how you add guardrails ("never run `rm -rf`"), automation ("format files after
every edit"), and quality gates ("don't stop until the tests pass") without changing the
agent itself.

## Where hooks live

Hooks are declared in JSON, in two places that are **merged** (global first, project last):

- **Global / user:** `hooks.json` in the Kolega Code state directory — always active. The
  state directory is platform-specific:
  - macOS: `~/Library/Application Support/kolega-code/hooks.json`
  - Linux: `$XDG_STATE_HOME/kolega-code/hooks.json` (defaults to `~/.local/state/kolega-code/hooks.json`)
  - Windows: `%LOCALAPPDATA%\kolega-code\hooks.json`

  Override the location with the `KOLEGA_CODE_STATE_DIR` environment variable or the
  `--state-dir <dir>` flag (the file is then `<dir>/hooks.json`).
- **Project:** `<project>/.kolega/hooks.json` — alongside `permissions.json`. Because a
  project file can run arbitrary commands from a cloned repo, project hooks are
  **disabled until you trust the project** (see [Trust](#trusting-project-hooks)).

Both files use the same shape:

```json
{
  "schema_version": 1,
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "execute_terminal_command|run_command_tracked",
        "hooks": [
          { "type": "command", "command": "./.kolega/guard-bash.sh", "timeout": 30 }
        ]
      }
    ]
  }
}
```

- The `hooks` map is keyed by **event name**.
- Each entry has a **`matcher`** and a list of **hook handlers**.
- `matcher` is tested against the tool name for tool events (or a source string for other
  events): `""` or `"*"` matches everything; `"Edit|Write"` is an exact-name OR list;
  anything else is a regular expression (e.g. `"mcp__.*"`).
- Handler lists from the global and project files are concatenated, so the global handler
  sees the action first and the project handler last.

## Lifecycle events

| Event | When it fires | What a "block" does |
|---|---|---|
| `SessionStart` | The agent session begins | Advisory; can inject starting context |
| `UserPromptSubmit` | You submit a prompt, before the agent sees it | Ends the turn; the reason is shown as a warning |
| `PreToolUse` | Before a tool runs (after the permission gate) | Denies the tool; the reason is returned to the agent as a tool error, so it can adjust |
| `PostToolUse` | After a tool succeeds (and on failure) | Ends the turn; the reason is shown as a warning |
| `PreCompact` | Before the conversation is compacted | Advisory |
| `PostCompact` | After the conversation is compacted | Advisory; payload reports before/after token counts |
| `Stop` | The agent is about to finish its turn | Keeps the agent working; the reason becomes its next instruction |
| `SubagentStop` | A dispatched sub-agent finished | Advisory; can annotate the result the parent sees |
| `Notification` | A permission prompt is about to be shown | Advisory (desktop notifications, sounds) |
| `SessionEnd` | The session ends | Advisory (cleanup, final logging) |

Hooks can also **modify** rather than block:

- `PreToolUse` can return `updatedInput` to rewrite the tool's arguments.
- `PostToolUse` can return `updatedToolOutput` to replace what the model sees, or
  `additionalContext` to append to it.
- `SessionStart` / `UserPromptSubmit` can return `additionalContext` to inject text.

## Hook types

### `command` — run a shell program

The event is sent as JSON on **stdin**. The hook communicates back through its exit code:

- **exit 0** — success. Optional JSON on stdout controls behavior:
  ```json
  {
    "hookSpecificOutput": {
      "permissionDecision": "deny",
      "permissionDecisionReason": "rm -rf is not allowed here",
      "updatedInput": { "command": "ls" },
      "updatedToolOutput": "…",
      "additionalContext": "…"
    },
    "systemMessage": "shown to the user",
    "continue": false
  }
  ```
- **exit 2** — block. Whatever the hook wrote to **stderr** becomes the reason.
- **any other code** — a non-blocking error: it is logged and the action proceeds.

```json
{ "type": "command", "command": "./.kolega/format.sh", "timeout": 30 }
```

Commands run with the project directory as the working directory and are split like a
shell argument list (no shell features such as pipes — call a script if you need them).

### `python` — run an in-process callable

Point at an importable `module.path:function`. It receives a `LifecycleEvent` and returns
a `HookOutcome` (or a dict with the same fields). Runs in the agent's process — best for
first-party checks.

```json
{ "type": "python", "callable": "myproject.hooks:block_secrets", "timeout": 15 }
```

```python
from kolega_code.hooks import HookOutcome

def block_secrets(event):
    inputs = event.payload.get("tool_input", {})
    if "AWS_SECRET" in str(inputs):
        return HookOutcome.deny("Refusing to write a secret to disk.")
    return HookOutcome.empty()
```

### `prompt` and `agent` — let an LLM decide

For judgment calls rather than deterministic rules. Both return a yes/no decision as
`{"ok": true}` (allow) or `{"ok": false, "reason": "…"}` (block); the per-event meaning of
a block is the same as the table above.

- **`prompt`** sends your prompt plus the event data to a model (the fast model by
  default; `"model"` may be `"fast"`, `"long"`, or `"thinking"`). Use `$EVENT` in the
  prompt to interpolate the event JSON.

  ```json
  {
    "type": "prompt",
    "prompt": "Have all of the user's requested tasks been completed? $EVENT",
    "model": "fast",
    "timeout": 30
  }
  ```

- **`agent`** spawns a full sub-agent that can read files, search the code, and run
  commands to verify a condition before answering. Heavier, with a longer default timeout.
  Because an agent hook uses tools, it is **not allowed on `PreToolUse`/`PostToolUse`**
  (that would recurse); use it on `Stop`, `SubagentStop`, `UserPromptSubmit`, or the
  session events.

  ```json
  {
    "type": "agent",
    "prompt": "Run the test suite. Only report ok:true if every test passes.",
    "timeout": 120
  }
  ```

  Tool calls made by an agent hook do **not** re-trigger tool hooks, so there is no
  infinite loop.

## Examples

**Block dangerous shell commands (project, command hook):**

```json
{
  "schema_version": 1,
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "execute_terminal_command|run_command|run_command_tracked",
        "hooks": [{ "type": "command", "command": "./.kolega/deny-rm.sh" }]
      }
    ]
  }
}
```

**Don't stop until the tests pass (prompt hook):**

```json
{
  "schema_version": 1,
  "hooks": {
    "Stop": [
      {
        "matcher": "*",
        "hooks": [{
          "type": "prompt",
          "prompt": "Based on the conversation, did the agent finish ALL requested work AND leave the tests passing? $EVENT"
        }]
      }
    ]
  }
}
```

**Verify tests actually pass before stopping (agent hook):**

```json
{
  "schema_version": 1,
  "hooks": {
    "Stop": [
      {
        "matcher": "*",
        "hooks": [{
          "type": "agent",
          "prompt": "Run the project's test command and confirm it exits cleanly.",
          "timeout": 120
        }]
      }
    ]
  }
}
```

## Trusting project hooks

A project's `.kolega/hooks.json` can run arbitrary commands, so it is **ignored until you
explicitly trust the project**. Global/user hooks are always trusted.

When an untrusted project defines hooks, Kolega Code tells you and runs only the global
hooks. To enable the project's hooks, launch with:

```bash
kolega-code --trust-hooks          # TUI
kolega-code ask "…" --trust-hooks  # non-interactive
```

Trust is recorded once (per resolved project path) in your user settings, so future runs
in that project enable its hooks automatically.

## Safety and behavior notes

- **Failure is isolated.** A hook that crashes, times out, or prints garbage is logged and
  the action proceeds — a broken hook never wedges the agent. Only a clean `exit 2`,
  `permissionDecision: "deny"`, or `ok: false` blocks.
- **Each handler has a `timeout`** (seconds). On timeout the process is killed and treated
  as a non-blocking error.
- **Tool hooks apply to sub-agents too**, so a `PreToolUse` guard also covers tools used by
  dispatched sub-agents.
- A `Stop` hook can force the agent to keep working only a bounded number of times per
  turn, so a misbehaving "don't stop" hook cannot loop forever.
