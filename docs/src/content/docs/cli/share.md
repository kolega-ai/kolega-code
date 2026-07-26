---
title: share
description: Export a session as a self-contained replay you can send to someone.
---

`kolega-code share export` turns a recorded session into a static replay: a small
directory of JSON and assets that plays in any browser, with pause, seek, and
variable-speed playback.

It is a way to show someone what an agent actually did — for a code review, a bug
report, a demo, or your own record of a long session — without them installing
anything.

```bash
kolega-code share export <session-id>
```

That writes `./<session-id>-replay/`. Open `index.html` in a browser, or drop the
directory on any static host and send the link.

## Options

| Option | Effect |
| --- | --- |
| `--out DIR` | Write the bundle to `DIR` instead of the default location. |
| `--zip` | Produce a single `.zip` archive instead of a directory. |
| `--title TEXT` | Title shown in the player. Defaults to the session title. |
| `--theme SLUG` | Theme the replay opens with. Defaults to your active theme. |
| `--state-dir DIR` | Read sessions from a different state directory. |

Find session IDs with [`kolega-code sessions list`](/cli/sessions/).

## What the player gives a viewer

- The full transcript: prompts, assistant responses, reasoning, tool calls and
  their results, and rendered diffs.
- Play, pause, and 1x / 2x / 4x / max playback. Long idle gaps are compressed, so
  a session with a 45-minute break does not replay a 45-minute break.
- A scrubber with a tick per turn, plus a turn list for jumping straight to a
  prompt.
- Live context-window usage, sub-agent activity, and terminal output.
- A theme switcher covering every theme the TUI ships.

Keyboard: <kbd>Space</kbd> plays and pauses, <kbd>←</kbd>/<kbd>→</kbd> seek five
seconds, <kbd>Home</kbd>/<kbd>End</kbd> jump to the start or end.

## What is removed before sharing

Export is filtered on the assumption that the bundle will leave your machine.

- **Secrets are scrubbed** from every exported string using the same detection the
  diagnostics bundle uses, including any value the agent happened to print.
- **Opaque provider payloads are dropped.** Only tool output and images are
  included. Reasoning signatures and encrypted reasoning exist solely so a
  conversation can be replayed to the model that produced it, carry no display
  value, and are never shared.
- **Local filesystem paths are stripped** from artifact references, so a bundle
  does not disclose your directory layout.

The command prints a summary of exactly what was redacted and dropped. Read it
before sending a bundle to someone else.

:::caution
Redaction is thorough but it cannot know what is sensitive in *your* project. A
replay contains real file contents, real command output, and real prompts. Review
a bundle before sharing it outside your team.
:::

## Sessions recorded before this feature

Replay is built from a session's event stream. Sessions recorded by earlier
versions contain conversation history but no presentation events, so there is
nothing to play back; the command reports this and exits rather than writing an
empty bundle.
