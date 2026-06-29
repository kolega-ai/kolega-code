---
title: Diagnostics & Bug Reports
description: How Kolega Code records what happened locally so a freeze or error becomes debuggable, and how to share it with /bug.
---

Kolega Code keeps an always-on, **local** diagnostics record so that hard-to-reproduce
problems — especially "it froze" or "it went unresponsive" — leave behind enough
evidence to debug. Nothing is uploaded anywhere; the record stays on your machine
and is only shared when you explicitly run `/bug` and send the result.

## Where it lives

The diagnostics folder sits alongside your sessions in the state directory:

| Platform | Location |
| --- | --- |
| macOS | `~/Library/Application Support/kolega-code/diagnostics/` |
| Linux | `~/.local/state/kolega-code/diagnostics/` (or `$XDG_STATE_HOME/kolega-code/diagnostics/`) |
| Windows | `%LOCALAPPDATA%\kolega-code\diagnostics\` |

Set [`KOLEGA_CODE_STATE_DIR`](../../configuration/environment-variables/) to move the
whole state directory, diagnostics included. The folder and its files are created
owner-only (`0700`/`0600`) where the platform supports it.

## What gets captured

| File | Contents |
| --- | --- |
| `session-<id>.jsonl` | The per-session **timeline**: one JSON line per event — startup snapshot, LLM request boundaries, errors, context/compaction status, tool calls, and stalls |
| `stalls.log` | Full thread-stack dumps taken automatically whenever the UI stops responding |
| `manual-dump.log` | Stacks captured on demand with the `SIGUSR1` escape hatch (below) |
| `crash-<ts>.log` | The traceback from an unhandled exception, if the app ever exits with one |
| `bug-<ts>.zip` | A zip assembled by `/bug` for sharing |

The most recent sessions are kept (older `session-*.jsonl` files are pruned), and a
single session file is capped in size, so the folder does not grow without bound.

## The responsiveness watchdog

The headline piece. When the UI "freezes," the cause is almost always the event
loop being blocked, and there are two very different reasons it can happen. The
watchdog tells them apart:

- **Blocked by synchronous work (a true hang).** A background thread expects a
  heartbeat roughly every second. If the heartbeat stops for ~5 seconds, the loop is
  stuck inside some synchronous call — the watchdog records an `event_loop_stalled`
  entry and **dumps every thread's stack** to `stalls.log`. The main thread's frame
  in that dump is *exactly* the code that was blocking. When the heartbeat resumes it
  records `event_loop_recovered` with how long the stall lasted.
- **Waiting on the network.** Here the loop is *not* blocked, so the watchdog stays
  quiet — but the timeline shows an LLM request that started and never finished. A
  request open for minutes with no matching completion points at a stalled network
  read rather than a CPU hang.

That distinction is usually the first thing you need to know, and the diagnostics
capture it without any action on your part.

### If the app is fully wedged

On macOS and Linux you can force a stack dump even when the process is completely
unresponsive:

```bash
kill -USR1 $(pgrep -f kolega-code)
```

The stacks are written to `manual-dump.log` in the diagnostics folder.

## `/diagnostics`

Run `/diagnostics` in the composer for an at-a-glance summary printed straight into
the transcript:

- version, platform, and terminal;
- active provider/model and endpoint;
- which providers have keys configured (never the keys themselves);
- how many loop stalls and LLM errors were recorded this session;
- the path to the diagnostics log.

## `/bug`

Run `/bug` to package everything into a single shareable zip at
`diagnostics/bug-<timestamp>.zip`:

- `summary.md` — the `/diagnostics` snapshot;
- the session timeline(s) and any stack dumps / crash logs;
- `session.json` — the full conversation for this session.

The zip path is copied to your clipboard, and Kolega Code prints a link to open a
new GitHub issue.

## Privacy model

The guiding rule: **content is kept, credentials are removed.**

- Diagnostics are **local-first** and shared only when you choose to send a `/bug`
  bundle.
- Ordinary content — prompts, file contents, tool output, error messages — is kept
  **unredacted**, because that is what makes a report actionable (and it is the same
  content already stored in your local session files).
- **Credentials are always scrubbed**, at write time and again when a bundle is
  assembled: API keys and tokens from your settings and environment, and common
  patterns such as `Authorization: Bearer …`, `sk-…`/`xai-…`, and `*_API_KEY=…` are
  replaced with `‹secret›`. The startup snapshot records *which* providers have keys,
  never the values.

:::caution
A `/bug` bundle contains your conversation and file contents. Credentials are
scrubbed, but review the bundle before posting it somewhere public.
:::

## Turning it off

Diagnostics are on by default. To disable the log and the watchdog entirely, set:

```bash
KOLEGA_CODE_NO_DIAGNOSTICS=1 kolega-code .
```

## Filing a good bug report

1. Reproduce the problem (or capture stacks with `SIGUSR1` if it is wedged).
2. Run `/bug` — or, if the app has exited, grab the `diagnostics/` folder directly.
3. Skim the bundle to confirm there is nothing private you would rather not share.
4. Attach it to a [new issue](https://github.com/kolega-ai/kolega-code/issues/new)
   with a short note on what you were doing.

For setup and configuration problems specifically, [`kolega-code doctor`](../../cli/doctor/)
is the faster first check.
