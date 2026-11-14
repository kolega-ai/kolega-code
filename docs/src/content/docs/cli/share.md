---
title: share
description: Send someone a session — as one replay file, or as a live link.
---

There are two ways to show someone what an agent did.

| You want | Use |
| --- | --- |
| To send a recording | `kolega-code share export` — one HTML file |
| Someone to watch you work now | [`/share`](/tui/slash-commands/#sharing-a-live-session) in the TUI |

## Sending a recording

```bash
kolega-code share export <session-id>
```

That writes `./<session-id>-replay.html`: a single file holding the whole session
— transcript, assets, and any images — with secrets scrubbed. Double-click it and
it plays. Email it, drop it in a chat, attach it to a bug report. The recipient
needs no server, no unzipping, and nothing installed.

Find session IDs with [`kolega-code sessions list`](/cli/sessions/).

The event log is the bulk of the file and is compressed before embedding. A short
session is a hundred kilobytes or so; a long one full of command output runs to a
megabyte or two. Images are embedded too, and very large ones are left out and
shown as a badge rather than making the file too big to email — the command says
when that happens.

User-attached images captured by the TUI or ACP appear beneath their prompt,
including prompts with no text. Unavailable images are shown as badges. Older
recordings cannot recover attachments they never captured, and replay never
downloads a remote image URL to fill in a missing attachment.

### Options

| Option | Effect |
| --- | --- |
| `--out PATH` | Write somewhere other than the default. |
| `--dir` | Write a directory of separate files instead of one HTML file. |
| `--zip` | Write a `.zip` of the directory form. Implies `--dir`. `.zip` is appended unless your path already ends in it. |
| `--title TEXT` | Title shown in the player. Defaults to the session title. |
| `--theme SLUG` | Theme the replay opens with. Defaults to your active theme. |
| `--state-dir DIR` | Read sessions from a different state directory. |

`--dir` exists for publishing a replay on a static host, where separate cacheable
files are the better shape.

:::caution
A directory bundle **must be served over HTTP**. Opening its `index.html` from a
`file://` URL shows a blank page, because browsers refuse to load module scripts
and run `fetch` from that origin. This is exactly why the single file is the
default — it has nothing to fetch.
:::

## What the player gives a viewer

- The full transcript: prompts, assistant responses, reasoning, tool calls and
  their results, rendered diffs, and any images captured during the session.
- Tool output is collapsed to a one-line header — the tool and how it ended —
  and opens when clicked, so a wall of command output never buries the
  conversation around it.
- Each delegated agent appears as one card saying who it is, what it was asked,
  whether it is still running, and how many steps it took. Click it, or its
  entry in the Sub-agents panel, to read that agent's thread on its own and then
  come back. Parallel agents interleave line by line, so a merged stream is
  unreadable once more than one is running.
- Opening an agent also selects its terminal output. Returning to the session
  restores the main agent's terminal; parallel and background workers do not
  mix their commands into it. Agent details retain reported parent links,
  delegation depth, model routing, background status and token totals.
- A gigacode run shows its own name, each phase as it starts, and how it
  finished, with its agents' cards in between.
- Play, pause, and 1x / 2x / 4x / max playback. Long idle gaps are compressed, so
  a session with a 45-minute break does not replay a 45-minute break.
- A scrubber with a tick per turn, plus a turn list for jumping straight to a
  prompt.
- Live context-window usage, sub-agent activity, and terminal output.
- A read-only, session-wide task list with pending and completed tasks, a
  completion count, and the source Markdown.
- A read-only, session-wide queue showing recorded pending-input previews in order.
- A read-only, session-wide Goal panel showing the latest recorded goal observation.
- Session-wide token and cost snapshots, including coverage and recording health.
- A theme switcher covering every theme the TUI ships.

Keyboard: <kbd>Space</kbd> plays and pauses, <kbd>←</kbd>/<kbd>→</kbd> seek five
seconds, <kbd>Home</kbd>/<kbd>End</kbd> jump to the start or end.

Older recordings remain playable, but missing parent, routing and terminal-owner
metadata cannot be reconstructed. Untagged terminal output stays in the main
terminal rather than being assigned to a guessed worker.

### Background reports

Live shares, single-file replays and directory exports show recorded background
reports in the main session transcript, separately from the worker's own steps.
Each entry retains the full report, task, job and completion IDs, and whether
the job completed, failed, was cancelled or was interrupted. Recorded launch,
model-routing and workspace details remain available when present; missing
metadata does not create a guessed worker.

A report appears when its delivery is recorded, not merely when the worker
finishes. **Context** means it was inserted into the live main-agent context.
**Archived** means it was retained while retiring that context, without being
inserted into the successor's live context. Both are read-only and follow
playback and seeking; repeated delivery records for the same completion appear
only once.

Large reports load their complete recorded text, including in single-file
exports. If those bytes are unavailable, the player labels the recorded preview
instead. Recording and displaying a report does not add another semantic
completion or change usage accounting, `ask --json`, or ATIF exports.

Older recordings without delivery events cannot establish whether or when a
report reached the main agent. That evidence is **unknown**, not proof that the
job failed or was never delivered. Resuming or exporting a session does not
backfill missing delivery history.

### Task list

The task-list panel shows the latest recorded update to the session's shared
list. It stays session-wide when you open a delegated agent's thread; a worker's
own updates do not replace it. Task-list updates do not appear again as prose in
the transcript.

The list follows playback and seeking, including when you seek backward or
return to the live edge. It is available in live shares, single-file replays and
directory bundles. Tasks and source Markdown are displayed as literal text,
not executable HTML. The panel is **read-only**: viewers cannot edit tasks or
check them off.

Before any task-list update has been recorded, the panel says **No task list
recorded yet**. That means the list is unknown, not empty or completed. An
explicit empty update instead shows **Task list cleared**. Older recordings can
show a list only where an update was actually captured. Resuming a session may
record its saved list at that point, but does not invent earlier task-list
history. Recording and displaying task updates does not add them to the model's
conversation history or change its prompts.

### Queue

The Queue panel shows the latest recorded observation of pending inputs, in queue
order. It follows live updates, playback and seeking in live shares, single-file
replays and directory bundles, and stays session-wide when viewing a worker.
Entries show bounded literal text previews, attachment image counts and recorded
peer attribution—not raw attachment bytes, local paths or model framing. The
panel does not load images or offer downloads or queue controls.

These are **read-only historical previews**, not recoverable pending inputs, a
durable queue or a way to resume queued work. A **dequeued** observation means an
input was removed for processing, **not** that semantic delivery succeeded; no
queue observation is a delivery acknowledgment. Queue updates do not create
conversation turns.

Before any queue observation was recorded, the queue is **unknown**, not empty.
`ask --json` remains semantic-only and does not record queue observations.
An explicitly recorded empty snapshot is shown as empty. Older sessions do not
gain invented queue history. Export redaction also applies to queue previews and
peer attribution; live shares remain unredacted.

### Goal

The Goal panel shows recorded goal observations: started, replaced, evaluated,
paused, resumed, completed and cleared, as well as saved-state snapshots. It
follows live updates, playback and seeking in live shares, single-file replays
and directory bundles. Opening a worker's thread does not narrow the panel or
replace the session's goal with a worker's state. Goal observations do not add
transcript rows or conversation turns.

The panel retains the full recorded condition, evaluation reason, counters,
timestamps, flags and status note. Text is displayed literally, not as
executable HTML. This is **read-only evidence**: there are no start, clear, pause
or resume controls. A recorded evaluation result or completion is not a new
verification by the player, and the panel cannot execute work, resume a goal or
recover goal-loop state.

Before the first recorded observation, the goal is **unknown**, not absent or
completed. An explicit snapshot with no goal or a recorded clear instead
establishes that no goal was set at that point. Older recordings do not gain
invented goal history. A saved-state snapshot records what was observed when
the session opened; it does not backfill earlier lifecycle events.

Normal `ask` can record goal observations without changing its existing goal
execution or recovery semantics. `ask --json` writes **no goal presentation
events**, preserving its semantic-only, gaplessly sequenced stream. Goal
observations do not change model history, semantic goal accounting or ATIF
exports. Export redaction applies to goal text; live shares remain unredacted.

### Token and cost usage

The read-only usage panel follows recorded session-wide snapshots during live
viewing, replay and seeking. Opening a worker's thread does not narrow these
totals. They include recorded requests across earlier turns and context resets;
the separate context-window gauge is not a lifetime token counter.

Unknown values appear as unavailable, not zero. Partial evidence is labelled as
a known subtotal, and request coverage is shown separately from persistence
health. An older recording without usage snapshots does not gain invented
totals. Resuming it can record a current snapshot from the available evidence,
but cannot reconstruct missing historical usage.

Costs retain their recorded decimal precision and distinguish reported values
from estimates. Currencies and non-monetary credits are displayed separately,
without conversion or a combined balance. Missing costs are not treated as free
requests. Sessions without full request accounting, including gateway sessions,
can therefore show incomplete coverage.

These snapshots are presentation data. They do not change model history,
semantic JSON output, or ATIF exports, and viewers cannot edit them.

## What is removed before sharing

Export is filtered on the assumption that the file will leave your machine.

- **Secrets are scrubbed** from every exported string using the same detection the
  diagnostics bundle uses, including any value the agent happened to print.
- **Opaque provider payloads are dropped.** Only tool output and images are
  included. Reasoning signatures and encrypted reasoning exist solely so a
  conversation can be replayed to the model that produced it, carry no display
  value, and are never shared.
- **Your home directory is rewritten to `~`** everywhere it appears, and local
  paths are dropped from artifact references, so a replay does not disclose your
  directory layout.

The command prints a summary of exactly what was redacted and dropped. Read it
before sending a replay to someone else.

:::caution
Secret detection is **pattern matching**, not proof. It knows the common
credential shapes and every API key you have configured in Kolega Code, but a
secret in a shape nobody anticipated can still get through. A replay also
contains real file contents, real command output, and real prompts. Review one
before sharing it outside your team. Image pixels are copied without redaction:
secret detection does not remove credentials or other private information
visible in a screenshot.
:::

## Sharing a session that is still running

Run [`/share`](/tui/slash-commands/#sharing-a-live-session) inside the TUI. It
starts a local server and hands you a link, already on your clipboard. Open it and
the player follows the session as it happens — reasoning, output, and tool calls
appear as they are produced, with a **LIVE** badge in the transport bar. Scrub back
to re-read something and the view stays where you put it; the badge becomes **JUMP
TO LIVE** to return to the edge.

Watching is **read-only**. A viewer cannot type, approve a permission prompt, or
interrupt a turn. A link reaches only the session you shared it from; every other
session on your machine is invisible to it.

Every HTTP request and WebSocket connection requires the link's token. The server
also validates Host and Origin headers: loopback shares accept loopback hosts,
and `/share lan` accepts the address it binds. For those native hosts, browser
origins must match the requested scheme, host and port. Tunnel browser origins
must match an explicitly configured origin, including its scheme and port.
Requests without Origin, including top-level browser navigation and non-browser
clients, still need a valid Host and token.

:::danger
**A live link is not redacted.** `share export` scrubs secrets and rewrites your
home directory; `/share` serves the session exactly as recorded, because it is
showing you what is happening rather than producing an artefact. Whatever the
agent printed — including credentials it read or echoed — is visible to anyone
holding the link. Send an exported replay instead when the recipient should not
see raw output, and run `/share stop` when you are done.
:::

### Letting someone off your machine watch

`/share` binds to loopback, so nothing outside your machine can reach it. There is
no built-in relay: to let someone else watch, you forward the port yourself
through a tunnel you control.

**Allow the tunnel's exact public origin before starting Kolega Code.** For
example, after your tunnel has assigned its address:

```bash
KOLEGA_CODE_SHARE_ALLOWED_ORIGINS=https://share.example.com kolega-code
```

Set the variable to the actual origin shown by Tailscale, ngrok or your SSH
forwarding setup, including a non-default port when needed. Multiple origins can
be comma-separated. Paths, credentials, queries, fragments and wildcards are
rejected; this is not a suffix allowlist.

Leave the server on **loopback** — the tunnel connects to it locally, so binding
wider only widens exposure. Give the share a fixed port so the tunnel keeps
pointing at the right place, running this in the process you just launched:

```bash
/share 8765
```

The tunnel gives you a new origin; everything after it stays the same. Take the
link `/share` gave you and swap the `scheme://host:port` for the tunnel's, keeping
the path and the `?token=` exactly as they are:

```
local    http://127.0.0.1:8765/s/<session-id>?token=<token>
tunnel   https://<tunnel-host>/s/<session-id>?token=<token>
```

The token has to survive the swap: the player needs it to load its own files and
open its WebSocket.

An approved origin also approves its hostname, so both proxies that preserve
the public Host and proxies that rewrite it to the local upstream work.
This does not approve other origins on that hostname or downgrade HTTPS to HTTP.
These are browser-Origin checks, not HTTPS-only transport enforcement; configure
that at your tunnel or reverse proxy.

Forwarded headers do not bypass these checks. Without explicit configuration,
the public Host or the browser's WebSocket Origin is refused even when the link
contains a valid token. The allowlist does not replace the token or change which
session is shared. Stop the share and restart Kolega Code to change its
environment-based allowlist.

#### Tailscale

Reachable by your own devices only, which is the option to prefer:

```bash
tailscale serve --bg 8765
tailscale serve status      # shows the https://<machine>.<tailnet>.ts.net address
tailscale serve reset       # stop serving
```

To reach someone who is not on your tailnet, `tailscale funnel --bg 8765` puts the
same address on the public internet. `tailscale funnel status` and
`tailscale funnel reset` manage it.

#### ngrok

A temporary public URL, useful when the other person cannot join your tailnet:

```bash
ngrok config add-authtoken <your-token>   # once
ngrok http 8765
```

ngrok prints a `https://<random>.ngrok.app` forwarding address; combine it with the
path and token as above. On free tunnels the first visit may show an ngrok
interstitial page — clicking through reaches the player.

#### SSH

If you already have a box both of you can reach, no extra tool is needed:

```bash
ssh -R 8765:127.0.0.1:8765 you@jump-host
```

:::danger
A tunnel makes the session readable by anyone who has the link, and the link
carries its token in the query string, so it lands in browser history and in any
chat log you paste it into. Sessions contain file contents, command output, and
prompts. Prefer Tailscale over a public tunnel, treat the link as a credential,
and stop the tunnel and the share when you are done.
:::

If you only need to send someone a recording rather than live access, export a
replay file instead — it needs no server and no tunnel at all.

## Sessions recorded before this feature

Replay is built from a session's event stream. Sessions recorded by earlier
versions contain conversation history but no presentation events, so there is
nothing to play back; the command reports this and exits rather than writing an
empty file.
