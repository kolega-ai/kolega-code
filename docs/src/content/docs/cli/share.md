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

The event log is the bulk of the file and is compressed before embedding, so a
long session usually lands well under a megabyte.

### Options

| Option | Effect |
| --- | --- |
| `--out PATH` | Write somewhere other than the default. |
| `--dir` | Write a directory of separate files instead of one HTML file. |
| `--zip` | Write a `.zip` of the directory form. Implies `--dir`. |
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
- Play, pause, and 1x / 2x / 4x / max playback. Long idle gaps are compressed, so
  a session with a 45-minute break does not replay a 45-minute break.
- A scrubber with a tick per turn, plus a turn list for jumping straight to a
  prompt.
- Live context-window usage, sub-agent activity, and terminal output.
- A theme switcher covering every theme the TUI ships.

Keyboard: <kbd>Space</kbd> plays and pauses, <kbd>←</kbd>/<kbd>→</kbd> seek five
seconds, <kbd>Home</kbd>/<kbd>End</kbd> jump to the start or end.

## What is removed before sharing

Export is filtered on the assumption that the file will leave your machine.

- **Secrets are scrubbed** from every exported string using the same detection the
  diagnostics bundle uses, including any value the agent happened to print.
- **Opaque provider payloads are dropped.** Only tool output and images are
  included. Reasoning signatures and encrypted reasoning exist solely so a
  conversation can be replayed to the model that produced it, carry no display
  value, and are never shared.
- **Local filesystem paths are stripped** from artifact references, so a replay
  does not disclose your directory layout.

The command prints a summary of exactly what was redacted and dropped. Read it
before sending a replay to someone else.

:::caution
Redaction is thorough but it cannot know what is sensitive in *your* project. A
replay contains real file contents, real command output, and real prompts. Review
one before sharing it outside your team.
:::

## Sharing a session that is still running

Run [`/share`](/tui/slash-commands/#sharing-a-live-session) inside the TUI. It
starts a local server and hands you a link, already on your clipboard. Open it and
the player follows the session as it happens — reasoning, output, and tool calls
appear as they are produced, with a **LIVE** badge in the transport bar. Scrub back
to re-read something and the view stays where you put it; the badge becomes **JUMP
TO LIVE** to return to the edge.

Watching is **read-only**. A viewer cannot type, approve a permission prompt, or
interrupt a turn.

### Letting someone off your machine watch

`/share` binds to loopback, so nothing outside your machine can reach it. There is
no built-in relay: to let someone else watch, you forward the port yourself
through a tunnel you control.

Leave the server on **loopback** — the tunnel connects to it locally, so binding
wider only widens exposure. Give the share a fixed port so the tunnel keeps
pointing at the right place:

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

The token has to survive the swap. It is the only thing gating access, and the
player needs it to load its own files and open its WebSocket.

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
