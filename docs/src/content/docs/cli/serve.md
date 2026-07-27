---
title: serve
description: Serve recorded sessions to browsers and other clients over HTTP.
---

`kolega-code serve` starts a local server that exposes your recorded sessions over
HTTP and WebSocket. It is built in — no extra install, no optional dependency.

```bash
kolega-code serve
```

```
Serving sessions on http://localhost:8765
  Session list   http://localhost:8765/
  API reference  http://localhost:8765/docs
```

Two things it is for: watching a session in a browser instead of a terminal, and
giving other frontends a stable API to build against.

## Options

| Option | Default | Effect |
| --- | --- | --- |
| `--port N` | `8765` | Port to listen on. |
| `--bind ADDR` | `127.0.0.1` | Address to bind. Anything other than loopback exposes your sessions. |
| `--token TOKEN` | none | Require this bearer token on every request. |
| `--state-dir DIR` | | Read sessions from a different state directory. |

## Following a live session

Open `/s/<session-id>` for a session that is still running and the player follows
it: reasoning, output, and tool calls appear as they happen, with a **LIVE** badge
in the transport bar. Scrub back to re-read something and the view stays where you
put it; the badge becomes **JUMP TO LIVE** to return to the edge. If the connection
drops it reattaches and resumes where it left off.

To share a session you are in the middle of, you do not need this command at all —
run [`/share`](/tui/slash-commands/#sharing-a-live-session) in the TUI and it starts
a server and hands you the link.

Building your own client? The same WebSocket the player uses replays a session's
backlog and then follows new events:

```
ws://localhost:8765/api/sessions/<session-id>/stream?from_seq=1
```

Reconnect with `from_seq` set to one past the last sequence number you saw and you
resume exactly where you left off — no gap, no duplicates. Sequence numbers are
strictly increasing but **not** contiguous, so page with the `next_from_seq` value
responses give you rather than adding one yourself.

## API surface

Full generated reference at `/docs`.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/sessions` | List sessions with replay metadata. |
| `GET /api/sessions/{id}` | One session, including its turn index. |
| `GET /api/sessions/{id}/events` | Read a range of raw events. |
| `GET /api/sessions/{id}/state` | Server-side folded view, if you would rather not implement the fold. |
| `WS /api/sessions/{id}/stream` | Replay then follow. |
| `GET /api/sessions/{id}/artifacts/{sha256}` | Fetch an image or long tool output an event points at. |
| `GET /api/themes` | Design tokens for every theme. |
| `GET /s/{id}` | The replay player. |

The server is **read-only**. It does not send messages to an agent, cancel a turn,
or answer permission prompts. Someone watching a live session can read everything
in it and change nothing.

When a token is set, a link carrying `?token=...` works in a browser: the server
hands the token to that tab as a cookie so the player's own files and its
WebSocket load. A wrong token is rejected outright rather than falling back to any
cookie already held.

## Sharing beyond your machine

The server binds to loopback, so nothing outside this machine can reach it. There
is no built-in relay: to let someone else watch you forward the port yourself,
through a tunnel you control.

Whichever tool you use, the shape is the same. Leave the server on **loopback** —
the tunnel connects to it locally, so binding wider only widens exposure — and
always set a token, because the tunnel's address is reachable by anyone who has
it.

```bash
kolega-code serve --token "$(openssl rand -hex 16)"
```

Or, for a session you are already in, run [`/share`](/tui/slash-commands/#sharing-a-live-session)
in the TUI. Both listen on port `8765` by default, which is the number the tunnel
commands below forward.

### Building the link you send

The tunnel gives you a new origin; everything after it stays the same. Take the
link you already have and swap the `scheme://host:port` for the tunnel's, keeping
the path and the `?token=` exactly as they are:

```
local    http://127.0.0.1:8765/s/<session-id>?token=<token>
tunnel   https://<tunnel-host>/s/<session-id>?token=<token>
```

The token has to survive the swap. It is the only thing gating access, and the
player needs it to load its own files and open its WebSocket.

### Tailscale

Reachable by your own devices only, which is the option to prefer:

```bash
tailscale serve --bg 8765
tailscale serve status      # shows the https://<machine>.<tailnet>.ts.net address
tailscale serve reset       # stop serving
```

The link becomes `https://<machine>.<tailnet>.ts.net/s/<session-id>?token=<token>`.

To reach someone who is not on your tailnet, `tailscale funnel --bg 8765` puts the
same address on the public internet. `tailscale funnel status` and
`tailscale funnel reset` manage it.

### ngrok

A temporary public URL, useful when the other person cannot join your tailnet:

```bash
ngrok config add-authtoken <your-token>   # once
ngrok http 8765
```

ngrok prints a `https://<random>.ngrok.app` forwarding address; combine it with the
path and token as above. On free tunnels the first visit may show an ngrok
interstitial page — clicking through reaches the player.

### SSH

If you already have a box both of you can reach, no extra tool is needed:

```bash
ssh -R 8765:127.0.0.1:8765 you@jump-host
```

:::danger
A tunnel makes the session readable by anyone who has the link, and the link
carries its token in the query string, so it lands in browser history and in any
chat log you paste it into. Sessions contain file contents, command output, and
prompts. Prefer Tailscale over a public tunnel, treat the link as a credential,
and stop the tunnel and the server when you are done.
:::

Watching is **read-only** in every case: a viewer cannot type, approve a permission
prompt, or interrupt a turn.

If you only need to send someone a recording rather than live access,
[`kolega-code share export`](/cli/share/) produces a static bundle with secrets
scrubbed and needs no server or tunnel at all.
