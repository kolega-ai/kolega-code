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

The server binds to loopback, so by default nothing outside your machine can reach
it. To let someone else watch, put it behind a private network or tunnel you
already trust and always set a token:

```bash
# Tailscale: reachable by devices on your tailnet only
kolega-code serve --bind 0.0.0.0 --token "$(openssl rand -hex 16)"
tailscale serve https / http://127.0.0.1:8765
```

```bash
# ngrok: a temporary public URL
kolega-code serve --token "$(openssl rand -hex 16)"
ngrok http 8765
```

:::danger
`--bind 0.0.0.0` without `--token` lets anyone who can reach the address read your
sessions, including file contents, command output, and prompts. The command warns
when you bind beyond loopback. Prefer a private network over a public tunnel, and
stop the server when you are done.
:::

If you only need to send someone a recording rather than live access,
[`kolega-code share export`](/cli/share/) produces a static bundle with secrets
scrubbed and needs no server at all.
