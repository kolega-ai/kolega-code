# Migration guide

Guidance for host applications that embed `kolega_code` as a library. Each entry
describes what changed, why, and the smallest upgrade that keeps a host working.

## Unreleased — the session event spine

Sessions are now a durable, ordered **event stream**. Previously `AgentEvent` was
broadcast for live delivery and then forgotten, so any client that attached late,
reconnected, or wanted to replay had nothing to read. Durability and ordering are
now a protocol that hosts implement against their own storage.

### What changed

**`AgentEvent` gained fields** (all optional, all defaulted):

| Field | Purpose |
| --- | --- |
| `schema_version` | Envelope version, currently `2`. |
| `session_id`, `workspace_id`, `thread_id` | Addressing carried *on* the event, so a persisted or forwarded event stays self-describing. |
| `seq` | Assigned by the store on append. `None` means "not durable". |
| `elapsed_ms` | Monotonic milliseconds since the stream began — the replay timing key. |
| `artifacts` | References to payloads held outside the event body. |

**`AgentEvent.event_type` is now `str`**, not a closed `Literal`. The literal
forced a release of this package for every new event type and prevented hosts from
emitting their own. Every consumer must treat an unrecognized type as inert rather
than an error.

**`AgentEvent.timestamp` is now UTC.** It was naive local time, which cannot be
ordered across machines or used for replay. It is an ISO-8601 *string* carrying a
`+00:00` offset, not a `datetime`, so it sorts lexicographically and survives a
round trip through JSON unchanged.

**New event types.** Assistant prose and reasoning previously reached the terminal
UI only through `process_message_stream`'s generator, so the event stream carried
tool activity and status but no conversation. `assistant_delta`, `thinking_delta`,
`turn_started`, `turn_ended`, and `stream_truncated` close that gap. A host that
renders from events will now receive assistant text it did not receive before; a
host that also renders the generator output should ignore these to avoid
duplicating each response.

**Sub-agent prose and reasoning moved to those types too — this one is breaking.**
`AgentTool` used to re-broadcast every chunk of a dispatched agent's generator as a
`chat_message` carrying `sub_agent_info` (and `message_type: "thinking"` for
reasoning). The dispatched agent is itself a `BaseAgent`, so it now mirrors the same
chunks as `assistant_delta`/`thinking_delta` with the same `sub_agent_info`, and
keeping both put every delegated sentence on the stream twice. The re-broadcast is
gone; chunk types the agent does not mirror are still broadcast as before.

The delta form is strictly better to render from: deltas of one segment share a
`uuid` so they accumulate into a single entry, whereas the `chat_message` copies had
to be re-assembled by the consumer and reasoning was distinguishable only by a
magic `message_type` string.

**Turn boundaries now carry `sub_agent_info`.** A dispatched agent runs turns of its
own. Without attribution they were indistinguishable from session turns, so the
sub-agent's task folded into the main transcript as a message the user never sent,
and its `turn_ended` reported the session idle while the main agent was still
working.

**A `chat_message` can carry image artifacts.** A tool that produces a picture —
reading an image, a browser screenshot — now passes the bytes to
`AgentEventEmitter.chat(images=[(media_type, base64_data)])`. The recording
wrapper turns them into `image` artifacts on the event and removes the payload
from the body, so nothing base64 is ever persisted inline. Previously the bytes
reached only the provider-facing history record, which no presentation client
reads, and a replay could not show a screenshot at all. A host that renders
`event.artifacts` gains images with no change; one that ignores artifacts is
unaffected. The text payload of the event is identical either way.

**`elapsed_ms` is now non-decreasing along the log.** Streaming segments are
coalesced into one record per run, and runs do not finish in the order they
began, so a record could carry a timestamp earlier than one already written.
Consumers treat `elapsed_ms` as a timeline and binary-search it, so the recorder
now writes runs in the order they started and clamps the value. Sessions recorded
before this change can still step backwards; clamp on read if that matters to you.

### Upgrading

#### 1. Take the additive changes (no work)

New fields are optional with defaults and JSON serialization is unchanged for
consumers that ignore unknown keys. Existing `AgentConnectionManager` subclasses
keep working untouched. If you match on `event_type` with an exhaustive
`if/elif`, add a fallback branch before upgrading.

#### 2. Re-point sub-agent rendering at the delta events (required if you render dispatches)

If you render delegated work, you were reading `chat_message` events that carried
`sub_agent_info`. Prose and reasoning no longer arrive that way. Tool calls, tool
results, edit previews, context updates, and the `GENERATING`/`STOPPED`/`ERROR`
lifecycle statuses are unchanged — only the streamed chunks moved.

```python
# before: one chat_message per chunk, reasoning flagged by a string
if event.event_type == "chat_message" and event.sub_agent_info:
    if event.content.get("message_type") == "thinking":
        ...

# after: typed deltas, grouped by uuid, completion on the envelope
if event.sub_agent_info and event.event_type in ("assistant_delta", "thinking_delta"):
    reasoning = event.event_type == "thinking_delta"
    append_to_segment(event.uuid, event.content["text"], complete=not event.is_streaming)
```

`kolega_code.session.projection.fold` already does this; hosts that render from it
need no change.

#### 3. Adopt the stores (opt in)

Implement the two protocols in `kolega_code.session.store` against your own
storage:

```python
class SessionEventStore(abc.ABC):
    async def append(self, event) -> int          # assign and return seq, atomically
    async def read(self, session_id, *, from_seq=1, to_seq=None, types=None)
    def tail(self, session_id, *, from_seq=1)     # backlog, then live
    async def head(self, session_id)              # cheap summary

class ArtifactStore(abc.ABC):
    async def put(self, data, *, media_type, purpose, encoding, chars=None)
    async def open(self, ref) -> bytes            # integrity-checked
```

Two contract points matter most:

- **`seq` is assigned by the store, never the emitter.** A session may be driven
  by several processes or workers, so use an atomic operation — an append under a
  lock, a database increment. Numbers must be strictly increasing and unique;
  they need not be contiguous, so consumers must treat `seq` as an opaque
  ordering key.
- **`tail` must deliver every seq exactly once, ascending.** Events appended while
  the backlog is being read must not fall into the gap between "end of backlog"
  and "start of live". This is the guarantee that lets a client reconnect at its
  last seen `seq` and lose nothing.

Then wrap your existing connection manager so everything it broadcasts is also
recorded:

```python
from kolega_code.session.recording import RecordingConnectionManager

manager = RecordingConnectionManager(
    your_connection_manager,
    your_event_store,
    session_id=session_id,
    artifact_store=your_artifact_store,
)
```

The wrapper persists before fan-out, so a client that sees `seq` N can always find
N in the store afterwards. It also coalesces high-volume streams: a verbose
command emitting thousands of output events is recorded as a much smaller number
of larger records that replay to the same text. Call `await manager.flush()` at
shutdown so a stream that is still buffered is not lost.

Validate your implementation with the shipped conformance suite, which is the same
one the first-party stores are held to:

```python
from kolega_code.testing.store_conformance import CONFORMANCE_CHECKS

async def factory():
    return YourStore(...), "some-session-id"

for check in CONFORMANCE_CHECKS:
    await check(factory)
```

#### 4. Replace bespoke reconnect buffers

If you keep a bounded in-memory cache of recent events so reconnecting clients can
catch up, `tail(from_seq=...)` supersedes it: durable, unbounded, and unaffected by
a process restart. Any hand-rolled reassembly of partial streaming messages is
likewise replaced by the recording wrapper's coalescing.

#### 5. Converge message persistence (optional, last)

If you already store per-thread ordered messages, that record can become a
*projection* of the event log rather than a parallel source of truth. Dual-write
during the transition, verify the derived view matches, then stop writing it
directly.

### Rendering without reimplementing the UI

`kolega_code.session.projection.fold` turns an event stream into
`PresentationState` — conversation, tool activity, sub-agent trajectories,
terminal buffer, context and compaction status, turn boundaries. It has no UI
dependencies, is deterministic, and treats unknown event types as inert.
`PresentationState.to_dict()` is JSON-safe for transport to a browser.

`fold` mutates the state it is given and returns it, so folding N events is linear
rather than quadratic. Do not share one state object between two independent
folds; use `replay(events)` to build state from scratch.

A JavaScript port ships at `kolega_code/web/assets/fold.js` and is held to
byte-equality with the Python implementation by a shared-fixture test.

### Hosting a session from inside a host application

`kolega_code.web.hosting.ShareServer` runs the read-only session server as a task
on the caller's event loop rather than owning the process, which is what lets a
running UI hand out a link without a second terminal. It binds a port the OS
picks, always mints an access token, and leaves the host's signal handlers alone.

```python
server = ShareServer(store, session_id=session_id)   # bind=ALL_INTERFACES to reach a LAN
await server.start()
link = server.session_url(session_id)                # carries ?token=...
await server.stop()                                  # request_stop() from sync teardown
```

`ALL_INTERFACES` is a request for "reachable on this machine's local network",
not the address bound. It resolves to the single local-network address the link
advertises, so a session is not also served on every other interface the host
happens to have — a VPN, a tether, a cloud NIC. With no such address, `start()`
raises rather than falling back to the wildcard.

**Pass `session_id` for anything you hand to another person.** The token gates
routes, not sessions. Without a scope, the link you gave one person reads every
session in the store, including ones recorded for unrelated projects. With it,
every other session answers 404 — not 403, which would confirm they exist.

The same scope is available on the app directly, as
`ServerConfig(store=..., session_ids=frozenset({...}))`. It defaults to `None`,
meaning unrestricted, so a host serving its own sessions needs no change:

```python
create_app(ServerConfig(store=store))                       # every session, as before
create_app(ServerConfig(store=store, session_ids=allowed))  # only these
```

**A served session is not redacted.** `export_bundle` scrubs every string it
writes; this serves recorded events as they are. Anything the agent printed —
credentials, file contents, command output — is visible to whoever holds the
link. Export a bundle instead when the recipient should not see raw output.

A link is only useful if it opens in a browser, so a correct `?token=` is now
handed to that tab as a cookie and accepted on later same-origin requests,
including the WebSocket handshake. Without it every relative subresource the
player loads was unauthorized and a token-protected server rendered a blank page.
An explicitly presented credential still has to be correct: a wrong header or
query token is rejected rather than falling back to a cookie.

### Sharing and export

`kolega_code.web.redaction` and `kolega_code.web.bundle` export a session as a
static replay. Redaction is **allowlist-based**: only `tool_result` and `image`
artifacts are ever shared, because the other four purposes are opaque provider
state — reasoning signatures and encrypted reasoning — with no display value.
Every exported string is scrubbed for secrets, the home directory is rewritten to
`~`, and local paths are dropped from artifact references. If you build your own
export path, reuse `redact_event` rather than writing events directly.

Secret detection is pattern matching, and plenty of real credentials match no
pattern at all. **Pass every secret you know about** as `extra_secrets` rather
than relying on the patterns:

```python
await export_bundle(events, destination, session_id=..., extra_secrets=my_api_keys)
```

`export_bundle(..., single_file=True)` writes the entire replay as one HTML
document — assets, manifest, gzipped events and any images embedded — instead of
a directory. This is now what `kolega-code share export` produces by default.

The reason is not tidiness. A directory bundle cannot be opened from a `file://`
URL at all: browsers block ES module imports and `fetch` on that origin, so a
recipient who double-clicks `index.html` gets a blank page, and not even an error,
because the script that would report the failure is the one that was blocked. A
directory therefore requires the recipient to run a web server, which is not a
sharing story. Pass `single_file=False` when you are publishing to a static host
and separate cacheable files are the better shape.

The player is not forked for this. It reads `globalThis.__KC_REPLAY__` when
present and fetches when it is absent, so the served player, the directory bundle
and the single file are all the same code. If you embed the player yourself, see
`kolega_code.web.singlefile` for the payload shape.

### Removed: the `kolega-code serve` command

`serve` is gone. It never appeared in a release, so nothing can depend on it.

Everything it did that people actually wanted is covered better elsewhere: `/share`
in the TUI starts the same server for a running session and hands you the link,
and `share export` produces a file you can send. Its remaining job — a documented
public HTTP API with no consumer — would have meant owning that surface's
compatibility forever.

`kolega_code.web.server.create_app` and `kolega_code.web.hosting.ShareServer` are
unchanged and remain the supported way to host sessions from your own process.

### On-disk sessions

The filesystem store is a CLI implementation detail, not the substrate. Nothing
outside `kolega_code.cli` assumes local disk, an in-process counter, or a single
writer. Sessions recorded before this change remain loadable and replay their
conversation history; they contain no presentation events, so clients should treat
them as not replayable rather than showing an empty player.
