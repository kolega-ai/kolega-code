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

**`AgentEvent.timestamp` is now timezone-aware UTC.** It was naive local time,
which cannot be ordered across machines or used for replay.

**New event types.** Assistant prose and reasoning previously reached the terminal
UI only through `process_message_stream`'s generator, so the event stream carried
tool activity and status but no conversation. `assistant_delta`, `thinking_delta`,
`turn_started`, `turn_ended`, and `stream_truncated` close that gap. A host that
renders from events will now receive assistant text it did not receive before; a
host that also renders the generator output should ignore these to avoid
duplicating each response.

### Upgrading

#### 1. Take the additive changes (no work)

New fields are optional with defaults and JSON serialization is unchanged for
consumers that ignore unknown keys. Existing `AgentConnectionManager` subclasses
keep working untouched. If you match on `event_type` with an exhaustive
`if/elif`, add a fallback branch before upgrading.

#### 2. Adopt the stores (opt in)

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

#### 3. Replace bespoke reconnect buffers

If you keep a bounded in-memory cache of recent events so reconnecting clients can
catch up, `tail(from_seq=...)` supersedes it: durable, unbounded, and unaffected by
a process restart. Any hand-rolled reassembly of partial streaming messages is
likewise replaced by the recording wrapper's coalescing.

#### 4. Converge message persistence (optional, last)

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

### Sharing and export

`kolega_code.web.redaction` and `kolega_code.web.bundle` export a session as a
static replay. Redaction is **allowlist-based**: only `tool_result` and `image`
artifacts are ever shared, because the other four purposes are opaque provider
state — reasoning signatures and encrypted reasoning — with no display value.
Every exported string is scrubbed for secrets and local filesystem paths are
stripped. If you build your own export path, reuse `redact_event` rather than
writing events directly.

### On-disk sessions

The filesystem store is a CLI implementation detail, not the substrate. Nothing
outside `kolega_code.cli` assumes local disk, an in-process counter, or a single
writer. Sessions recorded before this change remain loadable and replay their
conversation history; they contain no presentation events, so clients should treat
them as not replayable rather than showing an empty player.
