# Scan hangs: missing timeouts in the LLM client and agent loop

> **Status:** known issue, fix pending. This note documents two root causes that live in
> **kolega-code** and contribute to kolega-comply security scans hanging indefinitely. The
> guaranteed-recovery backstop (a watchdog that *recovers* hung scans) was already added in
> kolega-comply; the fixes below *prevent* the hangs at the source and benefit every caller of
> these APIs, not just scans.

## Background

kolega-comply runs security scans as Celery tasks. Several were observed stuck "running" for days.
A read-only audit found that once a scan worker blocks on a network/LLM call with no timeout (or an
agent loops without converging), the worker slot is pinned until Celery's hard time limit SIGKILLs
the process — which leaves the scan's DB row stranded in `RUNNING`. Two of those blocking points are
in this repo.

## Problem 1 — Anthropic client has no request timeout

`kolega_code/llm/providers/anthropic.py:112-113`

```python
self.async_client = AsyncAnthropic(api_key=api_key, base_url=base_url, max_retries=max_retries)
self.sync_client  = Anthropic(api_key=api_key, base_url=base_url, max_retries=max_retries)
```

No `timeout=` is passed to either client, and the awaited calls have no per-call cap:

- `kolega_code/llm/providers/anthropic.py:382` — `await self.async_client.messages.create(...)`
- `kolega_code/llm/providers/anthropic.py:361` — `self.async_client.messages.stream(...)`

A wedged TCP read (degraded API, stuck socket, network black-hole) blocks the caller indefinitely.
Combined with `max_retries=3` honoring `retry-after` on 429/529, a single logical request can stay
pending far longer than expected. Inside a Celery scan worker (which runs with
`worker_prefetch_multiplier=1`, i.e. one task per slot) this pins the slot until the hard time limit
kills it, stranding the scan as `RUNNING`.

**Fix:** pass an explicit timeout to both clients. The Anthropic SDK accepts either an `httpx.Timeout`
or a scalar:

```python
import httpx

timeout = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)
self.async_client = AsyncAnthropic(api_key=api_key, base_url=base_url, max_retries=max_retries, timeout=timeout)
self.sync_client  = Anthropic(api_key=api_key, base_url=base_url, max_retries=max_retries, timeout=timeout)
```

Consider a per-request override (`.with_options(timeout=...)`) for intentionally long generations, and
make the default configurable via the existing provider config / env.

## Problem 2 — Agent reasoning loop is unbounded

`kolega_code/agent/baseagent.py:1285`

```python
while stop_reason not in ["end_turn", "max_tokens", "stop_sequence"]:
    ...
```

`process_message_stream` (the single canonical loop shared by every agent) iterates until the model
emits a terminal stop reason. A model that keeps emitting tool calls without converging never exits,
so the agent runs until an external time limit that may not cleanly fire.

**Fix:** add a max-iteration guard and raise a clear, catchable error when exceeded so callers fail
fast instead of hanging:

```python
max_iterations = 50  # ideally a BaseAgent config/constructor param
iterations = 0
while stop_reason not in ["end_turn", "max_tokens", "stop_sequence"]:
    iterations += 1
    if iterations > max_iterations:
        raise MaxAgentIterationsExceeded(f"Agent exceeded {max_iterations} turns without converging")
    ...
```

## Suggested follow-ups

- Make the LLM request timeout configurable via the existing provider config / env.
- Surface the iteration cap as a `BaseAgent` constructor/config parameter (default ~50).
- Tests:
  - A mock provider that never returns a terminal `stop_reason` must raise (not loop forever).
  - A slow/blocking mock client must raise a timeout rather than hang.

## Cross-reference

The companion backstop work that *recovers* hung scans even when these calls do block lives in
**kolega-comply**:

- `backend/kolega_dev/celery_app/scan_watchdog_task.py` — periodic watchdog that fails stale scans and
  reconciles batches.
- `backend/kolega_dev/celery_app/scan_task_utils.py` — `finalize_scan_*`.
- `backend/kolega_dev/database/mongodb.py` / `mongodb_sync.py` — `recompute_batch_progress`,
  `update_scan_heartbeat`.
