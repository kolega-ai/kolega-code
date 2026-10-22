"""Shared HTTP timeout policy for LLM provider clients.

Provider SDKs default to a 600s read timeout, so a stalled stream blocks for ten
minutes before erroring. The agent loop runs its own stall guard over each event
wait instead: a short budget for "no first event" (connectivity, not a slow
model) and a longer one for gaps after output starts. The transport read timeout
sits above the idle guard so the guard wins the race and surfaces an actionable
error. Apply it per streaming request, not as the client default, so
non-streaming calls keep the SDK's generous budget.

Every budget is overridable by environment variable, because the people who need
to lower them are on a bad link: ``KOLEGA_STREAM_FIRST_CHUNK_TIMEOUT``,
``KOLEGA_STREAM_IDLE_TIMEOUT``, ``KOLEGA_STREAM_IDLE_HEARTBEAT``,
``KOLEGA_LLM_RETRY_BUDGET``.
"""

from __future__ import annotations

import os

import httpx

STREAM_CONNECT_TIMEOUT = 10.0
STREAM_WRITE_TIMEOUT = 30.0
STREAM_POOL_TIMEOUT = 10.0

#: No first event at all — treat the endpoint as unreachable.
STREAM_FIRST_CHUNK_TIMEOUT = 90.0
#: Gap between events once the stream has produced output.
STREAM_IDLE_TIMEOUT = 300.0
#: How often to report "still waiting, no data for Ns" while the stream is silent.
STREAM_IDLE_HEARTBEAT_SECONDS = 15.0
#: Total wall clock one turn may spend retrying transient errors. Attempt counts
#: alone are not a budget: 4 attempts x a 5 minute stall is 20 minutes of silence.
LLM_RETRY_BUDGET_SECONDS = 360.0

#: Transport read timeout sits above the idle guard so the guard wins the race and
#: the surfaced error is the actionable one.
STREAM_READ_TIMEOUT = STREAM_IDLE_TIMEOUT + 30.0


def _env_seconds(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def stream_first_chunk_timeout() -> float:
    return _env_seconds("KOLEGA_STREAM_FIRST_CHUNK_TIMEOUT", STREAM_FIRST_CHUNK_TIMEOUT)


def stream_idle_timeout() -> float:
    return _env_seconds("KOLEGA_STREAM_IDLE_TIMEOUT", STREAM_IDLE_TIMEOUT)


def stream_idle_heartbeat_seconds() -> float:
    return _env_seconds("KOLEGA_STREAM_IDLE_HEARTBEAT", STREAM_IDLE_HEARTBEAT_SECONDS)


def llm_retry_budget_seconds() -> float:
    return _env_seconds("KOLEGA_LLM_RETRY_BUDGET", LLM_RETRY_BUDGET_SECONDS)


def streaming_timeout() -> httpx.Timeout:
    """httpx.Timeout for streaming requests (bounded per-read inter-chunk wait)."""
    return httpx.Timeout(
        connect=STREAM_CONNECT_TIMEOUT,
        read=max(STREAM_READ_TIMEOUT, stream_idle_timeout() + 30.0),
        write=STREAM_WRITE_TIMEOUT,
        pool=STREAM_POOL_TIMEOUT,
    )
