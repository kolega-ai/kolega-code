"""Shared HTTP timeout policy for LLM provider clients.

Provider SDK clients otherwise inherit the SDK default of a 600s (10 min) read
timeout. On a streaming turn that means a stalled connection — a reasoning model
that goes silent, or an idle connection dropped by a load balancer in front of the
endpoint — blocks for up to 10 minutes before erroring, leaving sockets in CLOSE_WAIT
and the turn apparently frozen.

The httpx ``read`` timeout bounds a single socket read, which is the wrong shape for
a UI: it cannot distinguish "the endpoint never answered" from "a long gap
mid-answer", and it says nothing to the user while it waits. The agent loop therefore
runs its own stall guard over each event wait, with two budgets:

- ``STREAM_FIRST_CHUNK_TIMEOUT`` — no *first* event at all. That is a connectivity
  problem (VPN, proxy, DNS, firewall), not a slow model, so it is far shorter.
- ``STREAM_IDLE_TIMEOUT`` — a gap between events once output has started.

The transport timeout is deliberately *longer* than the idle guard so the guard wins
the race and produces an actionable error instead of a bare httpx read timeout.

Use per-request on the streaming call (not the client default) so non-streaming
``generate``/``count_tokens`` keep the SDK's generous read budget.

Every budget can be overridden by environment variable, because the people who need
to lower them are exactly the people on a bad link:
``KOLEGA_STREAM_FIRST_CHUNK_TIMEOUT``, ``KOLEGA_STREAM_IDLE_TIMEOUT``,
``KOLEGA_STREAM_IDLE_HEARTBEAT``, ``KOLEGA_LLM_RETRY_BUDGET``.
"""

from __future__ import annotations

import os

import httpx

# Seconds.
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
    """Seconds without a first event before the request is declared unreachable."""
    return _env_seconds("KOLEGA_STREAM_FIRST_CHUNK_TIMEOUT", STREAM_FIRST_CHUNK_TIMEOUT)


def stream_idle_timeout() -> float:
    """Seconds without any event before an in-progress stream is declared dead."""
    return _env_seconds("KOLEGA_STREAM_IDLE_TIMEOUT", STREAM_IDLE_TIMEOUT)


def stream_idle_heartbeat_seconds() -> float:
    """Seconds between "no data for Ns" status updates while the stream is silent."""
    return _env_seconds("KOLEGA_STREAM_IDLE_HEARTBEAT", STREAM_IDLE_HEARTBEAT_SECONDS)


def llm_retry_budget_seconds() -> float:
    """Total wall clock one turn may spend in transient-error retries."""
    return _env_seconds("KOLEGA_LLM_RETRY_BUDGET", LLM_RETRY_BUDGET_SECONDS)


def streaming_timeout() -> httpx.Timeout:
    """httpx.Timeout for streaming requests (bounded per-read inter-chunk wait)."""
    return httpx.Timeout(
        connect=STREAM_CONNECT_TIMEOUT,
        read=max(STREAM_READ_TIMEOUT, stream_idle_timeout() + 30.0),
        write=STREAM_WRITE_TIMEOUT,
        pool=STREAM_POOL_TIMEOUT,
    )
