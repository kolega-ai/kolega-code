"""Shared HTTP timeout policy for LLM provider clients.

Provider SDK clients otherwise inherit the SDK default of a 600s (10 min) read
timeout. On a streaming turn that means a stalled connection — a reasoning model
that goes silent, or an idle connection dropped by a load balancer in front of the
endpoint — blocks for up to 10 minutes before erroring, leaving sockets in CLOSE_WAIT
and the turn apparently frozen.

For streaming we bound the *per-read* (inter-chunk) wait instead: if no bytes arrive
for ``STREAM_READ_TIMEOUT`` seconds the connection is treated as dead, the request
fails, and the agent loop retries it (a stream read timeout is classified as a
retryable transport error). 300s is deliberately generous so it never cuts a slow
time-to-first-token on a large context, while still being far below the 600s hang.

Use per-request on the streaming call (not the client default) so non-streaming
``generate``/``count_tokens`` keep the SDK's generous read budget.
"""

from __future__ import annotations

import httpx

# Seconds.
STREAM_CONNECT_TIMEOUT = 10.0
STREAM_READ_TIMEOUT = 300.0
STREAM_WRITE_TIMEOUT = 30.0
STREAM_POOL_TIMEOUT = 10.0


def streaming_timeout() -> httpx.Timeout:
    """httpx.Timeout for streaming requests (bounded per-read inter-chunk wait)."""
    return httpx.Timeout(
        connect=STREAM_CONNECT_TIMEOUT,
        read=STREAM_READ_TIMEOUT,
        write=STREAM_WRITE_TIMEOUT,
        pool=STREAM_POOL_TIMEOUT,
    )
