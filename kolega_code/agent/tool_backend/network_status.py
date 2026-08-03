"""Session-level tracking and messaging for outbound connect failures.

Shared by web_fetch retrieval and the web_search tool so that both surfaces emit the
same causal, model-facing message when a connection never opens (TCP/DNS/TLS), and so
that repeated failures across distinct hosts escalate to a clear "network unavailable"
signal instead of encouraging retries against mirrors or other search engines.
"""

from __future__ import annotations

from typing import Optional

_TLS_MARKERS = ("ssl", "tls", "certificate", "handshake")

# Wording seen from httpx/httpcore plus requests/urllib3 (SDK-based search backends).
# Deliberately excludes bare "connect" and "connection reset": a mid-stream reset means
# the server WAS reachable, and a false "unreachable" claim is worse than a generic one.
_CONNECT_MARKERS = (
    "connection refused",
    "connection attempts failed",
    "establish a new connection",
    "name resolution",
    "name or service not known",
    "nodename nor servname",
    "getaddrinfo",
    "network is unreachable",
    "no route to host",
    "connect timeout",
    "connection timed out",
)

_LOCAL_PREFERENCE_HINT = (
    "Prefer resources available locally or hosts the task explicitly names instead of "
    "retrying mirrors or other search engines."
)


class ConnectFailureTracker:
    """Counts distinct endpoints that failed at the connect level in this session."""

    def __init__(self) -> None:
        self._subjects: set[str] = set()

    def record(self, subject: str) -> int:
        self._subjects.add(subject.strip().lower())
        return len(self._subjects)

    @property
    def distinct_count(self) -> int:
        return len(self._subjects)


def looks_like_tls_failure(detail: str) -> bool:
    lowered = detail.lower()
    return any(marker in lowered for marker in _TLS_MARKERS)


def looks_like_connect_failure(detail: str) -> bool:
    lowered = detail.lower()
    return any(marker in lowered for marker in _CONNECT_MARKERS)


def connect_failure_message(subject: str, detail: Optional[str], distinct_failed: int) -> str:
    """Compose the model-facing message for a connection that never opened."""
    lead = f"Could not connect to {subject}"
    if detail:
        lead += f" ({detail})"
    if distinct_failed >= 2:
        middle = (
            f"{distinct_failed} distinct external hosts have failed to connect in this "
            "session — treat outbound network access as unavailable."
        )
    else:
        middle = (
            "This environment may restrict outbound network access to an allowlist, in "
            "which case other external hosts will fail the same way."
        )
    return f"{lead}. {middle} {_LOCAL_PREFERENCE_HINT}"


def secure_connection_failure_message(subject: str, detail: str) -> str:
    """Compose the model-facing message for a TLS handshake or certificate failure."""
    lead = f"Could not establish a secure connection to {subject}"
    if detail:
        lead += f" ({detail})"
    return (
        f"{lead}. The TLS handshake or certificate validation failed, so retrying will "
        "not help; this could be a problem with the host's certificate or with a "
        "TLS-intercepting proxy in this environment."
    )
