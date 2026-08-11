"""Machine-readable output for ``kolega-code ask --json``.

``--json`` streams the public semantic session-event protocol: every stdout
line is one complete v2 event envelope, printed by ``SemanticStdoutPrinter``
registered as a journal listener. The printer receives the exact event objects
the journal persists, so a live line and the saved record share id, seq, and
timestamp — the later ``sessions export --format events-jsonl`` output is the
same records. Unsaved runs record through ``InMemorySessionJournal`` and emit
the identical protocol without touching disk.

The legacy ``{"kind": ...}`` record family (message/event/summary) was removed
with the protocol swap; there is no compatibility mode. See
docs/src/content/docs/cli/ask.md for the event reference.
"""

from __future__ import annotations

from .session_event_protocol import (
    InMemorySessionJournal,
    SemanticStdoutPrinter,
    to_public_event,
)

__all__ = ["InMemorySessionJournal", "SemanticStdoutPrinter", "to_public_event"]
