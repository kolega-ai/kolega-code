"""ACP tests drive the server directly, so production pacing windows only add wall time.

The session usage presenter batches ledger events over 100 ms so a busy UI does not
re-render on each one. No ACP test asserts that window: the same accounting and
notifications arrive, and a test that emits its events in one turn still gets one
batched usage update.
"""

from __future__ import annotations

import pytest

from kolega_code.session import usage


@pytest.fixture(autouse=True)
def _no_usage_pacing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(usage, "_COALESCE_SECONDS", 0.0)
