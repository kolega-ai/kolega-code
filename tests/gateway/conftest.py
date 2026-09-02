"""Hermetic environment for gateway tests.

Gateway configuration lives in settings.json now, so the only environment
influence left is the shared KOLEGA_CODE_STATE_DIR; keep it cleared so state
resolution never leaks a developer's real state dir into tests.
"""

import pytest


@pytest.fixture(autouse=True)
def isolated_gateway_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KOLEGA_CODE_STATE_DIR", raising=False)
