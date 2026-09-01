"""Hermetic environment for gateway tests.

The global fixtures clear ``KOLEGA_CODE_*`` config; gateway tests additionally
clear ``KOLEGA_GATEWAY_*`` so a developer's ambient gateway env never leaks in.
"""

import pytest

GATEWAY_ENV_KEYS = [
    "KOLEGA_GATEWAY_ADAPTER",
    "KOLEGA_GATEWAY_PROJECT",
    "KOLEGA_GATEWAY_STATE_DIR",
    "KOLEGA_GATEWAY_ALLOWED_USERS",
    "KOLEGA_GATEWAY_PERMISSION_MODE",
    "KOLEGA_GATEWAY_REQUEST_TIMEOUT_SECONDS",
    "KOLEGA_GATEWAY_MAX_SESSIONS",
    "KOLEGA_GATEWAY_SESSION_IDLE_TTL_SECONDS",
    "KOLEGA_GATEWAY_EDIT_THROTTLE_SECONDS",
    "KOLEGA_GATEWAY_STT",
    "KOLEGA_GATEWAY_TELEGRAM_TOKEN",
    "KOLEGA_GATEWAY_TELEGRAM_PROXY",
]


@pytest.fixture(autouse=True)
def isolated_gateway_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in GATEWAY_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("KOLEGA_CODE_STATE_DIR", raising=False)
