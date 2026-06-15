"""Shared pytest configuration for the repository."""

from pathlib import Path

import pytest
from dotenv import load_dotenv

from kolega_code.cli.config import API_KEY_ENV


CLI_CONFIG_ENV_KEYS = {
    "KOLEGA_CODE_PROVIDER",
    "KOLEGA_CODE_MODEL",
    "KOLEGA_CODE_FAST_PROVIDER",
    "KOLEGA_CODE_FAST_MODEL",
    "KOLEGA_CODE_EDIT_PROVIDER",
    "KOLEGA_CODE_EDIT_MODEL",
    "KOLEGA_CODE_THINKING_PROVIDER",
    "KOLEGA_CODE_THINKING_MODEL",
    "KOLEGA_CODE_THINKING_EFFORT",
    "KOLEGA_CODE_THINKING_TOKENS",
    "KOLEGA_CODE_ENVIRONMENT",
}


def pytest_configure() -> None:
    """Load local test environment variables before test modules import."""
    load_dotenv(Path(__file__).with_name(".env"), override=False)


@pytest.fixture
def isolated_cli_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove process CLI config so unit tests don't depend on a developer .env."""
    for key in {*API_KEY_ENV.values(), *CLI_CONFIG_ENV_KEYS}:
        monkeypatch.delenv(key, raising=False)
