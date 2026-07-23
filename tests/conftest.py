"""Shared pytest configuration for the repository."""

import tempfile
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
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)


@pytest.fixture
def isolated_cli_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Remove process CLI config so unit tests don't depend on a developer .env."""
    for key in {*API_KEY_ENV.values(), *CLI_CONFIG_ENV_KEYS}:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("KOLEGA_CODE_STATE_DIR", str(tmp_path / "state"))


@pytest.fixture(autouse=True)
def isolated_temp_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the process temp dir at the test's tmp_path for full isolation.

    tempfile caches its probe in ``tempfile.tempdir``; pre-seeding the cache
    redirects gettempdir/mkstemp/mkdtemp without env vars. Session scratchpads
    (kolega_code.scratchpad) resolve under it, so agent and TUI tests never
    touch the developer's real temp dir.
    """
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
