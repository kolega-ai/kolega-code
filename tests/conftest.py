"""Shared pytest configuration for the repository."""

import os
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
    "KOLEGA_CODE_LSP",
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


@pytest.fixture
def hermetic_git_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run git subprocesses without the developer's global/system config.

    CI runners have neither, so a test that silently depends on a personal
    ``~/.gitconfig`` (``init.defaultBranch``, ``merge.ff``, ``commit.gpgsign``,
    ``core.hooksPath``, ``pull.rebase``) passes locally and fails in CI. Opt in
    from any module that shells out to git.

    ``os.devnull`` is git's documented way to skip a configuration level, so no
    file is created and a test's ``tmp_path`` stays exactly as the test left it.
    """
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")


@pytest.fixture(autouse=True)
def isolated_temp_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the process temp dir at the test's tmp_path for full isolation.

    tempfile caches its probe in ``tempfile.tempdir``; pre-seeding the cache
    redirects gettempdir/mkstemp/mkdtemp without env vars. Session scratchpads
    (kolega_code.scratchpad) resolve under it, so agent and TUI tests never
    touch the developer's real temp dir.
    """
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
