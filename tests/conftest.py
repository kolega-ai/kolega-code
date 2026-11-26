"""Shared pytest configuration for the repository."""

import functools
import gc
import os
import ssl
import tempfile
from pathlib import Path
from typing import Any

import certifi
import pytest
from dotenv import load_dotenv

from kolega_code.cli.config import API_KEY_ENV


# A TUI test mounts and drops a whole Textual app, so the default thresholds trip
# a generation-2 pass about once per test, and each pass scans the live heap
# (~300 ms on the settings tests). Collecting once per interval instead of once
# per test drops that to a handful of passes; the peak heap tracks the interval.
GC_THRESHOLDS = (20_000, 50, 2_000)
GC_COLLECT_TEST_INTERVAL = 50
_tests_since_collection = 0


# Every LLM client the suite constructs (anthropic, openai, google-genai, all
# over httpx) builds a TLS context from the certifi bundle, and
# ``ssl.create_default_context`` parses the ~140 certificates in it: 22.7 ms per
# call here. The clients are then handed fake transports, so ``tests/agent``
# alone paid for 1,327 such parses. One shared context for exactly that bundle
# is the trust store each client would have built: httpx and google-genai pass
# it to httpcore without mutating it. Every other caller — ``verify=False``, a
# test's own ``SSL_CERT_FILE`` bundle, anyio's and websockets' no-argument
# calls — still gets a fresh context, so a library that tunes the object it
# created (ddgs shuffles ciphers) cannot leak into another test.
_ssl_create_default_context = ssl.create_default_context
_CERTIFI_BUNDLE = certifi.where()


@functools.cache
def _default_certifi_context() -> ssl.SSLContext:
    return _ssl_create_default_context(cafile=_CERTIFI_BUNDLE)


def _shared_create_default_context(
    purpose: ssl.Purpose = ssl.Purpose.SERVER_AUTH,
    *,
    cafile: str | None = None,
    capath: str | None = None,
    cadata: Any = None,
) -> ssl.SSLContext:
    if purpose is ssl.Purpose.SERVER_AUTH and cafile == _CERTIFI_BUNDLE and capath is None and cadata is None:
        return _default_certifi_context()
    return _ssl_create_default_context(purpose, cafile=cafile, capath=capath, cadata=cadata)


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
    "KOLEGA_CODE_SHARE_ALLOWED_ORIGINS",
    "KOLEGA_CODE_CUSTOM_ENDPOINTS",
    "KOLEGA_CODE_ENDPOINT_URL",
    "KOLEGA_CODE_ENDPOINT_STYLE",
    "KOLEGA_CODE_ENDPOINT_API_KEY",
    "KOLEGA_CODE_ENDPOINT_CONTEXT",
    "KOLEGA_CODE_ENDPOINT_MAX_OUTPUT",
    "KOLEGA_CODE_ENDPOINT_VISION",
    "KOLEGA_CODE_ENDPOINT_THINKING",
    "KOLEGA_CODE_ENDPOINT_REASONING",
    "KOLEGA_CODE_ENDPOINT_TEMPERATURE",
    "CLOUDFLARE_ACCOUNT_ID",
    "CLOUDFLARE_AI_GATEWAY_ID",
}


def pytest_configure() -> None:
    """Load local test environment variables before test modules import."""
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    # Installed before collection so a module-level client is covered too.
    ssl.create_default_context = _shared_create_default_context


def pytest_unconfigure(config: pytest.Config) -> None:
    ssl.create_default_context = _ssl_create_default_context


def pytest_sessionstart(session: pytest.Session) -> None:
    """Collect once up front, then let the widened thresholds stand."""
    gc.collect()
    gc.set_threshold(*GC_THRESHOLDS)


def pytest_runtest_teardown(item: pytest.Item, nextitem: pytest.Item | None) -> None:
    """Bound the postponed garbage: one full collection per interval."""
    global _tests_since_collection
    _tests_since_collection += 1
    if _tests_since_collection >= GC_COLLECT_TEST_INTERVAL:
        _tests_since_collection = 0
        gc.collect()


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
