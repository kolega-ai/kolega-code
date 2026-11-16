"""Unit tests for LspManager internal helpers (no real server needed).

Tests position resolution, status, capability checking, and dedupe/sort.
End-to-end tests with a fake server are in ``test_integration.py``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from kolega_code.services.lsp.client import LspClient, LspClientError, LspDiagnostic
import kolega_code.services.lsp.manager as manager_module
from kolega_code.services.lsp.config import LspConfig
from kolega_code.services.lsp.diagnostics import dedupe_and_sort
from kolega_code.services.lsp.detector import DetectionReport, DetectionResult, ResolvedLanguage
from kolega_code.services.lsp.manager import (
    LspManager,
    _LspSession,
    _LSP_SHUTDOWN_REQUEST_TIMEOUT_SECONDS,
    _path_to_uri,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def manager(tmp_path: Path) -> LspManager:
    """A minimal LspManager with no real server."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname='t'\n")
    m = LspManager(tmp_path, config=LspConfig(enabled=True, prompt_on_missing=False))
    # Populate a fake report so status() has data
    m.report = DetectionReport(
        detected=[DetectionResult("python", "Python", "pyproject.toml")],
        resolved=[
            ResolvedLanguage(
                language_id="python",
                display_name="Python",
                detection_reason="pyproject.toml",
                server_name="pyright",
                server_bin="/usr/bin/pyright-langserver",
                server_args=["--stdio"],
                install_commands=[],
                alternatives=[],
            )
        ],
        missing=[],
    )
    m._resolved = {r.language_id: r for r in m.report.resolved}
    return m


# ---------------------------------------------------------------------------
# shutdown()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shutdown_uses_short_graceful_timeout_and_stops_client(manager: LspManager):
    """Shutdown uses its own short deadline, then sends exit and stops the client."""
    client = MagicMock()
    client.running = True
    client.request = AsyncMock()
    client.notify = AsyncMock()
    client.stop = AsyncMock()
    manager._sessions["python"] = client
    manager._session_records["python"] = MagicMock()
    manager._diagnostics["file:///test.py"] = []
    manager._open_files["file:///test.py"] = "test.py"

    await manager.shutdown()

    client.request.assert_awaited_once_with("shutdown", timeout=_LSP_SHUTDOWN_REQUEST_TIMEOUT_SECONDS)
    client.notify.assert_awaited_once_with("exit")
    client.stop.assert_awaited_once_with()
    assert manager._sessions == {}
    assert manager._session_records == {}
    assert manager._diagnostics == {}
    assert manager._open_files == {}


@pytest.mark.asyncio
async def test_shutdown_stops_client_when_graceful_request_fails(manager: LspManager):
    """A failed graceful request skips exit but does not skip forced termination."""
    client = MagicMock()
    client.running = True
    client.request = AsyncMock(side_effect=LspClientError("shutdown timed out"))
    client.notify = AsyncMock()
    client.stop = AsyncMock()
    manager._sessions["python"] = client
    manager._session_records["python"] = MagicMock()

    await manager.shutdown()

    client.request.assert_awaited_once_with("shutdown", timeout=_LSP_SHUTDOWN_REQUEST_TIMEOUT_SECONDS)
    client.notify.assert_not_awaited()
    client.stop.assert_awaited_once_with()
    assert manager._sessions == {}
    assert manager._session_records == {}


# ---------------------------------------------------------------------------
# _resolve_position
# ---------------------------------------------------------------------------


def test_resolve_position_basic(manager: LspManager, tmp_path: Path):
    """Position resolution finds the symbol on the given line."""
    f = tmp_path / "test.py"
    f.write_text("def hello():\n    return world\n", encoding="utf-8")

    line, char = manager._resolve_position("test.py", 1, "hello")
    assert line == 0
    assert char == 4  # "def " is 4 chars, "hello" starts at index 4


def test_resolve_position_second_line(manager: LspManager, tmp_path: Path):
    """Position resolution works on lines beyond the first."""
    f = tmp_path / "test.py"
    f.write_text("def hello():\n    return world\n", encoding="utf-8")

    line, char = manager._resolve_position("test.py", 2, "world")
    assert line == 1
    assert char == 11  # "    return " is 11 chars


def test_resolve_position_name_hash_n(manager: LspManager, tmp_path: Path):
    """name#N syntax targets the Nth occurrence on the line."""
    f = tmp_path / "test.py"
    f.write_text("x = foo + foo + foo\n", encoding="utf-8")

    # First occurrence
    line, char1 = manager._resolve_position("test.py", 1, "foo")
    assert char1 == 4  # "x = " is 4 chars

    # Second occurrence
    line, char2 = manager._resolve_position("test.py", 1, "foo#2")
    assert char2 == 10  # "x = foo + " is 10 chars

    # Third occurrence
    line, char3 = manager._resolve_position("test.py", 1, "foo#3")
    assert char3 == 16  # "x = foo + foo + " is 16 chars


def test_resolve_position_not_found_raises(manager: LspManager, tmp_path: Path):
    """ValueError when symbol is not on the line."""
    f = tmp_path / "test.py"
    f.write_text("def hello():\n    pass\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not found on line 1"):
        manager._resolve_position("test.py", 1, "world")


def test_resolve_position_occurrence_not_found_raises(manager: LspManager, tmp_path: Path):
    """ValueError when name#N occurrence doesn't exist."""
    f = tmp_path / "test.py"
    f.write_text("x = foo\n", encoding="utf-8")

    with pytest.raises(ValueError, match="occurrence #2"):
        manager._resolve_position("test.py", 1, "foo#2")


def test_resolve_position_line_out_of_range(manager: LspManager, tmp_path: Path):
    """ValueError when line number is out of range."""
    f = tmp_path / "test.py"
    f.write_text("x = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="out of range"):
        manager._resolve_position("test.py", 99, "x")


# ---------------------------------------------------------------------------
# status()
# ---------------------------------------------------------------------------


def test_status_returns_expected_fields(manager: LspManager):
    """status() returns a dict with the expected structure."""
    status = manager.status()
    assert status["enabled"] is True
    assert status["initialized"] is False  # not initialized yet
    assert len(status["detected"]) == 1
    assert status["detected"][0]["display_name"] == "Python"
    assert status["sessions"] == []
    assert status["diagnostic_counts"] == {}


def test_status_exposes_incomplete_detection(manager: LspManager):
    assert manager.report is not None
    manager.report.scan_complete = False
    manager.report.scan_stop_reason = "deadline"
    manager.report.scanned_entries = 50_000
    manager.report.scan_elapsed_seconds = 5.0

    status = manager.status()

    assert status["scan_complete"] is False
    assert status["scan_stop_reason"] == "deadline"
    assert status["scanned_entries"] == 50_000


def test_status_includes_session_info(manager: LspManager):
    """status() includes active session details."""
    fake_client = MagicMock()
    fake_client.status = "initialized"
    fake_client.server_pid = 12345
    fake_client.running = True
    fake_client.last_error = None
    fake_client.active_root = "file:///project"
    manager._sessions["python"] = fake_client

    status = manager.status()
    assert len(status["sessions"]) == 1
    s = status["sessions"][0]
    assert s["server_name"] == "pyright"
    assert s["pid"] == 12345
    assert s["connected"] is True


def test_status_disabled_manager(tmp_path: Path):
    """status() for a disabled manager reports enabled=False."""
    m = LspManager(tmp_path, config=LspConfig(enabled=False))
    status = m.status()
    assert status["enabled"] is False


# ---------------------------------------------------------------------------
# _has_capability
# ---------------------------------------------------------------------------


def test_has_capability_true():
    """_has_capability returns True when the capability exists."""
    client = MagicMock()
    client.server_capabilities = {"definitionProvider": True}
    assert LspManager._has_capability(client, "definitionProvider") is True


def test_has_capability_nested():
    """_has_capability traverses nested paths."""
    client = MagicMock()
    client.server_capabilities = {"textDocument": {"sync": {"didSave": True}}}
    assert LspManager._has_capability(client, "textDocument", "sync", "didSave") is True


def test_has_capability_false_when_missing():
    """_has_capability returns False when the capability is absent."""
    client = MagicMock()
    client.server_capabilities = {"hoverProvider": True}
    assert LspManager._has_capability(client, "definitionProvider") is False


def test_has_capability_false_when_client_none():
    """_has_capability returns False when client is None."""
    assert LspManager._has_capability(None, "definitionProvider") is False


def test_has_capability_false_when_caps_none():
    """_has_capability returns False when capabilities are None."""
    client = MagicMock()
    client.server_capabilities = None
    assert LspManager._has_capability(client, "definitionProvider") is False


# ---------------------------------------------------------------------------
# dedupe_and_sort
# ---------------------------------------------------------------------------


def _make_diag(line: int, message: str, severity: int = 1, source: str = "test"):
    return LspDiagnostic(
        range={"start": {"line": line, "character": 0}, "end": {"line": line, "character": 80}},
        severity=severity,
        message=message,
        source=source,
    )


def test_dedupe_and_sort_removes_duplicates():
    """Duplicate diagnostics (same range + message + source) are removed."""
    diags = [
        _make_diag(0, "error", severity=1),
        _make_diag(0, "error", severity=1),  # exact duplicate
    ]
    result = dedupe_and_sort(diags)
    assert len(result) == 1


def test_dedupe_and_sort_sorts_by_severity():
    """Errors come before warnings, warnings before info."""
    diags = [
        _make_diag(5, "warning", severity=2),
        _make_diag(0, "info", severity=3),
        _make_diag(10, "error", severity=1),
    ]
    result = dedupe_and_sort(diags)
    assert result[0].severity == 1  # error
    assert result[1].severity == 2  # warning
    assert result[2].severity == 3  # info


def test_dedupe_and_sort_sorts_by_location_within_severity():
    """Within the same severity, diagnostics are sorted by line then character."""
    diags = [
        _make_diag(10, "a", severity=1),
        _make_diag(2, "b", severity=1),
        _make_diag(5, "c", severity=1),
    ]
    result = dedupe_and_sort(diags)
    assert result[0].message == "b"  # line 2
    assert result[1].message == "c"  # line 5
    assert result[2].message == "a"  # line 10


def test_dedupe_and_sort_caps_at_max():
    """Diagnostics are capped at max_count."""
    diags = [_make_diag(i, f"err{i}") for i in range(50)]
    result = dedupe_and_sort(diags, max_count=10)
    assert len(result) == 10


def test_dedupe_and_sort_different_sources_not_deduped():
    """Diagnostics with different sources are kept even if message/range match."""
    diags = [
        _make_diag(0, "error", severity=1, source="pyright"),
        _make_diag(0, "error", severity=1, source="ruff"),
    ]
    result = dedupe_and_sort(diags)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# document versioning
# ---------------------------------------------------------------------------


def test_next_version_increments(manager: LspManager):
    """_next_version increments per URI."""
    assert manager._next_version("file:///a") == 1
    assert manager._next_version("file:///a") == 2
    assert manager._next_version("file:///a") == 3
    # Different URI starts at 1
    assert manager._next_version("file:///b") == 1


@pytest.mark.asyncio
async def test_document_open_state_is_per_session(manager: LspManager, tmp_path: Path):
    """Each server session receives its own first didOpen for a URI."""
    (tmp_path / "test.py").write_text("x = 1\n", encoding="utf-8")
    uri = _path_to_uri(tmp_path, "test.py")

    client_a = MagicMock()
    client_a.notify = AsyncMock()
    client_b = MagicMock()
    client_b.notify = AsyncMock()
    manager._session_records["python"] = _LspSession("python", "python", "pyright", client_a, tmp_path.as_uri())
    manager._session_records["python:ruff"] = _LspSession("python:ruff", "python", "ruff", client_b, tmp_path.as_uri())

    await manager._ensure_document_open(client_a, uri, "test.py", "python", session_key="python")
    await manager._ensure_document_open(client_a, uri, "test.py", "python", session_key="python")
    await manager._ensure_document_open(client_b, uri, "test.py", "python", session_key="python:ruff")

    assert client_a.notify.await_args_list[0].args[0] == "textDocument/didOpen"
    assert client_a.notify.await_args_list[1].args[0] == "textDocument/didChange"
    assert client_b.notify.await_args_list[0].args[0] == "textDocument/didOpen"


def test_workspace_configuration_returns_configured_sections(manager: LspManager):
    """workspace/configuration replies with per-server section config."""
    manager._config.workspace_configuration = {
        "pyright": {
            "python": {"analysis": {"typeCheckingMode": "strict"}},
        }
    }
    handlers = {}
    client = MagicMock()
    client.on_request.side_effect = lambda method, handler: handlers.setdefault(method, handler)

    manager._register_server_request_handlers(client, server_name="pyright")
    result = handlers["workspace/configuration"]({"items": [{"section": "python"}, {"section": "missing"}, {}]})

    assert result == [
        {"analysis": {"typeCheckingMode": "strict"}},
        {},
        manager._config.workspace_configuration["pyright"],
    ]


def test_path_to_uri_escapes_spaces(tmp_path: Path):
    """file:// URIs are encoded correctly for paths with spaces."""
    uri = _path_to_uri(tmp_path, "space file.py")
    assert uri.startswith("file://")
    assert "space%20file.py" in uri


# ---------------------------------------------------------------------------
# F10: _has_capability treats empty options object {} as supported
# ---------------------------------------------------------------------------


def _client_with_caps(caps):
    """Build a fake client object exposing ``server_capabilities``."""
    client = MagicMock()
    client.server_capabilities = caps
    return client


def test_has_capability_true_is_supported():
    """A ``true`` capability value is supported."""
    client = _client_with_caps({"definitionProvider": True})
    assert LspManager._has_capability(client, "definitionProvider") is True


def test_has_capability_empty_object_is_supported():
    """F10: an empty options object ``{}`` is a valid 'enabled' advertisement."""
    client = _client_with_caps({"definitionProvider": {}})
    assert LspManager._has_capability(client, "definitionProvider") is True


def test_has_capability_populated_object_is_supported():
    """A populated options object is supported."""
    client = _client_with_caps({"definitionProvider": {"workDoneProgress": False}})
    assert LspManager._has_capability(client, "definitionProvider") is True


def test_has_capability_false_is_unsupported():
    """An explicit ``false`` is unsupported."""
    client = _client_with_caps({"definitionProvider": False})
    assert LspManager._has_capability(client, "definitionProvider") is False


def test_has_capability_absent_is_unsupported():
    """A missing capability is unsupported."""
    client = _client_with_caps({"hoverProvider": True})
    assert LspManager._has_capability(client, "definitionProvider") is False


def test_has_capability_none_client_is_unsupported():
    assert LspManager._has_capability(None, "definitionProvider") is False


# ---------------------------------------------------------------------------
# diagnostics routing and push waiters
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pull_diagnostics_skips_server_without_capability(manager: LspManager):
    client = MagicMock()
    client.server_capabilities = {"hoverProvider": True}
    client.request = AsyncMock()

    result = await manager._pull_diagnostics(client, "file:///test.py")

    assert result is None
    client.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_push_wait_returns_when_publish_already_arrived(manager: LspManager):
    uri = "file:///already.py"
    generation = manager._diagnostic_generation("python", uri)
    manager._on_publish_diagnostics(
        "python",
        {"uri": uri, "diagnostics": [], "version": None},
    )

    await asyncio.wait_for(
        manager._wait_for_push_diagnostics(
            "python",
            uri,
            after_generation=generation,
            timeout=3.0,
        ),
        timeout=1.0,
    )

    assert manager._diag_waiters == {}


@pytest.mark.asyncio
async def test_push_wait_is_released_by_later_publish(manager: LspManager):
    uri = "file:///later.py"
    generation = manager._diagnostic_generation("python", uri)
    waiting = asyncio.create_task(
        manager._wait_for_push_diagnostics(
            "python",
            uri,
            after_generation=generation,
            timeout=3.0,
        )
    )
    await asyncio.sleep(0)

    assert len(manager._diag_waiters[("python", uri)]) == 1
    manager._on_publish_diagnostics(
        "python",
        {"uri": uri, "diagnostics": [], "version": 1},
    )
    await asyncio.wait_for(waiting, timeout=1.0)

    assert manager._diag_waiters == {}


@pytest.mark.asyncio
async def test_push_publish_releases_all_concurrent_waiters(manager: LspManager):
    uri = "file:///concurrent.py"
    generation = manager._diagnostic_generation("python", uri)
    waiting = [
        asyncio.create_task(
            manager._wait_for_push_diagnostics(
                "python",
                uri,
                after_generation=generation,
                timeout=3.0,
            )
        )
        for _ in range(2)
    ]
    await asyncio.sleep(0)

    assert len(manager._diag_waiters[("python", uri)]) == 2
    manager._on_publish_diagnostics(
        "python",
        {"uri": uri, "diagnostics": [], "version": 1},
    )
    await asyncio.wait_for(asyncio.gather(*waiting), timeout=1.0)

    assert manager._diag_waiters == {}


@pytest.mark.asyncio
async def test_fresh_diagnostics_batch_is_ordered_unique_bounded_and_isolates_failures(
    manager: LspManager,
):
    manager._initialized = True
    active = 0
    max_active = 0
    calls: list[str] = []
    admitted = asyncio.Event()
    release = asyncio.Event()

    async def get_fresh(path: str) -> list[LspDiagnostic]:
        nonlocal active, max_active
        calls.append(path)
        active += 1
        max_active = max(max_active, active)
        if active == 4:
            admitted.set()
        try:
            await release.wait()
            if path == "bad.py":
                raise RuntimeError("broken server")
            return [_make_diag(0, path)]
        finally:
            active -= 1

    manager.get_fresh_diagnostics = get_fresh  # type: ignore[method-assign]
    paths = ["a.py", "b.py", "c.py", "d.py", "e.py", "bad.py", "a.py"]

    batch = asyncio.create_task(manager.get_fresh_diagnostics_for_paths(paths))
    await asyncio.wait_for(admitted.wait(), 10)
    release.set()
    results = await asyncio.wait_for(batch, 10)

    assert list(results) == paths[:-1]
    assert calls.count("a.py") == 1
    assert max_active == 4
    assert results["a.py"][0].message == "a.py"
    assert results["bad.py"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_on", ["lock", "detection"])
async def test_batch_budget_includes_lazy_initialization(
    manager: LspManager, monkeypatch: pytest.MonkeyPatch, blocked_on: str
) -> None:
    monkeypatch.setattr(manager_module, "_AUTO_DIAGNOSTICS_TIMEOUT_SECONDS", 0.05)
    detection_cancelled = asyncio.Event()

    async def detect(*args: object) -> None:
        try:
            await asyncio.Event().wait()
        finally:
            detection_cancelled.set()

    monkeypatch.setattr(manager_module, "detect_languages", detect)
    query = AsyncMock()
    monkeypatch.setattr(manager, "get_fresh_diagnostics", query)
    if blocked_on == "lock":
        await manager._init_lock.acquire()
    try:
        result = await asyncio.wait_for(manager.get_fresh_diagnostics_for_paths(["a.py", "a.py"]), 10)
    finally:
        if manager._init_lock.locked():
            manager._init_lock.release()

    assert result == {"a.py": []}
    assert not manager._initialized
    query.assert_not_awaited()
    assert detection_cancelled.is_set() == (blocked_on == "detection")


@pytest.mark.asyncio
async def test_batch_timeout_preserves_partial_results_and_settles_queued_work(
    manager: LspManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager._initialized = True
    monkeypatch.setattr(manager_module, "_AUTO_DIAGNOSTICS_TIMEOUT_SECONDS", 0.1)
    active = 0
    maximum = 0
    settled: set[str] = set()
    calls: list[str] = []

    async def query(path: str) -> list[LspDiagnostic]:
        nonlocal active, maximum
        calls.append(path)
        active += 1
        maximum = max(maximum, active)
        try:
            if path.startswith("healthy"):
                return [_make_diag(0, path)]
            await asyncio.Event().wait()
            return []
        finally:
            active -= 1
            settled.add(path)

    monkeypatch.setattr(manager, "get_fresh_diagnostics", query)
    paths = ["stuck1", "stuck2", "stuck3", "healthy1", "healthy2", "stuck4", "queued", "healthy1"]
    result = await asyncio.wait_for(manager.get_fresh_diagnostics_for_paths(paths), 10)

    assert list(result) == paths[:-1]
    assert result["healthy1"][0].message == "healthy1"
    assert result["healthy2"][0].message == "healthy2"
    assert result["queued"] == []
    assert "queued" not in calls
    assert maximum == 4
    assert active == 0
    assert settled == set(calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_batch_waits_for_cancelled_query_cleanup(
    manager: LspManager, monkeypatch: pytest.MonkeyPatch, cancel: bool
) -> None:
    manager._initialized = True
    monkeypatch.setattr(manager_module, "_AUTO_DIAGNOSTICS_TIMEOUT_SECONDS", 10 if cancel else 0.05)
    started = asyncio.Event()
    cleaning = asyncio.Event()
    release = asyncio.Event()

    async def query(path: str) -> list[LspDiagnostic]:
        started.set()
        try:
            await asyncio.Event().wait()
            return []
        finally:
            cleaning.set()
            await release.wait()

    monkeypatch.setattr(manager, "get_fresh_diagnostics", query)
    task = asyncio.create_task(manager.get_fresh_diagnostics_for_paths(["a.py"]))
    await asyncio.wait_for(started.wait(), 10)
    if cancel:
        task.cancel()
    await asyncio.wait_for(cleaning.wait(), 10)
    assert not task.done()
    release.set()
    if cancel:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 10)
    else:
        assert await asyncio.wait_for(task, 10) == {"a.py": []}


def _startup_client() -> MagicMock:
    client = MagicMock()
    client.running = True
    client.start = AsyncMock()
    client.request = AsyncMock(return_value={"capabilities": {}})
    client.notify = AsyncMock()
    client.stop = AsyncMock()
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["start", "request", "notify"])
async def test_startup_cancellation_settles_cleanup(
    manager: LspManager, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    client = _startup_client()
    monkeypatch.setattr(manager_module, "LspClient", lambda *args, **kwargs: client)
    entered = asyncio.Event()
    cleaning = asyncio.Event()
    release = asyncio.Event()

    async def block(*args: object, **kwargs: object) -> None:
        entered.set()
        await asyncio.Event().wait()

    async def stop() -> None:
        cleaning.set()
        await release.wait()

    getattr(client, stage).side_effect = block
    client.stop.side_effect = stop
    task = asyncio.create_task(manager._get_session("python", manager._resolved["python"]))
    await asyncio.wait_for(entered.wait(), 10)
    task.cancel()
    await asyncio.wait_for(cleaning.wait(), 10)
    assert manager._sessions["python"] is client
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 10)
    client.stop.assert_awaited_once()
    assert manager._sessions == {}
    assert manager._session_records == {}
    assert manager._starting_sessions == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["start", "request", "notify"])
async def test_startup_failure_retains_ownership_when_cleanup_fails(
    manager: LspManager, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    client = _startup_client()
    getattr(client, stage).side_effect = RuntimeError("startup failed")
    client.stop.side_effect = LspClientError("cleanup failed")
    monkeypatch.setattr(manager_module, "LspClient", lambda *args, **kwargs: client)

    assert await asyncio.wait_for(manager._get_session("python", manager._resolved["python"]), 10) is None
    assert manager._sessions["python"] is client
    assert manager._session_records["python"].client is client
    client.stop.assert_awaited_once()
    assert manager._starting_sessions == {}


@pytest.mark.asyncio
async def test_shutdown_cancels_pending_start_and_rejects_new_starts(
    manager: LspManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _startup_client()
    constructor = MagicMock(return_value=client)
    monkeypatch.setattr(manager_module, "LspClient", constructor)
    entered = asyncio.Event()
    cleaning = asyncio.Event()
    release = asyncio.Event()

    async def initialize(*args: object, **kwargs: object) -> None:
        entered.set()
        await asyncio.Event().wait()

    async def stop() -> None:
        cleaning.set()
        await release.wait()

    client.request.side_effect = initialize
    client.stop.side_effect = stop
    starting = asyncio.create_task(manager._get_session("python", manager._resolved["python"]))
    await asyncio.wait_for(entered.wait(), 10)
    shutdown = asyncio.create_task(manager.shutdown())
    await asyncio.wait_for(cleaning.wait(), 10)
    assert await manager._get_session("other", manager._resolved["python"]) is None
    assert not shutdown.done()
    release.set()
    await asyncio.wait_for(shutdown, 10)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(starting, 10)
    constructor.assert_called_once()
    assert manager._sessions == {}
    assert manager._starting_sessions == {}


@pytest.mark.asyncio
async def test_shutdown_deadline_includes_blocked_exit(manager: LspManager, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(manager_module, "_LSP_SHUTDOWN_REQUEST_TIMEOUT_SECONDS", 0.05)
    client = _startup_client()
    exit_cancelled = asyncio.Event()

    async def exit_notification(*args: object, **kwargs: object) -> None:
        try:
            await asyncio.Event().wait()
        finally:
            exit_cancelled.set()

    client.notify.side_effect = exit_notification
    manager._sessions["python"] = client
    await asyncio.wait_for(manager.shutdown(), 10)

    assert exit_cancelled.is_set()
    client.stop.assert_awaited_once()
    assert manager._sessions == {}


@pytest.mark.asyncio
async def test_shutdown_is_retained_through_caller_cancellation_and_stops_all_clients(manager: LspManager) -> None:
    clients = [_startup_client(), _startup_client()]
    both_cleaning = asyncio.Event()
    release = asyncio.Event()
    count = 0

    async def stop() -> None:
        nonlocal count
        count += 1
        if count == 2:
            both_cleaning.set()
        await release.wait()

    for key, client in zip(("python", "extra"), clients):
        client.stop.side_effect = stop
        manager._sessions[key] = client
    first = asyncio.create_task(manager.shutdown())
    await asyncio.wait_for(both_cleaning.wait(), 10)
    first.cancel()
    second = asyncio.create_task(manager.shutdown())
    assert not first.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(first, 10)
    await asyncio.wait_for(second, 10)
    await manager.shutdown()
    for client in clients:
        client.stop.assert_awaited_once()
    assert manager._sessions == {}
    assert manager._shutdown_task is not None and manager._shutdown_task.done()


@pytest.mark.asyncio
async def test_shutdown_preserves_failed_cleanup_maps_and_settles_other_clients(manager: LspManager) -> None:
    failed, healthy = _startup_client(), _startup_client()
    failed.stop.side_effect = LspClientError("unreaped process")
    manager._sessions.update(failed=failed, healthy=healthy)
    manager._session_records.update(failed=MagicMock(), healthy=MagicMock())
    manager._extra_sessions[("python", "failed")] = failed
    manager._extra_sessions[("python", "healthy")] = healthy

    await asyncio.wait_for(manager.shutdown(), 10)

    assert manager._sessions == {"failed": failed}
    assert set(manager._session_records) == {"failed"}
    assert manager._extra_sessions == {("python", "failed"): failed}
    healthy.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_restart_replaces_document_state_and_sends_did_open(
    manager: LspManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = manager._project_path / "test.py"
    path.write_text("value = 1\n")
    uri = path.as_uri()
    old = _startup_client()
    old.running = False
    manager._sessions["python"] = old
    manager._session_records["python"] = _LspSession("python", "python", "pyright", old, manager._project_path.as_uri())
    manager._session_records["python"].opened_docs[uri] = "python"
    manager._session_records["python"].doc_versions[uri] = 7
    manager._open_files[uri] = "python"
    new = _startup_client()
    monkeypatch.setattr(manager_module, "LspClient", lambda *args, **kwargs: new)

    assert await asyncio.wait_for(manager._get_session("python", manager._resolved["python"]), 10) is new
    await manager._ensure_document_open(new, uri, "test.py", "python", session_key="python")

    old.stop.assert_awaited_once()
    assert new.notify.await_args_list[-1].args[0] == "textDocument/didOpen"
    assert new.notify.await_args_list[-1].args[1]["textDocument"]["version"] == 1
    assert manager._session_records["python"].client is new


@pytest.mark.asyncio
async def test_shutdown_settles_start_semaphore_waiters_without_starting_clients(
    manager: LspManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager._start_semaphore = asyncio.Semaphore(0)
    queued = asyncio.Event()
    original_start = manager._start_session

    async def start(lang_id: str, rl: ResolvedLanguage, key: str, *, extra: bool) -> LspClient | None:
        queued.set()
        return await original_start(lang_id, rl, key, extra=extra)

    monkeypatch.setattr(manager, "_start_session", start)
    constructor = MagicMock()
    monkeypatch.setattr(manager_module, "LspClient", constructor)
    starting = asyncio.create_task(manager._get_session("python", manager._resolved["python"]))
    await asyncio.wait_for(queued.wait(), 10)

    await asyncio.wait_for(manager.shutdown(), 10)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(starting, 10)
    assert manager._sessions == {}
    assert manager._starting_sessions == {}
    constructor.assert_not_called()


@pytest.mark.asyncio
async def test_shutdown_rejects_session_lock_waiters(manager: LspManager, monkeypatch: pytest.MonkeyPatch) -> None:
    constructor = MagicMock()
    monkeypatch.setattr(manager_module, "LspClient", constructor)
    lock = asyncio.Lock()
    manager._session_locks["python"] = lock
    await lock.acquire()
    starting = asyncio.create_task(manager._get_session("python", manager._resolved["python"]))
    try:
        await asyncio.wait_for(manager.shutdown(), 10)
    finally:
        lock.release()
    assert await asyncio.wait_for(starting, 10) is None
    constructor.assert_not_called()
