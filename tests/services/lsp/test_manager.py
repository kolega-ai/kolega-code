"""Unit tests for LspManager internal helpers (no real server needed).

Tests position resolution, status, capability checking, and dedupe/sort.
End-to-end tests with a fake server are in ``test_integration.py``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from kolega_code.services.lsp.client import LspDiagnostic
from kolega_code.services.lsp.config import LspConfig
from kolega_code.services.lsp.diagnostics import dedupe_and_sort
from kolega_code.services.lsp.detector import DetectionReport, DetectionResult, ResolvedLanguage
from kolega_code.services.lsp.manager import LspManager, _LspSession, _path_to_uri


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
        detected=[DetectionResult("python", "Python", ["pyproject.toml"], 5, "pyproject.toml")],
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
