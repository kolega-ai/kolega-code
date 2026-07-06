"""Integration tests for the LSP client/manager against a fake stdio LSP server.

These tests launch ``_fake_server.py`` as a subprocess and verify the full
JSON-RPC round-trip: initialize, document sync, diagnostics (push + pull),
code intelligence queries, and server→client request handling.
"""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_initialize_and_capabilities(fake_lsp_manager):
    """The fake server initializes and capabilities are stored on the client."""
    manager = fake_lsp_manager
    # Start a session by querying diagnostics
    diags = await manager.get_diagnostics("src.py")
    assert isinstance(diags, list)

    # Check that a session was created
    assert len(manager._sessions) > 0
    client = next(iter(manager._sessions.values()))
    assert client.status == "initialized"
    assert client.server_capabilities is not None
    assert client.server_capabilities.get("definitionProvider") is True
    assert client.server_pid is not None


@pytest.mark.asyncio
async def test_server_request_handled_no_hang(fake_lsp_manager):
    """The workspace/configuration request from the server gets a response (no hang).

    The fake server sends workspace/configuration during initialized notification.
    If the client doesn't respond, the server would hang.  We verify the session
    is initialized (meaning the handshake completed without hanging).
    """
    manager = fake_lsp_manager
    await manager.get_diagnostics("src.py")
    client = next(iter(manager._sessions.values()))
    assert client.status == "initialized"


@pytest.mark.asyncio
async def test_push_diagnostics_on_open(fake_lsp_manager):
    """Diagnostics arrive via publishDiagnostics after didOpen."""
    manager = fake_lsp_manager
    # Write a file with a diagnostic trigger
    (manager._project_path / "bug.py").write_text("x = undefined_var\n", encoding="utf-8")
    diags = await manager.get_diagnostics("bug.py")
    assert len(diags) >= 1
    assert any("undefined_var" in d.message for d in diags)


@pytest.mark.asyncio
async def test_pull_diagnostics(fake_lsp_manager):
    """Pull diagnostics (textDocument/diagnostic) return results."""
    manager = fake_lsp_manager
    (manager._project_path / "pull_test.py").write_text("y = undefined_var\n", encoding="utf-8")
    diags = await manager.get_diagnostics("pull_test.py")
    assert len(diags) >= 1


@pytest.mark.asyncio
async def test_definition_query(fake_lsp_manager):
    """textDocument/definition returns a location."""
    manager = fake_lsp_manager
    (manager._project_path / "def_test.py").write_text("value = 42\n", encoding="utf-8")
    result = await manager.get_definition("def_test.py", 0, 0)
    assert result is not None
    assert isinstance(result, list)
    assert len(result) >= 1
    assert "uri" in result[0]


@pytest.mark.asyncio
async def test_references_query(fake_lsp_manager):
    """textDocument/references returns locations."""
    manager = fake_lsp_manager
    (manager._project_path / "ref_test.py").write_text("value = 42\n", encoding="utf-8")
    result = await manager.get_references("ref_test.py", 0, 0)
    assert result is not None
    assert len(result) >= 2


@pytest.mark.asyncio
async def test_hover_query(fake_lsp_manager):
    """textDocument/hover returns hover content."""
    manager = fake_lsp_manager
    (manager._project_path / "hover_test.py").write_text("value = 42\n", encoding="utf-8")
    result = await manager.get_hover("hover_test.py", 0, 0)
    assert result is not None
    contents = result.get("contents", {})
    assert "value" in contents or "value" in str(contents)


@pytest.mark.asyncio
async def test_document_symbols(fake_lsp_manager):
    """textDocument/documentSymbol returns parsed symbols."""
    manager = fake_lsp_manager
    (manager._project_path / "sym_test.py").write_text(
        "def foo():\n    pass\n\nclass Bar:\n    pass\n", encoding="utf-8"
    )
    result = await manager.get_document_symbols("sym_test.py")
    assert result is not None
    assert len(result) >= 2
    names = [s.get("name") for s in result]
    assert "foo" in names
    assert "Bar" in names


@pytest.mark.asyncio
async def test_workspace_symbols(fake_lsp_manager):
    """workspace/symbol returns matching symbols."""
    manager = fake_lsp_manager
    # Need an active session first
    await manager.get_diagnostics("src.py")
    result = await manager.get_workspace_symbols("test")
    assert result is not None
    assert len(result) >= 1


@pytest.mark.asyncio
async def test_status_returns_sessions(fake_lsp_manager):
    """status() returns active session info after a session is started."""
    manager = fake_lsp_manager
    await manager.get_diagnostics("src.py")
    status = manager.status()
    assert status["enabled"] is True
    assert status["initialized"] is True
    assert len(status["sessions"]) >= 1
    session = status["sessions"][0]
    assert session["connected"] is True
    assert session["pid"] is not None
    assert session["server_name"] == "fake-lsp"


@pytest.mark.asyncio
async def test_capabilities_for_path(fake_lsp_manager):
    """get_capabilities returns the server's capabilities for a file."""
    manager = fake_lsp_manager
    await manager.get_diagnostics("src.py")
    caps = manager.get_capabilities("src.py")
    assert caps.get("definitionProvider") is True
    assert caps.get("hoverProvider") is True


@pytest.mark.asyncio
async def test_fresh_diagnostics_after_edit(fake_lsp_manager):
    """get_fresh_diagnostics returns updated diagnostics after content change."""
    manager = fake_lsp_manager
    path = manager._project_path / "fresh.py"
    path.write_text("a = 1\n", encoding="utf-8")

    # First query — no diagnostics
    diags1 = await manager.get_fresh_diagnostics("fresh.py")
    assert len(diags1) == 0

    # Edit the file to introduce an error
    path.write_text("a = undefined_var\n", encoding="utf-8")
    diags2 = await manager.get_fresh_diagnostics("fresh.py")
    assert len(diags2) >= 1
    assert any("undefined_var" in d.message for d in diags2)


@pytest.mark.asyncio
async def test_reload_restarts_sessions(fake_lsp_manager):
    """reload() shuts down and re-initializes."""
    manager = fake_lsp_manager
    await manager.get_diagnostics("src.py")
    assert len(manager._sessions) >= 1

    await manager.reload()
    # After reload, sessions are cleared but can be restarted
    assert len(manager._sessions) == 0
    # Re-query to start a new session
    await manager.get_diagnostics("src.py")
    assert len(manager._sessions) >= 1


@pytest.mark.asyncio
async def test_dedupe_and_sort_applied(fake_lsp_manager):
    """Diagnostics are deduplicated and sorted by severity."""
    from kolega_code.services.lsp.diagnostics import dedupe_and_sort
    from kolega_code.services.lsp.client import LspDiagnostic

    diags = [
        LspDiagnostic(
            range={"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 5}},
            severity=2,
            message="warning",
            source="test",
        ),
        LspDiagnostic(
            range={"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 5}},
            severity=2,
            message="warning",
            source="test",
        ),  # duplicate
        LspDiagnostic(
            range={"start": {"line": 1, "character": 0}, "end": {"line": 1, "character": 5}},
            severity=1,
            message="error",
            source="test",
        ),
    ]
    result = dedupe_and_sort(diags, max_count=20)
    assert len(result) == 2
    # Error (severity 1) comes before warning (severity 2)
    assert result[0].severity == 1
    assert result[1].severity == 2
