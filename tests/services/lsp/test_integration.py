"""Integration tests for the LSP client/manager against a fake stdio LSP server.

These tests launch ``_fake_server.py`` as a subprocess and verify the full
JSON-RPC round-trip: initialize, document sync, diagnostics (push + pull),
code intelligence queries, and server→client request handling.
"""

from __future__ import annotations

import asyncio

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
    # The fake server returns a markdown code block; assert against the body.
    assert contents["value"].startswith("```python")


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
async def test_code_actions(fake_lsp_manager):
    """textDocument/codeAction returns read-only action metadata."""
    manager = fake_lsp_manager
    (manager._project_path / "actions.py").write_text("value = undefined_var\n", encoding="utf-8")
    result = await manager.get_code_actions("actions.py", 0, 8)
    assert result is not None
    assert any(action.get("title") == "Replace undefined_var with defined_var" for action in result)


@pytest.mark.asyncio
async def test_code_action_resolve(fake_lsp_manager):
    """codeAction/resolve returns a server-provided edit."""
    manager = fake_lsp_manager
    (manager._project_path / "resolve_actions.py").write_text("value = undefined_var\n", encoding="utf-8")
    actions = await manager.get_code_actions("resolve_actions.py", 0, 8)
    unresolved = next(action for action in actions if action.get("title") == "Resolve undefined_var with defined_var")

    resolved = await manager.resolve_code_action("resolve_actions.py", unresolved)

    assert "edit" in resolved
    assert resolved["edit"]["changes"]


@pytest.mark.asyncio
async def test_rename_returns_workspace_edit(fake_lsp_manager):
    """textDocument/rename returns a WorkspaceEdit."""
    manager = fake_lsp_manager
    (manager._project_path / "rename.py").write_text("old = 1\nprint(old)\n", encoding="utf-8")

    edit = await manager.get_rename("rename.py", 0, 0, "new")

    assert edit is not None
    changes = edit["changes"]
    edits = next(iter(changes.values()))
    assert len(edits) == 2


@pytest.mark.asyncio
async def test_formatting_returns_text_edits(fake_lsp_manager):
    """textDocument/formatting returns TextEdits."""
    manager = fake_lsp_manager
    (manager._project_path / "format_me.py").write_text("x = 1   \n", encoding="utf-8")

    edits = await manager.get_document_formatting("format_me.py")

    assert edits is not None
    assert edits[0]["newText"] == "x = 1\n"


@pytest.mark.asyncio
async def test_will_rename_files_returns_workspace_edit(fake_lsp_manager):
    """workspace/willRenameFiles returns edits for opened referencing documents."""
    manager = fake_lsp_manager
    (manager._project_path / "importer.py").write_text("from old import value\n", encoding="utf-8")
    await manager.get_diagnostics("importer.py")

    edits = await manager.will_rename_files("old.py", "new.py")

    assert len(edits) == 1
    assert edits[0]["changes"]


@pytest.mark.asyncio
async def test_workspace_apply_edit_denied_by_default(fake_lsp_manager):
    """Server-initiated workspace/applyEdit is denied unless a trusted edit tool scopes it."""
    manager = fake_lsp_manager
    path = manager._project_path / "imports.py"
    path.write_text("import unused\nvalue = 1\n", encoding="utf-8")
    await manager.get_diagnostics("imports.py")

    result = await manager.execute_command(
        "imports.py",
        {"command": "fake.organizeImports", "arguments": [path.as_uri()]},
    )

    assert result == {"applied": False}
    assert path.read_text(encoding="utf-8") == "import unused\nvalue = 1\n"


@pytest.mark.asyncio
async def test_call_hierarchy(fake_lsp_manager):
    """Call hierarchy prepare + incoming/outgoing calls are queried."""
    manager = fake_lsp_manager
    (manager._project_path / "calls.py").write_text("def example():\n    pass\n", encoding="utf-8")
    result = await manager.get_call_hierarchy("calls.py", 0, 4)
    assert result is not None
    assert result["items"]
    assert result["incoming"]
    assert result["outgoing"]


@pytest.mark.asyncio
async def test_extra_diagnostic_server_gets_its_own_did_open(fake_lsp_manager_with_extra_strict):
    """Extra diagnostic servers must receive didOpen before didChange."""
    manager = fake_lsp_manager_with_extra_strict
    (manager._project_path / "multi.py").write_text("value = undefined_var\n", encoding="utf-8")
    diags = await manager.get_diagnostics("multi.py")
    sources = {d.source for d in diags}
    assert {"primary", "extra"} <= sources


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


# ---------------------------------------------------------------------------
# T1: push-only freshness gate (previously untested)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fresh_diagnostics_push_only_accepts_fresh(fake_lsp_manager_no_pull):
    """T1a: with a push-only server, get_fresh_diagnostics returns fresh results.

    Exercises the push-diagnostics fallback path that was previously skipped
    because every other fixture enables pull diagnostics.
    """
    manager = fake_lsp_manager_no_pull
    path = manager._project_path / "push.py"
    path.write_text("a = 1\n", encoding="utf-8")

    # Introduce an error and query — push diagnostics should be accepted.
    path.write_text("a = undefined_var\n", encoding="utf-8")
    diags = await manager.get_fresh_diagnostics("push.py")
    assert len(diags) >= 1
    assert any("undefined_var" in d.message for d in diags)


@pytest.mark.asyncio
async def test_fresh_diagnostics_push_only_suppresses_stale(fake_lsp_manager_push_fixed_version):
    """T1b: push diagnostics whose version matches the pre-edit version are suppressed.

    The fixture's server always publishes with a fixed version, so after the first
    query the recorded version never advances. A second query must detect the
    diagnostics as stale and return an empty list.
    """
    manager = fake_lsp_manager_push_fixed_version
    path = manager._project_path / "stale.py"
    path.write_text("a = undefined_var\n", encoding="utf-8")

    # First query seeds the recorded version (fixed at 5) and returns diags.
    diags1 = await manager.get_fresh_diagnostics("stale.py")
    assert len(diags1) >= 1

    # Second query: the server republishes with the same fixed version (5), so the
    # freshness gate treats the push diagnostics as stale and suppresses them.
    diags2 = await manager.get_fresh_diagnostics("stale.py")
    assert diags2 == []


# ---------------------------------------------------------------------------
# T2: concurrent session starts (the _get_session race — F3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_same_language_starts_single_session(fake_lsp_manager):
    """T2: two concurrent diagnostics queries for one language start one session.

    Guards against the F3 race where concurrent callers each spawn + initialize a
    subprocess and the second overwrites (orphaning) the first.
    """
    manager = fake_lsp_manager
    (manager._project_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (manager._project_path / "b.py").write_text("y = 2\n", encoding="utf-8")

    # Fire two concurrent queries for the same language.
    await asyncio.gather(
        manager.get_diagnostics("a.py"),
        manager.get_diagnostics("b.py"),
    )

    # Exactly one session/subprocess should exist for the language.
    assert len(manager._sessions) == 1
