from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, Mock

import pytest

from kolega_code.agent.tool_backend.lsp_tool import LspEditTool


def _make_tool(manager) -> LspEditTool:
    caller = Mock(agent_name="test_agent", sub_agent=False, current_tool_execution_id=None)
    return LspEditTool(
        manager._project_path,
        "test_workspace",
        str(uuid.uuid4()),
        AsyncMock(),
        Mock(),
        caller,
        lsp_manager=manager,
    )


@pytest.mark.asyncio
async def test_lsp_edit_rename_applies_fake_server_workspace_edit(fake_lsp_manager):
    manager = fake_lsp_manager
    path = manager._project_path / "rename_tool.py"
    path.write_text("old = 1\nprint(old)\n", encoding="utf-8")

    result = await _make_tool(manager).lsp_edit(
        operation="rename",
        path="rename_tool.py",
        line=1,
        symbol="old",
        new_name="new",
    )

    assert result.startswith("Applied LSP edit `rename`.")
    assert path.read_text(encoding="utf-8") == "new = 1\nprint(new)\n"


@pytest.mark.asyncio
async def test_lsp_edit_rename_file_applies_will_rename_and_resource_rename(fake_lsp_manager):
    manager = fake_lsp_manager
    old_path = manager._project_path / "old.py"
    new_path = manager._project_path / "new.py"
    importer = manager._project_path / "importer.py"
    old_path.write_text("value = 1\n", encoding="utf-8")
    importer.write_text("from old import value\n", encoding="utf-8")
    await manager.get_diagnostics("importer.py")

    result = await _make_tool(manager).lsp_edit(
        operation="rename_file",
        path="old.py",
        new_path="new.py",
    )

    assert "Applied LSP edit `rename_file`." in result
    assert not old_path.exists()
    assert new_path.read_text(encoding="utf-8") == "value = 1\n"
    assert importer.read_text(encoding="utf-8") == "from new import value\n"


@pytest.mark.asyncio
async def test_lsp_edit_apply_code_action_allows_scoped_workspace_apply_edit(fake_lsp_manager):
    manager = fake_lsp_manager
    path = manager._project_path / "imports_tool.py"
    path.write_text("import unused\nvalue = 1\n", encoding="utf-8")

    result = await _make_tool(manager).lsp_edit(
        operation="apply_code_action",
        path="imports_tool.py",
        line=1,
        symbol="import",
        query="Organize imports",
    )

    assert "Executed LSP command `fake.organizeImports`." in result
    assert path.read_text(encoding="utf-8") == "value = 1\n"
