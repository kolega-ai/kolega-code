"""Tests for the generic ``lsp`` and ``lsp_edit`` tools.

The LspManager is fully mocked — no real language server is started.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from kolega_code.agent.tool_backend.lsp_tool import LspEditTool, LspTool
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.services.lsp import LspDiagnostic


# ---------------------------------------------------------------------------
# shared fixtures (mirror tests/agent/tool_backend/test_edit_tool.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_connection_manager():
    return AsyncMock()


@pytest.fixture
def project_path(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    return project


@pytest.fixture
def agent_config():
    return AgentConfig(
        anthropic_api_key="test_key",
        openai_api_key="test-key",
        long_context_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC, model="test-model", rate_limits=RateLimitConfig()
        ),
        fast_config=ModelConfig(provider=ModelProvider.ANTHROPIC, model="test-model", rate_limits=RateLimitConfig()),
        thinking_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model="test-model",
            rate_limits=RateLimitConfig(),
            thinking_effort="medium",
        ),
    )


@pytest.fixture
def mock_base_agent():
    mock = Mock()
    mock.agent_name = "test_agent"
    mock.sub_agent = False
    mock.current_tool_execution_id = "test-call-id"
    return mock


@pytest.fixture
def make_lsp_tool(project_path, mock_connection_manager, agent_config, mock_base_agent):
    """Factory that builds an ``LspTool`` bound to a given (mock) LspManager."""

    def _make(lsp_manager=None) -> LspTool:
        return LspTool(
            project_path,
            "test_workspace",
            str(uuid.uuid4()),
            mock_connection_manager,
            agent_config,
            mock_base_agent,
            lsp_manager=lsp_manager,
        )

    return _make


@pytest.fixture
def make_lsp_edit_tool(project_path, mock_connection_manager, agent_config, mock_base_agent):
    def _make(lsp_manager=None) -> LspEditTool:
        return LspEditTool(
            project_path,
            "test_workspace",
            str(uuid.uuid4()),
            mock_connection_manager,
            agent_config,
            mock_base_agent,
            lsp_manager=lsp_manager,
        )

    return _make


@pytest.fixture
def mock_lsp_manager():
    """A fully-mocked, enabled, initialized LspManager (no real server)."""
    manager = MagicMock()
    manager.enabled = True
    manager._initialized = True
    manager.initialize = AsyncMock(return_value=[])

    # Diagnostics / routing
    manager.server_for_path = Mock(return_value="pyright")
    manager.get_diagnostics = AsyncMock(return_value=[])

    # Position resolution (sync) + code-intelligence handlers (async)
    manager._resolve_position = Mock(return_value=(0, 0))
    manager.get_definition = AsyncMock(return_value=None)
    manager.get_type_definition = AsyncMock(return_value=None)
    manager.get_implementation = AsyncMock(return_value=None)
    manager.get_references = AsyncMock(return_value=None)
    manager.get_hover = AsyncMock(return_value=None)

    # Symbols
    manager.get_document_symbols = AsyncMock(return_value=None)
    manager.get_workspace_symbols = AsyncMock(return_value=None)
    manager.get_code_actions = AsyncMock(return_value=None)
    manager.get_call_hierarchy = AsyncMock(return_value=None)
    manager.get_rename = AsyncMock(return_value=None)
    manager.get_document_formatting = AsyncMock(return_value=None)
    manager.get_range_formatting = AsyncMock(return_value=None)

    async def _resolve_code_action(_path, action, **_kwargs):
        return action

    manager.resolve_code_action = AsyncMock(side_effect=_resolve_code_action)
    manager.execute_command = AsyncMock(return_value=None)
    manager.will_rename_files = AsyncMock(return_value=[])
    manager.did_rename_files = AsyncMock(return_value=None)
    manager.set_workspace_apply_edit_handler = Mock()
    manager._config = MagicMock(auto_diagnostics_on_edit=False)

    # Status / capabilities / reload
    manager.status = Mock(
        return_value={
            "enabled": True,
            "initialized": True,
            "detected": [],
            "missing": [],
            "sessions": [],
            "diagnostic_counts": {},
        }
    )
    manager.get_capabilities = Mock(return_value={})
    manager.reload = AsyncMock(return_value=[])

    manager._sessions = {}
    return manager


@pytest.fixture
def lsp_tool(make_lsp_tool, mock_lsp_manager):
    return make_lsp_tool(mock_lsp_manager)


@pytest.fixture
def lsp_edit_tool(make_lsp_edit_tool, mock_lsp_manager):
    return make_lsp_edit_tool(mock_lsp_manager)


# ---------------------------------------------------------------------------
# 1. Argument validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestLspArgumentValidation:
    async def test_diagnostics_requires_path(self, lsp_tool):
        result = await lsp_tool.lsp(operation="diagnostics")
        assert result == "Error: 'path' is required for the 'diagnostics' operation."

    async def test_document_symbols_requires_path(self, lsp_tool):
        result = await lsp_tool.lsp(operation="document_symbols")
        assert result == "Error: 'path' is required for the 'document_symbols' operation."

    async def test_workspace_symbols_requires_query(self, lsp_tool):
        result = await lsp_tool.lsp(operation="workspace_symbols")
        assert result == "Error: 'query' is required for the 'workspace_symbols' operation."

    @pytest.mark.parametrize("operation", ["definition", "type_definition", "implementation", "references", "hover"])
    async def test_position_ops_require_path(self, lsp_tool, operation):
        result = await lsp_tool.lsp(operation=operation)
        assert result == f"Error: 'path' is required for the '{operation}' operation."

    @pytest.mark.parametrize("operation", ["definition", "type_definition", "implementation", "references", "hover"])
    async def test_position_ops_require_line(self, lsp_tool, operation):
        result = await lsp_tool.lsp(operation=operation, path="foo.py")
        assert result == f"Error: 'line' (1-based) is required for the '{operation}' operation."

    @pytest.mark.parametrize("operation", ["definition", "type_definition", "implementation", "references", "hover"])
    async def test_position_ops_require_symbol(self, lsp_tool, operation):
        result = await lsp_tool.lsp(operation=operation, path="foo.py", line=1)
        assert result == f"Error: 'symbol' is required for the '{operation}' operation."

    async def test_unknown_operation_returns_error(self, lsp_tool):
        result = await lsp_tool.lsp(operation="bogus", path="foo.py")
        assert result.startswith("Unknown operation 'bogus'.")
        assert "Valid operations:" in result
        # The error must list the real operations.
        for op in ("diagnostics", "definition", "hover", "status", "workspace_symbols"):
            assert op in result

    async def test_empty_string_treated_as_missing(self, lsp_tool):
        # Empty path / query strings are falsy and should be rejected too.
        assert (
            await lsp_tool.lsp(operation="diagnostics", path="")
            == "Error: 'path' is required for the 'diagnostics' operation."
        )
        assert (
            await lsp_tool.lsp(operation="workspace_symbols", query="")
            == "Error: 'query' is required for the 'workspace_symbols' operation."
        )


# ---------------------------------------------------------------------------
# 2. Disabled / absent LSP
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestLspDisabled:
    async def test_lsp_returns_not_available_when_manager_is_none(self, make_lsp_tool):
        tool = make_lsp_tool(lsp_manager=None)
        result = await tool.lsp(operation="diagnostics", path="foo.py")
        assert result == "LSP is not available (disabled or not configured)."

    async def test_lsp_returns_not_available_when_manager_disabled(self, make_lsp_tool):
        manager = MagicMock()
        manager.enabled = False
        tool = make_lsp_tool(lsp_manager=manager)
        result = await tool.lsp(operation="status")
        assert result == "LSP is not available (disabled or not configured)."

    async def test_disabled_check_precedes_operation_validation(self, make_lsp_tool):
        """Even an unknown operation reports 'not available' when LSP is off."""
        tool = make_lsp_tool(lsp_manager=None)
        result = await tool.lsp(operation="bogus")
        assert result == "LSP is not available (disabled or not configured)."


@pytest.mark.asyncio
class TestLspDiagnosticsOperation:
    async def test_diagnostics_operation_no_diagnostics(self, lsp_tool, mock_lsp_manager):
        mock_lsp_manager.server_for_path.return_value = "pyright"
        mock_lsp_manager.get_diagnostics.return_value = []

        result = await lsp_tool.lsp(operation="diagnostics", path="foo.py")

        assert result == "\n✅ No LSP diagnostics."

    async def test_diagnostics_operation_with_diagnostics(self, lsp_tool, mock_lsp_manager):
        mock_lsp_manager.server_for_path.return_value = "pyright"
        mock_lsp_manager.get_diagnostics.return_value = [
            LspDiagnostic(
                range={"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 3}},
                severity=1,
                message="Undefined name",
                source="pyright",
            ),
        ]

        result = await lsp_tool.lsp(operation="diagnostics", path="foo.py")

        assert "LSP diagnostics (1 error):" in result
        assert "Undefined name" in result

    async def test_diagnostics_operation_when_no_server_configured(self, lsp_tool, mock_lsp_manager):
        mock_lsp_manager.server_for_path.return_value = None

        result = await lsp_tool.lsp(operation="diagnostics", path="data.csv")

        assert result == "No language server configured for data.csv."

    async def test_diagnostics_operation_initializes_manager_when_needed(self, make_lsp_tool):
        manager = MagicMock()
        manager.enabled = True
        manager._initialized = False
        manager.initialize = AsyncMock(return_value=[])
        manager.server_for_path = Mock(return_value="pyright")
        manager.get_diagnostics = AsyncMock(return_value=[])
        tool = make_lsp_tool(lsp_manager=manager)

        await tool.lsp(operation="diagnostics", path="foo.py")

        manager.initialize.assert_awaited_once()


# ---------------------------------------------------------------------------
# 3. Status operation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestLspStatus:
    async def test_status_formats_enabled_manager(self, lsp_tool, mock_lsp_manager):
        mock_lsp_manager.status.return_value = {
            "enabled": True,
            "initialized": True,
            "detected": [
                {"language_id": "python", "display_name": "Python", "detection_reason": "by extension"},
                {"language_id": "typescript", "display_name": "TypeScript", "detection_reason": "by extension"},
            ],
            "sessions": [
                {
                    "language_id": "python",
                    "server_name": "pyright",
                    "connected": True,
                    "pid": 1234,
                    "last_error": None,
                },
                {
                    "language_id": "typescript",
                    "server_name": "typescript-language-server",
                    "connected": False,
                    "pid": None,
                    "last_error": "boom",
                },
            ],
            "missing": [
                {"language_id": "rust", "display_name": "Rust", "server_name": "rust-analyzer"},
            ],
            "diagnostic_counts": {"file:///proj/foo.py": 3},
        }

        result = await lsp_tool.lsp(operation="status")

        assert result.startswith("## 🔍 LSP Status")
        assert "Detected 2 language(s)" in result
        assert "Python (by extension)" in result
        assert "Active sessions (2)" in result
        assert "✅ pyright (python) pid=1234" in result
        assert "❌ typescript-language-server (typescript) — boom" in result
        assert "Missing servers (1)" in result
        assert "Rust → rust-analyzer" in result
        assert "Last diagnostic counts" in result
        assert "/proj/foo.py: 3" in result

    async def test_status_reports_disabled(self, lsp_tool, mock_lsp_manager):
        mock_lsp_manager.status.return_value = {
            "enabled": False,
            "initialized": False,
            "detected": [],
            "missing": [],
            "sessions": [],
            "diagnostic_counts": {},
        }

        result = await lsp_tool.lsp(operation="status")

        assert result.startswith("## 🔍 LSP Status")
        assert "LSP is disabled." in result

    async def test_status_no_languages_detected(self, lsp_tool, mock_lsp_manager):
        mock_lsp_manager.status.return_value = {
            "enabled": True,
            "initialized": True,
            "detected": [],
            "missing": [],
            "sessions": [],
            "diagnostic_counts": {},
        }

        result = await lsp_tool.lsp(operation="status")

        assert "No languages detected." in result

    async def test_status_calls_manager_status(self, lsp_tool, mock_lsp_manager):
        await lsp_tool.lsp(operation="status")
        mock_lsp_manager.status.assert_called_once()


@pytest.mark.asyncio
class TestLspNewReadOnlyOperations:
    async def test_code_actions_formats_metadata_without_applying(self, lsp_tool, mock_lsp_manager):
        mock_lsp_manager._resolve_position.return_value = (0, 8)
        mock_lsp_manager.get_code_actions.return_value = [
            {
                "title": "Replace undefined_var with defined_var",
                "kind": "quickfix",
                "edit": {"changes": {"file:///repo/foo.py": [{"newText": "defined_var"}]}},
            }
        ]

        result = await lsp_tool.lsp(
            operation="code_actions",
            path="foo.py",
            line=1,
            symbol="undefined_var",
            kind="quickfix",
        )

        assert "## Code Actions (1 found)" in result
        assert "Replace undefined_var with defined_var" in result
        assert "action_id=" in result
        assert "1 text edits" in result
        mock_lsp_manager.get_code_actions.assert_awaited_once_with(
            "foo.py",
            0,
            8,
            end_line=None,
            kind="quickfix",
        )

    async def test_call_hierarchy_formats_incoming_and_outgoing(self, lsp_tool, mock_lsp_manager):
        item = {
            "name": "example",
            "kind": 12,
            "uri": "file:///repo/foo.py",
            "selectionRange": {"start": {"line": 0, "character": 4}, "end": {"line": 0, "character": 11}},
        }
        mock_lsp_manager._resolve_position.return_value = (0, 4)
        mock_lsp_manager.get_call_hierarchy.return_value = {
            "items": [item],
            "incoming": [{"from": {**item, "name": "caller"}}],
            "outgoing": [{"to": {**item, "name": "callee"}}],
        }

        result = await lsp_tool.lsp(operation="call_hierarchy", path="foo.py", line=1, symbol="example")

        assert "## Call Hierarchy" in result
        assert "caller" in result
        assert "callee" in result

    async def test_location_links_are_formatted_and_decoded(self, lsp_tool, mock_lsp_manager):
        mock_lsp_manager._resolve_position.return_value = (0, 0)
        mock_lsp_manager.get_definition.return_value = [
            {
                "targetUri": "file:///repo/space%20file.py",
                "targetSelectionRange": {"start": {"line": 2, "character": 4}, "end": {"line": 2, "character": 8}},
            }
        ]

        result = await lsp_tool.lsp(operation="definition", path="foo.py", line=1, symbol="foo")

        assert "/repo/space file.py:3:4" in result

    async def test_document_symbols_formats_children(self, lsp_tool, mock_lsp_manager):
        mock_lsp_manager.get_document_symbols.return_value = [
            {
                "name": "Parent",
                "kind": 5,
                "range": {"start": {"line": 0, "character": 0}},
                "children": [
                    {
                        "name": "child",
                        "kind": 12,
                        "range": {"start": {"line": 1, "character": 4}},
                    }
                ],
            }
        ]

        result = await lsp_tool.lsp(operation="document_symbols", path="foo.py")

        assert "`Parent` (Class)" in result
        assert "`child` (Function)" in result


@pytest.mark.asyncio
class TestLspEditTool:
    async def test_file_uri_preserves_parent_segments(self, lsp_edit_tool, project_path):
        uri = lsp_edit_tool._file_uri("link/../target.py")

        assert f"{project_path.as_uri()}/link/../target.py" == uri

    async def test_multi_path_diagnostics_uses_batch_and_preserves_order(
        self,
        lsp_edit_tool,
        mock_lsp_manager,
        project_path,
    ):
        for path in ("first.py", "second.py"):
            (project_path / path).write_text("value = 1\n", encoding="utf-8")
        mock_lsp_manager._config.auto_diagnostics_on_edit = True
        mock_lsp_manager.get_fresh_diagnostics_for_paths = AsyncMock(
            return_value={
                "first.py": [
                    LspDiagnostic(
                        range={"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 5}},
                        severity=1,
                        message="first issue",
                        source="pyright",
                    )
                ],
                "second.py": [
                    LspDiagnostic(
                        range={"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 6}},
                        severity=2,
                        message="second issue",
                        source="pyright",
                    )
                ],
            }
        )

        result = await lsp_edit_tool._diagnostics_for_paths(("first.py", "second.py", "first.py"))

        mock_lsp_manager.get_fresh_diagnostics_for_paths.assert_awaited_once_with(
            {"first.py": "pyright", "second.py": "pyright"}
        )
        assert result.index("first issue") < result.index("second issue")

    async def test_rename_preview_does_not_write_file(self, lsp_edit_tool, mock_lsp_manager, project_path):
        path = project_path / "foo.py"
        path.write_text("old = 1\nprint(old)\n", encoding="utf-8")
        mock_lsp_manager._resolve_position.return_value = (0, 0)
        mock_lsp_manager.get_rename.return_value = {
            "changes": {
                path.as_uri(): [
                    {
                        "range": {
                            "start": {"line": 0, "character": 0},
                            "end": {"line": 0, "character": 3},
                        },
                        "newText": "new",
                    }
                ]
            }
        }

        result = await lsp_edit_tool.lsp_edit(
            operation="rename",
            path="foo.py",
            line=1,
            symbol="old",
            new_name="new",
            apply=False,
        )

        assert result.startswith("Preview LSP edit `rename`.")
        assert path.read_text(encoding="utf-8") == "old = 1\nprint(old)\n"

    async def test_format_document_applies_server_text_edits(self, lsp_edit_tool, mock_lsp_manager, project_path):
        path = project_path / "foo.py"
        path.write_text("x = 1   \n", encoding="utf-8")
        mock_lsp_manager.get_document_formatting.return_value = [
            {
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 1, "character": 0},
                },
                "newText": "x = 1\n",
            }
        ]

        result = await lsp_edit_tool.lsp_edit(operation="format_document", path="foo.py")

        assert result.startswith("Applied LSP edit `format_document`.")
        assert path.read_text(encoding="utf-8") == "x = 1\n"

    async def test_apply_code_action_selects_by_action_id(self, lsp_edit_tool, mock_lsp_manager, project_path):
        path = project_path / "foo.py"
        path.write_text("value = undefined_var\n", encoding="utf-8")
        action = {
            "title": "Replace undefined_var with defined_var",
            "kind": "quickfix",
            "edit": {
                "changes": {
                    path.as_uri(): [
                        {
                            "range": {
                                "start": {"line": 0, "character": 8},
                                "end": {"line": 0, "character": 21},
                            },
                            "newText": "defined_var",
                        }
                    ]
                }
            },
        }
        mock_lsp_manager._resolve_position.return_value = (0, 8)
        mock_lsp_manager.get_code_actions.return_value = [action]

        result = await lsp_edit_tool.lsp_edit(
            operation="apply_code_action",
            path="foo.py",
            line=1,
            symbol="undefined_var",
            action_id=LspTool._action_id(action, 0),
        )

        assert result.startswith("Applied LSP edit `apply_code_action`.")
        assert path.read_text(encoding="utf-8") == "value = defined_var\n"

    async def test_external_workspace_edit_applies_without_snapshot_service(
        self, lsp_edit_tool, mock_lsp_manager, project_path
    ):
        outside_dir = project_path.parent / "outside"
        outside_dir.mkdir()
        outside = outside_dir / "external.py"
        outside.write_text("old = 1\n", encoding="utf-8")
        mock_lsp_manager._resolve_position.return_value = (0, 0)
        mock_lsp_manager.get_rename.return_value = {
            "changes": {
                outside.as_uri(): [
                    {
                        "range": {
                            "start": {"line": 0, "character": 0},
                            "end": {"line": 0, "character": 3},
                        },
                        "newText": "new",
                    }
                ]
            }
        }

        result = await lsp_edit_tool.lsp_edit(
            operation="rename",
            path=str(outside),
            line=1,
            symbol="old",
            new_name="new",
            apply=True,
        )

        assert result.startswith("Applied LSP edit `rename`.")
        assert outside.read_text(encoding="utf-8") == "new = 1\n"

    async def test_mixed_internal_external_workspace_edit_applies_every_change(
        self, lsp_edit_tool, mock_lsp_manager, project_path
    ):
        internal = project_path / "internal.py"
        outside_dir = project_path.parent / "outside"
        outside_dir.mkdir()
        external = outside_dir / "external.py"
        internal.write_text("old = 1\n", encoding="utf-8")
        external.write_text("old = 2\n", encoding="utf-8")
        replacement = [
            {
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 0, "character": 3},
                },
                "newText": "new",
            }
        ]
        mock_lsp_manager._resolve_position.return_value = (0, 0)
        mock_lsp_manager.get_rename.return_value = {
            "changes": {
                internal.as_uri(): replacement,
                external.as_uri(): replacement,
            }
        }

        result = await lsp_edit_tool.lsp_edit(
            operation="rename",
            path="internal.py",
            line=1,
            symbol="old",
            new_name="new",
            apply=True,
        )

        assert result.startswith("Applied LSP edit `rename`.")
        assert internal.read_text(encoding="utf-8") == "new = 1\n"
        assert external.read_text(encoding="utf-8") == "new = 2\n"


# ---------------------------------------------------------------------------
# T4: capabilities operation (no-path branch + valid JSON — F12/F13)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestLspCapabilities:
    async def test_capabilities_no_path_summarizes_sessions(self, lsp_tool, mock_lsp_manager):
        """T4: ``capabilities`` with no path lists each session's providers."""
        client = MagicMock()
        client.status = "initialized"
        client.server_capabilities = {
            "definitionProvider": True,
            "hoverProvider": {},
            "textDocumentSync": 1,
        }
        mock_lsp_manager._sessions = {"python": client}

        result = await lsp_tool.lsp(operation="capabilities")

        assert result.startswith("## LSP Capabilities")
        assert "**python** (initialized):" in result
        # Only *Provider keys are listed; textDocumentSync is excluded.
        assert "definitionProvider" in result
        assert "hoverProvider" in result
        assert "textDocumentSync" not in result

    async def test_capabilities_no_path_empty_sessions(self, lsp_tool, mock_lsp_manager):
        """T4: with no active sessions, capabilities reports none."""
        mock_lsp_manager._sessions = {}

        result = await lsp_tool.lsp(operation="capabilities")

        assert result.startswith("## LSP Capabilities")

    async def test_capabilities_with_path_emits_valid_json(self, lsp_tool, mock_lsp_manager):
        """F13: the per-path capabilities block contains valid JSON (not Python repr)."""
        mock_lsp_manager.get_capabilities.return_value = {
            "definitionProvider": True,
            "hoverProvider": {"contentFormat": ["markdown"]},
        }

        result = await lsp_tool.lsp(operation="capabilities", path="foo.py")

        assert "Server capabilities for foo.py:" in result
        # Extract the JSON body from the fenced block and confirm it parses.
        assert "```json" in result
        body = result.split("```json\n", 1)[1].rsplit("\n```", 1)[0]
        parsed = json.loads(body)
        assert parsed["definitionProvider"] is True
        assert parsed["hoverProvider"] == {"contentFormat": ["markdown"]}
