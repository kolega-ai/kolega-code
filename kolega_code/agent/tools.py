import asyncio
import inspect
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Union

from .common import LogMixin
from kolega_code.config import AgentConfig, EditProtocol
from kolega_code.llm.models import ImageBlock, ToolDefinition
from kolega_code.memory import ProjectMemoryManager
from kolega_code.tools import Tool, ToolRegistry, tool_definition_from_callable
from kolega_code.services.file_system import FileSystem, LocalFileSystem
from kolega_code.services.base import TerminalManager, BrowserManager
from kolega_code.services.terminal import LocalTerminalManager
from kolega_code.services.browser import PlaywrightBrowserManager
from .tool_backend.agent_tool import AgentTool
from .tool_backend.browser_tool import BROWSER_TOOL_SCHEMAS, BrowserTool
from .tool_backend.edit_tool import EditTool
from .tool_backend.codex_patch import CODEX_APPLY_PATCH_GRAMMAR
from .tool_backend.glob_tool import GlobTool
from .tool_backend.hashline_v2 import format_hash_lines, format_line_tag
from .tool_backend.list_directory_tool import ListDirectoryTool
from .tool_backend.memory_tool import MemoryTool
from .tool_backend.read_file_tool import ReadFileTool
from .tool_backend.read_image_tool import ReadImageTool
from .tool_backend.search_codebase_tool import SearchCodebaseTool
from .tool_backend.snapshot_tool import SnapshotTool
from .tool_backend.web_fetch_tool import WebFetchTool
from .tool_backend.web_search_tool import WebSearchTool
from .tool_backend.terminal_tool import TerminalTool
from .tool_backend.think_hard_tool import ThinkHardTool
from .tool_backend.workflow_tool import RUN_WORKFLOW_INPUT_SCHEMA, WorkflowTool
from .edit_protocols import EDIT_HANDLER_NAMES, edit_protocol_spec

# Import additional tools for consolidated functionality
from .tool_backend.build_tool import BuildTool
from .tool_backend.lsp_tool import LspEditTool, LspTool
from kolega_code.services.lsp import LspManager
from kolega_code.services.snapshots import SnapshotService

# Explicit input schema for the generic ``lsp`` tool.  The ``operation`` parameter
# is an enum that signature introspection cannot express.
_LSP_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "operation": {
            "type": "string",
            "enum": [
                "diagnostics",
                "definition",
                "type_definition",
                "implementation",
                "references",
                "hover",
                "call_hierarchy",
                "code_actions",
                "document_symbols",
                "workspace_symbols",
                "status",
                "capabilities",
                "reload",
            ],
            "description": (
                "The LSP operation to perform. Position operations (definition, "
                "type_definition, implementation, references, hover, call_hierarchy, "
                "code_actions) require path, "
                "line, and symbol. diagnostics and document_symbols require path. "
                "workspace_symbols requires query. status, capabilities, and reload "
                "need no additional args."
            ),
        },
        "path": {
            "type": "string",
            "description": "File path (relative to project root preferred). Required for most operations.",
        },
        "line": {
            "type": "integer",
            "description": "1-based line number for position operations.",
        },
        "symbol": {
            "type": "string",
            "description": "Symbol name to resolve on the line. Supports 'name#N' for the Nth occurrence.",
        },
        "query": {
            "type": "string",
            "description": "Search query for workspace_symbols.",
        },
        "end_line": {
            "type": "integer",
            "description": "Optional 1-based end line for code_actions.",
        },
        "kind": {
            "type": "string",
            "description": "Optional code action kind filter, such as quickfix or refactor.",
        },
        "timeout": {
            "type": "number",
            "description": "Per-call timeout in seconds (default: 30).",
        },
    },
    "required": ["operation"],
}

_LSP_EDIT_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "operation": {
            "type": "string",
            "enum": ["rename", "rename_file", "format_document", "format_range", "apply_code_action"],
            "description": (
                "Mutating LSP operation. rename requires path, line, symbol, and new_name. "
                "rename_file requires path and new_path. format_document requires path. "
                "format_range requires path and line, with optional end_line. "
                "apply_code_action requires path, line, symbol, and action_id or query."
            ),
        },
        "path": {
            "type": "string",
            "description": "File path (relative to project root preferred).",
        },
        "line": {
            "type": "integer",
            "description": "1-based line number for position/range operations.",
        },
        "symbol": {
            "type": "string",
            "description": "Symbol name to resolve on the line. Supports 'name#N' for the Nth occurrence.",
        },
        "new_name": {
            "type": "string",
            "description": "New symbol name for rename.",
        },
        "new_path": {
            "type": "string",
            "description": "Destination path for rename_file.",
        },
        "query": {
            "type": "string",
            "description": "Title substring or numeric index for apply_code_action when action_id is not provided.",
        },
        "action_id": {
            "type": "string",
            "description": "Stable action_id listed by lsp code_actions.",
        },
        "end_line": {
            "type": "integer",
            "description": "Optional 1-based end line for format_range and apply_code_action.",
        },
        "kind": {
            "type": "string",
            "description": "Optional code action kind filter, such as quickfix or refactor.",
        },
        "apply": {
            "type": "boolean",
            "description": "Apply the edit when true; preview only when false. Defaults to true.",
        },
        "timeout": {
            "type": "number",
            "description": "Per-call timeout in seconds (default: 30).",
        },
    },
    "required": ["operation"],
}

_SNAPSHOT_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["list", "show", "create", "restore"],
            "description": "Snapshot operation. Use restore with snapshot_id='latest' to undo the latest snapshot.",
        },
        "snapshot_id": {
            "type": "string",
            "description": "Snapshot id for show/restore. Use 'latest' to restore the newest snapshot.",
        },
        "paths": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Project-relative paths for action=create.",
        },
        "force": {
            "type": "boolean",
            "description": "Restore even when tracked files changed after the snapshot.",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of snapshots to list.",
        },
    },
}

_RESOLVE_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action_id": {
            "type": "string",
            "description": "Pending action id returned by a preview-only tool.",
        },
        "decision": {
            "type": "string",
            "enum": ["apply", "discard"],
            "description": "Apply or discard the pending preview action.",
        },
        "force": {
            "type": "boolean",
            "description": "Apply even if source hashes no longer match.",
        },
    },
    "required": ["action_id", "decision"],
}

_HASHLINE_REPLACE_CONTENT_SCHEMA: dict[str, Any] = {
    "anyOf": [
        {"type": "string"},
        {"type": "array", "items": {"type": "string"}},
        {"type": "null"},
    ],
    "description": (
        "Replacement file text as one string, an array of complete lines, or null to delete the line(s). "
        "Never include a display-only LINE#ID: prefix in this content."
    ),
}
_HASHLINE_INSERT_CONTENT_SCHEMA: dict[str, Any] = {
    "anyOf": [
        {"type": "string", "minLength": 1},
        {"type": "array", "items": {"type": "string"}, "minItems": 1},
    ],
    "description": (
        "Non-empty inserted file text as one string or an array of complete lines. "
        "Never include a display-only LINE#ID: prefix in this content."
    ),
}


def _hashline_operation_schema(
    op: str,
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"op": {"type": "string", "enum": [op]}, **properties},
        "required": ["op", *required],
        "additionalProperties": False,
    }


_HASHLINE_V2_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Project-relative path to edit."},
        "edits": {
            "type": "array",
            "description": (
                "All operations for this file, validated against one pre-edit snapshot. In displayed "
                "LINE#ID:CONTENT rows, pass LINE#ID to anchor fields and only CONTENT to content fields."
            ),
            "items": {
                "anyOf": [
                    _hashline_operation_schema(
                        "set",
                        {
                            "tag": {"type": "string", "description": "Target LINE#ID."},
                            "content": _HASHLINE_REPLACE_CONTENT_SCHEMA,
                        },
                        ["tag", "content"],
                    ),
                    _hashline_operation_schema(
                        "replace",
                        {
                            "first": {"type": "string", "description": "First LINE#ID, inclusive."},
                            "last": {"type": "string", "description": "Last LINE#ID, inclusive."},
                            "content": _HASHLINE_REPLACE_CONTENT_SCHEMA,
                        },
                        ["first", "last", "content"],
                    ),
                    _hashline_operation_schema(
                        "append",
                        {
                            "after": {"type": "string", "description": "Optional LINE#ID to insert after."},
                            "content": _HASHLINE_INSERT_CONTENT_SCHEMA,
                        },
                        ["content"],
                    ),
                    _hashline_operation_schema(
                        "prepend",
                        {
                            "before": {"type": "string", "description": "Optional LINE#ID to insert before."},
                            "content": _HASHLINE_INSERT_CONTENT_SCHEMA,
                        },
                        ["content"],
                    ),
                    _hashline_operation_schema(
                        "insert",
                        {
                            "after": {"type": "string", "description": "Optional preceding LINE#ID."},
                            "before": {"type": "string", "description": "Optional following LINE#ID."},
                            "content": _HASHLINE_INSERT_CONTENT_SCHEMA,
                        },
                        ["content"],
                    ),
                ]
            },
        },
        "delete": {"type": "boolean", "description": "Delete path; requires edits=[] and no rename."},
        "rename": {"type": "string", "description": "Move the edited result to this project-relative path."},
    },
    "required": ["path", "edits"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class ToolExtension:
    """Host-provided tool callbacks and named groups."""

    name: str
    tools: dict[str, Callable[..., Any]]
    tool_groups: dict[str, List[str]] = field(default_factory=dict)
    # Optional explicit JSON schemas keyed by tool name. When provided, the
    # schema is used verbatim instead of introspecting the callable signature,
    # allowing nested input shapes the introspector cannot express.
    tool_schemas: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Optional cleanup hook. May be sync or async; ToolCollection.cleanup awaits
    # it when needed.
    cleanup: Optional[Callable[[], Any]] = None
    # Whether this extension is inherited by sub-agents. Interactive or
    # session-shared host tools (task list, planning questions) belong to the
    # single top-level agent only; leaving them on for parallel sub-agents lets
    # them clobber shared state. Default True preserves inheritance.
    propagate_to_sub_agents: bool = True


class ToolCollectionConfig:
    """Configuration class for customizing tool availability per agent type."""

    def __init__(
        self,
        read_only: bool = False,
        browser_only: bool = False,
        include_agent_dispatch_tools: bool = False,
        include_memory_tools: bool = False,
        tool_exclusions: Optional[List[str]] = None,
        custom_tool_groups: Optional[List[str]] = None,
        enabled_tool_groups: Optional[List[str]] = None,
        restrict_to_tool_groups: bool = False,
        allowed_tools: Optional[List[str]] = None,
        memory_write_access: bool = False,
    ):
        """
        Initialize tool collection configuration.

        Args:
            read_only: Whether to restrict to read-only tools
            browser_only: Whether to only include browser tools
            include_agent_dispatch_tools: Whether to include agent dispatch tools (investigation, browser, coding)
            include_memory_tools: Whether to include memory management tools
            memory_write_access: Whether mutating memory tools (write/delete) may be exposed to this agent;
                subagent scope still strips them regardless.
            tool_exclusions: List of method names to exclude from tool list
            custom_tool_groups: Additional custom tool groups to include
            enabled_tool_groups: Additional custom tool groups to include
            restrict_to_tool_groups: If True, ONLY include tools from specified groups, excluding all other core tools
            allowed_tools: Optional exact allowlist applied after hard runtime gates. None inherits normal behavior;
                an empty list exposes no tools.
        """
        self.read_only = read_only
        self.browser_only = browser_only
        self.include_agent_dispatch_tools = include_agent_dispatch_tools
        self.include_memory_tools = include_memory_tools
        self.memory_write_access = memory_write_access
        self.tool_exclusions = tool_exclusions or []
        self.custom_tool_groups = list(dict.fromkeys((custom_tool_groups or []) + (enabled_tool_groups or [])))
        self.restrict_to_tool_groups = restrict_to_tool_groups
        self.allowed_tools = None if allowed_tools is None else list(dict.fromkeys(allowed_tools))


class ToolCollection(LogMixin):
    """
    A collection of tools for interacting with the project workspace.

    Provides utilities for file operations and workspace management.
    """

    read_only_tools = [
        "list_directory",
        "read_entire_file",
        "read_file_section",
        "read_memory",
        "list_memory",
        "search_codebase",
        "find_files_by_pattern",
        "think_hard",
        "web_fetch",
        "web_search",
        "sleep",
        "read_image",
        "lsp",
    ]

    browser_tools = [
        "browser_navigate",
        "browser_navigate_back",
        "browser_snapshot",
        "browser_find",
        "browser_wait_for",
        "browser_resize",
        "browser_click",
        "browser_type",
        "browser_fill_form",
        "browser_select_option",
        "browser_hover",
        "browser_drag",
        "browser_drop",
        "browser_press_key",
        "browser_tabs",
        "browser_handle_dialog",
        "browser_file_upload",
        "browser_console_messages",
        "browser_network_requests",
        "browser_network_request",
        "browser_take_screenshot",
        "browser_evaluate",
        "browser_close",
    ]

    # Agent dispatch tools group - includes all agent dispatch functionality
    agent_dispatch_tools = [
        "dispatch_investigation_agent",
        "dispatch_browser_agent",
        "dispatch_coding_agent",
        "dispatch_general_agent",
        "dispatch_custom_agent",
    ]

    # Legacy name for backward compatibility
    investigation_agent_tools = agent_dispatch_tools

    # CoderAgent specific dispatch tools
    coder_agent_tools = [
        "dispatch_investigation_agent",
        "dispatch_browser_agent",
        "dispatch_general_agent",
        "dispatch_custom_agent",
    ]

    custom_agent_tools = [
        "dispatch_custom_agent",
    ]

    # Memory tools group
    memory_tools = [
        "read_memory",
        "list_memory",
        "write_memory",
        "delete_memory",
    ]

    # gigacode workflow orchestration. Gated in _should_include_tool: only the
    # top-level (non-sub) agent gets it, and only when gigacode is enabled.
    orchestration_tools = [
        "run_workflow",
    ]

    # Shell execution + session management. Exposed to the planning and
    # investigation agents (via custom_tool_groups) so they can run investigative
    # commands even while read_only=True. Not read-only, so deliberately excluded
    # from the parallel-safe set in _build_tool.
    command_tools = [
        "exec_command",
        "write_stdin",
        "kill_command",
        "list_sessions",
    ]

    def __init__(
        self,
        project_path: Union[str, Path],
        workspace_id: str,
        thread_id: str,
        connection_manager,
        config: AgentConfig,
        caller,
        tool_config: Optional[ToolCollectionConfig] = None,
        read_only: bool = False,  # Keep for backward compatibility
        browser_only: bool = False,  # Keep for backward compatibility
        filesystem: Optional[FileSystem] = None,
        terminal_manager: Optional[TerminalManager] = None,
        browser_manager: Optional[BrowserManager] = None,
        langfuse_client=None,
        tool_extensions: Optional[List[ToolExtension]] = None,
    ) -> None:
        """
        Initialize a new ToolCollection instance.

        Args:
            project_path: File system path to the project root directory
            workspace_id: Unique identifier for the workspace
            thread_id: Unique identifier for the thread
            connection_manager: Connection manager for agent communication
            config: Agent configuration
            caller: The calling agent instance
            tool_config: Configuration for which tools to include (takes precedence over legacy flags)
            read_only: Whether tools should be read-only (legacy, use tool_config instead)
            browser_only: Whether to only include browser tools (legacy, use tool_config instead)
            filesystem: Optional filesystem implementation. If None, creates LocalFileSystem with project_path as root
            terminal_manager: Optional terminal manager implementation. If None, creates LocalTerminalManager
            browser_manager: Optional browser manager implementation. If None, creates PlaywrightBrowserManager
            tool_extensions: Host-provided tools and groups
        """
        # Handle backward compatibility - create tool_config from legacy parameters if not provided
        if tool_config is None:
            tool_config = ToolCollectionConfig(read_only=read_only, browser_only=browser_only)

        self.tool_config = tool_config
        self.workspace_id = workspace_id
        self.thread_id = thread_id

        # Convert string path to Path object if needed
        self.project_path = Path(project_path) if isinstance(project_path, str) else project_path

        # Create filesystem instance if not provided
        if filesystem is None:
            self.filesystem = LocalFileSystem(root_path=self.project_path)
        else:
            self.filesystem = filesystem

        # Create terminal manager instance if not provided
        if terminal_manager is None:
            self.terminal_manager = LocalTerminalManager(workspace_id, thread_id, connection_manager)
        else:
            self.terminal_manager = terminal_manager

        # Create browser manager instance if not provided
        if browser_manager is None:
            self.browser_manager = PlaywrightBrowserManager()
        else:
            self.browser_manager = browser_manager

        # Validate the filesystem root. Local filesystems check the directory
        # eagerly; sandbox filesystems are provisioned by their manager and no-op.
        self.filesystem.validate_root()

        self.connection_manager = connection_manager
        self.config = config
        self.caller = caller
        caller_protocol = getattr(caller, "edit_protocol", None)
        if isinstance(caller_protocol, EditProtocol):
            self.edit_protocol = caller_protocol
        else:
            model_config = getattr(caller, "primary_model_config", None)
            if not hasattr(model_config, "provider") or not hasattr(model_config, "model"):
                model_config = config.long_context_config
            resolved_edit_protocol = config.resolve_edit_protocol(model_config)
            self.edit_protocol = (
                resolved_edit_protocol
                if isinstance(resolved_edit_protocol, EditProtocol)
                else EditProtocol.SEARCH_REPLACE
            )
        self.langfuse_client = langfuse_client
        self.tool_extensions = tool_extensions or []
        self.extension_callbacks = {}
        self.extension_schemas = {}
        self._extension_group_names = set()

        # Set legacy attributes for backward compatibility
        self.read_only = tool_config.read_only
        self.browser_only = tool_config.browser_only

        # Build tool exclusions list from config. These are internal
        # management/logging APIs, not model-facing tools.
        self.tool_exclusions = [
            "execute_terminal_command",
            "get_tool_list",
            "registry",
            "has_tool",
            "call",
            "cleanup",
            "initialize",
            "log_error",
            "log_warning",
            "log_info",
        ]
        # Backend bindings, not these compatibility methods, own model-facing
        # memory schemas and capability registration.
        self._memory_compatibility_methods = {
            "read_memory",
            "list_memory",
            "write_memory",
            "delete_memory",
        }
        self.tool_exclusions.extend(tool_config.tool_exclusions)

        # Initialize tool backends
        self._initialize_tools()
        self._register_tool_extensions()

    def _register_tool_extensions(self):
        """Bind host-provided extension callbacks onto this collection."""
        for extension in self.tool_extensions:
            for tool_name, callback in extension.tools.items():
                if hasattr(self, tool_name):
                    raise ValueError(f"Tool extension '{extension.name}' conflicts with existing tool '{tool_name}'")
                setattr(self, tool_name, callback)
                self.extension_callbacks[tool_name] = callback

            for tool_name, schema in extension.tool_schemas.items():
                self.extension_schemas[tool_name] = schema

            for group_name, tool_names in extension.tool_groups.items():
                existing_group = list(getattr(self, group_name, []))
                merged_group = list(dict.fromkeys(existing_group + list(tool_names)))
                setattr(self, group_name, merged_group)
                self._extension_group_names.add(group_name)

    def _initialize_tools(self):
        """Initialize all tool backends based on configuration."""
        # Core tool backends (always available)

        # LSP manager (shared across EditTool and LspTool)
        lsp_config = getattr(self.config, "lsp", None)
        from kolega_code.services.lsp import LspConfig as _LspConfig

        if isinstance(lsp_config, _LspConfig) and lsp_config.enabled:
            self.lsp_manager = LspManager(
                self.project_path,
                config=lsp_config,
                trusted=getattr(self.config, "lsp_project_trusted", False),
            )
        else:
            self.lsp_manager = None

        snapshot_session_id = str(getattr(self.caller, "session_id", None) or self.thread_id)
        self.snapshot_service = SnapshotService(
            self.project_path,
            self.workspace_id,
            self.thread_id,
            snapshot_session_id,
            self.filesystem,
        )

        self.think_hard_tool = ThinkHardTool(
            self.project_path,
            self.workspace_id,
            self.thread_id,
            self.connection_manager,
            self.config,
            self.caller,
            self.filesystem,
        )
        self.edit_tool = EditTool(
            self.project_path,
            self.workspace_id,
            self.thread_id,
            self.connection_manager,
            self.config,
            self.caller,
            self.filesystem,
            lsp_manager=self.lsp_manager,
            snapshot_service=self.snapshot_service,
        )
        self.snapshot_tool = SnapshotTool(
            self.project_path,
            self.workspace_id,
            self.thread_id,
            self.connection_manager,
            self.config,
            self.caller,
            self.filesystem,
            snapshot_service=self.snapshot_service,
        )
        self.list_directory_tool = ListDirectoryTool(
            self.project_path,
            self.workspace_id,
            self.thread_id,
            self.connection_manager,
            self.config,
            self.caller,
            self.filesystem,
        )
        self.terminal_tool = TerminalTool(
            self.project_path,
            self.workspace_id,
            self.thread_id,
            self.connection_manager,
            self.config,
            self.caller,
            self.filesystem,
            terminal_manager=self.terminal_manager,
        )
        memory_manager = getattr(self.caller, "memory_manager", None)
        if not isinstance(memory_manager, ProjectMemoryManager):
            memory_manager = None
        self.memory_tool = MemoryTool(memory_manager, self.caller)
        self.search_codebase_tool = SearchCodebaseTool(
            self.project_path,
            self.workspace_id,
            self.thread_id,
            self.connection_manager,
            self.config,
            self.caller,
            self.filesystem,
        )
        self.web_fetch_tool = WebFetchTool(
            self.project_path,
            self.workspace_id,
            self.thread_id,
            self.connection_manager,
            self.config,
            self.caller,
            self.filesystem,
        )
        self.web_search_tool = WebSearchTool(
            self.project_path,
            self.workspace_id,
            self.thread_id,
            self.connection_manager,
            self.config,
            self.caller,
            self.filesystem,
        )
        self.glob_tool = GlobTool(
            self.project_path,
            self.workspace_id,
            self.thread_id,
            self.connection_manager,
            self.config,
            self.caller,
            self.filesystem,
        )
        self.read_file_tool = ReadFileTool(
            self.project_path,
            self.workspace_id,
            self.thread_id,
            self.connection_manager,
            self.config,
            self.caller,
            self.filesystem,
        )
        self.read_image_tool = ReadImageTool(
            self.project_path,
            self.workspace_id,
            self.thread_id,
            self.connection_manager,
            self.config,
            self.caller,
            self.filesystem,
        )
        self.agent_tool = AgentTool(
            self.project_path,
            self.workspace_id,
            self.thread_id,
            self.connection_manager,
            self.config,
            self.caller,
            self.filesystem,
            terminal_manager=self.terminal_manager,
            browser_manager=self.browser_manager,
            langfuse_client=self.langfuse_client,
            memory_manager=memory_manager,
        )
        self.workflow_tool = WorkflowTool(
            self.project_path,
            self.workspace_id,
            self.thread_id,
            self.connection_manager,
            self.config,
            self.caller,
            self.filesystem,
            terminal_manager=self.terminal_manager,
            browser_manager=self.browser_manager,
            langfuse_client=self.langfuse_client,
        )
        # run_workflow's `args` is free-form JSON, which the signature introspector
        # cannot express, so register its explicit input schema.
        self.extension_schemas["run_workflow"] = RUN_WORKFLOW_INPUT_SCHEMA
        # The `lsp` tool's `operation` is an enum that signature introspection
        # can't express, so register an explicit input schema.
        self.extension_schemas["lsp"] = _LSP_INPUT_SCHEMA
        self.extension_schemas["lsp_edit"] = _LSP_EDIT_INPUT_SCHEMA
        self.extension_schemas["snapshot"] = _SNAPSHOT_INPUT_SCHEMA
        self.extension_schemas["resolve"] = _RESOLVE_INPUT_SCHEMA
        self.extension_schemas.update(BROWSER_TOOL_SCHEMAS)
        self.browser_tool = BrowserTool(
            self.project_path,
            self.workspace_id,
            self.thread_id,
            self.connection_manager,
            self.config,
            self.caller,
            self.filesystem,
            browser_manager=self.browser_manager,
        )

        # Build tool
        self.build_tool = BuildTool(
            self.project_path,
            self.workspace_id,
            self.thread_id,
            self.connection_manager,
            self.config,
            self.caller,
            self.filesystem,
            terminal_manager=self.terminal_manager,
        )

        # LSP tool
        self.lsp_tool = LspTool(
            self.project_path,
            self.workspace_id,
            self.thread_id,
            self.connection_manager,
            self.config,
            self.caller,
            self.filesystem,
            lsp_manager=self.lsp_manager,
        )
        self.lsp_edit_tool = LspEditTool(
            self.project_path,
            self.workspace_id,
            self.thread_id,
            self.connection_manager,
            self.config,
            self.caller,
            self.filesystem,
            lsp_manager=self.lsp_manager,
            snapshot_service=self.snapshot_service,
        )

    async def browser_navigate(self, url: str) -> str:
        """Navigate the current browser tab to a URL, starting a session when needed.

        Args:
            url: HTTP or HTTPS URL to navigate to.
        """
        return await self.browser_tool.browser_navigate(url)

    async def browser_navigate_back(self) -> str:
        """Go back to the previous page and return the updated page snapshot."""
        return await self.browser_tool.browser_navigate_back()

    async def browser_snapshot(self, target: Optional[str] = None, depth: Optional[int] = None) -> str:
        """Capture the current page's accessibility snapshot.

        Prefer this over screenshots when deciding what to interact with. Interactive
        nodes include stable refs such as e12 that can be passed to action tools.

        Args:
            target: Optional snapshot ref or unique selector for a subtree.
            depth: Optional maximum accessibility-tree depth.
        """
        return await self.browser_tool.browser_snapshot(target, depth)

    async def browser_find(self, text: Optional[str] = None, regex: Optional[str] = None) -> str:
        """Find text or a regular expression in the accessibility snapshot.

        Provide exactly one of text or regex. This is cheaper than requesting a
        full snapshot when locating a specific element.

        Args:
            text: Case-insensitive text to find.
            regex: Regular expression to find.
        """
        return await self.browser_tool.browser_find(text, regex)

    async def browser_wait_for(
        self, time: Optional[float] = None, text: Optional[str] = None, text_gone: Optional[str] = None
    ) -> str:
        """Wait for time to pass, text to appear, or text to disappear.

        Args:
            time: Seconds to wait, capped at 30.
            text: Text to wait for until visible.
            text_gone: Text to wait for until hidden.
        """
        return await self.browser_tool.browser_wait_for(time, text, text_gone)

    async def browser_resize(self, width: int, height: int) -> str:
        """Resize the current browser viewport.

        Args:
            width: Viewport width in CSS pixels.
            height: Viewport height in CSS pixels.
        """
        return await self.browser_tool.browser_resize(width, height)

    async def browser_click(
        self,
        target: str,
        double_click: bool = False,
        button: str = "left",
        modifiers: Optional[list[str]] = None,
    ) -> str:
        """Click an element identified by a snapshot ref or unique selector.

        Args:
            target: Exact snapshot ref or unique selector.
            double_click: Perform a double click.
            button: Mouse button: left, right, or middle.
            modifiers: Keyboard modifiers held during the click.
        """
        return await self.browser_tool.browser_click(target, double_click, button, modifiers)

    async def browser_type(self, target: str, text: str, submit: bool = False, slowly: bool = False) -> str:
        """Enter text into an editable element.

        Args:
            target: Exact snapshot ref or unique selector.
            text: Text to enter.
            submit: Press Enter after entering text.
            slowly: Type character by character instead of filling.
        """
        return await self.browser_tool.browser_type(target, text, submit, slowly)

    async def browser_fill_form(self, fields: list[dict[str, Any]]) -> str:
        """Fill several textbox, checkbox, radio, combobox, or slider fields.

        Args:
            fields: Structured field descriptions with name, target, type, and value.
        """
        return await self.browser_tool.browser_fill_form(fields)

    async def browser_select_option(self, target: str, values: list[str]) -> str:
        """Select one or more values in a dropdown.

        Args:
            target: Exact snapshot ref or unique selector.
            values: Option values to select.
        """
        return await self.browser_tool.browser_select_option(target, values)

    async def browser_hover(self, target: str) -> str:
        """Hover over an element and return the updated snapshot.

        Args:
            target: Exact snapshot ref or unique selector.
        """
        return await self.browser_tool.browser_hover(target)

    async def browser_drag(self, start_target: str, end_target: str) -> str:
        """Drag one page element to another.

        Args:
            start_target: Source snapshot ref or unique selector.
            end_target: Destination snapshot ref or unique selector.
        """
        return await self.browser_tool.browser_drag(start_target, end_target)

    async def browser_drop(
        self, target: str, paths: Optional[list[str]] = None, data: Optional[dict[str, str]] = None
    ) -> str:
        """Drop workspace files or MIME-typed string data onto an element.

        Args:
            target: Destination snapshot ref or unique selector.
            paths: Workspace file paths to drop.
            data: MIME type to string value mapping.
        """
        return await self.browser_tool.browser_drop(target, paths, data)

    async def browser_press_key(self, key: str) -> str:
        """Press a keyboard key in the current tab.

        Args:
            key: Key name or character, such as ArrowLeft or a.
        """
        return await self.browser_tool.browser_press_key(key)

    async def browser_tabs(self, action: str, index: Optional[int] = None, url: Optional[str] = None) -> str:
        """List, create, close, or select browser tabs.

        Args:
            action: One of list, new, close, or select.
            index: Tab index for close or select.
            url: Optional URL for a new tab.
        """
        return await self.browser_tool.browser_tabs(action, index, url)

    async def browser_handle_dialog(self, accept: bool, prompt_text: Optional[str] = None) -> str:
        """Accept or dismiss the currently waiting JavaScript dialog.

        Args:
            accept: Accept rather than dismiss the dialog.
            prompt_text: Text to submit to a prompt dialog.
        """
        return await self.browser_tool.browser_handle_dialog(accept, prompt_text)

    async def browser_file_upload(self, paths: list[str]) -> str:
        """Upload workspace files through the currently waiting file chooser.

        Args:
            paths: Workspace file paths to upload. Use an empty list to cancel.
        """
        return await self.browser_tool.browser_file_upload(paths)

    async def browser_console_messages(self, level: str = "info", all_messages: bool = False) -> str:
        """Return console messages for the current tab.

        Args:
            level: Minimum severity: error, warning, info, or debug.
            all_messages: Include the full session instead of only messages since navigation.
        """
        return await self.browser_tool.browser_console_messages(level, all_messages)

    async def browser_network_requests(self, include_static: bool = False, filter_pattern: Optional[str] = None) -> str:
        """List network requests made by the current tab since navigation.

        Args:
            include_static: Include images, fonts, scripts, and styles.
            filter_pattern: Optional URL regular expression.
        """
        return await self.browser_tool.browser_network_requests(include_static, filter_pattern)

    async def browser_network_request(self, index: int, part: Optional[str] = None) -> str:
        """Return headers or body details for one indexed network request.

        Args:
            index: 1-based index from browser_network_requests.
            part: Optional request_headers, request_body, response_headers, or response_body.
        """
        return await self.browser_tool.browser_network_request(index, part)

    async def browser_take_screenshot(
        self,
        target: Optional[str] = None,
        image_type: str = "png",
        full_page: bool = False,
        scale: str = "css",
    ) -> List[ImageBlock]:
        """Capture a visual screenshot of the page or one element.

        Use browser_snapshot, not the screenshot, to choose interaction targets.

        Args:
            target: Optional snapshot ref or unique selector.
            image_type: png or jpeg.
            full_page: Capture the full scrollable page.
            scale: css or device pixel scale.
        """
        result = await self.browser_tool.browser_take_screenshot(target, image_type, full_page, scale)
        return [ImageBlock(image_type="base64", media_type=result["media_type"], data=result["image"])]

    async def read_image(self, path: str) -> List[Any]:
        """Read an image file from the project directory so you can see it.

        Use when the user references a screenshot, diagram, mockup, or other
        visual asset and text-based inspection is insufficient.

        Args:
            path: Path relative to the project root, or an allowed absolute path.
        """
        return await self.read_image_tool.read_image(path)

    async def browser_evaluate(self, function: str, target: Optional[str] = None) -> str:
        """Evaluate JavaScript in the page or against one target element.

        Args:
            function: JavaScript function to evaluate.
            target: Optional snapshot ref or unique selector passed as the function argument.
        """
        return await self.browser_tool.browser_evaluate(function, target)

    async def browser_close(self) -> str:
        """Close the current browser session and release its resources."""
        return await self.browser_tool.browser_close()

    async def build_backend(self) -> str:
        """
        Build the backend defined by the project manifest (.kolega-manifest.yaml).

        When to use this tool:
        - When you need to compile, bundle, or otherwise build the backend for the current workspace
        - When verifying that the backend build still succeeds after code changes

        Guidance:
        - Prefer this tool over manually running build commands in a terminal; it automatically selects the correct
          command from the manifest and works in both local and sandbox environments with standardized output

        Returns:
            Build output as markdown (combined stdout/stderr)
        """
        return await self.build_tool.build_backend()

    async def build_frontend(self) -> str:
        """
        Build the frontend defined by the project manifest (.kolega-manifest.yaml).

        When to use this tool:
        - When you need to compile, bundle, or otherwise build the frontend application
        - When you want a consistent build execution that adapts to local or sandbox contexts

        Guidance:
        - Prefer this tool over manually running build commands in a terminal; it reads the manifest to choose the
          correct command and standardizes execution and output across environments

        Returns:
            Build output as markdown (combined stdout/stderr)
        """
        return await self.build_tool.build_frontend()

    # Agent Dispatch Tools (available when include_agent_dispatch_tools is True)
    async def dispatch_investigation_agent(self, task: str) -> str:
        """
        Dispatch an investigation agent to perform a specific task with read-only access to the codebase.

        This tool launches a specialized agent that can analyze code, search for patterns, and investigate
        issues without modifying any files. The investigation agent has access to all read-only tools
        and will return a comprehensive report on its findings.

        When to use this tool:
        - When you need to perform complex searches across multiple files
        - When you need to analyze code patterns or understand how components interact
        - When you need to trace through code execution paths
        - When you need to gather information from multiple parts of the codebase

        Usage notes:
        1. Provide a detailed task description with specific questions or objectives for the agent
        2. The agent will work autonomously and return a single comprehensive report
        3. The agent cannot modify any files - it has read-only access to the codebase
        4. For best results, specify exactly what information you want the agent to find and include in its report
        5. The agent's report is not automatically shown to the user - you should summarize key findings

        IMPORTANT: The agent can only use these tools:
            - list_directory
            - read_entire_file
            - read_file_section
            - read_memory
            - search_codebase
            - find_files_by_pattern
            - think_hard
        If you need to do something that requires any other tool, you should call the tool directly.

        Args:
            task: A detailed description of the investigation task to perform

        Returns:
            A comprehensive report of the investigation findings
        """
        return await self.agent_tool.dispatch_investigation_agent(task)

    async def dispatch_browser_agent(self, task: str) -> str:
        """
        Dispatch a browser agent to perform web-based tasks and interactions.

        This tool launches a specialized agent that can navigate websites, interact with web elements,
        and extract information from web pages. The browser agent has access to all browser-related tools
        and will return a comprehensive report on its findings and actions.

        Use this ONLY when the user explicitly asks to browse, open, visit, or interact with a web page/URL,
        or explicitly requests a screenshot or web UI action. Do NOT use this for general research, docs lookup,
        or exploration unless the user clearly requests browsing.


        When to use this tool:
        - When you need to navigate and interact with websites
        - When you need to extract information from web pages
        - When you need to test web applications or interfaces
        - When you need to automate web-based workflows

        Usage notes:
        1. Provide a detailed task description with specific objectives for the browser agent
        2. The agent will work autonomously and return a single comprehensive report
        3. The agent can launch browsers, navigate pages, click elements, fill forms, and extract content
        4. For best results, specify exactly what information you want the agent to find or what actions to perform
        5. The agent's report is not automatically shown to the user - you should summarize key findings

        IMPORTANT: The browser agent specializes in these tools:
            - browser_navigate, browser_snapshot, and browser_find
            - browser_click, browser_type, browser_fill_form, and browser_select_option
            - browser_tabs, browser_wait_for, browser_handle_dialog, and browser_file_upload
            - browser_console_messages, browser_network_requests, and browser_take_screenshot
            - browser_close

        Args:
            task: A detailed description of the browser task to perform

        Returns:
            A comprehensive report of the browser agent's findings and actions
        """
        return await self.agent_tool.dispatch_browser_agent(task)

    async def dispatch_coding_agent(self, task: str) -> str:
        """
        Dispatch a coding agent for processing coding-related tasks with streaming output.

        Args:
            task: A detailed description of the coding task to perform

        Returns:
            A summary of the coding process outcome
        """
        return await self.agent_tool.dispatch_coding_agent(task)

    async def dispatch_general_agent(self, task: str) -> str:
        """
        Dispatch an autonomous general-purpose agent to complete a self-contained task.

        This tool launches a sub-agent with the full set of workspace tools (read, search,
        edit files, run commands). It works autonomously on the task you give it and returns
        a single final report. You will not see its intermediate steps, and you cannot send
        it follow-up messages, so the task description must contain everything it needs.

        PARALLEL EXECUTION: If you issue multiple dispatch_general_agent calls in a single
        response, the agents run CONCURRENTLY. Use this to fan out work that can proceed
        independently (e.g., "update module A's tests" and "update module B's tests").

        When to use this tool:
        - The work splits into independent subtasks that do not touch the same files
        - A subtask is large or noisy (broad searches, mechanical multi-file edits) and you
          only need the outcome, not every intermediate step
        - You want several independent investigations or changes done at once

        When NOT to use this tool:
        - Tasks that depend on each other's output or edit the same files - do those
          yourself sequentially, or dispatch them one at a time
        - Small tasks you can do directly with one or two tool calls
        - Anything requiring back-and-forth with the user

        Usage notes:
        1. Each task must be INDEPENDENT and SELF-CONTAINED: include the goal, relevant
           file paths, constraints, and exactly what the final report should contain.
        2. Never dispatch two parallel agents whose work could overlap on the same files.
        3. The agent cannot spawn further sub-agents.
        4. The agent's report is not automatically shown to the user - you should summarize
           the key results.

        Args:
            task: A detailed, self-contained description of the task to perform

        Returns:
            The agent's final report on the completed task
        """
        return await self.agent_tool.dispatch_general_agent(task)

    async def dispatch_custom_agent(self, agent: str, task: str) -> str:
        """Dispatch a named custom agent defined in project or user Markdown.

        Select an agent whose description matches a self-contained task. The agent
        runs in a fresh context, cannot spawn other agents, and returns one final
        report. Its tools can only be a subset of the tools available in this
        session. Multiple independent calls may run in parallel.

        Args:
            agent: Name of the custom agent to run.
            task: Detailed, self-contained task including relevant context and expected output.

        Returns:
            The custom agent's final report.
        """
        return await self.agent_tool.dispatch_custom_agent(agent, task)

    async def run_workflow(
        self,
        script: str = "",
        args: Any = None,
        token_budget: int = 0,
        script_path: str = "",
        resume_from_run_id: str = "",
    ) -> str:
        """Run a gigacode workflow: an authored Python script that orchestrates many
        sub-agents with deterministic control flow (parallel fan-out, pipelines,
        loop-until-dry, budget loops).

        The script's primitives are `agent()`, `parallel()`, `pipeline()`, `phase()`,
        and `log()`, plus the `args` and `budget` globals. See the gigacode authoring
        guide in your system prompt for the full API and patterns. Artifacts (script,
        full result and readable transcript; raw/per-agent debug artifacts are saved
        under the run directory but are not advertised by default) are written under the CLI state directory, and a run can be resumed with
        `resume_from_run_id`.

        Args:
            script: The Python orchestration script source (must define a top-level `meta` literal).
            args: Free-form JSON value exposed to the script as the global `args`.
            token_budget: Optional output-token ceiling for the whole run (0 = unbounded).
            script_path: Path to a script file on disk; takes precedence over `script`.
            resume_from_run_id: Resume a prior run, replaying cached agent() results for the
                unchanged prefix and running new/changed calls live.

        Returns:
            A compact artifact manifest: the runId, persisted scriptPath, token count,
            resultPath, and transcriptPath. The workflow result is written to
            resultPath rather than returned inline. Read resultPath for the workflow
            result, or transcriptPath for execution details. For normal workflow
            output, avoid reading individual sub-agent transcripts.
        """
        return await self.workflow_tool.run_workflow(
            script=script,
            args=args,
            token_budget=token_budget,
            script_path=script_path,
            resume_from_run_id=resume_from_run_id,
        )

    async def think_hard(self, problem_statement: str) -> str:
        """
        Uses Claude 3.7 Sonnet in extended thinking mode to analyze a problem deeply.

        This tool leverages Claude's extended thinking capabilities to perform in-depth
        analysis on complex problems. It sends the problem statement to the Claude API
        with specific parameters to enable extended thinking and returns the detailed response.

        Args:
            problem_statement: A clear statement of the problem to be analyzed, including ALL relevant details.

        Returns:
            The detailed analysis from Claude, including its extended thinking process
        """
        return await self.think_hard_tool.think_hard(problem_statement)

    async def sleep(self, seconds: float) -> str:
        """
        Pause execution for a specified number of seconds.

        This tool introduces a deliberate delay in execution, allowing time for external processes
        to complete, systems to stabilize, or operations to finish processing. It's particularly
        useful when working with asynchronous operations or waiting for long-running commands.

        When to use this tool:
        - After starting a long-running test suite and wanting to wait before checking results
        - When waiting for a development server or application to fully start up
        - After triggering a build process that needs time to complete
        - When waiting for file system operations to propagate (especially on networked drives)
        - After making configuration changes that need time to take effect
        - When working with rate-limited APIs and need to respect timing constraints

        Usage notes:
        1. Use this tool judiciously - unnecessary delays slow down overall task completion
        2. Consider checking process status first rather than using arbitrary wait times
        3. For very long operations (>5 minutes), consider breaking into smaller check intervals
        4. The tool accepts decimal values for sub-second precision (e.g., 0.5 for half a second)
        5. Maximum recommended sleep time is 300 seconds (5 minutes) to avoid excessive delays
        6. Prefer polling a running command with write_stdin (empty input) over sleeping to verify completion
        7. Consider using shorter initial sleeps and checking status rather than one long sleep

        Args:
            seconds: Number of seconds to sleep (must be positive, supports decimal values)

        Returns:
            A confirmation message indicating how long the execution was paused

        Raises:
            ValueError: If seconds is negative or exceeds the maximum allowed duration
        """
        if seconds <= 0:
            raise ValueError("Sleep duration must be positive")

        if seconds > 300:  # 5 minutes maximum
            raise ValueError("Sleep duration cannot exceed 300 seconds (5 minutes)")

        await asyncio.sleep(seconds)

        if seconds == 1:
            return f"✅ Paused execution for {seconds} second"
        else:
            return f"✅ Paused execution for {seconds} seconds"

    async def edit(self, path: str, block: str) -> str:
        """
        Edit a file using one search and replace block.

        The block should be formatted as follows:
        ```
        <<<<<<< SEARCH
        [original code to find]
        =======
        [new code to replace with]
        >>>>>>> REPLACE
        ```

        Before using this tool:

        1. Use the read_entire_file tool to understand the file's contents and context.

        To make a file edit, provide the following:
        1. The path to the file to modify.
        2. block: A single search and replace block.

        The tool replaces one uniquely matched occurrence. Matching is attempted in this order:
        1. Exact match.
        2. Per-line stripped match for indentation and trailing whitespace differences.
        3. Normalized line endings.
        4. Normalized smart quotes.

        CRITICAL REQUIREMENTS FOR USING THIS TOOL:

        1. UNIQUENESS: The old_string MUST uniquely identify the specific instance you want to change. This means:
        - Include AT LEAST 3-5 lines of context BEFORE the change point
        - Include AT LEAST 3-5 lines of context AFTER the change point
        - Include all whitespace, indentation, and surrounding code exactly as it appears in the file

        2. SINGLE INSTANCE: This tool can only change ONE instance at a time. If you need to change multiple instances:
        - Use multi_edit when all replacements are in the same file.
        - Each block must uniquely identify its specific instance using extensive context.

        3. VERIFICATION: Before using this tool:
        - Check how many instances of the target text exist in the file
        - If multiple instances exist, gather enough context to uniquely identify each one

        WARNING: If you do not follow these requirements:
        - The tool will fail if block matches multiple locations
        - The tool will fail if block doesn't match after all fallback passes
        - You may change the wrong instance if you don't include enough context

        When making edits:
        - Ensure the edit results in idiomatic, correct code
        - Do not leave the code in a broken state

        If you want to create or overwrite a file, use the write tool.

        Args:
            path: Path to the file to edit. Relative to the project root is preferred; an absolute path is also accepted.
            block: A single search and replace block formatted as shown above

        Returns:
            A summary of the update made to the file

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the search block doesn't match any content in the file
            ValueError: If the block is malformed or incorrectly formatted
            ValueError: If the block matches more than one place in the file
            PermissionError: If the file cannot be written to
        """
        return await self.edit_tool.edit(path, block)

    async def claude_edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """Perform an exact string replacement in a file.

        Read the file before editing it. ``old_string`` must match exactly and
        be unique unless ``replace_all`` is true.

        Args:
            file_path: Absolute or project-relative path to the file to modify.
            old_string: Exact text to replace.
            new_string: Replacement text, which must differ from old_string.
            replace_all: Replace every exact occurrence instead of requiring a unique match.

        Returns:
            A short summary of the edit.
        """

        return await self.edit_tool.claude_edit(file_path, old_string, new_string, replace_all)

    async def apply_patch(self, input: str) -> str:
        """Use the `apply_patch` tool to edit files.

        This is a FREEFORM tool, so do not wrap the patch in JSON. Supply one
        complete patch beginning with `*** Begin Patch` and ending with
        `*** End Patch`. A patch may add, update, move, or delete multiple files.

        Args:
            input: The raw Codex apply_patch payload.

        Returns:
            A summary of added, modified, moved, and deleted files.
        """
        return await self.edit_tool.apply_patch(input)

    async def hashline_edit(
        self,
        path: str,
        edits: list[dict[str, object]],
        delete: bool = False,
        rename: Optional[str] = None,
    ) -> str:
        """Apply precise edits using Hashline v2 ``LINE#ID`` anchors.

        Read the target range immediately before editing and copy its anchors
        exactly. Every operation is validated against the same pre-edit file
        snapshot and applied bottom-up. Re-read the file before a later edit
        call because successful edits change its anchors.

        Read results render each source line as ``LINE#ID:CONTENT``. The
        ``LINE#ID:`` prefix is display-only metadata, not part of the file. For
        example, given ``1#BM:MAX_RETRIES = 3``, use ``1#BM`` as the anchor and
        ``MAX_RETRIES = 5`` as replacement content. Never copy ``1#BM:`` or any
        other anchor prefix into ``content``.

        Use ``set`` for one line, ``replace`` for an inclusive range,
        ``append``/``prepend`` for insertion after/before an optional anchor,
        and ``insert`` for one- or two-sided anchored insertion. ``content`` may
        be a string or an array of complete lines; null deletes set/replace
        targets. Use ``delete=true`` with an empty edits array to delete a file,
        or ``rename`` to move the edited result.

        Args:
            path: Project-relative file path.
            edits: Hashline v2 operations for this file.
            delete: Delete the file; cannot be combined with edits or rename.
            rename: Optional project-relative destination path.

        Returns:
            A short summary, or fresh tagged context when an anchor is stale.
        """

        return await self.edit_tool.hashline_edit(path, edits, delete, rename)

    async def hashline_write(self, path: str, content: str) -> str:
        """Create or replace a complete file while using Hashline v2.

        Prefer the anchored `edit` tool for changes to an existing file. Use
        this tool for deliberate complete-file writes.

        Args:
            path: Project-relative path to create or replace.
            content: Complete file content.

        Returns:
            A short summary of the write.
        """

        return await self.edit_tool.hashline_write(path, content)

    async def multi_edit(self, path: str, blocks: str) -> str:
        """
        Edit a file using one or more search and replace blocks.

        Each block should be formatted as follows:
        ```
        <<<<<<< SEARCH
        [original code to find]
        =======
        [new code to replace with]
        >>>>>>> REPLACE
        ```

        All blocks are resolved against the original file contents before any changes are written.
        The tool fails without writing if any block is malformed, does not match, matches multiple locations,
        or overlaps with another block. Resolved replacements are applied from the end of the file toward
        the start to avoid offset shifts.

        Matching is attempted in this order for each block:
        1. Exact match.
        2. Per-line stripped match for indentation and trailing whitespace differences.
        3. Normalized line endings.
        4. Normalized smart quotes.

        Args:
            path: Path to the file to edit. Relative to the project root is preferred; an absolute path is also accepted.
            blocks: One or more search and replace blocks formatted as shown above

        Returns:
            A summary of the update made to the file

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If any search block doesn't match any content in the file
            ValueError: If any block is malformed or incorrectly formatted
            ValueError: If any block matches more than one place in the file
            ValueError: If resolved blocks overlap
            PermissionError: If the file cannot be written to
        """
        return await self.edit_tool.multi_edit(path, blocks)

    async def claude_write(self, file_path: str, content: str) -> str:
        """Write a file, overwriting it if it already exists.

        Existing files must be read first. Prefer edit for partial changes.

        Args:
            file_path: Absolute or project-relative path to create or overwrite.
            content: Complete content to write to the file.

        Returns:
            A short summary of the write.
        """

        return await self.edit_tool.claude_write(file_path, content)

    async def list_directory(self, path: str = "") -> str:
        """
        List files and directories at the specified path.

        Args:
            path: Path to list. Relative to the project root is preferred; an absolute path is also accepted.

        Returns:
            Markdown formatted list of files and directories with details

        Raises:
            NotADirectoryError: If the path is not a directory
        """
        return await self.list_directory_tool.list_directory(path)

    async def execute_terminal_command(self, command: str) -> str:
        """Execute a command and display output in terminal."""
        return await self.terminal_tool.execute_terminal_command(command)

    async def exec_command(
        self,
        command: str,
        workdir: Optional[str] = None,
        yield_time_ms: int = 10000,
        max_output_tokens: int = 10000,
        login: bool = False,
    ) -> str:
        """Run a shell command as a fresh process and return its output.

        The command runs under a pseudo-terminal so interactive programs behave
        normally. Output is collected for up to yield_time_ms milliseconds. If
        the process exits within that window, the full result with its real exit
        code is returned. If it is still running, a session_id is returned that
        you can drive with write_stdin (to send input or poll for more output)
        and stop with kill_command.

        The working directory does NOT persist between calls. Pass `workdir`, or
        chain commands in one call with `cd path && ...`. Defaults to the
        project root.

        Args:
            command: Shell command line, executed via `bash -c`.
            workdir: Working directory for the command. Defaults to project root.
            yield_time_ms: How long to wait for output/exit before returning, in
                           milliseconds (clamped to 250–30000).
            max_output_tokens: Maximum tokens of output to return in this call.
            login: Run the shell as a login shell (sources profile). Default false.

        Returns:
            A JSON object: {"status": "exited"|"running", "exit_code",
            "session_id", "output", "truncated", "original_token_count",
            "duration_ms"}.
        """
        return await self.terminal_tool.exec_command(
            command,
            workdir=workdir,
            yield_time_ms=yield_time_ms,
            max_output_tokens=max_output_tokens,
            login=login,
        )

    async def write_stdin(
        self,
        session_id: str,
        chars: str = "",
        yield_time_ms: int = 10000,
        max_output_tokens: int = 10000,
    ) -> str:
        """Write input to a running session's stdin and read recent output.

        Pass chars="" to poll (read new output without writing). Use this to
        answer prompts (e.g. send "y\\n"), drive a REPL, or send control
        characters (e.g. "\\x03" for Ctrl-C). The text is sent raw — include a
        trailing "\\n" to submit a line. Waits up to yield_time_ms (clamped to
        250–30000 when writing, 5000–300000 when polling) for more output or for
        the process to exit.

        Args:
            session_id: The id returned by exec_command when status == "running".
            chars: Bytes to write to stdin. An empty string polls only.
            yield_time_ms: Wait window in milliseconds.
            max_output_tokens: Maximum tokens of output to return in this call.

        Returns:
            A JSON object with the same shape as exec_command.
        """
        return await self.terminal_tool.write_stdin(
            session_id, chars, yield_time_ms=yield_time_ms, max_output_tokens=max_output_tokens
        )

    async def kill_command(self, session_id: str, signal: str = "TERM") -> str:
        """Terminate a running session and its process group.

        Sends SIGTERM (then SIGKILL after a short grace period). Use
        signal="INT" to send Ctrl-C (SIGINT) instead.

        Args:
            session_id: The id of the session to stop.
            signal: "TERM" (default, graceful) or "INT" (Ctrl-C).

        Returns:
            A JSON object describing the final state of the session.
        """
        return await self.terminal_tool.kill_command(session_id, signal)

    async def list_sessions(self) -> str:
        """List currently running exec sessions.

        Returns:
            A JSON object mapping each running session id to its command,
            working directory, and runtime in seconds.
        """
        return await self.terminal_tool.list_sessions()

    async def read_entire_file(self, path: str) -> str:
        """
        Read the contents of a file in the project.

        Note: Files exceeding 2000 lines will be truncated with a warning message.
        Use read_file_section to read specific portions of large files.

        Args:
            path: Path to the file. Relative to the project root is preferred; an absolute path is also accepted.

        Returns:
            The contents of the file as a string formatted as markdown

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        formatter = format_hash_lines if self._hashline_output_enabled() else None
        if formatter is None:
            result = await self.read_file_tool.read_entire_file(path)
        else:
            result = await self.read_file_tool.read_entire_file(path, line_formatter=formatter)
        if self.edit_protocol == EditProtocol.CLAUDE_CODE:
            self.edit_tool.observe_read(path)
        return result

    async def read_file_section(self, path: str, start_line: int, end_line: int) -> str:
        """
        Read a specific section of a file in the project from start_line to end_line (inclusive).

        Args:
            path: Path to the file. Relative to the project root is preferred; an absolute path is also accepted.
            start_line: The line number to start reading from (1-indexed)
            end_line: The line number to stop reading at (1-indexed, inclusive)

        Returns:
            The specified section of the file as a string formatted as markdown

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If start_line or end_line are invalid
        """
        formatter = format_hash_lines if self._hashline_output_enabled() else None
        if formatter is None:
            result = await self.read_file_tool.read_file_section(path, start_line, end_line)
        else:
            result = await self.read_file_tool.read_file_section(
                path,
                start_line,
                end_line,
                line_formatter=formatter,
            )
        if self.edit_protocol == EditProtocol.CLAUDE_CODE:
            self.edit_tool.observe_read(path)
        return result

    async def write(self, path: str, content: str) -> str:
        """
        Write content to a file in the project.

        This tool creates the file if it does not exist and replaces the entire file if it does.
        For small edits to existing files, prefer edit or multi_edit.

        Args:
            path: Path to the file to write. Relative to the project root is preferred; an absolute path is also accepted.
            content: Content to write to the file

        Returns:
            A summary of the write

        Raises:
            PermissionError: If the file cannot be written to
        """
        return await self.edit_tool.write(path, content)

    async def snapshot(
        self,
        action: str = "list",
        snapshot_id: str = "",
        paths: Optional[list[str]] = None,
        force: bool = False,
        limit: int = 20,
    ) -> str:
        """Manage file snapshots for undo, inspection, and manual checkpoints.

        Use action="list" to see recent snapshots, action="show" with a snapshot_id
        to inspect one, action="create" with paths to make a manual checkpoint, and
        action="restore" to restore a snapshot's before-state. Use snapshot_id="latest"
        with restore as an undo for the newest snapshot.

        Args:
            action: One of list, show, create, or restore.
            snapshot_id: Snapshot id for show/restore; use latest for newest.
            paths: Project-relative paths for create.
            force: Restore even when tracked files changed after the snapshot.
            limit: Maximum number of snapshots to list.

        Returns:
            Markdown summary of the snapshot operation.
        """
        return await self.snapshot_tool.snapshot(
            action=action,
            snapshot_id=snapshot_id,
            paths=paths,
            force=force,
            limit=limit,
        )

    async def resolve(self, action_id: str, decision: str, force: bool = False) -> str:
        """Apply or discard a pending preview action.

        Pending actions are created by preview-only tools such as lsp_edit(apply=false).
        Applying a pending action checks that the source files still match the preview
        inputs before writing, unless force=true is explicitly provided.

        Args:
            action_id: Pending action id returned by a preview-only tool.
            decision: apply or discard.
            force: Apply even if source hashes no longer match.

        Returns:
            Markdown summary of the resolve operation.
        """
        return await self.snapshot_tool.resolve(action_id=action_id, decision=decision, force=force)

    async def read_memory(self, path: str = "MEMORY.md") -> str:
        """Read a private project-memory Markdown entry."""
        return await self.memory_tool.read_memory(path)

    async def list_memory(self, query: str | None = None) -> str:
        """List private project-memory entries, optionally filtering by path or content."""
        return await self.memory_tool.list_memory(query)

    async def write_memory(
        self,
        memory_content: str,
        path: str = "MEMORY.md",
        mode: str = "append",
        expected_sha256: str | None = None,
    ) -> str:
        """Append to or replace a private project-memory entry using its current revision."""
        return await self.memory_tool.write_memory(
            memory_content,
            path,
            mode,
            expected_sha256,
        )

    async def delete_memory(self, path: str, expected_sha256: str) -> str:
        """Delete a private project-memory entry after reading its current revision."""
        return await self.memory_tool.delete_memory(path, expected_sha256)

    async def search_codebase(
        self, pattern: str, file_pattern: str = "*", case_sensitive: bool = False, literal: bool = False
    ) -> str:
        """
        Search the codebase for lines matching a regular expression (grep/ripgrep).

        The pattern is treated as a regular expression by default, so `|` is
        alternation: search for `TODO|FIXME|HACK` to match any of the three. Use
        ripgrep/POSIX-ERE syntax (alternation, character classes `[...]`, anchors
        `^ $`, quantifiers `* + ? {n,m}`, groups `(...)`). Set `literal=True` to match
        the pattern as plain text instead (e.g. to find `arr[0]` or `a||b` verbatim).

        Args:
            pattern: The regular expression to search for (use `literal=True` to match it as plain text)
            file_pattern: Optional glob to filter which files to search (default: all files)
            case_sensitive: Whether the search is case-sensitive (default: False)
            literal: Treat the pattern as plain text instead of a regular expression (default: False)

        Returns:
            Markdown formatted list of files and matches, limited to 128 results

        Raises:
            Exception: If any error occurs during the search operation
        """
        formatter: Callable[[int, str], str] | None = None
        if self._hashline_output_enabled():

            def hashline_formatter(line_number: int, content: str) -> str:
                if line_number == 1 and content.startswith("\ufeff"):
                    content = content[1:]
                return f"{format_line_tag(line_number, content)}:{content}"

            formatter = hashline_formatter

        kwargs = {
            "file_pattern": file_pattern,
            "case_sensitive": case_sensitive,
            "literal": literal,
        }
        if formatter is not None:
            kwargs["line_formatter"] = formatter
        return await self.search_codebase_tool.search_codebase(pattern, **kwargs)

    async def web_fetch(self, url: str, instruction: str) -> str:
        """
        Fetch URL content locally, follow an instruction, and return a grounded response.

        This tool handles HTML through a quality-gated local extractor chain, reads
        textual formats directly, converts PDF and modern Office documents locally,
        and asks the fast model to apply the instruction with source evidence. It does
        not run JavaScript or send content to a third-party reader service. For a page
        reported as JavaScript-rendered, use the browser tools instead.

        Args:
            url: Full http(s) URL to fetch.
            instruction: Guidance for how to use the extracted content.

        Returns:
            A source-attributed answer with evidence, or bounded extracted content if
            the internal answering stage cannot complete.
        """
        return await self.web_fetch_tool.web_fetch(url, instruction)

    async def web_search(self, query: str, max_results: int = 5) -> str:
        """
        Search the web and return a ranked list of results (title, URL, and a short snippet).

        Use this to discover relevant pages for a query when you don't already know the URL.
        The search backend (DuckDuckGo, Firecrawl, Tavily, or a self-hosted SearXNG instance)
        is whatever the user configured in Settings; the default works without an API key. To
        read a specific result in depth, follow up with the web_fetch tool on its URL.

        Args:
            query: The search query (natural language or keywords).
            max_results: Maximum number of results to return (clamped to 1-10, default 5).

        Returns:
            A markdown list of results, or a message if no results were found.
        """
        return await self.web_search_tool.web_search(query, max_results)

    async def find_files_by_pattern(
        self, pattern: str, include_directories: bool = True, show_details: bool = True
    ) -> str:
        """
        Find files by glob pattern in the project directory.

        Behavior:
        - Supports patterns like '*.py', 'src/**/*.js'. Leading '/' is ignored.
        - Bare filenames without wildcards or '/' (e.g., 'README.md') are treated as '**/README.md'.
        - include_directories=True (default) shows directories as well as files.
        - Returns 128 results max.

        Args:
            pattern: Glob pattern or filename to search for
            include_directories: Include directories in results (default: True)
            show_details: Include size/mtime/type metadata (default: True)

        Returns:
            Markdown with the matching items (max 128)
        """
        return await self.glob_tool.find_files_by_pattern(
            pattern, include_directories=include_directories, show_details=show_details
        )

    async def lsp(
        self,
        operation: str,
        path: Optional[str] = None,
        line: Optional[int] = None,
        symbol: Optional[str] = None,
        query: Optional[str] = None,
        end_line: Optional[int] = None,
        kind: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> str:
        """Query language server intelligence: diagnostics, definition, references, hover, symbols, status.

        This versatile read-only tool interacts with the project's language servers.
        Different operations require different arguments — see the operation list below.

        Operations and required arguments:
        - ``diagnostics`` — errors/warnings/hints for a file (``path``)
        - ``definition`` — go-to-definition (``path``, ``line``, ``symbol``)
        - ``type_definition`` — go-to-type-definition (``path``, ``line``, ``symbol``)
        - ``implementation`` — find implementations (``path``, ``line``, ``symbol``)
        - ``references`` — find all references (``path``, ``line``, ``symbol``)
        - ``hover`` — hover/type info (``path``, ``line``, ``symbol``)
        - ``call_hierarchy`` — incoming/outgoing calls (``path``, ``line``, ``symbol``)
        - ``code_actions`` — list fixes/refactors without applying them (``path``, ``line``, ``symbol``)
        - ``document_symbols`` — symbols in a file (``path``)
        - ``workspace_symbols`` — project-wide symbol search (``query``)
        - ``status`` — LSP server status (no args)
        - ``capabilities`` — server capabilities (optional ``path``)
        - ``reload`` — restart servers and re-detect (no args)

        For position operations, ``line`` is 1-based and ``symbol`` is the name to
        find on that line. Use ``name#N`` for the Nth occurrence.

        Args:
            operation: One of the operations listed above.
            path: File path (relative to project root preferred).
            line: 1-based line number for position operations.
            symbol: Symbol name to resolve on the line (supports ``name#N``).
            query: Search query for ``workspace_symbols``.
            end_line: Optional 1-based end line for ``code_actions``.
            kind: Optional code action kind filter, e.g. ``quickfix``.
            timeout: Per-call timeout in seconds (default: 30).

        Returns:
            Markdown-formatted results for the requested operation.
        """
        return await self.lsp_tool.lsp(operation, path, line, symbol, query, end_line, kind, timeout)

    async def lsp_edit(
        self,
        operation: str,
        path: Optional[str] = None,
        line: Optional[int] = None,
        symbol: Optional[str] = None,
        new_name: Optional[str] = None,
        new_path: Optional[str] = None,
        query: Optional[str] = None,
        action_id: Optional[str] = None,
        end_line: Optional[int] = None,
        kind: Optional[str] = None,
        apply: bool = True,
        timeout: Optional[float] = None,
    ) -> str:
        """Apply trusted LSP edits such as rename, file rename, formatting, and code actions.

        This is the mutating companion to the read-only ``lsp`` tool. Use
        ``apply=False`` to preview the server-provided WorkspaceEdit without
        writing files.
        """
        return await self.lsp_edit_tool.lsp_edit(
            operation,
            path,
            line,
            symbol,
            new_name,
            new_path,
            query,
            action_id,
            end_line,
            kind,
            apply,
            timeout,
        )

    async def get_host(self, port: int) -> str:
        """
        Get the hostname for accessing a service on the specified port.

        This tool returns the appropriate hostname based on the environment:
        - In local development: returns 'localhost:PORT'
        - In cloud sandbox (e2b): returns the sandbox-specific hostname

        When to use this tool:
        - Before accessing any web service or development server
        - When constructing URLs for HTTP requests
        - When providing URLs to users or other tools
        - When launching browsers to access local services

        Usage notes:
        1. Always call this tool before making HTTP requests to local services
        2. The port parameter is required - specify the port your service is running on
        3. Use the returned hostname to construct full URLs (e.g., http://{host}/api/endpoint)
        4. This ensures your code works in both local and cloud sandbox environments

        Args:
            port: The port number where the service is running

        Returns:
            The full hostname including port (e.g., 'localhost:3000' or 'xxxx.e2b.dev')
        """
        # Check if we're using a SandboxTerminalManager (indicates sandbox mode).
        # ``sandbox`` is only present on ``SandboxTerminalManager``; the base
        # ``TerminalManager``/``LocalTerminalManager`` do not declare it, so access it
        # via ``getattr`` rather than a direct attribute lookup.
        sandbox = getattr(self.terminal_manager, "sandbox", None)
        if sandbox is not None:
            # We're in sandbox mode, get the host from the sandbox
            # E2B AsyncSandbox has a get_host method that takes a port
            # The method is synchronous and returns a string directly
            host = sandbox.get_host(port)
            return host
        else:
            # Local mode, return localhost
            return f"localhost:{port}"

    def _tool_definition_from_callable(self, method_name: str, method: Callable[..., Any]) -> ToolDefinition:
        """Build a provider-agnostic tool definition from a Python callable."""
        definition = tool_definition_from_callable(method_name, method)
        if method_name == "apply_patch":
            definition.description = (
                "Use the `apply_patch` tool to edit files. This is a FREEFORM tool, so do not wrap the patch in JSON."
            )
            definition.input_kind = "freeform"
            definition.freeform_format = {
                "type": "grammar",
                "syntax": "lark",
                "definition": CODEX_APPLY_PATCH_GRAMMAR,
            }
        if method_name == "edit" and self.edit_protocol == EditProtocol.HASHLINE_V2:
            definition.input_schema = _HASHLINE_V2_INPUT_SCHEMA
        if method_name == "dispatch_custom_agent":
            catalog = getattr(self.caller, "custom_agent_catalog", None)
            if catalog is not None and catalog.has_agents():
                routing_catalog = catalog.model_catalog()
                definition.description = f"{definition.description}\n\nAvailable custom agents:\n{routing_catalog}"
                definition.input_schema = {
                    "type": "object",
                    "properties": {
                        "agent": {
                            "type": "string",
                            "enum": catalog.names(),
                            "description": "Name of the custom agent to run.",
                        },
                        "task": {
                            "type": "string",
                            "description": (
                                "Detailed, self-contained task including relevant context and expected output."
                            ),
                        },
                    },
                    "required": ["agent", "task"],
                }
        return definition

    def _hashline_output_enabled(self) -> bool:
        """Whether this collection actually exposes the Hashline edit binding."""

        return (
            self.edit_protocol == EditProtocol.HASHLINE_V2
            and "edit" not in self.tool_exclusions
            and self._should_include_tool("edit")
        )

    def _groups_for(self, method_name: str) -> frozenset:
        """Group tags for a tool, from the core group lists plus extension groups."""
        group_attrs = {
            "read_only_tools",
            "browser_tools",
            "agent_dispatch_tools",
            "coder_agent_tools",
            "custom_agent_tools",
            "memory_tools",
            "orchestration_tools",
            *self._extension_group_names,
        }
        return frozenset(
            group_name for group_name in group_attrs if method_name in (getattr(self, group_name, None) or [])
        )

    def registry(self) -> ToolRegistry:
        """
        Build the ToolRegistry of currently enabled tools.

        Rebuilt per call (matching the previous dynamic get_tool_list behavior)
        so tools added by subclasses or extensions after construction are seen.
        """
        registry = ToolRegistry()

        for method_name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if (
                method_name.startswith("_")
                or method_name in self.tool_exclusions
                or method_name in self._memory_compatibility_methods
            ):
                continue
            if method_name in EDIT_HANDLER_NAMES:
                continue
            if not self._should_include_tool(method_name):
                continue
            registry.add(self._build_tool(method_name, method))

        for binding in edit_protocol_spec(self.edit_protocol).tools:
            if binding.name in self.tool_exclusions or not self._should_include_tool(binding.name):
                continue
            registry.add(self._build_tool(binding.name, getattr(self, binding.handler_name)))

        for method_name, method in self.extension_callbacks.items():
            if method_name in registry or method_name in self.tool_exclusions:
                continue
            if not self._should_include_tool(method_name):
                continue
            registry.add(self._build_tool(method_name, method))

        for binding in self.memory_tool.bindings():
            if binding.name in registry or not self._should_include_memory_binding(
                binding.name, mutating=binding.mutating
            ):
                continue
            definition_data = binding.definition
            definition = ToolDefinition(
                name=binding.name,
                description=str(definition_data.get("description", "")),
                parameters=[],
                input_schema=dict(definition_data.get("input_schema", {"type": "object"})),
            )

            async def handler(_binding=binding, **inputs):
                return await self.memory_tool.invoke(_binding, **inputs)

            registry.add(
                Tool(
                    name=binding.name,
                    definition=definition,
                    handler=handler,
                    groups=frozenset({"memory_tools"}),
                    parallel_safe=not binding.mutating,
                )
            )

        return registry

    def _should_include_memory_binding(self, name: str, *, mutating: bool) -> bool:
        """Memory capability is backend-driven; exact host policy remains final."""
        if not self.tool_config.include_memory_tools:
            return False
        if name in self.tool_exclusions:
            return False
        if mutating and (getattr(self.caller, "sub_agent", False) or not self.tool_config.memory_write_access):
            return False
        allowed = self.tool_config.allowed_tools
        return allowed is None or name in allowed

    def _build_tool(self, method_name: str, method: Callable[..., Any]) -> Tool:
        definition = self._tool_definition_from_callable(method_name, method)
        explicit_schema = self.extension_schemas.get(method_name)
        if explicit_schema is not None:
            definition.input_schema = explicit_schema
        return Tool(
            name=method_name,
            definition=definition,
            handler=method,
            groups=self._groups_for(method_name),
            # Read-only tools have no side effects and agent dispatches operate
            # on independent sub-agents, so these may run concurrently.
            parallel_safe=(
                method_name in (self.read_only_tools or []) or method_name in (self.agent_dispatch_tools or [])
            ),
        )

    def has_tool(self, name: str) -> bool:
        """True if the named tool is currently enabled."""
        return name in self.registry()

    async def call(self, tool_name: str, /, **inputs: Any) -> Any:
        """Dispatch an enabled tool by name."""
        return await self.registry().call(tool_name, **inputs)

    def get_tool_list(self) -> List[ToolDefinition]:
        """
        Returns a list of tool definitions in the format required by the Anthropic API.

        Definitions are generated from the enabled tools' signatures and
        docstrings; the last definition carries the prompt-cache checkpoint.
        """
        return self.registry().definitions()

    def _should_include_tool(self, method_name: str) -> bool:
        """
        Determine if a tool method should be included based on the configuration.

        Args:
            method_name: Name of the method/tool to check

        Returns:
            True if the tool should be included, False otherwise
        """
        if method_name == "dispatch_custom_agent":
            catalog = getattr(self.caller, "custom_agent_catalog", None)
            if getattr(self.caller, "sub_agent", False) or catalog is None or not catalog.has_agents():
                return False

        if self.tool_config.allowed_tools is not None and method_name not in self.tool_config.allowed_tools:
            return False

        # gigacode orchestration: only the top-level (non-sub) agent may run
        # workflows, and only when gigacode has been enabled for the session.
        # This both prevents sub-agents from recursively spawning workflows and
        # keeps the expensive tool off until the user opts in.
        if method_name in self.orchestration_tools:
            if getattr(self.caller, "sub_agent", False):
                return False
            return bool(getattr(self.caller, "gigacode_enabled", False))

        # Vision gate: read_image is only surfaced to vision-capable models.
        if method_name == "read_image":
            return bool(getattr(self.caller, "supports_vision", False))

        # Check custom tool groups first
        if self.tool_config.custom_tool_groups:
            for group_name in self.tool_config.custom_tool_groups:
                if hasattr(self, group_name):
                    group_tools = getattr(self, group_name)
                    if method_name in group_tools:
                        return True

        # If restrict_to_tool_groups is True, only include tools from explicitly enabled groups
        if self.tool_config.restrict_to_tool_groups:
            # Check if tool belongs to any enabled group
            if method_name in self.agent_dispatch_tools and self.tool_config.include_agent_dispatch_tools:
                return True
            if method_name in self.memory_tools and self.tool_config.include_memory_tools:
                return method_name not in self.tool_exclusions
            if method_name in self.browser_tools and self.tool_config.browser_only:
                return True
            if method_name in self.read_only_tools and self.tool_config.read_only:
                return True
            # Tool doesn't belong to any enabled group
            return False

        # Original behavior for non-restricted mode
        # Handle legacy read-only filtering
        if self.tool_config.read_only and method_name not in self.read_only_tools:
            return False

        # Handle legacy browser-only filtering
        if self.tool_config.browser_only and method_name not in self.browser_tools:
            return False

        # Exclude browser tools unless this is a browser-only agent or investigation tools are enabled
        if (
            not self.tool_config.browser_only
            and not self.tool_config.include_agent_dispatch_tools
            and method_name in self.browser_tools
        ):
            return False

        # Check investigation agent tools
        if method_name in self.agent_dispatch_tools:
            return self.tool_config.include_agent_dispatch_tools

        # Check memory tools
        if method_name in self.memory_tools:
            # Include memory tools if explicitly enabled, or if memory tools are not excluded
            return self.tool_config.include_memory_tools or method_name not in self.tool_exclusions

        # Include all other core tools by default
        return True

    async def initialize(self) -> list[str]:
        """Perform async one-time initialization (LSP auto-detection, etc.).

        Safe to call multiple times — ``LspManager.initialize()`` is idempotent.
        Returns status messages that the caller may display (e.g., detected
        languages, install prompts for missing servers).
        """
        if self.lsp_manager is not None:
            return await self.lsp_manager.initialize()
        return []

    async def cleanup(self):
        """Clean up all tool resources"""
        try:
            # Clean up LSP resources
            if hasattr(self, "lsp_manager") and self.lsp_manager is not None:
                await self.lsp_manager.shutdown()
                await self.log_info("Cleaned up LSP resources", sender="ToolCollection")

            # Clean up terminal resources
            if hasattr(self, "terminal_tool") and hasattr(self.terminal_tool, "terminal_manager"):
                await self.terminal_tool.terminal_manager.cleanup_all()
                await self.log_info("Cleaned up terminal resources", sender="ToolCollection")

            # Clean up any browser resources
            if hasattr(self, "browser_tool") and hasattr(self.browser_tool, "cleanup"):
                await self.browser_tool.cleanup()
                await self.log_info("Cleaned up browser resources", sender="ToolCollection")

            # Clean up any sub-agents
            if hasattr(self, "agent_tool") and hasattr(self.agent_tool, "agents"):
                for agent_id, agent in list(self.agent_tool.agents.items()):
                    if hasattr(agent, "cleanup"):
                        try:
                            await agent.cleanup()
                            await self.log_info(f"Cleaned up sub-agent: {agent_id}", sender="ToolCollection")
                        except Exception as e:
                            await self.log_warning(
                                f"Error cleaning up sub-agent {agent_id}: {e}", sender="ToolCollection"
                            )

            # Clean up host-provided tool extensions (MCP transports, etc.).
            for extension in self.tool_extensions:
                cleanup = getattr(extension, "cleanup", None)
                if cleanup is None:
                    continue
                try:
                    result = cleanup()
                    if inspect.isawaitable(result):
                        await result
                except Exception as e:
                    await self.log_warning(
                        f"Error cleaning up tool extension {extension.name}: {e}", sender="ToolCollection"
                    )

        except Exception as e:
            await self.log_error(f"Error during tool cleanup: {str(e)}", sender="ToolCollection")

        self.extension_callbacks = {}
        self.extension_schemas = {}
