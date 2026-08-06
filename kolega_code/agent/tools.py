import inspect
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Union

from .common import LogMixin
from kolega_code.config import AgentConfig, EditProtocol
from kolega_code.llm.models import ImageBlock, ToolDefinition
from kolega_code.memory import ProjectMemoryManager
from kolega_code.tools import Tool, ToolRegistry, schema_has_property_descriptions, tool_definition_from_callable
from kolega_code.services.file_system import FileSystem, LocalFileSystem
from kolega_code.services.base import TerminalManager, BrowserManager
from kolega_code.services.terminal import LocalTerminalManager
from kolega_code.services.browser import PlaywrightBrowserManager
from .tool_backend.agent_tool import AgentTool
from .tool_backend.browser_tool import BROWSER_TOOL_SCHEMAS, BrowserTool
from .tool_backend.edit_tool import EditTool
from .tool_backend.eval_tool import EvalTool
from .tool_backend.codex_patch import CODEX_APPLY_PATCH_GRAMMAR
from .tool_backend.hashline_v2 import format_hash_lines
from .tool_backend.memory_tool import MemoryTool
from .tool_backend.network_status import ConnectFailureTracker
from .tool_backend.read_file_tool import ReadFileTool
from .tool_backend.read_image_tool import ReadImageTool
from .tool_backend.snapshot_tool import SnapshotTool
from .tool_backend.web_fetch_tool import WebFetchTool
from .tool_backend.web_search_tool import WebSearchTool
from .tool_backend.terminal_tool import TerminalTool
from .tool_backend.workflow_tool import RUN_WORKFLOW_INPUT_SCHEMA, WorkflowTool
from .edit_protocols import EDIT_HANDLER_NAMES, edit_protocol_spec
from .orchestration.context import has_workflow_context_marker, validated_workflow_depth

# Import additional tools for consolidated functionality
from .tool_backend.build_tool import BuildTool
from .tool_backend.lsp_tool import LspEditTool, LspTool
from kolega_code.services.lsp import LspManager
from kolega_code.services.snapshots import SnapshotService

# Atomic ordinary-dispatch routing cannot be represented by signature
# introspection: all nested fields are required together and effort is nullable.
_ORDINARY_MODEL_OVERRIDE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "provider": {
            "type": "string",
            "minLength": 1,
            "description": (
                "Non-empty configured provider name returned by list_subagent_models. "
                "Never infer it from the model name. "
                "`openai` and `openai_chatgpt` serve the same models; prefer `openai_chatgpt` when configured."
            ),
        },
        "model": {
            "type": "string",
            "minLength": 1,
            "description": "Non-empty exact model ID returned for that provider by list_subagent_models.",
        },
        "thinking_effort": {
            "anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}],
            "description": (
                "Exact supported effort string, or null only when the selected model has no effort control."
            ),
        },
    },
    "required": ["provider", "model", "thinking_effort"],
}


def _dispatch_agent_input_schema(
    agent_types: Sequence[str],
    *,
    browser_targets: tuple[str, ...] = (),
) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "agent_type": {
            "type": "string",
            "enum": list(agent_types),
            "description": "Which agent to dispatch. The tool description lists each value's role and tools.",
        },
        "task": {
            "type": "string",
            "description": "Detailed, self-contained task for the sub-agent.",
        },
        "model_override": {
            **_ORDINARY_MODEL_OVERRIDE_SCHEMA,
            "description": (
                'Usually omit this property entirely: a normal call is {"agent_type": "...", "task": "..."}. '
                "Only include it after calling list_subagent_models and selecting one exact route. "
                "Never send an empty object, blank strings, placeholder values, or a guessed provider/model. "
                "When present, all three nested fields are required."
            ),
        },
    }
    if "browser" in agent_types and len(browser_targets) > 1:
        properties["browser_target"] = {
            "type": "string",
            "enum": list(browser_targets),
            "description": (
                'Only for agent_type "browser". Omit for Playwright; choose Chrome only when the user directs you '
                "to use their configured Chrome browser."
            ),
        }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": ["agent_type", "task"],
    }


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
            "description": (
                "File path: project-relative, using ../ traversal, or absolute. Required for most operations."
            ),
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
            "description": "File path: project-relative, using ../ traversal, or absolute.",
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
            "description": ("Destination path for rename_file: project-relative, using ../ traversal, or absolute."),
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
        "path": {
            "type": "string",
            "description": "Path to edit: project-relative, using ../ traversal, or absolute.",
        },
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
        "rename": {
            "type": "string",
            "description": ("Move the edited result to this path: project-relative, using ../ traversal, or absolute."),
        },
    },
    "required": ["path", "edits"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class ToolExtension:
    """Host-provided tool callbacks and named groups.

    ``exclusive_tools`` declares session-control callbacks that must be the
    sole tool call in a model response. If one is batched with another call,
    the complete batch is rejected before any callback executes.
    """

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
    # Tools that may only be called as the sole tool in a model response. If
    # one appears in a larger batch, BaseAgent rejects the whole batch before
    # any callback runs.
    exclusive_tools: frozenset[str] = field(default_factory=frozenset)


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
        excluded_agent_types: Optional[List[str]] = None,
    ):
        """
        Initialize tool collection configuration.

        Args:
            read_only: Whether to restrict to read-only tools
            browser_only: Whether to only include browser tools
            include_agent_dispatch_tools: Whether to include the dispatch_agent tool (every agent_type)
            include_memory_tools: Whether to include memory management tools
            memory_write_access: Whether mutating memory tools (write/delete) may be exposed to this agent;
                subagent scope still strips them regardless.
            tool_exclusions: List of method names to exclude from tool list
            custom_tool_groups: Additional custom tool groups to include
            enabled_tool_groups: Additional custom tool groups to include
            restrict_to_tool_groups: If True, ONLY include tools from specified groups, excluding all other core tools
            allowed_tools: Optional exact allowlist applied after hard runtime gates. None inherits normal behavior;
                an empty list exposes no tools.
            excluded_agent_types: dispatch_agent agent_type values to withhold from this
                collection (e.g. "general" for a dispatched coder that must not fan out).
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
        self.excluded_agent_types = list(excluded_agent_types or [])


class ToolCollection(LogMixin):
    """
    A collection of tools for interacting with the project workspace.

    Provides utilities for file operations and workspace management.
    """

    read_only_tools = [
        "read",
        "read_memory",
        "list_memory",
        "web_fetch",
        "web_search",
        "read_image",
        "lsp",
        "list_subagent_models",
        "list_workflow_runs",
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
        "browser_scroll",
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

    # Agent dispatch group - a single dispatch_agent tool routes to every
    # dispatchable agent type; the agent_type enum it offers is computed per
    # collection in _available_agent_types.
    agent_dispatch_tools = [
        "dispatch_agent",
    ]

    # Legacy name for backward compatibility
    investigation_agent_tools = agent_dispatch_tools

    # CoderAgent specific dispatch tools
    coder_agent_tools = [
        "dispatch_agent",
    ]

    custom_agent_tools = [
        "dispatch_agent",
    ]

    # agent_type enum members each dispatch-enabled group admits. "custom"
    # stands for the names in the custom-agent catalog. The coder group omits
    # "coding" so the CoderAgent never dispatches itself.
    _dispatch_group_agent_types = {
        "agent_dispatch_tools": ("general", "investigation", "browser", "coding", "custom"),
        "investigation_agent_tools": ("general", "investigation", "browser", "coding", "custom"),
        "coder_agent_tools": ("general", "investigation", "browser", "custom"),
        "custom_agent_tools": ("custom",),
    }

    # Informational support for callers that can dispatch a child or author a
    # workflow. Inclusion is capability-gated separately so leaf agents never
    # receive the provider/model catalog tool.
    delegation_tools = ["list_subagent_models"]

    # Memory tools group
    memory_tools = [
        "read_memory",
        "list_memory",
        "write_memory",
        "edit_memory",
        "delete_memory",
    ]

    # gigacode workflow orchestration. Gated in _should_include_tool: only the
    # top-level (non-sub) agent gets it, and only when gigacode is enabled.
    orchestration_tools = [
        "run_workflow",
        "list_workflow_runs",
    ]

    # Shell execution + session management. Exposed to the planning and
    # investigation agents (via custom_tool_groups) so they can run investigative
    # commands even while read_only=True. Not read-only, so deliberately excluded
    # from the parallel-safe set in _build_tool. `eval` (arbitrary code in the
    # persistent kernels) carries the same power level as exec_command, so it
    # shares this group.
    command_tools = [
        "exec_command",
        "write_stdin",
        "kill_command",
        "list_sessions",
        "eval",
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
            self.terminal_manager = LocalTerminalManager(
                workspace_id, thread_id, connection_manager, default_workdir=self.project_path
            )
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
        self.exclusive_tools = frozenset()
        self._extension_group_names = set()
        self._extension_dispatch_tools: set[str] = set()
        self._legacy_only_extension_dispatch_tools: set[str] = set()
        # Extension declarations are per collection. Never mutate the shared
        # class inventories, and keep the legacy alias as the same list object.
        self.agent_dispatch_tools = list(type(self).agent_dispatch_tools)
        self.investigation_agent_tools = self.agent_dispatch_tools

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
            "cleanup_tool_extensions",
            "log_error",
            "log_warning",
            "log_info",
            "log_debug",
        ]
        # Backend bindings, not these compatibility methods, own model-facing
        # memory schemas and capability registration.
        self._memory_compatibility_methods = {
            "read_memory",
            "list_memory",
            "write_memory",
            "edit_memory",
            "delete_memory",
        }
        self.tool_exclusions.extend(tool_config.tool_exclusions)

        # Initialize tool backends
        self._initialize_tools()
        self._register_tool_extensions()

    def _register_tool_extensions(self) -> None:
        """Bind host-provided extension callbacks onto this collection."""
        self.exclusive_tools = frozenset(
            tool_name for extension in self.tool_extensions for tool_name in extension.exclusive_tools
        )
        for extension in self.tool_extensions:
            for tool_name, callback in extension.tools.items():
                if hasattr(self, tool_name):
                    raise ValueError(f"Tool extension '{extension.name}' conflicts with existing tool '{tool_name}'")
                setattr(self, tool_name, callback)
                self.extension_callbacks[tool_name] = callback

            for tool_name, schema in extension.tool_schemas.items():
                self.extension_schemas[tool_name] = schema

        canonical_declarations = {
            tool_name
            for extension in self.tool_extensions
            for tool_name in extension.tool_groups.get("agent_dispatch_tools", [])
        }
        legacy_declarations = {
            tool_name
            for extension in self.tool_extensions
            for tool_name in extension.tool_groups.get("investigation_agent_tools", [])
        }
        self._legacy_only_extension_dispatch_tools = (
            legacy_declarations - canonical_declarations
        ) & self.extension_callbacks.keys()

        for extension in self.tool_extensions:
            for group_name, tool_names in extension.tool_groups.items():
                if group_name in {"agent_dispatch_tools", "investigation_agent_tools"}:
                    for tool_name in tool_names:
                        if tool_name not in self.agent_dispatch_tools:
                            self.agent_dispatch_tools.append(tool_name)
                        if tool_name in self.extension_callbacks:
                            self._extension_dispatch_tools.add(tool_name)
                    self._extension_group_names.update({"agent_dispatch_tools", "investigation_agent_tools"})
                    continue
                existing_group = list(getattr(self, group_name, []))
                merged_group = list(dict.fromkeys(existing_group + list(tool_names)))
                setattr(self, group_name, merged_group)
                self._extension_group_names.add(group_name)

    async def cleanup_tool_extensions(self, extensions: List[ToolExtension]) -> None:
        """Release resources owned by extensions that are no longer installed."""
        for extension in extensions:
            cleanup = extension.cleanup
            if cleanup is None:
                continue
            try:
                result = cleanup()
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                await self.log_warning(
                    f"Error cleaning up tool extension {extension.name}: {exc}",
                    sender="ToolCollection",
                )

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
        self.eval_tool = EvalTool(
            self.project_path,
            self.workspace_id,
            self.thread_id,
            self.connection_manager,
            self.config,
            self.caller,
            self.filesystem,
        )
        memory_manager = getattr(self.caller, "memory_manager", None)
        if not isinstance(memory_manager, ProjectMemoryManager):
            memory_manager = None
        self.memory_tool = MemoryTool(memory_manager, self.caller)
        connect_failures = ConnectFailureTracker()
        self.web_fetch_tool = WebFetchTool(
            self.project_path,
            self.workspace_id,
            self.thread_id,
            self.connection_manager,
            self.config,
            self.caller,
            self.filesystem,
            connect_failures=connect_failures,
        )
        self.web_search_tool = WebSearchTool(
            self.project_path,
            self.workspace_id,
            self.thread_id,
            self.connection_manager,
            self.config,
            self.caller,
            self.filesystem,
            connect_failures=connect_failures,
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
        # dispatch_agent's schema is built at registration time in
        # _tool_definition_from_callable: its agent_type enum depends on the
        # custom-agent catalog and per-collection gates. browser_targets probes
        # live host configuration, so snapshot it here like the rest of the
        # construction-time tool surface.
        self._browser_targets = tuple(getattr(self.browser_manager, "browser_targets", ()))
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

        On a page too large to fit one snapshot, nodes nearest the viewport are shown
        first and a Coverage line states what was left out. That is an instruction to
        narrow the scope — pass a target, or browser_scroll and snapshot again — not a
        sign the page is unreadable.

        Args:
            target: Optional snapshot ref or unique selector for a subtree.
            depth: Optional maximum accessibility-tree depth. Counts emitted nodes
                rather than raw DOM nesting.
        """
        return await self.browser_tool.browser_snapshot(target, depth)

    async def browser_find(self, text: Optional[str] = None, regex: Optional[str] = None) -> str:
        """Find text or a regular expression in the accessibility snapshot.

        Provide exactly one of text or regex. This is cheaper than requesting a
        full snapshot when locating a specific element.

        A miss distinguishes three cases: absent from the page, present in the page
        but outside the region the snapshot covered, and undetermined because the
        search was truncated. Only the first is a reliable absence.

        Args:
            text: Case-insensitive text to find.
            regex: Regular expression to find. The Chrome backend accepts a
                restricted, linear-time subset: literals, '.', character classes,
                anchors, escapes, and the quantifiers ?, *, + and {n,m} applied to
                a single character or class, with at most 4 quantifiers and
                repetition counts up to 1000. Groups '(' ')', alternation '|', and
                backreferences are rejected, so write [0-9]{4} rather than
                (\\d{4}|\\d{2}). The Playwright backend accepts full Python
                regular expressions, so a pattern that works there may be
                rejected on Chrome.
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

        PageDown, PageUp, Home, End, ArrowDown, ArrowUp and Space also scroll the
        page, unless focus is in a text field or select, or the page handles the
        key itself. Use browser_scroll when you want to move the viewport
        deliberately rather than as a side effect of a keystroke.

        Args:
            key: Key name or character, such as ArrowLeft or a.
        """
        return await self.browser_tool.browser_press_key(key)

    async def browser_scroll(
        self,
        target: Optional[str] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
        by_pages: Optional[float] = None,
    ) -> str:
        """Move the viewport, then return the updated page snapshot.

        Supply exactly one movement. On a page too large to snapshot in one call,
        scroll and re-snapshot: the snapshot prioritises what is near the viewport,
        so moving the viewport is how you reach the rest.

        Args:
            target: Scroll this ref or unique selector into view.
            x: Absolute horizontal offset in CSS pixels.
            y: Absolute vertical offset in CSS pixels.
            by_pages: Scroll by this many viewport heights; negative scrolls up.
        """
        return await self.browser_tool.browser_scroll(target, x, y, by_pages)

    async def browser_tabs(self, action: str, index: Optional[int] = None, url: Optional[str] = None) -> str:
        """List, create, close, or select browser tabs.

        The action decides which other argument applies, and anything supplied for
        the rest is ignored: list uses neither, new uses url, and select and close
        use index. Tab indices shift after a close, so re-list before acting again.

        Args:
            action: One of list, new, close, or select.
            index: Tab index, required for select. For close it defaults to the
                current tab. 0 is a real tab index.
            url: URL for a new tab; omit it for a blank tab.
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

    # Agent dispatch (available when a dispatch-enabled tool group is active)
    async def dispatch_agent(
        self,
        agent_type: str,
        task: str,
        model_override: Any = None,
        browser_target: Optional[str] = None,
    ) -> str:
        """Dispatch an autonomous sub-agent to complete a self-contained task.

        The sub-agent works without further input and returns one final report. You
        cannot see its intermediate steps or send follow-up messages, so each task
        must be INDEPENDENT and SELF-CONTAINED: include the goal, relevant file
        paths, constraints, and exactly what the final report should contain. The
        report is not automatically shown to the user - summarize the key results.
        Sub-agents cannot spawn further sub-agents.

        PARALLEL EXECUTION: multiple dispatch_agent calls issued in a single
        response run CONCURRENTLY. Use this to fan out independent work, but never
        give two parallel agents work that could overlap on the same files. Do
        tasks that depend on each other's output sequentially or yourself, and
        skip dispatch for small tasks you can do directly with a couple of tool
        calls or anything needing back-and-forth with the user.

        Args:
            agent_type: Which agent to dispatch, from the listed values.
            task: A detailed, self-contained description of the task to perform
            model_override: Optional complete provider/model/thinking_effort route. Call
                list_subagent_models before selecting one; omit it to inherit defaults.
            browser_target: Only for agent_type "browser". Omit for Playwright; choose
                Chrome only when the user asks to use their Chrome browser.

        Returns:
            The sub-agent's final report
        """
        if agent_type not in self._available_agent_types():
            available = ", ".join(self._available_agent_types()) or "none"
            raise ValueError(f"Unknown or unavailable agent_type `{agent_type}`. Valid values: {available}.")
        if agent_type == "general":
            return await self.agent_tool.dispatch_general_agent(task, model_override)
        if agent_type == "investigation":
            return await self.agent_tool.dispatch_investigation_agent(task, model_override)
        if agent_type == "browser":
            return await self.agent_tool.dispatch_browser_agent(task, model_override, browser_target)
        if agent_type == "coding":
            return await self.agent_tool.dispatch_coding_agent(task, model_override)
        return await self.agent_tool.dispatch_custom_agent(agent_type, task, model_override)

    async def list_subagent_models(self, provider: Optional[str] = None) -> str:
        """List configured models available for ordinary and Gigacode sub-agents.

        Use this before choosing a non-default route. The result contains no
        credentials and dispatches revalidate every selection at execution time.

        Args:
            provider: Optional provider name to filter the catalog. Omit it to
                list every configured provider; a blank value is also treated
                as an unfiltered request.

        Returns:
            A compact Markdown catalog of agent defaults, supported
            provider/model pairs, exact effort options, nullable-effort rules,
            and vision support.
        """
        return await self.agent_tool.list_subagent_models(provider)

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
                A hard runaway backstop, not a target — review/investigation calls spend
                ~6-24k output tokens each and coder calls ~20-80k, so size generously
                or omit.
            script_path: Path to a script file on disk; takes precedence over `script`.
                Draft long scripts in the session scratchpad, not the project tree.
            resume_from_run_id: Resume a prior run, replaying cached agent() results for
                calls whose content is unchanged (matched by call content, not position)
                and running new/changed calls live.

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

    async def list_workflow_runs(self, limit: int = 20) -> str:
        """List this session's gigacode workflow runs, newest first, with each run's
        runId, name, status, timing, tokens, journaled agent calls, and artifact
        paths. Only runs started by the current session are shown.

        Use this to recover a run whose id you no longer have — for example after a
        `run_workflow` call was interrupted ("Operation was interrupted"). Do not
        re-run an interrupted workflow from scratch: resume it with
        `run_workflow(resume_from_run_id=...)` so its journaled calls replay instead
        of re-running. A run listed as "running" when no workflow is actually in
        flight died mid-run and is resumable the same way.

        Args:
            limit: Maximum number of runs to report (default 20).
        """
        return await self.workflow_tool.list_workflow_runs(limit=limit)

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

        1. Use the read tool to understand the file's contents and context.

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
            path: Path to edit: project-relative, using ../ traversal, or absolute.
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

        You must read the file first — or have just written it with write —
        before editing. ``old_string`` must match exactly and be unique unless
        ``replace_all`` is true.

        Args:
            file_path: Path to modify: project-relative, using ../ traversal, or absolute.
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
        Patch paths may be project-relative, use ``../`` traversal, or be absolute.

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
            path: File path: project-relative, using ../ traversal, or absolute.
            edits: Hashline v2 operations for this file.
            delete: Delete the file; cannot be combined with edits or rename.
            rename: Optional destination path: project-relative, using ../ traversal, or absolute.

        Returns:
            A short summary, or fresh tagged context when an anchor is stale.
        """

        return await self.edit_tool.hashline_edit(path, edits, delete, rename)

    async def hashline_write(self, path: str, content: str) -> str:
        """Create or replace a complete file while using Hashline v2.

        Prefer the anchored `edit` tool for changes to an existing file. Use
        this tool for deliberate complete-file writes.

        Args:
            path: Path to create or replace: project-relative, using ../ traversal, or absolute.
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
            path: Path to edit: project-relative, using ../ traversal, or absolute.
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
            file_path: Path to create or overwrite: project-relative, using ../ traversal, or absolute.
            content: Complete content to write to the file.

        Returns:
            A short summary of the write.
        """

        return await self.edit_tool.claude_write(file_path, content)

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
        background: bool = False,
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

        Running long-lived processes: pass background=true for dev servers,
        watchers, and long builds you want to keep running while you do other
        work. It returns after a short startup window with a session_id. The
        process is launched detached, so it keeps running until you stop it with
        kill_command — including after this agent session ends (it does NOT die
        when you finish). Do NOT use shell `&` for this — processes backgrounded
        that way are killed when the command that started them ends. Drive
        background sessions with write_stdin: chars sends real input, chars=""
        polls new output. Input is not echoed (their stdin is not a TTY) and
        never reaches EOF, so commands that read stdin run until kill_command
        stops them. Use list_sessions to see all running shells. Always verify
        a server answers (e.g. curl) before handing its URL to the browser
        agent — do not rely on its log, which may be buffered.

        Args:
            command: Shell command line, executed via `bash -c`.
            workdir: Working directory for the command. Defaults to project root.
            yield_time_ms: How long to wait for output/exit before returning, in
                           milliseconds (clamped to 250–30000). A `timeout`
                           argument is accepted as an alias for this window
                           (values under 1000 are read as seconds, larger
                           values as milliseconds).
            max_output_tokens: Maximum tokens of output to return in this call.
            login: Run the shell as a login shell (sources profile). Default false.
            background: Launch detached and return after a short startup window
                        (~2s) with a session_id. The process outlives this call
                        and the agent session until kill_command stops it; it
                        accepts write_stdin input (no echo; stdin never reaches
                        EOF). Commands that exit within the startup window
                        report their real exit code.

        Returns:
            A JSON object: {"status": "exited"|"running", "exit_code",
            "session_id", "output", "truncated", "original_token_count",
            "duration_ms"}. Background launches that are still running also
            include "background": true and a "note" with management hints.
        """
        return await self.terminal_tool.exec_command(
            command,
            workdir=workdir,
            yield_time_ms=yield_time_ms,
            max_output_tokens=max_output_tokens,
            login=login,
            background=background,
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
        trailing "\\n" to submit a line. Waits up to yield_time_ms for more
        output or for the process to exit.

        Works for background sessions too: input is delivered to their stdin
        but not echoed, so verify the effect from the command's output; their
        stdin never reaches EOF, so stop stdin-reading commands with
        kill_command.

        Args:
            session_id: The id returned by exec_command when status == "running".
            chars: Bytes to write to stdin. An empty string polls only.
            yield_time_ms: How long to wait for more output or the process to
                           exit, in milliseconds (clamped to 250–30000 when
                           writing, 5000–300000 when polling).
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

        Includes sessions started with exec_command background=true, annotated
        with "background": true.

        Returns:
            A JSON object mapping each running session id to its command,
            working directory, runtime in seconds, and background flag.
        """
        return await self.terminal_tool.list_sessions()

    async def eval(
        self,
        code: str,
        language: str = "py",
        title: Optional[str] = None,
        timeout: Optional[float] = None,
        reset: bool = False,
    ) -> Union[str, List[Any]]:
        """Run one step of code in a persistent kernel. State (imports, variables, functions) persists across eval calls, across tool calls, and across sub-agents in this session — each call is one logical step.

        language="py" runs Python in kolega-code's own managed environment (check python_info()): numpy, pandas, matplotlib, and pillow are preinstalled, and pip_install("scipy") adds more. Use tool.exec_command for anything that needs the project's own venv. language="js" runs JavaScript on Bun or Node (>= 18) when available.

        Work incrementally: imports → define → test → use, each its own cell. Re-run setup ONLY after reset or a kernel crash. On error, fix and re-run just the failing step. Prior top-level names survive into the next cell — NEVER re-import or re-declare them.

        Both kernels can call back into your own tools over a loopback bridge — loop tool.exec_command over shell checks, read images with tool.read_image, or dispatch sub-agents without leaving the cell. Bridge calls count as real tool calls (permissions and hooks apply). Use list_tools() in a cell to discover available tool names. tool.* results arrive in each tool's model-facing format (e.g. tool.read wraps content in a markdown header and code fence) — for raw file bytes like CSV/data loads, use the read()/write() helpers instead, which hit the filesystem directly.

        Python prelude (sync; pass kwargs):
          display(value)                 rich output: dict/list → JSON, matplotlib Figure → image
          print(value)                   shows in the cell's stdout
          read(path, offset=1, limit=None)   write(path, content)   env(key=None, value=None)
          tool.<name>(args_dict, **kwargs)   call any session tool (list_tools() shows names)
          parallel([lambda: ..., ...])   run thunks concurrently, results in order
          pip_install(*pkgs)   python_info()   log(msg)   phase(title)
        Top-level await works; do NOT call asyncio.run() inside a coroutine cell.

        JavaScript prelude (async; ONE trailing object literal, never positional args):
          display(value)   read(path, offset, limit)   write(path, content)   env(key, value)
          await tool.<name>({...})   await listTools()
          await parallel([() => ..., ...])   npm_install("pkg")   setGlobal(name, value)
          log(msg)   phase(title)
        Top-level await works; declarations inside await-wrapped cells do NOT persist — use setGlobal(name, value) for cross-cell state there. Redeclaring an existing top-level name errors: assign without redeclaring, or pass reset=true.

        Args:
            code: code to run in this eval call, verbatim. Top-level await is fine.
            language: which kernel to run in; defaults to "py" (the persistent Python kernel). Pass "js" for the persistent JavaScript kernel (needs bun or node >= 18 on PATH).
            title: short label for this step shown in the transcript (e.g. "load csv", "chart by region").
            timeout: timeout for this cell in seconds; 0 disables it. Default 120, max 600.
            reset: wipe this language's kernel before running (fresh state). The other language is untouched.

        Returns:
            The cell's stdout/stderr, the last expression's value (REPL echo), display() outputs (images included when the model supports vision), log()/phase() status lines, and any error with its traceback.
        """
        return await self.eval_tool.eval(language, code, title=title, timeout=timeout, reset=reset)

    async def read(self, file_path: str, offset: int = 1, limit: Optional[int] = None) -> str:
        """
        Read the contents of a file. Output is truncated to 2000 lines or 50KB (whichever is hit first). Use offset/limit to read specific sections of large files. When you need the full file, continue with offset until complete.

        Args:
            file_path: Path to the file. Relative to the project root is preferred; an absolute path is also accepted.
            offset: The 1-indexed first line to read (default 1).
            limit: The maximum number of lines to read; omitted reads from the top.

        Returns:
            The contents of the file as a string formatted as markdown.

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If offset or limit are invalid
        """
        formatter = format_hash_lines if self._hashline_output_enabled() else None
        result = await self.read_file_tool.read(
            file_path=file_path,
            offset=offset,
            limit=limit,
            line_formatter=formatter,
        )
        if self.edit_protocol == EditProtocol.CLAUDE_CODE:
            self.edit_tool.observe_read(file_path)
        return result

    async def write(self, path: str, content: str) -> str:
        """
        Write content to a file in the project.

        This tool creates the file if it does not exist and replaces the entire file if it does.
        For small edits to existing files, prefer edit or multi_edit.

        Args:
            path: Path to write: project-relative, using ../ traversal, or absolute.
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
        content: str,
        path: str = "MEMORY.md",
    ) -> str:
        """Create or overwrite a complete private project-memory entry."""
        return await self.memory_tool.write_memory(content, path)

    async def edit_memory(
        self,
        old_string: str,
        new_string: str,
        path: str = "MEMORY.md",
    ) -> str:
        """Replace one exact, unique occurrence in a private project-memory entry."""
        return await self.memory_tool.edit_memory(old_string, new_string, path)

    async def delete_memory(self, path: str) -> str:
        """Delete a private project-memory entry by path."""
        return await self.memory_tool.delete_memory(path)

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
            path: File path: project-relative, using ../ traversal, or absolute.
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
        writing files. ``path`` and ``new_path`` may be project-relative, use
        ``../`` traversal, or be absolute; local file URIs returned by the
        server may also target files outside the project.
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
        """Get the externally reachable hostname for a service listening on the given
        port in this sandbox. Use it to construct URLs instead of localhost."""
        # Terminal managers are injected at construction and never swapped
        # mid-session, so registration-time gating in `_should_include_tool`
        # guarantees a sandbox here. `sandbox` is intentionally duck-typed
        # (the gate uses getattr, not isinstance), so the base-class type does
        # not declare it.
        return self.terminal_manager.sandbox.get_host(port)  # pyright: ignore[reportAttributeAccessIssue]

    def _tool_definition_from_callable(self, method_name: str, method: Callable[..., Any]) -> ToolDefinition:
        """Build a provider-agnostic tool definition from a Python callable.

        The wire description drops the docstring's ``Args:`` block because the
        per-parameter schema carries the same text (``tool_definition_from_callable``
        strips it unless asked to keep it). Two shapes keep the block: freeform
        bindings, whose schema is a fallback ``input`` envelope rather than the
        real parameters, and tools with an explicit input schema that does not
        describe every property — in both cases the ``Args:`` block is the only
        place their parameters are documented on the wire.
        """
        explicit_schema = self.extension_schemas.get(method_name)
        freeform = method_name == "apply_patch"
        definition = tool_definition_from_callable(
            method_name,
            method,
            keep_args_in_description=freeform
            or (explicit_schema is not None and not schema_has_property_descriptions(explicit_schema)),
        )
        if method_name == "apply_patch":
            definition.description = (
                "Use the `apply_patch` tool to edit files. This is a FREEFORM tool, so do not wrap the patch in JSON. "
                "Patch paths may be project-relative, use ../ traversal, or be absolute."
            )
            definition.input_kind = "freeform"
            definition.freeform_format = {
                "type": "grammar",
                "syntax": "lark",
                "definition": CODEX_APPLY_PATCH_GRAMMAR,
            }
        if method_name == "edit" and self.edit_protocol == EditProtocol.HASHLINE_V2:
            definition.input_schema = _HASHLINE_V2_INPUT_SCHEMA
        if method_name == "dispatch_agent":
            agent_types = self._available_agent_types()
            definition.description = f"{definition.description}\n\n{self._dispatch_agent_type_catalog(agent_types)}"
            definition.input_schema = _dispatch_agent_input_schema(agent_types, browser_targets=self._browser_targets)
        return definition

    # One accurate line per built-in agent_type, appended to dispatch_agent's
    # description for the types this collection actually offers.
    _DISPATCH_AGENT_TYPE_LINES = {
        "general": (
            "- general: full workspace toolset (read, search, edit files, run commands). The default for "
            "independent subtasks such as broad searches or mechanical multi-file edits."
        ),
        "investigation": (
            "- investigation: read-only file access plus terminal commands and eval kernels (it can run and "
            "test code but has no file-editing tools), with lsp and web access. Use it to analyze code, "
            "trace execution paths, and gather findings into a report."
        ),
        "browser": (
            "- browser: navigates and interacts with web pages via the browser_* tools (snapshot, click, "
            "type, screenshot, ...). Use ONLY when the user explicitly asks to browse, visit, or interact "
            "with a web page or requests a screenshot - not for general research or docs lookup."
        ),
        "coding": "- coding: a full coding agent for coding tasks, with streaming output.",
    }

    def _dispatch_agent_type_catalog(self, agent_types: Sequence[str]) -> str:
        """Render the per-type description lines for the available agent_type values."""
        lines = ["Available agent_type values:"]
        custom_names = set(agent_types) - self._DISPATCH_AGENT_TYPE_LINES.keys()
        lines.extend(
            self._DISPATCH_AGENT_TYPE_LINES[agent_type]
            for agent_type in agent_types
            if agent_type in self._DISPATCH_AGENT_TYPE_LINES
        )
        if custom_names:
            catalog = self.caller.custom_agent_catalog
            lines.append(
                "Custom agents (run in a fresh context with a subset of this session's tools; "
                "pick one whose description matches the task):"
            )
            lines.append(catalog.model_catalog())
        return "\n".join(lines)

    def _available_agent_types(self) -> list[str]:
        """agent_type enum members dispatch_agent currently offers.

        Computed from the same conditions that used to gate whole per-type
        dispatch tools: which dispatch-enabled groups the config activates,
        config-level agent-type exclusions, the browser-agent vision gate, and
        the custom-agent catalog (never offered to sub-agents).
        """
        enabled_groups = set(self.tool_config.custom_tool_groups or [])
        if self.tool_config.include_agent_dispatch_tools:
            enabled_groups.add("agent_dispatch_tools")
        admitted: set[str] = set()
        for group_name in enabled_groups:
            admitted.update(self._dispatch_group_agent_types.get(group_name, ()))
        admitted.difference_update(self.tool_config.excluded_agent_types)

        agent_types = [agent_type for agent_type in ("general", "investigation") if agent_type in admitted]
        if "browser" in admitted:
            from kolega_code.agent.browseragent import BrowserAgent

            # Vision gate: the browser agent reads page screenshots, so gate on
            # the model resolved for the browser-agent role (which may differ
            # from the main one), not caller.supports_vision, rather than offer
            # a value that can only fail at dispatch.
            if BrowserAgent.resolved_model_supports_vision(self.config):
                agent_types.append("browser")
        if "coding" in admitted:
            agent_types.append("coding")
        if "custom" in admitted and not getattr(self.caller, "sub_agent", False):
            catalog = getattr(self.caller, "custom_agent_catalog", None)
            if catalog is not None and catalog.has_agents():
                # Built-in type names always win in routing, so a custom agent
                # shadowed by one never enters the enum as a dead value.
                agent_types.extend(name for name in catalog.names() if name not in self._DISPATCH_AGENT_TYPE_LINES)
        return agent_types

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
            "delegation_tools",
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
        context = getattr(self.caller, "sub_agent_context", None)
        workflow_dispatch = method_name in self.agent_dispatch_tools and (
            validated_workflow_depth(context) is not None or has_workflow_context_marker(context)
        )
        ordinary_parallel_dispatch = (
            method_name in self.agent_dispatch_tools and method_name not in self._legacy_only_extension_dispatch_tools
        )
        return Tool(
            name=method_name,
            definition=definition,
            handler=method,
            groups=self._groups_for(method_name),
            # Workflow parents lend their admitted execution-chain slot to one
            # child at a time. Ordinary delegation remains parallel-safe.
            parallel_safe=(method_name in (self.read_only_tools or []) or ordinary_parallel_dispatch)
            and not workflow_dispatch,
            log_debug=self._log_alias_hit,
        )

    async def _log_alias_hit(self, message: str) -> None:
        """Debug log_message sink for Tool param-alias hits (trajectory mining)."""
        await self.log_debug(message, sender=getattr(self.caller, "agent_name", "agent"))

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
        # Gigacode delegation depth is scoped through sub_agent_context. It only
        # narrows a workflow worker's existing dispatch inventory; callers with no
        # workflow context retain their normal behavior.
        if method_name in self.agent_dispatch_tools:
            sub_agent_context = getattr(self.caller, "sub_agent_context", None)
            workflow_depth = validated_workflow_depth(sub_agent_context)
            if has_workflow_context_marker(sub_agent_context):
                if workflow_depth is None:
                    return False
                depth, maximum = workflow_depth
                if method_name in self._extension_dispatch_tools or depth >= maximum:
                    return False

        if method_name in self.delegation_tools and not self._caller_can_delegate_or_author_workflow():
            return False

        # dispatch_agent only surfaces when it has at least one agent_type to
        # offer; the per-type gates live in _available_agent_types.
        if method_name == "dispatch_agent" and not self._available_agent_types():
            return False

        if self.tool_config.allowed_tools is not None and method_name not in self.tool_config.allowed_tools:
            return False

        if method_name in self.delegation_tools:
            return True

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

        # Sandbox gate: get_host needs a sandbox host provider, which only
        # sandbox terminal managers carry. Duck-type via getattr (not
        # isinstance) so kolega-code-e2b's injected SandboxTerminalManager —
        # possibly subclassed from a pinned older version — keeps qualifying.
        # Terminal managers are injected at construction and never swapped
        # mid-session, so registration-time gating is sufficient.
        if method_name == "get_host" and getattr(self.terminal_manager, "sandbox", None) is None:
            return False

        # Web tool gate: the client-side web tools drop out when the agent's
        # web_search_mode resolved to hosted (the provider's server-side tool
        # replaces them) or off. Exclusion-only — an enabled tool still passes
        # through the group/read-only/browser-only filtering below. Default True
        # so agents without the attribute (tests, custom callers) keep today's
        # tool set.
        if method_name in ("web_search", "web_fetch") and not getattr(self.caller, "client_web_tools_enabled", True):
            return False

        # Settings gate: eval can be disabled host-wide (AgentConfig.eval_enabled).
        # Strict `is False` so Mock configs in tests keep the default (enabled).
        if method_name == "eval" and getattr(self.config, "eval_enabled", True) is False:
            return False

        # Settings gate: LSP can be disabled host-wide (AgentConfig.lsp.enabled, from
        # settings.json or the --lsp CLI flag). The manager is never built when
        # disabled, so drop lsp/lsp_edit/resolve instead of advertising tools that
        # can only reply "not available". resolve is LSP-coupled: pending actions
        # are only ever created by lsp_edit(apply=false), so none can exist while
        # LSP is off. Strict `is False` so Mock configs keep the default.
        if (
            method_name in ("lsp", "lsp_edit", "resolve")
            and getattr(getattr(self.config, "lsp", None), "enabled", True) is False
        ):
            return False

        if method_name in self.browser_tools:
            supported_tools = getattr(self.browser_manager, "supported_tools", None)
            if supported_tools is not None and method_name not in supported_tools:
                return False

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
            if (
                method_name in self.agent_dispatch_tools
                and method_name not in self._legacy_only_extension_dispatch_tools
                and self.tool_config.include_agent_dispatch_tools
            ):
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
            if method_name in self._legacy_only_extension_dispatch_tools:
                return True
            return self.tool_config.include_agent_dispatch_tools

        # Check memory tools
        if method_name in self.memory_tools:
            # Include memory tools if explicitly enabled, or if memory tools are not excluded
            return self.tool_config.include_memory_tools or method_name not in self.tool_exclusions

        # Include all other core tools by default
        return True

    def _caller_can_delegate_or_author_workflow(self) -> bool:
        """Whether this collection belongs to a non-leaf routing-capable caller."""
        if not getattr(self.caller, "sub_agent", False) and bool(getattr(self.caller, "gigacode_enabled", False)):
            return True

        context = getattr(self.caller, "sub_agent_context", None)
        if has_workflow_context_marker(context):
            depth = validated_workflow_depth(context)
            if depth is None or depth[0] >= depth[1]:
                return False

        enabled_groups = set(self.tool_config.custom_tool_groups or [])
        dispatch_candidates = set()
        if self.tool_config.include_agent_dispatch_tools:
            dispatch_candidates.update(self.agent_dispatch_tools)
        for group_name in enabled_groups:
            dispatch_candidates.update(set(getattr(self, group_name, []) or []) & set(self.agent_dispatch_tools))

        dispatch_candidates.difference_update(self.tool_exclusions)
        if self.tool_config.allowed_tools is not None:
            dispatch_candidates.intersection_update(self.tool_config.allowed_tools)
        if "dispatch_agent" in dispatch_candidates and not self._available_agent_types():
            dispatch_candidates.discard("dispatch_agent")
        return bool(dispatch_candidates)

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

            # Clean up eval kernel resources (only the top-level agent owns the
            # session-shared kernel manager; EvalTool enforces that internally).
            if hasattr(self, "eval_tool"):
                await self.eval_tool.shutdown_if_owner()
                await self.log_info("Cleaned up eval kernel resources", sender="ToolCollection")

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
            await self.cleanup_tool_extensions(self.tool_extensions)

        except Exception as e:
            await self.log_error(f"Error during tool cleanup: {str(e)}", sender="ToolCollection")

        self.extension_callbacks = {}
        self.extension_schemas = {}
