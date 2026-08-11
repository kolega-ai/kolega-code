import inspect
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Union

from .common import LogMixin
from kolega_code.config import AgentConfig, EditProtocol
from kolega_code.llm.models import ImageBlock, ToolDefinition
from kolega_code.memory import ProjectMemoryManager
from kolega_code.tools import Tool, ToolRegistry
from kolega_code.scratchpad import expand_scratchpad_reference
from kolega_code.services.file_system import FileSystem, LocalFileSystem
from kolega_code.services.base import TerminalManager, BrowserManager
from kolega_code.services.terminal import LocalTerminalManager
from kolega_code.services.browser import PlaywrightBrowserManager
from .tool_backend.agent_tool import AgentTool
from .tool_backend.browser_tool import BrowserTool
from .tool_backend.edit_tool import EditTool
from .tool_backend.eval_tool import EvalTool
from .tool_backend.hashline_v2 import format_hash_lines
from .tool_backend.memory_tool import MemoryTool
from .tool_backend.network_status import ConnectFailureTracker
from .tool_backend.read_file_tool import ReadFileTool
from .tool_backend.read_image_tool import ReadImageTool
from .tool_backend.snapshot_tool import SnapshotTool
from .tool_backend.web_fetch_tool import WebFetchTool
from .tool_backend.web_search_tool import WebSearchTool
from .tool_backend.terminal_tool import TerminalTool
from .tool_backend.workflow_tool import WorkflowTool
from .edit_protocols import EDIT_HANDLER_NAMES, edit_protocol_spec
from .tool_definitions import builtin_tool_definition, dispatch_agent_input_schema
from .orchestration.context import has_workflow_context_marker, validated_workflow_depth

# Import additional tools for consolidated functionality
from .tool_backend.build_tool import BuildTool
from .tool_backend.lsp_tool import LspEditTool, LspTool
from kolega_code.services.lsp import LspManager
from kolega_code.services.snapshots import SnapshotService

# Canonical path parameters of the built-in file tools. The model-facing names
# differ across edit protocols (``path`` vs ``file_path``), so expansion covers
# every spelling the bound handler can accept; alias resolution has already
# collapsed wrong-name arguments onto these by the time they are expanded.
FILE_TOOL_PATH_PARAMS = frozenset({"path", "file_path", "rename"})

# Built-in tools whose path arguments may reference the session scratchpad as
# ``$KOLEGA_SCRATCHPAD``. The dispatch choke point (``Tool.call``) expands
# those references to the caller's real scratchpad directory before the
# handler runs, so a literal ``$KOLEGA_SCRATCHPAD`` directory can never appear
# in the workspace when the model passes the shell spelling to a file tool.
FILE_TOOLS_WITH_PATH_PARAMS = frozenset(
    {
        "write",
        "edit",
        "multi_edit",
        "read",
        "read_image",
        "claude_edit",
        "claude_write",
        "hashline_edit",
        "hashline_write",
    }
)


@dataclass(frozen=True)
class ToolExtension:
    """Host-provided tool callbacks and named groups.

    A tool's definition is declared data: ``tool_descriptions`` and
    ``tool_schemas`` must cover every callback in ``tools``, and both are used
    verbatim on the wire. Nothing is inferred from the callable.

    ``exclusive_tools`` declares session-control callbacks that must be the
    sole tool call in a model response. If one is batched with another call,
    the complete batch is rejected before any callback executes.
    """

    name: str
    tools: dict[str, Callable[..., Any]]
    tool_groups: dict[str, List[str]] = field(default_factory=dict)
    # Model-visible description per tool, used verbatim on the wire. Required
    # for every tool: registration fails without it. An empty string is a
    # deliberate declaration (used by internal report-only tools), a missing
    # key is an authoring error.
    tool_descriptions: dict[str, str] = field(default_factory=dict)
    # Complete JSON input schema per tool, used verbatim on the wire. Required
    # for every tool: registration fails without it.
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
        self.extension_descriptions = {}
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
                missing = [
                    part
                    for part, declared in (
                        ("tool_descriptions", tool_name in extension.tool_descriptions),
                        ("tool_schemas", tool_name in extension.tool_schemas),
                    )
                    if not declared
                ]
                if missing:
                    raise ValueError(
                        f"Tool extension '{extension.name}' declares tool '{tool_name}' without "
                        f"{' and '.join(missing)}. Extension tools must declare an explicit "
                        "model-visible description and a complete JSON input schema; nothing is "
                        "inferred from the callable."
                    )
                setattr(self, tool_name, callback)
                self.extension_callbacks[tool_name] = callback
                self.extension_descriptions[tool_name] = extension.tool_descriptions[tool_name]

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
        # dispatch_agent's schema is built at registration time in
        # _builtin_tool_definition: its agent_type enum depends on the
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
        """Navigate the current browser tab to a URL, starting a session when needed."""
        return await self.browser_tool.browser_navigate(url)

    async def browser_navigate_back(self) -> str:
        """Go back to the previous page and return the updated page snapshot."""
        return await self.browser_tool.browser_navigate_back()

    async def browser_snapshot(self, target: Optional[str] = None, depth: Optional[int] = None) -> str:
        """Capture the current page's accessibility snapshot."""
        return await self.browser_tool.browser_snapshot(target, depth)

    async def browser_find(self, text: Optional[str] = None, regex: Optional[str] = None) -> str:
        """Find text or a regular expression in the accessibility snapshot."""
        return await self.browser_tool.browser_find(text, regex)

    async def browser_wait_for(
        self, time: Optional[float] = None, text: Optional[str] = None, text_gone: Optional[str] = None
    ) -> str:
        """Wait for time to pass, text to appear, or text to disappear."""
        return await self.browser_tool.browser_wait_for(time, text, text_gone)

    async def browser_resize(self, width: int, height: int) -> str:
        """Resize the current browser viewport."""
        return await self.browser_tool.browser_resize(width, height)

    async def browser_click(
        self,
        target: str,
        double_click: bool = False,
        button: str = "left",
        modifiers: Optional[list[str]] = None,
    ) -> str:
        """Click an element identified by a snapshot ref or unique selector."""
        return await self.browser_tool.browser_click(target, double_click, button, modifiers)

    async def browser_type(self, target: str, text: str, submit: bool = False, slowly: bool = False) -> str:
        """Enter text into an editable element."""
        return await self.browser_tool.browser_type(target, text, submit, slowly)

    async def browser_fill_form(self, fields: list[dict[str, Any]]) -> str:
        """Fill several textbox, checkbox, radio, combobox, or slider fields."""
        return await self.browser_tool.browser_fill_form(fields)

    async def browser_select_option(self, target: str, values: list[str]) -> str:
        """Select one or more values in a dropdown."""
        return await self.browser_tool.browser_select_option(target, values)

    async def browser_hover(self, target: str) -> str:
        """Hover over an element and return the updated snapshot."""
        return await self.browser_tool.browser_hover(target)

    async def browser_drag(self, start_target: str, end_target: str) -> str:
        """Drag one page element to another."""
        return await self.browser_tool.browser_drag(start_target, end_target)

    async def browser_drop(
        self, target: str, paths: Optional[list[str]] = None, data: Optional[dict[str, str]] = None
    ) -> str:
        """Drop workspace files or MIME-typed string data onto an element."""
        return await self.browser_tool.browser_drop(target, paths, data)

    async def browser_press_key(self, key: str) -> str:
        """Press a keyboard key in the current tab."""
        return await self.browser_tool.browser_press_key(key)

    async def browser_scroll(
        self,
        target: Optional[str] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
        by_pages: Optional[float] = None,
    ) -> str:
        """Move the viewport, then return the updated page snapshot."""
        return await self.browser_tool.browser_scroll(target, x, y, by_pages)

    async def browser_tabs(self, action: str, index: Optional[int] = None, url: Optional[str] = None) -> str:
        """List, create, close, or select browser tabs."""
        return await self.browser_tool.browser_tabs(action, index, url)

    async def browser_handle_dialog(self, accept: bool, prompt_text: Optional[str] = None) -> str:
        """Accept or dismiss the currently waiting JavaScript dialog."""
        return await self.browser_tool.browser_handle_dialog(accept, prompt_text)

    async def browser_file_upload(self, paths: list[str]) -> str:
        """Upload workspace files through the currently waiting file chooser."""
        return await self.browser_tool.browser_file_upload(paths)

    async def browser_console_messages(self, level: str = "info", all_messages: bool = False) -> str:
        """Return console messages for the current tab."""
        return await self.browser_tool.browser_console_messages(level, all_messages)

    async def browser_network_requests(self, include_static: bool = False, filter_pattern: Optional[str] = None) -> str:
        """List network requests made by the current tab since navigation."""
        return await self.browser_tool.browser_network_requests(include_static, filter_pattern)

    async def browser_network_request(self, index: int, part: Optional[str] = None) -> str:
        """Return headers or body details for one indexed network request."""
        return await self.browser_tool.browser_network_request(index, part)

    async def browser_take_screenshot(
        self,
        target: Optional[str] = None,
        image_type: str = "png",
        full_page: bool = False,
        scale: str = "css",
    ) -> List[ImageBlock]:
        """Capture a visual screenshot of the page or one element."""
        result = await self.browser_tool.browser_take_screenshot(target, image_type, full_page, scale)
        return [ImageBlock(image_type="base64", media_type=result["media_type"], data=result["image"])]

    async def read_image(self, path: str) -> List[Any]:
        """Read an image file from the project directory so you can see it."""
        return await self.read_image_tool.read_image(path)

    async def browser_evaluate(self, function: str, target: Optional[str] = None) -> str:
        """Evaluate JavaScript in the page or against one target element."""
        return await self.browser_tool.browser_evaluate(function, target)

    async def browser_close(self) -> str:
        """Close the current browser session and release its resources."""
        return await self.browser_tool.browser_close()

    async def build_backend(self) -> str:
        """Build the backend defined by the project manifest."""
        return await self.build_tool.build_backend()

    async def build_frontend(self) -> str:
        """Build the frontend defined by the project manifest."""
        return await self.build_tool.build_frontend()

    # Agent dispatch (available when a dispatch-enabled tool group is active)
    async def dispatch_agent(
        self,
        agent_type: str,
        task: str,
        model_override: Any = None,
        browser_target: Optional[str] = None,
    ) -> str:
        """Dispatch an autonomous sub-agent to complete a self-contained task."""
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
        """List configured models available for ordinary and Gigacode sub-agents."""
        return await self.agent_tool.list_subagent_models(provider)

    async def run_workflow(
        self,
        script: str = "",
        args: Any = None,
        token_budget: int = 0,
        script_path: str = "",
        resume_from_run_id: str = "",
    ) -> str:
        """Run a gigacode workflow script that orchestrates many sub-agents."""
        return await self.workflow_tool.run_workflow(
            script=script,
            args=args,
            token_budget=token_budget,
            script_path=script_path,
            resume_from_run_id=resume_from_run_id,
        )

    async def list_workflow_runs(self, limit: int = 20) -> str:
        """List this session's gigacode workflow runs, newest first."""
        return await self.workflow_tool.list_workflow_runs(limit=limit)

    async def edit(self, path: str, block: str) -> str:
        """Edit a file using one search and replace block."""
        return await self.edit_tool.edit(path, block)

    async def claude_edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """Perform an exact string replacement in a file."""

        return await self.edit_tool.claude_edit(file_path, old_string, new_string, replace_all)

    async def apply_patch(self, input: str) -> str:
        """Use the `apply_patch` tool to edit files."""
        return await self.edit_tool.apply_patch(input)

    async def hashline_edit(
        self,
        path: str,
        edits: list[dict[str, object]],
        delete: bool = False,
        rename: Optional[str] = None,
    ) -> str:
        """Apply precise edits using Hashline v2 ``LINE#ID`` anchors."""

        return await self.edit_tool.hashline_edit(path, edits, delete, rename)

    async def hashline_write(self, path: str, content: str) -> str:
        """Create or replace a complete file while using Hashline v2."""

        return await self.edit_tool.hashline_write(path, content)

    async def multi_edit(self, path: str, blocks: str) -> str:
        """Edit a file using one or more search and replace blocks."""
        return await self.edit_tool.multi_edit(path, blocks)

    async def claude_write(self, file_path: str, content: str) -> str:
        """Write a file, overwriting it if it already exists."""

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
        """Run a shell command as a fresh process and return its output."""
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
        """Write input to a running session's stdin and read recent output."""
        return await self.terminal_tool.write_stdin(
            session_id, chars, yield_time_ms=yield_time_ms, max_output_tokens=max_output_tokens
        )

    async def kill_command(self, session_id: str, signal: str = "TERM") -> str:
        """Terminate a running session and its process group."""
        return await self.terminal_tool.kill_command(session_id, signal)

    async def list_sessions(self) -> str:
        """List currently running exec sessions."""
        return await self.terminal_tool.list_sessions()

    async def eval(
        self,
        code: str,
        language: str = "py",
        title: Optional[str] = None,
        timeout: Optional[float] = None,
        reset: bool = False,
    ) -> Union[str, List[Any]]:
        """Run one step of code in a persistent Python or JavaScript kernel."""
        return await self.eval_tool.eval(language, code, title=title, timeout=timeout, reset=reset)

    async def read(self, file_path: str, offset: int = 1, limit: Optional[int] = None) -> str:
        """Read the contents of a file."""
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
        """Write content to a file in the project."""
        return await self.edit_tool.write(path, content)

    async def snapshot(
        self,
        action: str = "list",
        snapshot_id: str = "",
        paths: Optional[list[str]] = None,
        force: bool = False,
        limit: int = 20,
    ) -> str:
        """Manage file snapshots for undo, inspection, and manual checkpoints."""
        return await self.snapshot_tool.snapshot(
            action=action,
            snapshot_id=snapshot_id,
            paths=paths,
            force=force,
            limit=limit,
        )

    async def resolve(self, action_id: str, decision: str, force: bool = False) -> str:
        """Apply or discard a pending preview action."""
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
        """Fetch URL content locally, follow an instruction, and return a grounded response."""
        return await self.web_fetch_tool.web_fetch(url, instruction)

    async def web_search(self, query: str, max_results: int = 5) -> str:
        """Search the web and return a ranked list of results (title, URL, and a short snippet)."""
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
        """Query language server intelligence: diagnostics, definition, references, hover, symbols, status."""
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
        """Apply trusted LSP edits such as rename, file rename, formatting, and code actions."""
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
        """Get the externally reachable hostname for a sandbox service port."""
        # Terminal managers are injected at construction and never swapped
        # mid-session, so registration-time gating in `_should_include_tool`
        # guarantees a sandbox here. `sandbox` is intentionally duck-typed
        # (the gate uses getattr, not isinstance), so the base-class type does
        # not declare it.
        return self.terminal_manager.sandbox.get_host(port)  # pyright: ignore[reportAttributeAccessIssue]

    def _builtin_tool_definition(self, spec_key: str, wire_name: str) -> ToolDefinition:
        """Build the declared definition for a built-in tool.

        Definitions are explicit artifacts (see ``kolega_code.agent.tool_definitions``);
        nothing is derived from the handler. dispatch_agent is template plus data:
        its static base description gains the per-collection agent_type catalog,
        and its schema is built from the same agent_type computation.
        """
        definition = builtin_tool_definition(spec_key, name=wire_name)
        if spec_key == "dispatch_agent":
            agent_types = self._available_agent_types()
            definition.description = f"{definition.description}\n\n{self._dispatch_agent_type_catalog(agent_types)}"
            definition.input_schema = dispatch_agent_input_schema(agent_types, browser_targets=self._browser_targets)
        return definition

    def _extension_tool_definition(self, method_name: str) -> ToolDefinition:
        """Build a host-extension tool definition from its declared artifacts."""
        return ToolDefinition(
            name=method_name,
            description=self.extension_descriptions[method_name],
            parameters=[],
            input_schema=self.extension_schemas[method_name],
        )

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
            registry.add(
                self._build_tool(binding.name, getattr(self, binding.handler_name), definition_key=binding.handler_name)
            )

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

    def _build_tool(self, method_name: str, method: Callable[..., Any], definition_key: Optional[str] = None) -> Tool:
        if method_name in self.extension_callbacks:
            definition = self._extension_tool_definition(method_name)
        else:
            definition = self._builtin_tool_definition(definition_key or method_name, method_name)
        context = getattr(self.caller, "sub_agent_context", None)
        workflow_dispatch = method_name in self.agent_dispatch_tools and (
            validated_workflow_depth(context) is not None or has_workflow_context_marker(context)
        )
        ordinary_parallel_dispatch = (
            method_name in self.agent_dispatch_tools and method_name not in self._legacy_only_extension_dispatch_tools
        )
        path_params = FILE_TOOL_PATH_PARAMS if method_name in FILE_TOOLS_WITH_PATH_PARAMS else frozenset()
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
            path_params=path_params,
            path_expander=self._expand_scratchpad_path if path_params else None,
        )

    def _expand_scratchpad_path(self, path: str) -> str:
        """Expand a ``$KOLEGA_SCRATCHPAD`` reference to the caller's scratchpad.

        Applied at the dispatch choke point to file-tool path arguments, so a
        literal ``$KOLEGA_SCRATCHPAD`` directory is never created in the
        workspace when the model passes the shell spelling to a file tool.
        Callers without a real scratchpad directory (including plain Mocks in
        tests) leave the path untouched.
        """
        scratchpad = getattr(self.caller, "scratchpad_dir", None)
        if not isinstance(scratchpad, (str, Path)):
            return path
        return expand_scratchpad_reference(path, scratchpad)

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

        Definitions come from the enabled tools' declared artifacts; the last
        definition carries the prompt-cache checkpoint.
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
        # Settings gate: sub-agent dispatch can be disabled host-wide
        # (AgentConfig.subagents_enabled, from settings.json or the --subagents
        # flag). Placed first so it also beats the custom_tool_groups whitelist
        # below (planning/investigation agents) and covers host extension
        # dispatch tools appended to this instance's agent_dispatch_tools.
        # Gigacode workflow workers are exempt: their dispatch chain is lent by
        # run_workflow (a separate opt-in) and depth-narrowed below.
        # Strict `is False` so Mock configs in tests keep the default (enabled).
        if (
            method_name in self.agent_dispatch_tools
            and getattr(self.config, "subagents_enabled", True) is False
            and not has_workflow_context_marker(getattr(self.caller, "sub_agent_context", None))
        ):
            return False

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
        # Sub-agent dispatch disabled host-wide (AgentConfig.subagents_enabled):
        # no dispatch candidates remain, so the informational
        # list_subagent_models tool drops out as well. Gigacode paths are
        # unaffected: the top-level workflow-authoring branch above already
        # qualified, and a workflow worker with delegation depth left keeps the
        # dispatch chain lent by run_workflow.
        if getattr(self.config, "subagents_enabled", True) is False and not has_workflow_context_marker(context):
            dispatch_candidates.clear()
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
