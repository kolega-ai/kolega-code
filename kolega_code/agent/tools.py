import asyncio
import inspect
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Union

from .common import LogMixin
from kolega_code.config import AgentConfig
from kolega_code.llm.models import ImageBlock, ToolDefinition
from kolega_code.tools import Tool, ToolRegistry, tool_definition_from_callable
from kolega_code.services.file_system import FileSystem, LocalFileSystem
from kolega_code.services.base import TerminalManager, BrowserManager
from kolega_code.services.terminal import LocalTerminalManager
from kolega_code.services.browser import PlaywrightBrowserManager
from .tool_backend.agent_tool import AgentTool
from .tool_backend.apply_patch_tool import APPLY_PATCH_TOOL_DESC, ApplyPatchTool
from .tool_backend.browser_tool import BrowserTool
from .tool_backend.create_file_tool import CreateFileTool
from .tool_backend.glob_tool import GlobTool
from .tool_backend.list_directory_tool import ListDirectoryTool
from .tool_backend.memory_tool import MemoryTool
from .tool_backend.read_file_tool import ReadFileTool
from .tool_backend.replace_entire_file_tool import (
    ReplaceEntireFileTool,
)
from .tool_backend.replace_lines_tool import ReplaceLinesTool
from .tool_backend.search_and_replace_tool import SearchAndReplaceTool
from .tool_backend.search_codebase_tool import SearchCodebaseTool
from .tool_backend.web_fetch_tool import WebFetchTool
from .tool_backend.terminal_tool import TerminalTool
from .tool_backend.think_hard_tool import ThinkHardTool

# Import additional tools for consolidated functionality
from .tool_backend.build_tool import BuildTool


@dataclass(frozen=True)
class ToolExtension:
    """Host-provided tool callbacks and named groups."""

    name: str
    tools: dict[str, Callable[..., Any]]
    tool_groups: dict[str, List[str]] = field(default_factory=dict)


class ToolCollectionConfig:
    """Configuration class for customizing tool availability per agent type."""

    def __init__(
        self,
        read_only: bool = False,
        browser_only: bool = False,
        include_agent_dispatch_tools: bool = False,
        include_memory_tools: bool = False,
        tool_exclusions: List[str] = None,
        custom_tool_groups: List[str] = None,
        enabled_tool_groups: List[str] = None,
        restrict_to_tool_groups: bool = False,
    ):
        """
        Initialize tool collection configuration.

        Args:
            read_only: Whether to restrict to read-only tools
            browser_only: Whether to only include browser tools
            include_agent_dispatch_tools: Whether to include agent dispatch tools (investigation, browser, coding)
            include_memory_tools: Whether to include memory management tools
            tool_exclusions: List of method names to exclude from tool list
            custom_tool_groups: Additional custom tool groups to include
            enabled_tool_groups: Additional custom tool groups to include
            restrict_to_tool_groups: If True, ONLY include tools from specified groups, excluding all other core tools
        """
        self.read_only = read_only
        self.browser_only = browser_only
        self.include_agent_dispatch_tools = include_agent_dispatch_tools
        self.include_memory_tools = include_memory_tools
        self.tool_exclusions = tool_exclusions or []
        self.custom_tool_groups = list(dict.fromkeys((custom_tool_groups or []) + (enabled_tool_groups or [])))
        self.restrict_to_tool_groups = restrict_to_tool_groups


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
        "search_codebase",
        "find_files_by_pattern",
        "think_hard",
        "web_fetch",
        "sleep",
    ]

    browser_tools = [
        "launch_browser",
        "list_browsers",
        "get_browser_interactive_elements",
        "get_browser_console_logs",
        "take_browser_screenshot",
        "interact_with_browser",
        "set_browser_select_value",
        "close_browser",
    ]

    # Agent dispatch tools group - includes all agent dispatch functionality
    agent_dispatch_tools = [
        "dispatch_investigation_agent",
        "dispatch_browser_agent",
        "dispatch_coding_agent",
        "dispatch_general_agent",
    ]

    # Legacy name for backward compatibility
    investigation_agent_tools = agent_dispatch_tools

    # CoderAgent specific dispatch tools
    coder_agent_tools = [
        "dispatch_investigation_agent",
        "dispatch_browser_agent",
        "dispatch_general_agent",
    ]

    # Memory tools group
    memory_tools = [
        "read_memory",
        "write_memory",
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
        self.langfuse_client = langfuse_client
        self.tool_extensions = tool_extensions or []
        self.extension_callbacks = {}
        self._extension_group_names = set()

        # Set legacy attributes for backward compatibility
        self.read_only = tool_config.read_only
        self.browser_only = tool_config.browser_only

        # Build tool exclusions list from config
        self.tool_exclusions = [
            "read_memory",
            "write_memory",
            "execute_terminal_command",
            "replace_lines",
            "apply_patch",
            "get_tool_list",
            "log_error",
            "log_warning",
            "log_info",
            "run_command",  # Disabled: unreliable completion detection, use run_command_tracked instead
        ]
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

            for group_name, tool_names in extension.tool_groups.items():
                existing_group = list(getattr(self, group_name, []))
                merged_group = list(dict.fromkeys(existing_group + list(tool_names)))
                setattr(self, group_name, merged_group)
                self._extension_group_names.add(group_name)

    def _initialize_tools(self):
        """Initialize all tool backends based on configuration."""
        # Core tool backends (always available)
        self.think_hard_tool = ThinkHardTool(
            self.project_path,
            self.workspace_id,
            self.thread_id,
            self.connection_manager,
            self.config,
            self.caller,
            self.filesystem,
        )
        self.search_and_replace_tool = SearchAndReplaceTool(
            self.project_path,
            self.workspace_id,
            self.thread_id,
            self.connection_manager,
            self.config,
            self.caller,
            self.filesystem,
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
        self.memory_tool = MemoryTool(
            self.project_path,
            self.workspace_id,
            self.thread_id,
            self.connection_manager,
            self.config,
            self.caller,
            self.filesystem,
        )
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
        self.create_file_tool = CreateFileTool(
            self.project_path,
            self.workspace_id,
            self.thread_id,
            self.connection_manager,
            self.config,
            self.caller,
            self.filesystem,
        )
        self.replace_entire_file_tool = ReplaceEntireFileTool(
            self.project_path,
            self.workspace_id,
            self.thread_id,
            self.connection_manager,
            self.config,
            self.caller,
            self.filesystem,
        )
        self.replace_lines_tool = ReplaceLinesTool(
            self.project_path,
            self.workspace_id,
            self.thread_id,
            self.connection_manager,
            self.config,
            self.caller,
            self.filesystem,
        )
        self.apply_patch_tool = ApplyPatchTool(
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
        )
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

    async def launch_browser(self, url: str) -> str:
        """
        Launch a browser and navigate to a specified URL.

        This tool opens a new browser window, navigates to the provided URL,
        and returns a unique browser ID that can be used to interact with this browser instance
        through other browser-related tools.

        When to use this tool:
        - When you need to visit a website to gather information
        - When you need to interact with web applications
        - When you need to test web functionality
        - When you need to demonstrate web-based features to the user

        Usage notes:
        1. The browser uses a standard viewport size (1280x720) and Chrome user agent
        2. The returned browser ID must be saved if you plan to interact with this browser later
        3. Each call creates a new browser instance - use judiciously to avoid resource consumption

        Args:
            url: The complete URL to navigate to (must include http:// or https://)

        Returns:
            A confirmation message with the unique browser ID for future reference
        """

        return await self.browser_tool.launch_browser(url)

    async def list_browsers(self) -> str:
        """
        List all currently running browser instances.

        This tool provides a formatted overview of all active browser sessions, displaying
        their unique browser IDs, the URLs they're currently visiting, and when they were launched.

        When to use this tool:
        - When you need to check which browser instances are currently active
        - When you need to retrieve a browser ID for use with other browser tools
        - When you want to see which URLs are currently being accessed
        - When you need to manage multiple browser sessions

        Usage notes:
        1. The output is formatted as a markdown table for easy readability
        2. If no browsers are running, the tool will indicate this
        3. Browser IDs can be used with other browser tools like close_browser
        4. This tool is useful for cleanup to ensure all browser instances are properly closed

        Returns:
            A markdown-formatted table listing all active browser instances with their details
        """
        return await self.browser_tool.list_browsers()

    async def get_browser_console_logs(
        self,
        browser_id: str,
        max_logs: int = 50,
        log_types: list = None,
        minutes_back: int = None,
        max_chars: int = 8000,
    ) -> str:
        """
        Retrieve filtered console logs from a browser instance by its browser ID.

        This tool captures console messages (info, warnings, errors) that have been logged
        in the browser's JavaScript console and returns them in a formatted markdown document.
        The logs are filtered to prevent context window overflow while focusing on the most relevant information.

        When to use this tool:
        - When you need to debug JavaScript errors on a webpage
        - When you want to see application messages logged to the console
        - When you need to diagnose network or rendering issues
        - When you're working with web applications that use console logging

        Usage notes:
        1. You must provide a valid browser_id from a previous launch_browser call
        2. Console logs are filtered by default to show only errors, warnings, and assertions
        3. By default, only the most recent 50 logs are returned with a character limit of 8000
        4. Each log entry includes its type, timestamp, and message text
        5. Use this after interacting with a page to see what messages were generated

        Args:
            browser_id: The unique identifier of the browser instance to get console logs from
            max_logs: Maximum number of logs to return (default: 50, most recent)
            log_types: List of log types to include (default: ['error', 'warning', 'assert'])
            minutes_back: Only return logs from the last N minutes (optional)
            max_chars: Maximum total character count for all log messages (default: 8000)

        Returns:
            A markdown-formatted document containing the filtered browser console logs
        """
        return await self.browser_tool.get_browser_console_logs(
            browser_id, max_logs=max_logs, log_types=log_types, minutes_back=minutes_back, max_chars=max_chars
        )

    async def get_browser_interactive_elements(self, browser_id: str) -> str:
        """
        Identify and extract all interactive elements from a browser page.

        This tool analyzes the current state of a browser page and identifies all interactive elements
        such as buttons, links, form inputs, and other clickable components, returning them in a
        structured markdown format with their selectors and attributes.

        When to use this tool:
        - When you need to discover what actions are possible on a webpage
        - When you need to find specific interactive elements to interact with
        - When you're exploring a new website and need to understand its interface
        - When you need to automate interactions with a webpage
        - When you need precise selectors for use with the interact_with_browser or set_browser_select_value tools

        Usage notes:
        1. You must provide a valid browser_id from a previous launch_browser call
        2. The tool returns a comprehensive list of all interactive elements with their types, text content, and selectors
        3. The selector column provides CSS selectors that can be used with interact_with_browser or set_browser_select_value
        4. The attributes column provides additional information about each element
        5. Use this tool before performing interactions to identify the correct elements to target

        Args:
            browser_id: The unique identifier of the browser instance to analyze

        Returns:
            A markdown-formatted document listing all interactive elements on the page with their details
        """
        return await self.browser_tool.get_browser_interactive_elements(browser_id)

    async def take_browser_screenshot(self, browser_id: str) -> str:
        """
        Take a screenshot of the current browser page.

        This tool captures the current visual state of a browser page and returns it as an image,
        along with relevant metadata such as the current URL and page title.

        When to use this tool:
        - When you need to visually inspect the current state of a webpage
        - When you need to capture visual evidence of a web application's behavior
        - When text-based content extraction is insufficient to understand the page layout
        - When you need to verify the visual appearance of a web interface

        Usage notes:
        1. You must provide a valid browser_id from a previous launch_browser call
        2. The screenshot captures the entire visible viewport of the browser
        3. The returned image is in base64-encoded format
        4. The tool also returns metadata about the page including title and URL

        Args:
            browser_id: The unique identifier of the browser instance to screenshot

        Returns:
            A list containing a text description and the screenshot image
        """
        result = await self.browser_tool.take_browser_screenshot(browser_id)

        # Create an image block with the screenshot data
        image_block = ImageBlock(image_type="base64", media_type="image/png", data=result["screenshot"])

        return [image_block]

    async def interact_with_browser(
        self, browser_id: str, action: str, selector: str, text: str, scroll_px: int
    ) -> str:
        """
        Interact with a browser by performing actions on web elements.

        This tool allows you to control a browser programmatically by executing common actions
        like clicking elements, typing text, or navigating to new URLs. It provides a way to
        automate web interactions within an existing browser session.

        When to use this tool:
        - When you need to click buttons, links, or other interactive elements on a webpage
        - When you need to fill out forms by typing text into input fields
        - When you need to navigate to a different URL within an existing browser session
        - When you need to automate a sequence of interactions with a web application

        When NOT to use this tool:
        - When you need to interact with a dropdown or select input. Use set_browser_select_value for that.

        Usage notes:
        1. You must provide a valid browser_id from a previous launch_browser call
        2. The action parameter must be one of: 'click', 'type', 'scroll' or 'navigate'
        3. For 'click' actions, provide a CSS or XPath selector that identifies the element to click
        4. For 'type' actions, provide both a selector for the input field and the text to type
        5. For 'scroll' actions, provide a scroll_px (positive to scroll down the page, negative to scroll up)
        5. For 'navigate' actions, provide the URL in the text parameter (selector can be empty)
        6. The tool waits for the page to stabilize after the action before returning
        7. The return value includes the current URL after the action is performed

        Args:
            browser_id: The unique identifier of the browser instance to interact with
            action: The type of interaction to perform ('click', 'type', or 'navigate')
            selector: CSS or XPath selector identifying the element to interact with
            text: Text to type (for 'type' action) or URL to navigate to (for 'navigate' action)

        Returns:
            A markdown-formatted report of the interaction result, including the current URL
        """
        return await self.browser_tool.interact_with_browser(browser_id, action, selector, text, scroll_px)

    async def set_browser_select_value(self, browser_id: str, selector: str, value: str) -> str:
        """
        Set the value of a select box (dropdown) in a browser page.

        This tool allows you to programmatically select an option from a dropdown menu (select element)
        on a webpage. It validates that the element is indeed a select box and that the specified value
        exists among the available options before making the selection.

        When to use this tool:
        - When you need to select an option from a dropdown menu on a form
        - When you need to change the selected value in a select box
        - When automating form filling that includes dropdown selections
        - When you need to test different options in a select element

        Usage notes:
        1. You must provide a valid browser_id from a previous launch_browser call
        2. The selector must identify a <select> HTML element - the tool will fail if used on other element types
        3. The value parameter should match the 'value' attribute of the <option> you want to select, not the visible text
        4. Use get_browser_interactive_elements first to find the correct selector and see available option values
        5. The tool validates that the specified value exists in the select options before attempting to set it
        6. The response will confirm whether the selection was successful and show the actual selected value

        Args:
            browser_id: The unique identifier of the browser instance to interact with
            selector: CSS selector that uniquely identifies the select element
            value: The value attribute of the option to select (not the display text)

        Returns:
            A markdown-formatted report showing the result of the selection, including success/error status
        """
        return await self.browser_tool.set_browser_select_value(browser_id, selector, value)

    async def close_browser(self, browser_id: str) -> str:
        """
        Close a specific browser instance by its ID.

        This tool terminates a browser session that was previously launched with the launch_browser tool,
        freeing up system resources and cleaning up the browser process.

        When to use this tool:
        - When you've completed tasks in a specific browser instance
        - When you need to clean up resources after web-based operations
        - When you want to start fresh with a new browser session
        - When you're managing multiple browser instances and need to close specific ones

        Usage notes:
        1. You must provide a valid browser ID that was returned from a previous launch_browser call
        2. Once closed, the browser ID becomes invalid and cannot be used again
        3. It's good practice to close browsers when you're done with them to free up resources
        4. If you're unsure which browser IDs are available, use the list_browsers tool first

        Args:
            browser_id: The unique identifier of the browser instance to close

        Returns:
            A confirmation message indicating the browser has been closed
        """
        return await self.browser_tool.close_browser(browser_id)

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
            - launch_browser
            - list_browsers
            - get_browser_content
            - get_browser_console_logs
            - take_browser_screenshot
            - interact_with_browser
            - set_browser_select_value
            - close_browser

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
        6. Use read_terminal or other status-checking tools after sleeping to verify completion
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

    async def search_and_replace(self, relative_path: str, block: str) -> str:
        """
        Edit a file using a search and replace block.

        The block should be formatted as follows:
        ```
        <<<<<<< SEARCH
        [original code to find]
        =======
        [new code to replace with]
        >>>>>>> REPLACE
        ```

        Before using this tool:

        1. Use the read_entire_file tool to understand the file's contents and context

        To make a file edit, provide the following:
        1. relative_path: The absolute path to the file to modify (must be absolute, not relative)
        2. block: The search and replace block, as specified above. The text to replace (must be unique within the file, and must match the file contents exactly, including all whitespace and indentation)

        The tool will replace ONE occurrence of old_string with new_string in the specified file.

        CRITICAL REQUIREMENTS FOR USING THIS TOOL:

        1. UNIQUENESS: The old_string MUST uniquely identify the specific instance you want to change. This means:
        - Include AT LEAST 3-5 lines of context BEFORE the change point
        - Include AT LEAST 3-5 lines of context AFTER the change point
        - Include all whitespace, indentation, and surrounding code exactly as it appears in the file

        2. SINGLE INSTANCE: This tool can only change ONE instance at a time. If you need to change multiple instances:
        - Make separate calls to this tool for each instance
        - Each call must uniquely identify its specific instance using extensive context

        3. VERIFICATION: Before using this tool:
        - Check how many instances of the target text exist in the file
        - If multiple instances exist, gather enough context to uniquely identify each one
        - Plan separate tool calls for each instance

        WARNING: If you do not follow these requirements:
        - The tool will fail if block matches multiple locations
        - The tool will fail if block doesn't match exactly (including whitespace)
        - You may change the wrong instance if you don't include enough context

        When making edits:
        - Ensure the edit results in idiomatic, correct code
        - Do not leave the code in a broken state

        If you want to create a new file, use the create_file tool.

        THE INDENTATION IN THE SEARCH BLOCK MUST BE IDENTICAL TO THE EXISTING FILE.

        Args:
            relative_path: Path to the file to edit, relative to the project root
            block: A single search and replace blocks formatted as shown above

        Returns:
            A summary of the update made to the file

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the search block doesn't match any content in the file
            ValueError: If the block is malformed or incorrectly formatted
            ValueError: If the block matches more than one place in the file
            PermissionError: If the file cannot be written to
        """
        return await self.search_and_replace_tool.search_and_replace(relative_path, block)

    async def list_directory(self, relative_path: str = "") -> str:
        """
        List files and directories at the specified path.

        Args:
            relative_path: Path to list, relative to the project root

        Returns:
            Markdown formatted list of files and directories with details

        Raises:
            NotADirectoryError: If the path is not a directory
        """
        return await self.list_directory_tool.list_directory(relative_path)

    async def execute_terminal_command(self, command: str) -> str:
        """Execute a command and display output in terminal."""
        return await self.terminal_tool.execute_terminal_command(command)

    async def launch_terminal(self, terminal_id: Optional[str] = None) -> str:
        """
        Launch a new terminal session.

        This tool creates a new terminal instance that can be used to execute commands.
        The terminal persists between commands, maintaining environment variables,
        working directory, and other state.

        Args:
            terminal_id: Optional ID for the terminal. If not provided, a random UUID will be generated.
                         Use this ID with run_command, read_terminal and close_terminal to interact with
                         this specific terminal.

        Returns:
            The ID of the created terminal that can be used in subsequent terminal operations
        """
        return await self.terminal_tool.launch_terminal(terminal_id)

    async def run_command(self, terminal_id: str, command: str, purpose: str) -> str:
        """
        Run a command in a specific terminal session and wait for output.

        This tool sends a command to an existing terminal session and returns true or false
        if the command was accepted. The tool does not return the terminal output. You must call
        read_terminal to get the output.

        This tool can be used to run long-running processes that do not exit, such as a development server.

        CRITICAL WARNINGS:
        1. **NEVER** send commands to a terminal that has a process still running (e.g., dev server, watch mode, etc.)
        2. If a terminal is running jest --watch, npm run dev, or any other persistent process, that terminal is BLOCKED
        3. To check if a terminal is blocked, use read_terminal first - if it shows an active process, DO NOT send commands
        4. For new commands while something is running, you MUST launch a new terminal with launch_terminal

        IMPORTANT NOTES:
        - If you change directory using cd in a terminal, the terminal will remain in that directory
        - Start a new terminal if you want to be sure of being in the project directory
        - Use list_terminals to see all active terminals and their last commands

        Args:
            terminal_id: The ID of the terminal session to use (must be created with launch_terminal first)
            command: The command to execute in the terminal
            purpose: The reason to run the command including what information you hope to get when you read the terminal

        Returns:
            Success message if command was accepted, or error if terminal is blocked/unavailable.
        """
        return await self.terminal_tool.run_command(terminal_id, command, purpose=purpose)

    async def read_terminal(self, terminal_id: str, num_chars: int = 1024, offset: int = 0) -> str:
        """
        Read the output from a specific terminal session.

        This tool retrieves output from a terminal session's persistent buffer, reading
        the most recent characters up to the specified limit. It does not wait for command
        completion - it returns whatever output is currently available.

        Args:
            terminal_id: The ID of the terminal session to read from (must be created with launch_terminal first)
            num_chars: Number of characters to read from the output buffer (default: 1024).
                      If buffer is smaller than num_chars, returns entire buffer.
            offset: Number of characters from the end to start reading from (default: 0).
                   If offset is 0, reads the last num_chars characters.
                   If offset is > 0, reads num_chars characters starting from that offset from the end.
                   Note: When offset > 0, compression is bypassed to allow reading specific portions.

        Returns:
            The terminal output as a string, formatted in markdown code blocks
        """
        return await self.terminal_tool.read_terminal(terminal_id, num_chars=num_chars, offset=offset)

    async def close_terminal(self, terminal_id: str) -> str:
        """
        Close a specific terminal session.

        This tool terminates a terminal session that was previously created with launch_terminal.
        It will kill any running processes in that terminal and clean up associated resources.
        Once closed, the terminal ID cannot be used again and a new terminal must be launched
        if needed.

        Args:
            terminal_id: The ID of the terminal session to close

        Returns:
            A confirmation message indicating the terminal was successfully closed
        """
        return await self.terminal_tool.close_terminal(terminal_id)

    async def list_terminals(self) -> str:
        """
        List all active terminal sessions and their status.

        This tool provides information about all currently active terminal sessions,
        including their IDs, running status, and the last command executed in each terminal.
        Use this to keep track of multiple terminal sessions and their state.

        Returns:
            A formatted string containing a table of all terminal sessions with their IDs,
            status (Running/Stopped), and last executed command
        """
        return await self.terminal_tool.list_terminals()

    async def run_command_tracked(self, terminal_id: str, command: str, purpose: str) -> str:
        """
        Run a command in a terminal with completion tracking.

        This is the standard tool for executing commands in terminals. It provides
        a command ID that allows you to monitor completion status and ensures
        reliable execution for both quick and long-running commands.

        The returned command ID enables you to:
        - Check if the command has finished with check_command_status
        - Wait for completion with wait_for_command_completion
        - Monitor progress for long-running operations

        Best practices:
        - Provide a clear purpose describing what you expect the command to accomplish
        - Save the returned command ID for monitoring if needed
        - Use wait_for_command_completion for commands that subsequent steps depend on

        Args:
            terminal_id: The ID of the terminal to run the command in
            command: The command to execute in the terminal
            purpose: Description of what the command is meant to accomplish

        Returns:
            Command ID for tracking completion, or error message if command couldn't be started
        """
        return await self.terminal_tool.run_command_tracked(terminal_id, command, purpose)

    async def send_terminal_input(
        self, terminal_id: str, text: str, submit: bool = True, command_id: Optional[str] = None
    ) -> str:
        """
        Send input to an already-running terminal command.

        Use this tool when a command started with run_command_tracked is waiting for
        interactive input, such as a confirmation prompt or a password prompt. Read
        the terminal first to confirm it is waiting, then send the exact response.

        This tool does not start a new command and does not echo or store the input
        text in terminal output.

        Args:
            terminal_id: The ID of the terminal where the command is running
            text: Text to send to the running process
            submit: Whether to append a newline before sending (default: true)
            command_id: Optional command ID when more than one command is active

        Returns:
            Confirmation that input was sent, or an error explaining why it could not be sent
        """
        return await self.terminal_tool.send_terminal_input(terminal_id, text, submit=submit, command_id=command_id)

    async def check_command_status(self, terminal_id: str, command_id: str) -> str:
        """
        Check if a command has finished running and get its results.

        Use this tool to check the status of commands started with run_command_tracked.
        The status will show whether the command is still running, completed successfully,
        or terminated with an error.

        Typical workflow:
        1. Start a command with run_command_tracked to get a command ID
        2. Use this tool to check if the command has finished
        3. Read the terminal output once the command is complete

        Args:
            terminal_id: The ID of the terminal where the command is running
            command_id: The command ID returned from run_command_tracked

        Returns:
            Formatted status showing completion state, duration, and exit code
        """
        return await self.terminal_tool.check_command_status(terminal_id, command_id)

    async def check_terminal_status(self, terminal_id: str) -> str:
        """
        Get an overview of a terminal's current state and active commands.

        Use this tool to see what commands are currently running in a terminal
        and whether the terminal is ready to accept new commands. This helps
        you avoid conflicts when managing multiple concurrent operations.

        When to use:
        - Before starting new commands to ensure the terminal is available
        - To see all active commands and their progress
        - To troubleshoot why a terminal might not be responding
        - To get an overview of terminal activity

        Args:
            terminal_id: The ID of the terminal to check

        Returns:
            Formatted report showing terminal status and all active commands with their progress
        """
        return await self.terminal_tool.check_terminal_status(terminal_id)

    async def wait_for_command_completion(self, terminal_id: str, command_id: str, timeout: Optional[int] = 120) -> str:
        """
        Wait for a command to finish before continuing with other tasks.

        Use this tool when you need to ensure a command completes before proceeding.
        This is essential for workflows where subsequent steps depend on the command
        results, such as running tests before deployment or building before serving.

        The tool will block execution until the command finishes or the timeout expires.
        Timeout defaults to 120 seconds and is capped at 300 seconds. If the timeout
        expires, the command is left running in the terminal and the response tells you
        how to check it again with check_command_status.
        After completion, you can read the terminal output to see the results.

        Common scenarios:
        - Wait for test suites to complete before analyzing results
        - Wait for build processes to finish before starting servers
        - Ensure setup commands complete before running the main application
        - Wait for package installations to finish before using new dependencies

        Args:
            terminal_id: The ID of the terminal where the command is running
            command_id: The command ID returned from run_command_tracked
            timeout: Maximum time to wait in seconds (default: 120, capped at 300)

        Returns:
            Completion status message or timeout notification with follow-up status-check guidance
        """
        return await self.terminal_tool.wait_for_command_completion(terminal_id, command_id, timeout)

    async def read_entire_file(self, relative_path: str) -> str:
        """
        Read the contents of a file in the project.

        Note: Files exceeding 2000 lines will be truncated with a warning message.
        Use read_file_section to read specific portions of large files.

        Args:
            relative_path: Path to the file, relative to the project root

        Returns:
            The contents of the file as a string formatted as markdown

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        return await self.read_file_tool.read_entire_file(relative_path)

    async def read_file_section(self, relative_path: str, start_line: int, end_line: int) -> str:
        """
        Read a specific section of a file in the project from start_line to end_line (inclusive).

        Args:
            relative_path: Path to the file, relative to the project root
            start_line: The line number to start reading from (1-indexed)
            end_line: The line number to stop reading at (1-indexed, inclusive)

        Returns:
            The specified section of the file as a string formatted as markdown

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If start_line or end_line are invalid
        """
        return await self.read_file_tool.read_file_section(relative_path, start_line, end_line)

    async def create_file(self, relative_path: str, content: str) -> str:
        """
        Create a new file in the project with the given content.

        Args:
            relative_path: Path to the file to create, relative to the project root
            content: Content to write to the new file

        Returns:
            The contents of the created file as a string formatted as markdown

        Raises:
            FileExistsError: If the file already exists
            ValueError: If the parent directory doesn't exist
            PermissionError: If the file cannot be created due to permissions
        """
        return await self.create_file_tool.create_file(relative_path, content)

    async def replace_entire_file(self, relative_path: str, content: str) -> str:
        """
        Replace the entire contents of a file in the project.

        Args:
            relative_path: Path to the file, relative to the project root
            content: New content to write to the file

        Returns:
            The updated contents of the file as a string formatted as markdown

        Raises:
            FileNotFoundError: If the file doesn't exist
            PermissionError: If the file cannot be written to
        """
        return await self.replace_entire_file_tool.replace_entire_file(relative_path, content)

    async def replace_lines(self, relative_path: str, start_line: int, end_line: int, new_content: str) -> str:
        """
        Replace a range of lines in a file with new content.

        Args:
            relative_path: Path to the file, relative to the project root
            start_line: The starting line number (1-indexed)
            end_line: The ending line number (1-indexed, inclusive)
            new_content: The new content to replace the specified lines with

        Returns:
            The updated contents of the file as a string formatted as markdown

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the line range is invalid
            PermissionError: If the file cannot be written to
        """
        return await self.replace_lines_tool.replace_lines(relative_path, start_line, end_line, new_content)

    async def apply_patch(self, input: str) -> str:
        """
        This is a custom utility that makes it more convenient to add, remove, move, or edit code files. `apply_patch` effectively allows you to execute a diff/patch against a file, but the format of the diff specification is unique to this task, so pay careful attention to these instructions. To use the `apply_patch` command, you should pass a message of the following structure as "input":

        %%bash
        apply_patch <<"EOF"
        *** Begin Patch
        [YOUR_PATCH]
        *** End Patch
        EOF

        Where [YOUR_PATCH] is the actual content of your patch, specified in the following V4A diff format.

        *** [ACTION] File: [path/to/file] -> ACTION can be one of Add, Update, or Delete.
        For each snippet of code that needs to be changed, repeat the following:
        [context_before] -> See below for further instructions on context.
        - [old_code] -> Precede the old code with a minus sign.
        + [new_code] -> Precede the new, replacement code with a plus sign.
        [context_after] -> See below for further instructions on context.

        For instructions on [context_before] and [context_after]:
        - By default, show 3 lines of code immediately above and 3 lines immediately below each change. If a change is within 3 lines of a previous change, do NOT duplicate the first change's [context_after] lines in the second change's [context_before] lines.
        - If 3 lines of context is insufficient to uniquely identify the snippet of code within the file, use the @@ operator to indicate the class or function to which the snippet belongs. For instance, we might have:
        @@ class BaseClass
        [3 lines of pre-context]
        - [old_code]
        + [new_code]
        [3 lines of post-context]

        - If a code block is repeated so many times in a class or function such that even a single @@ statement and 3 lines of context cannot uniquely identify the snippet of code, you can use multiple `@@` statements to jump to the right context. For instance:

        @@ class BaseClass
        @@ 	def method():
        [3 lines of pre-context]
        - [old_code]
        + [new_code]
        [3 lines of post-context]

        Note, then, that we do not use line numbers in this diff format, as the context is enough to uniquely identify code. An example of a message that you might pass as "input" to this function, in order to apply a patch, is shown below.

        %%bash
        apply_patch <<"EOF"
        *** Begin Patch
        *** Update File: pygorithm/searching/binary_search.py
        @@ class BaseClass
        @@     def search():
        -          pass
        +          raise NotImplementedError()

        @@ class Subclass
        @@     def search():
        -          pass
        +          raise NotImplementedError()

        *** End Patch
        EOF
        """
        return await self.apply_patch_tool.apply_patch(input)

    async def read_memory(self) -> str:
        """
        Read the contents of the AGENT_MEMORY.md file which serves as the agent's memory.

        Returns:
            The contents of the AGENT_MEMORY.md file as a string

        Raises:
            FileNotFoundError: If the AGENT_MEMORY.md file doesn't exist
        """
        return await self.memory_tool.read_memory()

    async def write_memory(self, memory_content: str) -> str:
        """
        Write a new memory to the AGENT_MEMORY.md file which serves as the agent's memory.

        The memory is added as a markdown bullet point to the file.

        Args:
            memory_content: The memory content to add to the file

        Returns:
            A confirmation message indicating success

        Raises:
            PermissionError: If the file cannot be written to
            Exception: If any other error occurs during writing
        """
        return await self.memory_tool.write_memory(memory_content)

    async def search_codebase(
        self, pattern: str, file_pattern: str = "*", case_sensitive: bool = False, literal: bool = True
    ) -> str:
        """
        Search the codebase for files containing a specific pattern (grep functionality).

        Args:
            pattern: The pattern to search for in files
            file_pattern: Optional glob pattern to filter which files to search (default: all files)
            case_sensitive: Whether the search should be case-sensitive (default: False)
            literal: Whether to treat the pattern as literal text (True) or as a regular expression (False) (default: True)

        Returns:
            Markdown formatted list of files and matches, limited to 128 results

        Raises:
            Exception: If any error occurs during the search operation
        """
        return await self.search_codebase_tool.search_codebase(
            pattern, file_pattern=file_pattern, case_sensitive=case_sensitive, literal=literal
        )

    async def web_fetch(self, url: str, instruction: str) -> str:
        """
        Fetch web page content, follow an instruction, and return a concise response.

        This tool downloads the specified URL, extracts readable text using Trafilatura,
        and asks the fast LLM model to apply the provided instruction. Useful for gathering
        information from public web pages without launching an interactive browser session.

        Args:
            url: Full http(s) URL to fetch.
            instruction: Guidance for how to use the extracted content.

        Returns:
            The model's response derived from the fetched content, truncated to an internal character limit if needed.
        """
        return await self.web_fetch_tool.web_fetch(url, instruction)

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
        # Check if we're using a SandboxTerminalManager (indicates sandbox mode)
        if hasattr(self.terminal_manager, "sandbox") and self.terminal_manager.sandbox:
            # We're in sandbox mode, get the host from the sandbox
            sandbox = self.terminal_manager.sandbox
            # E2B AsyncSandbox has a get_host method that takes a port
            # The method is synchronous and returns a string directly
            host = sandbox.get_host(port)
            return host
        else:
            # Local mode, return localhost
            return f"localhost:{port}"

    def _tool_definition_from_callable(self, method_name: str, method: Callable[..., Any]) -> ToolDefinition:
        """Build a provider-agnostic tool definition from a Python callable."""
        return tool_definition_from_callable(
            method_name, method, description_overrides={"apply_patch": APPLY_PATCH_TOOL_DESC}
        )

    def _groups_for(self, method_name: str) -> frozenset:
        """Group tags for a tool, from the core group lists plus extension groups."""
        group_attrs = {
            "read_only_tools",
            "browser_tools",
            "agent_dispatch_tools",
            "coder_agent_tools",
            "memory_tools",
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
            if method_name.startswith("_") or method_name in self.tool_exclusions:
                continue
            if not self._should_include_tool(method_name):
                continue
            registry.add(self._build_tool(method_name, method))

        for method_name, method in self.extension_callbacks.items():
            if method_name in registry or method_name in self.tool_exclusions:
                continue
            if not self._should_include_tool(method_name):
                continue
            registry.add(self._build_tool(method_name, method))

        return registry

    def _build_tool(self, method_name: str, method: Callable[..., Any]) -> Tool:
        return Tool(
            name=method_name,
            definition=self._tool_definition_from_callable(method_name, method),
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

    async def call(self, name: str, **inputs: Any) -> Any:
        """Dispatch an enabled tool by name."""
        return await self.registry().call(name, **inputs)

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

    async def cleanup(self):
        """Clean up all tool resources"""
        try:
            # Clean up terminal resources
            if hasattr(self, "terminal_tool") and hasattr(self.terminal_tool, "terminal_manager"):
                await self.terminal_tool.terminal_manager.cleanup_all()
                print("Cleaned up terminal resources")

            # Clean up any browser resources
            if hasattr(self, "browser_tool") and hasattr(self.browser_tool, "cleanup"):
                await self.browser_tool.cleanup()
                print("Cleaned up browser resources")

            # Clean up any sub-agents
            if hasattr(self, "agent_tool") and hasattr(self.agent_tool, "agents"):
                for agent_id, agent in list(self.agent_tool.agents.items()):
                    if hasattr(agent, "cleanup"):
                        try:
                            await agent.cleanup()
                            print(f"Cleaned up sub-agent: {agent_id}")
                        except Exception as e:
                            print(f"Error cleaning up sub-agent {agent_id}: {e}")

        except Exception as e:
            await self.log_error(f"Error during tool cleanup: {str(e)}", sender="ToolCollection")

        self.extension_callbacks = {}
