"""Shared Kolega agent runtime package.

Everything importable directly from ``kolega_code`` is the supported public
API for host applications (KolegaPlatform, kolega-comply) and sandbox
providers (kolega-code-e2b). Deeper module paths are internal and may move
between releases; import from here instead.
"""

# Agents
from .agent.baseagent import BaseAgent
from .agent.browseragent import BrowserAgent
from .agent.coder import CoderAgent
from .agent.generalagent import GeneralAgent
from .agent.investigationagent import InvestigationAgent
from .agent.planningagent import PlanningAgent

# Agent collaborators
from .agent.compression import HistoryCompressor
from .agent.context import AgentContext, AgentServices, Telemetry, WorkspaceInfo
from .agent.conversation import Conversation
from .events import AgentEventEmitter

# Configuration
from .config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig

# Events and connection management
from .events import AgentConnectionManager
from .events import AgentEvent, AgentStatus

# LLM clients and message models
from .llm.client import LLMClient
from .llm.instrumented_client import InstrumentedLLMClient
from .llm.models import (
    ImageBlock,
    Message,
    MessageHistory,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)

# Prompts
from .agent.prompt_provider import (
    AgentMode,
    AgentType,
    MissingPromptTemplateError,
    PromptContext,
    PromptExtension,
    PromptProvider,
)

# Tools
from .agent.tool_backend.base_tool import BaseTool
from .agent.tools import ToolCollection, ToolCollectionConfig, ToolExtension
from .tools import Tool, ToolError, ToolPolicy, ToolRegistry, tool_definition_from_callable

# Host environment services
from .agent.common import LogMixin
from .services.base import BrowserManager, TerminalManager
from .services.browser import PlaywrightBrowserManager
from .services.file_system import FileSystem, LocalFileSystem
from .services.terminal import LocalTerminalManager

# Sandbox abstractions (implemented by provider packages such as kolega-code-e2b)
from .sandbox.base import ProjectManifest, SandboxConfig, SandboxHandle, SandboxManager
from .sandbox.local import LocalSandboxManager
from .sandbox.serializer import TerminalStateSerializer
from .sandbox.utils import (
    get_git_diff_from_sandbox,
    get_modified_files_from_sandbox,
    parse_git_status_output,
)

__all__ = [
    # Agents
    "BaseAgent",
    "BrowserAgent",
    "CoderAgent",
    "GeneralAgent",
    "InvestigationAgent",
    "PlanningAgent",
    # Agent collaborators
    "AgentContext",
    "AgentServices",
    "Telemetry",
    "WorkspaceInfo",
    "Conversation",
    "HistoryCompressor",
    "AgentEventEmitter",
    # Configuration
    "AgentConfig",
    "ModelConfig",
    "ModelProvider",
    "RateLimitConfig",
    # Events and connection management
    "AgentConnectionManager",
    "AgentEvent",
    "AgentStatus",
    # LLM
    "LLMClient",
    "InstrumentedLLMClient",
    "ImageBlock",
    "Message",
    "MessageHistory",
    "TextBlock",
    "ThinkingBlock",
    "ToolCall",
    "ToolDefinition",
    "ToolParameter",
    "ToolResult",
    # Prompts
    "AgentMode",
    "AgentType",
    "MissingPromptTemplateError",
    "PromptContext",
    "PromptExtension",
    "PromptProvider",
    # Tools
    "BaseTool",
    "ToolCollection",
    "ToolCollectionConfig",
    "ToolExtension",
    "Tool",
    "ToolError",
    "ToolPolicy",
    "ToolRegistry",
    "tool_definition_from_callable",
    # Services
    "LogMixin",
    "BrowserManager",
    "TerminalManager",
    "PlaywrightBrowserManager",
    "FileSystem",
    "LocalFileSystem",
    "LocalTerminalManager",
    # Sandbox
    "ProjectManifest",
    "SandboxConfig",
    "SandboxHandle",
    "SandboxManager",
    "LocalSandboxManager",
    "TerminalStateSerializer",
    "get_git_diff_from_sandbox",
    "get_modified_files_from_sandbox",
    "parse_git_status_output",
]

__version__ = "0.3.2"
