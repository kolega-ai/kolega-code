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

# Configuration
from .agent.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig

# Events and connection management
from .agent.connection_manager import AgentConnectionManager
from .agent.models.public import AgentEvent, AgentStatus

# LLM clients and message models
from .agent.llm.client import LLMClient
from .agent.llm.instrumented_client import InstrumentedLLMClient
from .agent.llm.models import (
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
    PromptContext,
    PromptExtension,
    PromptProvider,
)

# Tools
from .agent.tool_backend.base_tool import BaseTool
from .agent.tools import ToolCollection, ToolCollectionConfig, ToolExtension

# Host environment services
from .agent.common import LogMixin
from .agent.services.base import BrowserManager, TerminalManager
from .agent.services.browser import PlaywrightBrowserManager
from .agent.services.file_system import FileSystem, LocalFileSystem
from .agent.services.terminal import LocalTerminalManager

# Sandbox abstractions (implemented by provider packages such as kolega-code-e2b)
from .agent.services.sandbox.base import ProjectManifest, SandboxConfig, SandboxManager
from .agent.services.sandbox.local_sandbox import LocalSandboxManager
from .agent.services.sandbox.terminal_state_serializer import TerminalStateSerializer
from .agent.services.sandbox.utils import (
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
    "PromptContext",
    "PromptExtension",
    "PromptProvider",
    # Tools
    "BaseTool",
    "ToolCollection",
    "ToolCollectionConfig",
    "ToolExtension",
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
    "SandboxManager",
    "LocalSandboxManager",
    "TerminalStateSerializer",
    "get_git_diff_from_sandbox",
    "get_modified_files_from_sandbox",
    "parse_git_status_output",
]

__version__ = "0.1.0"
