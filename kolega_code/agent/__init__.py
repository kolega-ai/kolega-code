# Export base agent
from .baseagent import BaseAgent

# Export concrete agents
from .coder import CoderAgent
from .investigationagent import InvestigationAgent
from .browseragent import BrowserAgent
from .generalagent import GeneralAgent
from .planningagent import PlanningAgent
from .custom_agents import CustomAgent, CustomAgentCatalog, CustomAgentDefinition

# Export agent errors
from .errors import AgentError, MaxAgentIterationsExceeded

# Export agent models
from kolega_code.events import AgentStatus, AgentEvent

# Export configuration
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider, RateLimitConfig
from kolega_code.events import AgentConnectionManager
from kolega_code.services.base import TerminalManager
from .prompt_provider import MissingPromptTemplateError, PromptExtension, PromptProvider
from .tools import ToolExtension

__all__ = [
    "BaseAgent",
    "CoderAgent",
    "InvestigationAgent",
    "BrowserAgent",
    "GeneralAgent",
    "PlanningAgent",
    "CustomAgent",
    "CustomAgentCatalog",
    "CustomAgentDefinition",
    "AgentError",
    "MaxAgentIterationsExceeded",
    "AgentStatus",
    "AgentEvent",
    "AgentConfig",
    "ModelConfig",
    "RateLimitConfig",
    "ModelProvider",
    "AgentConnectionManager",
    "TerminalManager",
    "MissingPromptTemplateError",
    "PromptExtension",
    "PromptProvider",
    "ToolExtension",
]

__version__ = "0.17.0"
