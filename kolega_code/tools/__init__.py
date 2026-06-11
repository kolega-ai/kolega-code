"""First-class tool primitives.

Tools are data: a Tool pairs a provider-facing ToolDefinition with the
coroutine that implements it, plus execution metadata (groups, parallel
safety). A ToolRegistry holds Tools and answers availability, dispatch,
and definition queries; ToolPolicy expresses name-based selection.

ToolCollection (kolega_code.agent.tools) remains the host-facing way to
assemble an agent's tools; internally it builds a ToolRegistry.
"""

from .core import Tool, ToolError
from .definitions import tool_definition_from_callable
from .registry import ToolPolicy, ToolRegistry

__all__ = [
    "Tool",
    "ToolError",
    "ToolPolicy",
    "ToolRegistry",
    "tool_definition_from_callable",
]
