"""Tool: a definition paired with its implementation and execution metadata."""

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, FrozenSet

from ..llm.models import ToolDefinition


class ToolError(Exception):
    """
    An expected tool failure.

    Raise this from a tool implementation when the failure is a normal
    outcome the model should see and react to (file not found, invalid
    arguments, command failed). The executor reports the message as an
    is_error tool result without treating it as an internal fault.
    """


@dataclass(frozen=True)
class Tool:
    """A single callable tool: provider-facing definition plus implementation."""

    name: str
    definition: ToolDefinition
    handler: Callable[..., Awaitable[Any]]
    groups: FrozenSet[str] = field(default_factory=frozenset)
    # Safe to run concurrently with other parallel-safe tools in one batch
    # (read-only operations and independent sub-agent dispatches).
    parallel_safe: bool = False

    async def call(self, **inputs: Any) -> Any:
        return await self.handler(**inputs)
