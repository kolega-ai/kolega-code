"""ToolRegistry: holds Tools; answers availability, dispatch, and definitions."""

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterator, List, Optional

from ..llm.models import ToolDefinition
from .core import Tool


@dataclass(frozen=True)
class ToolPolicy:
    """Name-based tool selection: optional allowlist plus exclusions."""

    include: Optional[FrozenSet[str]] = None  # None = all registered tools
    exclude: FrozenSet[str] = field(default_factory=frozenset)

    def allows(self, name: str) -> bool:
        if name in self.exclude:
            return False
        if self.include is not None and name not in self.include:
            return False
        return True


class ToolRegistry:
    """An ordered collection of Tools with lookup, selection, and dispatch."""

    def __init__(self, tools: Optional[List[Tool]] = None) -> None:
        self._tools: Dict[str, Tool] = {}
        for tool in tools or []:
            self.add(tool)

    def add(self, *tools: Tool) -> "ToolRegistry":
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"Tool '{tool.name}' is already registered")
            self._tools[tool.name] = tool
        return self

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def select(self, policy: ToolPolicy) -> "ToolRegistry":
        return ToolRegistry([tool for tool in self if policy.allows(tool.name)])

    async def call(self, tool_name: str, /, **inputs: Any) -> Any:
        """Dispatch a tool by name. Raises KeyError for unknown tools."""
        return await self.get(tool_name).call(**inputs)

    def definitions(self) -> List[ToolDefinition]:
        """
        Provider-facing definitions, with the prompt-cache checkpoint on the
        last definition (and cleared everywhere else, since definitions may
        be shared between registry views).
        """
        definitions = [tool.definition for tool in self]
        for definition in definitions:
            definition.cache_checkpoint = False
        if definitions:
            definitions[-1].cache_checkpoint = True
        return definitions
