"""Tool: a definition paired with its implementation and execution metadata."""

import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, FrozenSet

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
        self._reject_malformed_call(inputs)
        return await self.handler(**inputs)

    def _reject_malformed_call(self, inputs: Dict[str, Any]) -> None:
        """Convert an argument-shape mismatch into a clean, public ``ToolError``.

        Binding against the handler's signature *before* its body runs does two
        things a post-hoc ``except`` cannot: it separates "the model called this
        tool with the wrong arguments" from a genuine ``TypeError`` raised inside
        the tool, and it never turns a valid call into an error (``bind`` binds
        exactly when the real call would). ``Signature.bind``'s own message names
        only arguments — never the handler's internal qualname — and the usage
        string is built from the *public* tool name, so nothing internal leaks.
        """
        try:
            signature = inspect.signature(self.handler)
        except (TypeError, ValueError):
            return  # Uninspectable callable: let the real call proceed unchanged.
        try:
            signature.bind(**inputs)
        except TypeError as exc:
            raise ToolError(f"Invalid arguments for `{self.name}`: {exc}. Usage: {self._usage(signature)}") from None

    def _usage(self, signature: inspect.Signature) -> str:
        """Render the tool's public call signature, e.g. ``eval(code, language='py')``."""
        parts = []
        for param in signature.parameters.values():
            if param.name in ("self", "cls"):
                continue
            if param.kind is inspect.Parameter.VAR_POSITIONAL:
                parts.append(f"*{param.name}")
            elif param.kind is inspect.Parameter.VAR_KEYWORD:
                parts.append(f"**{param.name}")
            elif param.default is inspect.Parameter.empty:
                parts.append(param.name)
            else:
                parts.append(f"{param.name}={param.default!r}")
        return f"{self.name}({', '.join(parts)})"
