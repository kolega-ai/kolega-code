"""Tool: a definition paired with its implementation and execution metadata."""

import inspect
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Awaitable, Callable, Dict, FrozenSet, Optional

from ..llm.models import ToolDefinition
from .param_aliases import resolve_param_aliases

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1024)
def _function_signature(function: Callable[..., Any]) -> inspect.Signature:
    return inspect.signature(function)


@lru_cache(maxsize=1024)
def _bound_method_signature(function: Callable[..., Any]) -> inspect.Signature:
    """The signature a bound method is called with, i.e. without ``self``/``cls``."""

    signature = _function_signature(function)
    parameters = list(signature.parameters.values())
    if not parameters or parameters[0].kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        raise ValueError("invalid method signature")
    return signature.replace(parameters=parameters[1:])


def _handler_signature(handler: Callable[..., Any]) -> Optional[inspect.Signature]:
    """The signature ``handler`` is called with, or None when uninspectable.

    Parsing it is roughly half of rebuilding a tool collection, and a
    collection is rebuilt for every model request and tool call. Built-in
    handlers are stable class methods, so their signatures are cached by the
    underlying function and later rebuilds reuse them. Other callables keep
    the uncached path so host-supplied objects are never retained here.
    """
    try:
        if inspect.ismethod(handler):
            return _bound_method_signature(handler.__func__)
        if inspect.isfunction(handler):
            return _function_signature(handler)
        return inspect.signature(handler)
    except (TypeError, ValueError):
        return None


class ToolError(Exception):
    """
    An expected tool failure.

    Raise this from a tool implementation when the failure is a normal
    outcome the model should see and react to (file not found, invalid
    arguments, command failed). The executor reports the message as an
    is_error tool result without treating it as an internal fault.
    """


class ToolInputError(ToolError):
    """The model supplied arguments that cannot be bound to the tool handler."""


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
    # Debug-level log_message sink for param-alias hits, so trajectory mining
    # sees which aliases fire. None falls back to stdlib logging.
    log_debug: Optional[Callable[[str], Awaitable[None]]] = None
    # Expands filesystem-path arguments (scratchpad references such as
    # ``$KOLEGA_SCRATCHPAD/foo.py``) before the handler runs. Supplied at
    # construction for the built-in file tools; None leaves paths untouched.
    path_expander: Optional[Callable[[str], str]] = None
    # Canonical parameter names of ``handler`` that hold filesystem paths.
    # Only these receive expansion; a value is expanded only when it is a
    # plain string. Defaults to empty, so expansion is opt-in per tool.
    path_params: FrozenSet[str] = field(default_factory=frozenset)
    # Parameter names the bound handler accepts, computed once at construction
    # via ``inspect``. Alias resolution treats a registered alias as real only
    # when the binding does NOT accept the name: a name the handler itself
    # accepts is a legitimate argument for this binding, never a misspelling.
    accepted_params: FrozenSet[str] = field(init=False, repr=False)
    _signature: Optional[inspect.Signature] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        signature = _handler_signature(self.handler)
        accepted = frozenset(signature.parameters) if signature is not None else frozenset()
        object.__setattr__(self, "accepted_params", accepted)
        object.__setattr__(self, "_signature", signature)

    async def call(self, **inputs: Any) -> Any:
        inputs = await self._resolve_param_aliases(inputs)
        inputs = self._expand_path_params(inputs)
        self._reject_malformed_call(inputs)
        return await self.handler(**inputs)

    def _expand_path_params(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Expand scratchpad references in the tool's path arguments.

        Runs after alias resolution so the canonical parameter names are
        already in place, and before argument validation so the expanded path
        is what the handler (and its result strings) sees. A no-op unless the
        tool was constructed with a ``path_expander``.
        """
        expander = self.path_expander
        if expander is None or not self.path_params:
            return inputs
        for name in self.path_params:
            value = inputs.get(name)
            if isinstance(value, str):
                inputs[name] = expander(value)
        return inputs

    async def _resolve_param_aliases(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Map registered wrong-name arguments onto canonical parameters.

        Runs before ``_reject_malformed_call`` so a registered alias can never
        surface as an invalid-arguments error (or a raw ``TypeError``).
        """
        resolved, hits = resolve_param_aliases(self.name, inputs, accepted_params=self.accepted_params)
        for hit in hits:
            message = f"Param alias on `{hit.tool}`: `{hit.alias}` {hit.action}"
            if self.log_debug is not None:
                await self.log_debug(message)
            else:
                logger.debug(message)
        return resolved

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
        signature = self._signature
        if signature is None:
            return  # Uninspectable callable: let the real call proceed unchanged.
        try:
            signature.bind(**inputs)
        except TypeError as exc:
            raise ToolInputError(
                f"Invalid arguments for `{self.name}`: {exc}. Usage: {self._usage(signature)}"
            ) from None

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
