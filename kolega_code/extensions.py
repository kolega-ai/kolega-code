"""CLI-selected extension loading: public contracts and lifecycle helpers.

An extension is arbitrary trusted Python code, selected explicitly at launch
with ``--extension MODULE:FACTORY`` (plus an optional opaque
``--extension-config PATH``) on the interactive CLI and ``ask``. The named
factory is resolved once per process and called once per top-level agent
generation; it returns a :class:`KolegaExtensionBundle` of ordinary
``PromptExtension``/``ToolExtension`` objects plus optional lifecycle hooks
(``bind_agent``, ``llm_trace_sink``, ``cleanup``). The loader never interprets
the config path's contents and never lets a bundle override host runtime
configuration.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from .agent.prompt_provider import AgentMode, PromptExtension
from .agent.tools import ToolExtension

if TYPE_CHECKING:
    from .agent.baseagent import BaseAgent
    from .config import AgentConfig


class KolegaExtensionLoadError(Exception):
    """A CLI-selected extension could not be loaded, validated, or bound."""


@dataclass(frozen=True)
class KolegaExtensionHost:
    """Read-only host metadata handed to an extension factory.

    ``config`` is the host's already-resolved runtime configuration. It lets an
    extension observe or inherit the calling runtime; it is not an
    extension-owned configuration surface.
    """

    project_path: Path
    workspace_id: str
    thread_id: str
    config: "AgentConfig"
    agent_mode: AgentMode


@dataclass
class KolegaExtensionBundle:
    """One agent generation's contribution from the selected extension.

    ``bind_agent`` runs after the top-level agent and its tool collection are
    fully constructed and before that generation's first model request; it may
    return an awaitable. ``llm_trace_sink`` is forwarded to every LLM client
    created from the bound agent's context, including descendants. ``cleanup``
    runs exactly once per generation, after the agent's own cleanup.
    """

    prompt_extensions: list[PromptExtension] = field(default_factory=list)
    tool_extensions: list[ToolExtension] = field(default_factory=list)
    llm_trace_sink: Callable[[Any], None] | None = None
    bind_agent: Callable[["BaseAgent"], Any] | None = None
    cleanup: Callable[[], Any] | None = None


KolegaExtensionFactory = Callable[[KolegaExtensionHost, Path | None], KolegaExtensionBundle]


@dataclass(frozen=True)
class ExtensionSelection:
    """A launch-time resolved ``--extension``/``--extension-config`` pair."""

    spec: str
    factory: KolegaExtensionFactory
    config_path: Path | None


def resolve_extension_selection(spec: str, config_path: str | Path | None) -> ExtensionSelection:
    """Import ``MODULE:FACTORY`` once at launch and resolve the config path.

    The module must already be importable in the active environment; nothing is
    installed and ``sys.path`` is not modified. The config path is resolved to
    an absolute path without reading it.
    """
    module_name, sep, factory_name = spec.partition(":")
    if not sep or not module_name or not factory_name:
        raise KolegaExtensionLoadError(f"--extension expects MODULE:FACTORY, got {spec!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise KolegaExtensionLoadError(f"--extension: cannot import module {module_name!r}: {exc}") from exc
    factory = getattr(module, factory_name, None)
    if factory is None:
        raise KolegaExtensionLoadError(f"--extension: module {module_name!r} has no attribute {factory_name!r}")
    if not callable(factory):
        raise KolegaExtensionLoadError(f"--extension: {module_name}:{factory_name} is not callable")
    resolved = Path(config_path).expanduser().resolve() if config_path is not None else None
    return ExtensionSelection(spec=spec, factory=cast(KolegaExtensionFactory, factory), config_path=resolved)


def create_extension_bundle(selection: ExtensionSelection, host: KolegaExtensionHost) -> KolegaExtensionBundle:
    """Call the selected factory for one agent generation and validate its bundle.

    Returns a normalized copy: the contributed lists are copied so later
    mutation by the extension cannot alter the installed inventory.
    """
    try:
        bundle = selection.factory(host, selection.config_path)
    except KolegaExtensionLoadError:
        raise
    except Exception as exc:
        raise KolegaExtensionLoadError(
            f"extension factory {selection.spec!r} raised {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(bundle, KolegaExtensionBundle):
        raise KolegaExtensionLoadError(
            f"extension factory {selection.spec!r} returned {type(bundle).__name__}, expected KolegaExtensionBundle"
        )

    prompt_extensions = list(bundle.prompt_extensions)
    tool_extensions = list(bundle.tool_extensions)
    seen_prompt_ids: set[str] = set()
    for prompt_extension in prompt_extensions:
        if not isinstance(prompt_extension, PromptExtension):
            raise KolegaExtensionLoadError(
                f"extension {selection.spec!r} contributed a non-PromptExtension prompt entry: "
                f"{type(prompt_extension).__name__}"
            )
        if prompt_extension.id in seen_prompt_ids:
            raise KolegaExtensionLoadError(
                f"extension {selection.spec!r} contributed duplicate prompt extension id {prompt_extension.id!r}"
            )
        seen_prompt_ids.add(prompt_extension.id)
    for tool_extension in tool_extensions:
        if not isinstance(tool_extension, ToolExtension):
            raise KolegaExtensionLoadError(
                f"extension {selection.spec!r} contributed a non-ToolExtension tool entry: "
                f"{type(tool_extension).__name__}"
            )

    return KolegaExtensionBundle(
        prompt_extensions=prompt_extensions,
        tool_extensions=tool_extensions,
        llm_trace_sink=bundle.llm_trace_sink,
        bind_agent=bundle.bind_agent,
        cleanup=bundle.cleanup,
    )


async def bind_extension_agent(bundle: KolegaExtensionBundle, agent: "BaseAgent") -> None:
    """Run the bundle's agent binding, awaiting it when it returns an awaitable."""
    if bundle.bind_agent is None:
        return
    try:
        result = bundle.bind_agent(agent)
        if inspect.isawaitable(result):
            await result
    except Exception as exc:
        raise KolegaExtensionLoadError(f"extension bind_agent failed: {type(exc).__name__}: {exc}") from exc


async def cleanup_extension_bundle(bundle: KolegaExtensionBundle) -> None:
    """Run the bundle's cleanup, awaiting it when it returns an awaitable.

    Exceptions propagate; call sites report them without masking a primary
    exception from the agent run itself.
    """
    if bundle.cleanup is None:
        return
    result = bundle.cleanup()
    if inspect.isawaitable(result):
        await result
