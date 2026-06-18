"""Lifecycle events and hooks for kolega-code.

A hook runs a user-configured handler (shell command, in-process Python callable,
or an LLM prompt/agent) when a lifecycle event fires — to observe, block, or
modify the agent's behavior. See ``kolega_code/hooks/config.py`` for the on-disk
schema and ``dispatcher.py`` for how events are fired.

This is separate from ``kolega_code.events`` (the UI broadcast bus); the two never
share types.
"""

from __future__ import annotations

from .backends import AgentRunner, HookCapabilities, HookExecutionError, PromptRunner, run_hook
from .config import (
    GLOBAL_HOOKS_FILENAME,
    HOOKS_RELATIVE_PATH,
    HOOKS_SCHEMA_VERSION,
    HookConfig,
    HookConfigError,
    HookSpec,
    load_hook_config,
    project_hooks_present,
)
from .dispatcher import NO_OP_DISPATCHER, HookDispatcher
from .events import (
    BLOCKING_EVENTS,
    TOOL_EVENTS,
    HookEvent,
    LifecycleEvent,
    in_hook,
)
from .matcher import HookMatcher
from .outcome import HookOutcome, merge

__all__ = [
    "AgentRunner",
    "BLOCKING_EVENTS",
    "GLOBAL_HOOKS_FILENAME",
    "HOOKS_RELATIVE_PATH",
    "HOOKS_SCHEMA_VERSION",
    "HookCapabilities",
    "HookConfig",
    "HookConfigError",
    "HookDispatcher",
    "HookEvent",
    "HookExecutionError",
    "HookMatcher",
    "HookOutcome",
    "HookSpec",
    "LifecycleEvent",
    "NO_OP_DISPATCHER",
    "PromptRunner",
    "TOOL_EVENTS",
    "in_hook",
    "load_hook_config",
    "merge",
    "project_hooks_present",
    "run_hook",
]
