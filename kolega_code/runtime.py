"""Runtime adapter registry for host-provided services.

The shared agent package intentionally does not own product databases,
background job runners, or app-specific MCP/environment services. Host
applications can register those objects directly.
"""

from __future__ import annotations

from typing import Any

_registry: dict[str, Any] = {}


class RuntimeAdapterError(RuntimeError):
    """Raised when shared agent code needs a host service that was not registered."""


def register_runtime_adapter(name: str, value: Any) -> None:
    """Register a host-provided runtime dependency."""
    _registry[name] = value


def get_runtime_adapter(name: str) -> Any:
    """Return a registered host adapter."""
    if name in _registry:
        return _registry[name]

    raise RuntimeAdapterError(
        f"Host runtime adapter '{name}' is not registered. "
        "Call kolega_code.runtime.register_runtime_adapter during app startup."
    )


class RuntimeProxy:
    """Lazy proxy for a host-provided object."""

    def __init__(self, name: str) -> None:
        self._name = name

    def _target(self) -> Any:
        return get_runtime_adapter(self._name)

    def __getattr__(self, item: str) -> Any:
        if item.startswith("_"):
            raise AttributeError(item)
        return getattr(self._target(), item)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._target()(*args, **kwargs)
