"""Sandbox configuration and manager factory for local execution."""

from __future__ import annotations

import os

from kolega_code.agent.services.sandbox import LocalSandboxManager, SandboxManager

_sandbox_manager: SandboxManager | None = None


def get_sandbox_manager() -> SandboxManager:
    """Return the configured sandbox manager.

    Core `kolega-code` only provides local filesystem execution. Cloud sandbox
    providers such as E2B are implemented by companion packages.
    """
    global _sandbox_manager

    if _sandbox_manager is None:
        provider = os.getenv("SANDBOX_PROVIDER", "local")
        if provider != "local":
            raise ValueError(
                f"Sandbox provider '{provider}' is not available in kolega-code. "
                "Install and configure the provider package, for example kolega-code-e2b."
            )
        _sandbox_manager = LocalSandboxManager()

    return _sandbox_manager


def is_sandbox_enabled() -> bool:
    """Check if sandbox mode is enabled."""
    return os.getenv("USE_SANDBOX", "false").lower() == "true"

