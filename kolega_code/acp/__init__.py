"""ACP agent side for kolega-code.

Makes kolega-code usable from ACP clients (Zed, JetBrains, VS Code via
extensions, Neovim, Emacs) by serving the Agent Client Protocol v1 over
stdio. The client spawns this process and owns its lifecycle.

stdout carries only ACP JSON-RPC frames (owned by the SDK transport); all
diagnostics go to stderr. See ``acp-implementation-plan.md`` (repo root) for
the full design and phases.
"""

from __future__ import annotations

from kolega_code.acp.agent_factory import ACP_AGENT_MODE, ACP_PERMISSION_MODE, AgentFactory
from kolega_code.acp.bridge import TOOL_KINDS, AcpBridge
from kolega_code.acp.diffs import AcpDiffProvider
from kolega_code.acp.permissions import AcpPermissionBroker
from kolega_code.acp.server import AcpAgent
from kolega_code.acp.session import AcpSession
from kolega_code.acp.usage import build_usage_update, context_window_for

__all__ = [
    "ACP_AGENT_MODE",
    "ACP_PERMISSION_MODE",
    "AcpAgent",
    "AcpBridge",
    "AcpDiffProvider",
    "AcpPermissionBroker",
    "AcpSession",
    "AgentFactory",
    "TOOL_KINDS",
    "build_usage_update",
    "context_window_for",
]
