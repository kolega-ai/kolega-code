"""``kolega-code acp`` — serve kolega-code as an ACP agent on stdio.

The editor (or any ACP client) spawns this process and drives it over
JSON-RPC 2.0 on stdio. stdout is reserved for protocol frames; all logging
goes to stderr so client parsing is never corrupted.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

log = logging.getLogger("kolega_code.cli.acp")


def run_acp(args: Any) -> int:
    """Serve one client connection until the client closes stdio.

    The SDK import is lazy so every other ``kolega-code`` command stays
    importable without the ACP dependency resolving.
    """
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "debug", False) else logging.INFO,
        stream=sys.stderr,
        format="kolega-code[acp] %(levelname)s %(message)s",
    )
    from acp import run_agent

    from kolega_code import __version__ as kolega_version
    from kolega_code.acp.agent_factory import AgentFactory
    from kolega_code.acp.server import AcpAgent
    from kolega_code.permissions import PermissionMode, normalize_permission_mode

    log.info("kolega-code acp %s starting (permission-mode=%s)", kolega_version, getattr(args, "permission_mode", None))
    permission_mode = normalize_permission_mode(
        getattr(args, "permission_mode", None),
        default=PermissionMode.ASK,
    )
    try:
        asyncio.run(run_agent(AcpAgent(factory=AgentFactory(permission_mode=permission_mode))))
    except KeyboardInterrupt:
        return 130
    return 0
