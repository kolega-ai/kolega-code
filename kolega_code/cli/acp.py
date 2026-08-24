"""``kolega-code acp`` — serve kolega-code as an ACP agent on stdio.

The editor (or any ACP client) spawns this process and drives it over
JSON-RPC 2.0 on stdio. stdout is reserved for protocol frames; all logging
goes to stderr so client parsing is never corrupted.

``--acp-log PATH`` appends a JSON line per protocol update sent, for
client-side debugging independent of the editor's own log viewer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any

log = logging.getLogger("kolega_code.cli.acp")


def _json_logger(path: str) -> logging.Handler:
    handler = logging.FileHandler(path)
    handler.setLevel(logging.DEBUG)

    def format_record(record: logging.LogRecord) -> str:
        try:
            return json.dumps(
                {
                    "ts": record.created,
                    "level": record.levelname,
                    "message": record.getMessage(),
                },
            )
        except Exception:  # noqa: BLE001 — formatting must never break the process
            return f"{record.levelname} {record.getMessage()}"

    handler.setFormatter(logging.Formatter())  # type: ignore[arg-type]
    handler.format = format_record  # type: ignore[method-assign]
    return handler


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
    acp_log_path = getattr(args, "acp_log", None)
    if acp_log_path:
        root = logging.getLogger("kolega_code.acp")
        root.setLevel(logging.DEBUG)
        root.addHandler(_json_logger(acp_log_path))
        log.info("acp traffic log -> %s", acp_log_path)

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
