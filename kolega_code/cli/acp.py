"""``kolega-code acp`` — serve kolega-code as an ACP agent on stdio.

The editor (or any ACP client) spawns this process and drives it over
JSON-RPC 2.0 on stdio. Python stdout is redirected to stderr after the
protocol streams are bound. Saved stdout references, descriptor writes,
and native code writing directly to stdout are not intercepted.

``--acp-log PATH`` appends a JSON line per protocol update sent, for
client-side debugging independent of the editor's own log viewer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import platform
import sys
from collections.abc import Iterable
from typing import Any, BinaryIO

log = logging.getLogger("kolega_code.cli.acp")


class _StdoutTransport(asyncio.WriteTransport):
    def __init__(self, stdout: BinaryIO, protocol: asyncio.StreamReaderProtocol) -> None:
        super().__init__()
        self._stdout = stdout
        self._protocol = protocol
        self._closing = False

    def write(self, data: bytes | bytearray | memoryview) -> None:
        if self._closing:
            raise ConnectionResetError("ACP stdout is closed")
        self._stdout.write(data)
        self._stdout.flush()

    def writelines(self, list_of_data: Iterable[bytes | bytearray | memoryview]) -> None:
        for data in list_of_data:
            self.write(data)

    def can_write_eof(self) -> bool:
        return False

    def is_closing(self) -> bool:
        return self._closing

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        error: Exception | None = None
        try:
            self._stdout.flush()
        except Exception as exc:
            error = exc
        self._protocol.connection_lost(error)

    def abort(self) -> None:
        self.close()


def _windows_stdout_writer(stdout: BinaryIO) -> asyncio.StreamWriter:
    # SDK 0.12's Windows transport resolves sys.stdout on every write.
    loop = asyncio.get_running_loop()
    protocol = asyncio.StreamReaderProtocol(asyncio.StreamReader(), loop=loop)
    transport = _StdoutTransport(stdout, protocol)
    protocol.connection_made(transport)
    return asyncio.StreamWriter(transport, protocol, None, loop)


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

    async def serve() -> None:
        from acp import run_agent
        from acp.core import DEFAULT_STDIO_BUFFER_LIMIT_BYTES
        from acp.stdio import stdio_streams

        reader, writer = await stdio_streams(limit=DEFAULT_STDIO_BUFFER_LIMIT_BYTES)
        windows = platform.system() == "Windows"
        if windows:
            writer.close()
            writer = _windows_stdout_writer(sys.stdout.buffer)
        sys.stdout = sys.stderr

        try:
            from kolega_code import __version__ as kolega_version
            from kolega_code.acp.agent_factory import AgentFactory
            from kolega_code.acp.server import AcpAgent
            from kolega_code.permissions import PermissionMode, normalize_permission_mode

            log.info(
                "kolega-code acp %s starting (permission-mode=%s)",
                kolega_version,
                getattr(args, "permission_mode", None),
            )
            permission_mode = normalize_permission_mode(
                getattr(args, "permission_mode", None),
                default=PermissionMode.ASK,
            )
            agent = AcpAgent(factory=AgentFactory(permission_mode=permission_mode))
            try:
                await run_agent(
                    agent,
                    input_stream=writer,
                    output_stream=reader,
                    observers=[agent.on_transport_message],
                )
            finally:
                await agent.shutdown()
        finally:
            if windows:
                writer.close()
                await writer.wait_closed()

    stdout = sys.stdout
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        return 130
    finally:
        sys.stdout = stdout
    return 0
