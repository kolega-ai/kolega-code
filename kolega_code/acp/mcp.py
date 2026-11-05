"""Session-local ACP MCP configuration and tool discovery."""

from __future__ import annotations

import asyncio
import contextvars
import logging
import secrets
from collections.abc import Awaitable, Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from acp.exceptions import RequestError
from acp.schema import HttpMcpServer, McpServerStdio, SseMcpServer
from pydantic import ValidationError

from kolega_code.agent import ToolExtension
from kolega_code.mcp.config import LoadedMCPConfig, MCPServerConfig
from kolega_code.mcp.service import MCPService, exposed_tool_names
from kolega_code.mcp.state import MCPServerStatus, MCPStatusStore

logger = logging.getLogger(__name__)
_in_transport = contextvars.ContextVar("acp_mcp_transport", default=False)

ClientMCPServer = HttpMcpServer | SseMcpServer | McpServerStdio


class _TransportLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not (_in_transport.get() and record.name.split(".")[0] in {"mcp", "httpx", "httpcore"})


@contextmanager
def _safe_transport_diagnostics() -> Generator[None, None, None]:
    handlers = [*logging.getLogger().handlers]
    if logging.lastResort is not None:
        handlers.append(logging.lastResort)
    log_filter = _TransportLogFilter()
    for handler in handlers:
        handler.addFilter(log_filter)
    token = _in_transport.set(True)
    try:
        yield
    finally:
        _in_transport.reset(token)
        for handler in handlers:
            handler.removeFilter(log_filter)


def normalize_client_servers(
    servers: Sequence[ClientMCPServer | dict[str, Any]] | None, project_path: Path
) -> dict[str, MCPServerConfig]:
    normalized: dict[str, MCPServerConfig] = {}
    identities: set[str] = set()
    try:
        for entry in servers or []:
            payload = entry if isinstance(entry, dict) else entry.model_dump()
            transport = payload.get("type", "stdio")
            server_id = secrets.token_hex(16)
            while server_id in identities:
                server_id = secrets.token_hex(16)
            identities.add(server_id)
            if transport == "stdio":
                stdio = McpServerStdio.model_validate(payload)
                server = MCPServerConfig(
                    id=server_id,
                    name=stdio.name,
                    transport="stdio",
                    command=stdio.command,
                    args=stdio.args,
                    env={item.name: item.value for item in stdio.env},
                    cwd=str(project_path),
                    source="acp",
                )
                configured_name = stdio.name
            elif transport in {"http", "sse"}:
                remote = (
                    HttpMcpServer.model_validate(payload)
                    if transport == "http"
                    else SseMcpServer.model_validate(payload)
                )
                server = MCPServerConfig(
                    id=server_id,
                    name=remote.name,
                    transport="streamable_http" if transport == "http" else "sse",
                    url=remote.url,
                    headers={item.name: item.value for item in remote.headers},
                    source="acp",
                )
                configured_name = remote.name
            else:
                raise ValueError("unsupported transport")
            normalized[configured_name] = server
    except (ValidationError, ValueError, TypeError, AttributeError):
        raise RequestError.invalid_params({"message": "Invalid ACP MCP server configuration."}) from None
    return normalized


def merge_mcp_config(config: LoadedMCPConfig | None, clients: dict[str, MCPServerConfig]) -> LoadedMCPConfig:
    config = config or LoadedMCPConfig()
    diagnostics = [
        (
            "Project MCP servers are disabled because this project is not trusted for MCP."
            if "not trusted" in diagnostic
            else "MCP configuration warning; check the saved MCP configuration."
        )
        for diagnostic in config.diagnostics
    ]
    return replace(config, servers={**config.servers, **clients}, diagnostics=diagnostics)


class _SessionStatusStore(MCPStatusStore):
    def __init__(self, state_dir: Path, clients: set[str]) -> None:
        super().__init__(state_dir)
        self.clients = clients
        self.statuses: dict[str, MCPServerStatus] = {}

    def get(self, server_id: str) -> MCPServerStatus | None:
        return self.statuses.get(server_id) if server_id in self.clients else super().get(server_id)

    def update(self, server_id: str, status: MCPServerStatus) -> None:
        self.statuses[server_id] = status


async def build_session_mcp_extension(
    config: LoadedMCPConfig, project_path: Path, state_dir: Path
) -> ToolExtension | None:
    # Merge keys are editor names; service lookups and permissions use opaque IDs.
    service_config = replace(config, servers={server.id: server for server in config.servers.values()})
    if len(service_config.servers) != len(config.servers):
        raise RequestError.invalid_params({"message": "Conflicting ACP MCP server identities."})
    service = MCPService(service_config, state_dir=state_dir, project_path=project_path)
    clients = {server.id for server in config.enabled_servers if server.source == "acp"}
    service.status_store = _SessionStatusStore(state_dir, clients)
    try:
        for server_id in clients:
            server = service_config.servers[server_id]
            try:
                with _safe_transport_diagnostics():
                    async with asyncio.timeout(server.timeout_seconds):
                        result = await service.verify_server(server_id)
            except Exception:
                logger.warning("ACP MCP server discovery failed; check the client server configuration.")
            else:
                if not result.ok:
                    logger.warning("ACP MCP server discovery failed; check the client server configuration.")
        extension = _tool_extension(service)
        if extension is None:
            await service.cleanup()
        return extension
    except BaseException:
        await service.cleanup()
        raise


def _tool_extension(service: MCPService) -> ToolExtension | None:
    exposed = service.exposed_tools()
    if not exposed:
        return None
    callbacks: dict[str, Any] = {}
    descriptions: dict[str, str] = {}
    schemas: dict[str, dict[str, Any]] = {}

    def make_callback(server_id: str, tool_id: str) -> Callable[..., Awaitable[Any]]:
        async def call(**inputs: Any) -> Any:
            with _safe_transport_diagnostics():
                return await service.call_tool(server_id, tool_id, inputs)

        return call

    for item, (raw_name, name) in zip(exposed, exposed_tool_names(exposed), strict=True):
        callbacks[name] = make_callback(item.server.id, item.tool.id)
        description = item.description
        if raw_name != name:
            description += f"\n\nThe server-side tool name is `{item.tool.id}`; it is exposed here as `{name}`."
        descriptions[name] = description
        schemas[name] = item.tool.input_schema
    return ToolExtension(
        name="mcp",
        tools=callbacks,
        tool_descriptions=descriptions,
        tool_schemas=schemas,
        tool_groups={"mcp_tools": list(callbacks)},
        cleanup=service.cleanup,
        propagate_to_sub_agents=False,
    )
