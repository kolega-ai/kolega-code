"""Expose verified MCP tools through Kolega's ToolExtension API."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Optional

from kolega_code.agent.tools import ToolExtension

from .config import LoadedMCPConfig, load_mcp_config
from .service import MCPService, exposed_tool_names


def build_mcp_tool_extension(
    project_path: Path,
    state_dir: Path,
    *,
    project_trusted: bool,
    loaded_config: Optional[LoadedMCPConfig] = None,
) -> Optional[ToolExtension]:
    """Build a ToolExtension for currently verified MCP tools.

    This never performs network I/O or launches OAuth. It only reads MCP config and
    the local verification cache, so agent construction stays non-interactive.
    """
    config = loaded_config or load_mcp_config(project_path, state_dir, project_trusted=project_trusted)
    service = MCPService(config, state_dir=state_dir, project_path=project_path)
    exposed = service.exposed_tools()
    if not exposed:
        return None

    callbacks = {}
    descriptions = {}
    schemas = {}
    tool_names = []

    def make_callback(server_id: str, tool_id: str) -> Callable[..., Awaitable[Any]]:
        async def call_mcp_tool(**inputs: Any) -> Any:
            return await service.call_tool(server_id, tool_id, inputs)

        return call_mcp_tool

    for exposed_tool, (raw_name, tool_name) in zip(exposed, exposed_tool_names(exposed)):
        tool_id = exposed_tool.tool.id
        callbacks[tool_name] = make_callback(exposed_tool.server.id, tool_id)
        description = exposed_tool.description
        if raw_name != tool_name:
            description = (
                f"{description}\n\nThe server-side tool name is `{tool_id}`; it is exposed here as `{tool_name}`."
            )
        descriptions[tool_name] = description
        schemas[tool_name] = exposed_tool.tool.input_schema
        tool_names.append(tool_name)

    return ToolExtension(
        name="mcp",
        tools=callbacks,
        tool_groups={"mcp_tools": tool_names},
        tool_descriptions=descriptions,
        tool_schemas=schemas,
        cleanup=service.cleanup,
        propagate_to_sub_agents=False,
    )
