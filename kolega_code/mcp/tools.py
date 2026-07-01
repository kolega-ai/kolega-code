"""Expose verified MCP tools through Kolega's ToolExtension API."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from kolega_code.agent.tools import ToolExtension
from kolega_code.llm.models import ToolDefinition

from .config import LoadedMCPConfig, load_mcp_config
from .service import MCPService


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
    schemas = {}
    tool_names = []

    for exposed_tool in exposed:
        tool_name = exposed_tool.name
        server_id = exposed_tool.server.id
        tool_id = exposed_tool.tool.id

        async def _call_mcp_tool(_server_id=server_id, _tool_id=tool_id, **inputs):
            return await service.call_tool(_server_id, _tool_id, inputs)

        _call_mcp_tool.__name__ = tool_name
        _call_mcp_tool.__doc__ = exposed_tool.description
        callbacks[tool_name] = _call_mcp_tool
        schemas[tool_name] = exposed_tool.tool.input_schema
        tool_names.append(tool_name)

    return ToolExtension(
        name="mcp",
        tools=callbacks,
        tool_groups={"mcp_tools": tool_names},
        tool_schemas=schemas,
        cleanup=service.cleanup,
        propagate_to_sub_agents=False,
    )


def mcp_tool_definition(exposed_tool) -> ToolDefinition:
    """Build a ToolDefinition for tests and callers that need explicit definitions."""
    return ToolDefinition(
        name=exposed_tool.name,
        description=exposed_tool.description,
        parameters=[],
        input_schema=exposed_tool.tool.input_schema,
    )
