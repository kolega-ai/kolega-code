"""Expose verified MCP tools through Kolega's ToolExtension API."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from kolega_code.agent.tools import ToolExtension
from kolega_code.llm.models import ToolDefinition

from .config import LoadedMCPConfig, load_mcp_config
from .service import MCPExposedTool, MCPService, exposed_tool_names_for, sanitize_mcp_tool_name


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

    # Wire names are sanitized and deduped per server (names embed the server id,
    # so cross-server collisions are impossible). Dispatch is unaffected: the
    # callback closure captures the original server/tool ids, so renaming is
    # model-facing only.
    exposed_by_server: dict[str, list[MCPExposedTool]] = {}
    for exposed_tool in exposed:
        exposed_by_server.setdefault(exposed_tool.server.id, []).append(exposed_tool)

    for server_id, server_tools in exposed_by_server.items():
        name_pairs = exposed_tool_names_for(server_id, [exposed_tool.tool for exposed_tool in server_tools])
        for exposed_tool, (raw_name, tool_name) in zip(server_tools, name_pairs):
            tool_id = exposed_tool.tool.id

            async def _call_mcp_tool(_server_id=server_id, _tool_id=tool_id, **inputs):
                return await service.call_tool(_server_id, _tool_id, inputs)

            callbacks[tool_name] = _call_mcp_tool
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


def mcp_tool_definition(exposed_tool) -> ToolDefinition:
    """Build a ToolDefinition for tests and callers that need explicit definitions.

    Uses the sanitized wire name; ``build_mcp_tool_extension`` additionally
    dedupes collisions per server.
    """
    return ToolDefinition(
        name=sanitize_mcp_tool_name(exposed_tool.name),
        description=exposed_tool.description,
        parameters=[],
        input_schema=exposed_tool.tool.input_schema,
    )
