"""Model Context Protocol (MCP) integration for Kolega Code."""

from .config import (
    MCP_CONFIG_RELATIVE_PATH,
    MCP_GLOBAL_CONFIG_FILENAME,
    LoadedMCPConfig,
    MCPOAuthConfig,
    MCPScope,
    MCPServerConfig,
    load_mcp_config,
    load_mcp_config_for,
    server_fingerprint,
)
from .service import MCPService, MCPVerificationResult
from .tools import build_mcp_tool_extension

__all__ = [
    "MCP_CONFIG_RELATIVE_PATH",
    "MCP_GLOBAL_CONFIG_FILENAME",
    "LoadedMCPConfig",
    "MCPOAuthConfig",
    "MCPScope",
    "MCPServerConfig",
    "MCPService",
    "MCPVerificationResult",
    "build_mcp_tool_extension",
    "load_mcp_config",
    "load_mcp_config_for",
    "server_fingerprint",
]
