from __future__ import annotations

import sys
from pathlib import Path

import pytest

from kolega_code.acp.mcp import build_session_mcp_extension, merge_mcp_config, normalize_client_servers
from kolega_code.mcp.config import LoadedMCPConfig


@pytest.mark.asyncio
async def test_editor_stdio_server_discovers_and_invokes_real_mcp_tool(tmp_path: Path) -> None:
    server = tmp_path / "mcp_server.py"
    server.write_text(
        "from mcp.server.fastmcp import FastMCP\n"
        "mcp = FastMCP('integration')\n"
        "@mcp.tool()\n"
        "def echo(value: str) -> str:\n"
        "    return 'received:' + value\n"
        "if __name__ == '__main__':\n"
        "    mcp.run(transport='stdio')\n",
        encoding="utf-8",
    )
    clients = normalize_client_servers(
        [{"name": "editor-server", "command": sys.executable, "args": [str(server)], "env": []}],
        tmp_path,
    )
    extension = await build_session_mcp_extension(merge_mcp_config(LoadedMCPConfig(), clients), tmp_path, tmp_path)
    assert extension is not None
    try:
        names = extension.tool_groups["mcp_tools"]
        assert len(names) == 1
        result = await extension.tools[names[0]](value="hello")
        assert isinstance(result, str)
        assert result.startswith("received:hello")
    finally:
        if extension.cleanup is not None:
            await extension.cleanup()
