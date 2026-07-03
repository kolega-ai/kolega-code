from pathlib import Path

import pytest

from kolega_code.agent.tool_backend.build_tool import BuildTool
from kolega_code.config import AgentConfig
from kolega_code.events import AgentConnectionManager
from kolega_code.services.base import TerminalManager
from kolega_code.services.file_system import LocalFileSystem


class _StubConnectionManager(AgentConnectionManager):
    """Minimal no-op connection manager for build tool tests."""

    async def connect(self, websocket, workspace_id, thread_id, connection_type, user_info=None) -> None:
        return None

    def disconnect(self, websocket, workspace_id, thread_id, connection_type) -> None:
        return None

    async def broadcast_event(self, event, workspace_id, thread_id) -> None:
        return None

    def get_connection_count(self, workspace_id, thread_id) -> dict:
        return {}


class DummyFS(LocalFileSystem):
    def __init__(self, files: dict[str, str]):
        self._files = files

    def exists(self, path: str) -> bool:
        return path in self._files

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        if path not in self._files:
            raise FileNotFoundError(path)
        return self._files[path]


class DummyTM(TerminalManager):
    def __init__(self, outputs: dict[str, str] | None = None):
        self.outputs = outputs or {}
        self.calls = []

    async def run_command(self, command: str, cwd: str | None = None, timeout: int | None = None) -> str:
        self.calls.append((command, cwd, timeout))
        return self.outputs.get(command, f"ok: {command}")

    async def exec_command(
        self, command, *, workdir=None, yield_time_ms=10000, max_output_tokens=10000, login=False, env=None
    ):
        raise NotImplementedError

    async def write_stdin(self, session_id, chars="", *, yield_time_ms=10000, max_output_tokens=10000):
        raise NotImplementedError

    async def kill_session(self, session_id, signal="TERM"):
        raise NotImplementedError

    async def list_sessions(self):
        raise NotImplementedError

    async def close_all(self):
        raise NotImplementedError


def make_tool(fs_map: dict[str, str], tm_outputs: dict[str, str] | None = None) -> BuildTool:
    tool = BuildTool(
        project_path=Path("/repo"),
        workspace_id="ws",
        thread_id="th",
        connection_manager=_StubConnectionManager(),
        config=AgentConfig(anthropic_api_key="test_key"),
        caller=None,
        filesystem=DummyFS(fs_map),
        terminal_manager=DummyTM(tm_outputs),
    )
    return tool


@pytest.mark.asyncio
async def test_build_backend_specific_command():
    manifest = """
name: demo
runtime: node:18
backend_build_command: npm run build:api
"""
    tool = make_tool({".kolega-manifest.yaml": manifest})
    result = await tool.build_backend()
    assert "npm run build:api" in result
    assert "ok: npm run build:api" in result


@pytest.mark.asyncio
async def test_build_frontend_specific_command():
    manifest = """
name: demo
runtime: node:18
frontend_build_command: npm run build:web
"""
    tool = make_tool({".kolega-manifest.yaml": manifest})
    result = await tool.build_frontend()
    assert "npm run build:web" in result
    assert "ok: npm run build:web" in result


@pytest.mark.asyncio
async def test_build_fallback_to_generic_build_command():
    manifest = """
name: demo
runtime: node:18
build_command: npm run build
"""
    tool = make_tool({".kolega-manifest.yaml": manifest})
    be = await tool.build_backend()
    fe = await tool.build_frontend()
    assert "npm run build" in be
    assert "npm run build" in fe


@pytest.mark.asyncio
async def test_build_no_manifest_or_command():
    tool = make_tool({})
    be = await tool.build_backend()
    fe = await tool.build_frontend()
    assert "No backend_build_command or build_command" in be
    assert "No frontend_build_command or build_command" in fe
