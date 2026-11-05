from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import acp
import pytest
from acp.interfaces import Client

from kolega_code.acp.server import AcpAgent
from kolega_code.agent.tools import ToolCollection
from kolega_code.cli.acp import run_acp
from tests.acp.test_server import _FakeConn


def main() -> int:
    project = Path(sys.argv[1])
    records: dict[str, Any] = {}
    real_run_agent = acp.run_agent
    calls = 0

    async def initialize_tools(collection: ToolCollection) -> list[Any]:
        nonlocal calls
        calls += 1
        if calls == 2:
            manager = collection.terminal_tool.terminal_manager
            records["manager"] = manager
            result = await manager.exec_command("exec sleep 600", workdir=str(project), yield_time_ms=250)
            assert result.session_id is not None
            records["process"] = manager.sessions[result.session_id]
        return []

    async def prepared_run_agent(server: AcpAgent, **kwargs: Any) -> None:
        server.on_connect(cast(Client, _FakeConn()))
        patch.setattr(server, "_schedule_open_updates", lambda _sid: None)
        opened = await server.new_session(str(project))
        session = server._sessions[opened.session_id]
        records.update(server=server, session=session)
        patch.setattr(session.agent, "cleanup", AsyncMock(side_effect=RuntimeError("retired close failed")))
        await server.set_session_mode(session.session_id, "plan")
        adopted_cleanup = AsyncMock(wraps=session.agent.cleanup)
        assert session.accounting is not None
        accounting_close = AsyncMock(wraps=session.accounting.aclose)
        patch.setattr(session.agent, "cleanup", adopted_cleanup)
        patch.setattr(session.accounting, "aclose", accounting_close)
        records.update(adopted_cleanup=adopted_cleanup, accounting_close=accounting_close)
        healthy_id = (await server.new_session(str(project))).session_id
        records["healthy_id"] = healthy_id
        await real_run_agent(server, **kwargs)
        records["transport_eof"] = True

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ToolCollection, "initialize", initialize_tools)
        patch.setattr(acp, "run_agent", prepared_run_agent)
        try:
            return run_acp(SimpleNamespace(debug=False, acp_log=None, permission_mode="ask"))
        finally:
            try:
                process = records["process"]
                assert process.pid is not None
                os.kill(process.pid, 0)
                server = records["server"]
                session = records["session"]
                print(
                    json.dumps(
                        {
                            "transport_eof": records.get("transport_eof", False),
                            "pty_alive": not process.exited.is_set(),
                            "adopted_cleanup_calls": records["adopted_cleanup"].await_count,
                            "accounting_close_calls": records["accounting_close"].await_count,
                            "server_retained": server._sessions.get(session.session_id) is session,
                            "factory_retained": server._factory._acp_sessions.get(session.session_id) is session,
                            "healthy_drained": records["healthy_id"] not in server._sessions
                            and records["healthy_id"] not in server._factory._acp_sessions,
                        }
                    ),
                    flush=True,
                )
            finally:
                if "manager" in records:
                    asyncio.run(records["manager"].cleanup_all())


if __name__ == "__main__":
    raise SystemExit(main())
