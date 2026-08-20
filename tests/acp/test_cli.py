"""End-to-end: `kolega-code acp` driven by the SDK's programmatic client.

Spawns the real CLI subprocess exactly the way an editor would (stdio
JSON-RPC) and asserts the transport handshake plus a config-independent
round trip (``session/list``). Turn behavior and config-error conversion are
covered in-process in ``test_server.py``: whether ``session/new`` needs real
API keys depends on the machine (macOS keychain), so it cannot be asserted
here deterministically.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

import pytest
from acp import PROTOCOL_VERSION, spawn_agent_process
from acp.helpers import SessionUpdate
from acp.interfaces import Client
from acp.schema import ToolCallUpdate


class _CollectingClient(Client):
    def __init__(self) -> None:
        self.updates: list[SessionUpdate] = []

    async def session_update(self, session_id: str, update: SessionUpdate, **kwargs: Any) -> None:
        self.updates.append(update)

    async def request_permission(
        self, session_id: str, tool_call: ToolCallUpdate, options: list[Any], **kwargs: Any
    ) -> Any:
        return {"outcome": {"outcome": "cancelled"}}


@pytest.mark.asyncio
async def test_acp_subprocess_handshake_and_session_list(isolated_cli_env: None) -> None:
    client = _CollectingClient()  # pyright: ignore[reportAbstractUsage]
    # The SDK spawns children with a sanitized default environment, so the
    # fixture's hermetic env (isolated state dir, no API keys) must be passed
    # explicitly to reach the child.
    env = dict(os.environ)
    async with asyncio.timeout(60):
        async with spawn_agent_process(
            client,
            sys.executable,
            "-m",
            "kolega_code.cli.main",
            "acp",
            env=env,
        ) as (conn, proc):
            await conn.initialize(protocol_version=PROTOCOL_VERSION)
            # session/list is config-independent: an empty state dir lists no sessions.
            listed = await conn.list_sessions()
            assert listed.sessions == []
            # The transport keeps answering after the round trip.
            await conn.initialize(protocol_version=PROTOCOL_VERSION)
            assert proc.returncode is None
