from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

from acp.interfaces import Client
from acp.schema import ImageContentBlock, UserMessageChunk
import pytest

from kolega_code.acp.agent_factory import AgentFactory
from kolega_code.acp.server import AcpAgent
from kolega_code.cli.session_event_store import FileSessionEventStore
from kolega_code.cli.session_store import SessionStore
from kolega_code.events import AgentEvent, ArtifactPurpose

from .test_server import PNG_1X1, StreamingLLM, _FakeConn, _make_session

pytestmark = pytest.mark.usefixtures("isolated_cli_env")


@pytest.mark.asyncio
@pytest.mark.parametrize("user_text", ["inspect this", ""])
async def test_loaded_transcript_uses_session_confined_artifact_store(tmp_path: Path, user_text: str) -> None:
    session, _ = _make_session(tmp_path, StreamingLLM())
    assert session.agent.tool_collection is not None
    session.agent.tool_collection.cleanup = AsyncMock()
    store = SessionStore(tmp_path / "state")
    store.create(tmp_path, "build", {}, session_id=session.session_id)
    factory = AgentFactory()
    factory._store = store
    artifact_store = factory.artifact_store(session.session_id)
    png = base64.b64decode(PNG_1X1)
    ref = await artifact_store.put(png, media_type="image/png", purpose=ArtifactPurpose.IMAGE, encoding="base64")
    await factory.event_store(session.session_id).append(
        AgentEvent(
            sender="agent",
            event_type="turn_started",
            content={"turn_id": "image-turn", "user_text": user_text, "image_count": 1},
            artifacts=[ref],
        )
    )
    conn = _FakeConn()
    server = AcpAgent(factory=factory)
    server.on_connect(cast(Client, conn))
    try:
        await server._replay_transcript(session)
        updates = [update for update in conn.updates if isinstance(update, UserMessageChunk)]
        images = [update for update in updates if isinstance(update.content, ImageContentBlock)]
        assert len(images) == 1
        assert images[0].content.mime_type == "image/png"
        assert base64.b64decode(images[0].content.data) == png
        if user_text:
            assert updates[0].content.text == user_text
            assert updates[0].message_id == images[0].message_id
        else:
            assert updates == images
    finally:
        await session.agent.cleanup()
        store.release_session_locks()


@pytest.mark.asyncio
async def test_loaded_transcript_never_hydrates_another_sessions_image(tmp_path: Path) -> None:
    session, _ = _make_session(tmp_path, StreamingLLM())
    assert session.agent.tool_collection is not None
    session.agent.tool_collection.cleanup = AsyncMock()
    store = SessionStore(tmp_path / "state")
    store.create(tmp_path, "build", {}, session_id=session.session_id)
    foreign = store.create(tmp_path, "build", {})
    factory = AgentFactory()
    factory._store = store
    ref = await factory.artifact_store(foreign.session_id).put(
        base64.b64decode(PNG_1X1), media_type="image/png", purpose=ArtifactPurpose.IMAGE, encoding="base64"
    )
    await FileSessionEventStore(store.journal(session.session_id)).append(
        AgentEvent(
            sender="agent",
            event_type="turn_started",
            content={"turn_id": "image-turn", "user_text": "", "image_count": 1},
            artifacts=[ref],
        )
    )
    conn = _FakeConn()
    server = AcpAgent(factory=factory)
    server.on_connect(cast(Client, conn))
    try:
        await server._replay_transcript(session)
        updates: list[Any] = [update for update in conn.updates if isinstance(update, UserMessageChunk)]
        assert len(updates) == 1
        assert updates[0].content.text == "[1 image unavailable in replay]"
        assert not isinstance(updates[0].content, ImageContentBlock)
    finally:
        await session.agent.cleanup()
        store.release_session_locks()
