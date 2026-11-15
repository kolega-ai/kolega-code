from __future__ import annotations

import asyncio
import base64
from contextlib import nullcontext
from copy import deepcopy
from io import BytesIO
from pathlib import Path
import threading
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from kolega_code.agent.baseagent import BaseAgent, QueuedUserInput
from kolega_code.cli.session_event_store import FileArtifactStore, FileSessionEventStore
from kolega_code.cli.session_store import SessionStore
from kolega_code.events import (
    AGENT_EVENT_SCHEMA_VERSION,
    AgentConnectionManager,
    AgentEvent,
    AgentEventEmitter,
    ArtifactPurpose,
    KnownEventType,
)
from kolega_code.llm.models import ImageBlock, Message, TextBlock
from kolega_code.llm.exceptions import LLMError
from kolega_code.llm.usage import normalize_usage
from kolega_code.session.inmemory import InMemoryArtifactStore, InMemorySessionEventStore
from kolega_code.session.projection import replay
from kolega_code.session.recording import IMAGE_PAYLOAD_KEY, RecordingConnectionManager

from .compaction_helpers import FakeLLM, build_agent

pytestmark = pytest.mark.usefixtures("isolated_cli_env")


def _image(raw: bytes, media_type: str = "image/png") -> dict[str, Any]:
    return {"type": "image", "media_type": media_type, "data": base64.b64encode(raw).decode("ascii")}


def _picture(color: str, image_format: str = "PNG") -> bytes:
    output = BytesIO()
    Image.new("RGB", (1, 1), color).save(output, format=image_format)
    return output.getvalue()


def _build_agent(
    tmp_path: Path,
    *,
    sub_agent: bool = False,
) -> tuple[BaseAgent, AsyncMock, InMemorySessionEventStore, InMemoryArtifactStore]:
    inner = AsyncMock(spec=AgentConnectionManager)
    store = InMemorySessionEventStore()
    artifacts = InMemoryArtifactStore()
    manager = RecordingConnectionManager(inner, store, session_id="user-images", artifact_store=artifacts)
    llm = FakeLLM(
        token_script=[100],
        final_message=Message(role="assistant", content=[TextBlock(text="offline answer")], stop_reason="end_turn"),
    )
    agent, _ = build_agent(tmp_path, connection_manager=manager, sub_agent=sub_agent, llm=llm)
    return agent, inner, store, artifacts


def _live_events(inner: AsyncMock) -> list[AgentEvent]:
    return [call.args[0] for call in inner.broadcast_event.call_args_list]


@pytest.mark.asyncio
@pytest.mark.parametrize(("message", "include_file"), [("compare these", True), ("", False)])
async def test_agent_emits_multiple_user_images_only_on_started_and_preserves_history(
    tmp_path: Path, message: str, include_file: bool
) -> None:
    agent, inner, store, artifacts = _build_agent(tmp_path)
    raw_images = [_picture("red"), _picture("blue", "JPEG")]
    attachments = [_image(raw_images[0])]
    if include_file:
        attachments.append({"type": "file", "path": "notes.txt", "content": "attachment text"})
    attachments.append(_image(raw_images[1], "image/jpeg"))
    original = deepcopy(attachments)

    chunks = [chunk async for chunk in agent.process_message_stream(message, attachments)]

    assert any(chunk["type"] == "response" and chunk["complete"] for chunk in chunks)
    assert agent.history[-1].get_text_content() == "offline answer"
    events = await store.read("user-images")
    started = [event for event in events if event.event_type == KnownEventType.TURN_STARTED]
    assert len(started) == 1
    start = started[0]
    assert start.schema_version == AGENT_EVENT_SCHEMA_VERSION
    assert start.content["user_text"] == message
    assert start.content["image_count"] == 2
    assert start.content["status"] is None
    assert [ref.purpose for ref in start.artifacts] == [ArtifactPurpose.IMAGE, ArtifactPurpose.IMAGE]
    assert [ref.media_type for ref in start.artifacts] == ["image/png", "image/jpeg"]
    assert [await artifacts.open(ref) for ref in start.artifacts] == raw_images
    live_start = next(event for event in _live_events(inner) if event.event_type == KnownEventType.TURN_STARTED)
    assert live_start.content == start.content
    assert live_start.artifacts == start.artifacts
    assert live_start.seq == start.seq
    for event in events + _live_events(inner):
        assert IMAGE_PAYLOAD_KEY not in event.content
        if event.event_type != KnownEventType.TURN_STARTED:
            assert "image_count" not in event.content
            assert not any(ref.purpose == ArtifactPurpose.IMAGE for ref in event.artifacts)
    ended = next(event for event in events if event.event_type == KnownEventType.TURN_ENDED)
    assert ended.content["turn_id"] == start.content["turn_id"]
    assert ended.content["status"] == "completed"
    user_message = next(
        item
        for item in agent.history
        if isinstance(item.content, list) and any(isinstance(b, ImageBlock) for b in item.content)
    )
    image_blocks = [block for block in user_message.content if isinstance(block, ImageBlock)]
    assert [block.data for block in image_blocks] == [original[0]["data"], original[-1]["data"]]
    assert [block.media_type for block in image_blocks] == ["image/png", "image/jpeg"]
    assert user_message.get_text_content().startswith(message)
    assert ("attachment text" in user_message.get_text_content()) == include_file
    assert attachments == original
    restored, _, _, _ = _build_agent(tmp_path)
    restored.restore_message_history(agent.dump_message_history())
    assert [item.to_dict() for item in restored.history] == [item.to_dict() for item in agent.history]


@pytest.mark.asyncio
async def test_transient_boundary_has_existing_image_payload_shape(tmp_path: Path) -> None:
    inner = AsyncMock(spec=AgentConnectionManager)
    agent, _ = build_agent(tmp_path, connection_manager=inner, llm=FakeLLM(token_script=[100]))
    attachment = _image(b"inline")

    _ = [chunk async for chunk in agent.process_message_stream("look", [attachment])]

    start = next(event for event in _live_events(inner) if event.event_type == KnownEventType.TURN_STARTED)
    assert start.content[IMAGE_PAYLOAD_KEY] == [{"data": attachment["data"], "media_type": "image/png"}]
    assert start.content["image_count"] == 1
    assert start.content["user_text"] == "look"
    assert start.artifacts == []


@pytest.mark.asyncio
async def test_mixed_available_and_unavailable_images_keep_original_count(tmp_path: Path) -> None:
    agent, inner, store, artifacts = _build_agent(tmp_path)
    attachments = [
        _image(_picture("green")),
        {"type": "image", "image_type": "url", "data": "https://example.invalid/image.png"},
        {"type": "image", "data": "%%%invalid%%%"},
    ]
    original = deepcopy(attachments)

    _ = [chunk async for chunk in agent.process_message_stream("three attachments", attachments)]

    start = next(event for event in await store.read("user-images") if event.event_type == KnownEventType.TURN_STARTED)
    assert start.content["image_count"] == 3
    assert start.content["user_text"] == "three attachments"
    assert len(start.artifacts) == 1
    assert await artifacts.open(start.artifacts[0]) == base64.b64decode(original[0]["data"])
    live = next(event for event in _live_events(inner) if event.event_type == KnownEventType.TURN_STARTED)
    assert live.artifacts == start.artifacts
    assert live.content == start.content
    assert IMAGE_PAYLOAD_KEY not in live.content
    assert attachments == original


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "attachment",
    [
        {
            "type": "image",
            "image_type": "url",
            "data": "https://example.invalid/private.png",
            "media_type": "image/png",
        },
        {"type": "image", "data": b"\xffinvalid-bytes", "media_type": "image/png"},
        {"type": "image", "data": "%%%not-base64%%%", "media_type": "image/png"},
        {"type": "image", "data": "/unavailable/private.png", "media_type": "image/png"},
    ],
    ids=["url", "bytes", "malformed-base64", "path"],
)
async def test_unavailable_user_images_keep_count_without_fetching_or_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attachment: dict[str, Any]
) -> None:
    agent, inner, store, _ = _build_agent(tmp_path)
    original = deepcopy(attachment)
    original_open = Path.open

    def guarded_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        assert str(path) != "/unavailable/private.png", "presentation must not read attachment paths"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    import httpx

    get = AsyncMock(side_effect=AssertionError("presentation must not fetch attachment URLs"))
    monkeypatch.setattr(httpx.AsyncClient, "get", get)

    expected = pytest.raises(LLMError, match="bytes") if isinstance(attachment["data"], bytes) else nullcontext()
    with expected:
        _ = [chunk async for chunk in agent.process_message_stream("keep this text", [attachment])]

    events = await store.read("user-images")
    start = next(event for event in events if event.event_type == KnownEventType.TURN_STARTED)
    assert start.content["user_text"] == "keep this text"
    assert start.content["image_count"] == 1
    assert start.artifacts == []
    live_start = next(event for event in _live_events(inner) if event.event_type == KnownEventType.TURN_STARTED)
    assert live_start.content == start.content
    assert IMAGE_PAYLOAD_KEY not in live_start.content
    assert attachment == original
    get.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("attachments", [None, [], [{"type": "file", "path": "note", "content": "text"}]])
async def test_no_image_attachments_preserve_existing_boundary_content(
    tmp_path: Path, attachments: list[dict[str, Any]] | None
) -> None:
    agent, _, store, _ = _build_agent(tmp_path)

    _ = [chunk async for chunk in agent.process_message_stream("plain text", attachments)]

    start = next(event for event in await store.read("user-images") if event.event_type == KnownEventType.TURN_STARTED)
    assert set(start.content) == {"turn_id", "status", "user_text"}
    assert start.content["user_text"] == "plain text"
    assert start.artifacts == []


@pytest.mark.asyncio
async def test_subagent_image_boundaries_retain_dispatch_metadata(tmp_path: Path) -> None:
    agent, inner, store, _ = _build_agent(tmp_path, sub_agent=True)
    metadata = {"agent_id": "child", "agent_name": "worker", "dispatch_id": "dispatch", "parent_tool_call_id": "tool"}
    agent.sub_agent_context = deepcopy(metadata)

    _ = [chunk async for chunk in agent.process_message_stream("delegated task", [_image(b"child image")])]

    for source in (await store.read("user-images"), _live_events(inner)):
        boundaries = [
            event for event in source if event.event_type in (KnownEventType.TURN_STARTED, KnownEventType.TURN_ENDED)
        ]
        assert len(boundaries) == 2
        assert all(event.sub_agent_info == metadata for event in boundaries)
        assert boundaries[0].content["image_count"] == 1
        assert len(boundaries[0].artifacts) == 1
        assert "image_count" not in boundaries[1].content
        assert boundaries[1].artifacts == []
    assert agent.sub_agent_context == metadata


@pytest.mark.asyncio
async def test_continuation_does_not_reemit_prior_user_images(tmp_path: Path) -> None:
    agent, inner, store, _ = _build_agent(tmp_path)
    _ = [chunk async for chunk in agent.process_message_stream("look", [_image(b"original")])]
    before = [item.to_dict() for item in agent.history]
    first_count = len(_live_events(inner))

    _ = [chunk async for chunk in agent.continue_from_history_stream()]

    for event in _live_events(inner)[first_count:]:
        assert IMAGE_PAYLOAD_KEY not in event.content
        assert "image_count" not in event.content
        assert not any(ref.purpose == ArtifactPurpose.IMAGE for ref in event.artifacts)
    starts = [event for event in await store.read("user-images") if event.event_type == KnownEventType.TURN_STARTED]
    assert len(starts) == 2
    assert starts[1].content["user_text"] is None
    assert "image_count" not in starts[1].content
    assert starts[1].artifacts == []
    assert [item.to_dict() for item in agent.history][:-1] == before


@pytest.mark.asyncio
async def test_emitter_ignores_images_and_count_on_turn_end() -> None:
    inner = AsyncMock(spec=AgentConnectionManager)
    emitter = AgentEventEmitter(inner, "w", "t", "agent")

    await emitter.turn_boundary(
        "ended", turn_id="turn", status="completed", images=[("image/png", "aW1hZ2U=")], image_count=3
    )

    (event,) = _live_events(inner)
    assert event.content == {"turn_id": "turn", "status": "completed", "user_text": None}
    assert event.artifacts == []


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "cancelled"])
async def test_failed_and_cancelled_turns_do_not_repeat_user_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    async def _no_retry_delay(_delay: float) -> None:
        return None

    agent, _, store, _ = _build_agent(tmp_path)
    llm = FakeLLM(token_script=[100])
    llm.stream = AsyncMock(
        side_effect=asyncio.CancelledError() if status == "cancelled" else RuntimeError("offline error")
    )
    monkeypatch.setattr(agent, "llm", llm)

    # The turn must fail; the retry-backoff sleeps are not what this asserts.
    with (
        patch("kolega_code.agent.baseagent.asyncio.sleep", side_effect=_no_retry_delay),
        pytest.raises(asyncio.CancelledError if status == "cancelled" else LLMError),
    ):
        _ = [chunk async for chunk in agent.process_message_stream("look", [_image(_picture("red"))])]

    boundaries = [
        event
        for event in await store.read("user-images")
        if event.event_type in (KnownEventType.TURN_STARTED, KnownEventType.TURN_ENDED)
    ]
    assert len(boundaries) == 2
    assert boundaries[0].content["image_count"] == 1
    assert len(boundaries[0].artifacts) == 1
    assert boundaries[1].content["status"] == status
    assert "image_count" not in boundaries[1].content
    assert boundaries[1].artifacts == []


@pytest.mark.asyncio
@pytest.mark.parametrize("user_text", ["cancel this picture", ""])
async def test_cancellation_during_image_write_retains_presentation_and_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, user_text: str
) -> None:
    session_store = SessionStore(tmp_path / "state")
    session = session_store.create(tmp_path, "code", {})
    journal = session_store.journal(session.session_id)
    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    original_put = journal.put_artifact

    def gated_put(data: bytes, **kwargs: Any) -> dict[str, Any]:
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(30), "test did not release the artifact writer"
        return original_put(data, **kwargs)

    monkeypatch.setattr(journal, "put_artifact", gated_put)
    inner = AsyncMock(spec=AgentConnectionManager)
    events = FileSessionEventStore(journal)
    artifacts = FileArtifactStore(journal)
    recording = RecordingConnectionManager(inner, events, session_id=session.session_id, artifact_store=artifacts)
    llm = FakeLLM(token_script=[100])
    agent, _ = build_agent(tmp_path, connection_manager=recording, llm=llm)
    agent.session_recorder = session_store.recorder(session.session_id)
    png = _picture("red")

    async def consume() -> list[dict[str, Any]]:
        return [chunk async for chunk in agent.process_message_stream(user_text, [_image(png)])]

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(entered.wait(), 10)
        for _ in range(2):
            task.cancel()
            await asyncio.sleep(0)
        assert not task.done(), "cancellation abandoned a still-running artifact writer"
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 10)
        recorded = await events.read(session.session_id)
        boundaries = [
            event for event in recorded if event.event_type in (KnownEventType.TURN_STARTED, KnownEventType.TURN_ENDED)
        ]
        assert [event.event_type for event in boundaries] == [
            KnownEventType.TURN_STARTED,
            KnownEventType.TURN_ENDED,
        ]
        assert boundaries[0].content["user_text"] == user_text
        assert boundaries[0].content["image_count"] == 1
        assert await artifacts.open(boundaries[0].artifacts[0]) == png
        assert boundaries[1].content["status"] == "cancelled"
        live = [
            event
            for event in _live_events(inner)
            if event.event_type in (KnownEventType.TURN_STARTED, KnownEventType.TURN_ENDED)
        ]
        assert [(event.event_type, event.content, event.artifacts) for event in live] == [
            (event.event_type, event.content, event.artifacts) for event in boundaries
        ]
        assert all(IMAGE_PAYLOAD_KEY not in event.content for event in recorded + live)
        users = [item for item in replay(recorded).conversation if item.kind == "user"]
        assert [(item.text, item.image_count) for item in users] == [(user_text, 1)]
        loaded = session_store.load(session.session_id)
        assert any(
            isinstance(block, ImageBlock) and base64.b64decode(block.data) == png
            for item in loaded.history
            for block in Message.from_dict(item).content
        )
        assert llm.stream.await_count == 0
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await recording.flush()
        session_store.release_session_locks()


@pytest.mark.asyncio
async def test_queued_user_image_context_does_not_invent_turn_start_events(tmp_path: Path) -> None:
    agent, inner, store, _ = _build_agent(tmp_path)
    attachment = _image(_picture("red"))
    agent.queued_input_provider = AsyncMock(
        return_value=[QueuedUserInput(text="queued image", attachments=[attachment])]
    )

    await agent._deliver_queued_user_inputs()

    assert await store.read("user-images") == []
    assert _live_events(inner) == []
    assert agent.history[-1].get_text_content() == "queued image"
    assert any(
        isinstance(block, ImageBlock) and block.data == attachment["data"] for block in agent.history[-1].content
    )


@pytest.mark.asyncio
async def test_user_image_semantic_recording_and_resume_are_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    session_store = SessionStore(tmp_path / "sessions")
    session = session_store.create(project, "code", {})
    agent, _, _, _ = _build_agent(project)
    agent.session_recorder = session_store.recorder(session.session_id)
    usage = normalize_usage(
        {"input_tokens": 10, "output_tokens": 5},
        "anthropic",
        agent.primary_model_config.model,
    )
    llm = FakeLLM(
        token_script=[100],
        final_message=Message(
            role="assistant", content=[TextBlock(text="offline answer")], stop_reason="end_turn", usage=usage
        ),
    )
    monkeypatch.setattr(agent, "llm", llm)
    attachments = [_image(_picture("navy"))]
    original = deepcopy(attachments)
    try:
        _ = [chunk async for chunk in agent.process_message_stream("remember this", attachments)]

        loaded = session_store.load(session.session_id)
        messages = [Message.from_dict(item) for item in loaded.history]
        images = [
            block
            for item in messages
            if isinstance(item.content, list)
            for block in item.content
            if isinstance(block, ImageBlock)
        ]
        assert len(images) == 1
        assert images[0].data == original[0]["data"]
        assert images[0].media_type == original[0]["media_type"]
        assert attachments == original
        assert [item.to_dict() for item in messages] == [
            item.to_dict() for item in agent.history if item is not agent.session_context_message
        ]
        assert messages[-1].usage == usage
        assert llm.stream.await_count == 1
        assert loaded.usage["requests"] == loaded.usage["responses"] == 1
        assert loaded.usage["input_tokens"] == 10
        assert loaded.usage["output_tokens"] == 5
        assert loaded.usage["total_tokens"] == 15
        semantic_starts = [
            event
            for event in session_store.journal(session.session_id).read_events()
            if event.event_type == "turn.started"
        ]
        assert len(semantic_starts) == 1
        assert "image_count" not in semantic_starts[0].payload
    finally:
        session_store.release_session_locks()
