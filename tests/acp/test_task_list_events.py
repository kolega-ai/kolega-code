from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from acp.interfaces import Client
from acp.schema import AgentPlanUpdate

from kolega_code.acp.agent_factory import MODE_BUILD, MODE_PLAN, AgentFactory
from kolega_code.acp.bridge import AcpBridge
from kolega_code.acp.diffs import AcpDiffProvider
from kolega_code.acp.server import AcpAgent
from kolega_code.acp.session import AcpSession
from kolega_code.agent import ToolExtension
from kolega_code.agent.tools import ToolCollection
from kolega_code.cli.session_store import SessionRecord
from kolega_code.events import AgentEvent, KnownEventType
from kolega_code.tools import ToolError

from .test_server import _FakeConn

pytestmark = pytest.mark.usefixtures("isolated_cli_env", "hermetic_git_config")


@pytest_asyncio.fixture
async def factory(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[AgentFactory]:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-task-list-provider-key")
    monkeypatch.setenv("KOLEGA_CODE_PROVIDER", "anthropic")
    monkeypatch.setenv("KOLEGA_CODE_MODEL", "claude-haiku-4-5-20251001")
    monkeypatch.setattr(ToolCollection, "initialize", AsyncMock(return_value=[]))
    factory = AgentFactory()
    try:
        yield factory
    finally:
        for session_id in list(factory._acp_sessions):
            await factory.close_session(session_id)
        factory._store.release_session_locks()


def task_extension(session: AcpSession) -> ToolExtension:
    return next(extension for extension in session.agent.tool_extensions if extension.name == "acp-shared-task-list")


async def task_events(factory: AgentFactory, session_id: str) -> list[AgentEvent]:
    return [
        event
        for event in await factory.event_store(session_id).read(session_id)
        if event.event_type == KnownEventType.TASK_LIST_UPDATED
    ]


@pytest.mark.asyncio
async def test_factory_records_initial_update_and_clear_after_metadata_save(
    factory: AgentFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = await factory.open_session(str(tmp_path))
    initial = await task_events(factory, session.session_id)
    assert [event.content["markdown"] for event in initial] == [""]
    generation = getattr(session.agent, "_acp_generation")
    observed: list[str] = []
    broadcast = session.manager.broadcast_event

    async def observe(event: AgentEvent, workspace_id: str, thread_id: str) -> None:
        if event.event_type == KnownEventType.TASK_LIST_UPDATED:
            markdown = event.content["markdown"]
            durable = factory._store.load(session.session_id)
            assert durable.task_list_markdown == markdown
            assert session.record.updated_at == durable.updated_at
            recorded = await task_events(factory, session.session_id)
            assert recorded[-1].uuid == event.uuid
            observed.append(markdown)
        await broadcast(event, workspace_id, thread_id)

    monkeypatch.setattr(session.manager, "broadcast_event", observe)
    markdown = "- [x] inspect\n- [ ] implement"
    tools = task_extension(session).tools
    assert await tools["update_task_list"](f" \n{markdown}\n ") == "Task list updated."
    assert await tools["get_task_list"]() == markdown
    assert await tools["update_task_list"](" \n ") == "Task list updated."
    assert observed == [markdown, ""]
    events = await task_events(factory, session.session_id)
    assert [event.content["markdown"] for event in events[len(initial) :]] == [markdown, ""]
    for event in events:
        assert event.content == {"markdown": event.content["markdown"], "acp_generation": generation}
        assert event.session_id == session.session_id
        assert event.workspace_id == session.record.workspace_id
        assert event.thread_id == session.record.thread_id
        assert event.sub_agent_info is None
        assert not event.is_streaming
        assert event.seq is not None
    sequences = [event.seq for event in events if event.seq is not None]
    assert sequences == sorted(set(sequences))


@pytest.mark.asyncio
async def test_resumed_legacy_metadata_is_snapshotted_before_factory_returns(
    factory: AgentFactory, tmp_path: Path
) -> None:
    record = factory._store.create(tmp_path, "cli", {})
    record.task_list_markdown = "- [X] legacy completed\n- [ ] restored"
    factory._store.save(record)
    assert await task_events(factory, record.session_id) == []

    session = await factory.load_session(str(tmp_path), record.session_id)
    assert session is not None
    events = await task_events(factory, record.session_id)
    assert [event.content for event in events] == [
        {"markdown": record.task_list_markdown, "acp_generation": getattr(session.agent, "_acp_generation")}
    ]
    assert await task_extension(session).tools["get_task_list"]() == record.task_list_markdown


@pytest.mark.asyncio
async def test_rebuild_keeps_readonly_tools_and_rejects_retired_writer_without_emitting_snapshot(
    factory: AgentFactory, tmp_path: Path
) -> None:
    session = await factory.open_session(str(tmp_path))
    old_agent = session.agent
    old_writer = task_extension(session).tools["update_task_list"]
    await old_writer("- [x] retained")
    before = await task_events(factory, session.session_id)

    await factory.set_interaction_mode(session, MODE_PLAN)
    assert session.agent is not old_agent
    readonly = task_extension(session)
    assert set(readonly.tools) == {"get_task_list"}
    assert readonly.propagate_to_sub_agents is False
    assert await readonly.tools["get_task_list"]() == "- [x] retained"
    with pytest.raises(ToolError, match="inactive agent"):
        await old_writer("- [ ] stale plan-mode write")
    assert await task_events(factory, session.session_id) == before
    assert factory._store.load(session.session_id).task_list_markdown == "- [x] retained"

    await factory.set_interaction_mode(session, MODE_BUILD)
    assert await task_events(factory, session.session_id) == before
    with pytest.raises(ToolError, match="inactive agent"):
        await old_writer("- [ ] stale build-mode write")
    assert set(task_extension(session).tools) == {"get_task_list", "update_task_list"}
    await task_extension(session).tools["update_task_list"]("- [x] current generation")
    events = await task_events(factory, session.session_id)
    assert len(events) == len(before) + 1
    assert events[-1].content == {
        "markdown": "- [x] current generation",
        "acp_generation": getattr(session.agent, "_acp_generation"),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("retired_by", ["closed", "unregistered", "replaced"])
async def test_factory_writer_requires_registered_current_live_agent(
    factory: AgentFactory, tmp_path: Path, retired_by: str
) -> None:
    session = await factory.open_session(str(tmp_path))
    writer = task_extension(session).tools["update_task_list"]
    before = await task_events(factory, session.session_id)
    agent = session.agent
    if retired_by == "closed":
        session.closed = True
    elif retired_by == "unregistered":
        factory._acp_sessions.pop(session.session_id)
    else:
        session.agent = cast(Any, object())
    try:
        with pytest.raises(ToolError, match="inactive agent"):
            await writer("- [ ] forbidden")
        assert session.record.task_list_markdown == ""
        assert factory._store.load(session.session_id).task_list_markdown == ""
        assert await task_events(factory, session.session_id) == before
    finally:
        session.closed = False
        session.agent = agent
        factory._acp_sessions[session.session_id] = session


@pytest.mark.asyncio
async def test_factory_save_failure_never_publishes(
    factory: AgentFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = await factory.open_session(str(tmp_path))
    before = await task_events(factory, session.session_id)
    broadcast = AsyncMock(wraps=session.manager.broadcast_event)

    def fail_save(record: SessionRecord) -> None:
        raise OSError("save failed")

    with monkeypatch.context() as scoped:
        scoped.setattr(factory._store, "save", fail_save)
        scoped.setattr(session.manager, "broadcast_event", broadcast)
        with pytest.raises(OSError, match="save failed"):
            await task_extension(session).tools["update_task_list"]("- [ ] not durable")
        assert await task_events(factory, session.session_id) == before
        assert factory._store.load(session.session_id).task_list_markdown == ""
        broadcast.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", ["", "- [ ] failed replacement"])
async def test_session_pump_never_publishes_failed_save_from_earlier_queued_snapshot(
    factory: AgentFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replacement: str
) -> None:
    session = await factory.open_session(str(tmp_path))
    committed = "- [x] committed task list"
    writer = task_extension(session).tools["update_task_list"]
    await writer(committed)
    before = await task_events(factory, session.session_id)

    def fail_save(record: SessionRecord) -> None:
        raise OSError("replacement save failed")

    with monkeypatch.context() as scoped:
        scoped.setattr(factory._store, "save", fail_save)
        with pytest.raises(OSError, match="replacement save failed"):
            await writer(replacement)
    assert await task_events(factory, session.session_id) == before
    assert factory._store.load(session.session_id).task_list_markdown == committed

    conn = _FakeConn()
    delivered = asyncio.Event()
    notify = conn.session_update

    async def observe(session_id: str, update: Any, source: str = "") -> None:
        await notify(session_id, update, source)
        if sum(isinstance(item, AgentPlanUpdate) for item in conn.updates) == len(before):
            delivered.set()

    monkeypatch.setattr(conn, "session_update", observe)
    server = AcpAgent(factory)
    server.on_connect(cast(Client, conn))
    server._start_session_tasks(session)
    await asyncio.wait_for(delivered.wait(), timeout=10)
    plans = [update for update in conn.updates if isinstance(update, AgentPlanUpdate)]
    assert [[(entry.content, entry.status) for entry in plan.entries] for plan in plans] == [
        [("committed task list", "completed")]
    ] * len(before)
    assert session.record.task_list_markdown == committed


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", ["", "- [ ] pending replacement"])
async def test_session_pump_never_publishes_pending_save_from_earlier_queued_snapshot(
    factory: AgentFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replacement: str
) -> None:
    session = await factory.open_session(str(tmp_path))
    committed = "- [x] committed task list"
    await task_extension(session).tools["update_task_list"](committed)
    before = await task_events(factory, session.session_id)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def gated_save(candidate: SessionRecord) -> None:
        assert candidate is not session.record
        assert candidate.task_list_markdown == replacement
        entered.set()
        await release.wait()
        factory._store.save(candidate)

    extension = AgentFactory._task_list_extension(
        session.record, {"agent": session.agent}, MODE_BUILD, save_session=gated_save
    )
    caller = asyncio.create_task(extension.tools["update_task_list"](replacement))
    try:
        await asyncio.wait_for(entered.wait(), timeout=10)
        conn = _FakeConn()
        delivered = asyncio.Event()
        notify = conn.session_update

        async def observe(session_id: str, update: Any, source: str = "") -> None:
            await notify(session_id, update, source)
            if sum(isinstance(item, AgentPlanUpdate) for item in conn.updates) == len(before):
                delivered.set()

        monkeypatch.setattr(conn, "session_update", observe)
        server = AcpAgent(factory)
        server.on_connect(cast(Client, conn))
        server._start_session_tasks(session)
        await asyncio.wait_for(delivered.wait(), timeout=10)
        plans = [update for update in conn.updates if isinstance(update, AgentPlanUpdate)]
        assert [[(entry.content, entry.status) for entry in plan.entries] for plan in plans] == [
            [("committed task list", "completed")]
        ] * len(before)
        assert session.record.task_list_markdown == committed
        assert factory._store.load(session.session_id).task_list_markdown == committed
        release.set()
        await asyncio.wait_for(caller, timeout=10)
        assert session.record.task_list_markdown == replacement
        assert factory._store.load(session.session_id).task_list_markdown == replacement
    finally:
        release.set()
        await asyncio.gather(caller, return_exceptions=True)


@pytest.mark.asyncio
async def test_retired_emitter_keeps_old_generation_and_cannot_replace_current_bridge_plan(
    factory: AgentFactory, tmp_path: Path
) -> None:
    session = await factory.open_session(str(tmp_path))
    old_emitter = session.agent.emitter
    old_generation = getattr(session.agent, "_acp_generation")
    await factory._rebuild_agent(session, session.config)
    current_generation = getattr(session.agent, "_acp_generation")
    assert current_generation != old_generation

    await task_extension(session).tools["update_task_list"]("- [x] current work")
    await old_emitter.task_list_updated("- [ ] late retired work")
    events = await task_events(factory, session.session_id)
    assert events[-2].content == {"markdown": "- [x] current work", "acp_generation": current_generation}
    assert events[-1].content == {"markdown": "- [ ] late retired work", "acp_generation": old_generation}
    conn = _FakeConn()
    bridge = AcpBridge(cast(Client, conn), agent=session.agent)
    for event in events:
        await bridge.handle_event(session.session_id, event)
    assert len(conn.updates) == 1
    plan = conn.updates[0]
    assert isinstance(plan, AgentPlanUpdate)
    assert [(entry.content, entry.status) for entry in plan.entries] == [("current work", "completed")]
    assert factory._store.load(session.session_id).task_list_markdown == "- [x] current work"


@pytest.mark.asyncio
@pytest.mark.parametrize("markdown", ["", "- [x] committed before rebuild"])
@pytest.mark.parametrize("rebind_bridge", [False, True])
async def test_session_pump_preserves_queued_task_snapshot_across_generation_rebind(
    factory: AgentFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, markdown: str, rebind_bridge: bool
) -> None:
    session = await factory.open_session(str(tmp_path))
    while not session.manager.events.empty():
        session.manager.events.get_nowait()
    entered = asyncio.Event()
    release = asyncio.Event()
    delivered = asyncio.Event()
    conn = _FakeConn()
    notify = conn.session_update

    async def gated_notify(session_id: str, update: Any, source: str = "") -> None:
        if not isinstance(update, AgentPlanUpdate):
            entered.set()
            await release.wait()
        await notify(session_id, update, source)
        if isinstance(update, AgentPlanUpdate):
            delivered.set()

    monkeypatch.setattr(conn, "session_update", gated_notify)
    server = AcpAgent(factory)
    server.on_connect(cast(Client, conn))
    server._start_session_tasks(session)
    assert isinstance(session.bridge, AcpBridge)
    await session.agent.emitter.chat(message_type="tool_call", content="blocked notification")
    try:
        await asyncio.wait_for(entered.wait(), timeout=10)
        old_emitter = session.agent.emitter
        await task_extension(session).tools["update_task_list"](markdown)
        await factory._rebuild_agent(session, session.config)
        if rebind_bridge:
            session.bridge.begin_turn(session.agent, AcpDiffProvider.for_session(session))
        release.set()
        await asyncio.wait_for(delivered.wait(), timeout=10)
        plans = [update for update in conn.updates if isinstance(update, AgentPlanUpdate)]
        assert [[(entry.content, entry.status) for entry in plan.entries] for plan in plans] == (
            [[("committed before rebuild", "completed")]] if markdown else [[]]
        )
        delivered.clear()
        await old_emitter.task_list_updated("- [ ] forbidden retired replacement")
        await asyncio.wait_for(delivered.wait(), timeout=10)
        plans = [update for update in conn.updates if isinstance(update, AgentPlanUpdate)]
        assert all(
            [(entry.content, entry.status) for entry in plan.entries]
            == ([("committed before rebuild", "completed")] if markdown else [])
            for plan in plans
        )
        assert factory._store.load(session.session_id).task_list_markdown == markdown
    finally:
        release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_stage", ["save", "emit"])
async def test_acp_update_drains_repeated_cancellation_through_real_save_and_recording(
    factory: AgentFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, blocked_stage: str
) -> None:
    session = await factory.open_session(str(tmp_path))
    before = await task_events(factory, session.session_id)
    entered = asyncio.Event()
    release = asyncio.Event()
    owners: list[asyncio.Task | None] = []
    markdown = "- [x] saved through cancellation"
    emit = session.agent.emitter.task_list_updated

    async def pause() -> None:
        owners.append(asyncio.current_task())
        entered.set()
        await release.wait()

    async def save(candidate: SessionRecord) -> None:
        if blocked_stage == "save":
            await pause()
        factory._store.save(candidate)

    async def gated_emit(text: str) -> None:
        assert factory._store.load(session.session_id).task_list_markdown == markdown
        if blocked_stage == "emit":
            await pause()
        await emit(text)

    monkeypatch.setattr(session.agent.emitter, "task_list_updated", gated_emit)
    extension = AgentFactory._task_list_extension(
        session.record,
        {"agent": session.agent},
        MODE_BUILD,
        save_session=save,
        is_current=lambda: factory._acp_sessions.get(session.session_id) is session and not session.closed,
    )
    caller = asyncio.create_task(extension.tools["update_task_list"](markdown))
    try:
        await asyncio.wait_for(entered.wait(), timeout=10)
        for _ in range(3):
            caller.cancel()
            await asyncio.sleep(0)
            assert not caller.done()
        assert await task_events(factory, session.session_id) == before
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(caller, timeout=10)
        events = await task_events(factory, session.session_id)
        assert [event.content["markdown"] for event in events[len(before) :]] == [markdown]
        assert factory._store.load(session.session_id).task_list_markdown == markdown
        assert owners[0] is not None and owners[0].done()
        await asyncio.sleep(0)
        assert await task_events(factory, session.session_id) == events
    finally:
        release.set()
        await asyncio.gather(caller, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("markdown", ["", "- [x] restored\n- [ ] pending"])
async def test_transcript_replay_always_refreshes_plan_from_metadata(
    factory: AgentFactory, tmp_path: Path, markdown: str
) -> None:
    session = await factory.open_session(str(tmp_path))
    await task_extension(session).tools["update_task_list"]("- [ ] obsolete journal snapshot")
    session.record.task_list_markdown = markdown
    factory._store.save(session.record)
    conn = _FakeConn()
    server = AcpAgent(factory)
    server.on_connect(cast(Client, conn))

    await server._replay_transcript(session)

    plans = [update for update in conn.updates if isinstance(update, AgentPlanUpdate)]
    assert len(plans) == 1
    assert [(entry.content, entry.status) for entry in plans[0].entries] == (
        [("restored", "completed"), ("pending", "pending")] if markdown else []
    )


def task_event(kind: str, markdown: object, **kwargs: Any) -> AgentEvent:
    content = (
        {"markdown": markdown}
        if kind == "canonical"
        else {"message_type": "task_list_update", "text": markdown, "tool_call_id": "update-1"}
    )
    return AgentEvent(
        sender="agent",
        event_type=KnownEventType.TASK_LIST_UPDATED if kind == "canonical" else "chat_message",
        content=content,
        **kwargs,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["canonical", "legacy"])
async def test_bridge_accepts_task_strings_and_empty_clear(kind: str) -> None:
    conn = _FakeConn()
    bridge = AcpBridge(cast(Client, conn))
    await bridge.handle_event("session-1", task_event(kind, "- [X] done\n- [ ] next\n- plain"))
    await bridge.handle_event("session-1", task_event(kind, ""))
    plans = [update for update in conn.updates if isinstance(update, AgentPlanUpdate)]
    assert len(plans) == 2
    assert [(entry.content, entry.status) for entry in plans[0].entries] == [
        ("done", "completed"),
        ("next", "pending"),
        ("plain", "pending"),
    ]
    assert plans[1].entries == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["canonical", "legacy"])
@pytest.mark.parametrize("markdown", [None, False, 0, 7, [], ["- [ ] coerced"], {"text": "- [ ] coerced"}])
@pytest.mark.parametrize("use_metadata", [False, True])
async def test_bridge_rejects_non_string_task_payloads(kind: str, markdown: object, use_metadata: bool) -> None:
    conn = _FakeConn()
    bridge = AcpBridge(cast(Client, conn), task_list_markdown=(lambda: "- [x] durable") if use_metadata else None)
    await bridge.handle_event("session-1", task_event(kind, markdown))
    assert conn.updates == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["canonical", "legacy"])
async def test_bridge_never_publishes_worker_task_lists(kind: str) -> None:
    conn = _FakeConn()
    bridge = AcpBridge(cast(Client, conn))
    await bridge.handle_event(
        "session-1",
        task_event(kind, "- [ ] worker private work", sub_agent_info={"parent_tool_call_id": "dispatch-1"}),
    )
    assert conn.updates == []


@pytest.mark.asyncio
@pytest.mark.parametrize("generation", [None, "", "retired-generation", 7])
async def test_bridge_rejects_canonical_events_from_missing_or_stale_generations(generation: object) -> None:
    conn = _FakeConn()
    bridge = AcpBridge(cast(Client, conn), agent=SimpleNamespace(_acp_generation="current-generation"))
    stale = task_event("canonical", "- [ ] stale")
    if generation is not None:
        stale.content["acp_generation"] = generation
    await bridge.handle_event("session-1", stale)
    assert conn.updates == []
    current = task_event("canonical", "")
    current.content["acp_generation"] = "current-generation"
    await bridge.handle_event("session-1", current)
    assert len(conn.updates) == 1
    assert isinstance(conn.updates[0], AgentPlanUpdate) and conn.updates[0].entries == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["canonical", "legacy"])
@pytest.mark.parametrize("length", [499, 500, 501])
async def test_bridge_parses_tasks_before_capping_each_entry_at_500_characters(kind: str, length: int) -> None:
    conn = _FakeConn()
    bridge = AcpBridge(cast(Client, conn))
    text = "界" * length
    await bridge.handle_event("session-1", task_event(kind, f"# Work\n  - [X] {text}\n- [ ] next\n- [ ]"))
    assert len(conn.updates) == 1
    plan = conn.updates[0]
    assert isinstance(plan, AgentPlanUpdate)
    assert [(entry.content, entry.status) for entry in plan.entries] == [
        (text[:500], "completed"),
        ("next", "pending"),
    ]
