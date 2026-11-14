from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from kolega_code.gateway import daemon as daemon_module
from kolega_code.gateway import sessions
from kolega_code.gateway.control_relay import ControlRelay
from kolega_code.gateway.daemon import GatewayDaemon, GatewayDaemonError, GatewayShutdownIncomplete
from kolega_code.gateway.event_router import EventRouter
from kolega_code.gateway.sessions import AgentTurnHandler, ChatLifecycleIncomplete, ChatSession

from .test_sessions import RecordingAdapter, chat_ref, inbound, wait_for, wait_for_turn_done
from .test_shared_composition import CHAT, _RecorderAwareAgent
from .test_shared_composition import gateway as gateway
from .test_shared_composition import startup_resources as startup_resources

pytestmark = pytest.mark.usefixtures("isolated_cli_env")


@pytest_asyncio.fixture(autouse=True, params=[False, True], ids=["lazy", "eager"])
async def lifecycle_scheduling(request: pytest.FixtureRequest) -> AsyncIterator[None]:
    loop = asyncio.get_running_loop()
    original = loop.get_task_factory()
    if request.param:
        factory = getattr(asyncio, "eager_task_factory", None)
        if factory is None:
            pytest.skip("Eager task factories require Python 3.12")
        loop.set_task_factory(factory)
    try:
        yield
    finally:
        loop.set_task_factory(original)


async def settled(task: asyncio.Task[Any]) -> None:
    done, _ = await asyncio.wait({task}, timeout=10)
    assert done == {task}, "retained lifecycle did not settle after its gate was released"


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["agent", "relay", "router"])
@pytest.mark.parametrize("failure_stage", ["initialize", "background"])
async def test_startup_failure_reports_before_ordered_cleanup_and_serves_other_chats(
    gateway: AgentTurnHandler,
    monkeypatch: pytest.MonkeyPatch,
    startup_resources: tuple[list[EventRouter], list[ControlRelay]],
    stage: str,
    failure_stage: str,
) -> None:
    monkeypatch.setattr(sessions, "LIFECYCLE_WAIT_SECONDS", 60.0)
    entered, release = asyncio.Event(), asyncio.Event()
    order: list[str] = []
    agents: list[_RecorderAwareAgent] = []
    failure = RuntimeError("startup failed")

    class Agent(_RecorderAwareAgent):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            agents.append(self)
            if len(agents) == 1 and failure_stage == "initialize":
                self.tool_collection.initialize.side_effect = failure

        async def cleanup(self) -> None:
            self.cleanup_calls += 1
            if self is agents[0]:
                order.append("agent-start")
                if stage == "agent":
                    entered.set()
                    await release.wait()
                order.append("agent-end")

    relay_stop, router_stop = ControlRelay.stop, EventRouter.stop

    async def stop_relay(relay: ControlRelay) -> None:
        if relay is startup_resources[1][0]:
            order.append("relay-start")
            if stage == "relay":
                entered.set()
                await release.wait()
            await relay_stop(relay)
            order.append("relay-end")
        else:
            await relay_stop(relay)

    async def stop_router(router: EventRouter) -> None:
        if router is startup_resources[0][0]:
            order.append("router-start")
            if stage == "router":
                entered.set()
                await release.wait()
            await router_stop(router)
            order.append("router-end")
        else:
            await router_stop(router)

    monkeypatch.setattr(sessions, "CoderAgent", Agent)
    monkeypatch.setattr(ControlRelay, "stop", stop_relay)
    monkeypatch.setattr(EventRouter, "stop", stop_router)
    attach_background = gateway._attach_background

    async def attach(entry: ChatSession) -> None:
        if entry.chat_ref == CHAT and len(agents) == 1 and failure_stage == "background":
            raise failure
        await attach_background(entry)

    monkeypatch.setattr(gateway, "_attach_background", attach)
    request = asyncio.create_task(gateway.handle(CHAT, inbound("first")))
    try:
        await asyncio.wait_for(entered.wait(), 10)
        await settled(request)
        with pytest.raises(RuntimeError) as raised:
            request.result()
        assert raised.value is failure
        owner = gateway._operations[CHAT.key]
        assert owner.cleaning
        assert owner.task is not None and not owner.task.done()
        cancellations = owner.task.cancelling()
        for text in ("again", "/new", "/reset", "/model anthropic/other", "/permissions auto"):
            with pytest.raises(ChatLifecycleIncomplete):
                await gateway.handle(CHAT, inbound(text))
        await gateway.handle(CHAT, inbound("/stop"))
        assert owner.task.cancelling() == cancellations
        assert owner.task is gateway._operations[CHAT.key].task
        pending_background = SimpleNamespace(
            closed=False, chat_ref=CHAT, background_queued=False, background_retry=None
        )
        gateway._wake_background(cast(ChatSession, pending_background))
        assert CHAT.key not in gateway._workers
        assert CHAT.key not in gateway._queues
        assert not release.is_set()
        await gateway.handle(chat_ref("other"), inbound("other", chat_id="other"))
        adapter = cast(RecordingAdapter, gateway._adapter)
        assert await wait_for(lambda: ("other", "done") in adapter.sent)
        await wait_for_turn_done(gateway, chat_ref("other"))
        assert not owner.task.done()
        release.set()
        await settled(owner.task)
        with pytest.raises(RuntimeError) as raised:
            owner.task.result()
        assert raised.value is failure
        assert order == ["agent-start", "agent-end", "relay-start", "relay-end", "router-start", "router-end"]
        assert agents[0].cleanup_calls == 1
        assert startup_resources[0][0]._task is None
        assert startup_resources[1][0]._pump_task is None
        assert startup_resources[1][0]._runtime.control.controller is None
        assert CHAT.key not in gateway._operations
        await gateway._ensure_session(CHAT)
    finally:
        release.set()
        await asyncio.gather(request, return_exceptions=True)


@pytest.mark.asyncio
async def test_startup_timeout_interrupts_initialization_but_retains_cleanup(
    gateway: AgentTurnHandler, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The wait must expire while initialization is blocked; 0.05 s also covered
    # the session build, so a loaded runner cancelled before initialize was entered.
    monkeypatch.setattr(sessions, "LIFECYCLE_WAIT_SECONDS", 2.0)
    initializing, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    agents: list[_RecorderAwareAgent] = []
    wait_operation = AgentTurnHandler._wait_operation

    async def wait_after_initialization(handler: AgentTurnHandler, owner: sessions._LifecycleOperation) -> Any:
        # Exercise initialization timeout, not the preceding recording/presentation setup.
        await asyncio.wait_for(initializing.wait(), 10)
        return await wait_operation(handler, owner)

    async def initialize() -> list[str]:
        initializing.set()
        await asyncio.Future()
        return []

    class Agent(_RecorderAwareAgent):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            agents.append(self)
            self.tool_collection.initialize.side_effect = initialize

        async def cleanup(self) -> None:
            self.cleanup_calls += 1
            cleaning.set()
            await release.wait()

    monkeypatch.setattr(sessions, "CoderAgent", Agent)
    monkeypatch.setattr(AgentTurnHandler, "_wait_operation", wait_after_initialization)
    request = asyncio.create_task(gateway._ensure_session(CHAT))
    try:
        await asyncio.wait_for(initializing.wait(), 10)
        await settled(request)
        with pytest.raises(ChatLifecycleIncomplete):
            request.result()
        await asyncio.wait_for(cleaning.wait(), 10)
        owner = gateway._operations[CHAT.key]
        assert owner.task is not None and owner.task.cancelling() == 1
        await gateway.handle(CHAT, inbound("/stop"))
        assert owner.task.cancelling() == 1
        with pytest.raises(ChatLifecycleIncomplete):
            await gateway._ensure_session(CHAT)
        release.set()
        await settled(owner.task)
        assert owner.task.cancelled()
        assert agents[0].cleanup_calls == 1
        assert not gateway._operations
    finally:
        release.set()
        await asyncio.gather(request, return_exceptions=True)


@pytest.mark.asyncio
async def test_dispatch_cancellation_wins_ready_startup_failure_without_cancelling_cleanup(
    gateway: AgentTurnHandler, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    failure = RuntimeError("startup failed")
    agents: list[_RecorderAwareAgent] = []
    daemon = GatewayDaemon(replace(gateway._config, allowed_users=("7",)), gateway._adapter, gateway)

    class Agent(_RecorderAwareAgent):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            agents.append(self)
            self.tool_collection.initialize.side_effect = failure

        async def cleanup(self) -> None:
            self.cleanup_calls += 1
            owner = gateway._operations[CHAT.key]
            assert owner.cleaning and owner.startup_failure is not None and owner.startup_failure.done()
            assert daemon._dispatch_task is not None
            daemon._dispatch_task.cancel("cancel at failure handoff")
            entered.set()
            await release.wait()

    monkeypatch.setattr(sessions, "CoderAgent", Agent)
    await daemon.start()
    try:
        await gateway._adapter.inbound.put(inbound("first", "first"))
        await asyncio.wait_for(entered.wait(), 10)
        dispatch = daemon._dispatch_task
        assert dispatch is not None
        await settled(dispatch)
        with pytest.raises(asyncio.CancelledError, match="cancel at failure handoff"):
            dispatch.result()
        owner = gateway._operations[CHAT.key]
        assert owner.task is not None and not owner.task.done() and not owner.task.cancelling()
        await gateway._adapter.inbound.put(inbound("other", "other", chat_id="other"))
        with pytest.raises(ChatLifecycleIncomplete):
            await gateway._ensure_session(CHAT)
        release.set()
        await settled(owner.task)
        with pytest.raises(RuntimeError) as raised:
            owner.task.result()
        assert raised.value is failure
        assert len(agents) == agents[0].cleanup_calls == 1
        assert gateway._adapter.inbound.qsize() == 1
        assert not gateway._operations
    finally:
        release.set()
        await daemon.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["router", "relay"])
async def test_owner_precedes_partially_failing_transport_start(
    gateway: AgentTurnHandler, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    routers: list[EventRouter] = []
    relays: list[ControlRelay] = []
    original_router, original_relay = EventRouter.start, ControlRelay.start
    failure = RuntimeError("partially started transport")

    def start_router(router: EventRouter) -> None:
        owner = gateway._operations[CHAT.key]
        assert owner.router is router and owner.relay is not None
        routers.append(router)
        original_router(router)
        if stage == "router":
            raise failure

    def start_relay(relay: ControlRelay) -> None:
        owner = gateway._operations[CHAT.key]
        assert owner.relay is relay and owner.router is not None
        relays.append(relay)
        original_relay(relay)
        if stage == "relay":
            raise failure

    monkeypatch.setattr(EventRouter, "start", start_router)
    monkeypatch.setattr(ControlRelay, "start", start_relay)
    with pytest.raises(RuntimeError) as raised:
        await gateway._ensure_session(CHAT)
    assert raised.value is failure
    owner = gateway._operations.get(CHAT.key)
    if owner is not None:
        assert owner.task is not None
        await settled(owner.task)
    assert routers[0]._task is None
    assert all(relay._pump_task is None and relay._runtime.control.controller is None for relay in relays)
    assert not gateway._operations
    assert gateway._registry.get(CHAT) is None


@pytest.mark.asyncio
async def test_cancelled_late_startup_cannot_publish_after_shutdown(
    gateway: AgentTurnHandler,
    monkeypatch: pytest.MonkeyPatch,
    startup_resources: tuple[list[EventRouter], list[ControlRelay]],
) -> None:
    initialize_started, interrupted, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    agents: list[_RecorderAwareAgent] = []

    async def initialize() -> list[str]:
        initialize_started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            interrupted.set()
            await release.wait()
        return []

    class Agent(_RecorderAwareAgent):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            agents.append(self)
            self.tool_collection.initialize.side_effect = initialize

    monkeypatch.setattr(sessions, "CoderAgent", Agent)
    monkeypatch.setattr(sessions, "SHUTDOWN_WAIT_SECONDS", 0.05)
    request = asyncio.create_task(gateway._ensure_session(CHAT))
    try:
        await asyncio.wait_for(initialize_started.wait(), 10)
        request.cancel("request stopped")
        with pytest.raises(asyncio.CancelledError, match="request stopped"):
            await request
        await asyncio.wait_for(interrupted.wait(), 10)
        owner = gateway._operations[CHAT.key]
        assert owner.task is not None
        for _ in range(2):
            with pytest.raises(GatewayShutdownIncomplete):
                await gateway.shutdown()
        assert owner.task.cancelling() == 1
        with pytest.raises(ChatLifecycleIncomplete):
            await gateway.handle(chat_ref("other"), inbound("no admission", chat_id="other"))
        assert gateway._registry.get(CHAT) is None
        release.set()
        await asyncio.wait_for(gateway.wait_closed(), 10)
        assert owner.task.cancelled()
        assert gateway._registry.active_count() == 0
        assert startup_resources[1][0]._runtime.agent is None
        assert agents[0].cleanup_calls == 1
        assert startup_resources[0][0]._task is None
        assert startup_resources[1][0]._pump_task is None
        await gateway.shutdown()
        assert agents[0].cleanup_calls == 1
    finally:
        release.set()
        await asyncio.gather(request, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_failed_rebuild_owns_only_new_generation_and_fences_background(
    gateway: AgentTurnHandler, monkeypatch: pytest.MonkeyPatch, cleanup_fails: bool
) -> None:
    entry = await gateway._ensure_session(CHAT)
    old = entry.runtime.agent
    assert isinstance(old, _RecorderAwareAgent)
    entered, release = asyncio.Event(), asyncio.Event()
    agents: list[_RecorderAwareAgent] = []
    failure = RuntimeError("rebuild initialization failed")

    class Agent(_RecorderAwareAgent):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            agents.append(self)
            self.tool_collection.initialize.side_effect = failure

        async def cleanup(self) -> None:
            self.cleanup_calls += 1
            entered.set()
            await release.wait()
            if cleanup_fails:
                raise RuntimeError("cleanup is non-idempotent and failed")

    monkeypatch.setattr(sessions, "CoderAgent", Agent)
    monkeypatch.setattr(sessions, "LIFECYCLE_WAIT_SECONDS", 0.05)
    request = asyncio.create_task(gateway._apply_model(CHAT, entry, "anthropic/claude-haiku-4-5-20251001"))
    try:
        await asyncio.wait_for(entered.wait(), 10)
        await settled(request)
        with pytest.raises(ChatLifecycleIncomplete):
            request.result()
        owner = gateway._operations[CHAT.key]
        assert owner.task is not None
        for text in ("/reset", "/new", "/model anthropic/third", "queued input"):
            with pytest.raises(ChatLifecycleIncomplete):
                await gateway.handle(CHAT, inbound(text))
        gateway._wake_background(entry)
        assert CHAT.key not in gateway._workers
        assert old.cleanup_calls == 0
        assert entry.runtime.agent is old
        assert entry.config_holder["config"] is entry.agent_config
        release.set()
        await settled(owner.task)
        with pytest.raises(RuntimeError) as raised:
            owner.task.result()
        assert raised.value is failure
        assert agents[0].cleanup_calls == 1
        assert entry.runtime.agent is old
        if cleanup_fails:
            assert owner.cleanup_failed
            for _ in range(2):
                with pytest.raises(GatewayShutdownIncomplete):
                    await gateway.shutdown()
            assert old.cleanup_calls == 0
            assert agents[0].cleanup_calls == 1
        else:
            assert CHAT.key not in gateway._operations
            await gateway.handle(CHAT, inbound("old generation still works"))
            adapter = cast(RecordingAdapter, gateway._adapter)
            assert await wait_for(lambda: ("42", "done") in adapter.sent)
            await wait_for_turn_done(gateway, CHAT)
    finally:
        release.set()
        await asyncio.gather(request, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["agent", "relay", "router"])
async def test_reset_retains_exact_owner_through_shutdown(
    gateway: AgentTurnHandler, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    entry = await gateway._ensure_session(CHAT)
    agent = entry.runtime.agent
    assert agent is not None
    entered, release = asyncio.Event(), asyncio.Event()
    original = {"agent": agent.cleanup, "relay": entry.relay.stop, "router": entry.router.stop}[stage]

    async def cleanup() -> None:
        entered.set()
        await release.wait()
        await original()

    resource = {"agent": agent, "relay": entry.relay, "router": entry.router}[stage]
    cleanup_mock = AsyncMock(side_effect=cleanup)
    monkeypatch.setattr(resource, "cleanup" if stage == "agent" else "stop", cleanup_mock)
    monkeypatch.setattr(sessions, "LIFECYCLE_WAIT_SECONDS", 0.05)
    monkeypatch.setattr(sessions, "SHUTDOWN_WAIT_SECONDS", 0.05)
    request = asyncio.create_task(gateway._reset_session(CHAT))
    try:
        await asyncio.wait_for(entered.wait(), 10)
        await settled(request)
        with pytest.raises(ChatLifecycleIncomplete):
            request.result()
        owner = gateway._operations[CHAT.key]
        assert owner.task is not None and not owner.task.done()
        assert gateway._chat_sessions[CHAT.key] == entry.session_id
        with pytest.raises(ChatLifecycleIncomplete):
            await gateway._reset_session(CHAT)
        gateway._wake_background(entry)
        assert CHAT.key not in gateway._workers
        with pytest.raises(GatewayShutdownIncomplete):
            await gateway.shutdown()
        assert gateway._operations[CHAT.key] is owner
        release.set()
        await asyncio.wait_for(gateway.wait_closed(), 10)
        await gateway.shutdown()
        assert cleanup_mock.await_count == 1
        assert agent.cleanup_calls == 1
        assert CHAT.key not in gateway._chat_sessions
        assert not gateway._operations
    finally:
        release.set()
        await asyncio.gather(request, return_exceptions=True)


@pytest.mark.asyncio
async def test_shutdown_drains_chats_concurrently_and_survives_waiter_cancellation(
    gateway: AgentTurnHandler, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = await gateway._ensure_session(CHAT)
    second = await gateway._ensure_session(chat_ref("other"))
    assert first.runtime.agent is not None and second.runtime.agent is not None
    entered, release, other_closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    first_cleanup, second_cleanup = first.runtime.agent.cleanup, second.runtime.agent.cleanup

    async def close_first() -> None:
        entered.set()
        await release.wait()
        await first_cleanup()

    async def close_second() -> None:
        await second_cleanup()
        other_closed.set()

    first_mock, second_mock = AsyncMock(side_effect=close_first), AsyncMock(side_effect=close_second)
    monkeypatch.setattr(first.runtime.agent, "cleanup", first_mock)
    monkeypatch.setattr(second.runtime.agent, "cleanup", second_mock)
    monkeypatch.setattr(sessions, "SHUTDOWN_WAIT_SECONDS", 0.05)
    request = asyncio.create_task(gateway.shutdown())
    try:
        await asyncio.wait_for(entered.wait(), 10)
        await asyncio.wait_for(other_closed.wait(), 10)
        shutdown_owner = gateway._shutdown_task
        assert shutdown_owner is not None and not shutdown_owner.done()
        await settled(request)
        with pytest.raises(GatewayShutdownIncomplete):
            request.result()
        for _ in range(2):
            with pytest.raises(GatewayShutdownIncomplete):
                await gateway.shutdown()
        waiter = asyncio.create_task(gateway.shutdown())
        await asyncio.sleep(0)
        waiter.cancel("cancel shutdown wait")
        with pytest.raises(asyncio.CancelledError, match="cancel shutdown wait"):
            await waiter
        assert gateway._shutdown_task is shutdown_owner
        assert not shutdown_owner.cancelling()
        assert first_mock.await_count == second_mock.await_count == 1
        release.set()
        await asyncio.wait_for(gateway.wait_closed(), 10)
        await gateway.shutdown()
        assert first_mock.await_count == second_mock.await_count == 1
    finally:
        release.set()
        await asyncio.gather(request, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt_dispatch", [False, True])
async def test_daemon_keeps_adapter_lock_and_incomplete_status_until_cleanup_finishes(
    gateway: AgentTurnHandler, monkeypatch: pytest.MonkeyPatch, interrupt_dispatch: bool
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    agents: list[_RecorderAwareAgent] = []

    class Agent(_RecorderAwareAgent):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            agents.append(self)
            if len(agents) == 1:
                self.tool_collection.initialize.side_effect = RuntimeError("failed startup")

        async def cleanup(self) -> None:
            self.cleanup_calls += 1
            if self is agents[0]:
                entered.set()
                await release.wait()

    monkeypatch.setattr(sessions, "CoderAgent", Agent)
    monkeypatch.setattr(sessions, "LIFECYCLE_WAIT_SECONDS", 60.0)
    monkeypatch.setattr(daemon_module, "SHUTDOWN_WAIT_SECONDS", 0.05)
    adapter_stop = AsyncMock()
    monkeypatch.setattr(gateway._adapter, "stop", adapter_stop)
    config = replace(gateway._config, allowed_users=("7",))
    daemon = GatewayDaemon(config, gateway._adapter, gateway)
    await daemon.start()
    try:
        await gateway._adapter.inbound.put(inbound("first", "first"))
        await asyncio.wait_for(entered.wait(), 10)
        if not interrupt_dispatch:
            await gateway._adapter.inbound.put(inbound("other", "other", chat_id="other"))
            adapter = cast(RecordingAdapter, gateway._adapter)
            assert await wait_for(lambda: ("other", "done") in adapter.sent)
            await wait_for_turn_done(gateway, chat_ref("other"))
        for _ in range(2):
            with pytest.raises(GatewayShutdownIncomplete):
                await daemon.stop()
        stop_owner = daemon._stop_task
        assert stop_owner is not None and not stop_owner.done()
        assert gateway._closing
        assert daemon._dispatch_task is not None and daemon._dispatch_task.done()
        assert not daemon.is_message_allowed(inbound("no admission"))
        assert daemon._lock is not None and daemon._lock.is_locked
        assert daemon.status().shutdown_state == "incomplete"
        status = json.loads(daemon._status_path.read_text())
        assert status["shutdown_state"] == "incomplete" and status["pid"] is not None
        assert not status["running"]
        assert daemon._pid_path.exists()
        adapter_stop.assert_not_awaited()
        contender = GatewayDaemon(config, gateway._adapter, gateway)
        with pytest.raises(GatewayDaemonError, match="Another gateway instance"):
            await contender.start()
        release.set()
        await settled(stop_owner)
        await daemon.stop()
        assert agents[0].cleanup_calls == 1
        adapter_stop.assert_awaited_once()
        assert daemon._lock is None
        assert daemon.status().shutdown_state == "stopped"
        assert not daemon._status_path.exists() and not daemon._pid_path.exists()
    finally:
        release.set()
        await daemon.stop()


@pytest.mark.asyncio
async def test_retained_startup_admission_is_bounded(
    gateway: AgentTurnHandler, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gateway, "_config", replace(gateway._config, max_sessions=1))
    entered = [asyncio.Event(), asyncio.Event()]
    release = asyncio.Event()
    agents: list[_RecorderAwareAgent] = []

    class Agent(_RecorderAwareAgent):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            agents.append(self)
            self.tool_collection.initialize.side_effect = RuntimeError("failed startup")

        async def cleanup(self) -> None:
            self.cleanup_calls += 1
            entered[agents.index(self)].set()
            await release.wait()

    monkeypatch.setattr(sessions, "CoderAgent", Agent)
    # The wait returns as soon as the injected failure publishes; a 0.05 s
    # deadline raced the startup path and cancelled before cleanup was entered.
    monkeypatch.setattr(sessions, "LIFECYCLE_WAIT_SECONDS", 60.0)
    requests = [asyncio.create_task(gateway._ensure_session(chat_ref(str(index)))) for index in range(2)]
    try:
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in entered)), 10)
        for request in requests:
            await settled(request)
            with pytest.raises(RuntimeError, match="failed startup"):
                request.result()
        owners = list(gateway._operations.values())
        with pytest.raises(ChatLifecycleIncomplete, match="Too many"):
            await gateway._ensure_session(chat_ref("third"))
        assert len(gateway._operations) == len(agents) == 2
        release.set()
        for owner in owners:
            assert owner.task is not None
            await settled(owner.task)
        assert not gateway._operations
    finally:
        release.set()
        await asyncio.gather(*requests, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["relay", "router"])
@pytest.mark.parametrize("cancelled", [False, True])
async def test_failed_startup_transport_cleanup_is_fenced_and_never_retried(
    gateway: AgentTurnHandler,
    monkeypatch: pytest.MonkeyPatch,
    startup_resources: tuple[list[EventRouter], list[ControlRelay]],
    stage: str,
    cancelled: bool,
) -> None:
    failure = RuntimeError("initialization failed")
    order: list[str] = []
    agents: list[_RecorderAwareAgent] = []

    class Agent(_RecorderAwareAgent):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            agents.append(self)
            self.tool_collection.initialize.side_effect = failure

        async def cleanup(self) -> None:
            self.cleanup_calls += 1
            order.append("agent")

    relay_stop, router_stop = ControlRelay.stop, EventRouter.stop

    def fail() -> None:
        if cancelled:
            raise asyncio.CancelledError("transport cleanup interrupted")
        raise RuntimeError("transport cleanup failed")

    async def stop_relay(relay: ControlRelay) -> None:
        order.append("relay")
        await relay_stop(relay)
        if stage == "relay":
            fail()

    async def stop_router(router: EventRouter) -> None:
        order.append("router")
        await router_stop(router)
        if stage == "router":
            fail()

    monkeypatch.setattr(sessions, "CoderAgent", Agent)
    monkeypatch.setattr(ControlRelay, "stop", stop_relay)
    monkeypatch.setattr(EventRouter, "stop", stop_router)
    with pytest.raises(RuntimeError) as raised:
        await gateway._ensure_session(CHAT)
    assert raised.value is failure
    owner = gateway._operations[CHAT.key]
    assert owner.task is not None
    await settled(owner.task)
    assert owner.cleanup_failed
    assert order == ["agent", "relay", "router"]
    assert startup_resources[0][0]._task is None
    assert startup_resources[1][0]._pump_task is None
    for _ in range(2):
        with pytest.raises(ChatLifecycleIncomplete):
            await gateway._reset_session(CHAT)
        with pytest.raises(GatewayShutdownIncomplete):
            await gateway.shutdown()
    assert agents[0].cleanup_calls == 1
    assert order == ["agent", "relay", "router"]
    assert gateway._operations[CHAT.key] is owner


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_retired_generation_cleanup_keeps_its_owner_without_retry(
    gateway: AgentTurnHandler, monkeypatch: pytest.MonkeyPatch, cleanup_fails: bool
) -> None:
    entry = await gateway._ensure_session(CHAT)
    entered, release = asyncio.Event(), asyncio.Event()
    retired = SimpleNamespace(cleanup=AsyncMock())

    async def cleanup() -> None:
        entered.set()
        await release.wait()
        if cleanup_fails:
            raise RuntimeError("retired generation cleanup failed")

    retired.cleanup.side_effect = cleanup
    entry.retired_agents.append(retired)
    entry.background_jobs = SimpleNamespace(
        deliver=AsyncMock(),
        active_count=0,
        pending_count=0,
        has_work=False,
        close=AsyncMock(),
        set_callbacks=lambda **kwargs: None,
    )
    monkeypatch.setattr(sessions, "LIFECYCLE_WAIT_SECONDS", 0.05)
    request = asyncio.create_task(gateway._run_owned_turn(entry, CHAT, None))
    try:
        await asyncio.wait_for(entered.wait(), 10)
        await settled(request)
        with pytest.raises(ChatLifecycleIncomplete):
            request.result()
        owner = gateway._operations[CHAT.key]
        assert owner.partial_agent is retired and owner.entry is entry
        assert owner.task is not None
        await gateway._stop_turn(CHAT)
        assert not owner.task.cancelling()
        with pytest.raises(ChatLifecycleIncomplete):
            await gateway._reset_session(CHAT)
        release.set()
        await settled(owner.task)
        if cleanup_fails:
            assert owner.cleanup_failed and owner.partial_agent is retired
            for _ in range(2):
                with pytest.raises(GatewayShutdownIncomplete):
                    await gateway.shutdown()
        else:
            assert CHAT.key not in gateway._operations
            await gateway.shutdown()
        retired.cleanup.assert_awaited_once()
        assert not entry.retired_agents
    finally:
        release.set()
        await asyncio.gather(request, return_exceptions=True)


@pytest.mark.asyncio
async def test_pruning_retains_every_eviction_before_waiting_for_blocked_cleanup(
    gateway: AgentTurnHandler, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = await gateway._ensure_session(CHAT)
    other = chat_ref("other")
    second = await gateway._ensure_session(other)
    assert first.runtime.agent is not None and second.runtime.agent is not None
    entered, release, other_closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    first_cleanup, second_cleanup = first.runtime.agent.cleanup, second.runtime.agent.cleanup

    async def cleanup_first() -> None:
        entered.set()
        await release.wait()
        await first_cleanup()

    async def cleanup_second() -> None:
        await second_cleanup()
        other_closed.set()

    first_mock, second_mock = AsyncMock(side_effect=cleanup_first), AsyncMock(side_effect=cleanup_second)
    monkeypatch.setattr(first.runtime.agent, "cleanup", first_mock)
    monkeypatch.setattr(second.runtime.agent, "cleanup", second_mock)
    monkeypatch.setattr(gateway._registry, "_idle_ttl_seconds", 1.0)
    for entry in gateway._registry._entries.values():
        entry.last_active -= 10000
    monkeypatch.setattr(sessions, "LIFECYCLE_WAIT_SECONDS", 0.05)
    wait_operation = AgentTurnHandler._wait_operation

    async def wait_after_eviction(handler: AgentTurnHandler, owner: sessions._LifecycleOperation) -> Any:
        if owner.kind == "startup":
            await asyncio.wait_for(entered.wait(), 10)
        return await wait_operation(handler, owner)

    monkeypatch.setattr(AgentTurnHandler, "_wait_operation", wait_after_eviction)
    request = asyncio.create_task(gateway._ensure_session(chat_ref("third")))
    try:
        await asyncio.wait_for(entered.wait(), 10)
        await asyncio.wait_for(other_closed.wait(), 10)
        await settled(request)
        with pytest.raises(ChatLifecycleIncomplete):
            request.result()
        owner = gateway._operations[CHAT.key]
        assert owner.entry is first and owner.task is not None and not owner.task.done()
        assert gateway._registry.get(CHAT) is None and gateway._registry.get(other) is None
        assert gateway._registry.get(chat_ref("third")) is not None
        release.set()
        await settled(owner.task)
        await gateway.shutdown()
        first_mock.assert_awaited_once()
        second_mock.assert_awaited_once()
    finally:
        release.set()
        await asyncio.gather(request, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("restore_fails", [False, True])
async def test_failed_model_binding_restores_old_generation_before_cleaning_new(
    gateway: AgentTurnHandler, monkeypatch: pytest.MonkeyPatch, cancelled: bool, restore_fails: bool
) -> None:
    entry = await gateway._ensure_session(CHAT)
    old, old_config = entry.runtime.agent, entry.agent_config
    assert isinstance(old, _RecorderAwareAgent)
    entered, release = asyncio.Event(), asyncio.Event()
    agents: list[_RecorderAwareAgent] = []
    failure = asyncio.CancelledError("bind interrupted") if cancelled else RuntimeError("bind failed")

    def failed_restore(mode: Any) -> None:
        raise RuntimeError("old generation policy failed")

    if restore_fails:
        monkeypatch.setattr(old, "set_permission_mode", failed_restore)

    class Agent(_RecorderAwareAgent):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            agents.append(self)

        def set_permission_mode(self, mode: Any) -> None:
            raise failure

        async def cleanup(self) -> None:
            self.cleanup_calls += 1
            assert entry.runtime.agent is old
            assert entry.config_holder["config"] is old_config
            entered.set()
            await release.wait()

    monkeypatch.setattr(sessions, "CoderAgent", Agent)
    monkeypatch.setattr(sessions, "LIFECYCLE_WAIT_SECONDS", 0.05)
    request = asyncio.create_task(gateway._apply_model(CHAT, entry, "anthropic/claude-haiku-4-5-20251001"))
    try:
        await asyncio.wait_for(entered.wait(), 10)
        await settled(request)
        with pytest.raises(ChatLifecycleIncomplete):
            request.result()
        owner = gateway._operations[CHAT.key]
        assert owner.partial_agent is agents[0]
        assert old.cleanup_calls == 0
        assert owner.task is not None
        release.set()
        await settled(owner.task)
        if cancelled:
            assert owner.task.cancelled()
        else:
            with pytest.raises(RuntimeError) as raised:
                owner.task.result()
            assert raised.value is failure
        assert agents[0].cleanup_calls == 1 and old.cleanup_calls == 0
        assert entry.runtime.agent is old
        assert entry.agent_config is old_config
        if restore_fails:
            assert gateway._operations[CHAT.key] is owner and owner.cleanup_failed
            with pytest.raises(GatewayShutdownIncomplete):
                await gateway.shutdown()
            assert old.cleanup_calls == 0
        else:
            assert CHAT.key not in gateway._operations
    finally:
        release.set()
        await asyncio.gather(request, return_exceptions=True)
