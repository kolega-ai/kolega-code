"""The ACP agent server: kolega-code presented to editors as an ACP agent.

Phase 1 wires the transport to real ``CoderAgent`` turns: ``session/new``
and ``session/load`` create/resume durable kolega-code sessions, and
``session/prompt`` drives one turn, streaming text chunks and tool-call
lifecycle as ACP ``session/update`` notifications. ``session/cancel``
cancels the in-flight turn.

Phase 0 (transport skeleton) is preserved in the module's git history; see
``acp-implementation-plan.md`` for the full design.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable
from uuid import uuid4

from acp import Agent, InitializeResponse, NewSessionResponse, PromptResponse
from acp.connection import StreamEvent
from acp.exceptions import RequestError
from acp.helpers import (
    text_block,
    tool_content,
    update_agent_thought_text,
    update_available_commands,
    update_current_mode,
    update_plan,
    update_tool_call,
    update_user_message,
)
from acp.interfaces import Client
from acp.schema import (
    AgentCapabilities,
    AudioContentBlock,
    AvailableCommand,
    ClientCapabilities,
    ConfigOptionUpdate,
    EmbeddedResourceContentBlock,
    HttpMcpServer,
    ImageContentBlock,
    Implementation,
    ListSessionsResponse,
    LoadSessionResponse,
    McpServerStdio,
    PermissionOption,
    PromptCapabilities,
    ResourceContentBlock,
    SessionCapabilities,
    SessionCloseCapabilities,
    SessionListCapabilities,
    SetSessionConfigOptionResponse,
    SetSessionModeResponse,
    SseMcpServer,
    StopReason,
    TextContentBlock,
)

from kolega_code.acp import agent_factory
from kolega_code.acp.agent_factory import (
    CONFIG_MODE,
    CONFIG_MODEL,
    CONFIG_PERMISSION_AUTO,
    INTERACTION_MODES,
    MODE_BUILD,
    MODE_PLAN,
    AgentFactory,
)
from kolega_code.acp.accounting import USAGE_META_KEY, native_prompt_usage
from kolega_code.acp.bridge import AcpBridge, SUB_SESSION_PREFIX
from kolega_code.acp.control import AcpControlAdapter
from kolega_code.acp.diffs import AcpDiffProvider
from kolega_code.acp.plans import task_entries_from_markdown
from kolega_code.acp.session import AcpSession
from kolega_code.acp.usage import build_usage_update
from kolega_code.agent.prompts import build_implement_plan_prompt
from kolega_code.cli.config import CliConfigError
from kolega_code.cli.session_store import SessionStoreError
from kolega_code.cli.slash_commands import agent_command_entries
from kolega_code.llm.models import MessageHistory
from kolega_code.llm.ledger import logical_prompt_scope
from kolega_code.permissions import PermissionDecision, PermissionMode, PermissionRequest
from kolega_code.session.control import ControlChannel, DEFAULT_REQUEST_TIMEOUT_SECONDS
from kolega_code.session.projection import replay, sub_agent_key
from kolega_code.session.runtime import SessionRuntime

logger = logging.getLogger(__name__)

_PromptBlock = (
    TextContentBlock | ImageContentBlock | AudioContentBlock | ResourceContentBlock | EmbeddedResourceContentBlock
)

_McpServer = list[HttpMcpServer | SseMcpServer | McpServerStdio] | None

_HANDLED_CONFIG_ERRORS = (CliConfigError, SessionStoreError)

PermissionCallback = Callable[[PermissionRequest], Awaitable[PermissionDecision]]

#: Delay before post-response open updates so clients that only register the
#: session once the NewSessionResponse is processed (Zed drops every
#: pre-response notification for session/new) receive them after registration.
OPEN_UPDATES_DELAY_SECONDS = 0.25


class AcpAgent(Agent):
    """kolega-code as an ACP v1 agent, one client connection per process.

    The editor spawns this process, negotiates via ``initialize``, creates
    sessions, and drives turns with ``session/prompt``. One ``CoderAgent``
    per session; sessions persist as regular ``SessionRecord``s.
    """

    def __init__(self, factory: AgentFactory | None = None) -> None:
        self._factory = factory or AgentFactory()
        self._sessions: dict[str, AcpSession] = {}
        self._conn: Client
        self._client_capabilities: ClientCapabilities | None = None
        self._native_end_turn_usage = False
        self._shutdown_task: asyncio.Task[None] | None = None

    def on_connect(self, conn: Client) -> None:
        self._conn = conn

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        logger.info("acp initialize (protocol %s, client=%s)", protocol_version, client_info)
        self._client_capabilities = client_capabilities
        meta = getattr(client_capabilities, "field_meta", None) or kwargs.get("_meta") or kwargs.get("field_meta") or {}
        usage_meta = meta.get(USAGE_META_KEY, {})
        self._native_end_turn_usage = isinstance(usage_meta, dict) and usage_meta.get("endTurnUsage") == "disjoint-v1"
        return InitializeResponse(
            protocol_version=protocol_version,
            agent_capabilities=AgentCapabilities(
                load_session=True,
                prompt_capabilities=PromptCapabilities(image=True),
                session_capabilities=SessionCapabilities(
                    list=SessionListCapabilities(),
                    close=SessionCloseCapabilities(),
                ),
            ),
        )

    # -- session lifecycle ------------------------------------------------

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: _McpServer = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        if self._shutdown_task is not None:
            raise RequestError.invalid_request({"message": "ACP server is shutting down."})
        logger.info("acp new session (cwd=%s)", cwd)
        session_id = uuid4().hex
        try:
            session = await self._factory.open_session(
                cwd,
                session_id=session_id,
                permission_callback=self._permission_callback_for(session_id),
            )
        except _HANDLED_CONFIG_ERRORS as exc:
            raise AgentFactory.protocol_error(exc) from exc
        self._sessions[session.session_id] = session
        self._attach_runtime(session)
        response = NewSessionResponse(
            session_id=session.session_id,
            config_options=self._factory.config_options_for(session),
        )
        # Zed registers the session only once the NewSessionResponse is
        # processed and drops any pre-response notification, so the open-time
        # usage meter and command list go out just after the response instead.
        self._schedule_open_updates(session.session_id)
        return response

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: _McpServer = None,
        additional_directories: list[str] | None = None,
        **kwargs: Any,
    ) -> LoadSessionResponse | None:
        if self._shutdown_task is not None:
            raise RequestError.invalid_request({"message": "ACP server is shutting down."})
        if session_id.startswith(SUB_SESSION_PREFIX):
            # A synthetic sub-agent session Zed loads to render a delegated
            # turn as a nested transcript card under the dispatch tool call.
            return await self._load_sub_session(session_id)
        if session_id in self._sessions:
            existing = self._sessions[session_id]
            AgentFactory.check_active(existing)
            self._start_session_tasks(existing)
            return LoadSessionResponse(config_options=self._factory.config_options_for(existing))
        try:
            session = await self._factory.load_session(
                cwd,
                session_id,
                permission_callback=self._permission_callback_for(session_id),
            )
        except _HANDLED_CONFIG_ERRORS as exc:
            raise AgentFactory.protocol_error(exc) from exc
        if session is None:
            return None
        self._sessions[session.session_id] = session
        self._attach_runtime(session)
        await self._emit_initial_usage(session)
        await self._send_available_commands(session)
        await self._replay_transcript(session)
        self._start_session_tasks(session)
        return LoadSessionResponse(config_options=self._factory.config_options_for(session))

    async def _load_sub_session(self, session_id: str) -> LoadSessionResponse:
        """Replay one delegated turn as its synthetic sub-session.

        Zed renders ``_meta.subagent_session_info`` dispatch cards by loading
        the referenced child session id through ``session/load`` and nesting
        the replayed transcript under the dispatch. The child id encodes the
        parent session and the dispatching tool call
        (``sub-<parent>-<tool_call_id>``), which locates the delegate's events
        in the parent's journal.
        """
        rest = session_id[len(SUB_SESSION_PREFIX) :]
        parent_id, _, tool_call_id = rest.partition("-")
        if not parent_id or not tool_call_id:
            logger.info("acp sub-session load: malformed id %s", session_id)
            return LoadSessionResponse()
        try:
            events = await self._factory.event_store(parent_id).read(parent_id)
        except Exception:  # noqa: BLE001 — an unreadable journal yields an empty card
            logger.exception("acp sub-session load: journal read failed (parent=%s)", parent_id)
            return LoadSessionResponse()
        key: str | None = None
        for event in events:
            info = event.sub_agent_info or {}
            if str(info.get("parent_tool_call_id") or "") == tool_call_id:
                key = sub_agent_key(info)
                if key:
                    break
        if key is None:
            logger.info("acp sub-session load: no delegate for %s", session_id)
            return LoadSessionResponse()
        activity = replay(events).sub_agents.get(key)
        bridge = AcpBridge(self._conn)
        task = activity.task if activity is not None else ""
        if task:
            update = update_user_message(text_block(task))
            update.message_id = f"replay-sub-task-{tool_call_id}"
            await bridge.send(session_id, update)
        await bridge.replay_conversation(session_id, activity.steps if activity is not None else [])
        return LoadSessionResponse()

    async def list_sessions(
        self, cwd: str | None = None, cursor: str | None = None, **kwargs: Any
    ) -> ListSessionsResponse:
        logger.info("acp session/list (cwd=%s)", cwd)
        return ListSessionsResponse(sessions=self._factory.list_sessions(cwd))

    async def close_session(self, session_id: str, **kwargs: Any) -> None:
        logger.info("acp session/close (session=%s)", session_id)
        session = self._sessions.get(session_id)
        if session is None:
            await self._factory.close_session(session_id)
            return
        self._start_close(session)
        assert session.close_task is not None
        await agent_factory._finish_cleanup(session.close_task, timeout=agent_factory.CLEANUP_TIMEOUT)
        if not session.close_task.done():
            raise RequestError.invalid_request({"message": "Session close is still pending."})

    def _start_close(self, session: AcpSession) -> asyncio.Task[None]:
        session.closed = True
        session.auto_continue_paused = True
        session.approved_plan = None
        session.plan_approval_generation += 1
        if session.plan_approval_task is not None:
            session.plan_approval_task.cancel()
        session_id = session.session_id
        if session.close_task is None or (session.close_task.done() and session.close_retryable):
            session.close_retryable = False
            session.close_task = asyncio.create_task(self._close_session(session), name=f"acp-close-{session_id}")

            def completed(done: asyncio.Task[None]) -> None:
                if not done.cancelled() and done.exception() is not None:
                    logger.error("acp session close incomplete (session=%s)", session_id)

            session.close_task.add_done_callback(completed)
        return session.close_task

    async def _close_session(self, session: AcpSession) -> None:
        if session.turn_task is not None:
            session.turn_task.cancel()
            await asyncio.gather(session.turn_task, return_exceptions=True)
        if session.background_task is not None:
            session.background_task.cancel()
            await asyncio.gather(session.background_task, return_exceptions=True)
        async with session.execution_lock:
            await asyncio.gather(
                *(asyncio.shield(task) for task in tuple(session.cleanup_tasks.values())),
                return_exceptions=True,
            )
            AgentFactory.check_cleanup(session)
            if session.control_adapter is not None:
                try:
                    await session.control_adapter.shutdown()
                except Exception:
                    session.close_retryable = True
                    raise
            await self._factory.close_session(session.session_id)
        if session.event_task is not None:
            session.event_task.cancel()
            await asyncio.gather(session.event_task, return_exceptions=True)
            await self._drain_residual(session, session.bridge)
        self._sessions.pop(session.session_id, None)

    async def shutdown(self) -> None:
        """Join session work before the ACP transport and shared services close."""
        if self._shutdown_task is None:
            self._shutdown_task = asyncio.create_task(self._shutdown(), name="acp-shutdown")
        await agent_factory._finish_cleanup(self._shutdown_task)

    async def _shutdown(self) -> None:
        sessions = list(self._sessions.values())
        results = await asyncio.gather(*(self._start_close(session) for session in sessions), return_exceptions=True)
        failures: list[BaseException] = []
        for session, result in zip(sessions, results):
            if isinstance(result, BaseException):
                logger.warning("acp shutdown: session close incomplete (session=%s): %s", session.session_id, result)
                failures.append(result)
        if failures:
            raise RuntimeError("ACP shutdown incomplete") from BaseExceptionGroup("Session close failures", failures)

    # -- turns ------------------------------------------------------------

    async def prompt(self, session_id: str, prompt: list[_PromptBlock], **kwargs: Any) -> PromptResponse:
        session = self._sessions.get(session_id)
        if session is None:
            raise RequestError.resource_not_found(session_id)
        if not session.idle():
            raise RequestError.invalid_request({"message": "a turn is already running for this session"})
        text, attachments = self._extract_prompt(prompt)
        if not text and not attachments:
            raise RequestError.invalid_request({"message": "prompt contains no text or images"})
        self._start_session_tasks(session)
        async with session.execution_lock:
            AgentFactory.check_active(session)
            session.auto_continue_paused = False
            prompt_id = uuid4().hex
            with logical_prompt_scope(prompt_id):
                try:
                    stop_reason = await self._run_turn(session, text, attachments)
                    self._factory.persist(session)
                finally:
                    if session.accounting is not None:
                        await session.accounting.flush()
        if session.accounting is None:
            return PromptResponse(stop_reason=stop_reason)
        extension = await session.accounting.finish_prompt(prompt_id)
        return PromptResponse(
            stop_reason=stop_reason,
            field_meta={USAGE_META_KEY: extension},
            usage=native_prompt_usage(extension, enabled=self._native_end_turn_usage),
        )

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            return
        session.auto_continue_paused = True
        session.approved_plan = None
        session.plan_approval_generation += 1
        if session.plan_approval_task is not None:
            session.plan_approval_task.cancel()
        if session.turn_task is None:
            logger.debug("acp session/cancel: nothing to cancel (session=%s)", session_id)
            return
        logger.info("acp session/cancel (session=%s)", session_id)
        session.turn_task.cancel()

    # -- unsupported surfaces (null = "not supported" per ACP) -------------

    async def resume_session(
        self,
        session_id: str,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: _McpServer = None,
        **kwargs: Any,
    ) -> None:
        logger.info("acp session/resume not supported yet (session=%s)", session_id)
        return None

    async def fork_session(
        self,
        session_id: str,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: _McpServer = None,
        **kwargs: Any,
    ) -> None:
        logger.info("acp session/fork not supported yet (session=%s)", session_id)
        return None

    async def set_session_mode(self, session_id: str, mode_id: str, **kwargs: Any) -> SetSessionModeResponse | None:
        session = self._sessions.get(session_id)
        if session is None:
            raise RequestError.resource_not_found(session_id)
        if mode_id not in INTERACTION_MODES:
            raise RequestError.invalid_params({"message": f"unknown mode {mode_id!r}; available: build, plan"})
        if not session.idle():
            raise RequestError.invalid_request({"message": "modes apply between turns only"})
        async with session.execution_lock:
            await self._factory.set_interaction_mode(session, mode_id)
        logger.info("acp session mode -> %s (session=%s)", mode_id, session_id)
        await self._emit_current_mode(session)
        return SetSessionModeResponse()

    async def set_config_option(
        self, config_id: str, session_id: str, value: str | bool, **kwargs: Any
    ) -> SetSessionConfigOptionResponse:
        session = self._sessions.get(session_id)
        if session is None:
            raise RequestError.resource_not_found(session_id)
        if session.closed:
            raise RequestError.invalid_request({"message": "Session is closing."})
        if config_id == CONFIG_PERMISSION_AUTO:
            # Live flag flip (no rebuild), so it stays settable mid-turn.
            if isinstance(value, str):
                auto = value.strip().lower() in ("auto", "true", "1", "yes")
            else:
                auto = bool(value)
            mode = PermissionMode.AUTO if auto else PermissionMode.ASK
            await self._factory.set_permission_mode(session, mode)
            logger.info("acp session config: permission mode -> %s (session=%s)", mode.value, session_id)
            return SetSessionConfigOptionResponse(config_options=self._factory.config_options_for(session))
        if not session.idle():
            raise RequestError.invalid_request({"message": "config options apply between turns only"})
        if config_id == CONFIG_MODEL:
            if not isinstance(value, str) or "/" not in value:
                raise RequestError.invalid_params({"message": "model option expects 'provider/model'"})
            provider, model = value.split("/", 1)
            try:
                async with session.execution_lock:
                    await self._factory.apply_model(session, provider, model)
            except _HANDLED_CONFIG_ERRORS as exc:
                raise AgentFactory.protocol_error(exc) from exc
            logger.info("acp session config: model -> %s/%s (session=%s)", provider, model, session_id)
        elif config_id == CONFIG_MODE:
            if value not in INTERACTION_MODES:
                raise RequestError.invalid_params({"message": f"unknown mode {value!r}; available: build, plan"})
            async with session.execution_lock:
                await self._factory.set_interaction_mode(session, str(value))
            logger.info("acp session config: mode -> %s (session=%s)", value, session_id)
            await self._emit_current_mode(session)
        else:
            raise RequestError.invalid_params({"message": f"unknown config option {config_id!r}"})
        # The spec requires the full option set with current values so the
        # client can refresh its UI state.
        return SetSessionConfigOptionResponse(config_options=self._factory.config_options_for(session))

    async def authenticate(self, method_id: str, **kwargs: Any) -> None:
        logger.info("acp authenticate not supported yet (method=%s)", method_id)
        return None

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        raise RequestError.method_not_found(f"_{method}")

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        logger.debug("acp ignoring extension notification %s", method)

    # -- permissions --------------------------------------------------------

    def _permission_callback_for(self, session_id: str) -> PermissionCallback:
        """Late-bound callback used while the agent is initially constructed."""

        async def callback(request: PermissionRequest) -> PermissionDecision:
            session = self._sessions.get(session_id)
            if session is None or session.closed or session.runtime is None:
                return PermissionDecision(allowed=False, reason="Session is no longer active.")
            return await session.runtime.permission_callback(request)

        return callback

    def _attach_runtime(self, session: AcpSession) -> None:
        adapter = AcpControlAdapter(
            session,
            session_provider=lambda: self._sessions.get(session.session_id),
            connection_provider=lambda: getattr(self, "_conn", None),
            capabilities_provider=lambda: self._client_capabilities,
        )
        runtime = SessionRuntime(
            session_id=session.session_id,
            project_path=Path(session.record.project_path),
            control=ControlChannel(
                session_id=session.session_id,
                emit=adapter.emit,
                timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
            ),
            permission_mode=session.agent.permission_mode,
            on_notice=lambda notice: logger.warning("acp session %s: %s", session.session_id, notice),
        )
        runtime.adopt(session.agent)
        runtime.control.acquire(adapter.client_id)
        session.runtime = runtime
        session.control_adapter = adapter
        session.permission_callback = runtime.permission_callback

    async def on_transport_message(self, event: StreamEvent) -> None:
        """A successful ACP write acknowledges presentation before the human replies."""
        for session in self._sessions.values():
            if session.control_adapter is not None and await session.control_adapter.on_transport_message(event):
                return

    async def _emit_initial_usage(self, session: AcpSession) -> None:
        conn = getattr(self, "_conn", None)
        if conn is None:
            return
        if session.accounting is not None:
            session.accounting.start_publisher(lambda update: self._send_session_update(session, update))
            await session.accounting.publish(force=True)
            return
        update = build_usage_update(session.agent)
        if update is not None:
            await conn.session_update(session_id=session.session_id, update=update, source="kolega_code")

    async def _emit_current_mode(self, session: AcpSession) -> None:
        mode = session.record.interaction_mode or MODE_BUILD
        await self._send_session_update(session, update_current_mode(mode))
        # Mode is exposed as a config option (Zed 1.16 renders either the
        # config-option or session-mode surface, never both), so a server-side
        # switch — plan approval — must push refreshed options or the client's
        # mode select keeps showing the stale value.
        await self._send_session_update(
            session,
            ConfigOptionUpdate(
                session_update="config_option_update", config_options=list(self._factory.config_options_for(session))
            ),
        )

    async def _send_session_update(self, session: AcpSession, update: Any) -> None:
        conn = getattr(self, "_conn", None)
        if conn is None:
            return
        await conn.session_update(session_id=session.session_id, update=update, source="kolega_code")

    def _schedule_open_updates(self, session_id: str) -> None:
        """Send the open-time updates after the client has processed the
        ``session/new`` response (Zed drops pre-response notifications)."""
        asyncio.create_task(self._send_open_updates(session_id))

    async def _send_open_updates(self, session_id: str) -> None:
        await asyncio.sleep(OPEN_UPDATES_DELAY_SECONDS)
        session = self._sessions.get(session_id)
        if session is None:
            return
        await self._emit_initial_usage(session)
        await self._send_available_commands(session)
        self._start_session_tasks(session)

    async def _send_available_commands(self, session: AcpSession) -> None:
        """Advertise the agent's slash commands for the client's command UI.

        The agent's process loop dispatches exact-match commands itself (the
        same ``CommandProcessor`` decorator the CLI uses), so advertising is
        all the ACP server needs; the client sends ``/name args`` as a normal
        prompt and the agent turns it into the command result.
        """
        commands = [
            AvailableCommand(name=entry.name, description=entry.description) for entry in agent_command_entries()
        ]
        await self._send_session_update(session, update_available_commands(commands))

    async def _replay_transcript(self, session: AcpSession) -> None:
        """Replay the persisted transcript before responding to ``session/load``.

        The ACP spec requires the whole conversation to be re-sent as
        ``session/update`` notifications on load. The journal's presentation
        events fold through the same projection the TUI restores, so the editor
        sees the same transcript; live-only notices are excluded by the bridge.
        The plan view is then refreshed from the shared task list, mirroring the
        TUI's restored sidebar.
        """
        try:
            events = await self._factory.event_store(session.session_id).read(session.session_id)
        except Exception:  # noqa: BLE001 — a missing or unreadable journal must not block the load
            logger.exception("acp session/load: transcript replay failed (session=%s)", session.session_id)
            return
        state = replay(events)
        bridge = AcpBridge(self._conn)
        await bridge.replay_conversation(session.session_id, state.conversation)
        # Restore the compaction marker: the last finished summary renders as a
        # collapsible thought block, like the live event does.
        compaction = state.compaction or {}
        if str(compaction.get("phase") or "") == "finished":
            summary = str(compaction.get("summary") or "").strip()
            if summary:
                await self._send_session_update(
                    session, update_agent_thought_text(f"Conversation compacted:\n{summary}")
                )
        task_list = (session.record.task_list_markdown or "").strip()
        if task_list:
            entries = task_entries_from_markdown(task_list)
            if entries:
                await self._send_session_update(session, update_plan(entries))

    # -- turn machinery ----------------------------------------------------

    def _start_session_tasks(self, session: AcpSession) -> None:
        if session.closed:
            return
        if session.event_task is None:
            session.bridge = AcpBridge(
                self._conn,
                diff_provider=AcpDiffProvider.for_session(session),
                agent=session.agent,
                accounting=session.accounting,
            )
            session.event_task = asyncio.create_task(
                self._pump_events(session, session.bridge), name=f"acp-events-{session.session_id}"
            )
        # Explicit session activity rearms retained presentation work, including
        # after a transport failure. The worker must not retry itself on failure.
        if session.completions:
            session.background_wake.set()
        if session.background_task is not None and not session.background_task.done():
            return
        get_jobs = getattr(session.agent, "get_background_jobs", None)
        if callable(get_jobs):
            jobs: Any = get_jobs()

            def delivered(completion: dict[str, Any]) -> None:
                session.completions.append(completion)
                session.background_wake.set()

            jobs.set_callbacks(on_ready=session.background_wake.set, on_delivered=delivered)
            session.background_task = asyncio.create_task(
                self._background_worker(session, jobs), name=f"acp-background-{session.session_id}"
            )

    async def _background_worker(self, session: AcpSession, jobs: Any) -> None:
        await jobs.recover()
        if jobs.pending_count:
            session.background_wake.set()
        while not session.closed:
            await session.background_wake.wait()
            session.background_wake.clear()
            async with session.execution_lock:
                if session.closed:
                    return
                if session.cleanup_tasks:
                    continue
                try:
                    if session.approved_plan is not None:
                        with logical_prompt_scope(f"plan-{uuid4().hex}"):
                            await self._resume_approved_plan(session)
                        self._factory.persist(session)
                        if session.closed or session.cleanup_tasks:
                            continue
                        if jobs is not getattr(session.agent, "background_jobs", None):
                            return
                    await jobs.deliver(session.agent)
                    # Delivery callbacks also wake the worker for reports
                    # inserted by the running agent. Consume those wakeups now,
                    # not after sending: a failed send must wait for new activity,
                    # while readiness arriving during presentation must survive.
                    session.background_wake.clear()
                    if jobs.pending_count:
                        session.background_wake.set()
                    completions = bool(session.completions)
                    try:
                        while session.completions:
                            completion = session.completions[0]
                            await session.bridge.send(
                                session.session_id, update_agent_thought_text(str(completion["text"]))
                            )
                            # History insertion is already durable. Acknowledge
                            # presentation separately, only after this send
                            # succeeds; failure/cancellation retains the suffix.
                            session.completions.pop(0)
                    except Exception:
                        logger.exception("acp background presentation failed (session=%s)", session.session_id)
                        continue
                    if (
                        completions
                        and not session.auto_continue_paused
                        and getattr(session.agent, "background_needs_turn", True)
                    ):
                        with logical_prompt_scope(f"background-{uuid4().hex}"):
                            await self._run_turn(session, None)
                    self._factory.persist(session)
                    if jobs is not getattr(session.agent, "background_jobs", None):
                        return  # a context reset installed another epoch's owner
                    if not jobs.active_count:
                        while session.retired_agents and not session.cleanup_tasks:
                            await agent_factory._finish_cleanup(
                                AgentFactory.start_retired_cleanup(session, session.retired_agents[0]),
                                timeout=agent_factory.CLEANUP_TIMEOUT,
                            )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # A failed automatic turn must not spin or silently kill
                    # the prompt owner. Explicit input can resume it.
                    session.auto_continue_paused = True
                    logger.exception("acp background delivery failed (session=%s)", session.session_id)

    async def _run_turn(
        self,
        session: AcpSession,
        text: str | None,
        attachments: list[dict[str, Any]] | None = None,
    ) -> StopReason:
        """Drive one agent turn, mapping its output to ACP updates.

        The event pump belongs to the session, not this request. It keeps
        rendering child activity after a prompt's response has been returned.
        """
        if session.closed or session.cleanup_tasks:
            return "cancelled"
        self._start_session_tasks(session)
        bridge = session.bridge
        bridge.begin_turn(session.agent, AcpDiffProvider.for_session(session))
        session.turn_task = asyncio.create_task(self._drive_turn(session, bridge, text, attachments))
        reason: StopReason = "cancelled"
        try:
            reason = await session.turn_task
            return reason
        finally:
            session.turn_task = None
            await bridge.emit_usage(session.session_id)
            plan = self._consume_plan(session, bridge) if reason != "cancelled" and not session.closed else None
            if plan:
                session.record.latest_plan_markdown = plan
                session.record.plan_pending = True
                await self._request_plan_approval(session, plan)

    def _consume_plan(self, session: AcpSession, bridge: AcpBridge) -> str | None:
        """The plan from write_plan, else the planning turn's response text."""
        consume = getattr(session.agent, "consume_completed_plan", None)
        if callable(consume):
            plan = consume()
            if plan:
                return str(plan)
        if (session.record.interaction_mode or MODE_BUILD) == MODE_PLAN:
            text = bridge.response_text().strip()
            if text:
                return text
        return None

    async def _request_plan_approval(self, session: AcpSession, plan: str) -> None:
        """Plan approval mirroring the TUI's plan_actions: implement, implement
        with cleared context, or discuss. Approving switches the session to
        build mode and auto-starts the implementation turn."""
        if (
            session.closed
            or session.cleanup_tasks
            or session.auto_continue_paused
            or session.approved_plan is not None
            or session.plan_approval_task is not None
            or not session.record.plan_pending
        ):
            return
        generation = session.plan_approval_generation
        conn = getattr(self, "_conn", None)
        if conn is None:
            return
        tool_call = update_tool_call(
            f"plan-approval-{session.session_id[:8]}",
            title="Approve implementation plan?",
            kind="other",
            status="pending",
            content=[tool_content(text_block(plan))],
        )
        options = [
            PermissionOption(option_id="implement_plan", name="Implement plan", kind="allow_once"),
            PermissionOption(
                option_id="implement_plan_clear", name="Clear context and implement plan", kind="allow_once"
            ),
            PermissionOption(option_id="discuss_plan", name="Discuss further", kind="reject_once"),
        ]
        request = session.plan_approval_task = asyncio.create_task(
            conn.request_permission(session.session_id, tool_call, options)
        )
        try:
            response = await asyncio.wait_for(
                request,
                timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            if session.closed or generation != session.plan_approval_generation:
                return
            raise
        except Exception:  # noqa: BLE001 — approval prompts are best-effort
            logger.exception("acp plan approval prompt failed (session=%s)", session.session_id)
            return
        finally:
            if session.plan_approval_task is request:
                session.plan_approval_task = None
        outcome = getattr(response, "outcome", None)
        if isinstance(outcome, dict):
            selected = outcome.get("outcome") == "selected"
            option_id = str(outcome.get("optionId") or "")
        else:
            selected = getattr(outcome, "outcome", None) == "selected"
            option_id = str(getattr(outcome, "option_id", "") or "")
        if session.closed or session.cleanup_tasks or generation != session.plan_approval_generation:
            return
        if selected and option_id in ("implement_plan", "implement_plan_clear"):
            session.approved_plan = plan
            session.record.latest_plan_markdown = plan
            previous_agent = session.agent
            try:
                if option_id == "implement_plan_clear":
                    jobs = getattr(session.agent, "background_jobs", None)
                    if jobs is not None:
                        # The reset starts another epoch. Settle old-context work
                        # before clearing its history and creating a new owner.
                        jobs.set_callbacks(on_ready=None, on_delivered=None)
                        await jobs.close()
                        await jobs.deliver(session.agent)
                        session.agent.background_jobs = None
                        if (
                            session.background_task is not None
                            and session.background_task is not asyncio.current_task()
                        ):
                            session.background_task.cancel()
                            await asyncio.gather(session.background_task, return_exceptions=True)
                        session.background_task = None
                        session.completions.clear()
                        session.background_wake.clear()
                    if session.closed or generation != session.plan_approval_generation:
                        return
                    self._clear_agent_context(session)
                if (session.record.interaction_mode or MODE_BUILD) != MODE_BUILD:
                    await self._factory.set_interaction_mode(session, MODE_BUILD)
                if session.closed or generation != session.plan_approval_generation:
                    return
                await self._emit_current_mode(session)
                await self._resume_approved_plan(session)
            except asyncio.CancelledError:
                if generation == session.plan_approval_generation:
                    session.approved_plan = None
                    session.plan_approval_generation += 1
                raise
            except Exception:
                if session.agent is previous_agent:
                    session.approved_plan = None
                raise
            finally:
                if not session.closed and session.approved_plan is not None:
                    self._start_session_tasks(session)
                    session.background_wake.set()
            return
        if selected and option_id == "discuss_plan":
            session.record.plan_pending = False
            session.record.plan_reofferable = True
            logger.info("acp plan deferred for discussion (session=%s)", session.session_id)
            return
        # Unanswered, cancelled, or unknown: keep the plan pending.
        logger.info("acp plan approval pending (session=%s)", session.session_id)

    async def _resume_approved_plan(self, session: AcpSession) -> None:
        if (
            session.approved_plan is None
            or session.closed
            or session.cleanup_tasks
            or session.auto_continue_paused
            or (session.record.interaction_mode or MODE_BUILD) != MODE_BUILD
        ):
            return
        plan = session.approved_plan
        session.approved_plan = None
        session.record.plan_pending = False
        session.record.plan_reofferable = False
        await self._run_turn(session, build_implement_plan_prompt(plan))
        logger.info("acp plan approved and implementation started (session=%s)", session.session_id)

    def _clear_agent_context(self, session: AcpSession) -> None:
        if session.accounting is not None:
            session.accounting.invalidate_context()
        recorder = getattr(session.agent, "session_recorder", None)
        if recorder is not None and hasattr(recorder, "start_epoch"):
            recorder.start_epoch("acp_implement_plan_clear")
        session.agent.history = MessageHistory()
        session.agent.last_compression_index = None
        session.agent.reset_volatile_context()
        session.record.history = []
        session.record.compaction = {}

    async def _drive_turn(
        self,
        session: AcpSession,
        bridge: AcpBridge,
        text: str | None,
        attachments: list[dict[str, Any]] | None = None,
    ) -> StopReason:
        try:
            stream = (
                session.agent.process_message_stream(text, attachments)
                if text is not None
                else session.agent.continue_from_history_stream()
            )
            async for chunk in stream:
                await bridge.emit_chunk(session.session_id, chunk)
            return "end_turn"
        except asyncio.CancelledError:
            return "cancelled"

    async def _pump_events(self, session: AcpSession, bridge: AcpBridge) -> None:
        queue = session.manager.events
        while True:
            event = await queue.get()
            try:
                await bridge.handle_event(session.session_id, event)
            except Exception:  # noqa: BLE001 — a bad mapping must not kill the turn
                logger.exception("acp event mapping failed (session=%s)", session.session_id)

    async def _drain_residual(self, session: AcpSession, bridge: AcpBridge) -> None:
        while not session.manager.events.empty():
            try:
                event = session.manager.events.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                await bridge.handle_event(session.session_id, event)
            except Exception:  # noqa: BLE001
                logger.exception("acp residual event mapping failed (session=%s)", session.session_id)

    @staticmethod
    def _extract_prompt(prompt: list[_PromptBlock]) -> tuple[str, list[dict[str, Any]]]:
        parts: list[str] = []
        attachments: list[dict[str, Any]] = []
        for block in prompt:
            if isinstance(block, dict):
                block_type = block.get("type")
                text = block.get("text", "")
                data = block.get("data", "")
                media_type = block.get("mimeType") or block.get("mime_type")
            else:
                block_type = getattr(block, "type", None)
                text = getattr(block, "text", "")
                data = getattr(block, "data", "")
                media_type = getattr(block, "mime_type", None)
            if block_type == "text" and text:
                parts.append(str(text))
            elif block_type == "image" and isinstance(data, str) and isinstance(media_type, str):
                attachments.append({"type": "image", "media_type": media_type, "data": data})
        return "\n".join(parts), attachments
