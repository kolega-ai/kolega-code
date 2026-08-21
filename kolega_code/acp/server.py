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
from acp.exceptions import RequestError
from acp.interfaces import Client
from acp.schema import (
    AudioContentBlock,
    ClientCapabilities,
    EmbeddedResourceContentBlock,
    HttpMcpServer,
    ImageContentBlock,
    Implementation,
    ListSessionsResponse,
    LoadSessionResponse,
    McpServerStdio,
    ResourceContentBlock,
    SseMcpServer,
    StopReason,
    TextContentBlock,
)

from kolega_code.acp.agent_factory import AgentFactory
from kolega_code.acp.bridge import AcpBridge
from kolega_code.acp.permissions import AcpPermissionBroker
from kolega_code.acp.session import AcpSession
from kolega_code.cli.config import CliConfigError
from kolega_code.cli.session_store import SessionStoreError
from kolega_code.permissions import PermissionDecision, PermissionRequest

logger = logging.getLogger(__name__)

_PromptBlock = (
    TextContentBlock | ImageContentBlock | AudioContentBlock | ResourceContentBlock | EmbeddedResourceContentBlock
)

_McpServer = list[HttpMcpServer | SseMcpServer | McpServerStdio] | None

_HANDLED_CONFIG_ERRORS = (CliConfigError, SessionStoreError)

PermissionCallback = Callable[[PermissionRequest], Awaitable[PermissionDecision]]


class AcpAgent(Agent):
    """kolega-code as an ACP v1 agent, one client connection per process.

    The editor spawns this process, negotiates via ``initialize``, creates
    sessions, and drives turns with ``session/prompt``. One ``CoderAgent``
    per session; sessions persist as regular ``SessionRecord``s.
    """

    def __init__(self, factory: AgentFactory | None = None) -> None:
        self._factory = factory or AgentFactory()
        self._sessions: dict[str, AcpSession] = {}
        self._brokers: dict[str, AcpPermissionBroker] = {}
        self._conn: Client

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
        return InitializeResponse(protocol_version=protocol_version)

    # -- session lifecycle ------------------------------------------------

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: _McpServer = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
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
        return NewSessionResponse(session_id=session.session_id)

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: _McpServer = None,
        additional_directories: list[str] | None = None,
        **kwargs: Any,
    ) -> LoadSessionResponse | None:
        if session_id in self._sessions:
            return LoadSessionResponse()
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
        return LoadSessionResponse()

    async def list_sessions(
        self, cwd: str | None = None, cursor: str | None = None, **kwargs: Any
    ) -> ListSessionsResponse:
        logger.info("acp session/list (cwd=%s)", cwd)
        return ListSessionsResponse(sessions=self._factory.list_sessions(cwd))

    async def close_session(self, session_id: str, **kwargs: Any) -> None:
        logger.info("acp session/close (session=%s)", session_id)
        self._sessions.pop(session_id, None)
        self._brokers.pop(session_id, None)
        await self._factory.close_session(session_id)
        return None

    # -- turns ------------------------------------------------------------

    async def prompt(self, session_id: str, prompt: list[_PromptBlock], **kwargs: Any) -> PromptResponse:
        session = self._sessions.get(session_id)
        if session is None:
            raise RequestError.resource_not_found(session_id)
        if not session.idle():
            raise RequestError.invalid_request({"message": "a turn is already running for this session"})
        text = self._extract_text(prompt)
        if not text:
            raise RequestError.invalid_request({"message": "prompt contains no text"})
        stop_reason = await self._run_turn(session, text)
        self._factory.persist(session)
        return PromptResponse(stop_reason=stop_reason)

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        session = self._sessions.get(session_id)
        if session is None or session.turn_task is None:
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

    async def set_session_mode(self, session_id: str, mode_id: str, **kwargs: Any) -> None:
        logger.info("acp session/set_mode not supported yet (mode=%s)", mode_id)
        return None

    async def set_config_option(self, config_id: str, session_id: str, value: str | bool, **kwargs: Any) -> None:
        logger.info("acp session/set_config_option not supported yet (config=%s)", config_id)
        return None

    async def authenticate(self, method_id: str, **kwargs: Any) -> None:
        logger.info("acp authenticate not supported yet (method=%s)", method_id)
        return None

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        raise RequestError.method_not_found(f"_{method}")

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        logger.debug("acp ignoring extension notification %s", method)

    # -- permissions --------------------------------------------------------

    def _permission_callback_for(self, session_id: str) -> PermissionCallback:
        """One per-session agent callback that answers through the ACP client."""

        async def callback(request: PermissionRequest) -> PermissionDecision:
            session = self._sessions.get(session_id)
            if session is None:
                return PermissionDecision(allowed=False, reason="Session is no longer active.")
            broker = self._brokers.get(session_id)
            if broker is None:
                broker = AcpPermissionBroker(
                    self._conn,
                    session_id,
                    Path(session.record.project_path),
                    lambda: session.agent,
                )
                self._brokers[session_id] = broker
            return await broker(request)

        return callback

    # -- turn machinery ----------------------------------------------------

    async def _run_turn(self, session: AcpSession, text: str) -> StopReason:
        """Drive one agent turn, mapping its output to ACP updates.

        Returns the ACP stop reason. The event pump runs until the turn task
        completes, then any residual queued events are drained so late tool
        results still render.
        """
        session.drain_events()
        bridge = AcpBridge(self._conn)
        session.turn_task = asyncio.create_task(self._drive_turn(session, bridge, text))
        pump = asyncio.create_task(self._pump_events(session, bridge))
        try:
            return await session.turn_task
        finally:
            pump.cancel()
            try:
                await pump
            except asyncio.CancelledError:
                pass
            await self._drain_residual(session, bridge)
            session.turn_task = None

    async def _drive_turn(self, session: AcpSession, bridge: AcpBridge, text: str) -> str:
        try:
            async for chunk in session.agent.process_message_stream(text):
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
    def _extract_text(prompt: list[_PromptBlock]) -> str:
        parts: list[str] = []
        for block in prompt:
            text = block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
            if text:
                parts.append(str(text))
        return "\n".join(parts)
