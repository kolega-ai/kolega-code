"""Headless agent composition for ACP sessions.

Builds the same agent stack the CLI uses — settings, model config, durable
session recording — one ``CoderAgent`` per ACP session.

Sessions use ``AgentMode.CLI`` (the standard interactive coder template; the
editor thread has a human present) with an *explicit* permission mode:
``PermissionMode.AUTO`` in Phase 1, upgraded to ``ASK`` with the
``session/request_permission`` bridge in Phase 2. The ``ASK`` agent mode is
deliberately not used: it is the autonomous ``ask``-CLI template whose
permission handling is hardcoded to auto-approve.

Expected failures (missing model config, corrupt settings, unknown session)
are converted to ACP JSON-RPC errors so the client gets a clean message and
the process keeps serving.
"""

from __future__ import annotations

import logging
from pathlib import Path

from acp.exceptions import RequestError
from acp.schema import SessionInfo

from kolega_code.acp.session import AcpSession
from kolega_code.agent import CoderAgent
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.cli.config import build_agent_config, config_summary
from kolega_code.cli.connection import CliConnectionManager
from kolega_code.cli.session_event_store import FileArtifactStore, FileSessionEventStore
from kolega_code.cli.session_store import SessionStore, SessionStoreError
from kolega_code.cli.settings import SettingsStore
from kolega_code.permissions import PermissionMode
from kolega_code.session.recording import RecordingConnectionManager

logger = logging.getLogger(__name__)

ACP_AGENT_MODE = AgentMode.CLI
# Phase 1: auto-approve so turns complete without a permission bridge.
# Phase 2: PermissionMode.ASK + permission_callback -> session/request_permission.
ACP_PERMISSION_MODE = PermissionMode.AUTO


class AgentFactory:
    """Opens, loads, and persists ACP sessions backed by the session store."""

    def __init__(self) -> None:
        self._store = SessionStore()
        self._acp_sessions: dict[str, AcpSession] = {}

    async def open_session(self, cwd: str) -> AcpSession:
        project_path = Path(cwd).resolve()
        settings_store = SettingsStore()
        settings = settings_store.load()
        config = build_agent_config(project_path, settings=settings, settings_store=settings_store)
        record = self._store.create(project_path, ACP_AGENT_MODE.value, config_summary(config))
        return await self._build_session(record, config, restore=False)

    async def load_session(self, cwd: str, session_id: str) -> AcpSession | None:
        try:
            record = self._store.load(session_id)
        except SessionStoreError as exc:
            logger.info("acp session/load: unknown session %s (%s)", session_id, exc)
            return None
        settings_store = SettingsStore()
        settings = settings_store.load()
        config = build_agent_config(Path(record.project_path), settings=settings, settings_store=settings_store)
        return await self._build_session(record, config, restore=True)

    def persist(self, session: AcpSession) -> None:
        """Flush the agent's conversation state into the session record."""
        session.record.history = session.agent.dump_message_history()
        session.record.compaction = session.agent.dump_compaction_state()
        self._store.save(session.record)

    async def close_session(self, session_id: str) -> None:
        session = self._acp_sessions.pop(session_id, None)
        if session is None:
            return
        self.persist(session)
        try:
            await session.agent.cleanup()
        except Exception:  # noqa: BLE001 — closing must not mask the primary outcome
            logger.exception("acp session close: agent cleanup failed (session=%s)", session_id)

    def list_sessions(self, cwd: str | None) -> list[SessionInfo]:
        records = self._store.list(project_path=Path(cwd).resolve() if cwd else None)
        return [
            SessionInfo(
                session_id=record.session_id, cwd=record.project_path, title=record.title, updated_at=record.updated_at
            )
            for record in records
        ]

    async def _build_session(
        self,
        record,
        config,
        *,
        restore: bool,
    ) -> AcpSession:
        """Compose the durable agent stack for one session (the ``ask`` recipe + recording)."""
        journal = self._store.journal(record.session_id)
        recorder = self._store.recorder(record.session_id)
        manager = CliConnectionManager()
        recording = RecordingConnectionManager(
            manager,
            FileSessionEventStore(journal),
            session_id=record.session_id,
            artifact_store=FileArtifactStore(journal),
        )
        agent = CoderAgent(
            project_path=Path(record.project_path),
            workspace_id=record.workspace_id,
            thread_id=record.thread_id,
            connection_manager=recording,
            config=config,
            agent_mode=ACP_AGENT_MODE,
            permission_mode=ACP_PERMISSION_MODE,
            # Mirror the ask path: a resumed session hands the recorder over after
            # restoring history, so construction never double-records a resumed turn.
            session_recorder=None if restore else recorder,
        )
        lsp_messages = await agent.tool_collection.initialize()
        for message in lsp_messages:
            logger.debug("acp lsp: %s", message)
        if restore:
            agent.restore_message_history(record.history)
            agent.restore_compaction_state(record.compaction)
            agent.session_recorder = recorder
        session = AcpSession(
            session_id=record.session_id,
            record=record,
            agent=agent,
            manager=manager,
        )
        self._acp_sessions[record.session_id] = session
        return session

    # -- error conversion -------------------------------------------------

    @staticmethod
    def protocol_error(exc: BaseException) -> RequestError:
        return RequestError.internal_error({"message": str(exc) or exc.__class__.__name__})
