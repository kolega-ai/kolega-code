"""Headless agent composition for ACP sessions.

Builds the same agent stack the CLI uses — settings, model config, durable
session recording — one ``CoderAgent`` per ACP session.

Sessions use ``AgentMode.CLI`` (the standard interactive coder template; the
editor thread has a human present) with an *explicit* permission mode:
``PermissionMode.ASK`` with a client-bound ``permission_callback`` (the
``session/request_permission`` bridge). The ``ASK`` agent mode is
deliberately not used: it is the autonomous ``ask``-CLI template whose
permission handling is hardcoded to auto-approve.

Expected failures (missing model config, corrupt settings, unknown session)
are converted to ACP JSON-RPC errors so the client gets a clean message and
the process keeps serving.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Awaitable, Callable, cast

from acp.exceptions import RequestError
from acp.schema import (
    SessionConfigOptionBoolean,
    SessionConfigOptionSelect,
    SessionConfigSelectGroup,
    SessionConfigSelectOption,
    SessionInfo,
)

from kolega_code.acp.session import AcpSession
from kolega_code.agent import CoderAgent
from kolega_code.agent.planningagent import PlanningAgent
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.cli.config import CliConfigOverrides, build_agent_config, config_summary
from kolega_code.cli.connection import CliConnectionManager
from kolega_code.cli.provider_registry import PROVIDER_LABELS, ui_model_options
from kolega_code.cli.session_event_store import FileArtifactStore, FileSessionEventStore
from kolega_code.cli.session_store import SessionRecord, SessionStore, SessionStoreError
from kolega_code.cli.settings import CliSettings, SettingsStore
from kolega_code.permissions import PermissionDecision, PermissionMode, PermissionRequest, normalize_permission_mode
from kolega_code.session.recording import RecordingConnectionManager

logger = logging.getLogger(__name__)

ACP_AGENT_MODE = AgentMode.CLI
ACP_PERMISSION_MODE = PermissionMode.ASK
CONFIG_MODEL = "model"
CONFIG_PERMISSION_AUTO = "permission-auto"
MODE_BUILD = "build"
MODE_PLAN = "plan"
INTERACTION_MODES = {MODE_BUILD, MODE_PLAN}

PermissionCallback = Callable[[PermissionRequest], Awaitable[PermissionDecision]]
ConfigOption = SessionConfigOptionSelect | SessionConfigOptionBoolean


def agent_class_for(mode: str) -> type[CoderAgent]:
    return PlanningAgent if mode == MODE_PLAN else CoderAgent


class AgentFactory:
    """Opens, loads, and persists ACP sessions backed by the session store."""

    def __init__(self, permission_mode: PermissionMode = ACP_PERMISSION_MODE) -> None:
        self._permission_mode = permission_mode
        self._store = SessionStore()
        self._settings: CliSettings | None = None
        self._settings_store: SettingsStore | None = None
        self._acp_sessions: dict[str, AcpSession] = {}

    async def open_session(
        self,
        cwd: str,
        *,
        session_id: str | None = None,
        permission_callback: PermissionCallback | None = None,
    ) -> AcpSession:
        project_path = Path(cwd).resolve()
        self._load_settings()
        config = self._config_for(project_path)
        record = self._store.create(project_path, ACP_AGENT_MODE.value, config_summary(config), session_id=session_id)
        return await self._build_session(record, config, restore=False, permission_callback=permission_callback)

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        *,
        permission_callback: PermissionCallback | None = None,
    ) -> AcpSession | None:
        try:
            record = self._store.load(session_id)
        except SessionStoreError as exc:
            logger.info("acp session/load: unknown session %s (%s)", session_id, exc)
            return None
        self._load_settings()
        config = self._config_for(Path(record.project_path))
        return await self._build_session(record, config, restore=True, permission_callback=permission_callback)

    # -- session config options -------------------------------------------

    def config_options_for(self, session: AcpSession) -> list[ConfigOption]:
        return [self._model_option(session), self._permission_option(session)]

    async def set_permission_mode(self, session: AcpSession, mode: PermissionMode) -> None:
        session.agent.set_permission_mode(mode)
        session.record.permission_mode = mode.value

    async def set_interaction_mode(self, session: AcpSession, mode: str) -> None:
        """Switch plan/build mode by rebuilding the session agent with the mode's class."""
        session.record.interaction_mode = mode
        await self._rebuild_agent(session, session.config)

    async def apply_model(self, session: AcpSession, provider: str, model: str) -> None:
        config = self._config_for(Path(session.record.project_path), provider=provider, model=model)
        session.config = config
        await self._rebuild_agent(session, config)

    # -- persistence --------------------------------------------------------

    def persist(self, session: AcpSession) -> None:
        """Flush the agent's conversation state into the session record."""
        session.record.history = session.agent.dump_message_history()
        session.record.compaction = session.agent.dump_compaction_state()
        session.record.permission_mode = session.agent.permission_mode.value
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

    # -- internal -----------------------------------------------------------

    def _load_settings(self) -> None:
        settings_store = SettingsStore()
        self._settings_store = settings_store
        self._settings = settings_store.load()

    def _config_for(self, project_path: Path, *, provider: str | None = None, model: str | None = None) -> Any:
        assert self._settings is not None and self._settings_store is not None
        overrides = CliConfigOverrides(provider=provider, model=model) if provider and model else None
        return build_agent_config(
            project_path,
            overrides=overrides,
            settings=self._settings,
            settings_store=self._settings_store,
        )

    def _model_option(self, session: AcpSession) -> SessionConfigOptionSelect:
        assert self._settings is not None
        groups: list[SessionConfigSelectGroup] = []
        for provider, key in sorted(self._settings.api_keys.items()):
            if not key:
                continue
            options = [
                SessionConfigSelectOption(value=f"{provider}/{model}", name=label)
                for label, model in ui_model_options(provider)
            ]
            if options:
                groups.append(
                    SessionConfigSelectGroup(
                        group=provider,
                        name=str(PROVIDER_LABELS.get(cast(Any, provider), provider)),
                        options=options,
                    ),
                )
        primary = session.agent.primary_model_config
        current = f"{primary.provider.value}/{primary.model}"
        return SessionConfigOptionSelect(
            type="select",
            id=CONFIG_MODEL,
            name="Model",
            current_value=current,
            options=groups,
        )

    @staticmethod
    def _permission_option(session: AcpSession) -> SessionConfigOptionBoolean:
        return SessionConfigOptionBoolean(
            type="boolean",
            id=CONFIG_PERMISSION_AUTO,
            name="Auto-approve tools",
            current_value=session.agent.permission_mode == PermissionMode.AUTO,
        )

    async def _rebuild_agent(self, session: AcpSession, config: Any) -> None:
        """Replace the session's agent with a rebuild carrying the same history.

        Config options apply between turns, so the persisted record is the
        freshest history state; ``restore=True`` loads it into the rebuild.
        """
        old = session.agent
        agent, manager = await self._construct_agent(
            session.record,
            config,
            restore=True,
            permission_callback=session.permission_callback,
            permission_mode=old.permission_mode,
        )
        session.agent = agent
        session.manager = manager
        try:
            await old.cleanup()
        except Exception:  # noqa: BLE001
            logger.exception("acp model switch: old agent cleanup failed (session=%s)", session.session_id)

    async def _build_session(
        self,
        record: SessionRecord,
        config: Any,
        *,
        restore: bool,
        permission_callback: PermissionCallback | None,
    ) -> AcpSession:
        """Compose the durable agent stack for one session (the ``ask`` recipe + recording)."""
        permission_mode = (
            normalize_permission_mode(record.permission_mode, default=self._permission_mode)
            if restore
            else self._permission_mode
        )
        agent, manager = await self._construct_agent(
            record,
            config,
            restore=restore,
            permission_callback=permission_callback,
            permission_mode=permission_mode,
        )
        session = AcpSession(
            session_id=record.session_id,
            record=record,
            agent=agent,
            manager=manager,
            permission_callback=permission_callback,
            config=config,
        )
        self._acp_sessions[record.session_id] = session
        return session

    async def _construct_agent(
        self,
        record: SessionRecord,
        config: Any,
        *,
        restore: bool,
        permission_callback: PermissionCallback | None,
        permission_mode: PermissionMode | None = None,
    ) -> tuple[CoderAgent, CliConnectionManager]:
        journal = self._store.journal(record.session_id)
        recorder = self._store.recorder(record.session_id)
        manager = CliConnectionManager()
        recording = RecordingConnectionManager(
            manager,
            FileSessionEventStore(journal),
            session_id=record.session_id,
            artifact_store=FileArtifactStore(journal),
        )
        interaction_mode = record.interaction_mode or MODE_BUILD
        agent_class = agent_class_for(interaction_mode)
        agent = agent_class(
            project_path=Path(record.project_path),
            workspace_id=record.workspace_id,
            thread_id=record.thread_id,
            connection_manager=recording,
            config=config,
            agent_mode=ACP_AGENT_MODE,
            permission_mode=permission_mode or self._permission_mode,
            permission_callback=permission_callback,
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
        return agent, manager

    # -- error conversion -------------------------------------------------

    @staticmethod
    def protocol_error(exc: BaseException) -> RequestError:
        return RequestError.internal_error({"message": str(exc) or exc.__class__.__name__})
