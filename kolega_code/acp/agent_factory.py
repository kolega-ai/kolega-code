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

import asyncio
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Awaitable, Callable, cast
from uuid import uuid4

from acp.exceptions import RequestError
from acp.schema import (
    SessionConfigOptionBoolean,
    SessionConfigOptionSelect,
    SessionConfigSelectGroup,
    SessionConfigSelectOption,
    SessionInfo,
)

from kolega_code.acp.questions import build_question_extension
from kolega_code.acp.accounting import AcpUsageAccounting
from kolega_code.acp.mcp import (
    ClientMCPServer,
    build_session_mcp_extension,
    merge_mcp_config,
    normalize_client_servers,
)
from kolega_code.acp.session import AcpSession
from kolega_code.agent import CoderAgent, PromptExtension, ToolExtension
from kolega_code.agent.planningagent import PlanningAgent
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.agent.prompts import BUILD_QUESTION_PROMPT, PLANNING_QUESTION_PROMPT
from kolega_code.agent.tool_definitions import tool_description_asset
from kolega_code.agent.volatile_context import VolatileSection
from kolega_code.cli.config import API_KEY_ENV, CliConfigOverrides, build_agent_config, config_summary
from kolega_code.cli.connection import CliConnectionManager
from kolega_code.cli.provider_registry import PROVIDER_LABELS, ui_model_options
from kolega_code.cli.session_event_store import FileArtifactStore, FileSessionEventStore
from kolega_code.cli.session_store import SessionRecord, SessionStore, SessionStoreError
from kolega_code.cli.settings import CliSettings, SettingsStore
from kolega_code.cli.session_usage import SessionUsageSink
from kolega_code.events import AgentEvent
from kolega_code.hooks import HookDispatcher, HookEvent, load_hook_config
from kolega_code.llm.ledger import UsageLedger
from kolega_code.llm.models import Message, TextBlock
from kolega_code.llm.pricing import PricingResolver
from kolega_code.mcp.config import MCPServerConfig, load_mcp_config
from kolega_code.permissions import PermissionDecision, PermissionMode, PermissionRequest, normalize_permission_mode
from kolega_code.session.recording import RecordingConnectionManager
from kolega_code.session.composition import (
    CompositionHost,
    CompositionRequest,
    HostExtensions,
    construct_agent,
    prepare_composition,
    select_agent_class,
)

logger = logging.getLogger(__name__)

ACP_AGENT_MODE = AgentMode.CLI
ACP_PERMISSION_MODE = PermissionMode.ASK
CONFIG_MODEL = "model"
CONFIG_PERMISSION_AUTO = "permission-auto"
CONFIG_MODE = "mode"
MODE_BUILD = "build"
MODE_PLAN = "plan"
INTERACTION_MODES = {MODE_BUILD, MODE_PLAN}
CLEANUP_TIMEOUT = 5.0

PermissionCallback = Callable[[PermissionRequest], Awaitable[PermissionDecision]]
ConfigOption = SessionConfigOptionSelect | SessionConfigOptionBoolean


async def _finish_cleanup(cleanup: Awaitable[None], *, timeout: float | None = None) -> None:
    task = asyncio.ensure_future(cleanup)
    cancelled: asyncio.CancelledError | None = None
    deadline = asyncio.get_running_loop().time() + timeout if timeout is not None else None
    while True:
        try:
            if deadline is None:
                await asyncio.shield(task)
            else:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                done, _ = await asyncio.wait({task}, timeout=remaining)
                if not done:
                    break
                task.result()
            break
        except asyncio.CancelledError as exc:
            if task.cancelled():
                raise
            if cancelled is None:
                cancelled = exc
        except Exception:  # noqa: BLE001
            if cancelled is None:
                raise
            logger.exception("acp cleanup failed while cancellation was pending")
            break
    if cancelled is not None:
        raise cancelled


def agent_class_for(mode: str) -> type[CoderAgent]:
    return cast(
        type[CoderAgent], select_agent_class(mode, build_agent_class=CoderAgent, planning_agent_class=PlanningAgent)
    )


class _AcpRecordingConnectionManager(RecordingConnectionManager):
    generation: str

    async def broadcast_event(self, event: AgentEvent, workspace_id: str, thread_id: str) -> None:
        if event.event_type == "llm_context_update":
            event = event.model_copy(update={"content": {**event.content, "acp_generation": self.generation}})
        await super().broadcast_event(event, workspace_id, thread_id)


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
        mcp_servers: Sequence[ClientMCPServer | dict[str, Any]] | None = None,
    ) -> AcpSession:
        if session_id is not None and (existing := self._acp_sessions.get(session_id)) is not None:
            self.check_active(existing)
        project_path = Path(cwd).resolve()
        self._load_settings()
        config = self._config_for(project_path)
        clients = normalize_client_servers(mcp_servers, project_path)
        config.mcp_config = merge_mcp_config(config.mcp_config, clients)
        record = self._store.create(project_path, ACP_AGENT_MODE.value, config_summary(config), session_id=session_id)
        return await self._build_session(
            record, config, restore=False, permission_callback=permission_callback, client_mcp_servers=clients
        )

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        *,
        permission_callback: PermissionCallback | None = None,
        mcp_servers: Sequence[ClientMCPServer | dict[str, Any]] | None = None,
    ) -> AcpSession | None:
        existing = self._acp_sessions.get(session_id)
        if existing is not None:
            self.check_active(existing)
            if mcp_servers is not None:
                await self.update_client_servers(existing, mcp_servers)
            return existing
        try:
            record = self._store.load(session_id)
        except SessionStoreError as exc:
            logger.info("acp session/load: unknown session %s (%s)", session_id, exc)
            return None
        self._load_settings()
        config = self._config_for(Path(record.project_path))
        clients = normalize_client_servers(mcp_servers, Path(record.project_path))
        config.mcp_config = merge_mcp_config(config.mcp_config, clients)
        return await self._build_session(
            record, config, restore=True, permission_callback=permission_callback, client_mcp_servers=clients
        )

    async def update_client_servers(
        self, session: AcpSession, servers: Sequence[ClientMCPServer | dict[str, Any]]
    ) -> None:
        self.check_active(session)
        clients = normalize_client_servers(servers, Path(session.record.project_path))
        if session.execution_lock.locked():
            raise RequestError.invalid_request({"message": "MCP configuration applies between turns only."})
        async with session.execution_lock:
            self.check_active(session)
            for name, client in clients.items():
                previous = session.client_mcp_servers.get(name)
                if previous is not None and previous.model_dump(exclude={"id"}) == client.model_dump(exclude={"id"}):
                    clients[name] = previous
            if clients == session.client_mcp_servers:
                return
            if session.approved_plan is not None or (session.turn_task is not None and not session.turn_task.done()):
                raise RequestError.invalid_request({"message": "MCP configuration applies between turns only."})
            assert self._settings is not None and self._settings_store is not None
            configured = load_mcp_config(
                Path(session.record.project_path),
                self._settings_store.root,
                project_trusted=self._settings.is_mcp_project_trusted(Path(session.record.project_path)),
            )
            config = session.config.model_copy(update={"mcp_config": merge_mcp_config(configured, clients)})
            await self._rebuild_agent(session, config, client_mcp_servers=clients)

    # -- session config options -------------------------------------------

    def config_options_for(self, session: AcpSession) -> list[ConfigOption]:
        return [self._model_option(session), self._permission_option(session), self._mode_option(session)]

    async def set_permission_mode(self, session: AcpSession, mode: PermissionMode) -> None:
        self.check_active(session)
        if session.runtime is not None:
            session.runtime.set_permission_mode(mode)
        else:
            session.agent.set_permission_mode(mode)
        session.record.permission_mode = mode.value

    async def set_interaction_mode(self, session: AcpSession, mode: str) -> None:
        """Switch plan/build mode by rebuilding the session agent with the mode's class."""
        self.check_active(session)
        previous = session.record.interaction_mode
        previous_agent = session.agent
        session.record.interaction_mode = mode
        try:
            await self._rebuild_agent(session, session.config)
        except BaseException:
            if session.agent is previous_agent:
                session.record.interaction_mode = previous
            self.persist(session)
            raise

    async def apply_model(self, session: AcpSession, provider: str, model: str) -> None:
        config = self._config_for(Path(session.record.project_path), provider=provider, model=model)
        config.mcp_config = merge_mcp_config(config.mcp_config, session.client_mcp_servers)
        await self._rebuild_agent(session, config)

    # -- persistence --------------------------------------------------------

    def persist(self, session: AcpSession) -> None:
        """Flush the agent's conversation state into the session record."""
        session.record.history = session.agent.dump_message_history()
        session.record.compaction = session.agent.dump_compaction_state()
        session.record.permission_mode = session.agent.permission_mode.value
        self._store.save(session.record)

    async def close_session(self, session_id: str) -> None:
        existing = self._acp_sessions.get(session_id)
        if existing is not None:
            self.check_cleanup(existing)
        session = self._acp_sessions.get(session_id)
        if session is None:
            return
        session.closed = True
        try:
            self.persist(session)
            jobs = getattr(session.agent, "background_jobs", None)
            if jobs is not None:
                jobs.set_callbacks(on_ready=None, on_delivered=None)
                await jobs.close()
                await jobs.deliver(session.agent)
            self.persist(session)
            await session.agent.fire_hook(HookEvent.SESSION_END, {"reason": "session_close"})
        except Exception:  # noqa: BLE001 — closing must not mask the primary outcome
            logger.exception("acp session close: agent cleanup failed (session=%s)", session_id)
        finally:
            while session.retired_agents:
                await _finish_cleanup(self.start_retired_cleanup(session, session.retired_agents[0]))
            self.check_cleanup(session)
            try:
                await session.agent.cleanup()
            except Exception:  # noqa: BLE001
                logger.exception("acp agent cleanup failed (session=%s)", session_id)
            finally:
                if session.accounting is not None:
                    await session.accounting.aclose()
                session.client_mcp_servers.clear()
                session.config = None
                self._acp_sessions.pop(session_id, None)

    def list_sessions(self, cwd: str | None) -> list[SessionInfo]:
        records = self._store.list(project_path=Path(cwd).resolve() if cwd else None, recover=False)
        return [
            SessionInfo(
                session_id=record.session_id, cwd=record.project_path, title=record.title, updated_at=record.updated_at
            )
            for record in records
        ]

    def event_store(self, session_id: str) -> FileSessionEventStore:
        """The session's durable event journal as a readable event store."""
        return FileSessionEventStore(self._store.journal(session_id))

    # -- internal -----------------------------------------------------------

    @staticmethod
    def check_cleanup(session: AcpSession) -> None:
        if session.cleanup_tasks:
            raise RequestError.invalid_request({"message": "Agent cleanup is incomplete; retry after it finishes."})

    @staticmethod
    def check_active(session: AcpSession) -> None:
        if session.closed:
            raise RequestError.invalid_request({"message": "Session is closing."})
        AgentFactory.check_cleanup(session)

    @staticmethod
    def start_retired_cleanup(session: AcpSession, agent: Any) -> asyncio.Task[None]:
        existing = session.cleanup_tasks.get(id(agent))
        if existing is not None:
            return existing
        if not any(old is agent for old in session.retired_agents):
            session.retired_agents.append(agent)
        task = asyncio.create_task(agent.cleanup())
        session.cleanup_tasks[id(agent)] = task

        def completed(done: asyncio.Task[None]) -> None:
            if done.cancelled() or done.exception() is not None:
                # Failure does not prove resources are released; keep the fence rather than retrying cleanup.
                logger.error("acp old agent cleanup incomplete (session=%s)", session.session_id)
                return
            if session.cleanup_tasks.get(id(agent)) is done:
                session.cleanup_tasks.pop(id(agent))
                session.retired_agents[:] = [old for old in session.retired_agents if old is not agent]
                session.background_wake.set()

        task.add_done_callback(completed)
        return task

    @staticmethod
    def _task_list_extension(record: SessionRecord, agent_holder: dict[str, Any], mode: str) -> ToolExtension:
        """Shared task-list tools backed by the session record — the same list
        the TUI's ``update_task_list`` writes, so a session moves between the
        TUI and the editor with one task list.

        Build mode can read and update the list; plan mode gets a read-only
        view (same split as the TUI). Updates broadcast a ``task_list_update``
        chat event so the bridge can refresh the editor's plan view live.
        """

        async def get_task_list() -> str:
            return record.task_list_markdown or "No task list has been set."

        async def update_task_list(task_list_markdown: str) -> str:
            record.task_list_markdown = task_list_markdown.strip()
            agent = agent_holder.get("agent")
            if agent is not None:
                await agent.send_chat_message(
                    message_type="task_list_update",
                    content=task_list_markdown.strip(),
                    tool_call_id=str(getattr(agent, "current_tool_call_id", "") or ""),
                )
            return "Task list updated."

        tools: dict[str, Any] = {"get_task_list": get_task_list}
        if mode == MODE_BUILD:
            tools["update_task_list"] = update_task_list
        return ToolExtension(
            name="acp-shared-task-list",
            tools=tools,
            tool_descriptions={
                "get_task_list": tool_description_asset(
                    "get_task_list" if mode == MODE_BUILD else "get_task_list_readonly"
                ),
                **({"update_task_list": tool_description_asset("update_task_list")} if mode == MODE_BUILD else {}),
            },
            tool_schemas={
                "get_task_list": {"type": "object", "properties": {}, "required": []},
                "update_task_list": {
                    "type": "object",
                    "properties": {
                        "task_list_markdown": {
                            "type": "string",
                            "description": "The full current shared task list as Markdown.",
                        }
                    },
                    "required": ["task_list_markdown"],
                },
            },
            tool_groups={"planning_tools": list(tools)},
            propagate_to_sub_agents=False,
        )

    @staticmethod
    def _task_list_volatile_section(record: SessionRecord) -> Callable[[], VolatileSection]:
        """The current shared task list as a volatile section (TUI parity: the
        agent sees the latest list in context every turn without re-fetching).
        An empty list is an absent section, exactly like the TUI."""

        def provider() -> VolatileSection:
            return VolatileSection("task_list", (record.task_list_markdown or "").strip())

        return provider

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

    def _configured_providers(self) -> list[str]:
        """Providers the editor can offer: saved keys plus exported credentials.

        ACP spawns as a child of the editor, which inherits the shell
        environment. The CLI already treats an exported ``*_API_KEY`` as
        sufficient, so an env-only provider must not be invisible here just
        because it was never saved through onboarding.
        """
        assert self._settings is not None
        providers = {provider for provider, key in self._settings.api_keys.items() if key}
        providers.update(
            provider.value for provider, env_name in API_KEY_ENV.items() if (os.environ.get(env_name) or "").strip()
        )
        return sorted(providers)

    def _model_option(self, session: AcpSession) -> SessionConfigOptionSelect:
        groups: list[SessionConfigSelectGroup] = []
        for provider in self._configured_providers():
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
    def _permission_option(session: AcpSession) -> SessionConfigOptionSelect:
        # A select rather than the ACP boolean: vscode-acp renders only
        # select-type config options (its UI skips every other type), and Zed
        # renders selects just as well.
        return SessionConfigOptionSelect(
            type="select",
            id=CONFIG_PERMISSION_AUTO,
            name="Permissions",
            current_value="auto" if session.agent.permission_mode == PermissionMode.AUTO else "ask",
            options=[
                SessionConfigSelectOption(value="ask", name="Ask"),
                SessionConfigSelectOption(value="auto", name="Auto-approve"),
            ],
        )

    @staticmethod
    def _mode_option(session: AcpSession) -> SessionConfigOptionSelect:
        return SessionConfigOptionSelect(
            type="select",
            id=CONFIG_MODE,
            name="Mode",
            current_value=session.record.interaction_mode or MODE_BUILD,
            options=[
                SessionConfigSelectOption(value=MODE_BUILD, name="Build"),
                SessionConfigSelectOption(value=MODE_PLAN, name="Plan"),
            ],
        )

    async def _rebuild_agent(
        self, session: AcpSession, config: Any, *, client_mcp_servers: dict[str, MCPServerConfig] | None = None
    ) -> None:
        """Replace the session's agent with a rebuild carrying the same history.

        Config options apply between turns, so the persisted record is the
        freshest history state; ``restore=True`` loads it into the rebuild.
        """
        self.check_active(session)
        old = session.agent
        # The approval flow rebuilds mid-turn, before prompt() persists the
        # just-finished plan turn. Dump the live agent into the record first
        # (as the TUI does before rebuilds) so the rebuild restores the full
        # conversation, not the stale pre-turn snapshot.
        self.persist(session)
        if session.accounting is not None:
            session.accounting.invalidate_context()
        agent, manager = await self._construct_agent(
            session.record,
            config,
            restore=True,
            permission_callback=session.permission_callback,
            permission_mode=old.permission_mode,
            question_state=session.question_state,
            accounting=session.accounting,
        )
        # New generations must keep publishing into the session-lifetime event
        # consumer, including children still using the previous generation.
        if manager is not session.manager:
            raise RuntimeError("ACP rebuild must preserve the session connection manager")
        session.agent = agent
        session.config = config
        if client_mcp_servers is not None:
            session.client_mcp_servers = client_mcp_servers
        if session.accounting is not None:
            session.accounting.rebind(agent)
        if session.runtime is not None:
            session.runtime.adopt(agent)
        jobs = getattr(old, "background_jobs", None)
        if jobs is not None and agent is not old:
            agent.background_jobs = jobs
            agent.background_needs_turn = getattr(old, "background_needs_turn", False)
            old.background_jobs = None
        if jobs is not None and jobs.active_count and agent is not old:
            # Children borrow this generation's terminal/browser/extensions.
            # Keep it alive until the supervisor has joined them.
            session.retired_agents.append(old)
            return
        if agent is old:
            return
        task = self.start_retired_cleanup(session, old)
        try:
            await _finish_cleanup(task, timeout=CLEANUP_TIMEOUT)
        except Exception:  # noqa: BLE001
            logger.exception("acp model switch: old agent cleanup failed (session=%s)", session.session_id)
        except asyncio.CancelledError:
            raise

    async def _build_session(
        self,
        record: SessionRecord,
        config: Any,
        *,
        restore: bool,
        permission_callback: PermissionCallback | None,
        client_mcp_servers: dict[str, MCPServerConfig] | None = None,
    ) -> AcpSession:
        permission_mode = (
            normalize_permission_mode(record.permission_mode, default=self._permission_mode)
            if restore
            else self._permission_mode
        )
        question_state: dict[str, Any] = {}
        accounting = AcpUsageAccounting(
            record.session_id,
            baseline=record.usage,
            ledger=UsageLedger(pricing_resolver=PricingResolver(getattr(config, "usage_pricing", None))),
        )
        accounting.sink = SessionUsageSink(
            self._store.journal(record.session_id),
            self._store.recorder(record.session_id),
            accounting.ledger,
            mode="acp",
        )
        try:
            await accounting.sink.start()
            accounting.ledger.observer = accounting.sink
            agent, manager = await self._construct_agent(
                record,
                config,
                restore=restore,
                permission_callback=permission_callback,
                permission_mode=permission_mode,
                question_state=question_state,
                accounting=accounting,
            )
        except BaseException:
            await accounting.aclose()
            raise
        accounting.rebind(agent)
        session = AcpSession(
            session_id=record.session_id,
            record=record,
            agent=agent,
            manager=manager,
            permission_callback=permission_callback,
            config=config,
            question_state=question_state,
            accounting=accounting,
            client_mcp_servers=client_mcp_servers or {},
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
        question_state: dict[str, Any] | None = None,
        accounting: AcpUsageAccounting | None = None,
    ) -> tuple[CoderAgent, CliConnectionManager]:
        journal = self._store.journal(record.session_id)
        recorder = self._store.recorder(record.session_id)
        existing = self._acp_sessions.get(record.session_id)
        manager = existing.manager if existing is not None else CliConnectionManager()
        recording = _AcpRecordingConnectionManager(
            manager,
            FileSessionEventStore(journal),
            session_id=record.session_id,
            artifact_store=FileArtifactStore(journal),
        )
        recording.generation = uuid4().hex
        interaction_mode = record.interaction_mode or MODE_BUILD
        agent_class = agent_class_for(interaction_mode)
        agent_holder: dict[str, Any] = {}
        question_markdown = PLANNING_QUESTION_PROMPT if interaction_mode == MODE_PLAN else BUILD_QUESTION_PROMPT
        question_prompt = PromptExtension(
            id="acp-questions",
            title="Asking the User",
            markdown=question_markdown,
            modes=[ACP_AGENT_MODE],
            propagate_to_sub_agents=False,
        )
        assert self._settings is not None and self._settings_store is not None
        project_path = Path(record.project_path)
        hook_config = load_hook_config(
            project_path,
            self._settings_store.root,
            project_trusted=self._settings.is_hook_project_trusted(project_path),
        )
        for _diagnostic in hook_config.diagnostics:
            logger.warning("ACP hook configuration warning; check saved hooks and project trust.")
        hooks = HookDispatcher(hook_config)
        mcp_extension = await build_session_mcp_extension(config.mcp_config, project_path, self._settings_store.root)
        request = CompositionRequest(
            config=config,
            project_path=project_path,
            active_project_path=project_path,
            state_dir=self._settings_store.root,
            interaction_mode=interaction_mode,
            agent_class=agent_class,
            workspace_id=record.workspace_id,
            thread_id=record.thread_id,
            connection_manager=recording,
            agent_mode=ACP_AGENT_MODE,
            permission_mode=permission_mode or self._permission_mode,
            permission_callback=permission_callback,
            # Mirror the ask path: a resumed session hands the recorder over after
            # restoring history, so construction never double-records a resumed turn.
            session_recorder=None if restore else recorder,
            usage_ledger=accounting.ledger if accounting is not None else None,
        )
        host = CompositionHost(
            extensions=lambda: HostExtensions([], [self._task_list_extension(record, agent_holder, interaction_mode)]),
            questions=lambda: (
                question_prompt,
                build_question_extension(question_state if question_state is not None else {}),
            ),
            additional_prompts=lambda: [],
            history=lambda: agent_holder["agent"].history if "agent" in agent_holder else [],
            set_catalogs=lambda skills, agents: None,
            mcp_trusted=lambda: self._settings.is_mcp_project_trusted(project_path) if self._settings else False,
            mcp_diagnostic=lambda diagnostic: logger.warning("ACP MCP: %s", diagnostic),
            mcp_names=lambda loaded: None,
            own_bundle=lambda bundle: None,
            hook_dispatcher=lambda: hooks,
            mcp_builder=lambda project_path, state_dir, **kwargs: mcp_extension,
        )
        agent = None
        try:
            prepared = prepare_composition(request, host)
            composed, _bundle = construct_agent(request, host, prepared)
            agent = cast(CoderAgent, composed)
            setattr(agent, "_acp_generation", recording.generation)
            agent_holder["agent"] = agent
            agent.add_volatile_section(self._task_list_volatile_section(record))
            lsp_messages = await agent.tool_collection.initialize()
            for message in lsp_messages:
                logger.debug("acp lsp: %s", message)
            if restore:
                agent.restore_message_history(record.history)
                agent.restore_compaction_state(record.compaction)
                agent.session_recorder = recorder
            if existing is None:
                outcome = await agent.fire_hook(HookEvent.SESSION_START, {"source": "resume" if restore else "startup"})
                if outcome.additional_context:
                    if recorder is not None:
                        recorder.record_context_message(
                            Message(role="user", content=[TextBlock(text=outcome.additional_context)])
                        )
                    agent.append_user_message([TextBlock(text=outcome.additional_context)])
        except BaseException:
            try:
                if agent is not None:
                    await _finish_cleanup(agent.cleanup())
                elif mcp_extension is not None and mcp_extension.cleanup is not None:
                    await _finish_cleanup(mcp_extension.cleanup())
            except Exception:  # noqa: BLE001
                logger.exception("acp partial agent cleanup failed")
            raise
        return agent, manager

    # -- error conversion -------------------------------------------------

    @staticmethod
    def protocol_error(exc: BaseException) -> RequestError:
        return RequestError.internal_error({"message": str(exc) or exc.__class__.__name__})
