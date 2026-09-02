"""The agent session host: one Kolega session per chat, driven headlessly.

This is the gateway's real turn handler. Each chat gets a durable Kolega
session — a ``SessionRuntime`` owning a ``CoderAgent`` built with the same
recipe the ACP server uses (``CliConnectionManager`` + recording + journal) —
so recording, export, redaction, and the permission machinery all work
exactly as they do in the TUI.

Routing is a chat-key → session-id mapping persisted under the state dir, so
sessions survive daemon restarts. A per-chat worker task serializes turns
(messages queue behind the running one), and commands are handled *outside*
the queue so ``/stop`` can cancel the running turn while it runs.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from kolega_code.agent import CoderAgent, PromptExtension
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.agent.prompts import BUILD_QUESTION_PROMPT
from kolega_code.cli.config import CliConfigError, CliConfigOverrides, build_agent_config, config_summary
from kolega_code.cli.connection import CliConnectionManager
from kolega_code.cli.session_event_store import FileArtifactStore, FileSessionEventStore
from kolega_code.cli.session_store import SessionRecord, SessionStore, SessionStoreError
from kolega_code.cli.settings import CliSettings, SettingsStore
from kolega_code.events import AgentEvent, KnownEventType
from kolega_code.permissions import PermissionMode, normalize_permission_mode
from kolega_code.session.recording import RecordingConnectionManager
from kolega_code.session.runtime import SessionRuntime, control_channel_for
from kolega_code.tools import ToolError

from kolega_code.gateway.adapters.base import Attachment, ChatRef, GatewayAdapter, InboundMessage
from kolega_code.gateway.bridge import TurnRenderer
from kolega_code.gateway.commands import (
    COMMAND_HELP,
    COMMAND_MODEL,
    COMMAND_NEW,
    COMMAND_PERMISSIONS,
    COMMAND_RESET,
    COMMAND_STATUS,
    COMMAND_STOP,
    HELP_TEXT,
    UNKNOWN_COMMAND_REPLY,
    parse_command,
)
from kolega_code.gateway.config import GatewayConfig
from kolega_code.gateway.control_relay import ControlRelay
from kolega_code.gateway.daemon import ERROR_REPLY
from kolega_code.gateway.event_router import EventRouter
from kolega_code.gateway.questions import build_question_extension
from kolega_code.gateway.redaction import scrub
from kolega_code.gateway.router import SessionRegistry
from kolega_code.gateway.stt import (
    SttProvider,
    SttProviderError,
    SttProviderNotConfigured,
    build_transcriber,
)
from kolega_code.utils.images import encode_image_file

logger = logging.getLogger(__name__)

SESSION_MAP_FILE_NAME = "gateway_sessions.json"

#: Builds the agent for one session. Injecting this is how tests run the
#: whole host against a scripted LLM; the default mirrors the ACP recipe.
AgentBuilder = Callable[..., Awaitable[Any]]


@dataclass
class ChatSession:
    """Live state for one chat's agent session."""

    chat_ref: ChatRef
    session_id: str
    record: SessionRecord
    runtime: SessionRuntime
    manager: CliConnectionManager
    agent_config: Any
    router: EventRouter
    turn_events: asyncio.Queue[AgentEvent]
    relay: ControlRelay
    #: Mutable holder the agent factory reads at every build, so /model can
    #: swap the config and rebuild the session in place.
    config_holder: dict[str, Any]
    turn_task: Optional[asyncio.Task[None]] = None


class AgentTurnHandler:
    """The gateway's session host: chats in, turns out, sessions persisted."""

    def __init__(
        self,
        *,
        config: GatewayConfig,
        adapter: GatewayAdapter,
        store: SessionStore,
        settings_store: SettingsStore,
        overrides: Optional[CliConfigOverrides] = None,
        agent_builder: Optional[AgentBuilder] = None,
        transcriber: Optional[SttProvider] = None,
    ) -> None:
        self._config = config
        self._adapter = adapter
        self._store = store
        self._settings_store = settings_store
        self._settings: Optional[CliSettings] = None
        self._overrides = overrides or CliConfigOverrides()
        self._agent_builder = agent_builder
        self._transcriber = transcriber
        self._registry = SessionRegistry(
            max_sessions=config.max_sessions,
            idle_ttl_seconds=config.session_idle_ttl_seconds,
            on_evict=self._on_evict,
        )
        self._map_path = config.state_dir / SESSION_MAP_FILE_NAME
        self._chat_sessions = self._load_map()
        self._queues: dict[str, asyncio.Queue[InboundMessage]] = {}
        self._workers: dict[str, asyncio.Task[None]] = {}
        self._secret_values = self._collect_secret_values()

    # -- TurnHandler interface ---------------------------------------------

    def status(self) -> dict[str, Any]:
        return {"active_sessions": self._registry.active_count()}

    async def handle(self, chat_ref: ChatRef, message: InboundMessage) -> None:
        if message.callback_token:
            # A button tap answers a control prompt, never starts a turn.
            entry = self._registry.get(chat_ref)
            if entry is not None:
                await entry.payload.relay.handle_tap(message)
            return
        command = parse_command(message.text)
        if command is not None:
            await self._run_command(chat_ref, message, command)
            return
        await self._ensure_session(chat_ref)  # also prunes idle sessions
        queue = self._queues.setdefault(chat_ref.key, asyncio.Queue())
        queue.put_nowait(message)
        worker = self._workers.get(chat_ref.key)
        if worker is None or worker.done():
            self._workers[chat_ref.key] = asyncio.create_task(
                self._worker(chat_ref), name=f"gateway-worker-{chat_ref.key}"
            )

    async def shutdown(self) -> None:
        for worker in list(self._workers.values()):
            worker.cancel()
        for worker in list(self._workers.values()):
            try:
                await worker
            except asyncio.CancelledError:
                pass
        self._workers.clear()
        self._queues.clear()
        await self._registry.clear()
        self._save_map()

    # -- Per-chat worker ---------------------------------------------------

    async def _worker(self, chat_ref: ChatRef) -> None:
        queue = self._queues[chat_ref.key]
        try:
            while True:
                message = await queue.get()
                entry = await self._ensure_session(chat_ref)
                turn_task: asyncio.Task[None] = asyncio.create_task(
                    self._run_turn(entry, chat_ref, message),
                    name=f"gateway-turn-{entry.session_id}",
                )
                entry.turn_task = turn_task
                try:
                    # Shield breaks the cancellation coupling: cancelling the
                    # worker (shutdown/eviction) must not silently cancel the
                    # turn it awaits — the two have different outcomes.
                    await asyncio.shield(turn_task)
                except asyncio.CancelledError:
                    if turn_task.cancelled():
                        # /stop cancelled the turn; the worker keeps serving.
                        # A stopped turn's unwind can race the journal's
                        # cross-thread lock and end in a non-CancelledError
                        # exception, so settle with a broad catch.
                        try:
                            await turn_task
                        except BaseException:  # noqa: BLE001 — the outcome is already decided
                            pass
                    else:
                        # The worker itself was cancelled: stop the turn and
                        # propagate, so shutdown/eviction can proceed.
                        turn_task.cancel()
                        try:
                            await turn_task
                        except BaseException:  # noqa: BLE001
                            pass
                        entry.turn_task = None
                        raise asyncio.CancelledError()
                except Exception:
                    # A genuinely failed turn; a *stopped* turn can also land
                    # here when its cancellation unwind raced the journal's
                    # cross-thread lock, so check the task state first.
                    if not turn_task.cancelled():
                        logger.exception("gateway: turn failed for chat %s", chat_ref.key)
                        try:
                            await self._adapter.send_text(chat_ref.chat_id, ERROR_REPLY)
                        except Exception:  # noqa: BLE001 — the log already has the details
                            pass
                finally:
                    entry.turn_task = None
        except asyncio.CancelledError:
            raise
        finally:
            self._workers.pop(chat_ref.key, None)
            self._queues.pop(chat_ref.key, None)

    async def _run_turn(self, entry: ChatSession, chat_ref: ChatRef, message: InboundMessage) -> None:
        # A previous turn may have left unconsumed events (its pump was
        # cancelled); stale tool lines must not leak into this turn's status.
        while not entry.turn_events.empty():
            entry.turn_events.get_nowait()
        text = message.text
        attachments: list[dict[str, Any]] = []
        for attachment in message.attachments:
            if attachment.kind == "voice":
                source_path = Path(attachment.source)
                logger.info(
                    "gateway: transcribing voice note %s (%s bytes)",
                    attachment.source,
                    source_path.stat().st_size if source_path.exists() else "missing",
                )
                transcript = await self._transcribe(chat_ref, attachment.source)
                if transcript is None:
                    # _transcribe already told the user why there is no turn.
                    return
                # The model must know the medium, not just the words: a
                # transcribed voice note is labeled, and any caption the user
                # typed rides along.
                caption = text.strip()
                text = f"[voice message transcribed]: {transcript}"
                if caption:
                    text = f"{caption}\n\n{text}"
            elif attachment.kind == "image":
                encoded = encode_image_file(attachment.source)
                if encoded is not None:
                    attachments.append(encoded)
                else:
                    await self._adapter.send_text(
                        chat_ref.chat_id,
                        "🖼 That image could not be attached (unsupported format or too large).",
                    )
            elif attachment.kind == "document":
                text = self._document_note(text, attachment)
            else:
                logger.debug("gateway: ignoring attachment kind %r", attachment.kind)
        if message.reply_to is not None and message.reply_to.text:
            quoted_by = message.reply_to.sender_id or "user"
            text = f"[replying to {quoted_by}: {message.reply_to.text}]\n\n{text}" if text else ""
        text = text.strip()
        if not text and not attachments:
            return  # nothing the model can act on (e.g. an undecodable voice note)
        if not text:
            text = "[attachment]"
        renderer = TurnRenderer(
            self._adapter,
            chat_ref.chat_id,
            event_queue=entry.turn_events,
            chunk_limit=self._adapter.capabilities.text_chunk_limit,
            edit_throttle_seconds=self._config.edit_throttle_seconds,
        )
        stream = entry.runtime.send_message(text, attachments or None)
        await renderer.run(self._scrub_chunks(stream))
        self._persist(entry)

    async def _transcribe(self, chat_ref: ChatRef, source: str) -> Optional[str]:
        """Transcribe a voice note.

        Returns the transcript, or ``None`` when no turn should run because
        the user has already been told why (every failure path sends its own
        specific notice).
        """
        if not self._config.stt_enabled:
            await self._adapter.send_text(
                chat_ref.chat_id,
                "🎙 Voice transcription is disabled. Enable it in Settings → Tools → Voice transcription.",
            )
            return None
        transcriber = self._transcriber
        if transcriber is None:
            try:
                transcriber = build_transcriber(
                    self._config.stt_provider,
                    api_key=self._config.stt_api_key,
                    model=self._config.stt_model,
                )
            except SttProviderNotConfigured as exc:
                await self._adapter.send_text(chat_ref.chat_id, f"🎙 {exc}")
                return None
            except SttProviderError as exc:
                await self._adapter.send_text(chat_ref.chat_id, f"🎙 {exc}")
                return None
            self._transcriber = transcriber
        try:
            transcript = await transcriber.transcribe(Path(source))
        except Exception:  # noqa: BLE001
            logger.exception("gateway: voice transcription failed")
            await self._adapter.send_text(
                chat_ref.chat_id, "🎙 Transcription failed; the voice note was not understood."
            )
            return None
        logger.info("gateway: voice transcription succeeded (%d chars)", len(transcript))
        return transcript.strip()

    @staticmethod
    def _document_note(text: str, attachment: Attachment) -> str:
        """Tell the model where a document landed so it can read it with tools."""
        label = attachment.file_name or "document"
        note = f"[document: {label} saved at {attachment.source}]"
        return f"{note}\n\n{text}" if text else note

    async def _scrub_chunks(self, stream: Any) -> AsyncIterator[dict[str, Any]]:
        async for chunk in stream:
            content = chunk.get("content")
            if content:
                chunk["content"] = scrub(content, self._secret_values)
            yield chunk

    # -- Commands ----------------------------------------------------------

    async def _run_command(self, chat_ref: ChatRef, _message: InboundMessage, command: tuple[str, str]) -> None:
        name, args = command
        if name == COMMAND_HELP or name == "help":
            await self._adapter.send_text(chat_ref.chat_id, HELP_TEXT)
        elif name == COMMAND_STATUS:
            await self._adapter.send_text(chat_ref.chat_id, await self._status_text(chat_ref))
        elif name == COMMAND_MODEL:
            await self._model_command(chat_ref, args)
        elif name == COMMAND_PERMISSIONS:
            await self._permissions_command(chat_ref, args)
        elif name in (COMMAND_NEW, COMMAND_RESET):
            await self._reset_session(chat_ref)
            await self._adapter.send_text(chat_ref.chat_id, "🆕 New session started.")
        elif name == COMMAND_STOP:
            if await self._stop_turn(chat_ref):
                await self._adapter.send_text(chat_ref.chat_id, "🛑 Stopping the current turn.")
            else:
                await self._adapter.send_text(chat_ref.chat_id, "No turn is running.")
        else:
            await self._adapter.send_text(chat_ref.chat_id, UNKNOWN_COMMAND_REPLY)

    async def _status_text(self, chat_ref: ChatRef) -> str:
        entry = self._registry.get(chat_ref)
        if entry is None:
            return "No active session yet — send a message to start one."
        payload: ChatSession = entry.payload
        agent = payload.runtime.agent
        if agent is None:
            return f"Session {payload.session_id} is starting up."
        primary = agent.primary_model_config
        return (
            f"Session {payload.session_id}\n"
            f"Model: {primary.provider.value}/{primary.model}\n"
            f"Permissions: {payload.runtime.permission_mode.value}\n"
            f"Project: {payload.record.project_path}"
        )

    async def _reset_session(self, chat_ref: ChatRef) -> None:
        await self._stop_turn(chat_ref)
        entry = self._registry.remove(chat_ref)
        if entry is not None:
            await self._close_entry(entry.payload)
        self._chat_sessions.pop(chat_ref.key, None)
        self._save_map()
        queue = self._queues.get(chat_ref.key)
        if queue is not None:
            while not queue.empty():
                queue.get_nowait()

    async def _stop_turn(self, chat_ref: ChatRef) -> bool:
        entry = self._registry.get(chat_ref)
        if entry is None or entry.payload.turn_task is None:
            return False
        entry.payload.turn_task.cancel()
        return True

    async def _model_command(self, chat_ref: ChatRef, args: str) -> None:
        entry = self._registry.get(chat_ref)
        if entry is None or entry.payload.runtime.agent is None:
            await self._adapter.send_text(chat_ref.chat_id, "No active session yet — send a message to start one.")
            return
        if not args.strip():
            primary = entry.payload.runtime.agent.primary_model_config
            await self._adapter.send_text(
                chat_ref.chat_id,
                f"Model: {primary.provider.value}/{primary.model}\n"
                "Switch with /model <provider>/<model> (or /model <model> to keep the provider).",
            )
            return
        await self._apply_model(chat_ref, entry.payload, args.strip())

    async def _apply_model(self, chat_ref: ChatRef, entry: ChatSession, spec: str) -> None:
        turn_task = entry.turn_task
        if turn_task is not None and not turn_task.done():
            await self._adapter.send_text(chat_ref.chat_id, "A turn is running — /stop it first, then switch models.")
            return
        provider, model = self._parse_model_spec(spec, entry)
        if not provider:
            await self._adapter.send_text(
                chat_ref.chat_id,
                f"Could not switch models: name a provider as <provider>/<model>, got {spec!r}.",
            )
            return
        try:
            new_config = build_agent_config(
                Path(entry.record.project_path),
                overrides=CliConfigOverrides(provider=provider, model=model),
                settings=self._load_settings(),
                settings_store=self._settings_store,
            )
        except CliConfigError as exc:
            await self._adapter.send_text(chat_ref.chat_id, f"Could not switch models: {exc}")
            return
        # Persist the live conversation before the rebuild loads it back.
        self._persist(entry)
        entry.config_holder["config"] = new_config
        try:
            await entry.runtime.rebuild()
        except Exception:  # noqa: BLE001 — a failed switch must not kill the chat
            logger.exception("gateway: model switch failed for chat %s", chat_ref.key)
            await self._adapter.send_text(chat_ref.chat_id, ERROR_REPLY)
            return
        entry.agent_config = new_config
        entry.record.config = config_summary(new_config)
        self._store.save(entry.record)
        await self._adapter.send_text(chat_ref.chat_id, f"Model: {provider}/{model}")

    @staticmethod
    def _parse_model_spec(spec: str, entry: ChatSession) -> tuple[str, str]:
        """Split ``provider/model`` or resolve a bare model against the current provider."""
        if "/" in spec:
            provider, model = spec.split("/", 1)
            return provider, model
        agent = entry.runtime.agent
        if agent is None:
            return "", spec
        return agent.primary_model_config.provider.value, spec

    async def _permissions_command(self, chat_ref: ChatRef, args: str) -> None:
        entry = self._registry.get(chat_ref)
        if entry is None or entry.payload.runtime.agent is None:
            await self._adapter.send_text(chat_ref.chat_id, "No active session yet — send a message to start one.")
            return
        payload = entry.payload
        if not args.strip():
            await self._adapter.send_text(
                chat_ref.chat_id,
                f"Permissions: {payload.runtime.permission_mode.value}\n"
                "Change with /permissions ask (approve each tool via buttons) or "
                "/permissions auto (approve everything).",
            )
            return
        try:
            permission_mode = PermissionMode(args.strip().lower())
        except ValueError:
            await self._adapter.send_text(
                chat_ref.chat_id,
                f"Unknown permission mode {args.strip()!r}. Valid modes: ask, auto.",
            )
            return
        payload.runtime.set_permission_mode(permission_mode)
        payload.record.permission_mode = permission_mode.value
        self._store.save(payload.record)
        await self._adapter.send_text(chat_ref.chat_id, f"Permissions: {permission_mode.value}")

    # -- Session lifecycle -------------------------------------------------

    async def _ensure_session(self, chat_ref: ChatRef) -> ChatSession:
        entry = self._registry.get(chat_ref)
        if entry is not None:
            return entry.payload
        settings = self._load_settings()
        record = None
        restore = False
        session_id = self._chat_sessions.get(chat_ref.key)
        if session_id is not None:
            try:
                record = self._store.load(session_id)
                restore = True
            except SessionStoreError:
                logger.info("gateway: session %s gone; creating a fresh one", session_id)
                record = None
        agent_config = build_agent_config(
            Path(record.project_path) if record is not None else self._config.project_path,
            overrides=self._overrides,
            settings=settings,
            settings_store=self._settings_store,
        )
        if record is None:
            record = self._store.create(
                Path(self._config.project_path),
                AgentMode.CLI.value,
                config_summary(agent_config),
            )
            record.permission_mode = self._config.permission_mode
            record.title = f"{chat_ref.channel} chat {chat_ref.chat_id}"
            self._chat_sessions[chat_ref.key] = record.session_id
            self._save_map()
            restore = False
        payload = await self._build_chat_session(chat_ref, record, agent_config, restore=restore)
        entry = self._registry.put(chat_ref, payload)
        # The new entry is most-recently-active, so an over-capacity prune
        # evicts the least-recent neighbour — never the chat that just spoke.
        await self._registry.prune()
        return entry.payload

    async def _build_chat_session(
        self,
        chat_ref: ChatRef,
        record: SessionRecord,
        agent_config: Any,
        *,
        restore: bool,
    ) -> ChatSession:
        journal = self._store.journal(record.session_id)
        recorder = self._store.recorder(record.session_id)
        manager = CliConnectionManager()
        recording = RecordingConnectionManager(
            manager,
            FileSessionEventStore(journal),
            session_id=record.session_id,
            artifact_store=FileArtifactStore(journal),
        )
        permission_mode = normalize_permission_mode(
            record.permission_mode,
            default=PermissionMode(self._config.permission_mode),
        )
        control = control_channel_for(
            record.session_id,
            recording.broadcast_event,
            workspace_id=record.workspace_id,
            thread_id=record.thread_id,
            timeout=self._config.request_timeout_seconds,
        )
        # One fan-out task owns the session's event queue so the turn renderer
        # (tool activity) and the relay (control prompts) never steal each
        # other's events.
        router = EventRouter(manager.events)
        turn_events = router.subscribe(KnownEventType.CHAT_MESSAGE)
        control_events = router.subscribe(KnownEventType.CONTROL_REQUESTED, KnownEventType.CONTROL_RESOLVED)
        builder = self._agent_builder or self._default_agent_builder
        config_holder: dict[str, Any] = {"config": agent_config}
        build_state = {"restore": restore}

        async def factory() -> Any:
            built = await builder(
                record,
                config_holder["config"],
                manager=manager,
                recording=recording,
                recorder=recorder,
                permission_callback=runtime.permission_callback,
                permission_mode=PermissionMode.ASK,
                restore=build_state["restore"],
                question_state=question_state,
            )
            # Every build after the first must restore the conversation.
            build_state["restore"] = True
            return built

        runtime = SessionRuntime(
            session_id=record.session_id,
            project_path=Path(record.project_path),
            control=control,
            agent_factory=factory,
            permission_mode=permission_mode,
            on_notice=lambda notice: logger.info("gateway session %s: %s", record.session_id, notice),
        )
        relay = ControlRelay(
            chat_ref=chat_ref,
            runtime=runtime,
            adapter=self._adapter,
            event_queue=control_events,
            client_id=chat_ref.key,
            scrub=lambda text: scrub(text, self._secret_values),
        )
        question_state: dict[str, Any] = {}

        async def elicit(question: str, labels: list[str], descriptions: list[str]) -> str:
            # Same wire shape as the TUI's ask_user_choice, so the prompt lands
            # in the recording and the relay renders it like any other.
            response = await control.request(
                "question",
                {"question": question, "options": list(labels), "descriptions": list(descriptions)},
                default={"answer": None},
            )
            answer = response.get("answer")
            if not isinstance(answer, str) or not answer.strip():
                raise ToolError(
                    "No answer was given for this planning question. "
                    "Proceed without it, or state the assumption you are making instead."
                )
            return answer

        question_state["elicit"] = elicit
        router.start()
        relay.start()
        await runtime.start()
        return ChatSession(
            chat_ref=chat_ref,
            session_id=record.session_id,
            record=record,
            runtime=runtime,
            manager=manager,
            agent_config=agent_config,
            router=router,
            turn_events=turn_events,
            relay=relay,
            config_holder=config_holder,
        )

    async def _default_agent_builder(self, record: SessionRecord, agent_config: Any, **kwargs: Any) -> Any:
        recording: RecordingConnectionManager = kwargs["recording"]
        recorder: Any = kwargs["recorder"]
        permission_callback = kwargs["permission_callback"]
        permission_mode = kwargs["permission_mode"]
        restore: bool = kwargs["restore"]
        question_state: dict[str, Any] = kwargs["question_state"]
        agent = CoderAgent(
            project_path=Path(record.project_path),
            workspace_id=record.workspace_id,
            thread_id=record.thread_id,
            connection_manager=recording,
            config=agent_config,
            agent_mode=AgentMode.CLI,
            permission_mode=permission_mode,
            permission_callback=permission_callback,
            prompt_extensions=[
                PromptExtension(
                    id="gateway-questions",
                    title="Asking the User",
                    markdown=BUILD_QUESTION_PROMPT,
                    modes=[AgentMode.CLI],
                    propagate_to_sub_agents=False,
                ),
            ],
            tool_extensions=[build_question_extension(question_state)],
            # Mirror the ask/ACP path: a resumed session hands the recorder
            # over after restoring history so construction never
            # double-records a resumed turn.
            session_recorder=None if restore else recorder,
        )
        lsp_messages = await agent.tool_collection.initialize()
        for lsp_message in lsp_messages:
            logger.debug("gateway lsp: %s", lsp_message)
        if restore:
            agent.restore_message_history(record.history)
            agent.restore_compaction_state(record.compaction)
            agent.session_recorder = recorder
        return agent

    def _persist(self, entry: ChatSession) -> None:
        agent = entry.runtime.agent
        if agent is None:
            return
        record = entry.record
        record.history = agent.dump_message_history()
        record.compaction = agent.dump_compaction_state()
        record.permission_mode = agent.permission_mode.value
        self._store.save(record)

    async def _on_evict(self, payload: Any) -> None:
        chat_session: ChatSession = payload
        worker = self._workers.get(chat_session.chat_ref.key)
        if worker is not None and worker is not asyncio.current_task():
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass
        await self._close_entry(chat_session)

    async def _close_entry(self, chat_session: ChatSession) -> None:
        self._persist(chat_session)
        try:
            await chat_session.relay.stop()
        except Exception:  # noqa: BLE001 — closing must not mask the eviction
            logger.exception("gateway: relay stop failed (%s)", chat_session.session_id)
        await chat_session.router.stop()
        try:
            await chat_session.runtime.shutdown()
        except Exception:  # noqa: BLE001 — closing must not mask the eviction
            logger.exception("gateway: session close failed (%s)", chat_session.session_id)

    # -- Persisted chat→session mapping ------------------------------------

    def _load_map(self) -> dict[str, str]:
        try:
            raw = json.loads(self._map_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("gateway: unreadable session map %s (%s); starting fresh", self._map_path, exc)
            return {}
        return {str(key): str(value) for key, value in raw.items() if isinstance(value, str) and value}

    def _save_map(self) -> None:
        try:
            self._map_path.parent.mkdir(parents=True, exist_ok=True)
            self._map_path.write_text(json.dumps(self._chat_sessions, indent=2, sort_keys=True), encoding="utf-8")
        except OSError as exc:
            logger.warning("gateway: could not save session map (%s)", exc)

    # -- Settings and secrets ----------------------------------------------

    def _load_settings(self) -> CliSettings:
        if self._settings is None:
            self._settings = self._settings_store.load()
        return self._settings

    def _collect_secret_values(self) -> list[str]:
        secrets: list[str] = []
        if self._config.telegram_token:
            secrets.append(self._config.telegram_token)
        try:
            from kolega_code.cli.main import known_secret_values

            secrets.extend(
                known_secret_values(
                    self._load_settings(),
                    self._settings_store,
                    project_path=self._config.project_path,
                )
            )
        except Exception:  # noqa: BLE001 — redaction is best-effort, never fatal
            logger.debug("gateway: could not collect secret values", exc_info=True)
        return secrets
