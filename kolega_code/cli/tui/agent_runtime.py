"""Agent construction, turn execution, and event routing for the CLI TUI."""

from __future__ import annotations

import asyncio

from textual.widgets import TabbedContent

from kolega_code.agent import AgentConfig, AgentEvent, CoderAgent, PlanningAgent, PromptExtension, ToolExtension
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.hooks import HookDispatcher, HookEvent, load_hook_config, project_hooks_present
from kolega_code.llm.exceptions import LLMError, llm_error_message
from kolega_code.services.browser import PlaywrightBrowserManager

from .. import messages
from ..config import CliConfigError, build_agent_config, config_summary
from ..skills import build_skill_prompt_extension, build_skill_tool_extension, discover_skills
from . import constants as tui_constants
from . import state as tui_state
from . import widgets as tui_widgets


class AgentRuntimeMixin:
    async def _process_message(self, message: str, attachments: list[dict] | None = None) -> None:
        if self.agent is None:
            return
        self._begin_turn_progress()
        self._log_status(messages.GENERATING, "ok")
        try:
            stream = (
                self.agent.process_message_stream(message, attachments)
                if attachments
                else self.agent.process_message_stream(message)
            )
            async for chunk in stream:
                if chunk.get("type") == "response":
                    if chunk.get("content"):
                        self._update_progress(messages.READING_RESPONSE, complete=False, state=tui_state.TurnState.GENERATING)
                    self._apply_stream_chunk(chunk, kind="assistant")
                    continue

                content = chunk.get("content")
                if chunk.get("type") == "thinking":
                    self._update_progress(messages.THINKING, complete=False, state=tui_state.TurnState.THINKING)
                    self._apply_stream_chunk(chunk, kind="thinking")
                    if content:
                        self._write_log(content, "debug")
            await self._drain_pending_events()
            self._finalize_sub_agent_activities()
            self._finalize_workflow_activities()
            await self._save_session_history_async()
            self._finish_turn_progress(messages.FINISHED, tui_state.TurnState.IDLE)
            await self._capture_completed_plan()
            self._log_status(messages.FINISHED, "ok")
        except asyncio.CancelledError:
            self._cancel_pending_question()
            self._cancel_pending_approval()
            await self._drain_pending_events()
            self._finalize_sub_agent_activities()
            self._finalize_workflow_activities()
            await self._save_session_history_async()
            self._finish_turn_progress(messages.STOPPED_BY_USER, tui_state.TurnState.STOPPED)
            self._log_status(messages.STOPPED_BY_USER, "warn")
        except LLMError as exc:
            self._cancel_pending_question()
            self._cancel_pending_approval()
            await self._drain_pending_events()
            self._finalize_sub_agent_activities()
            self._finalize_workflow_activities()
            await self._save_session_history_async()
            model = self.config.long_context_config.model if self.config is not None else None
            message_text = llm_error_message(exc, model=model)
            self._finish_turn_progress(message_text, tui_state.TurnState.ERROR)
            self._log_status(message_text, "error")
        except Exception as exc:
            self._cancel_pending_question()
            self._cancel_pending_approval()
            await self._drain_pending_events()
            self._finalize_sub_agent_activities()
            self._finalize_workflow_activities()
            await self._save_session_history_async()
            self._finish_turn_progress(messages.STOPPED_WITH_ERROR.format(error=exc), tui_state.TurnState.ERROR)
            self._log_status(messages.STOPPED_WITH_ERROR.format(error=exc), "error")
            raise
        finally:
            self._flush_conversation_render()
            self._active_progress_entry = None
            self._turn_active = False
            self.agent_worker = None
            if self._plan_decision_active:
                self._set_composer_status(messages.PLAN_READY_PLACEHOLDER)
            else:
                self._restore_composer_placeholder()
            self._set_chat_enabled(self.agent is not None and not self._plan_decision_active)

    async def _consume_events(self) -> None:
        while True:
            event = await self.connection_manager.next_event()
            self._render_event(event)

    async def _drain_pending_events(self) -> None:
        while True:
            try:
                event = self.connection_manager.events.get_nowait()
            except asyncio.QueueEmpty:
                return
            self._render_event(event)

    def _render_event(self, event: AgentEvent) -> None:
        text = self._display_text_from_event(event)
        if event.event_type == "log_message":
            level = str(event.content.get("level", "info"))
            self._write_log(text, level)
        elif event.event_type == "terminal_output":
            self._terminal.write(event.content.get("output", ""))
            self._terminal_has_content = True
            self._mark_tab_activity("terminal_pane")
        elif event.event_type == "terminal_command":
            command = str(event.content.get("command") or "")
            self._write_terminal_command(command)
            if command:
                self._update_activity_progress(messages.RUNNING_TERMINAL_COMMAND, state=tui_state.TurnState.RUNNING_TOOL)
        elif event.event_type == "chat_message":
            if event.sub_agent_info:
                self._render_sub_agent_event(event)
                return
            message_text = event.content.get("text", "")
            message_type = event.content.get("message_type", "message")
            if message_type in {"tool_call", "tool_result", "tool_error"}:
                self._add_tool_message(message_type, event.content)
            elif message_type == "workflow_start":
                self._handle_workflow_start(event.content)
            elif message_type == "workflow_phase":
                self._handle_workflow_phase(event.content)
            elif message_type == "workflow_log":
                self._handle_workflow_log(event.content)
            elif message_type == "workflow_end":
                self._handle_workflow_end(event.content)
            elif message_text:
                self._add_conversation_entry(tui_state.ConversationEntry(kind="message", content=message_text))
        elif event.event_type == "tool_streaming_update":
            if event.sub_agent_info:
                self._note_sub_agent_tool_stream(event)
            else:
                self._apply_tool_streaming_update(event.content)
        elif event.event_type == "file_edit_preview":
            # UI-only inline diff/head preview. Sub-agent edits are not shown inline (v1).
            if not event.sub_agent_info:
                self._apply_edit_preview(event.content)
        elif event.event_type == "llm_context_update":
            if event.sub_agent_info:
                self._note_sub_agent_context(event)
            else:
                self._apply_context_status_update(event.content)
        elif event.event_type == "compaction_status":
            # Only the main agent's compaction drives the status dashboard; a
            # sub-agent's compaction must not stomp the main indicator.
            if not event.sub_agent_info:
                self._apply_compaction_status(event.content)
        elif event.event_type in {"llm_status_update", "status_update"}:
            if event.sub_agent_info:
                self._note_sub_agent_status(event)
            elif text:
                self._write_log(text, "info")
                self._update_activity_progress(text)
        else:
            if text:
                self._write_log(f"{event.event_type}: {text}", "info")
            else:
                self._write_log(messages.LOG_IGNORED_EVENT.format(event_type=event.event_type), "debug")

    def action_cancel_generation(self) -> None:
        if self.agent_worker is not None:
            self._update_progress(messages.STOP_REQUESTED, complete=False, state=tui_state.TurnState.STOPPING)
            self._cancel_pending_question()
            self._cancel_pending_approval()
            self.agent_worker.cancel()
            self._notify_user(messages.CANCEL_REQUESTED, severity="warning")

    async def _ensure_agent_from_settings(self, rebuild: bool = False) -> None:
        try:
            config = build_agent_config(
                self.project_path, self.overrides, settings=self.settings, settings_store=self.settings_store
            )
        except CliConfigError as exc:
            self.config = None
            self._set_chat_enabled(False)
            self._refresh_status_dashboard()
            self._set_settings_status(messages.SETTINGS_INCOMPLETE.format(error=exc), tone="error")
            self._ensure_startup_entry()
            self.query_one("#events", TabbedContent).active = "settings_pane"
            return

        self.config = config
        self.session.config = config_summary(config)
        await self._save_session_async()
        await self._build_agent(config, rebuild=rebuild)
        self._set_chat_enabled(True)
        self._update_settings_status()
        self._ensure_startup_entry()
        self.query_one("#composer", tui_widgets.ChatComposer).focus()

    async def _build_agent(
        self,
        config: AgentConfig,
        rebuild: bool = False,
        *,
        restore_transcript: bool = True,
    ) -> None:
        history = self.session.history
        compaction = self.session.compaction
        if self.agent is not None:
            await self._save_session_history_async()
            history = self.session.history
            compaction = self.session.compaction
            if rebuild:
                await self.agent.cleanup()

        browser_manager = PlaywrightBrowserManager()
        browser_manager.headless = not self.browser_visible
        agent_class = PlanningAgent if self.interaction_mode == tui_constants.PLAN_INTERACTION_MODE else CoderAgent
        self.skill_catalog = discover_skills(self.project_path)
        prompt_extensions: list[PromptExtension] = []
        tool_extensions: list[ToolExtension] = []
        # The shared task list is build-mode execution tracking; plan mode produces
        # a plan via write_plan and does not get the task-list tools.
        if self.interaction_mode == tui_constants.BUILD_INTERACTION_MODE:
            prompt_extensions.append(self._shared_task_list_prompt_extension())
            tool_extensions.append(self._shared_task_list_tool_extension())
        skill_prompt_extension = build_skill_prompt_extension(self.skill_catalog)
        skill_tool_extension = build_skill_tool_extension(
            self.skill_catalog,
            lambda: self.agent.history if self.agent is not None else [],
        )
        if skill_prompt_extension is not None:
            prompt_extensions.append(skill_prompt_extension)
        if skill_tool_extension is not None:
            tool_extensions.append(skill_tool_extension)
        if self.interaction_mode == tui_constants.PLAN_INTERACTION_MODE:
            prompt_extensions.append(self._planning_question_prompt_extension())
            tool_extensions.append(self._planning_question_tool_extension())

        # gigacode applies to any top-level agent and is carried across rebuilds.
        # In plan mode the orchestrating agent is read-only, so its workflow
        # sub-agents are forced read-only too (enforced in the dispatch adapter).
        gigacode_active = self._gigacode_enabled
        if gigacode_active:
            prompt_extensions.append(self._gigacode_prompt_extension())

        self.agent = agent_class(
            project_path=self.project_path,
            workspace_id=self.session.workspace_id,
            thread_id=self.session.thread_id,
            connection_manager=self.connection_manager,
            config=config,
            browser_manager=browser_manager,
            agent_mode=AgentMode(self.mode),
            prompt_extensions=prompt_extensions,
            tool_extensions=tool_extensions,
            permission_mode=self.permission_mode,
            permission_callback=self._permission_callback,
            hook_dispatcher=self._session_hook_dispatcher(),
        )
        self.agent.gigacode_enabled = gigacode_active
        if history:
            self.agent.restore_message_history(history)
            self.agent.restore_compaction_state(compaction)
            if restore_transcript:
                self._restore_conversation_history(history)
        self._update_mode_chrome()
        await self._fire_session_start_once()

    def _session_hook_dispatcher(self) -> HookDispatcher:
        """Build (once) the hook dispatcher for this session from global + project config."""
        if self._hook_dispatcher is None:
            trusted = self.settings.is_hook_project_trusted(self.project_path)
            config = load_hook_config(self.project_path, self.settings_store.root, project_trusted=trusted)
            self._hook_dispatcher = HookDispatcher(config)
            self._announce_hook_status(config)
        return self._hook_dispatcher

    def _announce_hook_status(self, config) -> None:
        """Surface hook diagnostics and an untrusted-project notice once at startup."""
        for diagnostic in config.diagnostics:
            self._log_status(f"hooks: {diagnostic}", level="warn")
        if project_hooks_present(self.project_path) and not self.settings.is_hook_project_trusted(self.project_path):
            self._notify_user(
                "This project defines hooks in .kolega/hooks.json, but they are not trusted, so they "
                "are disabled. Global hooks still run. Re-launch with `--trust-hooks` to enable them.",
                severity="warning",
                title="Untrusted project hooks",
            )

    async def _fire_session_start_once(self) -> None:
        if self._session_started or self.agent is None:
            return
        fire = getattr(self.agent, "fire_hook", None)
        if fire is None:
            return
        self._session_started = True
        outcome = await fire(HookEvent.SESSION_START, {"source": "startup"})
        if outcome.additional_context:
            self._add_conversation_entry(tui_state.ConversationEntry(kind="system", content=outcome.additional_context))
