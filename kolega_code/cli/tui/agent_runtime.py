"""Agent construction, turn execution, and event routing for the CLI TUI."""

from __future__ import annotations

import asyncio

from kolega_code.agent import AgentConfig, AgentEvent, CoderAgent, PlanningAgent, PromptExtension, ToolExtension
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.agent.prompts import build_current_plan_artifact_prompt
from kolega_code.hooks import HookDispatcher, HookEvent, load_hook_config, project_hooks_present
from kolega_code.llm.exceptions import LLMError, llm_error_message
from kolega_code.mcp.tools import build_mcp_tool_extension
from kolega_code.services.browser import PlaywrightBrowserManager

from .. import messages
from ..config import CliConfigError, build_agent_config, config_summary
from ..plan_artifacts import write_current_plan_artifact
from ..skills import build_skill_prompt_extension, build_skill_tool_extension, discover_skills
from . import constants as tui_constants
from . import state as tui_state


class AgentRuntimeMixin:
    def _agent_turn_stream(self, message: str, attachments: list[dict] | None = None):
        if attachments:
            return self.agent.process_message_stream(message, attachments)
        return self.agent.process_message_stream(message)

    async def _consume_agent_stream(self, stream) -> None:
        async for chunk in stream:
            if chunk.get("type") == "response":
                if chunk.get("content"):
                    self._update_progress(
                        messages.READING_RESPONSE, complete=False, state=tui_state.TurnState.GENERATING
                    )
                self._apply_stream_chunk(chunk, kind="assistant")
                continue

            content = chunk.get("content")
            if chunk.get("type") == "thinking":
                self._update_progress(messages.THINKING, complete=False, state=tui_state.TurnState.THINKING)
                self._apply_stream_chunk(chunk, kind="thinking")
                if content and self.show_logs:
                    self._write_log(content, "debug")

    async def _run_turn_stream(self, stream_factory) -> bool:
        """Run one agent stream and return whether it was cancelled by the user."""
        cancelled_by_user = False
        self._begin_turn_progress()
        self._log_status(messages.GENERATING, "ok")
        try:
            await self._consume_agent_stream(stream_factory())
            await self._drain_pending_events()
            self._refresh_session_diff()
            self._finalize_sub_agent_activities()
            self._finalize_workflow_activities()
            await self._save_session_history_async()
            self._finish_turn_progress(messages.FINISHED, tui_state.TurnState.IDLE)
            await self._capture_completed_plan()
            self._log_status(messages.FINISHED, "ok")
        except asyncio.CancelledError:
            cancelled_by_user = True
            self._cancel_pending_question()
            self._cancel_pending_approval()
            await self._drain_pending_events()
            self._refresh_session_diff()
            self._finalize_sub_agent_activities()
            self._finalize_workflow_activities()
            await self._save_session_history_async()
            self._finish_turn_progress(messages.STOPPED_BY_USER, tui_state.TurnState.STOPPED)
            self._log_status(messages.STOPPED_BY_USER, "warn")
        except LLMError as exc:
            self._cancel_pending_question()
            self._cancel_pending_approval()
            await self._drain_pending_events()
            self._refresh_session_diff()
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
            self._refresh_session_diff()
            self._finalize_sub_agent_activities()
            self._finalize_workflow_activities()
            await self._save_session_history_async()
            self._finish_turn_progress(messages.STOPPED_WITH_ERROR.format(error=exc), tui_state.TurnState.ERROR)
            self._log_status(messages.STOPPED_WITH_ERROR.format(error=exc), "error")
            raise
        finally:
            self._flush_terminal_output()
            self._flush_log_output()
            self._flush_conversation_render()
            self._active_progress_entry = None
            self._turn_active = False
            self.agent_worker = None
            if self._plan_decision_active:
                self._set_composer_status(messages.PLAN_READY_PLACEHOLDER)
            else:
                self._restore_composer_placeholder()
            self._set_chat_enabled(self.agent is not None and not self._plan_decision_active)
            if cancelled_by_user:
                self._schedule_primary_focus_restore()
        return cancelled_by_user

    async def _process_message(self, message: str, attachments: list[dict] | None = None) -> None:
        if self.agent is None:
            return
        cancelled_by_user = await self._run_turn_stream(lambda: self._agent_turn_stream(message, attachments))
        if not cancelled_by_user:
            self._schedule_maybe_start_queued_message()

    def _queue_user_message(self, text: str, attachments: list[dict] | None = None) -> None:
        self._queued_message_seq += 1
        entry = tui_state.ConversationEntry(kind="queued", content=text)
        queued = tui_state.QueuedMessage(
            queue_id=f"queued-{self._queued_message_seq}",
            text=text,
            attachments=[dict(item) for item in attachments] if attachments else None,
            entry=entry,
        )
        self._queued_messages.append(queued)
        self._add_conversation_entry(entry)
        self._refresh_queued_messages_panel()
        self._log_status(messages.QUEUED_MESSAGE, "info")

    def _queued_messages_preview(self) -> str:
        if not self._queued_messages:
            return messages.QUEUE_EMPTY
        lines = [f"{messages.QUEUE_LIST_TITLE} {len(self._queued_messages)}"]
        for index, queued in enumerate(self._queued_messages, start=1):
            preview = " ".join(queued.text.strip().split()) or "(empty)"
            if len(preview) > 120:
                preview = preview[:117] + "…"
            suffix = ""
            if queued.attachments:
                suffix = f" ({len(queued.attachments)} attachment(s))"
            lines.append(f"{index}. {preview}{suffix}")
        return "\n".join(lines)

    def _refresh_queued_messages_panel(self) -> None:
        try:
            panel = self.query_one("#queued_messages")
        except Exception:
            return
        if not self._queued_messages:
            panel.update("")
            panel.display = False
            return
        panel.update(self._queued_messages_preview())
        panel.display = (
            self._pending_approval is None and self._pending_question is None and not self._plan_decision_active
        )

    def _remove_queued_entries_from_transcript(self, queued: list[tui_state.QueuedMessage]) -> None:
        entry_ids = {item.entry.entry_id for item in queued}
        before = len(self.conversation_entries)
        self.conversation_entries = [entry for entry in self.conversation_entries if entry.entry_id not in entry_ids]
        if len(self.conversation_entries) != before:
            self._render_conversation()

    def _clear_queued_messages(self) -> int:
        queued = list(self._queued_messages)
        self._queued_messages.clear()
        self._remove_queued_entries_from_transcript(queued)
        self._refresh_queued_messages_panel()
        return len(queued)

    def _restore_queued_messages_to_composer(self) -> int:
        queued = list(self._queued_messages)
        if not queued:
            return 0

        queued_text = "\n\n".join(item.text for item in queued)
        try:
            composer = self.query_one("#composer")
            existing_text = getattr(composer, "text", "") or ""
            restored_text = f"{queued_text}\n\n{existing_text}" if existing_text else queued_text
            composer.load_text(restored_text)
        except Exception:
            pass

        self._queued_messages.clear()
        self._remove_queued_entries_from_transcript(queued)
        self._refresh_queued_messages_panel()
        return len(queued)

    def _schedule_maybe_start_queued_message(self) -> None:
        try:
            self.set_timer(0.01, self._maybe_start_queued_message, name="queued-message-drain")
        except Exception:
            self._maybe_start_queued_message()

    def _maybe_start_queued_message(self) -> bool:
        if not self._queued_messages or self.agent is None:
            return False
        if self._turn_active or self.agent_worker is not None:
            return False
        if (
            self._pending_question is not None
            or self._pending_approval is not None
            or self._pending_model_selection is not None
            or self._pending_effort_selection is not None
            or self._pending_theme_selection is not None
            or self._plan_decision_active
        ):
            return False

        queued = self._queued_messages.pop(0)
        queued.entry.kind = "user"
        queued.entry.tone = None
        if queued.entry not in self.conversation_entries:
            self._add_conversation_entry(queued.entry)
        else:
            # The entry is already mounted (added when queued); just re-render that one
            # widget for the queued->user transition instead of rebuilding the whole
            # transcript, which on a long thread is a synchronous O(entries) hitch.
            self._invalidate_conversation(queued.entry)
        self._refresh_queued_messages_panel()
        self._clear_composer_hint()
        self.agent_worker = self.run_worker(
            self._process_message(queued.text, queued.attachments), name="kolega-turn", group="turns", exclusive=True
        )
        return True

    async def _consume_events(self) -> None:
        while True:
            event = await self.connection_manager.next_event()
            self._render_event(event)

    async def _drain_pending_events(self) -> None:
        while True:
            try:
                event = self.connection_manager.events.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._render_event(event)
        self._flush_terminal_output()
        self._flush_log_output()

    def _diag_tee(self, event: AgentEvent) -> None:
        """Persist a curated, secret-scrubbed slice of the event stream to the diagnostics
        timeline so a hang/error report shows what the turn was doing. Never raises."""
        diag = getattr(self, "_diag", None)
        if diag is None:
            return
        et = event.event_type
        content = event.content or {}
        agent = "sub_agent" if event.sub_agent_info else None
        try:
            if et == "log_message":
                level = str(content.get("level", "info"))
                if level in ("warning", "error"):
                    diag.record("log", level=level, text=content.get("text") or content.get("message"), agent=agent)
            elif et in ("llm_status_update", "status_update"):
                diag.record(
                    "llm_status",
                    status=content.get("status"),
                    message=content.get("message") or content.get("text"),
                    agent=agent,
                )
            elif et == "llm_error":
                diag.record(
                    "llm_error",
                    provider=content.get("provider"),
                    model=content.get("model"),
                    endpoint=content.get("endpoint"),
                    http_status=content.get("http_status"),
                    error_type=content.get("error_type"),
                    raw_type=content.get("raw_type"),
                    attempt=content.get("attempt"),
                    message=content.get("message"),
                    agent=agent,
                )
            elif et == "llm_request":
                diag.record(
                    "llm_request",
                    phase=content.get("phase"),
                    provider=content.get("provider"),
                    model=content.get("model"),
                    endpoint=content.get("endpoint"),
                    elapsed_s=content.get("elapsed_s"),
                    stop_reason=content.get("stop_reason"),
                    agent=agent,
                )
            elif et == "llm_context_update":
                diag.record(
                    "context",
                    input_tokens=content.get("input_tokens"),
                    max_tokens=content.get("max_tokens"),
                    usage_percentage=content.get("usage_percentage"),
                    alert_level=content.get("alert_level"),
                    agent=agent,
                )
            elif et == "compaction_status":
                diag.record(
                    "compaction",
                    phase=content.get("phase") or content.get("status"),
                    message=content.get("message"),
                    agent=agent,
                )
            elif et == "chat_message" and content.get("message_type") in ("tool_call", "tool_error"):
                diag.record(
                    "tool",
                    message_type=content.get("message_type"),
                    tool=content.get("tool_description") or content.get("tool_name"),
                    tool_call_id=content.get("tool_call_id"),
                    text=content.get("text"),
                    agent=agent,
                )
        except Exception:
            pass

    def _render_event(self, event: AgentEvent) -> None:
        self._diag_tee(event)
        if event.event_type in ("llm_error", "llm_request"):
            return  # diagnostics-only events; persisted by the tee, nothing to render
        if event.event_type == "log_message":
            if self.show_logs:
                level = str(event.content.get("level", "info"))
                self._write_log(self._display_text_from_event(event), level)
        elif event.event_type == "terminal_output":
            output = event.content.get("display_output")
            if output is None:
                output = event.content.get("output", "")
            self._queue_terminal_output(str(output))
            self._refresh_session_diff()
        elif event.event_type == "terminal_command":
            command = str(event.content.get("command") or "")
            self._write_terminal_command(command)
            if command:
                self._update_activity_progress(
                    messages.RUNNING_TERMINAL_COMMAND, state=tui_state.TurnState.RUNNING_TOOL
                )
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
            # UI-only diff/head preview. Main-agent edits stay inline; every edit is
            # also captured in the session changes inspector. Sub-agent previews attach
            # to their trajectory steps so Ctrl+G and Ctrl+R can both reveal them.
            self._record_file_change_event(event)
            if event.sub_agent_info:
                self._apply_sub_agent_edit_preview(event)
            else:
                self._apply_edit_preview(event.content)
            self._refresh_session_diff()
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
            else:
                text = self._display_text_from_event(event)
                if text:
                    if self.show_logs:
                        self._write_log(text, "info")
                    self._update_activity_progress(text)
        elif self.show_logs:
            text = self._display_text_from_event(event)
            if text:
                self._write_log(f"{event.event_type}: {text}", "info")
            else:
                self._write_log(messages.LOG_IGNORED_EVENT.format(event_type=event.event_type), "debug")

    def action_cancel_generation(self) -> None:
        if self.agent_worker is not None:
            self._update_progress(messages.STOP_REQUESTED, complete=False, state=tui_state.TurnState.STOPPING)
            self._cancel_pending_question()
            self._cancel_pending_approval()
            self._restore_queued_messages_to_composer()
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
            return

        self.config = config
        self.session.config = config_summary(config)
        await self._save_session_async()
        await self._build_agent(config, rebuild=rebuild)
        self._set_chat_enabled(True)
        self._update_settings_status()
        self._ensure_startup_entry()
        self._schedule_primary_focus_restore()

    def _ensure_current_plan_artifact(self, plan: str | None = None):
        """Persist the latest plan to the session artifact path, returning the path if available."""
        plan_markdown = plan if plan is not None else getattr(self, "_latest_plan", None)
        if not plan_markdown:
            return None
        try:
            return write_current_plan_artifact(self.store.root, self.session.session_id, plan_markdown)
        except Exception as exc:  # noqa: BLE001 - artifact persistence is best-effort for continuity
            try:
                self._notify_user(
                    f"Could not persist current plan artifact: {exc}",
                    severity="warning",
                )
            except Exception:
                pass
            return None

    def _current_plan_artifact_prompt_extension(self) -> PromptExtension | None:
        """Return prompt context that points the build agent at the persisted plan artifact."""
        if self.interaction_mode != tui_constants.BUILD_INTERACTION_MODE or not getattr(self, "_latest_plan", None):
            return None
        artifact_path = self._ensure_current_plan_artifact()
        if artifact_path is None:
            return None
        return PromptExtension(
            id="cli-current-plan-artifact",
            title="Current Plan Artifact",
            markdown=build_current_plan_artifact_prompt(artifact_path),
            modes=[AgentMode.CLI],
            # Read-only continuity context is safe and useful for delegated agents,
            # especially if their own histories compact while implementing the plan.
            propagate_to_sub_agents=True,
        )

    async def _build_agent(
        self,
        config: AgentConfig,
        rebuild: bool = False,
        *,
        restore_transcript: bool = True,
    ) -> None:
        history = self.session.history
        compaction = self.session.compaction
        if rebuild:
            self._clear_queued_messages()
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
            plan_artifact_extension = self._current_plan_artifact_prompt_extension()
            if plan_artifact_extension is not None:
                prompt_extensions.append(plan_artifact_extension)
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

        mcp_config = getattr(config, "mcp_config", None)
        if mcp_config is not None:
            for diagnostic in getattr(mcp_config, "diagnostics", []) or []:
                self._log_status(f"mcp: {diagnostic}", level="warn")
            mcp_extension = build_mcp_tool_extension(
                self.project_path,
                self.settings_store.root,
                project_trusted=self.settings.is_mcp_project_trusted(self.project_path),
                loaded_config=mcp_config,
            )
            if mcp_extension is not None:
                tool_extensions.append(mcp_extension)

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
        self._ensure_startup_entry()
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
