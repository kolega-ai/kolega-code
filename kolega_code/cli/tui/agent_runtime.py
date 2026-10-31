"""Agent construction, turn execution, and event routing for the CLI TUI."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, cast

from kolega_code.agent import (
    AgentConfig,
    AgentEvent,
    CoderAgent,
    KnownEventType,
    PlanningAgent,
    PromptExtension,
    ToolExtension,
)
from kolega_code.agent.baseagent import BaseAgent, QueuedUserInput
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.agent.tool_definitions import tool_description_asset
from kolega_code.agent.volatile_context import VolatileSection
from kolega_code.extensions import (
    KolegaExtensionHost,
    KolegaExtensionLoadError,
    bind_extension_agent,
    cleanup_extension_bundle,
    create_extension_bundle,
)
from kolega_code.agent.custom_agents import discover_custom_agents, validate_custom_agent_models
from kolega_code.agent.prompts import (
    build_current_plan_artifact_prompt,
    build_scratchpad_prompt,
    build_worktree_control_prompt,
)
from kolega_code.hooks import HookDispatcher, HookEvent, load_hook_config, project_hooks_present
from kolega_code.scratchpad import SCRATCHPAD_PROMPT_EXTENSION_ID, ensure_scratchpad_dir, scratchpad_dir_for
from kolega_code.llm.exceptions import LLMError, llm_error_message
from kolega_code.mcp.config import LoadedMCPConfig, server_fingerprint
from kolega_code.mcp.service import MCPService, mcp_tool_name_adjustment_note
from kolega_code.mcp.tools import build_mcp_tool_extension
from kolega_code.permissions import PermissionDecision
from kolega_code.session.inbox import (
    PeerMessage,
    PeerMessageError,
    validate_peer_text as validate_peer_message_text,
)
from kolega_code.session.runtime import deserialize_permission_request
from kolega_code.session.background import BackgroundAgentManager
from kolega_code.tools import ToolError
from textual.widgets import Static

from .. import messages
from ..browser_backend import build_browser_manager
from ..config import CliConfigError, build_agent_config, config_summary
from ..goal import (
    GoalState,
    build_goal_control_prompt_extension_markdown,
    build_goal_nudge,
    build_goal_prompt_extension_markdown,
    now_iso,
)
from ..plan_artifacts import write_current_plan_artifact
from ..skills import (
    SkillCatalog,
    build_skill_prompt_extension,
    build_skill_tool_extension,
    context_window_tokens_for_skill_budget,
    discover_skills,
)
from . import app_base as tui_app_base
from . import constants as tui_constants
from . import state as tui_state
from . import widgets as tui_widgets


class _TurnOutcome(StrEnum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True)
class _TurnResult:
    outcome: _TurnOutcome
    error: str = ""


#: Event types carried on the stream purely so it is a complete, replayable
#: record of the session. The TUI renders this content from the agent's
#: generator instead, so it must not also render these or output would double.
_RECORDING_ONLY_EVENTS = frozenset(
    {
        KnownEventType.ASSISTANT_DELTA,
        KnownEventType.THINKING_DELTA,
        KnownEventType.TURN_STARTED,
        KnownEventType.TURN_ENDED,
        KnownEventType.STREAM_TRUNCATED,
    }
)

#: ...except when they belong to a sub-agent. Only the main agent's generator is
#: consumed here; a sub-agent's runs inside AgentTool, so these events are the
#: TUI's only view of what delegated work said.
_SUB_AGENT_STREAM_EVENTS = frozenset(
    {
        KnownEventType.ASSISTANT_DELTA,
        KnownEventType.THINKING_DELTA,
    }
)


class AgentRuntimeMixin(tui_app_base.KolegaAppBase):
    def _agent_execution_draining(self) -> bool:
        return bool(getattr(self, "_draining_agent_tasks", ()))

    def _track_agent_execution(self) -> None:
        task = asyncio.current_task()
        if task is not None and getattr(self, "_agent_execution_task", None) is not task:
            self._agent_execution_task = task
            task.add_done_callback(self._agent_execution_finished)

    def _agent_execution_finished(self, task: asyncio.Task) -> None:
        if not task.cancelled():
            task.exception()
        if getattr(self, "_agent_execution_task", None) is task:
            self._agent_execution_task = None
        draining = getattr(self, "_draining_agent_tasks", None)
        if draining is None or task not in draining:
            return
        draining.discard(task)
        if not self._exit and getattr(self, "_session_shutdown_task", None) is None:
            self._refresh_shortcut_context()
            if not draining and self._pending_workspace_switch is not None and self.agent_worker is None:
                operation: asyncio.Task[None]

                async def wait_for_recovery() -> None:
                    await asyncio.sleep(0)
                    await asyncio.shield(operation)

                owner = self.run_worker(
                    wait_for_recovery,
                    name="kolega-drained-workspace-switch",
                    group="turns",
                    exclusive=True,
                    exit_on_error=False,
                )
                self.agent_worker = owner
                # Stop can cancel the Worker before its coroutine starts. The
                # committed switch must survive and remain tracked regardless.
                operation = asyncio.create_task(self._finish_drained_workspace_switch(owner))
                self._agent_execution_task = operation
                operation.add_done_callback(self._agent_execution_finished)
            else:
                self._schedule_pending_inputs()

    async def _finish_drained_workspace_switch(self, owner: object) -> None:
        try:
            await self._apply_pending_workspace_switch()
        except Exception as exc:
            self._notify_user(f"Could not complete the workspace switch after Stop: {exc}", severity="error")
        finally:
            if self.agent_worker is owner:
                self.agent_worker = None
                if self._stop_pending:
                    self._force_settle_stop()
                self._refresh_shortcut_context()
                self._schedule_pending_inputs()

    def _handle_control_request(self, event: AgentEvent) -> None:
        """Show a prompt the agent raised on the control channel.

        Requests arrive as ordinary events on the same stream everything else
        does, which is why this client needs no privileged callback and why a
        replay can show that the agent stopped to ask.
        """
        request_id = str(event.content.get("request_id") or "")
        kind = str(event.content.get("kind") or "")
        payload = event.content.get("payload")
        if not request_id or not isinstance(payload, dict):
            return
        if not self._control_request_is_open(request_id):
            # The announcement is queued, so it can arrive after the request has
            # already been settled — by a fast answer, a timeout, or control
            # being released. Showing a prompt nobody is waiting on would leave
            # the user answering into the void.
            return
        if kind == "permission":
            if not isinstance(payload.get("request"), dict):
                return
            try:
                request = deserialize_permission_request(payload["request"])
            except Exception:
                # A request this client cannot render must not wedge the turn;
                # leave it to the channel's default rather than show a broken
                # prompt.
                return
            self._begin_approval(request_id, request)
            return
        if kind == "question":
            options = payload.get("options")
            question = str(payload.get("question") or "")
            if not question or not isinstance(options, list) or len(options) < 2:
                return
            descriptions = payload.get("descriptions")
            self._begin_question(
                request_id,
                question,
                [str(option) for option in options],
                [str(item) for item in descriptions] if isinstance(descriptions, list) else None,
            )

    def _control_request_is_open(self, request_id: str) -> bool:
        """Whether the channel is still waiting on this request.

        Only a client sharing a process with the channel can check this. A remote
        client relies on the ``control_resolved`` announcement instead, which is
        why that event clears a prompt as well.
        """
        try:
            return any(request.request_id == request_id for request in self.control_channel.pending())
        except Exception:
            return True

    def _answer_permission_request(self, request_id: str, decision: PermissionDecision) -> None:
        """Send a decision back over the control channel. Never raises."""
        try:
            self.session_runtime.answer_permission(
                request_id,
                decision,
                client_id=tui_constants.TUI_CLIENT_ID,
            )
        except Exception:
            pass

    def _answer_question_request(self, request_id: str, answer: str) -> None:
        """Send a planning answer back over the control channel. Never raises.

        An empty answer is a non-answer: the tool treats it as unanswered and
        tells the model to proceed without it rather than acting on a blank.
        """
        try:
            self.session_runtime.answer_question(
                request_id,
                {"answer": answer},
                client_id=tui_constants.TUI_CLIENT_ID,
            )
        except Exception:
            pass

    async def _prime_recording_offset(self) -> None:
        """Continue the replay clock from an existing recording, once per session.

        Without this a resumed session would restart at zero, so a replay would
        show the second sitting overlapping the first.
        """
        if self._recording_primed:
            return
        self._recording_primed = True
        try:
            await self.recording_connection_manager.prime_elapsed_offset()
        except Exception:
            pass

    async def _record_turn_checkpoint(self, label: str) -> None:
        """Capture the rewind boundary for the turn about to start.

        Runs before the agent stream begins, so the checkpoint strictly
        precedes the journal's turn.started event. A checkpoint is an
        enhancement; it must never block or fail a turn.
        """
        tracker = self._session_diff_tracker
        if tracker is None:
            return
        excerpt = checkpoint_excerpt(label)

        def _capture() -> None:
            with self._session_diff_lock:
                if not tracker.checkpoints():
                    # The startup baseline worker lost the race; skip rather
                    # than fabricate checkpoint 0 out of a mid-session state.
                    return
                tracker.capture_checkpoint(excerpt)

        try:
            started = time.monotonic()
            await asyncio.to_thread(_capture)
            if self.show_logs:
                self._write_log(f"Checkpoint captured in {time.monotonic() - started:.2f}s", "debug")
        except Exception:
            pass

    def _agent_turn_stream(self, message: str, attachments: list[dict] | None = None):
        assert self.agent is not None
        if attachments:
            return self.agent.process_message_stream(message, attachments)
        return self.agent.process_message_stream(message)

    async def _consume_agent_stream(self, stream) -> None:
        owner = self.agent_worker
        async for chunk in stream:
            if self.agent_worker is not owner:
                continue
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

    async def _finalize_turn_output(self, *, cancel_pending: bool) -> None:
        """Settle transcript state common to every turn outcome, before the final status."""
        owner = self.agent_worker
        if cancel_pending:
            self._cancel_pending_question()
            self._cancel_pending_approval()
        await self._drain_pending_events()
        if self.agent_worker is not owner:
            return
        self._start_session_diff_refresh()
        self._finalize_sub_agent_activities()
        self._finalize_workflow_activities()

    async def _run_turn_stream(
        self, stream_factory, *, checkpoint_label: str | None = None, preserve_inspector: bool = False
    ) -> _TurnResult:
        """Run one agent stream and return its terminal outcome.

        When ``checkpoint_label`` is given, the session-diff checkpoint is
        captured after progress begins, so the status strip flips off the
        previous turn's result immediately and Esc during a slow capture
        resolves through the normal cancelled path.
        """
        if self._agent_execution_draining():
            return _TurnResult(_TurnOutcome.CANCELLED)
        owner = self.agent_worker
        result = _TurnResult(_TurnOutcome.COMPLETED)
        self._begin_turn_progress(
            messages.PREPARING_CHECKPOINT if checkpoint_label is not None else messages.WORKING,
            preserve_inspector=preserve_inspector,
        )
        try:
            await self._stream_turn_body(owner, stream_factory, checkpoint_label)
        except asyncio.CancelledError:
            result = _TurnResult(_TurnOutcome.CANCELLED)
            await self._settle_turn(owner, messages.STOPPED_BY_USER, tui_state.TurnState.STOPPED, exiting=True)
        except LLMError as exc:
            model = self.config.long_context_config.model if self.config is not None else None
            error = llm_error_message(exc, model=model)
            result = _TurnResult(_TurnOutcome.FAILED, error)
            await self._settle_turn(owner, error, tui_state.TurnState.ERROR)
        except Exception as exc:
            await self._settle_turn(owner, messages.STOPPED_WITH_ERROR.format(error=exc), tui_state.TurnState.ERROR)
            raise
        finally:
            self._teardown_turn(owner, cancelled_by_user=result.outcome is _TurnOutcome.CANCELLED)
        return result

    async def _stream_turn_body(self, owner: object | None, stream_factory, checkpoint_label: str | None) -> None:
        """Drive a turn to completion, stopping early once another worker owns the slot."""
        if checkpoint_label is not None:
            await self._record_turn_checkpoint(checkpoint_label)
            if self.agent_worker is not owner:
                return
            self._update_progress(messages.WORKING, complete=False, state=tui_state.TurnState.GENERATING)
        self._log_status(messages.GENERATING, "ok")
        await self._consume_agent_stream(stream_factory())
        if self.agent_worker is not owner:
            return
        await self._finalize_turn_output(cancel_pending=False)
        if self.agent_worker is not owner:
            return
        self._finish_turn_progress(messages.FINISHED, tui_state.TurnState.IDLE)
        # Bounded: a journal write wedged on the session lock used to park the
        # turn here forever. The journal is canonical, so a skipped projection
        # refresh is recoverable; an invisible plan is not.
        await self._save_session_with_deadline(self._save_session_history_async())
        if self.agent_worker is not owner:
            return
        await self._capture_completed_plan()
        if self.agent_worker is not owner:
            return
        self._log_status(messages.FINISHED, "ok")
        self._maybe_trigger_session_metadata()

    async def _settle_turn(
        self, owner: object | None, message: str, state: tui_state.TurnState, *, exiting: bool = False
    ) -> None:
        if self.agent_worker is not owner:
            return
        self._background_paused = True
        # A quit already settled open prompts, saved the session and is tearing down the tree.
        if exiting and self._exit:
            return
        await self._finalize_turn_output(cancel_pending=True)
        if self.agent_worker is not owner:
            return
        self._finish_turn_progress(message, state)
        await self._save_session_history_async()
        if self.agent_worker is owner:
            self._log_status(message, "warn" if state is tui_state.TurnState.STOPPED else "error")

    def _teardown_turn(self, owner: object | None, *, cancelled_by_user: bool) -> None:
        # Teardown can yield while an exclusive replacement reserves the slot.
        if self.agent_worker is not owner:
            return
        self._active_progress_entry = None
        self._turn_active = False
        # The worker honored (or outlived) the stop, so no escalation is needed.
        self._stop_pending = False
        self._disarm_stop_watchdog()
        if self._exit:
            return
        self._flush_terminal_output()
        self._flush_log_output()
        self._flush_conversation_render()
        if self._plan_decision_active:
            self._set_composer_status(messages.PLAN_READY_PLACEHOLDER)
        else:
            self._restore_composer_placeholder()
        self._set_chat_enabled(self.agent is not None and not self._plan_decision_active)
        if cancelled_by_user:
            self._schedule_primary_focus_restore()

    def _maybe_trigger_session_metadata(self) -> None:
        """Trigger background session title and description generation if turn meets depth."""
        if self._exit or self.agent is None or not self.agent.history:
            return
        if len(self.agent.history) < 2:
            return
        self.run_worker(
            self._update_session_metadata_worker(),
            name="kolega-session-metadata",
            group="metadata",
            exclusive=True,
        )

    async def _update_session_metadata_worker(self, *, force_notify: bool = False) -> None:
        """Generate descriptive session title and description in the background."""
        if self.agent is None or not getattr(self.agent, "llm", None):
            if force_notify:
                self._notify_user("Cannot generate metadata: agent not ready.", severity="warning")
            return
        messages = list(self.agent.history)
        if len(messages) < 2:
            if force_notify:
                self._notify_user("Not enough conversation history to generate title.", severity="warning")
            return

        from kolega_code.agent.session_metadata import generate_session_metadata
        from ..main import known_secret_values

        secret_values: list[str] = []
        try:
            secret_values = known_secret_values(self.settings, self.settings_store, project_path=self.project_path)
        except Exception:
            pass

        model_cfg = getattr(self.agent, "primary_model_config", None)
        model = model_cfg.model if model_cfg is not None else "fast"
        metadata = await generate_session_metadata(
            messages,
            llm=self.agent.llm,
            model=model,
            secret_values=secret_values,
        )
        if metadata is None:
            if force_notify:
                self._notify_user("Could not generate session title.", severity="warning")
            return

        changed = False
        if metadata.title and self.session.title != metadata.title:
            self.session.title = metadata.title
            changed = True
        if metadata.description and self.session.description != metadata.description:
            self.session.description = metadata.description
            changed = True

        if changed:
            await self._save_session_async()
            self._update_mode_chrome()

        if force_notify:
            self._add_conversation_entry(
                tui_state.ConversationEntry(
                    kind="system",
                    content=f"Session retitled: **{self.session.title}**"
                    + (f"\n{self.session.description}" if self.session.description else ""),
                )
            )

    async def _process_message(
        self, message: str, attachments: list[dict] | None = None, *, turn_label: str | None = None
    ) -> bool:
        """Run one turn. Returns whether the user cancelled it.

        ``_run_turn_stream`` absorbs ``CancelledError`` so the transcript can be
        finalized, which means callers that need to react to a cancel — the
        scheduled-loop driver, for one — have to learn about it from this return
        value rather than from an exception.
        """
        # run_async uses eager tasks on Python 3.12+: let the caller publish
        # the Worker before capturing ownership (run_test doesn't do this).
        await asyncio.sleep(0)
        owner = self.agent_worker
        try:
            if self._agent_execution_draining():
                self._queue_user_message(message, attachments)
                return False
            self._track_agent_execution()
            if self.agent is None:
                return False
            if self._background_retirement is not None:
                await self._attach_background_jobs()
                if self.agent_worker is not owner:
                    return False
            self._background_paused = False
            self._background_delivery_failed = False
            self._schedule_background_status_refresh()
            if self._background_jobs is not None:
                await self._background_jobs.deliver(self.agent)
                if self.agent_worker is not owner:
                    return False
            turn_result = await self._run_turn_stream(
                lambda: self._agent_turn_stream(message, attachments),
                checkpoint_label=turn_label or message,
            )
            if self.agent_worker is not owner:
                return turn_result.outcome is _TurnOutcome.CANCELLED
            # A committed workspace switch applies between turns: the agent is
            # rebuilt in the new checkout, then handed one continuation turn so
            # the work that prompted the switch resumes there. A cancelled turn
            # still applies the committed rebuild but stays quiet afterwards.
            while True:
                continuation = await self._apply_pending_workspace_switch()
                if self.agent_worker is not owner:
                    return turn_result.outcome is _TurnOutcome.CANCELLED
                if turn_result.outcome is not _TurnOutcome.COMPLETED or continuation is None:
                    break
                turn_result = await self._run_turn_stream(
                    lambda prompt=continuation: self._agent_turn_stream(prompt),
                    checkpoint_label=continuation,
                )
            if turn_result.outcome is _TurnOutcome.CANCELLED:
                self._schedule_maybe_start_queued_message()
                return True
            if turn_result.outcome is _TurnOutcome.FAILED:
                if self._goal is not None and not self._goal.met:
                    await self._pause_goal_after_failed_turn(turn_result.error)
                self._schedule_maybe_start_queued_message()
                return False
            # Un-pause a paused goal on any user-initiated turn so the loop resumes.
            if self._goal is not None and self._goal.paused and not self._goal.met:
                self._goal.paused = False
                self._goal.status_note = ""
                self._sync_goal_to_session()
            if self._goal is not None and self._goal.is_active:
                await self._run_goal_loop()
            else:
                self._schedule_maybe_start_queued_message()
            return False
        finally:
            # An unexpected stream error must not strand a committed switch:
            # the journal already names the new workspace, so apply the rebuild
            # even though no continuation turn will run.
            if (
                self.agent_worker is owner
                and self._pending_workspace_switch is not None
                and not self._agent_execution_draining()
            ):
                await self._apply_pending_workspace_switch()
            # The turn worker owns ``agent_worker`` for its whole lifetime, including
            # the goal loop's verifier phase. Clearing it only here keeps the app
            # "busy" during evaluation: submissions queue instead of cancelling the
            # verifier via the exclusive ``turns`` group, and Esc can interrupt it.
            if self.agent_worker is owner:
                self.agent_worker = None
                # A stop can land outside ``_run_turn_stream``'s finalizer, leaving the
                # dashboard painted "Stopping…" with a live turn. Settle it here.
                if self._stop_pending and self._turn_started_at is not None:
                    self._force_settle_stop()
                self._refresh_shortcut_context()
                self._schedule_maybe_start_queued_message()

    # ------------------------------------------------------------------
    # Autonomous goal loop
    # ------------------------------------------------------------------

    def _worktree_control_prompt_extension(self) -> PromptExtension:
        return PromptExtension(
            id="cli-worktree-control",
            title="Active worktree",
            markdown=build_worktree_control_prompt(
                plan_mode=self.interaction_mode == tui_constants.PLAN_INTERACTION_MODE
            ),
            modes=[AgentMode.CLI],
            propagate_to_sub_agents=False,
        )

    def _worktree_control_tool_extension(self) -> ToolExtension:
        async def switch_worktree(path: str) -> str:
            clean_path = path.strip()
            if not clean_path:
                raise ToolError("'path' must identify a registered worktree.")
            return await self._switch_active_worktree(clean_path)

        return ToolExtension(
            name="cli-worktree-control",
            tools={"switch_worktree": switch_worktree},
            tool_descriptions={"switch_worktree": tool_description_asset("switch_worktree")},
            tool_schemas={
                "switch_worktree": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": (
                                "Registered worktree path (or nested path), resolved relative to the "
                                "immutable session launch checkout."
                            ),
                        }
                    },
                    "required": ["path"],
                }
            },
            # Deliberately not in ``planning_tools``: that group bypasses the
            # planning agent's read_only filter, and switching the workspace is
            # the most mutating thing plan mode could do. Plan mode's registry
            # therefore never contains this tool.
            tool_groups={"cli_worktree_tools": ["switch_worktree"]},
            propagate_to_sub_agents=False,
            exclusive_tools=frozenset({"switch_worktree"}),
        )

    def _goal_prompt_extension(self) -> PromptExtension:
        condition = self._goal.condition if self._goal else ""
        return PromptExtension(
            id="cli-active-goal",
            title="Active goal",
            markdown=build_goal_prompt_extension_markdown(condition),
            modes=None,
            # Read-only continuity context; safe and useful for delegated agents.
            propagate_to_sub_agents=True,
        )

    def _goal_control_prompt_extension(self) -> PromptExtension:
        return PromptExtension(
            id="cli-goal-control",
            title="Setting autonomous goals",
            markdown=build_goal_control_prompt_extension_markdown(
                plan_mode=self.interaction_mode == tui_constants.PLAN_INTERACTION_MODE
            ),
            modes=[AgentMode.CLI],
            # Only the top-level TUI agent owns goal and session state.
            propagate_to_sub_agents=False,
        )

    def _goal_control_tool_extension(self) -> ToolExtension:
        async def set_goal(condition: str) -> str:
            clean_condition = condition.strip()
            if not clean_condition:
                raise ToolError("'condition' must be a non-empty goal condition.")
            if self._scheduled_loop is not None and self._scheduled_loop.is_active:
                # A scheduled loop and a goal both drive the exclusive turn
                # worker; running them together makes iteration ownership
                # ambiguous, so the user has to pick one.
                return messages.GOAL_BLOCK_LOOP_ACTIVE

            replacing = await self._activate_goal(clean_condition)
            confirmation = messages.GOAL_REPLACED if replacing else messages.GOAL_SET
            self._add_conversation_entry(tui_state.ConversationEntry(kind="system", content=confirmation))
            return confirmation

        return ToolExtension(
            name="cli-goal-control",
            tools={"set_goal": set_goal},
            tool_descriptions={"set_goal": tool_description_asset("set_goal")},
            tool_schemas={
                "set_goal": {
                    "type": "object",
                    "properties": {
                        "condition": {
                            "type": "string",
                            "description": "The verifiable completion condition for the autonomous goal.",
                        }
                    },
                    "required": ["condition"],
                }
            },
            # Deliberately not in ``planning_tools``: that group bypasses the
            # planning agent's read_only filter. A planning turn must not start
            # an autonomous loop on its own; the user's `/goal` command works in
            # either mode, and a goal set in build mode keeps driving plan turns.
            tool_groups={"cli_goal_tools": ["set_goal"]},
            # Goal/session state belongs to the top-level TUI only.
            propagate_to_sub_agents=False,
        )

    def _peer_messaging_tool_extension(self) -> ToolExtension:
        async def _discovered_peers():
            from kolega_code.cli.peer_messaging import store_title_lookup
            from kolega_code.session.inbox import discover_peers, messaging_dir

            return await discover_peers(
                self.inbox_registry,
                messaging_dir(self.store.root),
                exclude_session_id=self.session.session_id,
                title_lookup=store_title_lookup(self.store),
            )

        async def list_agents() -> str:
            from kolega_code.cli.peer_messaging import format_peer_table

            return format_peer_table(await _discovered_peers())

        async def send_message(recipient: str, text: str) -> str:
            from kolega_code.session.inbox import (
                deliver_to_discovered_peer,
                resolve_summary,
            )

            clean_recipient = str(recipient or "").strip()
            if not clean_recipient:
                raise ToolError("'recipient' must name a live session (see list_agents).")
            try:
                cleaned_text = validate_peer_message_text(str(text or ""))
            except PeerMessageError as exc:
                raise ToolError(str(exc)) from exc

            message = PeerMessage.create(
                sender_session_id=self.session.session_id,
                sender_title=self.session.name,
                text=cleaned_text,
            )

            # In-process first; the socket directory answers for everyone else.
            try:
                registration = self.inbox_registry.resolve(
                    clean_recipient,
                    exclude_session_id=self.session.session_id,
                )
            except PeerMessageError:
                registration = None

            try:
                if registration is not None:
                    outcome = await self.inbox_registry.deliver(
                        registration.session_id,
                        message,
                        sender_session_id=self.session.session_id,
                    )
                    addressed = f"{registration.describe_title()} ({registration.session_id[:8]})"
                else:
                    peers = await _discovered_peers()
                    summary = resolve_summary(peers, clean_recipient)
                    _, outcome = await deliver_to_discovered_peer(
                        summary,
                        message,
                        sender_project=str(self.active_project_path),
                    )
                    addressed = f"{summary.title} ({summary.session_id[:8]})"
            except PeerMessageError as exc:
                # A delivery failure must reach the model as an error, never as a
                # plausible-looking success.
                raise ToolError(str(exc)) from exc

            return f"Message delivered to {addressed}."

        return ToolExtension(
            name="cli-peer-messaging",
            tools={"list_agents": list_agents, "send_message": send_message},
            tool_descriptions={
                "list_agents": tool_description_asset("list_agents"),
                "send_message": tool_description_asset("send_message"),
            },
            tool_schemas={
                "list_agents": {"type": "object", "properties": {}, "required": []},
                "send_message": {
                    "type": "object",
                    "properties": {
                        "recipient": {
                            "type": "string",
                            "description": (
                                "Peer session to message: exact name, unique name prefix, or session id "
                                "(from list_agents)."
                            ),
                        },
                        "text": {
                            "type": "string",
                            "description": "Plain-text message body. Context only — the recipient's user stays in control.",
                        },
                    },
                    "required": ["recipient", "text"],
                },
            },
            # Not in ``planning_tools``: like goal/worktree control, the planning
            # agent's read-only registry filters these out, so messaging is a
            # build-mode capability.
            tool_groups={"cli_peer_messaging_tools": ["list_agents", "send_message"]},
            # Session identity and the process-wide inbox belong to the top-level
            # TUI only; delegated agents never message peers.
            propagate_to_sub_agents=False,
            # Unlike set_goal, a send may batch with other calls: it does not
            # take over the turn.
        )

    def _sync_goal_to_session(self) -> None:
        """Mirror the live goal state into the session record (in-memory only)."""
        self.session.goal = self._goal.to_dict() if self._goal is not None else {}

    async def _persist_goal_async(self) -> None:
        self._sync_goal_to_session()
        await self._save_session_async()

    def _set_goal_state(self, goal: GoalState) -> None:
        self._goal = goal
        # A goal counts tokens from the moment it is set, not from process start.
        self._goal_usage_mark = self._usage_ledger.snapshot()
        if self.agent is not None:
            self.agent.apply_goal(goal.condition, self._goal_prompt_extension())
        self._refresh_status_dashboard()

    async def _activate_goal(self, condition: str, *, run_to_completion: bool = False) -> bool:
        """Create and persist a fresh goal, returning whether it replaced an unmet goal."""
        clean_condition = condition.strip()
        if not clean_condition:
            raise ValueError("Goal condition must not be empty.")

        replacing = bool(self._goal is not None and self._goal.condition and not self._goal.met)
        self._set_goal_state(GoalState.create(clean_condition, run_to_completion=run_to_completion))
        await self._persist_goal_async()
        return replacing

    async def _clear_active_goal(self, *, note: str | None = None) -> None:
        self._goal = None
        if self.agent is not None:
            self.agent.apply_goal(None)
        self._refresh_status_dashboard()
        await self._persist_goal_async()
        if note:
            self._add_conversation_entry(tui_state.ConversationEntry(kind="system", content=note))

    def _drain_goal_tokens(self) -> None:
        """Fold all LLM usage since the last drain into the active goal.

        The ledger covers every nested caller (verifier sub-agents, compression,
        web-fetch, hooks), so a drain at each goal boundary counts
        the whole command tree — including turns run while the goal was active.
        """
        current = self._usage_ledger.snapshot()
        if self._goal is not None:
            delta = current.since(self._goal_usage_mark)
            if delta.total_tokens > 0:
                self._goal.tokens_spent += delta.total_tokens
        self._goal_usage_mark = current

    async def _evaluate_active_goal(self):
        """Dispatch the read-only verifier and update goal bookkeeping. Returns the verdict."""
        assert self.agent is not None
        assert self._goal is not None
        owner = self.agent_worker
        self._set_status_activity(messages.GOAL_EVALUATING)
        self._refresh_status_dashboard()
        verdict = await self.agent.evaluate_goal_condition(self._goal.condition)
        if self.agent_worker is not owner:
            return verdict
        self._goal.last_reason = verdict.reason
        self._goal.last_evaluated_at = now_iso()
        return verdict

    async def _run_goal_loop(self) -> None:
        """Evaluate → continue loop, run after a successful (non-cancelled) turn."""
        owner = self.agent_worker
        try:
            while self._goal is not None and self._goal.is_active:
                self._drain_goal_tokens()
                verdict = await self._evaluate_active_goal()
                if self.agent_worker is not owner:
                    return
                if verdict.met:
                    await self._complete_goal(verdict)
                    break
                self._goal.turns_evaluated += 1
                self._sync_goal_to_session()
                if self._goal.turns_evaluated >= self._goal.max_turns:
                    await self._abort_goal()
                    break
                turns_remaining = self._goal.max_turns - self._goal.turns_evaluated
                nudge = build_goal_nudge(self._goal.condition, verdict, turns_remaining)
                self._add_conversation_entry(
                    tui_state.ConversationEntry(
                        kind="system",
                        content=messages.GOAL_NOT_MET_CONTINUE.format(
                            reason=verdict.reason or "see the verifier's assessment"
                        ),
                    )
                )
                turn_result = await self._run_turn_stream(lambda: self._agent_turn_stream(nudge))
                if self.agent_worker is not owner:
                    return
                # The goal nudge is itself the continuation, so a switch
                # committed during this turn only needs the rebuild applied.
                await self._apply_pending_workspace_switch()
                if self.agent_worker is not owner:
                    return
                if turn_result.outcome is _TurnOutcome.CANCELLED:
                    await self._pause_goal(messages.GOAL_PAUSED.format(reason="Stopped by user. "))
                    break
                if turn_result.outcome is _TurnOutcome.FAILED:
                    await self._pause_goal_after_failed_turn(turn_result.error)
                    break
        except asyncio.CancelledError:
            if self.agent_worker is not owner:
                return
            # Cancelled between work turns (e.g. Esc while the verifier was running).
            # Pause so the goal survives and can be resumed, like the nudge-turn cancel.
            if self._goal is not None and not self._goal.met and not self._goal.paused:
                await self._pause_goal(messages.GOAL_PAUSED.format(reason="Stopped by user. "))
        self._schedule_maybe_start_queued_message()

    async def _complete_goal(self, verdict) -> None:
        if self._goal is None:
            return
        self._drain_goal_tokens()  # count the final verifier before persisting
        condition = self._goal.condition
        self._goal.met = True
        self._goal.status_note = ""
        self._refresh_status_dashboard()
        await self._persist_goal_async()
        self._add_conversation_entry(
            tui_state.ConversationEntry(kind="system", content=messages.GOAL_MET.format(condition=condition))
        )
        self._notify_user(messages.GOAL_MET.format(condition=condition))
        # Goal achieved: drop the active-goal prompt extension for future turns.
        if self.agent is not None:
            self.agent.apply_goal(None)

    async def _abort_goal(self) -> None:
        if self._goal is None:
            return
        self._drain_goal_tokens()
        turns = self._goal.turns_evaluated
        self._goal.paused = True
        self._goal.status_note = f"Reached the turn cap ({self._goal.max_turns})."
        self._refresh_status_dashboard()
        await self._persist_goal_async()
        self._add_conversation_entry(
            tui_state.ConversationEntry(
                kind="system", content=messages.GOAL_MAX_TURNS.format(turns=turns), tone="warning"
            )
        )
        self._notify_user(messages.GOAL_MAX_TURNS.format(turns=turns), severity="warning")
        if self.agent is not None:
            self.agent.apply_goal(None)

    async def _pause_goal_after_failed_turn(self, error: str) -> None:
        detail = error.strip() or "The turn failed."
        await self._pause_goal(
            messages.GOAL_PAUSED_AFTER_FAILED_TURN.format(error=detail),
            status_note=detail,
        )

    async def _pause_goal(self, note: str, *, status_note: str = "Stopped by user.") -> None:
        if self._goal is None:
            return
        self._drain_goal_tokens()  # count the cancelled nudge turn's spend
        self._goal.paused = True
        self._goal.status_note = status_note
        self._refresh_status_dashboard()
        await self._persist_goal_async()
        self._add_conversation_entry(tui_state.ConversationEntry(kind="system", content=note, tone="warning"))

    def _background_completion_delivered(self, completion: dict[str, Any]) -> None:
        self._add_conversation_entry(
            tui_state.ConversationEntry(
                kind="system",
                content=(
                    f"Background {completion['agent_type']} {completion['status']} "
                    f"({completion['job_id']}):\n\n{completion['result']}"
                ),
                origin={"kind": "subagent_completion", "job_id": completion["job_id"]},
            )
        )
        # A completed card can have no terminal UI event on cancellation.
        activity = self._sub_agent_activities.get(completion["job_id"])
        if activity is not None and activity.status == "running":
            activity.status = completion["status"]
            activity.finished_at = self._now()
            activity.entry.complete = True
            activity.entry.content = self._format_sub_agent_content(activity)
            self._invalidate_conversation(activity.entry)
        if not self._background_rebuilding:
            self._schedule_maybe_start_queued_message()
        self._schedule_background_status_refresh()

    async def _attach_background_jobs(self) -> None:
        if self._background_retirement is not None:
            # Retry the actual retained reports with their original delivery
            # policy, never recover a replacement manager from stale disk rows.
            await self._retire_background_jobs()
        agent = self.agent
        getter = getattr(agent, "get_background_jobs", None)
        if agent is None or not callable(getter):
            return
        jobs = self._background_jobs
        if jobs is None:
            jobs = cast(BackgroundAgentManager, getter())
            self._background_jobs = jobs
        else:
            agent.background_jobs = jobs
        agent.background_needs_turn = self._background_resume_needed or agent.background_needs_turn
        self._background_resume_needed = False
        self._background_delivery_failed = False

        def ready() -> None:
            if self._background_jobs is jobs and not self._exit:
                self._schedule_pending_inputs()

        def changed() -> None:
            if self._background_jobs is jobs and not self._exit:
                self._schedule_background_status_refresh()

        jobs.set_callbacks(on_ready=ready, on_delivered=self._background_completion_delivered, on_changed=changed)
        await jobs.recover()
        ready()
        changed()

    async def _process_background_completions(self, jobs: BackgroundAgentManager) -> None:
        await asyncio.sleep(0)  # Publish agent_worker before eager execution captures it.
        owner = self.agent_worker
        try:
            if self._agent_execution_draining():
                return
            self._track_agent_execution()
            agent = self.agent
            if (
                agent is None
                or jobs is not self._background_jobs
                or self._background_rebuilding
                or self._background_retirement is not None
            ):
                return
            await self._drain_pending_events()
            if self.agent_worker is not owner:
                return
            await jobs.deliver(agent)
            if self.agent_worker is not owner:
                return
            if agent.background_needs_turn and not self._background_paused:
                await self._run_turn_stream(lambda: agent.continue_from_history_stream(), preserve_inspector=True)
            else:
                await self._save_session_history_async()
            if self.agent_worker is not owner:
                return
            await self._cleanup_retired_agent_generations()
        except asyncio.CancelledError:
            if self.agent_worker is owner:
                self._background_paused = True
            raise
        except Exception as exc:
            if self.agent_worker is owner:
                self._background_paused = True
                self._background_delivery_failed = True
                self._schedule_background_status_refresh()
                self._log_status(f"Background completion delivery failed: {exc}", "error")
        finally:
            if self.agent_worker is owner:
                self.agent_worker = None
                if self._stop_pending and self._turn_started_at is not None:
                    self._force_settle_stop()
                self._refresh_shortcut_context()
                self._schedule_maybe_start_queued_message()

    async def _cleanup_retired_agent_generations(self) -> None:
        if self._background_retirement is not None:
            return
        if self._background_jobs is not None and self._background_jobs.has_work:
            return
        retired, self._retired_agent_generations = self._retired_agent_generations, []
        for agent, bundle in retired:
            try:
                await agent.cleanup()
            except Exception as exc:
                self._agent_cleanup_failed = True
                self._log_status(f"retired agent cleanup failed: {exc}", "warn")
            finally:
                if bundle is not None:
                    try:
                        await cleanup_extension_bundle(bundle)
                    except Exception as exc:
                        self._agent_cleanup_failed = True
                        self._log_status(f"retired extension cleanup failed: {exc}", "warn")

    async def _retire_background_jobs(self, *, agent: BaseAgent | None = None, archive: bool = False) -> None:
        """Settle this epoch before destructive context/workspace transitions."""
        if self._background_retirement is None:
            target = agent if agent is not None else self.agent
            jobs = self._background_jobs
            if jobs is not None:
                self._background_retirement = (jobs, target, archive)
        else:
            jobs, target, archive = self._background_retirement
        self._background_resume_needed = False
        previous = self._background_rebuilding
        self._background_rebuilding = True

        async def retire() -> None:
            try:
                if jobs is not None:
                    jobs.set_callbacks(on_ready=None, on_delivered=None)
                    await jobs.close()
                    if target is not None:
                        pending = {row["job_id"] for row in jobs.snapshot() if not row["delivered"]}
                        await jobs.deliver(target, archive=archive)
                        if archive:
                            for outcome in jobs.snapshot():
                                if outcome["job_id"] in pending:
                                    self._background_completion_delivered(outcome)
                        target.background_jobs = None
                        target.background_needs_turn = False
                    self._background_jobs = None
                    self._background_retirement = None
                    self._schedule_background_status_refresh()
                    for outcome in jobs.snapshot():
                        activity = self._sub_agent_activities.get(outcome["job_id"])
                        if activity is not None and activity.status == "running":
                            activity.status = outcome["status"]
                            activity.finished_at = self._now()
                            activity.entry.complete = True
                            activity.entry.content = self._format_sub_agent_content(activity)
                            self._invalidate_conversation(activity.entry)
            finally:
                if jobs is not None and self._background_jobs is jobs:
                    self._schedule_background_status_refresh()
                await self._cleanup_retired_agent_generations()

        # Shield the host finalization too, not just the manager's durable writes.
        # Otherwise cancellation can strand the current root on a closed manager
        # after its completion has already been consumed.
        operation = asyncio.create_task(retire())
        cancelled = False
        try:
            while not operation.done():
                try:
                    await asyncio.shield(operation)
                except asyncio.CancelledError:
                    cancelled = True
            operation.result()
            if cancelled:
                if target is not None and target is self.agent and not self._exit:
                    # The transition did not run. Keep this root usable without
                    # reviving the consumed completion's automatic continuation.
                    recovery = asyncio.create_task(self._attach_background_jobs())
                    while not recovery.done():
                        try:
                            await asyncio.shield(recovery)
                        except asyncio.CancelledError:
                            pass
                    recovery.result()
                raise asyncio.CancelledError
        finally:
            self._background_rebuilding = previous

    def _prepare_background_agent_inspector(self) -> None:
        """Make queued/recovered jobs inspectable before any child UI event arrives."""
        self._sync_background_agent_status()
        self._sync_background_inspector_activities()

    def _sync_background_inspector_activities(self) -> None:
        """Project retained job results only when the inspector needs them."""
        jobs = self._background_jobs
        if jobs is None:
            return
        for row in self._background_status_rows:
            if row["archived"]:
                continue
            full = jobs.snapshot(row["job_id"])[0]
            activity = self._ensure_sub_agent_activity(
                AgentEvent(
                    event_type="chat_message",
                    sender=row["agent_type"],
                    content={},
                    sub_agent_info={
                        "agent_id": row["job_id"],
                        "agent_name": row["agent_type"],
                        "task": row["label"],
                        "task_full": full["task"],
                        "background": True,
                    },
                )
            )
            activity.status = row["status"]
            activity.current_action = "Queued" if row["status"] == "queued" else activity.current_action
            if row["status"] not in {"queued", "running"}:
                if activity.finished_at is None:
                    activity.finished_at = self._now()
                activity.entry.complete = True
                activity.current_action = ""
                activity.last_activity = full["result"][:240]
                result_key = f"background-result:{row['completion_id']}"
                if result_key not in activity.stream_steps:
                    result = tui_state.ConversationEntry(kind="assistant", content=full["result"])
                    activity.stream_steps[result_key] = result
                    activity.steps.append(result)
            elif activity.current_action == "Queued":
                activity.current_action = "" if row["status"] == "running" else "Queued"
            activity.entry.content = self._format_sub_agent_content(activity)
            self._invalidate_conversation(activity.entry)
            self._invalidate_sub_agent_detail(activity)

    def _queue_user_message(
        self,
        text: str,
        attachments: list[dict] | None = None,
        *,
        origin: dict | None = None,
        model_text: str | None = None,
    ) -> None:
        """Queue one follow-up message for delivery between turns or at a tool boundary.

        ``origin`` marks inputs that did not come from the local user (a peer
        session's message): they render as ``peer_message`` entries instead of
        user entries and are never restored into the composer. ``model_text``
        lets the caller send framing the model needs (the provenance preamble)
        while the transcript keeps the raw text.
        """
        self._queued_message_seq += 1
        entry = tui_state.ConversationEntry(kind="queued", content=text, tone=None)
        entry.origin = origin
        queued = tui_state.QueuedMessage(
            queue_id=f"queued-{self._queued_message_seq}",
            text=model_text if model_text is not None else text,
            attachments=[dict(item) for item in attachments] if attachments else None,
            entry=entry,
            display_text=text,
            origin=origin,
        )
        # Keep the future transcript entry unmounted while pending; the queue
        # widget is the only waiting-message preview until this item starts.
        self._queued_messages.append(queued)
        self._refresh_queued_messages_panel()
        if queued.is_peer:
            sender = str((origin or {}).get("title") or "peer")
            self._log_status(messages.PEER_MESSAGE_QUEUED.format(sender=sender), "info")
        else:
            self._log_status(messages.QUEUED_MESSAGE, "info")

    def _queued_messages_preview(self) -> str:
        if not self._queued_messages:
            return messages.QUEUE_EMPTY
        lines = [f"{messages.QUEUE_LIST_TITLE} {len(self._queued_messages)}"]
        for index, queued in enumerate(self._queued_messages, start=1):
            preview = " ".join(queued.display_text.strip().split()) or "(empty)"
            if len(preview) > 120:
                preview = preview[:117] + "…"
            suffix = ""
            if queued.is_peer:
                sender = str((queued.origin or {}).get("title") or "peer")
                suffix = f" ({messages.PEER_QUEUE_FROM.format(sender=sender)})"
            elif queued.attachments:
                suffix = f" ({len(queued.attachments)} attachment(s))"
            lines.append(f"{index}. {preview}{suffix}")
        return "\n".join(lines)

    def _refresh_queued_messages_panel(self) -> None:
        try:
            panel = self.query_one("#queued_messages", Static)
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

        # Peer messages carry model-facing framing (the provenance preamble) and
        # were never typed here; restoring their wrapped text into the composer
        # would paste machinery the user did not write. Drop them with a notice.
        peers = [item for item in queued if item.is_peer]
        own = [item for item in queued if not item.is_peer]
        if peers:
            self._log_status(messages.PEER_MESSAGES_DROPPED_ON_RESTORE.format(count=len(peers)), "info")

        if not own:
            self._queued_messages.clear()
            self._remove_queued_entries_from_transcript(queued)
            self._refresh_queued_messages_panel()
            return 0

        queued_text = "\n\n".join(item.display_text for item in own)
        try:
            composer = self.query_one("#composer", tui_widgets.ChatComposer)
            existing_text = getattr(composer, "text", "") or ""
            restored_text = f"{queued_text}\n\n{existing_text}" if existing_text else queued_text
            composer.load_text(restored_text)
        except Exception:
            pass

        self._queued_messages.clear()
        self._remove_queued_entries_from_transcript(queued)
        self._refresh_queued_messages_panel()
        return len(own)

    def _schedule_maybe_start_queued_message(self) -> None:
        try:
            self.set_timer(0.01, self._maybe_start_queued_message, name="queued-message-drain")
        except Exception:
            self._maybe_start_queued_message()

    def _schedule_pending_inputs(self) -> None:
        """Rearm on owner release/unblock only when there is actual work."""
        if (
            self._background_rebuilding
            or self._background_retirement is not None
            or getattr(self, "_session_shutdown_task", None) is not None
        ):
            return
        jobs = self._background_jobs
        if self._queued_messages or (
            jobs is not None
            and (
                jobs.pending_count
                or (getattr(self.agent, "background_needs_turn", False) and not self._background_paused)
            )
        ):
            self._schedule_maybe_start_queued_message()

    def _maybe_start_queued_message(self) -> bool:
        settings_screen = getattr(self, "_settings_screen", None)
        if settings_screen is not None and settings_screen.mcp_busy:
            return False
        if (
            self._startup_pending
            or self.agent is None
            or self._exit
            or getattr(self, "_session_shutdown_task", None) is not None
            or self._background_rebuilding
            or self._background_retirement is not None
            or self._pending_workspace_switch is not None
            or self._handoff_in_progress
            or self._rewind_running
            or self._agent_execution_draining()
        ):
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

        if not self._queued_messages:
            jobs = self._background_jobs
            if (
                jobs is None
                or self._background_delivery_failed
                or not (jobs.pending_count or (self.agent.background_needs_turn and not self._background_paused))
            ):
                return False
            self.agent_worker = self.run_worker(
                self._process_background_completions(jobs),
                name="kolega-background-completion",
                group="turns",
                exclusive=True,
            )
            return True

        queued = self._queued_messages.pop(0)
        if queued.is_peer:
            queued.entry.kind = "peer_message"
        else:
            queued.entry.kind = "user"
        queued.entry.tone = None
        if queued.entry not in self.conversation_entries:
            self._add_conversation_entry(queued.entry)
        else:
            # Defensive compatibility for any already-mounted queued entry (for
            # example from older in-memory behavior): re-render just that widget
            # for the queued->user transition instead of rebuilding the transcript.
            self._invalidate_conversation(queued.entry)
        self._refresh_queued_messages_panel()
        self._clear_composer_hint()
        self.agent_worker = self.run_worker(
            self._process_message(queued.text, queued.attachments), name="kolega-turn", group="turns", exclusive=True
        )
        return True

    def _maybe_start_background_completion(self) -> bool:
        """Level-triggered fallback for internal mail, not ordinary user input."""
        jobs = self._background_jobs
        if jobs is None or not (
            jobs.pending_count or (getattr(self.agent, "background_needs_turn", False) and not self._background_paused)
        ):
            return False
        return self._maybe_start_queued_message()

    async def _provide_queued_user_inputs(self) -> list[QueuedUserInput]:
        """Agent-driven mid-turn drain: the running turn pulls queued messages
        at a tool boundary instead of waiting for the turn to finish.

        Runs on the TUI event loop inside the turn worker, so mutating the
        queue here is race-free with the composer.
        """
        if self._agent_execution_draining() or not self._queued_messages:
            return []
        # Render tool events already emitted so tool output stays above the
        # interjected user entries in the transcript.
        await self._drain_pending_events()
        if self._agent_execution_draining():
            return []
        # Snapshot after the await: an Esc during the drain restores whatever
        # is still in the list, so anything taken here is genuinely delivered.
        taken = list(self._queued_messages)
        if not taken:
            return []
        self._queued_messages.clear()
        inputs: list[QueuedUserInput] = []
        for queued in taken:
            if queued.is_peer:
                queued.entry.kind = "peer_message"
            else:
                queued.entry.kind = "user"
            queued.entry.tone = None
            if queued.entry not in self.conversation_entries:
                self._add_conversation_entry(queued.entry)
            else:
                self._invalidate_conversation(queued.entry)
            inputs.append(QueuedUserInput(text=queued.text, attachments=queued.attachments, origin=queued.origin))
        self._refresh_queued_messages_panel()
        self._clear_composer_hint()
        self._log_status(messages.QUEUE_DELIVERED_MID_TURN.format(count=len(inputs)), "info")
        return inputs

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
        if event.event_type == KnownEventType.CONTROL_RESOLVED:
            resolved_id = str(event.content.get("request_id") or "")
            self._clear_approval_if_resolved(resolved_id)
            self._clear_question_if_resolved(resolved_id)
            return
        if (
            self._agent_execution_draining()
            and not event.sub_agent_info
            and event.event_type not in ("log_message", "terminal_output", "file_edit_preview")
        ):
            if (
                event.event_type == KnownEventType.CONTROL_REQUESTED
                and self.control_channel.controller == tui_constants.TUI_CLIENT_ID
            ):
                request_id = str(event.content.get("request_id") or "")
                for request in self.control_channel.pending():
                    if request.request_id == request_id:
                        self.control_channel.respond(request_id, request.default, client_id=tui_constants.TUI_CLIENT_ID)
                        break
            return
        if event.event_type == KnownEventType.CONTROL_REQUESTED:
            self._handle_control_request(event)
            return
        if event.sub_agent_info and event.event_type in _SUB_AGENT_STREAM_EVENTS:
            self._render_sub_agent_delta(event)
            return
        if event.event_type in _RECORDING_ONLY_EVENTS:
            # Assistant prose, reasoning, and turn boundaries exist on the event
            # stream so that non-TUI frontends and the replay player can render a
            # conversation. This TUI still renders them from
            # process_message_stream's generator, so rendering them here too would
            # duplicate every response.
            return
        if event.event_type == "log_message":
            if self.show_logs:
                level = str(event.content.get("level", "info"))
                self._write_log(self._display_text_from_event(event), level)
        elif event.event_type == "terminal_output":
            output = event.content.get("display_output")
            if output is None:
                output = event.content.get("output", "")
            self._queue_terminal_output(str(output))
            self._mark_session_diff_dirty()
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
            self._mark_session_diff_dirty()
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
                if event.content.get("resumed") and self._turn_status_text != "Waiting for response…":
                    return
                text = self._display_text_from_event(event)
                if text:
                    if self.show_logs:
                        self._write_log(text, "info")
                    self._update_activity_progress(text)
                    if event.content.get("retry"):
                        # A retry belongs in the transcript: the status line is
                        # overwritten by the next stream delta, leaving the user
                        # unable to tell a flapping connection from a slow model.
                        self._add_conversation_entry(
                            tui_state.ConversationEntry(kind="system", content=text, complete=True, tone="warning")
                        )
        elif self.show_logs:
            text = self._display_text_from_event(event)
            if text:
                self._write_log(f"{event.event_type}: {text}", "info")
            else:
                self._write_log(messages.LOG_IGNORED_EVENT.format(event_type=event.event_type), "debug")

    def action_cancel_generation(self) -> None:
        if self._startup_pending and self._pending_question is None and self._pending_approval is None:
            self.run_worker(self.action_quit, name="startup-quit", group="startup-quit")
            return
        # /handoff generation is a helper request, not an agent turn: Esc aborts
        # the handoff stream cooperatively and leaves the session untouched.
        if self._handoff_in_progress and self._handoff_cancel_event is not None:
            self._handoff_cancel_event.set()
            return
        self._background_paused = True
        if self.agent_worker is not None:
            self._update_progress(messages.STOP_REQUESTED, complete=False, state=tui_state.TurnState.STOPPING)
            self._cancel_pending_question()
            self._cancel_pending_approval()
            self._restore_queued_messages_to_composer()
            if self._stop_pending:
                # Escalation: the worker already had a stop request and has not
                # finished, because it is parked in a wait that swallows
                # CancelledError. Repeating cancel() would repaint "Stopping…"
                # forever, so detach the worker and settle the dashboard now.
                self._force_settle_stop()
                return
            self._stop_pending = True
            self.agent_worker.cancel()
            self._arm_stop_watchdog()
            self._notify_user(messages.CANCEL_REQUESTED, severity="warning")
            return
        # Nothing is running, but the dashboard may still show a live turn from a
        # worker that vanished before _finish_turn_progress. Settle it instead.
        if self._turn_started_at is not None:
            self._force_settle_stop()
            return
        # Clear a pending scheduled-loop wakeup the way Esc clears any other
        # pending action; with no turn and no loop this is a no-op.
        if self._scheduled_loop is not None and self._scheduled_loop.is_active:
            self._request_loop_stop(messages.LOOP_STOPPED_BY_USER.format(iterations=self._scheduled_loop.iterations))

    def _arm_stop_watchdog(self) -> None:
        """Settle the dashboard if the worker has not stopped after a grace period."""
        self._disarm_stop_watchdog()
        try:
            self._stop_watchdog = self.set_timer(
                max(0.0, self._stop_force_after_seconds), self._force_settle_stop, name="stop-watchdog"
            )
        except Exception:  # noqa: BLE001 - a timer is an enhancement; Esc still works
            self._stop_watchdog = None

    def _disarm_stop_watchdog(self) -> None:
        timer, self._stop_watchdog = self._stop_watchdog, None
        if timer is not None:
            try:
                timer.stop()
            except Exception:  # noqa: BLE001 - already fired or the loop is closing
                pass

    def _force_settle_stop(self) -> None:
        """Return the UI to a usable state even when the turn worker will not die.

        The dashboard's only path out of ``STOPPING`` used to be
        ``_finish_turn_progress`` inside the turn worker, so a worker that cannot
        honor a cancel left the app stuck on "Stopping…" indefinitely. Detach it:
        ownership checks make its eventual completion a no-op for the UI. Input
        remains editable, but execution and rebuilds stay fenced until the
        detached task has finished its durable cleanup.
        """
        if self._exit:
            return
        self._stop_pending = False
        self._disarm_stop_watchdog()
        worker, self.agent_worker = self.agent_worker, None
        if worker is not None:
            task = getattr(self, "_agent_execution_task", None)
            if task is not None and not task.done():
                if getattr(self, "_draining_agent_tasks", None) is None:
                    self._draining_agent_tasks = set()
                self._draining_agent_tasks.add(task)
            worker.exit_on_error = False
            worker.cancel()
        self._turn_active = False
        self._finalize_sub_agent_activities()
        self._finalize_workflow_activities()
        self._finish_turn_progress(messages.STOPPED_BY_USER, tui_state.TurnState.STOPPED)
        if self._plan_decision_active:
            self._set_composer_status(messages.PLAN_READY_PLACEHOLDER)
            self._set_chat_enabled(False)
        else:
            self._set_chat_enabled(self.agent is not None)
        self._refresh_shortcut_context()
        self._schedule_primary_focus_restore()
        self._schedule_maybe_start_queued_message()
        if worker is not None:
            self._log_status(messages.STOP_FORCED, "warn")
            self._notify_user(messages.STOP_FORCED, severity="warning")

    def _log_mcp_tool_name_warnings(self, mcp_config: LoadedMCPConfig) -> None:
        """Warn in the log pane when verified MCP tools get provider-adjusted wire names.

        Mirrors the ``mcp: {diagnostic}`` startup warnings: the model-facing names
        are sanitized, and silently renaming them would be confusing.
        """
        service = MCPService(mcp_config, state_dir=self.settings_store.root, project_path=self.active_project_path)
        for server in mcp_config.enabled_servers:
            status = service.server_status(server)
            if not status or status.status != "verified":
                continue
            if status.fingerprint != server_fingerprint(server):
                continue
            note = mcp_tool_name_adjustment_note(server.id, status.tools)
            if note:
                self._log_status(f"mcp: warning: server '{server.id}': {note}", level="warn")

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
        try:
            settings_screen = getattr(self, "_settings_screen", None)
            await self._build_agent(
                config,
                rebuild=rebuild,
                # Peer messages may arrive while an MCP operation awaits OAuth
                # or a remote server. Resume them against the rebuilt agent.
                preserve_queued=settings_screen is not None and settings_screen.mcp_busy,
            )
        except KolegaExtensionLoadError as exc:
            # The failed generation was already reclaimed by _build_agent; there
            # is no live agent, so surface the error instead of crashing the
            # worker that requested the rebuild.
            self._set_chat_enabled(False)
            self._refresh_status_dashboard()
            self._log_status(f"extension error: {exc}", level="error")
            self._ensure_startup_entry()
            return
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

    def _plan_volatile_section(self) -> VolatileSection:
        """Volatile section carrying a handle to the current plan artifact.

        Registered on the top-level agent so that after compaction (which summarizes the
        plan away) the model is reminded where the approved plan lives and can re-read it.
        Empty text when there is no captured plan; best-effort persistence is silent here
        because the prompt-extension path already surfaces write failures.
        """
        plan = getattr(self, "_latest_plan", None)
        if not plan:
            return VolatileSection("plan", "")
        try:
            path = write_current_plan_artifact(self.store.root, self.session.session_id, plan)
        except Exception:
            return VolatileSection("plan", "")
        return VolatileSection("plan", build_current_plan_artifact_prompt(path))

    def _task_list_volatile_section(self) -> VolatileSection:
        """Volatile section carrying the current shared task list.

        Injected as a ``<system-reminder>`` so the model always sees the latest todos —
        including after compaction and after every ``update_task_list`` call — without
        needing to re-fetch them. Empty text when no task list has been set.
        """
        return VolatileSection("task_list", (getattr(self.session, "task_list_markdown", "") or "").strip())

    def _scratchpad_prompt_extension(self) -> PromptExtension | None:
        """Return prompt context advertising this session's scratchpad directory.

        Applies to both interaction modes and is stable across resumes and
        plan/build switches because it is keyed by ``session.session_id``.
        Best-effort: if the temp directory cannot be created the agent simply
        runs without a scratchpad section.
        """
        try:
            scratchpad_path = ensure_scratchpad_dir(self.project_path, self.session.session_id)
        except (OSError, ValueError) as exc:
            try:
                self._notify_user(
                    f"Could not prepare the session scratchpad: {exc}",
                    severity="warning",
                )
            except Exception:
                pass
            return None
        return PromptExtension(
            id=SCRATCHPAD_PROMPT_EXTENSION_ID,
            title="Session Scratchpad",
            markdown=build_scratchpad_prompt(scratchpad_path),
            modes=[AgentMode.CLI],
            # Delegated agents share the parent's scratchpad; the prompt text
            # tells them to use unique filenames for concurrent safety.
            propagate_to_sub_agents=True,
        )

    async def _build_agent(
        self,
        config: AgentConfig,
        rebuild: bool = False,
        *,
        restore_transcript: bool = True,
        preserve_queued: bool = False,
    ) -> None:
        if self._agent_execution_draining() and asyncio.current_task() is not self._workspace_switch_task:
            raise RuntimeError("The previous turn is still finishing cleanup; wait before rebuilding the agent.")
        previous = self._background_rebuilding
        self._background_rebuilding = True
        try:
            await self._build_agent_impl(
                config, rebuild, restore_transcript=restore_transcript, preserve_queued=preserve_queued
            )
        finally:
            self._background_rebuilding = previous
            self._schedule_pending_inputs()

    async def _build_agent_impl(
        self,
        config: AgentConfig,
        rebuild: bool = False,
        *,
        restore_transcript: bool = True,
        preserve_queued: bool = False,
    ) -> None:
        self._set_startup_stage("Loading skills and agents…")
        await self._prime_recording_offset()
        history = self.session.history
        compaction = self.session.compaction
        restoring_history = rebuild or getattr(self, "_resuming_session", False) or bool(history)
        if rebuild and not preserve_queued:
            self._clear_queued_messages()
        if self.agent is not None:
            await self._save_session_history_async()
            history = self.session.history
            compaction = self.session.compaction
            if rebuild:
                await self._cleanup_agent_generation(preserve_background=True)

        browser_manager = build_browser_manager(
            self.store.root,
            self.session.session_id,
            browser_visible=self.browser_visible,
        )
        agent_class = PlanningAgent if self.interaction_mode == tui_constants.PLAN_INTERACTION_MODE else CoderAgent
        self.skills_enabled = config.skills_enabled
        self.skill_catalog = discover_skills(self.active_project_path) if self.skills_enabled else SkillCatalog()
        self.custom_agent_catalog = validate_custom_agent_models(
            discover_custom_agents(self.active_project_path, self.settings_store.root),
            config,
        ).for_mode(self.interaction_mode)
        prompt_extensions: list[PromptExtension] = []
        tool_extensions: list[ToolExtension] = []
        # The shared task list is build-mode execution tracking, so only build mode
        # can write it. Plan mode gets a read-only view so a re-plan started mid-build
        # can see what has already been completed.
        if self.interaction_mode == tui_constants.BUILD_INTERACTION_MODE:
            prompt_extensions.append(self._shared_task_list_prompt_extension())
            tool_extensions.append(self._shared_task_list_tool_extension())
            plan_artifact_extension = self._current_plan_artifact_prompt_extension()
            if plan_artifact_extension is not None:
                prompt_extensions.append(plan_artifact_extension)
        else:
            prompt_extensions.append(self._shared_task_list_readonly_prompt_extension())
            tool_extensions.append(self._shared_task_list_readonly_tool_extension())
        # Worktree and goal control belong to the top-level agent and are never
        # inherited by delegates. Both tool extensions are installed in either
        # mode; neither tool is declared in ``planning_tools``, so the planning
        # agent's registry filters both out and its dispatch cannot reach them.
        # The mode-aware prompts still apply in plan mode. They explicitly mark
        # both tools as unavailable there while teaching the planner where the
        # build-mode handoffs belong in an approved plan.
        prompt_extensions.append(self._worktree_control_prompt_extension())
        tool_extensions.append(self._worktree_control_tool_extension())
        prompt_extensions.append(self._goal_control_prompt_extension())
        tool_extensions.append(self._goal_control_tool_extension())
        # Cross-session messaging: discovery and plain-text delivery to peer
        # sessions sharing this process. Top-level only; the planning agent's
        # read-only registry filters it out of plan mode.
        tool_extensions.append(self._peer_messaging_tool_extension())
        # Both interaction modes get the scratchpad: plan-mode research and
        # build-mode execution both produce throwaway files.
        scratchpad_extension = self._scratchpad_prompt_extension()
        if scratchpad_extension is not None:
            prompt_extensions.append(scratchpad_extension)
        if self.skills_enabled:
            skill_prompt_extension = build_skill_prompt_extension(
                self.skill_catalog,
                context_window_tokens=context_window_tokens_for_skill_budget(
                    config,
                    getattr(agent_class, "agent_name", None),
                ),
            )
            skill_tool_extension = build_skill_tool_extension(
                self.skill_catalog,
                lambda: self.agent.history if self.agent is not None else [],
            )
            if skill_prompt_extension is not None:
                prompt_extensions.append(skill_prompt_extension)
            if skill_tool_extension is not None:
                tool_extensions.append(skill_tool_extension)
        # Both modes can ask the user; only the framing differs. Plan mode treats a
        # question as cheap, build mode reserves it for decisions that are expensive
        # to get wrong.
        if self.interaction_mode == tui_constants.PLAN_INTERACTION_MODE:
            prompt_extensions.append(self._planning_question_prompt_extension())
        else:
            prompt_extensions.append(self._build_question_prompt_extension())
        tool_extensions.append(self._planning_question_tool_extension())

        mcp_config = getattr(config, "mcp_config", None)
        if mcp_config is not None:
            for diagnostic in getattr(mcp_config, "diagnostics", []) or []:
                self._log_status(f"mcp: {diagnostic}", level="warn")
        mcp_extension = build_mcp_tool_extension(
            self.active_project_path,
            self.settings_store.root,
            project_trusted=self.settings.is_mcp_project_trusted(self.project_path),
            loaded_config=mcp_config if self.active_project_path == self.project_path else None,
        )
        if mcp_extension is not None:
            tool_extensions.append(mcp_extension)
            if mcp_config is not None and self.active_project_path == self.project_path:
                self._log_mcp_tool_name_warnings(mcp_config)

        # gigacode applies to any top-level agent and is carried across rebuilds.
        # In plan mode the orchestrating agent is read-only, so its workflow
        # sub-agents are forced read-only too (enforced in the dispatch adapter).
        gigacode_active = self._gigacode_enabled
        if gigacode_active:
            prompt_extensions.append(self._gigacode_prompt_extension())

        # An active autonomous goal is carried across rebuilds so the agent stays
        # goal-aware after a model switch. Live set/clear uses agent.apply_goal().
        if self._goal is not None and self._goal.condition:
            prompt_extensions.append(self._goal_prompt_extension())

        try:
            # Launch-selected extension: a fresh bundle for every agent generation,
            # from the same factory. Assigned to self immediately so every exit path
            # cleans up exactly this generation's bundle.
            extension_bundle = None
            if self.extension_selection is not None:
                extension_host = KolegaExtensionHost(
                    project_path=self.active_project_path,
                    workspace_id=self.session.workspace_id,
                    thread_id=self.session.thread_id,
                    config=config,
                    agent_mode=AgentMode(self.mode),
                )
                extension_bundle = create_extension_bundle(self.extension_selection, extension_host)
                self._extension_bundle = extension_bundle
                prompt_extensions.extend(extension_bundle.prompt_extensions)
                tool_extensions.extend(extension_bundle.tool_extensions)

            try:
                constructor_recorder = None if restoring_history else self._session_recorder
                self.agent = agent_class(
                    project_path=self.active_project_path,
                    workspace_id=self.session.workspace_id,
                    thread_id=self.session.thread_id,
                    connection_manager=self.recording_connection_manager,
                    config=config,
                    browser_manager=browser_manager,
                    agent_mode=AgentMode(self.mode),
                    prompt_extensions=prompt_extensions,
                    tool_extensions=tool_extensions,
                    memory_manager=self.memory_manager,
                    permission_mode=self.permission_mode,
                    # Permission policy lives in the session runtime, not the UI: mode
                    # checks and saved-rule matching are decisions about the session, so
                    # every frontend gets them and only real questions reach a person.
                    permission_callback=self.session_runtime.permission_callback,
                    session_recorder=constructor_recorder,
                    hook_dispatcher=self._session_hook_dispatcher(),
                    custom_agent_catalog=self.custom_agent_catalog,
                    usage_ledger=self._usage_ledger,
                    llm_trace_sink=extension_bundle.llm_trace_sink if extension_bundle else None,
                )
            except ValueError as exc:
                if extension_bundle is not None:
                    # With an extension loaded, a tool-name conflict at construction
                    # refuses the generation rather than continuing without the
                    # requested extension.
                    raise KolegaExtensionLoadError(str(exc)) from exc
                raise
            assert self.agent is not None
            # The runtime owns the agent for control purposes while the CLI keeps
            # composing it from settings, skills, hooks, and MCP configuration.
            self.session_runtime.adopt(self.agent)
            # The plan handle and shared task list are conversation-injected context (system
            # reminders), so a compaction that summarizes them away is followed by re-injection
            # on the next turn — see agent/volatile_context.py. Registered in both interaction
            # modes: the planner sees the plan artifact while re-planning mid-build, and both
            # modes keep the task list (read-only for the planner) in view. Each rebuild
            # constructs a fresh agent, so providers never accumulate across rebuilds.
            self.agent.add_volatile_section(self._plan_volatile_section)
            self.agent.add_volatile_section(self._task_list_volatile_section)
            if scratchpad_extension is not None:
                # Expose the resolved directory for the KOLEGA_SCRATCHPAD session env
                # and the terminal safety checker; the prompt extension above is what
                # the model sees.
                self.agent.scratchpad_dir = scratchpad_dir_for(self.project_path, self.session.session_id)
            # Mid-turn queue delivery: the running turn drains composer messages
            # queued while it works (main agent only; sub-agents never get this).
            self.agent.set_queued_input_provider(self._provide_queued_user_inputs)
            # Initialize LSP (language detection + server resolution)
            agent = self.agent
            assert agent.tool_collection is not None
            self._set_startup_stage("Initializing tools…")
            await agent.tool_collection.initialize()
            agent.gigacode_enabled = gigacode_active
            # A session-level /web-search override outlives agent rebuilds (model
            # switches); without one the config-resolved mode stands.
            if self._web_search_mode:
                agent.apply_web_search_mode(self._web_search_mode)
            self._initialize_ledger_diff_tracker()
            if restoring_history:
                self._set_startup_stage("Restoring session…")
                self.agent.restore_message_history(history)
                self.agent.restore_compaction_state(compaction)
                self.agent.session_recorder = self._session_recorder
                if restore_transcript:
                    archived = await asyncio.to_thread(self._session_recorder.read_archived_background_messages)
                    # UI-only reports must never enter restore_message_history:
                    # they belong to the retiring workspace, not its successor.
                    self._restore_conversation_history([*history, *archived])
            if extension_bundle is not None:
                await bind_extension_agent(extension_bundle, self.agent)
            await self._attach_background_jobs()
            self._update_mode_chrome()
            self._ensure_startup_entry()
            self._set_startup_stage("Running startup hooks…")
            await self._fire_session_start_once()
            # Refresh the Settings-tab LSP status now that the agent (and its
            # initialized lsp_manager) exists. Without this the status is stale
            # from startup, when it was computed before the agent was built.
            self._update_lsp_settings_status()
        except BaseException:
            # Reclaim the partial generation (bundle is already installed on
            # self; the agent may or may not be) before surfacing the failure.
            await self._cleanup_agent_generation()
            raise

    async def _cleanup_agent_generation(self, *, preserve_background: bool = False) -> None:
        """Tear down the current agent generation (agent, then bundle) exactly once.

        Retirement must succeed before detaching resources: a storage failure
        leaves the generation and its reports owned for a later retry. Cleanup
        failures after retirement are reported, never raised.
        """
        if self._agent_execution_draining() and asyncio.current_task() is not self._workspace_switch_task:
            self._agent_cleanup_failed = True
            return
        if not preserve_background or self._background_retirement is not None:
            await self._retire_background_jobs(agent=self.agent)
        agent = self.agent
        bundle = self._extension_bundle
        self.agent = None
        self._extension_bundle = None
        jobs = self._background_jobs
        if preserve_background and jobs is not None and agent is not None:
            self._background_resume_needed = agent.background_needs_turn
            agent.background_jobs = None
            if jobs.has_work:
                self._retired_agent_generations.append((agent, bundle))
                return
        try:
            if agent is not None:
                try:
                    await agent.cleanup()
                except Exception as exc:  # noqa: BLE001 — reported, never masks the primary failure
                    self._agent_cleanup_failed = True
                    self._log_status(f"agent cleanup failed: {exc}", level="warn")
        finally:
            if bundle is not None:
                try:
                    await cleanup_extension_bundle(bundle)
                except Exception as exc:  # noqa: BLE001 — reported, never masks the primary failure
                    self._agent_cleanup_failed = True
                    self._log_status(f"extension cleanup failed: {exc}", level="warn")

    async def _cleanup_extension_bundle(self) -> None:
        """Release the current extension bundle exactly once (rebuild or app exit)."""
        bundle = self._extension_bundle
        self._extension_bundle = None
        if bundle is None:
            return
        try:
            await cleanup_extension_bundle(bundle)
        except Exception as exc:  # noqa: BLE001 — reported, never masks the primary failure
            self._log_status(f"extension cleanup failed: {exc}", level="warn")

    def _format_lsp_status(self) -> str:
        """Build a human-readable LSP status message for the conversation or /lsp command."""
        agent = self.agent
        if agent is None or agent.tool_collection is None:
            return ""
        manager = agent.tool_collection.lsp_manager
        if manager is None or not manager.enabled:
            return ""

        lines = ["## 🔍 Language Servers (LSP)"]

        status = manager.status()
        if not status.get("initialized"):
            lines.append("\nLSP status will appear after the agent starts.")
            return "\n".join(lines)

        detected = status.get("detected", [])
        resolved = {r["language_id"]: r for r in status.get("resolved", [])}
        missing = status.get("missing", [])

        if not status.get("scan_complete", True):
            lines.append(
                "\n**⚠️ Detection incomplete:** filesystem scan stopped because of "
                f"{status.get('scan_stop_reason') or 'a scan limit'} after "
                f"{status.get('scanned_entries', 0)} entries."
            )

        if detected:
            lines.append(f"\n**Detected {len(detected)} language(s):**")
            for d in detected:
                rl = resolved.get(d["language_id"])
                missing_rl = next((m for m in missing if m["language_id"] == d["language_id"]), None)
                if rl:
                    lines.append(f"  - {d['display_name']} → **{rl['server_name']}** ({d['detection_reason']})")
                elif missing_rl:
                    lines.append(
                        f"  - {d['display_name']} → _{missing_rl['server_name']} not found_ ({d['detection_reason']})"
                    )
                else:
                    lines.append(f"  - {d['display_name']} ({d['detection_reason']})")
        else:
            lines.append("\nNo supported languages detected in this project.")
            return "\n".join(lines)

        if missing:
            lines.append(f"\n**⚠️ Missing servers ({len(missing)}):**")
            for m in missing:
                lines.append(f"  - **{m['display_name']}**: `{m['server_name']}`")

        # Active sessions with live state
        active_sessions = []
        for session in status.get("sessions", []):
            server_name = session["server_name"]
            lang_id = session["language_id"]
            if session.get("connected"):
                pid_str = f" (pid={session['pid']})" if session.get("pid") else ""
                active_sessions.append(f"  - ✅ **{server_name}** for {lang_id}{pid_str}")
            elif session.get("status") == "error" and session.get("last_error"):
                active_sessions.append(f"  - ❌ **{server_name}** for {lang_id} — {session['last_error']}")

        if active_sessions:
            lines.append(f"\n**Active sessions ({len(active_sessions)}):**")
            lines.extend(active_sessions)

        # Last diagnostic counts
        diag_counts = status.get("diagnostic_counts", {})
        if diag_counts:
            shown = list(diag_counts.items())[:5]
            count_parts = []
            for uri, count in shown:
                path_str = uri.replace("file://", "") if uri.startswith("file://") else uri
                count_parts.append(f"{path_str}: {count}")
            lines.append(f"\n**Last diagnostics:** {', '.join(count_parts)}")

        lines.append("\nRun `/lsp` to see this status again.")

        return "\n".join(lines)

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
        # Surface a restored active goal once at startup so the user knows the loop
        # is dormant and how to resume/clear it. The loop itself does not auto-start.
        if self._goal is not None and self._goal.condition and not self._goal.met:
            self._add_conversation_entry(
                tui_state.ConversationEntry(
                    kind="system",
                    content=messages.GOAL_RESUMED_NOTE.format(condition=self._goal.condition),
                )
            )
            self._refresh_status_dashboard()


def checkpoint_excerpt(label: str) -> str:
    collapsed = " ".join(label.split())
    if len(collapsed) > 60:
        collapsed = collapsed[:59] + "…"
    return collapsed
