"""Scheduler and iteration driver for the ``/loop`` command.

A loop is a prompt plus a schedule. A one-second Textual timer
(:meth:`LoopRuntimeMixin._loop_tick`) checks whether the loop is due and whether
the app is idle; when both hold it launches one iteration as an ordinary turn in
the exclusive ``turns`` worker group, so checkpointing, session saving, queue
draining, Esc handling and mode chrome all behave exactly as they do for a
message the user typed.

Two invariants keep this honest:

* **Fires happen between turns only.** The idle gate mirrors the one in
  :meth:`~kolega_code.cli.tui.agent_runtime.AgentRuntimeMixin._maybe_start_queued_message`,
  plus the queue itself, so a user's typed message always wins over a timer.
* **No catch-up.** If several due times pass while the app is busy, exactly one
  iteration runs once it goes idle.
"""

from __future__ import annotations

import asyncio

from kolega_code.agent import PromptExtension
from kolega_code.session.automation import AutomationDriver, LoopDecision, LoopHooks, LoopOutcome
from kolega_code.session.turns import TurnResult

from .. import messages
from ..loop import (
    PROMPT_SOURCE_INLINE,
    PROMPT_SOURCE_LOOP_MD,
    SUB_MINUTE_ADVISORY_SECONDS,
    LoopError,
    LoopSpec,
    LoopState,
    build_loop_prompt_extension_markdown,
    format_countdown,
    loop_status_summary,
    now_local,
    parse_schedule_text,
    read_loop_md,
)
from ..slash_commands import TUI_COMMAND_NAMES, agent_command_names
from ..skills import skill_activation_label
from . import app_base as tui_app_base
from . import state as tui_state


class LoopRuntimeMixin(tui_app_base.KolegaAppBase):
    # ------------------------------------------------------------------
    # Prompt extension and persistence
    # ------------------------------------------------------------------

    def _loop_prompt_extension(self) -> PromptExtension:
        assert self._scheduled_loop is not None
        return PromptExtension(
            id="cli-active-loop",
            title="Scheduled loop",
            markdown=build_loop_prompt_extension_markdown(self._scheduled_loop),
            modes=None,
            # Read-only context about *why* this turn is running; a delegated
            # sub-agent benefits from knowing nobody is watching too.
            propagate_to_sub_agents=True,
        )

    def _sync_loop_to_session(self) -> None:
        """Mirror the live loop state into the session record (in-memory only)."""
        self.session.loop = self._scheduled_loop.to_dict() if self._scheduled_loop is not None else {}

    async def _persist_loop_async(self) -> None:
        self._sync_loop_to_session()
        await self._save_session_async()

    def _loop_summary(self) -> str | None:
        """One-line loop summary for the status dashboard, or None when inactive.

        A stopped or finished loop drops off the dashboard but stays available
        through ``/loop status``.
        """
        state = self._scheduled_loop
        if state is None or state.stopped:
            return None
        return loop_status_summary(state)

    def _refresh_loop_status_if_changed(self) -> None:
        """Re-render the dashboard only when the coarse loop label actually moved.

        The scheduler ticks every second but the countdown is minute-resolution
        above a minute, so this keeps ticks from reflowing the screen — Textual
        layout cost scales with the number of mounted widgets.
        """
        if self._loop_summary() != self._status_state.loop:
            self._refresh_status_dashboard()

    def _loop_when_text(self, state: LoopState, now=None) -> str:
        countdown = format_countdown(state.seconds_until(now))
        return countdown if countdown == "now" else f"in {countdown}"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _activate_loop(self, spec: LoopSpec) -> bool:
        """Create and persist a loop, returning whether it replaced a running one.

        Raises :class:`~kolega_code.cli.loop.LoopError` when the schedule or the
        prompt cannot be resolved; the command handler renders the message.
        """
        schedule = spec.schedule
        prompt = spec.prompt.strip()
        prompt_source = PROMPT_SOURCE_INLINE
        truncated = False

        if not prompt or schedule is None:
            loop_md = read_loop_md(self.project_path)
            if loop_md is None:
                raise LoopError(messages.LOOP_MD_MISSING if not prompt else messages.LOOP_SCHEDULE_MISSING)
            if not prompt:
                prompt = loop_md.prompt
                prompt_source = PROMPT_SOURCE_LOOP_MD
                truncated = loop_md.truncated
            if schedule is None and loop_md.schedule_text:
                schedule = parse_schedule_text(loop_md.schedule_text)

        if schedule is None:
            raise LoopError(messages.LOOP_SCHEDULE_MISSING)

        replacing = bool(self._scheduled_loop is not None and self._scheduled_loop.is_active)
        self._scheduled_loop = LoopState.create(
            schedule,
            prompt,
            prompt_source=prompt_source,
            fresh=spec.fresh,
            max_iterations=spec.max_iterations,
            expires_seconds=spec.expires_seconds,
        )
        if self.agent is not None:
            self.agent.apply_loop(True, self._loop_prompt_extension())
        self._refresh_status_dashboard()
        await self._persist_loop_async()

        if truncated:
            self._add_conversation_entry(
                tui_state.ConversationEntry(kind="system", content=messages.LOOP_MD_TRUNCATED, tone="warning")
            )
        interval = schedule.interval_seconds
        if interval is not None and interval < SUB_MINUTE_ADVISORY_SECONDS:
            self._add_conversation_entry(
                tui_state.ConversationEntry(kind="system", content=messages.LOOP_SUB_MINUTE_ADVISORY, tone="warning")
            )
        if self.permission_mode.value == "ask":
            self._add_conversation_entry(
                tui_state.ConversationEntry(
                    kind="system", content=messages.LOOP_ASK_PERMISSION_ADVISORY, tone="warning"
                )
            )
        return replacing

    def _mark_loop_stopped(self, note: str, *, tone: str = "warning", notify: bool = True) -> bool:
        """Synchronously end the loop. Returns False when there was nothing to stop.

        Split from :meth:`_stop_loop` so key bindings (which are synchronous) can
        close the scheduling window immediately and persist afterwards.
        """
        state = self._scheduled_loop
        if state is None or state.stopped or self._agent_execution_draining():
            return False
        state.stopped = True
        state.deferred = False
        state.status_note = note
        if self.agent is not None:
            self.agent.apply_loop(False)
        self._sync_loop_to_session()
        self._add_conversation_entry(tui_state.ConversationEntry(kind="system", content=note, tone=tone))
        if notify:
            self._notify_user(note, severity="warning" if tone == "warning" else "information")
        self._refresh_status_dashboard()
        return True

    async def _stop_loop(self, note: str, *, tone: str = "warning", notify: bool = True) -> None:
        if self._mark_loop_stopped(note, tone=tone, notify=notify):
            await self._persist_loop_async()

    def _request_loop_stop(self, note: str) -> bool:
        """Stop the loop from a synchronous context (a key binding), then persist."""
        if not self._mark_loop_stopped(note):
            return False
        self.run_worker(self._persist_loop_async(), name="kolega-loop-persist", group="loop-persist")
        return True

    async def _restore_loop_on_startup(self) -> None:
        """Re-arm a loop restored from a resumed session.

        A ``next_fire_at`` already in the past is deliberately left alone: the
        loop fires once when the app goes idle, never once per missed window.
        """
        state = self._scheduled_loop
        if state is None or state.stopped:
            return
        if state.is_expired() or state.reached_cap:
            await self._stop_loop(messages.LOOP_EXPIRED.format(iterations=state.iterations), notify=False)
            return
        if self.agent is not None:
            self.agent.apply_loop(True, self._loop_prompt_extension())
        self._add_conversation_entry(
            tui_state.ConversationEntry(
                kind="system",
                content=messages.LOOP_RESTORED.format(
                    schedule=state.schedule_label(), when=self._loop_when_text(state)
                ),
            )
        )
        self._refresh_status_dashboard()

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def _loop_ready_to_fire(self, *, owned: bool = False) -> bool:
        """Whether the app is idle enough to start a scheduled iteration."""
        settings_screen = getattr(self, "_settings_screen", None)
        return (
            self.agent is not None
            and not (settings_screen is not None and settings_screen.mcp_busy)
            and not self._agent_execution_draining()
            and (owned or (not self._loop_iteration_active and not self._turn_active and self.agent_worker is None))
            and not self._startup_pending
            and not self._exit
            and getattr(self, "_session_shutdown_task", None) is None
            and not self._background_rebuilding
            and self._background_retirement is None
            and self._pending_workspace_switch is None
            and not self._handoff_in_progress
            and not self._rewind_running
            and self._pending_question is None
            and self._pending_approval is None
            and self._pending_model_selection is None
            and self._pending_effort_selection is None
            and self._pending_theme_selection is None
            and not self._plan_decision_active
        )

    async def _loop_tick(self) -> None:
        """One scheduler tick: end, defer, or fire."""
        state = self._scheduled_loop
        if state is None:
            return

        decision = AutomationDriver(self._turn_controller).loop_decision(
            state, ready=self._loop_ready_to_fire(), now=now_local()
        )
        if decision is LoopDecision.EXPIRED:
            await self._stop_loop(messages.LOOP_EXPIRED.format(iterations=state.iterations))
            return
        if decision is LoopDecision.CAPPED:
            await self._stop_loop(
                messages.LOOP_MAX_ITERATIONS.format(max_iterations=state.max_iterations),
                tone="info",
            )
            return

        if decision in (LoopDecision.WAIT, LoopDecision.DEFER):
            self._refresh_loop_status_if_changed()
            return
        if decision is not LoopDecision.FIRE:
            return

        self._launch_loop_iteration()

    def _launch_loop_iteration(self) -> bool:
        """Start one iteration in the exclusive ``turns`` worker group."""
        if (
            self._scheduled_loop is None
            or AutomationDriver(self._turn_controller).loop_decision(
                self._scheduled_loop, ready=self._loop_ready_to_fire(), now=now_local()
            )
            is not LoopDecision.FIRE
        ):
            return False
        # Set before run_worker so a tick landing between here and the worker's
        # first await cannot start a second iteration.
        self._loop_iteration_active = True
        self.agent_worker = self.run_worker(
            self._run_loop_iteration(), name="kolega-turn", group="turns", exclusive=True
        )
        return True

    def _resolve_loop_prompt(self, state: LoopState) -> str | None:
        """The prompt for this iteration; ``None`` when ``loop.md`` has vanished.

        ``loop.md`` is re-read on every fire so edits take effect on the next
        iteration.
        """
        if state.prompt_source != PROMPT_SOURCE_LOOP_MD:
            return state.prompt
        loop_md = read_loop_md(self.project_path)
        if loop_md is None:
            return None
        state.prompt = loop_md.prompt
        return loop_md.prompt

    def _apply_loop_skill_prefix(self, prompt: str) -> str:
        """Activate a leading ``/skill`` token, returning the remaining prompt.

        A leading token that names a built-in command is left alone and sent as
        plain text, matching how scheduled prompts behave elsewhere.
        """
        if not prompt.startswith("/"):
            return prompt
        token, _, remainder = prompt.partition(" ")
        command = token.lower()
        if command in agent_command_names() or command in TUI_COMMAND_NAMES:
            return prompt
        skill_name = command.removeprefix("/")
        if self.skill_catalog.get(skill_name) is None:
            return prompt
        self._activate_skill_in_agent(skill_name)
        self._add_conversation_entry(
            tui_state.ConversationEntry(kind="skill", content=skill_activation_label(skill_name))
        )
        return remainder.strip() or f"Apply the {skill_name} skill now."

    def _drain_loop_tokens(self, state: LoopState, mark) -> None:
        """Add all LLM usage since ``mark`` — the whole iteration's command tree —
        to the loop's counter."""
        delta = self._usage_ledger.snapshot().since(mark)
        if delta.total_tokens > 0:
            state.tokens_spent += delta.total_tokens

    async def _run_loop_iteration(self) -> None:
        """Run one scheduled iteration as an ordinary turn."""
        await asyncio.sleep(0)  # The caller must publish agent_worker before eager entry.
        self._track_agent_execution()
        owner = self.agent_worker
        state = self._scheduled_loop
        if state is None:
            self._loop_iteration_active = False
            self.agent_worker = None
            return
        iteration_mark = self._usage_ledger.snapshot()
        failure = ""

        async def reset_context() -> None:
            await self._reset_current_thread(preserve_loop=True)
            if self.agent_worker is owner:
                self._add_conversation_entry(
                    tui_state.ConversationEntry(kind="system", content=messages.LOOP_ITERATION_FRESH)
                )

        async def prepare_prompt() -> str | None:
            try:
                prompt = self._resolve_loop_prompt(state)
            except LoopError as exc:
                await self._stop_loop(str(exc))
                return None
            if prompt is None:
                await self._stop_loop(messages.LOOP_MD_GONE)
                return None
            return self._apply_loop_skill_prefix(prompt)

        async def started() -> None:
            self._sync_loop_to_session()
            self._add_conversation_entry(
                tui_state.ConversationEntry(
                    kind="system",
                    content=messages.LOOP_ITERATION_STARTED.format(
                        iteration=state.iterations,
                        max_iterations=state.max_iterations,
                        schedule=state.schedule_label(),
                    ),
                )
            )

        async def run_turn(prompt: str) -> TurnResult:
            nonlocal failure
            result = await self._process_message(
                prompt, turn_label=f"Loop {state.iterations}: {state.prompt}", release_owner=False
            )
            failure = result.error
            return result

        try:
            outcome = await AutomationDriver(self._turn_controller).run_loop_iteration(
                state,
                LoopHooks(
                    run_turn=run_turn,
                    prepare_prompt=prepare_prompt,
                    drain_tokens=lambda: self._drain_loop_tokens(state, iteration_mark),
                    reset_context=reset_context,
                    started=started,
                    is_current=lambda: self._scheduled_loop is state,
                    ready=lambda: self._loop_ready_to_fire(owned=True),
                ),
            )
            if self.agent_worker is not owner or self._agent_execution_draining():
                return
            if outcome is LoopOutcome.CANCELLED:
                await self._stop_loop(messages.LOOP_STOPPED_BY_USER.format(iterations=state.iterations))
                return
            if outcome is LoopOutcome.FAILED:
                await self._stop_loop(messages.LOOP_STOPPED_AFTER_FAILED_TURN.format(error=failure))
                return

            await self._persist_loop_async()
            self._refresh_status_dashboard()
        except asyncio.CancelledError:
            # Cancelled outside the turn stream (for example during a --fresh
            # thread reset), where _run_turn_stream never saw the cancellation.
            if self.agent_worker is owner and self._scheduled_loop is state:
                self._mark_loop_stopped(messages.LOOP_STOPPED_BY_USER.format(iterations=state.iterations))
                await self._persist_loop_async()
            raise
        except Exception as exc:
            if self.agent_worker is owner and self._scheduled_loop is state:
                await self._stop_loop(messages.LOOP_STOPPED_AFTER_FAILED_TURN.format(error=str(exc)))
            raise
        finally:
            if self.agent_worker is owner:
                self._loop_iteration_active = False
                self.agent_worker = None
                if self._stop_pending:
                    self._force_settle_stop()
                self._schedule_maybe_start_queued_message()
