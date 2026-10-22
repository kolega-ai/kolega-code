from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest

from kolega_code.agent.baseagent import BaseAgent
from kolega_code.cli import messages
from kolega_code.cli.tui import state as tui_state
from kolega_code.cli.tui.widgets import ChatComposer
from kolega_code.events import AgentEvent, KnownEventType
from kolega_code.llm.models import Message, TextBlock

from ._app_test_utils import FakeCoderAgent, _build_sub_agent_test_app, install_fake_agents, wait_for_turn_idle
from .test_app_cancel_escalation import _UnstoppableStreamAgent, _start_stuck_turn, _wait_for


class _RecordedFinalizationAgent(FakeCoderAgent):
    async def process_message_stream(self, message, attachments=None):
        self.messages.append(message)
        await BaseAgent.prepare_background_completions(cast(BaseAgent, self))
        user = Message(role="user", content=[TextBlock(message)])
        assistant = Message(role="assistant", content=[TextBlock("recorded answer")], stop_reason="end_turn")
        recorder = self.session_recorder
        assert recorder is not None
        recorder.start_turn(user)
        recorder.record_assistant(assistant)
        self.history.extend([user, assistant])
        yield {"type": "response", "content": "recorded answer", "complete": True, "uuid": message}
        await BaseAgent._finish_recorded_turn(cast(BaseAgent, self), "completed")


@pytest.mark.asyncio
@pytest.mark.parametrize("attempt_resume", [False, True], ids=["queue", "resume-then-queue"])
async def test_force_stop_fences_real_recorder_until_terminal_write_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attempt_resume: bool
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    install_fake_agents(monkeypatch, coder_cls=_RecordedFinalizationAgent)
    app._stop_force_after_seconds = 60
    entered = threading.Event()
    release = threading.Event()

    async with app.run_test() as pilot:
        agent = app.agent
        assert isinstance(agent, _RecordedFinalizationAgent)
        recorder = agent.session_recorder
        assert recorder is not None
        assert app.config is not None
        append = recorder.journal.append

        def held_append(event_type, **kwargs):
            if event_type == "turn.completed" and not release.is_set():
                entered.set()
                assert release.wait(10), "terminal-write gate was not released"
            return append(event_type, **kwargs)

        monkeypatch.setattr(recorder.journal, "append", held_append)
        app.agent_worker = app.run_worker(app._process_message("first"), name="kolega-turn", group="turns")
        old_task = None
        try:
            await _wait_for(pilot, entered.is_set)
            old_task = app._agent_execution_task
            assert old_task is not None
            app.action_cancel_generation()
            await pilot.pause(0.01)
            app.action_cancel_generation()
            await _wait_for(pilot, lambda: app.agent_worker is None)
            assert app._agent_execution_draining()
            assert not old_task.done()

            if attempt_resume:
                target = app.store.create(app.project_path, "code", {})
                current_session = app.session
                with monkeypatch.context() as resume_patch:
                    save_history = AsyncMock()
                    quit_app = AsyncMock()
                    resume_patch.setattr(app, "_save_session_history_async", save_history)
                    resume_patch.setattr(app, "_save_session_async", AsyncMock())
                    resume_patch.setattr(app, "action_quit", quit_app)
                    resume_patch.setattr(app, "_recent_sessions_available", True)
                    await app._resume_recent_session(target.session_id)
                    save_history.assert_not_awaited()
                    quit_app.assert_not_awaited()
                    assert app.session is current_session
                    assert app.resume_session is None
                    assert app.agent is agent
                    assert agent.session_recorder is recorder

            composer = app.query_one("#composer", ChatComposer)
            composer.load_text("next")
            await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
            await _wait_for(pilot, lambda: len(app._queued_messages) == 1 and app.agent_worker is None)
            assert agent.messages == ["first"]
            assert not app._maybe_start_queued_message()
            assert await app._provide_queued_user_inputs() == []
            with pytest.raises(RuntimeError, match="still finishing cleanup"):
                await app._build_agent(app.config, rebuild=True)
            assert app.agent is agent
            assert agent.cleanup_calls == 0
            assert app._turn_final_state is tui_state.TurnState.STOPPED
            assert composer.disabled is False
        finally:
            release.set()
            if old_task is not None:
                await _wait_for(pilot, old_task.done)
            await _wait_for(pilot, lambda: not app._agent_execution_draining())
            await wait_for_turn_idle(app, pilot)

        await _wait_for(pilot, lambda: agent.messages == ["first", "next"])
        await wait_for_turn_idle(app, pilot)
        terminal = [event for event in recorder.journal.read_events() if event.event_type == "turn.completed"]
        assert len(terminal) == 2
        assert len({event.turn_id for event in terminal}) == 2
        assert not app._queued_messages


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_after_release", [False, True], ids=["success", "late-error"])
async def test_detached_chunks_cannot_overwrite_replacement_status_or_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_after_release: bool
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    install_fake_agents(monkeypatch, coder_cls=_UnstoppableStreamAgent)
    app._stop_force_after_seconds = 60

    async with app.run_test() as pilot:
        agent = await _start_stuck_turn(app, pilot)
        agent.fail_after_release = fail_after_release
        old_worker = app.agent_worker
        assert old_worker is not None
        old_task = app._agent_execution_task
        assert old_task is not None
        replacement_release = asyncio.Event()

        async def replacement():
            await replacement_release.wait()

        replacement_worker = None
        try:
            app.action_cancel_generation()
            await _wait_for(pilot, lambda: agent.cancellations == 1)
            app.action_cancel_generation()
            await _wait_for(pilot, lambda: app._agent_execution_draining())
            replacement_worker = app.run_worker(replacement(), name="replacement", group="replacement")
            app.agent_worker = replacement_worker
            app._begin_turn_progress("replacement status")
            timer = app._turn_timer
            agent.release.set()
            await _wait_for(pilot, old_task.done)
            assert not app._agent_execution_draining()
            assert app.agent_worker is replacement_worker
            assert app._turn_status_text == "replacement status"
            assert app._turn_timer is timer
            assert not any("late " in entry.content for entry in app.conversation_entries)
            if fail_after_release:
                assert isinstance(old_worker.error, RuntimeError)
        finally:
            agent.release.set()
            replacement_release.set()
            await _wait_for(pilot, old_task.done)
            if replacement_worker is not None:
                await replacement_worker.wait()
            app.agent_worker = None
            app._finish_turn_progress(messages.STOPPED_BY_USER, tui_state.TurnState.STOPPED)


@pytest.mark.asyncio
async def test_detached_completion_delivery_cannot_start_another_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def held_delivery(agent):
        entered.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                pass

    async with app.run_test() as pilot:
        assert app.agent is not None
        jobs = AsyncMock()
        jobs.deliver.side_effect = held_delivery
        monkeypatch.setattr(app, "_background_jobs", jobs)
        app.agent.background_needs_turn = True
        continuation = AsyncMock()
        monkeypatch.setattr(app, "_run_turn_stream", continuation)
        app.agent_worker = app.run_worker(app._process_background_completions(jobs), name="completion", group="turns")
        old_task = None
        try:
            await _wait_for(pilot, entered.is_set)
            old_task = app._agent_execution_task
            app.action_cancel_generation()
            app.action_cancel_generation()
            assert app._agent_execution_draining()
        finally:
            # Avoid the mailbox fallback starting fresh work during teardown.
            app.agent.background_needs_turn = False
            jobs.pending_count = 0
            release.set()
            if old_task is not None:
                await _wait_for(pilot, old_task.done)
            await _wait_for(pilot, lambda: not app._agent_execution_draining())
            monkeypatch.setattr(app, "_background_jobs", None)
        continuation.assert_not_awaited()


@pytest.mark.asyncio
async def test_detached_bus_events_cannot_revive_turn_or_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    install_fake_agents(monkeypatch, coder_cls=_UnstoppableStreamAgent)
    app._stop_force_after_seconds = 60

    async with app.run_test() as pilot:
        agent = await _start_stuck_turn(app, pilot)
        task = app._agent_execution_task
        assert task is not None
        try:
            app.action_cancel_generation()
            app.action_cancel_generation()
            assert app._agent_execution_draining()
            entries = list(app.conversation_entries)
            status_text = app._turn_status_text
            context_tokens = app._status_state.input_tokens, app._status_state.max_tokens
            diag = Mock()
            monkeypatch.setattr(app, "_diag_tee", diag)
            for event_type, content in [
                ("llm_status_update", {"text": "late retry", "retry": True}),
                ("status_update", {"text": "late status"}),
                ("llm_context_update", {"input_tokens": 123, "max_tokens": 456}),
                ("compaction_status", {"phase": "started", "message": "late compaction"}),
                ("compaction_status", {"phase": "finished", "summary": "late summary"}),
                ("chat_message", {"message_type": "message", "text": "late message"}),
                ("chat_message", {"message_type": "tool_call", "tool_name": "late tool", "tool_call_id": "late"}),
                ("tool_streaming_update", {"tool_name": "late tool", "tool_call_id": "late", "text": "late tool"}),
                ("terminal_command", {"command": "late command"}),
            ]:
                event = AgentEvent(sender="coder", event_type=event_type, content=content)
                app._render_event(event)
                diag.assert_called_with(event)
                assert app.conversation_entries == entries, event_type
                assert app._turn_status_text == status_text, event_type
                assert app._turn_final_state is tui_state.TurnState.STOPPED
                assert app._turn_timer is None
                assert not app._status_state.is_compacting
                assert (app._status_state.input_tokens, app._status_state.max_tokens) == context_tokens

            background_status = Mock()
            terminal_output = Mock()
            file_change = Mock()
            monkeypatch.setattr(app, "_note_sub_agent_status", background_status)
            monkeypatch.setattr(app, "_queue_terminal_output", terminal_output)
            monkeypatch.setattr(app, "_record_file_change_event", file_change)
            monkeypatch.setattr(app, "_apply_edit_preview", Mock())
            background_event = AgentEvent(
                sender="coder",
                event_type="llm_status_update",
                content={"text": "background still running"},
                sub_agent_info={"agent_id": "background"},
            )
            app._render_event(background_event)
            background_status.assert_called_once_with(background_event)
            app._render_event(AgentEvent(sender="terminal", event_type="terminal_output", content={"output": "data"}))
            terminal_output.assert_called_once_with("data")
            edit = AgentEvent(sender="coder", event_type="file_edit_preview", content={"path": "src/changed.py"})
            app._render_event(edit)
            file_change.assert_called_once_with(edit)
        finally:
            agent.release.set()
            await _wait_for(pilot, task.done)
            await _wait_for(pilot, lambda: not app._agent_execution_draining())


@pytest.mark.asyncio
async def test_control_request_during_detached_cleanup_defaults_without_opening_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    install_fake_agents(monkeypatch, coder_cls=_UnstoppableStreamAgent)
    app._stop_force_after_seconds = 60

    async with app.run_test() as pilot:
        agent = await _start_stuck_turn(app, pilot)
        task = app._agent_execution_task
        assert task is not None
        request_task = None
        try:
            app.action_cancel_generation()
            app.action_cancel_generation()
            assert app._agent_execution_draining()
            default = {"cancelled": True}
            request_task = asyncio.create_task(
                app.control_channel.request(
                    "question", {"question": "Late question?", "options": ["Yes", "No"]}, default=default
                )
            )
            assert await asyncio.wait_for(asyncio.shield(request_task), 1) == default
            assert not app.control_channel.pending()
            assert app._pending_question is None
            assert app._pending_approval is None
            clear_question = Mock()
            clear_approval = Mock()
            monkeypatch.setattr(app, "_clear_question_if_resolved", clear_question)
            monkeypatch.setattr(app, "_clear_approval_if_resolved", clear_approval)
            app._render_event(
                AgentEvent(sender="system", event_type=KnownEventType.CONTROL_RESOLVED, content={"request_id": "late"})
            )
            clear_question.assert_called_once_with("late")
            clear_approval.assert_called_once_with("late")
        finally:
            if request_task is not None:
                request_task.cancel()
                await asyncio.gather(request_task, return_exceptions=True)
            agent.release.set()
            await _wait_for(pilot, task.done)
            await _wait_for(pilot, lambda: not app._agent_execution_draining())


@pytest.mark.asyncio
@pytest.mark.parametrize("detached", [False, True], ids=["active-worker", "already-detached"])
async def test_quit_bounds_cancellation_resistant_turn_without_tearing_down_its_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, detached: bool
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    install_fake_agents(monkeypatch, coder_cls=_UnstoppableStreamAgent)
    app._stop_force_after_seconds = 60

    async with app.run_test() as pilot:
        agent = await _start_stuck_turn(app, pilot)
        task = app._agent_execution_task
        assert task is not None
        arm_watchdog = Mock()
        exit_app = Mock()
        monkeypatch.setattr(app, "_on_shutdown_requested", arm_watchdog)
        monkeypatch.setattr(app, "exit", exit_app)
        if detached:
            app.action_cancel_generation()
            app.action_cancel_generation()
            assert app.agent_worker is None
        app._stop_force_after_seconds = 0
        try:
            await asyncio.wait_for(app.action_quit(), 2)
            arm_watchdog.assert_called_once_with()
            exit_app.assert_called_once_with()
            assert not task.done()
            assert app._agent_execution_draining()
            assert app.agent is agent
            assert agent.cleanup_calls == 0
            assert not app._quit_cleanly
            await app.action_quit()
            arm_watchdog.assert_called_once_with()
        finally:
            agent.release.set()
            await _wait_for(pilot, task.done)
            await _wait_for(pilot, lambda: not app._agent_execution_draining())
            await _wait_for(pilot, lambda: not app._shutdown_worker_waiters)
