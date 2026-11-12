from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from pathlib import Path
import threading
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from textual.pilot import Pilot

from kolega_code.agent.goal import GoalVerdict
from kolega_code.cli.goal import GoalState, build_goal_task_prompt
from kolega_code.cli.session_event_store import FileSessionEventStore
from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.tui.widgets import ChatComposer
from kolega_code.events import AgentEvent
from kolega_code.session.projection import replay
from tests.cli.test_app_goal_command import GoalFakeAgent, _build_goal_test_app
from tests.cli.test_app_handoff import _build_handoff_test_app
from tests.cli.test_ask_goal import GoalAskFakeAgent, _reset_fake, _setup_project
from tests.cli.test_ask_teardown import _claim_in_child

pytestmark = [pytest.mark.usefixtures("isolated_cli_env", "hermetic_git_config")]


async def _events(app: Any, session_id: str | None = None) -> list[AgentEvent]:
    session_id = str(session_id or app.session.session_id)
    return [
        event
        for event in await FileSessionEventStore(app.store.journal(session_id)).read(session_id)
        if event.event_type == "goal_updated"
    ]


async def _flush(app: Any) -> None:
    assert app._goal_presentation is not None
    await app._goal_presentation.flush()


@pytest.mark.asyncio
async def test_tui_start_pause_implicit_resume_evaluations_and_completion_preserve_counting_and_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kolega_code.llm.usage import normalize_usage

    class LedgerGoalAgent(GoalFakeAgent):
        def record(self, tokens: int) -> None:
            ledger = self.kwargs["usage_ledger"]
            request = ledger.begin("openai", "gpt-5.4-mini")
            ledger.record_response(
                request, normalize_usage({"prompt_tokens": tokens, "completion_tokens": 0}, "openai", "gpt-5.4-mini")
            )

        async def process_message_stream(self, message: str, attachments: Any = None) -> Any:
            self.record(100)
            async for chunk in super().process_message_stream(message, attachments):
                yield chunk

        async def evaluate_goal_condition(self, condition: str) -> GoalVerdict:
            self.record(7)
            return await super().evaluate_goal_condition(condition)

    app = _build_goal_test_app(tmp_path, monkeypatch, agent_cls=LedgerGoalAgent)
    async with app.run_test():
        await app._startup_complete.wait()
        assert app.agent is not None
        agent = GoalFakeAgent.instances[-1]
        agent._goal_evaluate_results = [GoalVerdict(met=False, reason="gap"), GoalVerdict(met=True, reason="done")]
        assert not await app._activate_goal("all deliverables")
        await app._pause_goal("Stopped", status_note="literal <stop>")
        await app._process_message("continue")
        await _flush(app)
        observed = await _events(app)
        assert [event.content["reason"] for event in observed] == [
            "snapshot",
            "started",
            "paused",
            "resumed",
            "evaluated",
            "evaluated",
            "completed",
        ]
        assert [event.content["evaluation_met"] for event in observed] == [None, None, None, None, False, True, True]
        assert observed[0].content["goal"] is None
        assert observed[2].content["goal"]["status_note"] == "literal <stop>"
        assert observed[3].content["goal"]["paused"] is False
        assert observed[3].content["goal"]["status_note"] == ""
        assert [event.content["goal"]["turns_evaluated"] for event in observed[4:]] == [1, 1, 1]
        assert [event.content["goal"]["tokens_spent"] for event in observed[4:]] == [100, 207, 214]
        assert observed[-1].content["goal"]["met"] is True
        assert observed[-1].content["goal"] == app.session.goal
        assert app.store.load(app.session.session_id).goal == app.session.goal
        assert agent._evaluate_calls == ["all deliverables", "all deliverables"]
        assert agent._stream_call_count == 2 and agent.messages[0] == "continue"
        assert app._usage_ledger.snapshot().total_tokens == 214
        assert all(event.sub_agent_info is None and event.content["source"] == "tui" for event in observed)
        await app.action_quit()


@pytest.mark.asyncio
async def test_presentation_alone_changes_no_history_resume_usage_atif_transcript_or_goal_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kolega_code.cli.atif_export import export_atif_to_text
    from kolega_code.cli.session_usage import derive_session_usage

    app = _build_goal_test_app(tmp_path, monkeypatch)
    async with app.run_test():
        await app._startup_complete.wait()
        await app._activate_goal("actual state")
        assert app._goal is not None and app.agent is not None
        await _flush(app)
        journal = app.store.journal(app.session.session_id)
        original_goal = app._goal.to_dict()
        semantic = [event.to_dict() for event in journal.read_events() if not event.event_type.startswith("ui.")]
        history = deepcopy(app.agent.history)
        resume = app.store._replay(journal.read_events(), journal)
        ledger = app._usage_ledger.snapshot()
        usage = derive_session_usage(journal.read_events())
        atif = export_atif_to_text(journal, kolega_version="test", secret_values=(), state_dirs=())
        transcript = list(app.conversation_entries)
        app._publish_goal("evaluated", evaluation_met=False)
        app._publish_goal("paused")
        app._publish_goal("completed")
        await _flush(app)
        assert app._goal.to_dict() == original_goal and app.session.goal == original_goal
        assert app.agent.history == history and app._usage_ledger.snapshot() == ledger
        assert app.store._replay(journal.read_events(), journal) == resume
        assert derive_session_usage(journal.read_events()) == usage
        assert export_atif_to_text(journal, kolega_version="test", secret_values=(), state_dirs=()) == atif
        assert [
            event.to_dict() for event in journal.read_events() if not event.event_type.startswith("ui.")
        ] == semantic
        assert app.conversation_entries == transcript
        write_log = Mock()
        monkeypatch.setattr(app, "_write_log", write_log)
        app._render_event(
            AgentEvent(
                sender="cli",
                event_type="goal_updated",
                content={"goal": None},
                sub_agent_info={"agent_id": "untrusted-child"},
            )
        )
        write_log.assert_not_called()
        assert app.conversation_entries == transcript and not app._sub_agent_activities
        await app.action_quit()


@pytest.mark.asyncio
async def test_replacement_clear_reset_and_restored_snapshot_do_not_invent_prior_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_goal_test_app(tmp_path, monkeypatch)
    saved = GoalState(
        condition="restored goal",
        started_at="original-start",
        turns_evaluated=4,
        tokens_spent=123,
        paused=True,
        last_reason="prior gap",
        status_note="prior stop",
    )
    app._goal = saved
    app.session.goal = saved.to_dict()
    async with app.run_test():
        await app._startup_complete.wait()
        first = await _events(app)
        assert len(first) == 1 and first[0].content["reason"] == "snapshot"
        assert first[0].content["goal"] == saved.to_dict()
        assert first[0].content["evaluation_met"] is None
        assert not GoalFakeAgent.instances[-1]._evaluate_calls
        assert await app._activate_goal("replacement")
        with pytest.raises(ValueError):
            await app._activate_goal(" \n ")
        await app._clear_active_goal()
        assert not await app._activate_goal("after clear")
        await app._reset_current_thread()
        await _flush(app)
        observed = await _events(app)
        assert [event.content["reason"] for event in observed] == [
            "snapshot",
            "replaced",
            "cleared",
            "started",
            "cleared",
        ]
        assert observed[1].content["goal"]["turns_evaluated"] == 0
        assert observed[1].content["goal"]["tokens_spent"] == 0
        assert observed[-1].content["goal"] is None
        assert not replay(observed).conversation
        await app.action_quit()


@pytest.mark.asyncio
async def test_handoff_keeps_old_goal_observations_on_captured_recorder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_handoff_test_app(tmp_path, monkeypatch)
    async with app.run_test():
        await app._startup_complete.wait()
        old_session = app.session.session_id
        await app._activate_goal("old context goal")
        old_owner = app._goal_presentation
        await app._handoff_session("successor context")
        await _flush(app)
        old = await _events(app, old_session)
        new = await _events(app)
        assert [event.content["reason"] for event in old] == ["snapshot", "started"]
        assert old[-1].content["goal"]["condition"] == "old context goal"
        assert all(event.session_id == old_session for event in old)
        assert len(new) == 1 and new[0].session_id == app.session.session_id
        assert new[0].content["reason"] == "snapshot" and new[0].content["goal"] is None
        assert new[0].content["generation_id"] != old[0].content["generation_id"]
        assert app._goal_presentation is not old_owner and app._goal is None
        await app.action_quit()


@pytest.mark.asyncio
async def test_failed_observer_cannot_change_goal_or_stop_existing_verifier_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_goal_test_app(tmp_path, monkeypatch)
    async with app.run_test():
        await app._startup_complete.wait()
        monkeypatch.setattr(app._goal_presentation, "_emit", AsyncMock(side_effect=OSError("observer failed")))
        await app._activate_goal("verified normally")
        await app._process_message("work")
        await _flush(app)
        assert app._goal is not None
        assert app._goal.met is True and app._goal.turns_evaluated == 0
        assert GoalFakeAgent.instances[-1]._evaluate_calls == ["verified normally"]
        await app.action_quit()


@pytest.mark.asyncio
async def test_real_launcher_records_goal_and_drains_presentation_without_retaining_turn_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_goal_test_app(tmp_path, monkeypatch)
    loop = asyncio.get_running_loop()
    original_factory = loop.get_task_factory()

    async def work(_pilot: Pilot) -> None:
        await app._startup_complete.wait()
        eager_factory = getattr(asyncio, "eager_task_factory", None)
        if eager_factory is not None:
            assert loop.get_task_factory() is eager_factory
        await app._activate_goal("real launcher goal")
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("work")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        worker = app.agent_worker
        assert worker is not None
        await worker.wait()
        assert app.agent_worker is None and not app._turn_active
        assert GoalFakeAgent.instances[-1]._evaluate_calls == ["real launcher goal"]
        await app.action_quit()

    try:
        await asyncio.wait_for(app.run_async(headless=True, auto_pilot=work), 30)
    finally:
        loop.set_task_factory(original_factory)
    observed = await _events(app)
    assert [event.content["reason"] for event in observed] == ["snapshot", "started", "evaluated", "completed"]
    assert observed[-1].content["goal"]["met"] is True
    assert app._goal_presentation is not None and app._goal_presentation._worker is None


@pytest.mark.asyncio
async def test_stale_verifier_owner_does_not_record_a_new_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_goal_test_app(tmp_path, monkeypatch)
    async with app.run_test():
        await app._startup_complete.wait()
        await app._activate_goal("owner goal")
        agent = GoalFakeAgent.instances[-1]
        agent._verifier_started, agent._release_verifier = asyncio.Event(), asyncio.Event()
        running = asyncio.create_task(app._run_goal_loop())
        await agent._verifier_started.wait()
        replacement = Mock()
        app.agent_worker = replacement
        agent._release_verifier.set()
        await running
        await _flush(app)
        assert app._goal is not None
        assert [event.content["reason"] for event in await _events(app)] == ["snapshot", "started"]
        assert app._goal.last_reason == "" and app._goal.last_evaluated_at is None
        app.agent_worker = None
        await app.action_quit()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["cap", "failed-nudge", "cancelled-nudge", "cancelled-verifier"])
async def test_tui_existing_stop_boundaries_record_pause_without_another_work_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    app = _build_goal_test_app(tmp_path, monkeypatch)
    async with app.run_test():
        await app._startup_complete.wait()
        await app._activate_goal("stop boundary")
        assert app._goal is not None
        agent = GoalFakeAgent.instances[-1]
        agent._goal_evaluate_results = [GoalVerdict(met=False, reason="gap")]
        if outcome == "cap":
            app._goal.max_turns = 1
        elif outcome == "failed-nudge":
            agent._fail_on_calls = {2}
        elif outcome == "cancelled-nudge":
            agent._cancel_on_calls = {2}
        if outcome == "cancelled-verifier":
            agent._verifier_started, agent._release_verifier = asyncio.Event(), asyncio.Event()
            running = asyncio.create_task(app._process_message("work"))
            await agent._verifier_started.wait()
            running.cancel()
            await running
        else:
            await app._process_message("work")
        await _flush(app)
        observed = await _events(app)
        assert [event.content["reason"] for event in observed] == (
            ["snapshot", "started", "paused"]
            if outcome == "cancelled-verifier"
            else ["snapshot", "started", "evaluated", "paused"]
        )
        final = observed[-1].content["goal"]
        assert final["paused"] is True and final["met"] is False
        assert final == app.session.goal == app.store.load(app.session.session_id).goal
        assert len(agent._evaluate_calls) == 1
        assert agent._stream_call_count == (1 if outcome in ("cap", "cancelled-verifier") else 2)
        if outcome == "cap":
            assert final["status_note"] == "Reached the turn cap (1)."
        elif outcome == "failed-nudge":
            assert "provider rejected the request" in final["status_note"]
        else:
            assert final["status_note"] == "Stopped by user."
        await app.action_quit()


@pytest.mark.asyncio
async def test_headless_goal_writer_drains_before_terminal_export_and_cross_process_lock_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kolega_code.cli import main as main_module

    _reset_fake(monkeypatch)
    project = _setup_project(tmp_path, monkeypatch)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", {})
    monkeypatch.setattr(main_module, "_store_from_args", lambda _args: store)
    monkeypatch.setattr(main_module, "start_headless_peer_inbox", AsyncMock(return_value=None))
    entered, release = threading.Event(), threading.Event()
    append = FileSessionEventStore._append_sync

    def blocked(events: FileSessionEventStore, event: AgentEvent) -> int:
        if event.event_type == "goal_updated" and event.content["reason"] == "started":
            entered.set()
            assert release.wait(30)
        return append(events, event)

    monkeypatch.setattr(FileSessionEventStore, "_append_sync", blocked)
    output = tmp_path / "goal-trajectory.json"
    args = main_module.parse_args(
        [
            "ask",
            "--goal",
            "owned goal",
            "--project",
            str(project),
            "--session",
            session.session_id,
            "--atif-output",
            str(output),
        ]
    )
    command = asyncio.create_task(main_module._run_ask(args))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        command.cancel()
        await asyncio.sleep(0)
        command.cancel()
        await asyncio.sleep(0)
        assert not command.done()
        assert await _claim_in_child(store.root, session.session_id, project) == 3
        assert not output.exists()
        raw = store.events_path_for(session.session_id).read_text()
        assert not any(json.loads(line)["type"].startswith("run.") for line in raw.splitlines())
        release.set()
        assert await asyncio.wait_for(command, 20) == 0
        assert await _claim_in_child(store.root, session.session_id, project) == 0
        events = await FileSessionEventStore(store.journal(session.session_id)).read(session.session_id)
        goals = [event for event in events if event.event_type == "goal_updated"]
        assert [event.content["reason"] for event in goals] == ["started", "evaluated", "completed"]
        terminal = next(
            event for event in store.journal(session.session_id).read_events() if event.event_type == "run.completed"
        )
        assert all(event.seq is not None and event.seq < terminal.seq for event in goals)
        trajectory = json.loads(output.read_text())
        assert trajectory["extra"]["kolega"]["status"]["outcome"] == "completed"
        assert GoalAskFakeAgent.instances[-1].evaluate_calls == 1
    finally:
        release.set()
        if not command.done():
            command.cancel()
        await asyncio.wait_for(asyncio.gather(command, return_exceptions=True), 20)
        store.release_session_locks()


@pytest.mark.parametrize("json_mode", [False, True])
@pytest.mark.parametrize("outcome", ["met", "cap", "failed-first", "failed-nudge"])
def test_headless_observations_preserve_verifier_nudge_terminal_and_gapless_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any, json_mode: bool, outcome: str
) -> None:
    from kolega_code.cli import main as main_module

    _reset_fake(monkeypatch)
    project = _setup_project(tmp_path, monkeypatch)
    GoalAskFakeAgent.evaluate_queue = [GoalVerdict(met=False, reason="gap"), GoalVerdict(met=True, reason="done")]
    if outcome == "cap":
        GoalAskFakeAgent.evaluate_queue = []
        GoalAskFakeAgent.default_met = False
    elif outcome.startswith("failed"):
        GoalAskFakeAgent.fail_on_stream_calls = {1 if outcome == "failed-first" else 2}
    args = ["ask", "--goal", "deliverables", "--goal-max-turns", "2", "--project", str(project), "--save"]
    if json_mode:
        args.append("--json")
    code = main_module.main(args)
    assert code == (0 if outcome == "met" else 1)
    agent = GoalAskFakeAgent.instances[-1]
    expected_calls = {"met": (2, 2), "cap": (2, 3), "failed-first": (0, 1), "failed-nudge": (1, 2)}
    assert (agent.evaluate_calls, agent.stream_calls) == expected_calls[outcome]
    if outcome != "failed-first":
        assert agent.messages[0] == build_goal_task_prompt("deliverables")
    output = capsys.readouterr().out
    store = SessionStore()
    sessions = store.list(recover=False)
    assert len(sessions) == 1
    journal = store.journal(sessions[0].session_id)
    raw = journal.read_events(repair_tail=False)
    ui = asyncio.run(FileSessionEventStore(journal).read(sessions[0].session_id))
    goals = [event for event in ui if event.event_type == "goal_updated"]
    if json_mode:
        assert not goals
        lines = [json.loads(line) for line in output.splitlines() if line.strip()]
        assert [line["seq"] for line in lines] == list(range(1, len(lines) + 1))
        assert [line["seq"] for line in lines] == [event.seq for event in raw]
        assert lines[-1]["type"] == ("run.failed" if outcome.startswith("failed") else "run.completed")
        assert not any(line["type"].startswith("ui.") for line in lines)
    else:
        expected = {
            "met": ["started", "evaluated", "evaluated", "completed"],
            "cap": ["started", "evaluated", "evaluated", "completed"],
            "failed-first": ["started", "paused"],
            "failed-nudge": ["started", "evaluated", "paused"],
        }
        assert [event.content["reason"] for event in goals] == expected[outcome]
        assert all(event.content["source"] == "ask" and event.sub_agent_info is None for event in goals)
        final = goals[-1].content["goal"]
        assert final["met"] is (outcome == "met")
        assert final["paused"] is outcome.startswith("failed")
        if outcome == "cap":
            assert final["turns_evaluated"] == 2 and final["paused"] is False
        if outcome.startswith("failed"):
            assert "provider rejected the request" in final["status_note"]
            assert store.load(sessions[0].session_id).goal == final
        assert not replay(goals).conversation
    terminals = [event for event in raw if event.event_type == "goal.completed"]
    assert len(terminals) == 1
    assert terminals[0].payload["met"] is (outcome == "met")
