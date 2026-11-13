from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
import json
from pathlib import Path
import threading
from typing import Any, Callable, cast

import pytest
from textual.pilot import Pilot
from textual.widgets import OptionList

from kolega_code.agent.goal import GoalVerdict
from kolega_code.cli.app import KolegaCodeApp
from kolega_code.cli.config import config_summary
from kolega_code.cli.plan_artifacts import current_plan_artifact_path
from kolega_code.cli.provider_registry import UI_DEFAULT_MODEL, UI_DEFAULT_PROVIDER
from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.tui.constants import TUI_CLIENT_ID
from kolega_code.cli.tui.plan_approval import PlanApprovalOwner
from kolega_code.cli.tui.widgets import ActionList, ChatComposer, PlanningMarkdown
from kolega_code.events import AgentEvent, KnownEventType
from kolega_code.llm.models import Message, TextBlock
from kolega_code.permissions import PermissionMode
from kolega_code.session.control import ControlRequest
from kolega_code.session.plans import PLAN_APPROVAL_OPTIONS, PlanApprovalAction, plan_approval_payload

from ._app_test_utils import (
    FakeBuildAgentWithRegistry,
    FakePlanAgentWithRegistry,
    build_test_config,
    install_fake_agents,
    wait_for_layout,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("isolated_cli_env")]

PLAN = "# Full plan\n\n## Design\n\n- Keep **all** Markdown.\n- Ship once.\n"


class PlanAgent(FakePlanAgentWithRegistry):
    completed_plan: str | None = None

    def consume_completed_plan(self) -> str | None:
        plan, self.completed_plan = self.completed_plan, None
        return plan


def _app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, restored: bool = False) -> KolegaCodeApp:
    install_fake_agents(monkeypatch, coder_cls=FakeBuildAgentWithRegistry, planning_cls=PlanAgent)
    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    if restored:
        session.interaction_mode = "plan"
        session.latest_plan_markdown = PLAN
        session.plan_pending = True
        session.plan_reofferable = True
        store.save(session)
    return KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)


async def _wait(predicate: Callable[[], object]) -> None:
    async with asyncio.timeout(15):
        while not predicate():
            await asyncio.sleep(0.001)


def _requests(app: KolegaCodeApp) -> list[ControlRequest]:
    return [request for request in app.control_channel.pending() if request.kind == "plan_approval"]


async def _request(app: KolegaCodeApp) -> ControlRequest:
    await _wait(lambda: _requests(app))
    requests = _requests(app)
    assert len(requests) == 1
    return requests[0]


def _owner(app: KolegaCodeApp) -> PlanApprovalOwner:
    owner = app._plan_approval_owner
    assert owner is not None and owner.task is not None
    assert owner.task in app._plan_approval_tasks
    return owner


async def _joined(owner: PlanApprovalOwner) -> None:
    assert owner.task is not None
    await asyncio.wait_for(asyncio.shield(owner.task), 15)


async def _capture(app: KolegaCodeApp) -> PlanApprovalOwner:
    assert isinstance(app.agent, PlanAgent)
    app.agent.completed_plan = PLAN
    await app._capture_completed_plan()
    return _owner(app)


async def _submit(app: KolegaCodeApp, text: str) -> None:
    composer = app.query_one("#composer", ChatComposer)
    composer.load_text(text)
    await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, text))


def _observe(app: KolegaCodeApp, monkeypatch: pytest.MonkeyPatch) -> list[AgentEvent]:
    events: list[AgentEvent] = []
    emit = app.control_channel._emit

    async def observe(event: AgentEvent) -> None:
        events.append(event)
        await emit(event)

    monkeypatch.setattr(app.control_channel, "_emit", observe)
    return events


async def _choose(app: KolegaCodeApp, pilot: Pilot, action: PlanApprovalAction) -> None:
    actions = app.query_one("#plan_actions", ActionList)
    await wait_for_layout(pilot, lambda: actions.display and actions.region.width > 0)
    actions.focus()
    actions.highlighted = next(index for index, (value, _) in enumerate(PLAN_APPROVAL_OPTIONS) if value == action)
    await pilot.press("enter")


@asynccontextmanager
async def _task_factory(eager: bool) -> AsyncIterator[None]:
    factory = getattr(asyncio, "eager_task_factory", None)
    if eager and factory is None:
        pytest.skip("Eager tasks require Python 3.12+")
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    loop.set_task_factory(factory if eager else None)
    try:
        yield
    finally:
        loop.set_task_factory(previous)


@pytest.mark.parametrize("action", [value for value, _ in PLAN_APPROVAL_OPTIONS])
async def test_plan_choices_use_one_real_request_and_one_accepted_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: PlanApprovalAction
) -> None:
    app = _app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        await app.action_toggle_interaction_mode()
        events = _observe(app, monkeypatch)
        original = cast(PlanAgent, app.agent)
        prior = Message(role="user", content=[TextBlock("Prior planning history")])
        original.history.append(prior)
        app._session_recorder.record_context_message(prior)
        app._gigacode_enabled = True
        owner = await _capture(app)
        request = await _request(app)
        actions = app.query_one("#plan_actions", ActionList)
        assert [(option.id, str(option.prompt)) for option in actions.options] == list(PLAN_APPROVAL_OPTIONS)
        assert request.payload == plan_approval_payload(PLAN)
        assert request.default == {"action": None}
        assert app.query_one("#composer", ChatComposer).disabled

        await _choose(app, pilot, action)
        await _joined(owner)
        await _wait(lambda: app.agent_worker is None)
        await _wait(lambda: any(event.event_type == KnownEventType.CONTROL_RESOLVED for event in events))
        assert not app.session_runtime.answer_plan_approval(request.request_id, action, client_id=TUI_CLIENT_ID)
        assert len(events) == 2
        requested, resolved = events
        assert requested.event_type == KnownEventType.CONTROL_REQUESTED
        assert requested.content == {
            "request_id": request.request_id,
            "kind": "plan_approval",
            "payload": plan_approval_payload(PLAN),
            "has_controller": True,
        }
        assert resolved.event_type == KnownEventType.CONTROL_RESOLVED
        assert resolved.content == {
            "request_id": request.request_id,
            "kind": "plan_approval",
            "payload": plan_approval_payload(PLAN),
            "response": {"action": action},
            "reason": "answered",
        }
        assert app._latest_plan == PLAN
        assert app.query_one("#planning_plan_markdown", PlanningMarkdown).source == PLAN
        assert not actions.display
        assert not app._plan_pending
        assert not app._plan_decision_active
        assert not app._plan_approval_tasks
        saved = app.store.load(app.session.session_id)
        assert request.request_id not in json.dumps(asdict(saved))
        assert saved.latest_plan_markdown == PLAN
        if action == "discuss_plan":
            assert app.agent is original
            assert not original.messages
            assert app._plan_reofferable
            assert not app.query_one("#composer", ChatComposer).disabled
        else:
            build = cast(FakeBuildAgentWithRegistry, app.agent)
            assert app.interaction_mode == "build"
            assert len(build.messages) == 1
            assert PLAN.rstrip() in build.messages[0]
            assert not app._plan_reofferable
            artifact = current_plan_artifact_path(app.store.root, app.session.session_id)
            assert artifact.read_text() == PLAN.rstrip() + "\n"
            assert str(artifact) in build.messages[0]
            history = [message.get_text_content() for message in build.history]
            assert ("Prior planning history" in history) == (action == "implement_plan")


async def test_discussion_reoffers_and_identical_new_plan_has_a_new_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _app(tmp_path, monkeypatch)
    async with app.run_test():
        await app.action_toggle_interaction_mode()
        first = await _capture(app)
        first_request = await _request(app)
        assert app.session_runtime.answer_plan_approval(
            first_request.request_id, "discuss_plan", client_id=TUI_CLIENT_ID
        )
        await _joined(first)
        assert not cast(PlanAgent, app.agent).messages
        await app._process_message("Discuss the tradeoffs")
        second = _owner(app)
        second_request = await _request(app)
        assert second_request.request_id != first_request.request_id
        assert second.revision != first.revision
        assert app._plan_decision_active

        third = await _capture(app)
        third_request = await _request(app)
        assert second.task is not None and second.task.done()
        assert not second.valid
        assert third.plan == second.plan == first.plan
        assert third.revision > second.revision
        assert third_request.request_id != second_request.request_id
        assert not app.session_runtime.answer_plan_approval(
            second_request.request_id, "implement_plan", client_id=TUI_CLIENT_ID
        )
        for event_type in (KnownEventType.CONTROL_REQUESTED, KnownEventType.CONTROL_RESOLVED):
            app._consume_non_rendered_event(
                AgentEvent(
                    sender="system",
                    event_type=event_type,
                    session_id=app.session.session_id,
                    content={
                        "request_id": second_request.request_id,
                        "kind": "plan_approval",
                        "payload": plan_approval_payload(PLAN),
                        "response": {"action": "implement_plan"},
                    },
                )
            )
        assert app._plan_approval_owner is third
        assert app.query_one("#plan_actions").display
        assert app.control_channel.is_pending(third_request.request_id)


@pytest.mark.parametrize("eager", [False, True], ids=["lazy", "eager"])
@pytest.mark.parametrize("same_plan", [False, True], ids=["new-text", "same-text"])
@pytest.mark.parametrize("action", [value for value, _ in PLAN_APPROVAL_OPTIONS])
async def test_queued_plan_selection_cannot_approve_a_replacement_offer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: PlanApprovalAction, same_plan: bool, eager: bool
) -> None:
    app = _app(tmp_path, monkeypatch)
    async with app.run_test(), _task_factory(eager):
        await app.action_toggle_interaction_mode()
        old = await _capture(app)
        await _request(app)
        actions = app.query_one("#plan_actions", ActionList)
        index = next(index for index, (value, _) in enumerate(PLAN_APPROVAL_OPTIONS) if value == action)
        queued = OptionList.OptionSelected(actions, actions.get_option_at_index(index), index)
        replacement_plan = PLAN if same_plan else "# Replacement plan\n\nApprove only this revision."
        await app._show_plan_for_decision(replacement_plan, notification="Replacement plan ready.")
        owner, request = _owner(app), await _request(app)
        assert owner is not old and owner.revision > old.revision
        assert owner.plan == replacement_plan
        assert request.payload == plan_approval_payload(replacement_plan)
        assert actions.get_option_at_index(index) is not queued.option
        await app.on_option_list_option_selected(queued)
        assert app.control_channel.is_pending(request.request_id), "the old widget selection has no new authority"
        assert owner.action is None
        assert app._plan_pending and app._plan_decision_active
        assert not cast(PlanAgent, app.agent).messages
        await app.on_option_list_option_selected(
            OptionList.OptionSelected(actions, actions.get_option_at_index(index), index)
        )
        await _joined(owner)
        await _wait(lambda: app.agent_worker is None)
        assert not app._plan_pending
        if action == "discuss_plan":
            assert app.interaction_mode == "plan" and not cast(PlanAgent, app.agent).messages
        else:
            assert app.interaction_mode == "build"
            assert replacement_plan.rstrip() in cast(FakeBuildAgentWithRegistry, app.agent).messages[0]


@pytest.mark.parametrize("permission_mode", list(PermissionMode))
async def test_plan_approval_is_explicit_in_every_permission_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, permission_mode: PermissionMode
) -> None:
    app = _app(tmp_path, monkeypatch)
    app.permission_mode = permission_mode
    app.session_runtime.set_permission_mode(permission_mode)
    async with app.run_test():
        await app.action_toggle_interaction_mode()
        owner = await _capture(app)
        request = await _request(app)
        assert owner.action is None
        assert app._plan_pending and app._plan_decision_active
        assert app.permission_mode == permission_mode
        assert not cast(PlanAgent, app.agent).messages
        assert app.session_runtime.answer_plan_approval(request.request_id, "discuss_plan", client_id=TUI_CLIENT_ID)
        await _joined(owner)


async def test_plan_card_never_answers_an_unrelated_identical_text_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _app(tmp_path, monkeypatch)
    async with app.run_test():
        await app.action_toggle_interaction_mode()
        foreign = asyncio.create_task(app.session_runtime.request_plan_approval(PLAN))
        try:
            foreign_request = await _request(app)
            owner = await _capture(app)
            await _wait(lambda: owner.request_id is not None)
            assert owner.request_id != foreign_request.request_id
            assert await app._answer_pending_plan("discuss_plan")
            await _joined(owner)
            assert not foreign.done()
            assert app.control_channel.is_pending(foreign_request.request_id)
            assert not cast(PlanAgent, app.agent).messages
        finally:
            foreign.cancel()
            await asyncio.gather(foreign, return_exceptions=True)


@pytest.mark.parametrize("restored", [False, True])
async def test_delayed_plan_announcement_preserves_selection_and_editable_focus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restored: bool
) -> None:
    app = _app(tmp_path, monkeypatch, restored=restored)
    async with app.run_test():
        if not restored:
            await app.action_toggle_interaction_mode()
            await _capture(app)
        request = await _request(app)
        actions = app.query_one("#plan_actions", ActionList)
        actions.highlighted = 2
        if restored:
            app.query_one("#composer", ChatComposer).focus()
        focused = app.screen.focused
        app._consume_non_rendered_event(
            AgentEvent(
                sender="system",
                event_type=KnownEventType.CONTROL_REQUESTED,
                session_id=app.session.session_id,
                content={**request.describe(), "has_controller": True},
            )
        )
        assert actions.highlighted == 2
        assert app.screen.focused is focused


async def test_restored_plan_allows_editing_and_fences_the_old_request_before_followup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _app(tmp_path, monkeypatch, restored=True)
    async with app.run_test() as pilot:
        request = await _request(app)
        owner = _owner(app)
        composer = app.query_one("#composer", ChatComposer)
        assert not composer.disabled
        assert not app._plan_decision_active
        assert app.query_one("#plan_actions", ActionList).option_count == 3
        await wait_for_layout(pilot, lambda: composer.region.width > 0)
        app.screen.set_focus(composer)
        app._heal_prompt_focus()
        assert app.screen.focused is composer
        await pilot.press("r")
        assert composer.text == "r"

        await _submit(app, "Revise the approach")
        assert not app.session_runtime.answer_plan_approval(
            request.request_id, "implement_plan", client_id=TUI_CLIENT_ID
        )
        await _wait(lambda: app.agent_worker is None and app._plan_approval_owner is not None)
        next_request = await _request(app)
        assert next_request.request_id != request.request_id
        assert owner.task is not None and owner.task.done()
        assert app._latest_plan == PLAN
        assert cast(PlanAgent, app.agent).messages == ["Revise the approach"]
        assert app._plan_decision_active
        assert composer.disabled
        await wait_for_layout(pilot, lambda: app.query_one("#plan_actions").region.width > 0)
        assert app.query_one("#plan_actions", ActionList).option_count == 3


@pytest.mark.parametrize("transition", ["rebuild", "model", "settings", "mode", "reset", "handoff"])
async def test_rebuild_and_session_transitions_reject_old_plan_responses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transition: str
) -> None:
    app = _app(tmp_path, monkeypatch, restored=True)
    async with app.run_test():
        old_request = await _request(app)
        old_owner = _owner(app)
        old_runtime = app.session_runtime
        if transition == "rebuild":
            assert app.config is not None
            await app._build_agent(app.config, rebuild=True)
        elif transition == "model":
            app.settings.set_api_key(UI_DEFAULT_PROVIDER, "fake-key")
            await app._switch_model(UI_DEFAULT_PROVIDER, UI_DEFAULT_MODEL)
            assert app.agent is not old_owner.agent
        elif transition == "settings":
            candidate = replace(app.settings, active_provider=UI_DEFAULT_PROVIDER, active_model=UI_DEFAULT_MODEL)
            candidate.set_api_key(UI_DEFAULT_PROVIDER, "fake-key")
            applied, error = await app._apply_settings_candidate(candidate)
            assert applied, error
            assert app.agent is not old_owner.agent
        elif transition == "mode":
            await app._set_interaction_mode("build")
            assert app._plan_pending and app._plan_reofferable
            assert not _requests(app)
            await app._set_interaction_mode("plan")
        elif transition == "reset":
            await app._reset_current_thread()
        else:
            await app._handoff_session("A fresh session, not an approved plan.")
        assert old_owner.task is not None and old_owner.task.done()
        assert not old_owner.valid
        assert not old_runtime.answer_plan_approval(old_request.request_id, "implement_plan", client_id=TUI_CLIENT_ID)
        if transition in {"mode", "rebuild", "model", "settings"}:
            request = await _request(app)
            assert request.request_id != old_request.request_id
            assert app._plan_pending and app._plan_reofferable
            assert not app._plan_decision_active
            assert not app.query_one("#composer", ChatComposer).disabled
        else:
            assert not _requests(app)
            assert not app._plan_approval_tasks
            assert not app._plan_pending
            assert app._latest_plan is None
        assert app.interaction_mode == "plan"
        assert not cast(PlanAgent, app.agent).messages


async def test_live_rebuild_preserves_fresh_plan_typing_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app(tmp_path, monkeypatch)
    async with app.run_test():
        await app.action_toggle_interaction_mode()
        owner = await _capture(app)
        request = await _request(app)
        assert app.config is not None
        await app._build_agent(app.config, rebuild=True)
        replacement = await _request(app)
        assert replacement.request_id != request.request_id
        assert owner.task is not None and owner.task.done()
        assert app._plan_decision_active
        assert app.query_one("#composer", ChatComposer).disabled


@pytest.mark.parametrize("action", [value for value, _ in PLAN_APPROVAL_OPTIONS])
async def test_early_response_waits_for_origin_save_and_actual_worker_teardown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: PlanApprovalAction
) -> None:
    app = _app(tmp_path, monkeypatch)
    save_entered, save_release = asyncio.Event(), asyncio.Event()
    teardown_entered, teardown_release = asyncio.Event(), asyncio.Event()
    async with app.run_test():
        await app.action_toggle_interaction_mode()
        origin_agent = cast(PlanAgent, app.agent)
        origin_agent.completed_plan = PLAN
        original_save = app._save_session_async
        original_process = app._process_message

        async def hold_plan_save() -> None:
            if app._plan_pending and not save_entered.is_set():
                save_entered.set()
                await save_release.wait()
            await original_save()

        async def hold_origin_teardown() -> None:
            await original_process("Write the plan")
            teardown_entered.set()
            await teardown_release.wait()

        monkeypatch.setattr(app, "_save_session_async", hold_plan_save)
        app.agent_worker = app.run_worker(hold_origin_teardown(), group="turns")
        origin_worker = app.agent_worker
        try:
            await asyncio.wait_for(save_entered.wait(), 15)
            owner = _owner(app)
            request = await _request(app)
            assert app.session_runtime.answer_plan_approval(request.request_id, action, client_id=TUI_CLIENT_ID)
            await _wait(lambda: owner.action == action)
            assert not owner.ready.is_set()
            assert app.agent_worker is origin_worker
            assert app.agent is origin_agent
            save_release.set()
            await asyncio.wait_for(teardown_entered.wait(), 15)
            assert owner.task is not None and not owner.task.done()
            assert app.agent is origin_agent
            assert app._plan_pending
            teardown_release.set()
            await _joined(owner)
            await _wait(lambda: app.agent_worker is None)
            if action != "discuss_plan":
                assert len(cast(FakeBuildAgentWithRegistry, app.agent).messages) == 1
                assert app.interaction_mode == "build"
            else:
                assert app.agent is origin_agent
                assert origin_agent.messages == ["Write the plan"]
        finally:
            save_release.set()
            teardown_release.set()
            await origin_worker.wait()


async def test_plan_renders_and_origin_releases_before_a_stalled_control_announcement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _app(tmp_path, monkeypatch)
    entered, release = asyncio.Event(), asyncio.Event()
    async with app.run_test() as pilot:
        await app.action_toggle_interaction_mode()
        emit = app.control_channel._emit

        async def stalled_announcement(event: AgentEvent) -> None:
            await emit(event)
            if event.event_type == KnownEventType.CONTROL_REQUESTED:
                entered.set()
                await release.wait()

        monkeypatch.setattr(app.control_channel, "_emit", stalled_announcement)
        cast(PlanAgent, app.agent).completed_plan = PLAN
        await _submit(app, "Write a plan")
        try:
            await asyncio.wait_for(entered.wait(), 15)
            await _wait(lambda: app.agent_worker is None)
            actions = app.query_one("#plan_actions", ActionList)
            await wait_for_layout(pilot, lambda: actions.display and actions.region.width > 0)
            assert app.query_one("#planning_plan_markdown", PlanningMarkdown).source == PLAN
            assert any(entry.kind == "plan" and entry.content == PLAN for entry in app.conversation_entries)
            owner = _owner(app)
            await _choose(app, pilot, "discuss_plan")
            assert owner.action is None
            assert not await app._answer_pending_plan("implement_plan")
            release.set()
            await _joined(owner)
            assert app._plan_reofferable and not app._plan_pending
        finally:
            release.set()


async def test_plan_timeout_transparently_renews_without_discussing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _app(tmp_path, monkeypatch)
    async with app.run_test():
        await app.action_toggle_interaction_mode()
        events = _observe(app, monkeypatch)
        app.control_channel.timeout = 0.05
        owner = await _capture(app)
        await _wait(lambda: sum(event.event_type == KnownEventType.CONTROL_REQUESTED for event in events) >= 2)
        app.control_channel.timeout = 30
        request = await _request(app)
        assert app._plan_approval_owner is owner
        assert app._plan_pending and app._plan_decision_active
        assert app.query_one("#plan_actions").display
        assert not cast(PlanAgent, app.agent).messages
        assert any(
            event.event_type == KnownEventType.CONTROL_RESOLVED
            and event.content["reason"] == "timeout"
            and event.content["response"] == {"action": None}
            for event in events
        )
        assert app.session_runtime.answer_plan_approval(request.request_id, "discuss_plan", client_id=TUI_CLIENT_ID)
        await _joined(owner)
        assert app._plan_reofferable and not app._plan_pending


async def test_accepted_plan_waits_for_a_timed_out_durable_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _app(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    async with app.run_test():
        await app.action_toggle_interaction_mode()
        original_save = app.store.save

        def held_save(record: Any) -> None:
            if record.plan_pending and not entered.is_set():
                entered.set()
                if not release.wait(30):
                    raise TimeoutError("The test did not release the plan save")
            original_save(record)

        monkeypatch.setattr(app.store, "save", held_save)
        app._turn_save_timeout_seconds = 0.03
        cast(PlanAgent, app.agent).completed_plan = PLAN
        await _submit(app, "Write a durable plan")
        try:
            await _wait(entered.is_set)
            await _wait(lambda: app.agent_worker is None)
            owner = _owner(app)
            request = await _request(app)
            assert owner.ready.is_set()
            assert app._pending_session_saves
            assert app.session_runtime.answer_plan_approval(
                request.request_id, "implement_plan", client_id=TUI_CLIENT_ID
            )
            await _wait(lambda: owner.action == "implement_plan")
            assert owner.task is not None and not owner.task.done()
            assert app.interaction_mode == "plan"
            assert app._plan_pending
            assert cast(PlanAgent, app.agent).messages == ["Write a durable plan"]
            release.set()
            await _joined(owner)
            await _wait(lambda: app.agent_worker is None)
            assert app.interaction_mode == "build"
            assert len(cast(FakeBuildAgentWithRegistry, app.agent).messages) == 1
            saved = app.store.load(app.session.session_id)
            assert not saved.plan_pending and saved.interaction_mode == "build"
        finally:
            release.set()


@pytest.mark.parametrize("eager", [False, True], ids=["lazy", "eager"])
@pytest.mark.parametrize("action", [value for value, _ in PLAN_APPROVAL_OPTIONS])
async def test_goal_revision_releases_accepted_owner_without_cancelling_its_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: PlanApprovalAction, eager: bool
) -> None:
    app = _app(tmp_path, monkeypatch)
    second_entered, second_release = asyncio.Event(), asyncio.Event()
    verifier_entered, verifier_release = asyncio.Event(), asyncio.Event()
    async with app.run_test(), _task_factory(eager):
        await app.action_toggle_interaction_mode()
        agent = cast(PlanAgent, app.agent)
        stream = agent.process_message_stream
        calls = 0

        async def produce(message: str, attachments: list[dict] | None = None) -> AsyncIterator[dict[str, Any]]:
            nonlocal calls
            calls += 1
            if calls == 2:
                second_entered.set()
                await second_release.wait()
            agent.completed_plan = f"# Goal plan revision {calls}"
            async for chunk in stream(message, attachments):
                yield chunk

        async def evaluate(_condition: str) -> GoalVerdict:
            if calls == 2:
                verifier_entered.set()
                await verifier_release.wait()
                return GoalVerdict(met=True, reason="The planning goal is complete")
            return GoalVerdict(met=False, reason="Another planning pass is needed")

        monkeypatch.setattr(agent, "process_message_stream", produce)
        monkeypatch.setattr(agent, "evaluate_goal_condition", evaluate, raising=False)
        await app._activate_goal("Finish planning")
        await _submit(app, "Plan the work")
        origin_worker = app.agent_worker
        assert origin_worker is not None
        first: PlanApprovalOwner | None = None
        try:
            await asyncio.wait_for(second_entered.wait(), 15)
            first = _owner(app)
            request = await _request(app)
            origin = app._agent_execution_task
            assert origin is not None and first.origin_task is origin
            assert app.session_runtime.answer_plan_approval(request.request_id, action, client_id=TUI_CLIENT_ID)
            await _wait(lambda: first.action == action)
            second_release.set()
            await _joined(first)
            await asyncio.wait_for(verifier_entered.wait(), 15)
            current = _owner(app)
            current_request = await _request(app)
            assert not first.valid
            assert first.task is not None and not first.task.cancelled()
            assert current is not first and current.revision > first.revision
            assert current.plan == "# Goal plan revision 2"
            assert current.origin_task is origin
            assert current_request.request_id != request.request_id
            assert not app.session_runtime.answer_plan_approval(request.request_id, action, client_id=TUI_CLIENT_ID)
            assert not origin.done() and not origin.cancelling()
            assert app.agent_worker is origin_worker
            assert app.agent is agent
            assert app.interaction_mode == "plan"
            assert not agent.clear_history_calls
            assert len(agent.messages) == 2
            assert not any(entry.content == "Implement the approved plan." for entry in app.conversation_entries)

            assert app.session_runtime.answer_plan_approval(current_request.request_id, action, client_id=TUI_CLIENT_ID)
            await _wait(lambda: current.action == action)
            assert current.task is not None and not current.task.done()
            verifier_release.set()
            await _joined(current)
            await origin_worker.wait()
            await _wait(lambda: app.agent_worker is None)
            assert origin.done() and not origin.cancelled() and not origin.cancelling()
            if action == "discuss_plan":
                assert app.agent is agent
                assert len(agent.messages) == 2
                assert not app._plan_pending and app._plan_reofferable
            else:
                assert app.interaction_mode == "build"
                assert len(cast(FakeBuildAgentWithRegistry, app.agent).messages) == 1
                assert "# Goal plan revision 2" in cast(FakeBuildAgentWithRegistry, app.agent).messages[0]
        finally:
            second_release.set()
            verifier_release.set()
            if first is not None and first.task is not None and not first.task.done():
                first.task.cancel()
                await asyncio.gather(first.task, return_exceptions=True)
            await origin_worker.wait()


@pytest.mark.parametrize("eager", [False, True], ids=["lazy", "eager"])
@pytest.mark.parametrize("restored", [False, True], ids=["fresh", "restored"])
@pytest.mark.parametrize("action", [value for value, _ in PLAN_APPROVAL_OPTIONS])
async def test_failed_retained_plan_writer_restores_offer_and_requires_an_explicit_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: PlanApprovalAction, restored: bool, eager: bool
) -> None:
    app = _app(tmp_path, monkeypatch, restored=restored)
    entered, release = threading.Event(), threading.Event()
    notices: list[str] = []
    async with app.run_test(), _task_factory(eager):
        if not restored:
            await app.action_toggle_interaction_mode()
        agent = cast(PlanAgent, app.agent)
        original_save = app.store.save
        writes: list[bool] = []

        def failed_save(record: Any) -> None:
            if record.plan_pending and not entered.is_set():
                entered.set()
                if not release.wait(30):
                    raise TimeoutError("The test did not release the failed plan save")
                raise OSError("held plan writer failed")
            original_save(record)
            writes.append(record.plan_pending)

        def notify(message: str, **_kwargs: Any) -> None:
            notices.append(message)

        monkeypatch.setattr(app.store, "save", failed_save)
        monkeypatch.setattr(app, "_notify_user", notify)
        app._turn_save_timeout_seconds = 0.03
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("Keep this draft")
        try:
            if restored:
                owner = _owner(app)
                assert not await app._save_session_with_deadline(app._save_session_async())
            else:
                owner = await _capture(app)
            request = await _request(app)
            assert entered.is_set() and app._pending_session_saves
            saves = list(app._pending_session_saves)
            assert app.session_runtime.answer_plan_approval(request.request_id, action, client_id=TUI_CLIENT_ID)
            await _wait(lambda: owner.action == action)
            assert owner.task is not None and not owner.task.done()
            assert all(not save.done() and not save.cancelling() for save in saves)
            release.set()
            result = await asyncio.gather(owner.task, return_exceptions=True)
            assert isinstance(result[0], OSError)
            assert not owner.valid and owner.invalidated.is_set()
            assert app._plan_approval_owner is not None, "a failed prerequisite must leave an actionable offer"
            retry_owner = _owner(app)
            retry_request = await _request(app)
            assert retry_owner is not owner and retry_owner.revision > owner.revision
            assert retry_request.request_id != request.request_id
            assert app.query_one("#plan_actions", ActionList).display
            assert composer.disabled is (not restored)
            assert composer.text == "Keep this draft"
            assert app._plan_decision_active is (not restored)
            assert app._plan_pending and app._plan_reofferable
            assert app.agent is agent and app.agent_worker is None
            assert not agent.messages and not agent.clear_history_calls
            assert not any(entry.content == "Implement the approved plan." for entry in app.conversation_entries)
            assert any("held plan writer failed" in notice and "retry" in notice for notice in notices)
            assert all(save.done() and not save.cancelled() for save in saves)
            assert not app.session_runtime.answer_plan_approval(request.request_id, action, client_id=TUI_CLIENT_ID)

            assert app.session_runtime.answer_plan_approval(retry_request.request_id, action, client_id=TUI_CLIENT_ID)
            await _joined(retry_owner)
            await _wait(lambda: app.agent_worker is None)
            assert not app.session_runtime.answer_plan_approval(
                retry_request.request_id, action, client_id=TUI_CLIENT_ID
            )
            assert writes and writes[0], "the retry must persist its prerequisite before applying the choice"
            assert not app._plan_pending and not app._plan_decision_active
            assert not app._plan_approval_tasks
            if action == "discuss_plan":
                assert app.agent is agent and not agent.messages
                assert app._plan_reofferable
                assert not composer.disabled
            else:
                build = cast(FakeBuildAgentWithRegistry, app.agent)
                assert app.interaction_mode == "build"
                assert len(build.messages) == 1
                assert PLAN.rstrip() in build.messages[0]
                assert sum(entry.content == "Implement the approved plan." for entry in app.conversation_entries) == 1
                assert not app._plan_reofferable
        finally:
            release.set()


@pytest.mark.parametrize("eager", [False, True], ids=["lazy", "eager"])
@pytest.mark.parametrize("restored", [False, True], ids=["fresh", "restored"])
@pytest.mark.parametrize(
    ("action", "phase"),
    [
        ("discuss_plan", "action_save"),
        ("implement_plan", "action_save"),
        ("implement_plan_clear", "action_save"),
        ("implement_plan", "mode_save"),
        ("implement_plan_clear", "mode_save"),
        ("implement_plan", "build_history_save"),
        ("implement_plan_clear", "build_history_save"),
    ],
)
async def test_failed_plan_action_save_restores_offer_and_requires_a_new_choice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action: PlanApprovalAction,
    phase: str,
    restored: bool,
    eager: bool,
) -> None:
    app = _app(tmp_path, monkeypatch, restored=restored)
    async with app.run_test(), _task_factory(eager):
        if restored:
            owner = _owner(app)
        else:
            await app.action_toggle_interaction_mode()
            owner = await _capture(app)
        request = await _request(app)
        agent = app.agent
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("Preserve this draft after an action-save failure")
        save = app.store.save
        failed = False
        writes: list[tuple[bool, str]] = []
        notices: list[str] = []

        def fail_action_save(record: Any) -> None:
            nonlocal failed
            if (
                not failed
                and not record.plan_pending
                and (
                    phase == "action_save"
                    or (
                        record.interaction_mode == "build"
                        and (phase == "mode_save" or any(mode == "build" for _, mode in writes))
                    )
                )
            ):
                failed = True
                raise OSError(f"{phase} failed")
            save(record)
            writes.append((record.plan_pending, record.interaction_mode))

        monkeypatch.setattr(app.store, "save", fail_action_save)
        monkeypatch.setattr(app, "_notify_user", lambda message, **_kwargs: notices.append(message))
        assert app.session_runtime.answer_plan_approval(request.request_id, action, client_id=TUI_CLIENT_ID)
        assert owner.task is not None
        results = await asyncio.gather(owner.task, return_exceptions=True)
        assert failed and isinstance(results[0], OSError)
        assert writes and writes[0] == (True, "plan"), "the prerequisite save must have succeeded"
        assert app._plan_pending and app._plan_reofferable
        assert app.interaction_mode == "plan" and app.agent is agent
        assert app._plan_decision_active is (not restored)
        assert composer.disabled is (not restored)
        assert composer.text == "Preserve this draft after an action-save failure"
        assert app.query_one("#plan_actions", ActionList).display
        retry_owner, retry = _owner(app), await _request(app)
        assert retry_owner is not owner and not owner.valid and owner.invalidated.is_set()
        assert retry.request_id != request.request_id and retry.payload["plan"] == PLAN
        assert app.agent_worker is None and not cast(PlanAgent, app.agent).messages
        assert not any(entry.content == "Implement the approved plan." for entry in app.conversation_entries)
        assert any(f"{phase} failed" in notice and "retry" in notice for notice in notices)
        assert not app.session_runtime.answer_plan_approval(request.request_id, action, client_id=TUI_CLIENT_ID)
        assert app.session_runtime.answer_plan_approval(retry.request_id, action, client_id=TUI_CLIENT_ID)
        await _joined(retry_owner)
        await _wait(lambda: app.agent_worker is None)
        assert not app._plan_pending and not app._plan_decision_active
        if action == "discuss_plan":
            assert app.agent is agent and app._plan_reofferable and not composer.disabled
        else:
            assert app.interaction_mode == "build" and not app._plan_reofferable
            assert len(cast(FakeBuildAgentWithRegistry, app.agent).messages) == 1


@pytest.mark.parametrize("action", [value for value, _ in PLAN_APPROVAL_OPTIONS])
async def test_real_launcher_immediate_answer_is_eager_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: PlanApprovalAction
) -> None:
    app = _app(tmp_path, monkeypatch)
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    owners: list[PlanApprovalOwner] = []

    async def pilot_turn(_pilot: Pilot) -> None:
        await app._startup_complete.wait()
        eager_factory = getattr(asyncio, "eager_task_factory", None)
        if eager_factory is not None:
            assert loop.get_task_factory() is eager_factory
        await app.action_toggle_interaction_mode()
        emit = app.control_channel._emit

        async def answer_immediately(event: AgentEvent) -> None:
            if event.event_type == KnownEventType.CONTROL_REQUESTED and event.content["kind"] == "plan_approval":
                owners.append(_owner(app))
                assert app.session_runtime.answer_plan_approval(
                    event.content["request_id"], action, client_id=TUI_CLIENT_ID
                )
            await emit(event)

        monkeypatch.setattr(app.control_channel, "_emit", answer_immediately)
        cast(PlanAgent, app.agent).completed_plan = PLAN
        await _submit(app, "Plan it")
        await _wait(lambda: owners)
        await _joined(owners[0])
        await _wait(lambda: app.agent_worker is None)
        assert not app._plan_approval_tasks
        assert not app._turn_active
        if action == "discuss_plan":
            assert cast(PlanAgent, app.agent).messages == ["Plan it"]
            assert app.interaction_mode == "plan"
        else:
            assert len(cast(FakeBuildAgentWithRegistry, app.agent).messages) == 1
            assert app.interaction_mode == "build"
        app.exit()

    try:
        await asyncio.wait_for(app.run_async(headless=True, auto_pilot=pilot_turn), 30)
    finally:
        loop.set_task_factory(previous_factory)


@pytest.mark.parametrize("phase", ["save", "mode_save", "retire", "rebuild"])
@pytest.mark.parametrize("transition", ["reset", "handoff"])
async def test_reset_and_handoff_invalidate_an_action_before_awaited_transition_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str, transition: str
) -> None:
    app = _app(tmp_path, monkeypatch)
    entered, release = asyncio.Event(), asyncio.Event()
    async with app.run_test():
        await app.action_toggle_interaction_mode()
        owner = await _capture(app)
        request = await _request(app)
        method_name = {
            "save": "_save_session_async",
            "mode_save": "_save_session_async",
            "retire": "_retire_background_jobs",
            "rebuild": "_build_agent",
        }[phase]
        original = getattr(app, method_name)

        async def held(*args: Any, **kwargs: Any) -> None:
            if (
                asyncio.current_task() is owner.task
                and not entered.is_set()
                and (phase != "mode_save" or app.interaction_mode == "build")
            ):
                entered.set()
                await release.wait()
            await original(*args, **kwargs)

        monkeypatch.setattr(app, method_name, held)
        action: PlanApprovalAction = "implement_plan_clear" if phase == "retire" else "implement_plan"
        assert app.session_runtime.answer_plan_approval(request.request_id, action, client_id=TUI_CLIENT_ID)
        await asyncio.wait_for(entered.wait(), 15)
        assert app.agent_worker is None
        assert app._agent_execution_draining()
        reset = asyncio.create_task(
            app._reset_current_thread() if transition == "reset" else app._handoff_session("Replacement context")
        )
        try:
            await _wait(lambda: not owner.valid)
            assert not reset.done()
            assert owner.task in app._plan_approval_tasks
            assert not app.session_runtime.answer_plan_approval(
                request.request_id, "implement_plan", client_id=TUI_CLIENT_ID
            )
            release.set()
            await asyncio.wait_for(reset, 15)
            assert not app._plan_approval_tasks
            assert app.agent_worker is None
            assert app._latest_plan is None
            assert not cast(FakeBuildAgentWithRegistry, app.agent).messages
            assert not any(entry.content == "Implement the approved plan." for entry in app.conversation_entries)
        finally:
            release.set()
            await reset


@pytest.mark.parametrize("phase", ["plan_save", "mode_save"])
async def test_stop_retains_and_fences_an_action_until_its_save_releases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    app = _app(tmp_path, monkeypatch)
    entered, release = asyncio.Event(), asyncio.Event()
    async with app.run_test():
        await app.action_toggle_interaction_mode()
        owner = await _capture(app)
        request = await _request(app)
        original_save = app._save_session_async

        async def held_save() -> None:
            if asyncio.current_task() is owner.task and (phase == "plan_save" or app.interaction_mode == "build"):
                entered.set()
                await release.wait()
            await original_save()

        monkeypatch.setattr(app, "_save_session_async", held_save)
        assert app.session_runtime.answer_plan_approval(request.request_id, "implement_plan", client_id=TUI_CLIENT_ID)
        await asyncio.wait_for(entered.wait(), 15)
        try:
            app.action_cancel_generation()
            assert not owner.valid
            assert owner.task in app._plan_approval_tasks
            assert app._agent_execution_draining()
            await _submit(app, "Continue discussing after Stop")
            assert app.agent_worker is None
            assert len(app._queued_messages) == 1
            assert not cast(PlanAgent, app.agent).messages
            release.set()
            await _joined(owner)
            await _wait(lambda: cast(PlanAgent, app.agent).messages and app.agent_worker is None)
            assert app.interaction_mode == "plan"
            assert isinstance(app.agent, PlanAgent)
            assert app.store.load(app.session.session_id).interaction_mode == "plan"
            assert cast(PlanAgent, app.agent).messages == ["Continue discussing after Stop"]
            assert app._plan_pending
            assert _owner(app) is not owner
        finally:
            release.set()


async def test_shutdown_joins_retained_action_before_agent_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _app(tmp_path, monkeypatch)
    entered, release = asyncio.Event(), asyncio.Event()
    async with app.run_test():
        await app.action_toggle_interaction_mode()
        original_agent = cast(PlanAgent, app.agent)
        owner = await _capture(app)
        request = await _request(app)
        original_save = app._save_session_async

        async def held_save() -> None:
            if asyncio.current_task() is owner.task:
                entered.set()
                await release.wait()
            await original_save()

        monkeypatch.setattr(app, "_save_session_async", held_save)
        assert app.session_runtime.answer_plan_approval(request.request_id, "implement_plan", client_id=TUI_CLIENT_ID)
        await asyncio.wait_for(entered.wait(), 15)
        quitting = asyncio.create_task(app.action_quit())
        try:
            await _wait(lambda: not owner.valid)
            assert not quitting.done()
            assert owner.task in app._plan_approval_tasks
            assert not original_agent.cleanup_calls
            release.set()
            await asyncio.wait_for(quitting, 15)
            assert owner.task is not None and owner.task.done()
            assert not app._plan_approval_tasks
            assert original_agent.cleanup_calls == 1
            assert not original_agent.messages
        finally:
            release.set()
            await quitting
