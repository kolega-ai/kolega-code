# ruff: noqa: F401,F811,E402
"""Tests for the ``/loop`` TUI slash command and the scheduled-loop driver."""

from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from kolega_code.cli import messages
from kolega_code.cli import loop as loop_module
from kolega_code.cli.config import config_summary
from kolega_code.cli.loop import LOOP_MD_RELATIVE_PATH, LoopState, parse_schedule_text
from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.tui import loop_runtime as loop_runtime_module

from ._app_test_utils import FakeCoderAgent, build_test_config, extension_by_name, install_fake_agents

START = datetime(2026, 7, 27, 10, 0, 0)  # a Monday


# --------------------------------------------------------------------------- #
# Fakes and helpers
# --------------------------------------------------------------------------- #


class LoopFakeAgent(FakeCoderAgent):
    """Coder-agent stand-in that tracks loop prompt extensions and turn prompts."""

    instances: list["LoopFakeAgent"] = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_extensions = list(kwargs.get("prompt_extensions", []))
        self._stream_call_count = 0
        #: Stream calls (1-indexed) that raise CancelledError instead of yielding.
        self._cancel_on_calls: set[int] = set()
        #: When set, a stream blocks on this event so a test can act mid-turn.
        self._hold_stream: asyncio.Event | None = None
        LoopFakeAgent.instances.append(self)

    def apply_loop(self, active, prompt_extension=None):
        self.loop_active = active
        self.loop_prompt_extension = prompt_extension if active else None
        exts = [e for e in (self.prompt_extensions or []) if getattr(e, "id", None) != "cli-active-loop"]
        if active and prompt_extension is not None:
            exts.append(prompt_extension)
        self.prompt_extensions = exts

    def apply_goal(self, condition, prompt_extension=None):
        self.active_goal_condition = condition
        exts = [e for e in (self.prompt_extensions or []) if getattr(e, "id", None) != "cli-active-goal"]
        if condition and prompt_extension is not None:
            exts.append(prompt_extension)
        self.prompt_extensions = exts

    async def evaluate_goal_condition(self, condition):
        from kolega_code.agent.goal import GoalVerdict

        return GoalVerdict(met=True, reason="done")

    def restore_message_history(self, history):
        self.history = []

    def dump_message_history(self):
        return []

    async def process_message_stream(self, message, attachments=None):
        self._stream_call_count += 1
        if self._stream_call_count in self._cancel_on_calls:
            raise asyncio.CancelledError()
        self.messages.append(message)
        self.attachments.append(attachments)
        if self._hold_stream is not None:
            await self._hold_stream.wait()
        yield {"type": "response", "content": "checked", "complete": True, "uuid": "resp-1"}

    @property
    def primary_model_config(self):
        return None


class FakeClock:
    """Deterministic stand-in for :func:`kolega_code.cli.loop.now_local`."""

    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now += timedelta(**kwargs)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    """Freeze the loop clock in both modules that bind ``now_local``."""
    fake = FakeClock(START)
    monkeypatch.setattr(loop_module, "now_local", fake)
    monkeypatch.setattr(loop_runtime_module, "now_local", fake)
    return fake


def _build_loop_test_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    agent_cls=LoopFakeAgent,
    loop_md: str | None = None,
    with_skill: bool = False,
    session_loop: dict | None = None,
):
    pytest.importorskip("textual")
    from kolega_code.cli.app import KolegaCodeApp

    LoopFakeAgent.instances = []
    install_fake_agents(monkeypatch, coder_cls=agent_cls)

    project = tmp_path / "project"
    project.mkdir()
    if loop_md is not None:
        path = project / LOOP_MD_RELATIVE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(loop_md, encoding="utf-8")
    if with_skill:
        skill_dir = project / ".agents" / "skills" / "demo-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: demo-skill\ndescription: Use this demo skill.\n---\n\nFollow demo instructions.\n",
            encoding="utf-8",
        )
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    if session_loop is not None:
        session.loop = session_loop
        store.save(session)
        session = store.load(session.session_id)
    return KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)


async def _submit(app, text: str) -> None:
    from kolega_code.cli.tui.widgets import ChatComposer

    composer = app.query_one("#composer", ChatComposer)
    composer.load_text(text)
    await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))


async def _wait_for(app, pilot, predicate, *, timeout: float = 6.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await pilot.pause(0.02)
        if predicate():
            return
    raise AssertionError(f"condition not met within {timeout}s")


async def _ready(app, pilot):
    """Wait for the agent to mount, then silence the real 1s scheduler timer.

    Tests drive ``_loop_tick`` explicitly so firing is deterministic.
    """
    await _wait_for(app, pilot, lambda: app.agent is not None)
    if app._loop_scheduler_timer is not None:
        app._loop_scheduler_timer.stop()
        app._loop_scheduler_timer = None
    return LoopFakeAgent.instances[-1]


async def _wait_loop_idle(app, pilot, *, timeout: float = 6.0) -> None:
    def _idle():
        return app.agent_worker is None and not app._turn_active and not app._loop_iteration_active

    await _wait_for(app, pilot, _idle, timeout=timeout)
    # The turn that just completed scheduled a 10ms queued-message-drain
    # timer; let it fire before returning. If it fires after a test queues a
    # message, it would start that message as a real turn which could still be
    # running at teardown — colliding with widget pruning as a
    # NoMatches("#composer") WorkerFailed flake in CI.
    await pilot.pause(0.05)


async def _tick(app, pilot) -> None:
    await app._loop_tick()
    await _wait_loop_idle(app, pilot)


def _system_contents(app) -> list[str]:
    return [entry.content for entry in app.conversation_entries if entry.kind == "system"]


def _turn_prompts(agent) -> list[str]:
    return [message for message in agent.messages]


# --------------------------------------------------------------------------- #
# Starting a loop
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_interval_loop_runs_first_iteration_immediately(tmp_path, monkeypatch, clock):
    app = _build_loop_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        agent = await _ready(app, pilot)

        await _submit(app, "/loop 5m check if CI went green")
        await _wait_loop_idle(app, pilot)

        assert app._scheduled_loop is not None
        assert app._scheduled_loop.iterations == 1
        assert app._scheduled_loop.prompt == "check if CI went green"
        assert len(agent.messages) == 1
        assert "[Scheduled loop iteration 1 of 100]" in agent.messages[0]
        assert "check if CI went green" in agent.messages[0]
        # The next fire is one interval after the iteration completed.
        assert app._scheduled_loop.next_fire_at == (START + timedelta(minutes=5)).isoformat()
        assert agent.loop_active is True
        assert "cli-active-loop" in {getattr(e, "id", None) for e in agent.prompt_extensions}


@pytest.mark.asyncio
async def test_cron_loop_waits_for_the_first_match(tmp_path, monkeypatch, clock):
    app = _build_loop_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        agent = await _ready(app, pilot)

        await _submit(app, '/loop --cron "0 9 * * *" morning briefing')
        await _wait_loop_idle(app, pilot)

        assert app._scheduled_loop is not None
        assert app._scheduled_loop.iterations == 0
        assert agent.messages == []
        assert app._scheduled_loop.next_fire_at == datetime(2026, 7, 28, 9, 0).isoformat()

        # Not due yet.
        clock.advance(hours=10)
        await _tick(app, pilot)
        assert app._scheduled_loop.iterations == 0

        # Due.
        clock.now = datetime(2026, 7, 28, 9, 0, 0)
        await _tick(app, pilot)
        assert app._scheduled_loop.iterations == 1
        assert len(agent.messages) == 1
        assert app._scheduled_loop.next_fire_at == datetime(2026, 7, 29, 9, 0).isoformat()


@pytest.mark.asyncio
async def test_second_iteration_fires_after_the_interval(tmp_path, monkeypatch, clock):
    app = _build_loop_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        agent = await _ready(app, pilot)
        await _submit(app, "/loop 5m poll")
        await _wait_loop_idle(app, pilot)

        clock.advance(minutes=2)
        await _tick(app, pilot)
        assert app._scheduled_loop is not None and app._scheduled_loop.iterations == 1

        clock.advance(minutes=4)
        await _tick(app, pilot)
        assert app._scheduled_loop.iterations == 2
        assert len(agent.messages) == 2
        assert "iteration 2 of 100" in agent.messages[1]


@pytest.mark.asyncio
async def test_replacing_a_loop_reports_it(tmp_path, monkeypatch, clock):
    app = _build_loop_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        await _submit(app, "/loop 5m first")
        await _wait_loop_idle(app, pilot)
        await _submit(app, "/loop 10m second")
        await _wait_loop_idle(app, pilot)

        assert app._scheduled_loop is not None
        assert app._scheduled_loop.prompt == "second"
        assert app._scheduled_loop.iterations == 1  # counter restarts with the new loop
        assert any("Replaced the previous loop" in content for content in _system_contents(app))


@pytest.mark.asyncio
async def test_sub_minute_interval_warns(tmp_path, monkeypatch, clock):
    app = _build_loop_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        await _submit(app, "/loop 30s poll")
        await _wait_loop_idle(app, pilot)
        assert messages.LOOP_SUB_MINUTE_ADVISORY in _system_contents(app)


@pytest.mark.asyncio
async def test_ask_permission_mode_adds_an_advisory_without_blocking(tmp_path, monkeypatch, clock):
    app = _build_loop_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        assert app.permission_mode.value == "ask"

        await _submit(app, "/loop 5m poll")
        await _wait_loop_idle(app, pilot)

        assert messages.LOOP_ASK_PERMISSION_ADVISORY in _system_contents(app)
        assert app._scheduled_loop is not None and app._scheduled_loop.iterations == 1


# --------------------------------------------------------------------------- #
# Idle gate and catch-up
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_no_fire_while_a_turn_is_running(tmp_path, monkeypatch, clock):
    app = _build_loop_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        agent = await _ready(app, pilot)
        await _submit(app, "/loop 5m poll")
        await _wait_loop_idle(app, pilot)

        gate = asyncio.Event()
        agent._hold_stream = gate
        await _submit(app, "a message from the user")
        await _wait_for(app, pilot, lambda: app._turn_active)

        clock.advance(minutes=10)
        await app._loop_tick()
        await pilot.pause(0.05)
        assert app._scheduled_loop is not None
        assert app._scheduled_loop.iterations == 1
        assert app._scheduled_loop.deferred is True

        gate.set()
        agent._hold_stream = None
        await _wait_loop_idle(app, pilot)

        await _tick(app, pilot)
        assert app._scheduled_loop.iterations == 2
        assert app._scheduled_loop.deferred is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "block",
    [
        "pending_question",
        "pending_approval",
        "pending_model_selection",
        "pending_effort_selection",
        "pending_theme_selection",
        "plan_decision",
        "queued_message",
    ],
)
async def test_no_fire_while_a_prompt_or_queue_is_pending(tmp_path, monkeypatch, clock, block):
    from kolega_code.cli.tui import state as tui_state

    app = _build_loop_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        await _submit(app, "/loop 5m poll")
        await _wait_loop_idle(app, pilot)
        clock.advance(minutes=10)

        if block == "pending_question":
            app._pending_question = tui_state.PendingQuestion(
                request_id="r1", question="pick", options=["a", "b"], descriptions=None
            )
        elif block == "pending_approval":
            app._pending_approval = object()  # type: ignore[assignment]
        elif block == "pending_model_selection":
            app._pending_model_selection = object()  # type: ignore[assignment]
        elif block == "pending_effort_selection":
            app._pending_effort_selection = object()  # type: ignore[assignment]
        elif block == "pending_theme_selection":
            app._pending_theme_selection = object()  # type: ignore[assignment]
        elif block == "plan_decision":
            app._plan_decision_active = True
        elif block == "queued_message":
            app._queue_user_message("a queued message")

        await app._loop_tick()
        await pilot.pause(0.05)

        assert app._scheduled_loop is not None
        assert app._scheduled_loop.iterations == 1
        assert app._scheduled_loop.deferred is True


@pytest.mark.asyncio
async def test_missed_windows_do_not_accumulate(tmp_path, monkeypatch, clock):
    """Three interval windows pass while busy; exactly one iteration runs after."""
    app = _build_loop_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        agent = await _ready(app, pilot)
        await _submit(app, "/loop 5m poll")
        await _wait_loop_idle(app, pilot)
        assert len(agent.messages) == 1

        app._plan_decision_active = True
        for _ in range(3):
            clock.advance(minutes=5)
            await app._loop_tick()
            await pilot.pause(0.02)
        assert len(agent.messages) == 1

        app._plan_decision_active = False
        await _tick(app, pilot)
        assert len(agent.messages) == 2

        # A further tick at the same instant must not fire again.
        await _tick(app, pilot)
        assert len(agent.messages) == 2


# --------------------------------------------------------------------------- #
# Status and stopping
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_status_without_a_loop(tmp_path, monkeypatch, clock):
    app = _build_loop_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        await _submit(app, "/loop status")
        assert app.conversation_entries[-1].content == messages.LOOP_NONE_ACTIVE


@pytest.mark.asyncio
async def test_status_reports_the_running_loop(tmp_path, monkeypatch, clock):
    app = _build_loop_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        await _submit(app, "/loop --max-iterations 4 5m check CI")
        await _wait_loop_idle(app, pilot)

        clock.advance(minutes=2)
        await _submit(app, "/loop status")
        text = app.conversation_entries[-1].content
        assert "Loop (armed): check CI" in text
        assert "Schedule: every 5m" in text
        assert "Iterations: 1/4" in text
        assert "Next iteration: in 3m" in text


@pytest.mark.asyncio
@pytest.mark.parametrize("alias", ["stop", "clear", "off", "cancel", "none", "reset"])
async def test_stop_aliases_end_the_loop(tmp_path, monkeypatch, clock, alias):
    app = _build_loop_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        await _submit(app, "/loop 5m poll")
        await _wait_loop_idle(app, pilot)

        await _submit(app, f"/loop {alias}")
        await pilot.pause()

        assert app._scheduled_loop is not None and app._scheduled_loop.stopped is True
        assert app.agent is not None and app.agent.loop_active is False
        assert app.store.load(app.session.session_id).loop["stopped"] is True

        # A stopped loop no longer fires.
        clock.advance(hours=1)
        await _tick(app, pilot)
        assert app._scheduled_loop.iterations == 1


@pytest.mark.asyncio
async def test_stop_without_a_loop_reports_none_active(tmp_path, monkeypatch, clock):
    app = _build_loop_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        await _submit(app, "/loop stop")
        assert app.conversation_entries[-1].content == messages.LOOP_NONE_ACTIVE


@pytest.mark.asyncio
async def test_usage_is_shown_for_a_bad_invocation(tmp_path, monkeypatch, clock):
    app = _build_loop_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        await _submit(app, "/loop 5s far too fast")
        assert messages.LOOP_USAGE in app.conversation_entries[-1].content
        assert "at least" in app.conversation_entries[-1].content
        assert app._scheduled_loop is None


@pytest.mark.asyncio
async def test_esc_while_armed_and_idle_stops_the_loop(tmp_path, monkeypatch, clock):
    app = _build_loop_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        await _submit(app, "/loop 5m poll")
        await _wait_loop_idle(app, pilot)

        app.action_cancel_generation()
        await pilot.pause()

        assert app._scheduled_loop is not None and app._scheduled_loop.stopped is True
        assert any("stopped by user" in content.lower() for content in _system_contents(app))


@pytest.mark.asyncio
async def test_esc_with_nothing_running_is_a_noop(tmp_path, monkeypatch, clock):
    app = _build_loop_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        before = list(app.conversation_entries)

        app.action_cancel_generation()
        await pilot.pause()

        assert app._scheduled_loop is None
        assert app.conversation_entries == before


@pytest.mark.asyncio
async def test_cancelling_an_iteration_stops_the_loop(tmp_path, monkeypatch, clock):
    app = _build_loop_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        agent = await _ready(app, pilot)
        agent._cancel_on_calls = {1}

        await _submit(app, "/loop 5m poll")
        await _wait_loop_idle(app, pilot)

        assert app._scheduled_loop is not None
        assert app._scheduled_loop.stopped is True
        assert app._scheduled_loop.iterations == 1
        assert app.agent is not None and app.agent.loop_active is False


# --------------------------------------------------------------------------- #
# --fresh
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_fresh_resets_the_thread_from_the_second_iteration(tmp_path, monkeypatch, clock):
    app = _build_loop_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        agent = await _ready(app, pilot)

        await _submit(app, "/loop --fresh 5m poll")
        await _wait_loop_idle(app, pilot)
        first_entries = len(app.conversation_entries)
        assert first_entries > 0
        assert messages.LOOP_ITERATION_FRESH not in _system_contents(app)

        clock.advance(minutes=6)
        await _tick(app, pilot)

        # The reset cleared the transcript, and the loop survived it.
        assert app._scheduled_loop is not None
        assert app._scheduled_loop.iterations == 2
        assert app._scheduled_loop.fresh is True
        assert app.session.loop["iterations"] == 2
        assert messages.LOOP_ITERATION_FRESH in _system_contents(app)
        assert "fresh conversation thread" in agent.messages[-1]
        assert app.agent is not None and app.agent.loop_active is True


@pytest.mark.asyncio
async def test_non_fresh_loop_keeps_the_thread(tmp_path, monkeypatch, clock):
    app = _build_loop_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        await _submit(app, "/loop 5m poll")
        await _wait_loop_idle(app, pilot)
        clock.advance(minutes=6)
        await _tick(app, pilot)

        assert messages.LOOP_ITERATION_FRESH not in _system_contents(app)
        assert messages.THREAD_RESET_MESSAGE not in [e.content for e in app.conversation_entries]


# --------------------------------------------------------------------------- #
# .kolega/loop.md
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_bare_loop_uses_loop_md_including_the_schedule_header(tmp_path, monkeypatch, clock):
    app = _build_loop_test_app(tmp_path, monkeypatch, loop_md="schedule: 15m\n\ntend the branch\n")
    async with app.run_test() as pilot:
        agent = await _ready(app, pilot)

        await _submit(app, "/loop")
        await _wait_loop_idle(app, pilot)

        assert app._scheduled_loop is not None
        assert app._scheduled_loop.prompt == "tend the branch"
        assert app._scheduled_loop.prompt_source == "loop_md"
        assert app._scheduled_loop.schedule_label() == "every 15m"
        assert "tend the branch" in agent.messages[0]


@pytest.mark.asyncio
async def test_bare_loop_without_loop_md_prints_usage(tmp_path, monkeypatch, clock):
    app = _build_loop_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        await _submit(app, "/loop")
        assert messages.LOOP_MD_MISSING in app.conversation_entries[-1].content
        assert messages.LOOP_USAGE in app.conversation_entries[-1].content
        assert app._scheduled_loop is None


@pytest.mark.asyncio
async def test_explicit_schedule_overrides_the_loop_md_header(tmp_path, monkeypatch, clock):
    app = _build_loop_test_app(tmp_path, monkeypatch, loop_md="schedule: 15m\n\ntend the branch\n")
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        await _submit(app, "/loop 30m")
        await _wait_loop_idle(app, pilot)
        assert app._scheduled_loop is not None
        assert app._scheduled_loop.schedule_label() == "every 30m"
        assert app._scheduled_loop.prompt == "tend the branch"


@pytest.mark.asyncio
async def test_loop_md_without_a_schedule_and_no_flag_reports_the_gap(tmp_path, monkeypatch, clock):
    app = _build_loop_test_app(tmp_path, monkeypatch, loop_md="tend the branch\n")
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        await _submit(app, "/loop")
        assert messages.LOOP_SCHEDULE_MISSING in app.conversation_entries[-1].content
        assert app._scheduled_loop is None


@pytest.mark.asyncio
async def test_loop_md_edits_take_effect_on_the_next_iteration(tmp_path, monkeypatch, clock):
    app = _build_loop_test_app(tmp_path, monkeypatch, loop_md="schedule: 5m\n\nfirst body\n")
    async with app.run_test() as pilot:
        agent = await _ready(app, pilot)
        await _submit(app, "/loop")
        await _wait_loop_idle(app, pilot)
        assert "first body" in agent.messages[0]

        (app.project_path / LOOP_MD_RELATIVE_PATH).write_text("schedule: 5m\n\nsecond body\n", encoding="utf-8")
        clock.advance(minutes=6)
        await _tick(app, pilot)

        assert "second body" in agent.messages[1]
        assert app._scheduled_loop is not None and app._scheduled_loop.prompt == "second body"


@pytest.mark.asyncio
async def test_loop_md_deleted_mid_loop_stops_the_loop(tmp_path, monkeypatch, clock):
    app = _build_loop_test_app(tmp_path, monkeypatch, loop_md="schedule: 5m\n\ntend the branch\n")
    async with app.run_test() as pilot:
        agent = await _ready(app, pilot)
        await _submit(app, "/loop")
        await _wait_loop_idle(app, pilot)

        (app.project_path / LOOP_MD_RELATIVE_PATH).unlink()
        clock.advance(minutes=6)
        await _tick(app, pilot)

        assert app._scheduled_loop is not None and app._scheduled_loop.stopped is True
        assert messages.LOOP_MD_GONE in _system_contents(app)
        assert len(agent.messages) == 1


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="symlink creation needs elevation on Windows")
async def test_symlinked_loop_md_is_refused(tmp_path, monkeypatch, clock):
    app = _build_loop_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        real = tmp_path / "elsewhere.md"
        real.write_text("do something\n", encoding="utf-8")
        link = app.project_path / LOOP_MD_RELATIVE_PATH
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(real)

        await _submit(app, "/loop")
        assert "symlink" in app.conversation_entries[-1].content
        assert app._scheduled_loop is None


# --------------------------------------------------------------------------- #
# Skill prompts
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_skill_prefixed_prompt_activates_the_skill(tmp_path, monkeypatch, clock):
    app = _build_loop_test_app(tmp_path, monkeypatch, with_skill=True)
    async with app.run_test() as pilot:
        agent = await _ready(app, pilot)

        await _submit(app, "/loop 5m /demo-skill review the diff")
        await _wait_loop_idle(app, pilot)

        skill_entries = [entry for entry in app.conversation_entries if entry.kind == "skill"]
        assert len(skill_entries) == 1
        assert skill_entries[0].content == "Activated skill `/demo-skill`."
        # Activation is injected into the agent's history, not the turn prompt.
        history_text = "\n".join(
            getattr(block, "text", "") or "" for message in agent.history for block in (message.content or [])
        )
        assert "Follow demo instructions." in history_text
        # The skill token is stripped; only the remainder is the turn prompt.
        assert "review the diff" in agent.messages[0]
        assert "/demo-skill" not in agent.messages[0]


@pytest.mark.asyncio
async def test_builtin_command_prefix_is_sent_as_plain_text(tmp_path, monkeypatch, clock):
    app = _build_loop_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        agent = await _ready(app, pilot)

        await _submit(app, "/loop 5m /model tell me the model")
        await _wait_loop_idle(app, pilot)

        assert not any(entry.kind == "skill" for entry in app.conversation_entries)
        assert "/model tell me the model" in agent.messages[0]


# --------------------------------------------------------------------------- #
# Mutual exclusion with /goal
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_loop_is_refused_while_a_goal_is_active(tmp_path, monkeypatch, clock):
    app = _build_loop_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        monkeypatch.setattr(type(app), "_run_goal_loop", lambda self: asyncio.sleep(0), raising=False)
        await _submit(app, "/goal ship the release")
        await _wait_for(app, pilot, lambda: app._goal is not None)
        await _wait_loop_idle(app, pilot)
        assert app._goal is not None and app._goal.is_active

        await _submit(app, "/loop 5m poll")
        await pilot.pause()

        assert app._scheduled_loop is None
        assert (
            messages.LOOP_BLOCK_GOAL_ACTIVE
            in [entry.content for entry in app.conversation_entries if entry.kind == "system"]
            or app._scheduled_loop is None
        )


@pytest.mark.asyncio
async def test_goal_command_is_refused_while_a_loop_is_active(tmp_path, monkeypatch, clock):
    app = _build_loop_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        await _submit(app, "/loop 5m poll")
        await _wait_loop_idle(app, pilot)

        await _submit(app, "/goal ship the release")
        await pilot.pause()

        assert app._goal is None


@pytest.mark.asyncio
async def test_set_goal_tool_is_refused_while_a_loop_is_active(tmp_path, monkeypatch, clock):
    app = _build_loop_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        agent = await _ready(app, pilot)
        await _submit(app, "/loop 5m poll")
        await _wait_loop_idle(app, pilot)

        extension = extension_by_name(agent.kwargs["tool_extensions"], "cli-goal-control")
        result = await extension.tools["set_goal"]("ship the release")

        assert result == messages.GOAL_BLOCK_LOOP_ACTIVE
        assert app._goal is None


# --------------------------------------------------------------------------- #
# Caps, expiry, reset, rewind
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_iteration_cap_ends_the_loop(tmp_path, monkeypatch, clock):
    app = _build_loop_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        agent = await _ready(app, pilot)
        await _submit(app, "/loop --max-iterations 2 5m poll")
        await _wait_loop_idle(app, pilot)

        clock.advance(minutes=6)
        await _tick(app, pilot)
        assert app._scheduled_loop is not None and app._scheduled_loop.iterations == 2

        clock.advance(minutes=6)
        await _tick(app, pilot)

        assert app._scheduled_loop.stopped is True
        assert len(agent.messages) == 2
        assert messages.LOOP_MAX_ITERATIONS.format(max_iterations=2) in _system_contents(app)
        assert app.agent is not None and app.agent.loop_active is False


@pytest.mark.asyncio
async def test_expiry_ends_the_loop(tmp_path, monkeypatch, clock):
    app = _build_loop_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        agent = await _ready(app, pilot)
        await _submit(app, "/loop --expires 1h 5m poll")
        await _wait_loop_idle(app, pilot)

        clock.advance(hours=2)
        await _tick(app, pilot)

        assert app._scheduled_loop is not None and app._scheduled_loop.stopped is True
        assert len(agent.messages) == 1
        assert any("expired" in content.lower() for content in _system_contents(app))
        assert app.agent is not None and app.agent.loop_active is False


@pytest.mark.asyncio
async def test_thread_reset_clears_the_loop(tmp_path, monkeypatch, clock):
    app = _build_loop_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        await _submit(app, "/loop 5m poll")
        await _wait_loop_idle(app, pilot)

        await _submit(app, "/clear")
        await pilot.pause()

        assert app._scheduled_loop is None
        assert app.session.loop == {}
        assert app.agent is not None and app.agent.loop_active is False


@pytest.mark.asyncio
async def test_rewind_stops_the_loop(tmp_path, monkeypatch, clock):
    app = _build_loop_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        await _submit(app, "/loop 5m poll")
        await _wait_loop_idle(app, pilot)
        assert app._scheduled_loop is not None and app._scheduled_loop.is_active

        await app._stop_loop(messages.REWIND_LOOP_STOPPED, notify=False)

        assert app._scheduled_loop.stopped is True
        assert messages.REWIND_LOOP_STOPPED in _system_contents(app)


# --------------------------------------------------------------------------- #
# Persistence and resume
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_loop_state_is_persisted(tmp_path, monkeypatch, clock):
    app = _build_loop_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        await _submit(app, "/loop 5m check CI")
        await _wait_loop_idle(app, pilot)

        stored = app.store.load(app.session.session_id).loop
        assert stored["prompt"] == "check CI"
        assert stored["schedule_kind"] == "interval"
        assert stored["schedule_value"] == "300"
        assert stored["iterations"] == 1


@pytest.mark.asyncio
async def test_resume_rearms_an_unexpired_loop(tmp_path, monkeypatch, clock):
    state = LoopState.create(parse_schedule_text("5m"), "check CI", now=START)
    state.mark_fired(START)
    state.advance_after_completion(START)
    app = _build_loop_test_app(tmp_path, monkeypatch, session_loop=state.to_dict())

    async with app.run_test() as pilot:
        agent = await _ready(app, pilot)
        await _wait_for(app, pilot, lambda: agent.loop_active is True)

        assert app._scheduled_loop is not None
        assert app._scheduled_loop.prompt == "check CI"
        assert app._scheduled_loop.iterations == 1
        assert "cli-active-loop" in {getattr(e, "id", None) for e in agent.prompt_extensions}
        assert any("Loop restored" in content for content in _system_contents(app))

        clock.advance(minutes=6)
        await _tick(app, pilot)
        assert app._scheduled_loop.iterations == 2


@pytest.mark.asyncio
async def test_resume_expires_a_stale_loop_instead_of_firing_it(tmp_path, monkeypatch, clock):
    old = START - timedelta(days=30)
    state = LoopState.create(parse_schedule_text("5m"), "check CI", now=old, expires_seconds=3600)
    state.mark_fired(old)
    state.advance_after_completion(old)
    app = _build_loop_test_app(tmp_path, monkeypatch, session_loop=state.to_dict())

    async with app.run_test() as pilot:
        agent = await _ready(app, pilot)
        await _wait_for(app, pilot, lambda: app._scheduled_loop is not None and app._scheduled_loop.stopped)

        assert agent.loop_active is False
        assert agent.messages == []
        await _tick(app, pilot)
        assert agent.messages == []


# --------------------------------------------------------------------------- #
# Status dashboard
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_dashboard_shows_and_hides_the_loop_line(tmp_path, monkeypatch, clock):
    from textual.widgets import Static

    app = _build_loop_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        await _submit(app, "/loop 5m check CI")
        await _wait_loop_idle(app, pilot)

        rendered = str(app.query_one("#status_dashboard", Static).render())
        assert "Loop" in rendered
        assert "check CI" in rendered
        assert "every 5m" in rendered

        await _submit(app, "/loop stop")
        await pilot.pause()
        rendered = str(app.query_one("#status_dashboard", Static).render())
        assert "check CI" not in rendered


@pytest.mark.asyncio
async def test_ticks_do_not_re_render_when_the_coarse_label_is_unchanged(tmp_path, monkeypatch, clock):
    app = _build_loop_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        await _ready(app, pilot)
        await _submit(app, "/loop 1h check CI")
        await _wait_loop_idle(app, pilot)

        # Settle the label first: the iteration's own refresh rendered "1h", and
        # the first tick a second later legitimately moves it to "59m".
        clock.advance(seconds=1)
        await app._loop_tick()
        await pilot.pause()

        calls = {"count": 0}
        original = app._refresh_status_dashboard

        def counting_refresh():
            calls["count"] += 1
            original()

        monkeypatch.setattr(app, "_refresh_status_dashboard", counting_refresh)

        # Several more sub-minute ticks: the coarse countdown stays "59m".
        for _ in range(5):
            clock.advance(seconds=1)
            await app._loop_tick()
        await pilot.pause()
        assert calls["count"] == 0

        # Crossing a whole minute changes the label and triggers exactly one render.
        clock.advance(minutes=1)
        await app._loop_tick()
        await pilot.pause()
        assert calls["count"] == 1


@pytest.mark.asyncio
async def test_loop_tokens_accumulate_from_ledger_delta_per_iteration(tmp_path, monkeypatch, clock):
    """Each loop iteration adds the ledger delta for its whole command tree."""
    from kolega_code.llm.usage import normalize_usage

    class LedgerLoopFakeAgent(LoopFakeAgent):
        async def process_message_stream(self, message, attachments=None):
            ledger = self.kwargs["usage_ledger"]
            request_id = ledger.begin("anthropic", "m")
            ledger.record_response(
                request_id, normalize_usage({"input_tokens": 40, "output_tokens": 2}, "anthropic", "m")
            )
            async for item in LoopFakeAgent.process_message_stream(self, message, attachments):
                yield item

    app = _build_loop_test_app(tmp_path, monkeypatch, agent_cls=LedgerLoopFakeAgent)
    async with app.run_test() as pilot:
        agent = await _ready(app, pilot)
        assert agent.kwargs["usage_ledger"] is app._usage_ledger

        await _submit(app, "/loop 5m check if CI went green")
        await _wait_loop_idle(app, pilot)

        assert app._scheduled_loop is not None
        assert app._scheduled_loop.iterations == 1
        assert app._scheduled_loop.tokens_spent == 42
