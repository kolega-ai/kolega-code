# ruff: noqa: F401,F811,E402
"""Tests for the ``/goal`` TUI slash command and the autonomous goal loop."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from kolega_code.agent.goal import GoalVerdict
from kolega_code.cli import messages
from kolega_code.cli.config import build_agent_config, config_summary
from kolega_code.cli.goal import GOAL_CLEAR_ALIASES, GoalState, build_goal_task_prompt
from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.tui import agent_runtime as agent_runtime_module
from kolega_code.llm.models import Message

from ._app_test_utils import build_test_config


# --------------------------------------------------------------------------- #
# Fake agent
# --------------------------------------------------------------------------- #


class GoalFakeAgent:
    """A minimal coder-agent stand-in that supports the goal operations.

    ``process_message_stream`` yields a single response chunk. Goal evaluation
    results are dequeued from ``_goal_evaluate_results`` so each test can script
    the verdict sequence the loop will see.
    """

    instances: list["GoalFakeAgent"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.history: list = []
        self.messages: list = []
        self.attachments: list = []
        self.prompt_extensions = list(kwargs.get("prompt_extensions", []))
        self.active_goal_condition = None
        # Queue of GoalVerdict values returned by evaluate_goal_condition (FIFO).
        self._goal_evaluate_results: list[GoalVerdict] = []
        self._evaluate_calls: list[str] = []
        # Stream call counter (1-indexed); calls listed in ``_cancel_on_calls``
        # raise asyncio.CancelledError instead of yielding.
        self._stream_call_count = 0
        self._cancel_on_calls: set[int] = set()
        # Optional verifier gate: when set, evaluate_goal_condition signals
        # ``_verifier_started`` then blocks on ``_release_verifier`` so tests can
        # act *during* the verifier phase (e.g. submit a message or cancel).
        self._verifier_started: asyncio.Event | None = None
        self._release_verifier: asyncio.Event | None = None
        GoalFakeAgent.instances.append(self)

    # -- goal plumbing ----------------------------------------------------- #

    def apply_goal(self, condition, prompt_extension=None):
        self.active_goal_condition = condition
        exts = [e for e in (self.prompt_extensions or []) if getattr(e, "id", None) != "cli-active-goal"]
        if condition and prompt_extension is not None:
            exts.append(prompt_extension)
        self.prompt_extensions = exts

    async def evaluate_goal_condition(self, condition):
        self._evaluate_calls.append(condition)
        if self._verifier_started is not None:
            self._verifier_started.set()
        if self._release_verifier is not None:
            await self._release_verifier.wait()  # CancelledError propagates here on cancel
        if self._goal_evaluate_results:
            return self._goal_evaluate_results.pop(0)
        return GoalVerdict(met=True, reason="done")

    # -- message history --------------------------------------------------- #

    def append_user_message(self, content):
        self.history.append(Message(role="user", content=content))

    def restore_message_history(self, history):
        self.history = []

    def dump_message_history(self):
        return []

    def dump_compaction_state(self):
        return {}

    def restore_compaction_state(self, data):
        pass

    # -- streaming --------------------------------------------------------- #

    async def process_message_stream(self, message, attachments=None):
        self._stream_call_count += 1
        if self._stream_call_count in self._cancel_on_calls:
            # The turn is cancelled before any processing/output happens.
            raise asyncio.CancelledError()
        self.messages.append(message)
        self.attachments.append(attachments)
        yield {"type": "response", "content": "working on it", "complete": True, "uuid": "resp-1"}

    async def cleanup(self):
        pass

    @property
    def primary_model_config(self):
        return None


# --------------------------------------------------------------------------- #
# App builder + async helpers
# --------------------------------------------------------------------------- #


def _build_goal_test_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("textual")
    from kolega_code.cli.app import KolegaCodeApp

    GoalFakeAgent.instances = []
    monkeypatch.setattr(agent_runtime_module, "CoderAgent", GoalFakeAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    return KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)


async def _submit(app, pilot, text: str) -> None:
    """Type a slash command into the composer and submit it."""
    from kolega_code.cli.tui.widgets import ChatComposer

    composer = app.query_one("#composer", ChatComposer)
    composer.load_text(text)
    await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))


async def _wait_for(app, pilot, predicate, *, timeout: float = 6.0) -> None:
    """Poll until ``predicate()`` is truthy or time out."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await pilot.pause(0.02)
        if predicate():
            return
    raise AssertionError(f"condition not met within {timeout}s")


async def _wait_goal_terminal(app, pilot, *, timeout: float = 8.0) -> None:
    """Wait for the goal loop to reach a terminal state (met/paused/None)."""

    def _terminal():
        goal = app._goal
        return goal is None or goal.met or goal.paused

    await _wait_for(app, pilot, _terminal, timeout=timeout)
    # Let the worker finish any post-terminal bookkeeping / notify calls.
    for _ in range(5):
        await pilot.pause(0.02)


async def _wait_turn_idle(app, pilot, *, timeout: float = 6.0) -> None:
    """Wait for the current turn worker to finish and the app to be idle."""

    def _idle():
        return not app._turn_active and app.agent_worker is None

    await _wait_for(app, pilot, _idle, timeout=timeout)
    for _ in range(3):
        await pilot.pause(0.02)


async def _noop_goal_loop(self) -> None:
    """A no-op replacement for ``_run_goal_loop`` used when a test only cares
    about the *set* side-effects (not the autonomous continuation)."""
    return None


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_goal_set_condition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``/goal <condition>`` sets the goal, applies the prompt extension,
    persists to ``session.goal``, adds a transcript entry, and kicks off a turn."""
    app = _build_goal_test_app(tmp_path, monkeypatch)
    # Keep the autonomous loop from running so the set-side-effects stay visible.
    monkeypatch.setattr(app, "_run_goal_loop", lambda: _noop_goal_loop(app))

    async with app.run_test() as pilot:
        await _wait_for(app, pilot, lambda: app.agent is not None)

        await _submit(app, pilot, "/goal make all tests pass")
        await _wait_turn_idle(app, pilot)

        # Goal state on the app.
        assert app._goal is not None
        assert app._goal.condition == "make all tests pass"
        assert app._goal.met is False
        assert app._goal.paused is False

        # Agent was told about the goal + carries the prompt extension.
        assert app.agent is not None
        assert app.agent.active_goal_condition == "make all tests pass"
        ext_ids = [getattr(e, "id", None) for e in app.agent.prompt_extensions]
        assert "cli-active-goal" in ext_ids

        # Persisted to the session record.
        assert app.session.goal
        assert app.session.goal.get("condition") == "make all tests pass"

        # Transcript: a user entry echoing the /goal command.
        user_entries = [e for e in app.conversation_entries if e.kind == "user"]
        assert any("make all tests pass" in e.content for e in user_entries)

        # First turn was kicked off with the goal task prompt.
        agent = GoalFakeAgent.instances[-1]
        assert agent.messages
        assert agent.messages[0] == build_goal_task_prompt("make all tests pass")


@pytest.mark.asyncio
async def test_goal_status_no_active_goal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``/goal`` with no args and no active goal reports that none is active."""
    app = _build_goal_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        await _wait_for(app, pilot, lambda: app.agent is not None)

        await _submit(app, pilot, "/goal")
        await pilot.pause()

        assert app._goal is None
        last = app.conversation_entries[-1]
        assert last.kind == "system"
        assert last.content == messages.GOAL_NONE_ACTIVE


@pytest.mark.asyncio
async def test_goal_status_active_goal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``/goal`` with an active goal renders the status block."""
    app = _build_goal_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        await _wait_for(app, pilot, lambda: app.agent is not None)

        # Seed an active goal directly (no turn kicked off).
        app._set_goal_state(GoalState.create("make all tests pass"))
        await app._persist_goal_async()
        await pilot.pause()

        await _submit(app, pilot, "/goal")
        await pilot.pause()

        last = app.conversation_entries[-1]
        assert last.kind == "system"
        assert "make all tests pass" in last.content
        assert "active" in last.content


@pytest.mark.asyncio
@pytest.mark.parametrize("alias", sorted(GOAL_CLEAR_ALIASES))
async def test_goal_clear_aliases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alias: str) -> None:
    """Every clear alias removes the active goal and clears the session record."""
    app = _build_goal_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        await _wait_for(app, pilot, lambda: app.agent is not None)

        app._set_goal_state(GoalState.create("make all tests pass"))
        await app._persist_goal_async()
        await pilot.pause()
        assert app._goal is not None

        await _submit(app, pilot, f"/goal {alias}")
        await pilot.pause()

        assert app._goal is None
        assert app.agent is not None
        assert app.agent.active_goal_condition is None
        assert app.session.goal == {}
        assert any(e.kind == "system" and messages.GOAL_CLEARED in e.content for e in app.conversation_entries)


@pytest.mark.asyncio
async def test_goal_clear_with_no_active_goal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``/goal clear`` with nothing active reports no active goal (no error)."""
    app = _build_goal_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        await _wait_for(app, pilot, lambda: app.agent is not None)

        await _submit(app, pilot, "/goal clear")
        await pilot.pause()

        assert app._goal is None
        assert app.conversation_entries[-1].kind == "system"
        assert app.conversation_entries[-1].content == messages.GOAL_NONE_ACTIVE


@pytest.mark.asyncio
async def test_goal_print_flag_sets_run_to_completion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``/goal -p <condition>`` sets ``run_to_completion=True``."""
    app = _build_goal_test_app(tmp_path, monkeypatch)
    monkeypatch.setattr(app, "_run_goal_loop", lambda: _noop_goal_loop(app))

    async with app.run_test() as pilot:
        await _wait_for(app, pilot, lambda: app.agent is not None)

        await _submit(app, pilot, "/goal -p make all tests pass")
        await _wait_turn_idle(app, pilot)

        assert app._goal is not None
        assert app._goal.run_to_completion is True
        assert app._goal.condition == "make all tests pass"
        # A system entry announcing run-to-completion was added.
        assert any(
            e.kind == "system" and messages.GOAL_RUN_TO_COMPLETION in e.content for e in app.conversation_entries
        )


@pytest.mark.asyncio
async def test_goal_loop_completes_after_nudges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The goal loop continues until the verifier reports ``met``; nudges are
    injected for each not-yet-met evaluation and a "Goal met" entry is added."""
    app = _build_goal_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        await _wait_for(app, pilot, lambda: app.agent is not None)
        agent = GoalFakeAgent.instances[-1]

        # Two not-met evaluations (→ two nudges), then met.
        agent._goal_evaluate_results = [
            GoalVerdict(met=False, reason="not yet"),
            GoalVerdict(met=False, reason="still not"),
            GoalVerdict(met=True, reason="done"),
        ]

        await _submit(app, pilot, "/goal make all tests pass")
        await _wait_goal_terminal(app, pilot)

        assert app._goal is not None
        assert app._goal.met is True
        assert app._goal.paused is False

        # The evaluator was called three times with the condition.
        assert len(agent._evaluate_calls) == 3
        assert all(c == "make all tests pass" for c in agent._evaluate_calls)

        # First stream call = goal task prompt; calls 2 and 3 = nudges.
        assert len(agent.messages) == 3
        assert agent.messages[0] == build_goal_task_prompt("make all tests pass")
        assert "not yet met" in agent.messages[1]
        assert "not yet met" in agent.messages[2]

        # A system entry announcing completion exists.
        assert any(
            e.kind == "system" and messages.GOAL_MET.format(condition="make all tests pass") in e.content
            for e in app.conversation_entries
        )


@pytest.mark.asyncio
async def test_goal_loop_turn_cap_aborts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The loop pauses after reaching ``max_turns`` and warns the user."""
    app = _build_goal_test_app(tmp_path, monkeypatch)

    # Force a small turn cap for every goal created during this test.
    original_create = GoalState.create

    @classmethod
    def capped_create(cls, condition, *, max_turns=2, run_to_completion=False):  # type: ignore[override]
        return original_create.__func__(cls, condition, max_turns=2, run_to_completion=run_to_completion)

    monkeypatch.setattr(GoalState, "create", capped_create)

    async with app.run_test() as pilot:
        await _wait_for(app, pilot, lambda: app.agent is not None)
        agent = GoalFakeAgent.instances[-1]

        # Always not-met so the only exit is the turn cap.
        agent._goal_evaluate_results = [
            GoalVerdict(met=False, reason="nope"),
            GoalVerdict(met=False, reason="nope"),
            GoalVerdict(met=False, reason="nope"),
        ]

        await _submit(app, pilot, "/goal make all tests pass")
        await _wait_goal_terminal(app, pilot)

        assert app._goal is not None
        assert app._goal.paused is True
        assert app._goal.met is False
        assert app._goal.turns_evaluated == 2
        # Two evaluations (cap hit on the second).
        assert len(agent._evaluate_calls) == 2

        # A warning-tone system entry about the turn cap exists.
        warning = [
            e
            for e in app.conversation_entries
            if e.kind == "system" and messages.GOAL_MAX_TURNS.format(turns=2) in e.content
        ]
        assert warning, [e.content for e in app.conversation_entries if e.kind == "system"]


@pytest.mark.asyncio
async def test_goal_cancel_pauses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancelling the nudge turn pauses the goal instead of clearing it."""
    app = _build_goal_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        await _wait_for(app, pilot, lambda: app.agent is not None)
        agent = GoalFakeAgent.instances[-1]

        # First evaluation: not met → a nudge turn is scheduled.
        agent._goal_evaluate_results = [GoalVerdict(met=False, reason="keep going")]
        # Cancel the nudge turn (the 2nd stream call).
        agent._cancel_on_calls = {2}

        await _submit(app, pilot, "/goal make all tests pass")
        await _wait_goal_terminal(app, pilot)

        assert app._goal is not None
        assert app._goal.paused is True
        assert app._goal.met is False
        # Only the initial task-prompt turn ran; the nudge was cancelled.
        assert len(agent.messages) == 1

        # A paused system entry exists.
        assert any(e.kind == "system" and "Goal paused" in e.content for e in app.conversation_entries)


@pytest.mark.asyncio
async def test_clear_command_clears_active_goal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``/clear`` (thread reset) wipes an active goal and the session record."""
    app = _build_goal_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        await _wait_for(app, pilot, lambda: app.agent is not None)

        # Seed an active goal directly (no running turn to block /clear).
        app._set_goal_state(GoalState.create("make all tests pass"))
        await app._persist_goal_async()
        await pilot.pause()
        assert app._goal is not None

        await _submit(app, pilot, "/clear")
        await pilot.pause()

        assert app._goal is None
        assert app.session.goal == {}
        assert app.agent is not None
        assert app.agent.active_goal_condition is None
        ext_ids = [getattr(e, "id", None) for e in app.agent.prompt_extensions]
        assert "cli-active-goal" not in ext_ids


@pytest.mark.asyncio
async def test_goal_blocked_while_turn_active(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``/goal clear`` is refused while a turn is running."""
    app = _build_goal_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        await _wait_for(app, pilot, lambda: app.agent is not None)

        app._set_goal_state(GoalState.create("make all tests pass"))
        await app._persist_goal_async()
        # Simulate an in-flight turn.
        app._turn_active = True
        await pilot.pause()

        await _submit(app, pilot, "/goal clear")
        await pilot.pause()

        # Goal is untouched and a stop-first warning was surfaced.
        assert app._goal is not None
        assert app._goal.condition == "make all tests pass"


@pytest.mark.asyncio
async def test_submission_during_verifier_is_queued(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A message submitted while the verifier runs is queued, not processed
    immediately (which would cancel the verifier via the exclusive worker group)."""
    app = _build_goal_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        await _wait_for(app, pilot, lambda: app.agent is not None)
        agent = GoalFakeAgent.instances[-1]

        # First evaluation: not met (blocks on the gate); second: met.
        agent._goal_evaluate_results = [
            GoalVerdict(met=False, reason="not yet"),
            GoalVerdict(met=True, reason="done"),
        ]
        verifier_started = asyncio.Event()
        release_verifier = asyncio.Event()
        agent._verifier_started = verifier_started
        agent._release_verifier = release_verifier

        await _submit(app, pilot, "/goal make all tests pass")
        await _wait_for(app, pilot, lambda: verifier_started.is_set())

        # The worker is alive during the verifier (the fix). Without it,
        # agent_worker is None and a submission would start a new turn worker
        # that cancels the verifier via the exclusive ``turns`` group.
        assert app.agent_worker is not None
        assert app._turn_active is False

        # Submit a steering message mid-verifier → it must queue, not run.
        await _submit(app, pilot, "steer: do X")
        assert any(item.text == "steer: do X" for item in app._queued_messages)
        assert "steer: do X" not in agent.messages

        # Release the verifier so the loop can proceed to completion, and clear
        # the gate so the second evaluation doesn't block.
        release_verifier.set()
        agent._release_verifier = None
        agent._verifier_started = None

        await _wait_goal_terminal(app, pilot)
        # The queued message is drained and processed after the goal loop ends.
        await _wait_for(app, pilot, lambda: "steer: do X" in agent.messages, timeout=8.0)
        await _wait_turn_idle(app, pilot)

        assert "steer: do X" in agent.messages


@pytest.mark.asyncio
async def test_cancel_during_verifier_pauses_goal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Esc while the verifier is running pauses the goal instead of doing nothing."""
    app = _build_goal_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        await _wait_for(app, pilot, lambda: app.agent is not None)
        agent = GoalFakeAgent.instances[-1]

        # First evaluation: not met → blocks on the gate so we can cancel mid-verifier.
        agent._goal_evaluate_results = [GoalVerdict(met=False, reason="keep going")]
        verifier_started = asyncio.Event()
        release_verifier = asyncio.Event()
        agent._verifier_started = verifier_started
        agent._release_verifier = release_verifier

        await _submit(app, pilot, "/goal make all tests pass")
        await _wait_for(app, pilot, lambda: verifier_started.is_set())

        # The worker is alive during the verifier (the fix). Without it,
        # agent_worker is None and action_cancel_generation is a no-op.
        assert app.agent_worker is not None

        app.action_cancel_generation()
        await _wait_goal_terminal(app, pilot)

        assert app._goal is not None
        assert app._goal.paused is True
        assert app._goal.met is False
        assert any(e.kind == "system" and "Goal paused" in e.content for e in app.conversation_entries)


# --------------------------------------------------------------------------- #
# Pure unit test — no app needed
# --------------------------------------------------------------------------- #


def test_goal_state_persistence_roundtrip() -> None:
    """``GoalState`` round-trips losslessly through ``to_dict``/``from_dict``."""
    state = GoalState.create("make all tests pass", run_to_completion=True)
    state.turns_evaluated = 7
    state.tokens_spent = 54_321
    state.last_reason = "two tests still failing"
    state.last_evaluated_at = "2026-01-02T03:04:05+00:00"
    state.paused = False
    state.met = False
    state.status_note = ""

    restored = GoalState.from_dict(state.to_dict())

    assert restored.condition == state.condition
    assert restored.started_at == state.started_at
    assert restored.turns_evaluated == state.turns_evaluated
    assert restored.tokens_spent == state.tokens_spent
    assert restored.last_reason == state.last_reason
    assert restored.last_evaluated_at == state.last_evaluated_at
    assert restored.max_turns == state.max_turns
    assert restored.run_to_completion is state.run_to_completion
    assert restored.paused is state.paused
    assert restored.met is state.met
    assert restored.status_note == state.status_note
    assert restored.is_active is state.is_active


def test_goal_state_from_dict_tolerates_missing_keys() -> None:
    """``from_dict`` tolerates empty/partial dicts (older sessions)."""
    restored = GoalState.from_dict({})
    assert restored.condition == ""
    assert restored.met is False
    assert restored.paused is False
    assert restored.run_to_completion is False
    assert restored.max_turns > 0
    assert restored.is_active is False  # no condition → not active

    partial = GoalState.from_dict({"condition": "do something", "met": True})
    assert partial.condition == "do something"
    assert partial.met is True
    assert partial.is_active is False  # met → not active
