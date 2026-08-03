"""Turn-checkpoint UX: the status strip must flip the moment a turn starts.

The session-diff checkpoint can read every dirty file in the repo, so it runs
inside ``_run_turn_stream`` *after* progress begins. These tests pin the
ordering (strip shows "Preparing checkpoint…" while the capture is in flight),
the Esc path during a slow capture, and the goal-nudge turn that deliberately
takes no checkpoint.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest

from kolega_code.cli import messages
from kolega_code.cli.tui import state as tui_state

from ._app_test_utils import (
    FakeCoderAgent,
    _build_sub_agent_test_app,
    install_fake_agents,
    wait_for_session_diff_baseline,
    wait_for_turn_idle,
)
from .test_app_changes_inspector import _init_git_project


pytestmark = pytest.mark.usefixtures("hermetic_git_config")


class GatedStreamAgent(FakeCoderAgent):
    """Holds the response stream until the test releases it."""

    stream_gate: asyncio.Event | None = None

    async def process_message_stream(self, message, attachments=None):
        gate = type(self).stream_gate
        if gate is not None:
            await gate.wait()
        async for chunk in super().process_message_stream(message, attachments):
            yield chunk


async def _wait_for_thread_event(pilot, event: threading.Event, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if event.is_set():
            return
        await pilot.pause(0.01)
    raise AssertionError("Timed out waiting for the checkpoint capture to start")


def _gate_checkpoint_capture(monkeypatch: pytest.MonkeyPatch, tracker):
    """Make capture_checkpoint block until released; returns (started, release)."""
    started = threading.Event()
    release = threading.Event()
    original = tracker.capture_checkpoint

    def capture(label):
        started.set()
        if not release.wait(timeout=5.0):
            raise AssertionError("Timed out waiting to release the checkpoint capture")
        return original(label)

    monkeypatch.setattr(tracker, "capture_checkpoint", capture)
    return started, release


@pytest.mark.asyncio
async def test_status_flips_before_checkpoint_capture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    GatedStreamAgent.stream_gate = asyncio.Event()
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    # _build_sub_agent_test_app installs the plain fake; the agent is only
    # constructed on mount, so overriding again here makes the gate stick.
    install_fake_agents(monkeypatch, coder_cls=GatedStreamAgent)
    _init_git_project(app.project_path)

    async with app.run_test() as pilot:
        await wait_for_session_diff_baseline(app)
        tracker = app._session_diff_tracker
        assert tracker is not None
        started, release = _gate_checkpoint_capture(monkeypatch, tracker)

        app.agent_worker = app.run_worker(app._process_message("hello"), name="kolega-turn", group="turns")
        await _wait_for_thread_event(pilot, started)

        # The capture is still blocked, yet the previous turn's result is gone
        # and the spinner is already running with a truthful activity.
        assert app._turn_active is True
        assert app._turn_started_at is not None
        assert app._turn_status_text == messages.PREPARING_CHECKPOINT

        release.set()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and app._turn_status_text != messages.WORKING:
            await pilot.pause(0.01)
        assert app._turn_status_text == messages.WORKING

        GatedStreamAgent.stream_gate.set()
        await wait_for_turn_idle(app, pilot)
        assert app._turn_final_text.startswith("Done in")


@pytest.mark.asyncio
async def test_esc_during_checkpoint_finalizes_stopped_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    _init_git_project(app.project_path)

    async with app.run_test() as pilot:
        await wait_for_session_diff_baseline(app)
        tracker = app._session_diff_tracker
        assert tracker is not None
        started, release = _gate_checkpoint_capture(monkeypatch, tracker)

        app.agent_worker = app.run_worker(app._process_message("hello"), name="kolega-turn", group="turns")
        await _wait_for_thread_event(pilot, started)

        app.action_cancel_generation()
        release.set()
        await wait_for_turn_idle(app, pilot)

        agent = app.agent
        assert isinstance(agent, FakeCoderAgent)
        assert app._turn_final_state is tui_state.TurnState.STOPPED
        assert app._turn_final_text.startswith("Stopped after")
        assert app._turn_active is False
        assert app.agent_worker is None
        assert agent.messages == []  # the stream never started


@pytest.mark.asyncio
async def test_goal_nudge_turn_takes_no_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    _init_git_project(app.project_path)

    async with app.run_test():
        await wait_for_session_diff_baseline(app)
        tracker = app._session_diff_tracker
        assert tracker is not None
        agent = app.agent
        assert isinstance(agent, FakeCoderAgent)
        captures: list[str] = []
        monkeypatch.setattr(tracker, "capture_checkpoint", lambda label: captures.append(label))

        cancelled = await app._run_turn_stream(lambda: agent.process_message_stream("nudge"))

        assert cancelled is False
        assert captures == []
        # Progress still began and completed normally without a checkpoint.
        assert app._turn_final_text.startswith("Done in")
