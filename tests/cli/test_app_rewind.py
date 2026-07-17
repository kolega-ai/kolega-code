from pathlib import Path

import pytest

from kolega_code.cli import messages
from kolega_code.llm.models import Message, TextBlock

from ._app_test_utils import _build_sub_agent_test_app, settle_changes_inspector
from .test_app_changes_inspector import _init_git_project


def _record_turn(recorder, user_text: str) -> None:
    recorder.start_turn(Message(role="user", content=[TextBlock(user_text)]))
    recorder.record_assistant(Message(role="assistant", content=[TextBlock("done")], stop_reason="end_turn"))
    recorder.finish_turn("completed")


async def _wait_for_agent(app, pilot, timeout: float = 5.0) -> None:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if app.agent is not None:
            return
        await pilot.pause(0.05)
    raise AssertionError("Timed out waiting for the test agent to build")


@pytest.mark.asyncio
async def test_baseline_stepping_and_reset_on_close(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    _init_git_project(app.project_path)

    async with app.run_test() as pilot:
        tracker = app._session_diff_tracker
        assert tracker is not None
        (app.project_path / "src" / "a.py").write_text("pre-turn edit\n", encoding="utf-8")
        turn = tracker.capture_checkpoint("fix the bug")
        (app.project_path / "src" / "b.py").write_text("turn edit\n", encoding="utf-8")

        app.action_open_changes()
        await settle_changes_inspector(app, pilot)
        screen = app._changes_inspector
        assert screen is not None
        assert {change.path for change in app._session_diff_files} == {"src/a.py", "src/b.py"}

        screen.action_baseline_newer()
        await settle_changes_inspector(app, pilot)
        assert app._session_diff_baseline_id == turn.checkpoint_id
        assert {change.path for change in app._session_diff_files} == {"src/b.py"}
        assert "Turn 1" in app._changes_baseline_label()

        screen.action_baseline_older()
        await settle_changes_inspector(app, pilot)
        assert app._session_diff_baseline_id is None
        assert app._changes_baseline_label() == messages.CHANGES_BASELINE_SESSION_START

        screen.action_baseline_newer()
        await settle_changes_inspector(app, pilot)
        screen.action_close()
        await pilot.pause(0.1)
        assert app._session_diff_baseline_id is None


@pytest.mark.asyncio
async def test_command_rewind_opens_at_latest_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    _init_git_project(app.project_path)

    async with app.run_test() as pilot:
        notices: list[str] = []
        monkeypatch.setattr(app, "_notify_user", lambda message, **kwargs: notices.append(message))

        await app._command_rewind("")
        assert notices == [messages.REWIND_NO_CHECKPOINTS]
        assert app._changes_inspector is None

        tracker = app._session_diff_tracker
        assert tracker is not None
        first = tracker.capture_checkpoint("turn one")
        second = tracker.capture_checkpoint("turn two")

        await app._command_rewind("")
        await settle_changes_inspector(app, pilot)
        assert app._changes_inspector is not None
        assert app._session_diff_baseline_id == second.checkpoint_id

        await app._command_rewind("5")  # clamps to Turn 1, never session start
        await settle_changes_inspector(app, pilot)
        assert app._session_diff_baseline_id == first.checkpoint_id


@pytest.mark.asyncio
async def test_rewind_worker_restores_files_and_conversation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    _init_git_project(app.project_path)

    async with app.run_test() as pilot:
        await _wait_for_agent(app, pilot)
        tracker = app._session_diff_tracker
        recorder = app._session_recorder
        assert tracker is not None

        tracker.capture_checkpoint("turn one")
        _record_turn(recorder, "turn one")
        (app.project_path / "src" / "a.py").write_text("turn1 edit\n", encoding="utf-8")

        second = tracker.capture_checkpoint("turn two")
        _record_turn(recorder, "turn two")
        (app.project_path / "src" / "b.py").write_text("turn2 edit\n", encoding="utf-8")

        await app._rewind_worker(second.checkpoint_id, "Turn 2")
        await settle_changes_inspector(app, pilot)

        # Files: only the turn-2 change is reverted.
        assert (app.project_path / "src" / "a.py").read_text(encoding="utf-8") == "turn1 edit\n"
        assert (app.project_path / "src" / "b.py").read_text(encoding="utf-8") == "old b\n"

        # Conversation: the journal replays without turn two.
        record = app.store.load(app.session.session_id)
        texts = [message["content"][0]["text"] for message in record.history]
        assert texts == ["turn one", "done"]

        # Transcript marker and composer prefill.
        contents = [entry.content for entry in app.conversation_entries]
        assert any(messages.REWOUND_MARKER.format(excerpt="turn two") in content for content in contents)
        composer = app.query_one("#composer")
        assert "turn two" in getattr(composer, "text", "")


@pytest.mark.asyncio
async def test_rewind_worker_blocked_while_turn_active(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    _init_git_project(app.project_path)

    async with app.run_test() as pilot:
        await _wait_for_agent(app, pilot)
        tracker = app._session_diff_tracker
        assert tracker is not None
        checkpoint = tracker.capture_checkpoint("turn one")
        (app.project_path / "src" / "a.py").write_text("edited\n", encoding="utf-8")

        notices: list[str] = []
        monkeypatch.setattr(app, "_notify_user", lambda message, **kwargs: notices.append(message))
        app._turn_active = True
        try:
            await app._rewind_worker(checkpoint.checkpoint_id, "Turn 1")
        finally:
            app._turn_active = False

        assert notices == [messages.REWIND_BLOCKED_TURN]
        assert (app.project_path / "src" / "a.py").read_text(encoding="utf-8") == "edited\n"
