from __future__ import annotations

import asyncio
from pathlib import Path
import threading
from typing import Any

import pytest

from kolega_code.cli.session_journal import SessionJournal
from kolega_code.cli.session_store import SessionLockedError, SessionStore, default_state_dir
from kolega_code.cli.session_usage import LLM_RUN_FINISHED_EVENT

from .test_app_handoff import _build_handoff_test_app, _wait_for
from .test_ask_usage_persistence import _setup


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_waiter", [False, True])
async def test_headless_terminal_and_lock_release_wait_for_actual_finish_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None, cancel_waiter: bool
) -> None:
    main_module, project = _setup(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    terminals: list[str] = []
    original_append = SessionJournal.append

    def append(journal: SessionJournal, event_type: str, **kwargs: Any) -> Any:
        if event_type == LLM_RUN_FINISHED_EVENT:
            entered.set()
            if not release.wait(20):
                raise AssertionError("test did not release finish writer")
        if event_type in ("run.completed", "run.failed", "run.cancelled"):
            terminals.append(event_type)
        return original_append(journal, event_type, **kwargs)

    monkeypatch.setattr(SessionJournal, "append", append)
    args = main_module.parse_args(["ask", "offline turn", "--project", str(project), "--session", "shutdown-test"])
    task = asyncio.create_task(main_module._run_ask(args))
    contender = SessionStore(default_state_dir())
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        if cancel_waiter:
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
        assert not task.done()
        assert terminals == []
        with pytest.raises(SessionLockedError):
            contender.claim_for_resume("shutdown-test", project)
        release.set()
        assert await asyncio.wait_for(task, 10) == 0
        assert terminals == ["run.completed"]
        contender.claim_for_resume("shutdown-test", project)
        events = contender.journal("shutdown-test").read_events()
        kinds = [event.event_type for event in events]
        assert kinds.index(LLM_RUN_FINISHED_EVENT) < kinds.index("run.completed")
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        contender.release_session_locks()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["handoff", "quit"])
async def test_tui_rebind_and_shutdown_wait_for_actual_finish_after_repeated_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None, action: str
) -> None:
    app = _build_handoff_test_app(tmp_path, monkeypatch)
    old_id = app.session.session_id
    app.store.claim_for_resume(old_id, app.project_path)
    contender = SessionStore(tmp_path / "state")
    entered, release = threading.Event(), threading.Event()
    try:
        async with app.run_test() as pilot:
            await _wait_for(app, pilot, lambda: app.agent is not None and app._usage_sink is not None)
            sink = app._usage_sink
            assert sink is not None
            journal = app.store.journal(old_id)
            append = journal.append

            def blocked(event_type: str, **kwargs: Any) -> Any:
                if event_type == LLM_RUN_FINISHED_EVENT:
                    entered.set()
                    assert release.wait(20)
                return append(event_type, **kwargs)

            monkeypatch.setattr(journal, "append", blocked)
            task = asyncio.create_task(
                app._handoff_session("offline handoff") if action == "handoff" else app._shutdown_session()
            )
            try:
                assert await asyncio.to_thread(entered.wait, 10)
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
                assert app.session.session_id == old_id
                assert not app._quit_cleanly
                with pytest.raises(SessionLockedError):
                    contender.claim_for_resume(old_id, app.project_path)
                release.set()
                await asyncio.wait_for(task, 10)
                assert sink.close_result is not None and sink.close_result.committed
                assert sink._drainer is not None and sink._drainer.done()
                if action == "handoff":
                    assert app.session.session_id != old_id
                else:
                    assert app._quit_cleanly
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)
                if app._usage_sink is not None:
                    await app._usage_sink.wait_closed()
    finally:
        release.set()
        app.store.release_session_locks()
        contender.release_session_locks()
