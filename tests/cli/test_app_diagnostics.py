# ruff: noqa: F401,E402
"""App-level diagnostics: the event tee persists structured LLM errors, and /bug bundles them."""

import asyncio
import json
import threading
import zipfile
from concurrent.futures import CancelledError as ExportCancelledError
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from textual.pilot import Pilot

from kolega_code.events import AgentEvent
from kolega_code.cli import app as app_module
from kolega_code.cli.session_store import SessionStoreError
from kolega_code.cli.tui import bug_report
from kolega_code.cli.tui.state import ConversationEntry
from kolega_code.cli.tui.widgets import ChatComposer, ConversationEntryWidget

from ._app_test_utils import _build_mention_test_app, renderable_text, wait_for_layout
from .test_app_plan_approval_channel import _task_factory


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rendered(app: app_module.KolegaCodeApp, text: str) -> bool:
    return any(
        widget.region.height and text in renderable_text(widget._formatted)
        for widget in app.query(ConversationEntryWidget)
    )


def _submit(app: app_module.KolegaCodeApp, text: str) -> None:
    composer = app.query_one("#composer", ChatComposer)
    assert app.post_message(ChatComposer.Submitted(composer, text))


class _ExportGate:
    def __init__(self, app: app_module.KolegaCodeApp, monkeypatch: pytest.MonkeyPatch) -> None:
        self.loop = asyncio.get_running_loop()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.exited = threading.Event()
        self._entered_notification = asyncio.Event()
        self.calls = 0
        self.original = app.store.bug_export
        monkeypatch.setattr(app.store, "bug_export", self.export)

    def export(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        self.entered.set()
        self.loop.call_soon_threadsafe(self._entered_notification.set)
        try:
            assert self.release.wait(60), "test did not release the export thread"
            return self.original(*args, **kwargs)
        finally:
            self.exited.set()

    async def wait_entered(self) -> None:
        await asyncio.wait_for(self._entered_notification.wait(), 30)
        assert self.entered.is_set()

    async def drain(self, app: app_module.KolegaCodeApp) -> None:
        self.release.set()
        job = app._bug_export_job
        if job is not None:
            try:
                await asyncio.wait_for(job.wait(), 30)
            except (asyncio.CancelledError, ExportCancelledError):
                pass


async def _checkpoint() -> None:
    ready = asyncio.Event()
    asyncio.get_running_loop().call_soon(ready.set)
    await ready.wait()


@pytest.mark.asyncio
@pytest.mark.parametrize("eager", [False, True], ids=["normal", "eager"])
async def test_bug_export_releases_app_message_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None, eager: bool
) -> None:
    app = _build_mention_test_app(tmp_path, monkeypatch)
    gate = _ExportGate(app, monkeypatch)
    dispatched = asyncio.Event()
    original_diagnostics = app._command_diagnostics

    async def diagnostics(args: str) -> None:
        await original_diagnostics(args)
        dispatched.set()

    monkeypatch.setattr(app, "_command_diagnostics", diagnostics)
    async with _task_factory(eager), app.run_test() as pilot:
        _submit(app, "/bug")
        try:
            await gate.wait_entered()
            _submit(app, "/diagnostics")
            await asyncio.wait_for(dispatched.wait(), 30)
            await wait_for_layout(pilot, lambda: _rendered(app, "Preparing bug report"))
            assert not gate.release.is_set()
        finally:
            await gate.drain(app)
        await wait_for_layout(pilot, lambda: app._bug_export_job is None and _rendered(app, "Bug report written to:"))


@pytest.mark.asyncio
@pytest.mark.parametrize("eager", [False, True], ids=["normal", "eager"])
async def test_bug_export_admits_one_job_and_allows_a_later_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None, eager: bool
) -> None:
    app = _build_mention_test_app(tmp_path, monkeypatch)
    gate = _ExportGate(app, monkeypatch)
    async with _task_factory(eager), app.run_test() as pilot:
        _submit(app, "/bug")
        try:
            await gate.wait_entered()
            owner = app._bug_export_job
            assert owner is not None and owner.worker is not None
            assert owner.worker.name == "kolega-bug-export"
            assert owner.worker.group == "bug-export"
            _submit(app, "/bug")
            await wait_for_layout(pilot, lambda: _rendered(app, "already being prepared"))
            assert app._bug_export_job is owner
            assert gate.calls == 1
            assert sum(entry.kind == "progress" for entry in app.conversation_entries) == 1
        finally:
            await gate.drain(app)
        await wait_for_layout(pilot, lambda: app._bug_export_job is None and _rendered(app, "Bug report written to:"))
        _submit(app, "/bug")
        assert app._diag is not None
        diag_dir = app._diag.dir
        await wait_for_layout(
            pilot,
            lambda: (
                app._bug_export_job is None
                and len(list(diag_dir.glob("bug-*.zip"))) == 2
                and sum("Bug report written to:" in entry.content for entry in app.conversation_entries) == 2
            ),
        )
        assert gate.calls == 2
        assert not any(entry.kind == "progress" for entry in app.conversation_entries)


@pytest.mark.asyncio
async def test_bug_export_summary_runs_in_the_export_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    app = _build_mention_test_app(tmp_path, monkeypatch)
    main_thread = threading.get_ident()
    threads = []
    original_summary = bug_report.diagnostics_summary_lines
    original_export = app.store.bug_export

    def summary(*args: Any, **kwargs: Any) -> list[str]:
        threads.append(threading.get_ident())
        return original_summary(*args, **kwargs)

    def export(*args: Any, **kwargs: Any) -> Any:
        threads.append(threading.get_ident())
        return original_export(*args, **kwargs)

    def forbidden_summary() -> list[str]:
        raise AssertionError("/bug must not run the app's synchronous summary reader")

    monkeypatch.setattr(bug_report, "diagnostics_summary_lines", summary)
    monkeypatch.setattr(app.store, "bug_export", export)
    monkeypatch.setattr(app, "_diagnostics_summary_lines", forbidden_summary)
    async with app.run_test() as pilot:
        assert app._diag is not None
        app._diag.record("event_loop_stalled")
        app._diag.record("llm_error")
        _submit(app, "/bug")
        await wait_for_layout(pilot, lambda: app._bug_export_job is None and _rendered(app, "Bug report written to:"))
        assert len(threads) == 2 and threads[0] == threads[1] != main_thread
        [bundle] = app._diag.dir.glob("bug-*.zip")
        with zipfile.ZipFile(bundle) as archive:
            assert b"1 loop stall(s), 1 LLM error(s)" in archive.read("summary.md")


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_diag", [False, True])
async def test_bug_export_diagnostics_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None, missing_diag: bool
) -> None:
    app = _build_mention_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        assert app._diag is not None
        diag_dir = app._diag.dir
        if missing_diag:
            app._diag = None
        else:
            app._diag.enabled = False
        _submit(app, "/bug")
        await wait_for_layout(pilot, lambda: _rendered(app, "Diagnostics are disabled"))
        assert app._bug_export_job is None
        assert not list(diag_dir.glob("bug-*.zip"))


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [SessionStoreError, OSError, ValueError])
async def test_bug_export_failure_settles_progress_and_releases_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None, error_type: type[Exception]
) -> None:
    app = _build_mention_test_app(tmp_path, monkeypatch)
    copied = []
    original_export = app.store.bug_export

    def failing_export(*args: Any, **kwargs: Any) -> Any:
        raise error_type("bad journal Authorization: Bearer private-credential-123456789")

    monkeypatch.setattr(app.store, "bug_export", failing_export)
    monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
    async with app.run_test() as pilot:
        _submit(app, "/bug")
        await wait_for_layout(
            pilot, lambda: app._bug_export_job is None and _rendered(app, "Could not assemble the bug bundle.")
        )
        assert not copied
        assert not any(entry.kind == "progress" for entry in app.conversation_entries)
        assert app._diag is not None
        assert not list(app._diag.dir.glob("bug-*.zip"))
        failures = [row for row in _read(app._diag.path) if row["kind"] == "bug_export_failed"]
        assert failures[-1]["error_type"] == error_type.__name__
        assert "private-credential-123456789" not in app._diag.path.read_text(encoding="utf-8")
        monkeypatch.setattr(app.store, "bug_export", original_export)
        _submit(app, "/bug")
        await wait_for_layout(pilot, lambda: app._bug_export_job is None and _rendered(app, "Bug report written to:"))
        assert len(copied) == 1


@pytest.mark.asyncio
async def test_bug_export_session_switch_does_not_change_successor_transcript_or_clipboard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    app = _build_mention_test_app(tmp_path, monkeypatch)
    gate = _ExportGate(app, monkeypatch)
    copied = []
    notices = []
    monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
    monkeypatch.setattr(app, "_notify_user", lambda message, **kwargs: notices.append(message))
    async with app.run_test() as pilot:
        original_session = app.session
        original_entries = app.conversation_entries
        _submit(app, "/bug")
        try:
            await gate.wait_entered()
            successor = ConversationEntry(kind="system", content="Successor session")
            app.session = replace(original_session, session_id="successor-session", name="Successor")
            app.conversation_entries = [successor]
            await gate.drain(app)
            await wait_for_layout(pilot, lambda: app._bug_export_job is None)
            assert app.conversation_entries == [successor]
            assert not copied
            assert len(notices) == 1
            assert original_session.session_id in notices[0]
            assert original_session.name in notices[0]
            assert app._diag is not None
            [bundle] = app._diag.dir.glob("bug-*.zip")
            assert str(bundle) in notices[0]
            with zipfile.ZipFile(bundle) as archive:
                assert json.loads(archive.read("session.json"))["session_id"] == original_session.session_id
        finally:
            await gate.drain(app)
            app.session = original_session
            app.conversation_entries = original_entries


@pytest.mark.asyncio
async def test_bug_export_corrupt_journal_does_not_publish_a_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    app = _build_mention_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        journal_path = app.store.events_path_for(app.session.session_id)
        damaged = journal_path.read_bytes() + b"{not-json}\n"
        journal_path.write_bytes(damaged)
        _submit(app, "/bug")
        await wait_for_layout(
            pilot, lambda: app._bug_export_job is None and _rendered(app, "Could not assemble the bug bundle.")
        )
        assert journal_path.read_bytes() == damaged
        assert app._diag is not None
        assert not list(app._diag.dir.glob("bug-*.zip"))
        assert not list(app._diag.dir.glob("*.tmp"))


@pytest.mark.asyncio
@pytest.mark.parametrize("eager", [False, True], ids=["normal", "eager"])
async def test_bug_export_repeated_cancellation_drains_before_dropping_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None, eager: bool
) -> None:
    app = _build_mention_test_app(tmp_path, monkeypatch)
    gate = _ExportGate(app, monkeypatch)
    copied = []
    monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
    async with _task_factory(eager), app.run_test() as pilot:
        _submit(app, "/bug")
        drain = None
        try:
            await gate.wait_entered()
            job = app._bug_export_job
            assert job is not None and job.worker is not None and job.future is not None
            await wait_for_layout(pilot, lambda: _rendered(app, "Preparing bug report"))
            before = job.entry.content
            drain = asyncio.create_task(app._cancel_bug_export())
            await _checkpoint()
            for _ in range(2):
                drain.cancel()
                job.worker.cancel()
                await _checkpoint()
                assert not drain.done()
                assert app._bug_export_job is job
                assert job.cancel_event.is_set()
                assert not job.future.done()
            app._bug_export_progress(job, "Late progress must not appear")
            assert job.entry.content == before
        finally:
            await gate.drain(app)
            if drain is not None:
                await asyncio.wait_for(drain, 30)
        assert gate.exited.is_set()
        assert app._bug_export_job is None
        assert job.future.done()
        assert not copied
        assert job.entry.content == before
        assert app._diag is not None
        assert not list(app._diag.dir.glob("bug-*.zip"))
        assert not list(app._diag.dir.glob("*.tmp"))
        await app._cancel_bug_export()  # Idempotent once drained.


@pytest.mark.asyncio
@pytest.mark.parametrize("shutdown", ["quit", "unmount"])
@pytest.mark.parametrize("eager", [False, True], ids=["normal", "eager"])
async def test_bug_export_shutdown_drains_without_late_ui(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None, shutdown: str, eager: bool
) -> None:
    app = _build_mention_test_app(tmp_path, monkeypatch)
    gate = _ExportGate(app, monkeypatch)
    copied = []
    monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
    cancelled = asyncio.Event()
    release_task = None
    try:
        async with _task_factory(eager), app.run_test():
            _submit(app, "/bug")
            await gate.wait_entered()
            job = app._bug_export_job
            assert job is not None and job.future is not None
            future = job.future
            before = job.entry.content
            original_cancel = job.cancel_event.set

            def cancel() -> None:
                original_cancel()
                cancelled.set()

            monkeypatch.setattr(job.cancel_event, "set", cancel)

            async def release_after_shutdown() -> None:
                await asyncio.wait_for(cancelled.wait(), 30)
                assert app._bug_export_job is job
                assert not future.done()
                gate.release.set()

            release_task = asyncio.create_task(release_after_shutdown())
            if shutdown == "quit":
                await asyncio.wait_for(app.action_quit(), 30)
        await asyncio.wait_for(release_task, 30)
        assert job.future.done()
        assert gate.exited.is_set()
        assert app._bug_export_job is None
        assert job.entry.content == before
        assert not copied
        assert app._diag is not None
        assert not list(app._diag.dir.glob("bug-*.zip"))
        assert not list(app._diag.dir.glob("*.tmp"))
    finally:
        await gate.drain(app)
        if release_task is not None and not release_task.done():
            release_task.cancel()
            await asyncio.gather(release_task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("eager", [False, True], ids=["normal", "eager"])
async def test_bug_export_shutdown_request_fences_progress_and_published_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None, eager: bool
) -> None:
    app = _build_mention_test_app(tmp_path, monkeypatch)
    gate = _ExportGate(app, monkeypatch)
    copied = []
    monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
    shutdown_started = asyncio.Event()
    finish_shutdown = asyncio.Event()
    original_shutdown = app._shutdown_session

    async def held_shutdown() -> None:
        shutdown_started.set()
        await finish_shutdown.wait()
        await original_shutdown()

    monkeypatch.setattr(app, "_shutdown_session", held_shutdown)
    async with _task_factory(eager), app.run_test() as pilot:
        _submit(app, "/bug")
        shutdown_task = None
        try:
            await gate.wait_entered()
            job = app._bug_export_job
            assert job is not None
            await wait_for_layout(pilot, lambda: _rendered(app, "Preparing bug report"))
            before = job.entry.content
            shutdown_task = asyncio.create_task(app.action_quit())
            await asyncio.wait_for(shutdown_started.wait(), 30)
            app._bug_export_progress(job, "Progress after a shutdown request")
            assert job.entry.content == before
            await gate.drain(app)
            await wait_for_layout(pilot, lambda: app._bug_export_job is None)
            assert not copied
            assert job.entry.content == before
            assert app._diag is not None
            [published] = app._diag.dir.glob("bug-*.zip")
            with zipfile.ZipFile(published) as archive:
                assert archive.testzip() is None
        finally:
            await gate.drain(app)
            finish_shutdown.set()
            if shutdown_task is not None:
                await asyncio.wait_for(shutdown_task, 30)


@pytest.mark.asyncio
async def test_real_launcher_bug_export_releases_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    app = _build_mention_test_app(tmp_path, monkeypatch)
    loop = asyncio.get_running_loop()
    original_factory = loop.get_task_factory()

    async def drive(pilot: Pilot) -> None:
        await app._startup_complete.wait()
        eager_factory = getattr(asyncio, "eager_task_factory", None)
        if eager_factory is not None:
            assert loop.get_task_factory() is eager_factory
        for count in (1, 2):
            _submit(app, "/bug")
            await wait_for_layout(
                pilot,
                lambda: (
                    app._bug_export_job is None
                    and sum("Bug report written to:" in entry.content for entry in app.conversation_entries) == count
                    and _rendered(app, "Bug report written to:")
                ),
            )
        app.exit()

    try:
        await asyncio.wait_for(app.run_async(headless=True, auto_pilot=drive), 60)
    finally:
        loop.set_task_factory(original_factory)


@pytest.mark.asyncio
async def test_event_tee_persists_structured_llm_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")
    app = _build_mention_test_app(tmp_path, monkeypatch)
    async with app.run_test():
        assert app._diag is not None and app._diag.enabled
        # Simulate the agent emitting a structured error (e.g. the DeepSeek /v1 image 400).
        app._render_event(
            AgentEvent(
                event_type="llm_error",
                sender="agent",
                content={
                    "provider": "deepseek",
                    "model": "deepseek-v4-pro",
                    "endpoint": "https://api.deepseek.com/v1",
                    "http_status": 400,
                    "error_type": "LLMInvalidRequestError",
                    "message": "unknown variant `image_url`, expected `text`",
                },
            )
        )
        rows = _read(app._diag.path)

    errors = [r for r in rows if r["kind"] == "llm_error"]
    assert errors, "structured llm_error was not persisted to the diagnostics timeline"
    err = errors[0]
    assert err["http_status"] == 400
    assert err["provider"] == "deepseek"
    assert "api.deepseek.com/v1" in err["endpoint"]
    assert "image_url" in err["message"]


@pytest.mark.asyncio
async def test_journal_stall_callback_persists_the_failure_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    app = _build_mention_test_app(tmp_path, monkeypatch)
    async with app.run_test():
        app._record_journal_stall({"kind": "timeout", "event_type": "turn_ended", "elapsed_s": 30.0, "timeout_s": 30.0})
        assert app._diag is not None
        rows = _read(app._diag.path)

    stalls = [row for row in rows if row["kind"] == "journal_stall"]
    assert len(stalls) == 1
    assert stalls[0]["failure_kind"] == "timeout"
    assert stalls[0]["event_type"] == "turn_ended"


@pytest.mark.asyncio
async def test_diagnostics_and_bug_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    pytest.importorskip("textual")
    app = _build_mention_test_app(tmp_path, monkeypatch)
    async with app.run_test() as pilot:
        _submit(app, "/diagnostics")
        _submit(app, "/bug")
        await wait_for_layout(pilot, lambda: app._bug_export_job is None and _rendered(app, "Bug report written to:"))
        assert app._diag is not None
        diag_dir = app._diag.dir

    # /bug wrote a shareable zip with the projection, canonical events, and artifact manifest.
    bundles = list(diag_dir.glob("bug-*.zip"))
    assert bundles, "/bug did not write a zip"
    bundle = bundles[0]
    with zipfile.ZipFile(bundle) as zf:
        names = zf.namelist()
    assert "summary.md" in names
    assert "session.json" in names
    assert "session-events.jsonl" in names
    assert "session-artifacts.json" in names

    system_entries = [e.content for e in app.conversation_entries if e.kind == "system"]
    assert any("Diagnostics log" in c for c in system_entries)  # /diagnostics summary
    assert any("written to" in c for c in system_entries)  # /bug output
