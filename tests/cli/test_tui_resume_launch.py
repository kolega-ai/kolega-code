from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.cli import main as main_module
from kolega_code.cli.main import _run_tui, parse_args
from kolega_code.cli.session_store import SessionRecord, SessionStoreError


CLI_MODE = AgentMode.CLI.value


class _Config:
    mcp_config = None


class _UsageSink:
    def __init__(self, events: list[str], app: Any) -> None:
        self._events = events
        self._app = app

    async def aclose(self) -> None:
        self._events.append(f"sink:{self._app.session.session_id}")


class _FakeStore:
    def __init__(self, root: Path, events: list[str]) -> None:
        self.root = root
        self._events = events
        self.records: dict[str, SessionRecord] = {}
        self.held_locks: set[str] = set()
        self.created = 0
        self.saved_snapshots: list[tuple[str, str, str]] = []

    def create(self, project_path: Path, mode: str, summary: dict[str, Any]) -> SessionRecord:
        del summary
        self.created += 1
        session_id = f"created-{self.created}"
        record = SessionRecord.create(project_path, mode, {}, session_id=session_id, name=session_id)
        self.records[record.session_id] = record
        self._events.append(f"create:{record.session_id}")
        return record

    def load(self, session_id: str) -> SessionRecord:
        try:
            return self.records[session_id]
        except KeyError as exc:
            raise SessionStoreError(f"Session not found: {session_id}") from exc

    def load_session_or_thread(self, session_or_thread_id: str) -> SessionRecord:
        if session_or_thread_id in self.records:
            return self.records[session_or_thread_id]
        for record in self.records.values():
            if record.thread_id == session_or_thread_id:
                return record
        raise SessionStoreError(f"Session not found: {session_or_thread_id}")

    def latest_for_project(self, project_path: Path) -> SessionRecord | None:
        resolved = str(project_path.resolve())
        for record in reversed(list(self.records.values())):
            if record.project_path == resolved:
                return record
        return None

    def save(self, record: SessionRecord) -> None:
        self.records[record.session_id] = record
        self.saved_snapshots.append((record.session_id, record.name, record.permission_mode))
        self._events.append(f"save:{record.session_id}:{record.name}:{record.permission_mode}")

    def lock_session(self, session_id: str) -> None:
        self.held_locks.add(session_id)
        self._events.append(f"lock:{session_id}")

    def release_session_lock(self, session_id: str) -> None:
        self._events.append(f"release_one:{session_id}:target_held={'selected' in self.held_locks}")
        self.held_locks.discard(session_id)

    def release_session_locks(self) -> None:
        held = ",".join(sorted(self.held_locks))
        self._events.append(f"release_all:{held}")
        self.held_locks.clear()


def _install_harness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    events: list[str],
    scripts: list[dict[str, Any]],
) -> _FakeStore:
    store = _FakeStore(tmp_path / "state", events)
    original_find_spec = main_module.importlib.util.find_spec
    monkeypatch.setattr(
        main_module.importlib.util,
        "find_spec",
        lambda name: object() if name == "textual" else original_find_spec(name),
    )

    def store_from_args(_args: Any) -> _FakeStore:
        del _args
        return store

    def build_agent_config(*_args: Any, **_kwargs: Any) -> _Config:
        del _args, _kwargs
        return _Config()

    def config_summary(_config: _Config) -> dict[str, Any]:
        del _config
        return {}

    monkeypatch.setattr(main_module, "_store_from_args", store_from_args)
    monkeypatch.setattr(main_module, "build_agent_config", build_agent_config)
    monkeypatch.setattr(main_module, "config_summary", config_summary)
    monkeypatch.setattr(main_module, "_print_quit_resume_hint", lambda session_id: events.append(f"hint:{session_id}"))

    class FakeApp:
        instances: list[FakeApp] = []

        def __init__(self, **kwargs: Any) -> None:
            script = scripts[len(self.instances)]
            self.__class__.instances.append(self)
            self.kwargs = kwargs
            self.store: _FakeStore = kwargs["store"]
            self.session: SessionRecord = kwargs["session"]
            self.resume_session: SessionRecord | None = None
            self._quit_cleanly = bool(script.get("clean", True))
            self._script = script
            self._usage_sink = _UsageSink(events, self)
            target_held = "selected" in self.store.held_locks
            self.store.lock_session(self.session.session_id)
            events.append(
                "init:"
                f"{self.session.session_id}:"
                f"resuming={kwargs['resuming_session']}:"
                f"permission={kwargs['permission_mode']}:"
                f"name={self.session.name}:"
                f"target_held={target_held}:"
                f"show_logs={kwargs['show_logs']}"
            )

        async def run_async(self) -> None:
            events.append(f"run:{self.session.session_id}")
            selected = self._script.get("select")
            if selected is not None:
                self.store.lock_session(selected.session_id)
                self.resume_session = selected
            final_session = self._script.get("final_session")
            if final_session is not None:
                self.store.records[final_session.session_id] = final_session
                self.session = final_session
            if self._script.get("pending_shutdown"):
                self._quit_cleanly = False

                async def finish_shutdown() -> None:
                    await asyncio.sleep(0.01)
                    events.append(f"shutdown-finished:{self.session.session_id}")
                    self._quit_cleanly = True

                self._session_shutdown_task = asyncio.create_task(finish_shutdown())

        async def _cleanup_agent_generation(self) -> None:
            events.append(f"cleanup:{self.session.session_id}")
            failure = self._script.get("cleanup_error")
            if failure is not None:
                raise failure

    monkeypatch.setitem(sys.modules, "kolega_code.cli.app", types.SimpleNamespace(KolegaCodeApp=FakeApp))
    return store


def _record(project: Path, session_id: str, *, name: str = "", permission_mode: str = "ask") -> SessionRecord:
    record = SessionRecord.create(project, CLI_MODE, {}, session_id=session_id, name=name or session_id)
    record.permission_mode = permission_mode
    return record


def test_tui_resume_selection_relaunches_after_teardown_and_prints_final_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
) -> None:
    del isolated_cli_env
    project = tmp_path / "project"
    project.mkdir()
    events: list[str] = []
    selected = _record(project, "selected", name="selected-name", permission_mode="ask")
    final = _record(project, "handoff-final", name="handoff-final", permission_mode="auto")
    scripts = [
        {"select": selected, "clean": True},
        {"final_session": final, "clean": True},
    ]
    store = _install_harness(monkeypatch, tmp_path, events, scripts)
    store.records[selected.session_id] = selected

    args = parse_args(
        [
            "tui",
            str(project),
            "--state-dir",
            str(tmp_path / "state"),
            "--name",
            "launch-name",
            "--permission-mode",
            "auto",
            "--show-logs",
        ]
    )

    assert _run_tui(args) == 0

    assert [event for event in events if event.startswith("init:")] == [
        "init:created-1:resuming=False:permission=auto:name=launch-name:target_held=False:show_logs=True",
        "init:selected:resuming=True:permission=auto:name=selected-name:target_held=True:show_logs=True",
    ]
    assert [event for event in events if event.startswith("hint:")] == ["hint:handoff-final"]
    assert store.records[selected.session_id].name == "selected-name"
    assert store.records[selected.session_id].permission_mode == "auto"
    assert (
        events.index("cleanup:created-1")
        < events.index("sink:created-1")
        < events.index("release_one:created-1:target_held=True")
        < events.index("init:selected:resuming=True:permission=auto:name=selected-name:target_held=True:show_logs=True")
    )
    assert events[-1] == "release_all:selected"


def test_tui_shutdown_failure_does_not_launch_selected_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
) -> None:
    del isolated_cli_env
    project = tmp_path / "project"
    project.mkdir()
    events: list[str] = []
    selected = _record(project, "selected")
    failure = RuntimeError("cleanup failed")
    scripts = [{"select": selected, "clean": True, "cleanup_error": failure}]
    store = _install_harness(monkeypatch, tmp_path, events, scripts)
    store.records[selected.session_id] = selected

    def write_crash_log(_root: Path, *, exc: BaseException, header: str, secret_values: list[str]) -> None:
        del _root, exc, secret_values
        events.append(f"crash:{header}")

    monkeypatch.setattr(main_module, "write_crash_log", write_crash_log)
    args = parse_args(["tui", str(project), "--state-dir", str(tmp_path / "state")])

    with pytest.raises(RuntimeError, match="cleanup failed"):
        _run_tui(args)

    assert [event for event in events if event.startswith("init:")] == [
        "init:created-1:resuming=False:permission=ask:name=created-1:target_held=False:show_logs=False"
    ]
    assert "sink:created-1" in events
    assert not any(event == "init:selected" for event in events)
    assert "release_one:created-1:target_held=True" not in events
    assert "crash:kolega-code crash | session created-1" in events
    assert events[-1] == "release_all:created-1,selected"


def test_tui_waits_for_inflight_shutdown_before_releasing_old_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_cli_env: None
) -> None:
    del isolated_cli_env
    project = tmp_path / "project"
    project.mkdir()
    events: list[str] = []
    selected = _record(project, "selected")
    store = _install_harness(
        monkeypatch, tmp_path, events, [{"select": selected, "pending_shutdown": True}, {"clean": True}]
    )
    store.records[selected.session_id] = selected
    assert _run_tui(parse_args(["tui", str(project)])) == 0
    assert (
        events.index("shutdown-finished:created-1")
        < events.index("cleanup:created-1")
        < events.index("release_one:created-1:target_held=True")
    )
    assert [event for event in events if event.startswith("hint:")] == ["hint:selected"]
    assert events[-1] == "release_all:selected"


def test_tui_normal_no_selection_remains_single_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_cli_env: None,
) -> None:
    del isolated_cli_env
    project = tmp_path / "project"
    project.mkdir()
    events: list[str] = []
    scripts = [{"clean": True}]
    _install_harness(monkeypatch, tmp_path, events, scripts)
    args = parse_args(["tui", str(project), "--state-dir", str(tmp_path / "state"), "--name", "solo"])

    assert _run_tui(args) == 0

    assert [event for event in events if event.startswith("init:")] == [
        "init:created-1:resuming=False:permission=ask:name=solo:target_held=False:show_logs=False"
    ]
    assert [event for event in events if event.startswith("hint:")] == ["hint:created-1"]
    assert not any(event.startswith("release_one:") for event in events)
    assert events[-1] == "release_all:created-1"
