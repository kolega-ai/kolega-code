from pathlib import Path

import pytest

from kolega_code.cli.main import CLI_AGENT_MODE, RESUME_LATEST, _resolve_tui_session, main, parse_args
from kolega_code.cli.provider_registry import UI_DEFAULT_MODEL, UI_DEFAULT_PROVIDER
from kolega_code.cli.session_store import SessionStore, SessionStoreError
from kolega_code.cli.settings import CliSettings, SettingsStore


def test_parse_default_command_as_tui() -> None:
    args = parse_args(["/tmp/project", "--new"])

    assert args.command == "tui"
    assert args.project_path == Path("/tmp/project")
    assert args.new is True
    assert args.resume is None
    assert args.mode == CLI_AGENT_MODE


def test_parse_tui_resume_latest() -> None:
    args = parse_args(["/tmp/project", "--resume"])

    assert args.command == "tui"
    assert args.resume == RESUME_LATEST


def test_parse_tui_resume_specific_thread() -> None:
    args = parse_args(["/tmp/project", "--resume", "thread-123"])

    assert args.command == "tui"
    assert args.resume == "thread-123"


def test_parse_tui_legacy_session_alias() -> None:
    args = parse_args(["/tmp/project", "--session", "session-123"])

    assert args.command == "tui"
    assert args.session == "session-123"


def test_parse_ask_subcommand() -> None:
    args = parse_args(["ask", "hello", "--project", "/tmp/project", "--save", "--json"])

    assert args.command == "ask"
    assert args.prompt == "hello"
    assert args.project == Path("/tmp/project")
    assert args.save is True
    assert args.json is True
    assert args.mode == CLI_AGENT_MODE


def test_parse_sessions_list_subcommand() -> None:
    args = parse_args(["sessions", "list", "--project", "/tmp/project"])

    assert args.command == "sessions"
    assert args.sessions_command == "list"
    assert args.project == Path("/tmp/project")


def test_doctor_uses_stored_kimi_settings(tmp_path: Path, capsys, isolated_cli_env: None) -> None:
    project = tmp_path / "project"
    project.mkdir()
    state_dir = tmp_path / "state"
    settings = CliSettings(active_provider=UI_DEFAULT_PROVIDER, active_model=UI_DEFAULT_MODEL)
    settings.set_api_key(UI_DEFAULT_PROVIDER, "moonshot-key")
    SettingsStore(state_dir).save(settings)

    exit_code = main(["doctor", "--project", str(project), "--state-dir", str(state_dir)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert f"Stored active model: {UI_DEFAULT_PROVIDER}/{UI_DEFAULT_MODEL}" in output
    assert "present in local settings" in output
    assert "moonshot-key" not in output


def test_tui_default_creates_new_session_even_when_latest_exists(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    existing = store.create(project, "code", {})

    session = _resolve_tui_session(store, project, {}, resume=None, legacy_session_id=None)

    assert session.session_id != existing.session_id
    assert session.thread_id != existing.thread_id
    assert session.mode == CLI_AGENT_MODE


def test_tui_resume_without_id_loads_latest_project_session(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    store.create(project, "code", {}, title="older")
    newer = store.create(project, "code", {}, title="newer")

    session = _resolve_tui_session(store, project, {}, resume=RESUME_LATEST, legacy_session_id=None)

    assert session.session_id == newer.session_id
    assert session.mode == CLI_AGENT_MODE
    assert store.load(newer.session_id).mode == CLI_AGENT_MODE


def test_tui_resume_specific_session_id(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    existing = store.create(project, "code", {})

    session = _resolve_tui_session(store, project, {}, resume=existing.session_id, legacy_session_id=None)

    assert session.session_id == existing.session_id
    assert session.mode == CLI_AGENT_MODE
    assert store.load(existing.session_id).mode == CLI_AGENT_MODE


def test_tui_resume_specific_thread_id(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    existing = store.create(project, "code", {})

    session = _resolve_tui_session(store, project, {}, resume=existing.thread_id, legacy_session_id=None)

    assert session.session_id == existing.session_id
    assert session.mode == CLI_AGENT_MODE


def test_tui_resume_missing_id_raises(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")

    with pytest.raises(SessionStoreError):
        _resolve_tui_session(store, project, {}, resume="missing-thread", legacy_session_id=None)


def test_tui_resume_project_mismatch_raises(tmp_path: Path) -> None:
    project = tmp_path / "project"
    other_project = tmp_path / "other"
    project.mkdir()
    other_project.mkdir()
    store = SessionStore(tmp_path / "state")
    existing = store.create(other_project, "code", {})

    with pytest.raises(SessionStoreError, match="belongs to project"):
        _resolve_tui_session(store, project, {}, resume=existing.thread_id, legacy_session_id=None)


def test_tui_legacy_session_alias_loads_specific_session(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    existing = store.create(project, "code", {})

    session = _resolve_tui_session(
        store,
        project,
        {},
        resume=None,
        legacy_session_id=existing.session_id,
    )

    assert session.session_id == existing.session_id
    assert session.mode == CLI_AGENT_MODE
