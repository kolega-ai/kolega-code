import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from kolega_code.agent.session_metadata import SessionMetadata
from kolega_code.cli.config import config_summary
from kolega_code.cli.session_store import SessionStore
from tests.cli._app_test_utils import build_test_config, install_fake_agents


@pytest.mark.asyncio
async def test_retitle_command_with_explicit_title(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")
    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import Static

    install_fake_agents(monkeypatch)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config), name="worker-01")
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        assert app.session.name == "worker-01"
        assert app.session.title == "worker-01"

        await app._command_retitle("Implement User OAuth Flow")
        await pilot.pause()

        assert app.session.name == "worker-01"
        assert app.session.title == "Implement User OAuth Flow"

        # Compact metadata reserves space for mode/permissions; the full session
        # title and name remain available in the Session card and startup details.
        session_content = str(app.query_one("#status_session", Static).render())
        assert "Implement User OAuth Flow" in session_content
        assert "worker-01" in session_content
        assert "Implement User OAuth Flow" in app.conversation_entries[0].content

        assert app.session.title_pinned is True
        assert any(
            "Session title pinned: 'Implement User OAuth Flow'" in str(entry.content)
            for entry in app.conversation_entries
        )


@pytest.mark.asyncio
async def test_retitle_command_automatic_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")
    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import Static

    install_fake_agents(monkeypatch)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config), name="worker-02")
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        from unittest.mock import MagicMock
        from kolega_code.llm.models import Message, TextBlock

        assert app.agent is not None
        app.agent.llm = MagicMock()
        app.agent.history.append(Message(role="user", content=[TextBlock(text="Help me debug memory leak")]))
        app.agent.history.append(Message(role="assistant", content=[TextBlock(text="Found unclosed files.")]))

        # Mock metadata generator to return structured metadata
        fake_metadata = SessionMetadata(
            title="Investigate Memory Leak",
            description="Profiled heap usage and identified unclosed file handles.",
        )
        with patch("kolega_code.agent.session_metadata.generate_session_metadata", return_value=fake_metadata):
            await app._update_session_metadata_worker(user_requested=True)
            await pilot.pause()

        assert app.session.name == "worker-02"
        assert app.session.title == "Investigate Memory Leak"
        assert app.session.description == "Profiled heap usage and identified unclosed file handles."

        # Session card in sidebar displays title and handle
        session_card = str(app.query_one("#status_session", Static).render())
        assert "Investigate Memory Leak" in session_card
        assert "worker-02" in session_card


@pytest.mark.asyncio
async def test_rename_preserves_custom_title(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")
    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config), name="worker-old", title="Custom Feature Work")
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        assert app.session.name == "worker-old"
        assert app.session.title == "Custom Feature Work"

        await app._command_rename("worker-renamed")
        await pilot.pause()

        # Handle changed, but title remained intact
        assert app.session.name == "worker-renamed"
        assert app.session.title == "Custom Feature Work"


GENERATOR = "kolega_code.agent.session_metadata.generate_session_metadata"


def _offered_title(generate: Any) -> str:
    assert generate.await_args is not None
    return generate.await_args.kwargs["current_title"]


def _app_with_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, SessionStore]:
    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch)
    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config), name="worker-03")
    return KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session), store


def _seed_history(app: Any) -> None:
    from kolega_code.llm.models import Message, TextBlock

    assert app.agent is not None
    app.agent.llm = MagicMock()
    app.agent.history.append(Message(role="user", content=[TextBlock(text="Refactor the payments platform")]))
    app.agent.history.append(Message(role="assistant", content=[TextBlock(text="Why is the webhook retrying?")]))


@pytest.mark.asyncio
async def test_pinned_title_survives_automatic_refresh_and_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")
    app, store = _app_with_history(tmp_path, monkeypatch)

    async with app.run_test():
        _seed_history(app)
        await app._command_retitle("Payments Platform Refactor")

        refreshed = SessionMetadata(title="Debug Webhook Retry Loop", description="Traced webhook retries.")
        with patch(GENERATOR, return_value=refreshed) as generate:
            await app._update_session_metadata_worker()

        assert _offered_title(generate) == ""
        assert app.session.title == "Payments Platform Refactor"
        assert app.session.description == "Traced webhook retries."

    reloaded = store.load(app.session.session_id)
    assert reloaded.title == "Payments Platform Refactor"
    assert reloaded.title_pinned is True
    assert reloaded.description == "Traced webhook retries."


@pytest.mark.asyncio
async def test_title_pinned_during_generation_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")
    app, _store = _app_with_history(tmp_path, monkeypatch)
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_generate(*_args: Any, **_kwargs: Any) -> SessionMetadata:
        started.set()
        await release.wait()
        return SessionMetadata(title="Debug Webhook Retry Loop", description="Traced webhook retries.")

    async with app.run_test():
        _seed_history(app)
        with patch(GENERATOR, new=slow_generate):
            refresh = asyncio.create_task(app._update_session_metadata_worker())
            await started.wait()
            await app._command_retitle("Payments Platform Refactor")
            release.set()
            await refresh

        assert app.session.title == "Payments Platform Refactor"


@pytest.mark.asyncio
async def test_bare_retitle_unpins_and_asks_for_a_fresh_title(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")
    app, store = _app_with_history(tmp_path, monkeypatch)

    async with app.run_test():
        _seed_history(app)
        await app._command_retitle("Payments Platform Refactor")

        fresh = SessionMetadata(title="Debug Webhook Retry Loop", description="Traced webhook retries.")
        with patch(GENERATOR, return_value=fresh) as generate:
            await app._command_retitle("")
            (regenerate,) = [worker for worker in app.workers if worker.name == "kolega-session-metadata"]
            await regenerate.wait()

        assert _offered_title(generate) == ""
        assert app.session.title == "Debug Webhook Retry Loop"
        assert app.session.title_pinned is False

    assert store.load(app.session.session_id).title_pinned is False


@pytest.mark.asyncio
async def test_automatic_refresh_offers_the_current_title_to_keep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")
    app, _store = _app_with_history(tmp_path, monkeypatch)

    async with app.run_test():
        _seed_history(app)
        first = SessionMetadata(title="Payments Platform Refactor", description="Refactoring payments.")
        with patch(GENERATOR, return_value=first) as generate:
            await app._update_session_metadata_worker()
            # The placeholder title (the session name) is not offered to keep.
            assert _offered_title(generate) == ""

            await app._update_session_metadata_worker()
            assert _offered_title(generate) == "Payments Platform Refactor"
