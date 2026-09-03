from pathlib import Path
from unittest.mock import patch

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

        # Verify header reflects the title and name
        meta_content = str(app.query_one("#session_meta", Static).render())
        assert "Implement User OAuth Flow" in meta_content
        assert "worker-01" in meta_content

        # Verify conversation entry
        assert any(
            "Session title updated: **Implement User OAuth Flow**" in str(entry.content)
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
            await app._update_session_metadata_worker(force_notify=True)
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
