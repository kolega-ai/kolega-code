# ruff: noqa: F401,F811,E402
from pathlib import Path
import asyncio
import json
import time

import pytest

from kolega_code.config import ModelProvider
from kolega_code.llm.exceptions import (
    LLMBillingError,
    LLMAuthenticationError,
    LLMContextWindowExceededError,
    LLMError,
    LLMInternalServerError,
)
from kolega_code.llm.models import Message, TextBlock, ToolCall, ToolResult
from kolega_code.events import AgentEvent
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.cli.config import build_agent_config, config_summary
from kolega_code.cli.provider_registry import (
    DEEPSEEK_DEFAULT_MODEL,
    MOONSHOT_K26_MODEL,
    UI_DEFAULT_MODEL,
    UI_DEFAULT_PROVIDER,
)
from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.settings import CliSettings, SettingsStore

from ._app_test_utils import (
    FakeCoderAgent,
    _build_mention_test_app,
    _build_sub_agent_test_app,
    _sub_agent_context_event,
    _sub_agent_entries,
    _sub_agent_event,
    _workflow_event,
    build_test_config,
    extension_by_name,
    first_text_styles,
    install_fake_agents,
    question_payload,
    renderable_text,
)


@pytest.mark.asyncio
async def test_textual_app_plan_and_build_slash_commands_switch_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

    class FakePlanningAgent(FakeCoderAgent):
        pass

    install_fake_agents(monkeypatch, planning_cls=FakePlanningAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        assert app.interaction_mode == "build"

        composer.load_text("/plan")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        assert app.interaction_mode == "plan"
        assert isinstance(app.agent, FakePlanningAgent)
        assert composer.text == ""

        composer.load_text("/build")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        assert app.interaction_mode == "build"
        assert isinstance(app.agent, FakeCoderAgent)


@pytest.mark.asyncio
async def test_textual_app_sidebar_slash_command_toggles_sidebar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        side_panel = app.query_one("#side_panel")

        composer.load_text("/sidebar")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        assert composer.text == ""
        assert app.sidebar_visible is False
        assert side_panel.display is False

        composer.load_text("/sidebar")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        assert composer.text == ""
        assert app.sidebar_visible is True
        assert side_panel.display is True


@pytest.mark.asyncio
async def test_textual_app_init_slash_command_starts_agents_md_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("/init focus on test commands")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        await pilot.pause()

        assert composer.text == ""
        assert app.agent is not None
        messages = getattr(app.agent, "messages")
        assert len(messages) == 1
        prompt = messages[0]
        assert "Create or update `AGENTS.md` for this repository." in prompt
        assert "`focus on test commands`" in prompt
        assert "$ARGUMENTS" not in prompt
        assert any(
            entry.kind == "user" and entry.content == "/init focus on test commands"
            for entry in app.conversation_entries
        )


@pytest.mark.asyncio
async def test_textual_app_init_slash_command_switches_from_plan_to_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

    class FakePlanningAgent(FakeCoderAgent):
        pass

    install_fake_agents(monkeypatch, planning_cls=FakePlanningAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("/plan")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        assert app.interaction_mode == "plan"
        assert isinstance(app.agent, FakePlanningAgent)

        composer.load_text("/init focus on docs")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        await pilot.pause()

        assert app.interaction_mode == "build"
        assert isinstance(app.agent, FakeCoderAgent)
        assert app.agent.messages
        assert "`focus on docs`" in app.agent.messages[0]


@pytest.mark.asyncio
async def test_textual_app_init_slash_command_blocks_during_active_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Static

    from kolega_code.cli.tui.widgets import ChatComposer

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        app._turn_active = True
        composer.load_text("/init")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        assert app.agent is not None
        assert getattr(app.agent, "messages") == []
        assert "Stop the current turn before running /init." in str(app.query_one("#composer_hint", Static).render())
        app._turn_active = False
