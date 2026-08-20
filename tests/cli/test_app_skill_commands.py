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
from kolega_code.llm.models import TextBlock, ToolCall, ToolResult
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
async def test_textual_app_skill_slash_commands_list_and_activate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

    install_fake_agents(monkeypatch)

    project = tmp_path / "project"
    skill_dir = project / ".agents" / "skills" / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: Use this demo skill.\n---\n\nFollow demo instructions.\n",
        encoding="utf-8",
    )
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("/skills")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        assert app.conversation_entries[-1].kind == "system"
        assert "`/demo-skill`" in app.conversation_entries[-1].content

        composer.load_text("/demo-skill")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        assert app.conversation_entries[-1].kind == "skill"
        assert app.conversation_entries[-1].content == "Activated skill `/demo-skill`."
        assert "Follow demo instructions." not in app.conversation_entries[-1].content
        assert app.agent is not None
        assert '<skill_content name="demo-skill">' in app.agent.history[-1].get_text_content()
        assert '<skill_content name="demo-skill">' in store.load(session.session_id).history[-1]["content"][0]["text"]


@pytest.mark.asyncio
async def test_textual_app_skill_slash_command_with_prompt_starts_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

    install_fake_agents(monkeypatch)

    project = tmp_path / "project"
    skill_dir = project / ".agents" / "skills" / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: Use this demo skill.\n---\n\nFollow demo instructions.\n",
        encoding="utf-8",
    )
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("/demo-skill Build the feature")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        await pilot.pause()

        assert app.agent is not None
        assert getattr(app.agent, "messages") == ["/demo-skill Build the feature"]
        assert any(
            entry.kind == "skill" and entry.content == "Activated skill `/demo-skill`."
            for entry in app.conversation_entries
        )
        assert any(
            entry.kind == "user" and entry.content == "/demo-skill Build the feature"
            for entry in app.conversation_entries
        )
        # The skill body must never leak into the transcript as a wall of text.
        assert all("Follow demo instructions." not in entry.content for entry in app.conversation_entries)


def _text_history_message(role: str, text: str) -> dict:
    return {"role": role, "content": [{"type": "text", "text": text}]}


@pytest.mark.asyncio
async def test_skill_invocation_with_arguments_restores_identically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch)

    project = tmp_path / "project"
    skill_dir = project / ".agents" / "skills" / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: Use this demo skill.\n---\n\nFollow demo instructions.\n",
        encoding="utf-8",
    )
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    # History exactly as the live path records it: the activation context
    # message, then the submitted command line as the user turn message.
    history = [
        _text_history_message("user", '<skill_content name="demo-skill">Follow demo instructions.</skill_content>'),
        _text_history_message("user", "/demo-skill Build the feature"),
    ]

    async with app.run_test():
        app._restore_conversation_history(history)

        restored = [entry for entry in app.conversation_entries if entry.kind in {"skill", "user"}]
        assert [(entry.kind, entry.content) for entry in restored] == [
            ("skill", "Activated skill `/demo-skill`."),
            ("user", "/demo-skill Build the feature"),
        ]
        assert all("Follow demo instructions." not in entry.content for entry in app.conversation_entries)
