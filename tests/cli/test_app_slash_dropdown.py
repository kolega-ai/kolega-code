# ruff: noqa: F401,F811,E402
from pathlib import Path
import asyncio
import json
import time

import pytest

from kolega_code.cli.tui import agent_runtime as agent_runtime_module

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
    _build_mention_test_app,
    _build_sub_agent_test_app,
    _sub_agent_context_event,
    _sub_agent_entries,
    _sub_agent_event,
    _workflow_event,
    build_test_config,
    extension_by_name,
    first_text_styles,
    question_payload,
    renderable_text,
)


@pytest.mark.asyncio
async def test_textual_app_slash_dropdown_opens_filters_and_tab_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer, CompletionDropdown
    from kolega_code.cli.slash_commands import SlashCommandEntry

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        dropdown = app.query_one("#completion_dropdown", CompletionDropdown)
        composer.focus()
        await pilot.pause()

        composer.insert("/")
        await pilot.pause()
        assert dropdown.is_open
        assert dropdown.option_count > 1
        assert isinstance(dropdown.highlighted_entry(), SlashCommandEntry)

        composer.insert("pl")
        await pilot.pause()
        assert dropdown.is_open
        highlighted = dropdown.highlighted_entry()
        assert isinstance(highlighted, SlashCommandEntry)
        assert highlighted.name == "plan"

        await pilot.press("tab")
        assert composer.text == "/plan "
        assert not dropdown.is_open


@pytest.mark.asyncio
async def test_textual_app_slash_dropdown_lists_skills_with_descriptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer, CompletionDropdown
    from kolega_code.cli.slash_commands import SlashCommandEntry

    app = _build_mention_test_app(tmp_path, monkeypatch)
    skill_dir = app.project_path / ".agents" / "skills" / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: Use this demo skill.\n---\n\nFollow demo instructions.\n",
        encoding="utf-8",
    )

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        dropdown = app.query_one("#completion_dropdown", CompletionDropdown)
        composer.focus()
        composer.insert("/demo")
        await pilot.pause()

        assert dropdown.is_open
        entry = dropdown.highlighted_entry()
        assert isinstance(entry, SlashCommandEntry)
        assert entry.name == "demo-skill"
        assert entry.description == "Use this demo skill."

        await pilot.press("tab")
        assert composer.text == "/demo-skill "


@pytest.mark.asyncio
async def test_textual_app_slash_dropdown_enter_completes_instead_of_submitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer, CompletionDropdown

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        dropdown = app.query_one("#completion_dropdown", CompletionDropdown)
        composer.focus()
        composer.insert("/versio")
        await pilot.pause()
        assert dropdown.is_open

        await pilot.press("enter")
        await pilot.pause()
        assert composer.text == "/version "
        assert not dropdown.is_open
        assert app.agent is not None
        # The helper's fake coder agent records submitted messages on ``messages``;
        # completing a slash command must not submit anything.
        assert getattr(app.agent, "messages") == []


@pytest.mark.asyncio
async def test_textual_app_slash_dropdown_does_not_open_mid_text_or_after_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer, CompletionDropdown

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        dropdown = app.query_one("#completion_dropdown", CompletionDropdown)
        composer.focus()

        composer.insert("see src/")
        await pilot.pause()
        assert not dropdown.is_open

        composer.load_text("")
        composer.insert("first line")
        composer.action_insert_newline()
        composer.insert("/")
        await pilot.pause()
        assert not dropdown.is_open

        composer.load_text("")
        composer.insert("/skills extra")
        await pilot.pause()
        assert not dropdown.is_open
