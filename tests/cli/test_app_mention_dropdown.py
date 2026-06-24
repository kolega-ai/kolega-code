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
async def test_textual_app_mention_dropdown_opens_and_escape_dismisses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer, CompletionDropdown

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        dropdown = app.query_one("#completion_dropdown", CompletionDropdown)
        composer.focus()
        await pilot.pause()

        composer.insert("@alp")
        await pilot.pause()
        assert dropdown.is_open
        assert dropdown.option_count > 0

        await pilot.press("escape")
        assert not dropdown.is_open
        assert composer.text == "@alp"


@pytest.mark.asyncio
async def test_textual_app_mention_dropdown_not_opened_by_email_address(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer, CompletionDropdown

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        dropdown = app.query_one("#completion_dropdown", CompletionDropdown)
        composer.focus()
        composer.insert("mail user@example")
        await pilot.pause()
        assert not dropdown.is_open


@pytest.mark.asyncio
async def test_textual_app_mention_dropdown_down_and_tab_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer, CompletionDropdown

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        dropdown = app.query_one("#completion_dropdown", CompletionDropdown)
        composer.focus()
        composer.insert("@alp")
        await pilot.pause()
        assert dropdown.is_open

        expected = dropdown.entry_at(1).path
        await pilot.press("down")
        assert dropdown.highlighted == 1
        await pilot.press("tab")
        assert composer.text == f"@{expected} "
        assert not dropdown.is_open


@pytest.mark.asyncio
async def test_textual_app_mention_enter_completes_instead_of_submitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer, CompletionDropdown

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        dropdown = app.query_one("#completion_dropdown", CompletionDropdown)
        composer.focus()
        composer.insert("@README")
        await pilot.pause()
        assert dropdown.is_open

        await pilot.press("enter")
        await pilot.pause()
        assert composer.text == "@README.md "
        assert not dropdown.is_open
        # No message was submitted, only the completion was applied.
        assert app.agent.messages == []


@pytest.mark.asyncio
async def test_textual_app_submitting_mention_attaches_file_and_keeps_short_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("summarize @src/alpha.py please")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        await pilot.pause()

        assert app.agent.messages == ["summarize @src/alpha.py please"]
        attachments = app.agent.attachments[0]
        assert attachments is not None and len(attachments) == 1
        assert attachments[0]["type"] == "file"
        assert attachments[0]["path"] == "src/alpha.py"
        assert attachments[0]["content"] == "print('alpha')\n"
        assert any(
            entry.kind == "user" and entry.content == "summarize @src/alpha.py please"
            for entry in app.conversation_entries
        )


@pytest.mark.asyncio
async def test_textual_app_unresolved_mention_clears_hint_and_sends_plain_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Static

    from kolega_code.cli.tui.widgets import ChatComposer

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("look at @does/not/exist.py")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))

        # Unresolved mentions are sent as plain text; the compose-time hint must
        # not linger after the message has been submitted.
        hint = app.query_one("#composer_hint", Static)
        assert str(hint.render()) == ""

        await pilot.pause()
        assert app.agent.messages == ["look at @does/not/exist.py"]
        assert app.agent.attachments == [None]
