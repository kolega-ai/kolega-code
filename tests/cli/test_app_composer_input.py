# ruff: noqa: F401,F811,E402
from pathlib import Path
import asyncio
import json
import time

import pytest

from kolega_code.cli import messages
from kolega_code.cli.tui import agent_runtime as agent_runtime_module
from kolega_code.cli.tui import state as tui_state

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
async def test_textual_app_composer_shift_enter_inserts_line_break_and_enter_submits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.messages: list[str] = []

        def restore_message_history(self, history):
            return None

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

        async def process_message_stream(self, message):
            self.messages.append(message)
            yield {"type": "response", "content": "ok", "complete": True, "uuid": "response-1"}

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.focus()

        await pilot.press("h", "i")
        await pilot.press("shift+enter")
        await pilot.press("t", "h", "e", "r", "e")
        assert composer.text == "hi\nthere"

        await pilot.press("enter")
        await pilot.pause()

        assert app.agent is not None
        assert getattr(app.agent, "messages") == ["hi\nthere"]
        assert composer.text == ""
        user_entries = [entry for entry in app.conversation_entries if entry.kind == "user"]
        assert user_entries[-1].content == "hi\nthere"


@pytest.mark.asyncio
async def test_textual_app_composer_ctrl_enter_still_inserts_line_break(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.focus()

        await pilot.press("h", "i")
        await pilot.press("ctrl+enter")
        await pilot.press("t", "h", "e", "r", "e")

        assert composer.text == "hi\nthere"


@pytest.mark.asyncio
async def test_textual_app_composer_repeated_ctrl_u_clears_multiline_draft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.focus()
        composer.load_text("one\ntwo\nthree")
        composer.move_cursor(composer.document.end, record_width=False)

        await pilot.press("ctrl+u", "ctrl+u", "ctrl+u")

        assert composer.text == ""
        assert composer.cursor_location == (0, 0)


@pytest.mark.asyncio
async def test_textual_app_composer_ctrl_u_preserves_within_line_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.focus()

        composer.load_text("abc def")
        composer.move_cursor(composer.document.end, record_width=False)
        await pilot.press("ctrl+u")
        assert composer.text == ""
        assert composer.cursor_location == (0, 0)

        composer.load_text("one\ntwo\nthree")
        composer.move_cursor((1, 2), record_width=False)
        await pilot.press("ctrl+u")
        assert composer.text == "one\no\nthree"
        assert composer.cursor_location == (1, 0)


@pytest.mark.asyncio
async def test_textual_app_composer_ctrl_u_boundaries_and_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.focus()

        composer.load_text("abc")
        composer.move_cursor((0, 0), record_width=False)
        await pilot.press("ctrl+u")
        assert composer.text == "abc"
        assert composer.cursor_location == (0, 0)

        composer.load_text("one\ntwo")
        composer.selection = composer.selection.__class__((0, 1), (0, 3))
        await pilot.press("ctrl+u")
        assert composer.text == "o\ntwo"
        assert composer.cursor_location == (0, 1)


@pytest.mark.asyncio
async def test_textual_app_composer_ctrl_l_selects_all_and_backspace_clears_draft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.focus()
        draft = "one\ntwo\nthree"
        composer.load_text(draft)
        composer.move_cursor(composer.document.end, record_width=False)

        await pilot.press("ctrl+l")

        assert composer.selected_text == draft
        assert composer.selection.start == (0, 0)
        assert composer.selection.end == composer.document.end

        await pilot.press("backspace")

        assert composer.text == ""
        assert composer.cursor_location == (0, 0)


@pytest.mark.asyncio
async def test_textual_app_composer_cmd_a_selects_all_and_backspace_clears_draft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.focus()
        draft = "one\ntwo\nthree"
        composer.load_text(draft)
        composer.move_cursor(composer.document.end, record_width=False)

        await pilot.press("super+a")

        assert composer.selected_text == draft
        assert composer.selection.start == (0, 0)
        assert composer.selection.end == composer.document.end

        await pilot.press("backspace")

        assert composer.text == ""
        assert composer.cursor_location == (0, 0)


@pytest.mark.asyncio
async def test_textual_app_composer_ctrl_a_still_moves_to_start_of_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.focus()
        composer.load_text("one\ntwo\nthree")
        composer.move_cursor((1, 2), record_width=False)

        await pilot.press("ctrl+a")

        assert composer.cursor_location == (1, 0)
        assert composer.selected_text == ""
        assert composer.text == "one\ntwo\nthree"


@pytest.mark.asyncio
async def test_textual_app_composer_preserves_multiline_paste(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from textual import events

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.messages: list[str] = []

        def restore_message_history(self, history):
            return None

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

        async def process_message_stream(self, message):
            self.messages.append(message)
            yield {"type": "response", "content": "ok", "complete": True, "uuid": "response-1"}

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    pasted = "line one\n    line two\nline three"
    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        composer.focus()

        await composer._on_paste(events.Paste(pasted))
        assert composer.text == pasted

        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        worker = app.agent_worker
        assert worker is not None
        await worker.wait()

        assert app.agent is not None
        assert getattr(app.agent, "messages") == [pasted]
        assert composer.text == ""


@pytest.mark.asyncio
async def test_textual_app_composer_auto_grows_caps_and_shrinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test(size=(120, 30)) as pilot:
        composer = app.query_one("#composer", ChatComposer)
        conversation = app.query_one("#conversation")

        initial_conversation_height = conversation.region.height
        assert composer.region.height == 5

        composer.load_text("short draft")
        await pilot.pause()
        assert composer.region.height == 5
        assert conversation.region.height == initial_conversation_height

        composer.load_text("\n".join(f"line {index}" for index in range(6)))
        await pilot.pause()
        assert 5 < composer.region.height < 15
        assert conversation.region.height < initial_conversation_height

        long_draft = "\n".join(f"line {index}" for index in range(50))
        composer.load_text(long_draft)
        await pilot.pause()
        assert composer.region.height == 15
        assert conversation.region.height > 0

        for index in range(20):
            app._add_conversation_entry(
                tui_state.ConversationEntry(
                    kind="agent",
                    content=f"transcript entry {index}\nmore content\nmore content",
                    complete=True,
                )
            )
        await pilot.pause()
        assert composer.vertical_scrollbar.display is True
        assert conversation.vertical_scrollbar.display is True
        assert composer.vertical_scrollbar.region.x == conversation.vertical_scrollbar.region.x
        assert composer.vertical_scrollbar.region.width == conversation.vertical_scrollbar.region.width

        composer.load_text("")
        await pilot.pause()
        assert composer.region.height == 5
        assert conversation.region.height == initial_conversation_height

        composer.load_text(long_draft)
        await pilot.pause()
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        worker = app.agent_worker
        assert worker is not None
        await worker.wait()
        await pilot.pause()

        assert composer.text == ""
        assert composer.region.height == 5
        assert conversation.region.height > 0


@pytest.mark.asyncio
async def test_textual_app_composer_auto_grows_for_soft_wrapped_long_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test(size=(60, 30)) as pilot:
        composer = app.query_one("#composer", ChatComposer)

        assert composer.region.height == 5

        composer.load_text("x" * 100)
        await pilot.pause()

        assert composer.virtual_size.height > 3
        assert composer.region.height > 5
        assert composer.region.height <= 15

        composer.load_text("")
        await pilot.pause()
        assert composer.region.height == 5


@pytest.mark.asyncio
async def test_chat_composer_active_slash_query(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.widgets import ChatComposer

    app = _build_mention_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        composer.focus()

        composer.insert("/")
        assert composer.active_slash_query() == ("", 0, 1)

        composer.insert("mod")
        assert composer.active_slash_query() == ("mod", 0, 4)

        composer.load_text("")
        composer.insert("  /he")
        assert composer.active_slash_query() == ("he", 2, 5)

        composer.load_text("")
        composer.insert("hello /he")
        assert composer.active_slash_query() is None

        composer.load_text("")
        composer.insert("/model kimi")
        assert composer.active_slash_query() is None


@pytest.mark.asyncio
async def test_textual_app_queues_multiple_active_turn_messages_fifo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

    first_started = asyncio.Event()
    release_first = asyncio.Event()

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.messages: list[str] = []

        def restore_message_history(self, history):
            return None

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

        async def process_message_stream(self, message, attachments=None):
            self.messages.append(message)
            first_started.set()
            await release_first.wait()
            yield {"type": "response", "content": "first done", "complete": True, "uuid": "response-1"}

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.focus()

        composer.load_text("first")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        first_worker = app.agent_worker
        assert first_worker is not None
        await asyncio.wait_for(first_started.wait(), timeout=2)

        composer.load_text("second")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        composer.load_text("third")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        await pilot.pause()

        assert [item.text for item in app._queued_messages] == ["second", "third"]
        assert not [entry for entry in app.conversation_entries if entry.content in {"second", "third"}]
        assert not [entry for entry in app.conversation_entries if entry.kind == "queued"]
        queued_panel = app.query_one("#queued_messages")
        assert queued_panel.display is True
        queued_text = renderable_text(queued_panel.render())
        assert "second" in queued_text
        assert "third" in queued_text

        release_first.set()
        await first_worker.wait()
        for _ in range(20):
            await pilot.pause()
            if (
                app.agent is not None
                and getattr(app.agent, "messages") == ["first", "second", "third"]
                and app.agent_worker is None
            ):
                break
        if app.agent_worker is not None:
            await app.agent_worker.wait()
        await pilot.pause()

        assert app.agent is not None
        assert getattr(app.agent, "messages") == ["first", "second", "third"]
        assert [entry.kind for entry in app.conversation_entries if entry.content in {"second", "third"}] == [
            "user",
            "user",
        ]
        user_contents = [entry.content for entry in app.conversation_entries if entry.kind == "user"]
        assert user_contents.count("second") == 1
        assert user_contents.count("third") == 1


@pytest.mark.asyncio
async def test_textual_app_queue_clear_command_discards_active_turn_followups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

    first_started = asyncio.Event()
    release_first = asyncio.Event()

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.messages: list[str] = []

        def restore_message_history(self, history):
            return None

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

        async def process_message_stream(self, message, attachments=None):
            self.messages.append(message)
            first_started.set()
            await release_first.wait()
            yield {"type": "response", "content": "first done", "complete": True, "uuid": "response-1"}

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)
        composer.focus()

        composer.load_text("first")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        first_worker = app.agent_worker
        assert first_worker is not None
        await asyncio.wait_for(first_started.wait(), timeout=2)

        for text in ("second", "third"):
            composer.load_text(text)
            await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        assert len(app._queued_messages) == 2
        assert not [entry for entry in app.conversation_entries if entry.content in {"second", "third"}]
        assert not [entry for entry in app.conversation_entries if entry.kind == "queued"]

        composer.load_text("/queue-clear")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        assert app._queued_messages == []

        release_first.set()
        await first_worker.wait()
        await pilot.pause()
        if app.agent_worker is not None:
            await app.agent_worker.wait()

        assert app.agent is not None
        assert getattr(app.agent, "messages") == ["first"]
        assert app.query_one("#queued_messages").display is False
        assert not [entry for entry in app.conversation_entries if entry.content in {"second", "third"}]
        assert any(messages.QUEUE_CLEARED.format(count=2) in entry.content for entry in app.conversation_entries)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("draft", "expected_text"),
    [
        ("draft", "second\n\nthird\n\ndraft"),
        ("", "second\n\nthird"),
    ],
)
async def test_textual_app_cancel_restores_queued_followups_to_composer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, draft: str, expected_text: str
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

    first_started = asyncio.Event()

    class FakeCoderAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def restore_message_history(self, history):
            return None

        def dump_compaction_state(self):
            return {}

        def restore_compaction_state(self, data):
            pass

        def dump_message_history(self):
            return []

        async def cleanup(self):
            return None

        async def process_message_stream(self, message, attachments=None):
            first_started.set()
            await asyncio.Event().wait()
            yield {"type": "response", "content": "unreachable", "complete": True, "uuid": "response-1"}

    monkeypatch.setattr(agent_runtime_module, "CoderAgent", FakeCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", ChatComposer)

        composer.load_text("first")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        assert app.agent_worker is not None
        await asyncio.wait_for(first_started.wait(), timeout=2)

        for text in ("second", "third"):
            composer.load_text(text)
            await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        assert [item.text for item in app._queued_messages] == ["second", "third"]
        assert app.query_one("#queued_messages").display is True
        assert not [entry for entry in app.conversation_entries if entry.content in {"second", "third"}]
        assert not [entry for entry in app.conversation_entries if entry.kind == "queued"]

        composer.load_text(draft)
        app.action_cancel_generation()
        for _ in range(10):
            await pilot.pause()
            if app.agent_worker is None:
                break

        assert app.agent_worker is None
        assert app._queued_messages == []
        assert app.query_one("#queued_messages").display is False
        assert composer.text == expected_text
        assert not [entry for entry in app.conversation_entries if entry.content in {"second", "third"}]
