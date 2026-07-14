# ruff: noqa: F401,F811,E402
from pathlib import Path
import asyncio
import json
import time

import pytest

from kolega_code.cli import messages
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
async def test_textual_app_composer_shift_enter_inserts_line_break_and_enter_submits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

    install_fake_agents(monkeypatch)

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

    install_fake_agents(monkeypatch)

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

    install_fake_agents(monkeypatch)

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
async def test_textual_app_composer_long_cr_separated_paste_does_not_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual import events

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

    install_fake_agents(monkeypatch)

    # macOS Terminal delivers pasted line breaks as carriage returns, not LF.
    # A few thousand chars reproduces the OSError [Errno 63] from GitHub #218.
    pasted = "\r".join("line of text " * 50 for _ in range(30))
    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        composer.focus()

        # The custom handler must not raise and must not queue an image attachment.
        await composer.on_paste(events.Paste(pasted))
        assert app._pending_image_attachments == []

        # Text is inserted by the default TextArea handler. TextArea normalizes
        # CR line breaks to LF on insertion, so compare against the normalized form.
        await composer._on_paste(events.Paste(pasted))
        assert composer.text == pasted.replace("\r", "\n")


@pytest.mark.asyncio
async def test_textual_app_composer_long_single_line_paste_does_not_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual import events

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

    install_fake_agents(monkeypatch)

    # A single very long line with no line breaks and no image extension.
    pasted = "x" * 5000
    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        composer.focus()

        await composer.on_paste(events.Paste(pasted))
        assert app._pending_image_attachments == []


@pytest.mark.asyncio
async def test_textual_app_composer_paste_image_file_path_still_attaches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual import events

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

    install_fake_agents(monkeypatch)

    png = tmp_path / "shot.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nfake-image-bytes")

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        composer.focus()

        await composer.on_paste(events.Paste(str(png)))
        assert len(app._pending_image_attachments) == 1
        assert app._pending_image_attachments[0]["path"] == str(png)


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

    class _GatedCoderAgent(FakeCoderAgent):
        async def process_message_stream(self, message, attachments=None):
            self.messages.append(message)
            first_started.set()
            await release_first.wait()
            yield {"type": "response", "content": "first done", "complete": True, "uuid": "response-1"}

    install_fake_agents(monkeypatch, coder_cls=_GatedCoderAgent)

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

    class _GatedCoderAgent(FakeCoderAgent):
        async def process_message_stream(self, message, attachments=None):
            self.messages.append(message)
            first_started.set()
            await release_first.wait()
            yield {"type": "response", "content": "first done", "complete": True, "uuid": "response-1"}

    install_fake_agents(monkeypatch, coder_cls=_GatedCoderAgent)

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

    class _BlockingCoderAgent(FakeCoderAgent):
        async def process_message_stream(self, message, attachments=None):
            first_started.set()
            await asyncio.Event().wait()
            yield {"type": "response", "content": "unreachable", "complete": True, "uuid": "response-1"}

    install_fake_agents(monkeypatch, coder_cls=_BlockingCoderAgent)

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


@pytest.mark.asyncio
async def test_textual_app_queued_message_delivered_mid_turn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

    turn_started = asyncio.Event()
    reach_boundary = asyncio.Event()
    delivered_ready = asyncio.Event()
    release_finish = asyncio.Event()

    class _ToolBoundaryCoderAgent(FakeCoderAgent):
        """Simulates a turn that hits a tool boundary and pulls queued input."""

        async def process_message_stream(self, message, attachments=None):
            self.messages.append(message)
            turn_started.set()
            await reach_boundary.wait()
            provider = self.queued_input_provider
            assert provider is not None
            inputs = await provider()
            self.delivered = [(item.text, item.attachments) for item in inputs]
            delivered_ready.set()
            await release_finish.wait()
            yield {"type": "response", "content": "done", "complete": True, "uuid": "response-1"}

    install_fake_agents(monkeypatch, coder_cls=_ToolBoundaryCoderAgent)

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
        worker = app.agent_worker
        assert worker is not None
        await asyncio.wait_for(turn_started.wait(), timeout=2)

        composer.load_text("second")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        await pilot.pause()
        assert [item.text for item in app._queued_messages] == ["second"]
        assert app.query_one("#queued_messages").display is True

        reach_boundary.set()
        await asyncio.wait_for(delivered_ready.wait(), timeout=2)
        await pilot.pause()

        # Mid-turn: the message was handed to the running turn, not a new one.
        assert app.agent_worker is not None
        assert app.agent is not None
        assert getattr(app.agent, "delivered") == [("second", None)]
        assert app._queued_messages == []
        assert app.query_one("#queued_messages").display is False
        second_entries = [entry for entry in app.conversation_entries if entry.content == "second"]
        assert [entry.kind for entry in second_entries] == ["user"]

        release_finish.set()
        await worker.wait()
        for _ in range(10):
            await pilot.pause()
            if app.agent_worker is None:
                break

        # No second turn started for the delivered message.
        assert app.agent_worker is None
        assert getattr(app.agent, "messages") == ["first"]
        user_contents = [entry.content for entry in app.conversation_entries if entry.kind == "user"]
        assert user_contents.count("second") == 1
        assert not [entry for entry in app.conversation_entries if entry.kind == "queued"]


@pytest.mark.asyncio
async def test_textual_app_cancel_restores_only_undelivered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer

    turn_started = asyncio.Event()
    reach_boundary = asyncio.Event()
    delivered_ready = asyncio.Event()

    class _ToolBoundaryCoderAgent(FakeCoderAgent):
        """Delivers queued input at one boundary, then blocks until cancelled."""

        async def process_message_stream(self, message, attachments=None):
            self.messages.append(message)
            turn_started.set()
            await reach_boundary.wait()
            provider = self.queued_input_provider
            assert provider is not None
            inputs = await provider()
            self.delivered = [item.text for item in inputs]
            delivered_ready.set()
            await asyncio.Event().wait()
            yield {"type": "response", "content": "unreachable", "complete": True, "uuid": "response-1"}

    install_fake_agents(monkeypatch, coder_cls=_ToolBoundaryCoderAgent)

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
        assert app.agent_worker is not None
        await asyncio.wait_for(turn_started.wait(), timeout=2)

        composer.load_text("second")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        reach_boundary.set()
        await asyncio.wait_for(delivered_ready.wait(), timeout=2)
        await pilot.pause()
        assert app.agent is not None
        assert getattr(app.agent, "delivered") == ["second"]

        composer.load_text("third")
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))
        assert [item.text for item in app._queued_messages] == ["third"]

        composer.load_text("")
        app.action_cancel_generation()
        for _ in range(10):
            await pilot.pause()
            if app.agent_worker is None:
                break

        # Only the undelivered follow-up returns to the composer; the delivered
        # one stays in the transcript as a user message.
        assert app.agent_worker is None
        assert app._queued_messages == []
        assert composer.text == "third"
        second_entries = [entry for entry in app.conversation_entries if entry.content == "second"]
        assert [entry.kind for entry in second_entries] == ["user"]
        assert not [entry for entry in app.conversation_entries if entry.content == "third"]
