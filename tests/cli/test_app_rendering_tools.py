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
async def test_textual_app_renders_tool_events_in_chat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.theme import TOOL_RESULT_PREVIEW_CHARS

    install_fake_agents(monkeypatch)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        app._render_event(
            AgentEvent(
                event_type="chat_message",
                sender="coder",
                content={
                    "message_type": "tool_call",
                    "text": "Calling read_file",
                    "tool_description": "read_file",
                    "tool_call_id": "tool-1",
                },
            )
        )
        app._render_event(
            AgentEvent(
                event_type="chat_message",
                sender="coder",
                content={
                    "message_type": "tool_result",
                    "text": "short result",
                    "tool_description": "read_file",
                    "tool_call_id": "tool-1",
                },
            )
        )
        app._render_event(
            AgentEvent(
                event_type="chat_message",
                sender="coder",
                content={
                    "message_type": "tool_error",
                    "text": "x" * (TOOL_RESULT_PREVIEW_CHARS + 10),
                    "tool_description": "read_file",
                    "tool_call_id": "tool-2",
                },
            )
        )

        tool_entries = [entry for entry in app.conversation_entries if entry.kind.startswith("tool")]
        assert [entry.kind for entry in tool_entries] == ["tool_result", "tool_error"]
        assert tool_entries[0].content == "short result"
        assert tool_entries[0].tool_call_id == "tool-1"
        assert tool_entries[1].content.endswith("…")
        assert tool_entries[1].tool_call_id == "tool-2"
        assert len(tool_entries[1].content) == TOOL_RESULT_PREVIEW_CHARS + 1


@pytest.mark.asyncio
async def test_parallel_same_name_tool_calls_keep_separate_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)
    tool_call_ids = [f"web-fetch-{index}" for index in range(5)]

    async with app.run_test():
        for tool_call_id in tool_call_ids:
            app._render_event(
                AgentEvent(
                    event_type="chat_message",
                    sender="coder",
                    content={
                        "message_type": "tool_call",
                        "text": "Calling web_fetch",
                        "tool_description": "web_fetch",
                        "tool_call_id": tool_call_id,
                    },
                )
            )

        tool_entries = [entry for entry in app.conversation_entries if entry.kind.startswith("tool")]
        assert len(tool_entries) == 5
        assert [entry.tool_call_id for entry in tool_entries] == tool_call_ids
        assert len({id(entry) for entry in tool_entries}) == 5

        for index in reversed(range(5)):
            app._render_event(
                AgentEvent(
                    event_type="chat_message",
                    sender="coder",
                    content={
                        "message_type": "tool_result",
                        "text": f"result {index}",
                        "tool_description": "web_fetch",
                        "tool_call_id": tool_call_ids[index],
                    },
                )
            )

        assert [entry.kind for entry in tool_entries] == ["tool_result"] * 5
        assert [entry.content for entry in tool_entries] == [f"result {index}" for index in range(5)]
        assert all(
            app._tool_entries[tool_call_id] is tool_entries[index] for index, tool_call_id in enumerate(tool_call_ids)
        )


@pytest.mark.asyncio
async def test_textual_app_appends_append_mode_tool_streaming_events_in_chat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        app._render_event(
            AgentEvent(
                event_type="chat_message",
                sender="coder",
                content={
                    "message_type": "tool_call",
                    "text": "Calling think_hard",
                    "tool_description": "think_hard",
                    "tool_call_id": "tool-1",
                },
            )
        )
        app._render_event(
            AgentEvent(
                event_type="tool_streaming_update",
                sender="coder",
                content={
                    "text": "partial analysis",
                    "tool_name": "think_hard",
                    "tool_call_id": "tool-1",
                    "is_complete": False,
                    "stream_mode": "append",
                },
            )
        )
        app._render_event(
            AgentEvent(
                event_type="tool_streaming_update",
                sender="coder",
                content={
                    "text": "\ncontinued analysis",
                    "tool_name": "think_hard",
                    "tool_call_id": "tool-1",
                    "is_complete": False,
                    "stream_mode": "append",
                },
            )
        )

        tool_entries = [entry for entry in app.conversation_entries if entry.kind.startswith("tool")]
        assert len(tool_entries) == 1
        assert tool_entries[0].kind == "tool_call"
        assert tool_entries[0].content == "partial analysis\ncontinued analysis"
        assert tool_entries[0].complete is False

        app._render_event(
            AgentEvent(
                event_type="tool_streaming_update",
                sender="coder",
                content={
                    "text": "final analysis",
                    "tool_name": "think_hard",
                    "tool_call_id": "tool-1",
                    "is_complete": True,
                },
            )
        )

        tool_entries = [entry for entry in app.conversation_entries if entry.kind.startswith("tool")]
        assert len(tool_entries) == 1
        assert tool_entries[0].kind == "tool_result"
        assert tool_entries[0].content == "final analysis"
        assert tool_entries[0].complete is True
        assert app._tool_stream_buffers == {}


@pytest.mark.asyncio
async def test_textual_app_replaces_default_tool_streaming_events_in_chat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        app._render_event(
            AgentEvent(
                event_type="tool_streaming_update",
                sender="coder",
                content={
                    "text": "Fetching content...",
                    "tool_name": "web_fetch",
                    "tool_call_id": "tool-1",
                    "is_complete": False,
                },
            )
        )
        app._render_event(
            AgentEvent(
                event_type="tool_streaming_update",
                sender="coder",
                content={
                    "text": "Processing content...",
                    "tool_name": "web_fetch",
                    "tool_call_id": "tool-1",
                    "is_complete": False,
                },
            )
        )

        tool_entries = [entry for entry in app.conversation_entries if entry.kind.startswith("tool")]
        assert len(tool_entries) == 1
        assert tool_entries[0].kind == "tool_call"
        assert tool_entries[0].content == "Processing content..."


@pytest.mark.asyncio
async def test_textual_app_caps_long_append_mode_tool_streaming_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.theme import TOOL_STREAM_PREVIEW_CHARS

    install_fake_agents(monkeypatch)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        app._render_event(
            AgentEvent(
                event_type="tool_streaming_update",
                sender="coder",
                content={
                    "text": "a" * (TOOL_STREAM_PREVIEW_CHARS + 10),
                    "tool_name": "think_hard",
                    "tool_call_id": "tool-1",
                    "is_complete": False,
                    "stream_mode": "append",
                },
            )
        )

        tool_entries = [entry for entry in app.conversation_entries if entry.kind.startswith("tool")]
        assert len(tool_entries) == 1
        assert tool_entries[0].content.startswith(
            f"[stream truncated to the last {TOOL_STREAM_PREVIEW_CHARS} characters]"
        )
        assert tool_entries[0].content.endswith("a" * TOOL_STREAM_PREVIEW_CHARS)


@pytest.mark.asyncio
async def test_textual_app_renders_queued_tool_events_during_active_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Static

    from kolega_code.cli.app import KolegaCodeApp
    from kolega_code.cli.tui.widgets import ChatComposer
    from kolega_code.cli.messages import COMPOSER_PLACEHOLDER, QUEUE_PLACEHOLDER

    started = asyncio.Event()
    release = asyncio.Event()

    class _QueuingCoderAgent(FakeCoderAgent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.connection_manager = kwargs["connection_manager"]
            self.workspace_id = kwargs["workspace_id"]
            self.thread_id = kwargs["thread_id"]

        async def process_message_stream(self, message):
            await self.connection_manager.broadcast_event(
                AgentEvent(
                    event_type="chat_message",
                    sender="coder",
                    content={
                        "message_type": "tool_call",
                        "text": "Calling read_file",
                        "tool_description": "read_file",
                        "tool_call_id": "tool-1",
                    },
                ),
                self.workspace_id,
                self.thread_id,
            )
            started.set()
            await release.wait()
            await self.connection_manager.broadcast_event(
                AgentEvent(
                    event_type="chat_message",
                    sender="coder",
                    content={
                        "message_type": "tool_result",
                        "text": "README contents",
                        "tool_description": "read_file",
                        "tool_call_id": "tool-1",
                    },
                ),
                self.workspace_id,
                self.thread_id,
            )
            yield {"type": "response", "content": "done", "complete": True, "uuid": "response-1"}

    async def wait_for_tool_entries(app: KolegaCodeApp, count: int) -> list:
        while True:
            entries = [entry for entry in app.conversation_entries if entry.kind.startswith("tool")]
            if len(entries) >= count:
                return entries
            await asyncio.sleep(0.01)

    install_fake_agents(monkeypatch, coder_cls=_QueuingCoderAgent)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)
    now = 10.0
    monkeypatch.setattr(app, "_now", lambda: now)

    async with app.run_test():
        composer = app.query_one("#composer", ChatComposer)
        turn_status = app.query_one("#turn_status", Static)
        await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, "hi"))
        worker = app.agent_worker
        assert worker is not None
        assert worker.group == "turns"

        await started.wait()
        event_worker = next(worker for worker in app.workers if worker.name == "kolega-events")
        assert event_worker.group == "events"
        assert not event_worker.is_cancelled

        tool_entries = await asyncio.wait_for(wait_for_tool_entries(app, 1), timeout=1)
        assert tool_entries[0].kind == "tool_call"
        assert tool_entries[0].content == "Calling read_file"
        assert composer.placeholder == QUEUE_PLACEHOLDER
        assert "Running read_file…" in str(turn_status.render())

        now = 25.0
        release.set()
        await worker.wait()

        tool_entries = [entry for entry in app.conversation_entries if entry.kind.startswith("tool")]
        assert len(tool_entries) == 1
        assert tool_entries[0].kind == "tool_result"
        assert tool_entries[0].content == "README contents"
        assert composer.placeholder == COMPOSER_PLACEHOLDER
        assert "Done in 15s" in str(turn_status.render())


@pytest.mark.asyncio
async def test_textual_app_late_tool_result_updates_existing_tool_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    app = KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)

    async with app.run_test():
        app._render_event(
            AgentEvent(
                event_type="chat_message",
                sender="coder",
                content={
                    "message_type": "tool_call",
                    "text": "Calling read_file",
                    "tool_description": "read_file",
                    "tool_call_id": "tool-1",
                },
            )
        )
        app._active_progress_entry = None
        app._turn_active = False

        app._render_event(
            AgentEvent(
                event_type="chat_message",
                sender="coder",
                content={
                    "message_type": "tool_result",
                    "text": "late result",
                    "tool_description": "read_file",
                    "tool_call_id": "tool-1",
                },
            )
        )

        tool_entries = [entry for entry in app.conversation_entries if entry.kind.startswith("tool")]
        assert len(tool_entries) == 1
        assert tool_entries[0].kind == "tool_result"
        assert tool_entries[0].content == "late result"


@pytest.mark.asyncio
async def test_tool_entries_render_as_collapsibles_with_full_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Collapsible

    from kolega_code.cli.tui.widgets import ToolEntryWidget
    from kolega_code.cli.theme import TOOL_RESULT_PREVIEW_CHARS

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        app._turn_active = True
        long_output = "x" * (TOOL_RESULT_PREVIEW_CHARS + 200)
        app._add_tool_message(
            "tool_call", {"tool_name": "read_file", "tool_call_id": "tc-1", "text": "Calling read_file"}
        )
        app._flush_conversation_render()
        await pilot.pause()

        widget = app.query(ToolEntryWidget).last()
        collapsible = widget.query_one(Collapsible)
        assert collapsible.collapsed is True
        assert "running" in str(collapsible.title)

        app._add_tool_message("tool_result", {"tool_name": "read_file", "tool_call_id": "tc-1", "text": long_output})
        app._flush_conversation_render()
        await pilot.pause()

        # The same widget is updated in place: title flips to done, body holds full output
        same_widget = app.query(ToolEntryWidget).last()
        assert same_widget is widget
        assert "done" in str(widget.query_one(Collapsible).title)
        entry = widget.entry
        assert len(entry.content) == TOOL_RESULT_PREVIEW_CHARS + 1  # preview stays truncated
        assert entry.full_content == long_output  # expand-on-demand shows everything


@pytest.mark.asyncio
async def test_tool_edit_preview_stays_between_title_and_expanded_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from textual.widgets import Collapsible, Static
    from textual.widgets._collapsible import CollapsibleTitle

    from kolega_code.cli.tui.state import ConversationEntry
    from kolega_code.cli.tui.widgets import ToolEntryWidget

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)
    entry = ConversationEntry(
        kind="tool_result",
        content="Edited src/a.py",
        full_content="raw tool output",
        tool_name="edit",
        edit_preview={
            "kind": "diff",
            "path": "src/a.py",
            "language": "python",
            "lines": [["meta", "@@ -1 +1 @@"], ["del", "-old"], ["add", "+new"]],
            "more": 0,
            "adds": 1,
            "dels": 1,
        },
    )

    async with app.run_test(size=(100, 45)) as pilot:
        app.conversation_entries = [entry]
        app._render_conversation()
        await pilot.pause()

        widget = app.query(ToolEntryWidget).last()
        collapsible = widget.query_one(Collapsible)
        title = widget.query_one(CollapsibleTitle)
        preview = widget.query_one(".tool-preview", Static)
        body = widget.query_one(".tool-body", Static)

        assert collapsible.collapsed is True
        assert preview.display is True
        assert title.region.y < preview.region.y
        assert preview.content_region.x == widget.region.x + 4

        collapsible.collapsed = False
        await pilot.pause()

        assert title.region.y < preview.region.y < body.region.y
        assert body.region.x == widget.region.x + 4


@pytest.mark.asyncio
async def test_tool_entry_title_shows_lsp_badge_for_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tool result carrying LSP diagnostics surfaces a severity badge in its title."""
    pytest.importorskip("textual")

    from textual.widgets import Collapsible

    from kolega_code.cli.tui.widgets import ToolEntryWidget

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        app._turn_active = True
        result_text = (
            "Edited foo.py\n\n"
            "LSP diagnostics (2 warnings):\n"
            "🟡 Line 5: 'foo' is not defined (pyright)\n"
            "🟡 Line 10: Unused import 'os' (pyright)"
        )
        app._add_tool_message("tool_call", {"tool_name": "edit", "tool_call_id": "tc-lsp", "text": "Calling edit"})
        app._add_tool_message("tool_result", {"tool_name": "edit", "tool_call_id": "tc-lsp", "text": result_text})
        app._flush_conversation_render()
        await pilot.pause()

        widget = app.query(ToolEntryWidget).last()
        title = str(widget.query_one(Collapsible).title)
        assert "done" in title
        # The badge is visible without expanding and reads "2 LSP warnings".
        assert "2 LSP warnings" in title
        # The full diagnostics block is available in the expandable body.
        assert "LSP diagnostics (2 warnings):" in widget.entry.full_content


@pytest.mark.asyncio
async def test_tool_entry_title_has_no_lsp_badge_without_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tool result without diagnostics shows no LSP badge."""
    pytest.importorskip("textual")

    from textual.widgets import Collapsible

    from kolega_code.cli.tui.widgets import ToolEntryWidget

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        app._turn_active = True
        app._add_tool_message("tool_call", {"tool_name": "edit", "tool_call_id": "tc-clean", "text": "Calling edit"})
        app._add_tool_message("tool_result", {"tool_name": "edit", "tool_call_id": "tc-clean", "text": "Edited foo.py"})
        app._flush_conversation_render()
        await pilot.pause()

        widget = app.query(ToolEntryWidget).last()
        title = str(widget.query_one(Collapsible).title)
        assert "done" in title
        assert "LSP" not in title
