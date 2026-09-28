"""Production integration of context hints; no changes to actual key dispatch."""

import asyncio
import time

import pytest

from kolega_code.cli.tui.shortcut_bar import ContextFooter, ShortcutHelpScreen
from kolega_code.cli.tui.state import ConversationEntry
from kolega_code.cli.tui.widgets import ChatComposer, CompletionDropdown

from ._app_test_utils import FakeCoderAgent, _build_mention_test_app, install_fake_agents


async def wait_for(pilot, predicate) -> None:
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline:
        await pilot.pause(0.02)
        if predicate():
            return
    raise AssertionError("footer context did not settle")


def descriptions(app) -> list[str]:
    return [hint.description for hint in app.query_one(ContextFooter).context_shortcuts()]


@pytest.mark.asyncio
async def test_footer_tracks_real_worker_queue_and_completion_escape(tmp_path, monkeypatch) -> None:
    gate = asyncio.Event()

    class WaitingAgent(FakeCoderAgent):
        async def process_message_stream(self, message, attachments=None):
            await gate.wait()
            async for chunk in super().process_message_stream(message, attachments):
                yield chunk

    app = _build_mention_test_app(tmp_path, monkeypatch)
    install_fake_agents(monkeypatch, coder_cls=WaitingAgent)
    async with app.run_test(size=(120, 40)) as pilot:
        composer = app.query_one(ChatComposer)
        composer.focus()
        await wait_for(pilot, lambda: "send" in descriptions(app))
        composer.load_text("first task")
        await pilot.press("enter")
        await wait_for(pilot, lambda: app.agent_worker is not None and "queue" in descriptions(app))
        await pilot.press("/")
        await wait_for(pilot, lambda: "select" in descriptions(app))
        assert "queue" not in descriptions(app) and "dismiss" in descriptions(app)
        worker = app.agent_worker
        assert worker is not None
        await pilot.press("escape")
        await wait_for(pilot, lambda: "queue" in descriptions(app))
        assert not app.query_one(CompletionDropdown).display
        assert app.agent_worker is worker and not worker.is_cancelled
        composer.load_text("follow-up")
        await pilot.press("enter")
        await wait_for(pilot, lambda: bool(app._queued_messages))
        assert "queue" in descriptions(app)
        # Clear the test follow-up before releasing the turn's queue-drain timer.
        app._clear_queued_messages()
        gate.set()
        await wait_for(pilot, lambda: app.agent_worker is None and "send" in descriptions(app))


@pytest.mark.asyncio
async def test_shortcut_help_preserves_draft_and_resyncs_covered_transcript(tmp_path, monkeypatch) -> None:
    app = _build_mention_test_app(tmp_path, monkeypatch)
    async with app.run_test(size=(120, 40)) as pilot:
        composer = app.query_one(ChatComposer)
        composer.load_text("keep this draft")
        composer.focus()
        await pilot.press("f1")
        await wait_for(pilot, lambda: isinstance(app.screen, ShortcutHelpScreen))
        entry = ConversationEntry(kind="assistant", content="Output received while help was open.")
        app._add_conversation_entry(entry)
        app._flush_conversation_render()
        assert app._transcript_sync_pending
        await pilot.press("escape")
        await wait_for(pilot, lambda: len(app.screen_stack) == 1 and entry.entry_id in app._entry_widgets)
        assert composer.text == "keep this draft" and app.screen.focused is composer
        assert not app._transcript_sync_pending
        assert "send" in descriptions(app)
        # Local help must not turn /help into a local command.
        assert not any(e.kind == "user" for e in app.conversation_entries)


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt_before_help", [False, True])
@pytest.mark.parametrize("close_key", ["escape", "f1", "enter"])
async def test_help_cannot_focus_or_answer_a_covered_permission_prompt(
    tmp_path, monkeypatch, prompt_before_help: bool, close_key: str
) -> None:
    from textual.widgets import Button

    from kolega_code.cli.tui.widgets import ActionList
    from kolega_code.permissions import permission_request_for_tool

    from .test_app_permission_prompts import _open_permission_request

    app = _build_mention_test_app(tmp_path, monkeypatch)
    async with app.run_test(size=(120, 40)) as pilot:
        if not prompt_before_help:
            await pilot.press("f1")
            await wait_for(pilot, lambda: isinstance(app.screen, ShortcutHelpScreen))
        request = permission_request_for_tool("exec_command", {"command": "echo never-executed"})
        assert request is not None
        _, decision = await _open_permission_request(app, request)
        try:
            await wait_for(pilot, lambda: app._pending_approval is not None)
            actions = app.screen_stack[0].query_one("#approval_actions", ActionList)
            if prompt_before_help:
                await wait_for(pilot, lambda: app.screen.focused is actions)
                await pilot.press("f1")
            await wait_for(pilot, lambda: isinstance(app.screen, ShortcutHelpScreen))
            help_screen = app.screen
            help_screen.query_one(Button).focus()
            await pilot.pause()
            assert help_screen.focused is not actions
            # Number keys must not select a covered option either.
            await pilot.press("1")
            assert not decision.done() and app._pending_approval is not None
            await pilot.press(close_key)
            await wait_for(pilot, lambda: len(app.screen_stack) == 1 and app.screen.focused is actions)
            assert not decision.done() and app._pending_approval is not None
        finally:
            app._cancel_pending_approval()
            decision.cancel()
            await asyncio.gather(decision, return_exceptions=True)
