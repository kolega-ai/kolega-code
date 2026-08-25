"""Cross-session messaging, phase 1: provenance through the TUI queue.

Peer messages ride the existing queued-message machinery but must stay
distinguishable end to end: the human sees the raw text under a sender badge,
the model receives the provenance preamble, and nothing peer-authored ever
lands back in the composer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kolega_code.agent.baseagent import QueuedUserInput

from ._app_test_utils import (
    _build_sub_agent_test_app,
    renderable_text,
    wait_for_turn_idle,
)

PEER_ORIGIN = {"kind": "peer", "session_id": "peer-1", "title": "deploy-bot"}
PREAMBLE = "[Peer message from deploy-bot"
RAW_TEXT = "Please rerun the nightly job"


def _wrapped(text: str, suffix: str = "]") -> str:
    return f"{PREAMBLE}{suffix}\n\n{text}"


@pytest.mark.asyncio
async def test_peer_message_entry_renders_sender_badge_and_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kolega_code.cli.tui.state import ConversationEntry

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        entry = ConversationEntry(kind="peer_message", content=f"{RAW_TEXT}\nsecond line")
        entry.origin = dict(PEER_ORIGIN)
        lines = renderable_text(app._format_conversation_entry(entry)).splitlines()

        assert lines[0].startswith("◆ ← deploy-bot")
        assert RAW_TEXT not in lines[0], "the body starts on its own line beneath the header"
        assert lines[1] == f"  {RAW_TEXT}"
        assert lines[2] == "  second line"

        # Without origin metadata the badge still renders, generically.
        anonymous = renderable_text(
            app._format_conversation_entry(ConversationEntry(kind="peer_message", content=RAW_TEXT))
        )
        assert "← peer" in anonymous
        assert RAW_TEXT in anonymous


@pytest.mark.asyncio
async def test_queue_preview_attributes_peer_messages_to_their_sender(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        app._queue_user_message(RAW_TEXT, origin=dict(PEER_ORIGIN), model_text=_wrapped(RAW_TEXT))
        app._queue_user_message("plain follow-up")

        preview = app._queued_messages_preview()

        assert "(from deploy-bot)" in preview
        assert PREAMBLE not in preview, "the preview shows the raw text, never the preamble"
        assert RAW_TEXT in preview


@pytest.mark.asyncio
async def test_started_peer_turn_sends_preamble_but_renders_raw_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        model_text = _wrapped(RAW_TEXT)
        app._queue_user_message(RAW_TEXT, origin=dict(PEER_ORIGIN), model_text=model_text)

        assert app._maybe_start_queued_message() is True

        await wait_for_turn_idle(app, pilot)
        # Settle the 10ms queued-message-drain timer before teardown.
        await pilot.pause(0.05)

        agent = app.agent
        assert agent is not None
        sent: list[Any] = getattr(agent, "messages")
        assert sent == [model_text]

        peer_entries = [entry for entry in app.conversation_entries if entry.kind == "peer_message"]
        assert len(peer_entries) == 1
        assert peer_entries[0].content == RAW_TEXT
        assert peer_entries[0].origin == PEER_ORIGIN
        user_entries = [entry for entry in app.conversation_entries if entry.kind == "user"]
        assert user_entries == []


@pytest.mark.asyncio
async def test_mid_turn_drain_carries_origin_through_queued_user_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        app._queue_user_message(RAW_TEXT, origin=dict(PEER_ORIGIN), model_text=_wrapped(RAW_TEXT))
        app._queue_user_message("typed later")

        inputs: list[QueuedUserInput] = await app._provide_queued_user_inputs()

        assert [(item.origin, item.text) for item in inputs] == [
            (PEER_ORIGIN, _wrapped(RAW_TEXT)),
            (None, "typed later"),
        ]
        kinds_by_content = {
            entry.content: entry.kind
            for entry in app.conversation_entries
            if entry.content in (RAW_TEXT, "typed later")
        }
        assert kinds_by_content == {RAW_TEXT: "peer_message", "typed later": "user"}
        assert app._queued_messages == []


@pytest.mark.asyncio
async def test_restore_to_composer_drops_peer_and_keeps_typed_messages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from textual.widgets import TextArea

    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        app._queue_user_message(RAW_TEXT, origin=dict(PEER_ORIGIN), model_text=_wrapped(RAW_TEXT))
        app._queue_user_message("my typed follow-up")

        restored = app._restore_queued_messages_to_composer()

        assert restored == 1
        composer_text = app.query_one("#composer", TextArea).text
        assert "my typed follow-up" in composer_text
        assert PREAMBLE not in composer_text, "the wrapped peer message never reaches the composer"
        assert app._queued_messages == []
