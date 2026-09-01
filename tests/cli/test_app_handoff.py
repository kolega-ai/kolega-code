# ruff: noqa: F401,F811,E402
"""Tests for the ``/handoff`` TUI slash command and the session switch it drives."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock

import pytest

from kolega_code.cli import messages
from kolega_code.cli.config import config_summary
from kolega_code.cli.session_store import SessionStore
from kolega_code.llm.models import Message, TextBlock

from ._app_test_utils import FakeCoderAgent, build_test_config, install_fake_agents
from tests.agent.compaction_helpers import FakeLLM

HANDOFF_DOC = "## Goal\nShip the handoff feature"


def _user_text(text: str) -> Message:
    return Message(role="user", content=[TextBlock(text=text)])


def _assistant_text(text: str) -> Message:
    return Message(role="assistant", content=[TextBlock(text=text)], stop_reason="end_turn")


class HandoffFakeAgent(FakeCoderAgent):
    """A coder-agent stand-in carrying the surface ``/handoff`` reads.

    ``primary_model_config``/``model_default_temperature``/``llm`` mirror the
    real agent's helper-request inputs; ``llm`` is a scripted FakeLLM whose
    stream returns the handoff document.
    """

    instances: list["HandoffFakeAgent"] = []

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.primary_model_config = SimpleNamespace(model="claude-opus-5", thinking_effort=None)
        self.model_default_temperature = 1.0
        self.llm = FakeLLM(summary_text=HANDOFF_DOC)
        HandoffFakeAgent.instances.append(self)


class BlockingStream:
    """Stream that yields deltas until the app's handoff cancel event fires."""

    def __init__(self, owner: "BlockingHandoffFakeLLM") -> None:
        self._owner = owner
        self._yielded = False

    async def __aenter__(self) -> "BlockingStream":
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    def __aiter__(self) -> "BlockingStream":
        return self

    async def __anext__(self) -> dict[str, str]:
        if not self._yielded:
            self._yielded = True
            return {"type": "delta", "text": "x"}
        await asyncio.sleep(0.01)
        event = self._owner.event_provider()
        if event is not None and event.is_set():
            raise StopAsyncIteration
        return {"type": "delta", "text": "x"}

    async def get_final_message(self) -> Message:
        raise AssertionError("must not be called after cancellation")


class BlockingHandoffFakeLLM:
    """LLM whose stream runs until the test signals cancellation."""

    def __init__(self) -> None:
        self.provider = MagicMock(base_url="https://api.test.example/v1")
        self.event_provider: Callable[[], asyncio.Event | None] = lambda: None
        self.stream = AsyncMock(side_effect=self._stream)
        self.stream_entered = False

    async def _stream(self, *args: Any, **kwargs: Any) -> BlockingStream:
        self.stream_entered = True
        return BlockingStream(self)


class BlockingHandoffFakeAgent(HandoffFakeAgent):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.llm = BlockingHandoffFakeLLM()


def _build_handoff_test_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    agent_cls: type[FakeCoderAgent] = HandoffFakeAgent,
):
    pytest.importorskip("textual")
    from kolega_code.cli.app import KolegaCodeApp

    HandoffFakeAgent.instances = []
    install_fake_agents(monkeypatch, coder_cls=agent_cls)

    project = tmp_path / "project"
    project.mkdir()
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", config_summary(config))
    return KolegaCodeApp(project_path=project, config=config, mode="code", store=store, session=session)


async def _submit(app, pilot, text: str) -> None:
    """Type a slash command into the composer and submit it."""
    from kolega_code.cli.tui.widgets import ChatComposer

    composer = app.query_one("#composer", ChatComposer)
    composer.load_text(text)
    await app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text))


async def _wait_for(app, pilot, predicate, *, timeout: float = 6.0) -> None:
    """Poll until ``predicate()`` is truthy or time out."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await pilot.pause(0.02)
        if predicate():
            return
    raise AssertionError(f"condition not met within {timeout}s")


@pytest.mark.asyncio
async def test_handoff_refuses_while_a_turn_is_active(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_handoff_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        await _wait_for(app, pilot, lambda: app.agent is not None)
        old_id = app.session.session_id
        app._turn_active = True

        await _submit(app, pilot, "/handoff")

        assert app.session.session_id == old_id
        assert len(app.store.list()) == 1
        assert app._handoff_in_progress is False
        assert getattr(HandoffFakeAgent.instances[-1], "cleanup_calls") == 0


@pytest.mark.asyncio
async def test_handoff_with_empty_history_warns_and_stays(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _build_handoff_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        await _wait_for(app, pilot, lambda: app.agent is not None)
        old_id = app.session.session_id

        await _submit(app, pilot, "/handoff")

        assert app.session.session_id == old_id
        assert len(app.store.list()) == 1
        assert app._handoff_in_progress is False


@pytest.mark.asyncio
async def test_handoff_switches_to_a_fresh_parent_linked_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_handoff_test_app(tmp_path, monkeypatch)

    async with app.run_test() as pilot:
        await _wait_for(app, pilot, lambda: app.agent is not None)
        old_id = app.session.session_id
        old_agent = app.agent
        assert old_agent is not None
        old_agent.history = [_user_text("ship the handoff feature"), _assistant_text("on it")]

        await _submit(app, pilot, "/handoff focus on the tests")

        # A new session is active and records its parent.
        new_id = app.session.session_id
        assert new_id != old_id
        assert app.session.parent_session_id == old_id
        records = {record.session_id: record for record in app.store.list()}
        assert old_id in records and new_id in records
        assert records[new_id].parent_session_id == old_id
        assert records[old_id].parent_session_id is None
        assert app._handoff_in_progress is False

        # The new agent's entire conversation is the handoff document.
        new_agent = app.agent
        assert new_agent is not None
        assert new_agent is not old_agent
        texts = [message.get_text_content() for message in new_agent.history]
        assert len(texts) == 1
        assert texts[0].startswith("<handoff-context>")
        assert HANDOFF_DOC in texts[0]
        assert getattr(old_agent, "cleanup_calls") == 1

        # The transcript shows the document and the success line.
        system_entries = [entry.content for entry in app.conversation_entries if entry.kind == "system"]
        assert any(messages.HANDOFF_SUCCESS.format(title=app.session.title) in content for content in system_entries)
        user_entries = [entry.content for entry in app.conversation_entries if entry.kind == "user"]
        assert any(HANDOFF_DOC in content for content in user_entries)

        # The old session's journal carries the lineage notice and epoch boundary
        # and still loads; the new session replays with the handoff document.
        events = list(app.store.journal(old_id).read_events())
        assert any(event.event_type == "context.message" for event in events)
        assert any(
            event.event_type == "context.epoch_started" and (event.payload or {}).get("reason") == "handoff"
            for event in events
        )
        assert app.store.load(old_id) is not None
        loaded_new = app.store.load(new_id)
        assert loaded_new.parent_session_id == old_id
        history_text = " ".join(
            block.get("text", "")
            for message in loaded_new.history
            for block in message.get("content", [])
            if isinstance(block, dict)
        )
        assert HANDOFF_DOC in history_text


@pytest.mark.asyncio
async def test_handoff_cancellation_leaves_the_session_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_handoff_test_app(tmp_path, monkeypatch, agent_cls=BlockingHandoffFakeAgent)

    async with app.run_test() as pilot:
        await _wait_for(app, pilot, lambda: app.agent is not None)
        old_id = app.session.session_id
        agent = app.agent
        assert agent is not None and isinstance(agent.llm, BlockingHandoffFakeLLM)
        agent.history = [_user_text("hello"), _assistant_text("hi there")]
        agent.llm.event_provider = lambda: app._handoff_cancel_event

        from kolega_code.cli.tui.widgets import ChatComposer

        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("/handoff")
        task = asyncio.create_task(app.on_chat_composer_submitted(ChatComposer.Submitted(composer, composer.text)))

        await _wait_for(app, pilot, lambda: app._handoff_in_progress)
        await _wait_for(app, pilot, lambda: getattr(agent.llm, "stream_entered", False))
        app.action_cancel_generation()
        await _wait_for(app, pilot, lambda: not app._handoff_in_progress, timeout=8.0)
        await task

        assert app.session.session_id == old_id
        assert len(app.store.list()) == 1
        assert app.agent is agent
