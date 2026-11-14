from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from kolega_code.cli.peer_messaging import HeadlessPeerInbox
from kolega_code.cli.session_event_store import FileSessionEventStore
from kolega_code.cli.tui.widgets import ChatComposer
from kolega_code.events import AgentEvent
from kolega_code.session.inbox import PeerMessage, peer_model_text
from kolega_code.session.projection import replay
from tests.cli.test_app_handoff import _build_handoff_test_app

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("isolated_cli_env", "hermetic_git_config")]


async def _events(app: Any, session_id: str | None = None) -> list[AgentEvent]:
    session_id = session_id or str(app.session.session_id)
    return [
        event
        for event in await FileSessionEventStore(app.store.journal(session_id)).read(session_id)
        if event.event_type == "queue_updated"
    ]


async def _flush(app: Any) -> None:
    assert app._queue_presentation is not None
    await app._queue_presentation.flush()


async def test_tui_snapshot_fifo_images_peer_text_and_no_semantic_or_accounting_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kolega_code.cli.atif_export import export_atif_to_text
    from kolega_code.cli.session_usage import derive_session_usage

    app = _build_handoff_test_app(tmp_path, monkeypatch)
    async with app.run_test():
        await app._startup_complete.wait()
        assert app.agent is not None
        initial = await _events(app)
        assert initial[-1].content["items"] == [] and initial[-1].content["reason"] == "snapshot"
        journal = app.store.journal(app.session.session_id)
        semantic = [event.to_dict() for event in journal.read_events() if not event.event_type.startswith("ui.")]
        history = deepcopy(app.agent.history)
        usage = app._usage_ledger.snapshot()
        resumed_history, _ = app.store._replay(journal.read_events(), journal)
        recorded_usage = derive_session_usage(journal.read_events())
        atif = export_atif_to_text(journal, kolega_version="test", secret_values=(), state_dirs=())
        app._turn_active = True
        attachments = [
            {"type": "image", "media_type": "image/png", "data": "PRIVATE_IMAGE_BYTES", "path": "/private/image"},
            {"type": "text", "content": "PRIVATE_FILE_CONTENT"},
        ]
        app._queue_user_message(" \tfirst\nrequest  ", attachments)
        peer = PeerMessage.create(sender_session_id="peer", sender_title="<literal>", text="peer body")
        app._queue_user_message(
            peer.text,
            origin={"kind": "peer", "session_id": "peer", "title": "<literal>"},
            model_text=peer_model_text(peer),
        )
        app._queue_user_message("", [{"type": "image", "data": "PRIVATE_IMAGE_BYTES"}])
        assert not any(entry.kind in ("user", "peer_message", "queued") for entry in app.conversation_entries)
        await _flush(app)
        events = (await _events(app))[len(initial) :]
        assert [len(event.content["items"]) for event in events] == [1, 2, 3]
        assert events[-1].content["items"] == [
            {"queue_id": "queued-1", "text": "first request", "image_count": 1, "origin": None},
            {
                "queue_id": "queued-2",
                "text": "peer body",
                "image_count": 0,
                "origin": {"kind": "peer", "session_id": "peer", "title": "<literal>"},
            },
            {"queue_id": "queued-3", "text": "(empty)", "image_count": 1, "origin": None},
        ]
        assert "PRIVATE_" not in str([event.content for event in events])
        assert "[Peer message" not in str([event.content for event in events])
        assert [
            event.to_dict() for event in journal.read_events() if not event.event_type.startswith("ui.")
        ] == semantic
        assert app.agent.history == history and app._usage_ledger.snapshot() == usage
        monkeypatch.setattr(app, "_drain_pending_events", AsyncMock())
        inputs = await app._provide_queued_user_inputs()
        assert [item.text for item in inputs] == [" \tfirst\nrequest  ", peer_model_text(peer), ""]
        assert inputs[0].attachments == attachments
        assert [entry.kind for entry in app.conversation_entries if entry.kind in ("user", "peer_message")] == [
            "user",
            "peer_message",
            "user",
        ]
        assert await app._provide_queued_user_inputs() == []
        await _flush(app)
        final = (await _events(app))[-1]
        assert final.content["reason"] == "dequeued" and final.content["items"] == []
        assert app.agent.history == history and app._usage_ledger.snapshot() == usage
        assert app.store._replay(journal.read_events(), journal)[0] == resumed_history
        assert derive_session_usage(journal.read_events()) == recorded_usage
        assert export_atif_to_text(journal, kolega_version="test", secret_values=(), state_dirs=()) == atif
        app._turn_active = False
        await app.action_quit()


async def test_restore_clear_and_reset_observe_mutations_without_restoring_peers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_handoff_test_app(tmp_path, monkeypatch)
    async with app.run_test():
        await app._startup_complete.wait()
        app._turn_active = True
        composer = app.query_one("#composer", ChatComposer)
        composer.load_text("draft")
        app._queue_user_message("own")
        app._queue_user_message(
            "", origin={"kind": "peer", "session_id": "p", "title": "peer"}, model_text="PRIVATE_FRAME"
        )
        assert app._restore_queued_messages_to_composer() == 1
        assert composer.text == "own\n\ndraft"
        assert not app._queued_messages
        await _flush(app)
        assert (await _events(app))[-1].content["reason"] == "removed"
        app._queue_user_message("removed by clear")
        assert app._clear_queued_messages() == 1
        await _flush(app)
        assert (await _events(app))[-1].content["reason"] == "cleared"
        app._queue_user_message("reset input")
        app._turn_active = False
        before = app._queue_presentation
        await app._reset_current_thread()
        await _flush(app)
        assert app._queue_presentation is before and not app._queued_messages
        assert (await _events(app))[-1].content["items"] == []
        assert not replay(await _events(app)).conversation
        await app.action_quit()


async def test_between_turn_dequeue_and_failed_observer_do_not_change_execution_or_render_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_handoff_test_app(tmp_path, monkeypatch)
    async with app.run_test():
        await app._startup_complete.wait()
        app._turn_active = True
        monkeypatch.setattr(app._queue_presentation, "_emit", AsyncMock(side_effect=OSError("observer failed")))
        app._queue_user_message("display", [{"type": "image", "data": "image"}], model_text="model-facing")
        app._queue_user_message("second")
        await _flush(app)
        app._turn_active = False
        process = AsyncMock()
        monkeypatch.setattr(app, "_process_message", process)

        def start(coroutine: Any, **_kwargs: Any) -> Mock:
            coroutine.close()
            return Mock()

        monkeypatch.setattr(app, "run_worker", start)
        assert app._maybe_start_queued_message()
        assert [item.display_text for item in app._queued_messages] == ["second"]
        process.assert_called_once_with("model-facing", [{"type": "image", "data": "image"}])
        assert any(entry.kind == "user" and entry.content == "display" for entry in app.conversation_entries)
        entries = list(app.conversation_entries)
        write_log = Mock()
        monkeypatch.setattr(app, "_write_log", write_log)
        app._render_event(AgentEvent(event_type="queue_updated", sender="cli", content={"items": []}))
        assert app.conversation_entries == entries
        write_log.assert_not_called()
        app.agent_worker = None
        app._clear_queued_messages()


async def test_handoff_drains_old_queue_observations_before_session_rebinding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _build_handoff_test_app(tmp_path, monkeypatch)
    async with app.run_test():
        await app._startup_complete.wait()
        old_session = app.session.session_id
        app._turn_active = True
        app._queue_user_message("old pending request")
        await _flush(app)
        old_owner = app._queue_presentation
        old_generation = (await _events(app))[-1].content["queue_id"]
        app._turn_active = False
        await app._handoff_session("New context document")
        await _flush(app)
        assert app.session.session_id != old_session
        assert app._queue_presentation is not old_owner
        old = await _events(app, old_session)
        new = await _events(app)
        assert old[-1].content["reason"] == "cleared" and old[-1].content["items"] == []
        assert all(event.session_id == old_session for event in old)
        assert all(event.session_id == app.session.session_id and event.content["items"] == [] for event in new)
        assert all(event.content["queue_id"] != old_generation for event in new)
        assert new[-1].content["reason"] == "snapshot"
        assert not app._queued_messages
        await app.action_quit()


async def test_headless_queue_records_fifo_observations_without_printing_or_changing_inputs(
    tmp_path: Path, capsys: Any
) -> None:
    from kolega_code.cli.session_store import SessionStore

    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    session = store.create(project, "code", {})
    inbox = HeadlessPeerInbox(
        store_root=store.root,
        session_id=session.session_id,
        json_mode=False,
        journal_factory=lambda: FileSessionEventStore(store.journal(session.session_id)),
    )
    inbox._publish_queue("snapshot")
    one = PeerMessage.create(sender_session_id="one", sender_title="one", text="first")
    two = PeerMessage.create(sender_session_id="two", sender_title="two", text="second")
    assert await inbox._deliver(one) == "accepted"
    assert await inbox._deliver(two) == "accepted"
    assert inbox.pending_texts() == [peer_model_text(one), peer_model_text(two)]
    assert inbox.pop_all() == [peer_model_text(one), peer_model_text(two)]
    assert inbox.pop_all() == []
    assert await inbox._provide_queued_inputs() == []
    assert await inbox._deliver(one) == "accepted"  # repeat suppression remains silent toward the sender
    assert await inbox._provide_queued_inputs() == []
    third = PeerMessage.create(sender_session_id="one", sender_title="one", text="third")
    assert await inbox._deliver(third) == "accepted"
    inputs = await inbox._provide_queued_inputs()
    assert len(inputs) == 1 and inputs[0].text == peer_model_text(third)
    assert inputs[0].origin == {"kind": "peer", "session_id": "one", "title": "one"}
    await inbox.stop()
    events = [
        event
        for event in await FileSessionEventStore(store.journal(session.session_id)).read(session.session_id)
        if event.event_type == "queue_updated"
    ]
    assert [event.content["reason"] for event in events] == [
        "snapshot",
        "enqueued",
        "enqueued",
        "dequeued",
        "enqueued",
        "dequeued",
    ]
    assert [len(event.content["items"]) for event in events] == [0, 1, 2, 0, 1, 0]
    assert [item["text"] for item in events[2].content["items"]] == ["first", "second"]
    assert all(event.content["source"] == "ask" for event in events)
    assert capsys.readouterr() == ("", "")
