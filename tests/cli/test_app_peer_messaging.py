"""Cross-session messaging, phase 1: provenance through the TUI queue.

Peer messages ride the existing queued-message machinery but must stay
distinguishable end to end: the human sees the raw text under a sender badge,
the model receives the provenance preamble, and nothing peer-authored ever
lands back in the composer.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from kolega_code.agent.baseagent import QueuedUserInput
from kolega_code.cli.config import config_summary
from kolega_code.cli.session_event_store import FileSessionEventStore
from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.settings import CliSettings, SettingsStore
from kolega_code.events import KnownEventType
from kolega_code.permissions import PermissionKind, PermissionMode
from kolega_code.session.inbox import (
    MESSAGING_PROTOCOL_VERSION,
    InboxRegistry,
    PeerMessage,
    send_over_socket,
)

from kolega_code.tools import ToolError

from ._app_test_utils import (
    _build_sub_agent_test_app,
    build_test_config,
    install_fake_agents,
    renderable_text,
    wait_for_question_prompt,
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


# ---------------------------------------------------------------------------
# Commit 5: app wiring — registry registration, inbound policy, hold approval,
# recorded PEER_MESSAGE_* events.
# ---------------------------------------------------------------------------


#: State roots created by _two_apps; drained by the autouse fixture below.
_SHORT_STATE_ROOTS: list[Path] = []


@pytest.fixture(autouse=True)
def _cleanup_short_state_roots():
    yield
    import shutil

    while _SHORT_STATE_ROOTS:
        shutil.rmtree(_SHORT_STATE_ROOTS.pop(), ignore_errors=True)


def _short_state_root() -> Path:
    """Short root so real socket binds stay under the AF_UNIX limit."""
    import tempfile

    try:
        root = Path(tempfile.mkdtemp(prefix="kolega-peer-", dir="/tmp"))
    except OSError:
        root = Path(tempfile.mkdtemp(prefix="kolega-peer-"))
    _SHORT_STATE_ROOTS.append(root)
    return root


def _two_apps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    settings: CliSettings | None = None,
):
    """Two mounted-able apps sharing one injected registry and state dir."""
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch)
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    config = build_test_config(project)
    state_root = _short_state_root()
    store = SessionStore(state_root)
    if settings is not None:
        SettingsStore(state_root).save(settings)
    registry = InboxRegistry()
    session_a = store.create(project, "code", config_summary(config), title="alpha")
    session_b = store.create(project, "code", config_summary(config), title="beta")
    common = {"mode": "code", "store": store, "inbox_registry": registry}
    app_a = KolegaCodeApp(project_path=project, config=config, session=session_a, **common)
    app_b = KolegaCodeApp(project_path=project, config=config, session=session_b, **common)
    return registry, app_a, app_b


def _message_from(app_a, text="hello from alpha"):
    return PeerMessage.create(
        sender_session_id=app_a.session.session_id,
        sender_title="alpha",
        text=text,
    )


async def _poll(predicate, pilot, *, timeout: float = 6.0, description: str):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await pilot.pause(0.01)
    raise AssertionError(f"Timed out waiting for {description}")


@pytest.mark.asyncio
async def test_sessions_register_on_mount_and_are_visible_to_peers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, app_a, app_b = _two_apps(tmp_path, monkeypatch)

    async with app_a.run_test():
        async with app_b.run_test():
            assert registry.is_registered(app_a.session.session_id)
            assert registry.is_registered(app_b.session.session_id)

            peers_seen_by_a = {
                agent.session_id: agent for agent in registry.list_agents(exclude_session_id=app_a.session.session_id)
            }
            assert set(peers_seen_by_a) == {app_b.session.session_id}
            assert peers_seen_by_a[app_b.session.session_id].title == "beta"
            assert peers_seen_by_a[app_b.session.session_id].status == "idle"

            # Live callables: status is read per query, so flipping B's state
            # shows up on the next discovery, not in stale snapshots.
            peers_snapshot = {
                agent.session_id: agent for agent in registry.list_agents(exclude_session_id=app_a.session.session_id)
            }
            assert peers_snapshot[app_b.session.session_id].status == "idle"
            app_b._turn_active = True
            fresh = {
                agent.session_id: agent for agent in registry.list_agents(exclude_session_id=app_a.session.session_id)
            }
            assert fresh[app_b.session.session_id].status == "busy"
            app_b._turn_active = False

    # Unregistration is the shutdown backstop; direct call keeps sessions private.
    registry.unregister(app_b.session.session_id)
    assert not registry.is_registered(app_b.session.session_id)


@pytest.mark.asyncio
async def test_deliver_to_idle_recipient_starts_a_turn_and_records_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, app_a, app_b = _two_apps(tmp_path, monkeypatch)

    async with app_a.run_test():
        async with app_b.run_test() as pilot_b:
            outcome = await registry.deliver(
                app_b.session.session_id,
                _message_from(app_a),
                sender_session_id=app_a.session.session_id,
            )
            assert outcome == "accepted"

            await _poll(
                lambda: any(entry.kind == "peer_message" for entry in app_b.conversation_entries),
                pilot_b,
                description="the peer turn to start",
            )
            await wait_for_turn_idle(app_b, pilot_b)
            await pilot_b.pause(0.05)

            agent = app_b.agent
            assert agent is not None
            sent = getattr(agent, "messages")
            assert len(sent) == 1
            assert "[Peer message from session 'alpha'" in sent[0]
            assert sent[0].endswith("hello from alpha")

            (entry,) = [entry for entry in app_b.conversation_entries if entry.kind == "peer_message"]
            assert entry.content == "hello from alpha"

            store_events = FileSessionEventStore(app_b.store.journal(app_b.session.session_id))
            recorded = await store_events.read(
                app_b.session.session_id,
                types={
                    KnownEventType.PEER_MESSAGE_RECEIVED,
                    KnownEventType.PEER_MESSAGE_DELIVERED,
                },
            )
            assert [event.event_type for event in recorded] == [
                KnownEventType.PEER_MESSAGE_RECEIVED,
                KnownEventType.PEER_MESSAGE_DELIVERED,
            ]
            assert recorded[0].content["sender_id"] == app_a.session.session_id
            assert recorded[0].content["text"] == "hello from alpha"


@pytest.mark.asyncio
async def test_deliver_while_busy_queues_and_starts_when_idle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry, app_a, app_b = _two_apps(tmp_path, monkeypatch)

    async with app_a.run_test():
        async with app_b.run_test() as pilot_b:
            app_b._turn_active = True  # simulate a running turn
            outcome = await registry.deliver(
                app_b.session.session_id,
                _message_from(app_a),
                sender_session_id=app_a.session.session_id,
            )
            assert outcome == "accepted"
            assert len(app_b._queued_messages) == 1
            assert "(from alpha)" in app_b._queued_messages_preview()

            app_b._turn_active = False
            app_b._schedule_maybe_start_queued_message()
            await _poll(
                lambda: any(entry.kind == "peer_message" for entry in app_b.conversation_entries),
                pilot_b,
                description="the queued peer turn to start",
            )
            await wait_for_turn_idle(app_b, pilot_b)
            await pilot_b.pause(0.05)


@pytest.mark.asyncio
async def test_refuse_policy_drops_silently_without_erroring_the_sender(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, app_a, app_b = _two_apps(tmp_path, monkeypatch, settings=CliSettings(cross_session_inbound="refuse"))

    async with app_a.run_test():
        async with app_b.run_test() as pilot_b:
            outcome = await registry.deliver(
                app_b.session.session_id,
                _message_from(app_a),
                sender_session_id=app_a.session.session_id,
            )

            assert outcome == "refused"
            assert app_b._queued_messages == []
            await pilot_b.pause(0.05)
            assert not any(entry.kind == "peer_message" for entry in app_b.conversation_entries)


@pytest.mark.asyncio
async def test_hold_policy_asks_and_acceptance_delivers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry, app_a, app_b = _two_apps(tmp_path, monkeypatch, settings=CliSettings(cross_session_inbound="hold"))

    async with app_a.run_test():
        async with app_b.run_test() as pilot_b:
            outcome = await registry.deliver(
                app_b.session.session_id,
                _message_from(app_a),
                sender_session_id=app_a.session.session_id,
            )
            assert outcome == "held"
            assert app_b._queued_messages == [], "a held message must not enter the queue before approval"

            await wait_for_question_prompt(app_b, pilot_b)
            assert app_b._pending_question is not None
            assert "alpha" in app_b._pending_question.question

            await app_b._answer_pending_question("Accept")

            await _poll(
                lambda: any(entry.kind == "peer_message" for entry in app_b.conversation_entries),
                pilot_b,
                description="the accepted peer turn to start",
            )
            await wait_for_turn_idle(app_b, pilot_b)
            await pilot_b.pause(0.05)

            store_events = FileSessionEventStore(app_b.store.journal(app_b.session.session_id))
            delivered = await store_events.read(app_b.session.session_id, types={KnownEventType.PEER_MESSAGE_DELIVERED})
            assert len(delivered) == 1


@pytest.mark.asyncio
async def test_hold_policy_drop_leaves_nothing_behind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry, app_a, app_b = _two_apps(tmp_path, monkeypatch, settings=CliSettings(cross_session_inbound="hold"))

    async with app_a.run_test():
        async with app_b.run_test() as pilot_b:
            outcome = await registry.deliver(
                app_b.session.session_id,
                _message_from(app_a),
                sender_session_id=app_a.session.session_id,
            )
            assert outcome == "held"

            await wait_for_question_prompt(app_b, pilot_b)
            await app_b._answer_pending_question("Drop")

            await _poll(
                lambda: app_b.control_channel.pending() == [],
                pilot_b,
                description="the hold request to settle",
            )
            await pilot_b.pause(0.05)
            assert app_b._queued_messages == []
            assert not any(entry.kind == "peer_message" for entry in app_b.conversation_entries)


@pytest.mark.asyncio
async def test_hold_approval_expires_after_dialog_expiry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry, app_a, app_b = _two_apps(
        tmp_path,
        monkeypatch,
        settings=CliSettings(cross_session_inbound="hold", dialog_expiry=0.3),
    )

    async with app_a.run_test():
        async with app_b.run_test() as pilot_b:
            outcome = await registry.deliver(
                app_b.session.session_id,
                _message_from(app_a),
                sender_session_id=app_a.session.session_id,
            )
            assert outcome == "held"
            await wait_for_question_prompt(app_b, pilot_b)

            # Never answered: the short dialog expiry must settle it to Drop.
            await _poll(
                lambda: app_b.control_channel.pending() == [],
                pilot_b,
                timeout=5.0,
                description="the hold approval to expire",
            )
            await _poll(
                lambda: app_b._pending_question is None,
                pilot_b,
                description="the expired prompt to clear",
            )
            await pilot_b.pause(0.05)
            assert app_b._queued_messages == []


@pytest.mark.asyncio
async def test_auto_matrix_holds_mixed_permission_modes_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ask recipient + bypassing sender holds; equals accept immediately."""
    registry, app_a, app_b = _two_apps(tmp_path, monkeypatch)

    async with app_a.run_test():
        async with app_b.run_test() as pilot_b:
            mixed = PeerMessage.create(
                sender_session_id=app_a.session.session_id,
                sender_title="alpha",
                sender_mode="auto",
                text="from a bypassing sender",
            )
            outcome = await registry.deliver(
                app_b.session.session_id, mixed, sender_session_id=app_a.session.session_id
            )
            assert outcome == "held"
            await wait_for_question_prompt(app_b, pilot_b)
            await app_b._answer_pending_question("Drop")
            await _poll(lambda: app_b.control_channel.pending() == [], pilot_b, description="settle")

            # Recipient switches to bypass mode; a fellow bypasser is accepted.
            app_b.permission_mode = PermissionMode.AUTO
            outcome = await registry.deliver(
                app_b.session.session_id, mixed, sender_session_id=app_a.session.session_id
            )
            assert outcome == "accepted"
            await _poll(
                lambda: any(entry.kind == "peer_message" for entry in app_b.conversation_entries),
                pilot_b,
                description="the accepted peer turn to start",
            )
            await wait_for_turn_idle(app_b, pilot_b)
            await pilot_b.pause(0.05)


# ---------------------------------------------------------------------------
# Commit 6: the list_agents / send_message model-facing tools.
# ---------------------------------------------------------------------------


def _extension(app):
    """Build the real tool extension; closures only need app attributes."""
    return app._peer_messaging_tool_extension()


@pytest.mark.asyncio
async def test_peer_messaging_extension_reaches_agent_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_build_agent must hand the extension to every top-level agent generation."""
    app = _build_sub_agent_test_app(tmp_path, monkeypatch)

    async with app.run_test():
        agent = app.agent
        assert agent is not None
        # FakeCoderAgent records its constructor kwargs, so the extension list
        # _build_agent assembled is inspectable without building a real agent.
        constructed_kwargs: dict = getattr(agent, "kwargs")
        extensions = {ext.name: ext for ext in constructed_kwargs["tool_extensions"]}
        assert "cli-peer-messaging" in extensions
        extension = extensions["cli-peer-messaging"]
        assert set(extension.tools) == {"list_agents", "send_message"}
        assert extension.propagate_to_sub_agents is False
        assert not extension.exclusive_tools
        # Descriptions are declared data, loaded verbatim from the assets.
        from kolega_code.agent.tool_definitions import tool_description_asset

        assert extension.tool_descriptions["list_agents"] == tool_description_asset("list_agents")
        assert extension.tool_descriptions["send_message"] == tool_description_asset("send_message")


@pytest.mark.asyncio
async def test_list_agents_tool_formats_live_peers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry, app_a, app_b = _two_apps(tmp_path, monkeypatch)

    async with app_a.run_test():
        async with app_b.run_test():
            result = await _extension(app_a).tools["list_agents"]()

            assert "beta" in result
            assert "idle" in result
            assert "project" in result  # project basename
            assert app_b.session.session_id[:8] in result
            assert "alpha" not in result.split("Live peer sessions:")[1], "self is never listed"

            # Empty registry says so explicitly instead of returning "".
            registry.unregister(app_b.session.session_id)
            empty = await _extension(app_a).tools["list_agents"]()
            assert "No other live sessions" in empty


@pytest.mark.asyncio
async def test_send_message_tool_resolves_names_and_reports_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, app_a, app_b = _two_apps(tmp_path, monkeypatch)

    async with app_a.run_test():
        async with app_b.run_test():
            result = await _extension(app_a).tools["send_message"](recipient="beta", text="ping from the tool")

            assert result.startswith("Message delivered to beta (")
            assert "awaiting" not in result
            assert len(app_b._queued_messages) == 1
            queued = app_b._queued_messages[0]
            # The model text carries the preamble; the transcript keeps it raw.
            assert "[Peer message from session 'alpha'" in queued.text
            assert queued.text.endswith("ping from the tool")
            assert queued.display_text == "ping from the tool"
            assert queued.entry.content == "ping from the tool"

            # Prefix addressing works too.
            result = await _extension(app_a).tools["send_message"](recipient="bet", text="again")
            assert result.startswith("Message delivered to beta (")


@pytest.mark.asyncio
async def test_send_message_tool_errors_are_explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from kolega_code.tools import ToolError

    registry, app_a, app_b = _two_apps(tmp_path, monkeypatch)

    async with app_a.run_test():
        async with app_b.run_test():
            send = _extension(app_a).tools["send_message"]

            with pytest.raises(ToolError, match="No reachable session"):
                await send(recipient="ghost", text="hi")
            with pytest.raises(ToolError, match="must name a live session"):
                await send(recipient="  ", text="hi")
            with pytest.raises(ToolError):
                # Self-addressing: "alpha" resolves to nothing once self is excluded.
                await send(recipient="alpha", text="note to self")
            with pytest.raises(ToolError, match="must not be empty"):
                await send(recipient="beta", text="   ")
            with pytest.raises(ToolError, match="character limit"):
                await send(recipient="beta", text="x" * 64_001)

            assert app_b._queued_messages == [], "failed sends must not enqueue anything"


@pytest.mark.asyncio
async def test_send_message_tool_ambiguous_name_fails_loudly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two same-named peers make the name ambiguous for everyone else."""
    pytest.importorskip("textual")

    from kolega_code.cli.app import KolegaCodeApp

    install_fake_agents(monkeypatch)
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    config = build_test_config(project)
    store = SessionStore(tmp_path / "state")
    registry = InboxRegistry()
    session_a = store.create(project, "code", config_summary(config), title="observer")
    session_b = store.create(project, "code", config_summary(config), title="worker")
    session_c = store.create(project, "code", config_summary(config), title="worker")
    common = {"mode": "code", "store": store, "inbox_registry": registry}
    app_a = KolegaCodeApp(project_path=project, config=config, session=session_a, **common)
    app_b = KolegaCodeApp(project_path=project, config=config, session=session_b, **common)
    app_c = KolegaCodeApp(project_path=project, config=config, session=session_c, **common)

    async with app_a.run_test():
        async with app_b.run_test():
            async with app_c.run_test():
                with pytest.raises(ToolError, match="Multiple sessions are named"):
                    await _extension(app_a).tools["send_message"](recipient="worker", text="hi")

                # The id form still disambiguates.
                result = await _extension(app_a).tools["send_message"](recipient=session_c.session_id[:8], text="hi")
                assert f"Message delivered to worker ({session_c.session_id[:8]})" in result


@pytest.mark.asyncio
async def test_held_receipt_tells_the_sender_review_is_pending(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry, app_a, app_b = _two_apps(tmp_path, monkeypatch, settings=CliSettings(cross_session_inbound="hold"))

    async with app_a.run_test():
        async with app_b.run_test() as pilot_b:
            result = await _extension(app_a).tools["send_message"](recipient="beta", text="held please")

            assert "awaiting the recipient's review" in result
            await wait_for_question_prompt(app_b, pilot_b)
            await app_b._answer_pending_question("Drop")
            await _poll(lambda: app_b.control_channel.pending() == [], pilot_b, description="settle")


@pytest.mark.asyncio
async def test_send_message_approval_dialog_shows_recipient_and_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")

    from kolega_code.cli.tui.state import PendingApproval
    from kolega_code.cli.tui.widgets import ActionList
    from kolega_code.permissions import allow_rule_options, permission_request_for_tool

    registry, app_a, app_b = _two_apps(tmp_path, monkeypatch)

    async with app_a.run_test():
        async with app_b.run_test():
            request = permission_request_for_tool(
                "send_message",
                {"recipient": "deploy-bot", "text": "Please rerun the nightly job"},
            )
            assert request is not None and request.kind == PermissionKind.MESSAGE

            app_b._pending_approval = PendingApproval(
                request=request,
                request_id="req-peer",
                rule_options=allow_rule_options(request),
            )
            app_b._set_approval_actions_visible(True)
            try:
                approval_actions = app_b.query_one("#approval_actions", ActionList)
                prompts = [
                    str(approval_actions.get_option(f"approval_option_{index}").prompt)
                    for index in range(approval_actions.option_count)
                ]
                assert prompts[:2] == ["1. Allow once", "2. Deny"]
                joined = "\n".join(prompts)
                assert "Always allow messages to `deploy-bot`" in joined
                assert "Always allow `send_message`" in joined

                body = app_b._format_permission_content(request)
                assert "deploy-bot" in body
                assert "Please rerun the nightly job" in body
            finally:
                app_b._pending_approval = None
                app_b._set_approval_actions_visible(False)


# ---------------------------------------------------------------------------
# Phase 2: cross-process transport — the TUI binds its inbox socket.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sessions_bind_owner_only_sockets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    from kolega_code.session.inbox import parse_socket_name

    registry, app_a, app_b = _two_apps(tmp_path, monkeypatch)

    async with app_a.run_test():
        async with app_b.run_test():
            for app in (app_a, app_b):
                assert app.messaging_socket_path is not None
                assert app.messaging_socket_path.exists()
                parsed = parse_socket_name(app.messaging_socket_path.name)
                assert parsed is not None
                session_id, pid = parsed
                assert session_id == app.session.session_id
                assert pid == os.getpid()

            # The bound path is what child processes will see.
            agent = app_a.agent
            assert agent is not None
            assert getattr(agent, "messaging_socket_path") == app_a.messaging_socket_path


@pytest.mark.asyncio
async def test_envelope_over_the_socket_lands_in_the_recipient_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raw socket client (what another process would be) delivers into B."""
    registry, app_a, app_b = _two_apps(tmp_path, monkeypatch)

    async with app_a.run_test():
        async with app_b.run_test() as pilot_b:
            assert app_b.messaging_socket_path is not None
            response = await send_over_socket(
                app_b.messaging_socket_path,
                {
                    "v": MESSAGING_PROTOCOL_VERSION,
                    "kind": "message",
                    "sender_id": app_a.session.session_id,
                    "sender_title": "alpha",
                    "sender_mode": "ask",
                    "text": "over the wire",
                },
            )

            assert response["ok"] is True
            assert response["outcome"] == "accepted"

            await _poll(
                lambda: any(entry.kind == "peer_message" for entry in app_b.conversation_entries),
                pilot_b,
                description="the delivered peer turn to start",
            )
            await wait_for_turn_idle(app_b, pilot_b)
            await pilot_b.pause(0.05)

            store_events = FileSessionEventStore(app_b.store.journal(app_b.session.session_id))
            recorded = await store_events.read(app_b.session.session_id, types={KnownEventType.PEER_MESSAGE_DELIVERED})
            assert len(recorded) == 1


@pytest.mark.asyncio
async def test_feature_gate_disables_the_socket_transport(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from kolega_code.session.inbox import MESSAGING_ENV_FLAG, messaging_enabled

    registry, app_a, app_b = _two_apps(tmp_path, monkeypatch)
    monkeypatch.setenv(MESSAGING_ENV_FLAG, "off")

    assert messaging_enabled() is False

    async with app_a.run_test():
        async with app_b.run_test():
            assert app_a.messaging_socket_path is None
            agent = app_a.agent
            assert agent is not None
            assert getattr(agent, "messaging_socket_path") is None


@pytest.mark.asyncio
async def test_messaging_gate_defaults_on(monkeypatch: pytest.MonkeyPatch) -> None:
    from kolega_code.session.inbox import MESSAGING_ENV_FLAG, messaging_enabled

    monkeypatch.delenv(MESSAGING_ENV_FLAG, raising=False)
    assert messaging_enabled() is True
    monkeypatch.setenv(MESSAGING_ENV_FLAG, "ON")
    assert messaging_enabled() is True
    monkeypatch.setenv(MESSAGING_ENV_FLAG, "off")
    assert messaging_enabled() is False
