"""In-process peer inbox: registration, addressing, policy, and delivery.

Like the control-channel tests, the failure modes matter as much as the happy
path: a send that silently goes nowhere, an ambiguous address that reaches the
wrong peer, or a policy that lets a bypassing sender ride its authority into a
cautious recipient are all pinned here.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

from kolega_code.session.inbox import (
    MAX_PEER_TEXT_BYTES,
    MAX_QUEUED_PEER_MESSAGES,
    MESSAGING_PROTOCOL_VERSION,
    DeliveryOutcome,
    ENVELOPE_MAX_BYTES,
    InboxRegistration,
    InboxRegistry,
    PeerMessage,
    PeerMessageError,
    PeerRepeatGuard,
    PeerSocketServer,
    encode_envelope,
    is_pid_alive,
    messaging_dir,
    parse_envelope,
    parse_socket_name,
    provenance_preamble,
    recipient_queue_full,
    resolve_inbound_decision,
    send_over_socket,
    socket_path_for,
    sweep_stale_sockets,
    validate_peer_text,
)


def _registration(
    session_id: str = "s1",
    title: str = "myapp",
    project_path: str = "/tmp/myapp",
    status: str = "idle",
    outcomes: list[str] | None = None,
) -> InboxRegistration:
    def deliver(_message: PeerMessage) -> str:
        if outcomes is not None:
            return outcomes.pop(0)
        return DeliveryOutcome.ACCEPTED.value

    async def deliver_async(message: PeerMessage) -> str:
        return deliver(message)

    return InboxRegistration(
        session_id=session_id,
        describe_title=lambda: title,
        describe_project_path=lambda: project_path,
        describe_status=lambda: status,
        deliver_message=deliver_async,
    )


def _message(text: str = "hello", sender_session_id: str = "s2", **kwargs) -> PeerMessage:
    return PeerMessage.create(sender_session_id=sender_session_id, sender_title="peer", text=text, **kwargs)


# -- Registration lifecycle --------------------------------------------------


def test_register_and_unregister_round_trip() -> None:
    registry = InboxRegistry()
    registry.register(_registration())
    assert registry.is_registered("s1")

    registry.unregister("s1")
    assert not registry.is_registered("s1")
    assert registry.list_agents() == []


def test_unregister_unknown_session_is_a_no_op() -> None:
    InboxRegistry().unregister("never-registered")


def test_reregistration_replaces_the_entry() -> None:
    registry = InboxRegistry()
    registry.register(_registration(status="idle"))
    registry.register(_registration(status="busy"))

    (agent,) = registry.list_agents()
    assert agent.status == "busy"


# -- Discovery ---------------------------------------------------------------


def test_list_agents_reports_live_callables_and_excludes_self() -> None:
    registry = InboxRegistry()
    registry.register(_registration(session_id="self", title="self-session", status="busy"))
    registry.register(_registration(session_id="b", title="zeta"))
    registry.register(_registration(session_id="a", title="alpha", project_path="/tmp/alpha"))

    agents = registry.list_agents(exclude_session_id="self")

    assert [agent.title for agent in agents] == ["alpha", "zeta"]  # sorted by name
    assert [agent.status for agent in agents] == ["idle", "idle"]
    assert agents[0].session_id == "a"
    assert agents[0].project_path == "/tmp/alpha"
    # Self never appears in discovery.
    assert all(agent.session_id != "self" for agent in agents)


# -- Addressing --------------------------------------------------------------


def test_resolve_by_exact_name_case_insensitive() -> None:
    registry = InboxRegistry()
    registry.register(_registration(title="MyApp"))
    registry.register(_registration(session_id="other", title="other"))

    resolved = registry.resolve(" myapp ", exclude_session_id="other")
    assert resolved.session_id == "s1"


def test_resolve_by_unique_name_prefix_and_by_full_id() -> None:
    registry = InboxRegistry()
    registry.register(_registration(title="deploy-bot", session_id="aaaa-bbbb"))
    registry.register(_registration(session_id="cccc-dddd", title="unrelated"))

    assert registry.resolve("dep").session_id == "aaaa-bbbb"
    assert registry.resolve("aaaa-bbbb").session_id == "aaaa-bbbb"
    assert registry.resolve("cccc").session_id == "cccc-dddd"  # id prefix


@pytest.mark.parametrize("query", ["", "   ", "ghost"])
@pytest.mark.asyncio
async def test_resolve_unknown_recipient_raises(query: str) -> None:
    registry = InboxRegistry()
    registry.register(_registration())

    with pytest.raises(PeerMessageError):
        registry.resolve(query)


def test_resolve_ambiguous_prefix_lists_candidates() -> None:
    registry = InboxRegistry()
    registry.register(_registration(session_id="1", title="deploy-api"))
    registry.register(_registration(session_id="2", title="deploy-web"))

    with pytest.raises(PeerMessageError, match="deploy-api.*deploy-web|deploy-web.*deploy-api"):
        registry.resolve("deploy")


def test_resolve_duplicate_titles_are_ambiguous_not_silent() -> None:
    """Two sessions sharing a name must fail loudly, not message the wrong one."""
    registry = InboxRegistry()
    registry.register(_registration(session_id="1", title="myapp"))
    registry.register(_registration(session_id="2", title="myapp"))

    with pytest.raises(PeerMessageError, match="address one by session id"):
        registry.resolve("myapp")

    # The id form still works.
    assert registry.resolve("2").session_id == "2"


# -- Inbound policy ----------------------------------------------------------


@pytest.mark.parametrize(
    "recipient_mode,sender_mode,expected",
    [
        ("ask", "ask", DeliveryOutcome.ACCEPTED),
        ("auto", "auto", DeliveryOutcome.ACCEPTED),
        ("ask", "auto", DeliveryOutcome.HELD),
        ("auto", "ask", DeliveryOutcome.HELD),
    ],
)
def test_auto_policy_is_asymmetric_by_permission_mode(
    recipient_mode: str, sender_mode: str, expected: DeliveryOutcome
) -> None:
    assert resolve_inbound_decision("auto", recipient_mode, sender_mode) is expected


@pytest.mark.parametrize("recipient_mode,sender_mode", [("ask", "ask"), ("ask", "auto"), ("auto", "ask")])
def test_explicit_policies_override_the_matrix(recipient_mode: str, sender_mode: str) -> None:
    assert resolve_inbound_decision("accept", recipient_mode, sender_mode) is DeliveryOutcome.ACCEPTED
    assert resolve_inbound_decision("hold", recipient_mode, sender_mode) is DeliveryOutcome.HELD
    assert resolve_inbound_decision("refuse", recipient_mode, sender_mode) is DeliveryOutcome.REFUSED


@pytest.mark.parametrize("policy", [None, "", "nonsense"])
def test_unknown_policy_degrades_to_auto(policy: object) -> None:
    assert resolve_inbound_decision(policy, "ask", "ask") is DeliveryOutcome.ACCEPTED  # type: ignore[arg-type]
    assert resolve_inbound_decision(policy, "auto", "ask") is DeliveryOutcome.HELD  # type: ignore[arg-type]


# -- Delivery ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_deliver_routes_to_the_registration_outcome() -> None:
    outcomes = [DeliveryOutcome.HELD.value]
    registry = InboxRegistry()
    registry.register(_registration(outcomes=outcomes))

    outcome = await registry.deliver("s1", _message(), sender_session_id="s2")

    assert outcome == DeliveryOutcome.HELD.value


@pytest.mark.asyncio
async def test_deliver_to_unknown_recipient_raises() -> None:
    registry = InboxRegistry()

    with pytest.raises(PeerMessageError, match="No live session"):
        await registry.deliver("missing", _message())


@pytest.mark.asyncio
async def test_deliver_rejects_self_addressing_even_when_registered() -> None:
    """The tool rejects self-sends first; the registry is the backstop."""
    registry = InboxRegistry()
    registry.register(_registration(session_id="me"))

    with pytest.raises(PeerMessageError, match="cannot message itself"):
        await registry.deliver("me", _message(sender_session_id="me"), sender_session_id="me")


@pytest.mark.asyncio
async def test_deliver_rejects_empty_text() -> None:
    registry = InboxRegistry()
    registry.register(_registration())

    with pytest.raises(PeerMessageError, match="must not be empty"):
        await registry.deliver("s1", _message(text="   "))


@pytest.mark.asyncio
async def test_deliver_rejects_oversized_text() -> None:
    registry = InboxRegistry()
    registry.register(_registration())

    oversized = PeerMessage.create(sender_session_id="s2", sender_title="p", text="x" * (MAX_PEER_TEXT_BYTES + 1))
    with pytest.raises(PeerMessageError, match="byte limit"):
        await registry.deliver("s1", oversized)


def test_validate_peer_text_strips_whitespace() -> None:
    assert validate_peer_text("  hi \n") == "hi"


# -- Message shape -----------------------------------------------------------


def test_peer_message_create_stamps_identity_and_time() -> None:
    message = PeerMessage.create(sender_session_id="s1", sender_title="a", text="hi")

    assert message.message_id
    assert message.sender_mode == "ask"
    assert message.created_at  # utc_now_iso stamped by default factory
    assert message.reply_to is None


@pytest.mark.asyncio
async def test_delivery_failure_inside_callback_propagates() -> None:
    """A recipient-side crash must surface to the sender, never fake success."""

    async def boom(_message: PeerMessage) -> str:
        raise RuntimeError("queue exploded")

    registry = InboxRegistry()
    registry.register(_registration())
    registry.register(
        InboxRegistration(
            session_id="broken",
            describe_title=lambda: "broken",
            describe_project_path=lambda: "/x",
            describe_status=lambda: "idle",
            deliver_message=boom,
        )
    )

    with pytest.raises(RuntimeError, match="queue exploded"):
        await registry.deliver("broken", _message())


# The shared module-level registry exists so ordinary hosts need no wiring.
def test_shared_registry_is_a_registry() -> None:
    from kolega_code.session.inbox import SHARED_INBOX_REGISTRY

    assert isinstance(SHARED_INBOX_REGISTRY, InboxRegistry)


# ---------------------------------------------------------------------------
# Phase 2: same-machine cross-process transport.
# ---------------------------------------------------------------------------


def _short_state_root() -> Path:
    """A state root short enough for real AF_UNIX binds on macOS.

    pytest's tmp_path nests too deep for the 104-byte sun_path limit; the
    production default state dir sits far below it. Each call gets a fresh
    /tmp directory so tests never share a messaging dir.
    """
    try:
        # /tmp explicitly: pytest rewrites TMPDIR into its deep per-test tree,
        # which blows the 104-byte AF_UNIX limit on macOS.
        return Path(tempfile.mkdtemp(prefix="kolega-msg-", dir="/tmp"))
    except (OSError, FileNotFoundError):
        return Path(tempfile.mkdtemp(prefix="kolega-msg-"))


@pytest.fixture
def short_state_root() -> Any:
    root = _short_state_root()
    yield root
    import shutil

    shutil.rmtree(root, ignore_errors=True)


def _make_server(
    outcomes: list[str] | None = None,
    *,
    state_root: Path | None = None,
    session_id: str = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
) -> tuple[PeerSocketServer, list[PeerMessage]]:
    received: list[PeerMessage] = []

    async def deliver(message: PeerMessage) -> str:
        received.append(message)
        if outcomes:
            return outcomes.pop(0)
        return DeliveryOutcome.ACCEPTED.value

    server = PeerSocketServer(
        directory=messaging_dir(state_root if state_root is not None else _short_state_root()),
        session_id=session_id,
        pid=os.getpid(),
        describe_status=(lambda: "idle"),
        deliver_message=deliver,
    )
    return server, received


# -- Socket naming, liveness, sweeping ----------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee.4242.sock", ("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", 4242)),
        ("shortid.1.sock", None),
        ("notes.txt", None),
        ("missing-pid.sock", None),
        ("aaaa-bbbb.notapid.sock", None),
    ],
)
def test_parse_socket_name(name: str, expected: tuple[str, int] | None) -> None:
    assert parse_socket_name(name) == expected


def test_is_pid_alive() -> None:
    assert is_pid_alive(os.getpid()) is True
    assert is_pid_alive(-1) is False


def test_sweep_removes_only_dead_pid_sockets(tmp_path: Path) -> None:
    directory = messaging_dir(tmp_path / "state")
    dead = socket_path_for(directory, "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", 2**22 - 1)
    foreign = directory / "someone-elses-file.txt"
    dead.write_bytes(b"")
    foreign.write_bytes(b"keep me")
    live = socket_path_for(directory, "bbbbbbbb-bbbb-cccc-dddd-eeeeeeeeeeee", os.getpid())
    live.write_bytes(b"")

    removed = sweep_stale_sockets(directory)

    assert removed == [dead]
    assert not dead.exists()
    assert foreign.exists(), "unparsable files are never touched"
    assert live.exists(), "a live pid's socket survives the sweep"


# -- Envelope validation -------------------------------------------------------


def test_encode_and_parse_message_round_trip() -> None:
    envelope = {
        "v": MESSAGING_PROTOCOL_VERSION,
        "kind": "message",
        "sender_id": "s-1",
        "sender_title": "alpha",
        "sender_project": "/tmp/p",
        "sender_mode": "auto",
        "text": "hello",
        "reply_to": None,
    }
    parsed = parse_envelope(encode_envelope(envelope))

    assert parsed["kind"] == "message"
    assert parsed["sender_id"] == "s-1"
    assert parsed["sender_mode"] == "auto"
    assert parsed["text"] == "hello"
    assert parsed["reply_to"] is None


def test_parse_status_envelope() -> None:
    parsed = parse_envelope({"v": 1, "kind": "status"})
    assert parsed == {"v": 1, "kind": "status"}


@pytest.mark.parametrize(
    "payload",
    [
        {"v": 2, "kind": "message", "sender_id": "s", "text": "hi"},
        {"v": 1, "kind": "explode", "sender_id": "s", "text": "hi"},
        {"v": 1, "kind": "message", "text": "hi"},
        {"v": 1, "kind": "message", "sender_id": "s"},
        {"v": 1, "kind": "message", "sender_id": "s", "text": ""},
        {"v": 1, "kind": "message", "sender_id": "s", "text": "x" * (MAX_PEER_TEXT_BYTES + 1)},
        {"v": 1, "kind": "message", "sender_id": "", "text": "hi"},
        {"v": 1, "kind": "message", "sender_id": " ", "text": "hi"},
        {"v": 1, "kind": "message", "sender_id": 7, "text": "hi"},
        {"v": 1, "kind": "message", "sender_id": "s", "text": 42},
    ],
)
@pytest.mark.asyncio
async def test_malformed_envelopes_are_rejected(payload: dict) -> None:
    with pytest.raises(PeerMessageError):
        parse_envelope(encode_envelope(payload))


@pytest.mark.asyncio
async def test_non_json_and_oversized_lines_are_rejected() -> None:
    for bad in [b"not json\n", b"[1,2]\n", b"", b"   \n"]:
        with pytest.raises(PeerMessageError):
            parse_envelope(bad)
    with pytest.raises(PeerMessageError):
        encode_envelope({"text": "x" * (ENVELOPE_MAX_BYTES + 10)})


def test_multibyte_text_at_the_byte_cap_encodes_cleanly() -> None:
    """Regression: CJK-heavy text once inflated ~3x past the wire cap at encode.

    The limit is measured on the JSON-encoded form, so a max-size multibyte
    message must validate AND survive encode/parse within ENVELOPE_MAX_BYTES.
    """
    text = "消" * (MAX_PEER_TEXT_BYTES // 3)  # exactly fills the byte budget
    assert validate_peer_text(text) == text

    envelope = {
        "v": MESSAGING_PROTOCOL_VERSION,
        "kind": "message",
        "sender_id": "s",
        "sender_title": "t",
        "sender_project": "",
        "sender_mode": "ask",
        "text": text,
        "reply_to": None,
    }
    data = encode_envelope(envelope)
    assert len(data) <= ENVELOPE_MAX_BYTES + 1  # + trailing newline
    assert parse_envelope(data)["text"] == text


def test_control_char_heavy_text_is_measured_after_escaping() -> None:
    """Each quote doubles on encode; validation must catch that, not just raw size."""
    text = '"' * (MAX_PEER_TEXT_BYTES // 2 + 16)
    with pytest.raises(PeerMessageError, match="byte limit"):
        validate_peer_text(text)


# -- Server round trip ---------------------------------------------------------


@pytest.mark.asyncio
async def test_socket_delivers_message_and_reports_outcome(short_state_root: Any) -> None:
    server, received = _make_server(state_root=short_state_root)
    path = await server.start()
    try:
        response = await send_over_socket(
            path,
            {
                "v": MESSAGING_PROTOCOL_VERSION,
                "kind": "message",
                "sender_id": "remote-1",
                "sender_title": "remote",
                "text": "over the wire",
            },
        )

        assert response["ok"] is True
        assert response["outcome"] == DeliveryOutcome.ACCEPTED.value
        assert response["message_id"]
        (received,) = received
        assert received.text == "over the wire"
        assert received.sender_session_id == "remote-1"
    finally:
        await server.stop()
    assert not path.exists(), "stop unlinks the socket file"


@pytest.mark.asyncio
async def test_socket_status_query_reports_live_state(short_state_root: Any) -> None:
    server, _received = _make_server(state_root=short_state_root)
    path = await server.start()
    try:
        response = await send_over_socket(path, {"v": MESSAGING_PROTOCOL_VERSION, "kind": "status"})
        assert response == {"ok": True, "status": "idle"}
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_socket_surfaces_recipient_policy_errors_to_the_sender(short_state_root: Any) -> None:
    server, _received = _make_server(outcomes=[DeliveryOutcome.REFUSED.value], state_root=short_state_root)
    path = await server.start()
    try:
        response = await send_over_socket(
            path,
            {"v": MESSAGING_PROTOCOL_VERSION, "kind": "message", "sender_id": "r", "text": "hi"},
        )
        assert response["ok"] is True
        assert response["outcome"] == "refused"

        bad = await send_over_socket(
            path,
            {"v": MESSAGING_PROTOCOL_VERSION, "kind": "message", "sender_id": "r", "text": ""},
        )
        assert bad["ok"] is False
        assert "empty" in bad["error"].lower()
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_unreachable_socket_raises_instead_of_faking_success(tmp_path: Path) -> None:
    with pytest.raises(PeerMessageError, match="Could not reach"):
        await send_over_socket(tmp_path / "nothing-here.sock", {"v": 1, "kind": "status"})


@pytest.mark.asyncio
async def test_start_refuses_to_steal_a_live_owner_socket(short_state_root: Any) -> None:
    first, _ = _make_server(state_root=short_state_root)
    second, _ = _make_server(state_root=short_state_root)
    # Same session id and pid => same path. The second bind must fail loudly.
    await first.start()
    try:
        with pytest.raises(PeerMessageError, match="refusing to bind"):
            await second.start()
    finally:
        await first.stop()


@pytest.mark.asyncio
async def test_start_sweeps_a_dead_owner_socket_at_same_path(short_state_root: Any) -> None:
    directory = messaging_dir(short_state_root)
    stale = socket_path_for(directory, "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", 2**22 - 1)
    stale.write_bytes(b"")
    server, _received = _make_server(state_root=short_state_root)
    path = await server.start()
    try:
        assert path.exists()
    finally:
        await server.stop()


# -- Genuine cross-process exchange -------------------------------------------


CHILD_SERVER_PROGRAM = """
import asyncio, json, os, sys
from pathlib import Path

sys.path.insert(0, {repo!r})

from kolega_code.session.inbox import PeerSocketServer, messaging_dir

state_root = Path({state_root!r})
received = []

async def deliver(message):
    received.append(message.to_dict() if hasattr(message, "to_dict") else message.text)
    return "accepted"

async def main():
    server = PeerSocketServer(
        directory=messaging_dir(state_root),
        session_id={session_id!r},
        pid=os.getpid(),
        describe_status=lambda: "busy",
        deliver_message=deliver,
    )
    path = await server.start()
    print(json.dumps({{"ready": str(path)}}), flush=True)
    await asyncio.sleep(30)

asyncio.run(main())
"""


@pytest.mark.asyncio
async def test_cross_process_delivery_over_a_real_socket() -> None:
    """A child process owns the socket; this process sends to it.

    Hermetic two-process proof of the transport: no LLM, no TUI — protocol,
    liveness naming, and honest error responses across a real process boundary.
    """
    import shutil

    repo = str(Path(__file__).resolve().parents[2])
    state_root = _short_state_root()
    session_id = "cccccccc-dddd-eeee-ffff-000000000000"

    program = CHILD_SERVER_PROGRAM.format(repo=repo, state_root=str(state_root), session_id=session_id)
    proc = subprocess.Popen(
        [sys.executable, "-c", program],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ},
    )
    try:
        assert proc.stdout is not None
        deadline = time.monotonic() + 15
        ready: dict = {}
        while time.monotonic() < deadline:
            line = proc.stdout.readline().decode()
            if line.strip():
                ready = json.loads(line)
                break
            if proc.poll() is not None:
                raise AssertionError(f"child exited early: {proc.stderr.read().decode() if proc.stderr else ''}")
            await asyncio.sleep(0.05)
        assert ready.get("ready"), f"child never reported ready: {ready}"

        status = await send_over_socket(Path(ready["ready"]), {"v": 1, "kind": "status"})
        assert status == {"ok": True, "status": "busy"}

        sent = await send_over_socket(
            Path(ready["ready"]),
            {
                "v": 1,
                "kind": "message",
                "sender_id": "parent-session",
                "sender_title": "parent",
                "text": "cross-process hello",
            },
        )
        assert sent["ok"] is True
        assert sent["outcome"] == "accepted"
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    # After the child dies, its socket file is stale and gets swept.
    await asyncio.sleep(0.2)
    try:
        removed = sweep_stale_sockets(messaging_dir(state_root))
        assert removed, "the dead child's socket must be swept"
    finally:
        shutil.rmtree(state_root, ignore_errors=True)


# ---------------------------------------------------------------------------
# Phase 2: loop protection and recipient framing.
# ---------------------------------------------------------------------------


class _FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_repeat_guard_drops_identical_sender_text_inside_window() -> None:
    clock = _FakeClock()
    guard = PeerRepeatGuard(window_seconds=60.0)
    guard._monotonic = clock

    assert guard.allow("s1", "ping") is True
    assert guard.allow("s1", "ping") is False
    clock.advance(61)
    assert guard.allow("s1", "ping") is True, "the window expires"


def test_repeat_guard_keys_on_sender_and_text() -> None:
    guard = PeerRepeatGuard(window_seconds=60.0)
    guard._monotonic = _FakeClock()

    assert guard.allow("s1", "ping") is True
    assert guard.allow("s1", "different") is True, "new text from the same sender passes"
    assert guard.allow("s2", "ping") is True, "the same text from another sender passes"
    assert guard.allow("s1", "ping") is False


def test_provenance_preamble_carries_the_full_trust_model() -> None:
    message = PeerMessage.create(sender_session_id="s1", sender_title="alpha", text="hi")
    preamble = provenance_preamble(message)

    assert "'alpha'" in preamble
    for clause in ["context", "no", "authority"]:
        assert any(clause in word.lower() or clause in preamble.lower() for word in preamble.split())
    assert "permission settings" in preamble
    assert "configuration" in preamble
    assert "normal permission flow" in preamble


def test_recipient_queue_cap_constant_is_fifty() -> None:
    assert MAX_QUEUED_PEER_MESSAGES == 50
    assert recipient_queue_full(49) is False
    assert recipient_queue_full(50) is True
